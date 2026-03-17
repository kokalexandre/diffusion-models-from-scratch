from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.diffusion.ddim import DDIMSampler
from src.diffusion.ddpm_cifar import DDPMCIFAR
from src.metrics.inception_cifar import InceptionScoreAccumulator, InceptionV3FeatureExtractor
from src.metrics.stats import StreamingMeanCov, frechet_distance
from src.models.unet_cifar10 import UNetCIFAR10, UNetCIFAR10Config
from src.utils import EMA, load_yaml, set_seed


class InferenceModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, channels_last: bool = False):
        super().__init__()
        self.model = model
        self.channels_last = bool(channels_last)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if self.channels_last and x.ndim == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        return self.model(x, timesteps)


def build_model_from_cfg(cfg):
    mcfg = cfg["model"]
    return UNetCIFAR10(
        UNetCIFAR10Config(
            in_channels=int(mcfg["in_channels"]),
            out_channels=int(mcfg["out_channels"]),
            base_channels=int(mcfg["base_channels"]),
            channel_mults=tuple(int(v) for v in mcfg["channel_mults"]),
            num_res_blocks=int(mcfg["num_res_blocks"]),
            attn_resolutions=tuple(int(v) for v in mcfg["attn_resolutions"]),
            dropout=float(mcfg["dropout"]),
            resolution=int(mcfg["resolution"]),
        )
    )


def build_diffusion_from_cfg(cfg):
    dcfg = cfg["diffusion"]
    return DDPMCIFAR(
        timesteps=int(dcfg["timesteps"]),
        beta_start=float(dcfg["beta_start"]),
        beta_end=float(dcfg["beta_end"]),
        beta_schedule=str(dcfg.get("beta_schedule", "linear")),
        var_type=str(dcfg.get("var_type", "fixedlarge")),
    )


def configure_runtime(cfg, device: torch.device) -> dict:
    tcfg = cfg["training"]

    mixed_precision = str(tcfg.get("mixed_precision", "bf16")).lower()
    allow_tf32 = bool(tcfg.get("allow_tf32", True))
    cudnn_benchmark = bool(tcfg.get("cudnn_benchmark", True))
    channels_last = bool(tcfg.get("channels_last", True))
    compile_enabled = bool(tcfg.get("compile", True))
    compile_mode = str(tcfg.get("compile_mode", "default"))
    compile_backend = str(tcfg.get("compile_backend", "inductor"))

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if allow_tf32:
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        if mixed_precision == "bf16":
            bf16_ok = True
            if hasattr(torch.cuda, "is_bf16_supported"):
                bf16_ok = bool(torch.cuda.is_bf16_supported())
            if not bf16_ok:
                print("[runtime] BF16 requested but not supported on this GPU -> fallback to FP16")
                mixed_precision = "fp16"
    else:
        mixed_precision = "none"
        compile_enabled = False
        channels_last = False

    if compile_enabled and not hasattr(torch, "compile"):
        print("[runtime] torch.compile unavailable in this PyTorch build -> disabled")
        compile_enabled = False

    return {
        "mixed_precision": mixed_precision,
        "allow_tf32": allow_tf32,
        "cudnn_benchmark": cudnn_benchmark,
        "channels_last": channels_last,
        "compile_enabled": compile_enabled,
        "compile_mode": compile_mode,
        "compile_backend": compile_backend,
    }


def make_autocast_context(runtime: dict, device: torch.device):
    mp = runtime["mixed_precision"]
    enabled = device.type == "cuda" and mp in {"fp16", "bf16"}
    if not enabled:
        return nullcontext()

    dtype = torch.float16 if mp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_cifar10_eval_loader(
    root: str,
    train: bool,
    batch_size: int,
    num_workers: int,
    real_max: int | None,
    seed: int,
    download: bool = True,
):
    ds = CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ]
        ),
    )

    if real_max is not None and 0 < real_max < len(ds):
        g = torch.Generator()
        g.manual_seed(int(seed))
        idx = torch.randperm(len(ds), generator=g)[: int(real_max)].tolist()
        ds = Subset(ds, idx)

    loader_kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    dl = DataLoader(**loader_kwargs)
    return dl, ds


@torch.inference_mode()
def extract_real_stats(extractor, loader, device: torch.device):
    extractor.eval()
    stats = StreamingMeanCov(dim=2048)
    seen = 0

    for x, _ in tqdm(loader, desc="real features"):
        x = x.to(device, non_blocking=True).float()
        logits, feat = extractor(x)
        del logits
        stats.update(feat)
        seen += x.shape[0]

    mu, cov = stats.finalize()
    return mu, cov, seen


@torch.inference_mode()
def sample_images(
    *,
    sampler: str,
    diffusion: DDPMCIFAR,
    ddim: DDIMSampler,
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    runtime: dict,
    steps: int,
    schedule: str,
    eta: float,
    shape: tuple[int, int, int],
):
    with make_autocast_context(runtime, device):
        if sampler == "ddpm":
            imgs = diffusion.sample(model, batch_size=batch_size, device=device, shape=shape)
        elif sampler == "ddim":
            imgs = ddim.sample(
                model,
                batch_size=batch_size,
                device=device,
                shape=shape,
                steps=int(steps),
                schedule=schedule,
                eta=float(eta),
            )
        elif sampler == "ddpm_noisy":
            imgs = diffusion.sample_subsequence_ddpm_noisy(
                model,
                batch_size=batch_size,
                device=device,
                shape=shape,
                steps=int(steps),
                schedule=schedule,
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

    return imgs.float()


@torch.inference_mode()
def extract_gen_stats_and_is(
    *,
    sampler: str,
    diffusion: DDPMCIFAR,
    ddim: DDIMSampler,
    model: nn.Module,
    extractor,
    device: torch.device,
    runtime: dict,
    num_gen: int,
    gen_batch: int,
    is_splits: int,
    steps: int,
    schedule: str,
    eta: float,
    shape: tuple[int, int, int],
):
    model.eval()
    extractor.eval()

    stats = StreamingMeanCov(dim=2048)
    is_acc = InceptionScoreAccumulator()

    remaining = int(num_gen)
    pbar = tqdm(total=int(num_gen), desc=f"generate+features [{sampler}]")
    while remaining > 0:
        b = min(int(gen_batch), remaining)
        imgs = sample_images(
            sampler=sampler,
            diffusion=diffusion,
            ddim=ddim,
            model=model,
            batch_size=b,
            device=device,
            runtime=runtime,
            steps=int(steps),
            schedule=schedule,
            eta=eta,
            shape=shape,
        )
        logits, feat = extractor(imgs)
        stats.update(feat)
        is_acc.update_logits(logits)
        remaining -= b
        pbar.update(b)
    pbar.close()

    mu, cov = stats.finalize()
    is_mean, is_std = is_acc.finalize(splits=is_splits)
    return mu, cov, is_mean, is_std


def get_real_stats_cache_path(out_dir: Path, real_split: str, real_count: int) -> Path:
    return out_dir / f"real_stats_cifar10_{real_split}_{real_count}_inceptionv3.npz"


def load_or_compute_real_stats(
    *,
    out_dir: Path,
    extractor,
    root: str,
    real_split: str,
    real_batch: int,
    num_workers: int,
    real_max: int | None,
    real_seed: int,
    device: torch.device,
    download: bool,
):
    real_count = 50000 if real_max is None else int(real_max)
    cache_path = get_real_stats_cache_path(out_dir, real_split, real_count)

    if cache_path.exists():
        blob = np.load(cache_path)
        mu = torch.from_numpy(blob["mu"]).to(torch.float64)
        cov = torch.from_numpy(blob["cov"]).to(torch.float64)
        n_real = int(blob["n_real"])
        return mu, cov, n_real, cache_path

    loader, _ = make_cifar10_eval_loader(
        root=root,
        train=(real_split == "train"),
        batch_size=real_batch,
        num_workers=num_workers,
        real_max=real_max,
        seed=real_seed,
        download=download,
    )
    mu, cov, n_real = extract_real_stats(extractor, loader, device=device)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        mu=mu.cpu().numpy(),
        cov=cov.cpu().numpy(),
        n_real=np.array(n_real, dtype=np.int64),
    )
    return mu, cov, n_real, cache_path


def run_single_eval(
    *,
    sampler: str,
    steps: int,
    schedule: str,
    eta: float,
    diffusion: DDPMCIFAR,
    ddim: DDIMSampler,
    model: nn.Module,
    extractor,
    device: torch.device,
    runtime: dict,
    mu_r: torch.Tensor,
    cov_r: torch.Tensor,
    num_gen: int,
    gen_batch: int,
    is_splits: int,
    shape: tuple[int, int, int],
):
    mu_g, cov_g, is_mean, is_std = extract_gen_stats_and_is(
        sampler=sampler,
        diffusion=diffusion,
        ddim=ddim,
        model=model,
        extractor=extractor,
        device=device,
        runtime=runtime,
        num_gen=num_gen,
        gen_batch=gen_batch,
        is_splits=is_splits,
        steps=steps,
        schedule=schedule,
        eta=eta,
        shape=shape,
    )
    fid = frechet_distance(mu_r, cov_r, mu_g, cov_g)
    return {
        "sampler": sampler,
        "steps": int(steps),
        "schedule": schedule,
        "eta": float(eta),
        "fid": float(fid),
        "inception_score_mean": float(is_mean),
        "inception_score_std": float(is_std),
    }


def write_markdown_table(path: Path, rows: list[dict]) -> None:
    lines = [
        "| sampler | steps | eta | FID | IS |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        is_text = f"{row['inception_score_mean']:.3f} ± {row['inception_score_std']:.3f}"
        lines.append(
            f"| {row['sampler']} | {row['steps']} | {row['eta']:.1f} | {row['fid']:.3f} | {is_text} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sampler", "steps", "schedule", "eta", "fid", "inception_score_mean", "inception_score_std"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/cifar10_ddpm.yaml")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--use_ema", action="store_true")

    p.add_argument("--mode", type=str, default="single", choices=["single", "table"])
    p.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim", "ddpm_noisy"])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--schedule", type=str, default="quadratic", choices=["linear", "quadratic"])
    p.add_argument("--eta", type=float, default=0.0)

    p.add_argument("--table_steps", type=str, default="10,20,50,100,1000")
    p.add_argument("--table_etas", type=str, default="0.0,0.2,0.5,1.0")
    p.add_argument("--include_ddpm_noisy", action="store_true")

    p.add_argument("--num_gen", type=int, default=50000)
    p.add_argument("--gen_batch", type=int, default=256)
    p.add_argument("--is_splits", type=int, default=10)
    p.add_argument("--gen_seed", type=int, default=0)

    p.add_argument("--real_split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--real_batch", type=int, default=256)
    p.add_argument("--real_max", type=int, default=50000)
    p.add_argument("--real_seed", type=int, default=0)

    p.add_argument("--out", type=str, default="runs/cifar10_metrics/metrics.json")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(args.gen_seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = configure_runtime(cfg, device)

    print(
        "[runtime]",
        {
            "device": str(device),
            "mixed_precision_gen": runtime["mixed_precision"],
            "allow_tf32": runtime["allow_tf32"],
            "cudnn_benchmark": runtime["cudnn_benchmark"],
            "channels_last": runtime["channels_last"],
            "compile_enabled": runtime["compile_enabled"],
            "compile_mode": runtime["compile_mode"],
            "compile_backend": runtime["compile_backend"],
            "extractor_precision": "fp32",
        },
    )

    model = build_model_from_cfg(cfg).to(device)
    diffusion = build_diffusion_from_cfg(cfg).to(device)
    ddim = DDIMSampler(diffusion)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    if args.use_ema and "ema" in ckpt:
        ema = EMA(model, decay=float(cfg["training"].get("ema_decay", 0.9999)))
        ema.load_state_dict(ckpt["ema"])
        model = ema.make_ema_model(model).to(device)

    model.eval()
    if runtime["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    model_for_eval: nn.Module = InferenceModelWrapper(
        model,
        channels_last=runtime["channels_last"],
    ).to(device).eval()

    if runtime["compile_enabled"]:
        print("[runtime] compiling model for metric generation...")
        model_for_eval = torch.compile(
            model_for_eval,
            backend=runtime["compile_backend"],
            mode=runtime["compile_mode"],
        )

    extractor = InceptionV3FeatureExtractor(
        channels_last=runtime["channels_last"],
    ).to(device).eval()

    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_r, cov_r, n_real, real_stats_cache = load_or_compute_real_stats(
        out_dir=out_dir,
        extractor=extractor,
        root=cfg["data"]["root"],
        real_split=args.real_split,
        real_batch=int(args.real_batch),
        num_workers=int(cfg["data"]["num_workers"]),
        real_max=int(args.real_max) if int(args.real_max) > 0 else None,
        real_seed=int(args.real_seed),
        device=device,
        download=bool(cfg["data"].get("download", True)),
    )

    shape = tuple(int(v) for v in cfg["sampling"].get("shape", [3, 32, 32]))

    if args.mode == "single":
        rows = [
            run_single_eval(
                sampler=args.sampler,
                steps=int(args.steps),
                schedule=args.schedule,
                eta=float(args.eta),
                diffusion=diffusion,
                ddim=ddim,
                model=model_for_eval,
                extractor=extractor,
                device=device,
                runtime=runtime,
                mu_r=mu_r,
                cov_r=cov_r,
                num_gen=int(args.num_gen),
                gen_batch=int(args.gen_batch),
                is_splits=int(args.is_splits),
                shape=shape,
            )
        ]
    else:
        steps_list = [int(x.strip()) for x in args.table_steps.split(",") if x.strip()]
        eta_list = [float(x.strip()) for x in args.table_etas.split(",") if x.strip()]
        rows = []

        for steps in steps_list:
            for eta in eta_list:
                rows.append(
                    run_single_eval(
                        sampler="ddim",
                        steps=steps,
                        schedule=args.schedule,
                        eta=eta,
                        diffusion=diffusion,
                        ddim=ddim,
                        model=model_for_eval,
                        extractor=extractor,
                        device=device,
                        runtime=runtime,
                        mu_r=mu_r,
                        cov_r=cov_r,
                        num_gen=int(args.num_gen),
                        gen_batch=int(args.gen_batch),
                        is_splits=int(args.is_splits),
                        shape=shape,
                    )
                )

            if args.include_ddpm_noisy:
                rows.append(
                    run_single_eval(
                        sampler="ddpm_noisy",
                        steps=steps,
                        schedule=args.schedule,
                        eta=1.0,
                        diffusion=diffusion,
                        ddim=ddim,
                        model=model_for_eval,
                        extractor=extractor,
                        device=device,
                        runtime=runtime,
                        mu_r=mu_r,
                        cov_r=cov_r,
                        num_gen=int(args.num_gen),
                        gen_batch=int(args.gen_batch),
                        is_splits=int(args.is_splits),
                        shape=shape,
                    )
                )

    payload = {
        "dataset": "CIFAR10",
        "real_split": args.real_split,
        "n_real": int(n_real),
        "real_stats_cache": str(real_stats_cache),
        "num_gen": int(args.num_gen),
        "gen_batch": int(args.gen_batch),
        "is_splits": int(args.is_splits),
        "use_ema": bool(args.use_ema),
        "ckpt": args.ckpt,
        "mode": args.mode,
        "rows": rows,
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out_dir / (out_path.stem + ".csv"), rows)
    write_markdown_table(out_dir / (out_path.stem + ".md"), rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()