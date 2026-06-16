"""FID + Inception Score grid sweep for the Memory-DDIM sampler (CIFAR-10).

Port of the supervisor's Memory-DDIM sampler into the repo eval pipeline.
Same infrastructure as eval_cifar10_metrics_ddim_aug.py (streaming FID,
50K real-stats cache, resume, SIGUSR1 graceful exit, per-row JSON/CSV/MD),
but driving src/diffusion/ddim_memory.py.

Grid axes (comma-separated lists):
  shared : steps, eta, lambda_f2, lambda_h, lambda_beta, lambda_r
  mode=barycenter : lambda_d, lambda_e
  mode=c0         : lambda_h2

lambda_f2=0 reduces the sampler EXACTLY to DDIM (no t=0 model call), so it
is the fair DDIM baseline measured in the same pipeline. Keep it in the grid.

PSD is guaranteed by construction for interior barycenter weights; we keep
a scalar pre-check and skip any config that still violates PSD.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from src.diffusion.ddim_memory import (
    MemoryDDIMConfig,
    MemoryDDIMSampler,
    count_psd_violations,
)
from src.diffusion.ddpm_cifar import DDPMCIFAR
from src.metrics.inception_cifar import InceptionScoreAccumulator, InceptionV3FeatureExtractor
from src.metrics.stats import StreamingMeanCov, frechet_distance
from src.models.unet_cifar10 import UNetCIFAR10, UNetCIFAR10Config
from src.utils import EMA, load_yaml, set_seed


# ---------------------------------------------------------------------------
# Runtime helpers (identical to the augmented eval; sampler-agnostic)
# ---------------------------------------------------------------------------


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
                print("[runtime] BF16 requested but not supported -> fallback FP16")
                mixed_precision = "fp16"
    else:
        mixed_precision = "none"
        compile_enabled = False
        channels_last = False

    if compile_enabled and not hasattr(torch, "compile"):
        print("[runtime] torch.compile unavailable -> disabled")
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


# ---------------------------------------------------------------------------
# Real-stats cache (identical format to the augmented/standard eval)
# ---------------------------------------------------------------------------


def make_cifar10_eval_loader(root, train, batch_size, num_workers, real_max, seed, download=True):
    ds = CIFAR10(
        root=root, train=train, download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]),
    )
    if real_max is not None and 0 < real_max < len(ds):
        g = torch.Generator(); g.manual_seed(int(seed))
        idx = torch.randperm(len(ds), generator=g)[: int(real_max)].tolist()
        ds = Subset(ds, idx)
    loader_kwargs = dict(dataset=ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(**loader_kwargs), ds


@torch.inference_mode()
def extract_real_stats(extractor, loader, device):
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


def get_real_stats_cache_path(out_dir, real_split, real_count):
    return out_dir / f"real_stats_cifar10_{real_split}_{real_count}_inceptionv3.npz"


def load_or_compute_real_stats(*, out_dir, extractor, root, real_split, real_batch,
                               num_workers, real_max, real_seed, device, download):
    real_count = 50000 if real_max is None else int(real_max)
    cache_path = get_real_stats_cache_path(out_dir, real_split, real_count)
    if cache_path.exists():
        blob = np.load(cache_path)
        mu = torch.from_numpy(blob["mu"]).to(torch.float64)
        cov = torch.from_numpy(blob["cov"]).to(torch.float64)
        return mu, cov, int(blob["n_real"]), cache_path
    loader, _ = make_cifar10_eval_loader(
        root=root, train=(real_split == "train"), batch_size=real_batch,
        num_workers=num_workers, real_max=real_max, seed=real_seed, download=download)
    mu, cov, n_real = extract_real_stats(extractor, loader, device=device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, mu=mu.cpu().numpy(), cov=cov.cpu().numpy(),
             n_real=np.array(n_real, dtype=np.int64))
    return mu, cov, n_real, cache_path


# ---------------------------------------------------------------------------
# Sampling + feature extraction for one config
# ---------------------------------------------------------------------------


@torch.inference_mode()
def extract_gen_stats_and_is(*, sampler, model, extractor, device, runtime,
                             num_gen, gen_batch, is_splits, mem_cfg, shape, progress_desc):
    model.eval()
    extractor.eval()
    stats = StreamingMeanCov(dim=2048)
    is_acc = InceptionScoreAccumulator()
    remaining = int(num_gen)
    pbar = tqdm(total=int(num_gen), desc=progress_desc, leave=False)
    while remaining > 0:
        b = min(int(gen_batch), remaining)
        with make_autocast_context(runtime, device):
            imgs = sampler.sample(model, batch_size=b, device=device, shape=shape, cfg=mem_cfg)
        imgs = imgs.float()
        logits, feat = extractor(imgs)
        stats.update(feat)
        is_acc.update_logits(logits)
        remaining -= b
        pbar.update(b)
    pbar.close()
    mu, cov = stats.finalize()
    is_mean, is_std = is_acc.finalize(splits=is_splits)
    return mu, cov, is_mean, is_std


def run_single_eval(*, mem_cfg, sampler, model, extractor, device, runtime,
                    mu_r, cov_r, num_gen, gen_batch, is_splits, shape, abar, T, progress_desc):
    psd_count = count_psd_violations(mem_cfg, abar, T)
    mu_g, cov_g, is_mean, is_std = extract_gen_stats_and_is(
        sampler=sampler, model=model, extractor=extractor, device=device, runtime=runtime,
        num_gen=num_gen, gen_batch=gen_batch, is_splits=is_splits, mem_cfg=mem_cfg,
        shape=shape, progress_desc=progress_desc)
    fid = frechet_distance(mu_r, cov_r, mu_g, cov_g)
    return {
        "sampler": "ddim_memory",
        "mode": mem_cfg.mode,
        "steps": int(mem_cfg.steps),
        "schedule": mem_cfg.schedule,
        "eta": float(mem_cfg.eta),
        "lambda_f2": float(mem_cfg.lambda_f2),
        "lambda_h": float(mem_cfg.lambda_h),
        "lambda_beta": float(mem_cfg.lambda_beta),
        "lambda_r": float(mem_cfg.lambda_r),
        "lambda_d": float(mem_cfg.lambda_d),
        "lambda_e": float(mem_cfg.lambda_e),
        "lambda_h2": float(mem_cfg.lambda_h2),
        "psd_violations": int(psd_count),
        "fid": float(fid),
        "inception_score_mean": float(is_mean),
        "inception_score_std": float(is_std),
    }


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


CSV_FIELDS = [
    "sampler", "mode", "steps", "schedule", "eta",
    "lambda_f2", "lambda_h", "lambda_beta", "lambda_r",
    "lambda_d", "lambda_e", "lambda_h2",
    "psd_violations", "fid", "inception_score_mean", "inception_score_std",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_existing_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    rows: list[dict] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            try:
                rows.append({
                    "sampler": str(raw["sampler"]),
                    "mode": str(raw["mode"]),
                    "steps": int(raw["steps"]),
                    "schedule": str(raw["schedule"]),
                    "eta": float(raw["eta"]),
                    "lambda_f2": float(raw["lambda_f2"]),
                    "lambda_h": float(raw["lambda_h"]),
                    "lambda_beta": float(raw["lambda_beta"]),
                    "lambda_r": float(raw["lambda_r"]),
                    "lambda_d": float(raw["lambda_d"]),
                    "lambda_e": float(raw["lambda_e"]),
                    "lambda_h2": float(raw["lambda_h2"]),
                    "psd_violations": int(raw["psd_violations"]),
                    "fid": float(raw["fid"]),
                    "inception_score_mean": float(raw["inception_score_mean"]),
                    "inception_score_std": float(raw["inception_score_std"]),
                })
            except (KeyError, ValueError) as exc:
                print(f"[resume] skipping malformed CSV row: {exc}")
                continue
    return rows


def row_key(row_or_combo, mode: str):
    """Identity of a config for resume/dedup. Mode-dependent."""
    if isinstance(row_or_combo, dict):
        steps = int(row_or_combo["steps"]); eta = float(row_or_combo["eta"])
        lf2 = float(row_or_combo["lambda_f2"]); lh = float(row_or_combo["lambda_h"])
        lb = float(row_or_combo["lambda_beta"]); lr = float(row_or_combo["lambda_r"])
        ld = float(row_or_combo["lambda_d"]); le = float(row_or_combo["lambda_e"])
        lh2 = float(row_or_combo["lambda_h2"]); m = str(row_or_combo["mode"])
    else:
        steps, eta, lf2, lh, lb, lr, ld, le, lh2 = row_or_combo
        m = mode
    if m == "c0":
        return (m, int(steps), float(eta), float(lf2), float(lh), float(lb), float(lr), float(lh2))
    return (m, int(steps), float(eta), float(lf2), float(lh), float(lb), float(lr), float(ld), float(le))


def write_markdown_table(path: Path, rows: list[dict]) -> None:
    header = "| mode | steps | eta | lf2 | lh | lb | lr | ld | le | lh2 | psd! | FID | IS |"
    sep = "|" + "---:|" * 13
    lines = [header, sep]
    rows_sorted = sorted(rows, key=lambda r: (
        r["mode"], r["steps"], r["eta"], r["lambda_f2"],
        r["lambda_h"], r["lambda_beta"], r["lambda_r"],
        r["lambda_d"], r["lambda_e"], r["lambda_h2"]))
    for r in rows_sorted:
        is_text = f"{r['inception_score_mean']:.3f}\u00b1{r['inception_score_std']:.3f}"
        lines.append(
            f"| {r['mode']} | {r['steps']} | {r['eta']:.2f} | {r['lambda_f2']:.3g} "
            f"| {r['lambda_h']:.3g} | {r['lambda_beta']:.3g} | {r['lambda_r']:.3g} "
            f"| {r['lambda_d']:.3g} | {r['lambda_e']:.3g} | {r['lambda_h2']:.3g} "
            f"| {r['psd_violations']} | {r['fid']:.3f} | {is_text} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Graceful interruption
# ---------------------------------------------------------------------------


class _InterruptHandler:
    def __init__(self):
        self.stop = False
        self.received_signal = None
        signal.signal(signal.SIGUSR1, self._handle)
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):
        self.stop = True
        self.received_signal = signum


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/cifar10_ddpm.yaml")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--schedule", type=str, default="quadratic", choices=["linear", "quadratic"])

    p.add_argument("--mode", type=str, default="c0", choices=["barycenter", "c0"])

    p.add_argument("--table_steps", type=str, default="10,20,50,100")
    p.add_argument("--table_etas", type=str, default="0.0,0.2,0.5,1.0")
    p.add_argument("--table_lambda_f2", type=str, default="0.0,0.3,0.6,0.9")
    p.add_argument("--table_lambda_h", type=str, default="0.5,1.0")
    p.add_argument("--table_lambda_beta", type=str, default="0.4")
    p.add_argument("--table_lambda_r", type=str, default="0.1")
    p.add_argument("--table_lambda_d", type=str, default="0.5")
    p.add_argument("--table_lambda_e", type=str, default="0.5")
    p.add_argument("--table_lambda_h2", type=str, default="0.5")

    p.add_argument("--skip_invalid", action="store_true")
    p.add_argument("--skip_invalid_threshold", type=int, default=0)

    p.add_argument("--num_gen", type=int, default=3000)
    p.add_argument("--gen_batch", type=int, default=256)
    p.add_argument("--is_splits", type=int, default=10)
    p.add_argument("--gen_seed", type=int, default=0)

    p.add_argument("--real_split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--real_batch", type=int, default=256)
    p.add_argument("--real_max", type=int, default=50000)
    p.add_argument("--real_seed", type=int, default=0)

    p.add_argument("--out", type=str, default="runs/cifar10_metrics_ddim_memory/metrics.json")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--time_budget_seconds", type=float, default=0.0)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(args.gen_seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = configure_runtime(cfg, device)
    print("[runtime]", {"device": str(device), "mp": runtime["mixed_precision"],
                        "channels_last": runtime["channels_last"],
                        "compile": runtime["compile_enabled"], "mode": args.mode})

    model = build_model_from_cfg(cfg).to(device)
    diffusion = build_diffusion_from_cfg(cfg).to(device)
    sampler = MemoryDDIMSampler(diffusion)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if args.use_ema and "ema" in ckpt:
        ema = EMA(model, decay=float(cfg["training"].get("ema_decay", 0.9999)))
        ema.load_state_dict(ckpt["ema"])
        model = ema.make_ema_model(model).to(device)
    model.eval()
    if runtime["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    model_for_eval = InferenceModelWrapper(model, channels_last=runtime["channels_last"]).to(device).eval()
    if runtime["compile_enabled"]:
        print("[runtime] compiling model...")
        model_for_eval = torch.compile(model_for_eval, backend=runtime["compile_backend"],
                                       mode=runtime["compile_mode"])

    extractor = InceptionV3FeatureExtractor(channels_last=runtime["channels_last"]).to(device).eval()

    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_r, cov_r, n_real, real_stats_cache = load_or_compute_real_stats(
        out_dir=out_dir, extractor=extractor, root=cfg["data"]["root"],
        real_split=args.real_split, real_batch=int(args.real_batch),
        num_workers=int(cfg["data"]["num_workers"]),
        real_max=int(args.real_max) if int(args.real_max) > 0 else None,
        real_seed=int(args.real_seed), device=device,
        download=bool(cfg["data"].get("download", True)))

    shape = tuple(int(v) for v in cfg["sampling"].get("shape", [3, 32, 32]))
    abar = diffusion.alphas_cumprod
    T = int(diffusion.timesteps)

    steps_list = parse_int_list(args.table_steps)
    etas = parse_float_list(args.table_etas)
    lf2_list = parse_float_list(args.table_lambda_f2)
    lh_list = parse_float_list(args.table_lambda_h)
    lb_list = parse_float_list(args.table_lambda_beta)
    lr_list = parse_float_list(args.table_lambda_r)
    ld_list = parse_float_list(args.table_lambda_d)
    le_list = parse_float_list(args.table_lambda_e)
    lh2_list = parse_float_list(args.table_lambda_h2)

    # Mode-dependent: don't multiply by irrelevant axes
    if args.mode == "c0":
        ld_list, le_list = [0.5], [0.5]
    else:
        lh2_list = [0.5]

    combos = list(itertools.product(
        steps_list, etas, lf2_list, lh_list, lb_list, lr_list, ld_list, le_list, lh2_list))
    total = len(combos)
    print(f"[grid] mode={args.mode} total configurations: {total}")

    csv_partial = out_dir / (out_path.stem + ".csv")
    md_path = out_dir / (out_path.stem + ".md")

    rows: list[dict] = []
    done_keys: set = set()
    if args.resume:
        existing = read_existing_rows(csv_partial)
        if existing:
            rows = existing
            done_keys = {row_key(r, args.mode) for r in rows}
            print(f"[resume] {len(existing)} rows already done at {csv_partial}; skipping them.")
    elif csv_partial.exists():
        print(f"[resume] --no_resume: overwriting {csv_partial}.")

    interrupt = _InterruptHandler()
    wall_start = time.monotonic()
    time_budget = float(args.time_budget_seconds)
    skipped: list[dict] = []

    def _payload():
        return {
            "dataset": "CIFAR10", "sampler": "ddim_memory", "mode": args.mode,
            "schedule": args.schedule, "real_split": args.real_split, "n_real": int(n_real),
            "real_stats_cache": str(real_stats_cache), "num_gen": int(args.num_gen),
            "gen_batch": int(args.gen_batch), "is_splits": int(args.is_splits),
            "use_ema": bool(args.use_ema), "ckpt": args.ckpt,
            "grid": {"steps": steps_list, "etas": etas, "lambda_f2": lf2_list,
                     "lambda_h": lh_list, "lambda_beta": lb_list, "lambda_r": lr_list,
                     "lambda_d": ld_list, "lambda_e": le_list, "lambda_h2": lh2_list},
            "n_configs_total": total, "n_configs_run": len(rows),
            "n_configs_skipped": len(skipped), "skipped_configs": skipped,
            "elapsed_seconds": round(time.monotonic() - wall_start, 1), "rows": rows,
        }

    def _write_all():
        write_csv(csv_partial, rows)
        out_path.write_text(json.dumps(_payload(), indent=2), encoding="utf-8")
        write_markdown_table(md_path, rows)

    early_exit_reason = None
    pbar = tqdm(combos, total=total, desc="grid")
    for k, combo in enumerate(pbar):
        steps, eta, lf2, lh, lb, lr, ld, le, lh2 = combo
        if row_key(combo, args.mode) in done_keys:
            pbar.set_postfix_str("done (resume)")
            continue

        mem_cfg = MemoryDDIMConfig(
            steps=int(steps), schedule=args.schedule, eta=float(eta),
            lambda_f2=float(lf2), lambda_h=float(lh), lambda_beta=float(lb),
            lambda_r=float(lr), mode=args.mode,
            lambda_d=float(ld), lambda_e=float(le), lambda_h2=float(lh2),
            clip_denoised=True, verbose_pd_warnings=False)

        psd_count = count_psd_violations(mem_cfg, abar, T)
        if args.skip_invalid and psd_count > int(args.skip_invalid_threshold):
            skipped.append({"steps": int(steps), "eta": float(eta), "lambda_f2": float(lf2),
                            "lambda_h": float(lh), "lambda_beta": float(lb), "lambda_r": float(lr),
                            "lambda_d": float(ld), "lambda_e": float(le), "lambda_h2": float(lh2),
                            "psd_violations": int(psd_count)})
            pbar.set_postfix_str(f"skip ({psd_count} non-PSD)")
            continue

        desc = (f"[{k+1}/{total}] {args.mode} S={steps} eta={eta:.2f} "
                f"lf2={lf2:.2g} lh={lh:.2g}")
        row = run_single_eval(
            mem_cfg=mem_cfg, sampler=sampler, model=model_for_eval, extractor=extractor,
            device=device, runtime=runtime, mu_r=mu_r, cov_r=cov_r,
            num_gen=int(args.num_gen), gen_batch=int(args.gen_batch),
            is_splits=int(args.is_splits), shape=shape, abar=abar, T=T, progress_desc=desc)
        rows.append(row)
        done_keys.add(row_key(row, args.mode))
        pbar.set_postfix_str(f"FID={row['fid']:.2f} IS={row['inception_score_mean']:.2f}")
        _write_all()

        if interrupt.stop:
            early_exit_reason = f"signal {interrupt.received_signal} -> graceful exit"
            break
        if time_budget > 0 and (time.monotonic() - wall_start) >= time_budget:
            early_exit_reason = "time budget exhausted -> graceful exit"
            break
    pbar.close()
    _write_all()

    if early_exit_reason is not None:
        print(f"\n[early-exit] {early_exit_reason}")
        print(f"[early-exit] {len(rows)}/{total} done. Relaunch (resume on) to continue.")

    if rows:
        print("\n[summary] top-5 lowest FID:")
        for r in sorted(rows, key=lambda r: r["fid"])[:5]:
            print(f"  FID={r['fid']:.3f} IS={r['inception_score_mean']:.3f} "
                  f"mode={r['mode']} S={r['steps']} eta={r['eta']} lf2={r['lambda_f2']} "
                  f"lh={r['lambda_h']} ld={r['lambda_d']} le={r['lambda_e']} lh2={r['lambda_h2']}")
    print(f"\n[done] wrote {out_path}, {csv_partial}, {md_path}")


if __name__ == "__main__":
    main()