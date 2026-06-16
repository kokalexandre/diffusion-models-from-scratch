"""FID + Inception Score grid sweep for the augmented DDIM sampler (CIFAR-10).

Mirrors tools/eval_cifar10_metrics.py but:
  - uses DDIMAugmentedSampler from src/diffusion/ddim_augmented.py;
  - iterates the FULL cartesian product over (steps, eta, lambda_h,
    lambda_beta, lambda_r, lambda_c, lambda_h2, rho_g), each given as
    a comma-separated list;
  - pre-computes a PSD-violation count per config (scalar-only, fast)
    so invalid configs are surfaced in the output table;
  - reuses the same real-stats cache file format as the original script
    (real_stats_cifar10_<split>_<count>_inceptionv3.npz);
  - writes JSON + CSV + Markdown table, with the CSV refreshed after
    every row so partial results survive a job timeout;
  - supports graceful interruption: on SIGUSR1/SIGTERM/SIGINT it
    finishes the current eval, writes all outputs, then exits cleanly;
  - supports auto-resume: if the target CSV already exists, already
    completed configs are loaded and skipped on relaunch.

The model is loaded once, the Inception V3 extractor is built once,
and the real CIFAR-10 stats are computed (or loaded from cache) once.
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

from src.diffusion.ddim_augmented import (
    DDIMAugmentedConfig,
    DDIMAugmentedSampler,
    _compute_step_params,
)
from src.diffusion.ddim import _make_timesteps
from src.diffusion.ddpm_cifar import DDPMCIFAR
from src.metrics.inception_cifar import InceptionScoreAccumulator, InceptionV3FeatureExtractor
from src.metrics.stats import StreamingMeanCov, frechet_distance
from src.models.unet_cifar10 import UNetCIFAR10, UNetCIFAR10Config
from src.utils import EMA, load_yaml, set_seed


# ---------------------------------------------------------------------------
# Runtime helpers (kept aligned with the existing eval script)
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


# ---------------------------------------------------------------------------
# Real-stats cache (identical format to eval_cifar10_metrics.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PSD-violation pre-check (fast, scalar-only, no GPU)
# ---------------------------------------------------------------------------


def count_psd_violations(
    aug_cfg: DDIMAugmentedConfig,
    abar: torch.Tensor,
    T: int,
) -> int:
    """Return the number of trajectory steps where Sigma_t fails PSD.

    Pure scalar computation: doesn't run the network. Useful to filter
    configurations before launching expensive sampling.
    """
    ts = _make_timesteps(T=T, S=int(aug_cfg.steps), schedule=aug_cfg.schedule)
    abar = abar.detach().cpu()
    tol = aug_cfg.pd_tol
    count = 0
    for j in range(len(ts) - 1, 0, -1):
        t = int(ts[j].item())
        t_prev = int(ts[j - 1].item())
        a_t = float(abar[t].item())
        a_prev = float(abar[t_prev].item())
        p = _compute_step_params(a_t, a_prev, aug_cfg)
        det = p["sigma11"] * p["sigma22"] - p["sigma12"] ** 2
        if (
            p["sigma11"] < -tol
            or p["sigma22"] < -tol
            or det < -tol
        ):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Sampling + feature extraction for a single config
# ---------------------------------------------------------------------------


@torch.inference_mode()
def extract_gen_stats_and_is_aug(
    *,
    sampler_aug: DDIMAugmentedSampler,
    model: nn.Module,
    extractor: InceptionV3FeatureExtractor,
    device: torch.device,
    runtime: dict,
    num_gen: int,
    gen_batch: int,
    is_splits: int,
    aug_cfg: DDIMAugmentedConfig,
    shape: tuple[int, int, int],
    progress_desc: str,
):
    model.eval()
    extractor.eval()

    stats = StreamingMeanCov(dim=2048)
    is_acc = InceptionScoreAccumulator()

    remaining = int(num_gen)
    pbar = tqdm(total=int(num_gen), desc=progress_desc, leave=False)
    while remaining > 0:
        b = min(int(gen_batch), remaining)
        with make_autocast_context(runtime, device):
            imgs = sampler_aug.sample(
                model,
                batch_size=b,
                device=device,
                shape=shape,
                cfg=aug_cfg,
            )
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


def run_single_eval_aug(
    *,
    aug_cfg: DDIMAugmentedConfig,
    sampler_aug: DDIMAugmentedSampler,
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
    abar: torch.Tensor,
    T: int,
    progress_desc: str,
) -> dict:
    psd_count = count_psd_violations(aug_cfg, abar, T)
    mu_g, cov_g, is_mean, is_std = extract_gen_stats_and_is_aug(
        sampler_aug=sampler_aug,
        model=model,
        extractor=extractor,
        device=device,
        runtime=runtime,
        num_gen=num_gen,
        gen_batch=gen_batch,
        is_splits=is_splits,
        aug_cfg=aug_cfg,
        shape=shape,
        progress_desc=progress_desc,
    )
    fid = frechet_distance(mu_r, cov_r, mu_g, cov_g)
    return {
        "sampler": "ddim_aug",
        "steps": int(aug_cfg.steps),
        "schedule": aug_cfg.schedule,
        "eta": float(aug_cfg.eta),
        "lambda_h": float(aug_cfg.lambda_h),
        "lambda_beta": float(aug_cfg.lambda_beta),
        "lambda_r": float(aug_cfg.lambda_r),
        "lambda_c": float(aug_cfg.lambda_c),
        "lambda_h2": float(aug_cfg.lambda_h2),
        "rho_g": float(aug_cfg.rho_g),
        "psd_violations": int(psd_count),
        "fid": float(fid),
        "inception_score_mean": float(is_mean),
        "inception_score_std": float(is_std),
    }


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


CSV_FIELDS = [
    "sampler",
    "steps",
    "schedule",
    "eta",
    "lambda_h",
    "lambda_beta",
    "lambda_r",
    "lambda_c",
    "lambda_h2",
    "rho_g",
    "psd_violations",
    "fid",
    "inception_score_mean",
    "inception_score_std",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_existing_rows(csv_path: Path) -> list[dict]:
    """Re-read rows from a partial CSV (same format as write_csv).

    Used for auto-resume: when the script is relaunched and finds a CSV
    at the target path, it re-loads the rows so already-evaluated configs
    can be skipped and final outputs include them.
    """
    if not csv_path.exists():
        return []
    rows: list[dict] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            try:
                rows.append(
                    {
                        "sampler": str(raw["sampler"]),
                        "steps": int(raw["steps"]),
                        "schedule": str(raw["schedule"]),
                        "eta": float(raw["eta"]),
                        "lambda_h": float(raw["lambda_h"]),
                        "lambda_beta": float(raw["lambda_beta"]),
                        "lambda_r": float(raw["lambda_r"]),
                        "lambda_c": float(raw["lambda_c"]),
                        "lambda_h2": float(raw["lambda_h2"]),
                        "rho_g": float(raw["rho_g"]),
                        "psd_violations": int(raw["psd_violations"]),
                        "fid": float(raw["fid"]),
                        "inception_score_mean": float(raw["inception_score_mean"]),
                        "inception_score_std": float(raw["inception_score_std"]),
                    }
                )
            except (KeyError, ValueError) as exc:
                # Malformed line (corrupted partial write) -> ignore it
                print(f"[resume] skipping malformed CSV row: {exc}")
                continue
    return rows


def row_key(row_or_combo) -> tuple:
    """Identifier used to detect "already done" configs across runs.

    Accepts either a dict (loaded from CSV) or a tuple
    (steps, eta, lh, lb, lr, lc, lh2, rho).
    """
    if isinstance(row_or_combo, dict):
        return (
            int(row_or_combo["steps"]),
            float(row_or_combo["eta"]),
            float(row_or_combo["lambda_h"]),
            float(row_or_combo["lambda_beta"]),
            float(row_or_combo["lambda_r"]),
            float(row_or_combo["lambda_c"]),
            float(row_or_combo["lambda_h2"]),
            float(row_or_combo["rho_g"]),
        )
    steps, eta, lh, lb, lr, lc, lh2, rho = row_or_combo
    return (
        int(steps),
        float(eta),
        float(lh),
        float(lb),
        float(lr),
        float(lc),
        float(lh2),
        float(rho),
    )


# ---------------------------------------------------------------------------
# Graceful interruption (SLURM wall-time + Ctrl-C)
# ---------------------------------------------------------------------------


class _InterruptHandler:
    """Sets a stop flag on SIGUSR1 / SIGTERM / SIGINT.

    SLURM sends SIGUSR1 a configurable amount of time before the wall (via
    `#SBATCH --signal=B:SIGUSR1@<seconds>` in the sbatch). When the flag
    fires the main loop finishes the current evaluation and exits cleanly,
    so partial JSON / MD outputs are written and the CSV stays consistent.
    """

    def __init__(self):
        self.stop = False
        self.received_signal: int | None = None
        signal.signal(signal.SIGUSR1, self._handle)
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):
        # Called from the signal-handler context: keep it minimal.
        self.stop = True
        self.received_signal = signum


def write_markdown_table(path: Path, rows: list[dict]) -> None:
    header = (
        "| steps | eta | lh | lb | lr | lc | lh2 | rho_g | psd! | FID | IS |"
    )
    sep = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    # Sort: steps asc, eta asc, then other hypers for readability
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r["steps"],
            r["eta"],
            r["lambda_h"],
            r["lambda_beta"],
            r["lambda_r"],
            r["lambda_c"],
            r["lambda_h2"],
            r["rho_g"],
        ),
    )
    for row in rows_sorted:
        is_text = f"{row['inception_score_mean']:.3f}\u00b1{row['inception_score_std']:.3f}"
        lines.append(
            f"| {row['steps']} "
            f"| {row['eta']:.2f} "
            f"| {row['lambda_h']:.3g} "
            f"| {row['lambda_beta']:.3g} "
            f"| {row['lambda_r']:.3g} "
            f"| {row['lambda_c']:.3g} "
            f"| {row['lambda_h2']:.3g} "
            f"| {row['rho_g']:.3g} "
            f"| {row['psd_violations']} "
            f"| {row['fid']:.3f} "
            f"| {is_text} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI list parsers
# ---------------------------------------------------------------------------


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/cifar10_ddpm.yaml")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--use_ema", action="store_true")

    # Grid axes (comma-separated)
    p.add_argument("--table_steps", type=str, default="10,20,50,100")
    p.add_argument("--table_etas", type=str, default="0.0,0.2,0.5,1.0")
    p.add_argument("--table_lambda_h", type=str, default="0.3,0.7")
    p.add_argument("--table_lambda_beta", type=str, default="0.3,0.7")
    p.add_argument("--table_lambda_r", type=str, default="0.02,0.1")
    p.add_argument("--table_lambda_c", type=str, default="0.0,0.005")
    p.add_argument("--table_lambda_h2", type=str, default="0.0,0.02")
    p.add_argument("--table_rho_g", type=str, default="0.3,0.7")
    p.add_argument("--schedule", type=str, default="quadratic", choices=["linear", "quadratic"])

    # Behaviour
    p.add_argument(
        "--skip_invalid",
        action="store_true",
        help="Skip configs where psd_violations > skip_invalid_threshold.",
    )
    p.add_argument("--skip_invalid_threshold", type=int, default=0)

    # Generation/evaluation budget
    p.add_argument("--num_gen", type=int, default=1000)
    p.add_argument("--gen_batch", type=int, default=256)
    p.add_argument("--is_splits", type=int, default=10)
    p.add_argument("--gen_seed", type=int, default=0)

    # Real stats
    p.add_argument("--real_split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--real_batch", type=int, default=256)
    p.add_argument(
        "--real_max",
        type=int,
        default=50000,
        help="Use 50000 to keep FID comparable to standard CIFAR-10 reports.",
    )
    p.add_argument("--real_seed", type=int, default=0)

    p.add_argument("--out", type=str, default="runs/cifar10_metrics_ddim_aug/metrics.json")

    # Resume + graceful interrupt
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="If the target CSV already exists, skip configs already done. "
             "Enabled by default; use --no_resume to disable.",
    )
    p.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        help="Disable auto-resume: ignore any existing CSV and overwrite.",
    )
    p.add_argument(
        "--time_budget_seconds",
        type=float,
        default=0.0,
        help="If > 0, the script will exit cleanly after the current eval "
             "once this many seconds of wallclock have elapsed since startup. "
             "Use a value slightly under the SLURM wall (e.g. 35400 for 10h). "
             "Defaults to 0 = no time check (rely on SIGUSR1/SIGTERM only).",
    )

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

    # ---- Model + diffusion + samplers (one shot) ----
    model = build_model_from_cfg(cfg).to(device)
    diffusion = build_diffusion_from_cfg(cfg).to(device)
    sampler_aug = DDIMAugmentedSampler(diffusion)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    if args.use_ema and "ema" in ckpt:
        ema = EMA(model, decay=float(cfg["training"].get("ema_decay", 0.9999)))
        ema.load_state_dict(ckpt["ema"])
        model = ema.make_ema_model(model).to(device)

    model.eval()
    if runtime["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    model_for_eval: nn.Module = (
        InferenceModelWrapper(model, channels_last=runtime["channels_last"]).to(device).eval()
    )

    if runtime["compile_enabled"]:
        print("[runtime] compiling model for metric generation...")
        model_for_eval = torch.compile(
            model_for_eval,
            backend=runtime["compile_backend"],
            mode=runtime["compile_mode"],
        )

    extractor = (
        InceptionV3FeatureExtractor(channels_last=runtime["channels_last"]).to(device).eval()
    )

    # ---- Real stats (cached on disk) ----
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
    abar = diffusion.alphas_cumprod
    T = int(diffusion.timesteps)

    # ---- Grid expansion ----
    steps_list = parse_int_list(args.table_steps)
    etas = parse_float_list(args.table_etas)
    lh_list = parse_float_list(args.table_lambda_h)
    lb_list = parse_float_list(args.table_lambda_beta)
    lr_list = parse_float_list(args.table_lambda_r)
    lc_list = parse_float_list(args.table_lambda_c)
    lh2_list = parse_float_list(args.table_lambda_h2)
    rho_list = parse_float_list(args.table_rho_g)

    combos = list(
        itertools.product(steps_list, etas, lh_list, lb_list, lr_list, lc_list, lh2_list, rho_list)
    )
    total = len(combos)
    print(f"[grid] total configurations: {total}")
    print(
        f"[grid] axes: steps={steps_list} eta={etas} "
        f"lh={lh_list} lb={lb_list} lr={lr_list} lc={lc_list} "
        f"lh2={lh2_list} rho_g={rho_list}"
    )

    # ---- Resume: load any rows already in the target CSV ----
    csv_partial = out_dir / (out_path.stem + ".csv")
    md_path = out_dir / (out_path.stem + ".md")

    rows: list[dict] = []
    done_keys: set = set()
    if args.resume:
        existing = read_existing_rows(csv_partial)
        if existing:
            rows = existing
            done_keys = {row_key(r) for r in rows}
            print(
                f"[resume] found {len(existing)} previously completed rows "
                f"at {csv_partial}; they will be skipped."
            )
    elif csv_partial.exists():
        print(
            f"[resume] --no_resume set: overwriting {csv_partial} "
            "(existing rows will be lost)."
        )

    # ---- Signal handler + wallclock budget ----
    interrupt = _InterruptHandler()
    wall_start = time.monotonic()
    time_budget = float(args.time_budget_seconds)
    if time_budget > 0:
        print(
            f"[time] soft budget: {time_budget:.0f}s ; the loop will exit "
            "after the current eval once this elapses."
        )

    def _full_payload() -> dict:
        return {
            "dataset": "CIFAR10",
            "sampler": "ddim_aug",
            "schedule": args.schedule,
            "real_split": args.real_split,
            "n_real": int(n_real),
            "real_stats_cache": str(real_stats_cache),
            "num_gen": int(args.num_gen),
            "gen_batch": int(args.gen_batch),
            "is_splits": int(args.is_splits),
            "use_ema": bool(args.use_ema),
            "ckpt": args.ckpt,
            "grid": {
                "steps": steps_list,
                "etas": etas,
                "lambda_h": lh_list,
                "lambda_beta": lb_list,
                "lambda_r": lr_list,
                "lambda_c": lc_list,
                "lambda_h2": lh2_list,
                "rho_g": rho_list,
            },
            "n_configs_total": total,
            "n_configs_run": len(rows),
            "n_configs_skipped": len(skipped),
            "skipped_configs": skipped,
            "elapsed_seconds": round(time.monotonic() - wall_start, 1),
            "rows": rows,
        }

    def _write_all_outputs() -> None:
        write_csv(csv_partial, rows)
        out_path.write_text(json.dumps(_full_payload(), indent=2), encoding="utf-8")
        write_markdown_table(md_path, rows)

    # ---- Loop ----
    skipped: list[dict] = []
    early_exit_reason: str | None = None

    pbar = tqdm(combos, total=total, desc="grid")
    for k, combo in enumerate(pbar):
        steps, eta, lh, lb, lr, lc, lh2, rho = combo

        # Already done in a previous run? (resume)
        if row_key(combo) in done_keys:
            pbar.set_postfix_str("done (resume)")
            continue

        aug_cfg = DDIMAugmentedConfig(
            steps=int(steps),
            schedule=args.schedule,
            eta=float(eta),
            lambda_h=float(lh),
            lambda_beta=float(lb),
            lambda_r=float(lr),
            lambda_c=float(lc),
            lambda_h2=float(lh2),
            rho_g=float(rho),
            clip_denoised=True,
            verbose_pd_warnings=False,
        )

        # Pre-check PSD validity without touching the GPU
        psd_count = count_psd_violations(aug_cfg, abar, T)
        if args.skip_invalid and psd_count > int(args.skip_invalid_threshold):
            skipped.append(
                {
                    "steps": int(steps), "eta": float(eta),
                    "lambda_h": float(lh), "lambda_beta": float(lb),
                    "lambda_r": float(lr), "lambda_c": float(lc),
                    "lambda_h2": float(lh2), "rho_g": float(rho),
                    "psd_violations": int(psd_count),
                }
            )
            pbar.set_postfix_str(f"skip ({psd_count} non-PSD)")
            continue

        desc = (
            f"[{k + 1}/{total}] S={steps} eta={eta:.2f} "
            f"rho={rho:.2g} lh={lh:.2g} lb={lb:.2g} lr={lr:.2g} "
            f"lc={lc:.2g} lh2={lh2:.2g}"
        )

        row = run_single_eval_aug(
            aug_cfg=aug_cfg,
            sampler_aug=sampler_aug,
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
            abar=abar,
            T=T,
            progress_desc=desc,
        )
        rows.append(row)
        done_keys.add(row_key(row))

        pbar.set_postfix_str(
            f"FID={row['fid']:.2f} IS={row['inception_score_mean']:.2f} "
            f"psd!={row['psd_violations']}"
        )

        # Persist all outputs after each row -> nothing lost on SIGKILL
        _write_all_outputs()

        # Stop conditions: external signal or soft time budget exceeded
        if interrupt.stop:
            early_exit_reason = (
                f"signal {interrupt.received_signal} received (SLURM wall, "
                "SIGTERM or Ctrl-C); exiting gracefully after current eval."
            )
            break
        if time_budget > 0 and (time.monotonic() - wall_start) >= time_budget:
            early_exit_reason = (
                f"soft time budget exhausted "
                f"({time.monotonic() - wall_start:.0f}s >= {time_budget:.0f}s); "
                "exiting gracefully."
            )
            break
    pbar.close()

    # Make absolutely sure the final state is on disk even if no early exit fired
    _write_all_outputs()

    if early_exit_reason is not None:
        print(f"\n[early-exit] {early_exit_reason}")
        print(
            f"[early-exit] {len(rows)} / {total} configs done so far. "
            "Relaunch the same sbatch (with --resume, on by default) to continue."
        )

    # Brief summary (best-FID rows)
    if rows:
        best = sorted(rows, key=lambda r: r["fid"])[:5]
        print("\n[summary] top-5 lowest FID configs:")
        for r in best:
            print(
                f"  FID={r['fid']:.3f}  IS={r['inception_score_mean']:.3f}  "
                f"steps={r['steps']} eta={r['eta']} rho_g={r['rho_g']} "
                f"lh={r['lambda_h']} lb={r['lambda_beta']} lr={r['lambda_r']} "
                f"lc={r['lambda_c']} lh2={r['lambda_h2']}"
            )

    print(f"\n[done] wrote {out_path}, {csv_partial}, and {md_path}")


if __name__ == "__main__":
    main()