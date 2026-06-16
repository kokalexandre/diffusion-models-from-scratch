"""CLI script : sampler Memory-DDIM (Petetin) avec variable auxiliaire y_t (CIFAR-10).

Strictement aligne sur sample_cifar10_ddim_augmented.py (runtime,
build_model_from_cfg, EMA, channels_last, torch.compile, autocast). Genere
une grille d'images PNG. Sert a :
  - produire la grille d'images pour le rapport/README ;
  - le check visuel f2=0 => DDIM a meme seed (mettre --lambda_f2 0) ;
  - l'inspection qualitative quand un FID est suspect.

Hyperparametres specifiques au sampler memory :
  --eta         : stochasticite DDIM (q_t)
  --lambda_f2   : f2_t = lambda_f2 * sqrt(Q_t / r_t),  0 <= lambda_f2 < 1
                  (lambda_f2 = 0 => DDIM exact)
  --lambda_h    : h_t = lambda_h * sqrt(alpha_t)
  --lambda_beta : beta_t = lambda_beta * sqrt(alpha_t)
  --lambda_r    : r_t = lambda_r * (1 - alpha_t) + eps_r
  --mode        : "barycenter" (h2,h1 barycentres des racines PSD) ou
                  "c0" (c_PJ=0, h2 barycentre analytique)
  --lambda_d    : barycentre h2 (mode barycenter), dans [0.1, 0.9]
  --lambda_e    : barycentre h1 (mode barycenter), dans [0.1, 0.9]
  --lambda_h2   : barycentre h2 (mode c0), dans [0.1, 0.9] ; 0.5 = centre
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

from src.diffusion.ddim_memory import MemoryDDIMConfig, MemoryDDIMSampler
from src.diffusion.ddpm_cifar import DDPMCIFAR
from src.models.unet_cifar10 import UNetCIFAR10, UNetCIFAR10Config
from src.utils import EMA, load_yaml, save_image_grid, set_seed


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


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cifar10_ddpm.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--schedule", type=str, default="quadratic", choices=["linear", "quadratic"])
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="")

    # Hyperparametres du sampler memory
    parser.add_argument("--mode", type=str, default="c0", choices=["barycenter", "c0"])
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--lambda_f2", type=float, default=0.3)
    parser.add_argument("--lambda_h", type=float, default=1.0)
    parser.add_argument("--lambda_beta", type=float, default=0.4)
    parser.add_argument("--lambda_r", type=float, default=0.1)
    parser.add_argument("--eps_r", type=float, default=1e-6)
    parser.add_argument("--lambda_d", type=float, default=0.5)
    parser.add_argument("--lambda_e", type=float, default=0.5)
    parser.add_argument("--lambda_h2", type=float, default=0.5)
    parser.add_argument("--no_clip_denoised", action="store_true")
    parser.add_argument("--verbose_pd", action="store_true",
                        help="Log warnings when Sigma_t is not PSD.")

    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = configure_runtime(cfg, device)

    print(
        "[runtime]",
        {
            "device": str(device),
            "mixed_precision": runtime["mixed_precision"],
            "channels_last": runtime["channels_last"],
            "compile_enabled": runtime["compile_enabled"],
        },
    )

    mem_cfg = MemoryDDIMConfig(
        steps=int(args.steps),
        schedule=str(args.schedule),
        eta=float(args.eta),
        lambda_f2=float(args.lambda_f2),
        lambda_h=float(args.lambda_h),
        lambda_beta=float(args.lambda_beta),
        lambda_r=float(args.lambda_r),
        eps_r=float(args.eps_r),
        mode=str(args.mode),
        lambda_d=float(args.lambda_d),
        lambda_e=float(args.lambda_e),
        lambda_h2=float(args.lambda_h2),
        clip_denoised=not bool(args.no_clip_denoised),
        verbose_pd_warnings=bool(args.verbose_pd),
    )
    print(
        "[ddim_memory]",
        {
            "mode": mem_cfg.mode,
            "steps": mem_cfg.steps,
            "schedule": mem_cfg.schedule,
            "eta": mem_cfg.eta,
            "lambda_f2": mem_cfg.lambda_f2,
            "lambda_h": mem_cfg.lambda_h,
            "lambda_beta": mem_cfg.lambda_beta,
            "lambda_r": mem_cfg.lambda_r,
            "lambda_d": mem_cfg.lambda_d,
            "lambda_e": mem_cfg.lambda_e,
            "lambda_h2": mem_cfg.lambda_h2,
            "clip_denoised": mem_cfg.clip_denoised,
        },
    )

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

    model_for_sample: nn.Module = InferenceModelWrapper(
        model,
        channels_last=runtime["channels_last"],
    ).to(device).eval()

    if runtime["compile_enabled"]:
        print("[runtime] compiling model for Memory-DDIM sampling...")
        model_for_sample = torch.compile(
            model_for_sample,
            backend=runtime["compile_backend"],
            mode=runtime["compile_mode"],
        )

    shape = tuple(int(v) for v in cfg["sampling"].get("shape", [3, 32, 32]))

    with make_autocast_context(runtime, device):
        imgs = sampler.sample(
            model_for_sample,
            batch_size=args.num_samples,
            device=device,
            shape=shape,
            cfg=mem_cfg,
        )

    imgs = imgs.float().cpu()

    out_path = args.out
    if not out_path:
        save_dir = Path(cfg["sampling"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.ckpt).stem
        if args.mode == "c0":
            tag = (
                f"ddim_memory_c0_steps{args.steps}_eta{args.eta}"
                f"_lf2{args.lambda_f2}_lh{args.lambda_h}_lb{args.lambda_beta}"
                f"_lr{args.lambda_r}_lh2{args.lambda_h2}"
            )
        else:
            tag = (
                f"ddim_memory_bary_steps{args.steps}_eta{args.eta}"
                f"_lf2{args.lambda_f2}_lh{args.lambda_h}_lb{args.lambda_beta}"
                f"_lr{args.lambda_r}_ld{args.lambda_d}_le{args.lambda_e}"
            )
        out_path = str(save_dir / f"{stem}_{tag}.png")

    nrow = int(max(1, round(args.num_samples ** 0.5)))
    save_image_grid(imgs, out_path, nrow=nrow)
    print(out_path)


if __name__ == "__main__":
    main()