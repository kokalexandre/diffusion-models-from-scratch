from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

from src.diffusion.ddim import DDIMSampler
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
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="")
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
            "allow_tf32": runtime["allow_tf32"],
            "cudnn_benchmark": runtime["cudnn_benchmark"],
            "channels_last": runtime["channels_last"],
            "compile_enabled": runtime["compile_enabled"],
            "compile_mode": runtime["compile_mode"],
            "compile_backend": runtime["compile_backend"],
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

    model_for_sample: nn.Module = InferenceModelWrapper(
        model,
        channels_last=runtime["channels_last"],
    ).to(device).eval()

    if runtime["compile_enabled"]:
        print("[runtime] compiling model for DDIM sampling...")
        model_for_sample = torch.compile(
            model_for_sample,
            backend=runtime["compile_backend"],
            mode=runtime["compile_mode"],
        )

    shape = tuple(int(v) for v in cfg["sampling"].get("shape", [3, 32, 32]))

    with make_autocast_context(runtime, device):
        imgs = ddim.sample(
            model_for_sample,
            batch_size=args.num_samples,
            device=device,
            shape=shape,
            steps=int(args.steps),
            schedule=str(args.schedule),
            eta=float(args.eta),
        )

    imgs = imgs.float().cpu()

    out_path = args.out
    if not out_path:
        save_dir = Path(cfg["sampling"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.ckpt).stem
        out_path = str(save_dir / f"{stem}_ddim_steps{args.steps}_eta{args.eta}.png")

    nrow = int(max(1, round(args.num_samples ** 0.5)))
    save_image_grid(imgs, out_path, nrow=nrow)
    print(out_path)


if __name__ == "__main__":
    main()