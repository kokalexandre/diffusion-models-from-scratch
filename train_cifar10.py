from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.cifar10 import make_cifar10_loader
from src.diffusion.ddpm_cifar import DDPMCIFAR
from src.models.unet_cifar10 import UNetCIFAR10, UNetCIFAR10Config
from src.utils import EMA, load_yaml, set_seed


def save_checkpoint(path, model, optimizer, global_step: int, ema: EMA | None, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "global_step": int(global_step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, str(path))


def load_checkpoint(path, model, optimizer=None, ema: EMA | None = None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    start_step = int(ckpt.get("global_step", 0))
    return start_step, ckpt.get("cfg", None)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _load_loss_csv(path: Path) -> list[tuple[int, float]]:
    if not path.exists():
        return []
    out = []
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append((int(row["global_step"]), float(row["loss"])))
    except Exception:
        return []
    return out


def _write_loss_csv(path: Path, rows: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["global_step", "loss"])
        writer.writeheader()
        for step, loss in rows:
            writer.writerow({"global_step": step, "loss": loss})


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


def make_autocast_kwargs(runtime: dict, device: torch.device) -> dict:
    mp = runtime["mixed_precision"]
    enabled = device.type == "cuda" and mp in {"fp16", "bf16"}
    dtype = None
    if mp == "fp16":
        dtype = torch.float16
    elif mp == "bf16":
        dtype = torch.bfloat16
    return {"device_type": "cuda", "enabled": enabled, "dtype": dtype}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cifar10_ddpm.yaml")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 0)))

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

    loader = make_cifar10_loader(
    root=cfg["data"]["root"],
    train=True,
    batch_size=int(cfg["data"]["batch_size"]),
    num_workers=int(cfg["data"]["num_workers"]),
    pin_memory=bool(cfg["data"]["pin_memory"]),
    download=bool(cfg["data"]["download"]),
    random_flip=bool(cfg["data"].get("random_flip", True)),
    persistent_workers=bool(cfg["data"].get("persistent_workers", False)),
    prefetch_factor=int(cfg["data"].get("prefetch_factor", 2)),
    )
    data_iter = _cycle(loader)

    model = build_model_from_cfg(cfg).to(device)
    if runtime["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    diffusion = build_diffusion_from_cfg(cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and runtime["mixed_precision"] == "fp16")
    )

    ema_decay = float(cfg["training"].get("ema_decay", 0.0))
    ema = EMA(model, decay=ema_decay) if ema_decay > 0.0 else None

    log_dir = Path(cfg["training"]["log_dir"])
    ckpt_dir = Path(cfg["training"]["ckpt_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    loss_csv_path = log_dir / "loss_steps.csv"
    loss_rows = _load_loss_csv(loss_csv_path)

    start_step = 0
    if args.resume:
        print(f"[resume] loading checkpoint: {args.resume}")
        start_step, _ = load_checkpoint(args.resume, model, optimizer=optimizer, ema=ema, map_location=device)
        loss_rows = [(step, loss) for step, loss in loss_rows if step <= start_step]
        print(f"[resume] resumed from global_step={start_step}")

    model_train = model
    if runtime["compile_enabled"]:
        print("[runtime] compiling model...")
        model_train = torch.compile(
            model,
            backend=runtime["compile_backend"],
            mode=runtime["compile_mode"],
        )

    max_steps = int(cfg["training"]["max_steps"])
    warmup_steps = int(cfg["training"].get("warmup_steps", 0))
    grad_clip = float(cfg["training"].get("grad_clip", 0.0))
    log_every = int(cfg["training"].get("log_every_steps", 100))
    save_every = int(cfg["training"].get("save_every_steps", 10000))
    base_lr = float(cfg["training"]["lr"])

    running_loss = 0.0
    model.train()
    model_train.train()

    autocast_kwargs = make_autocast_kwargs(runtime, device)

    pbar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc="train cifar10 ddpm")

    for global_step in pbar:
        x, _ = next(data_iter)
        x = x.to(device, non_blocking=True)
        if runtime["channels_last"]:
            x = x.contiguous(memory_format=torch.channels_last)

        if warmup_steps > 0:
            lr = base_lr * min(1.0, float(global_step + 1) / float(warmup_steps))
        else:
            lr = base_lr

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(**autocast_kwargs):
            loss = diffusion.loss(model_train, x)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        loss_val = float(loss.item())
        running_loss += loss_val

        if (global_step + 1) % log_every == 0:
            avg_loss = running_loss / log_every
            running_loss = 0.0
            writer.add_scalar("train/loss", avg_loss, global_step + 1)
            writer.add_scalar("train/lr", lr, global_step + 1)
            loss_rows.append((global_step + 1, avg_loss))
            _write_loss_csv(loss_csv_path, loss_rows)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

        if save_every > 0 and ((global_step + 1) % save_every == 0 or (global_step + 1) == max_steps):
            ckpt_path = ckpt_dir / f"ddpm_cifar10_step_{global_step + 1:07d}.pt"
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                global_step + 1,
                ema,
                cfg,
            )
            print(f"[checkpoint] saved: {ckpt_path}")

    writer.close()


if __name__ == "__main__":
    main()