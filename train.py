import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils import load_yaml, set_seed, EMA, save_image_grid
from src.data.emnist import make_emnist_loader
from src.models.unet import UNet
from src.diffusion.ddpm import DDPM


def save_checkpoint(path, model, optimizer, epoch, ema: EMA | None, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
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
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    return start_epoch, ckpt.get("cfg", None)


@torch.no_grad()
def run_sampling(diffusion: DDPM, model: torch.nn.Module, cfg, device, epoch: int):
    save_dir = Path(cfg["sampling"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    n = int(cfg["sampling"]["num_samples"])
    bs = int(cfg["sampling"]["batch_size"])
    all_imgs = []
    remaining = n
    while remaining > 0:
        cur = min(bs, remaining)
        x = diffusion.sample(model, batch_size=cur, device=device)
        all_imgs.append(x.cpu())
        remaining -= cur

    imgs = torch.cat(all_imgs, dim=0)
    out_path = save_dir / f"epoch_{epoch:04d}.png"
    save_image_grid(imgs, str(out_path), nrow=int(n ** 0.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/emnist_ddpm.yaml")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    loader = make_emnist_loader(
        root=cfg["data"]["root"],
        split=cfg["data"]["split"],
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        download=bool(cfg["data"]["download"]),
    )

    # Model
    model = UNet(
        in_channels=int(cfg["model"]["in_channels"]),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        channel_mults=list(cfg["model"]["channel_mults"]),
        num_res_blocks=int(cfg["model"]["num_res_blocks"]),
        dropout=float(cfg["model"]["dropout"]),
        time_emb_dim=int(cfg["model"]["time_emb_dim"]),
    ).to(device)

    diffusion = DDPM(
        timesteps=int(cfg["diffusion"]["timesteps"]),
        beta_start=float(cfg["diffusion"]["beta_start"]),
        beta_end=float(cfg["diffusion"]["beta_end"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    use_amp = bool(cfg["training"]["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ema = EMA(model, decay=float(cfg["training"]["ema_decay"])) if float(cfg["training"]["ema_decay"]) > 0 else None

    log_dir = Path(cfg["training"]["log_dir"])
    ckpt_dir = Path(cfg["training"]["ckpt_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer=optimizer, ema=ema, map_location=device)

    epochs = int(cfg["training"]["epochs"])
    grad_clip = float(cfg["training"]["grad_clip"])
    save_every = int(cfg["training"]["save_every_epochs"])
    sample_every = int(cfg["training"]["sample_every_epochs"])

    global_step = 0
    model.train()
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs-1}", leave=True)
        running_loss = 0.0

        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = diffusion.loss(model, x)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            running_loss += float(loss.item())
            global_step += 1

            if global_step % 50 == 0:
                avg = running_loss / 50.0
                running_loss = 0.0
                writer.add_scalar("train/loss", avg, global_step)
                pbar.set_postfix(loss=f"{avg:.4f}")

        epoch_time = time.time() - t0
        writer.add_scalar("train/epoch_time_sec", epoch_time, epoch)

        # Save
        if (epoch + 1) % save_every == 0:
            save_checkpoint(ckpt_dir / f"ddpm_emnist_epoch_{epoch:04d}.pt", model, optimizer, epoch, ema, cfg)

        # Sample
        if (epoch + 1) % sample_every == 0:
            model_to_sample = model
            if bool(cfg["sampling"]["use_ema"]) and ema is not None:
                model_to_sample = ema.make_ema_model(model).to(device).eval()
            else:
                model_to_sample = model.eval()

            run_sampling(diffusion, model_to_sample, cfg, device, epoch)
            model.train()

    writer.close()


if __name__ == "__main__":
    main()
