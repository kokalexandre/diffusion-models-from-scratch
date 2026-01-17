import argparse
from pathlib import Path

import torch

from src.utils import load_yaml, save_image_grid, EMA
from src.models.unet import UNet
from src.diffusion.ddpm import DDPM


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/emnist_ddpm.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    if args.use_ema and "ema" in ckpt:
        ema = EMA(model, decay=float(cfg["training"]["ema_decay"]))
        ema.load_state_dict(ckpt["ema"])
        model = ema.make_ema_model(model).to(device).eval()

    imgs = diffusion.sample(model, batch_size=args.num_samples, device=device)

    out_path = args.out
    if not out_path:
        save_dir = Path(cfg["sampling"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(save_dir / "sample.png")

    save_image_grid(imgs.cpu(), out_path, nrow=int(args.num_samples ** 0.5))


if __name__ == "__main__":
    main()
