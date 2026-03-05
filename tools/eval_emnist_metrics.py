import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from torchvision.datasets import EMNIST
from torchvision import transforms
import torchvision.transforms.functional as TF

from src.utils import load_yaml, EMA
from src.models.unet import UNet
from src.diffusion.ddpm import DDPM
from src.diffusion.ddim import DDIMSampler
from src.metrics.backbones import emnist_resnet34
from src.metrics.stats import StreamingMeanCov, StreamingInceptionScore, frechet_distance


def _fix_emnist_orientation(img):
    img = TF.rotate(img, -90)
    img = TF.hflip(img)
    return img


def _make_emnist_loader(
    root: str,
    split: str,
    train: bool,
    batch_size: int,
    num_workers: int,
    real_max: int | None,
    seed: int,
):
    tfm = transforms.Compose([
        transforms.Lambda(_fix_emnist_orientation),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1,1]
    ])
    ds = EMNIST(root=root, split=split, train=train, download=False, transform=tfm)

    if real_max is not None and real_max > 0 and real_max < len(ds):
        g = torch.Generator()
        g.manual_seed(int(seed))
        idx = torch.randperm(len(ds), generator=g)[: int(real_max)].tolist()
        ds = Subset(ds, idx)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl, ds


@torch.no_grad()
def _extract_real_stats(classifier, loader, device, feature_dim: int):
    classifier.eval()
    stats = StreamingMeanCov(feature_dim)
    seen = 0

    for x, _ in tqdm(loader, desc="real features"):
        x = x.to(device, non_blocking=True)
        _, feat = classifier(x, return_features=True)
        stats.update(feat)
        seen += x.shape[0]

    mu, cov = stats.finalize()
    return mu, cov, seen


@torch.no_grad()
def _sample_images(
    *,
    sampler: str,
    diffusion: DDPM,
    ddim: DDIMSampler,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    eta: float,
    ddim_steps: int,
    ddim_schedule: str,
):
    if sampler == "ddpm":
        return diffusion.sample(model, batch_size=batch_size, device=device)

    if sampler == "ddim":
        return ddim.sample(
            model,
            batch_size=batch_size,
            device=device,
            shape=(1, 28, 28),
            steps=int(ddim_steps),
            schedule=ddim_schedule,
            eta=float(eta),
        )

    raise ValueError(f"Unknown sampler: {sampler}")


@torch.no_grad()
def _extract_gen_stats_and_is(
    *,
    sampler: str,
    diffusion: DDPM,
    ddim: DDIMSampler,
    ddpm_model: torch.nn.Module,
    classifier,
    device,
    num_gen: int,
    gen_batch: int,
    feature_dim: int,
    num_classes: int,
    eta: float,
    ddim_steps: int,
    ddim_schedule: str,
):
    ddpm_model.eval()
    classifier.eval()

    stats = StreamingMeanCov(feature_dim)
    is_acc = StreamingInceptionScore(num_classes=num_classes)

    remaining = int(num_gen)
    pbar = tqdm(total=int(num_gen), desc="generate+features")

    while remaining > 0:
        b = min(int(gen_batch), remaining)
        imgs = _sample_images(
            sampler=sampler,
            diffusion=diffusion,
            ddim=ddim,
            model=ddpm_model,
            batch_size=b,
            device=device,
            eta=eta,
            ddim_steps=ddim_steps,
            ddim_schedule=ddim_schedule,
        )
        logits, feat = classifier(imgs, return_features=True)
        stats.update(feat)
        is_acc.update_logits(logits)
        remaining -= b
        pbar.update(b)

    pbar.close()
    mu, cov = stats.finalize()
    is_score = is_acc.finalize()
    return mu, cov, is_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ddpm_config", type=str, default="configs/emnist_ddpm.yaml")
    p.add_argument("--ddpm_ckpt", type=str, required=True)
    p.add_argument("--use_ema", action="store_true")

    p.add_argument("--cls_ckpt", type=str, required=True)

    p.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--ddim_schedule", type=str, default="linear", choices=["linear", "quadratic"])

    p.add_argument("--num_gen", type=int, default=5000)
    p.add_argument("--gen_batch", type=int, default=64)

    p.add_argument("--real_splits", type=str, default="train,test")
    p.add_argument("--real_batch", type=int, default=512)
    p.add_argument("--real_max", type=int, default=10000)
    p.add_argument("--real_seed", type=int, default=0)

    p.add_argument("--gen_seed", type=int, default=-1)

    p.add_argument("--out", type=str, default="runs/emnist_metrics/metrics.json")
    args = p.parse_args()

    if int(args.gen_seed) >= 0:
        torch.manual_seed(int(args.gen_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.gen_seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_yaml(args.ddpm_config)

    ddpm_model = UNet(
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

    ckpt = torch.load(args.ddpm_ckpt, map_location=device)
    ddpm_model.load_state_dict(ckpt["model"], strict=True)
    ddpm_model.eval()

    if args.use_ema and "ema" in ckpt:
        ema = EMA(ddpm_model, decay=float(cfg["training"]["ema_decay"]))
        ema.load_state_dict(ckpt["ema"])
        ddpm_model = ema.make_ema_model(ddpm_model).to(device).eval()

    cls_blob = torch.load(args.cls_ckpt, map_location=device)
    backbone = str(cls_blob.get("backbone", "emnist_resnet34"))
    if backbone != "emnist_resnet34":
        raise ValueError(f"Expected emnist_resnet34 ckpt, got backbone={backbone}")

    num_classes = int(cls_blob["num_classes"])
    feature_dim = int(cls_blob["feature_dim"])
    base_width = int(cls_blob.get("base_width", 96))

    classifier = emnist_resnet34(num_classes=num_classes, feature_dim=feature_dim, base_width=base_width).to(device)
    classifier.load_state_dict(cls_blob["model"], strict=True)
    classifier.eval()

    ddim = DDIMSampler(diffusion)

    mu_g, cov_g, is_score = _extract_gen_stats_and_is(
        sampler=args.sampler,
        diffusion=diffusion,
        ddim=ddim,
        ddpm_model=ddpm_model,
        classifier=classifier,
        device=device,
        num_gen=args.num_gen,
        gen_batch=args.gen_batch,
        feature_dim=feature_dim,
        num_classes=num_classes,
        eta=float(args.eta),
        ddim_steps=int(args.ddim_steps),
        ddim_schedule=args.ddim_schedule,
    )

    root = cfg["data"]["root"]
    split = cfg["data"]["split"]
    num_workers = int(cfg["data"]["num_workers"])

    splits = [s.strip() for s in args.real_splits.split(",") if s.strip()]
    results = {}

    for s in splits:
        is_train = (s == "train")
        split_seed = int(args.real_seed) + (0 if is_train else 1)

        real_loader, _ = _make_emnist_loader(
            root=root,
            split=split,
            train=is_train,
            batch_size=args.real_batch,
            num_workers=num_workers,
            real_max=int(args.real_max) if int(args.real_max) > 0 else None,
            seed=split_seed,
        )

        mu_r, cov_r, n_real = _extract_real_stats(
            classifier=classifier,
            loader=real_loader,
            device=device,
            feature_dim=feature_dim,
        )

        fid = frechet_distance(mu_r, cov_r, mu_g, cov_g)
        results[s] = {"fid": float(fid), "n_real": int(n_real), "real_seed": int(split_seed)}

    out = {
        "dataset": f"EMNIST/{split}",
        "sampler": args.sampler,
        "eta": float(args.eta),
        "ddim_steps": int(args.ddim_steps),
        "ddim_schedule": args.ddim_schedule,
        "n_gen": int(args.num_gen),
        "gen_batch": int(args.gen_batch),
        "gen_seed": int(args.gen_seed),
        "feature_extractor": "emnist_resnet34",
        "feature_dim": int(feature_dim),
        "inception_score": float(is_score),
        "fid_by_real_split": results,
        "ddpm_ckpt": args.ddpm_ckpt,
        "cls_ckpt": args.cls_ckpt,
        "use_ema": bool(args.use_ema),
        "real_max": int(args.real_max),
        "real_seed": int(args.real_seed),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()