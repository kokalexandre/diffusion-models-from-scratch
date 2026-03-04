import argparse
import time
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchvision.datasets import EMNIST
from torchvision import transforms
import torchvision.transforms.functional as TF

from src.utils import load_yaml, set_seed
from src.metrics.backbones import emnist_resnet34


def _fix_emnist_orientation(img):
    img = TF.rotate(img, -90)
    img = TF.hflip(img)
    return img


def _make_transforms(cfg, train: bool):
    tfms = [transforms.Lambda(_fix_emnist_orientation)]

    if train and bool(cfg["data"].get("augment", False)):
        deg = float(cfg["data"].get("aug_degrees", 10))
        tr = float(cfg["data"].get("aug_translate", 0.10))
        smin = float(cfg["data"].get("aug_scale_min", 0.90))
        smax = float(cfg["data"].get("aug_scale_max", 1.10))
        sh = float(cfg["data"].get("aug_shear", 5))

        tfms.append(
            transforms.RandomAffine(
                degrees=deg,
                translate=(tr, tr),
                scale=(smin, smax),
                shear=sh,
                fill=0,
            )
        )

    tfms += [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ]
    return transforms.Compose(tfms)


def make_loaders(cfg):
    root = cfg["data"]["root"]
    split = cfg["data"]["split"]
    download = bool(cfg["data"]["download"])

    val_fraction = float(cfg["data"].get("val_fraction", 0.05))
    val_seed = int(cfg["data"].get("val_seed", 0))

    tfm_tr = _make_transforms(cfg, train=True)
    tfm_eval = _make_transforms(cfg, train=False)

    ds_train_aug = EMNIST(root=root, split=split, train=True, download=download, transform=tfm_tr)
    ds_train_eval = EMNIST(root=root, split=split, train=True, download=download, transform=tfm_eval)

    n = len(ds_train_aug)
    n_val = int(round(n * val_fraction))
    n_val = max(1, min(n_val, n - 1))

    g = torch.Generator()
    g.manual_seed(val_seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    tr_ds = Subset(ds_train_aug, train_idx)
    val_ds = Subset(ds_train_eval, val_idx)

    tr = DataLoader(
        tr_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=True,
    )
    val = DataLoader(
        val_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=False,
    )

    ds_test = EMNIST(root=root, split=split, train=False, download=download, transform=tfm_eval)
    test = DataLoader(
        ds_test,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=False,
    )

    return tr, val, test, ds_train_aug


@torch.no_grad()
def eval_loss_acc(model, loader, device, label_smoothing: float = 0.0):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x, return_features=False)
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        total_loss += float(loss.item()) * x.shape[0]
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    model.train()
    return total_loss / max(1, total), correct / max(1, total)


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_losses(path_png: Path, rows):
    if not rows:
        return
    e = [r["epoch"] for r in rows]
    tl = [r["train_loss"] for r in rows]
    vl = [r["val_loss"] for r in rows]

    plt.figure()
    plt.plot(e, tl, label="train")
    plt.plot(e, vl, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_png, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/emnist_classifier.yaml")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = str(cfg["model"].get("backbone", "emnist_resnet34"))
    if backbone != "emnist_resnet34":
        raise ValueError(f"This script is ResNet-34 only. Got model.backbone={backbone}")

    tr_loader, val_loader, test_loader, ds_train = make_loaders(cfg)
    num_classes = len(ds_train.classes)

    feature_dim = int(cfg["model"]["feature_dim"])
    base_width = int(cfg["model"]["base_width"])
    model = emnist_resnet34(num_classes=num_classes, feature_dim=feature_dim, base_width=base_width).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    use_amp = bool(cfg["training"]["amp"]) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast = lambda: torch.amp.autocast("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    epochs = int(cfg["training"]["epochs"])
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))

    scheduler = None
    if bool(cfg["training"].get("use_onecycle", False)):
        max_lr = float(cfg["training"]["max_lr"])
        pct_start = float(cfg["training"].get("pct_start", 0.1))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=len(tr_loader),
            pct_start=pct_start,
            anneal_strategy="cos",
        )

    log_dir = Path(cfg["training"]["log_dir"]); log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["training"]["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    csv_path = log_dir / "loss_epoch.csv"
    png_path = log_dir / "loss_epoch.png"
    rows = []

    best_acc = -1.0
    global_step = 0
    split = cfg["data"]["split"]

    for ep in range(epochs):
        t0 = time.time()
        pbar = tqdm(tr_loader, desc=f"cls epoch {ep}/{epochs-1}")

        train_loss_sum = 0.0
        train_count = 0

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with autocast():
                logits = model(x, return_features=False)
                loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            train_loss_sum += float(loss.item()) * x.shape[0]
            train_count += int(x.shape[0])

            global_step += 1
            if global_step % 50 == 0:
                writer.add_scalar("train/loss_step", float(loss.item()), global_step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], global_step)

        train_loss = train_loss_sum / max(1, train_count)
        val_loss, val_acc = eval_loss_acc(model, val_loader, device, label_smoothing=0.0)

        writer.add_scalar("train/loss_epoch", train_loss, ep)
        writer.add_scalar("val/loss", val_loss, ep)
        writer.add_scalar("val/acc", val_acc, ep)
        writer.add_scalar("train/epoch_time_sec", time.time() - t0, ep)

        rows.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
        _write_csv(csv_path, rows)
        _plot_losses(png_path, rows)

        payload = {
            "model": model.state_dict(),
            "backbone": "emnist_resnet34",
            "num_classes": num_classes,
            "feature_dim": feature_dim,
            "base_width": base_width,
            "split": split,
            "val_fraction": float(cfg["data"].get("val_fraction", 0.05)),
            "val_seed": int(cfg["data"].get("val_seed", 0)),
        }

        if bool(cfg["training"].get("save_best", True)):
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(payload, ckpt_dir / f"emnist_resnet34_best_{split}.pt")
        else:
            torch.save(payload, ckpt_dir / f"emnist_resnet34_epoch_{ep:04d}_{split}.pt")

        print(f"[epoch {ep}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    test_loss, test_acc = eval_loss_acc(model, test_loader, device, label_smoothing=0.0)
    writer.add_scalar("test/loss", test_loss, epochs)
    writer.add_scalar("test/acc", test_acc, epochs)
    print(f"[final] test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    writer.close()


if __name__ == "__main__":
    main()