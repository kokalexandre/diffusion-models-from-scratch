from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms
import torchvision.transforms.functional as TF


def _fix_emnist_orientation(img):
    # torchvision EMNIST est souvent transposé; ce fix donne des glyphes “droits”
    img = TF.rotate(img, -90)
    img = TF.hflip(img)
    return img


def make_emnist_loader(
    root: str,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    download: bool,
) -> DataLoader:
    tfm = transforms.Compose([
        transforms.Lambda(_fix_emnist_orientation),
        transforms.ToTensor(),                    # [0,1]
        transforms.Lambda(lambda x: x * 2.0 - 1.0) # [-1,1]
    ])

    ds = EMNIST(
        root=root,
        split=split,
        train=True,
        download=download,
        transform=tfm,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return loader
