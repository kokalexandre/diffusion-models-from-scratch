from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms


def make_cifar10_loader(
    root: str,
    train: bool,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    download: bool,
    random_flip: bool = True,
    drop_last: bool | None = None,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    tfms = []
    if train and random_flip:
        tfms.append(transforms.RandomHorizontalFlip())
    tfms.extend(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [0,1] -> [-1,1]
        ]
    )
    tfm = transforms.Compose(tfms)

    ds = CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=tfm,
    )

    if drop_last is None:
        drop_last = bool(train)

    loader_kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    loader = DataLoader(**loader_kwargs)
    return loader