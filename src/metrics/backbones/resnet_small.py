from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = F.silu(out + identity)
        return out


class EMNISTResNet(nn.Module):
    """
    ResNet-34-like
    """
    def __init__(
        self,
        num_classes: int,
        layers=(3, 4, 6, 3),  
        feature_dim: int = 512,
        base_width: int = 96,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        bw = int(base_width)
        l1, l2, l3, l4 = layers

        self.conv1 = nn.Conv2d(1, bw, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bw)

        self.layer1 = self._make_layer(bw, bw, blocks=l1, stride=1)
        self.layer2 = self._make_layer(bw, bw * 2, blocks=l2, stride=2)
        self.layer3 = self._make_layer(bw * 2, bw * 4, blocks=l3, stride=2)
        self.layer4 = self._make_layer(bw * 4, bw * 8, blocks=l4, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feat_in = bw * 8
        self.proj = nn.Identity() if feat_in == self.feature_dim else nn.Linear(feat_in, self.feature_dim)
        self.fc = nn.Linear(self.feature_dim, self.num_classes)

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int):
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_features: bool = True):
        x = F.silu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x).flatten(1)
        feat = self.proj(x)
        if not isinstance(self.proj, nn.Identity):
            feat = F.silu(feat)
        logits = self.fc(feat)

        if return_features:
            return logits, feat
        return logits


def emnist_resnet34(num_classes: int, feature_dim: int = 512, base_width: int = 96) -> EMNISTResNet:
    return EMNISTResNet(num_classes=num_classes, layers=(3, 4, 6, 3), feature_dim=feature_dim, base_width=base_width)