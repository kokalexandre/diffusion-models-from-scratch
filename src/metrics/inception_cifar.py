from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self, channels_last: bool = True):
        super().__init__()

        from torchvision.models import Inception_V3_Weights, inception_v3

        weights = Inception_V3_Weights.IMAGENET1K_V1

        # Important:
        # torchvision inception_v3 with pretrained weights expects aux_logits=True
        # during construction. We can disable the auxiliary head afterwards.
        self.model = inception_v3(
            weights=weights,
            transform_input=False,
            aux_logits=True,
        )
        self.model.aux_logits = False
        self.model.AuxLogits = None

        self.channels_last = bool(channels_last)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(-1.0, 1.0)
        x = (x + 1.0) / 2.0
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._preprocess(x)
        m = self.model

        x = m.Conv2d_1a_3x3(x)
        x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x)
        x = m.maxpool1(x)
        x = m.Conv2d_3b_1x1(x)
        x = m.Conv2d_4a_3x3(x)
        x = m.maxpool2(x)
        x = m.Mixed_5b(x)
        x = m.Mixed_5c(x)
        x = m.Mixed_5d(x)
        x = m.Mixed_6a(x)
        x = m.Mixed_6b(x)
        x = m.Mixed_6c(x)
        x = m.Mixed_6d(x)
        x = m.Mixed_6e(x)
        x = m.Mixed_7a(x)
        x = m.Mixed_7b(x)
        x = m.Mixed_7c(x)
        x = m.avgpool(x)
        features = torch.flatten(x, 1)
        x = m.dropout(x)
        logits = m.fc(torch.flatten(x, 1))
        return logits, features


class InceptionScoreAccumulator:
    def __init__(self, eps: float = 1e-16):
        self.eps = float(eps)
        self.probs: list[torch.Tensor] = []

    @torch.no_grad()
    def update_logits(self, logits: torch.Tensor) -> None:
        probs = torch.softmax(logits.detach(), dim=1).to("cpu", dtype=torch.float64)
        self.probs.append(probs)

    @torch.no_grad()
    def finalize(self, splits: int = 10) -> tuple[float, float]:
        if not self.probs:
            raise RuntimeError("No logits were accumulated for Inception Score.")

        probs = torch.cat(self.probs, dim=0)
        n = probs.shape[0]
        splits = max(1, min(int(splits), n))
        chunk = n // splits
        scores = []

        for i in range(splits):
            start = i * chunk
            end = n if i == splits - 1 else (i + 1) * chunk
            part = probs[start:end]
            p_y = part.mean(dim=0, keepdim=True)
            kl = part * (torch.log(part + self.eps) - torch.log(p_y + self.eps))
            score = torch.exp(kl.sum(dim=1).mean())
            scores.append(float(score.item()))

        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(var)
        return float(mean), float(std)