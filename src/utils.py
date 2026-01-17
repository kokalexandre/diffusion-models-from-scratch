from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.utils import make_grid, save_image


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_image_grid(x: torch.Tensor, path: str, nrow: int = 8):
    # x in [-1, 1]
    x = x.clamp(-1, 1)
    x = (x + 1) / 2
    grid = make_grid(x, nrow=nrow)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self._init_from(model)

    def _init_from(self, model: torch.nn.Module):
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state):
        self.decay = float(state["decay"])
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}

    def make_ema_model(self, model: torch.nn.Module) -> torch.nn.Module:
        m = copy.deepcopy(model)
        sd = m.state_dict()
        for k in sd.keys():
            if k in self.shadow:
                sd[k] = self.shadow[k].to(sd[k].device).to(sd[k].dtype)
        m.load_state_dict(sd, strict=True)
        return m
