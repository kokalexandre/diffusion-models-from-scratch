from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


def _make_timesteps(
    T: int,
    S: int,
    schedule: Literal["linear", "quadratic"] = "linear",
) -> torch.Tensor:
    T = int(T)
    S = int(S)
    if S < 2:
        S = 2
    if S > T:
        S = T

    if schedule == "linear":
        ts = torch.linspace(0, T - 1, S)
    elif schedule == "quadratic":
        ts = torch.linspace(0, math.sqrt(T - 1), S) ** 2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    idx = []
    for v in ts.tolist():
        i = int(round(v))
        i = max(0, min(T - 1, i))
        if len(idx) == 0 or i != idx[-1]:
            idx.append(i)

    if idx[0] != 0:
        idx = [0] + idx
    if idx[-1] != T - 1:
        idx = idx + [T - 1]

    idx = sorted(set(idx))
    if len(idx) > S:
        idx = idx[: S - 1] + [T - 1]
    if len(idx) < S:
        fill = torch.linspace(0, T - 1, S).round().to(torch.long).tolist()
        for i in fill:
            i = int(i)
            if i not in idx:
                idx.append(i)
            if len(idx) >= S:
                break
        idx = sorted(set(idx))
        if idx[-1] != T - 1:
            idx[-1] = T - 1
        if idx[0] != 0:
            idx[0] = 0

    return torch.tensor(idx, dtype=torch.long)


@dataclass
class DDIMConfig:
    steps: int = 50
    schedule: Literal["linear", "quadratic"] = "linear"
    eta: float = 0.0


class DDIMSampler:
    def __init__(self, ddpm):
        self.ddpm = ddpm

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        device: torch.device,
        shape=(1, 28, 28),
        *,
        steps: int = 50,
        schedule: Literal["linear", "quadratic"] = "linear",
        eta: float = 0.0,
        clip_denoised: bool = True,
        return_timesteps: bool = False,
    ):
        T = int(self.ddpm.timesteps)
        ts = _make_timesteps(T=T, S=int(steps), schedule=schedule).to(device)
        x = torch.randn((batch_size, *shape), device=device)

        abar = self.ddpm.alphas_cumprod.to(device)

        for j in range(len(ts) - 1, -1, -1):
            t = int(ts[j].item())
            t_prev = int(ts[j - 1].item()) if j > 0 else -1

            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            eps = model(x, t_batch)

            a_t = abar[t]
            sqrt_a_t = torch.sqrt(a_t)
            sqrt_1m_a_t = torch.sqrt(torch.clamp(1.0 - a_t, min=0.0))

            x0_pred = (x - sqrt_1m_a_t * eps) / torch.clamp(sqrt_a_t, min=1e-12)

            if clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)
                eps = (x - sqrt_a_t * x0_pred) / torch.clamp(sqrt_1m_a_t, min=1e-12)

            if t_prev < 0:
                x = x0_pred
                break

            a_prev = abar[t_prev]
            sqrt_a_prev = torch.sqrt(a_prev)

            sigma = 0.0
            if eta > 0:
                sigma = eta * torch.sqrt(
                    torch.clamp(
                        (1.0 - a_prev) / torch.clamp(1.0 - a_t, min=1e-12)
                        * (1.0 - a_t / torch.clamp(a_prev, min=1e-12)),
                        min=0.0,
                    )
                )

            sigma2 = sigma * sigma
            c = torch.sqrt(torch.clamp(1.0 - a_prev - sigma2, min=0.0))

            x_prev = sqrt_a_prev * x0_pred + c * eps
            if eta > 0:
                x_prev = x_prev + sigma * torch.randn_like(x_prev)

            x = x_prev

        if return_timesteps:
            return x, ts.detach().cpu()
        return x