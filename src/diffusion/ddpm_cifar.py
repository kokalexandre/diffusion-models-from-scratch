from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.diffusion.ddim import _make_timesteps


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(0, t).reshape(b, *((1,) * (len(x_shape) - 1)))
    return out


class DDPMCIFAR(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        beta_schedule: Literal["linear"] = "linear",
        var_type: Literal["fixedsmall", "fixedlarge"] = "fixedlarge",
    ):
        super().__init__()
        self.timesteps = int(timesteps)
        self.var_type = str(var_type)

        if beta_schedule != "linear":
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")

        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]],
            dim=0,
        )

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / torch.clamp(1.0 - alphas_cumprod, min=1e-20)
        posterior_log_variance_clipped = torch.log(
            torch.clamp(
                torch.cat([posterior_variance[1:2], posterior_variance[1:]], dim=0),
                min=1e-20,
            )
        )
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / torch.clamp(1.0 - alphas_cumprod, min=1e-20)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / torch.clamp(1.0 - alphas_cumprod, min=1e-20)

        if self.var_type == "fixedlarge":
            model_variance = torch.cat([posterior_variance[1:2], betas[1:]], dim=0)
            model_log_variance = torch.log(torch.clamp(model_variance, min=1e-20))
        elif self.var_type == "fixedsmall":
            model_variance = posterior_variance
            model_log_variance = posterior_log_variance_clipped
        else:
            raise ValueError(f"Unknown var_type: {self.var_type}")

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register_buffer("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)
        self.register_buffer("model_variance", model_variance)
        self.register_buffer("model_log_variance", model_log_variance)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + _extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x0.shape,
        ) * noise

    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return _extract(self.sqrt_recip_alphas_cumprod, t, xt.shape) * xt - _extract(
            self.sqrt_recipm1_alphas_cumprod,
            t,
            xt.shape,
        ) * eps

    def q_posterior_mean(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return _extract(self.posterior_mean_coef1, t, xt.shape) * x0 + _extract(self.posterior_mean_coef2, t, xt.shape) * xt

    def loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.timesteps, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise=noise)
        pred_noise = model(xt, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_mean_variance(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = model(xt, t)
        x0_pred = self.predict_x0_from_eps(xt, t, eps)
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)
        model_mean = self.q_posterior_mean(x0_pred, xt, t)
        model_log_variance = _extract(self.model_log_variance, t, xt.shape)
        return model_mean, model_log_variance, x0_pred

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        model_mean, model_log_variance, _ = self.p_mean_variance(
            model,
            xt,
            t,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).float().reshape(xt.shape[0], *((1,) * (xt.ndim - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        device: torch.device,
        shape: tuple[int, int, int] = (3, 32, 32),
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        x = torch.randn((batch_size, *shape), device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, clip_denoised=clip_denoised)
        return x

    @torch.no_grad()
    def sample_subsequence_ddpm_noisy(
        self,
        model: nn.Module,
        batch_size: int,
        device: torch.device,
        shape: tuple[int, int, int] = (3, 32, 32),
        *,
        steps: int = 100,
        schedule: Literal["linear", "quadratic"] = "quadratic",
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        seq = _make_timesteps(self.timesteps, int(steps), schedule=schedule).to(device)
        seq_next = torch.cat([torch.tensor([-1], device=device, dtype=torch.long), seq[:-1]], dim=0)

        x = torch.randn((batch_size, *shape), device=device)
        for i, j in zip(reversed(seq.tolist()), reversed(seq_next.tolist())):
            t = torch.full((batch_size,), int(i), device=device, dtype=torch.long)
            eps = model(x, t)

            at = _extract(self.alphas_cumprod, t, x.shape)
            x0_pred = x / torch.sqrt(at) - torch.sqrt(torch.clamp(1.0 / at - 1.0, min=0.0)) * eps
            if clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            if j < 0:
                x = x0_pred
                continue

            t_prev = torch.full((batch_size,), int(j), device=device, dtype=torch.long)
            a_prev = _extract(self.alphas_cumprod, t_prev, x.shape)
            beta_t = 1.0 - at / a_prev

            mean = ((torch.sqrt(a_prev) * beta_t) * x0_pred + (torch.sqrt(1.0 - beta_t) * (1.0 - a_prev)) * x) / torch.clamp(
                1.0 - at,
                min=1e-12,
            )
            logvar = torch.log(torch.clamp(beta_t, min=1e-20))
            noise = torch.randn_like(x)
            x = mean + torch.exp(0.5 * logvar) * noise

        return x