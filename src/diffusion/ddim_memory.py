"""Memory-DDIM sampler (Petetin), ported into the repo infrastructure.

Port of the supervisor's `memory_sampler2.py` / `symbolic_constraints.py`,
adapted to this repo's conventions:
  - the network predicts epsilon (via the InferenceModelWrapper), not .sample;
  - alphas_cumprod come from DDPMCIFAR;
  - timesteps use the repo's `_make_timesteps` so the comparison against the
    existing DDIM eval is apples-to-apples (same trajectory, same schedule);
  - the loop stops at the transition sampling x_0 (NO extra model call at
    t=0), matching the supervisor's fair DDIM baseline.

Two parametrizations of the free coefficients (f2, h2=d, h1=e), both
guaranteeing Sigma_t PSD by construction:

  mode = "barycenter"  (supervisor's default)
    f2 = lambda_f2 * sqrt(Q_t / r_t)
    h2 = barycentre(lambda_d) entre les racines d1,d2 du discriminant en e
    h1 = barycentre(lambda_e) entre les racines e1,e2 de det(Sigma)=0
    c_PJ = f_t*h_prev - h_t*h2 - h1   (implicite, non nul en general)

  mode = "c0"  (reparametrisation (f2, h2, c) avec c=0)
    f2 = lambda_f2 * sqrt(Q_t / r_t)
    c_PJ = 0   =>   h1 = f_t*h_prev - h_t*h2
    h2 = f2*h_prev + (2*lambda_h2 - 1) * delta,
         delta = sqrt(r_prev * sigma11 / (r_t * Q_t))
    Forme close : det(Sigma) = Q*r_prev - f2^2*r*r_prev - r*Q*(h2 - f2*h_prev)^2
    (verifiee symboliquement). PSD garantie pour lambda_h2 dans [0,1].

Convention de notation (alignee sur les notes du superviseur) :
    a = f_t, b = h_t, b_prev = h_prev, c = f2_t, d = h2_t, e = h1_t,
    Q = Q_t, R = r_t, R_prev = r_prev, P = 1 - alpha_t.

Les coefficients des racines (mode barycenter) sont la forme analytique
fermee des fonctions sympy-lambdifiees du superviseur, verifiee a 3e-14
pres -> aucune dependance sympy au runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from src.diffusion.ddim import _make_timesteps


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MemoryDDIMConfig:
    """Hyperparametres du Memory-DDIM sampler.

    Noms alignes sur le notebook du superviseur.
    """

    steps: int = 20
    schedule: Literal["linear", "quadratic"] = "quadratic"
    eta: float = 0.2

    # f2 = lambda_f2 * sqrt(Q_t / r_t),  0 <= lambda_f2 < 1
    lambda_f2: float = 0.3

    # Schedules memoire : h_t = lambda_h*sqrt(a_t), beta_t = lambda_beta*sqrt(a_t),
    #                     r_t = lambda_r*(1-a_t) + eps_r
    lambda_h: float = 1.0
    lambda_beta: float = 0.4
    lambda_r: float = 0.1
    eps_r: float = 1e-6

    # Mode de choix de (h2, h1)
    mode: Literal["barycenter", "c0"] = "barycenter"

    # mode "barycenter" : barycentres des racines PSD
    lambda_d: float = 0.5   # h2 = (1-lambda_d)*d1 + lambda_d*d2
    lambda_e: float = 0.5   # h1 = (1-lambda_e)*e1 + lambda_e*e2

    # mode "c0" : barycentre de h2 dans [f2*h_prev - delta, f2*h_prev + delta]
    lambda_h2: float = 0.5  # 0.5 => h2 = f2*h_prev (centre)

    clip_denoised: bool = True

    # Securite PSD (les modes garantissent PSD par construction, mais on
    # garde un garde-fou numerique aux bords).
    pd_tol: float = 1e-9
    verbose_pd_warnings: bool = False


# ---------------------------------------------------------------------------
# Roots of the PSD region (analytic, = supervisor's lambdified sympy roots)
# Verified to 3e-14 against symbolic_constraints.py
# ---------------------------------------------------------------------------


def _roots_d(a, Q, c, b, b_prev, R, R_prev, P):
    """Racines d1,d2 du discriminant (en e) vu comme polynome en d (=h2).

    Entre d1 et d2, le polynome det(Sigma) en e admet des racines reelles.
    Ordre identique a celui du superviseur : d1 = (-Bd - sqrt)/(2Ad).
    """
    s11 = Q - R * c * c
    common = P * a * a + Q
    Ad = -4.0 * P * R * s11 * common
    Bd = 8.0 * P * R * b_prev * c * s11 * common
    Cd = -4.0 * P * s11 * (
        P * R * a * a * b_prev * b_prev * c * c
        - P * R_prev * a * a
        + Q * R * b_prev * b_prev * c * c
        - Q * R_prev
        + R * R_prev * c * c
    )
    disc = Bd * Bd - 4.0 * Ad * Cd
    if disc < 0.0 or Ad == 0.0:
        return float("nan"), float("nan")
    sq = math.sqrt(disc)
    d1 = (-Bd - sq) / (2.0 * Ad)
    d2 = (-Bd + sq) / (2.0 * Ad)
    return d1, d2


def _roots_e(a, Q, c, d, b, b_prev, R, R_prev, P):
    """Racines e1,e2 de det(Sigma)=0 vu comme polynome en e (=h1).

    Entre e1 et e2, det(Sigma) >= 0. Ordre identique au superviseur.
    """
    Ae = -P * (P * a * a + Q - R * c * c)
    Be = 2.0 * P * (
        P * a ** 3 * b_prev
        - P * a * a * b * d
        + Q * a * b_prev
        - Q * b * d
        - R * a * c * d
        + R * b * c * c * d
    )
    Ce = (
        -P * P * a ** 4 * b_prev * b_prev
        - P * Q * a * a * b_prev * b_prev
        - P * R * a * a * b_prev * b_prev * c * c
        - Q * R * b_prev * b_prev * c * c
        + Q * R_prev
        - R * R_prev * c * c
        + d * d * (
            -P * P * a * a * b * b
            - P * Q * b * b
            - 2.0 * P * R * a * b * c
            + P * R * b * b * c * c
            - Q * R
        )
        + d * (
            2.0 * P * P * a ** 3 * b * b_prev
            + 2.0 * P * Q * a * b * b_prev
            + 2.0 * P * R * a * a * b_prev * c
            + 2.0 * Q * R * b_prev * c
        )
    )
    disc = Be * Be - 4.0 * Ae * Ce
    if disc < 0.0 or Ae == 0.0:
        return float("nan"), float("nan")
    sq = math.sqrt(disc)
    e1 = (-Be - sq) / (2.0 * Ae)
    e2 = (-Be + sq) / (2.0 * Ae)
    return e1, e2


# ---------------------------------------------------------------------------
# DDIM scalar coefficients
# ---------------------------------------------------------------------------


def _ddim_Q(alpha_prev: float, alpha_t: float, eta: float) -> float:
    if eta <= 0.0:
        return 0.0
    eps = 1e-12
    term1 = (1.0 - alpha_prev) / (1.0 - alpha_t + eps)
    term2 = 1.0 - alpha_t / (alpha_prev + eps)
    return eta * eta * term1 * term2


def _ddim_f(alpha_prev: float, alpha_t: float, Q: float) -> float:
    eps = 1e-12
    num = max(eps, 1.0 - alpha_prev - Q)
    den = max(eps, 1.0 - alpha_t)
    return math.sqrt(num / den)


# ---------------------------------------------------------------------------
# Per-step coefficient computation
# ---------------------------------------------------------------------------


def _compute_step(alpha_t: float, alpha_prev: float, cfg: MemoryDDIMConfig) -> dict:
    """Calcule tous les coefficients du pas t -> t_prev pour le mode choisi."""
    Q = _ddim_Q(alpha_prev, alpha_t, cfg.eta)
    f = _ddim_f(alpha_prev, alpha_t, Q)
    P = max(1.0 - alpha_t, 0.0)
    sqrt_a_t = math.sqrt(max(alpha_t, 0.0))
    sqrt_a_prev = math.sqrt(max(alpha_prev, 0.0))

    # Schedules memoire (varient en t)
    h_t = cfg.lambda_h * sqrt_a_t
    h_prev = cfg.lambda_h * sqrt_a_prev
    beta_t = cfg.lambda_beta * sqrt_a_t
    beta_prev = cfg.lambda_beta * sqrt_a_prev
    r_t = cfg.lambda_r * (1.0 - alpha_t) + cfg.eps_r
    r_prev = cfg.lambda_r * (1.0 - alpha_prev) + cfg.eps_r

    # f2 = lambda_f2 * sqrt(Q / r_t)  (assure sigma11 = Q - r_t*f2^2 >= 0)
    f2_max = math.sqrt(max(Q, 0.0) / max(r_t, 1e-12))
    f2 = cfg.lambda_f2 * f2_max

    if cfg.mode == "c0":
        # c_PJ = 0  =>  h1 = f*h_prev - h_t*h2
        s11 = Q - r_t * f2 * f2
        # det = Q*r_prev - f2^2*r*r_prev - r*Q*(h2 - f2*h_prev)^2 >= 0
        #   <=> (h2 - f2*h_prev)^2 <= r_prev*s11/(r*Q)
        if Q > 1e-12 and r_t > 1e-12:
            delta = math.sqrt(max(r_prev * s11 / (r_t * Q), 0.0))
        else:
            delta = 0.0
        h2 = f2 * h_prev + (2.0 * cfg.lambda_h2 - 1.0) * delta
        h1 = f * h_prev - h_t * h2
    else:  # "barycenter"
        d1, d2 = _roots_d(f, Q, f2, h_t, h_prev, r_t, r_prev, P)
        if math.isnan(d1):
            # pas de region reelle -> on retombe sur c=0 centre (h2 = f2*h_prev)
            h2 = f2 * h_prev
        else:
            h2 = (1.0 - cfg.lambda_d) * d1 + cfg.lambda_d * d2
        e1, e2 = _roots_e(f, Q, f2, h2, h_t, h_prev, r_t, r_prev, P)
        if math.isnan(e1):
            h1 = f * h_prev - h_t * h2  # fallback c=0
        else:
            h1 = (1.0 - cfg.lambda_e) * e1 + cfg.lambda_e * e2

    # f1 (contrainte de coherence f1 + f2*h_t = f)
    f1 = f - f2 * h_t

    # b1, b2 (termes de moyenne)
    b1 = sqrt_a_prev - f * sqrt_a_t - f2 * beta_t
    b2 = (
        h_prev * sqrt_a_prev
        + beta_prev
        - h1 * sqrt_a_t
        - h2 * (h_t * sqrt_a_t + beta_t)
    )

    # Covariance Sigma
    sigma11 = Q - r_t * f2 * f2
    sigma12 = (
        f * (f * h_prev - h_t * h2 - h1) * P
        + h_prev * Q
        - h2 * f2 * r_t
    )
    sigma22 = (
        r_prev
        + h_prev * h_prev * Q
        + P * (f * h_prev) ** 2
        - P * (h1 + h_t * h2) ** 2
        - h2 * h2 * r_t
    )

    return {
        "Q": Q, "f": f, "f1": f1, "f2": f2,
        "h_t": h_t, "h_prev": h_prev,
        "beta_t": beta_t, "beta_prev": beta_prev,
        "r_t": r_t, "r_prev": r_prev,
        "h1": h1, "h2": h2,
        "b1": b1, "b2": b2,
        "sigma11": sigma11, "sigma12": sigma12, "sigma22": sigma22,
    }


def count_psd_violations(cfg: MemoryDDIMConfig, abar: torch.Tensor, T: int) -> int:
    """Nombre de pas ou Sigma_t echoue la PSD (scalaire, sans GPU)."""
    ts = _make_timesteps(T=T, S=int(cfg.steps), schedule=cfg.schedule)
    abar = abar.detach().cpu()
    n = 0
    for j in range(len(ts) - 1, 0, -1):
        t = int(ts[j].item())
        tp = int(ts[j - 1].item())
        p = _compute_step(float(abar[t]), float(abar[tp]), cfg)
        det = p["sigma11"] * p["sigma22"] - p["sigma12"] ** 2
        if (
            p["sigma11"] < -cfg.pd_tol
            or p["sigma22"] < -cfg.pd_tol
            or det < -cfg.pd_tol
        ):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class MemoryDDIMSampler:
    """Memory-DDIM sampler (variable auxiliaire y_t, filtrage de Kalman)."""

    def __init__(self, ddpm):
        self.ddpm = ddpm

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        device: torch.device,
        shape: tuple[int, int, int] = (3, 32, 32),
        *,
        cfg: MemoryDDIMConfig | None = None,
        return_aux_state: bool = False,
    ):
        if cfg is None:
            cfg = MemoryDDIMConfig()

        T = int(self.ddpm.timesteps)
        ts = _make_timesteps(T=T, S=int(cfg.steps), schedule=cfg.schedule).to(device)
        abar = self.ddpm.alphas_cumprod.to(device)

        x = torch.randn((batch_size, *shape), device=device)
        m: torch.Tensor | None = None
        P_cov: float = 0.0

        for j in range(len(ts) - 1, 0, -1):
            t = int(ts[j].item())
            t_prev = int(ts[j - 1].item())
            alpha_t = float(abar[t].item())
            alpha_prev = float(abar[t_prev].item())

            # Prediction reseau -> xhat_0 (interface eps)
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            eps = model(x, t_batch)
            sqrt_a_t = torch.sqrt(abar[t]).clamp(min=1e-12)
            sqrt_1m_a_t = torch.sqrt(torch.clamp(1.0 - abar[t], min=0.0))
            x0_hat = (x - sqrt_1m_a_t * eps) / sqrt_a_t
            if cfg.clip_denoised:
                x0_hat = x0_hat.clamp(-1.0, 1.0)

            # Initialisation memoire q(y_T | x_T)
            if m is None:
                sa = math.sqrt(max(alpha_t, 0.0))
                h_T = cfg.lambda_h * sa
                beta_T = cfg.lambda_beta * sa
                r_T = cfg.lambda_r * (1.0 - alpha_t) + cfg.eps_r
                m = h_T * x + beta_T * x0_hat
                P_cov = float(r_T)

            p = _compute_step(alpha_t, alpha_prev, cfg)
            f1, f2, b1 = p["f1"], p["f2"], p["b1"]
            h1, h2, b2 = p["h1"], p["h2"], p["b2"]
            s11, s12, s22 = p["sigma11"], p["sigma12"], p["sigma22"]

            if cfg.verbose_pd_warnings:
                det = s11 * s22 - s12 * s12
                if s11 < -cfg.pd_tol or s22 < -cfg.pd_tol or det < -cfg.pd_tol:
                    print(
                        f"[mem_ddim] Sigma non PSD t={t}->{t_prev}: "
                        f"s11={s11:.3e} s22={s22:.3e} det={det:.3e}"
                    )

            s11 = max(s11, 0.0)
            s22 = max(s22, 0.0)

            # x_{t-1} | x_t:T
            mu_x = f1 * x + f2 * m + b1 * x0_hat
            Vx = max(s11 + f2 * f2 * P_cov, 0.0)
            if Vx > 0.0:
                x_prev = mu_x + math.sqrt(Vx) * torch.randn_like(x)
            else:
                x_prev = mu_x

            # Prediction jointe de y_{t-1}
            mu_y = h1 * x + h2 * m + b2 * x0_hat
            Vy = max(s22 + h2 * h2 * P_cov, 0.0)
            Cyx = s12 + f2 * h2 * P_cov

            # Kalman
            if Vx > cfg.pd_tol:
                K = Cyx / Vx
                m_new = mu_y + K * (x_prev - mu_x)
                P_new = Vy - K * Cyx
            else:
                m_new = mu_y
                P_new = Vy
            P_new = max(P_new, 0.0)

            x = x_prev
            m = m_new
            P_cov = float(P_new)

        if return_aux_state:
            return x, {"m": m, "P": P_cov}
        return x