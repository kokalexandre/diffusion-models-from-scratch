"""Augmented DDIM sampler with auxiliary variable y_t.

L'etat augmente est z_t = (x_t, y_t). La transition est lineaire-gaussienne :

    x_{t-1} = f^1_t * x_t + f^2_t * y_t + b^1_t * xhat_0 + xi^x_t
    y_{t-1} = h^1_t * x_t + h^2_t * y_t + b^2_t * xhat_0 + xi^y_t
    (xi^x_t, xi^y_t) ~ N(0, Sigma_t)

Les coefficients sont choisis pour preserver exactement la marginale DDIM
sur x_t et la forme z_t | x_0 ~ N(m_t(x_0), C_t) attendue.
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
class DDIMAugmentedConfig:

    # Trajectoire de timesteps (identique a DDIM standard)
    steps: int = 50
    schedule: Literal["linear", "quadratic"] = "quadratic"

    # Parametre DDIM de stochasticite (controle q_t = sigma_t^2)
    eta: float = 0.2

    # Parametrisation de y_t | x_t, x_0 ~ N(h_t x_t + beta_t x_0, r_t I)
    # (PJ section 8)
    lambda_h: float = 0.5      # h_t = lambda_h
    lambda_beta: float = 0.5   # beta_t = lambda_beta * sqrt(alpha_t)
    lambda_r: float = 0.05     # r_t = lambda_r * (1 - alpha_t) + eps_r
    eps_r: float = 1e-3        # plancher de r_t (sinon r_0 = 0)

    # Cas general. 0 = cas simple recommande.
    lambda_c: float = 0.0      # c_t = lambda_c   (defaut d'alignement)
    lambda_h2: float = 0.0     # h^2_t = lambda_h2 (couplage y_t -> y_{t-1})

    # Coefficient f^2_t = rho_g * f^2_t_max, avec 0 <= rho_g < 1
    # (f^2_t_max = borne assurant det(Sigma_t) >= 0)
    rho_g: float = 0.5

    # Clippage de xhat_0 a [-1, 1] (comme dans DDIM standard)
    clip_denoised: bool = True

    # Verifications de PSD
    pd_tol: float = 1e-8
    check_pd: bool = True
    verbose_pd_warnings: bool = False


# ---------------------------------------------------------------------------
# Helpers (scalaires)
# ---------------------------------------------------------------------------


def _aux_params(alpha_t: float, cfg: DDIMAugmentedConfig) -> tuple[float, float, float]:
    """Retourne (h_t, beta_t, r_t) au timestep correspondant a alpha_t.
 
        h_t    = lambda_h
        beta_t = lambda_beta * sqrt(alpha_t)
        r_t    = lambda_r * (1 - alpha_t) + eps_r
    """
    a = max(alpha_t, 0.0)
    h_t = cfg.lambda_h
    beta_t = cfg.lambda_beta * math.sqrt(a)
    r_t = cfg.lambda_r * (1.0 - alpha_t) + cfg.eps_r
    return h_t, beta_t, r_t


def _ddim_eta_sigma2(alpha_t: float, alpha_prev: float, eta: float) -> float:

    if eta <= 0.0:
        return 0.0
    ratio = (1.0 - alpha_prev) / max(1.0 - alpha_t, 1e-20)
    fac = 1.0 - alpha_t / max(alpha_prev, 1e-20)
    inner = max(ratio * fac, 0.0)
    return (eta ** 2) * inner


def _f2_upper_bound(
    q_t: float,
    r_t: float,
    r_tprev: float,
    h_tprev: float,
    h2: float,
    c: float,
    f_t: float,
    one_minus_alpha_t: float,
    sigma22: float,
) -> float:
    """Borne superieure sur f^2_t telle que det(Sigma_t) >= 0.

    Cas simple (c = 0, h^2_t = 0) -> formule fermee PJ section 8 :

        f^2_t_max = sqrt( q_t * r_{t-1} / (r_t * (r_{t-1} + h_{t-1}^2 * q_t)) ).

    Cas general : on resout la quadratique en f^2_t obtenue depuis
    det(Sigma_t) = sigma11 * sigma22 - sigma12^2 >= 0,
    en notant sigma11 = q_t - f^2_t^2 * r_t, sigma12 = B0 - f^2_t * B1
    avec B0 = f_t * c * (1 - alpha_t) + h_{t-1} * q_t, B1 = h^2_t * r_t.
    On retourne la racine superieure.
    """
    A = sigma22
    if A <= 0.0:
        return 0.0

    B0 = f_t * c * one_minus_alpha_t + h_tprev * q_t
    B1 = h2 * r_t

    denom = r_t * A + B1 * B1
    if denom <= 0.0:
        return 0.0

    disc = A * (q_t * denom - r_t * B0 * B0)
    if disc < 0.0:
        # det(Sigma_t) ne peut pas etre rendu positif -> on prend f^2_t = 0
        return 0.0

    return (B0 * B1 + math.sqrt(disc)) / denom


def _compute_step_params(
    alpha_t: float,
    alpha_prev: float,
    cfg: DDIMAugmentedConfig,
) -> dict:
    """Calcule tous les coefficients du pas augmente t -> t_prev.

    Retourne un dict de scalaires Python suivant les notations de la PJ.
    """
    # ---- DDIM de reference (sections 2 et 5) ----
    q_t = _ddim_eta_sigma2(alpha_t, alpha_prev, cfg.eta)
    one_minus_alpha_t = max(1.0 - alpha_t, 0.0)
    one_minus_alpha_prev = max(1.0 - alpha_prev, 0.0)

    sqrt_1m_at = math.sqrt(one_minus_alpha_t)
    f_t = math.sqrt(max(one_minus_alpha_prev - q_t, 0.0)) / max(sqrt_1m_at, 1e-20)
    sqrt_a_t = math.sqrt(max(alpha_t, 0.0))
    sqrt_a_prev = math.sqrt(max(alpha_prev, 0.0))
    d_t = sqrt_a_prev - f_t * sqrt_a_t

    # ---- Parametres auxiliaires (sections 3 et 8) ----
    h_t, beta_t, r_t = _aux_params(alpha_t, cfg)
    h_tprev, beta_tprev, r_tprev = _aux_params(alpha_prev, cfg)

    # ---- Parametres libres generaux ----
    c = cfg.lambda_c
    h2 = cfg.lambda_h2

    # sigma22 ne depend pas de f^2_t (PJ section 6.3)
    sigma22 = (
        r_tprev
        + (h_tprev ** 2) * q_t
        + (2.0 * h_tprev * f_t * c - c * c) * one_minus_alpha_t
        - (h2 ** 2) * r_t
    )

    # Borne sur f^2_t puis choix f^2_t = rho_g * f^2_t_max
    f2_max = _f2_upper_bound(
        q_t=q_t,
        r_t=r_t,
        r_tprev=r_tprev,
        h_tprev=h_tprev,
        h2=h2,
        c=c,
        f_t=f_t,
        one_minus_alpha_t=one_minus_alpha_t,
        sigma22=sigma22,
    )
    f2 = cfg.rho_g * f2_max

    # ---- Coefficients imposes (PJ section 7.2) ----
    # h^1_t = h_{t-1} * f_t - h^2_t * h_t - c_t
    h1 = h_tprev * f_t - h2 * h_t - c
    # f^1_t = f_t - f^2_t * h_t
    f1 = f_t - f2 * h_t
    # b^1_t = d_t - f^2_t * beta_t
    b1 = d_t - f2 * beta_t
    # b^2_t = c_t * sqrt(alpha_t) + h_{t-1} * d_t + beta_{t-1} - h^2_t * beta_t
    b2 = c * sqrt_a_t + h_tprev * d_t + beta_tprev - h2 * beta_t

    # ---- Bruit de transition Sigma_t (PJ sections 5 et 6) ----
    sigma11 = q_t - (f2 ** 2) * r_t
    sigma12 = f_t * c * one_minus_alpha_t + h_tprev * q_t - f2 * h2 * r_t

    return {
        # DDIM
        "q_t": q_t,
        "f_t": f_t,
        "d_t": d_t,
        "sqrt_a_t": sqrt_a_t,
        "sqrt_a_prev": sqrt_a_prev,
        # Aux courant et precedent
        "h_t": h_t,
        "beta_t": beta_t,
        "r_t": r_t,
        "h_tprev": h_tprev,
        "beta_tprev": beta_tprev,
        "r_tprev": r_tprev,
        # Libres
        "c": c,
        "h2": h2,
        "f2": f2,
        "f2_max": f2_max,
        # Imposes
        "h1": h1,
        "f1": f1,
        "b1": b1,
        "b2": b2,
        # Covariance
        "sigma11": sigma11,
        "sigma12": sigma12,
        "sigma22": sigma22,
    }


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class DDIMAugmentedSampler:
    """Sampler DDIM augmente avec variable auxiliaire y_t.

    Le reseau de debruitage `model` n'est pas modifie : il est utilise
    exactement comme dans DDIM pour predire epsilon (et donc xhat_0).
    La variable auxiliaire y_t est integree sous forme de loi
    conditionnelle gaussienne q(y_t | x_t:T) = N(m_t, P_t I) maintenue
    le long de la trajectoire, et mise a jour par filtrage de Kalman.
    """

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
        cfg: DDIMAugmentedConfig | None = None,
        return_aux_state: bool = False,
        return_timesteps: bool = False,
    ):
        if cfg is None:
            cfg = DDIMAugmentedConfig()

        T = int(self.ddpm.timesteps)
        ts = _make_timesteps(T=T, S=int(cfg.steps), schedule=cfg.schedule).to(device)
        abar = self.ddpm.alphas_cumprod.to(device)

        x = torch.randn((batch_size, *shape), device=device)
        m: torch.Tensor | None = None
        P: float = 0.0                 

        # On parcourt j = S, S-1, ..., 1. A j=1 on echantillonne x_0
        # via la transition ; on n'appelle PAS le reseau en t=0.
        for j in range(len(ts) - 1, 0, -1):
            t = int(ts[j].item())
            t_prev = int(ts[j - 1].item())

            alpha_t = float(abar[t].item())
            alpha_prev = float(abar[t_prev].item())

            # ---- Prediction de xhat_0 par le reseau (DDIM) ----
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            eps = model(x, t_batch)
            sqrt_a_t_buf = torch.sqrt(abar[t]).clamp(min=1e-12)
            sqrt_1m_a_t_buf = torch.sqrt(torch.clamp(1.0 - abar[t], min=0.0))
            x0_pred = (x - sqrt_1m_a_t_buf * eps) / sqrt_a_t_buf
            if cfg.clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            # ---- Initialisation de q(y_T | x_T) (PJ section 11.1) ----
            if m is None:
                h_T, beta_T, r_T = _aux_params(alpha_t, cfg)
                m = h_T * x + beta_T * x0_pred
                P = float(r_T)

            # ---- Coefficients du pas t -> t_prev ----
            p = _compute_step_params(alpha_t, alpha_prev, cfg)
            f1, f2, b1 = p["f1"], p["f2"], p["b1"]
            h1, h2, b2 = p["h1"], p["h2"], p["b2"]
            sigma11, sigma12, sigma22 = p["sigma11"], p["sigma12"], p["sigma22"]

            # ---- Verification PSD (PJ section 7.3) ----
            det_sigma = sigma11 * sigma22 - sigma12 * sigma12
            if cfg.check_pd and cfg.verbose_pd_warnings:
                if (
                    sigma11 < -cfg.pd_tol
                    or sigma22 < -cfg.pd_tol
                    or det_sigma < -cfg.pd_tol
                ):
                    print(
                        f"[ddim_aug] Sigma non PSD a t={t} -> t_prev={t_prev}: "
                        f"s11={sigma11:.3e}, s22={sigma22:.3e}, "
                        f"det={det_sigma:.3e}, f2={f2:.3e}"
                    )

            sigma11 = max(sigma11, 0.0)
            sigma22 = max(sigma22, 0.0)

            # ---- Marginalisation de y_t pour generer x_{t-1} (PJ 11.3) ----
            # x_{t-1} | x_t:T ~ N(mu_x, V_x I)
            mu_x = f1 * x + f2 * m + b1 * x0_pred
            V_x = sigma11 + (f2 ** 2) * P
            V_x = max(V_x, 0.0)

            if V_x > 0.0:
                x_prev = mu_x + math.sqrt(V_x) * torch.randn_like(x)
            else:
                x_prev = mu_x

            # ---- Prediction jointe de y_{t-1} (PJ 11.4) ----
            mu_y = h1 * x + h2 * m + b2 * x0_pred
            V_y = sigma22 + (h2 ** 2) * P
            V_y = max(V_y, 0.0)
            C_yx = sigma12 + f2 * h2 * P

            # ---- Mise a jour de Kalman (PJ 11.5) ----
            if V_x > cfg.pd_tol:
                K = C_yx / V_x
                m_new = mu_y + K * (x_prev - mu_x)
                P_new = V_y - K * C_yx
            else:
                # Cas degenere (q_t = 0 et f^2_t = 0) : x_{t-1} = mu_x
                # n'apporte aucune information sur y_{t-1}.
                m_new = mu_y
                P_new = V_y

            P_new = max(P_new, 0.0)

            # ---- Iteration ----
            x = x_prev
            m = m_new
            P = float(P_new)

        # Sortie : x = x_{t_0} = x_0 echantillonne via la transition

        out = (x,)
        if return_aux_state:
            out = out + ({"m": m, "P": P},)
        if return_timesteps:
            out = out + (ts.detach().cpu(),)
        return out[0] if len(out) == 1 else out