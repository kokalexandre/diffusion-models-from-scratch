from __future__ import annotations

import math
import torch


class StreamingMeanCov:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n = 0
        self.sum_x = torch.zeros(self.dim, dtype=torch.float64)
        self.sum_xx = torch.zeros(self.dim, self.dim, dtype=torch.float64)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.detach().to("cpu", dtype=torch.float64) 
        b, d = x.shape
        assert d == self.dim
        self.n += int(b)
        self.sum_x += x.sum(dim=0)
        self.sum_xx += x.t().mm(x)

    @torch.no_grad()
    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.n > 0
        mu = self.sum_x / self.n
        exx = self.sum_xx / self.n
        cov = exx - torch.outer(mu, mu)
        cov = 0.5 * (cov + cov.t())  
        return mu, cov


@torch.no_grad()
def _sqrtm_psd(mat: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Matrix square root for symmetric PSD matrix using eigendecomposition."""
    mat = 0.5 * (mat + mat.t())
    vals, vecs = torch.linalg.eigh(mat)
    vals = torch.clamp(vals, min=eps)
    sqrt_vals = torch.sqrt(vals)
    return (vecs * sqrt_vals.unsqueeze(0)) @ vecs.t()


@torch.no_grad()
def frechet_distance(mu1: torch.Tensor, cov1: torch.Tensor, mu2: torch.Tensor, cov2: torch.Tensor) -> float:
    mu1 = mu1.to(dtype=torch.float64, device="cpu")
    mu2 = mu2.to(dtype=torch.float64, device="cpu")
    cov1 = cov1.to(dtype=torch.float64, device="cpu")
    cov2 = cov2.to(dtype=torch.float64, device="cpu")

    diff = mu1 - mu2
    diff_sq = float(diff.dot(diff).item())

    s1 = _sqrtm_psd(cov1)
    mid = s1 @ cov2 @ s1
    mid = 0.5 * (mid + mid.t())
    s_mid = _sqrtm_psd(mid)

    tr = float(torch.trace(cov1 + cov2 - 2.0 * s_mid).item())
    return diff_sq + tr


class StreamingInceptionScore:
    def __init__(self, num_classes: int, eps: float = 1e-10):
        self.num_classes = int(num_classes)
        self.eps = float(eps)
        self.n = 0
        self.sum_p = torch.zeros(self.num_classes, dtype=torch.float64)
        self.sum_p_log_p = 0.0

    @torch.no_grad()
    def update_logits(self, logits: torch.Tensor):
        probs = torch.softmax(logits.detach(), dim=1).to("cpu", dtype=torch.float64)
        self.n += int(probs.shape[0])
        self.sum_p += probs.sum(dim=0)
        self.sum_p_log_p += float((probs * torch.log(probs + self.eps)).sum().item())

    @torch.no_grad()
    def finalize(self) -> float:
        assert self.n > 0
        p_bar = self.sum_p / self.n
        term1 = self.sum_p_log_p / self.n
        term2 = float((p_bar * torch.log(p_bar + self.eps)).sum().item())
        avg_kl = term1 - term2
        return float(math.exp(avg_kl))