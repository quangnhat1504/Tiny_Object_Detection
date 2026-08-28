"""NWD — Normalized Wasserstein Distance (Wang et al. 2022).

NWD models both boxes as 2D Gaussians:
    N_p = N(mu_p, Sigma_p)   with mu_p = (xp, yp), Sigma_p = diag(w_p/2, h_p/2)^2
The 2nd Wasserstein distance between two Gaussians:
    W2^2 = ||mu_p - mu_t||^2 + ||Sigma_p^(1/2) - Sigma_t^(1/2)||_F^2
For diagonal Sigma:
    W2^2 = (xp-xt)^2 + (yp-yt)^2 + (wp/2 - wt/2)^2 + (hp/2 - ht/2)^2
Normalize by C (dataset constant):
    NWD = exp(- sqrt(W2^2) / C)

C is the average object size in the dataset (we use median sqrt-area
by default; can be overridden by passing C explicitly).
"""
from __future__ import annotations
import torch

from ..config import METRIC_BETA, NWD_C

EPS = 1e-6


def _wasserstein_sq(xn, yn, wn, hn, xg, yg, wg, hg):
    """W2² between two diagonal Gaussians."""
    dx = xn.unsqueeze(1) - xg.unsqueeze(0)
    dy = yn.unsqueeze(1) - yg.unsqueeze(0)
    dw = (wn.unsqueeze(1) - wg.unsqueeze(0)) / 2.0
    dh = (hn.unsqueeze(1) - hg.unsqueeze(0)) / 2.0
    return dx * dx + dy * dy + dw * dw + dh * dh


def compute_rfd(xn, yn, wn, hn, xg, yg, wg, hg, C: float = NWD_C,
                beta: float = METRIC_BETA, chunk_size: int = 16384, **kwargs) -> torch.Tensor:
    """NWD similarity with memory chunking (Wang et al. 2022)."""
    N = xn.shape[0]
    M = xg.shape[0]
    if N == 0 or M == 0:
        return torch.zeros((N, M), device=xn.device, dtype=xn.dtype)

    if N <= chunk_size:
        w2sq = _wasserstein_sq(xn, yn, wn, hn, xg, yg, wg, hg).clamp(min=0.0)
        w2 = w2sq.sqrt().clamp(max=30.0 / max(beta, EPS))
        return torch.exp(-beta * w2 / max(C, EPS))

    chunks = []
    for i in range(0, N, chunk_size):
        end_idx = min(i + chunk_size, N)
        w2sq_chunk = _wasserstein_sq(
            xn[i:end_idx], yn[i:end_idx], wn[i:end_idx], hn[i:end_idx],
            xg, yg, wg, hg
        ).clamp(min=0.0)
        w2_chunk = w2sq_chunk.sqrt().clamp(max=30.0 / max(beta, EPS))
        chunks.append(torch.exp(-beta * w2_chunk / max(C, EPS)))

    return torch.cat(chunks, dim=0)


name = "nwd"
needs_reliability_thr = False