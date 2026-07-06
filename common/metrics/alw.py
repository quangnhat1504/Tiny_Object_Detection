"""ALW — Anisotropic Log-Wasserstein (proposed metric).

ALW² = (Δx)²/Sx + (Δy)²/Sy + [ln(wn/wg)]² + [ln(hn/hg)]²

where:
  Sx = (wn² + wg²) / 2     ← anisotropic, RMS-squared horizontal
  Sy = (hn² + hg²) / 2     ← anisotropic, RMS-squared vertical

Two improvements over IGWD:
  1. Anisotropic: Sx ≠ Sy → correct for elongated objects
  2. Log-ratio shape: [ln(wn/wg)]² → truly scale-invariant

Three additional mechanisms (the "RG-Robust" part):
  R. Reliability gate: shape weight λ depends on GT size
  G. Charbonnier penalty: smooth L1 for outlier robustness
  Robust: combines both

Variants for ablation:
  alw_full            : all 3 (anisotropic + log + reliability + Charbonnier)
  alw_aniso_only      : anisotropic + log-ratio only (no reliability, no Charbonnier)
  alw_reliability_only: + reliability gate (no Charbonnier)
  alw_charbonnier_only: + Charbonnier (no reliability gate)
"""
from __future__ import annotations
import torch

from ..config import (
    ALW_SHAPE_LAMBDA_MIN, ALW_SHAPE_LAMBDA_POWER,
    ALW_CHARBONNIER_EPS_MIN, ALW_CHARBONNIER_EPS_MAX,
    METRIC_BETA, SA_ALW_LOG_CLAMP,
)

EPS = 1e-6


# ── Core ALW: full version ─────────────────────────────────────────────────
def compute_rfd(xn, yn, wn, hn, xg, yg, wg, hg,
                beta: float = METRIC_BETA,
                reliability_thr: float = 16.0, **kwargs) -> torch.Tensor:
    """ALW (full): anisotropic + log-ratio + reliability + Charbonnier."""
    return _alw(xn, yn, wn, hn, xg, yg, wg, hg,
                beta=beta, reliability_thr=reliability_thr,
                use_reliability=True, use_charbonnier=True)


# ── Ablation 1: anisotropic + log-ratio only (no R, no Charbonnier) ─────
def compute_aniso_only(xn, yn, wn, hn, xg, yg, wg, hg,
                       beta: float = METRIC_BETA, **kwargs) -> torch.Tensor:
    """Pure anisotropic + log-ratio shape (no reliability, no Charbonnier)."""
    return _alw(xn, yn, wn, hn, xg, yg, wg, hg,
                beta=beta, reliability_thr=16.0,
                use_reliability=False, use_charbonnier=False)


# ── Ablation 2: + reliability gate only ───────────────────────────────────
def compute_reliability_only(xn, yn, wn, hn, xg, yg, wg, hg,
                             beta: float = METRIC_BETA,
                             reliability_thr: float = 16.0, **kwargs
                             ) -> torch.Tensor:
    """Anisotropic + log-ratio + reliability gate (no Charbonnier)."""
    return _alw(xn, yn, wn, hn, xg, yg, wg, hg,
                beta=beta, reliability_thr=reliability_thr,
                use_reliability=True, use_charbonnier=False)


# ── Ablation 3: + Charbonnier only ────────────────────────────────────────
def compute_charbonnier_only(xn, yn, wn, hn, xg, yg, wg, hg,
                             beta: float = METRIC_BETA,
                             reliability_thr: float = 16.0, **kwargs
                             ) -> torch.Tensor:
    """Anisotropic + log-ratio + Charbonnier (no reliability gate)."""
    return _alw(xn, yn, wn, hn, xg, yg, wg, hg,
                beta=beta, reliability_thr=reliability_thr,
                use_reliability=False, use_charbonnier=True)


# ── Core implementation ───────────────────────────────────────────────────
def _alw(xn, yn, wn, hn, xg, yg, wg, hg,
         beta: float, reliability_thr: float,
         use_reliability: bool, use_charbonnier: bool) -> torch.Tensor:
    wn = wn.clamp(min=EPS); hn = hn.clamp(min=EPS)
    wg = wg.clamp(min=EPS); hg = hg.clamp(min=EPS)

    # Position (anisotropic)
    dx = xn.unsqueeze(1) - xg.unsqueeze(0)
    dy = yn.unsqueeze(1) - yg.unsqueeze(0)
    Sx = (wn.unsqueeze(1) ** 2 + wg.unsqueeze(0) ** 2) / 2.0
    Sy = (hn.unsqueeze(1) ** 2 + hg.unsqueeze(0) ** 2) / 2.0
    pos_x = dx * dx / Sx.clamp(min=EPS)
    pos_y = dy * dy / Sy.clamp(min=EPS)

    # Shape (log-ratio) — clamp for numerical stability
    log_ratio_w = torch.log(wn.unsqueeze(1) / wg.unsqueeze(0)).abs().clamp(0, SA_ALW_LOG_CLAMP)
    log_ratio_h = torch.log(hn.unsqueeze(1) / hg.unsqueeze(0)).abs().clamp(0, SA_ALW_LOG_CLAMP)

    if use_reliability:
        gt_size = torch.sqrt((wg * hg).clamp(min=EPS)).unsqueeze(0)
        size_gate = (gt_size / reliability_thr).clamp(0.0, 1.0)
        shape_lambda = (
            ALW_SHAPE_LAMBDA_MIN
            + (1.0 - ALW_SHAPE_LAMBDA_MIN) * (size_gate ** ALW_SHAPE_LAMBDA_POWER)
        )
    else:
        shape_lambda = 1.0

    if use_charbonnier:
        if use_reliability:
            robust_eps = (
                ALW_CHARBONNIER_EPS_MIN
                + ALW_CHARBONNIER_EPS_MAX * (1.0 - size_gate)
            )
        else:
            robust_eps = torch.tensor(ALW_CHARBONNIER_EPS_MIN,
                                       dtype=wg.dtype, device=wg.device)
        sw = torch.sqrt(log_ratio_w * log_ratio_w + robust_eps * robust_eps) - robust_eps
        sh = torch.sqrt(log_ratio_h * log_ratio_h + robust_eps * robust_eps) - robust_eps
        shape_term = shape_lambda * (sw + sh)
    else:
        # Pure L1 on log-ratio
        shape_term = shape_lambda * (log_ratio_w + log_ratio_h)

    alw_sq = (pos_x + pos_y + shape_term).clamp(min=0.0)
    alw = alw_sq.sqrt().clamp(max=30.0 / max(beta, EPS))
    return torch.exp(-beta * alw)


name = "alw"
needs_reliability_thr = True