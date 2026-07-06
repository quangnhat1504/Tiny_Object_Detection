"""IGWD — Improved Gaussian Wasserstein Distance (paper IGWD).

From paper:
    W2^2(N_p, N_t) = (xp-xt)^2 + (yp-yt)^2 + ((wp-wt)/2)^2 + ((hp-ht)/2)^2
    S(Op, Ot) = wp*hp + wt*ht
    IGWD = sqrt(W2 / S)
    sim = exp(-beta * sqrt(IGWD))     (Equation 8)

Plus an ablation variant: IGWD + reliability-gated shape (so we can
prove the reliability gate alone helps).
"""
from __future__ import annotations
import torch

from ..config import (
    METRIC_BETA, SA_ALW_LOG_CLAMP,
    ALW_SHAPE_LAMBDA_MIN, ALW_SHAPE_LAMBDA_POWER,
    ALW_CHARBONNIER_EPS_MIN, ALW_CHARBONNIER_EPS_MAX,
)

EPS = 1e-6


def _wasserstein_sq(xn, yn, wn, hn, xg, yg, wg, hg):
    dx = xn.unsqueeze(1) - xg.unsqueeze(0)
    dy = yn.unsqueeze(1) - yg.unsqueeze(0)
    dw = (wn.unsqueeze(1) - wg.unsqueeze(0)) / 2.0
    dh = (hn.unsqueeze(1) - hg.unsqueeze(0)) / 2.0
    return dx * dx + dy * dy + dw * dw + dh * dh


def _area_sum(wn, hn, wg, hg):
    return wn.unsqueeze(1) * hn.unsqueeze(1) + wg.unsqueeze(0) * hg.unsqueeze(0)


def compute_rfd(xn, yn, wn, hn, xg, yg, wg, hg,
                beta: float = METRIC_BETA, **kwargs) -> torch.Tensor:
    """IGWD similarity, Eq (8) of paper IGWD.

    IGWD = sqrt(W2^2 / S)
    sim  = exp(-beta * sqrt(IGWD))
    """
    wn = wn.clamp(min=EPS); hn = hn.clamp(min=EPS)
    wg = wg.clamp(min=EPS); hg = hg.clamp(min=EPS)
    w2sq = _wasserstein_sq(xn, yn, wn, hn, xg, yg, wg, hg).clamp(min=0.0)
    S = _area_sum(wn, hn, wg, hg).clamp(min=EPS)
    igwd = (w2sq / S).clamp(min=0.0)
    igwd = igwd.clamp(max=30.0 / max(beta, EPS))
    return torch.exp(-beta * igwd)


# ── Ablation: IGWD + reliability gate ────────────────────────────────────
def compute_with_reliability(xn, yn, wn, hn, xg, yg, wg, hg,
                             beta: float = METRIC_BETA,
                             reliability_thr: float = 16.0, **kwargs
                             ) -> torch.Tensor:
    """IGWD where the 'shape' term is reliability-gated.

    We don't change W2/S (those are fixed by paper), but instead
    multiply the (effectively shape-related) distance by a small
    reliability-gated factor when gt is very tiny.

    NOTE: For fair comparison, we keep the paper formula intact and
    just add a Charbonnier + gate on the relative-size deviation term.
    """
    wn = wn.clamp(min=EPS); hn = hn.clamp(min=EPS)
    wg = wg.clamp(min=EPS); hg = hg.clamp(min=EPS)

    # Position part
    dx = xn.unsqueeze(1) - xg.unsqueeze(0)
    dy = yn.unsqueeze(1) - yg.unsqueeze(0)
    dw = (wn.unsqueeze(1) - wg.unsqueeze(0)) / 2.0
    dh = (hn.unsqueeze(1) - hg.unsqueeze(0)) / 2.0
    pos = dx * dx + dy * dy

    # Shape part with reliability gate (rescales shape contribution)
    gt_size = torch.sqrt((wg * hg).clamp(min=EPS)).unsqueeze(0)
    size_gate = (gt_size / reliability_thr).clamp(0.0, 1.0)
    shape_lambda = (
        ALW_SHAPE_LAMBDA_MIN
        + (1.0 - ALW_SHAPE_LAMBDA_MIN) * (size_gate ** ALW_SHAPE_LAMBDA_POWER)
    )
    robust_eps = (
        ALW_CHARBONNIER_EPS_MIN
        + ALW_CHARBONNIER_EPS_MAX * (1.0 - size_gate)
    )

    # Paper shape term (relative size deviation, no log)
    sw = torch.sqrt(dw * dw + robust_eps * robust_eps) - robust_eps
    sh = torch.sqrt(dh * dh + robust_eps * robust_eps) - robust_eps
    shape = shape_lambda * (sw + sh)

    w2sq = (pos + shape).clamp(min=0.0)
    S = _area_sum(wn, hn, wg, hg).clamp(min=EPS)
    igwd = (w2sq / S).clamp(min=0.0).clamp(max=30.0 / max(beta, EPS))
    return torch.exp(-beta * igwd)


# ── Ablation: IGWD + log-ratio shape (Phase 2.4) ────────────────────────
def compute_log_shape(xn, yn, wn, hn, xg, yg, wg, hg,
                      beta: float = METRIC_BETA, **kwargs) -> torch.Tensor:
    """IGWD position (isotropic) + ALW log-ratio shape.

    Position: same as IGWD — normalized by S = wp·hp + wt·ht
    Shape: log-ratio from ALW — [ln(wp/wt)]² + [ln(hp/ht)]²
    """
    wn = wn.clamp(min=EPS); hn = hn.clamp(min=EPS)
    wg = wg.clamp(min=EPS); hg = hg.clamp(min=EPS)

    dx = xn.unsqueeze(1) - xg.unsqueeze(0)
    dy = yn.unsqueeze(1) - yg.unsqueeze(0)

    log_w = torch.log(wn.unsqueeze(1) / wg.unsqueeze(0)).clamp(-SA_ALW_LOG_CLAMP, SA_ALW_LOG_CLAMP)
    log_h = torch.log(hn.unsqueeze(1) / hg.unsqueeze(0)).clamp(-SA_ALW_LOG_CLAMP, SA_ALW_LOG_CLAMP)

    # Log-shape replaces (dw²+dh²)/4 in W2²
    w2sq = (dx*dx + dy*dy + log_w*log_w + log_h*log_h).clamp(min=0.0)
    S = _area_sum(wn, hn, wg, hg).clamp(min=EPS)
    igwd = (w2sq / S).clamp(min=0.0).clamp(max=30.0 / max(beta, EPS))
    return torch.exp(-beta * igwd)


# ── Ablation: IGWD + anisotropic position normalization (Phase 2.5) ────
def compute_anisotropic_s(xn, yn, wn, hn, xg, yg, wg, hg,
                          beta: float = METRIC_BETA, **kwargs) -> torch.Tensor:
    """ALW anisotropic position + IGWD Euclidean shape.

    Position: anisotropic — dx²/Sx + dy²/Sy (like ALW)
    Shape: Euclidean — (dw²+dh²)/4 (like IGWD, no S normalization)
    """
    wn = wn.clamp(min=EPS); hn = hn.clamp(min=EPS)
    wg = wg.clamp(min=EPS); hg = hg.clamp(min=EPS)

    dx = xn.unsqueeze(1) - xg.unsqueeze(0)
    dy = yn.unsqueeze(1) - yg.unsqueeze(0)
    dw = (wn.unsqueeze(1) - wg.unsqueeze(0)) / 2.0
    dh = (hn.unsqueeze(1) - hg.unsqueeze(0)) / 2.0

    # Anisotropic denominators (like ALW)
    Sx = (wn.unsqueeze(1)**2 + wg.unsqueeze(0)**2) / 2.0
    Sy = (hn.unsqueeze(1)**2 + hg.unsqueeze(0)**2) / 2.0
    pos_x = dx*dx / Sx.clamp(min=EPS)
    pos_y = dy*dy / Sy.clamp(min=EPS)

    # Euclidean shape (like IGWD, no global S normalization)
    igwd_sq = (pos_x + pos_y + dw*dw + dh*dh).clamp(min=0.0)
    igwd = igwd_sq.sqrt().clamp(max=30.0 / max(beta, EPS))
    return torch.exp(-beta * igwd)


name = "igwd"
needs_reliability_thr = False