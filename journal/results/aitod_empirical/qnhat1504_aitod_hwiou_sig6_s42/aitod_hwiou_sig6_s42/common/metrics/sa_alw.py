"""
SA-ALW — Scale-Adaptive Anisotropic Log-Wasserstein [Proposal — chưa công bố]

SA-ALW = ALW + 2 cơ chế scale-adaptive độc lập:

(a) Scale-Adaptive beta(s): object càng nhỏ → beta càng lớn → similarity giảm
    nhanh hơn khi lệch vị trí. Tận dụng insight từ IGWD paper (beta=8 cho AP
    chung, beta=10 cho AP_vt) nhưng biến thành hàm liên tục theo scale.

(b) Scale-Adaptive position weight w_pos(s): object càng nhỏ → trọng số
    position term càng lớn. Lý do: shape term không đáng tin cho object siêu
    nhỏ (2-3px nhiễu annotation đã đổi hẳn aspect ratio), position term vẫn
    còn ý nghĩa.

Công thức:
    ALW²(s) = w_pos(s) · [pos_term] + [shape_term]
    SA_ALW_sim = exp(-beta(s) · sqrt(ALW²))

[Unverified/Chưa công bố] — đây là công trình riêng, không được trình bày
như kết quả đã qua peer-review trong bất kỳ tài liệu phái sinh nào.
"""
from __future__ import annotations

import torch

from ..config import (
    ALW_SHAPE_LAMBDA_MIN, ALW_SHAPE_LAMBDA_POWER,
    ALW_CHARBONNIER_EPS_MIN, ALW_CHARBONNIER_EPS_MAX,
    METRIC_BETA,
    SA_ALW_BETA_MIN, SA_ALW_BETA_MAX,
    SA_ALW_S_MIN, SA_ALW_S_MAX,
    SA_ALW_POS_WEIGHT_MIN, SA_ALW_POS_WEIGHT_MAX,
    SA_ALW_LOG_CLAMP,
)

EPS = 1e-6


# ── Scale-Adaptive beta ─────────────────────────────────────────────────────
def scale_adaptive_beta(gt_wh: torch.Tensor,
                         s_min: float = SA_ALW_S_MIN,
                         s_max: float = SA_ALW_S_MAX,
                         beta_min: float = SA_ALW_BETA_MIN,
                         beta_max: float = SA_ALW_BETA_MAX) -> torch.Tensor:
    """Object càng nhỏ → beta càng lớn.

    Args:
        gt_wh: (M, 2) tensor of GT widths and heights
        s_min, s_max: percentile P10/P90 from dataset
        beta_min: beta cho object lớn nhất
        beta_max: beta cho object nhỏ nhất
    Returns:
        (M,) tensor of per-GT beta values
    """
    scale = torch.sqrt((gt_wh[:, 0] * gt_wh[:, 1]).clamp(min=EPS))
    t = ((scale - s_min) / (s_max - s_min)).clamp(0.0, 1.0)
    return beta_max - (beta_max - beta_min) * t


# ── Scale-Adaptive position weight ──────────────────────────────────────────
def scale_adaptive_pos_weight(gt_wh: torch.Tensor,
                               s_min: float = SA_ALW_S_MIN,
                               s_max: float = SA_ALW_S_MAX,
                               w_min: float = SA_ALW_POS_WEIGHT_MIN,
                               w_max: float = SA_ALW_POS_WEIGHT_MAX) -> torch.Tensor:
    """Object càng nhỏ → w_pos càng lớn (position term được ưu tiên hơn).

    Args:
        gt_wh: (M, 2) tensor of GT widths and heights
        w_min: weight cho object lớn (≈1.0)
        w_max: weight cho object siêu nhỏ (≈1.5)
    Returns:
        (M,) tensor of per-GT position weights
    """
    scale = torch.sqrt((gt_wh[:, 0] * gt_wh[:, 1]).clamp(min=EPS))
    t = ((scale - s_min) / (s_max - s_min)).clamp(0.0, 1.0)
    return w_max - (w_max - w_min) * t


# ── SA-ALW Full (Scale-Adaptive beta + position weight) ──────────────────────
def compute_rfd(xn, yn, wn, hn, xg, yg, wg, hg,
                beta: float = METRIC_BETA,
                reliability_thr: float = 16.0,
                use_sa_beta: bool = True,
                use_sa_pos_weight: bool = True,
                **kwargs) -> torch.Tensor:
    """SA-ALW (full): ALW + Scale-Adaptive beta + Scale-Adaptive position weight.

    Nếu cả 2 SA cơ chế đều tắt, fallback về ALW-full.
    """
    wn = wn.clamp(min=EPS); hn = hn.clamp(min=EPS)
    wg = wg.clamp(min=EPS); hg = hg.clamp(min=EPS)

    # ── Position (anisotropic, giống ALW) ──
    N = xn.shape[0]
    M = xg.shape[0]
    dx = xn.unsqueeze(1) - xg.unsqueeze(0)   # (N, M)
    dy = yn.unsqueeze(1) - yg.unsqueeze(0)
    Sx = (wn.unsqueeze(1) ** 2 + wg.unsqueeze(0) ** 2) / 2.0  # (N, M)
    Sy = (hn.unsqueeze(1) ** 2 + hg.unsqueeze(0) ** 2) / 2.0
    pos_x = dx * dx / Sx.clamp(min=EPS)
    pos_y = dy * dy / Sy.clamp(min=EPS)

    # ── Shape (log-ratio, giống ALW) ──
    log_ratio_w = torch.log(wn.unsqueeze(1) / wg.unsqueeze(0))
    log_ratio_h = torch.log(hn.unsqueeze(1) / hg.unsqueeze(0))

    # Clamp for numerical stability (H2.4)
    log_ratio_w = log_ratio_w.clamp(-SA_ALW_LOG_CLAMP, SA_ALW_LOG_CLAMP)
    log_ratio_h = log_ratio_h.clamp(-SA_ALW_LOG_CLAMP, SA_ALW_LOG_CLAMP)

    log_ratio_w_sq = log_ratio_w * log_ratio_w
    log_ratio_h_sq = log_ratio_h * log_ratio_h

    # ── Reliability gate (giống ALW) ──
    gt_size = torch.sqrt((wg * hg).clamp(min=EPS)).unsqueeze(0)  # (1, M)
    size_gate = (gt_size / reliability_thr).clamp(0.0, 1.0)
    shape_lambda = (
        ALW_SHAPE_LAMBDA_MIN
        + (1.0 - ALW_SHAPE_LAMBDA_MIN) * (size_gate ** ALW_SHAPE_LAMBDA_POWER)
    )  # (1, M)

    # Charbonnier robust loss (always adaptive — SA-ALW always has reliability gate)
    robust_eps = (
        ALW_CHARBONNIER_EPS_MIN
        + ALW_CHARBONNIER_EPS_MAX * (1.0 - size_gate)
    )

    sw = torch.sqrt(log_ratio_w_sq + robust_eps * robust_eps) - robust_eps
    sh = torch.sqrt(log_ratio_h_sq + robust_eps * robust_eps) - robust_eps
    shape_term = shape_lambda * (sw + sh)  # (N, M)

    # ── Scale-Adaptive position weight ──
    if use_sa_pos_weight:
        w_pos = scale_adaptive_pos_weight(
            torch.stack([wg, hg], dim=1)  # (M, 2)
        ).unsqueeze(0)  # (1, M)
    else:
        w_pos = 1.0

    # ── ALW² weighted ──
    alw_sq = (w_pos * (pos_x + pos_y) + shape_term).clamp(min=0.0)
    alw = alw_sq.sqrt()

    # ── Scale-Adaptive beta ──
    if use_sa_beta:
        gt_wh = torch.stack([wg, hg], dim=1)  # (M, 2)
        per_gt_beta = scale_adaptive_beta(gt_wh).unsqueeze(0)  # (1, M)
        beta_val = float(per_gt_beta.min())
    else:
        per_gt_beta = beta
        beta_val = beta

    # Clamp alw để tránh underflow trong exp
    alw = alw.clamp(max=30.0 / max(beta_val, EPS))
    return torch.exp(-per_gt_beta * alw)


# ── Ablation: SA-beta only (no SA-position-weight) ─────────────────────────
def compute_sa_beta_only(xn, yn, wn, hn, xg, yg, wg, hg,
                          beta: float = METRIC_BETA,
                          reliability_thr: float = 16.0,
                          **kwargs) -> torch.Tensor:
    """ALW + Scale-Adaptive beta (w_pos=1.0 cố định). Tương ứng Phase 2.7."""
    return compute_rfd(xn, yn, wn, hn, xg, yg, wg, hg,
                       beta=beta, reliability_thr=reliability_thr,
                       use_sa_beta=True, use_sa_pos_weight=False)


# ── Ablation: SA-position-weight only ──────────────────────────────────────
def compute_sa_pos_only(xn, yn, wn, hn, xg, yg, wg, hg,
                         beta: float = METRIC_BETA,
                         reliability_thr: float = 16.0,
                         **kwargs) -> torch.Tensor:
    """ALW + Scale-Adaptive position weight (beta cố định)."""
    return compute_rfd(xn, yn, wn, hn, xg, yg, wg, hg,
                       beta=beta, reliability_thr=reliability_thr,
                       use_sa_beta=False, use_sa_pos_weight=True)


name = "sa_alw"
needs_reliability_thr = True
