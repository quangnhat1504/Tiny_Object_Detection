"""
Spectral Wavelet-Enhanced Homotopy Wasserstein-IoU (SW-HWIoU) Module.

Exploits high-frequency edge energy from 2D Discrete Wavelet Transform (DWT)
to prevent micro-scale feature gradient collapse:
    rho_HF(F) = (||LH||_2^2 + ||HL||_2^2 + ||HH||_2^2) / (||LL||_2^2 + eps)
    gamma_SW(s, rho) = (s^2 / (s^2 + sigma_0^2)) * (1.0 - alpha * sigmoid(rho_HF))
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torchvision.ops import box_iou

EPS = 1e-6


def haar_wavelet_2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 2D Haar Wavelet Decomposition without trainable parameters.
    Args:
        x: [B, C, H, W]
    Returns:
        LL (Low-Low approx), LH (Horizontal edges), HL (Vertical edges), HH (Diagonal edges)
    """
    x01 = x[:, :, 0::2, :] / 2.0
    x02 = x[:, :, 1::2, :] / 2.0
    x_low = x01 + x02
    x_high = x01 - x02

    ll = x_low[:, :, :, 0::2] + x_low[:, :, :, 1::2]
    lh = x_low[:, :, :, 0::2] - x_low[:, :, :, 1::2]
    hl = x_high[:, :, :, 0::2] + x_high[:, :, :, 1::2]
    hh = x_high[:, :, :, 0::2] - x_high[:, :, :, 1::2]

    return ll, lh, hl, hh


def compute_spectral_high_frequency_ratio(feature_map: torch.Tensor) -> torch.Tensor:
    """Compute spatial spectral high-frequency energy ratio rho_HF in [0, inf).
    Args:
        feature_map: [B, C, H, W]
    Returns:
        rho: [B] scalar spectral ratio
    """
    ll, lh, hl, hh = haar_wavelet_2d(feature_map)
    energy_ll = ll.square().mean(dim=(1, 2, 3)).clamp_min(EPS)
    energy_hf = (lh.square() + hl.square() + hh.square()).mean(dim=(1, 2, 3))
    return (energy_hf / energy_ll).clamp(0.0, 10.0)


def compute_wavelet_spectral_homotopy(
    gt_wh: torch.Tensor,
    sigma_0: float = 8.0,
    spectral_rho: torch.Tensor | float = 0.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Compute Wavelet Spectral Homotopy Parameter gamma_SW in [0, 1]."""
    safe_wh = gt_wh.clamp_min(EPS)
    s = torch.sqrt(safe_wh[:, 0] * safe_wh[:, 1])
    s_sq = s.square()
    sigma_sq = float(sigma_0) ** 2

    gamma_base = s_sq / (s_sq + sigma_sq)

    if isinstance(spectral_rho, torch.Tensor):
        mod = 1.0 - alpha * torch.sigmoid(spectral_rho.to(gt_wh.device, gt_wh.dtype))
    elif float(spectral_rho) > 0.0:
        mod = 1.0 - alpha * (1.0 / (1.0 + math.exp(-float(spectral_rho))))
    else:
        mod = 1.0

    gamma_sw = gamma_base * mod
    return gamma_sw.clamp(0.0, 1.0)


def aligned_wavelet_spectral_h_wiou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    sigma_0: float = 8.0,
    spectral_rho: torch.Tensor | float = 0.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Compute aligned Spectral Wavelet H-WIoU loss."""
    if pred_boxes.ndim != 2 or pred_boxes.shape[-1] != 4:
        raise ValueError("pred_boxes must have shape [N, 4]")
    if target_boxes.ndim != 2 or target_boxes.shape[-1] != 4:
        raise ValueError("target_boxes must have shape [N, 4]")

    xa = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
    ya = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0
    wa = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=EPS)
    ha = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=EPS)

    xb = (target_boxes[:, 0] + target_boxes[:, 2]) / 2.0
    yb = (target_boxes[:, 1] + target_boxes[:, 3]) / 2.0
    wb = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=EPS)
    hb = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=EPS)

    dx = xa - xb
    dy = ya - yb
    sx = (wa.square() + wb.square()) / 2.0
    sy = (ha.square() + hb.square()) / 2.0
    position_term = (dx.square() / sx.clamp_min(EPS)) + (dy.square() / sy.clamp_min(EPS))
    log_w = torch.log(wa) - torch.log(wb)
    log_h = torch.log(ha) - torch.log(hb)
    shape_term = log_w.square() + log_h.square()
    dw_sq = (position_term + shape_term).clamp_min(0.0)

    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    union = (wa * ha + wb * hb - inter).clamp_min(EPS)
    iou = (inter / union).clamp(min=EPS, max=1.0)

    target_wh = torch.stack([wb, hb], dim=1)
    gamma = compute_wavelet_spectral_homotopy(target_wh, sigma_0=sigma_0, spectral_rho=spectral_rho, alpha=alpha)

    iou_part = torch.pow(iou, gamma)
    ot_part = torch.exp(-(1.0 - gamma) * dw_sq)
    similarity = (iou_part * ot_part).clamp(0.0, 1.0)

    return 1.0 - similarity
