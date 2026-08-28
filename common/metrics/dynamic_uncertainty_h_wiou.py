"""
Dynamic Uncertainty-Aware Homotopy Wasserstein-IoU (DU-HWIoU) Module.

Formulates instance-conditioned adaptive scale homotopy parameter sigma_0(z):
    sigma_0(z) = sigma_base * (1.0 + tanh(w^T z + b))
    gamma(s, z) = s^2 / (s^2 + sigma_0(z)^2)
    S_DUHWIoU(A, B) = [IoU(A, B)]^gamma(s, z) * exp( - (1 - gamma(s, z)) * D_W^2(A, B) )
    L_DUHWIoU(A, B) = (1 - S_DUHWIoU(A, B)) + lambda_unc * (log(sigma_0(z)) - log(sigma_base))^2
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torchvision.ops import box_iou

EPS = 1e-6


class UncertaintyHomotopyPredictor(nn.Module):
    """Lightweight 2-layer MLP predicting instance-level sigma_0 from RoI feature embeddings."""

    def __init__(self, in_features: int = 1024, hidden_dim: int = 128, sigma_base: float = 8.0):
        super().__init__()
        self.sigma_base = float(sigma_base)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        # Initialize to near-zero deviation
        nn.init.zeros_(self.mlp[-2].weight)
        nn.init.zeros_(self.mlp[-2].bias)

    def forward(self, roi_features: torch.Tensor) -> torch.Tensor:
        """Args:
            roi_features: [N, in_features] feature tensor.
        Returns:
            sigma_0: [N] instance-level scale parameter in [0.5 * sigma_base, 2.0 * sigma_base].
        """
        delta = self.mlp(roi_features).squeeze(-1)  # [N] in (-1, 1)
        sigma_0 = self.sigma_base * (1.0 + 0.5 * delta)
        return sigma_0.clamp_min(1.0)


def compute_dynamic_scale_homotopy(
    gt_wh: torch.Tensor,
    sigma_0: torch.Tensor | float = 8.0,
) -> torch.Tensor:
    """Compute dynamic scale homotopy parameter gamma(s, sigma_0) in [0, 1].

    Args:
        gt_wh: Tensor [N, 2] of (width, height)
        sigma_0: Tensor [N] or float scalar representing instance scale threshold
    Returns:
        gamma: Tensor [N] in [0, 1]
    """
    safe_wh = gt_wh.clamp_min(EPS)
    s = torch.sqrt(safe_wh[:, 0] * safe_wh[:, 1])
    s_sq = s.square()

    if isinstance(sigma_0, torch.Tensor):
        sigma_sq = sigma_0.to(gt_wh.device, gt_wh.dtype).square()
    else:
        sigma_sq = float(sigma_0) ** 2

    gamma = s_sq / (s_sq + sigma_sq)
    return gamma.clamp(0.0, 1.0)


def aligned_dynamic_uncertainty_h_wiou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    sigma_0: torch.Tensor | float = 8.0,
    sigma_base: float = 8.0,
    lambda_unc: float = 0.05,
) -> torch.Tensor:
    """Compute aligned Dynamic Uncertainty H-WIoU loss with uncertainty regularization."""
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

    # 1. Aligned 2-Wasserstein Distance Squared
    dx = xa - xb
    dy = ya - yb
    sx = (wa.square() + wb.square()) / 2.0
    sy = (ha.square() + hb.square()) / 2.0
    position_term = (dx.square() / sx.clamp_min(EPS)) + (dy.square() / sy.clamp_min(EPS))
    log_w = torch.log(wa) - torch.log(wb)
    log_h = torch.log(ha) - torch.log(hb)
    shape_term = log_w.square() + log_h.square()
    dw_sq = (position_term + shape_term).clamp_min(0.0)

    # 2. IoU
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    union = (wa * ha + wb * hb - inter).clamp_min(EPS)
    iou = (inter / union).clamp(min=EPS, max=1.0)

    # 3. Dynamic Homotopy Gamma
    target_wh = torch.stack([wb, hb], dim=1)
    gamma = compute_dynamic_scale_homotopy(target_wh, sigma_0=sigma_0)

    # 4. Homotopy Loss
    iou_part = torch.pow(iou, gamma)
    ot_part = torch.exp(-(1.0 - gamma) * dw_sq)
    similarity = (iou_part * ot_part).clamp(0.0, 1.0)
    loss = 1.0 - similarity

    # 5. Uncertainty Regularization
    if isinstance(sigma_0, torch.Tensor) and lambda_unc > 0.0:
        reg = lambda_unc * (torch.log(sigma_0 / float(sigma_base))).square()
        loss = loss + reg

    return loss
