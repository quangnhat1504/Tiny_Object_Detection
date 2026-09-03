"""
Entropy-Modulated Homotopy Wasserstein-IoU (EH-WIoU) Metric & Bounded Regression Loss.

Implements the novel Feature-Metric Symbiosis paradigm:
1. Shannon Information Entropy Guidance: H(x, y) = -sum_c p_c log(p_c)
2. 2D Scale-Entropy Homotopy Operator:
   gamma(s, H) = [s^2 * (1 + beta * H)] / [s^2 * (1 + beta * H) + sigma_0^2]
3. Entropy-Modulated Homotopy Similarity:
   S_EH-WIoU(A, B) = [IoU(A, B)]^gamma(s_B, H) * exp(-(1 - gamma(s_B, H)) * D_W^2(A, B))
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_shannon_entropy(feature_map: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute pixel-wise Shannon information entropy across feature channels.

    Args:
        feature_map: Tensor of shape [B, C, H, W]
        eps: Numerical stability constant
    Returns:
        entropy_map: Normalized entropy map of shape [B, 1, H, W] in [0, 1]
    """
    # Channel-wise softmax probability distribution
    probs = F.softmax(feature_map, dim=1)  # [B, C, H, W]
    # Shannon Entropy: H = -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1, keepdim=True)  # [B, 1, H, W]
    # Max possible entropy for C channels: log(C)
    max_entropy = math.log(feature_map.shape[1])
    # Normalize to [0, 1]
    norm_entropy = entropy / max(max_entropy, eps)
    return norm_entropy.clamp(0.0, 1.0)


class EntropyGuidanceModule(nn.Module):
    """
    Spatial Entropy Guidance Module (EGM) that extracts pixel-wise Shannon
    information entropy and modulates multi-scale FPN features.
    """
    def __init__(self, in_channels: int = 256, reduction: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.conv_refine = nn.Sequential(
            nn.Conv2d(1, in_channels // reduction, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature_map: [B, C, H, W]
        Returns:
            enhanced_feature: [B, C, H, W]
            entropy_weights: [B, 1, H, W] in [0, 1]
        """
        raw_entropy = compute_shannon_entropy(feature_map)
        entropy_weights = self.conv_refine(raw_entropy)
        # Residual modulation: amplify high-entropy micro-edge signals
        enhanced_feature = feature_map * (1.0 + entropy_weights)
        return enhanced_feature, entropy_weights


def _compute_entropy_homotopy_similarity_chunk(
    cx_p: torch.Tensor,
    cy_p: torch.Tensor,
    w_p: torch.Tensor,
    h_p: torch.Tensor,
    cx_g: torch.Tensor,
    cy_g: torch.Tensor,
    w_g: torch.Tensor,
    h_g: torch.Tensor,
    s_g_sq: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute EH-WIoU similarity for a manageable chunk of predicted boxes."""
    # 1. Boxes for IoU
    x1_a = cx_p - w_p * 0.5
    y1_a = cy_p - h_p * 0.5
    x2_a = cx_p + w_p * 0.5
    y2_a = cy_p + h_p * 0.5

    x1_g = cx_g - w_g * 0.5
    y1_g = cy_g - h_g * 0.5
    x2_g = cx_g + w_g * 0.5
    y2_g = cy_g + h_g * 0.5

    inter_w = (torch.min(x2_a[:, None], x2_g[None, :]) - torch.max(x1_a[:, None], x1_g[None, :])).clamp(min=0.0)
    inter_h = (torch.min(y2_a[:, None], y2_g[None, :]) - torch.max(y1_a[:, None], y1_g[None, :])).clamp(min=0.0)
    inter_area = inter_w * inter_h
    union_area = (w_p * h_p)[:, None] + (w_g * h_g)[None, :] - inter_area
    iou = (inter_area / (union_area + eps)).clamp(min=eps, max=1.0)

    # 2. Wasserstein distance
    dx2 = (cx_p[:, None] - cx_g[None, :]) ** 2
    dy2 = (cy_p[:, None] - cy_g[None, :]) ** 2
    dw2 = ((w_p[:, None] - w_g[None, :]) * 0.5) ** 2
    dh2 = ((h_p[:, None] - h_g[None, :]) * 0.5) ** 2
    w2_sq = dx2 + dy2 + dw2 + dh2
    d_w_sq = w2_sq / (2.0 * s_g_sq + eps)
    wasserstein_similarity = torch.exp(-d_w_sq.clamp(max=50.0))

    # 3. Additive Scale-Entropy Convex Interpolation: S = gamma * IoU + (1 - gamma) * S_W
    similarity = gamma * iou + (1.0 - gamma) * wasserstein_similarity
    return similarity.clamp(0.0, 1.0)


def compute_entropy_homotopy_similarity(
    *args: torch.Tensor,
    entropy_prior: Optional[torch.Tensor] = None,
    sigma_0: float = 8.0,
    beta: float = 0.5,
    eps: float = 1e-7,
    chunk_size: int = 16384,
    **_,
) -> torch.Tensor:
    """
    Compute pairwise Entropy-Modulated Homotopy Similarity (S_EH-WIoU) matrix [N, M].
    Supports both 2-tensor calling convention (pred_boxes [N, 4], gt_boxes [M, 4])
    and 8-coordinate convention (xa, ya, wa, ha, xg, yg, wg, hg [N] and [M]).
    Implements memory-safe chunking along N to prevent CUDA OOM on dense FPN anchor grids.
    """
    if len(args) == 2:
        pred_boxes, gt_boxes = args
        N = pred_boxes.shape[0]
        M = gt_boxes.shape[0]
        if N == 0 or M == 0:
            return torch.empty((N, M), dtype=pred_boxes.dtype, device=pred_boxes.device)

        cx_p = (pred_boxes[:, 0] + pred_boxes[:, 2]) * 0.5
        cy_p = (pred_boxes[:, 1] + pred_boxes[:, 3]) * 0.5
        w_p = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=eps)
        h_p = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=eps)

        cx_g = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
        cy_g = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
        w_g = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=eps)
        h_g = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=eps)

    elif len(args) == 8:
        xa, ya, wa, ha, xg, yg, wg, hg = args
        N = xa.shape[0]
        M = xg.shape[0]
        if N == 0 or M == 0:
            return torch.zeros((N, M), device=xa.device, dtype=xa.dtype)

        cx_p, cy_p, w_p, h_p = xa, ya, wa.clamp(min=eps), ha.clamp(min=eps)
        cx_g, cy_g, w_g, h_g = xg, yg, wg.clamp(min=eps), hg.clamp(min=eps)
    else:
        raise ValueError(f"compute_entropy_homotopy_similarity expects 2 or 8 positional arguments, got {len(args)}")

    # Ground-truth characteristic scale s_g^2 = w_g * h_g [1, M]
    s_g_sq = (w_g * h_g).clamp(min=eps)[None, :]

    # 2D Scale-Entropy Homotopy Parameter gamma(s, H) [1, M]
    if entropy_prior is not None:
        if entropy_prior.dim() == 1:
            h_prior = entropy_prior[None, :]
        else:
            h_prior = entropy_prior
        mod_scale_sq = s_g_sq * (1.0 + beta * h_prior)
    else:
        mod_scale_sq = s_g_sq

    gamma = mod_scale_sq / (mod_scale_sq + (sigma_0 ** 2))
    gamma = gamma.clamp(0.001, 0.999)

    # Memory-safe chunked execution
    if N <= chunk_size:
        return _compute_entropy_homotopy_similarity_chunk(
            cx_p, cy_p, w_p, h_p,
            cx_g, cy_g, w_g, h_g,
            s_g_sq, gamma, eps=eps
        )

    sims = []
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        sims.append(_compute_entropy_homotopy_similarity_chunk(
            cx_p[i:end_i], cy_p[i:end_i], w_p[i:end_i], h_p[i:end_i],
            cx_g, cy_g, w_g, h_g,
            s_g_sq, gamma, eps=eps
        ))
    return torch.cat(sims, dim=0)


class EntropyHomotopyLoss(nn.Module):
    """
    Entropy-Modulated Homotopy Bounded Regression Loss for Stage 2 RoI Heads.
    L_EH-WIoU = 1 - S_EH-WIoU = gamma * (1 - IoU) + (1 - gamma) * (1 - S_W)
    """
    def __init__(self, sigma_0: float = 8.0, beta: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.sigma_0 = float(sigma_0)
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        entropy_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes: [N, 4]
            target_boxes: [N, 4]
            entropy_prior: Optional [N]
        Returns:
            loss: Scalar loss
        """
        if pred_boxes.numel() == 0:
            return pred_boxes.sum() * 0.0

        # Diagonal computation for paired regression boxes
        N = pred_boxes.shape[0]
        # IoU diagonal
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
        area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0.0) * (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0.0)
        area_g = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0.0) * (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0.0)
        union = area_p + area_g - inter
        iou = (inter / (union + self.eps)).clamp(0.0, 1.0)

        # Wasserstein diagonal
        cx_p = (pred_boxes[:, 0] + pred_boxes[:, 2]) * 0.5
        cy_p = (pred_boxes[:, 1] + pred_boxes[:, 3]) * 0.5
        w_p = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=self.eps)
        h_p = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=self.eps)

        cx_g = (target_boxes[:, 0] + target_boxes[:, 2]) * 0.5
        cy_g = (target_boxes[:, 1] + target_boxes[:, 3]) * 0.5
        w_g = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=self.eps)
        h_g = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=self.eps)

        dx2 = (cx_p - cx_g) ** 2
        dy2 = (cy_p - cy_g) ** 2
        dw2 = ((w_p - w_g) * 0.5) ** 2
        dh2 = ((h_p - h_g) * 0.5) ** 2

        s_g = torch.sqrt((w_g * h_g).clamp(min=self.eps))
        w2_sq = dx2 + dy2 + dw2 + dh2
        d_w_sq = w2_sq / (2.0 * (s_g ** 2) + self.eps)
        wasserstein_sim = torch.exp(-d_w_sq.clamp(max=50.0))

        # Scale-Entropy Homotopy Parameter
        if entropy_prior is not None:
            mod_scale_sq = (s_g ** 2) * (1.0 + self.beta * entropy_prior)
        else:
            mod_scale_sq = s_g ** 2

        gamma = (mod_scale_sq / (mod_scale_sq + (self.sigma_0 ** 2))).clamp(0.0, 1.0)
        similarity = gamma * iou + (1.0 - gamma) * wasserstein_sim
        loss = 1.0 - similarity
        return loss.mean()


def aligned_entropy_homotopy_loss(
    *args: torch.Tensor,
    sigma_0: float = 8.0,
    beta: float = 0.5,
    entropy_prior: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
    **_,
) -> torch.Tensor:
    """Compute aligned Entropy-Modulated Homotopy Wasserstein-IoU loss for bounding box regression.
    Supports both 2-tensor calling convention (pred_boxes [N,4], target_boxes [N,4])
    and 8-coordinate convention (xa, ya, wa, ha, xb, yb, wb, hb [N]).
    """
    if len(args) == 2:
        pred_boxes, target_boxes = args
        if pred_boxes.ndim != 2 or pred_boxes.shape[-1] != 4:
            raise ValueError("pred_boxes must have shape [N, 4]")
        if target_boxes.ndim != 2 or target_boxes.shape[-1] != 4:
            raise ValueError("target_boxes must have shape [N, 4]")

        xa = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
        ya = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0
        wa = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=eps)
        ha = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=eps)

        xb = (target_boxes[:, 0] + target_boxes[:, 2]) / 2.0
        yb = (target_boxes[:, 1] + target_boxes[:, 3]) / 2.0
        wb = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=eps)
        hb = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=eps)

        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
        area_a = wa * ha
        area_b = wb * hb
        union = (area_a + area_b - inter).clamp_min(eps)
        iou = (inter / union).clamp(min=0.0, max=1.0)
    elif len(args) == 8:
        xa, ya, wa, ha, xb, yb, wb, hb = args
        x1_a, y1_a, x2_a, y2_a = xa - wa / 2.0, ya - ha / 2.0, xa + wa / 2.0, ya + ha / 2.0
        x1_b, y1_b, x2_b, y2_b = xb - wb / 2.0, yb - hb / 2.0, xb + wb / 2.0, yb + hb / 2.0
        inter_w = (torch.min(x2_a, x2_b) - torch.max(x1_a, x1_b)).clamp(min=0.0)
        inter_h = (torch.min(y2_a, y2_b) - torch.max(y1_a, y1_b)).clamp(min=0.0)
        inter = inter_w * inter_h
        union = (wa * ha + wb * hb - inter).clamp_min(eps)
        iou = (inter / union).clamp(min=0.0, max=1.0)
    else:
        raise ValueError(f"aligned_entropy_homotopy_loss expects 2 or 8 arguments, got {len(args)}")

    dx2 = (xa - xb) ** 2
    dy2 = (ya - yb) ** 2
    dw2 = ((wa - wb) * 0.5) ** 2
    dh2 = ((ha - hb) * 0.5) ** 2

    s_g = torch.sqrt((wb * hb).clamp(min=eps))
    w2_sq = dx2 + dy2 + dw2 + dh2
    d_w_sq = w2_sq / (2.0 * (s_g ** 2) + eps)
    wasserstein_sim = torch.exp(-d_w_sq.clamp(max=50.0))

    if entropy_prior is not None:
        mod_scale_sq = (s_g ** 2) * (1.0 + beta * entropy_prior)
    else:
        mod_scale_sq = s_g ** 2

    gamma = (mod_scale_sq / (mod_scale_sq + (sigma_0 ** 2))).clamp(0.001, 0.999)
    similarity = gamma * iou + (1.0 - gamma) * wasserstein_sim
    loss = 1.0 - similarity
    return loss

