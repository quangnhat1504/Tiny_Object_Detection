"""
Cascade Multi-Stage Homotopy Loss & Assigner Module.
Provides progressive scale-homotopy schedules across multi-stage Cascade R-CNN architectures.

Stage Homotopy Schedule:
    Stage 1 (Coarse / RPN Proposals): sigma_1 = 8.0 px (Optimal Transport dominant, high recall)
    Stage 2 (Refinement):              sigma_2 = 4.0 px (Balanced transport & geometry)
    Stage 3 (Strict Localization):     sigma_3 = 2.0 px -> 0.0 px (Lebesgue IoU dominant, tight boundary)
"""
from __future__ import annotations
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision.ops import box_iou

from .h_wiou import (
    EPS,
    compute_scale_homotopy,
    pairwise_wasserstein_distance_squared,
    aligned_wasserstein_distance_squared,
    compute_h_wiou_similarity,
    aligned_h_wiou_loss,
)


class CascadeHomotopyLoss(nn.Module):
    """Multi-stage Homotopy Wasserstein-IoU loss for Cascade R-CNN regression heads.

    Args:
        sigmas: List of characteristic scale thresholds per stage, e.g. [8.0, 4.0, 2.0]
        loss_weights: List of loss multipliers per stage, e.g. [1.0, 1.0, 1.0]
        form: Homotopy deformation function form ('rational', 'exponential', etc.)
    """

    def __init__(
        self,
        sigmas: List[float] = [8.0, 4.0, 2.0],
        loss_weights: List[float] = [1.0, 1.0, 1.0],
        form: str = "rational",
    ):
        super().__init__()
        if len(sigmas) != len(loss_weights):
            raise ValueError(f"Length of sigmas ({len(sigmas)}) must match loss_weights ({len(loss_weights)})")
        self.sigmas = [float(s) for s in sigmas]
        self.loss_weights = [float(w) for w in loss_weights]
        self.form = form

    def forward(
        self,
        stage_idx: int,
        *args: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute aligned Homotopy Wasserstein-IoU loss for a specific cascade stage.

        Args:
            stage_idx: Zero-indexed cascade stage (0 <= stage_idx < len(sigmas))
            args: Either (pred_boxes, target_boxes) [N, 4] or (xa, ya, wa, ha, xb, yb, wb, hb) [N]
        Returns:
            weighted_loss: Scalar loss tensor [N]
        """
        if not (0 <= stage_idx < len(self.sigmas)):
            raise IndexError(f"stage_idx {stage_idx} out of range [0, {len(self.sigmas) - 1}]")

        sigma_k = self.sigmas[stage_idx]
        weight_k = self.loss_weights[stage_idx]

        loss = aligned_h_wiou_loss(*args, sigma_0=sigma_k, form=self.form, **kwargs)
        return weight_k * loss


def cascade_homotopy_stage_matcher(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    stage_idx: int = 0,
    sigmas: List[float] = [8.0, 4.0, 2.0],
    pos_thresh: float = 0.35,
    neg_thresh: float = 0.15,
    topk_fallback: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Homotopy-guided sample matching for Cascade stages.
    Prevents microscopic sample starvation in Stage 2 and Stage 3 by replacing
    discrete IoU thresholding with continuous scale-homotopy similarity.
    """
    N = proposals.shape[0]
    M = gt_boxes.shape[0]
    device = proposals.device

    if N == 0 or M == 0:
        matched_gt_boxes = torch.zeros((N, 4), dtype=proposals.dtype, device=device)
        matched_labels = torch.zeros((N,), dtype=torch.int64, device=device)
        pos_mask = torch.zeros((N,), dtype=torch.bool, device=device)
        return matched_gt_boxes, matched_labels, pos_mask

    sigma_k = sigmas[min(stage_idx, len(sigmas) - 1)]

    # 1. Convert to center format (cx, cy, w, h)
    xa = (proposals[:, 0] + proposals[:, 2]) / 2.0
    ya = (proposals[:, 1] + proposals[:, 3]) / 2.0
    wa = (proposals[:, 2] - proposals[:, 0]).clamp_min(EPS)
    ha = (proposals[:, 3] - proposals[:, 1]).clamp_min(EPS)

    xg = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
    yg = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
    wg = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp_min(EPS)
    hg = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp_min(EPS)

    # 2. Compute Stage Homotopy Similarity Matrix [N, M]
    sim_matrix = compute_h_wiou_similarity(
        xa, ya, wa, ha,
        xg, yg, wg, hg,
        sigma_0=sigma_k,
    )

    # 3. Match each proposal to best GT
    max_sim_per_proposal, best_gt_idx = sim_matrix.max(dim=1)  # [N]

    matched_gt_boxes = gt_boxes[best_gt_idx]
    matched_labels = torch.zeros(N, dtype=torch.int64, device=device)

    # Positive threshold
    pos_mask = max_sim_per_proposal >= pos_thresh
    matched_labels[pos_mask] = gt_labels[best_gt_idx[pos_mask]]

    # 4. Top-K Fallback guarantee per GT box
    if topk_fallback > 0:
        for m in range(M):
            gt_sims = sim_matrix[:, m]
            k = min(topk_fallback, N)
            topk_vals, topk_inds = torch.topk(gt_sims, k)
            valid_topk = topk_inds[topk_vals > neg_thresh]
            if len(valid_topk) > 0:
                pos_mask[valid_topk] = True
                matched_labels[valid_topk] = gt_labels[m]
                matched_gt_boxes[valid_topk] = gt_boxes[m]

    return matched_gt_boxes, matched_labels, pos_mask
