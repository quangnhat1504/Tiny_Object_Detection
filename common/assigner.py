"""
SAALWAssigner — threshold-based label assignment using SA-ALW similarity.

Khác với MetricRPN (dùng hierarchical top-k):
  - SAALWAssigner dùng ngưỡng similarity để phân loại positive/negative
  - Mỗi anchor: sim > pos_sim_thr với 1 GT nào đó -> positive
  - Mỗi anchor: sim < neg_sim_thr với tất cả GT -> negative
  - topk_fallback: nếu GT có < k positive, lấy top-k anchors bất kể ngưỡng
  - Có dynamic pos_thr theo scale của GT (object càng nhỏ -> threshold thấp hơn)

Ablation targets:
  H3.1: topk_fallback có optimum non-monotonic (~6-9)
  H3.2: SAALWAssigner > ATSS trên AP_vt, <= ATSS trên general AP
"""
from __future__ import annotations
from typing import Callable, Optional

import torch
import torch.nn as nn

from .config import (
    SA_ALW_S_MIN, SA_ALW_S_MAX,
    TINY_THRESHOLD_PX,
)

# Default thresholds (sẽ grid-search)
DEFAULT_POS_SIM_THR      = 0.45
DEFAULT_NEG_SIM_THR      = 0.20
DEFAULT_TOPK_FALLBACK    = 6
DEFAULT_DYNAMIC_THR      = True   # threshold thấp hơn cho tiny GT


class SAALWAssigner(nn.Module):
    """Threshold-based label assignment via SA-ALW similarity.

    Args:
        metric_fn: SA-ALW similarity function (returns exp(-β·d) ∈ (0,1])
        pos_sim_thr: ngưỡng similarity để positive (anchor có sim > thr với GT)
        neg_sim_thr: ngưỡng similarity để negative (anchor có sim < thr với mọi GT)
        topk_fallback: số anchor tối thiểu mỗi GT được gán (bất kể threshold)
        dynamic_thr: nếu True, pos_thr giảm theo sqrt(area) của GT
        reliability_thr: adaptive threshold từ dataset (cho ALW reliability gate)
    """

    def __init__(
        self,
        metric_fn: Callable,
        pos_sim_thr: float = DEFAULT_POS_SIM_THR,
        neg_sim_thr: float = DEFAULT_NEG_SIM_THR,
        topk_fallback: int = DEFAULT_TOPK_FALLBACK,
        dynamic_thr: bool = DEFAULT_DYNAMIC_THR,
        reliability_thr: float = 16.0,
    ):
        super().__init__()
        self.metric_fn = metric_fn
        self.pos_sim_thr = pos_sim_thr
        self.neg_sim_thr = neg_sim_thr
        self.topk_fallback = topk_fallback
        self.dynamic_thr = dynamic_thr
        self.reliability_thr = reliability_thr

    def forward(self, anchors, gt_boxes):
        """Compute assignment labels and matched boxes.

        Args:
            anchors: Tensor (N, 4) in (x1, y1, x2, y2) format
            gt_boxes: Tensor (M, 4) in (x1, y1, x2, y2) format
        Returns:
            labels: (N,) float tensor (1=positive, 0=negative/ignore)
            matched_boxes: (N, 4) tensor of matched GT boxes
        """
        N = anchors.shape[0]
        M = gt_boxes.shape[0]
        dev = anchors.device

        labels = torch.zeros(N, dtype=torch.float32, device=dev)
        matched_boxes = torch.zeros(N, 4, device=dev)

        if M == 0:
            return labels, matched_boxes

        # Convert to (cx, cy, w, h) cho metric computation
        xa = (anchors[:, 0] + anchors[:, 2]) / 2.0
        ya = (anchors[:, 1] + anchors[:, 3]) / 2.0
        wa = (anchors[:, 2] - anchors[:, 0]).clamp(min=1.0)
        ha = (anchors[:, 3] - anchors[:, 1]).clamp(min=1.0)

        xg = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
        yg = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
        wg = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1.0)
        hg = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1.0)

        # Compute SA-ALW similarity matrix (N, M)
        sim = self.metric_fn(
            xa, ya, wa, ha,
            xg, yg, wg, hg,
            reliability_thr=self.reliability_thr,
        )

        # ── Per-GT dynamic threshold ──
        if self.dynamic_thr:
            gt_size = torch.sqrt((wg * hg).clamp(min=1.0))
            # Normalize to [0, 1]: 0 = smallest, 1 = P90
            t = ((gt_size - SA_ALW_S_MIN) / (SA_ALW_S_MAX - SA_ALW_S_MIN)).clamp(0, 1)
            # Tiny objects: thr_low = 0.6 * base   Large objects: thr = base
            dyn_pos_thr = self.pos_sim_thr * (0.6 + 0.4 * t)  # (M,)
        else:
            dyn_pos_thr = torch.full((M,), self.pos_sim_thr, device=dev)

        # ── Assign positive: anchor i -> GT j if sim[i,j] > pos_thr[j] ──
        pos_mask = sim > dyn_pos_thr.unsqueeze(0)  # (N, M)

        # Greedy: assign each anchor to the GT with highest similarity
        best_sim, best_gt = sim.max(dim=1)

        is_pos = torch.zeros(N, dtype=torch.bool, device=dev)

        # Assign anchors that clear threshold for their best GT
        for j in range(M):
            anchor_candidates = (best_gt == j) & (best_sim > dyn_pos_thr[j])
            is_pos |= anchor_candidates

        # ── topk_fallback: ensure each GT gets at least k anchors ──
        per_gt_counts = torch.zeros(M, dtype=torch.long, device=dev)
        per_gt_counts.scatter_add_(0, best_gt[is_pos],
                                    torch.ones(is_pos.sum(), dtype=torch.long, device=dev))

        for j in range(M):
            deficit = self.topk_fallback - int(per_gt_counts[j])
            if deficit <= 0:
                continue
            # Get top-deficit anchors by sim for this GT (excluding already assigned)
            sim_j = sim[:, j]
            # Zero out already assigned anchors
            sim_j_masked = sim_j.clone()
            sim_j_masked[is_pos] = 0.0
            _, top_indices = sim_j_masked.topk(min(deficit, N))
            is_pos[top_indices] = True

        labels[is_pos] = 1.0
        matched_boxes[is_pos] = gt_boxes[best_gt[is_pos]]

        # ── Hard negative mining: anchors with sim < neg_thr for ALL GTs ──
        # (standard RPN will sample from these; others are ignored)
        all_below_neg = (sim < self.neg_sim_thr).all(dim=1)  # (N,)
        labels[all_below_neg & ~is_pos] = -1  # mark as candidate negative

        return labels, matched_boxes


class ScaleDecoupledAssigner(nn.Module):
    """Scale-Decoupled Assigner for Tiny Object Detection (SDA-FRCNN).

    Preserves 100% standard IoU anchor assignment for normal/tiny objects (>= cutoff px)
    while activating SA-ALW dynamic top-k assignment for microscopic objects (< cutoff px).
    Prevents assignment distortion on normal scales and guarantees positive anchors for micro scales.
    """

    def __init__(
        self,
        metric_fn: Callable,
        micro_cutoff_px: float = 8.0,
        fg_iou_thresh: float = 0.7,
        bg_iou_thresh: float = 0.3,
        micro_topk: int = 4,
        micro_pos_sim_thr: float = 0.35,
        max_center_dist_ratio: float = 4.0,
        reliability_thr: float = 16.0,
    ):
        super().__init__()
        self.metric_fn = metric_fn
        self.micro_cutoff_px = float(micro_cutoff_px)
        self.fg_iou_thresh = float(fg_iou_thresh)
        self.bg_iou_thresh = float(bg_iou_thresh)
        self.micro_topk = int(micro_topk)
        self.micro_pos_sim_thr = float(micro_pos_sim_thr)
        self.max_center_dist_ratio = float(max_center_dist_ratio)
        self.reliability_thr = float(reliability_thr)

    def forward(self, anchors: torch.Tensor, gt_boxes: torch.Tensor):
        """Assign anchors to ground truth boxes in a scale-decoupled manner.

        Args:
            anchors: Tensor (N, 4) in (x1, y1, x2, y2)
            gt_boxes: Tensor (M, 4) in (x1, y1, x2, y2)
        Returns:
            labels: (N,) tensor (1=pos, 0=neg, -1=ignore)
            matched_boxes: (N, 4) tensor
        """
        from torchvision.ops import box_iou
        N = anchors.shape[0]
        M = gt_boxes.shape[0]
        dev = anchors.device

        labels = torch.full((N,), -1, dtype=torch.float32, device=dev)
        matched_boxes = torch.zeros((N, 4), device=dev)

        if M == 0:
            labels.zero_()
            return labels, matched_boxes

        # Compute GT sizes
        wg = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1.0)
        hg = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1.0)
        gt_scales = torch.sqrt(wg * hg)

        is_micro_gt = gt_scales < self.micro_cutoff_px
        is_standard_gt = ~is_micro_gt

        # 1. Standard IoU Branch for Standard GTs (>= cutoff px)
        if is_standard_gt.any():
            std_gt_indices = torch.where(is_standard_gt)[0]
            std_gts = gt_boxes[std_gt_indices]
            iou_matrix = box_iou(anchors, std_gts)  # (N, M_std)

            # Standard max IoU per anchor
            matched_vals, matched_idx = iou_matrix.max(dim=1)
            below_bg = matched_vals < self.bg_iou_thresh
            labels[below_bg] = 0.0

            above_fg = matched_vals >= self.fg_iou_thresh
            labels[above_fg] = 1.0
            matched_boxes[above_fg] = std_gts[matched_idx[above_fg]]

            # Best anchor per GT rule
            gt_max_vals, gt_max_idx = iou_matrix.max(dim=0)
            for j_idx, max_anchor_idx in enumerate(gt_max_idx):
                if gt_max_vals[j_idx] > 0.1:  # non-degenerate
                    labels[max_anchor_idx] = 1.0
                    matched_boxes[max_anchor_idx] = std_gts[j_idx]

        # 2. Micro SA-ALW Branch for Micro GTs (< cutoff px)
        if is_micro_gt.any():
            micro_gt_indices = torch.where(is_micro_gt)[0]
            micro_gts = gt_boxes[micro_gt_indices]

            xa = (anchors[:, 0] + anchors[:, 2]) / 2.0
            ya = (anchors[:, 1] + anchors[:, 3]) / 2.0
            wa = (anchors[:, 2] - anchors[:, 0]).clamp(min=1.0)
            ha = (anchors[:, 3] - anchors[:, 1]).clamp(min=1.0)

            xg_m = (micro_gts[:, 0] + micro_gts[:, 2]) / 2.0
            yg_m = (micro_gts[:, 1] + micro_gts[:, 3]) / 2.0
            wg_m = (micro_gts[:, 2] - micro_gts[:, 0]).clamp(min=1.0)
            hg_m = (micro_gts[:, 3] - micro_gts[:, 1]).clamp(min=1.0)
            scale_m = torch.sqrt(wg_m * hg_m)

            sim_m = self.metric_fn(
                xa, ya, wa, ha,
                xg_m, yg_m, wg_m, hg_m,
                reliability_thr=self.reliability_thr,
            )  # (N, M_micro)

            for j_local, j_global in enumerate(micro_gt_indices):
                sim_col = sim_m[:, j_local]
                s_gt = scale_m[j_local]
                max_radius = max(32.0, float(s_gt.item()) * self.max_center_dist_ratio)
                spatial_dist_sq = (xa - xg_m[j_local]).square() + (ya - yg_m[j_local]).square()
                is_spatial_candidate = spatial_dist_sq <= (max_radius ** 2)

                # High sim matches within spatial radius
                high_sim_mask = (sim_col >= self.micro_pos_sim_thr) & is_spatial_candidate & (labels != 1.0)
                labels[high_sim_mask] = 1.0
                matched_boxes[high_sim_mask] = micro_gts[j_local]

                # Dynamic Top-k fallback on spatially valid candidates if deficit
                curr_assigned = (labels == 1.0) & (matched_boxes == micro_gts[j_local]).all(dim=-1)
                deficit = self.micro_topk - int(curr_assigned.sum().item())
                if deficit > 0:
                    sim_masked = sim_col.clone()
                    # Do not steal anchors already assigned to standard GTs
                    sim_masked[labels == 1.0] = -1.0
                    # Filter out anchors outside spatial radius
                    sim_masked[~is_spatial_candidate] = -1.0
                    valid_candidates = (sim_masked >= 0.0).sum().item()
                    k_val = min(deficit, valid_candidates)
                    if k_val > 0:
                        _, topk_idx = sim_masked.topk(k_val)
                        labels[topk_idx] = 1.0
                        matched_boxes[topk_idx] = micro_gts[j_local]

        # If an anchor is neither positive nor marked negative, check global background
        unassigned = labels == -1
        if unassigned.any():
            labels[unassigned] = 0.0

        return labels, matched_boxes


def build_saalw_assigner(metric_name: str = "sa_alw_full", **kwargs) -> SAALWAssigner:
    """Build SAALWAssigner with specified metric.

    Args:
        metric_name: key in METRIC_REGISTRY (default: sa_alw_full)
        **kwargs: pos_sim_thr, neg_sim_thr, topk_fallback, dynamic_thr, reliability_thr
    """
    from common.metrics import get_metric_fn
    metric_fn = get_metric_fn(metric_name)
    return SAALWAssigner(metric_fn=metric_fn, **kwargs)


def build_scale_decoupled_assigner(
    metric_name: str = "sa_alw_full",
    micro_cutoff_px: float = 8.0,
    **kwargs,
) -> ScaleDecoupledAssigner:
    """Build ScaleDecoupledAssigner."""
    from common.metrics import get_metric_fn
    metric_fn = get_metric_fn(metric_name)
    return ScaleDecoupledAssigner(
        metric_fn=metric_fn,
        micro_cutoff_px=micro_cutoff_px,
        **kwargs,
    )

