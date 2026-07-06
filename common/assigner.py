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


def build_saalw_assigner(metric_name: str = "sa_alw_full", **kwargs) -> SAALWAssigner:
    """Build SAALWAssigner with specified metric.

    Args:
        metric_name: key in METRIC_REGISTRY (default: sa_alw_full)
        **kwargs: pos_sim_thr, neg_sim_thr, topk_fallback, dynamic_thr, reliability_thr
    """
    from common.metrics import get_metric_fn
    metric_fn = get_metric_fn(metric_name)
    return SAALWAssigner(metric_fn=metric_fn, **kwargs)
