"""CIoU baseline.

For fair comparison, when METRIC="ciou" we DON'T use custom metric
in label assignment (RPN). Instead, we use torchvision's default
IoU-based assignment. The model is built differently via build_model().

This file exposes a stub `compute_rfd` that returns IoU as similarity,
for use ONLY in ablation experiments where someone wants to plug CIoU
into the custom RPN pipeline.

Normal CIoU baseline experiments use standard torchvision model (see model.py).
"""
from __future__ import annotations
import torch
from torchvision.ops import box_iou

EPS = 1e-6


def compute_rfd(xn, yn, wn, hn, xg, yg, wg, hg, **kwargs) -> torch.Tensor:
    """Treat anchors + GT as boxes, return IoU matrix as 'similarity'.

    boxes_n = [xn-wn/2, yn-hn/2, xn+wn/2, yn+hn/2]
    boxes_g = [xg-wg/2, yg-hg/2, xg+wg/2, yg+hg/2]
    """
    boxes_n = torch.stack([xn - wn/2, yn - hn/2, xn + wn/2, yn + hn/2], dim=1)
    boxes_g = torch.stack([xg - wg/2, yg - hg/2, xg + wg/2, yg + hg/2], dim=1)
    return box_iou(boxes_n, boxes_g).clamp(min=EPS)


name = "ciou"
needs_reliability_thr = False