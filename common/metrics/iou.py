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


def compute_rfd(xn, yn, wn, hn, xg, yg, wg, hg, chunk_size: int = 16384, **kwargs) -> torch.Tensor:
    """Treat anchors + GT as boxes, return chunked IoU matrix as 'similarity'."""
    boxes_n = torch.stack([xn - wn/2, yn - hn/2, xn + wn/2, yn + hn/2], dim=1)
    boxes_g = torch.stack([xg - wg/2, yg - hg/2, xg + wg/2, yg + hg/2], dim=1)
    N = boxes_n.shape[0]
    if N <= chunk_size:
        return box_iou(boxes_n, boxes_g).clamp(min=EPS)
    sims = []
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        sims.append(box_iou(boxes_n[i:end_i], boxes_g).clamp(min=EPS))
    return torch.cat(sims, dim=0)


name = "ciou"
needs_reliability_thr = False