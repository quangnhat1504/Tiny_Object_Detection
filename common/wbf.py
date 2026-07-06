"""
SmartWBF — extends standard WBF with extent_hull fusion and adaptive IoU.
No tile adjacency (too slow) — relies on scale-adaptive thresholds instead.

Usage:
    from common.wbf import wbf_fusion_smart
    fused = wbf_fusion_smart(tile_preds, tile_coords, img_size,
                             iou_thr=0.55, fusion_mode="extent_hull",
                             adaptive_thr=True)
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import torch
from torchvision.ops import box_iou


def _extent_hull(boxes: torch.Tensor, scores: torch.Tensor,
                 iqr_mult: float = 1.5) -> torch.Tensor | None:
    if len(boxes) <= 1:
        return boxes[0] if len(boxes) == 1 else None

    def _robust_min(v):
        q1, q3 = torch.quantile(v, torch.tensor([0.25, 0.75]))
        iqr = q3 - q1
        lo = q1 - iqr_mult * iqr
        valid = v >= lo
        return v[valid].min().item() if valid.any() else v.min().item()

    def _robust_max(v):
        q1, q3 = torch.quantile(v, torch.tensor([0.25, 0.75]))
        iqr = q3 - q1
        hi = q3 + iqr_mult * iqr
        valid = v <= hi
        return v[valid].max().item() if valid.any() else v.max().item()

    return torch.tensor([
        _robust_min(boxes[:, 0]), _robust_min(boxes[:, 1]),
        _robust_max(boxes[:, 2]), _robust_max(boxes[:, 3]),
    ])


def _weighted_avg(boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    w = scores / scores.sum()
    return (boxes.T @ w).T


def _adaptive_threshold(area: float, base_iou_thr: float,
                        use_adaptive: bool) -> float:
    """LARGE objects get LOWER threshold to merge partial tile views."""
    if not use_adaptive:
        return base_iou_thr
    if area < 256:
        return base_iou_thr
    if area < 1024:
        return max(base_iou_thr - 0.05, 0.25)
    if area < 4096:
        return max(base_iou_thr - 0.15, 0.20)
    return max(base_iou_thr - 0.25, 0.10)


def wbf_fusion_smart(
    tile_preds: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    tile_coords: List[Tuple[int, int, int, int]],
    img_size: Tuple[int, int],
    iou_thr: float = 0.55,
    fusion_mode: str = "weighted_avg",
    adaptive_thr: bool = False,
) -> Dict[str, torch.Tensor]:

    W, H = img_size
    all_boxes, all_scores, all_labels = [], [], []

    for (tx, ty, tw, th), (boxes, scores, labels) in zip(tile_coords, tile_preds):
        if boxes.numel() == 0:
            continue
        rm = boxes.clone()
        rm[:, 0] = rm[:, 0] * tw / 512 + tx
        rm[:, 1] = rm[:, 1] * th / 512 + ty
        rm[:, 2] = rm[:, 2] * tw / 512 + tx
        rm[:, 3] = rm[:, 3] * th / 512 + ty
        rm[:, 0].clamp_(0, W); rm[:, 1].clamp_(0, H)
        rm[:, 2].clamp_(0, W); rm[:, 3].clamp_(0, H)
        valid = (rm[:, 2] - rm[:, 0] >= 2) & (rm[:, 3] - rm[:, 1] >= 2)
        if valid.any():
            all_boxes.append(rm[valid])
            all_scores.append(scores[valid])
            all_labels.append(labels[valid])

    empty = {
        "boxes": torch.zeros(0, 4),
        "scores": torch.zeros(0),
        "labels": torch.zeros(0, dtype=torch.int64),
    }

    if not all_boxes:
        return empty

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    if boxes.numel() <= 1:
        return {"boxes": boxes, "scores": scores, "labels": labels}

    ious = box_iou(boxes, boxes)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # DSU cluster with adaptive IoU
    parent = list(range(len(boxes)))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x, y):
        xr, yr = _find(x), _find(y)
        if xr != yr:
            parent[yr] = xr

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _find(i) == _find(j):
                continue
            if labels[i] != labels[j]:
                continue
            thr = _adaptive_threshold(
                min(areas[i].item(), areas[j].item()), iou_thr, adaptive_thr)
            if ious[i, j] > thr:
                _union(i, j)

    groups = {}
    for i in range(len(boxes)):
        root = _find(i)
        groups.setdefault(root, []).append(i)

    fused_b, fused_s, fused_l = [], [], []
    for indices in groups.values():
        idx = torch.tensor(indices)
        cb, cs, cl = boxes[idx], scores[idx], labels[idx]
        if fusion_mode == "extent_hull":
            fb = _extent_hull(cb, cs)
        else:
            fb = _weighted_avg(cb, cs)
        if fb is not None:
            fused_b.append(fb)
            fused_s.append(cs.mean())
            fused_l.append(torch.mode(cl).values)

    if not fused_b:
        return empty

    return {
        "boxes": torch.stack(fused_b),
        "scores": torch.stack(fused_s),
        "labels": torch.stack(fused_l),
    }
