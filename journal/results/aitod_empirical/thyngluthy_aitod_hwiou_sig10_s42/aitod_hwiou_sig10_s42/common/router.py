"""
Uncertainty Router — decides which regions need FRCNN refinement.

3 routing criteria (from Cascaded Uncertainty Routing concept):
  1. Grey-zone score: score in [conf_low, conf_high] — uncertain detections
  2. Tiny box: area < area_thr — small objects benefit from FRCNN's multi-scale RPN
  3. Blind-spot prior: grid-based regions where YOLO may have zero detections
                       (horizon line, far water) — critical fallback (H4.1)

Output: list of patch regions (xyxy in full-image coords) to send to FRCNN.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
import torch
from torchvision.ops import box_iou


class UncertaintyRouter:
    """Decides which image regions need FRCNN refinement.

    Args:
        conf_low: lower bound of grey zone (detections below this are ignored)
        conf_high: upper bound of grey zone (detections above this are trusted)
        area_thr: area threshold for tiny-box routing (pixels)
        context_ratio: padding around box when extracting patches
        patch_size: output patch size (resized to this)
        blind_spot_grid: grid size for blind-spot scanning (0 = disabled)
        blind_spot_overlap: overlap ratio for blind-spot grid
        iou_merge_thr: merge overlapping patches with IoU > this
    """

    def __init__(
        self,
        conf_low: float = 0.20,
        conf_high: float = 0.55,
        area_thr: int = 256,   # 16x16 px
        context_ratio: float = 1.5,
        patch_size: int = 512,
        blind_spot_grid: int = 0,  # 0 = disabled for now (Phase 4 step 1)
        blind_spot_overlap: float = 0.3,
        iou_merge_thr: float = 0.5,
    ):
        self.conf_low = conf_low
        self.conf_high = conf_high
        self.area_thr = area_thr
        self.context_ratio = context_ratio
        self.patch_size = patch_size
        self.blind_spot_grid = blind_spot_grid
        self.blind_spot_overlap = blind_spot_overlap
        self.iou_merge_thr = iou_merge_thr

    def route(
        self,
        detections: List[dict],   # YOLO output: [{"boxes": (K,4), "scores": (K,), "labels": (K,)}]
        img_shapes: List[Tuple[int, int, int]],  # [(C, H, W), ...]
    ) -> List[List[Tuple[int, int, int, int]]]:
        """Route each image and return patch regions.

        Args:
            detections: per-image YOLO predictions
            img_shapes: per-image shapes (C, H, W) or (H, W)

        Returns:
            per-image list of patch regions (x1, y1, x2, y2) in pixel coords
        """
        all_patches = []
        for det, shape in zip(detections, img_shapes):
            H, W = shape[0], shape[1] if len(shape) >= 2 else (shape[0], shape[0])
            patches = []

            boxes = det.get("boxes", torch.empty(0, 4))
            scores = det.get("scores", torch.empty(0))
            if boxes.numel() == 0:
                all_patches.append(patches)
                continue

            # Criterion 1: Grey-zone — uncertain detections
            grey_mask = (scores >= self.conf_low) & (scores <= self.conf_high)
            for box in boxes[grey_mask]:
                patches.append(self._box_to_patch(box, W, H))

            # Criterion 2: Tiny box — small objects
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            tiny_mask = (areas < self.area_thr) & (~grey_mask)  # avoid duplicates
            for box in boxes[tiny_mask]:
                patches.append(self._box_to_patch(box, W, H))

            # Criterion 3: Blind-spot grid (optional)
            if self.blind_spot_grid > 0:
                gs = self.blind_spot_grid
                stride = int(self.patch_size * (1 - self.blind_spot_overlap))
                for y in range(0, H - gs + 1, stride):
                    for x in range(0, W - gs + 1, stride):
                        x1, y1 = x, y
                        x2, y2 = min(x + gs, W), min(y + gs, H)
                        # Check if this grid cell has zero detections
                        grid_box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
                        ious = box_iou(grid_box, boxes)
                        if ious.max() < 0.1:  # no detections in this cell
                            patches.append(self._box_to_patch(
                                torch.tensor([x1, y1, x2, y2], dtype=torch.float32), W, H))

            # Merge overlapping patches
            patches = self._merge_overlapping(patches, self.iou_merge_thr)

            # Clamp to image bounds
            clamped = []
            for x1, y1, x2, y2 in patches:
                x1c = max(0, int(x1))
                y1c = max(0, int(y1))
                x2c = min(W, int(x2))
                y2c = min(H, int(y2))
                if x2c - x1c >= 10 and y2c - y1c >= 10:
                    clamped.append((x1c, y1c, x2c, y2c))
            all_patches.append(clamped)

        return all_patches

    def _box_to_patch(self, box, W: int, H: int) -> Tuple[int, int, int, int]:
        """Convert a detection box to a patch region with context padding."""
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        bw = box[2] - box[0]
        bh = box[3] - box[1]

        pw = max(bw * self.context_ratio, self.patch_size)
        ph = max(bh * self.context_ratio, self.patch_size)

        x1 = cx - pw / 2
        y1 = cy - ph / 2
        x2 = cx + pw / 2
        y2 = cy + ph / 2

        return (x1, y1, x2, y2)

    @staticmethod
    def _merge_overlapping(patches: List[Tuple], iou_thr: float) -> List[Tuple]:
        """Merge patches with IoU > iou_thr (union bounding box)."""
        if len(patches) <= 1:
            return patches

        merged = []
        used = [False] * len(patches)

        for i in range(len(patches)):
            if used[i]:
                continue
            x1, y1, x2, y2 = patches[i]
            used[i] = True
            for j in range(i + 1, len(patches)):
                if used[j]:
                    continue
                px1, py1, px2, py2 = patches[j]
                ix1 = max(x1, px1)
                iy1 = max(y1, py1)
                ix2 = min(x2, px2)
                iy2 = min(y2, py2)
                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (px2 - px1) * (py2 - py1)
                    union = area1 + area2 - inter
                    if union > 0 and inter / union > iou_thr:
                        x1 = min(x1, px1)
                        y1 = min(y1, py1)
                        x2 = max(x2, px2)
                        y2 = max(y2, py2)
                        used[j] = True
            merged.append((x1, y1, x2, y2))
        return merged
