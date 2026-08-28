"""
Cascaded Pipeline — FRCNN tile-scan -> Router -> FRCNN patch-refine -> WBF.

Architecture (Phase 4, revised):
  1. FRCNN does full-image tiling (512x512 stride 448), low score_thr=0.05
  2. Router: uncertain detections (score 0.05-0.30) -> crop patches
  3. FRCNN refines each patch at native resolution
  4. WBF fuses tile predictions + patch predictions

YOLO removed as stage 1 — too weak on this dataset (mAP@50=0.17).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.ops import box_iou

from .router import UncertaintyRouter


class CascadedDetector(nn.Module):
    def __init__(
        self,
        frcnn_model: nn.Module,
        router: UncertaintyRouter,
        device: torch.device,
        tile_size: int = 512,
        tile_overlap: int = 64,
        frcnn_scan_thr: float = 0.05,
        frcnn_refine_thr: float = 0.30,
        wbf_tile_weight: float = 1.0,
        wbf_patch_weight: float = 2.0,
        wbf_iou_thr: float = 0.55,
    ):
        super().__init__()
        self.frcnn = frcnn_model
        self.router = router
        self.device = device
        self.tile_size = tile_size
        self.stride = tile_size - tile_overlap
        self.frcnn_scan_thr = frcnn_scan_thr
        self.frcnn_refine_thr = frcnn_refine_thr
        self.wbf_tile_weight = wbf_tile_weight
        self.wbf_patch_weight = wbf_patch_weight
        self.wbf_iou_thr = wbf_iou_thr

    @torch.no_grad()
    def forward(self, images: List[Image.Image]) -> List[Dict]:
        results = []
        for img in images:
            W, H = img.size

            # Stage 1: FRCNN tile-scan
            tiles = self._tile_image(img)
            tile_detections = self._frcnn_on_tiles(tiles)

            # Remap to full-image coords
            all_boxes, all_scores, all_labels = [], [], []
            for (tx, ty), (boxes, scores, labels) in zip(self._tile_coords(img), tile_detections):
                if boxes.numel() == 0:
                    continue
                rm = boxes.clone()
                rm[:, 0] += tx; rm[:, 2] += tx
                rm[:, 1] += ty; rm[:, 3] += ty
                rm[:, 0].clamp_(0, W); rm[:, 2].clamp_(0, W)
                rm[:, 1].clamp_(0, H); rm[:, 3].clamp_(0, H)
                all_boxes.append(rm)
                all_scores.append(scores)
                all_labels.append(labels)

            if all_boxes:
                scan_boxes = torch.cat(all_boxes)
                scan_scores = torch.cat(all_scores)
                scan_labels = torch.cat(all_labels)
            else:
                scan_boxes = torch.empty(0, 4, device=self.device)
                scan_scores = torch.empty(0, device=self.device)
                scan_labels = torch.zeros(0, dtype=torch.int64, device=self.device)

            # Stage 2: Route uncertain detections to patches
            uncertain_mask = (scan_scores >= self.frcnn_scan_thr) & \
                             (scan_scores <= 0.30)  # upper bound for uncertainty
            if uncertain_mask.any():
                patches = []
                for box in scan_boxes[uncertain_mask]:
                    patches.append(self._box_to_patch(box, W, H))
                patches = UncertaintyRouter._merge_overlapping(patches, 0.5)
                patches = [(max(0,int(p[0])), max(0,int(p[1])), min(W,int(p[2])), min(H,int(p[3])))
                          for p in patches]
            else:
                patches = []

            # Stage 3: FRCNN refine on patches
            patch_dets = self._refine_patches(img, patches)

            # Stage 4: WBF
            fused = self._weighted_boxes_fusion(
                scan_boxes, scan_scores, scan_labels, patch_dets)

            results.append(fused)
        return results

    def _tile_image(self, img: Image.Image) -> List[Image.Image]:
        tiles = []
        for (tx, ty) in self._tile_coords(img):
            tile = img.crop((tx, ty, min(tx+self.tile_size, img.width),
                             min(ty+self.tile_size, img.height)))
            tile = tile.resize((self.tile_size, self.tile_size))
            tiles.append(tile)
        return tiles

    def _tile_coords(self, img: Image.Image) -> List[Tuple[int, int]]:
        coords = []
        W, H = img.size
        for y in range(0, max(1, H - self.tile_size + self.stride), self.stride):
            for x in range(0, max(1, W - self.tile_size + self.stride), self.stride):
                coords.append((
                    min(x, max(0, W - self.tile_size)),
                    min(y, max(0, H - self.tile_size)),
                ))
        return coords

    def _frcnn_on_tiles(self, tiles) -> List[Tuple]:
        results = []
        self.frcnn.eval()
        prev_thr = self.frcnn.roi_heads.score_thresh
        self.frcnn.roi_heads.score_thresh = self.frcnn_scan_thr

        for tile in tiles:
            t = TF.to_tensor(tile).to(self.device)
            pred = self.frcnn([t])[0]
            keep = pred["scores"] >= self.frcnn_scan_thr
            results.append((
                pred["boxes"][keep], pred["scores"][keep], pred["labels"][keep]))

        self.frcnn.roi_heads.score_thresh = prev_thr
        return results

    def _box_to_patch(self, box, W, H):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        pw = max(float(bw) * self.router.context_ratio, float(self.router.patch_size))
        ph = max(float(bh) * self.router.context_ratio, float(self.router.patch_size))
        return (cx - pw/2, cy - ph/2, cx + pw/2, cy + ph/2)

    def _refine_patches(self, img, patches):
        if not patches:
            return []
        all_boxes, all_scores, all_labels = [], [], []
        W, H = img.size
        for (x1, y1, x2, y2) in patches:
            pw, ph = x2 - x1, y2 - y1
            if pw < 10 or ph < 10:
                continue
            patch_img = img.crop((x1, y1, x2, y2))
            patch_img = patch_img.resize((self.router.patch_size, self.router.patch_size))
            pt = TF.to_tensor(patch_img).to(self.device)
            self.frcnn.eval()
            pred = self.frcnn([pt])[0]
            keep = pred["scores"] >= self.frcnn_refine_thr
            boxes = pred["boxes"][keep]; scores = pred["scores"][keep]; labels = pred["labels"][keep]
            if boxes.numel() == 0:
                continue
            sx = pw / self.router.patch_size; sy = ph / self.router.patch_size
            rm = boxes.clone()
            rm[:,0] = boxes[:,0]*sx + x1; rm[:,1] = boxes[:,1]*sy + y1
            rm[:,2] = boxes[:,2]*sx + x1; rm[:,3] = boxes[:,3]*sy + y1
            rm[:,0].clamp_(0,W); rm[:,1].clamp_(0,H); rm[:,2].clamp_(0,W); rm[:,3].clamp_(0,H)
            valid = (rm[:,2]-rm[:,0]>=2) & (rm[:,3]-rm[:,1]>=2)
            all_boxes.append(rm[valid]); all_scores.append(scores[valid]); all_labels.append(labels[valid])
        if all_boxes:
            return [{"boxes": torch.cat(all_boxes), "scores": torch.cat(all_scores), "labels": torch.cat(all_labels)}]
        return []

    def _weighted_boxes_fusion(self, scan_boxes, scan_scores, scan_labels, patch_dets):
        all_boxes = []; all_scores = []; all_labels = []
        if scan_boxes.numel() > 0:
            all_boxes.append(scan_boxes)
            all_scores.append(scan_scores * self.wbf_tile_weight)
            all_labels.append(scan_labels)
        for pd in patch_dets:
            if pd["boxes"].numel() > 0:
                all_boxes.append(pd["boxes"])
                all_scores.append(pd["scores"] * self.wbf_patch_weight)
                all_labels.append(pd["labels"])
        if not all_boxes:
            return {"boxes": torch.empty(0,4,device=self.device),
                    "scores": torch.empty(0,device=self.device),
                    "labels": torch.zeros(0,dtype=torch.int64,device=self.device)}
        boxes = torch.cat(all_boxes); scores = torch.cat(all_scores); labels = torch.cat(all_labels)
        if boxes.numel() <= 1:
            return {"boxes": boxes, "scores": scores, "labels": labels}
        ious = box_iou(boxes, boxes)
        cid = torch.zeros(len(boxes), dtype=torch.long, device=self.device)
        nc = 1
        for i in range(len(boxes)):
            if cid[i] > 0: continue
            cid[i] = nc
            for j in range(i+1, len(boxes)):
                if cid[j] > 0: continue
                if ious[i,j] > self.wbf_iou_thr: cid[j] = nc
            nc += 1
        fb, fs, fl = [], [], []
        for c in range(1, nc):
            m = cid == c
            cb = boxes[m]; cs = scores[m]; cl = labels[m]
            w = cs / cs.sum()
            avg = (cb.T @ w).T
            fb.append(avg); fs.append(cs.mean()); fl.append(torch.mode(cl).values)
        return {"boxes": torch.stack(fb), "scores": torch.stack(fs), "labels": torch.stack(fl)}
