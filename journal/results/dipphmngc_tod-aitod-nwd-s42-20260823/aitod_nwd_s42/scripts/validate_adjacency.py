"""
Quick validation: compare adjacency on vs off for one iou/score config.
"""
from __future__ import annotations
import json, sys, time, itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
import tqdm

from common.config import DEVICE, seed_all, SEED
from common.metrics import get_metric_fn
from common.model import build_model
from common.dataset import YOLOTinyDataset, collate_fn
from common.eval_utils import compute_scale_ap

FRCNN_CKPT = ROOT / "runs/sa_alw_full__la_loss__seed42/best.pt"


class _DSU:
    def __init__(self, n): self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr != yr: self.parent[yr] = xr


def _weighted_avg(boxes, scores):
    w = scores / scores.sum()
    return (boxes.T @ w).T


def fuse_one_image(tile_preds, tile_coords, img_size,
                   iou_thr=0.55, score_thr=0.10, use_adjacency=True):
    W, H = img_size
    all_boxes, all_scores, all_labels = [], [], []
    tile_ids = []
    tile_rects = []

    for tid, ((tx, ty, tw, th), (boxes, scores, labels)) in enumerate(
        zip(tile_coords, tile_preds)
    ):
        tile_rects.append((float(tx), float(ty),
                           float(tx + tw), float(ty + th)))
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
            tile_ids.extend([tid] * valid.sum().item())

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
    n = len(boxes)

    # Score filter
    keep = scores >= score_thr
    if not keep.any():
        return empty

    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    tile_ids = [tile_ids[i] for i in keep.nonzero(as_tuple=False).squeeze(-1).tolist()]
    n = len(boxes)

    if n <= 1:
        return {"boxes": boxes, "scores": scores, "labels": labels}

    dsu = _DSU(n)

    # Step 1: adjacency force-merge
    if use_adjacency and len(tile_rects) >= 2:
        tile_to_indices = {}
        for i, tid in enumerate(tile_ids):
            tile_to_indices.setdefault(tid, []).append(i)

        centers = torch.stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2,
        ], dim=1)

        for ta in range(len(tile_rects)):
            for tb in range(ta + 1, len(tile_rects)):
                ox1 = max(tile_rects[ta][0], tile_rects[tb][0])
                oy1 = max(tile_rects[ta][1], tile_rects[tb][1])
                ox2 = min(tile_rects[ta][2], tile_rects[tb][2])
                oy2 = min(tile_rects[ta][3], tile_rects[tb][3])
                if ox1 >= ox2 or oy1 >= oy2:
                    continue
                list_a = tile_to_indices.get(ta, [])
                list_b = tile_to_indices.get(tb, [])
                if not list_a or not list_b:
                    continue
                for i in list_a:
                    cxi, cyi = centers[i, 0].item(), centers[i, 1].item()
                    if not (ox1 <= cxi <= ox2 and oy1 <= cyi <= oy2):
                        continue
                    for j in list_b:
                        if dsu.find(i) == dsu.find(j):
                            continue
                        if labels[i] != labels[j]:
                            continue
                        cxj, cyj = centers[j, 0].item(), centers[j, 1].item()
                        if ox1 <= cxj <= ox2 and oy1 <= cyj <= oy2:
                            dsu.union(i, j)

    # Step 2: IoU-based union
    ious = box_iou(boxes, boxes)
    for i in range(n):
        for j in range(i + 1, n):
            if dsu.find(i) == dsu.find(j):
                continue
            if labels[i] != labels[j]:
                continue
            if ious[i, j] > iou_thr:
                dsu.union(i, j)

    # Step 3: fuse
    groups = {}
    for i in range(n):
        root = dsu.find(i)
        groups.setdefault(root, []).append(i)

    fused_b, fused_s, fused_l = [], [], []
    for indices in groups.values():
        idx = torch.tensor(indices)
        cb, cs, cl = boxes[idx], scores[idx], labels[idx]
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


def main():
    seed_all(SEED)
    ck = torch.load(FRCNN_CKPT, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    mt = get_metric_fn(cfg.get("metric", "sa_alw_full"))
    model = build_model(mt, "la_loss", cfg.get("reliability_thr", 16.0)).to(DEVICE)
    model.load_state_dict(ck["model"], strict=False)

    td = YOLOTinyDataset(
        img_dir=ROOT / "data/test/images",
        lbl_dir=ROOT / "data/test/labels",
        is_train=False,
    )
    print(f"Test tiles: {len(td)}")

    print("Collecting tile predictions...")
    model.eval()
    prev_thr = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = 0.001
    all_tile_preds = []
    loader = DataLoader(td, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=(DEVICE.type == "cuda"))
    for imgs, _ in tqdm.tqdm(loader, desc="FRCNN tiles"):
        imgs = [i.to(DEVICE) for i in imgs]
        with torch.no_grad():
            preds = model(imgs)
        for p in preds:
            all_tile_preds.append(
                (p["boxes"].cpu(), p["scores"].cpu(), p["labels"].cpu()))
    model.roi_heads.score_thresh = prev_thr

    img_groups = {}
    for idx in range(len(td)):
        img_idx = td.tile_index[idx][0]
        if img_idx not in img_groups:
            img_groups[img_idx] = {"tiles": [], "coords": []}
        img_groups[img_idx]["tiles"].append(all_tile_preds[idx])
        img_groups[img_idx]["coords"].append(td.tile_index[idx])
    print(f"Images: {len(img_groups)}")

    for iou_thr in [0.55, 0.60]:
        for score_thr in [0.10, 0.20]:
            for use_adj in [False, True]:
                all_preds, all_gts = [], []
                for img_idx, group in sorted(img_groups.items()):
                    cache_entry = td.labels_cache[img_idx]
                    if len(cache_entry) == 2:
                        boxes_raw, (W, H) = cache_entry
                    else:
                        boxes_raw, W, H = cache_entry

                    img_coords = [(tx1, ty1, tx2 - tx1, ty2 - ty1)
                                  for (_i, tx1, ty1, tx2, ty2) in group["coords"]]

                    fused = fuse_one_image(
                        group["tiles"], img_coords, (W, H),
                        iou_thr=iou_thr, score_thr=score_thr,
                        use_adjacency=use_adj)
                    all_preds.append(fused)

                    gt_boxes = torch.tensor(
                        [[b[1], b[2], b[3], b[4]] for b in boxes_raw
                         if b[3] > b[1] and b[4] > b[2]],
                        dtype=torch.float32)
                    gt_labels = torch.tensor(
                        [b[0] + 1 for b in boxes_raw if b[3] > b[1] and b[4] > b[2]],
                        dtype=torch.int64)
                    areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]) *
                             (gt_boxes[:, 3] - gt_boxes[:, 1])) if gt_boxes.numel() > 0 \
                        else torch.zeros(0)
                    all_gts.append({
                        "boxes": gt_boxes, "labels": gt_labels,
                        "area": areas,
                        "iscrowd": torch.zeros(len(gt_labels), dtype=torch.int64),
                        "image_id": torch.tensor([img_idx], dtype=torch.int64),
                    })

                sap = compute_scale_ap(all_preds, all_gts)
                tgt = sum(sap.get(f"n_gt_{s}", 0)
                          for s in ("micro", "tiny", "small", "large"))
                prim = (sum(sap.get(f"AP_{s}", 0) * sap.get(f"n_gt_{s}", 0)
                            for s in ("micro", "tiny", "small", "large"))
                        / max(tgt, 1))
                print(f"iou={iou_thr:.2f} scr={score_thr:.2f} adj={use_adj} "
                      f"-> mAP={prim:.4f} "
                      f"micro={sap.get('AP_micro',0):.4f} "
                      f"tiny={sap.get('AP_tiny',0):.4f} "
                      f"small={sap.get('AP_small',0):.4f} "
                      f"large={sap.get('AP_large',0):.4f}")


if __name__ == "__main__":
    main()
