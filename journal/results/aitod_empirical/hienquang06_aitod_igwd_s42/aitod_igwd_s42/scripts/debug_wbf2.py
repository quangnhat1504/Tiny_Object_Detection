"""Debug: same tile predictions, standard vs smart WBF, one image."""
from __future__ import annotations
import sys
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

FRCNN_CKPT = ROOT / "runs/sa_alw_full__la_loss__seed42/best.pt"

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
        all_tile_preds.append((p["boxes"].cpu(), p["scores"].cpu(), p["labels"].cpu()))
model.roi_heads.score_thresh = prev_thr

# Group by image
img_groups = {}
for idx in range(len(td)):
    img_idx = td.tile_index[idx][0]
    if img_idx not in img_groups:
        img_groups[img_idx] = {"tiles": [], "coords": []}
    img_groups[img_idx]["tiles"].append(all_tile_preds[idx])
    img_groups[img_idx]["coords"].append(td.tile_index[idx])

# Standard WBF
def standard_wbf(tile_preds, tile_coords, img_size, iou_thr=0.55):
    W, H = img_size
    all_boxes, all_scores, all_labels = [], [], []
    for (tx, ty, tw, th), (boxes, scores, labels) in zip(tile_coords, tile_preds):
        if boxes.numel() == 0:
            continue
        rm = boxes.clone()
        rm[:, 0] = rm[:, 0] * tw/512 + tx
        rm[:, 1] = rm[:, 1] * th/512 + ty
        rm[:, 2] = rm[:, 2] * tw/512 + tx
        rm[:, 3] = rm[:, 3] * th/512 + ty
        rm[:, 0].clamp_(0, W); rm[:, 1].clamp_(0, H)
        rm[:, 2].clamp_(0, W); rm[:, 3].clamp_(0, H)
        valid = (rm[:, 2]-rm[:, 0]>=2) & (rm[:, 3]-rm[:, 1]>=2)
        if valid.any():
            all_boxes.append(rm[valid])
            all_scores.append(scores[valid])
            all_labels.append(labels[valid])
    if not all_boxes:
        return torch.zeros(0,4), torch.zeros(0), torch.zeros(0,dtype=torch.int64)
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    if boxes.numel() <= 1:
        return boxes, scores, labels
    ious = box_iou(boxes, boxes)
    cid = torch.zeros(len(boxes), dtype=torch.long)
    nc = 1
    for i in range(len(boxes)):
        if cid[i] > 0: continue
        cid[i] = nc
        for j in range(i+1, len(boxes)):
            if cid[j] > 0: continue
            if ious[i,j] > iou_thr: cid[j] = nc
        nc += 1
    fb, fs, fl = [], [], []
    for c in range(1, nc):
        m = cid == c
        cb, cs, cl = boxes[m], scores[m], labels[m]
        w = cs / cs.sum()
        avg = (cb.T @ w).T
        fb.append(avg); fs.append(cs.mean()); fl.append(torch.mode(cl).values)
    return torch.stack(fb), torch.stack(fs), torch.stack(fl)

iou_thr = 0.60
score_thr = 0.10
total_std = 0
total_smart = 0

for img_idx in sorted(img_groups):
    group = img_groups[img_idx]
    cache_entry = td.labels_cache[img_idx]
    if len(cache_entry) == 2:
        boxes, (W, H) = cache_entry
    else:
        boxes, W, H = cache_entry

    img_coords = [(tx1, ty1, tx2-tx1, ty2-ty1)
                  for (_i, tx1, ty1, tx2, ty2) in group["coords"]]

    # Standard path: filter by score, then WBF
    filtered = []
    for (b, s, l) in group["tiles"]:
        keep = s >= score_thr
        filtered.append((b[keep], s[keep], l[keep]))
    sb, ss, sl = standard_wbf(filtered, img_coords, (W, H), iou_thr)
    total_std += len(sb)

    # Smart path: precomputed data + smart fuse
    from scripts.tune_smart_wbf import (
        precompute_image_data, smart_fuse_from_precomputed,
        _adaptive_threshold, _extent_hull, _weighted_avg,
    )
    pre = precompute_image_data(group["tiles"], img_coords, (W, H))
    if pre is None:
        continue
    fused = smart_fuse_from_precomputed(
        pre, base_iou_thr=iou_thr, score_thr=score_thr,
        fusion_mode="weighted_avg", adaptive_thr=False)
    total_smart += len(fused["boxes"])

    # Compare: number of input boxes (after score filter)
    n_std_input = sum(b.shape[0] for (b, _, _) in filtered)
    n_smart_input = (pre["scores"] >= score_thr).sum().item()

    if len(sb) != len(fused["boxes"]) or n_std_input != n_smart_input:
        print(f"Image {img_idx}: std={n_std_input}->{len(sb)} smart={n_smart_input}->{len(fused['boxes'])}  DIFF!")
    elif img_idx <= 3:  # Print first few for sanity
        print(f"Image {img_idx}: {n_std_input}->{len(sb)} OK")

print(f"\nTotal: std={total_std} boxes, smart={total_smart} boxes")
