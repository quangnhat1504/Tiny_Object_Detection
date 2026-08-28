"""Minimal test: load cached preds, test 6 configs."""
import pickle, sys, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torchvision.ops import box_iou

from common.eval_utils import compute_scale_ap

PRED_FILE = ROOT / "runs/tile_preds_seed42.pkl"

def _weighted_avg(boxes, scores):
    w = scores / scores.sum()
    return (boxes.T @ w).T

def _extent_hull(boxes, scores, iqr_mult=1.5):
    if len(boxes) <= 1:
        return boxes[0] if len(boxes) == 1 else None
    def _robust(v, fn, mult):
        q1, q3 = torch.quantile(v, torch.tensor([0.25, 0.75]))
        iqr = q3 - q1
        bound = q1 - mult * iqr if fn == "min" else q3 + mult * iqr
        valid = v >= bound if fn == "min" else v <= bound
        return (v[valid].min().item() if fn == "min"
                else v[valid].max().item()) if valid.any() else fn(v)
    return torch.tensor([
        _robust(boxes[:, 0], "min", iqr_mult),
        _robust(boxes[:, 1], "min", iqr_mult),
        _robust(boxes[:, 2], "max", iqr_mult),
        _robust(boxes[:, 3], "max", iqr_mult),
    ])

def _adaptive_iou_thr(area, base_thr, use_adaptive):
    if not use_adaptive: return base_thr
    if area < 256: return base_thr
    if area < 1024: return max(base_thr - 0.05, 0.25)
    if area < 4096: return max(base_thr - 0.15, 0.20)
    return max(base_thr - 0.25, 0.10)

def fuse_one(tile_preds, tile_coords, img_size, iou_thr=0.55,
             score_thr=0.10, fusion_mode="weighted_avg", use_adaptive=False):
    W, H = img_size
    all_boxes, all_scores, all_labels = [], [], []
    for (tx, ty, tw, th), (boxes, scores, labels) in zip(tile_coords, tile_preds):
        if boxes.numel() == 0: continue
        rm = boxes.clone()
        rm[:, 0] = rm[:, 0] * tw / 512 + tx; rm[:, 1] = rm[:, 1] * th / 512 + ty
        rm[:, 2] = rm[:, 2] * tw / 512 + tx; rm[:, 3] = rm[:, 3] * th / 512 + ty
        rm[:, 0].clamp_(0, W); rm[:, 1].clamp_(0, H)
        rm[:, 2].clamp_(0, W); rm[:, 3].clamp_(0, H)
        valid = (rm[:, 2] - rm[:, 0] >= 2) & (rm[:, 3] - rm[:, 1] >= 2)
        if valid.any():
            all_boxes.append(rm[valid])
            all_scores.append(scores[valid])
            all_labels.append(labels[valid])
    empty = {"boxes": torch.zeros(0,4), "scores": torch.zeros(0),
             "labels": torch.zeros(0, dtype=torch.int64)}
    if not all_boxes: return empty
    boxes = torch.cat(all_boxes); scores = torch.cat(all_scores); labels = torch.cat(all_labels)
    keep = scores >= score_thr
    if not keep.any(): return empty
    boxes = boxes[keep]; scores = scores[keep]; labels = labels[keep]
    n = len(boxes)
    if n <= 1: return {"boxes": boxes, "scores": scores, "labels": labels}
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ious = box_iou(boxes, boxes)
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        xr, yr = find(x), find(y)
        if xr != yr: parent[yr] = xr
    for i in range(n):
        for j in range(i+1, n):
            if find(i) == find(j): continue
            if labels[i] != labels[j]: continue
            thr = _adaptive_iou_thr(min(areas[i].item(), areas[j].item()), iou_thr, use_adaptive)
            if ious[i,j] > thr: union(i,j)
    groups = {}
    for i in range(n):
        root = find(i); groups.setdefault(root, []).append(i)
    fused_b, fused_s, fused_l = [], [], []
    for indices in groups.values():
        idx = torch.tensor(indices); cb, cs, cl = boxes[idx], scores[idx], labels[idx]
        fb = _extent_hull(cb, cs) if fusion_mode == "extent_hull" else _weighted_avg(cb, cs)
        if fb is not None:
            fused_b.append(fb); fused_s.append(cs.mean()); fused_l.append(torch.mode(cl).values)
    if not fused_b: return empty
    return {"boxes": torch.stack(fused_b), "scores": torch.stack(fused_s), "labels": torch.stack(fused_l)}

# ---- Main ----
print("Loading cached predictions...")
with open(PRED_FILE, "rb") as f: data = pickle.load(f)
all_tile_preds = [(p["boxes"], p["scores"], p["labels"]) for p in data["preds"]]
tile_index = data["tile_index"]; labels_cache = data["labels_cache"]

img_groups = {}
for idx in range(len(tile_index)):
    img_idx = tile_index[idx][0]
    if img_idx not in img_groups:
        img_groups[img_idx] = {"tiles": [], "coords": []}
    img_groups[img_idx]["tiles"].append(all_tile_preds[idx])
    img_groups[img_idx]["coords"].append(tile_index[idx])
print(f"Images: {len(img_groups)}")

test_configs = [
    # Standard WBF (baseline)
    {"iou_thr": 0.55, "score_thr": 0.20, "fusion_mode": "weighted_avg", "use_adaptive": False},
    {"iou_thr": 0.60, "score_thr": 0.20, "fusion_mode": "weighted_avg", "use_adaptive": False},
    # Adaptive only
    {"iou_thr": 0.55, "score_thr": 0.20, "fusion_mode": "weighted_avg", "use_adaptive": True},
    {"iou_thr": 0.50, "score_thr": 0.10, "fusion_mode": "weighted_avg", "use_adaptive": True},
    # Extent hull
    {"iou_thr": 0.55, "score_thr": 0.20, "fusion_mode": "extent_hull", "use_adaptive": False},
    {"iou_thr": 0.50, "score_thr": 0.10, "fusion_mode": "extent_hull", "use_adaptive": True},
]

print(f"\nTesting {len(test_configs)} configs...")
for cfg in test_configs:
    t0 = time.time()
    all_preds, all_gts = [], []
    for img_idx, group in sorted(img_groups.items()):
        cache_entry = labels_cache[img_idx]
        if len(cache_entry) == 2: boxes_raw, (W, H) = cache_entry
        else: boxes_raw, W, H = cache_entry
        filtered = []
        for (b, s, l) in group["tiles"]:
            keep = s >= cfg["score_thr"]; filtered.append((b[keep], s[keep], l[keep]))
        img_coords = [(tx1, ty1, tx2-tx1, ty2-ty1)
                      for (_i, tx1, ty1, tx2, ty2) in group["coords"]]
        fused = fuse_one(filtered, img_coords, (W, H), **cfg)
        all_preds.append(fused)
        gt_boxes = torch.tensor(
            [[b[1], b[2], b[3], b[4]] for b in boxes_raw if b[3] > b[1] and b[4] > b[2]],
            dtype=torch.float32)
        gt_labels = torch.tensor(
            [b[0] + 1 for b in boxes_raw if b[3] > b[1] and b[4] > b[2]], dtype=torch.int64)
        areas = ((gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:,3]-gt_boxes[:,1])) \
            if gt_boxes.numel() > 0 else torch.zeros(0)
        all_gts.append({"boxes": gt_boxes, "labels": gt_labels, "area": areas,
                        "iscrowd": torch.zeros(len(gt_labels), dtype=torch.int64),
                        "image_id": torch.tensor([img_idx], dtype=torch.int64)})
    sap = compute_scale_ap(all_preds, all_gts)
    tgt = sum(sap.get(f"n_gt_{s}", 0) for s in ("micro","tiny","small","large"))
    prim = sum(sap.get(f"AP_{s}",0)*sap.get(f"n_gt_{s}",0)
               for s in ("micro","tiny","small","large"))/max(tgt,1)
    dt = time.time() - t0
    print(f"  iou={cfg['iou_thr']:.2f} scr={cfg['score_thr']:.2f} "
          f"{cfg['fusion_mode']:>13} adapt={cfg['use_adaptive']} "
          f"-> mAP={prim:.4f} ({dt:.1f}s)")
    del all_preds, all_gts  # free memory
