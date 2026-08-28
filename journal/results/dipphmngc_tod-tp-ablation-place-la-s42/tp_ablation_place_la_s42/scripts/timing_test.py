"""Timing test: how long does one WBF+metric config take?"""
import pickle, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from common.wbf import wbf_fusion_smart
from common.eval_utils import compute_scale_ap

PRED_FILE = ROOT / "runs/tile_preds_seed42.pkl"

with open(PRED_FILE, "rb") as f:
    data = pickle.load(f)

all_tile_preds = [(p["boxes"], p["scores"], p["labels"]) for p in data["preds"]]
tile_index = data["tile_index"]
labels_cache = data["labels_cache"]

img_groups = {}
for idx in range(len(tile_index)):
    img_idx = tile_index[idx][0]
    if img_idx not in img_groups:
        img_groups[img_idx] = {"tiles": [], "coords": []}
    img_groups[img_idx]["tiles"].append(all_tile_preds[idx])
    img_groups[img_idx]["coords"].append(tile_index[idx])

t0 = time.time()
all_preds, all_gts = [], []
for img_idx, group in sorted(img_groups.items()):
    cache_entry = labels_cache[img_idx]
    if len(cache_entry) == 2:
        boxes_raw, (W, H) = cache_entry
    else:
        boxes_raw, W, H = cache_entry
    filtered = []
    for (b, s, l) in group["tiles"]:
        keep = s >= 0.10
        filtered.append((b[keep], s[keep], l[keep]))
    img_coords = [(tx1, ty1, tx2 - tx1, ty2 - ty1)
                  for (_i, tx1, ty1, tx2, ty2) in group["coords"]]
    fused = wbf_fusion_smart(
        filtered, img_coords, (W, H),
        iou_thr=0.60, fusion_mode="weighted_avg", adaptive_thr=False)
    all_preds.append(fused)
    gt_boxes = torch.tensor(
        [[b[1], b[2], b[3], b[4]] for b in boxes_raw
         if b[3] > b[1] and b[4] > b[2]], dtype=torch.float32)
    gt_labels = torch.tensor(
        [b[0] + 1 for b in boxes_raw if b[3] > b[1] and b[4] > b[2]],
        dtype=torch.int64)
    areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]) *
             (gt_boxes[:, 3] - gt_boxes[:, 1])) \
        if gt_boxes.numel() > 0 else torch.zeros(0)
    all_gts.append({
        "boxes": gt_boxes, "labels": gt_labels,
        "area": areas,
        "iscrowd": torch.zeros(len(gt_labels), dtype=torch.int64),
        "image_id": torch.tensor([img_idx], dtype=torch.int64),
    })
dt_wbf = time.time() - t0
print(f"WBF: {dt_wbf:.1f}s", flush=True)

t0 = time.time()
sap = compute_scale_ap(all_preds, all_gts)
dt_metric = time.time() - t0
tgt = sum(sap.get(f"n_gt_{s}", 0) for s in ("micro", "tiny", "small", "large"))
prim = (sum(sap.get(f"AP_{s}", 0) * sap.get(f"n_gt_{s}", 0)
            for s in ("micro", "tiny", "small", "large")) / max(tgt, 1))
print(f"Metric: {dt_metric:.1f}s", flush=True)
print(f"Total: {dt_wbf+dt_metric:.1f}s, mAP={prim:.4f}", flush=True)
