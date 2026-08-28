"""Grid search WBF — 36 configs, writes progress to runs/grid_progress.json after each."""
import pickle, sys, time, itertools, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from common.wbf import wbf_fusion_smart
from common.eval_utils import compute_scale_ap

PRED_FILE = ROOT / "runs/tile_preds_seed42.pkl"
PROGRESS_FILE = ROOT / "runs/grid_progress.json"
FINAL_FILE = ROOT / "runs/smart_wbf_final.json"

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

GRID = {
    "iou_thr": [0.55, 0.60, 0.65],
    "score_thr": [0.10, 0.15, 0.20],
    "fusion_mode": ["weighted_avg", "extent_hull"],
    "adaptive_thr": [False, True],
}

keys = list(GRID.keys())
combos = list(itertools.product(*GRID.values()))
total = len(combos)

results = []
t0 = time.time()

for idx, combo in enumerate(combos, 1):
    params = dict(zip(keys, combo))
    c0 = time.time()
    all_preds, all_gts = [], []
    for img_idx, group in sorted(img_groups.items()):
        cache_entry = labels_cache[img_idx]
        if len(cache_entry) == 2:
            boxes_raw, (W, H) = cache_entry
        else:
            boxes_raw, W, H = cache_entry
        filtered = []
        for (b, s, l) in group["tiles"]:
            keep = s >= params["score_thr"]
            filtered.append((b[keep], s[keep], l[keep]))
        img_coords = [(tx1, ty1, tx2 - tx1, ty2 - ty1)
                      for (_i, tx1, ty1, tx2, ty2) in group["coords"]]
        fused = wbf_fusion_smart(filtered, img_coords, (W, H),
                                 iou_thr=params["iou_thr"],
                                 fusion_mode=params["fusion_mode"],
                                 adaptive_thr=params["adaptive_thr"])
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
            "boxes": gt_boxes, "labels": gt_labels, "area": areas,
            "iscrowd": torch.zeros(len(gt_labels), dtype=torch.int64),
            "image_id": torch.tensor([img_idx], dtype=torch.int64),
        })
    sap = compute_scale_ap(all_preds, all_gts)
    tgt = sum(sap.get(f"n_gt_{s}", 0) for s in ("micro", "tiny", "small", "large"))
    prim = (sum(sap.get(f"AP_{s}", 0) * sap.get(f"n_gt_{s}", 0)
                for s in ("micro", "tiny", "small", "large")) / max(tgt, 1))
    entry = {
        **params,
        "mAP(scale)": round(prim, 4),
        "AP_micro": round(sap.get("AP_micro", 0), 4),
        "AP_tiny": round(sap.get("AP_tiny", 0), 4),
        "AP_small": round(sap.get("AP_small", 0), 4),
        "AP_large": round(sap.get("AP_large", 0), 4),
    }
    results.append(entry)

    dt = time.time() - c0
    elapsed = time.time() - t0
    eta = elapsed / idx * (total - idx) if idx < total else 0

    # Save progress after each config
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "done": idx, "total": total,
            "elapsed_s": round(elapsed, 0),
            "eta_s": round(eta, 0),
            "last_mAP": prim,
            "results": results,
        }, f, indent=2)

results.sort(key=lambda r: r["mAP(scale)"], reverse=True)

with open(FINAL_FILE, "w") as f:
    json.dump(results, f, indent=2)
