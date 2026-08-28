"""Quick grid: 12 configs, foreground."""
import pickle, sys, time, json
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

# 12 configs: iou x3, scr x2, mode x2
configs = []
for iou in [0.55, 0.60, 0.65]:
    for scr in [0.10, 0.20]:
        for mode in ["weighted_avg", "extent_hull"]:
            configs.append((iou, scr, mode))

results = []
t0 = time.time()
for idx, (iou, scr, mode) in enumerate(configs, 1):
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
            keep = s >= scr
            filtered.append((b[keep], s[keep], l[keep]))
        img_coords = [(tx1, ty1, tx2 - tx1, ty2 - ty1)
                      for (_i, tx1, ty1, tx2, ty2) in group["coords"]]
        fused = wbf_fusion_smart(
            filtered, img_coords, (W, H),
            iou_thr=iou, fusion_mode=mode, adaptive_thr=False)
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
        "iou_thr": iou, "score_thr": scr, "fusion_mode": mode,
        "mAP(scale)": round(prim, 4),
        "AP_micro": round(sap.get("AP_micro", 0), 4),
        "AP_tiny": round(sap.get("AP_tiny", 0), 4),
        "AP_small": round(sap.get("AP_small", 0), 4),
        "AP_large": round(sap.get("AP_large", 0), 4),
    }
    results.append(entry)
    dt = time.time() - c0
    elapsed = time.time() - t0
    eta = elapsed / idx * (len(configs) - idx) if idx < len(configs) else 0
    print(f"[{idx:>2}/{len(configs)}] iou={iou:.2f} scr={scr:.2f} {mode:>13} "
          f"-> mAP={prim:.4f} ({dt:.0f}s, eta={eta:.0f}s)",
          flush=True)

results.sort(key=lambda r: r["mAP(scale)"], reverse=True)
print("\nTOP 5:")
for i, r in enumerate(results[:5], 1):
    print(f"  #{i}: mAP={r['mAP(scale)']:.4f} iou={r['iou_thr']:.2f} "
          f"scr={r['score_thr']:.2f} {r['fusion_mode']:>13} "
          f"micro={r['AP_micro']:.4f} tiny={r['AP_tiny']:.4f} "
          f"small={r['AP_small']:.4f} large={r['AP_large']:.4f}",
          flush=True)

with open(ROOT / "runs/smart_wbf_quick.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved runs/smart_wbf_quick.json")
