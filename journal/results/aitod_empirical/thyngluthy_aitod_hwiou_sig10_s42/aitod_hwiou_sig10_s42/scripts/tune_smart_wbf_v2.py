"""Grid search SmartWBF params — loads cached tile predictions (fast)."""
from __future__ import annotations
import argparse, json, sys, time, itertools, pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import tqdm

from common.config import seed_all, SEED
from common.eval_utils import compute_scale_ap, evaluate_coco
from common.wbf import wbf_fusion_smart


GRID = {
    "iou_thr": [0.50, 0.55, 0.60],
    "score_thr": [0.10, 0.20, 0.30],
    "fusion_mode": ["weighted_avg", "extent_hull", "ap75_hybrid"],
    "adaptive_thr": [False, True],
}


def _parse_args():
    p = argparse.ArgumentParser(description="Grid search SmartWBF params from cached predictions")
    p.add_argument("--split", choices=["valid", "test"], default="valid",
                   help="Dataset split (default: valid — only tune on valid never test)")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None,
                   help="Device override")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Checkpoint used for cache naming (default auto)")
    p.add_argument("--pred-file", type=Path, default=None,
                   help="Prediction cache path (default: runs/tile_preds_<split>_seed42.pkl)")
    p.add_argument("--max-images", type=int, default=None,
                   help="Evaluate only first N images")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON path")
    return p.parse_args()


def _topk_for_coco(preds, k=100):
    out = []
    for pred in preds:
        scores = pred["scores"]
        if len(scores) > k:
            keep = torch.topk(scores, k=k).indices
            out.append({
                "boxes": pred["boxes"][keep],
                "scores": pred["scores"][keep],
                "labels": pred["labels"][keep],
            })
        else:
            out.append(pred)
    return out


def run_config(tile_preds, tile_index, labels_cache, max_images=None, **params):
    img_groups = {}
    for idx in range(len(tile_index)):
        img_idx = tile_index[idx][0]
        if img_idx not in img_groups:
            img_groups[img_idx] = {"tiles": [], "coords": []}
        img_groups[img_idx]["tiles"].append(tile_preds[idx])
        img_groups[img_idx]["coords"].append(tile_index[idx])

    if max_images is not None:
        keep_ids = sorted(img_groups)[:max_images]
        img_groups = {img_idx: img_groups[img_idx] for img_idx in keep_ids}

    all_preds, all_gts = [], []
    for img_idx in sorted(img_groups):
        group = img_groups[img_idx]
        cache_entry = labels_cache[img_idx]
        if len(cache_entry) == 2:
            boxes, (W, H) = cache_entry
        else:
            boxes, W, H = cache_entry

        filtered = []
        for (b, s, l) in group["tiles"]:
            keep = s >= params["score_thr"]
            filtered.append((b[keep], s[keep], l[keep]))

        img_coords = [(tx1, ty1, tx2 - tx1, ty2 - ty1)
                      for (_i, tx1, ty1, tx2, ty2) in group["coords"]]

        fused = wbf_fusion_smart(
            filtered, img_coords, (W, H),
            iou_thr=params["iou_thr"],
            fusion_mode=params["fusion_mode"],
            adaptive_thr=params["adaptive_thr"],
        )
        all_preds.append(fused)

        gt_boxes = torch.tensor(
            [[b[1], b[2], b[3], b[4]] for b in boxes if b[3] > b[1] and b[4] > b[2]],
            dtype=torch.float32)
        gt_labels = torch.tensor(
            [b[0] + 1 for b in boxes if b[3] > b[1] and b[4] > b[2]],
            dtype=torch.int64)
        areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]) *
                 (gt_boxes[:, 3] - gt_boxes[:, 1])) if gt_boxes.numel() > 0 else torch.zeros(0)
        all_gts.append({
            "boxes": gt_boxes, "labels": gt_labels,
            "area": areas,
            "iscrowd": torch.zeros(len(gt_labels), dtype=torch.int64),
            "image_id": torch.tensor([img_idx], dtype=torch.int64),
        })

    sap = compute_scale_ap(all_preds, all_gts)
    coco = evaluate_coco(_topk_for_coco(all_preds, 100), all_gts, class_metrics=False)
    tgt = sum(sap.get(f"n_gt_{s}", 0) for s in ("micro", "tiny", "small", "large"))
    prim = (sum(sap.get(f"AP_{s}", 0) * sap.get(f"n_gt_{s}", 0)
                for s in ("micro", "tiny", "small", "large")) / max(tgt, 1))

    return {
        "mAP(scale)": round(prim, 4),
        "AP_micro": round(sap.get("AP_micro", 0), 4),
        "AP_tiny": round(sap.get("AP_tiny", 0), 4),
        "AP_small": round(sap.get("AP_small", 0), 4),
        "AP_large": round(sap.get("AP_large", 0), 4),
        "coco_AP": round(coco.get("coco_AP", 0), 4),
        "coco_AP50": round(coco.get("coco_AP50", 0), 4),
        "coco_AP75": round(coco.get("coco_AP75", 0), 4),
        "coco_AP_small": round(coco.get("coco_AP_small", 0), 4),
        "coco_AP_medium": round(coco.get("coco_AP_medium", 0), 4),
        "coco_AP_large": round(coco.get("coco_AP_large", 0), 4),
        "coco_AR100": round(coco.get("coco_AR100", 0), 4),
    }


def main():
    args = _parse_args()
    seed_all(SEED)

    device = torch.device("cuda" if (args.device or "cuda") == "cuda"
                          and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pred_file = args.pred_file or (ROOT / f"runs/tile_preds_{args.split}_seed42.pkl")
    if not pred_file.is_absolute():
        pred_file = ROOT / pred_file
    if not pred_file.exists():
        raise SystemExit(f"Cache not found: {pred_file}\n"
                         f"Run: python scripts/cache_predictions.py --split {args.split} --device {device.type}")

    print(f"Loading cached predictions: {pred_file.name}")
    with open(pred_file, "rb") as f:
        data = pickle.load(f)
    if data.get("meta"):
        print(f"Cache meta: {data['meta']}")
    all_tile_preds = [(p["boxes"], p["scores"], p["labels"]) for p in data["preds"]]
    tile_index = data["tile_index"]
    labels_cache = data["labels_cache"]
    n_images = len(set(ti[0] for ti in tile_index))
    if args.max_images is not None:
        print(f"  Tiles: {len(all_tile_preds)}, Images: {n_images}, debug subset: {args.max_images}")
    else:
        print(f"  Tiles: {len(all_tile_preds)}, Images: {n_images}")

    keys = list(GRID.keys())
    combinations = list(itertools.product(*GRID.values()))
    total = len(combinations)
    print(f"Configs: {total}")

    results = []
    t0 = time.time()
    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        metrics = run_config(
            all_tile_preds, tile_index, labels_cache,
            max_images=args.max_images, **params)
        entry = {**params, **metrics}
        results.append(entry)
        elapsed = time.time() - t0
        eta = elapsed / idx * (total - idx) if idx < total else 0
        print(f"  [{idx:>2}/{total}] {params['fusion_mode']:>13} "
              f"iou={params['iou_thr']:.2f} scr={params['score_thr']:.2f} "
              f"adapt={str(params['adaptive_thr']):>5} "
              f"-> coco75={metrics['coco_AP75']:.4f} "
              f"cocoAP={metrics['coco_AP']:.4f} mAP={metrics['mAP(scale)']:.4f} "
              f"({elapsed:.0f}s, eta={eta:.0f}s)")

    results.sort(key=lambda r: (r.get("coco_AP75", 0),
                                r.get("coco_AP", 0),
                                r["mAP(scale)"]),
                 reverse=True)

    print("\n" + "=" * 100)
    print("TOP 5 — SmartWBF Grid Search")
    print("=" * 100)
    for i, r in enumerate(results[:5], 1):
        print(f"  #{i}: {r['fusion_mode']:>13} iou={r['iou_thr']:.2f} scr={r['score_thr']:.2f} "
              f"adaptive={r['adaptive_thr']} "
              f"coco75={r['coco_AP75']:.4f} cocoAP={r['coco_AP']:.4f} "
              f"mAP={r['mAP(scale)']:.4f} micro={r['AP_micro']:.4f} large={r['AP_large']:.4f}")

    out_file = args.out or (ROOT / f"runs/smart_wbf_v2_{args.split}.json")
    if not out_file.is_absolute():
        out_file = ROOT / out_file
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
