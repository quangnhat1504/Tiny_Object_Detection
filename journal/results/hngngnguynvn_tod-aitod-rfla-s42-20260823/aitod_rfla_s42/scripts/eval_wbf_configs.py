"""Evaluate WBF configs from cached tile predictions."""
import argparse, sys, time, pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from common.wbf import wbf_fusion_smart
from common.eval_utils import compute_scale_ap, evaluate_coco


DEFAULT_CONFIGS = [
    ("weighted_avg", 0.55, 0.20, False),
    ("weighted_avg", 0.60, 0.10, False),
    ("extent_hull", 0.55, 0.20, False),
    ("extent_hull", 0.50, 0.10, False),
    ("ap75_hybrid", 0.55, 0.10, False),
    ("ap75_hybrid", 0.55, 0.20, False),
    ("ap75_hybrid", 0.60, 0.10, False),
]


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate cached WBF configs")
    p.add_argument("--split", choices=["valid", "test"], default="valid",
                   help="Dataset split")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None,
                   help="Device override")
    p.add_argument("--max-images", type=int, default=None,
                   help="Evaluate only first N images")
    p.add_argument("--modes", nargs="*", default=None,
                   help="Fusion modes to evaluate")
    p.add_argument("--iou-thrs", nargs="*", type=float, default=None,
                   help="IoU thresholds")
    p.add_argument("--score-thrs", nargs="*", type=float, default=None,
                   help="Score thresholds")
    p.add_argument("--only", nargs="*", default=None,
                   help="Exact configs as mode:iou:score[:adaptive]")
    p.add_argument("--pred-file", type=Path, default=None,
                   help="Prediction cache path (default: runs/tile_preds_<split>_seed42.pkl)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output text path")
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


def _filter_configs(configs, args):
    if args.only:
        wanted = set()
        for spec in args.only:
            parts = spec.split(":")
            if len(parts) not in (3, 4):
                raise SystemExit(f"Bad --only config: {spec}")
            mode, iou, score = parts[:3]
            adaptive = parts[3].lower() in ("1", "true", "yes", "y") if len(parts) == 4 else False
            wanted.add((mode, float(iou), float(score), adaptive))
        return [cfg for cfg in configs if cfg in wanted]

    modes = set(args.modes) if args.modes else None
    ious = set(args.iou_thrs) if args.iou_thrs else None
    scores = set(args.score_thrs) if args.score_thrs else None
    out = []
    for mode, iou, score, adaptive in configs:
        if modes is not None and mode not in modes:
            continue
        if ious is not None and iou not in ious:
            continue
        if scores is not None and score not in scores:
            continue
        out.append((mode, iou, score, adaptive))
    return out


def main():
    args = _parse_args()

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

    img_groups = {}
    for idx in range(len(tile_index)):
        img_idx = tile_index[idx][0]
        if img_idx not in img_groups:
            img_groups[img_idx] = {"tiles": [], "coords": []}
        img_groups[img_idx]["tiles"].append(all_tile_preds[idx])
        img_groups[img_idx]["coords"].append(tile_index[idx])
    print(f"Images: {len(img_groups)}")

    if args.max_images is not None:
        keep_ids = sorted(img_groups)[:args.max_images]
        img_groups = {img_idx: img_groups[img_idx] for img_idx in keep_ids}
        print(f"Debug subset: {len(img_groups)} images")

    configs = _filter_configs(DEFAULT_CONFIGS, args)
    if not configs:
        raise SystemExit("No configs selected")
    print(f"Selected configs: {configs}")

    results = []
    for cfg_idx, (mode, iou, scr, adaptive) in enumerate(configs, 1):
        print(f"Config {cfg_idx}/{len(configs)}: mode={mode} iou={iou:.2f} score={scr:.2f} adaptive={adaptive}", flush=True)
        all_preds, all_gts = [], []
        t0 = time.time()
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
                iou_thr=iou, fusion_mode=mode, adaptive_thr=adaptive)
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

        t0 = time.time()
        sap = compute_scale_ap(all_preds, all_gts)
        coco = evaluate_coco(_topk_for_coco(all_preds, 100), all_gts, class_metrics=False)
        dt_metric = time.time() - t0

        tgt = sum(sap.get(f"n_gt_{s}", 0) for s in ("micro", "tiny", "small", "large"))
        prim = (sum(sap.get(f"AP_{s}", 0) * sap.get(f"n_gt_{s}", 0)
                    for s in ("micro", "tiny", "small", "large")) / max(tgt, 1))

        line = (f"{mode:>13} iou={iou:.2f} scr={scr:.2f} adapt={adaptive} "
                f"-> mAP={prim:.4f} "
                f"micro={sap.get('AP_micro',0):.4f} "
                f"tiny={sap.get('AP_tiny',0):.4f} "
                f"small={sap.get('AP_small',0):.4f} "
                f"large={sap.get('AP_large',0):.4f} "
                f"cocoAP={coco.get('coco_AP',0):.4f} "
                f"coco50={coco.get('coco_AP50',0):.4f} "
                f"coco75={coco.get('coco_AP75',0):.4f} "
                f"cocoS={coco.get('coco_AP_small',0):.4f} "
                f"cocoM={coco.get('coco_AP_medium',0):.4f} "
                f"cocoL={coco.get('coco_AP_large',0):.4f} "
                f"ar100={coco.get('coco_AR100',0):.4f} "
                f"(wbf={dt_wbf:.0f}s, metric={dt_metric:.0f}s)")
        print(line)
        results.append(line)

    out_file = args.out or (ROOT / f"runs/wbf_{args.split}_results.txt")
    if not out_file.is_absolute():
        out_file = ROOT / out_file
    with open(out_file, "w") as f:
        f.write("\n".join(results))
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
