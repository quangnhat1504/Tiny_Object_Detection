"""Evaluate prediction-level ensembles from cached tile predictions."""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.eval_utils import compute_scale_ap, evaluate_coco
from common.wbf import wbf_fusion_smart


def _parse_float_list(values: list[str] | None, default: list[float]) -> list[float]:
    return [float(v) for v in values] if values else default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate cached multi-checkpoint ensembles")
    p.add_argument("--split", choices=["valid", "test"], default="valid")
    p.add_argument("--pred-files", type=Path, nargs="+", required=True)
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--score-thrs", nargs="*", default=None)
    p.add_argument("--iou-thrs", nargs="*", default=None)
    p.add_argument("--modes", nargs="*", default=["weighted_avg", "ap75_hybrid"])
    p.add_argument("--weights", nargs="*", default=None,
                   help="Weight specs like 1,1 or 1,0.9. Defaults to all 1.")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--tile-topk", type=int, default=150,
                   help="Keep at most K scored predictions per tile/model before WBF")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _load_cache(path: Path) -> dict:
    path = path if path.is_absolute() else ROOT / path
    with path.open("rb") as f:
        data = pickle.load(f)
    data["_path"] = str(path)
    return data


def _validate_caches(caches: list[dict]) -> None:
    first_index = caches[0]["tile_index"]
    first_labels = caches[0]["labels_cache"]
    for cache in caches[1:]:
        if cache["tile_index"] != first_index:
            raise SystemExit(f"tile_index mismatch: {cache['_path']}")
        if len(cache["labels_cache"]) != len(first_labels):
            raise SystemExit(f"labels_cache length mismatch: {cache['_path']}")


def _parse_weight_specs(specs: list[str] | None, n_models: int) -> list[tuple[float, ...]]:
    if not specs:
        return [tuple([1.0] * n_models)]
    out = []
    for spec in specs:
        weights = tuple(float(v) for v in spec.split(","))
        if len(weights) != n_models:
            raise SystemExit(f"Weight spec must have {n_models} values: {spec}")
        out.append(weights)
    return out


def _topk_for_coco(preds: list[dict], k: int) -> list[dict]:
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


def _build_img_groups(tile_index: list[tuple[int, int, int, int, int]]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for idx, tile in enumerate(tile_index):
        groups.setdefault(tile[0], []).append(idx)
    return groups


def _gt_from_cache_entry(cache_entry, img_idx: int) -> dict:
    if len(cache_entry) == 2:
        boxes_raw, _size = cache_entry
    else:
        boxes_raw, _w, _h = cache_entry
    gt_boxes = torch.tensor(
        [[b[1], b[2], b[3], b[4]] for b in boxes_raw if b[3] > b[1] and b[4] > b[2]],
        dtype=torch.float32,
    )
    gt_labels = torch.tensor(
        [b[0] + 1 for b in boxes_raw if b[3] > b[1] and b[4] > b[2]],
        dtype=torch.int64,
    )
    areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])) \
        if gt_boxes.numel() > 0 else torch.zeros(0)
    return {
        "boxes": gt_boxes,
        "labels": gt_labels,
        "area": areas,
        "iscrowd": torch.zeros(len(gt_labels), dtype=torch.int64),
        "image_id": torch.tensor([img_idx], dtype=torch.int64),
    }


def evaluate_config(
    caches: list[dict],
    img_groups: dict[int, list[int]],
    mode: str,
    iou_thr: float,
    score_thr: float,
    weights: tuple[float, ...],
    max_images: int | None,
    topk: int,
    tile_topk: int,
) -> dict:
    tile_index = caches[0]["tile_index"]
    labels_cache = caches[0]["labels_cache"]
    keep_img_ids = sorted(img_groups)
    if max_images is not None:
        keep_img_ids = keep_img_ids[:max_images]

    all_preds, all_gts = [], []
    for img_idx in keep_img_ids:
        tile_preds, tile_coords = [], []
        for tile_idx in img_groups[img_idx]:
            _i, tx1, ty1, tx2, ty2 = tile_index[tile_idx]
            for cache, weight in zip(caches, weights):
                pred = cache["preds"][tile_idx]
                keep = pred["scores"] >= score_thr
                boxes = pred["boxes"][keep]
                scores = pred["scores"][keep]
                labels = pred["labels"][keep]
                if len(scores) > tile_topk:
                    top_idx = torch.topk(scores, k=tile_topk).indices
                    boxes = boxes[top_idx]
                    scores = scores[top_idx]
                    labels = labels[top_idx]
                tile_preds.append((
                    boxes,
                    scores * weight,
                    labels,
                ))
                tile_coords.append((tx1, ty1, tx2 - tx1, ty2 - ty1))

        cache_entry = labels_cache[img_idx]
        if len(cache_entry) == 2:
            _boxes_raw, (W, H) = cache_entry
        else:
            _boxes_raw, W, H = cache_entry
        fused = wbf_fusion_smart(
            tile_preds,
            tile_coords,
            (W, H),
            iou_thr=iou_thr,
            fusion_mode=mode,
            adaptive_thr=False,
        )
        all_preds.append(fused)
        all_gts.append(_gt_from_cache_entry(cache_entry, img_idx))

    sap = compute_scale_ap(all_preds, all_gts)
    coco = evaluate_coco(_topk_for_coco(all_preds, topk), all_gts, class_metrics=False)
    tgt = sum(sap.get(f"n_gt_{s}", 0) for s in ("micro", "tiny", "small", "large"))
    primary = (
        sum(sap.get(f"AP_{s}", 0.0) * sap.get(f"n_gt_{s}", 0)
            for s in ("micro", "tiny", "small", "large")) / max(tgt, 1)
    )
    return {
        "fusion_mode": mode,
        "iou_thr": iou_thr,
        "score_thr": score_thr,
        "weights": list(weights),
        "mAP_scale": round(primary, 4),
        "AP_micro": round(sap.get("AP_micro", 0), 4),
        "AP_tiny": round(sap.get("AP_tiny", 0), 4),
        "AP_small": round(sap.get("AP_small", 0), 4),
        "AP_large": round(sap.get("AP_large", 0), 4),
        "coco_AP": round(coco.get("coco_AP", 0), 4),
        "coco_AP50": round(coco.get("coco_AP50", 0), 4),
        "coco_AP75": round(coco.get("coco_AP75", 0), 4),
        "coco_AR100": round(coco.get("coco_AR100", 0), 4),
    }


def main() -> None:
    args = parse_args()
    caches = [_load_cache(p) for p in args.pred_files]
    _validate_caches(caches)

    names = args.names or [Path(c["_path"]).stem for c in caches]
    if len(names) != len(caches):
        raise SystemExit("--names length must match --pred-files")

    print("Caches:")
    for name, cache in zip(names, caches):
        print(f"  {name}: {cache['_path']}")
        if cache.get("meta"):
            print(f"    meta={cache['meta']}")

    score_thrs = _parse_float_list(args.score_thrs, [0.05, 0.10, 0.20])
    iou_thrs = _parse_float_list(args.iou_thrs, [0.55, 0.60, 0.65])
    weight_specs = _parse_weight_specs(args.weights, len(caches))
    img_groups = _build_img_groups(caches[0]["tile_index"])

    configs = list(itertools.product(args.modes, iou_thrs, score_thrs, weight_specs))
    print(f"Images: {len(img_groups)}")
    if args.max_images is not None:
        print(f"Debug subset: {args.max_images}")
    print(f"Configs: {len(configs)}")

    results = []
    t0 = time.time()
    for idx, (mode, iou_thr, score_thr, weights) in enumerate(configs, 1):
        metrics = evaluate_config(
            caches, img_groups, mode, iou_thr, score_thr, weights,
            args.max_images, args.topk, args.tile_topk,
        )
        results.append(metrics)
        elapsed = time.time() - t0
        print(
            f"[{idx:>2}/{len(configs)}] {mode:>13} iou={iou_thr:.2f} "
            f"score={score_thr:.2f} weights={','.join(str(w) for w in weights)} "
            f"-> AP75={metrics['coco_AP75']:.4f} AP={metrics['coco_AP']:.4f} "
            f"mAP={metrics['mAP_scale']:.4f} AR100={metrics['coco_AR100']:.4f} "
            f"({elapsed:.0f}s)",
            flush=True,
        )

    results.sort(
        key=lambda r: (r["coco_AP75"], r["coco_AP"], r["mAP_scale"], r["coco_AR100"]),
        reverse=True,
    )

    out_path = args.out or (ROOT / f"runs/cache_ensemble_{args.split}.json")
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    payload = {
        "split": args.split,
        "names": names,
        "pred_files": [c["_path"] for c in caches],
        "max_images": args.max_images,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print("\nTop 5:")
    for row in results[:5]:
        print(row)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
