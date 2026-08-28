"""Tune boundary-aware rescoring for cached tile prediction fusion."""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate tile-boundary score penalties before WBF")
    p.add_argument("--split", choices=["valid", "test"], default="valid")
    p.add_argument("--pred-files", type=Path, nargs="+", required=True)
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--modes", nargs="*", default=["weighted_avg", "ap75_hybrid"])
    p.add_argument("--iou-thrs", nargs="*", type=float, default=[0.55, 0.60, 0.65])
    p.add_argument("--score-thrs", nargs="*", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument("--edge-pxs", nargs="*", type=float, default=[4.0, 8.0, 16.0])
    p.add_argument("--edge-weights", nargs="*", type=float, default=[0.0, 0.25, 0.50, 0.75, 1.0])
    p.add_argument("--micro-weights", nargs="*", type=float, default=[1.0])
    p.add_argument("--micro-px", type=float, default=6.0)
    p.add_argument("--weights", nargs="*", default=None, help="Model weights like 1,1 or 1,0.9")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--tile-topk", type=int, default=150)
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
        vals = tuple(float(v) for v in spec.split(","))
        if len(vals) != n_models:
            raise SystemExit(f"Weight spec must have {n_models} values: {spec}")
        out.append(vals)
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
    areas = (
        (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        if gt_boxes.numel() > 0
        else torch.zeros(0)
    )
    return {
        "boxes": gt_boxes,
        "labels": gt_labels,
        "area": areas,
        "iscrowd": torch.zeros(len(gt_labels), dtype=torch.int64),
        "image_id": torch.tensor([img_idx], dtype=torch.int64),
    }


def _topk_for_coco(preds: list[dict], k: int) -> list[dict]:
    out = []
    for pred in preds:
        scores = pred["scores"]
        if len(scores) > k:
            keep = torch.topk(scores, k=k).indices
            out.append(
                {
                    "boxes": pred["boxes"][keep],
                    "scores": pred["scores"][keep],
                    "labels": pred["labels"][keep],
                }
            )
        else:
            out.append(pred)
    return out


def _rescore_tile(
    pred: dict,
    tx1: int,
    ty1: int,
    tx2: int,
    ty2: int,
    score_thr: float,
    edge_px: float,
    edge_weight: float,
    micro_px: float,
    micro_weight: float,
    model_weight: float,
    tile_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes = pred["boxes"]
    scores = pred["scores"].clone()
    labels = pred["labels"]
    if len(scores) == 0:
        return boxes, scores, labels

    tile_w = tx2 - tx1
    tile_h = ty2 - ty1
    edge_dist = torch.stack(
        [
            boxes[:, 0],
            boxes[:, 1],
            tile_w - boxes[:, 2],
            tile_h - boxes[:, 3],
        ],
        dim=1,
    ).min(dim=1).values
    pred_size = ((boxes[:, 2] - boxes[:, 0]).clamp(0) * (boxes[:, 3] - boxes[:, 1]).clamp(0)).sqrt()
    scores[edge_dist <= edge_px] *= edge_weight
    scores[pred_size < micro_px] *= micro_weight
    scores *= model_weight

    keep = scores >= score_thr
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if len(scores) > tile_topk:
        keep_top = torch.topk(scores, k=tile_topk).indices
        boxes = boxes[keep_top]
        scores = scores[keep_top]
        labels = labels[keep_top]
    return boxes, scores, labels


def evaluate_config(
    caches: list[dict],
    img_groups: dict[int, list[int]],
    mode: str,
    iou_thr: float,
    score_thr: float,
    edge_px: float,
    edge_weight: float,
    micro_px: float,
    micro_weight: float,
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
                boxes, scores, labels = _rescore_tile(
                    cache["preds"][tile_idx],
                    tx1,
                    ty1,
                    tx2,
                    ty2,
                    score_thr,
                    edge_px,
                    edge_weight,
                    micro_px,
                    micro_weight,
                    weight,
                    tile_topk,
                )
                tile_preds.append((boxes, scores, labels))
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
        sum(
            sap.get(f"AP_{s}", 0.0) * sap.get(f"n_gt_{s}", 0)
            for s in ("micro", "tiny", "small", "large")
        )
        / max(tgt, 1)
    )
    return {
        "fusion_mode": mode,
        "iou_thr": iou_thr,
        "score_thr": score_thr,
        "edge_px": edge_px,
        "edge_weight": edge_weight,
        "micro_px": micro_px,
        "micro_weight": micro_weight,
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
    weight_specs = _parse_weight_specs(args.weights, len(caches))
    img_groups = _build_img_groups(caches[0]["tile_index"])

    print("Caches:")
    for name, cache in zip(names, caches):
        print(f"  {name}: {cache['_path']}")
        if cache.get("meta"):
            print(f"    meta={cache['meta']}")
    print(f"Images: {len(img_groups)}")
    if args.max_images is not None:
        print(f"Debug subset: {args.max_images}")

    configs = list(
        itertools.product(
            args.modes,
            args.iou_thrs,
            args.score_thrs,
            args.edge_pxs,
            args.edge_weights,
            args.micro_weights,
            weight_specs,
        )
    )
    print(f"Configs: {len(configs)}")

    results = []
    t0 = time.time()
    for idx, (mode, iou_thr, score_thr, edge_px, edge_weight, micro_weight, weights) in enumerate(configs, 1):
        row = evaluate_config(
            caches,
            img_groups,
            mode,
            iou_thr,
            score_thr,
            edge_px,
            edge_weight,
            args.micro_px,
            micro_weight,
            weights,
            args.max_images,
            args.topk,
            args.tile_topk,
        )
        results.append(row)
        elapsed = time.time() - t0
        print(
            f"[{idx:>3}/{len(configs)}] {mode:>13} iou={iou_thr:.2f} "
            f"score={score_thr:.2f} edge={edge_px:g} ew={edge_weight:.2f} "
            f"mw={micro_weight:.2f} "
            f"weights={','.join(str(w) for w in weights)} -> "
            f"AP75={row['coco_AP75']:.4f} AP={row['coco_AP']:.4f} "
            f"mAP={row['mAP_scale']:.4f} AR100={row['coco_AR100']:.4f} "
            f"({elapsed:.0f}s)",
            flush=True,
        )

    results.sort(
        key=lambda r: (r["coco_AP75"], r["coco_AP"], r["mAP_scale"], r["coco_AR100"]),
        reverse=True,
    )
    out_path = args.out or (ROOT / f"runs/boundary_rescore_{args.split}.json")
    out_path = out_path if out_path.is_absolute() else ROOT / out_path
    payload = {
        "split": args.split,
        "names": names,
        "pred_files": [c["_path"] for c in caches],
        "max_images": args.max_images,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print("\nTop 10:")
    for row in results[:10]:
        print(row)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
