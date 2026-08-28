"""Diagnose score-vs-localization quality from cached tile predictions."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import torch
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


SCORE_BINS = [
    (0.001, 0.01),
    (0.01, 0.05),
    (0.05, 0.10),
    (0.10, 0.20),
    (0.20, 0.50),
    (0.50, 1.01),
]

SCALE_BINS = {
    "micro": (0.0, 6.0),
    "tiny": (6.0, 16.0),
    "small": (16.0, 64.0),
    "large": (64.0, 1e9),
    "unmatched": (-1.0, 0.0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze whether detection score tracks IoU quality and tile-boundary failures"
    )
    p.add_argument("--pred-file", type=Path, required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--score-thr", type=float, default=0.001)
    p.add_argument("--edge-px", type=float, default=8.0)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-csv", type=Path, default=None)
    return p.parse_args()


def _load_cache(path: Path) -> dict:
    path = path if path.is_absolute() else ROOT / path
    with path.open("rb") as f:
        data = pickle.load(f)
    data["_path"] = str(path)
    return data


def _gt_from_cache(entry: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    if len(entry) == 2:
        boxes_raw, _size = entry
    else:
        boxes_raw, _w, _h = entry
    rows = [b for b in boxes_raw if b[3] > b[1] and b[4] > b[2]]
    boxes = torch.tensor([[b[1], b[2], b[3], b[4]] for b in rows], dtype=torch.float32)
    labels = torch.tensor([int(b[0]) + 1 for b in rows], dtype=torch.int64)
    return boxes, labels


def _score_bin(score: float) -> str:
    for lo, hi in SCORE_BINS:
        if lo <= score < hi:
            return f"{lo:.3g}-{hi:.3g}"
    return "out"


def _scale_bin(size: float | None) -> str:
    if size is None:
        return "unmatched"
    for name, (lo, hi) in SCALE_BINS.items():
        if lo <= size < hi:
            return name
    return "large"


def _pred_scale_bin(size: float) -> str:
    for name, (lo, hi) in SCALE_BINS.items():
        if name == "unmatched":
            continue
        if lo <= size < hi:
            return name
    return "large"


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    den = math.sqrt(vx * vy)
    return float(num / den) if den > 0 else 0.0


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return _pearson(_rank(xs), _rank(ys))


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "hit_iou25": 0.0,
            "hit_iou50": 0.0,
            "hit_iou75": 0.0,
            "mean_score": 0.0,
        }
    ious = [r["best_iou"] for r in rows]
    scores = [r["score"] for r in rows]
    n = len(rows)
    return {
        "n": n,
        "mean_iou": round(sum(ious) / n, 4),
        "median_iou": round(float(median(ious)), 4),
        "hit_iou25": round(sum(v >= 0.25 for v in ious) / n, 4),
        "hit_iou50": round(sum(v >= 0.50 for v in ious) / n, 4),
        "hit_iou75": round(sum(v >= 0.75 for v in ious) / n, 4),
        "mean_score": round(sum(scores) / n, 4),
    }


def _append_full_box(box: torch.Tensor, tx1: int, ty1: int) -> torch.Tensor:
    out = box.clone()
    out[[0, 2]] += tx1
    out[[1, 3]] += ty1
    return out


def _gt_visibility_in_tile(gt_box: torch.Tensor, tx1: int, ty1: int, tx2: int, ty2: int) -> float:
    gx1, gy1, gx2, gy2 = [float(v) for v in gt_box]
    ix1 = max(gx1, tx1)
    iy1 = max(gy1, ty1)
    ix2 = min(gx2, tx2)
    iy2 = min(gy2, ty2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area = max(gx2 - gx1, 0.0) * max(gy2 - gy1, 0.0)
    return inter / max(area, 1e-9)


def analyze(cache: dict, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    tile_index = cache["tile_index"]
    labels_cache = cache["labels_cache"]
    preds = cache["preds"]

    keep_images = None
    if args.max_images is not None:
        keep_images = set(sorted({t[0] for t in tile_index})[: args.max_images])

    gt_by_img = {img_idx: _gt_from_cache(entry) for img_idx, entry in labels_cache.items()}
    rows: list[dict] = []
    gt_hits = {
        img_idx: {
            "best_any": torch.zeros(len(boxes), dtype=torch.float32),
            "best_edge": torch.zeros(len(boxes), dtype=torch.float32),
            "best_far": torch.zeros(len(boxes), dtype=torch.float32),
        }
        for img_idx, (boxes, _labels) in gt_by_img.items()
    }

    for tile_idx, (img_idx, tx1, ty1, tx2, ty2) in enumerate(tile_index):
        if keep_images is not None and img_idx not in keep_images:
            continue
        pred = preds[tile_idx]
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]
        if len(scores) == 0:
            continue
        keep = scores >= args.score_thr
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if len(scores) == 0:
            continue

        gt_boxes, gt_labels = gt_by_img[img_idx]
        tile_w = tx2 - tx1
        tile_h = ty2 - ty1
        for j in range(len(scores)):
            local_box = boxes[j].float()
            full_box = _append_full_box(local_box, tx1, ty1)
            label = labels[j]
            score = float(scores[j])
            edge_dist = min(
                float(local_box[0]),
                float(local_box[1]),
                float(tile_w - local_box[2]),
                float(tile_h - local_box[3]),
            )
            pred_size = float(
                ((local_box[2] - local_box[0]).clamp(0) * (local_box[3] - local_box[1]).clamp(0)).sqrt()
            )
            near_edge = edge_dist <= args.edge_px
            same = torch.where(gt_labels == label)[0]
            best_iou = 0.0
            best_gt = -1
            gt_size = None
            gt_visibility = 0.0
            if len(same) > 0:
                ious = box_iou(full_box.view(1, 4), gt_boxes[same]).view(-1)
                val, local_idx = ious.max(dim=0)
                best_iou = float(val)
                best_gt = int(same[int(local_idx)])
                gt_box = gt_boxes[best_gt]
                gt_size = float(((gt_box[2] - gt_box[0]).clamp(0) * (gt_box[3] - gt_box[1]).clamp(0)).sqrt())
                gt_visibility = _gt_visibility_in_tile(gt_box, tx1, ty1, tx2, ty2)
                gt_hits[img_idx]["best_any"][best_gt] = max(gt_hits[img_idx]["best_any"][best_gt], best_iou)
                key = "best_edge" if near_edge else "best_far"
                gt_hits[img_idx][key][best_gt] = max(gt_hits[img_idx][key][best_gt], best_iou)

            rows.append(
                {
                    "image_id": img_idx,
                    "tile_id": tile_idx,
                    "label": int(label),
                    "score": score,
                    "score_bin": _score_bin(score),
                    "pred_scale_bin": _pred_scale_bin(pred_size),
                    "pred_size": round(pred_size, 4),
                    "best_iou": best_iou,
                    "best_gt": best_gt,
                    "scale_bin": _scale_bin(gt_size),
                    "gt_size": round(gt_size, 4) if gt_size is not None else "",
                    "near_tile_edge": near_edge,
                    "edge_dist": round(edge_dist, 4),
                    "gt_visibility_in_tile": round(gt_visibility, 4),
                    "partial_gt_in_tile": gt_visibility < 0.95 if best_gt >= 0 else False,
                }
            )

    scores = [r["score"] for r in rows]
    ious = [r["best_iou"] for r in rows]
    by_score = defaultdict(list)
    by_scale = defaultdict(list)
    by_pred_scale = defaultdict(list)
    by_edge = defaultdict(list)
    by_partial = defaultdict(list)
    for r in rows:
        by_score[r["score_bin"]].append(r)
        by_scale[r["scale_bin"]].append(r)
        by_pred_scale[r["pred_scale_bin"]].append(r)
        by_edge["near_edge" if r["near_tile_edge"] else "far_from_edge"].append(r)
        by_partial["partial_gt_tile" if r["partial_gt_in_tile"] else "full_or_unmatched_tile"].append(r)

    gt_total = 0
    gt_any75 = gt_edge75 = gt_far75 = 0
    gt_any50 = gt_edge50 = gt_far50 = 0
    for hit in gt_hits.values():
        n = len(hit["best_any"])
        gt_total += n
        gt_any75 += int((hit["best_any"] >= 0.75).sum())
        gt_edge75 += int((hit["best_edge"] >= 0.75).sum())
        gt_far75 += int((hit["best_far"] >= 0.75).sum())
        gt_any50 += int((hit["best_any"] >= 0.50).sum())
        gt_edge50 += int((hit["best_edge"] >= 0.50).sum())
        gt_far50 += int((hit["best_far"] >= 0.50).sum())

    summary = {
        "name": args.name or Path(cache["_path"]).stem,
        "pred_file": cache["_path"],
        "cache_meta": cache.get("meta", {}),
        "score_thr": args.score_thr,
        "edge_px": args.edge_px,
        "max_images": args.max_images,
        "n_predictions": len(rows),
        "n_gt": gt_total,
        "score_iou_pearson": round(_pearson(scores, ious), 4),
        "score_iou_spearman": round(_spearman(scores, ious), 4),
        "overall": _summarize(rows),
        "by_score_bin": {k: _summarize(v) for k, v in sorted(by_score.items())},
        "by_scale_bin": {k: _summarize(v) for k, v in sorted(by_scale.items())},
        "by_pred_scale_bin": {k: _summarize(v) for k, v in sorted(by_pred_scale.items())},
        "by_tile_edge": {k: _summarize(v) for k, v in sorted(by_edge.items())},
        "by_gt_visibility": {k: _summarize(v) for k, v in sorted(by_partial.items())},
        "high_score_low_iou": {
            "score_ge_0_20_iou_lt_0_50": sum(r["score"] >= 0.20 and r["best_iou"] < 0.50 for r in rows),
            "score_ge_0_20_iou_lt_0_75": sum(r["score"] >= 0.20 and r["best_iou"] < 0.75 for r in rows),
            "score_ge_0_50_iou_lt_0_50": sum(r["score"] >= 0.50 and r["best_iou"] < 0.50 for r in rows),
            "score_ge_0_50_iou_lt_0_75": sum(r["score"] >= 0.50 and r["best_iou"] < 0.75 for r in rows),
        },
        "gt_coverage_from_raw_tiles": {
            "any_pred_iou50": round(gt_any50 / max(gt_total, 1), 4),
            "any_pred_iou75": round(gt_any75 / max(gt_total, 1), 4),
            "edge_pred_iou50": round(gt_edge50 / max(gt_total, 1), 4),
            "edge_pred_iou75": round(gt_edge75 / max(gt_total, 1), 4),
            "far_pred_iou50": round(gt_far50 / max(gt_total, 1), 4),
            "far_pred_iou75": round(gt_far75 / max(gt_total, 1), 4),
        },
    }
    return summary, rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_id",
        "tile_id",
        "label",
        "score",
        "score_bin",
        "pred_scale_bin",
        "pred_size",
        "best_iou",
        "best_gt",
        "scale_bin",
        "gt_size",
        "near_tile_edge",
        "edge_dist",
        "gt_visibility_in_tile",
        "partial_gt_in_tile",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cache = _load_cache(args.pred_file)
    summary, rows = analyze(cache, args)

    base = args.name or Path(cache["_path"]).stem
    out_json = args.out_json or (ROOT / "runs" / f"quality_diagnosis_{base}.json")
    out_csv = args.out_csv or (ROOT / "runs" / f"quality_diagnosis_{base}.csv")
    out_json = out_json if out_json.is_absolute() else ROOT / out_json
    out_csv = out_csv if out_csv.is_absolute() else ROOT / out_csv

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    _write_csv(out_csv, rows)

    print(f"Predictions: {summary['n_predictions']}  GT: {summary['n_gt']}")
    print(
        "Score/IoU correlation: "
        f"pearson={summary['score_iou_pearson']:.4f} "
        f"spearman={summary['score_iou_spearman']:.4f}"
    )
    print(f"Overall: {summary['overall']}")
    print(f"Tile edge: {summary['by_tile_edge']}")
    print(f"GT coverage: {summary['gt_coverage_from_raw_tiles']}")
    print(f"Saved JSON: {out_json}")
    print(f"Saved CSV : {out_csv}")


if __name__ == "__main__":
    main()
