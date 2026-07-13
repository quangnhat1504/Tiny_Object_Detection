"""AP75 failure analysis for Faster R-CNN checkpoints.

Produces a compact JSON summary plus CSV files for the worst localization,
false-positive, and false-negative cases. The analysis runs on the same tiled
dataset representation used by training/evaluation, so tile-boundary artifacts
are visible in the exported rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import SEED, TILE_SIZE, TINY_THRESHOLD_PX, seed_all
from common.dataset import YOLOTinyDataset, collate_fn
from common.eval_utils import evaluate_coco
from common.metrics import get_metric_fn
from common.model import build_model


SCALE_BINS = {
    "micro": (0.0, 6.0),
    "tiny": (6.0, 16.0),
    "small": (16.0, 64.0),
    "large": (64.0, float("inf")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AP75 failure modes")
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--metric", default=None,
                        help="Metric override if checkpoint config lacks it")
    parser.add_argument("--placement", default=None,
                        help="Placement override if checkpoint config lacks it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--edge-margin", type=float, default=4.0,
                        help="Pixels from tile boundary counted as edge-touching")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def scale_name(box: torch.Tensor) -> str:
    size = float(((box[2] - box[0]).clamp(min=0) *
                  (box[3] - box[1]).clamp(min=0)).sqrt())
    for name, (lo, hi) in SCALE_BINS.items():
        if lo < size <= hi:
            return name
    return "micro"


def touches_edge(box: torch.Tensor, width: int, height: int, margin: float) -> bool:
    return bool(
        box[0] <= margin or box[1] <= margin or
        box[2] >= width - margin or box[3] >= height - margin
    )


def box_list(box: torch.Tensor) -> List[float]:
    return [round(float(v), 2) for v in box.tolist()]

def center_distance_matrix(boxes: torch.Tensor) -> torch.Tensor:
    if len(boxes) == 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    centers = torch.stack(((boxes[:, 0] + boxes[:, 2]) / 2,
                           (boxes[:, 1] + boxes[:, 3]) / 2), dim=1)
    return torch.cdist(centers, centers)

def nearest_center_distance(boxes: torch.Tensor, index: int) -> float:
    if len(boxes) <= 1:
        return float("inf")
    distances = center_distance_matrix(boxes)[index]
    distances[index] = float("inf")
    return float(distances.min())

def crowd_bucket(distance: float) -> str:
    if distance <= 8:
        return "d_le_8"
    if distance <= 16:
        return "d_8_16"
    if distance <= 32:
        return "d_16_32"
    return "isolated"

def class_agnostic(items: List[Dict]) -> List[Dict]:
    out = []
    for item in items:
        copied = dict(item)
        copied["labels"] = torch.ones_like(item["labels"])
        out.append(copied)
    return out

def dataset_diagnosis(dataset: YOLOTinyDataset, edge_margin: float) -> Dict:
    summary = {
        "tiles": len(dataset),
        "images": len(dataset.img_files),
        "gt": 0,
        "edge_gt": 0,
        "tiny_tiles": int(sum(dataset.tile_has_tiny)),
        "by_scale": {name: {"gt": 0, "edge_gt": 0} for name in SCALE_BINS},
        "crowd": {"d_le_8": 0, "d_8_16": 0, "d_16_32": 0, "isolated": 0},
        "class_counts": {},
    }
    for tile_idx in range(len(dataset)):
        _img_idx, tx1, ty1, tx2, ty2 = dataset.tile_index[tile_idx]
        tile_w = tx2 - tx1
        tile_h = ty2 - ty1
        _img, target = dataset[tile_idx]
        boxes = target["boxes"]
        labels = target["labels"]
        summary["gt"] += len(boxes)
        for label in labels.tolist():
            key = str(int(label))
            summary["class_counts"][key] = summary["class_counts"].get(key, 0) + 1
        for gt_idx, box in enumerate(boxes):
            scale = scale_name(box)
            edge = touches_edge(box, tile_w, tile_h, edge_margin)
            nearest = nearest_center_distance(boxes, gt_idx)
            summary["by_scale"][scale]["gt"] += 1
            summary["by_scale"][scale]["edge_gt"] += int(edge)
            summary["edge_gt"] += int(edge)
            summary["crowd"][crowd_bucket(nearest)] += 1
    summary["edge_gt_rate"] = round(summary["edge_gt"] / max(summary["gt"], 1), 6)
    summary["tiny_tile_rate"] = round(summary["tiny_tiles"] / max(summary["tiles"], 1), 6)
    return summary


@torch.no_grad()
def collect_predictions(model, loader, device, score_thr: float, topk: int):
    model.eval()
    prev_score = getattr(model.roi_heads, "score_thresh", None)
    model.roi_heads.score_thresh = min(score_thr, 0.001)

    preds_all, gts_all = [], []
    for imgs, targets in loader:
        preds = model([img.to(device) for img in imgs])
        for pred, target in zip(preds, targets):
            pred = {key: value.detach().cpu() for key, value in pred.items()}
            keep = pred["scores"] >= score_thr
            if int(keep.sum()) > topk:
                kept_scores = pred["scores"][keep]
                top_idx = torch.topk(kept_scores, k=topk).indices
                keep_indices = torch.where(keep)[0][top_idx]
            else:
                keep_indices = torch.where(keep)[0]
            pred = {key: value[keep_indices] for key, value in pred.items()}
            preds_all.append(pred)
            gts_all.append({key: value.cpu() if isinstance(value, torch.Tensor) else value
                            for key, value in target.items()})

    if prev_score is not None:
        model.roi_heads.score_thresh = prev_score
    return preds_all, gts_all


def analyze_matches(preds: List[Dict], gts: List[Dict], dataset: YOLOTinyDataset,
                    edge_margin: float) -> Tuple[Dict, List[Dict], List[Dict], List[Dict]]:
    summary = {
        "n_gt": 0, "n_pred": 0, "tp75": 0, "fn75": 0,
        "fp_duplicate": 0, "fp_class_mismatch": 0, "fp_localization_50_75": 0,
        "fp_localization_25_50": 0, "fp_background": 0,
        "edge_gt": 0, "edge_fn75": 0,
        "class_match_iou50": 0, "class_mismatch_iou50": 0,
        "class_match_iou75": 0, "class_mismatch_iou75": 0,
        "class_confusion": {},
        "fn75_by_crowd": {"d_le_8": 0, "d_8_16": 0, "d_16_32": 0, "isolated": 0},
        "loc_50_75_by_crowd": {"d_le_8": 0, "d_8_16": 0, "d_16_32": 0, "isolated": 0},
        "by_scale": {name: {"gt": 0, "tp75": 0, "fn75": 0,
                             "loc_50_75": 0, "loc_25_50": 0}
                     for name in SCALE_BINS},
    }
    localization_rows, fp_rows, fn_rows = [], [], []

    for tile_idx, (pred, gt) in enumerate(zip(preds, gts)):
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]
        summary["n_gt"] += len(gt_boxes)
        summary["n_pred"] += len(pred_boxes)

        if tile_idx < len(dataset.tile_index):
            orig_img, tx1, ty1, tx2, ty2 = dataset.tile_index[tile_idx]
            tile_w, tile_h = tx2 - tx1, ty2 - ty1
        else:
            orig_img, tx1, ty1, tile_w, tile_h = -1, 0, 0, TILE_SIZE, TILE_SIZE

        for gt_box in gt_boxes:
            scale = scale_name(gt_box)
            summary["by_scale"][scale]["gt"] += 1
            if touches_edge(gt_box, tile_w, tile_h, edge_margin):
                summary["edge_gt"] += 1

        if len(gt_boxes) == 0:
            for pred_idx, pred_box in enumerate(pred_boxes):
                fp_rows.append({
                    "tile_idx": tile_idx, "orig_img": orig_img,
                    "score": round(float(pred_scores[pred_idx]), 6),
                    "best_iou": 0.0, "reason": "background_no_gt",
                    "pred_box": box_list(pred_box), "gt_box": [],
                })
                summary["fp_background"] += 1
            continue

        if len(pred_boxes) == 0:
            for gt_idx, gt_box in enumerate(gt_boxes):
                scale = scale_name(gt_box)
                edge = touches_edge(gt_box, tile_w, tile_h, edge_margin)
                nearest = nearest_center_distance(gt_boxes, gt_idx)
                crowd = crowd_bucket(nearest)
                summary["fn75"] += 1
                summary["by_scale"][scale]["fn75"] += 1
                summary["edge_fn75"] += int(edge)
                summary["fn75_by_crowd"][crowd] += 1
                fn_rows.append({
                    "tile_idx": tile_idx, "orig_img": orig_img,
                    "gt_idx": gt_idx, "scale": scale, "edge_gt": edge,
                    "crowd": crowd,
                    "nearest_gt_center": round(nearest, 4) if nearest != float("inf") else None,
                    "gt_label": int(gt_labels[gt_idx]),
                    "best_iou": 0.0, "gt_box": box_list(gt_box),
                })
            continue

        ious = box_iou(pred_boxes, gt_boxes)
        matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
        order = pred_scores.argsort(descending=True)

        for pred_idx_t in order:
            pred_idx = int(pred_idx_t)
            row = ious[pred_idx]
            best_iou, best_gt_t = row.max(dim=0)
            best_gt = int(best_gt_t)
            best_iou_f = float(best_iou)
            gt_box = gt_boxes[best_gt]
            gt_label = int(gt_labels[best_gt])
            pred_label = int(pred_labels[pred_idx])
            scale = scale_name(gt_box)
            edge = touches_edge(gt_box, tile_w, tile_h, edge_margin)
            nearest = nearest_center_distance(gt_boxes, best_gt)
            crowd = crowd_bucket(nearest)

            if best_iou_f >= 0.50:
                if pred_label == gt_label:
                    summary["class_match_iou50"] += 1
                else:
                    summary["class_mismatch_iou50"] += 1
                    key = f"{gt_label}->{pred_label}"
                    summary["class_confusion"][key] = summary["class_confusion"].get(key, 0) + 1
            if best_iou_f >= 0.75:
                if pred_label == gt_label:
                    summary["class_match_iou75"] += 1
                else:
                    summary["class_mismatch_iou75"] += 1

            if best_iou_f >= 0.75 and pred_label == gt_label and not matched_gt[best_gt]:
                matched_gt[best_gt] = True
                summary["tp75"] += 1
                summary["by_scale"][scale]["tp75"] += 1
                continue

            if best_iou_f >= 0.75 and pred_label != gt_label:
                reason = "class_mismatch"
                summary["fp_class_mismatch"] += 1
            elif best_iou_f >= 0.75 and matched_gt[best_gt]:
                reason = "duplicate"
                summary["fp_duplicate"] += 1
            elif best_iou_f >= 0.50:
                reason = "localization_50_75"
                summary["fp_localization_50_75"] += 1
                summary["by_scale"][scale]["loc_50_75"] += 1
                summary["loc_50_75_by_crowd"][crowd] += 1
            elif best_iou_f >= 0.25:
                reason = "localization_25_50"
                summary["fp_localization_25_50"] += 1
                summary["by_scale"][scale]["loc_25_50"] += 1
            else:
                reason = "background"
                summary["fp_background"] += 1

            row_out = {
                "tile_idx": tile_idx, "orig_img": orig_img,
                "tile_xyxy": [tx1, ty1, tx1 + tile_w, ty1 + tile_h],
                "pred_idx": pred_idx, "gt_idx": best_gt,
                "score": round(float(pred_scores[pred_idx]), 6),
                "best_iou": round(best_iou_f, 6), "reason": reason,
                "scale": scale, "edge_gt": edge, "crowd": crowd,
                "nearest_gt_center": round(nearest, 4) if nearest != float("inf") else None,
                "gt_label": gt_label, "pred_label": pred_label,
                "pred_box": box_list(pred_boxes[pred_idx]),
                "gt_box": box_list(gt_box),
            }
            fp_rows.append(row_out)
            if reason.startswith("localization"):
                localization_rows.append(row_out)

        for gt_idx, gt_box in enumerate(gt_boxes):
            if matched_gt[gt_idx]:
                continue
            scale = scale_name(gt_box)
            edge = touches_edge(gt_box, tile_w, tile_h, edge_margin)
            best_iou = float(ious[:, gt_idx].max()) if len(pred_boxes) else 0.0
            nearest = nearest_center_distance(gt_boxes, gt_idx)
            crowd = crowd_bucket(nearest)
            summary["fn75"] += 1
            summary["by_scale"][scale]["fn75"] += 1
            summary["edge_fn75"] += int(edge)
            summary["fn75_by_crowd"][crowd] += 1
            fn_rows.append({
                "tile_idx": tile_idx, "orig_img": orig_img,
                "tile_xyxy": [tx1, ty1, tx1 + tile_w, ty1 + tile_h],
                "gt_idx": gt_idx, "scale": scale, "edge_gt": edge,
                "crowd": crowd,
                "nearest_gt_center": round(nearest, 4) if nearest != float("inf") else None,
                "gt_label": int(gt_labels[gt_idx]),
                "best_iou": round(best_iou, 6), "gt_box": box_list(gt_box),
            })

    summary["recall75"] = round(summary["tp75"] / max(summary["n_gt"], 1), 6)
    summary["precision75_greedy"] = round(
        summary["tp75"] / max(summary["tp75"] + len(fp_rows), 1), 6)
    summary["edge_fn75_rate"] = round(
        summary["edge_fn75"] / max(summary["fn75"], 1), 6)
    summary["class_mismatch_iou50_rate"] = round(
        summary["class_mismatch_iou50"] /
        max(summary["class_match_iou50"] + summary["class_mismatch_iou50"], 1), 6)
    summary["class_mismatch_iou75_rate"] = round(
        summary["class_mismatch_iou75"] /
        max(summary["class_match_iou75"] + summary["class_mismatch_iou75"], 1), 6)
    return summary, localization_rows, fp_rows, fn_rows


def write_csv(path: Path, rows: List[Dict], limit: int) -> None:
    rows = rows[:limit]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seed_all(SEED)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    metric_name = args.metric or config.get("metric", "iou")
    placement = args.placement or config.get("placement", "everywhere")
    box_loss = config.get("box_loss", "metric")
    reliability_thr = config.get("reliability_thr", 16.0)

    metric_fn = None if metric_name == "iou" else get_metric_fn(metric_name)
    if metric_fn is None:
        placement = "everywhere"

    data_dir = ROOT / "data" / args.split
    dataset = YOLOTinyDataset(data_dir / "images", data_dir / "labels", is_train=False)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=(device.type == "cuda"))

    model = build_model(metric_fn=metric_fn, placement=placement,
                        reliability_thr=reliability_thr,
                        box_loss_type=box_loss).to(device)
    model.load_state_dict(ckpt["model"])

    preds, gts = collect_predictions(model, loader, device, args.score_thr, args.topk)
    coco = evaluate_coco(preds, gts, class_metrics=False)
    coco_class_agnostic = evaluate_coco(
        class_agnostic(preds), class_agnostic(gts), class_metrics=False)
    dataset_summary = dataset_diagnosis(dataset, args.edge_margin)
    summary, loc_rows, fp_rows, fn_rows = analyze_matches(
        preds, gts, dataset, args.edge_margin)
    summary.update({
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "metric": metric_name,
        "placement": placement,
        "box_loss": box_loss,
        "score_thr": args.score_thr,
        "topk": args.topk,
        "coco": coco,
        "coco_class_agnostic": coco_class_agnostic,
        "dataset": dataset_summary,
    })

    out_dir = args.out_dir or ROOT / "runs" / f"ap75_analysis_{ckpt_path.parent.name}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    loc_rows.sort(key=lambda row: (row["best_iou"], -row["score"]))
    fp_rows.sort(key=lambda row: (row["reason"], -row["score"]))
    fn_rows.sort(key=lambda row: (row["best_iou"], row["scale"]))

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(out_dir / "localization_errors.csv", loc_rows, limit=500)
    write_csv(out_dir / "false_positives.csv", fp_rows, limit=500)
    write_csv(out_dir / "false_negatives.csv", fn_rows, limit=500)

    print(json.dumps({
        "out_dir": str(out_dir),
        "coco_AP75": coco.get("coco_AP75", 0),
        "class_agnostic_AP75": coco_class_agnostic.get("coco_AP75", 0),
        "recall75": summary["recall75"],
        "precision75_greedy": summary["precision75_greedy"],
        "fp_localization_50_75": summary["fp_localization_50_75"],
        "fp_duplicate": summary["fp_duplicate"],
        "fp_class_mismatch": summary["fp_class_mismatch"],
        "class_mismatch_iou50_rate": summary["class_mismatch_iou50_rate"],
        "edge_fn75_rate": summary["edge_fn75_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
