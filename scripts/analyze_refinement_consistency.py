"""Measure whether CBL box-refinement consistency predicts localization quality."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import common.model as model_module
from common.config import SEED, seed_all
from common.dataset import YOLOTinyDataset, collate_fn
from common.metrics import get_metric_fn
from common.model import build_model


SELF_IOU_BINS = (
    (0.00, 0.50),
    (0.50, 0.75),
    (0.75, 0.90),
    (0.90, 0.97),
    (0.97, 1.01),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pair detections before and after iterative CBL refinement and "
            "measure whether box stability predicts true IoU."
        )
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    x_var = sum((x - mx) ** 2 for x in xs)
    y_var = sum((y - my) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    return float(numerator / denominator) if denominator > 0 else 0.0


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while (
            end + 1 < len(order)
            and values[order[end + 1]] == values[order[start]]
        ):
            end += 1
        rank = (start + end + 2) / 2.0
        for position in range(start, end + 1):
            ranks[order[position]] = rank
        start = end + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return _pearson(_ranks(xs), _ranks(ys))


def _binary_auc(values: list[float], positives: list[bool]) -> float:
    n_pos = sum(positives)
    n_neg = len(positives) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    ranks = _ranks(values)
    positive_rank_sum = sum(
        rank for rank, positive in zip(ranks, positives) if positive
    )
    return float(
        (positive_rank_sum - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


def _same_class_best_iou(
    box: torch.Tensor,
    label: int,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
) -> tuple[float, int]:
    candidates = torch.where(gt_labels == label)[0]
    if len(candidates) == 0:
        return 0.0, -1
    values = box_iou(box.view(1, 4), gt_boxes[candidates]).view(-1)
    best_iou, best_index = values.max(dim=0)
    return float(best_iou), int(candidates[int(best_index)])


def _rescale_boxes(
    boxes: torch.Tensor,
    transformed_shape: tuple[int, int],
    original_shape: tuple[int, int],
) -> torch.Tensor:
    transformed_h, transformed_w = transformed_shape
    original_h, original_w = original_shape
    scale = boxes.new_tensor(
        [
            original_w / transformed_w,
            original_h / transformed_h,
            original_w / transformed_w,
            original_h / transformed_h,
        ]
    )
    return boxes * scale


def _pair_by_label_and_score(
    before_boxes: torch.Tensor,
    before_scores: torch.Tensor,
    before_labels: torch.Tensor,
    after_boxes: torch.Tensor,
    after_scores: torch.Tensor,
    after_labels: torch.Tensor,
) -> tuple[list[tuple[int, int]], int]:
    """Pair surviving boxes; refinement preserves label and exact score."""
    available: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(before_labels.tolist()):
        available[int(label)].append(index)

    pairs: list[tuple[int, int]] = []
    unmatched = 0
    for after_index, label in enumerate(after_labels.tolist()):
        candidates = available.get(int(label), [])
        if not candidates:
            unmatched += 1
            continue
        differences = torch.tensor(
            [
                abs(float(before_scores[index]) - float(after_scores[after_index]))
                for index in candidates
            ]
        )
        exact = torch.where(differences <= 1e-7)[0]
        if len(exact) == 0:
            unmatched += 1
            continue
        exact_candidates = [candidates[int(index)] for index in exact]
        if len(exact_candidates) == 1:
            before_index = exact_candidates[0]
        else:
            overlaps = box_iou(
                after_boxes[after_index].view(1, 4),
                before_boxes[exact_candidates],
            ).view(-1)
            before_index = exact_candidates[int(overlaps.argmax())]
        pairs.append((before_index, after_index))
        candidates.remove(before_index)
    return pairs, unmatched


def _build_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device,
):
    config = checkpoint.get("config", {})
    metric = config.get("metric", "sa_alw_full")
    placement = config.get("placement", "la_loss")
    metric_fn = None if metric == "iou" else get_metric_fn(metric)
    model = build_model(
        metric_fn=metric_fn,
        placement="everywhere" if metric == "iou" else placement,
        reliability_thr=float(config.get("reliability_thr", 16.0)),
        box_loss_type=config.get("box_loss", "metric"),
        box_loss_warmup_epochs=int(
            config.get("box_loss_warmup_epochs", 3)
        ),
        use_quality_score=bool(config.get("quality_score", False)),
        quality_loss_weight=float(
            config.get("quality_loss_weight", 0.0) or 0.0
        ),
        use_quality_focal=bool(config.get("quality_focal", False)),
        quality_focal_beta=float(config.get("quality_focal_beta", 2.0)),
        use_rank_sort=bool(config.get("rank_sort", False)),
        rank_sort_delta=float(config.get("rank_sort_delta", 0.5)),
        use_double_head=bool(config.get("double_head", False)),
        double_head_reg_roi_scale=float(
            config.get("double_head_reg_roi_scale", 1.3)
        ),
        double_head_num_convs=int(config.get("double_head_num_convs", 4)),
        cbl_refine_steps=int(config.get("cbl_refine_steps", 0)),
        cbl_refine_blend=float(config.get("cbl_refine_blend", 1.0)),
        cbl_refine_last_step_blend=config.get(
            "cbl_refine_last_step_blend"
        ),
        cbl_refine_score_threshold=float(
            config.get("cbl_refine_score_threshold", 0.0)
        ),
        cbl_refine_extra_min_size_ratio=float(
            config.get("cbl_refine_extra_min_size_ratio", 0.0)
        ),
        cbl_refine_train_weight=float(
            config.get("cbl_refine_train_weight", 0.0)
        ),
        cbl_refine_train_steps=int(
            config.get("cbl_refine_train_steps", 1)
        ),
        cbl_alpha=float(config.get("cbl_alpha", 5.0)),
        cbl_num_bins=int(config.get("cbl_num_bins", 6)),
        cbl_grid_beta=float(config.get("cbl_grid_beta", 1.0)),
        cbl_um_weight=float(config.get("cbl_um_weight", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    keys = (
        "pre_iou",
        "post_iou",
        "iou_gain",
        "self_iou",
        "center_shift",
        "size_change",
        "score",
    )
    result = {"n": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows]
        result[f"mean_{key}"] = round(sum(values) / len(values), 6)
    result.update(
        {
            "pre_iou50_rate": round(
                sum(row["pre_iou"] >= 0.50 for row in rows) / len(rows), 6
            ),
            "post_iou50_rate": round(
                sum(row["post_iou"] >= 0.50 for row in rows) / len(rows), 6
            ),
            "pre_iou75_rate": round(
                sum(row["pre_iou"] >= 0.75 for row in rows) / len(rows), 6
            ),
            "post_iou75_rate": round(
                sum(row["post_iou"] >= 0.75 for row in rows) / len(rows), 6
            ),
            "cross_up_iou75": sum(
                row["pre_iou"] < 0.75 <= row["post_iou"] for row in rows
            ),
            "cross_down_iou75": sum(
                row["post_iou"] < 0.75 <= row["pre_iou"] for row in rows
            ),
            "improved": sum(row["iou_gain"] > 1e-6 for row in rows),
            "regressed": sum(row["iou_gain"] < -1e-6 for row in rows),
        }
    )
    return result


def _bin_label(value: float) -> str:
    for lower, upper in SELF_IOU_BINS:
        if lower <= value < upper:
            return f"{lower:.2f}-{upper:.2f}"
    return "out"


def _metric_diagnostics(rows: list[dict]) -> dict:
    post_ious = [float(row["post_iou"]) for row in rows]
    gains = [float(row["iou_gain"]) for row in rows]
    candidates = {
        "class_score": [float(row["score"]) for row in rows],
        "self_iou": [float(row["self_iou"]) for row in rows],
        "score_x_self_iou": [
            float(row["score"] * row["self_iou"]) for row in rows
        ],
        "score_x_sqrt_self_iou": [
            float(row["score"] * math.sqrt(max(row["self_iou"], 0.0)))
            for row in rows
        ],
        "score_x_half_stability": [
            float(row["score"] * (0.5 + 0.5 * row["self_iou"]))
            for row in rows
        ],
    }
    result = {}
    for name, values in candidates.items():
        result[name] = {
            "post_iou_pearson": round(_pearson(values, post_ious), 6),
            "post_iou_spearman": round(_spearman(values, post_ious), 6),
            "iou50_auc": round(
                _binary_auc(values, [iou >= 0.50 for iou in post_ious]), 6
            ),
            "iou75_auc": round(
                _binary_auc(values, [iou >= 0.75 for iou in post_ious]), 6
            ),
            "gain_pearson": round(_pearson(values, gains), 6),
            "gain_spearman": round(_spearman(values, gains), 6),
        }
    return result


def main() -> None:
    args = parse_args()
    seed_all(SEED)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    checkpoint_path = (
        args.ckpt if args.ckpt.is_absolute() else ROOT / args.ckpt
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model, config = _build_model_from_checkpoint(checkpoint, device)
    if int(config.get("cbl_refine_steps", 0)) != 1:
        raise ValueError("This paired diagnostic requires exactly one refine step")

    data_dir = ROOT / "data" / args.split
    dataset = YOLOTinyDataset(
        img_dir=data_dir / "images",
        lbl_dir=data_dir / "labels",
        is_train=False,
    )
    if args.max_tiles is not None:
        dataset = torch.utils.data.Subset(
            dataset, range(min(args.max_tiles, len(dataset)))
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    captures: list[dict] = []
    original_refine = model_module._iteratively_refine_cbl_detections

    def observed_refine(
        roi_heads,
        features,
        boxes,
        scores,
        labels,
        image_shapes,
        steps,
        blend,
        last_step_blend,
        score_threshold,
        extra_min_size_ratio,
    ):
        result = original_refine(
            roi_heads,
            features,
            boxes,
            scores,
            labels,
            image_shapes,
            steps,
            blend,
            last_step_blend,
            score_threshold,
            extra_min_size_ratio,
        )
        captures.append(
            {
                "before_boxes": [value.detach().cpu() for value in boxes],
                "before_scores": [value.detach().cpu() for value in scores],
                "before_labels": [value.detach().cpu() for value in labels],
                "after_boxes": [
                    value.detach().cpu() for value in result[0]
                ],
                "after_scores": [
                    value.detach().cpu() for value in result[1]
                ],
                "after_labels": [
                    value.detach().cpu() for value in result[2]
                ],
                "image_shapes": [tuple(shape) for shape in image_shapes],
                "score_threshold": float(score_threshold),
            }
        )
        return result

    model_module._iteratively_refine_cbl_detections = observed_refine
    rows: list[dict] = []
    unmatched = 0
    total_after = 0
    tile_index = 0
    try:
        for images, targets in tqdm(loader, desc="Paired CBL refinement"):
            capture_start = len(captures)
            original_shapes = [
                (int(image.shape[-2]), int(image.shape[-1]))
                for image in images
            ]
            with torch.no_grad():
                model([image.to(device) for image in images])
            if len(captures) != capture_start + 1:
                raise RuntimeError("Expected exactly one refinement capture per batch")
            capture = captures.pop()

            for batch_index, target in enumerate(targets):
                transformed_shape = capture["image_shapes"][batch_index]
                original_shape = original_shapes[batch_index]
                before_boxes = _rescale_boxes(
                    capture["before_boxes"][batch_index],
                    transformed_shape,
                    original_shape,
                )
                after_boxes = _rescale_boxes(
                    capture["after_boxes"][batch_index],
                    transformed_shape,
                    original_shape,
                )
                before_scores = capture["before_scores"][batch_index]
                after_scores = capture["after_scores"][batch_index]
                before_labels = capture["before_labels"][batch_index]
                after_labels = capture["after_labels"][batch_index]
                pairs, tile_unmatched = _pair_by_label_and_score(
                    before_boxes,
                    before_scores,
                    before_labels,
                    after_boxes,
                    after_scores,
                    after_labels,
                )
                unmatched += tile_unmatched
                total_after += len(after_boxes)
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()

                for before_index, after_index in pairs:
                    score = float(after_scores[after_index])
                    if score < capture["score_threshold"]:
                        continue
                    before_box = before_boxes[before_index]
                    after_box = after_boxes[after_index]
                    label = int(after_labels[after_index])
                    pre_iou, pre_gt = _same_class_best_iou(
                        before_box, label, gt_boxes, gt_labels
                    )
                    post_iou, post_gt = _same_class_best_iou(
                        after_box, label, gt_boxes, gt_labels
                    )
                    self_iou = float(
                        box_iou(
                            before_box.view(1, 4),
                            after_box.view(1, 4),
                        )[0, 0]
                    )
                    pre_width = max(
                        float(before_box[2] - before_box[0]), 1e-6
                    )
                    pre_height = max(
                        float(before_box[3] - before_box[1]), 1e-6
                    )
                    pre_center = torch.stack(
                        (
                            (before_box[0] + before_box[2]) / 2,
                            (before_box[1] + before_box[3]) / 2,
                        )
                    )
                    post_center = torch.stack(
                        (
                            (after_box[0] + after_box[2]) / 2,
                            (after_box[1] + after_box[3]) / 2,
                        )
                    )
                    center_shift = float(
                        torch.linalg.vector_norm(post_center - pre_center)
                        / math.sqrt(pre_width * pre_height)
                    )
                    post_width = max(
                        float(after_box[2] - after_box[0]), 1e-6
                    )
                    post_height = max(
                        float(after_box[3] - after_box[1]), 1e-6
                    )
                    size_change = (
                        abs(math.log(post_width / pre_width))
                        + abs(math.log(post_height / pre_height))
                    ) / 2.0
                    gt_index = post_gt if post_iou >= pre_iou else pre_gt
                    gt_size = 0.0
                    if gt_index >= 0:
                        gt_box = gt_boxes[gt_index]
                        gt_size = math.sqrt(
                            max(
                                float(
                                    (gt_box[2] - gt_box[0])
                                    * (gt_box[3] - gt_box[1])
                                ),
                                0.0,
                            )
                        )
                    rows.append(
                        {
                            "tile_index": tile_index,
                            "label": label,
                            "score": score,
                            "pre_iou": pre_iou,
                            "post_iou": post_iou,
                            "iou_gain": post_iou - pre_iou,
                            "self_iou": self_iou,
                            "self_iou_bin": _bin_label(self_iou),
                            "center_shift": center_shift,
                            "size_change": size_change,
                            "pre_gt": pre_gt,
                            "post_gt": post_gt,
                            "same_best_gt": pre_gt == post_gt and pre_gt >= 0,
                            "gt_size": gt_size,
                        }
                    )
                tile_index += 1
    finally:
        model_module._iteratively_refine_cbl_detections = original_refine

    by_self_iou: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_self_iou[row["self_iou_bin"]].append(row)
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_model_source": checkpoint.get(
            "model_source", "legacy_unspecified"
        ),
        "split": args.split,
        "tiles": tile_index,
        "refine_score_threshold": float(
            config.get("cbl_refine_score_threshold", 0.0)
        ),
        "paired_refined_detections": len(rows),
        "all_post_detections": total_after,
        "unmatched_post_detections": unmatched,
        "unmatched_rate": round(unmatched / max(total_after, 1), 8),
        "overall": _summarize(rows),
        "by_self_iou": {
            key: _summarize(value)
            for key, value in sorted(by_self_iou.items())
        },
        "ranking_diagnostics": _metric_diagnostics(rows),
    }

    out_json = (
        args.out_json
        if args.out_json.is_absolute()
        else ROOT / args.out_json
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    if args.out_csv is not None:
        out_csv = (
            args.out_csv
            if args.out_csv.is_absolute()
            else ROOT / args.out_csv
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

    print(json.dumps(summary, indent=2))
    print(f"Saved JSON: {out_json}")
    if args.out_csv is not None:
        print(f"Saved CSV : {out_csv}")


if __name__ == "__main__":
    main()
