from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.metrics import sa_alw_canonical as canonical
from common.model import _hierarchical_assignment
from paper_a.tools.fit_train_scale_schedule import (
    _is_valid_positive,
    detector_resize_shape,
    sha256_file,
)


ANCHOR_SIZES = (
    (4, 8, 16),
    (16, 32, 64),
    (64, 128, 256),
    (128, 256, 512),
    (256, 512, 1024),
)
ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * 5
FPN_STRIDES = (4, 8, 16, 32, 64)
VARIANTS = ("alw", "beta_only", "position_only", "full")


def _columns(boxes: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(boxes[:, index] for index in range(4))


def _geometry_xywh(boxes: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return (
        (boxes[:, 0] + boxes[:, 2]) / 2.0,
        (boxes[:, 1] + boxes[:, 3]) / 2.0,
        (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0),
        (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0),
    )


def deterministic_image_sample(
    image_ids: list[int], *, sample_size: int, seed: int
) -> list[int]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(image_ids):
        raise ValueError("sample_size exceeds eligible image count")
    generator = random.Random(seed)
    return sorted(generator.sample(sorted(image_ids), sample_size))


def _anchor_lattice(height: int, width: int, device: torch.device) -> torch.Tensor:
    grid_sizes = [(height // stride, width // stride) for stride in FPN_STRIDES]
    strides = [
        (
            torch.tensor(height // grid_height, device=device),
            torch.tensor(width // grid_width, device=device),
        )
        for grid_height, grid_width in grid_sizes
    ]
    generator = AnchorGenerator(sizes=ANCHOR_SIZES, aspect_ratios=ASPECT_RATIOS)
    generator.set_cell_anchors(dtype=torch.float32, device=device)
    return torch.cat(generator.grid_anchors(grid_sizes, strides)).to(device)


def _metric_functions(schedule: dict[str, Any]) -> dict[str, Callable[..., torch.Tensor]]:
    def alw(*args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return canonical.compute_alw_similarity(*args, beta=8.0, **kwargs)

    def beta_only(*args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return canonical.compute_sa_alw_beta_only_similarity(
            *args, **schedule, **kwargs
        )

    def position_only(*args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return canonical.compute_sa_alw_pos_only_similarity(
            *args, beta=8.0, **schedule, **kwargs
        )

    def full(*args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return canonical.compute_sa_alw_similarity(*args, **schedule, **kwargs)

    return {
        "alw": alw,
        "beta_only": beta_only,
        "position_only": position_only,
        "full": full,
    }


def _scale_bin(scale: float, *, s_min: float, s_max: float) -> str:
    if scale < s_min:
        return "below_s_min"
    if scale < s_max:
        return "adaptive_interval"
    return "above_s_max"


def _comparison_stats(
    baseline: torch.Tensor, candidate: torch.Tensor
) -> dict[str, int]:
    baseline_positive = baseline >= 0
    candidate_positive = candidate >= 0
    both_positive = baseline_positive & candidate_positive
    return {
        "changed_anchor_count": int((baseline != candidate).sum()),
        "positive_set_change_count": int(
            (baseline_positive != candidate_positive).sum()
        ),
        "added_positive_count": int((~baseline_positive & candidate_positive).sum()),
        "dropped_positive_count": int((baseline_positive & ~candidate_positive).sum()),
        "owner_change_count": int(
            (both_positive & (baseline != candidate)).sum()
        ),
    }


def audit_anchor_assignment(
    annotation_file: Path,
    schedule_file: Path,
    *,
    sample_size: int,
    seed: int,
    device: torch.device,
    target_height: int = 640,
    target_width: int = 640,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schedule_audit = json.loads(schedule_file.read_text(encoding="utf-8"))
    if schedule_audit.get("split") != "train":
        raise ValueError("assignment audit requires a train-only schedule")
    annotation_sha256 = sha256_file(annotation_file)
    if annotation_sha256 != schedule_audit.get("annotation_sha256"):
        raise ValueError("annotation hash does not match the train schedule audit")

    bounds = schedule_audit["schedule_bounds"]
    schedule = {
        "s_min": float(bounds["s_min"]),
        "s_max": float(bounds["s_max"]),
        "beta_min": 8.0,
        "beta_max": 10.0,
        "w_min": 1.0,
        "w_max": 1.5,
        "schedule_form": "linear",
    }
    transform = schedule_audit["transform"]
    min_size = int(transform["min_size"])
    max_size = int(transform["max_size"])

    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in payload["images"]}
    annotations_by_image: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        image = images[image_id]
        if _is_valid_positive(annotation, image):
            annotations_by_image[image_id].append(annotation)

    eligible_ids = []
    for image_id, annotations in annotations_by_image.items():
        image = images[image_id]
        resized_height, resized_width = detector_resize_shape(
            int(image["height"]), int(image["width"]), min_size, max_size
        )
        if (
            annotations
            and resized_height == target_height
            and resized_width == target_width
        ):
            eligible_ids.append(image_id)
    selected_ids = deterministic_image_sample(
        eligible_ids, sample_size=sample_size, seed=seed
    )

    anchors = _anchor_lattice(target_height, target_width, device)
    anchor_geometry = _geometry_xywh(anchors)
    metric_functions = _metric_functions(schedule)
    variant_positive_counts: defaultdict[str, int] = defaultdict(int)
    variant_bin_positive: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    variant_bin_covered_gt: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    gt_bin_counts: defaultdict[str, int] = defaultdict(int)
    comparison_totals: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    per_image: list[dict[str, Any]] = []
    started = time.perf_counter()

    with torch.inference_mode():
        for image_id in selected_ids:
            image = images[image_id]
            scale_x = float(target_width) / float(image["width"])
            scale_y = float(target_height) / float(image["height"])
            target_rows = []
            for annotation in annotations_by_image[image_id]:
                x, y, width, height = [float(value) for value in annotation["bbox"]]
                target_rows.append(
                    [
                        x * scale_x,
                        y * scale_y,
                        (x + width) * scale_x,
                        (y + height) * scale_y,
                    ]
                )
            targets = torch.tensor(target_rows, dtype=torch.float32, device=device)
            target_geometry = _geometry_xywh(targets)
            target_scales = torch.sqrt(target_geometry[2] * target_geometry[3])
            target_bins = [
                _scale_bin(float(value), s_min=schedule["s_min"], s_max=schedule["s_max"])
                for value in target_scales.cpu()
            ]
            for bin_name in target_bins:
                gt_bin_counts[bin_name] += 1

            matches: dict[str, torch.Tensor] = {}
            image_variant_counts: dict[str, int] = {}
            for variant in VARIANTS:
                metric_fn = metric_functions[variant]
                similarity = metric_fn(*anchor_geometry, *target_geometry)
                matched = _hierarchical_assignment(
                    similarity,
                    *anchor_geometry,
                    *target_geometry,
                    metric_fn=metric_fn,
                )
                matches[variant] = matched
                positive_count = int((matched >= 0).sum())
                image_variant_counts[variant] = positive_count
                variant_positive_counts[variant] += positive_count
                if positive_count:
                    per_gt = torch.bincount(
                        matched[matched >= 0], minlength=len(targets)
                    ).cpu()
                    for gt_index, count in enumerate(per_gt.tolist()):
                        variant_bin_positive[variant][target_bins[gt_index]] += int(count)
                        if count > 0:
                            variant_bin_covered_gt[variant][target_bins[gt_index]] += 1

            image_comparisons = {}
            for baseline_name, candidate_name in (
                ("alw", "beta_only"),
                ("alw", "position_only"),
                ("alw", "full"),
                ("position_only", "full"),
            ):
                comparison_name = f"{candidate_name}_vs_{baseline_name}"
                stats = _comparison_stats(
                    matches[baseline_name], matches[candidate_name]
                )
                image_comparisons[comparison_name] = stats
                for key, value in stats.items():
                    comparison_totals[comparison_name][key] += value

            per_image.append(
                {
                    "image_id": image_id,
                    "gt_count": len(targets),
                    "positive_counts": image_variant_counts,
                    "comparisons": image_comparisons,
                }
            )

    csv_rows: list[dict[str, Any]] = []
    bin_order = ("below_s_min", "adaptive_interval", "above_s_max")
    for baseline_name, candidate_name in (
        ("alw", "beta_only"),
        ("alw", "position_only"),
        ("alw", "full"),
        ("position_only", "full"),
    ):
        comparison_name = f"{candidate_name}_vs_{baseline_name}"
        for bin_name in bin_order:
            baseline_count = variant_bin_positive[baseline_name][bin_name]
            candidate_count = variant_bin_positive[candidate_name][bin_name]
            gt_count = gt_bin_counts[bin_name]
            baseline_covered = variant_bin_covered_gt[baseline_name][bin_name]
            candidate_covered = variant_bin_covered_gt[candidate_name][bin_name]
            csv_rows.append(
                {
                    "comparison": comparison_name,
                    "scale_bin": bin_name,
                    "gt_count": gt_count,
                    "baseline_positive_count": baseline_count,
                    "candidate_positive_count": candidate_count,
                    "delta_positive_count": candidate_count - baseline_count,
                    "relative_delta": (
                        (candidate_count / baseline_count - 1.0)
                        if baseline_count
                        else None
                    ),
                    "baseline_mean_positive_per_gt": (
                        baseline_count / gt_count if gt_count else None
                    ),
                    "candidate_mean_positive_per_gt": (
                        candidate_count / gt_count if gt_count else None
                    ),
                    "baseline_covered_gt_count": baseline_covered,
                    "candidate_covered_gt_count": candidate_covered,
                    "delta_covered_gt_count": candidate_covered - baseline_covered,
                }
            )

    result = {
        "status": "TRAIN_ANNOTATION_ANCHOR_PREFLIGHT",
        "evidence_class": "validation_evidence_not_submission_evidence",
        "annotation_file": annotation_file.name,
        "annotation_sha256": annotation_sha256,
        "schedule_file": schedule_file.name,
        "schedule_audit_sha256": schedule_audit["audit_sha256"],
        "schedule": schedule,
        "device": str(device),
        "seed": seed,
        "eligible_image_count": len(eligible_ids),
        "sample_size": sample_size,
        "selected_image_ids": selected_ids,
        "selection_rule": (
            "seeded sample over valid "
            f"{target_width}x{target_height} resized train images; "
            "labels not used for selection"
        ),
        "anchor_contract": {
            "transformed_shape": [target_height, target_width],
            "anchor_count": len(anchors),
            "fpn_strides": list(FPN_STRIDES),
            "sizes": [list(values) for values in ANCHOR_SIZES],
            "aspect_ratios": [list(values) for values in ASPECT_RATIOS],
            "hierarchical_assignment_passes": 2,
        },
        "gt_count": sum(gt_bin_counts.values()),
        "gt_scale_bin_counts": dict(gt_bin_counts),
        "variant_positive_counts": dict(variant_positive_counts),
        "variant_gt_coverage_counts": {
            variant: sum(variant_bin_covered_gt[variant].values())
            for variant in VARIANTS
        },
        "variant_gt_coverage_by_scale": {
            variant: dict(variant_bin_covered_gt[variant]) for variant in VARIANTS
        },
        "comparison_totals": {
            key: dict(value) for key, value in comparison_totals.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
        "per_image": per_image,
        "restrictions": [
            "training annotations only",
            "exact frozen anchor and HLA geometry but no image features or optimizer updates",
            "bounded seeded sample rather than full training split",
            "AI-TOD-v2 mechanism evidence cannot freeze TinyPerson endpoints",
            "no performance or final-test claim",
        ],
    }
    return result, csv_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path("paper_a/schedules/aitodv2_train_p10_p90.json"),
    )
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-height", type=int, default=640)
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_a/diagnostics/aitodv2_anchor_assignment_preflight.json"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("paper_a/diagnostics/aitodv2_anchor_assignment_by_scale.csv"),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    result, rows = audit_anchor_assignment(
        args.annotations.resolve(),
        args.schedule.resolve(),
        sample_size=args.sample_size,
        seed=args.seed,
        device=torch.device(args.device),
        target_height=args.target_height,
        target_width=args.target_width,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: value for key, value in result.items() if key != "per_image"}
    summary["csv_rows"] = len(rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
