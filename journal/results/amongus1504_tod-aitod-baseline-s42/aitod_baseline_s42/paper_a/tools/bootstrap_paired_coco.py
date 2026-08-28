"""Paired original-image bootstrap for Paper A COCO AP metrics."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
from typing import Sequence

import numpy as np


METRICS = ("AP", "AP50", "AP75")


def evaluate_detection_files(annotation_path: Path, detection_path: Path) -> dict[str, object]:
    """Evaluate one COCO detection file and retain all-area per-image matches."""

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotation_path))
        detections = ground_truth.loadRes(str(detection_path))
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    if len(evaluator._paramsEval.catIds) != 1:
        raise ValueError("paired Paper A bootstrap requires exactly one category")
    image_ids = list(evaluator._paramsEval.imgIds)
    image_count = len(image_ids)
    eval_images = evaluator.evalImgs[:image_count]
    empty_entry = {
        "dtScores": np.empty((0,), dtype=float),
        "dtMatches": np.empty((len(evaluator.params.iouThrs), 0), dtype=float),
        "dtIgnore": np.empty((len(evaluator.params.iouThrs), 0), dtype=bool),
        "gtIgnore": np.empty((0,), dtype=bool),
    }
    eval_images = [entry if entry is not None else empty_entry for entry in eval_images]
    metrics = accumulate_ap(
        eval_images,
        np.ones(image_count, dtype=int),
        iou_thresholds=evaluator.params.iouThrs,
        recall_thresholds=evaluator.params.recThrs,
    )
    return {
        "image_ids": image_ids,
        "eval_images": eval_images,
        "iou_thresholds": evaluator.params.iouThrs,
        "recall_thresholds": evaluator.params.recThrs,
        "metrics": metrics,
    }


def accumulate_ap(
    eval_images: Sequence[dict],
    image_weights: np.ndarray,
    *,
    iou_thresholds: np.ndarray,
    recall_thresholds: np.ndarray,
    max_detections: int = 100,
) -> dict[str, float]:
    """Accumulate COCO AP after repeating images by integer weights."""

    if len(eval_images) != len(image_weights):
        raise ValueError("eval-image and weight counts must match")
    weighted = [
        entry
        for entry, weight in zip(eval_images, image_weights, strict=True)
        for _ in range(int(weight))
        if int(weight) > 0
    ]
    if not weighted:
        raise ValueError("at least one image must have positive weight")

    scores = np.concatenate([entry["dtScores"][:max_detections] for entry in weighted])
    order = np.argsort(-scores, kind="mergesort")
    matches = np.concatenate(
        [entry["dtMatches"][:, :max_detections] for entry in weighted], axis=1
    )[:, order]
    ignored = np.concatenate(
        [entry["dtIgnore"][:, :max_detections] for entry in weighted], axis=1
    )[:, order]
    gt_ignored = np.concatenate([entry["gtIgnore"] for entry in weighted])
    positive_gt = int(np.count_nonzero(gt_ignored == 0))
    if positive_gt == 0:
        raise ValueError("bootstrap sample contains no non-ignored ground truth")

    true_positives = np.logical_and(matches, np.logical_not(ignored))
    false_positives = np.logical_and(np.logical_not(matches), np.logical_not(ignored))
    tp_sum = np.cumsum(true_positives, axis=1, dtype=float)
    fp_sum = np.cumsum(false_positives, axis=1, dtype=float)
    precision = np.zeros((len(iou_thresholds), len(recall_thresholds)), dtype=float)
    for threshold_index, (tp, fp) in enumerate(zip(tp_sum, fp_sum, strict=True)):
        recall = tp / positive_gt
        curve = tp / (tp + fp + np.spacing(1))
        curve = np.maximum.accumulate(curve[::-1])[::-1]
        positions = np.searchsorted(recall, recall_thresholds, side="left")
        valid = positions < len(curve)
        precision[threshold_index, valid] = curve[positions[valid]]

    def metric_at(threshold: float) -> float:
        indices = np.flatnonzero(np.isclose(iou_thresholds, threshold))
        if len(indices) != 1:
            raise ValueError(f"expected one IoU threshold at {threshold}")
        return float(np.mean(precision[indices[0]]))

    return {
        "AP": float(np.mean(precision)),
        "AP50": metric_at(0.50),
        "AP75": metric_at(0.75),
    }


def bootstrap_metric_deltas(
    paired_eval_images: Sequence[tuple[Sequence[dict], Sequence[dict]]],
    *,
    iou_thresholds: np.ndarray,
    recall_thresholds: np.ndarray,
    replicates: int,
    rng_seed: int,
) -> dict[str, object]:
    """Bootstrap mean method-B minus method-A COCO metric deltas by image."""

    if not paired_eval_images:
        raise ValueError("at least one seed pair is required")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    image_count = len(paired_eval_images[0][0])
    if image_count < 1:
        raise ValueError("paired evaluations must contain images")
    if any(len(method_a) != image_count or len(method_b) != image_count
           for method_a, method_b in paired_eval_images):
        raise ValueError("all paired evaluations must have the same image count")

    full_weights = np.ones(image_count, dtype=int)
    point_deltas = {metric: [] for metric in METRICS}
    for method_a, method_b in paired_eval_images:
        metrics_a = accumulate_ap(
            method_a, full_weights, iou_thresholds=iou_thresholds,
            recall_thresholds=recall_thresholds,
        )
        metrics_b = accumulate_ap(
            method_b, full_weights, iou_thresholds=iou_thresholds,
            recall_thresholds=recall_thresholds,
        )
        for metric in METRICS:
            point_deltas[metric].append(metrics_b[metric] - metrics_a[metric])

    rng = np.random.default_rng(rng_seed)
    distributions = {metric: np.empty(replicates, dtype=float) for metric in METRICS}
    probabilities = np.full(image_count, 1.0 / image_count)
    for replicate in range(replicates):
        weights = rng.multinomial(image_count, probabilities)
        replicate_deltas = {metric: [] for metric in METRICS}
        for method_a, method_b in paired_eval_images:
            metrics_a = accumulate_ap(
                method_a, weights, iou_thresholds=iou_thresholds,
                recall_thresholds=recall_thresholds,
            )
            metrics_b = accumulate_ap(
                method_b, weights, iou_thresholds=iou_thresholds,
                recall_thresholds=recall_thresholds,
            )
            for metric in METRICS:
                replicate_deltas[metric].append(metrics_b[metric] - metrics_a[metric])
        for metric in METRICS:
            distributions[metric][replicate] = float(np.mean(replicate_deltas[metric]))

    return {
        "replicates": replicates,
        "rng_seed": rng_seed,
        "image_count": image_count,
        "seed_pair_count": len(paired_eval_images),
        "metrics": {
            metric: {
                "delta_mean": float(np.mean(point_deltas[metric])),
                "ci_low": float(np.quantile(distributions[metric], 0.025)),
                "ci_high": float(np.quantile(distributions[metric], 0.975)),
            }
            for metric in METRICS
        },
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=20260814)
    parser.add_argument("--replay-tolerance", type=float, default=5e-6)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs = manifest.get("pairs", [])
    if not pairs:
        raise SystemExit("bootstrap manifest contains no pairs")

    paired_eval_images = []
    provenance = []
    reference_image_ids = None
    reference_iou_thresholds = None
    reference_recall_thresholds = None
    ground_truth_hash = None
    for pair in pairs:
        seed_record: dict[str, object] = {"seed": int(pair["seed"]), "runs": {}}
        evaluations = {}
        for method_key in ("method_a", "method_b"):
            run_dir = Path(pair[f"{method_key}_run"])
            annotation_path = run_dir / "val_gt_paper_primary.json"
            detection_path = run_dir / "detections_best.json"
            results_path = run_dir / "results.json"
            for required_path in (annotation_path, detection_path, results_path):
                if not required_path.is_file():
                    raise SystemExit(f"required bootstrap input missing: {required_path}")

            current_gt_hash = file_sha256(annotation_path)
            if ground_truth_hash is None:
                ground_truth_hash = current_gt_hash
            elif current_gt_hash != ground_truth_hash:
                raise SystemExit("paired runs do not share byte-identical ground truth")

            evaluated = evaluate_detection_files(annotation_path, detection_path)
            results = json.loads(results_path.read_text(encoding="utf-8"))
            stored = results["reloaded_evaluation"]["paper_primary_coco"]["metrics"]
            replay_deltas = {
                metric: abs(float(evaluated["metrics"][metric]) - float(stored[metric]))
                for metric in METRICS
            }
            if max(replay_deltas.values()) > args.replay_tolerance:
                raise SystemExit(
                    f"metric replay exceeds tolerance for {run_dir}: {replay_deltas}"
                )

            image_ids = evaluated["image_ids"]
            if reference_image_ids is None:
                reference_image_ids = image_ids
                reference_iou_thresholds = evaluated["iou_thresholds"]
                reference_recall_thresholds = evaluated["recall_thresholds"]
            elif image_ids != reference_image_ids:
                raise SystemExit("paired runs do not share identical ordered image IDs")
            elif not np.array_equal(evaluated["iou_thresholds"], reference_iou_thresholds):
                raise SystemExit("paired runs do not share IoU thresholds")
            elif not np.array_equal(evaluated["recall_thresholds"], reference_recall_thresholds):
                raise SystemExit("paired runs do not share recall thresholds")

            evaluations[method_key] = evaluated["eval_images"]
            seed_record["runs"][method_key] = {
                "run_dir": str(run_dir),
                "ground_truth_sha256": current_gt_hash,
                "detections_sha256": file_sha256(detection_path),
                "results_sha256": file_sha256(results_path),
                "metrics": {metric: float(evaluated["metrics"][metric]) for metric in METRICS},
                "maximum_replay_delta": max(replay_deltas.values()),
            }
        paired_eval_images.append((evaluations["method_a"], evaluations["method_b"]))
        seed_record["deltas"] = {
            metric: (
                seed_record["runs"]["method_b"]["metrics"][metric]
                - seed_record["runs"]["method_a"]["metrics"][metric]
            )
            for metric in METRICS
        }
        provenance.append(seed_record)

    bootstrap = bootstrap_metric_deltas(
        paired_eval_images,
        iou_thresholds=reference_iou_thresholds,
        recall_thresholds=reference_recall_thresholds,
        replicates=args.replicates,
        rng_seed=args.rng_seed,
    )
    artifact = {
        "comparison_id": manifest["comparison_id"],
        "dataset": manifest["dataset"],
        "method_a": manifest["method_a"],
        "method_b": manifest["method_b"],
        "delta_definition": "method_b_minus_method_a",
        "bootstrap_unit": "original_image",
        "sampling": "same multinomial image multiplicities shared across all methods and fixed seeds",
        "confidence_interval": "two-sided percentile 95%",
        "ground_truth_sha256": ground_truth_hash,
        "manifest_sha256": file_sha256(args.manifest),
        "provenance": provenance,
        **bootstrap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "sha256": file_sha256(args.output),
        "metrics": artifact["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
