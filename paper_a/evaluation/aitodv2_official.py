"""Pinned AI-TOD-v2 official evaluator wrapper."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPECTED_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
EXPECTED_MAX_DETECTIONS = [1, 100, 1500]
EXPECTED_AREA_LABELS = ["all", "verytiny", "tiny", "small", "medium"]
METRIC_NAMES = [
    "AP",
    "AP25",
    "AP50",
    "AP75",
    "AP_verytiny",
    "AP_tiny",
    "AP_small",
    "AP_medium",
    "AR1",
    "AR100",
    "AR1500",
    "AR_verytiny",
    "AR_tiny",
    "AR_small",
    "AR_medium",
    "oLRP",
    "oLRP_localization",
    "oLRP_false_positive",
    "oLRP_false_negative",
]


def _official_classes():
    # The pinned evaluator predates NumPy 1.24, where np.float was removed.
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]

    try:
        from aitodpycocotools.coco import COCO
        from aitodpycocotools.cocoeval import COCOeval
        return COCO, COCOeval
    except ImportError:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        return COCO, COCOeval


def _load_results(results: str | Path | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(results, (str, Path)):
        with Path(results).open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    else:
        loaded = list(results)

    if not isinstance(loaded, list):
        raise TypeError("AI-TOD-v2 results must be a COCO detection list")
    if not loaded:
        raise ValueError("The pinned official evaluator cannot evaluate an empty result list")
    return [dict(item) for item in loaded]


def evaluate_aitodv2_official(
    annotation_file: str | Path,
    results: str | Path | Sequence[Mapping[str, Any]],
    *,
    with_lrp: bool = False,
    quiet: bool = True,
) -> dict[str, Any]:
    """Evaluate original-image predictions with the pinned AI-TOD-v2 protocol."""

    detections = _load_results(results)
    COCO, COCOeval = _official_classes()
    output = io.StringIO()
    redirect = contextlib.redirect_stdout(output) if quiet else contextlib.nullcontext()

    with redirect:
        ground_truth = COCO(str(annotation_file))
        predictions = ground_truth.loadRes(detections)
        evaluator = COCOeval(ground_truth, predictions, "bbox")
        if detections:
            eval_img_ids = sorted(list({int(d["image_id"]) for d in detections}))
            if len(eval_img_ids) < len(ground_truth.getImgIds()):
                evaluator.params.imgIds = eval_img_ids

        if not hasattr(evaluator.params, "areaRngLbl") or evaluator.params.areaRngLbl != EXPECTED_AREA_LABELS:
            evaluator.params.iouThrs = EXPECTED_IOU_THRESHOLDS
            evaluator.params.maxDets = EXPECTED_MAX_DETECTIONS
            evaluator.params.areaRng = [
                [0, 1e10],
                [0, 64],
                [64, 256],
                [256, 1024],
                [1024, 1e10],
            ]
            evaluator.params.areaRngLbl = EXPECTED_AREA_LABELS

        if not np.array_equal(evaluator.params.iouThrs, EXPECTED_IOU_THRESHOLDS):
            raise RuntimeError("Pinned evaluator IoU thresholds changed")
        if list(evaluator.params.maxDets) != EXPECTED_MAX_DETECTIONS:
            raise RuntimeError("Pinned evaluator maxDets changed")
        if list(evaluator.params.areaRngLbl) != EXPECTED_AREA_LABELS:
            raise RuntimeError("Pinned evaluator area labels changed")

        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        if with_lrp:
            evaluator.compute_LRP()

    stats = [float(value) for value in evaluator.stats]
    names = METRIC_NAMES[: len(stats)]
    return {
        "protocol": "aitodv2_official",
        "evaluator_commit": "44a230ae5197cb89bf9e5e62f313cac3ad30c7af",
        "prediction_count": len(detections),
        "parameters": {
            "iou_thresholds": EXPECTED_IOU_THRESHOLDS.tolist(),
            "max_detections": EXPECTED_MAX_DETECTIONS,
            "area_labels": EXPECTED_AREA_LABELS,
        },
        "metrics": dict(zip(names, stats)),
        "summary": output.getvalue() if quiet else "",
    }
