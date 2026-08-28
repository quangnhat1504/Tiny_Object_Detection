"""Pinned TinyPerson official AP evaluator wrapper."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EVALUATOR_COMMIT = "bf6b83aa9a149ae15087eed4e9a7283f5cc67603"
EVALUATOR_SHA256 = "222b3173510e7a89bd03d077dce5d4a11e23ea6a7cd22afbbe930817b0886557"
EVALUATOR_RELATIVE_PATH = Path(
    "tiny_benchmark/maskrcnn_benchmark/data/datasets/evaluation/coco/cocoeval.py"
)
EXPECTED_IOU_THRESHOLDS = np.array([0.25, 0.5, 0.75])
EXPECTED_MAX_DETECTIONS = [200]
EXPECTED_AREA_LABELS = [
    "all",
    "tiny",
    "tiny1",
    "tiny2",
    "tiny3",
    "small",
    "reasonable",
]


def default_evaluator_source() -> Path:
    configured = os.environ.get("PAPER_A_TINYPERSON_EVALUATOR")
    if configured:
        return Path(configured)
    root = Path(__file__).resolve().parents[2]
    return (
        root
        / ".runtime"
        / "paper_a_sources"
        / "PointTinyBenchmark-pinned"
        / EVALUATOR_RELATIVE_PATH
    )


def _verified_source(path: Path) -> str:
    payload = path.read_bytes()
    # Git's Windows checkout may use CRLF, while the locked Git blob uses LF.
    canonical_payload = payload.replace(b"\r\n", b"\n")
    actual_hash = hashlib.sha256(canonical_payload).hexdigest()
    if actual_hash != EVALUATOR_SHA256:
        raise RuntimeError(
            f"TinyPerson evaluator hash mismatch: {actual_hash} != {EVALUATOR_SHA256}"
        )
    return canonical_payload.decode("utf-8")


def _load_official_module(source_path: Path):
    source = _verified_source(source_path)

    # The 2020 evaluator passes np.float64 counts to np.linspace. NumPy 2 rejects
    # those counts, so cast only the two locked expressions after hash validation.
    replacements = {
        "np.round((1.00 - .0) / .01) + 1": (
            "int(np.round((1.00 - .0) / .01) + 1)"
        ),
        "np.round((0.95 - .5) / .05) + 1": (
            "int(np.round((0.95 - .5) / .05) + 1)"
        ),
    }
    expected_counts = {
        "np.round((1.00 - .0) / .01) + 1": 3,
        "np.round((0.95 - .5) / .05) + 1": 2,
    }
    for old, new in replacements.items():
        if source.count(old) != expected_counts[old]:
            raise RuntimeError("Unexpected TinyPerson NumPy compatibility surface")
        source = source.replace(old, new)

    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]

    module = types.ModuleType("paper_a_tinyperson_cocoeval")
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    return module


def _load_results(
    results: str | Path | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(results, (str, Path)):
        with Path(results).open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    else:
        loaded = list(results)
    if not isinstance(loaded, list):
        raise TypeError("TinyPerson results must be a COCO detection list")
    if not loaded:
        raise ValueError("The pinned official evaluator cannot evaluate an empty result list")
    return [dict(item) for item in loaded]


def _metric_names() -> list[str]:
    names: list[str] = []
    for metric_type in ("AP", "AR"):
        for threshold in (25, 50, 75):
            for area_label in EXPECTED_AREA_LABELS:
                names.append(f"{metric_type}{threshold}_{area_label}")
    return names


def evaluate_tinyperson_official(
    annotation_file: str | Path,
    results: str | Path | Sequence[Mapping[str, Any]],
    *,
    evaluator_source: str | Path | None = None,
    quiet: bool = True,
) -> dict[str, Any]:
    """Evaluate original-image predictions with TinyPerson's official AP rules."""

    from pycocotools.coco import COCO

    detections = _load_results(results)
    source_path = Path(evaluator_source) if evaluator_source else default_evaluator_source()
    module = _load_official_module(source_path.resolve())
    module.Params.EVAL_STRANDARD = "tiny"

    output = io.StringIO()
    redirect = contextlib.redirect_stdout(output) if quiet else contextlib.nullcontext()
    with redirect:
        ground_truth = COCO(str(annotation_file))
        predictions = ground_truth.loadRes(detections)
        evaluator = module.COCOeval(
            ground_truth,
            predictions,
            "bbox",
            True,
            True,
            True,
        )
        if not np.array_equal(evaluator.params.iouThrs, EXPECTED_IOU_THRESHOLDS):
            raise RuntimeError("Pinned evaluator IoU thresholds changed")
        if list(evaluator.params.maxDets) != EXPECTED_MAX_DETECTIONS:
            raise RuntimeError("Pinned evaluator maxDets changed")
        if list(evaluator.params.areaRngLbl) != EXPECTED_AREA_LABELS:
            raise RuntimeError("Pinned evaluator area labels changed")
        if not evaluator.ignore_uncertain or not evaluator.use_iod_for_ignore:
            raise RuntimeError("Pinned evaluator ignore policy changed")

        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    stats = [float(value) for value in evaluator.stats]
    names = _metric_names()
    if len(stats) != len(names):
        raise RuntimeError(f"Unexpected TinyPerson metric count: {len(stats)}")
    return {
        "protocol": "tinyperson_official",
        "evaluator_commit": EVALUATOR_COMMIT,
        "evaluator_sha256": EVALUATOR_SHA256,
        "prediction_count": len(detections),
        "parameters": {
            "iou_thresholds": EXPECTED_IOU_THRESHOLDS.tolist(),
            "max_detections": EXPECTED_MAX_DETECTIONS,
            "area_labels": EXPECTED_AREA_LABELS,
            "ignore_uncertain": True,
            "use_iod_for_ignore": True,
        },
        "runtime_compatibility": "numpy2_integer_linspace_counts_and_np_float_alias",
        "metrics": dict(zip(names, stats)),
        "summary": output.getvalue() if quiet else "",
    }
