"""Manifest-backed reconstruction helpers for Program B tiled validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from paper_a.evaluation.tile_to_original import TileRecord


def records_from_tile_manifest(
    manifest_path: str | Path,
    tile_names: Sequence[str | Path],
) -> tuple[list[TileRecord], dict[int, tuple[int, int]]]:
    """Return records in loader order and original `(width, height)` by ID."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records_by_name: dict[str, dict[str, Any]] = {
        str(item["tile_name"]): dict(item) for item in payload["tiles"]
    }
    records: list[TileRecord] = []
    sizes: dict[int, tuple[int, int]] = {}
    for raw_name in tile_names:
        name = Path(raw_name).name
        item = records_by_name.get(name)
        if item is None:
            raise ValueError(f"missing tile in manifest: {name}")
        original_id = int(item["original_image_id"])
        size = (int(item["original_width"]), int(item["original_height"]))
        previous = sizes.setdefault(original_id, size)
        if previous != size:
            raise ValueError(f"inconsistent original dimensions for image {original_id}")
        records.append(
            TileRecord(
                original_id=original_id,
                x_offset=int(item["x1"]),
                y_offset=int(item["y1"]),
                x_end=int(item["x2"]),
                y_end=int(item["y2"]),
            )
        )
    return records, sizes


def reconstructed_predictions_to_coco(
    predictions: Sequence[dict[str, torch.Tensor]],
    original_ids: Iterable[int],
) -> list[dict[str, float | int | list[float]]]:
    """Convert original-coordinate torchvision predictions to COCO detections."""
    original_ids = list(original_ids)
    if len(predictions) != len(original_ids):
        raise ValueError("original prediction and ID counts must match")
    detections: list[dict[str, float | int | list[float]]] = []
    for image_id, prediction in zip(original_ids, predictions):
        boxes = prediction["boxes"].detach().cpu().float()
        scores = prediction["scores"].detach().cpu().float()
        labels = prediction["labels"].detach().cpu().long()
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError("prediction fields must have equal lengths")
        for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            x1, y1, x2, y2 = (float(value) for value in box)
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            if width == 0.0 or height == 0.0:
                continue
            detections.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [x1, y1, width, height],
                    "score": float(score),
                }
            )
    return detections


@torch.no_grad()
def evaluate_tiled_model(
    model: torch.nn.Module,
    loader: Iterable[tuple[Sequence[torch.Tensor], Sequence[Mapping[str, Any]]]],
    device: torch.device,
    annotation_file: str | Path,
    manifest_path: str | Path,
    tile_names: Sequence[str | Path],
    *,
    evaluator: Callable[[str | Path, Sequence[Mapping[str, Any]]], Mapping[str, Any]],
    score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
) -> dict[str, Any]:
    """Run tile inference in loader order before original-image evaluation."""
    model.eval()
    tile_predictions: list[dict[str, torch.Tensor]] = []
    for images, _ in loader:
        outputs = model([image.to(device) for image in images])
        tile_predictions.extend(
            {
                key: value.detach().cpu()
                for key, value in output.items()
                if isinstance(value, torch.Tensor)
            }
            for output in outputs
        )
    if len(tile_predictions) != len(tile_names):
        raise ValueError(
            "model prediction count does not match the frozen tile manifest order"
        )
    return evaluate_tiled_predictions(
        annotation_file,
        manifest_path,
        tile_names,
        tile_predictions,
        evaluator=evaluator,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )


def evaluate_tiled_predictions(
    annotation_file: str | Path,
    manifest_path: str | Path,
    tile_names: Sequence[str | Path],
    tile_predictions: Sequence[Mapping[str, torch.Tensor]],
    *,
    evaluator: Callable[[str | Path, Sequence[Mapping[str, Any]]], Mapping[str, Any]],
    score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
) -> dict[str, Any]:
    """Reconstruct tile predictions before dispatching one original-image evaluation."""
    from paper_a.evaluation.tile_to_original import reconstruct_predictions

    records, original_sizes = records_from_tile_manifest(manifest_path, tile_names)
    original_ids = sorted(original_sizes)
    predictions = reconstruct_predictions(
        tile_predictions,
        records,
        original_sizes,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )
    detections = reconstructed_predictions_to_coco(predictions, original_ids)
    if not detections:
        raise ValueError("no detections remain after original-image reconstruction")
    evaluation = dict(evaluator(annotation_file, detections))
    return {
        "original_image_count": len(original_ids),
        "tile_prediction_count": len(tile_predictions),
        "detection_count": len(detections),
        "postprocess": {
            "score_threshold": score_threshold,
            "nms_iou_threshold": nms_iou_threshold,
            "max_detections": max_detections,
        },
        "evaluation": evaluation,
    }
