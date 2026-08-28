"""Map tile detections to original images and deduplicate them once per image."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torchvision.ops import batched_nms


@dataclass(frozen=True)
class TileRecord:
    original_id: int
    x_offset: int
    y_offset: int
    x_end: int
    y_end: int


def _empty_prediction() -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "scores": torch.zeros((0,), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }


def records_from_dataset(dataset) -> list[TileRecord]:
    return [
        TileRecord(
            original_id=int(original_id),
            x_offset=int(x1),
            y_offset=int(y1),
            x_end=int(x2),
            y_end=int(y2),
        )
        for original_id, x1, y1, x2, y2 in dataset.tile_index
    ]


def original_sizes_from_dataset(dataset) -> dict[int, tuple[int, int]]:
    sizes = {}
    for original_id, (_, width, height) in dataset.labels_cache.items():
        sizes[int(original_id)] = (int(width), int(height))
    return sizes


def original_targets_from_dataset(dataset) -> list[dict[str, torch.Tensor]]:
    """Build each original-image GT once, never from clipped tile annotations."""
    targets = []
    for original_id in range(len(dataset.img_files)):
        boxes, _, _ = dataset.labels_cache[original_id]
        if boxes:
            boxes_tensor = torch.tensor(
                [[box[1], box[2], box[3], box[4]] for box in boxes],
                dtype=torch.float32,
            )
            labels_tensor = torch.tensor(
                [int(box[0]) + 1 for box in boxes], dtype=torch.int64
            )
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        area = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0]).clamp_min(0)
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1]).clamp_min(0)
        )
        targets.append(
            {
                "boxes": boxes_tensor,
                "labels": labels_tensor,
                "area": area,
                "iscrowd": torch.zeros(len(labels_tensor), dtype=torch.int64),
                "image_id": torch.tensor([original_id], dtype=torch.int64),
            }
        )
    return targets


def reconstruct_predictions(
    tile_predictions: Sequence[Mapping[str, torch.Tensor]],
    tile_records: Sequence[TileRecord],
    original_sizes: Mapping[int, tuple[int, int]],
    *,
    score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
) -> list[dict[str, torch.Tensor]]:
    """Map tile boxes to original coordinates and apply fixed class-aware NMS."""
    if len(tile_predictions) != len(tile_records):
        raise ValueError("tile prediction and metadata counts must match")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1]")
    if not 0.0 <= nms_iou_threshold <= 1.0:
        raise ValueError("nms_iou_threshold must be in [0, 1]")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")

    grouped: dict[int, dict[str, list[torch.Tensor]]] = {
        int(original_id): {"boxes": [], "scores": [], "labels": []}
        for original_id in original_sizes
    }
    for prediction, record in zip(tile_predictions, tile_records):
        if record.original_id not in grouped:
            raise ValueError(f"unknown original image id: {record.original_id}")
        boxes = prediction.get("boxes", torch.zeros((0, 4))).detach().cpu().float()
        scores = prediction.get("scores", torch.zeros((0,))).detach().cpu().float()
        labels = prediction.get(
            "labels", torch.zeros((0,), dtype=torch.int64)
        ).detach().cpu().long()
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise ValueError("prediction boxes must have shape [N, 4]")
        if scores.shape != (len(boxes),) or labels.shape != (len(boxes),):
            raise ValueError("prediction scores and labels must match boxes")

        keep = scores >= score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if boxes.numel() == 0:
            continue
        boxes[:, (0, 2)] += float(record.x_offset)
        boxes[:, (1, 3)] += float(record.y_offset)
        width, height = original_sizes[record.original_id]
        boxes[:, (0, 2)] = boxes[:, (0, 2)].clamp(0, width)
        boxes[:, (1, 3)] = boxes[:, (1, 3)].clamp(0, height)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not valid.any():
            continue
        grouped[record.original_id]["boxes"].append(boxes[valid])
        grouped[record.original_id]["scores"].append(scores[valid])
        grouped[record.original_id]["labels"].append(labels[valid])

    reconstructed = []
    for original_id in sorted(original_sizes):
        group = grouped[original_id]
        if not group["boxes"]:
            reconstructed.append(_empty_prediction())
            continue
        boxes = torch.cat(group["boxes"], dim=0)
        scores = torch.cat(group["scores"], dim=0)
        labels = torch.cat(group["labels"], dim=0)
        keep = batched_nms(boxes, scores, labels, nms_iou_threshold)
        keep = keep[:max_detections]
        reconstructed.append(
            {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
        )
    return reconstructed


def reconstruct_dataset_predictions(
    dataset,
    tile_predictions: Sequence[Mapping[str, torch.Tensor]],
    *,
    score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    predictions = reconstruct_predictions(
        tile_predictions,
        records_from_dataset(dataset),
        original_sizes_from_dataset(dataset),
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )
    targets = original_targets_from_dataset(dataset)
    if not (len(predictions) == len(targets) == len(dataset.img_files)):
        raise AssertionError("original prediction and target counts must match")
    return predictions, targets
