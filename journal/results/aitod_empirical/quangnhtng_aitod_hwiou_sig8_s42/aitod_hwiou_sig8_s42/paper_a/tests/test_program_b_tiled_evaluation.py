"""Contract tests for Program B manifest-backed original-image evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from paper_a.evaluation.program_b_tiled import (
    evaluate_tiled_model,
    evaluate_tiled_predictions,
    records_from_tile_manifest,
    reconstructed_predictions_to_coco,
)
from paper_a.evaluation.tile_to_original import reconstruct_predictions


def prediction(boxes, scores, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


class _OneTileDataset(Dataset):
    tile_names = ["3_x0_y0.jpg"]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return torch.zeros((3, 32, 32)), {"image_id": torch.tensor([index])}


class _FixedPredictionModel(torch.nn.Module):
    def forward(self, images):
        return [prediction([[10, 20, 30, 50]], [0.7], [1]) for _ in images]


class ProgramBTiledEvaluationTests(unittest.TestCase):
    def test_manifest_records_reconstruct_to_original_coco_detections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "tile_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "tile_size": 512,
                        "tile_overlap": 64,
                        "tiles": [
                            {
                                "tile_name": "7_x0_y0.jpg",
                                "original_image_id": 7,
                                "original_width": 1000,
                                "original_height": 600,
                                "x1": 0,
                                "y1": 0,
                                "x2": 512,
                                "y2": 512,
                            },
                            {
                                "tile_name": "7_x488_y0.jpg",
                                "original_image_id": 7,
                                "original_width": 1000,
                                "original_height": 600,
                                "x1": 488,
                                "y1": 0,
                                "x2": 1000,
                                "y2": 512,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            records, sizes = records_from_tile_manifest(manifest_path, ["7_x0_y0.jpg", "7_x488_y0.jpg"])
            reconstructed = reconstruct_predictions(
                [
                    prediction([[490, 20, 510, 40]], [0.90], [1]),
                    prediction([[2, 20, 22, 40]], [0.80], [1]),
                ],
                records,
                sizes,
                score_threshold=0.05,
                nms_iou_threshold=0.5,
                max_detections=200,
            )
            detections = reconstructed_predictions_to_coco(reconstructed, sorted(sizes))

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["image_id"], 7)
        self.assertEqual(detections[0]["category_id"], 1)
        self.assertEqual(detections[0]["bbox"], [490.0, 20.0, 20.0, 20.0])
        self.assertAlmostEqual(detections[0]["score"], 0.90)

    def test_manifest_records_reject_unknown_dataset_tile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "tile_manifest.json"
            manifest_path.write_text(json.dumps({"tiles": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing tile"):
                records_from_tile_manifest(manifest_path, ["missing.jpg"])

    def test_evaluate_tiled_predictions_reconstructs_before_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "tile_manifest.json"
            annotation_path = root / "validation.json"
            manifest_path.write_text(
                json.dumps(
                    {"tiles": [{"tile_name": "3_x0_y0.jpg", "original_image_id": 3,
                                "original_width": 512, "original_height": 512,
                                "x1": 0, "y1": 0, "x2": 512, "y2": 512}]}
                ), encoding="utf-8"
            )
            annotation_path.write_text("{}", encoding="utf-8")
            called = {}
            def evaluator(annotation_file, detections):
                called["annotation_file"] = Path(annotation_file)
                called["detections"] = detections
                return {"protocol": "fixture"}

            result = evaluate_tiled_predictions(
                annotation_path,
                manifest_path,
                ["3_x0_y0.jpg"],
                [prediction([[10, 20, 30, 50]], [0.7], [1])],
                evaluator=evaluator,
                score_threshold=0.05,
                nms_iou_threshold=0.5,
                max_detections=200,
            )

        self.assertEqual(result["original_image_count"], 1)
        self.assertEqual(result["tile_prediction_count"], 1)
        self.assertEqual(result["detection_count"], 1)
        self.assertEqual(called["annotation_file"].name, "validation.json")
        self.assertEqual(called["detections"][0]["image_id"], 3)
        self.assertEqual(called["detections"][0]["bbox"], [10.0, 20.0, 20.0, 30.0])

    def test_evaluate_tiled_model_uses_loader_order_for_manifest_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "tile_manifest.json"
            annotation_path = root / "validation.json"
            manifest_path.write_text(
                json.dumps(
                    {"tiles": [{"tile_name": "3_x0_y0.jpg", "original_image_id": 3,
                                "original_width": 512, "original_height": 512,
                                "x1": 0, "y1": 0, "x2": 512, "y2": 512}]}
                ), encoding="utf-8"
            )
            annotation_path.write_text("{}", encoding="utf-8")
            loader = DataLoader(_OneTileDataset(), batch_size=1,
                                collate_fn=lambda batch: tuple(zip(*batch)))
            result = evaluate_tiled_model(
                _FixedPredictionModel(), loader, torch.device("cpu"), annotation_path,
                manifest_path, ["3_x0_y0.jpg"],
                evaluator=lambda _, detections: {"detections": len(detections)},
                score_threshold=0.05, nms_iou_threshold=0.5, max_detections=200,
            )

        self.assertEqual(result["tile_prediction_count"], 1)
        self.assertEqual(result["original_image_count"], 1)
        self.assertEqual(result["evaluation"]["detections"], 1)


if __name__ == "__main__":
    unittest.main()
