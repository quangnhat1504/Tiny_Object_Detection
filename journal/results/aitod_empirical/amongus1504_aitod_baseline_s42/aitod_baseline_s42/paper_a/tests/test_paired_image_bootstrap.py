"""Tests for paired original-image COCO AP bootstrap helpers."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper_a.tools.bootstrap_paired_coco import (
    accumulate_ap,
    bootstrap_metric_deltas,
    evaluate_detection_files,
)


class PairedImageBootstrapTests(unittest.TestCase):
    @staticmethod
    def _image(scores: list[float], matches: list[list[int]], gt_count: int) -> dict:
        return {
            "dtScores": np.asarray(scores, dtype=float),
            "dtMatches": np.asarray(matches, dtype=float),
            "dtIgnore": np.zeros((len(matches), len(scores)), dtype=bool),
            "gtIgnore": np.zeros((gt_count,), dtype=bool),
        }

    def test_weighted_images_match_explicit_repetition(self) -> None:
        images = [
            self._image([0.9, 0.3], [[1, 0], [1, 0], [1, 0]], 1),
            self._image([0.8], [[0], [1], [1]], 1),
        ]
        iou_thresholds = np.asarray([0.50, 0.75, 0.95])
        recall_thresholds = np.linspace(0.0, 1.0, 101)

        weighted = accumulate_ap(
            images,
            np.asarray([2, 1]),
            iou_thresholds=iou_thresholds,
            recall_thresholds=recall_thresholds,
        )
        repeated = accumulate_ap(
            [images[0], images[0], images[1]],
            np.ones(3, dtype=int),
            iou_thresholds=iou_thresholds,
            recall_thresholds=recall_thresholds,
        )

        self.assertEqual(set(weighted), {"AP", "AP50", "AP75"})
        for metric in weighted:
            self.assertAlmostEqual(weighted[metric], repeated[metric], places=12)

    def test_accumulation_replays_cocoeval_metrics(self) -> None:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        ground_truth = COCO()
        ground_truth.dataset = {
            "info": {},
            "images": [{"id": 1}, {"id": 2}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 1,
                 "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            ],
            "categories": [{"id": 1, "name": "person"}],
        }
        ground_truth.createIndex()
        detections = ground_truth.loadRes([
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 2, "category_id": 1, "bbox": [5, 0, 10, 10], "score": 0.8},
        ])
        evaluator = COCOeval(ground_truth, detections, "bbox")
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()

        image_count = len(evaluator._paramsEval.imgIds)
        all_area_images = evaluator.evalImgs[:image_count]
        replay = accumulate_ap(
            all_area_images,
            np.ones(image_count, dtype=int),
            iou_thresholds=evaluator.params.iouThrs,
            recall_thresholds=evaluator.params.recThrs,
        )

        self.assertAlmostEqual(replay["AP"], float(evaluator.stats[0]), places=12)
        self.assertAlmostEqual(replay["AP50"], float(evaluator.stats[1]), places=12)
        self.assertAlmostEqual(replay["AP75"], float(evaluator.stats[2]), places=12)

    def test_identical_methods_have_zero_paired_interval(self) -> None:
        images = [
            self._image([0.9], [[1], [1], [1]], 1),
            self._image([0.8], [[0], [1], [1]], 1),
        ]
        pairs = [(images, images), (images, images), (images, images)]
        result = bootstrap_metric_deltas(
            pairs,
            iou_thresholds=np.asarray([0.50, 0.75, 0.95]),
            recall_thresholds=np.linspace(0.0, 1.0, 101),
            replicates=10,
            rng_seed=42,
        )

        self.assertEqual(result["replicates"], 10)
        self.assertEqual(result["rng_seed"], 42)
        for metric in ("AP", "AP50", "AP75"):
            self.assertEqual(result["metrics"][metric]["delta_mean"], 0.0)
            self.assertEqual(result["metrics"][metric]["ci_low"], 0.0)
            self.assertEqual(result["metrics"][metric]["ci_high"], 0.0)

    def test_detection_file_evaluation_preserves_image_order(self) -> None:
        ground_truth = {
            "info": {},
            "images": [{"id": 7}, {"id": 3}, {"id": 11}],
            "annotations": [
                {"id": 1, "image_id": 7, "category_id": 1,
                 "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
                {"id": 2, "image_id": 3, "category_id": 1,
                 "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            ],
            "categories": [{"id": 1, "name": "person"}],
        }
        detections = [
            {"image_id": 7, "category_id": 1,
             "bbox": [0, 0, 10, 10], "score": 0.9},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_path = root / "gt.json"
            detection_path = root / "detections.json"
            gt_path.write_text(json.dumps(ground_truth), encoding="utf-8")
            detection_path.write_text(json.dumps(detections), encoding="utf-8")

            evaluated = evaluate_detection_files(gt_path, detection_path)

        self.assertEqual(evaluated["image_ids"], [3, 7, 11])
        self.assertEqual(len(evaluated["eval_images"]), 3)
        self.assertAlmostEqual(evaluated["metrics"]["AP50"], 0.5049504950495048)


if __name__ == "__main__":
    unittest.main()
