from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from paper_a.datasets.coco_original import CocoOriginalDataset


class CocoOriginalDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.image_root = root / "images"
        self.image_root.mkdir()
        Image.new("RGB", (10, 10), color=(10, 20, 30)).save(
            self.image_root / "sample.png"
        )
        payload = {
            "images": [
                {"id": 7, "file_name": "sample.png", "width": 10, "height": 10}
            ],
            "categories": [
                {"id": 0, "name": "zero"},
                {"id": 7, "name": "seven"},
            ],
            "annotations": [
                {"id": 1, "image_id": 7, "category_id": 0, "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0},
                {"id": 2, "image_id": 7, "category_id": 7, "bbox": [4, 5, 2, 2], "area": 4, "iscrowd": 0},
                {"id": 3, "image_id": 7, "category_id": 0, "bbox": [1, 1, 0, 2], "area": 0, "iscrowd": 0},
                {"id": 4, "image_id": 7, "category_id": 0, "bbox": [20, 20, 2, 2], "area": 4, "iscrowd": 0},
                {"id": 5, "image_id": 7, "category_id": 0, "bbox": [2, 2, 3, 3], "area": 9, "iscrowd": 1},
                {"id": 6, "image_id": 7, "category_id": 0, "bbox": [3, 3, 2, 2], "area": 4, "iscrowd": 0, "ignore": 1},
            ],
        }
        self.annotation_file = root / "annotations.json"
        self.annotation_file.write_text(json.dumps(payload), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_zero_based_categories_are_mapped_away_from_background(self) -> None:
        dataset = CocoOriginalDataset(self.image_root, self.annotation_file)
        image, target = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 10, 10))
        self.assertEqual(target["labels"].tolist(), [1, 2])
        self.assertEqual(target["category_ids"].tolist(), [0, 7])
        self.assertEqual(target["image_id"].item(), 7)

    def test_official_filter_and_crowd_routing_contract(self) -> None:
        dataset = CocoOriginalDataset(self.image_root, self.annotation_file)
        _, target = dataset[0]
        self.assertEqual(target["annotation_ids"].tolist(), [1, 2])
        self.assertEqual(target["ignore_boxes"].tolist(), [[2.0, 2.0, 5.0, 5.0]])
        self.assertEqual(target["ignore_category_ids"].tolist(), [0])

    def test_prediction_mapping_is_invertible(self) -> None:
        dataset = CocoOriginalDataset(self.image_root, self.annotation_file)
        results = dataset.predictions_to_coco(
            [
                {
                    "boxes": torch.tensor([[1.0, 2.0, 4.0, 6.0], [4.0, 5.0, 6.0, 7.0]]),
                    "scores": torch.tensor([0.9, 0.8]),
                    "labels": torch.tensor([1, 2]),
                }
            ],
            [7],
        )
        self.assertEqual([row["category_id"] for row in results], [0, 7])
        self.assertEqual(results[0]["bbox"], [1.0, 2.0, 3.0, 4.0])

    def test_unknown_training_label_is_rejected(self) -> None:
        dataset = CocoOriginalDataset(self.image_root, self.annotation_file)
        with self.assertRaisesRegex(ValueError, "unknown contiguous"):
            dataset.predictions_to_coco(
                [
                    {
                        "boxes": torch.tensor([[1.0, 1.0, 2.0, 2.0]]),
                        "scores": torch.tensor([0.5]),
                        "labels": torch.tensor([0]),
                    }
                ],
                [7],
            )


if __name__ == "__main__":
    unittest.main()
