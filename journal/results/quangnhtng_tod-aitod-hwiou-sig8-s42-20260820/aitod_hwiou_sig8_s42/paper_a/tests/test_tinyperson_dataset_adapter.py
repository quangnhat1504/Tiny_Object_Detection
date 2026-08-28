import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from paper_a.datasets.tinyperson_original import TinyPersonOriginalDataset


class TinyPersonDatasetAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.image_root = root / "images"
        self.image_root.mkdir()
        Image.new("RGB", (20, 20), color=(10, 20, 30)).save(
            self.image_root / "sample.png"
        )
        self.annotation_file = root / "annotations.json"
        self.payload = {
            "images": [
                {
                    "id": 7,
                    "file_name": "sample.png",
                    "width": 10,
                    "height": 10,
                    "corner": [5, 5, 15, 15],
                }
            ],
            "categories": [{"id": 1, "name": "person"}],
            "annotations": [
                {"id": 1, "image_id": 7, "category_id": 1, "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0, "ignore": 0, "uncertain": 0},
                {"id": 2, "image_id": 7, "category_id": 1, "bbox": [5, 5, 2, 2], "area": 4, "iscrowd": 0, "ignore": 0, "uncertain": 1},
                {"id": 3, "image_id": 7, "category_id": 1, "bbox": [8, 8, 2, 2], "area": 4, "iscrowd": 0, "ignore": 1, "uncertain": 0},
                {"id": 4, "image_id": 7, "category_id": 1, "bbox": [2, 7, 2, 2], "area": 4, "iscrowd": 1, "ignore": 0, "uncertain": 0},
                {"id": 5, "image_id": 7, "category_id": 1, "bbox": [30, 30, 2, 2], "area": 4, "iscrowd": 0, "ignore": 0, "uncertain": 0},
            ],
        }
        self.annotation_file.write_text(json.dumps(self.payload), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_uncertain_ignore_and_crowd_are_not_training_positives(self) -> None:
        dataset = TinyPersonOriginalDataset(self.image_root, self.annotation_file)
        image, target = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 10, 10))
        self.assertEqual(target["annotation_ids"].tolist(), [1])
        self.assertEqual(target["labels"].tolist(), [1])
        self.assertEqual(len(target["ignore_boxes"]), 3)
        self.assertEqual(target["ignore_category_ids"].tolist(), [1, 1, 1])

    def test_prediction_mapping_preserves_official_category(self) -> None:
        dataset = TinyPersonOriginalDataset(self.image_root, self.annotation_file)
        result = dataset.predictions_to_coco(
            [
                {
                    "boxes": torch.tensor([[1.0, 2.0, 4.0, 6.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([1]),
                }
            ],
            [7],
        )
        self.assertEqual(result[0]["category_id"], 1)
        self.assertEqual(result[0]["bbox"], [1.0, 2.0, 3.0, 4.0])

    def test_non_binary_task_annotation_is_rejected(self) -> None:
        self.payload["categories"] = [
            {"id": 1, "name": "sea_person"},
            {"id": 2, "name": "earth_person"},
        ]
        self.annotation_file.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "binary task-all"):
            TinyPersonOriginalDataset(self.image_root, self.annotation_file)

    def test_missing_corner_annotation_is_rejected_for_training(self) -> None:
        del self.payload["images"][0]["corner"]
        self.annotation_file.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "corner annotation"):
            TinyPersonOriginalDataset(self.image_root, self.annotation_file)


if __name__ == "__main__":
    unittest.main()
