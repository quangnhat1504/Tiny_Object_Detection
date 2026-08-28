import json
import tempfile
import unittest
from pathlib import Path

import torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from paper_a.tools.fit_train_scale_schedule import (
    detector_resize_shape,
    fit_schedule,
)


class TrainScaleScheduleTest(unittest.TestCase):
    def test_resize_shape_matches_torchvision_transform(self) -> None:
        transform = GeneralizedRCNNTransform(640, 800, [0, 0, 0], [1, 1, 1])
        for height, width in ((512, 640), (800, 800), (513, 777), (1200, 500)):
            expected = detector_resize_shape(height, width, 640, 800)
            image = torch.zeros((3, height, width), dtype=torch.float32)
            resized, _ = transform.resize(image)
            self.assertEqual(tuple(resized.shape[-2:]), expected)

    def test_fit_uses_only_valid_transformed_train_positives(self) -> None:
        payload = {
            "images": [
                {"id": 1, "file_name": "a.png", "width": 640, "height": 512}
            ],
            "categories": [{"id": 1, "name": "person"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 8, 8], "area": 64, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 16, 16], "area": 256, "iscrowd": 0},
                {"id": 3, "image_id": 1, "category_id": 1, "bbox": [0, 0, 100, 100], "area": 10000, "iscrowd": 0, "ignore": 1},
                {"id": 4, "image_id": 1, "category_id": 1, "bbox": [0, 0, 100, 100], "area": 10000, "iscrowd": 0, "uncertain": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotation_file = Path(temporary_directory) / "train.json"
            annotation_file.write_text(json.dumps(payload), encoding="utf-8")
            result = fit_schedule(
                annotation_file,
                split="train",
                min_size=640,
                max_size=800,
                lower_percentile=0,
                upper_percentile=100,
            )
        self.assertEqual(result["valid_positive_count"], 2)
        self.assertEqual(result["resize_shape_counts"], {"640x512->800x640": 1})
        self.assertAlmostEqual(result["schedule_bounds"]["s_min"], 10.0)
        self.assertAlmostEqual(result["schedule_bounds"]["s_max"], 20.0)

    def test_non_train_split_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "only on split=train"):
            fit_schedule(
                Path("unused.json"),
                split="validation",
                min_size=640,
                max_size=800,
                lower_percentile=10,
                upper_percentile=90,
            )


if __name__ == "__main__":
    unittest.main()
