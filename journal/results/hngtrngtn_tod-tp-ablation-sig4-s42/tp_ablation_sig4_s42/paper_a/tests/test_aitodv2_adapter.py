"""
Unit test for AI-TOD-v2 Dataset Adapter.
"""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from paper_a.datasets.aitodv2_adapter import AITODv2Dataset, AITODV2_CATEGORIES


class TestAITODv2Adapter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.img_dir = self.root / "images"
        self.img_dir.mkdir()

        # Create dummy image
        img = Image.new("RGB", (800, 800), color=(128, 128, 128))
        img.save(self.img_dir / "test_001.jpg")

        # Create dummy COCO JSON for AI-TOD-v2
        self.ann_file = self.root / "aitodv2_dummy.json"
        data = {
            "images": [{"id": 1, "file_name": "test_001.jpg", "width": 800, "height": 800}],
            "annotations": [
                {"id": 101, "image_id": 1, "category_id": 1, "bbox": [100, 100, 12, 14], "area": 168, "ignore": 0},
                {"id": 102, "image_id": 1, "category_id": 6, "bbox": [200, 200, 6, 6], "area": 36, "ignore": 0},
                {"id": 103, "image_id": 1, "category_id": 7, "bbox": [300, 300, 4, 8], "area": 32, "ignore": 1},
            ],
            "categories": AITODV2_CATEGORIES,
        }
        self.ann_file.write_text(json.dumps(data), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_loading_and_filtering(self):
        ds = AITODv2Dataset(self.img_dir, self.ann_file)
        self.assertEqual(len(ds), 1)

        img_tensor, target = ds[0]
        self.assertEqual(img_tensor.shape[0], 3)
        self.assertIn("boxes", target)
        self.assertIn("labels", target)
        self.assertIn("ignore_boxes", target)

        # 2 positive boxes, 1 ignored
        self.assertEqual(len(target["boxes"]), 2)
        self.assertEqual(len(target["labels"]), 2)
        self.assertEqual(target["labels"][0].item(), 1)  # airplane -> label 1
        self.assertEqual(target["labels"][1].item(), 6)  # vehicle -> label 6
        self.assertEqual(len(target["ignore_boxes"]), 1)


if __name__ == "__main__":
    unittest.main()
