"""Behavioral tests for the Program B COCO-to-tile preparation contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.build_program_b_coco_tiles import build_coco_tiles


class ProgramBCocoTileTests(unittest.TestCase):
    def test_tiles_preserve_original_offsets_and_clip_yolo_boxes(self) -> None:
        payload = {
            "images": [
                {
                    "id": 7,
                    "file_name": "labeled_images/bb_V0001_I0000010.jpg",
                    "width": 800,
                    "height": 512,
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 7,
                    "category_id": 1,
                    "bbox": [480, 100, 80, 40],
                }
            ],
            "categories": [{"id": 1, "name": "person"}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "images"
            (image_root / "labeled_images").mkdir(parents=True)
            Image.new("RGB", (800, 512)).save(
                image_root / "labeled_images/bb_V0001_I0000010.jpg"
            )
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps(payload), encoding="utf-8")

            report = build_coco_tiles(annotations, image_root, root / "tiles", side="train")
            records = json.loads((root / "tiles" / "tile_manifest.json").read_text())
            labels = (root / "tiles" / "labels" / "7_x288_y0.txt").read_text().split()

        self.assertEqual(report["tile_count"], 2)
        self.assertEqual(records["original_count"], 1)
        self.assertEqual(
            [(tile["x1"], tile["y1"]) for tile in records["tiles"]],
            [(0, 0), (288, 0)],
        )
        self.assertEqual(records["tiles"][1]["original_image_id"], 7)
        self.assertEqual(labels[0], "0")
        self.assertAlmostEqual(float(labels[1]), 232 / 512, places=6)
        self.assertAlmostEqual(float(labels[3]), 80 / 512, places=6)
        self.assertEqual(records["tiles"][1]["label_count"], 1)


if __name__ == "__main__":
    unittest.main()
