"""Behavioral tests for the independent Program B TinyPerson split builder."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_program_b_tinyperson_split import build_program_b_split


class ProgramBTinyPersonSplitTests(unittest.TestCase):
    def test_split_keeps_every_video_and_image_group_on_one_side(self) -> None:
        payload = {
            "images": [
                {"id": 1, "file_name": "bb_V0001_I0000010.jpg"},
                {"id": 2, "file_name": "bb_V0001_I0000020.jpg"},
                {"id": 3, "file_name": "bb_V0002_I0000010.jpg"},
                {"id": 4, "file_name": "photo_a.jpg"},
                {"id": 5, "file_name": "photo_b.jpg"},
            ],
            "annotations": [
                {"id": 11, "image_id": 1},
                {"id": 12, "image_id": 2},
                {"id": 13, "image_id": 3},
                {"id": 14, "image_id": 4},
                {"id": 15, "image_id": 5},
            ],
            "categories": [{"id": 1, "name": "person"}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            annotations = root / "source.json"
            annotations.write_text(json.dumps(payload), encoding="utf-8")

            report = build_program_b_split(
                annotations,
                root / "split",
                namespace="program_b_test",
                val_fraction=0.5,
            )

            train_payload = json.loads(
                (root / "split" / "program_b_train.json").read_text(encoding="utf-8")
            )
            val_payload = json.loads(
                (root / "split" / "program_b_val.json").read_text(encoding="utf-8")
            )

        train_ids = {image["id"] for image in train_payload["images"]}
        val_ids = {image["id"] for image in val_payload["images"]}
        self.assertFalse(train_ids & val_ids)
        self.assertIn(1, train_ids | val_ids)
        self.assertEqual((1 in train_ids), (2 in train_ids))
        self.assertEqual((1 in val_ids), (2 in val_ids))
        self.assertEqual(report["status"], "FROZEN_PROGRAM_B_VALIDATION_SPLIT")
        self.assertEqual(report["source_group_overlap"], [])
        self.assertEqual(report["counts"]["train"]["images"] + report["counts"]["val"]["images"], 5)


if __name__ == "__main__":
    unittest.main()
