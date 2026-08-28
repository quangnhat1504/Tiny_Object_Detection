"""Tests for explicit Program B tiled data paths in the shared dataset factory."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from common.dataset import build_tiled_datasets


class ProgramBTrainingDataContractTests(unittest.TestCase):
    def test_explicit_tile_paths_build_train_and_validation_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for side in ("train", "validation"):
                (root / side / "images").mkdir(parents=True)
                (root / side / "labels").mkdir()
                Image.new("RGB", (512, 512)).save(root / side / "images" / "tile.jpg")
                (root / side / "labels" / "tile.txt").write_text(
                    "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
                )
            train, validation = build_tiled_datasets(
                root / "train" / "images", root / "train" / "labels",
                root / "validation" / "images", root / "validation" / "labels",
            )
        self.assertTrue(train.is_train)
        self.assertFalse(validation.is_train)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(validation), 1)


if __name__ == "__main__":
    unittest.main()
