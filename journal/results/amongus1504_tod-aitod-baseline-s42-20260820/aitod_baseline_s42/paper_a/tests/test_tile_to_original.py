from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from paper_a.evaluation.tile_to_original import (
    TileRecord,
    original_targets_from_dataset,
    reconstruct_dataset_predictions,
    reconstruct_predictions,
)


def prediction(boxes, scores, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


class TileToOriginalTests(unittest.TestCase):
    def reconstruct(self, predictions, records, sizes=None):
        return reconstruct_predictions(
            predictions,
            records,
            sizes or {0: (100, 80)},
            score_threshold=0.01,
            nms_iou_threshold=0.5,
            max_detections=100,
        )

    def test_offsets_and_clipping(self) -> None:
        result = self.reconstruct(
            [prediction([[-5, -4, 30, 40]], [0.9], [1])],
            [TileRecord(0, 80, 60, 100, 80)],
        )[0]
        torch.testing.assert_close(
            result["boxes"], torch.tensor([[75.0, 56.0, 100.0, 80.0]])
        )

    def test_overlapping_tile_duplicate_is_removed(self) -> None:
        result = self.reconstruct(
            [
                prediction([[40, 20, 60, 40]], [0.9], [1]),
                prediction([[0, 20, 20, 40]], [0.8], [1]),
            ],
            [TileRecord(0, 0, 0, 60, 60), TileRecord(0, 40, 0, 100, 60)],
        )[0]
        self.assertEqual(len(result["boxes"]), 1)
        torch.testing.assert_close(
            result["boxes"], torch.tensor([[40.0, 20.0, 60.0, 40.0]])
        )
        torch.testing.assert_close(result["scores"], torch.tensor([0.9]))

    def test_different_classes_are_not_deduplicated(self) -> None:
        result = self.reconstruct(
            [
                prediction([[40, 20, 60, 40]], [0.9], [1]),
                prediction([[0, 20, 20, 40]], [0.8], [2]),
            ],
            [TileRecord(0, 0, 0, 60, 60), TileRecord(0, 40, 0, 100, 60)],
        )[0]
        self.assertEqual(len(result["boxes"]), 2)

    def test_empty_original_image_is_retained(self) -> None:
        result = self.reconstruct([], [], sizes={0: (50, 50), 1: (60, 40)})
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["boxes"].shape, (0, 4))
        self.assertEqual(result[1]["boxes"].shape, (0, 4))

    def test_single_tile_is_equivalent_except_fixed_filtering(self) -> None:
        source = prediction(
            [[3, 4, 20, 25], [40, 30, 55, 47]], [0.8, 0.7], [1, 2]
        )
        result = self.reconstruct(
            [source], [TileRecord(0, 0, 0, 100, 80)]
        )[0]
        torch.testing.assert_close(result["boxes"], source["boxes"])
        torch.testing.assert_close(result["scores"], source["scores"])
        torch.testing.assert_close(result["labels"], source["labels"])

    def test_prediction_metadata_count_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "counts must match"):
            self.reconstruct([prediction([], [], [])], [])

    def test_max_detections_is_enforced_after_nms(self) -> None:
        result = reconstruct_predictions(
            [prediction([[i * 5, 0, i * 5 + 2, 2] for i in range(5)],
                        [0.9, 0.8, 0.7, 0.6, 0.5], [1] * 5)],
            [TileRecord(0, 0, 0, 100, 80)],
            {0: (100, 80)},
            score_threshold=0.0,
            nms_iou_threshold=0.5,
            max_detections=3,
        )[0]
        self.assertEqual(len(result["boxes"]), 3)

    def test_border_crossing_gt_is_counted_once_from_original_annotations(self) -> None:
        dataset = SimpleNamespace(
            img_files=["original.jpg"],
            labels_cache={0: ([[0, 40.0, 20.0, 60.0, 40.0]], 100, 80)},
            tile_index=[(0, 0, 0, 60, 60), (0, 40, 0, 100, 60)],
        )
        tile_predictions = [
            prediction([[40, 20, 60, 40]], [0.9], [1]),
            prediction([[0, 20, 20, 40]], [0.8], [1]),
        ]
        predictions, targets = reconstruct_dataset_predictions(
            dataset,
            tile_predictions,
            score_threshold=0.01,
            nms_iou_threshold=0.5,
            max_detections=100,
        )
        self.assertEqual(len(predictions[0]["boxes"]), 1)
        self.assertEqual(len(targets[0]["boxes"]), 1)
        torch.testing.assert_close(
            targets[0]["boxes"], torch.tensor([[40.0, 20.0, 60.0, 40.0]])
        )
        self.assertEqual(int(targets[0]["iscrowd"].sum()), 0)

    def test_original_gt_total_matches_source_annotations(self) -> None:
        dataset = SimpleNamespace(
            img_files=["a.jpg", "b.jpg"],
            labels_cache={
                0: ([[0, 1.0, 2.0, 3.0, 4.0]], 10, 10),
                1: (
                    [
                        [1, 2.0, 2.0, 5.0, 6.0],
                        [0, 6.0, 1.0, 9.0, 4.0],
                    ],
                    10,
                    10,
                ),
            },
        )
        targets = original_targets_from_dataset(dataset)
        self.assertEqual(sum(len(target["boxes"]) for target in targets), 3)


if __name__ == "__main__":
    unittest.main()
