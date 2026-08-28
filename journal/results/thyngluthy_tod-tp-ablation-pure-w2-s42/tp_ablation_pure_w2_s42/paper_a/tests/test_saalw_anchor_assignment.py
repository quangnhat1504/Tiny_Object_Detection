from __future__ import annotations

import unittest

import torch

from paper_a.tools.audit_saalw_anchor_assignment import (
    _comparison_stats,
    _scale_bin,
    deterministic_image_sample,
)


class SaAlwAnchorAssignmentAuditTests(unittest.TestCase):
    def test_seeded_image_sample_is_deterministic_and_ordered(self) -> None:
        image_ids = list(range(100))
        first = deterministic_image_sample(image_ids, sample_size=12, seed=42)
        second = deterministic_image_sample(image_ids, sample_size=12, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(set(first)), 12)

    def test_scale_bins_use_frozen_schedule_bounds(self) -> None:
        self.assertEqual(_scale_bin(4.9, s_min=5.0, s_max=20.0), "below_s_min")
        self.assertEqual(_scale_bin(5.0, s_min=5.0, s_max=20.0), "adaptive_interval")
        self.assertEqual(_scale_bin(20.0, s_min=5.0, s_max=20.0), "above_s_max")

    def test_assignment_change_decomposition_is_exact(self) -> None:
        baseline = torch.tensor([-1, 0, 1, 1, -1, 2])
        candidate = torch.tensor([0, 0, 2, -1, -1, 2])
        stats = _comparison_stats(baseline, candidate)
        self.assertEqual(stats["changed_anchor_count"], 3)
        self.assertEqual(stats["positive_set_change_count"], 2)
        self.assertEqual(stats["added_positive_count"], 1)
        self.assertEqual(stats["dropped_positive_count"], 1)
        self.assertEqual(stats["owner_change_count"], 1)


if __name__ == "__main__":
    unittest.main()
