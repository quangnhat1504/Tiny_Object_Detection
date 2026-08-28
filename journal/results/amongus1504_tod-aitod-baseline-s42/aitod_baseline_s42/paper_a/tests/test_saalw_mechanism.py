from __future__ import annotations

import unittest

from paper_a.tools.audit_saalw_mechanism import run_audit


class SaAlwMechanismPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_audit()

    def test_preflight_passes_all_declared_checks(self) -> None:
        self.assertEqual(self.result["status"], "PASS_TECHNICAL_PREFLIGHT")
        self.assertTrue(all(self.result["checks"].values()))

    def test_beta_preserves_ranking_but_changes_threshold_count(self) -> None:
        rows = self.result["within_target_ranking"]
        self.assertTrue(all(row["rank_changes"] == 0 for row in rows))
        threshold = self.result["quality_threshold"]
        self.assertEqual(threshold["fixed_assigned_count"], 4)
        self.assertEqual(threshold["adaptive_assigned_count"], 3)
        self.assertLess(threshold["small_target_beta_distance_margin"], threshold["fixed_beta_distance_margin"])

    def test_beta_and_position_have_distinct_assignment_paths(self) -> None:
        ownership = self.result["cross_target_ownership"]
        self.assertNotEqual(ownership["fixed_owner"], ownership["adaptive_owner"])
        position = self.result["position_ranking"]
        self.assertNotEqual(position["alw_winner"], position["adaptive_position_winner"])
        regression = self.result["regression"]
        self.assertEqual(regression["alw_distance"], regression["beta_only_distance"])


if __name__ == "__main__":
    unittest.main()
