from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from paper_a.tools.validate_team_shards import load_rows, validate_rows


ROOT = Path(__file__).resolve().parents[2]
BOARD = ROOT / "paper_a" / "experiments" / "team_run_shards.csv"


class TeamRunShardTests(unittest.TestCase):
    def test_checked_in_board_passes_while_unassigned(self) -> None:
        rows, fieldnames = load_rows(BOARD)
        result = validate_rows(rows, fieldnames)
        self.assertEqual(result["status"], "PASS", result["errors"])
        self.assertEqual(result["assigned_training_shards"], 0)
        self.assertGreater(result["unassigned_training_shards"], 0)

    def test_ready_shard_requires_owner_account_and_gpu_hours(self) -> None:
        rows, fieldnames = load_rows(BOARD)
        rows[0] = dict(rows[0])
        rows[0]["status"] = "READY_FOR_PUSH"
        result = validate_rows(rows, fieldnames)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("requires owner and account" in e for e in result["errors"]))
        self.assertTrue(any("requires numeric GPU-hours" in e for e in result["errors"]))

    def test_ordinary_shard_cannot_request_final_test(self) -> None:
        rows, fieldnames = load_rows(BOARD)
        rows[0] = dict(rows[0])
        rows[0]["test_access"] = "final_test"
        result = validate_rows(rows, fieldnames)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("ordinary shard test_access" in e for e in result["errors"]))

    def test_component_pilot_must_match_main_owner_and_account(self) -> None:
        rows, fieldnames = load_rows(BOARD)
        by_id = {row["shard_id"]: row for row in rows}
        by_id["PILOT-D1-S42"]["owner"] = "member_a"
        by_id["PILOT-D1-S42"]["kaggle_account"] = "account_a"
        by_id["PILOT-COMP-D1-S42"]["owner"] = "member_b"
        by_id["PILOT-COMP-D1-S42"]["kaggle_account"] = "account_b"
        result = validate_rows(rows, fieldnames)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("must match PILOT-D1-S42" in e for e in result["errors"]))


if __name__ == "__main__":
    unittest.main()
