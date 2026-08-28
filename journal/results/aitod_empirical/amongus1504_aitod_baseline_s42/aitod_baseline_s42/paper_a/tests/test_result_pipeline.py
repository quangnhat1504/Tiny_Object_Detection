import unittest
from pathlib import Path

from paper_a.tools.build_result_tables import _latex_escape, aggregate
from paper_a.tools.validate_result_ledgers import RUN_COLUMNS, _validate_run_row


class ResultAggregationTest(unittest.TestCase):
    def _row(self, seed: int, ap: float) -> dict[str, str]:
        row = {
            "dataset": "TinyPerson",
            "detector": "Faster R-CNN",
            "backbone": "ResNet-50-FPN",
            "method": "sa_alw_canonical",
            "placement": "la_loss",
            "seed": str(seed),
        }
        for metric in ("AP", "AP50", "AP75", "APS", "APM", "APL", "AR100"):
            row[metric] = str(ap)
        return row

    def test_matched_seeds_are_aggregated_deterministically(self):
        summaries = aggregate([self._row(2024, 0.3), self._row(42, 0.1), self._row(123, 0.2)])
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["seeds"], "42;123;2024")
        self.assertEqual(summaries[0]["n"], "3")
        self.assertEqual(summaries[0]["AP_mean"], "0.200000")
        self.assertEqual(summaries[0]["AP_std"], "0.100000")

    def test_latex_fields_are_escaped(self):
        self.assertEqual(_latex_escape("sa_alw & 50%"), r"sa\_alw \& 50\%")

    def _accepted_row(self) -> dict[str, str]:
        row = dict.fromkeys(RUN_COLUMNS, "")
        row.update(
            {
                "run_id": "paper-a-fixture",
                "dataset": "TinyPerson",
                "dataset_version": "fixture",
                "split_hash": "split-hash",
                "detector": "Faster R-CNN",
                "backbone": "ResNet-50-FPN",
                "method": "sa_alw_canonical",
                "placement": "la_loss",
                "seed": "42",
                "code_commit": "code-hash",
                "config_hash": "config-hash",
                "train_budget": "12 epochs",
                "checkpoint_rule": "coco_ap",
                "best_epoch": "9",
                "AP": "0.2",
                "AP50": "0.4",
                "AP75": "0.1",
                "status": "ACCEPTED",
            }
        )
        return row

    def test_accepted_row_requires_registered_seed(self):
        row = self._accepted_row()
        row["seed"] = "7"
        with self.assertRaisesRegex(ValueError, "unregistered seed"):
            _validate_run_row(Path("fixture.csv"), 2, row, {row["run_id"]})

    def test_accepted_row_requires_coco_ap_checkpoint(self):
        row = self._accepted_row()
        row["checkpoint_rule"] = "ap50"
        with self.assertRaisesRegex(ValueError, "checkpoint_rule"):
            _validate_run_row(Path("fixture.csv"), 2, row, {row["run_id"]})

    def test_accepted_row_requires_manifest_entry(self):
        row = self._accepted_row()
        with self.assertRaisesRegex(ValueError, "missing from manifest"):
            _validate_run_row(Path("fixture.csv"), 2, row, set())


if __name__ == "__main__":
    unittest.main()
