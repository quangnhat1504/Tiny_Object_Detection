import json
import tempfile
import unittest
from pathlib import Path

from paper_a.evaluation.tinyperson_official import (
    default_evaluator_source,
    evaluate_tinyperson_official,
)


class TinyPersonOfficialEvaluatorTest(unittest.TestCase):
    def _annotation_file(self, directory: Path) -> Path:
        annotation = {
            "images": [
                {"id": 1, "file_name": "fixture.png", "width": 800, "height": 800}
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 100, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                    "ignore": 0,
                    "uncertain": 0,
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [300, 300, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                    "ignore": 0,
                    "uncertain": 1,
                },
            ],
            "categories": [{"id": 1, "name": "person"}],
        }
        path = directory / "fixture.json"
        path.write_text(json.dumps(annotation), encoding="utf-8")
        return path

    def test_perfect_prediction_and_uncertain_iod_match_pinned_protocol(self):
        if not default_evaluator_source().exists():
            self.skipTest("Pinned TinyPerson evaluator checkout is unavailable")

        with tempfile.TemporaryDirectory() as temporary_directory:
            annotation_file = self._annotation_file(Path(temporary_directory))
            result = evaluate_tinyperson_official(
                annotation_file,
                [
                    {
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [310, 310, 5, 5],
                        "score": 1.0,
                    },
                    {
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [100, 100, 10, 10],
                        "score": 0.9,
                    },
                ],
            )

        self.assertEqual(result["parameters"]["iou_thresholds"], [0.25, 0.5, 0.75])
        self.assertEqual(result["parameters"]["max_detections"], [200])
        self.assertTrue(result["parameters"]["ignore_uncertain"])
        self.assertTrue(result["parameters"]["use_iod_for_ignore"])
        self.assertAlmostEqual(result["metrics"]["AP25_all"], 1.0)
        self.assertAlmostEqual(result["metrics"]["AP50_all"], 1.0)
        self.assertAlmostEqual(result["metrics"]["AP75_all"], 1.0)
        self.assertAlmostEqual(result["metrics"]["AP50_tiny"], 1.0)
        self.assertAlmostEqual(result["metrics"]["AP50_tiny2"], 1.0)
        self.assertAlmostEqual(result["metrics"]["AP50_tiny1"], -1.0)

    def test_empty_results_are_rejected_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotation_file = self._annotation_file(Path(temporary_directory))
            with self.assertRaisesRegex(ValueError, "empty result list"):
                evaluate_tinyperson_official(annotation_file, [])

    def test_modified_evaluator_source_is_rejected(self):
        source = default_evaluator_source()
        if not source.exists():
            self.skipTest("Pinned TinyPerson evaluator checkout is unavailable")
        with tempfile.TemporaryDirectory() as temporary_directory:
            modified_source = Path(temporary_directory) / "cocoeval.py"
            modified_source.write_bytes(source.read_bytes() + b"\n# modified\n")
            annotation_file = self._annotation_file(Path(temporary_directory))
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                evaluate_tinyperson_official(
                    annotation_file,
                    [
                        {
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [100, 100, 10, 10],
                            "score": 1.0,
                        }
                    ],
                    evaluator_source=modified_source,
                )


if __name__ == "__main__":
    unittest.main()
