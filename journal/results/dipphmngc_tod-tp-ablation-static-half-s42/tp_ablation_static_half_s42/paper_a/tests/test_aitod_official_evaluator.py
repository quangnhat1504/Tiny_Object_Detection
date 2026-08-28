import json
import tempfile
import unittest
from pathlib import Path

from paper_a.evaluation.aitodv2_official import evaluate_aitodv2_official


class AITODOfficialEvaluatorTest(unittest.TestCase):
    def _annotation_file(self, directory: Path) -> Path:
        annotation = {
            "images": [{"id": 1, "file_name": "fixture.png", "width": 800, "height": 800}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [100, 100, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 0, "name": "airplane"}],
        }
        path = directory / "fixture.json"
        path.write_text(json.dumps(annotation), encoding="utf-8")
        return path

    def test_perfect_tiny_prediction_matches_pinned_protocol(self):
        try:
            import aitodpycocotools  # noqa: F401
        except ImportError:
            self.skipTest("Pinned aitodpycocotools package is not installed")

        with tempfile.TemporaryDirectory() as temporary_directory:
            annotation_file = self._annotation_file(Path(temporary_directory))
            result = evaluate_aitodv2_official(
                annotation_file,
                [
                    {
                        "image_id": 1,
                        "category_id": 0,
                        "bbox": [100, 100, 10, 10],
                        "score": 1.0,
                    }
                ],
            )

        self.assertEqual(result["parameters"]["max_detections"], [1, 100, 1500])
        self.assertEqual(result["parameters"]["area_labels"], ["all", "verytiny", "tiny", "small", "medium"])
        self.assertAlmostEqual(result["metrics"]["AP"], 1.0)
        self.assertAlmostEqual(result["metrics"]["AP50"], 1.0)
        self.assertAlmostEqual(result["metrics"]["AP75"], 1.0)
        self.assertAlmostEqual(result["metrics"]["AP_tiny"], 1.0)
        self.assertAlmostEqual(result["metrics"]["AP_verytiny"], -1.0)

    def test_empty_results_are_rejected_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotation_file = self._annotation_file(Path(temporary_directory))
            with self.assertRaisesRegex(ValueError, "empty result list"):
                evaluate_aitodv2_official(annotation_file, [])


if __name__ == "__main__":
    unittest.main()
