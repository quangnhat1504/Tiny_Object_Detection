"""Synthetic checks for class-aware custom scale AP."""
from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.eval_utils import compute_class_aware_scale_ap, compute_scale_ap


def sample(predicted_label: int):
    predictions = [{
        "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
        "scores": torch.tensor([0.99]),
        "labels": torch.tensor([predicted_label]),
    }]
    targets = [{
        "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
        "labels": torch.tensor([1]),
    }]
    return predictions, targets


def main() -> None:
    wrong_predictions, targets = sample(predicted_label=2)
    legacy = compute_scale_ap(wrong_predictions, targets)
    corrected = compute_class_aware_scale_ap(wrong_predictions, targets)
    assert legacy["AP_micro"] == 1.0
    assert corrected["AP_micro_class_aware"] == 0.0

    correct_predictions, targets = sample(predicted_label=1)
    corrected = compute_class_aware_scale_ap(correct_predictions, targets)
    assert corrected["AP_micro_class_aware"] == 1.0
    assert corrected["n_gt_micro_class_aware"] == 1
    print("class-aware scale AP synthetic checks PASSED")


if __name__ == "__main__":
    main()
