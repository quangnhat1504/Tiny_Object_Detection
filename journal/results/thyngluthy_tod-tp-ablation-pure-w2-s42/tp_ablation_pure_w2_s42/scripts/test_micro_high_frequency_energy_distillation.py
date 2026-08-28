"""Algebra tests for the PC-MHED audit target."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.audit_micro_object_feature_distillation import (
    _micro_high_frequency_energy_loss,
)


def main() -> None:
    torch.manual_seed(42)
    student = torch.randn(3, 8, 7, 7, requires_grad=True)
    teacher = student.detach().clone()
    identical = _micro_high_frequency_energy_loss(student, teacher)
    assert identical.shape == (3,)
    assert torch.allclose(identical, torch.zeros_like(identical), atol=1e-6)

    changed_teacher = teacher.roll(shifts=1, dims=-1).requires_grad_(True)
    changed = _micro_high_frequency_energy_loss(student, changed_teacher)
    assert torch.isfinite(changed).all()
    assert (changed > 0).all()
    changed.mean().backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert changed_teacher.grad is None

    try:
        _micro_high_frequency_energy_loss(student, teacher[:, :, :-1])
    except ValueError:
        pass
    else:
        raise AssertionError("Shape mismatch was accepted")
    print("PC-MHED high-frequency energy algebra: PASS")


if __name__ == "__main__":
    main()
