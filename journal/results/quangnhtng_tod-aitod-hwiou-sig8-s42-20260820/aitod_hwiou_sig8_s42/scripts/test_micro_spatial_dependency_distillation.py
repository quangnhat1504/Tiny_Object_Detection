"""Focused algebra tests for the PC-MSDD spatial relation target."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.model import _micro_high_frequency_feature_loss
from scripts.audit_micro_object_feature_distillation import (
    _spatial_dependency_loss,
)


def main() -> None:
    generator = torch.Generator().manual_seed(42)
    teacher = torch.randn(3, 8, 3, 3, generator=generator)

    identical = _spatial_dependency_loss(
        teacher.clone().requires_grad_(True),
        teacher,
        temperature=0.20,
    )
    assert identical.shape == (3,)
    assert float(identical.detach().abs().max()) < 1e-6

    q, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
    rotated = torch.einsum("dc,nchw->ndhw", q, teacher)
    invariant = _spatial_dependency_loss(
        rotated,
        teacher,
        temperature=0.20,
    )
    assert float(invariant.detach().abs().max()) < 1e-6

    student = teacher.flip(-1).clone().requires_grad_(True)
    changed = _spatial_dependency_loss(
        student,
        teacher.requires_grad_(True),
        temperature=0.20,
    ).mean()
    assert torch.isfinite(changed) and float(changed.detach()) > 1e-5
    changed.backward()
    assert student.grad is not None and float(student.grad.norm()) > 0
    assert teacher.grad is None

    try:
        _spatial_dependency_loss(student, teacher, temperature=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-positive temperature should be rejected")

    high_teacher = torch.randn(3, 8, 5, 5, generator=generator)
    high_identical = _micro_high_frequency_feature_loss(
        high_teacher.clone().requires_grad_(True),
        high_teacher,
    )
    assert float(high_identical.detach().abs().max()) < 1e-6

    constant_teacher = torch.randn(2, 8, 1, 1, generator=generator).expand(
        -1, -1, 5, 5)
    constant_student = torch.randn(2, 8, 1, 1, generator=generator).expand(
        -1, -1, 5, 5)
    constant_loss = _micro_high_frequency_feature_loss(
        constant_student,
        constant_teacher,
    )
    assert float(constant_loss.detach().abs().max()) < 1e-6

    high_teacher.requires_grad_(True)
    high_student = high_teacher.detach().flip(-1).clone().requires_grad_(True)
    high_changed = _micro_high_frequency_feature_loss(
        high_student,
        high_teacher,
    ).mean()
    assert torch.isfinite(high_changed) and float(high_changed.detach()) > 1e-5
    high_changed.backward()
    assert high_student.grad is not None and float(high_student.grad.norm()) > 0
    assert high_teacher.grad is None

    print("PC-MSDD/PC-MHFD feature-target algebra tests passed")


if __name__ == "__main__":
    main()
