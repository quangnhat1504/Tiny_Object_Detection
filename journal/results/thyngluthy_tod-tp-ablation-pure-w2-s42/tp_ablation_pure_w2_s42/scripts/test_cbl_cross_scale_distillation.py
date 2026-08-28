"""Focused unit and gradient test for cross-scale CBL distillation."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.model import (
    _align_horizontal_flip_cbl_logits,
    _cbl_cross_scale_distillation_loss,
    _coordinate_reliable_cbl_weights,
    _teacher_flip_consensus_weights,
    _teacher_localization_advantage_mask,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(42)
    student = torch.randn(
        5, 4, 6, generator=generator, device=device, requires_grad=True)
    teacher = torch.randn(
        5, 4, 6, generator=generator, device=device, requires_grad=True)

    identical_loss = _cbl_cross_scale_distillation_loss(
        student, student.detach().clone())
    assert torch.isfinite(identical_loss)
    assert identical_loss.abs() < 1e-6

    mask = torch.tensor([True, False, True, False, True], device=device)
    weights = torch.tensor([2.0, 1.0, 1.5, 1.0, 0.5], device=device)
    loss = _cbl_cross_scale_distillation_loss(
        student,
        teacher,
        temperature=2.0,
        roi_weights=weights,
        roi_mask=mask,
    )
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None

    empty_loss = _cbl_cross_scale_distillation_loss(
        student.detach(), teacher, roi_mask=torch.zeros_like(mask))
    assert empty_loss == 0

    coordinate_student = student.detach().clone().requires_grad_(True)
    coordinate_weights = torch.zeros(5, 4, device=device)
    coordinate_weights[0, 0] = 1.0
    coordinate_loss = _cbl_cross_scale_distillation_loss(
        coordinate_student,
        teacher,
        coordinate_weights=coordinate_weights,
    )
    coordinate_loss.backward()
    assert coordinate_student.grad is not None
    assert coordinate_student.grad[0, 0].abs().sum() > 0
    assert coordinate_student.grad[0, 1:].abs().sum() == 0
    assert coordinate_student.grad[1:].abs().sum() == 0

    ordered_grid = torch.tensor(
        [-2.0, -0.5, 0.0, 0.5, 2.0], device=device)
    ordered_student = torch.full(
        (1, 4, 5), -8.0, device=device, requires_grad=True)
    with torch.no_grad():
        ordered_student[..., 0] = 8.0
    adjacent_teacher = torch.full((1, 4, 5), -8.0, device=device)
    adjacent_teacher[..., 1] = 8.0
    far_teacher = torch.full((1, 4, 5), -8.0, device=device)
    far_teacher[..., 4] = 8.0
    adjacent_w1 = _cbl_cross_scale_distillation_loss(
        ordered_student,
        adjacent_teacher,
        grid=ordered_grid,
        distance="ordered_w1",
    )
    far_w1 = _cbl_cross_scale_distillation_loss(
        ordered_student,
        far_teacher,
        grid=ordered_grid,
        distance="ordered_w1",
    )
    identical_w1 = _cbl_cross_scale_distillation_loss(
        ordered_student,
        ordered_student.detach().clone(),
        grid=ordered_grid,
        distance="ordered_w1",
    )
    assert identical_w1.abs() < 1e-6
    assert far_w1 > adjacent_w1 > 0
    far_w1.backward()
    assert ordered_student.grad is not None
    try:
        _cbl_cross_scale_distillation_loss(
            ordered_student.detach(),
            far_teacher,
            grid=torch.tensor([-1.0, 0.0, 0.0, 1.0], device=device),
            distance="ordered_w1",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Repeated ordered grid value was accepted")

    bounded_student = torch.full(
        (2, 4, 5), -8.0, device=device, requires_grad=True)
    with torch.no_grad():
        bounded_student[..., 0] = 8.0
    bounded_targets = torch.tensor(
        [[2.0, 0.5, -0.5, 0.0], [0.0, 0.0, 0.0, 0.0]],
        device=device,
    )
    bounded_weights = torch.zeros(2, 4, device=device)
    bounded_weights[0, :2] = torch.tensor([1.0, 0.5], device=device)
    bounded_loss = _cbl_cross_scale_distillation_loss(
        bounded_student,
        far_teacher.expand(2, -1, -1),
        coordinate_weights=bounded_weights,
        grid=ordered_grid,
        distance="teacher_bounded_gt",
        target_deltas=bounded_targets,
    )
    bounded_loss.backward()
    assert torch.isfinite(bounded_loss) and bounded_loss > 0
    assert bounded_student.grad is not None
    assert bounded_student.grad[0, :2].abs().sum() > 0
    assert bounded_student.grad[0, 2:].abs().sum() == 0
    assert bounded_student.grad[1].abs().sum() == 0

    symmetric_grid = torch.tensor(
        [-2.0, -1.0, -0.25, 0.25, 1.0, 2.0], device=device)
    flip_logits = torch.arange(
        24, dtype=torch.float32, device=device).reshape(1, 4, 6)
    aligned_flip = _align_horizontal_flip_cbl_logits(
        flip_logits, symmetric_grid)
    assert torch.equal(aligned_flip[:, 0], flip_logits[:, 0].flip(-1))
    assert torch.equal(aligned_flip[:, 1:], flip_logits[:, 1:])
    perfect_agreement = _teacher_flip_consensus_weights(
        flip_logits, flip_logits.clone())
    disagreeing = flip_logits.clone()
    disagreeing[:, 0] = disagreeing[:, 0].flip(-1)
    imperfect_agreement = _teacher_flip_consensus_weights(
        flip_logits, disagreeing)
    assert torch.allclose(perfect_agreement, torch.ones_like(perfect_agreement))
    assert imperfect_agreement[0, 0] < perfect_agreement[0, 0]
    assert not imperfect_agreement.requires_grad

    reliability_grid = torch.tensor([-1.0, 0.0, 1.0], device=device)
    reliability_student = torch.full((1, 4, 3), -10.0, device=device)
    reliability_teacher = torch.full((1, 4, 3), -10.0, device=device)
    reliability_student[0, :, 0] = 10.0
    reliability_teacher[0, 0, 2] = 10.0
    reliability_teacher[0, 1, 0] = 10.0
    reliability_teacher[0, 2, 0] = 10.0
    reliability_teacher[0, 3, 1] = 10.0
    reliability_targets = torch.tensor(
        [[1.0, 1.0, -1.0, 1.0]], device=device)
    reliability_weights = _coordinate_reliable_cbl_weights(
        reliability_student,
        reliability_teacher,
        reliability_targets,
        reliability_grid,
    )
    assert reliability_weights.shape == (1, 4)
    assert reliability_weights[0, 0] > 0.99
    assert reliability_weights[0, 1] == 0
    assert reliability_weights[0, 2] == 0
    assert reliability_weights[0, 3] > 0

    target_boxes = torch.tensor(
        [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]],
        dtype=torch.float32,
        device=device,
    )
    student_boxes = torch.tensor(
        [[0, 0, 8, 8], [0, 0, 9, 9], [0, 0, 10, 10]],
        dtype=torch.float32,
        device=device,
    )
    teacher_boxes = torch.tensor(
        [[0, 0, 10, 10], [0, 0, 9.05, 9.05], [0, 0, 9, 9]],
        dtype=torch.float32,
        device=device,
    )
    advantage = _teacher_localization_advantage_mask(
        student_boxes, teacher_boxes, target_boxes, margin=0.02)
    assert advantage.tolist() == [True, False, False]

    print(
        f"Cross-scale CBL distillation test PASSED on {device.type}: "
        f"loss={float(loss.detach()):.6f}, selected={int(mask.sum())}, "
        f"reliable_coordinates={int((reliability_weights > 0).sum())}"
    )


if __name__ == "__main__":
    main()
