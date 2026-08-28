"""Focused exactness tests for the cross-head PCGrad backward helper."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.train_utils import (
    _backward_with_disjoint_pcgrad,
    _backward_with_pcgrad,
    train_one_epoch,
)


def _run_case(auxiliary_sign: float) -> tuple[dict[str, float], torch.Tensor]:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    detector_loss = parameter.square().sum()
    auxiliary_loss = auxiliary_sign * parameter.square().sum()
    metrics = _backward_with_pcgrad(
        detector_loss, auxiliary_loss, (parameter,), scaler)
    assert parameter.grad is not None
    return metrics, parameter.grad.detach().clone()


def main() -> None:
    conflict_metrics, conflict_gradient = _run_case(-0.5)
    assert conflict_metrics["conflict"] == 1.0
    assert abs(conflict_metrics["cosine"] + 1.0) < 1e-6
    assert torch.allclose(conflict_gradient, torch.tensor([2.0, 4.0]))

    aligned_metrics, aligned_gradient = _run_case(0.5)
    assert aligned_metrics["conflict"] == 0.0
    assert abs(aligned_metrics["cosine"] - 1.0) < 1e-6
    assert torch.allclose(aligned_gradient, torch.tensor([3.0, 6.0]))

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.roi_heads = torch.nn.Module()
            self.roi_heads.box_head = torch.nn.Module()
            self.roi_heads.box_head.weight = torch.nn.Parameter(
                torch.tensor([1.0, 2.0]))
            self.roi_heads._cbl_scale_distill_pcgrad = True

        def forward(self, images, targets):
            del images, targets
            weight = self.roi_heads.box_head.weight
            detector_loss = weight.square().sum()
            return {
                "loss_classifier": detector_loss,
                "loss_box_scale_distill": -0.5 * detector_loss,
            }

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    _, breakdown = train_one_epoch(
        model,
        optimizer,
        [([torch.zeros(1)], [{}])],
        scaler,
        torch.device("cpu"),
        epoch=1,
    )
    assert torch.allclose(
        model.roi_heads.box_head.weight,
        torch.tensor([0.8, 1.6]),
        atol=1e-6,
    )
    assert breakdown["pcgrad_conflict_rate"] == 1.0

    class DummyRPNModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rpn = torch.nn.Module()
            self.rpn.head = torch.nn.Module()
            self.rpn.head.weight = torch.nn.Parameter(
                torch.tensor([1.0, 2.0]))
            self.rpn._micro_rescue_pcgrad = True
            self.rpn._micro_rescue_stats = {
                "micro_gt": 4,
                "selected_gt": 2,
            }

        def forward(self, images, targets):
            del images, targets
            detector_loss = self.rpn.head.weight.square().sum()
            return {
                "loss_rpn_box_reg": detector_loss,
                "loss_rpn_micro_rescue": -0.5 * detector_loss,
            }

    rpn_model = DummyRPNModel()
    rpn_optimizer = torch.optim.SGD(rpn_model.parameters(), lr=0.1)
    _, rpn_breakdown = train_one_epoch(
        rpn_model,
        rpn_optimizer,
        [([torch.zeros(1)], [{}])],
        torch.amp.GradScaler("cuda", enabled=False),
        torch.device("cpu"),
        epoch=1,
    )
    assert torch.allclose(
        rpn_model.rpn.head.weight,
        torch.tensor([0.8, 1.6]),
        atol=1e-6,
    )
    assert rpn_breakdown["pcgrad_conflict_rate"] == 1.0
    assert rpn_breakdown["micro_rescue_valid_batch_rate"] == 1.0
    assert rpn_breakdown["micro_rescue_selection_coverage"] == 0.5

    class DummyFPNModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.fpn = torch.nn.Module()
            self.backbone.fpn.weight = torch.nn.Parameter(
                torch.tensor([1.0, 2.0]))
            self.backbone._moc_feature_pcgrad = True
            self.backbone._moc_feature_stats = {
                "micro_gt": 5,
                "selected_gt": 2,
            }

        def forward(self, images, targets):
            del images, targets
            detector_loss = self.backbone.fpn.weight.square().sum()
            return {
                "loss_classifier": detector_loss,
                "loss_fpn_micro_feature": -0.5 * detector_loss,
            }

    fpn_model = DummyFPNModel()
    fpn_optimizer = torch.optim.SGD(fpn_model.parameters(), lr=0.1)
    _, fpn_breakdown = train_one_epoch(
        fpn_model,
        fpn_optimizer,
        [([torch.zeros(1)], [{}])],
        torch.amp.GradScaler("cuda", enabled=False),
        torch.device("cpu"),
        epoch=1,
    )
    assert torch.allclose(
        fpn_model.backbone.fpn.weight,
        torch.tensor([0.8, 1.6]),
        atol=1e-6,
    )
    assert fpn_breakdown["pcgrad_conflict_rate"] == 1.0
    assert fpn_breakdown["micro_feature_valid_batch_rate"] == 1.0
    assert fpn_breakdown["micro_feature_selection_coverage"] == 0.4

    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    detector = first.square().sum() + second.square().sum()
    disjoint_metrics = _backward_with_disjoint_pcgrad(
        detector,
        (
            ("first", -0.5 * first.square().sum(), (first,)),
            ("second", 0.5 * second.square().sum(), (second,)),
        ),
        torch.amp.GradScaler("cuda", enabled=False),
    )
    assert disjoint_metrics["first"]["conflict"] == 1.0
    assert disjoint_metrics["second"]["conflict"] == 0.0
    assert torch.allclose(first.grad, torch.tensor([2.0, 4.0]))
    assert torch.allclose(second.grad, torch.tensor([9.0, 12.0]))
    try:
        detector = first.square().sum()
        _backward_with_disjoint_pcgrad(
            detector,
            (
                ("first", first.sum(), (first,)),
                ("overlap", first.square().sum(), (first,)),
            ),
            torch.amp.GradScaler("cuda", enabled=False),
        )
    except ValueError as error:
        assert "disjoint" in str(error)
    else:
        raise AssertionError("Overlapping PCGrad scopes were accepted")

    class DummyDualModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rpn = torch.nn.Module()
            self.rpn.head = torch.nn.Module()
            self.rpn.head.weight = torch.nn.Parameter(
                torch.tensor([1.0, 2.0]))
            self.rpn._micro_rescue_pcgrad = True
            self.rpn._micro_rescue_stats = {
                "micro_gt": 4,
                "selected_gt": 2,
            }
            self.backbone = torch.nn.Module()
            self.backbone.fpn = torch.nn.Module()
            self.backbone.fpn.weight = torch.nn.Parameter(
                torch.tensor([3.0, 4.0]))
            self.backbone._moc_feature_pcgrad = True
            self.backbone._moc_feature_stats = {
                "micro_gt": 4,
                "selected_gt": 2,
            }

        def forward(self, images, targets):
            del images, targets
            rpn_loss = self.rpn.head.weight.square().sum()
            fpn_loss = self.backbone.fpn.weight.square().sum()
            return {
                "loss_classifier": rpn_loss + fpn_loss,
                "loss_rpn_micro_rescue": -0.5 * rpn_loss,
                "loss_fpn_micro_feature": 0.5 * fpn_loss,
            }

    dual_model = DummyDualModel()
    dual_optimizer = torch.optim.SGD(dual_model.parameters(), lr=0.1)
    _, dual_breakdown = train_one_epoch(
        dual_model,
        dual_optimizer,
        [([torch.zeros(1)], [{}])],
        torch.amp.GradScaler("cuda", enabled=False),
        torch.device("cpu"),
        epoch=1,
        grad_clip=100.0,
    )
    assert torch.allclose(
        dual_model.rpn.head.weight,
        torch.tensor([0.8, 1.6]),
        atol=1e-6,
    )
    assert torch.allclose(
        dual_model.backbone.fpn.weight,
        torch.tensor([2.1, 2.8]),
        atol=1e-6,
    )
    assert dual_breakdown["pcgrad_rpn_conflict_rate"] == 1.0
    assert dual_breakdown["pcgrad_fpn_conflict_rate"] == 0.0
    assert dual_breakdown["pcgrad_rpn_batches"] == 1
    assert dual_breakdown["pcgrad_fpn_batches"] == 1
    print("PCGrad exactness tests PASSED")


if __name__ == "__main__":
    main()
