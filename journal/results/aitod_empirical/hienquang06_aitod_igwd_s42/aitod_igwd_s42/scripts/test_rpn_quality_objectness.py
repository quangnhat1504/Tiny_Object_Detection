"""Focused CUDA test for localization-quality RPN objectness training."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torchvision.models.detection._utils import BoxCoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.metrics import get_metric_fn
from common.model import (
    _aligned_delta_iou_quality,
    _binary_quality_focal_loss,
    build_model,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this focused test")

    coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
    predicted = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.0],
    ])
    target = torch.zeros_like(predicted)
    quality = _aligned_delta_iou_quality(coder, predicted, target)
    if not torch.allclose(quality[:1], torch.ones(1)):
        raise AssertionError(f"Exact delta match must have IoU 1: {quality}")
    if not 0 < quality[1] < 1:
        raise AssertionError(f"Shifted delta must have fractional IoU: {quality}")

    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.25])
    qfl = _binary_quality_focal_loss(logits, targets, beta=2.0)
    qfl.backward()
    if not torch.isfinite(qfl) or not torch.isfinite(logits.grad).all():
        raise AssertionError("Binary RPN QFL produced non-finite values")

    torch.manual_seed(42)
    device = torch.device("cuda")
    model = build_model(
        metric_fn=get_metric_fn("sa_alw_full"),
        placement="la_loss",
        box_loss_type="cbl",
        box_loss_warmup_epochs=0,
        cbl_refine_steps=1,
        cbl_refine_score_threshold=0.3,
        cbl_refine_train_weight=0.5,
        rpn_quality_objectness=True,
        rpn_quality_beta=2.0,
        rpn_quality_preserve_below_size_ratio=0.015625,
    ).to(device)
    model.train()

    image = torch.rand(3, 256, 256, device=device)
    target_dict = {
        "boxes": torch.tensor([
            [72.0, 80.0, 104.0, 116.0],
            [140.0, 140.0, 143.0, 143.0],
        ], device=device),
        "labels": torch.tensor(
            [1, 1], dtype=torch.int64, device=device),
    }
    losses = model([image], [target_dict])
    total_loss = sum(losses.values())
    if not torch.isfinite(total_loss):
        raise AssertionError(f"Non-finite training loss: {losses}")
    total_loss.backward()

    objectness_parameters = [
        parameter
        for name, parameter in model.rpn.head.named_parameters()
        if "cls_logits" in name and parameter.requires_grad
    ]
    if not objectness_parameters or not all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in objectness_parameters
    ):
        raise AssertionError("RPN quality objectness gradients are missing")

    stats = model.rpn._rpn_quality_stats
    if stats.get("sampled_positive", 0) < 1:
        raise AssertionError(f"No sampled positive quality targets: {stats}")
    mean_quality = stats.get("positive_quality_mean", -1.0)
    if not 0 <= mean_quality <= 1:
        raise AssertionError(f"Invalid RPN quality target stats: {stats}")
    if stats.get("preserved_positive", 0) < 1:
        raise AssertionError(
            f"Micro positive targets were not preserved: {stats}")

    print(
        f"QFL={float(qfl.detach()):.6f}; "
        f"training_loss={float(total_loss.detach()):.4f}; "
        f"quality_stats={stats}"
    )
    print("RPN quality objectness CUDA smoke PASSED")


if __name__ == "__main__":
    main()
