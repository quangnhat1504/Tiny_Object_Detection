"""Focused CUDA test for evaluation-only iterative RPN proposal refinement."""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.metrics import get_metric_fn
from common.model import build_model, iterative_rpn_proposals


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this focused test")

    torch.manual_seed(42)
    device = torch.device("cuda")
    model = build_model(
        metric_fn=get_metric_fn("sa_alw_full"),
        placement="la_loss",
        box_loss_type="cbl",
        box_loss_warmup_epochs=0,
        rpn_refine_steps=1,
        rpn_refine_min_size_ratio=0.03125,
    ).to(device)

    image = torch.rand(3, 256, 256, device=device)
    target = {
        "boxes": torch.tensor(
            [[72.0, 80.0, 104.0, 116.0]], device=device),
        "labels": torch.tensor([1], dtype=torch.int64, device=device),
    }

    model.train()
    losses = model([image], [target])
    total_loss = sum(losses.values())
    if not torch.isfinite(total_loss):
        raise AssertionError(f"Non-finite training loss: {losses}")
    total_loss.backward()
    if not any(
        parameter.grad is not None
        for parameter in model.rpn.parameters()
        if parameter.requires_grad
    ):
        raise AssertionError("Training did not produce RPN gradients")

    model.eval()
    with torch.inference_mode():
        image_list, _ = model.transform([image], None)
        features = model.backbone(image_list.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        actual, losses = model.rpn(image_list, features)
        expected = iterative_rpn_proposals(
            model.rpn,
            image_list,
            features,
            total_passes=2,
            min_refine_size_ratio=0.03125,
        )

    if losses:
        raise AssertionError(f"Evaluation RPN returned losses: {losses}")
    if len(actual) != len(expected):
        raise AssertionError("Evaluation proposal batch length differs")
    for actual_boxes, expected_boxes in zip(actual, expected):
        if not torch.equal(actual_boxes, expected_boxes):
            max_diff = (
                float((actual_boxes - expected_boxes).abs().max().item())
                if actual_boxes.shape == expected_boxes.shape
                and actual_boxes.numel()
                else float("inf")
            )
            raise AssertionError(
                f"Wrapped pass-2 proposals differ; max diff={max_diff}")

    print(f"Training loss={float(total_loss.detach()):.4f}")
    print("Iterative RPN train/eval wrapper smoke PASSED")


if __name__ == "__main__":
    main()
