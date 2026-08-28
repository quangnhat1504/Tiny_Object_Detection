"""Focused CUDA test for the learned two-stage metric RPN cascade."""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.metrics import get_metric_fn
from common.model import build_model


def _rpn_proposals(model, images):
    image_list, _ = model.transform(images)
    features = model.backbone(image_list.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    proposals, losses = model.rpn(image_list, features)
    if losses:
        raise AssertionError(f"Evaluation RPN returned losses: {losses.keys()}")
    return proposals


def _assert_finite_nonzero_gradient(name: str, parameter: torch.Tensor) -> None:
    if parameter.grad is None:
        raise AssertionError(f"{name} did not receive a gradient")
    if not torch.isfinite(parameter.grad).all():
        raise AssertionError(f"{name} gradient is non-finite")
    if parameter.grad.abs().sum() == 0:
        raise AssertionError(f"{name} gradient is zero")


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
        cbl_refine_train_weight=0.5,
        cbl_refine_steps=1,
        cbl_refine_score_threshold=0.3,
        rpn_cascade=True,
        rpn_cascade_stage1_weight=1.0,
    ).to(device)
    if not model.rpn.cascade_refinement:
        raise AssertionError("RPN cascade was not enabled")

    image = torch.rand(3, 256, 256, device=device)
    target = {
        "boxes": torch.tensor(
            [
                [72.0, 80.0, 104.0, 116.0],
                [140.0, 140.0, 144.0, 144.0],
            ],
            device=device,
        ),
        "labels": torch.ones(2, dtype=torch.int64, device=device),
    }

    model.train()
    losses = model([image], [target])
    expected_losses = {
        "loss_rpn_stage1_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
        "loss_box_refine",
    }
    missing = expected_losses.difference(losses)
    if missing:
        raise AssertionError(f"Missing cascade losses: {sorted(missing)}")
    for name, loss in losses.items():
        if not torch.isfinite(loss):
            raise AssertionError(f"Non-finite {name}: {loss}")

    stage1_parameters = list(model.rpn.cascade_stage1_head.parameters())
    stage2_only = losses["loss_objectness"] + losses["loss_rpn_box_reg"]
    detached_gradients = torch.autograd.grad(
        stage2_only,
        stage1_parameters,
        allow_unused=True,
        retain_graph=True,
    )
    if any(
        gradient is not None and gradient.abs().sum() > 0
        for gradient in detached_gradients
    ):
        raise AssertionError("Stage-2 loss propagated through refined anchors")

    model.zero_grad(set_to_none=True)
    sum(losses.values()).backward()
    _assert_finite_nonzero_gradient(
        "stage1 bbox predictor", model.rpn.cascade_stage1_head.bbox_pred.weight)
    _assert_finite_nonzero_gradient(
        "stage2 objectness", model.rpn.head.cls_logits.weight)
    _assert_finite_nonzero_gradient(
        "stage2 bbox predictor", model.rpn.head.bbox_pred.weight)

    saved_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    model.eval()
    with torch.inference_mode():
        proposals_before = [box.cpu() for box in _rpn_proposals(model, [image])]
        model.rpn.cascade_stage1_head.bbox_pred.bias.add_(0.25)
        model.load_state_dict(saved_state)
        proposals_after = [box.cpu() for box in _rpn_proposals(model, [image])]

    for before, after in zip(proposals_before, proposals_after):
        if not torch.equal(before, after):
            raise AssertionError("Reloaded cascade changed RPN proposals")
        if before.ndim != 2 or before.shape[1] != 4:
            raise AssertionError(f"Invalid proposal shape: {before.shape}")
        if not torch.isfinite(before).all():
            raise AssertionError("Cascade produced non-finite proposals")

    summary = {name: float(loss.detach()) for name, loss in losses.items()}
    print(f"Cascade losses: {summary}")
    print(f"Evaluation proposals: {[len(box) for box in proposals_after]}")
    print("RPN cascade CUDA smoke PASSED")


if __name__ == "__main__":
    main()
