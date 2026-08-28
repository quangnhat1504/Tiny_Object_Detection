"""Focused CUDA test for PAA-style RPN localization-IoU prediction."""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.metrics import get_metric_fn
from common.model import _fuse_rpn_presence_iou_logits, build_model


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

    presence_probability = torch.tensor([0.2, 0.8])
    quality_probability = torch.tensor([0.5, 0.25])
    fused_logits = _fuse_rpn_presence_iou_logits(
        torch.logit(presence_probability),
        torch.logit(quality_probability),
    )
    expected_probability = (presence_probability * quality_probability).sqrt()
    if not torch.allclose(fused_logits.sigmoid(), expected_probability):
        raise AssertionError("RPN unified-score fusion does not match PAA")
    blended_logits = _fuse_rpn_presence_iou_logits(
        torch.logit(presence_probability),
        torch.logit(quality_probability),
        fusion_weight=0.5,
    )
    blended_expected = (
        presence_probability.pow(0.75) * quality_probability.pow(0.25))
    if not torch.allclose(
        blended_logits.sigmoid(), blended_expected
    ):
        raise AssertionError("Weighted RPN IoU fusion is incorrect")
    presence_only_logits = _fuse_rpn_presence_iou_logits(
        torch.logit(presence_probability),
        torch.logit(quality_probability),
        fusion_weight=0.0,
    )
    if not torch.allclose(
        presence_only_logits.sigmoid(), presence_probability
    ):
        raise AssertionError("Zero-weight RPN IoU fusion changed presence")

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
        rpn_iou_prediction=True,
        rpn_iou_prediction_loss_weight=0.5,
        rpn_iou_prediction_fusion_weight=0.5,
        rpn_iou_prediction_detached_tower=True,
    ).to(device)
    if not model.rpn.iou_prediction:
        raise AssertionError("RPN IoU prediction was not enabled")
    if model.rpn.iou_prediction_fusion_weight != 0.5:
        raise AssertionError("RPN IoU fusion weight was not stored")
    if model.rpn.head.iou_conv is None:
        raise AssertionError("Detached RPN IoU tower was not built")

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
        "loss_objectness",
        "loss_rpn_box_reg",
        "loss_rpn_iou_pred",
        "loss_box_refine",
    }
    missing = expected_losses.difference(losses)
    if missing:
        raise AssertionError(f"Missing RPN IoU losses: {sorted(missing)}")
    for name, loss in losses.items():
        if not torch.isfinite(loss):
            raise AssertionError(f"Non-finite {name}: {loss}")
    if losses["loss_rpn_iou_pred"] <= 0:
        raise AssertionError("RPN IoU-prediction loss must be positive")

    bbox_quality_gradient = torch.autograd.grad(
        losses["loss_rpn_iou_pred"],
        model.rpn.head.bbox_pred.weight,
        allow_unused=True,
        retain_graph=True,
    )[0]
    if bbox_quality_gradient is not None and bbox_quality_gradient.abs().sum() > 0:
        raise AssertionError("IoU target propagated into the RPN bbox predictor")
    shared_quality_gradient = torch.autograd.grad(
        losses["loss_rpn_iou_pred"],
        next(model.rpn.head.conv.parameters()),
        allow_unused=True,
        retain_graph=True,
    )[0]
    if shared_quality_gradient is not None:
        raise AssertionError("Detached IoU loss reached the shared RPN tower")
    backbone_quality_gradient = torch.autograd.grad(
        losses["loss_rpn_iou_pred"],
        next(
            parameter
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ),
        allow_unused=True,
        retain_graph=True,
    )[0]
    if backbone_quality_gradient is not None:
        raise AssertionError("Detached IoU loss reached the backbone")

    model.zero_grad(set_to_none=True)
    sum(losses.values()).backward()
    _assert_finite_nonzero_gradient(
        "IoU predictor", model.rpn.head.iou_pred.weight)
    _assert_finite_nonzero_gradient(
        "IoU tower", model.rpn.head.iou_conv.weight)
    _assert_finite_nonzero_gradient(
        "objectness predictor", model.rpn.head.cls_logits.weight)
    _assert_finite_nonzero_gradient(
        "bbox predictor", model.rpn.head.bbox_pred.weight)

    stats = model.rpn._rpn_iou_prediction_stats
    if stats.get("positive", 0) < 1:
        raise AssertionError(f"No positive IoU targets: {stats}")
    for key in ("target_mean", "prediction_mean", "mae"):
        if not 0 <= stats[key] <= 1:
            raise AssertionError(f"Invalid IoU-prediction stats: {stats}")

    empty_logits = torch.randn(4, device=device, requires_grad=True)
    empty_loss = model.rpn._iou_prediction_loss(
        empty_logits,
        torch.zeros(4, 4, device=device),
        [torch.zeros(4, device=device)],
        [torch.zeros(4, 4, device=device)],
    )
    empty_loss.backward()
    if empty_loss != 0 or empty_logits.grad is None:
        raise AssertionError("Empty-positive IoU loss is not graph-safe zero")

    saved_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    model.eval()
    with torch.inference_mode():
        proposals_before = [box.cpu() for box in _rpn_proposals(model, [image])]
        model.rpn.head.iou_pred.bias.add_(0.25)
        model.load_state_dict(saved_state)
        proposals_after = [box.cpu() for box in _rpn_proposals(model, [image])]

    for before, after in zip(proposals_before, proposals_after):
        if not torch.equal(before, after):
            raise AssertionError("Reloaded RPN IoU head changed proposals")
        if before.ndim != 2 or before.shape[1] != 4:
            raise AssertionError(f"Invalid proposal shape: {before.shape}")
        if not torch.isfinite(before).all():
            raise AssertionError("RPN IoU prediction produced non-finite proposals")

    summary = {name: float(loss.detach()) for name, loss in losses.items()}
    print(f"RPN IoU losses: {summary}")
    print(f"RPN IoU stats: {stats}")
    print(f"Evaluation proposals: {[len(box) for box in proposals_after]}")
    print("RPN IoU-prediction CUDA smoke PASSED")


if __name__ == "__main__":
    main()
