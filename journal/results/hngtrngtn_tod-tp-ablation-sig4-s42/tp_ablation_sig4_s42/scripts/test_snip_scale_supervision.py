"""Focused synthetic checks for SNIP-like scale-normalized supervision."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.metrics import get_metric_fn
from common.model import build_model
from common.train_utils import ModelEMA


MIN_SIZES = (640, 800, 960)
VALID_RANGES = ((20.0, float("inf")), (12.5, 50.0), (0.0, 30.0))


def _synthetic_sample(device: torch.device):
    image = torch.rand((3, 512, 512), device=device)
    boxes = torch.tensor(
        [
            [20.0, 20.0, 24.0, 24.0],
            [80.0, 80.0, 92.0, 92.0],
            [160.0, 160.0, 184.0, 184.0],
            [280.0, 280.0, 344.0, 344.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    target = {
        "boxes": boxes,
        "labels": torch.ones(4, dtype=torch.int64, device=device),
        "area": torch.tensor(
            [16.0, 144.0, 576.0, 4096.0],
            dtype=torch.float32,
            device=device,
        ),
        "iscrowd": torch.zeros(4, dtype=torch.int64, device=device),
        "image_id": torch.tensor([0], dtype=torch.int64, device=device),
    }
    return image, target


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        metric_fn=get_metric_fn("sa_alw_full"),
        placement="la_loss",
        box_loss_type="cbl",
        box_loss_warmup_epochs=0,
        cbl_refine_steps=1,
        cbl_refine_train_weight=0.5,
        cbl_refine_score_threshold=0.3,
        transform_min_sizes=MIN_SIZES,
        transform_max_size=1200,
        snip_valid_ranges=VALID_RANGES,
        snip_rpn_ignore_iou_thresh=0.4,
        snip_collect_stats=True,
    ).to(device)
    model.train()

    expected_masks = {
        640: [False, False, True, True],
        800: [False, True, True, False],
        960: [True, True, False, False],
    }
    transformed_by_scale = {}
    for min_size in MIN_SIZES:
        image, target = _synthetic_sample(device)
        model.transform.min_size = (min_size,)
        _, transformed_targets = model.transform([image], [target])
        transformed = transformed_targets[0]
        actual = transformed["_snip_valid"].tolist()
        if actual != expected_masks[min_size]:
            raise AssertionError(
                f"scale {min_size}: expected {expected_masks[min_size]}, "
                f"got {actual}"
            )
        transformed_by_scale[min_size] = transformed

    target_800 = transformed_by_scale[800]
    anchors = target_800["boxes"].clone()
    anchors = torch.cat(
        [anchors, anchors.new_tensor([[400.0, 400.0, 420.0, 420.0]])],
        dim=0,
    )
    rpn_labels, _ = model.rpn.assign_targets_to_anchors(
        [anchors], [target_800])
    rpn_labels = rpn_labels[0]
    if rpn_labels[1].item() != 1 or rpn_labels[2].item() != 1:
        raise AssertionError(f"valid GT anchors are not positive: {rpn_labels}")
    if rpn_labels[0].item() != -1 or rpn_labels[3].item() != -1:
        raise AssertionError(f"invalid GT anchors are not ignored: {rpn_labels}")

    proposals = [
        torch.cat([
            target_800["boxes"],
            target_800["boxes"].new_tensor(
                [[400.0, 400.0, 416.0, 416.0]]),
        ])
    ]
    model.roi_heads._snip_current_ranges = [
        target_800["_snip_valid_range"]]
    _, roi_labels = model.roi_heads.assign_targets_to_proposals(
        proposals,
        [target_800["boxes"]],
        [target_800["labels"]],
    )
    expected_roi_labels = [-1, 1, 1, -1, 0]
    if roi_labels[0].tolist() != expected_roi_labels:
        raise AssertionError(
            f"expected RoI labels {expected_roi_labels}, got {roi_labels[0]}")
    model.roi_heads._snip_current_ranges = None

    model.transform.min_size = (800,)
    image, target = _synthetic_sample(device)
    losses = model([image], [target])
    total_loss = sum(losses.values())
    if not torch.isfinite(total_loss):
        raise AssertionError(f"non-finite total loss: {losses}")
    total_loss.backward()

    model.transform.min_size = MIN_SIZES
    ema = ModelEMA(model)
    ema_model = ema.get_model()
    ema_model.transform.min_size = (640,)
    ema_image, _ = _synthetic_sample(device)
    transformed_images, transformed_targets = ema_model.transform(
        [ema_image], None)
    if transformed_images.image_sizes[0] != (640, 640):
        raise AssertionError(
            "EMA transform did not use its independent fixed eval size")
    if transformed_targets is not None:
        raise AssertionError("EMA inference transform unexpectedly made targets")

    print("SNIP scale masks:", expected_masks)
    print("RPN labels:", rpn_labels.tolist())
    print("RoI labels:", roi_labels[0].tolist())
    print(
        "Losses:",
        {name: round(float(value.detach()), 6)
         for name, value in losses.items()},
    )
    print("EMA eval image size:", transformed_images.image_sizes[0])
    print("PASS")


if __name__ == "__main__":
    main()
