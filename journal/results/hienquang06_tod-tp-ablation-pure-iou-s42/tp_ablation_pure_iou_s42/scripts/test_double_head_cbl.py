"""Focused geometry, gradient, inference, and reload tests for Double-Head CBL."""
from __future__ import annotations

import gc
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import DEVICE, seed_all
from common.dataset import build_training_datasets, collate_fn
from common.metrics import get_metric_fn
from common.model import (
    DoubleHeadCBLPredictor,
    _scale_roi_boxes,
    build_model,
)


def _test_roi_scaling() -> None:
    boxes = [torch.tensor([
        [10.0, 20.0, 30.0, 40.0],
        [0.0, 0.0, 10.0, 10.0],
    ])]
    enlarged = _scale_roi_boxes(boxes, [(45, 35)], 1.3)[0]
    expected = torch.tensor([
        [7.0, 17.0, 33.0, 43.0],
        [0.0, 0.0, 11.5, 11.5],
    ])
    assert torch.allclose(enlarged, expected)
    assert torch.equal(_scale_roi_boxes(boxes, [(45, 35)], 1.0)[0], boxes[0])
    print("RoI scaling: center-preserving enlargement and clipping PASSED")


def main() -> None:
    seed_all(42)
    _test_roi_scaling()

    build_kwargs = {
        "metric_fn": get_metric_fn("sa_alw_full"),
        "placement": "la_loss",
        "reliability_thr": 16.0,
        "box_loss_type": "cbl",
        "box_loss_warmup_epochs": 0,
        "use_double_head": True,
        "double_head_reg_roi_scale": 1.3,
        "double_head_num_convs": 4,
    }
    model = build_model(**build_kwargs).to(DEVICE)
    predictor = model.roi_heads.box_predictor
    assert isinstance(predictor, DoubleHeadCBLPredictor)
    assert predictor.is_distributional and predictor.is_double_head

    dataset = build_training_datasets(use_patches=False, is_train=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )
    images, targets_batch = next(iter(loader))
    images = [image.to(DEVICE) for image in images]
    targets_batch = [
        {
            key: value.to(DEVICE) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets_batch
    ]

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
        loss_dict = model(images, targets_batch)
        total_loss = sum(loss_dict.values())
    total_loss.backward()

    gradient_parameters = {
        "classification": predictor.cls_score.weight,
        "reg_projection": predictor.reg_projection.conv1.weight,
        "reg_bottleneck": predictor.reg_convs[-1].conv3.weight,
        "distribution": predictor.bbox_dist.weight,
    }
    for name, parameter in gradient_parameters.items():
        assert parameter.grad is not None, f"{name} did not receive a gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} gradient is non-finite"
        assert parameter.grad.abs().sum() > 0, f"{name} gradient is zero"
    assert all(torch.isfinite(value) for value in loss_dict.values())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    print(
        f"{DEVICE.type.upper()} batch: total={float(total_loss.detach()):.4f}, "
        f"cls={float(loss_dict['loss_classifier'].detach()):.4f}, "
        f"box={float(loss_dict['loss_box_reg'].detach()):.4f}"
    )

    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=(DEVICE.type == "cuda")
    ):
        predictions = model(images)
    assert len(predictions) == len(images)
    assert all(torch.isfinite(pred["boxes"]).all() for pred in predictions)
    assert all(torch.isfinite(pred["scores"]).all() for pred in predictions)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "cbl_double_head_smoke.pt"
        torch.save({"model": model.state_dict()}, checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location=DEVICE, weights_only=False)
        reloaded = build_model(**build_kwargs).to(DEVICE)
        reloaded.load_state_dict(checkpoint["model"])
        reloaded.eval()
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=(DEVICE.type == "cuda")
        ):
            reloaded_predictions = reloaded(images)
        assert torch.allclose(
            predictions[0]["boxes"],
            reloaded_predictions[0]["boxes"],
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.allclose(
            predictions[0]["scores"],
            reloaded_predictions[0]["scores"],
            atol=1e-4,
            rtol=1e-4,
        )

    del model, reloaded, optimizer, loss_dict, total_loss
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    print("Double-Head CBL forward/backward/inference/reload smoke PASSED")


if __name__ == "__main__":
    main()
