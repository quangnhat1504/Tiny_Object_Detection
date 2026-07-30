"""Focused CUDA, gradient, inference, and reload test for iterative CBL training."""
from __future__ import annotations

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
from common.model import build_model


def run_variant(images, targets, *, separate_head: bool) -> None:
    seed_all(42)
    build_kwargs = {
        "metric_fn": get_metric_fn("sa_alw_full"),
        "placement": "la_loss",
        "reliability_thr": 16.0,
        "box_loss_type": "cbl",
        "box_loss_warmup_epochs": 0,
        "cbl_refine_train_weight": 0.5,
        "cbl_refine_steps": 1,
        "cbl_refine_blend": 1.0,
        "cbl_refine_score_threshold": 0.3,
        "cbl_refine_separate_head": separate_head,
    }
    model = build_model(**build_kwargs).to(DEVICE)
    model.train()
    loss_dict = model(images, targets)
    assert "loss_box_refine" in loss_dict
    refine_loss = loss_dict["loss_box_refine"]
    assert torch.isfinite(refine_loss) and refine_loss > 0

    model.zero_grad(set_to_none=True)
    refine_loss.backward()
    if separate_head:
        gradient_parameters = {
            "refine_box_head": model.roi_heads.refine_box_head.fc7.weight,
            "refine_distribution": (
                model.roi_heads.refine_box_predictor.bbox_dist.weight),
        }
        assert model.roi_heads.refine_box_predictor.cls_score.weight.grad is None
    else:
        gradient_parameters = {
            "box_head": model.roi_heads.box_head.fc7.weight,
            "distribution": model.roi_heads.box_predictor.bbox_dist.weight,
        }
    for name, parameter in gradient_parameters.items():
        assert parameter.grad is not None, f"{name} did not receive a gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} gradient is non-finite"
        assert parameter.grad.abs().sum() > 0, f"{name} gradient is zero"
    print(
        f"{DEVICE.type.upper()} "
        f"{'stage-specific' if separate_head else 'shared-head'} "
        f"iterative loss={float(refine_loss.detach()):.4f}"
    )

    model.eval()
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=(DEVICE.type == "cuda")
    ):
        predictions = model(images)
    assert len(predictions) == len(images)
    assert all(torch.isfinite(prediction["boxes"]).all()
               for prediction in predictions)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / (
            "cbl_stage2_train_smoke.pt"
            if separate_head else "cbl_iterative_train_smoke.pt")
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
    print(
        f"Iterative CBL "
        f"{'stage-specific' if separate_head else 'shared-head'} "
        "train/inference/reload smoke PASSED"
    )


def main() -> None:
    seed_all(42)
    dataset = build_training_datasets(use_patches=False, is_train=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )
    images, targets = next(iter(loader))
    images = [image.to(DEVICE) for image in images]
    targets = [
        {
            key: value.to(DEVICE) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]
    run_variant(images, targets, separate_head=False)
    run_variant(images, targets, separate_head=True)


if __name__ == "__main__":
    main()
