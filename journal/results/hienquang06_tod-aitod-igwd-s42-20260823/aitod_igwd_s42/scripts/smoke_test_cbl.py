"""Focused CPU/CUDA smoke test for confidence-driven RoI localization."""
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
from common.model import (
    _cbl_localization_loss,
    _make_cbl_grid,
    build_model,
)


def main() -> None:
    seed_all(42)

    grid = _make_cbl_grid(alpha=5.0, num_bins=6, beta=1.0)
    assert torch.all(grid[1:] > grid[:-1])
    assert torch.allclose(grid, -grid.flip(0), atol=1e-6)

    logits = torch.randn(8, 4, 6, requires_grad=True)
    targets = torch.empty(8, 4).uniform_(-3.0, 3.0)
    unit_loss = _cbl_localization_loss(
        logits, targets, grid, uncertainty_weight=1.0)
    unit_loss.backward()
    assert torch.isfinite(unit_loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    print(f"CPU unit: loss={float(unit_loss.detach()):.4f}, grid={grid.tolist()}")

    dataset = build_training_datasets(use_patches=False, is_train=True)
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=0,
        collate_fn=collate_fn, pin_memory=(DEVICE.type == "cuda"),
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

    model = build_model(
        metric_fn=get_metric_fn("sa_alw_full"),
        placement="la_loss",
        reliability_thr=16.0,
        box_loss_type="cbl",
        box_loss_warmup_epochs=0,
    ).to(DEVICE)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
        loss_dict = model(images, targets_batch)
        loss = sum(loss_dict.values())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    assert all(torch.isfinite(value) for value in loss_dict.values())
    print(
        f"{DEVICE.type.upper()} batch: total={float(loss.detach()):.4f}, "
        f"box={float(loss_dict['loss_box_reg'].detach()):.4f}"
    )

    model.eval()
    with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=(DEVICE.type == "cuda")):
        predictions = model(images)
    assert len(predictions) == len(images)
    assert all(torch.isfinite(pred["boxes"]).all() for pred in predictions)

    model.roi_heads._cbl_refine_steps = 2
    model.roi_heads._cbl_refine_blend = 0.75
    model.roi_heads._cbl_refine_last_step_blend = 0.5
    model.roi_heads._cbl_refine_last_center_blend = 0.25
    model.roi_heads._cbl_refine_last_size_blend = 0.5
    model.roi_heads._cbl_refine_score_threshold = 0.05
    with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=(DEVICE.type == "cuda")):
        refined_predictions = model(images)
    assert len(refined_predictions) == len(images)
    assert all(
        torch.isfinite(prediction["boxes"]).all()
        for prediction in refined_predictions
    )
    assert all(
        len(prediction["boxes"]) <= model.roi_heads.detections_per_img
        for prediction in refined_predictions
    )
    model.roi_heads._cbl_refine_steps = 0
    model.roi_heads._cbl_refine_blend = 1.0
    model.roi_heads._cbl_refine_last_step_blend = 1.0
    model.roi_heads._cbl_refine_last_center_blend = 1.0
    model.roi_heads._cbl_refine_last_size_blend = 1.0
    model.roi_heads._cbl_refine_score_threshold = 0.0

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "cbl_smoke.pt"
        torch.save({"model": model.state_dict()}, checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location=DEVICE, weights_only=False)
        reloaded = build_model(
            metric_fn=get_metric_fn("sa_alw_full"),
            placement="la_loss",
            reliability_thr=16.0,
            box_loss_type="cbl",
            box_loss_warmup_epochs=0,
        ).to(DEVICE)
        reloaded.load_state_dict(checkpoint["model"])
        reloaded.eval()
        with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=(DEVICE.type == "cuda")):
            reloaded_predictions = reloaded(images)
        assert torch.allclose(
            predictions[0]["boxes"],
            reloaded_predictions[0]["boxes"],
            atol=1e-3,
            rtol=1e-3,
        )
    print("CBL forward/backward/inference/reload smoke PASSED")


if __name__ == "__main__":
    main()
