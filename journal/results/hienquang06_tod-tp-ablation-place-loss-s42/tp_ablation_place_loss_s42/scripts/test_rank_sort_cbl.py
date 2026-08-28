"""Focused gradient and model smoke tests for sampled RoI Rank & Sort with CBL."""
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
from common.model import _RankSortFunction, _rank_sort_loss, build_model


def _test_rank_sort_gradients() -> None:
    logits = torch.tensor(
        [
            [0.0, 2.0, -3.0],
            [0.0, 1.5, -3.0],
            [0.0, 0.5, -3.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 1, 1])
    quality = torch.tensor([0.0, 0.9, 0.3])
    loss = _rank_sort_loss(logits, labels, quality, delta=0.5)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1] > 0
    assert logits.grad[1:, 1].sum() < 0
    assert logits.grad[:, 0].abs().sum() == 0
    assert logits.grad[:, 2].abs().sum() == 0

    ordered_logits = torch.tensor([2.0, 1.0], requires_grad=True)
    reversed_logits = torch.tensor([1.0, 2.0], requires_grad=True)
    targets = torch.tensor([0.9, 0.3])
    ordered_rank, ordered_sort = _RankSortFunction.apply(
        ordered_logits, targets, 0.5
    )
    reversed_rank, reversed_sort = _RankSortFunction.apply(
        reversed_logits, targets, 0.5
    )
    assert torch.allclose(ordered_rank, torch.zeros_like(ordered_rank))
    assert torch.allclose(reversed_rank, torch.zeros_like(reversed_rank))
    assert ordered_sort.abs() < 1e-7
    assert reversed_sort > ordered_sort
    print(
        "RankSort gradient/order: "
        f"loss={float(loss.detach()):.6f}, "
        f"reversed_sort={float(reversed_sort.detach()):.6f}"
    )


def main() -> None:
    seed_all(42)
    _test_rank_sort_gradients()

    dataset = build_training_datasets(use_patches=False, is_train=True)
    loader = DataLoader(
        dataset,
        batch_size=2,
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

    build_kwargs = {
        "metric_fn": get_metric_fn("sa_alw_full"),
        "placement": "la_loss",
        "reliability_thr": 16.0,
        "box_loss_type": "cbl",
        "box_loss_warmup_epochs": 0,
        "use_rank_sort": True,
        "rank_sort_delta": 0.5,
    }
    model = build_model(**build_kwargs).to(DEVICE)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
        loss_dict = model(images, targets_batch)
        total_loss = sum(loss_dict.values())
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    assert all(torch.isfinite(value) for value in loss_dict.values())
    assert "loss_quality" not in loss_dict
    print(
        f"{DEVICE.type.upper()} batch: total={float(total_loss.detach()):.4f}, "
        f"rank_sort={float(loss_dict['loss_classifier'].detach()):.4f}, "
        f"box={float(loss_dict['loss_box_reg'].detach()):.4f}"
    )

    model.eval()
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=(DEVICE.type == "cuda")
    ):
        predictions = model(images)
    assert len(predictions) == len(images)
    assert all(torch.isfinite(pred["scores"]).all() for pred in predictions)
    assert all(
        ((pred["scores"] >= 0) & (pred["scores"] <= 1)).all()
        for pred in predictions
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "cbl_rank_sort_smoke.pt"
        torch.save({"model": model.state_dict()}, checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location=DEVICE, weights_only=False
        )
        reloaded = build_model(**build_kwargs).to(DEVICE)
        reloaded.load_state_dict(checkpoint["model"])
        reloaded.eval()
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=(DEVICE.type == "cuda")
        ):
            reloaded_predictions = reloaded(images)
        assert torch.allclose(
            predictions[0]["scores"],
            reloaded_predictions[0]["scores"],
            atol=1e-4,
            rtol=1e-4,
        )
    print("CBL+RankSort forward/backward/inference/reload smoke PASSED")


if __name__ == "__main__":
    main()
