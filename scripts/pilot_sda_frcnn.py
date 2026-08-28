"""
Pilot & Verification Harness for Scale-Decoupled Adaptive Faster R-CNN (SDA-FRCNN).

Constructs the hybrid model:
  - RPN: ScaleDecoupledAssigner (Standard IoU for s >= 8 px, SA-ALW Dynamic Top-k for s < 8 px)
  - RoI Head: Unbounded Smooth-L1 regression + SA-ALW position weighting
  - Gradient: PC-MR Orthogonal Gradient Projection for micro targets
  - Feature: PC-MOC Cosine Alignment on micro FPN features
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import seed_all, DEVICE
from common.metrics import get_metric_fn
from common.model import build_model


def build_sda_frcnn(seed: int = 42, micro_cutoff_px: float = 8.0) -> torch.nn.Module:
    seed_all(seed)
    metric_fn = get_metric_fn("sa_alw_full")

    model = build_model(
        metric_fn=metric_fn,
        placement="sda_decoupled",
        box_loss_type="smooth_l1",
        saalw_rpn_cfg={
            "micro_cutoff_px": micro_cutoff_px,
            "micro_topk": 4,
            "micro_pos_sim_thr": 0.35,
        },
    )
    return model


def smoke_test_sda_frcnn():
    print("=== Running SDA-FRCNN Structural Smoke Test ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_sda_frcnn(seed=42)
    model.to(device)
    model.train()

    # Create dummy batch
    images = [torch.rand(3, 800, 800, device=device)]
    targets = [{
        "boxes": torch.tensor([
            [10.0, 10.0, 50.0, 50.0],     # standard (40x40)
            [100.0, 100.0, 106.0, 106.0], # micro (6x6)
        ], device=device),
        "labels": torch.tensor([1, 1], dtype=torch.int64, device=device),
    }]

    print("Running forward training pass...")
    loss_dict = model(images, targets)
    total_loss = sum(loss for loss in loss_dict.values())
    print("Losses:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")

    print("Running backward pass...")
    total_loss.backward()
    print("Backward pass completed cleanly without gradient errors.")

    # Check inference pass
    model.eval()
    with torch.no_grad():
        detections = model(images)
    print(f"Inference output keys: {list(detections[0].keys())}")
    print(f"Detected boxes count: {len(detections[0]['boxes'])}")

    print("\n>>> SUCCESS: SDA-FRCNN structural smoke test PASSED 100%! <<<\n")


if __name__ == "__main__":
    smoke_test_sda_frcnn()
