"""
Faster R-CNN Training & Evaluation Pipeline on AI-TOD-v2 Benchmark.
Supports Standard IoU Baseline, NWD, RFLA, and Homotopy Wasserstein-IoU (H-WIoU).
Evaluates with official AI-TOD-v2 scales: AP, AP50, AP75, AP_vt, AP_t, AP_s, AP_m, AR100, AR1500.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.metrics import configure_metric
from common.model import build_model
from paper_a.datasets.aitodv2_adapter import AITODv2Dataset, AITODV2_CATEGORIES
from paper_a.evaluation.aitodv2_official import evaluate_aitodv2_official


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def train_aitod(
    metric: str = "h_wiou",
    placement: str = "h_wiou",
    box_loss: str = "h_wiou",
    seed: int = 42,
    epochs: int = 12,
    batch_size: int = 4,
    lr: float = 0.005,
    h_wiou_sigma_0: float = 8.0,
    h_wiou_form: str = "rational",
    h_wiou_static_gamma: float = 0.5,
    data_root: Path | None = None,
    output_dir: Path | None = None,
    tag: str = "",
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== AI-TOD-v2 Training: Metric={metric}, Placement={placement}, Loss={box_loss}, Seed={seed}, Device={device} ===")

    # 1. Resolve dataset paths
    if data_root is None:
        # Standard Kaggle dataset path for AI-TOD
        candidates = [
            Path("/kaggle/input/datasets/simplestzyp/tiny-object-detection-in-aerial-images/AI-TOD"),
            Path("/kaggle/input/tiny-object-detection-in-aerial-images/AI-TOD"),
            Path("/kaggle/input/aitoddatasets/AI-TOD"),
            ROOT / "data/AI-TOD",
        ]
        for c in candidates:
            if c.exists():
                data_root = c
                break
        if data_root is None:
            data_root = candidates[0]

    print(f"Data root: {data_root}")
    train_img_dir = data_root / "images/train"
    val_img_dir = data_root / "images/val"
    train_ann_file = data_root / "annotations/train.json"
    val_ann_file = data_root / "annotations/val.json"

    # 2. Output directory
    if output_dir is None:
        tag_str = f"_{tag}" if tag else ""
        output_dir = ROOT / f"runs/aitod_{metric}_{placement}_{box_loss}_s{seed}{tag_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Datasets & Loaders
    print("Loading datasets...")
    train_dataset = AITODv2Dataset(train_img_dir, train_ann_file, drop_empty=True)
    val_dataset = AITODv2Dataset(val_img_dir, val_ann_file, drop_empty=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 4. Metric & Model configuration
    if metric == "standard" or not metric:
        metric_fn, metric_dist_fn, meta = None, None, {}
    else:
        metric_fn, metric_dist_fn, meta = configure_metric(
            metric,
            h_wiou_sigma_0=h_wiou_sigma_0,
            h_wiou_form=h_wiou_form,
            h_wiou_static_gamma=h_wiou_static_gamma,
        )

    print(f"Building Faster R-CNN with 9 classes (8 foreground + background)...")
    model = build_model(
        num_classes=9,
        metric_fn=metric_fn,
        metric_distance_fn=metric_dist_fn,
        placement=placement,
        box_loss_type=box_loss,
    )
    model.to(device)

    # 5. Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 6. Training Loop
    best_map50 = 0.0
    metrics_history = []

    print("\n=== Beginning Training ===")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print(f"[Warning] Loss is {losses.item()}, skipping batch")
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()

            total_loss += losses.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        lr_scheduler.step()
        train_time = time.time() - t0

        # Validation & Official Evaluation
        print(f"\nEvaluating Epoch {epoch}/{epochs} (Train loss: {avg_loss:.4f}, time: {train_time:.1f}s)...")
        model.eval()
        coco_results = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                predictions = model(images)

                for pred, target in zip(predictions, targets):
                    img_id = int(target["image_id"].item() if isinstance(target["image_id"], torch.Tensor) else target["image_id"])
                    boxes = pred["boxes"].cpu().numpy()
                    scores = pred["scores"].cpu().numpy()
                    labels = pred["labels"].cpu().numpy()

                    for b, s, l in zip(boxes, scores, labels):
                        if s < 0.05:
                            continue
                        coco_results.append({
                            "image_id": img_id,
                            "category_id": int(l),
                            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                            "score": float(s),
                        })

        eval_res = {}
        if coco_results:
            try:
                eval_res = evaluate_aitodv2_official(val_ann_file, coco_results, quiet=True)
                metrics = eval_res.get("metrics", {})
                m_ap = metrics.get("AP", 0.0)
                m_ap50 = metrics.get("AP50", 0.0)
                m_ap75 = metrics.get("AP75", 0.0)
                m_ap_vt = metrics.get("AP_verytiny", 0.0)
                m_ap_t = metrics.get("AP_tiny", 0.0)
                m_ap_s = metrics.get("AP_small", 0.0)
                m_ap_m = metrics.get("AP_medium", 0.0)
                m_ar100 = metrics.get("AR100", 0.0)

                print(f"Epoch {epoch:2d}/{epochs:2d} | AP={m_ap:.4f} | AP50={m_ap50:.4f} | AP75={m_ap75:.4f} | AP_vt={m_ap_vt:.4f} | AP_t={m_ap_t:.4f} | AP_s={m_ap_s:.4f} | AR100={m_ar100:.4f}")

                record = {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "AP": m_ap,
                    "AP50": m_ap50,
                    "AP75": m_ap75,
                    "AP_verytiny": m_ap_vt,
                    "AP_tiny": m_ap_t,
                    "AP_small": m_ap_s,
                    "AP_medium": m_ap_m,
                    "AR100": m_ar100,
                }
                metrics_history.append(record)

                if m_ap50 > best_map50:
                    best_map50 = m_ap50
                    torch.save(model.state_dict(), output_dir / "best.pt")
                    print(f"  --> Saved new best checkpoint with AP50={best_map50:.4f}")

            except Exception as e:
                print(f"[Error evaluating AI-TOD]: {e}")
        else:
            print("No detections above score threshold")

        # Save last checkpoint
        torch.save(model.state_dict(), output_dir / "last.pt")

    # Save metrics JSON
    (output_dir / "metrics.json").write_text(json.dumps(metrics_history, indent=2), encoding="utf-8")
    print(f"\n=== Training Complete! Best AP50 = {best_map50:.4f} ===")


def main():
    parser = argparse.ArgumentParser(description="AI-TOD-v2 Faster R-CNN Training")
    parser.add_argument("--metric", type=str, default="h_wiou",
                        choices=["standard", "nwd", "rfla", "alw_canonical", "sa_alw_canonical", "h_wiou"])
    parser.add_argument("--placement", type=str, default="h_wiou",
                        choices=["iou_smooth_l1", "la", "loss", "la_loss", "h_wiou"])
    parser.add_argument("--box-loss", type=str, default="h_wiou",
                        choices=["smooth_l1", "metric", "h_wiou"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--h-wiou-sigma-0", type=float, default=8.0)
    parser.add_argument("--h-wiou-form", type=str, default="rational")
    parser.add_argument("--h-wiou-static-gamma", type=float, default=0.5)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    train_aitod(
        metric=args.metric,
        placement=args.placement,
        box_loss=args.box_loss,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        h_wiou_sigma_0=args.h_wiou_sigma_0,
        h_wiou_form=args.h_wiou_form,
        h_wiou_static_gamma=args.h_wiou_static_gamma,
        data_root=args.data_root,
        output_dir=args.output_dir,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
