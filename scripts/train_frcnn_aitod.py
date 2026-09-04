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
    use_amp: bool = True,
    eval_interval: int = 3,
    resume: Path | None = None,
    rpn_cascade: bool = False,
    use_quality_focal: bool = False,
    quality_focal_beta: float = 2.0,
    use_homotopy_roi: bool = False,
    use_egm: bool = False,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = use_amp and (device.type == "cuda")
    if metric == "standard" or not metric:
        if placement == "iou_smooth_l1":
            placement = "everywhere"
        if box_loss == "iou_smooth_l1":
            box_loss = "smooth_l1"
    print(f"=== AI-TOD-v2 Training: Metric={metric}, Placement={placement}, Loss={box_loss}, Seed={seed}, AMP={amp_enabled}, BatchSize={batch_size}, Device={device} ===")

    # 1. Resolve dataset paths
    if data_root is None:
        candidates = [
            Path(r"D:\paper_a_data\AI-TOD-v2\AI-TOD"),
            Path(r"D:\paper_a_data\AI-TOD-v2"),
            Path("/kaggle/input/datasets/simplestzyp/tiny-object-detection-in-aerial-images/AI-TOD"),
            Path("/kaggle/input/tiny-object-detection-in-aerial-images/AI-TOD"),
            Path("/kaggle/input/aitoddatasets/AI-TOD"),
            ROOT / "data/AI-TOD",
        ]
        for c in candidates:
            if c.exists():
                data_root = c
                break
        if data_root is None and Path("/kaggle/input").exists():
            for c in Path("/kaggle/input").glob("**/AI-TOD"):
                if c.is_dir():
                    data_root = c
                    break
            if data_root is None:
                for c in Path("/kaggle/input").glob("*tiny-object*"):
                    if c.is_dir():
                        data_root = c
                        break
        if data_root is None:
            data_root = candidates[0]

    print(f"Data root: {data_root}")
    
    # Auto-discover train / val / test annotations
    train_ann_candidates = [
        data_root / "annotations/train.json",
        data_root / "annotations/aitodv2_train.json",
        data_root / "annotations/aitod_train.json",
        data_root.parent / "annotations/aitodv2_train.json",
    ]
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("*train*.json"):
            train_ann_candidates.append(p)
    train_ann_file = next((p for p in train_ann_candidates if p.is_file()), train_ann_candidates[0])

    val_ann_candidates = [
        data_root / "annotations/val.json",
        data_root / "annotations/aitodv2_val.json",
        data_root / "annotations/aitod_val.json",
        data_root.parent / "annotations/aitodv2_val.json",
    ]
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("*val*.json"):
            val_ann_candidates.append(p)
    val_ann_file = next((p for p in val_ann_candidates if p.is_file()), val_ann_candidates[0])

    # Image dirs
    train_img_candidates = [
        data_root / "AI-TOD/images/train",
        data_root / "images/train",
        data_root / "AI-TOD/images",
        data_root / "images",
        data_root,
    ]
    train_img_dir = next((p for p in train_img_candidates if p.is_dir()), train_img_candidates[0])

    val_img_candidates = [
        data_root / "AI-TOD/images/val",
        data_root / "images/val",
        data_root / "AI-TOD/images",
        data_root / "images",
        data_root,
    ]
    val_img_dir = next((p for p in val_img_candidates if p.is_dir()), val_img_candidates[0])

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
        num_workers=4 if os.name != "nt" else 2,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4 if os.name != "nt" else 2,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 4. Metric & Model configuration
    if metric in ("standard", "rfla") or not metric:
        from common.metrics import iou
        metric_fn, metric_dist_fn, meta = iou.compute_rfd, None, {}
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
        rpn_cascade=rpn_cascade,
        use_quality_focal=use_quality_focal,
        quality_focal_beta=quality_focal_beta,
        use_homotopy_roi_matching=use_homotopy_roi,
        use_egm=use_egm,
    )
    model.to(device)

    # 5. Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    best_map50 = 0.0
    metrics_history = []

    # Resume checkpoint if provided
    if resume is not None and Path(resume).exists():
        print(f"Loading checkpoint for resume: {resume}")
        ckpt = torch.load(resume, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_map50 = ckpt.get("best_map50", 0.0)
            print(f"  -> Resumed from epoch {start_epoch-1}, best AP50={best_map50:.4f}")
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
            print(f"  -> Loaded raw model weights from {resume}")

    # 6. Training Loop
    print("\n=== Beginning Training ===")
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print(f"[Warning] Loss is {losses.item()}, skipping batch")
                continue

            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += losses.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        lr_scheduler.step()
        train_time = time.time() - t0
        print(f"Epoch {epoch:2d}/{epochs:2d} finished | Train Loss = {avg_loss:.4f} | Time = {train_time:.1f}s ({train_time/60:.1f} min)")

        # Save checkpoint after EVERY epoch
        torch.save(model.state_dict(), output_dir / "last.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_map50": best_map50,
            "train_loss": avg_loss,
        }, output_dir / "checkpoint_state.pt")

        if epoch % eval_interval == 0 or epoch == epochs:
            print(f"\n--- Validation at Epoch {epoch} ---")
            model.eval()
            coco_results = []
            with torch.no_grad():
                for images, targets in val_loader:
                    img_tensors = [img.to(device) for img in images]
                    preds = model(img_tensors)
                    for target, pred in zip(targets, preds):
                        img_id = int(target["image_id"])
                        boxes = pred["boxes"].detach().cpu().numpy()
                        scores = pred["scores"].detach().cpu().numpy()
                        labels = pred["labels"].detach().cpu().numpy()
                        for box, score, label in zip(boxes, scores, labels):
                            int_l = int(label)
                            if score > 0.05 and int_l in val_dataset.label_to_category_id:
                                cat_id = val_dataset.label_to_category_id[int_l]
                                x1, y1, x2, y2 = [float(v) for v in box]
                                w = max(x2 - x1, 0.0)
                                h = max(y2 - y1, 0.0)
                                coco_results.append({
                                    "image_id": img_id,
                                    "category_id": cat_id,
                                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                                    "score": round(float(score), 4),
                                })

            if coco_results:
                eval_res = evaluate_aitodv2_official(val_ann_file, coco_results, quiet=True)
                metrics = eval_res.get("metrics", {})
                current_ap50 = metrics.get("AP50", 0.0)
                current_ap = metrics.get("AP", 0.0)
                print(f"Val Epoch {epoch:2d}: AP50 = {current_ap50:.4f}, AP = {current_ap:.4f}")

                metrics_history.append({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "AP": current_ap,
                    "AP50": current_ap50,
                    "AP75": metrics.get("AP75", 0.0),
                    "AP_verytiny": metrics.get("AP_verytiny", 0.0),
                    "AP_tiny": metrics.get("AP_tiny", 0.0),
                    "AP_small": metrics.get("AP_small", 0.0),
                    "AP_medium": metrics.get("AP_medium", 0.0),
                    "AR100": metrics.get("AR100", 0.0),
                })

                if current_ap50 > best_map50:
                    best_map50 = current_ap50
                    torch.save(model.state_dict(), output_dir / "best.pt")
                    print(f"  -> New Best Model saved! (AP50 = {best_map50:.4f})")
            else:
                print("No detections above score threshold")

    # Save metrics JSON
    (output_dir / "metrics.json").write_text(json.dumps(metrics_history, indent=2), encoding="utf-8")
    print(f"\n=== Training Complete! Best AP50 = {best_map50:.4f} ===")


def main():
    parser = argparse.ArgumentParser(description="AI-TOD-v2 Faster R-CNN Training")
    parser.add_argument("--metric", type=str, default="h_wiou",
                        choices=["standard", "nwd", "igwd", "rfla", "alw_canonical", "sa_alw_canonical", "h_wiou", "du_hwiou", "sw_hwiou", "oriented_h_wiou", "eh_wiou"])
    parser.add_argument("--placement", type=str, default="h_wiou",
                        choices=["iou_smooth_l1", "everywhere", "la", "loss", "la_loss", "h_wiou"])
    parser.add_argument("--box-loss", type=str, default="h_wiou",
                        choices=["smooth_l1", "metric", "h_wiou", "eh_wiou"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--h-wiou-sigma-0", type=float, default=8.0)
    parser.add_argument("--h-wiou-form", type=str, default="rational")
    parser.add_argument("--h-wiou-static-gamma", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=3)
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--rpn-cascade", action="store_true", help="Enable multi-stage RPN cascade")
    parser.add_argument("--use-quality-focal", action="store_true", help="Enable Task-Aligned Quality Focal Loss")
    parser.add_argument("--quality-focal-beta", type=float, default=2.0, help="QFL modulating exponent")
    parser.add_argument("--use-homotopy-roi", action="store_true", help="Enable continuous Homotopy-Aware RoI Head Matching")
    parser.add_argument("--use-egm", action="store_true", help="Enable Feature-Level Entropy Guidance Module on FPN P2/P3")
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
        use_amp=not args.no_amp,
        eval_interval=args.eval_interval,
        resume=args.resume,
        rpn_cascade=args.rpn_cascade,
        use_quality_focal=args.use_quality_focal,
        quality_focal_beta=args.quality_focal_beta,
        use_homotopy_roi=args.use_homotopy_roi,
        use_egm=args.use_egm,
    )


if __name__ == "__main__":
    main()
