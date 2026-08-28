"""
Phase 1: Faster R-CNN baseline training (standard torchvision).

Usage:
    # Full-image
    python scripts/train_frcnn_baseline.py --seed 42

    # Patches (Phase 0 output)
    python scripts/train_frcnn_baseline.py --use-patches --seed 42

    # Custom seed
    python scripts/train_frcnn_baseline.py --seed 123
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import (
    EPOCHS, LR, MOMENTUM, WEIGHT_DECAY,
    WARMUP_EPOCHS, WARMUP_START_LR,
    BATCH_SIZE, NUM_WORKERS, DEVICE,
    USE_EMA,
    seed_all, make_output_dir,
)
from common.dataset import (
    YOLOTinyDataset, collate_fn, build_training_datasets,
    build_copy_paste_pool,
)
from common.model import build_model
from common.train_utils import ModelEMA, build_optim_sched, train_one_epoch
from common.eval_utils import evaluate

warnings.filterwarnings("ignore")


def train_frcnn(seed: int, use_patches: bool, tag: str = ""):
    tag_suffix = f"__{tag}" if tag else ""
    placement_tag = f"patches{tag_suffix}" if use_patches else f"full{tag_suffix}"
    output_name = f"frcnn_standard__{placement_tag}__seed{seed}"
    OUTPUT_DIR = ROOT / "runs" / output_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"FRCNN BASELINE — standard torchvision")
    print(f"  Dataset: {'patches' if use_patches else 'full-image'}")
    print(f"  Seed: {seed}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    seed_all(seed)

    # ── Data ──
    train_ds = build_training_datasets(use_patches=use_patches, is_train=True)
    val_ds   = build_training_datasets(use_patches=use_patches, is_train=False)
    cp_pool  = build_copy_paste_pool(train_ds)
    if cp_pool:
        train_ds.copy_paste_pool = cp_pool

    sampler = WeightedRandomSampler(
        train_ds.get_sample_weights(), len(train_ds), replacement=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=2, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"))

    # ── Model (standard torchvision — no metric) ──
    model = build_model(metric_fn=None, placement="everywhere").to(DEVICE)

    # ── Optimizer ──
    opt = torch.optim.SGD(
        model.parameters(), lr=WARMUP_START_LR,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    from common.train_utils import WarmupCosineLR
    sched = WarmupCosineLR(
        opt, warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS,
        base_lr=LR, warmup_start_lr=WARMUP_START_LR)
    sched.step_epoch()  # epoch 0

    ema = ModelEMA(model) if USE_EMA else None

    # ── Training loop ──
    best_mAP50 = 0.0
    best_ap75 = 0.0
    best_coco_ap = 0.0
    best_epoch = 0
    best_ap75_epoch = 0
    best_coco_ap_epoch = 0
    csv_path = OUTPUT_DIR / "metrics.csv"
    fields = ["epoch", "train_loss", "val_loss", "mAP_50", "mAP_primary",
              "coco_AP", "coco_AP50", "coco_AP75", "coco_AR100",
              "AP_micro", "AP_tiny", "AP_small", "AP_large",
              "FPS", "Precision", "Recall",
              "lr", "seconds"]

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tloss, breakdown = train_one_epoch(
            model, opt, train_loader, scaler, DEVICE, epoch, ema=ema)
        sched.step_epoch()
        cur_lr = opt.param_groups[0]["lr"]

        # Eval every epoch
        eval_model = ema.get_model() if ema else model
        met = evaluate(eval_model, val_loader, DEVICE, measure_fps_flag=(epoch == EPOCHS))
        elapsed = time.time() - t0

        # Logging
        mAP50 = met.get("mAP_50", 0) or 0
        coco_ap = met.get("coco_AP", 0) or 0
        coco_ap75 = met.get("coco_AP75", 0) or 0
        print(f"  Epoch {epoch}/{EPOCHS} | {elapsed:.1f}s | "
              f"mAP@50={mAP50:.4f} | AP75={coco_ap75:.4f} | "
              f"best75={best_ap75:.4f} @ ep{best_ap75_epoch}")

        # CSV
        row = {
            "epoch": epoch, "train_loss": round(tloss, 6),
            "val_loss": met.get("val_loss", ""),
            "mAP_50": round(mAP50, 6),
            "mAP_primary": met.get("mAP_primary", ""),
            "coco_AP": met.get("coco_AP", ""),
            "coco_AP50": met.get("coco_AP50", ""),
            "coco_AP75": met.get("coco_AP75", ""),
            "coco_AR100": met.get("coco_AR100", ""),
            "AP_micro": met.get("AP_micro", ""),
            "AP_tiny": met.get("AP_tiny", ""),
            "AP_small": met.get("AP_small", ""),
            "AP_large": met.get("AP_large", ""),
            "FPS": met.get("FPS", ""),
            "Precision": met.get("Precision", ""),
            "Recall": met.get("Recall", ""),
            "lr": cur_lr, "seconds": round(elapsed, 2),
        }
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerow(row)

        # Best checkpoint
        if mAP50 > best_mAP50:
            best_mAP50 = mAP50
            best_epoch = epoch
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "metrics": met, "best_mAP50": best_mAP50,
                "best_coco_AP75": best_ap75,
                "best_coco_AP": best_coco_ap,
                "config": {"seed": seed, "use_patches": use_patches},
            }, OUTPUT_DIR / "best.pt")

        if coco_ap75 > best_ap75:
            best_ap75 = coco_ap75
            best_ap75_epoch = epoch
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "metrics": met, "best_mAP50": best_mAP50,
                "best_coco_AP75": best_ap75,
                "best_coco_AP": best_coco_ap,
                "config": {"seed": seed, "use_patches": use_patches},
            }, OUTPUT_DIR / "best_ap75.pt")

        if coco_ap > best_coco_ap:
            best_coco_ap = coco_ap
            best_coco_ap_epoch = epoch
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "metrics": met, "best_mAP50": best_mAP50,
                "best_coco_AP75": best_ap75,
                "best_coco_AP": best_coco_ap,
                "config": {"seed": seed, "use_patches": use_patches},
            }, OUTPUT_DIR / "best_coco_ap.pt")

    # Final summary
    print(f"\n{'='*70}")
    print(f"DONE: best mAP@50 = {best_mAP50:.4f} @ epoch {best_epoch}")
    print(f"      best AP75   = {best_ap75:.4f} @ epoch {best_ap75_epoch}")
    print(f"      best COCO AP= {best_coco_ap:.4f} @ epoch {best_coco_ap_epoch}")
    print(f"Logs: {csv_path}")
    print(f"{'='*70}\n")

    return best_mAP50


def main():
    parser = argparse.ArgumentParser(description="FRCNN baseline training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-patches", action="store_true",
                        help="Train on patches (Phase 0) instead of full-image")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional tag for output dir name")
    args = parser.parse_args()

    train_frcnn(seed=args.seed, use_patches=args.use_patches, tag=args.tag)


if __name__ == "__main__":
    main()
