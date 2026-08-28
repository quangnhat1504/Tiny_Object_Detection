"""
Phase 3: Train with SAALWAssigner (threshold-based RPN assignment).

Usage:
    python scripts/train_saalw.py --metric sa_alw_full --seed 42
    python scripts/train_saalw.py --metric sa_alw_full --pos_sim_thr 0.50 --neg_sim_thr 0.15 --topk 6
    python scripts/train_saalw.py --metric sa_alw_full --grid-search  # run full grid
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import (
    EPOCHS, LR, MOMENTUM, WEIGHT_DECAY,
    WARMUP_EPOCHS, WARMUP_START_LR,
    BATCH_SIZE, NUM_WORKERS, DEVICE,
    USE_EMA, seed_all,
)
from common.dataset import (
    collate_fn, build_training_datasets, build_copy_paste_pool,
    compute_reliability_threshold,
)
from common.metrics import get_metric_fn, NEEDS_RELIABILITY
from common.model import build_model
from common.train_utils import ModelEMA, WarmupCosineLR, train_one_epoch
from common.eval_utils import evaluate


# Grid search space
GRID_POS_THR  = [0.35, 0.40, 0.45, 0.50, 0.55]
GRID_NEG_THR  = [0.15, 0.20, 0.25]
GRID_TOPK     = [3, 6, 9, 12]


def train_one_config(metric_name: str, seed: int, pos_thr: float,
                     neg_thr: float, topk: int, dynamic_thr: bool,
                     tag: str = ""):
    cfg_label = f"p{pos_thr:.0f}_n{neg_thr:.0f}_t{topk}{'_dyn' if dynamic_thr else ''}"
    run_name = f"{metric_name}__saalw_{cfg_label}__seed{seed}"
    if tag:
        run_name += f"__{tag}"
    output_dir = ROOT / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SAALW Assigner — {metric_name}")
    print(f"  pos_thr={pos_thr}, neg_thr={neg_thr}, topk={topk}, dynamic={dynamic_thr}")
    print(f"  Seed: {seed}  |  Output: {output_dir}")
    print(f"{'='*70}\n")

    seed_all(seed)

    train_ds = build_training_datasets(use_patches=False, is_train=True)
    val_ds   = build_training_datasets(use_patches=False, is_train=False)
    cp_pool  = build_copy_paste_pool(train_ds)
    if cp_pool:
        train_ds.copy_paste_pool = cp_pool

    reliability_thr = 16.0
    if metric_name in NEEDS_RELIABILITY:
        reliability_thr = compute_reliability_threshold(train_ds)

    sampler = WeightedRandomSampler(
        train_ds.get_sample_weights(), len(train_ds), replacement=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"), drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=2, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"))

    metric_fn = get_metric_fn(metric_name)
    model = build_model(
        metric_fn=metric_fn,
        placement="saalw_assigner",
        reliability_thr=reliability_thr,
        saalw_rpn_cfg={
            "pos_sim_thr": pos_thr, "neg_sim_thr": neg_thr,
            "topk_fallback": topk, "dynamic_thr": dynamic_thr,
        },
    ).to(DEVICE)

    opt = torch.optim.SGD(model.parameters(), lr=WARMUP_START_LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    sched = WarmupCosineLR(opt, WARMUP_EPOCHS, EPOCHS, LR, WARMUP_START_LR)
    sched.step_epoch()
    ema = ModelEMA(model) if USE_EMA else None

    best_mAP50 = 0.0
    best_epoch = 0
    csv_path = output_dir / "metrics.csv"
    fields = ["epoch", "train_loss", "val_loss", "mAP_50", "mAP_primary",
              "AP_micro", "AP_tiny", "AP_small", "AP_large", "lr", "seconds"]

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tloss, _ = train_one_epoch(model, opt, train_loader, scaler, DEVICE, epoch, ema=ema)
        sched.step_epoch()
        cur_lr = opt.param_groups[0]["lr"]

        eval_model = ema.get_model() if ema else model
        met = evaluate(eval_model, val_loader, DEVICE)
        elapsed = time.time() - t0

        mAP50 = met.get("mAP_50", 0) or 0
        print(f"  Ep {epoch}/{EPOCHS} | {elapsed:.0f}s | "
              f"mAP@50={mAP50:.4f} | best={best_mAP50:.4f} @ ep{best_epoch}")

        row = {
            "epoch": epoch, "train_loss": round(tloss, 6),
            "val_loss": met.get("val_loss", ""),
            "mAP_50": round(mAP50, 6),
            "mAP_primary": met.get("mAP_primary", ""),
            "AP_micro": met.get("AP_micro", ""),
            "AP_tiny": met.get("AP_tiny", ""),
            "AP_small": met.get("AP_small", ""),
            "AP_large": met.get("AP_large", ""),
            "lr": cur_lr, "seconds": round(elapsed, 2),
        }
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerow(row)

        if mAP50 > best_mAP50:
            best_mAP50 = mAP50
            best_epoch = epoch
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": opt.state_dict(), "metrics": met,
                "best_mAP50": best_mAP50,
                "config": {"metric": metric_name, "placement": "saalw_assigner",
                           "seed": seed, "pos_sim_thr": pos_thr,
                           "neg_sim_thr": neg_thr, "topk_fallback": topk,
                           "dynamic_thr": dynamic_thr,
                           "reliability_thr": reliability_thr},
            }, output_dir / "best.pt")

    print(f"\n  BEST: mAP@50={best_mAP50:.4f} @ ep{best_epoch}")
    return {"name": run_name, "best_mAP50": best_mAP50, "best_epoch": best_epoch,
            "pos_thr": pos_thr, "neg_thr": neg_thr, "topk": topk, "dynamic": dynamic_thr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="sa_alw_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos_sim_thr", type=float, default=0.45)
    parser.add_argument("--neg_sim_thr", type=float, default=0.20)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--no-dynamic", action="store_true")
    parser.add_argument("--grid-search", action="store_true")
    args = parser.parse_args()

    if args.grid_search:
        results = []
        total = len(GRID_POS_THR) * len(GRID_NEG_THR) * len(GRID_TOPK) * 2
        i = 0
        for pos in GRID_POS_THR:
            for neg in GRID_NEG_THR:
                for topk in GRID_TOPK:
                    for dyn in [True, False]:
                        i += 1
                        print(f"\n--- Grid [{i}/{total}] ---")
                        r = train_one_config(args.metric, args.seed, pos, neg, topk, dyn)
                        results.append(r)
        # Save grid results
        grid_file = ROOT / "runs" / f"saalw_grid_search__seed{args.seed}.json"
        with open(grid_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nGrid results saved: {grid_file}")
    else:
        train_one_config(args.metric, args.seed, args.pos_sim_thr, args.neg_sim_thr,
                         args.topk, not args.no_dynamic)


if __name__ == "__main__":
    main()
