"""
Run SA-ALW seed 2024 standalone — overrides config để tránh OOM với RPN=3000.

Giảm COPY_PASTE_MAX_PER=1, RPN_NUM_PROPOSALS_TRAIN=2000.
Chạy: python scripts/run_seed2024.py
"""
import sys, os, csv, time, argparse
from pathlib import Path
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import common.config as cfg

# Override cấu hình an toàn cho seed 2024
cfg.RPN_NUM_PROPOSALS_TRAIN = 2000
cfg.RPN_NUM_PROPOSALS_TEST  = 1000
cfg.COPY_PASTE_MAX_PER      = 1
cfg.COPY_PASTE_PROB         = 0.20

from common.config import (
    EPOCHS, LR, MOMENTUM, WEIGHT_DECAY,
    WARMUP_EPOCHS, WARMUP_START_LR,
    BATCH_SIZE, NUM_WORKERS, DEVICE,
    USE_EMA, seed_all, make_output_dir,
    RPN_NUM_PROPOSALS_TRAIN, RPN_NUM_PROPOSALS_TEST,
    COPY_PASTE_MAX_PER, COPY_PASTE_PROB,
)
from common.dataset import (
    collate_fn, build_training_datasets, build_copy_paste_pool,
    compute_reliability_threshold,
)
from common.metrics import get_metric_fn, NEEDS_RELIABILITY
from common.model import build_model
from common.train_utils import ModelEMA, WarmupCosineLR, train_one_epoch
from common.eval_utils import evaluate

METRIC = "sa_alw_full"
PLACEMENT = "la_loss"
SEED = 2024

OUTPUT_DIR = make_output_dir(METRIC, PLACEMENT, SEED)
seed_all(SEED)

print(f"RPN proposals: train={RPN_NUM_PROPOSALS_TRAIN}, test={RPN_NUM_PROPOSALS_TEST}")
print(f"Copy-paste: prob={COPY_PASTE_PROB}, max_per={COPY_PASTE_MAX_PER}")

train_ds = build_training_datasets(use_patches=False, is_train=True)
val_ds   = build_training_datasets(use_patches=False, is_train=False)
cp_pool  = build_copy_paste_pool(train_ds)
if cp_pool:
    train_ds.copy_paste_pool = cp_pool

reliability_thr = compute_reliability_threshold(train_ds)

from torch.utils.data import DataLoader, WeightedRandomSampler
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

metric_fn = get_metric_fn(METRIC)
model = build_model(
    metric_fn=metric_fn, placement=PLACEMENT,
    reliability_thr=reliability_thr,
).to(DEVICE)

opt = torch.optim.SGD(
    model.parameters(), lr=WARMUP_START_LR,
    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
sched = WarmupCosineLR(
    opt, warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS,
    base_lr=LR, warmup_start_lr=WARMUP_START_LR)
sched.step_epoch()
ema = ModelEMA(model) if USE_EMA else None

best_mAP50 = 0.0
best_epoch = 0

csv_path = OUTPUT_DIR / "metrics.csv"
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
    print(f"  Epoch {epoch}/{EPOCHS} | {elapsed:.1f}s | "
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
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.state_dict() if ema else None,
        "metrics": met, "best_mAP50": best_mAP50,
        "best_epoch": best_epoch,
        "config": {"metric": METRIC, "placement": PLACEMENT, "seed": SEED,
                   "reliability_thr": reliability_thr},
    }, OUTPUT_DIR / "last.pt")

    if mAP50 == best_mAP50:
        import shutil
        shutil.copy2(OUTPUT_DIR / "last.pt", OUTPUT_DIR / "best.pt")

print(f"\nDone. Best mAP@50={best_mAP50:.4f} at epoch {best_epoch}")
