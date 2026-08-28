from __future__ import annotations
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Robust path discovery for jupyter nbconvert (cwd may differ)
for _candidate in [
    os.environ.get("CPV_ROOT"),
    os.getcwd(),
    str(Path("/root")),
    str(Path(__file__).resolve().parent.parent if "__file__" in dir() else "/root"),
]:
    if _candidate and Path(_candidate).is_dir() and (Path(_candidate) / "common").is_dir():
        if str(Path(_candidate)) not in sys.path:
            sys.path.insert(0, str(Path(_candidate)))
        break

from common.config import (
    DATA_ROOT, TRAIN_DIR, VALID_DIR, TEST_DIR, NUM_CLASSES,
    EPOCHS, LR, MOMENTUM, WEIGHT_DECAY,
    WARMUP_EPOCHS, WARMUP_START_LR, LR_SCHEDULER,
    LR_STEPS, LR_GAMMA,
    EARLY_STOP_PATIENCE, BEST_METRIC, BEST_MODE, MIN_DELTA,
    BATCH_SIZE, NUM_WORKERS, DEVICE, SEED,
    USE_EMA, EVAL_EVERY, EMPTY_CACHE_EVERY,
    USE_TORCH_COMPILE, USE_CHANNELS_LAST, PREFETCH_FACTOR,
    seed_all, make_output_dir,
)
from common.dataset import (
    YOLOTinyDataset, collate_fn,
    build_copy_paste_pool, compute_reliability_threshold,
)
from common.metrics import get_metric_fn, METRIC_DISPLAY_NAME, NEEDS_RELIABILITY
from common.model import build_model
from common.train_utils import ModelEMA, build_optim_sched, train_one_epoch
from common.eval_utils import evaluate

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG (auto-filled by generate_notebooks.py)
# ═══════════════════════════════════════════════════════════════════════
METRIC_NAME = "__METRIC__"
PLACEMENT   = "__PLACEMENT__"
SEED        = __SEED__

print(f"\n{'='*70}\n"
      f"EXPERIMENT: {METRIC_DISPLAY_NAME.get(METRIC_NAME, METRIC_NAME)}  "
      f"@ placement={PLACEMENT}  seed={SEED}\n"
      f"{'='*70}\n")

seed_all(SEED)
OUTPUT_DIR = make_output_dir(METRIC_NAME, PLACEMENT, SEED)
RESUME_CKPT = OUTPUT_DIR / "last.pt"

# Build metric function
metric_fn = get_metric_fn(METRIC_NAME)
reliability_thr = 16.0

# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════
td = YOLOTinyDataset(TRAIN_DIR/"images", TRAIN_DIR/"labels", is_train=True)
vd = YOLOTinyDataset(VALID_DIR/"images", VALID_DIR/"labels", is_train=False)

if METRIC_NAME in NEEDS_RELIABILITY:
    reliability_thr = compute_reliability_threshold(td)
    print(f"  reliability_thr = {reliability_thr:.2f}")

cp_pool = build_copy_paste_pool(td)
if cp_pool:
    td.copy_paste_pool = cp_pool

sam = WeightedRandomSampler(td.get_sample_weights(), len(td), replacement=True)
tl = DataLoader(td, batch_size=BATCH_SIZE, sampler=sam,
                num_workers=NUM_WORKERS, collate_fn=collate_fn,
                pin_memory=(DEVICE.type == "cuda"),
                prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
                drop_last=True, persistent_workers=(NUM_WORKERS > 0))
vl = DataLoader(vd, batch_size=2, shuffle=False,
                num_workers=NUM_WORKERS, collate_fn=collate_fn,
                pin_memory=(DEVICE.type == "cuda"),
                prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
                persistent_workers=(NUM_WORKERS > 0))

# ═══════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════
model = build_model(
    metric_fn=metric_fn if PLACEMENT != "everywhere" else None,
    placement=PLACEMENT,
    reliability_thr=reliability_thr,
    channels_last=USE_CHANNELS_LAST,
).to(DEVICE)

# Optional torch.compile (PyTorch 2.0+ with Triton)
if USE_TORCH_COMPILE:
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print(f"  [speed] torch.compile enabled (mode=reduce-overhead)")
    except Exception as e:
        print(f"  [speed] torch.compile failed: {e}")

# ═══════════════════════════════════════════════════════════════════════
# OPTIMIZER / SCHEDULER / EMA
# ═══════════════════════════════════════════════════════════════════════
opt, scaler, sch = build_optim_sched(
    model, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
    warmup_start_lr=WARMUP_START_LR, warmup_epochs=WARMUP_EPOCHS,
    total_epochs=EPOCHS, scheduler_type=LR_SCHEDULER,
    lr_steps=LR_STEPS, lr_gamma=LR_GAMMA,
)
ema = ModelEMA(model) if USE_EMA else None

# ═══════════════════════════════════════════════════════════════════════
# RESUME
# ═══════════════════════════════════════════════════════════════════════
start = 1
best  = -1.0 if BEST_MODE == "max" else float("inf")
best_epoch = 0
es_counter = 0
hist = []

if RESUME_CKPT.exists():
    ck = torch.load(RESUME_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    scaler.load_state_dict(ck["scaler"])
    if ema and ck.get("ema"):
        ema.load_state_dict(ck["ema"])
    start      = ck.get("epoch", 0) + 1
    best       = ck.get("best_metric_value", best)
    best_epoch = ck.get("best_epoch", 0)
    es_counter = ck.get("es_counter", 0)
    hist       = ck.get("history", [])
    print(f"[Resume] ep={start} | best({BEST_METRIC})={best:.4f} @ ep{best_epoch} | es={es_counter}")

# ═══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════
early_stop = False
train_started_at = time.time()
epoch_times: list = []      # for ETA = running avg × remaining epochs
CSV_PATH = OUTPUT_DIR / "metrics.csv"
CSV_FIELDS = [
    "epoch", "train_loss", "val_loss", "mAP_50", "mAP_primary",
    "AP_micro", "AP_tiny", "AP_small", "AP_large",
    "lr", "seconds",
    "loss_classifier", "loss_box_reg", "loss_objectness",
    "loss_rpn_box_reg", "loss_metric",
]

for epoch in range(start, EPOCHS + 1):
    t0 = time.time()
    tloss, breakdown = train_one_epoch(model, opt, tl, scaler, DEVICE, epoch, ema=ema)
    sch.step_epoch() if hasattr(sch, "step_epoch") else sch.step()
    cur_lr = opt.param_groups[0]["lr"]

    do_eval = (epoch % EVAL_EVERY == 0) or (epoch == EPOCHS)
    eval_model = ema.get_model() if ema else model
    met = (evaluate(eval_model, vl, DEVICE) if do_eval
           else {"mAP_primary": None, "mAP_50": None, "val_loss": None})

    elapsed = time.time() - t0
    epoch_times.append(elapsed)
    # Running average for ETA (smoothes first slow epochs)
    avg_ep = sum(epoch_times) / len(epoch_times)
    remaining_ep = max(EPOCHS - epoch, 0)
    eta_sec = avg_ep * remaining_ep

    # Throughput from train loader: batches * BATCH_SIZE / wall time
    n_train_batches = max(len(tl), 1)
    throughput = (n_train_batches * BATCH_SIZE) / max(elapsed, 1e-6)

    def _fmt_dur(secs: float) -> str:
        secs = max(0.0, secs)
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        if h > 0:
            return f"{h}h {m}m"
        if m > 0:
            return f"{m}m {s}s"
        return f"{s}s"

    hist.append({"epoch": epoch, "train_loss": tloss, "lr": cur_lr,
                 "seconds": elapsed, "loss_breakdown": breakdown, **met})

    # Improved per-epoch header
    wall_elapsed = time.time() - train_started_at
    cur_mAP50 = met.get("mAP_50")
    cur_mAP_str = f"{cur_mAP50:.4f}" if cur_mAP50 is not None else "n/a"
    val_loss = met.get("val_loss")

    print()
    print("=" * 60)
    print(
        f"EPOCH {epoch}/{EPOCHS} | {_fmt_dur(elapsed)} | "
        f"{throughput:.1f}it/s | ETA {_fmt_dur(eta_sec)} | "
        f"wall={_fmt_dur(wall_elapsed)} | LR={cur_lr:.2e}"
    )
    print(
        f"  train_loss: {tloss:.4f} | cls={breakdown['loss_classifier']:.3f} "
        f"box={breakdown['loss_box_reg']:.3f} "
        f"obj={breakdown['loss_objectness']:.3f} "
        f"rpn_box={breakdown['loss_rpn_box_reg']:.3f}"
    )
    if val_loss is not None:
        print(
            f"  val_loss:   {val_loss:.4f} | mAP_50: {cur_mAP_str} "
            f"(best: {best:.4f} @ ep{best_epoch})"
        )
    else:
        print(f"  (eval skipped this epoch)")
    if cur_mAP50 is not None:
        print(
            f"  mAP(scale): {met.get('mAP_primary', 0):.4f} | "
            f"COCO AP50:75: {met.get('coco_AP', 0):.4f} | "
            f"AP50: {met.get('coco_AP50', 0):.4f}"
        )
    print("=" * 60)

    # ── Append to CSV ──────────────────────────────────────────────
    csv_row = {
        "epoch": epoch,
        "train_loss": round(tloss, 6),
        "val_loss": val_loss if val_loss is not None else "",
        "mAP_50": cur_mAP50 if cur_mAP50 is not None else "",
        "mAP_primary": met.get("mAP_primary") if met.get("mAP_primary") is not None else "",
        "AP_micro": met.get("AP_micro", ""),
        "AP_tiny":  met.get("AP_tiny", ""),
        "AP_small": met.get("AP_small", ""),
        "AP_large": met.get("AP_large", ""),
        "lr": cur_lr,
        "seconds": round(elapsed, 2),
        "loss_classifier": round(breakdown.get("loss_classifier", 0.0), 6),
        "loss_box_reg":    round(breakdown.get("loss_box_reg", 0.0), 6),
        "loss_objectness": round(breakdown.get("loss_objectness", 0.0), 6),
        "loss_rpn_box_reg": round(breakdown.get("loss_rpn_box_reg", 0.0), 6),
        "loss_metric":     round(breakdown.get("loss_metric", 0.0), 6),
    }
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(csv_row)

    ck = {
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.state_dict() if ema else None,
        "metrics": met, "best_metric_value": best,
        "best_metric_name": BEST_METRIC, "best_epoch": best_epoch,
        "es_counter": es_counter, "history": hist,
        "config": {"metric": METRIC_NAME, "placement": PLACEMENT,
                   "seed": SEED, "reliability_thr": reliability_thr},
    }
    torch.save(ck, OUTPUT_DIR / "last.pt")

    # ── Best metric + early stopping ───────────────────────────────
    if do_eval and met.get("val_loss") is not None:
        cur_metric = met.get(BEST_METRIC)
        if cur_metric is not None:
            improved = (cur_metric > best + MIN_DELTA) if BEST_MODE == "max" \
                       else (cur_metric < best - MIN_DELTA)
            if improved:
                best = cur_metric
                best_epoch = epoch
                torch.save(ck, OUTPUT_DIR / "best.pt")
                print(f"  ★ New best ({BEST_METRIC}): {best:.4f} @ ep{epoch}")
                es_counter = 0
            else:
                es_counter += 1
                print(f"  ES counter: {es_counter}/{EARLY_STOP_PATIENCE}")

        # Early stopping theo val_loss
        val_losses = [h.get("val_loss") for h in hist
                      if h.get("val_loss") is not None]
        if len(val_losses) >= 2:
            best_so_far = float("inf")
            no_improve = 0
            for vl_ in reversed(val_losses):
                if vl_ < best_so_far - MIN_DELTA:
                    best_so_far = vl_
                    break
                no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stop @ ep{epoch}: val_loss did not improve for "
                      f"{EARLY_STOP_PATIENCE} epochs")
                early_stop = True

    if early_stop:
        break

print(f"\n{'='*70}\n"
      f"DONE: best {BEST_METRIC}={best:.4f} @ ep{best_epoch}\n"
      f"Metrics CSV: {CSV_PATH}\n"
      f"{'='*70}")