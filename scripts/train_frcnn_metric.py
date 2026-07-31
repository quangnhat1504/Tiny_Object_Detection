"""
Phase 2: Faster R-CNN with metric-based RPN + loss (la/la_loss/la_loss_nms).

Usage:
    python scripts/train_frcnn_metric.py --metric nwd --placement la_loss --seed 42
    python scripts/train_frcnn_metric.py --metric alw_full --placement la_loss --seed 123
    python scripts/train_frcnn_metric.py --metric sa_alw_full --placement la_loss --seed 2024
"""
from __future__ import annotations
import argparse
import csv
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
    USE_EMA, BOX_LOSS_WARMUP_EPOCHS,
    CBL_ALPHA, CBL_NUM_BINS, CBL_GRID_BETA, CBL_UM_WEIGHT,
    MIN_SIZE, MAX_SIZE,
    seed_all, make_output_dir,
)
from common.dataset import (
    collate_fn, build_training_datasets, build_copy_paste_pool,
    compute_reliability_threshold,
)
from common.metrics import get_metric_fn, NEEDS_RELIABILITY
from common.model import build_model
from common.train_utils import ModelEMA, WarmupCosineLR, train_one_epoch
from common.eval_utils import evaluate


def _select_evaluation_model(model, ema):
    """Return the exact model used for validation and its checkpoint label."""
    if ema is None:
        return model, "raw"
    return ema.get_model(), "ema"


def _set_transform_sizes(model, min_sizes, max_size):
    model.transform.min_size = tuple(int(size) for size in min_sizes)
    model.transform.max_size = int(max_size)


def train_metric(metric: str, placement: str, seed: int, resume: bool = False,
                 box_loss: str = "metric", tag: str = "",
                 box_loss_warmup_epochs: int | None = None,
                 quality_score: bool = False,
                 quality_loss_weight: float = 0.5,
                 quality_focal: bool = False,
                 quality_focal_beta: float = 2.0,
                 rank_sort: bool = False,
                 rank_sort_delta: float = 0.5,
                 double_head: bool = False,
                 double_head_reg_roi_scale: float = 1.3,
                 double_head_num_convs: int = 4,
                 cbl_refine_train_weight: float = 0.0,
                 cbl_refine_steps: int = 0,
                 cbl_refine_blend: float = 1.0,
                 cbl_refine_last_step_blend: float | None = None,
                 cbl_refine_last_center_blend: float | None = None,
                 cbl_refine_last_size_blend: float | None = None,
                 cbl_refine_score_threshold: float = 0.0,
                 cbl_refine_extra_min_size_ratio: float = 0.0,
                 cbl_alpha: float = CBL_ALPHA,
                 cbl_num_bins: int = CBL_NUM_BINS,
                 cbl_grid_beta: float = CBL_GRID_BETA,
                 cbl_um_weight: float = CBL_UM_WEIGHT,
                 train_min_sizes: tuple[int, ...] | None = None,
                 train_max_size: int | None = None):
    train_min_sizes = tuple(train_min_sizes or (MIN_SIZE,))
    if any(size <= 0 for size in train_min_sizes):
        raise ValueError("Training minimum sizes must be positive")
    if train_max_size is None:
        scale_ratio = max(train_min_sizes) / MIN_SIZE
        train_max_size = max(MAX_SIZE, round(MAX_SIZE * scale_ratio))
    if train_max_size < max(train_min_sizes):
        raise ValueError(
            "Training max size must be at least the largest minimum size")

    metric_name = metric if box_loss == "metric" else f"{metric}__{box_loss}"
    if quality_score:
        metric_name = f"{metric_name}__q{quality_loss_weight:g}"
    if quality_focal:
        metric_name = f"{metric_name}__qflb{quality_focal_beta:g}"
    if rank_sort:
        metric_name = f"{metric_name}__rsd{rank_sort_delta:g}"
    if double_head:
        metric_name = (
            f"{metric_name}__dh{double_head_num_convs}"
            f"s{double_head_reg_roi_scale:g}"
        )
    if cbl_refine_train_weight > 0:
        metric_name = (
            f"{metric_name}__irtw{cbl_refine_train_weight:g}"
            f"ir{cbl_refine_steps}s{cbl_refine_score_threshold:g}"
        )
        if cbl_refine_extra_min_size_ratio > 0:
            metric_name += f"m{cbl_refine_extra_min_size_ratio:g}"
    output_name = f"{metric_name}__{placement}__seed{seed}"
    if tag:
        output_name = f"{output_name}__{tag}"
    OUTPUT_DIR = ROOT / "runs" / output_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"METRIC ABLATION — {metric} @ {placement}")
    print(f"  Seed: {seed}")
    print(f"  Box loss: {box_loss}")
    if box_loss_warmup_epochs is not None:
        print(f"  Box loss warmup epochs: {box_loss_warmup_epochs}")
    print(f"  Quality score: {quality_score} (weight={quality_loss_weight:g})")
    print(f"  Quality focal: {quality_focal} (beta={quality_focal_beta:g})")
    print(f"  Rank & Sort: {rank_sort} (delta={rank_sort_delta:g})")
    print(
        f"  Double-Head: {double_head} "
        f"(scale={double_head_reg_roi_scale:g}, "
        f"bottlenecks={double_head_num_convs})"
    )
    print(
        f"  CBL iterative train: weight={cbl_refine_train_weight:g}; "
        f"inference steps={cbl_refine_steps}, "
        f"blend={cbl_refine_blend:g}, "
        f"last step blend={cbl_refine_last_step_blend}, "
        f"last center blend={cbl_refine_last_center_blend}, "
        f"last size blend={cbl_refine_last_size_blend}, "
        f"score threshold={cbl_refine_score_threshold:g}, "
        f"extra min size ratio={cbl_refine_extra_min_size_ratio:g}"
    )
    print(
        f"  Transform: train min={train_min_sizes}, "
        f"train max={train_max_size}; eval={MIN_SIZE}/{MAX_SIZE}"
    )
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Resume: {resume}")
    print(f"{'='*70}\n")

    seed_all(seed)

    # ── Data (always full-image for Phase 2) ──
    train_ds = build_training_datasets(use_patches=False, is_train=True)
    val_ds   = build_training_datasets(use_patches=False, is_train=False)
    cp_pool  = build_copy_paste_pool(train_ds)
    if cp_pool:
        train_ds.copy_paste_pool = cp_pool

    # Reliability threshold (for ALW/SA-ALW)
    reliability_thr = 16.0
    if metric in NEEDS_RELIABILITY:
        reliability_thr = compute_reliability_threshold(train_ds)
        print(f"  reliability_thr = {reliability_thr:.2f}")

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

    # ── Model ──
    metric_fn = get_metric_fn(metric)
    model = build_model(
        metric_fn=metric_fn,
        placement=placement,
        reliability_thr=reliability_thr,
        box_loss_type=box_loss,
        box_loss_warmup_epochs=(
            BOX_LOSS_WARMUP_EPOCHS if box_loss_warmup_epochs is None
            else box_loss_warmup_epochs
        ),
        use_quality_score=quality_score,
        quality_loss_weight=quality_loss_weight,
        use_quality_focal=quality_focal,
        quality_focal_beta=quality_focal_beta,
        use_rank_sort=rank_sort,
        rank_sort_delta=rank_sort_delta,
        use_double_head=double_head,
        double_head_reg_roi_scale=double_head_reg_roi_scale,
        double_head_num_convs=double_head_num_convs,
        cbl_refine_train_weight=cbl_refine_train_weight,
        cbl_refine_steps=cbl_refine_steps,
        cbl_refine_blend=cbl_refine_blend,
        cbl_refine_last_step_blend=cbl_refine_last_step_blend,
        cbl_refine_last_center_blend=cbl_refine_last_center_blend,
        cbl_refine_last_size_blend=cbl_refine_last_size_blend,
        cbl_refine_score_threshold=cbl_refine_score_threshold,
        cbl_refine_extra_min_size_ratio=cbl_refine_extra_min_size_ratio,
        cbl_alpha=cbl_alpha,
        cbl_num_bins=cbl_num_bins,
        cbl_grid_beta=cbl_grid_beta,
        cbl_um_weight=cbl_um_weight,
        transform_min_sizes=train_min_sizes,
        transform_max_size=train_max_size,
    ).to(DEVICE)

    # ── Optimizer ──
    opt = torch.optim.SGD(
        model.parameters(), lr=WARMUP_START_LR,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    sched = WarmupCosineLR(
        opt, warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS,
        base_lr=LR, warmup_start_lr=WARMUP_START_LR)
    sched.step_epoch()

    ema = ModelEMA(model) if USE_EMA else None

    # ── Resume checkpoint ──
    best_mAP50 = 0.0
    best_ap75 = 0.0
    best_coco_ap = 0.0
    best_epoch = 0
    best_ap75_epoch = 0
    best_coco_ap_epoch = 0
    start_epoch = 1
    history = []

    ckpt_path = OUTPUT_DIR / "last.pt"
    if resume and ckpt_path.exists():
        print(f"[RESUME] Loading checkpoint: {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        if ema and ck.get("ema"):
            ema.load_state_dict(ck["ema"])
        sched.set_epoch(ck.get("epoch", 0) + 1)
        best_mAP50 = ck.get("best_mAP50", 0.0)
        best_ap75 = ck.get("best_coco_AP75", ck.get("best_ap75", 0.0))
        best_coco_ap = ck.get("best_coco_AP", 0.0)
        best_epoch = ck.get("best_epoch", 0)
        best_ap75_epoch = ck.get("best_ap75_epoch", 0)
        best_coco_ap_epoch = ck.get("best_coco_ap_epoch", 0)
        history = ck.get("history", [])
        start_epoch = ck.get("epoch", 0) + 1
        print(f"[RESUME] Resuming at epoch {start_epoch}, "
              f"best mAP@50={best_mAP50:.4f} @ ep{best_epoch}, "
              f"best AP75={best_ap75:.4f} @ ep{best_ap75_epoch}")
    elif resume:
        print(f"[RESUME] No checkpoint found at {ckpt_path}, starting fresh.")

    csv_path = OUTPUT_DIR / "metrics.csv"
    fields = ["epoch", "train_loss", "val_loss", "mAP_50", "mAP_primary",
              "coco_AP", "coco_AP50", "coco_AP75", "coco_AR100",
              "AP_micro", "AP_tiny", "AP_small", "AP_large", "lr", "seconds"]
    effective_box_loss_warmup_epochs = (
        BOX_LOSS_WARMUP_EPOCHS if box_loss_warmup_epochs is None
        else box_loss_warmup_epochs
    )
    run_config = {
        "metric": metric,
        "placement": placement,
        "seed": seed,
        "box_loss": box_loss,
        "box_loss_warmup_epochs": effective_box_loss_warmup_epochs,
        "tag": tag,
        "reliability_thr": reliability_thr,
        "quality_score": quality_score,
        "quality_loss_weight": quality_loss_weight,
        "quality_focal": quality_focal,
        "quality_focal_beta": quality_focal_beta,
        "rank_sort": rank_sort,
        "rank_sort_delta": rank_sort_delta,
        "double_head": double_head,
        "double_head_reg_roi_scale": double_head_reg_roi_scale,
        "double_head_num_convs": double_head_num_convs,
        "cbl_refine_train_weight": cbl_refine_train_weight,
        "cbl_refine_steps": cbl_refine_steps,
        "cbl_refine_blend": cbl_refine_blend,
        "cbl_refine_last_step_blend": cbl_refine_last_step_blend,
        "cbl_refine_last_center_blend": cbl_refine_last_center_blend,
        "cbl_refine_last_size_blend": cbl_refine_last_size_blend,
        "cbl_refine_score_threshold": cbl_refine_score_threshold,
        "cbl_refine_extra_min_size_ratio": (
            cbl_refine_extra_min_size_ratio
        ),
        "cbl_alpha": cbl_alpha,
        "cbl_num_bins": cbl_num_bins,
        "cbl_grid_beta": cbl_grid_beta,
        "cbl_um_weight": cbl_um_weight,
        "train_min_sizes": list(train_min_sizes),
        "train_max_size": train_max_size,
        "eval_min_size": MIN_SIZE,
        "eval_max_size": MAX_SIZE,
        "use_ema": USE_EMA,
    }

    for epoch in range(start_epoch, EPOCHS + 1):
        _set_transform_sizes(model, train_min_sizes, train_max_size)
        # Set current epoch on model for box loss warmup
        if hasattr(model, 'roi_heads'):
            model.roi_heads._current_epoch = epoch
        t0 = time.time()
        tloss, breakdown = train_one_epoch(
            model, opt, train_loader, scaler, DEVICE, epoch, ema=ema)
        sched.step_epoch()
        cur_lr = opt.param_groups[0]["lr"]

        eval_model, eval_model_source = _select_evaluation_model(model, ema)
        _set_transform_sizes(eval_model, (MIN_SIZE,), MAX_SIZE)
        met = evaluate(eval_model, val_loader, DEVICE, measure_fps_flag=(epoch == EPOCHS))
        elapsed = time.time() - t0

        mAP50 = met.get("mAP_50", 0) or 0
        coco_ap = met.get("coco_AP", 0) or 0
        coco_ap75 = met.get("coco_AP75", 0) or 0
        print(f"  Epoch {epoch}/{EPOCHS} | {elapsed:.1f}s | "
              f"mAP@50={mAP50:.4f} | AP75={coco_ap75:.4f} | "
              f"best75={best_ap75:.4f} @ ep{best_ap75_epoch}")

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
        if coco_ap75 > best_ap75:
            best_ap75 = coco_ap75
            best_ap75_epoch = epoch
        if coco_ap > best_coco_ap:
            best_coco_ap = coco_ap
            best_coco_ap_epoch = epoch

        history.append(row)

        # always save last.pt for resume
        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "model_source": "raw",
            "eval_model_source": eval_model_source,
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.state_dict() if ema else None,
            "metrics": met, "best_mAP50": best_mAP50,
            "best_coco_AP75": best_ap75,
            "best_coco_AP": best_coco_ap,
            "best_epoch": best_epoch,
            "best_ap75_epoch": best_ap75_epoch,
            "best_coco_ap_epoch": best_coco_ap_epoch,
            "history": history,
            "config": run_config,
        }, OUTPUT_DIR / "last.pt")

        if mAP50 == best_mAP50 and epoch == best_epoch:
            torch.save({
                "epoch": epoch, "model": eval_model.state_dict(),
                "model_source": eval_model_source,
                "optimizer": opt.state_dict(),
                "metrics": met, "best_mAP50": best_mAP50,
                "best_coco_AP75": best_ap75,
                "best_coco_AP": best_coco_ap,
                "config": run_config,
            }, OUTPUT_DIR / "best.pt")

        if coco_ap75 == best_ap75 and epoch == best_ap75_epoch:
            torch.save({
                "epoch": epoch, "model": eval_model.state_dict(),
                "model_source": eval_model_source,
                "optimizer": opt.state_dict(),
                "metrics": met, "best_mAP50": best_mAP50,
                "best_coco_AP75": best_ap75,
                "best_coco_AP": best_coco_ap,
                "config": run_config,
            }, OUTPUT_DIR / "best_ap75.pt")

        if coco_ap == best_coco_ap and epoch == best_coco_ap_epoch:
            torch.save({
                "epoch": epoch, "model": eval_model.state_dict(),
                "model_source": eval_model_source,
                "optimizer": opt.state_dict(),
                "metrics": met, "best_mAP50": best_mAP50,
                "best_coco_AP75": best_ap75,
                "best_coco_AP": best_coco_ap,
                "config": run_config,
            }, OUTPUT_DIR / "best_coco_ap.pt")

    print(f"\n{'='*70}")
    print(f"DONE: best mAP@50 = {best_mAP50:.4f} @ epoch {best_epoch}")
    print(f"      best AP75   = {best_ap75:.4f} @ epoch {best_ap75_epoch}")
    print(f"      best COCO AP= {best_coco_ap:.4f} @ epoch {best_coco_ap_epoch}")
    print(f"Logs: {csv_path}")
    print(f"{'='*70}\n")

    return best_mAP50


def main():
    parser = argparse.ArgumentParser(description="FRCNN metric ablation")
    parser.add_argument("--metric", type=str, required=True,
                        choices=["nwd", "igwd", "igwd_log_shape", "igwd_anisotropic_s",
                                 "alw_full", "sa_alw_beta_only", "sa_alw_full",
                                 "sa_alw_pos_only"],
                        help="Metric name")
    parser.add_argument("--placement", type=str, default="la_loss",
                        choices=["la", "la_loss", "la_loss_nms"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last.pt checkpoint")
    parser.add_argument("--box-loss", type=str, default="metric",
                        choices=["metric", "smooth_l1", "side_smooth_l1", "ciou", "diou", "cbl"],
                        help="Box regression loss type (decoupled from metric)")
    parser.add_argument("--box-loss-warmup-epochs", type=int, default=None,
                        help="Override metric-loss warmup epochs before decoupled box loss")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix for output dir")
    parser.add_argument("--quality-score", action="store_true",
                        help="Enable auxiliary RoI localization-quality head")
    parser.add_argument("--quality-loss-weight", type=float, default=0.5,
                        help="Weight for quality IoU target loss")
    parser.add_argument("--quality-focal", action="store_true",
                        help="Train a joint class-IoU score with Quality Focal Loss")
    parser.add_argument("--quality-focal-beta", type=float, default=2.0,
                        help="Quality Focal Loss modulating exponent")
    parser.add_argument("--rank-sort", action="store_true",
                        help="Train sampled RoI classification with Rank & Sort loss")
    parser.add_argument("--rank-sort-delta", type=float, default=0.5,
                        help="Rank & Sort comparison smoothing width")
    parser.add_argument("--double-head", action="store_true",
                        help="Use a convolutional CBL box-regression head")
    parser.add_argument("--double-head-reg-roi-scale", type=float, default=1.3,
                        help="Proposal scale used for Double-Head regression")
    parser.add_argument("--double-head-num-convs", type=int, default=4,
                        help="Residual bottlenecks in Double-Head regression")
    parser.add_argument("--cbl-refine-train-weight", type=float, default=0.0,
                        help="Auxiliary shared-head second-pass CBL loss weight")
    parser.add_argument("--cbl-refine-steps", type=int, default=0,
                        help="Inference-time repeated CBL regression passes")
    parser.add_argument("--cbl-refine-blend", type=float, default=1.0,
                        help="Fraction of each inference refinement update")
    parser.add_argument(
        "--cbl-refine-last-step-blend",
        type=float,
        default=None,
        help="Final inference update fraction; defaults to the shared blend",
    )
    parser.add_argument(
        "--cbl-refine-last-center-blend",
        type=float,
        default=None,
        help="Final center-update fraction; defaults to final step blend",
    )
    parser.add_argument(
        "--cbl-refine-last-size-blend",
        type=float,
        default=None,
        help="Final width/height-update fraction; defaults to final step blend",
    )
    parser.add_argument("--cbl-refine-score-threshold", type=float, default=0.0,
                        help="Preserve detections below this score")
    parser.add_argument(
        "--cbl-refine-extra-min-size-ratio",
        type=float,
        default=0.0,
        help=(
            "After pass one, refine only boxes at or above this normalized "
            "sqrt-area size; zero disables the gate"
        ),
    )
    parser.add_argument("--cbl-alpha", type=float, default=CBL_ALPHA,
                        help="CBL normalized delta range")
    parser.add_argument("--cbl-num-bins", type=int, default=CBL_NUM_BINS,
                        help="CBL distribution bins per coordinate")
    parser.add_argument("--cbl-grid-beta", type=float, default=CBL_GRID_BETA,
                        help="CBL interval-nonuniform grid density")
    parser.add_argument("--cbl-um-weight", type=float, default=CBL_UM_WEIGHT,
                        help="CBL uncertainty matching loss weight")
    parser.add_argument(
        "--train-min-sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Stochastic shorter-side resize choices during training; "
            "validation remains fixed at the project default"
        ),
    )
    parser.add_argument(
        "--train-max-size",
        type=int,
        default=None,
        help=(
            "Training maximum image side; defaults to the project aspect "
            "ratio scaled to the largest training minimum size"
        ),
    )
    args = parser.parse_args()

    train_metric(args.metric, args.placement, args.seed, args.resume,
                 box_loss=args.box_loss, tag=args.tag,
                 box_loss_warmup_epochs=args.box_loss_warmup_epochs,
                 quality_score=args.quality_score,
                 quality_loss_weight=args.quality_loss_weight,
                 quality_focal=args.quality_focal,
                 quality_focal_beta=args.quality_focal_beta,
                 rank_sort=args.rank_sort,
                 rank_sort_delta=args.rank_sort_delta,
                 double_head=args.double_head,
                 double_head_reg_roi_scale=args.double_head_reg_roi_scale,
                 double_head_num_convs=args.double_head_num_convs,
                 cbl_refine_train_weight=args.cbl_refine_train_weight,
                 cbl_refine_steps=args.cbl_refine_steps,
                 cbl_refine_blend=args.cbl_refine_blend,
                 cbl_refine_last_step_blend=(
                     args.cbl_refine_last_step_blend),
                 cbl_refine_last_center_blend=(
                     args.cbl_refine_last_center_blend),
                 cbl_refine_last_size_blend=(
                     args.cbl_refine_last_size_blend),
                 cbl_refine_score_threshold=(
                     args.cbl_refine_score_threshold),
                 cbl_refine_extra_min_size_ratio=(
                     args.cbl_refine_extra_min_size_ratio),
                 cbl_alpha=args.cbl_alpha,
                 cbl_num_bins=args.cbl_num_bins,
                 cbl_grid_beta=args.cbl_grid_beta,
                 cbl_um_weight=args.cbl_um_weight,
                 train_min_sizes=(
                     tuple(args.train_min_sizes)
                     if args.train_min_sizes is not None
                     else None
                 ),
                 train_max_size=args.train_max_size)


if __name__ == "__main__":
    main()
