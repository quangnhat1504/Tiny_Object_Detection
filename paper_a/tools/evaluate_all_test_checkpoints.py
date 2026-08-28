"""
Master Test-Set Inference & Comprehensive Metric Evaluator for All Downloaded Checkpoints.
Runs full evaluation on NVIDIA GPU with COCO & TOD scale-aware metrics.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import DEVICE, seed_all, SEED
from common.dataset import YOLOTinyDataset, collate_fn
from common.metrics import get_metric_fn
from common.model import build_model
from common.eval_utils import evaluate

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "journal/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


CHECKPOINT_REGISTRY = [
    # 1. AI-TOD & Cascade Runs
    {
        "id": "faster_rcnn_baseline",
        "name": "Faster R-CNN Baseline (AI-TOD/TP)",
        "group": "Baselines",
        "path": RESULTS_DIR / "amongus1504_tod-aitod-baseline-s42-20260823/aitod_baseline_s42/runs/aitod_standard_everywhere_smooth_l1_s42/last.pt",
        "default_metric": "iou",
        "placement": "everywhere",
    },
    {
        "id": "cascade_rcnn_baseline",
        "name": "Cascade R-CNN Baseline",
        "group": "Baselines",
        "path": RESULTS_DIR / "hngtrngtn_tod-aitod-cascade-s42-20260824/aitod_cascade_s42/runs/aitod_standard_everywhere_smooth_l1_s42_cascade_rcnn/last.pt",
        "default_metric": "iou",
        "placement": "everywhere",
    },
    # 2. SOTA Baselines
    {
        "id": "nwd_sota",
        "name": "NWD (Wasserstein Distance)",
        "group": "SOTA Benchmarks",
        "path": ROOT / "runs/nwd__la_loss__seed42/best.pt",
        "default_metric": "nwd",
        "placement": "la_loss",
    },
    {
        "id": "igwd_sota",
        "name": "IGWD (Gaussian Wasserstein)",
        "group": "SOTA Benchmarks",
        "path": ROOT / "runs/igwd__la_loss__seed42/best.pt",
        "default_metric": "igwd",
        "placement": "la_loss",
    },
    {
        "id": "sa_alw_canonical",
        "name": "SA-ALW (Scale-Adaptive Anisotropic)",
        "group": "SOTA Benchmarks",
        "path": ROOT / "runs/sa_alw_full__la_loss__seed42/best.pt",
        "default_metric": "sa_alw_full",
        "placement": "la_loss",
    },
    # 3. Core Homotopy H-WIoU Ablations
    {
        "id": "h_wiou_pure_w2",
        "name": "H-WIoU Ablation: Pure W2 (gamma=0)",
        "group": "H-WIoU Ablations",
        "path": RESULTS_DIR / "thyngluthy_tod-tp-ablation-pure-w2-s42/tp_ablation_pure_w2_s42/runs/h_wiou__h_wiou__pure_w2__h_wiou__seed42__ablation_pure_w2/best.pt",
        "default_metric": "h_wiou",
        "placement": "la_loss",
    },
    {
        "id": "h_wiou_pure_iou",
        "name": "H-WIoU Ablation: Pure IoU (gamma=1)",
        "group": "H-WIoU Ablations",
        "path": RESULTS_DIR / "hienquang06_tod-tp-ablation-pure-iou-s42/tp_ablation_pure_iou_s42/runs/h_wiou__h_wiou__pure_iou__h_wiou__seed42__ablation_pure_iou/best.pt",
        "default_metric": "h_wiou",
        "placement": "la_loss",
    },
    {
        "id": "h_wiou_static_half",
        "name": "H-WIoU Ablation: Static gamma=0.5",
        "group": "H-WIoU Ablations",
        "path": RESULTS_DIR / "dipphmngc_tod-tp-ablation-static-half-s42/tp_ablation_static_half_s42/runs/h_wiou__h_wiou__static__h_wiou__seed42__ablation_static_half/best.pt",
        "default_metric": "h_wiou",
        "placement": "la_loss",
    },
    {
        "id": "h_wiou_form_sigmoid",
        "name": "H-WIoU Transition: Sigmoid Form",
        "group": "H-WIoU Ablations",
        "path": RESULTS_DIR / "pptlyn11_tod-tp-ablation-form-sigmoid-s42/tp_ablation_form_sigmoid_s42/runs/h_wiou__h_wiou__sigmoid__la_loss__seed42__ablation_form_sigmoid/best.pt",
        "default_metric": "h_wiou",
        "placement": "la_loss",
    },
    {
        "id": "h_wiou_form_exp",
        "name": "H-WIoU Transition: Exponential Form",
        "group": "H-WIoU Ablations",
        "path": RESULTS_DIR / "hngngnguynvn_tod-tp-ablation-exp-form-s42/tp_ablation_exp_form_s42/runs/h_wiou__h_wiou__exponential__h_wiou__seed42__ablation_exp_form/best.pt",
        "default_metric": "h_wiou",
        "placement": "la_loss",
    },
    {
        "id": "h_wiou_sigma_4",
        "name": "H-WIoU Scale Pivot: sigma_0=4.0px",
        "group": "H-WIoU Ablations",
        "path": RESULTS_DIR / "hngtrngtn_tod-tp-ablation-sig4-s42/tp_ablation_sig4_s42/runs/h_wiou__h_wiou__sig4__h_wiou__seed42__ablation_sig4/best.pt",
        "default_metric": "h_wiou",
        "placement": "la_loss",
    },
    {
        "id": "h_wiou_sigma_12",
        "name": "H-WIoU Scale Pivot: sigma_0=12.0px",
        "group": "H-WIoU Ablations",
        "path": RESULTS_DIR / "luongsythanh_tod-tp-ablation-sig12-s42/tp_ablation_sig12_s42/runs/h_wiou__h_wiou__sig12__la_loss__seed42__ablation_sig12/best.pt",
        "default_metric": "h_wiou",
        "placement": "la_loss",
    },
    {
        "id": "h_wiou_place_la",
        "name": "H-WIoU Placement: LA Only",
        "group": "H-WIoU Ablations",
        "path": RESULTS_DIR / "dipphmngc_tod-tp-ablation-place-la-s42/tp_ablation_place_la_s42/runs/h_wiou__smooth_l1__la__seed42__ablation_place_la/best.pt",
        "default_metric": "h_wiou",
        "placement": "la",
    },
    {
        "id": "h_wiou_place_loss",
        "name": "H-WIoU Placement: Loss Only",
        "group": "H-WIoU Ablations",
        "path": RESULTS_DIR / "hienquang06_tod-tp-ablation-place-loss-s42/tp_ablation_place_loss_s42/runs/h_wiou__h_wiou__loss__seed42__ablation_place_loss/best.pt",
        "default_metric": "h_wiou",
        "placement": "loss",
    },
]


def evaluate_single_checkpoint(entry: dict, test_loader: DataLoader, device: torch.device):
    ckpt_path = Path(entry["path"])
    if not ckpt_path.exists():
        print(f"⚠️ Checkpoint not found: {ckpt_path}")
        return None

    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load checkpoint {ckpt_path}: {e}")
        return None

    ep = ck.get("epoch", "?")
    cfg = ck.get("config", {})
    metric_name = cfg.get("metric", entry.get("default_metric", "iou"))
    placement = cfg.get("placement", entry.get("placement", "la_loss"))

    metric_fn = None if metric_name == "iou" else get_metric_fn(metric_name)

    model = build_model(
        metric_fn=metric_fn,
        placement=placement,
        reliability_thr=cfg.get("reliability_thr", 16.0),
        box_loss_type=cfg.get("box_loss", "metric"),
        use_quality_score=bool(cfg.get("quality_score", False)),
        quality_loss_weight=float(cfg.get("quality_loss_weight", 0.0) or 0.0),
        use_quality_focal=bool(cfg.get("quality_focal", False)),
        quality_focal_beta=float(cfg.get("quality_focal_beta", 2.0)),
        use_rank_sort=bool(cfg.get("rank_sort", False)),
        rank_sort_delta=float(cfg.get("rank_sort_delta", 0.5)),
        use_double_head=bool(cfg.get("double_head", False)),
        double_head_reg_roi_scale=float(cfg.get("double_head_reg_roi_scale", 1.3)),
        double_head_num_convs=int(cfg.get("double_head_num_convs", 4)),
        cbl_alpha=float(cfg.get("cbl_alpha", 5.0)),
        cbl_num_bins=int(cfg.get("cbl_num_bins", 6)),
        cbl_grid_beta=float(cfg.get("cbl_grid_beta", 1.0)),
        cbl_um_weight=float(cfg.get("cbl_um_weight", 1.0)),
    ).to(device)

    if "model" in ck:
        model.load_state_dict(ck["model"])
    elif "model_state_dict" in ck:
        model.load_state_dict(ck["model_state_dict"])
    elif "state_dict" in ck:
        model.load_state_dict(ck["state_dict"])
    else:
        print(f"⚠️ Unrecognized state dict in {ckpt_path}")
        return None

    model.eval()

    t0 = time.time()
    metrics = evaluate(model, test_loader, device, measure_fps_flag=True)
    t_eval = time.time() - t0

    res = {
        "id": entry["id"],
        "name": entry["name"],
        "group": entry["group"],
        "epoch": ep,
        "eval_time_sec": round(t_eval, 2),
        "mAP50": round(float(metrics.get("coco_AP50", metrics.get("AP50", 0.0))), 4),
        "mAP": round(float(metrics.get("coco_AP", metrics.get("AP", 0.0))), 4),
        "AP75": round(float(metrics.get("coco_AP75", metrics.get("AP75", 0.0))), 4),
        "APS": round(float(metrics.get("coco_AP_small", metrics.get("APS", 0.0))), 4),
        "APM": round(float(metrics.get("coco_AP_medium", metrics.get("APM", 0.0))), 4),
        "AR100": round(float(metrics.get("coco_AR100", metrics.get("AR100", 0.0))), 4),
        "FPS": round(float(metrics.get("FPS", 0.0)), 1),
    }
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    seed_all(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print(f"🚀 RUNNING INFERENCE ON TEST DATASET (Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 115)

    test_dir = ROOT / "data/test"
    if not test_dir.exists() or not (test_dir / "images").exists():
        test_dir = ROOT / "data/valid"
        print(f"ℹ️ Using validation set for evaluation: {test_dir}")
    else:
        print(f"ℹ️ Using official test set: {test_dir}")

    test_dataset = YOLOTinyDataset(img_dir=test_dir / "images", lbl_dir=test_dir / "labels", is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"Total Test Images/Tiles: {len(test_dataset)}\n")

    results = []
    print(f"{'METHOD / VARIANT':<40} | {'GROUP':<18} | {'mAP50':<7} | {'mAP':<7} | {'AP75':<7} | {'APS':<7} | {'AR100':<7} | {'FPS':<6}")
    print("-" * 115)

    for entry in CHECKPOINT_REGISTRY:
        res = evaluate_single_checkpoint(entry, test_loader, device)
        if res:
            results.append(res)
            print(f"{res['name']:<40} | {res['group']:<18} | {res['mAP50']:<7.4f} | {res['mAP']:<7.4f} | {res['AP75']:<7.4f} | {res['APS']:<7.4f} | {res['AR100']:<7.4f} | {res['FPS']:<6.1f}")

    print("=" * 115)

    # Save summary JSON
    out_json = RESULTS_DIR / "test_evaluation_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved full evaluation summary to {out_json}")


if __name__ == "__main__":
    main()
