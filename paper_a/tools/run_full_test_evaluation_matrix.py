"""
Master Script: Run Test & Validation Inference Across All Downloaded Checkpoints.
Computes Official COCO Metrics, TinyPerson Scale APs, Precision/Recall, and FPS.
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
sys.stdout.reconfigure(encoding="utf-8")


MODELS_TO_EVALUATE = [
    # 1. Official Baselines
    {
        "id": "frcnn_baseline",
        "name": "Faster R-CNN Baseline",
        "category": "Baseline",
        "ckpt": RESULTS_DIR / "amongus1504_tod-aitod-baseline-s42-20260823/aitod_baseline_s42/runs/aitod_standard_everywhere_smooth_l1_s42/last.pt",
        "fallback_ckpt": ROOT / "runs/frcnn_standard__full__seed42/best.pt",
        "metric": "iou",
        "placement": "everywhere",
    },
    {
        "id": "cascade_baseline",
        "name": "Cascade R-CNN Baseline",
        "category": "Baseline",
        "ckpt": RESULTS_DIR / "hngtrngtn_tod-aitod-cascade-s42-20260824/aitod_cascade_s42/runs/aitod_standard_everywhere_smooth_l1_s42_cascade_rcnn/last.pt",
        "metric": "iou",
        "placement": "everywhere",
    },
    # 2. SOTA Distance Metrics
    {
        "id": "rfla_sota",
        "name": "RFLA (Gaussian Assignment)",
        "category": "SOTA Baseline",
        "ckpt": RESULTS_DIR / "hngngnguynvn_tod-aitod-rfla-s42-20260823/aitod_rfla_s42/runs/aitod_rfla_la_smooth_l1_s42/last.pt",
        "metric": "iou",
        "placement": "la",
    },
    {
        "id": "nwd_sota",
        "name": "NWD (Normalized Wasserstein)",
        "category": "SOTA Baseline",
        "ckpt": RESULTS_DIR / "dipphmngc_tod-aitod-nwd-s42-20260823/aitod_nwd_s42/runs/aitod_nwd_everywhere_metric_s42/last.pt",
        "fallback_ckpt": ROOT / "runs/nwd__la_loss__seed42/best.pt",
        "metric": "nwd",
        "placement": "everywhere",
    },
    {
        "id": "igwd_sota",
        "name": "IGWD (Gaussian Wasserstein)",
        "category": "SOTA Baseline",
        "ckpt": RESULTS_DIR / "hienquang06_tod-aitod-igwd-s42-20260823/aitod_igwd_s42/runs/aitod_igwd_everywhere_metric_s42/last.pt",
        "fallback_ckpt": ROOT / "runs/igwd__la_loss__seed42/best.pt",
        "metric": "igwd",
        "placement": "everywhere",
    },
    {
        "id": "sa_alw_canonical",
        "name": "SA-ALW (Ours - Paper A)",
        "category": "SOTA Proposed",
        "ckpt": ROOT / "runs/sa_alw_full__la_loss__seed42/best.pt",
        "metric": "sa_alw_full",
        "placement": "la_loss",
    },
    # 3. Homotopy H-WIoU Ablations (Journal)
    {
        "id": "h_wiou_pure_w2",
        "name": "H-WIoU: Pure W2 (gamma=0)",
        "category": "H-WIoU Ablation",
        "ckpt": RESULTS_DIR / "thyngluthy_tod-tp-ablation-pure-w2-s42/tp_ablation_pure_w2_s42/runs/h_wiou__h_wiou__pure_w2__h_wiou__seed42__ablation_pure_w2/best.pt",
        "metric": "h_wiou",
        "placement": "la_loss",
    },
    {
        "id": "h_wiou_pure_iou",
        "name": "H-WIoU: Pure IoU (gamma=1)",
        "category": "H-WIoU Ablation",
        "ckpt": RESULTS_DIR / "hienquang06_tod-tp-ablation-pure-iou-s42/tp_ablation_pure_iou_s42/runs/h_wiou__h_wiou__pure_iou__h_wiou__seed42__ablation_pure_iou/best.pt",
        "metric": "h_wiou",
        "placement": "la_loss",
    },
    {
        "id": "h_wiou_static_half",
        "name": "H-WIoU: Static gamma=0.5",
        "category": "H-WIoU Ablation",
        "ckpt": RESULTS_DIR / "dipphmngc_tod-tp-ablation-static-half-s42/tp_ablation_static_half_s42/runs/h_wiou__h_wiou__static__h_wiou__seed42__ablation_static_half/best.pt",
        "metric": "h_wiou",
        "placement": "la_loss",
    },
    {
        "id": "h_wiou_form_sigmoid",
        "name": "H-WIoU: Sigmoid Form",
        "category": "H-WIoU Ablation",
        "ckpt": RESULTS_DIR / "pptlyn11_tod-tp-ablation-form-sigmoid-s42/tp_ablation_form_sigmoid_s42/runs/h_wiou__h_wiou__sigmoid__la_loss__seed42__ablation_form_sigmoid/best.pt",
        "metric": "h_wiou",
        "placement": "la_loss",
    },
    {
        "id": "h_wiou_form_exp",
        "name": "H-WIoU: Exponential Form",
        "category": "H-WIoU Ablation",
        "ckpt": RESULTS_DIR / "hngngnguynvn_tod-tp-ablation-exp-form-s42/tp_ablation_exp_form_s42/runs/h_wiou__h_wiou__exponential__h_wiou__seed42__ablation_exp_form/best.pt",
        "metric": "h_wiou",
        "placement": "la_loss",
    },
    {
        "id": "h_wiou_sigma_4",
        "name": "H-WIoU: sigma_0=4.0px",
        "category": "H-WIoU Ablation",
        "ckpt": RESULTS_DIR / "hngtrngtn_tod-tp-ablation-sig4-s42/tp_ablation_sig4_s42/runs/h_wiou__h_wiou__sig4__h_wiou__seed42__ablation_sig4/best.pt",
        "metric": "h_wiou",
        "placement": "la_loss",
    },
    {
        "id": "h_wiou_sigma_12",
        "name": "H-WIoU: sigma_0=12.0px",
        "category": "H-WIoU Ablation",
        "ckpt": RESULTS_DIR / "luongsythanh_tod-tp-ablation-sig12-s42/tp_ablation_sig12_s42/runs/h_wiou__h_wiou__sig12__la_loss__seed42__ablation_sig12/best.pt",
        "metric": "h_wiou",
        "placement": "la_loss",
    },
    {
        "id": "h_wiou_place_la",
        "name": "H-WIoU: LA Only",
        "category": "H-WIoU Ablation",
        "ckpt": RESULTS_DIR / "dipphmngc_tod-tp-ablation-place-la-s42/tp_ablation_place_la_s42/runs/h_wiou__smooth_l1__la__seed42__ablation_place_la/best.pt",
        "metric": "h_wiou",
        "placement": "la",
    },
    {
        "id": "h_wiou_place_loss",
        "name": "H-WIoU: Loss Only",
        "category": "H-WIoU Ablation",
        "ckpt": RESULTS_DIR / "hienquang06_tod-tp-ablation-place-loss-s42/tp_ablation_place_loss_s42/runs/h_wiou__h_wiou__loss__seed42__ablation_place_loss/best.pt",
        "metric": "h_wiou",
        "placement": "loss",
    },
]


def eval_one(m_info: dict, loader: DataLoader, device: torch.device):
    ckpt_p = m_info["ckpt"]
    if not ckpt_p.exists():
        fallback = m_info.get("fallback_ckpt")
        if fallback and fallback.exists():
            ckpt_p = fallback
        else:
            return None

    try:
        ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ Error loading {ckpt_p}: {e}")
        return None

    ep = ck.get("epoch", "?")
    cfg = ck.get("config", {})
    metric_name = cfg.get("metric", m_info.get("metric", "iou"))
    placement = cfg.get("placement", m_info.get("placement", "la_loss"))

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

    sd = ck.get("model", ck.get("model_state_dict", ck.get("state_dict", {})))
    model.load_state_dict(sd, strict=False)
    model.eval()

    t0 = time.perf_counter()
    metrics = evaluate(model, loader, device, measure_fps_flag=False)
    elapsed = time.perf_counter() - t0

    return {
        "id": m_info["id"],
        "name": m_info["name"],
        "category": m_info["category"],
        "epoch": ep,
        "eval_seconds": round(elapsed, 2),
        "mAP50": round(float(metrics.get("coco_AP50", metrics.get("mAP_50", 0.0))), 4),
        "mAP": round(float(metrics.get("coco_AP", 0.0)), 4),
        "AP75": round(float(metrics.get("coco_AP75", 0.0)), 4),
        "APS": round(float(metrics.get("coco_AP_small", 0.0)), 4),
        "APM": round(float(metrics.get("coco_AP_medium", 0.0)), 4),
        "AP_micro": round(float(metrics.get("AP_micro", 0.0)), 4),
        "AP_tiny": round(float(metrics.get("AP_tiny", 0.0)), 4),
        "mAP_scale": round(float(metrics.get("mAP_primary", 0.0)), 4),
        "AR100": round(float(metrics.get("coco_AR100", 0.0)), 4),
        "Precision": round(float(metrics.get("Precision", 0.0)), 4),
        "Recall": round(float(metrics.get("Recall", 0.0)), 4),
    }


def main():
    seed_all(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 135)
    print(f"🔥 MASTER TEST-SET INFERENCE (Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 135)

    test_ds = YOLOTinyDataset(img_dir=Path("data/test/images"), lbl_dir=Path("data/test/labels"), is_train=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"Test Images: {len(test_ds)}\n")

    test_results = []
    headers = ["Method / Variant", "Category", "mAP@50", "mAP", "AP75", "APS", "AP_micro", "AP_tiny", "mAP(scale)", "AR100", "P", "R"]
    print(f"{headers[0]:<35} | {headers[1]:<16} | {headers[2]:<7} | {headers[3]:<6} | {headers[4]:<6} | {headers[5]:<6} | {headers[6]:<8} | {headers[7]:<7} | {headers[8]:<10} | {headers[9]:<6} | {headers[10]:<6} | {headers[11]:<6}")
    print("-" * 135)

    for m in MODELS_TO_EVALUATE:
        res = eval_one(m, test_loader, device)
        if res:
            test_results.append(res)
            print(f"{res['name']:<35} | {res['category']:<16} | {res['mAP50']:<7.4f} | {res['mAP']:<6.4f} | {res['AP75']:<6.4f} | {res['APS']:<6.4f} | {res['AP_micro']:<8.4f} | {res['AP_tiny']:<7.4f} | {res['mAP_scale']:<10.4f} | {res['AR100']:<6.4f} | {res['Precision']:<6.4f} | {res['Recall']:<6.4f}")

    print("=" * 135)

    out_file = RESULTS_DIR / "official_test_benchmark_table.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    print(f"\n💾 Saved full evaluation summary to {out_file}")


if __name__ == "__main__":
    main()
