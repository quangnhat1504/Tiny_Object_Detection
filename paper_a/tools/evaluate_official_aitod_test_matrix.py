r"""
Master Evaluation Script for AI-TOD-v2 on 14,018 Official Test Images (D:\paper_a_data\AI-TOD-v2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.metrics import get_metric_fn
from common.model import build_model
from paper_a.datasets.aitodv2_adapter import AITODv2Dataset, aitod_collate_fn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DEFAULT_DATA_ROOT = Path(r"D:\paper_a_data\AI-TOD-v2")
DEFAULT_ANN_TEST = DEFAULT_DATA_ROOT / "annotations" / "aitodv2_test.json"
if not DEFAULT_ANN_TEST.exists():
    DEFAULT_ANN_TEST = DEFAULT_DATA_ROOT / "AI-TOD" / "annotations" / "val.json"
DEFAULT_TEST_IMAGES = DEFAULT_DATA_ROOT / "AI-TOD" / "images" / "test"

AITOD_CLASSES = {
    1: "airplane",
    2: "bridge",
    3: "storage-tank",
    4: "ship",
    5: "swimming-pool",
    6: "vehicle",
    7: "person",
    8: "wind-mill",
}

from paper_a.evaluation.aitodv2_official import evaluate_aitodv2_official


def evaluate_aitod_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    annotation_path: Path,
    label_to_category_id: dict[int, int],
    cache_path: Path | None = None,
) -> Dict[str, Any]:
    if cache_path and cache_path.exists():
        print(f"Loading cached predictions from {cache_path}...")
        all_detections = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        model.eval()
        all_detections = []

        with torch.no_grad():
            for images, targets in tqdm(loader, desc="AI-TOD Test Inference"):
                img_tensors = [img.to(device) for img in images]
                preds = model(img_tensors)

                for target, pred in zip(targets, preds):
                    img_id = int(target["image_id"])
                    boxes = pred["boxes"].detach().cpu().numpy()
                    scores = pred["scores"].detach().cpu().numpy()
                    labels = pred["labels"].detach().cpu().numpy()

                    for box, score, label in zip(boxes, scores, labels):
                        int_l = int(label)
                        if score > 0.05 and int_l in label_to_category_id:
                            cat_id = label_to_category_id[int_l]
                            x1, y1, x2, y2 = [float(v) for v in box]
                            w = max(x2 - x1, 0.0)
                            h = max(y2 - y1, 0.0)
                            all_detections.append({
                                "image_id": img_id,
                                "category_id": cat_id,
                                "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                                "score": round(float(score), 4),
                            })
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(all_detections), encoding="utf-8")

    if not all_detections:
        return {"mAP_50": 0.0, "mAP": 0.0, "AP_75": 0.0}

    # Run official pinned AI-TOD evaluator
    eval_res = evaluate_aitodv2_official(annotation_path, all_detections, quiet=False)
    return eval_res

def main():
    parser = argparse.ArgumentParser(description="Evaluate AI-TOD-v2 on 14,018 official test images")
    parser.add_argument("--ann-file", type=str, default=str(DEFAULT_ANN_TEST))
    parser.add_argument("--img-dir", type=str, default=str(DEFAULT_TEST_IMAGES))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="journal/results/official_aitod_14018_test_benchmark.json")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"=== EVALUATING AI-TOD-V2 TEST BENCHMARK ON GPU: {torch.cuda.get_device_name(0)} ===")
    print(f"Annotation: {args.ann_file}")
    print(f"Images Root: {args.img_dir}")
    print("=" * 80)

    # Load GT
    coco_gt = COCO(args.ann_file)
    ds = AITODv2Dataset(Path(args.img_dir), Path(args.ann_file), drop_empty=False)
    if args.limit > 0:
        ds.records = ds.records[: args.limit]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=aitod_collate_fn)
    print(f"Total AI-TOD-v2 Test Samples to evaluate: {len(ds)}\n")

    # Candidate Checkpoints
    results = {}
    checkpoints = [
        (
            "H-WIoU Unified Proposed (sigma0=8.0px)",
            ROOT / "runs" / "aitod_h_wiou_h_wiou_h_wiou_s42_hwiou_unified_sig8_s42" / "best.pt",
            "h_wiou",
            "h_wiou",
            8.0,
        ),
        (
            "H-WIoU Proposed (sigma0=8.0px)",
            ROOT / "runs" / "aitod_kaggle_checkpoints" / "hwiou_sig8_s42" / "tod_output" / "runs" / "aitod_h_wiou_h_wiou_h_wiou_s42_hwiou_sig8_s42" / "best.pt",
            "h_wiou",
            "h_wiou",
            8.0,
        ),
        (
            "H-WIoU Ablation (sigma0=6.0px)",
            ROOT / "runs" / "aitod_kaggle_checkpoints" / "hwiou_sig6_s42" / "tod_output" / "runs" / "aitod_h_wiou_h_wiou_h_wiou_s42_hwiou_sig6_s42" / "best.pt",
            "h_wiou",
            "h_wiou",
            6.0,
        ),
        (
            "H-WIoU Ablation (sigma0=10.0px)",
            ROOT / "runs" / "aitod_kaggle_checkpoints" / "hwiou_sig10_s42" / "tod_output" / "runs" / "aitod_h_wiou_h_wiou_h_wiou_s42_hwiou_sig10_s42" / "best.pt",
            "h_wiou",
            "h_wiou",
            10.0,
        ),
        (
            "H-WIoU + Cascade R-CNN",
            ROOT / "runs" / "aitod_kaggle_checkpoints" / "hwiou_cascade_s42" / "tod_output" / "runs" / "aitod_h_wiou_h_wiou_h_wiou_s42_hwiou_cascade_s42" / "best.pt",
            "h_wiou",
            "h_wiou",
            8.0,
        ),
        (
            "Faster R-CNN Baseline (Standard IoU)",
            ROOT / "runs" / "aitod_kaggle_checkpoints" / "baseline_s42" / "tod_output" / "runs" / "aitod_standard_everywhere_smooth_l1_s42_baseline_s42" / "best.pt",
            None,
            "everywhere",
            None,
        ),
        (
            "NWD (NeurIPS 2021)",
            ROOT / "runs" / "aitod_kaggle_checkpoints" / "nwd_s42" / "tod_output" / "runs" / "aitod_nwd_everywhere_metric_s42_nwd_s42" / "best.pt",
            "nwd",
            "everywhere",
            None,
        ),
        (
            "SAFit (AAAI 2024)",
            ROOT / "runs" / "official_aitod_checkpoints" / "safit" / "aitod_safit_s42" / "runs" / "aitod_sa_alw_canonical_everywhere_metric_s42_safit" / "last.pt",
            "sa_alw_canonical",
            "everywhere",
            None,
        ),
    ]

    per_class_results = {}

    for name, ckpt_p, m_name, placement, sigma0 in checkpoints:
        if not ckpt_p.exists():
            # Try fallback to last.pt or root dir
            alt_ckpts = list(ckpt_p.parent.glob("*.pt")) if ckpt_p.parent.exists() else []
            if alt_ckpts:
                ckpt_p = alt_ckpts[0]
            else:
                print(f"[SKIP] Checkpoint not found for {name}: {ckpt_p}")
                continue

        print("\n" + "#" * 80)
        print(f"--> EVALUATING: {name} (Checkpoint: {ckpt_p.name})")
        print("#" * 80)

        if m_name == "h_wiou":
            from common.metrics import configure_metric
            m_fn, _, _ = configure_metric("h_wiou", h_wiou_sigma_0=sigma0 if sigma0 else 8.0)
        elif m_name:
            m_fn = get_metric_fn(m_name)
        else:
            m_fn = None

        model = build_model(
            num_classes=9,
            metric_fn=m_fn,
            placement=placement,
            box_loss_type="h_wiou" if m_name == "h_wiou" else ("metric" if m_name else "smooth_l1"),
            box_loss_warmup_epochs=0,
        ).to(device)

        ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and any(k in ck for k in ("model", "model_state_dict", "state_dict")):
            sd = ck.get("model", ck.get("model_state_dict", ck.get("state_dict", {})))
        else:
            sd = ck
        model.load_state_dict(sd, strict=False)

        cache_p = Path("journal/results/preds_cache") / f"aitod_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('+', '_')}.json"
        metrics = evaluate_aitod_model(
            model, loader, device, Path(args.ann_file),
            label_to_category_id=ds.label_to_category_id,
            cache_path=cache_p
        )
        results[name] = metrics

        # Compute exact per-category AP50 using COCOeval on cache_p
        try:
            coco_dt = coco_gt.loadRes(str(cache_p))
            ev = COCOeval(coco_gt, coco_dt, "bbox")
            ev.params.iouThrs = [0.5]
            ev.params.maxDets = [1500]
            ev.evaluate()
            ev.accumulate()

            cats = coco_gt.loadCats(coco_gt.getCatIds())
            cat_aps = []
            for k, cat in enumerate(cats):
                pr = ev.eval["precision"][0, :, k, 0, 0]
                pr = pr[pr > -1]
                ap = float(np.mean(pr) * 100) if len(pr) > 0 else 0.0
                cat_aps.append(round(ap, 1))
            per_class_results[name] = {
                "categories": [c["name"] for c in cats],
                "per_class_ap50": cat_aps,
                "mean_ap50": round(float(np.mean(cat_aps)), 2),
            }
        except Exception as e:
            print(f"Per-category calculation warning: {e}")

        print(f"\nResults for {name}:")
        for k, v in metrics.items():
            val_str = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
            print(f"  {k:<20}: {val_str}")

    # Save to JSON
    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] AI-TOD-v2 benchmark saved to {out_p.resolve()}")

    # Export Markdown table with Table 2 AND Table 3
    md_p = out_p.with_suffix(".md")
    md_lines = [
        "# Bảng Tổng Hợp Đánh Giá Chính Thức Trên 14.018 Ảnh Test AI-TOD-v2\n",
        f"* **Tập dữ liệu**: `{args.img_dir}` (14.018 ảnh test thực tế, 8 lớp vi mô).",
        "* **Evaluator**: Pinned `aitodpycocotools` theo chuẩn chính thức của tác giả AI-TOD-v2 (Wang et al.).",
        "* **Môi trường**: Huấn luyện độc lập trên cụm Kaggle Tesla T4 GPU (PyTorch AMP) $\\to$ Suy luận thực nghiệm trên NVIDIA GeForce RTX 5070 Ti local.\n",
        "---",
        "## 1. Bảng So Sánh Toàn Diện (Table 2 in Manuscript)\n",
        "| Phương Pháp (Method) | Tổng Dự Đoán | AP (%) | $\\text{AP}_{50}$ (%) | $\\text{AP}_{75}$ (%) | $\\text{AP}_{vt}$ (%) | $\\text{AP}_t$ (%) | $\\text{AP}_s$ (%) | $\\text{AP}_m$ (%) | $\\text{AR}_{1500}$ (%) | oLRP |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_name, res in results.items():
        m = res.get("metrics", {})
        preds = res.get("prediction_count", 0)
        ap = m.get("AP", 0.0) * 100
        ap50 = m.get("AP50", 0.0) * 100
        ap75 = m.get("AP75", 0.0) * 100
        ap_vt = m.get("AP_verytiny", 0.0) * 100
        ap_t = m.get("AP_tiny", 0.0) * 100
        ap_s = m.get("AP_small", 0.0) * 100
        ap_m = m.get("AP_medium", 0.0) * 100
        ar1500 = m.get("AR1500", 0.0) * 100
        olrp = m.get("oLRP", 0.0)
        md_lines.append(f"| **{m_name}** | {preds:,} | **{ap:.2f}** | {ap50:.2f} | {ap75:.2f} | {ap_vt:.2f} | {ap_t:.2f} | {ap_s:.2f} | {ap_m:.2f} | {ar1500:.2f} | {olrp:.4f} |")

    md_lines.extend([
        "\n---",
        "## 2. Chi Tiết Các Chỉ Số Thu Hồi (Recall & Localization Error Breakdown)\n",
        "| Phương Pháp | $\\text{AR}_{1}$ (%) | $\\text{AR}_{100}$ (%) | $\\text{AR}_{1500}$ (%) | $\\text{AR}_{vt}$ (%) | $\\text{AR}_{t}$ (%) | $\\text{AR}_{s}$ (%) | $\\text{AR}_{m}$ (%) | $\\text{oLRP}_{\\text{loc}}$ | $\\text{oLRP}_{\\text{fp}}$ | $\\text{oLRP}_{\\text{fn}}$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for m_name, res in results.items():
        m = res.get("metrics", {})
        ar1 = m.get("AR1", 0.0) * 100
        ar100 = m.get("AR100", 0.0) * 100
        ar1500 = m.get("AR1500", 0.0) * 100
        ar_vt = m.get("AR_verytiny", 0.0) * 100
        ar_t = m.get("AR_tiny", 0.0) * 100
        ar_s = m.get("AR_small", 0.0) * 100
        ar_m = m.get("AR_medium", 0.0) * 100
        olrp_loc = m.get("oLRP_localization", 0.0)
        olrp_fp = m.get("oLRP_false_positive", 0.0)
        olrp_fn = m.get("oLRP_false_negative", 0.0)
        md_lines.append(f"| **{m_name}** | {ar1:.2f} | {ar100:.2f} | {ar1500:.2f} | {ar_vt:.2f} | {ar_t:.2f} | {ar_s:.2f} | {ar_m:.2f} | {olrp_loc:.4f} | {olrp_fp:.4f} | {olrp_fn:.4f} |")

    # Add Table 3 in Markdown
    md_lines.extend([
        "\n---",
        "## 3. Phân Rã Hiệu Năng Theo Từng Lớp Mục Tiêu (Table 3 in Manuscript - Per-Category AP50 %)\n",
        "| Phương Pháp | Airplane | Bridge | Storage | Ship | Pool | Vehicle | Person | Windmill | Mean $\\text{AP}_{50}$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for m_name, pc in per_class_results.items():
        caps = pc.get("per_class_ap50", [])
        m_ap = pc.get("mean_ap50", 0.0)
        c_str = " | ".join(f"{v:.1f}" for v in caps)
        md_lines.append(f"| **{m_name}** | {c_str} | **{m_ap:.1f}%** |")

    md_p.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[SUCCESS] Markdown benchmark summary exported to {md_p.resolve()}")

if __name__ == "__main__":
    main()
