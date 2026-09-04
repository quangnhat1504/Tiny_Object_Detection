"""
Evaluate Downloaded EH-WIoU Checkpoints on Official AI-TOD-v2 Test Set.
Supports full 14,018 image evaluation or quick sample evaluation via --limit.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.metrics import configure_metric
from common.model import build_model
from paper_a.datasets.aitodv2_adapter import AITODv2Dataset, aitod_collate_fn
from paper_a.evaluation.aitodv2_official import evaluate_aitodv2_official
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DEFAULT_ANN_TEST = Path(r"D:\paper_a_data\AI-TOD-v2\annotations\aitodv2_test.json")
DEFAULT_IMG_TEST = Path(r"D:\paper_a_data\AI-TOD-v2\AI-TOD\images\test")

MODELS_TO_EVAL = [
    {
        "name": "EH-WIoU Proposed (sigma0=6.0px, s42)",
        "ckpt": ROOT / "runs/matrix_kaggle_outputs/aitod_ehwiou_sig6_s42/runs/aitod_eh_wiou_h_wiou_eh_wiou_s42_ehwiou_sig6_s42/best.pt",
        "sigma0": 6.0,
        "qfl": False,
    },
    {
        "name": "EH-WIoU Proposed (sigma0=8.0px, s42)",
        "ckpt": ROOT / "runs/matrix_kaggle_outputs/aitod_ehwiou_s42_chunked/runs/aitod_eh_wiou_h_wiou_eh_wiou_s42_ehwiou_sig8_s42/best.pt",
        "sigma0": 8.0,
        "qfl": False,
    },
    {
        "name": "EH-WIoU Proposed (sigma0=8.0px, s123)",
        "ckpt": ROOT / "runs/matrix_kaggle_outputs/aitod_ehwiou_sig8_s123/runs/aitod_eh_wiou_h_wiou_eh_wiou_s123_ehwiou_sig8_s123/best.pt",
        "sigma0": 8.0,
        "qfl": False,
    },
    {
        "name": "SW-HWIoU Proposed (sigma0=8.0px, s42)",
        "ckpt": ROOT / "runs/matrix_kaggle_outputs/aitod_sw_hwiou_s42/runs/aitod_sw_hwiou_h_wiou_h_wiou_s42_sw_hwiou_s42/best.pt",
        "sigma0": 8.0,
        "qfl": False,
    },
    {
        "name": "QFL + DU-HWIoU Proposed (sigma0=8.0px, s42)",
        "ckpt": ROOT / "runs/matrix_kaggle_outputs/aitod_qfl_duhwiou_s42/runs/aitod_du_hwiou_h_wiou_h_wiou_s42_qfl_duhwiou_s42/best.pt",
        "sigma0": 8.0,
        "qfl": True,
    },
]


def evaluate_single_model(model, loader, device, ann_path, label_to_cat, desc=""):
    model.eval()
    all_detections = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=desc):
            img_tensors = [img.to(device) for img in images]
            preds = model(img_tensors)
            for target, pred in zip(targets, preds):
                img_id = int(target["image_id"])
                boxes = pred["boxes"].detach().cpu().numpy()
                scores = pred["scores"].detach().cpu().numpy()
                labels = pred["labels"].detach().cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    int_l = int(label)
                    if score > 0.05 and int_l in label_to_cat:
                        cat_id = label_to_cat[int_l]
                        x1, y1, x2, y2 = [float(v) for v in box]
                        w = max(x2 - x1, 0.0)
                        h = max(y2 - y1, 0.0)
                        all_detections.append({
                            "image_id": img_id,
                            "category_id": cat_id,
                            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                            "score": round(float(score), 4),
                        })

    if not all_detections:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}

    eval_res = evaluate_aitodv2_official(ann_path, all_detections, quiet=True)
    return eval_res.get("metrics", {})


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Number of images to evaluate (0 for full test set)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if already cached")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Evaluation on Device: {device} ({torch.cuda.get_device_name(0)}) ===")
    print(f"Annotation file: {DEFAULT_ANN_TEST}")
    print(f"Images folder: {DEFAULT_IMG_TEST}")

    ds = AITODv2Dataset(DEFAULT_IMG_TEST, DEFAULT_ANN_TEST, drop_empty=False)
    if args.limit > 0:
        ds.records = ds.records[:args.limit]
        print(f"[NOTE] Quick Sanity Mode: Limited to {len(ds)} test images.")
    else:
        print(f"Full Official Evaluation: {len(ds)} test images.")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=aitod_collate_fn)

    out_p = ROOT / "journal/results/ehwiou_downloaded_evaluated_metrics.json"
    results_dict = {}
    if out_p.exists() and not args.force and args.limit == 0:
        try:
            cached = json.loads(out_p.read_text(encoding="utf-8"))
            for item in cached:
                if "name" in item:
                    results_dict[item["name"]] = item
            print(f"Loaded {len(results_dict)} existing results from {out_p}")
        except Exception as e:
            print(f"Warning loading cached results: {e}")

    for item in MODELS_TO_EVAL:
        name = item["name"]
        ckpt_path = item["ckpt"]
        sigma0 = item["sigma0"]
        qfl = item.get("qfl", False)

        if not ckpt_path.exists():
            print(f"[SKIP] Checkpoint not found: {ckpt_path}")
            continue

        if name in results_dict and not args.force and args.limit == 0:
            print(f"[CACHED] {name} already evaluated (mAP={results_dict[name].get('AP', 0.0):.4f}). Skipping.")
            continue

        print(f"\nEvaluating: {name} (qfl={qfl})...")
        m_fn, _, _ = configure_metric("h_wiou", h_wiou_sigma_0=sigma0)
        model = build_model(
            num_classes=9,
            metric_fn=m_fn,
            placement="h_wiou",
            box_loss_type="h_wiou",
            box_loss_warmup_epochs=0,
            use_quality_focal=qfl,
            quality_focal_beta=2.0 if qfl else 2.0,
        ).to(device)

        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and any(k in sd for k in ("model", "model_state_dict", "state_dict")):
            sd = sd.get("model", sd.get("model_state_dict", sd.get("state_dict", {})))
        model.load_state_dict(sd, strict=False)

        metrics = evaluate_single_model(model, loader, device, DEFAULT_ANN_TEST, ds.label_to_category_id, desc=name)
        metrics["name"] = name
        results_dict[name] = metrics

        print(f"--> Results for {name}:")
        print(f"    mAP:         {metrics.get('AP', 0.0):.4f}")
        print(f"    mAP50:       {metrics.get('AP50', 0.0):.4f}")
        print(f"    mAP75:       {metrics.get('AP75', 0.0):.4f}")
        print(f"    AP_verytiny: {metrics.get('AP_verytiny', 0.0):.4f}")
        print(f"    AP_tiny:     {metrics.get('AP_tiny', 0.0):.4f}")
        print(f"    AP_small:    {metrics.get('AP_small', 0.0):.4f}")
        print(f"    AP_medium:   {metrics.get('AP_medium', 0.0):.4f}")
        print(f"    AR100:       {metrics.get('AR100', 0.0):.4f}")
        print(f"    AR1500:      {metrics.get('AR1500', 0.0):.4f}")

        # Save immediately after each model
        if args.limit == 0:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(list(results_dict.values()), indent=2), encoding="utf-8")
            print(f"Updated results saved to {out_p}")

    results_table = list(results_dict.values())
    if args.limit == 0:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results_table, indent=2), encoding="utf-8")
        print(f"\nFinal evaluation results saved to {out_p}")


if __name__ == "__main__":
    run()
