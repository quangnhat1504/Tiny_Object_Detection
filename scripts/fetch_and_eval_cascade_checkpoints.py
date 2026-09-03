"""
Download and evaluate completed Cascade & RFLA checkpoints from Kaggle cluster on AI-TOD-v2 Test Set.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.metrics import configure_metric
from common.model import build_model
from paper_a.datasets.aitodv2_adapter import AITODv2Dataset
from paper_a.evaluation.aitodv2_official import evaluate_aitodv2_official

PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_cascade_profiles")
DOWNLOAD_DIR = ROOT / "runs/cascade_kaggle_checkpoints"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = ROOT / "journal/results/cascade_aitod_benchmark.json"
RESULTS_MD = ROOT / "journal/results/cascade_aitod_benchmark.md"

CASCADE_KERNELS = [
    {
        "name": "Cascade Homotopy Proposed",
        "account": "luongsythanh",
        "slug": "tod-cascade-homotopy-s42-proposed",
        "metric": "h_wiou",
        "placement": "la_loss",
        "box_loss": "h_wiou",
        "rpn_cascade": True,
    },
    {
        "name": "Cascade Baseline Standard",
        "account": "amongus1504",
        "slug": "tod-cascade-baseline-s42",
        "metric": "h_wiou",
        "placement": "la_loss",
        "box_loss": "smooth_l1",
        "rpn_cascade": True,
    },
    {
        "name": "RFLA + H-WIoU Proposed",
        "account": "qnhat1504",
        "slug": "tod-rfla-hwiou-s42-proposed",
        "metric": "rfla",
        "placement": "la_loss",
        "box_loss": "h_wiou",
        "rpn_cascade": False,
    },
    {
        "name": "RFLA + Smooth-L1 Baseline",
        "account": "thyngluthy",
        "slug": "tod-rfla-baseline-s42",
        "metric": "rfla",
        "placement": "la_loss",
        "box_loss": "smooth_l1",
        "rpn_cascade": False,
    },
    {
        "name": "DU-HWIoU Proposed",
        "account": "quangnhtng",
        "slug": "tod-aitod-du-hwiou-s42-proposed",
        "metric": "h_wiou",
        "placement": "h_wiou",
        "box_loss": "h_wiou",
        "rpn_cascade": False,
    },
    {
        "name": "QFL + H-WIoU Proposed",
        "account": "trieuvo123",
        "slug": "tod-aitod-qfl-hwiou-s42-proposed",
        "metric": "h_wiou",
        "placement": "h_wiou",
        "box_loss": "h_wiou",
        "rpn_cascade": False,
        "use_quality_focal": True,
    },
]


def download_outputs():
    print("=== DOWNLOADING COMPLETED CASCADE RUNS FROM KAGGLE ===")
    for item in CASCADE_KERNELS:
        account = item["account"]
        slug = item["slug"]
        profile = PROFILE_ROOT / account
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(profile)

        dest = DOWNLOAD_DIR / slug
        dest.mkdir(parents=True, exist_ok=True)

        full_slug = f"{account}/{slug}"
        print(f"Checking & downloading {full_slug} -> {dest}")
        cmd = ["kaggle", "kernels", "output", full_slug, "-p", str(dest)]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        print("Output:", res.stdout.strip())
def aitod_collate_fn(batch):
    return [x[0] for x in batch], [x[1] for x in batch]


def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== EVALUATING DOWNLOADED CHECKPOINTS ON AI-TOD-V2 TEST SET ({device}) ===")
    
    test_img_dir = Path(r"D:\paper_a_data\AI-TOD-v2\AI-TOD\images\test")
    test_ann_file = Path(r"D:\paper_a_data\AI-TOD-v2\AI-TOD\annotations\aitodv2_test.json")
    if not test_ann_file.exists():
        test_ann_file = Path(r"D:\paper_a_data\AI-TOD-v2\annotations\aitodv2_test.json")

    test_dataset = AITODv2Dataset(test_img_dir, test_ann_file, drop_empty=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=aitod_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    print(f"Loaded AI-TOD-v2 test dataset: {len(test_dataset)} images.")

    all_metrics = {}
    if RESULTS_JSON.exists():
        try:
            all_metrics = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
            print(f"Loaded existing results for {len(all_metrics)} models from {RESULTS_JSON}")
        except Exception:
            all_metrics = {}

    for item in CASCADE_KERNELS:
        name = item["name"]
        slug = item["slug"]

        if name in all_metrics and "--force" not in sys.argv:
            m = all_metrics[name].get("metrics", {})
            print(f"[{name}] Already evaluated (AP = {m.get('AP', 0.0):.4f}, AP50 = {m.get('AP50', 0.0):.4f}). Skipping.")
            continue

        ckpt_dir = DOWNLOAD_DIR / slug
        best_pt = None
        for p in ckpt_dir.rglob("*.pt"):
            if "best" in p.name:
                best_pt = p
                break
        if best_pt is None:
            for p in ckpt_dir.rglob("*.pt"):
                if "last" in p.name:
                    best_pt = p
                    break

        if best_pt is None:
            print(f"Skipping {name}: No checkpoint found in {ckpt_dir}")
            continue

        print(f"\nEvaluating {name} using checkpoint: {best_pt}")
        if item["metric"] in ("h_wiou", "du_hwiou"):
            similarity_fn, distance_fn, _ = configure_metric("h_wiou", h_wiou_sigma_0=8.0)
        elif item["metric"] == "rfla":
            from common.metrics import iou
            similarity_fn, distance_fn = iou.compute_rfd, None
        elif item["metric"] == "nwd":
            similarity_fn, distance_fn, _ = configure_metric("nwd")
        else:
            similarity_fn, distance_fn = None, None

        model = build_model(
            num_classes=9,
            metric_fn=similarity_fn if item["metric"] != "standard" else None,
            metric_distance_fn=distance_fn if item["metric"] != "standard" else None,
            placement=item["placement"],
            box_loss_type=item["box_loss"],
            rpn_cascade=item["rpn_cascade"],
            use_quality_focal=item.get("use_quality_focal", False),
        ).to(device)

        state = torch.load(best_pt, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict):
            model.load_state_dict(state)
        model.eval()

        coco_results = []
        total_batches = len(test_loader)
        t0 = time.time()
        with torch.no_grad():
            for b_idx, (images, targets) in enumerate(test_loader):
                img_tensors = [img.to(device) for img in images]
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    preds = model(img_tensors)
                for target, pred in zip(targets, preds):
                    img_id = int(target["image_id"])
                    boxes = pred["boxes"].detach().cpu().numpy()
                    scores = pred["scores"].detach().cpu().numpy()
                    labels = pred["labels"].detach().cpu().numpy()
                    for box, score, label in zip(boxes, scores, labels):
                        int_l = int(label)
                        if score > 0.05 and int_l in test_dataset.label_to_category_id:
                            cat_id = test_dataset.label_to_category_id[int_l]
                            x1, y1, x2, y2 = [float(v) for v in box]
                            w = max(x2 - x1, 0.0)
                            h = max(y2 - y1, 0.0)
                            coco_results.append({
                                "image_id": img_id,
                                "category_id": cat_id,
                                "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                                "score": round(float(score), 4),
                            })
                if (b_idx + 1) % 500 == 0 or (b_idx + 1) == total_batches:
                    elapsed = time.time() - t0
                    fps = ((b_idx + 1) * 4) / max(elapsed, 1e-5)
                    print(f"  [{name}] Processed {b_idx + 1}/{total_batches} batches ({len(coco_results):,} preds) - {fps:.1f} FPS")

        if coco_results:
            eval_res = evaluate_aitodv2_official(test_ann_file, coco_results, quiet=False)
            all_metrics[name] = eval_res
            m = eval_res.get("metrics", {})
            print(f"[{name}] Completed. AP = {m.get('AP', 0.0):.4f}, AP50 = {m.get('AP50', 0.0):.4f}, AP_vt = {m.get('AP_verytiny', 0.0):.4f}")

    if all_metrics:
        RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n[SUCCESS] Results saved to {RESULTS_JSON}")

        # Write markdown summary table
        lines = [
            "# Bảng Đánh Giá Chính Thức AI-TOD-v2 (Test Set 14.018 Ảnh)",
            "",
            "| Phương Pháp (Method) | AP (%) | AP50 (%) | AP75 (%) | AP_vt (%) | AP_t (%) | AP_s (%) | AR1500 (%) | AR_vt (%) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for name, res in all_metrics.items():
            m = res.get("metrics", {})
            lines.append(
                f"| **{name}** | {m.get('AP', 0)*100:.2f} | {m.get('AP50', 0)*100:.2f} | "
                f"{m.get('AP75', 0)*100:.2f} | {m.get('AP_verytiny', 0)*100:.2f} | "
                f"{m.get('AP_tiny', 0)*100:.2f} | {m.get('AP_small', 0)*100:.2f} | "
                f"{m.get('AR1500', 0)*100:.2f} | {m.get('AR_verytiny', 0)*100:.2f} |"
            )
        with open(RESULTS_MD, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[SUCCESS] Markdown table saved to {RESULTS_MD}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        evaluate_all()
    else:
        download_outputs()
        if len(sys.argv) > 1 and "--eval" in sys.argv:
            evaluate_all()
