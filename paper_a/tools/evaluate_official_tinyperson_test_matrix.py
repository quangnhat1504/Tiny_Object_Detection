"""Master evaluation script for all models on the 786 Official TinyPerson Test Images."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.metrics import get_metric_fn
from common.model import build_model
from paper_a.datasets.coco_original import CocoOriginalDataset
from paper_a.evaluation.tinyperson_official import evaluate_tinyperson_official

DEFAULT_DATA_ROOT = Path(r"D:\paper_a_data\TinyPerson\tiny_set\erase_with_uncertain_dataset")
DEFAULT_ANN_TEST = DEFAULT_DATA_ROOT / "annotations" / "mini_annotations" / "tiny_set_test_all.json"
DEFAULT_TEST_IMAGES = DEFAULT_DATA_ROOT / "test"

METHODS = [
    ("standard", "Faster R-CNN Baseline", "smooth_l1", "everywhere", None),
    ("nwd", "NWD (NeurIPS 2021)", "metric", "la_loss", "nwd"),
    ("igwd", "IGWD (IEEE TMM 2022)", "metric", "la_loss", "igwd"),
    ("saalw", "SA-ALW (Paper A)", "metric", "la_loss", "sa_alw_canonical"),
    ("rfla", "RFLA (ECCV 2022)", "metric", "h_wiou", "h_wiou"),
    ("hwiou_sig6", "H-WIoU (sig=6.0px) (Ablation)", "metric", "h_wiou", "h_wiou"),
    ("hwiou_sig8", "H-WIoU (sig=8.0px) (Proposed Ours)", "metric", "h_wiou", "h_wiou"),
]

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate_model_on_official_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    annotation_path: Path,
    cache_path: Path | None = None,
) -> Dict[str, float]:
    if cache_path and cache_path.exists():
        print(f"Loading cached predictions from {cache_path}...")
        all_predictions = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for images, targets in tqdm(loader, desc="Inference on 786 test images"):
                img_tensors = [img.to(device) for img in images]
                preds = model(img_tensors)
                
                for target, pred in zip(targets, preds):
                    img_id = int(target["image_id"])
                    boxes = pred["boxes"].detach().cpu().numpy()
                    scores = pred["scores"].detach().cpu().numpy()
                    labels = pred["labels"].detach().cpu().numpy()
                    
                    # Format: COCO detection dict: {"image_id": img_id, "category_id": 1, "bbox": [x1, y1, w, h], "score": score}
                    for box, score, label in zip(boxes, scores, labels):
                        if score > 0.05 and label == 1:
                            x1 = float(box[0])
                            y1 = float(box[1])
                            x2 = float(box[2])
                            y2 = float(box[3])
                            w = max(0.0, x2 - x1)
                            h = max(0.0, y2 - y1)
                            all_predictions.append({
                                "image_id": img_id,
                                "category_id": 1,
                                "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                                "score": round(float(score), 4),
                            })
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(all_predictions), encoding="utf-8")
                            
    # Run official TinyPerson evaluator
    eval_res = evaluate_tinyperson_official(annotation_path, all_predictions)
    return eval_res

def main():
    parser = argparse.ArgumentParser(description="Evaluate all models on 786 Official TinyPerson Test Images")
    parser.add_argument("--runs-root", type=str, default="runs/official_tinyperson_runs", help="Directory with trained model runs")
    parser.add_argument("--output-json", type=str, default="journal/results/official_tinyperson_786_test_benchmark.json", help="Path to save output JSON table")
    parser.add_argument("--batch-size", type=int, default=2, help="Inference batch size")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== EVALUATING ON OFFICIAL 786 TINYPERSON TEST SET (GPU: {torch.cuda.get_device_name(0)}) ===")
    print(f"Annotation: {DEFAULT_ANN_TEST}")
    print(f"Images Root: {DEFAULT_TEST_IMAGES}")

    # Load Dataset
    ds = CocoOriginalDataset(DEFAULT_TEST_IMAGES, DEFAULT_ANN_TEST)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"Loaded {len(ds)} test images successfully.\n")

    runs_root = Path(args.runs_root)
    final_table = {}

    for method_key, method_name, box_loss, placement, metric_name in METHODS:
        # Find checkpoint
        candidate_files = list(runs_root.rglob(f"*{method_key}*/**/best.pt"))
        if not candidate_files:
            candidate_files = list(runs_root.rglob(f"*{method_key}*/**/best_coco_ap.pt"))
        if not candidate_files:
            print(f"[SKIP] Checkpoint for {method_name} not found in {runs_root}")
            continue
            
        ckpt_path = candidate_files[0]

        print("=" * 80)
        print(f"--> EVALUATING: {method_name} ({ckpt_path})")
        print("=" * 80)

        # Build Model
        m_fn = None if metric_name is None else get_metric_fn(metric_name)
        model = build_model(
            num_classes=2,
            metric_fn=m_fn,
            placement=placement,
            box_loss_type=box_loss,
            box_loss_warmup_epochs=0,
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", {})))
        model.load_state_dict(sd, strict=False)

        metrics = evaluate_model_on_official_test(model, loader, device, DEFAULT_ANN_TEST)
        final_table[method_name] = metrics

        print(f"\nResults for {method_name}:")
        for k, v in metrics.items():
            val_str = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
            print(f"  {k:<20}: {val_str}")
        print("-" * 80)

    # Save to JSON
    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(final_table, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved official evaluation benchmark to: {out_p.resolve()}")

if __name__ == "__main__":
    main()
