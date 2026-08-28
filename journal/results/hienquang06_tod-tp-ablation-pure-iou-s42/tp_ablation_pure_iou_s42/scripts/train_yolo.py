"""
Phase 1: YOLO baseline training (full-image, no router, no RCNN).

Usage:
    python scripts/train_yolo.py                    # YOLOv8n default
    python scripts/train_yolo.py --model yolo11n    # YOLOv11n
    python scripts/train_yolo.py --epochs 50
    python scripts/train_yolo.py --imgsz 640
"""
from __future__ import annotations
import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import torch
from ultralytics import YOLO
logging.getLogger("ultralytics").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "data.yaml"
OUTPUT_DIR = ROOT / "runs" / "yolo_baseline"

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640
DEFAULT_LR = 0.01
DEFAULT_BATCH = 16


def train_yolo(model_name: str, epochs: int, imgsz: int,
               lr: float, batch: int, device: str):
    """Train YOLO baseline and log metrics."""
    print("=" * 60)
    print(f"YOLO BASELINE — {model_name}")
    print("=" * 60)
    print(f"Data: {DATA_YAML}")
    print(f"Epochs: {epochs} | ImgSize: {imgsz} | LR: {lr} | Batch: {batch}")

    # Load pretrained model
    model = YOLO(model_name)
    print(f"Model loaded: {model_name}")

    # Train
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        lr0=lr,
        batch=batch,
        device=device,
        patience=10,
        seed=42,
        project=str(OUTPUT_DIR),
        name=model_name.replace(".pt", ""),
        exist_ok=True,
        amp=True,
        val=True,
        verbose=True,
    )

    print(f"\nTraining complete. Results in: {OUTPUT_DIR}")

    # ── Evaluation on val set ──
    print("\n" + "=" * 60)
    print("VALIDATION EVALUATION")
    print("=" * 60)
    val_metrics = model.val(data=str(DATA_YAML), device=device)

    # ── FPS benchmark ──
    print("\n" + "=" * 60)
    print("FPS BENCHMARK")
    print("=" * 60)
    model.model.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model.predict(dummy, verbose=False)
    # Measure
    n_warmup = 50
    n_iters = 200
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(n_iters):
            _ = model.predict(dummy, verbose=False)
            if i == n_warmup:
                torch.cuda.synchronize() if device == "cuda" else None
                t0 = time.perf_counter()
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.perf_counter() - t0
    effective_iters = n_iters - n_warmup
    fps = effective_iters / elapsed

    # ── Compile results ──
    results_summary = {
        "model": model_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "optimizer": "SGD",
        "data": str(DATA_YAML),
        # Validation metrics
        "mAP@50": round(float(val_metrics.box.map50), 4),
        "mAP@50:95": round(float(val_metrics.box.map), 4),
        "mAP@75": round(float(val_metrics.box.map75), 4),
        "Precision": round(float(val_metrics.box.mp), 4),
        "Recall": round(float(val_metrics.box.mr), 4),
        # Speed
        "FPS": round(fps, 1),
        "inference_ms": round(1000.0 / fps, 2),
    }

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for k, v in results_summary.items():
        print(f"  {k:20s}: {v}")

    # Save
    (OUTPUT_DIR / model_name.replace(".pt", "") / "results.json").write_text(
        json.dumps(results_summary, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nResults saved to: {OUTPUT_DIR / model_name.replace('.pt', '') / 'results.json'}")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="YOLO baseline training")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="YOLO model name (yolov8n.pt, yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_yolo(args.model, args.epochs, args.imgsz,
               args.lr, args.batch, args.device)


if __name__ == "__main__":
    main()
