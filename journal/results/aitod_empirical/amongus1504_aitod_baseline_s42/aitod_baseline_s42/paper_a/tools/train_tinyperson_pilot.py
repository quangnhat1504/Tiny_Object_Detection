"""Paper A WP01 TinyPerson G3 pilot trainer (seed-matched reduced budget).

Canonical pilot harness for the six frozen pilot methods:

    standard          Standard IoU assignment + Smooth-L1 regression
    igwd              Verified direct predecessor (IGWD, assignment + loss)
    alw_canonical     Pure canonical ALW (beta = 8, no reliability wrapping)
    sa_alw_beta_only  SA-ALW beta schedule only
    sa_alw_pos_only   SA-ALW position-weight schedule only
    sa_alw_full       Full SA-ALW (beta + position schedules)

All methods share: TinyPersonOriginalDataset with the frozen PL-001 split,
seeded horizontal flip as the only augmentation, seeded data order, the
common.config optimizer schedule, the validation-COCO-AP checkpoint selector
(`paper_primary_coco` family, ignored/uncertain GT as iscrowd=1), and the
pinned TinyPerson official evaluator (`benchmark_official` family).

Usage:
    python paper_a/tools/train_tinyperson_pilot.py --method sa_alw_full
    python paper_a/tools/train_tinyperson_pilot.py --method standard --epochs 12
    python paper_a/tools/train_tinyperson_pilot.py --method alw_canonical \
        --epochs 1 --limit-train 8 --limit-val 4 --batch-size 1   # smoke
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.config import (  # noqa: E402
    LR, MOMENTUM, WEIGHT_DECAY,
    WARMUP_EPOCHS, WARMUP_START_LR,
    MIN_SIZE, MAX_SIZE,
    SCORE_THRESH_TRAIN, NMS_THRESH_TEST, BOX_DETECTIONS_PER_IMG,
    RPN_FG_IOU, RPN_BG_IOU, ROI_FG_IOU_THRESH, ROI_BG_IOU_THRESH,
    seed_all,
)
from common.metrics import configure_metric, get_metric_fn  # noqa: E402
from common.model import build_model  # noqa: E402
from common.train_utils import WarmupCosineLR, train_one_epoch  # noqa: E402
from paper_a.datasets.tinyperson_original import TinyPersonOriginalDataset  # noqa: E402
from paper_a.evaluation.tinyperson_official import (  # noqa: E402
    EVALUATOR_COMMIT, EVALUATOR_SHA256, evaluate_tinyperson_official,
)

DEFAULT_DATA_ROOT = Path(r"D:\paper_a_data\TinyPerson\tiny_set")
DEFAULT_SPLITS = ROOT / "paper_a" / "splits"
DEFAULT_SCHEDULE = ROOT / "paper_a" / "schedules" / "tinyperson_train_p10_p90.json"

METHODS = (
    "standard", "igwd", "alw_canonical",
    "sa_alw_beta_only", "sa_alw_pos_only", "sa_alw_full",
    "rfla", "nwd",
)

# Frozen reference endpoints (schedules/endpoint_protocol.md).
BETA_MIN, BETA_MAX = 8.0, 10.0
W_MIN, W_MAX = 1.0, 1.5


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


class FlipTransform:
    """Seeded random horizontal flip; the only pilot augmentation."""

    def __init__(self, seed: int, probability: float = 0.5) -> None:
        self.rng = random.Random(seed)
        self.probability = probability

    def __call__(self, image, target):
        if self.rng.random() < self.probability and len(target["boxes"]) > 0:
            width = image.width
            boxes = target["boxes"].clone()
            boxes[:, 0] = width - target["boxes"][:, 2]
            boxes[:, 2] = width - target["boxes"][:, 0]
            target = dict(target)
            target["boxes"] = boxes
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        tensor = pil_to_tensor(image).to(dtype=torch.float32) / 255.0
        return tensor, target


def load_schedule(schedule_path: Path) -> dict:
    payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    bounds = payload["schedule_bounds"]
    if (bounds["lower_percentile"], bounds["upper_percentile"]) != (10, 90):
        raise ValueError("The pilot reference schedule requires the P10/P90 bounds")
    return {
        "s_min": float(bounds["s_min"]),
        "s_max": float(bounds["s_max"]),
        "beta_min": BETA_MIN,
        "beta_max": BETA_MAX,
        "w_min": W_MIN,
        "w_max": W_MAX,
        "schedule_form": "linear",
        "bounds_audit_sha256": payload["audit_sha256"],
        "bounds_file_sha256": file_sha256(schedule_path),
    }


def build_method_model(method: str, schedule: dict) -> torch.nn.Module:
    # TinyPerson is binary task-all: one foreground class only.
    common_kwargs = {"num_classes": 1}
    if method == "standard":
        return build_model(
            metric_fn=None, placement="everywhere",
            box_loss_type="smooth_l1", box_loss_warmup_epochs=0,
            **common_kwargs,
        )
    if method == "igwd":
        return build_model(
            metric_fn=get_metric_fn("igwd"), placement="la_loss",
            box_loss_type="metric", **common_kwargs,
        )
    if method == "rfla":
        # RFLA: receptive-field hierarchical assignment with CIoU similarity,
        # Smooth-L1 regression. Hyperparams from RFLA paper: k=3, beta=0.9,
        # dynamic-k per scale, quality_ratio=0.60.
        return build_model(
            metric_fn=get_metric_fn("ciou"), placement="la",
            box_loss_type="smooth_l1", box_loss_warmup_epochs=0,
            **common_kwargs,
        )
    if method == "nwd":
        # NWD: Normalized Wasserstein Distance (Wang et al., CVPR 2022).
        # Fidelity: official formula exp(-W2/C), no beta multiplier.
        # Override registry default beta=8.0 to beta=1.0.
        from functools import partial
        nwd_fn = get_metric_fn("nwd")
        nwd_faithful = partial(nwd_fn, beta=1.0)
        return build_model(
            metric_fn=nwd_faithful, placement="la_loss_nms",
            box_loss_type="metric", **common_kwargs,
        )
    canonical_name = {
        "alw_canonical": "alw_canonical",
        "sa_alw_beta_only": "sa_alw_canonical_beta_only",
        "sa_alw_pos_only": "sa_alw_canonical_pos_only",
        "sa_alw_full": "sa_alw_canonical",
    }[method]
    if method == "alw_canonical":
        similarity, distance, _ = configure_metric(
            canonical_name, beta=BETA_MIN)
    else:
        similarity, distance, _ = configure_metric(
            canonical_name,
            s_min=schedule["s_min"], s_max=schedule["s_max"],
            beta_min=schedule["beta_min"], beta_max=schedule["beta_max"],
            w_min=schedule["w_min"], w_max=schedule["w_max"],
            schedule_form=schedule["schedule_form"],
        )
    return build_model(
        metric_fn=similarity, metric_distance_fn=distance,
        placement="la_loss", box_loss_type="metric", **common_kwargs,
    )


def build_datasets(args, transform_seed: int):
    image_root = args.data_root / "erase_with_uncertain_dataset" / "train"
    train_ann = args.splits_dir / "tinyperson_train_sub.json"
    val_ann = args.splits_dir / "tinyperson_val.json"
    train_ds = TinyPersonOriginalDataset(
        image_root, train_ann,
        transform=FlipTransform(transform_seed),
    )
    val_ds = TinyPersonOriginalDataset(image_root, val_ann)
    if args.limit_train > 0:
        train_ds.records = train_ds.records[: args.limit_train]
    if args.limit_val > 0:
        val_ds.records = val_ds.records[: args.limit_val]
    return train_ds, val_ds, val_ann


def collate_fn(batch):
    return tuple(zip(*batch))


def build_paper_primary_gt(val_ds, output_path: Path) -> dict:
    """paper_primary_coco GT: positives iscrowd=0, ignored/uncertain iscrowd=1."""
    images, annotations, ann_id = [], [], 1
    for record in val_ds.records:
        image = record["image"]
        images.append({
            "id": int(image["id"]), "width": int(image["width"]),
            "height": int(image["height"]), "file_name": image["file_name"],
        })
        for group, iscrowd in ((record["positives"], 0), (record["ignored"], 1)):
            for ann in group:
                x, y, w, h = ann["bbox"]
                annotations.append({
                    "id": ann_id, "image_id": int(image["id"]),
                    "category_id": 1, "bbox": [x, y, w, h],
                    "area": float(ann.get("area", w * h)),
                    "iscrowd": iscrowd,
                })
                ann_id += 1
    payload = {
        "info": {"description": "Paper A WP01 paper_primary_coco validation GT"},
        "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def evaluate_model(model, val_ds, val_ann: Path, gt_path: Path,
                   device: torch.device) -> dict:
    """Run both frozen evaluation families on the validation crops."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    model.eval()
    predictions, image_ids = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            predictions.extend(outputs)
            image_ids.extend(int(target["image_id"].item()) for target in targets)
    detections = val_ds.predictions_to_coco(predictions, image_ids)

    gt_coco = COCO(str(gt_path))
    if detections:
        dt_coco = gt_coco.loadRes(detections)
        evaluator = COCOeval(gt_coco, dt_coco, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        paper_primary_ap = float(evaluator.stats[0])
        paper_primary = {
            "AP": paper_primary_ap,
            "AP50": float(evaluator.stats[1]),
            "AP75": float(evaluator.stats[2]),
            "AP_small": float(evaluator.stats[3]),
            "AP_medium": float(evaluator.stats[4]),
            "AP_large": float(evaluator.stats[5]),
        }
    else:
        paper_primary_ap = 0.0
        paper_primary = {key: 0.0 for key in (
            "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")}

    if detections:
        official = evaluate_tinyperson_official(val_ann, detections)
    else:
        official = {
            "protocol": "tinyperson_official",
            "evaluator_commit": EVALUATOR_COMMIT,
            "evaluator_sha256": EVALUATOR_SHA256,
            "prediction_count": 0,
            "metrics": {}, "summary": "",
        }
    return {
        "paper_primary_coco": {
            "protocol": "paper_primary_coco",
            "parameters": {
                "iou_thresholds": [0.5 + 0.05 * i for i in range(10)],
                "max_detections": [1, 10, 100],
                "ignored_gt_as_iscrowd": True,
            },
            "prediction_count": len(detections),
            "metrics": paper_primary,
        },
        "benchmark_official": official,
        "detections": detections,
        "selector_ap": paper_primary_ap,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "paper_a" / "runs")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    seed_all(args.seed)

    schedule = load_schedule(args.schedule) if args.method != "standard" else {}
    run_id = f"wp01_pilot_{args.method}__seed{args.seed}"
    if args.tag:
        run_id += f"__{args.tag}"
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, val_ann = build_datasets(args, transform_seed=args.seed)
    gt_path = output_dir / "val_gt_paper_primary.json"
    build_paper_primary_gt(val_ds, gt_path)

    config = {
        "work_package": "WP01",
        "shards": ["PILOT-D1-S42", "PILOT-COMP-D1-S42"],
        "method": args.method,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": {"type": "SGD", "lr": LR, "momentum": MOMENTUM,
                      "weight_decay": WEIGHT_DECAY},
        "lr_schedule": {"type": "warmup_cosine",
                        "warmup_epochs": WARMUP_EPOCHS,
                        "warmup_start_lr": WARMUP_START_LR},
        "ema": False,
        "augmentation": "horizontal_flip_p0.5_only",
        "detector": {
            "backbone": "resnet50_fpn_pretrained",
            "min_size": MIN_SIZE, "max_size": MAX_SIZE,
            "rpn_fg_iou": RPN_FG_IOU, "rpn_bg_iou": RPN_BG_IOU,
            "roi_fg_iou": ROI_FG_IOU_THRESH, "roi_bg_iou": ROI_BG_IOU_THRESH,
            "score_thresh_test": SCORE_THRESH_TRAIN,
            "nms_thresh_test": NMS_THRESH_TEST,
            "detections_per_img": BOX_DETECTIONS_PER_IMG,
        },
        "checkpoint_selector": "validation paper_primary_coco AP (max over epochs)",
        "evaluation_families": ["paper_primary_coco", "benchmark_official"],
        "schedule": schedule,
        "dataset": {
            "name": "TinyPerson binary task-all (erase_with_uncertain)",
            "split_protocol": "PL-001",
            "train_annotation_sha256": file_sha256(
                args.splits_dir / "tinyperson_train_sub.json"),
            "val_annotation_sha256": file_sha256(
                args.splits_dir / "tinyperson_val.json"),
            "train_records": len(train_ds),
            "val_records": len(val_ds),
        },
        "code": {
            "trainer_sha256": file_sha256(Path(__file__)),
            "git_commit": git_commit(),
        },
    }
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    config["config_sha256"] = config_sha256
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")

    model = build_method_model(args.method, schedule).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=WARMUP_START_LR,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    scheduler = WarmupCosineLR(optimizer, WARMUP_EPOCHS, args.epochs,
                               LR, WARMUP_START_LR)

    csv_path = output_dir / "metrics.csv"
    fields = ["epoch", "train_loss", "selector_ap", "ap50_official",
              "seconds", "lr"]
    best_ap, best_epoch = -1.0, 0
    print(f"[pilot] {run_id} | device={device} | train={len(train_ds)} "
          f"val={len(val_ds)} | config_sha256={config_sha256}")

    for epoch in range(1, args.epochs + 1):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn,
            generator=generator, drop_last=False)
        model.roi_heads._current_epoch = epoch
        start = time.time()
        train_loss, _ = train_one_epoch(
            model, optimizer, train_loader, scaler, device, epoch)
        scheduler.step_epoch()

        evaluation = evaluate_model(model, val_ds, val_ann, gt_path, device)
        selector_ap = evaluation["selector_ap"]
        elapsed = time.time() - start
        official_metrics = evaluation["benchmark_official"].get("metrics", {})
        row = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "selector_ap": round(selector_ap, 6),
            "ap50_official": official_metrics.get("AP50_all", ""),
            "seconds": round(elapsed, 2),
            "lr": optimizer.param_groups[0]["lr"],
        }
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"[pilot] ep {epoch}/{args.epochs} | loss={train_loss:.4f} | "
              f"selector_AP={selector_ap:.4f} | {elapsed:.0f}s")

        if selector_ap > best_ap:
            best_ap, best_epoch = selector_ap, epoch
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "selector_ap": selector_ap,
                "config_sha256": config_sha256,
            }, output_dir / "best.pt")

    print(f"[pilot] best selector AP={best_ap:.4f} @ epoch {best_epoch}; "
          f"reloading checkpoint independently")
    reload_model = build_method_model(args.method, schedule).to(device)
    state = torch.load(output_dir / "best.pt", map_location=device)
    incompatible = reload_model.load_state_dict(state["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict pilot checkpoint reload failed")
    final = evaluate_model(reload_model, val_ds, val_ann, gt_path, device)
    (output_dir / "detections_best.json").write_text(
        json.dumps(final["detections"]), encoding="utf-8")
    final.pop("detections", None)

    results = {
        "run_id": run_id,
        "config_sha256": config_sha256,
        "best_epoch": best_epoch,
        "best_selector_ap_in_loop": best_ap,
        "reloaded_evaluation": final,
        "test_access": "validation_only",
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"[pilot] done: {output_dir}")


if __name__ == "__main__":
    main()
