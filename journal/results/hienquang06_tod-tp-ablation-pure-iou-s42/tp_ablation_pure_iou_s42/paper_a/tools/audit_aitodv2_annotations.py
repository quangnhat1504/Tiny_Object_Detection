from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FILES = {
    "train": "aitodv2_train.json",
    "val": "aitodv2_val.json",
    "trainval": "aitodv2_trainval.json",
    "test": "aitodv2_test.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def audit_file(path: Path) -> tuple[dict[str, Any], set[str], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload["images"]
    annotations = payload["annotations"]
    categories = payload["categories"]
    image_by_id = {image["id"]: image for image in images}
    category_ids = {category["id"] for category in categories}
    if len(image_by_id) != len(images):
        raise AssertionError(f"duplicate image IDs in {path.name}")

    category_counts: Counter[int] = Counter()
    valid_category_counts: Counter[int] = Counter()
    invalid_reasons: Counter[str] = Counter()
    valid_scales: list[float] = []
    images_with_valid_gt: set[int] = set()
    annotation_ids: set[int] = set()

    for annotation in annotations:
        annotation_id = annotation["id"]
        if annotation_id in annotation_ids:
            invalid_reasons["duplicate_annotation_id"] += 1
        annotation_ids.add(annotation_id)
        image = image_by_id.get(annotation["image_id"])
        if image is None:
            invalid_reasons["missing_image_reference"] += 1
            continue
        category_id = annotation["category_id"]
        category_counts[category_id] += 1
        if category_id not in category_ids:
            invalid_reasons["unknown_category"] += 1
            continue
        x, y, width, height = annotation["bbox"]
        intersection_width = max(
            0.0, min(x + width, image["width"]) - max(x, 0.0)
        )
        intersection_height = max(
            0.0, min(y + height, image["height"]) - max(y, 0.0)
        )
        if annotation.get("ignore", False):
            invalid_reasons["ignore"] += 1
            continue
        if intersection_width * intersection_height == 0:
            invalid_reasons["no_image_intersection"] += 1
            continue
        if annotation.get("area", width * height) <= 0:
            invalid_reasons["nonpositive_area"] += 1
            continue
        if width < 1 or height < 1:
            invalid_reasons["width_or_height_below_one"] += 1
            continue
        if annotation.get("iscrowd", False):
            invalid_reasons["crowd"] += 1
            continue
        valid_category_counts[category_id] += 1
        images_with_valid_gt.add(annotation["image_id"])
        valid_scales.append(math.sqrt(width * height))

    size_histogram = Counter(
        f"{image['width']}x{image['height']}" for image in images
    )
    quantiles = {
        f"p{int(q * 100):02d}": percentile(valid_scales, q)
        for q in (0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95)
    }
    summary = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "images": len(images),
        "annotations": len(annotations),
        "categories": categories,
        "category_counts": dict(sorted(category_counts.items())),
        "valid_category_counts": dict(sorted(valid_category_counts.items())),
        "valid_annotations_under_official_loader_contract": sum(
            valid_category_counts.values()
        ),
        "images_with_valid_gt": len(images_with_valid_gt),
        "empty_images_after_filter": len(images) - len(images_with_valid_gt),
        "filtered_reason_counts": dict(sorted(invalid_reasons.items())),
        "image_size_histogram": dict(sorted(size_histogram.items())),
        "sqrt_bbox_area_px_quantiles_after_filter": quantiles,
    }
    return summary, {image["file_name"] for image in images}, images


def write_manifest(
    path: Path,
    audited: dict[str, dict[str, Any]],
    images_by_split: dict[str, list[dict[str, Any]]],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "dataset_version",
                "split",
                "official_image_id",
                "file_name",
                "width",
                "height",
                "annotation_file_sha256",
                "image_sha256",
                "image_status",
            ],
        )
        writer.writeheader()
        for split in ("train", "val", "test"):
            for image in sorted(
                images_by_split[split], key=lambda item: item["file_name"]
            ):
                writer.writerow(
                    {
                        "dataset": "AI-TOD-v2",
                        "dataset_version": "2.0 official annotations",
                        "split": split,
                        "official_image_id": image["id"],
                        "file_name": image["file_name"],
                        "width": image["width"],
                        "height": image["height"],
                        "annotation_file_sha256": audited[split]["sha256"],
                        "image_sha256": "",
                        "image_status": "NOT_ACQUIRED",
                    }
                )
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "paper_a" / "datasets" / "aitodv2_annotation_audit.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "paper_a" / "splits" / "aitodv2_annotation_manifest.csv",
    )
    args = parser.parse_args()
    annotation_root = args.root.resolve()

    audited: dict[str, dict[str, Any]] = {}
    filenames: dict[str, set[str]] = {}
    images_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, filename in EXPECTED_FILES.items():
        path = annotation_root / filename
        if not path.exists():
            raise FileNotFoundError(path)
        audited[split], filenames[split], images_by_split[split] = audit_file(path)

    if filenames["train"] & filenames["val"]:
        raise AssertionError("official train and validation filenames overlap")
    if filenames["trainval"] != filenames["train"] | filenames["val"]:
        raise AssertionError("trainval is not the exact train/validation union")
    if filenames["trainval"] & filenames["test"]:
        raise AssertionError("official trainval and test filenames overlap")

    manifest_hash = write_manifest(args.manifest, audited, images_by_split)
    output = {
        "status": "ANNOTATIONS_ACQUIRED_IMAGES_PENDING",
        "source_url": "https://drive.google.com/drive/folders/1Er14atDO1cBraBD4DSFODZV1x7NHO_PY",
        "evaluator_commit": "44a230ae5197cb89bf9e5e62f313cac3ad30c7af",
        "annotation_root": str(annotation_root),
        "splits": audited,
        "filename_contract": {
            "train_val_overlap": len(filenames["train"] & filenames["val"]),
            "trainval_test_overlap": len(
                filenames["trainval"] & filenames["test"]
            ),
            "trainval_is_exact_train_val_union": True,
        },
        "torchvision_label_mapping": {
            "annotation_category_ids": list(range(8)),
            "training_labels": list(range(1, 9)),
            "rule": "training_label = annotation_category_id + 1",
        },
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "images_acquired": False,
        "g2_ready": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
