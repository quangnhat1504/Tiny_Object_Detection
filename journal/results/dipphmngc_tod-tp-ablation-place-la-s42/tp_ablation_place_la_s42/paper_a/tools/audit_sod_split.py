from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data"
DEFAULT_MANIFEST = ROOT / "paper_a" / "splits" / "legacy_split_manifest.csv"
DEFAULT_AUDIT = ROOT / "paper_a" / "splits" / "split_audit.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ROBOFLOW_SUFFIX = re.compile(r"_jpg\.rf\.([0-9a-f]+)$", re.IGNORECASE)
VIDEO_SOURCE = re.compile(
    r"^(bb|youtube|chitube)_V(\d+)_I(\d+)$", re.IGNORECASE
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_and_augmentation(path: Path) -> tuple[str, str]:
    match = ROBOFLOW_SUFFIX.search(path.stem)
    if not match:
        return path.stem, "unknown"
    return path.stem[: match.start()], match.group(1)


def sequence_id(source_id: str) -> str:
    match = VIDEO_SOURCE.match(source_id)
    if match:
        return f"{match.group(1).lower()}_V{match.group(2)}"
    return source_id


def count_objects(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    return sum(bool(line.strip()) for line in label_path.read_text().splitlines())


def build_rows(data_root: Path) -> list[dict[str, str | int]]:
    rows = []
    exposure = {
        "train": "historical_train",
        "valid": "historical_validation_tuned",
        "test": "historical_test_reused",
    }
    for split in ("train", "valid", "test"):
        image_dir = data_root / split / "images"
        label_dir = data_root / split / "labels"
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            source_id, augmentation_id = source_and_augmentation(image_path)
            label_path = label_dir / f"{image_path.stem}.txt"
            with Image.open(image_path) as image:
                width, height = image.size
            rows.append(
                {
                    "source_id": source_id,
                    "sequence_id": sequence_id(source_id),
                    "original_path": "",
                    "derived_path": image_path.relative_to(ROOT).as_posix(),
                    "split": f"legacy_{split}",
                    "augmentation_id": augmentation_id,
                    "image_hash": sha256(image_path),
                    "annotation_hash": sha256(label_path) if label_path.exists() else "MISSING",
                    "width": width,
                    "height": height,
                    "num_objects": count_objects(label_path),
                    "exposure_status": exposure[split],
                }
            )
    return rows


def pairwise_overlap(sets: dict[str, set[str]]) -> dict[str, dict[str, object]]:
    result = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = sorted(sets[left] & sets[right])
        result[f"{left}_{right}"] = {"count": len(overlap), "values": overlap}
    return result


def audit(rows: list[dict[str, str | int]], manifest_hash: str) -> dict:
    by_split = {
        split: [row for row in rows if row["split"] == f"legacy_{split}"]
        for split in ("train", "valid", "test")
    }
    source_sets = {
        split: {str(row["source_id"]) for row in split_rows}
        for split, split_rows in by_split.items()
    }
    sequence_sets = {
        split: {str(row["sequence_id"]) for row in split_rows}
        for split, split_rows in by_split.items()
    }
    image_hash_sets = {
        split: {str(row["image_hash"]) for row in split_rows}
        for split, split_rows in by_split.items()
    }
    counts = {}
    for split, split_rows in by_split.items():
        variant_counts = Counter(str(row["source_id"]) for row in split_rows)
        counts[split] = {
            "processed_images": len(split_rows),
            "independent_source_ids": len(source_sets[split]),
            "sequence_groups": len(sequence_sets[split]),
            "objects_in_processed_annotations": sum(
                int(row["num_objects"]) for row in split_rows
            ),
            "variants_per_source_histogram": {
                str(key): value
                for key, value in sorted(Counter(variant_counts.values()).items())
            },
        }
    sequence_overlap = pairwise_overlap(sequence_sets)
    source_overlap = pairwise_overlap(source_sets)
    exact_image_overlap = pairwise_overlap(image_hash_sets)
    return {
        "status": "NO_GO_CURRENT_DERIVATIVE",
        "manifest_sha256": manifest_hash,
        "counts": counts,
        "total_processed_images": len(rows),
        "total_independent_source_ids": len(
            {str(row["source_id"]) for row in rows}
        ),
        "source_id_overlap": source_overlap,
        "sequence_id_overlap": sequence_overlap,
        "exact_image_hash_overlap": exact_image_overlap,
        "missing_annotations": sum(
            row["annotation_hash"] == "MISSING" for row in rows
        ),
        "submission_test_available": False,
        "reasons": [
            "video/sequence groups overlap across all legacy splits",
            "all current splits have prior train/tuning/test exposure",
            "legacy evaluation treats tiles as independent COCO images",
            "upstream original paths and ignore/crowd provenance are unavailable",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    rows = build_rows(args.data_root.resolve())
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest_hash = sha256(args.manifest)
    result = audit(rows, manifest_hash)
    args.audit.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

