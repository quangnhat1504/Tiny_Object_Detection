"""Deterministic TinyPerson validation split builder (protocol ledger PL-001).

Builds a video/source-disjoint split of the official erased corner-task train
annotation. Group identity: filename stem without the trailing ``_I<frame>``
suffix for video frames (``bb_V0032_I0001640`` -> ``bb_V0032``); the full
filename stem otherwise. Each group class is ordered by
``sha256("tinyperson_<video|image>_<identity>")`` and the first ``val_fraction``
of each class becomes validation.

Outputs (all in --output-dir):
- tinyperson_validation_split.json : membership, counts, artifact hashes
- tinyperson_train_sub.json        : train-side COCO subset annotation
- tinyperson_val.json              : validation-side COCO subset annotation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from paper_a.tools.fit_train_scale_schedule import sha256_file

FRAME_SUFFIX = re.compile(r"_I\d+$")
VIDEO_PATTERN = re.compile(r"_V\d+_I\d+$")


def group_identity(file_name: str) -> tuple[str, str]:
    stem = Path(file_name).stem
    if VIDEO_PATTERN.search(stem):
        return "video", FRAME_SUFFIX.sub("", stem)
    return "image", stem


def group_hash(kind: str, identity: str) -> str:
    return hashlib.sha256(f"tinyperson_{kind}_{identity}".encode()).hexdigest()


def build_split(
    annotation_file: Path,
    output_dir: Path,
    *,
    val_fraction: float = 0.20,
) -> dict:
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = payload["categories"]
    if len(categories) != 1 or categories[0].get("id") != 1:
        raise ValueError("PL-001 applies only to the binary task-all annotation")

    groups: dict[tuple[str, str], list[int]] = {}
    for image in payload["images"]:
        key = group_identity(image["file_name"])
        groups.setdefault(key, []).append(int(image["id"]))

    sides: dict[str, set[int]] = {"train_sub": set(), "val": set()}
    membership: dict[str, list[str]] = {"video": [], "image": []}
    val_groups: dict[str, list[str]] = {"video": [], "image": []}
    for kind in ("video", "image"):
        ordered = sorted(
            (identity for (k, identity) in groups if k == kind),
            key=lambda identity: group_hash(kind, identity),
        )
        n_val = max(1, round(len(ordered) * val_fraction))
        for rank, identity in enumerate(ordered):
            side = "val" if rank < n_val else "train_sub"
            sides[side].update(groups[(kind, identity)])
            membership[kind].append(identity)
            if side == "val":
                val_groups[kind].append(identity)

    def subset(side: str) -> dict:
        image_ids = sides[side]
        images = [im for im in payload["images"] if int(im["id"]) in image_ids]
        annotations = [
            ann
            for ann in payload["annotations"]
            if int(ann["image_id"]) in image_ids
        ]
        return {"images": images, "annotations": annotations, "categories": categories}

    output_dir.mkdir(parents=True, exist_ok=True)
    train_sub = subset("train_sub")
    val = subset("val")
    train_path = output_dir / "tinyperson_train_sub.json"
    val_path = output_dir / "tinyperson_val.json"
    train_path.write_text(json.dumps(train_sub), encoding="utf-8")
    val_path.write_text(json.dumps(val), encoding="utf-8")

    positives = lambda side: sum(  # noqa: E731
        1 for ann in payload["annotations"] if int(ann["image_id"]) in sides[side]
    )
    report = {
        "protocol_entry": "PL-001",
        "status": "FROZEN_VALIDATION_SPLIT",
        "annotation_file": annotation_file.name,
        "annotation_sha256": sha256_file(annotation_file),
        "val_fraction": val_fraction,
        "group_rule": (
            "video = stem without trailing _I<frame>; image = full stem; "
            "sha256(tinyperson_<kind>_<identity>) ordering per class"
        ),
        "group_counts": {
            kind: len(membership[kind]) for kind in ("video", "image")
        },
        "val_groups": val_groups,
        "sides": {
            side: {"crops": len(sides[side]), "positives": positives(side)}
            for side in ("train_sub", "val")
        },
        "train_sub_annotation_sha256": sha256_file(train_path),
        "val_annotation_sha256": sha256_file(val_path),
        "restrictions": [
            "validation is for run/checkpoint/method selection only",
            "never a submission or final-test surface",
            "A3 official test material remains unmounted",
        ],
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_a/splits"),
    )
    parser.add_argument("--val-fraction", type=float, default=0.20)
    args = parser.parse_args()
    report = build_split(
        args.annotations.resolve(), args.output_dir, val_fraction=args.val_fraction
    )
    report_path = args.output_dir / "tinyperson_validation_split.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
