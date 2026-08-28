"""Build a deterministic source/video-disjoint TinyPerson Program B split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


FRAME_SUFFIX = re.compile(r"_I\d+$")
VIDEO_PATTERN = re.compile(r"_V\d+_I\d+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_identity(file_name: str) -> tuple[str, str]:
    stem = Path(file_name).stem
    if VIDEO_PATTERN.search(stem):
        return "video", FRAME_SUFFIX.sub("", stem)
    return "image", stem


def group_sort_key(namespace: str, kind: str, identity: str) -> str:
    return hashlib.sha256(f"{namespace}:{kind}:{identity}".encode()).hexdigest()


def build_program_b_split(
    annotation_file: str | Path,
    output_dir: str | Path,
    *,
    namespace: str,
    val_fraction: float,
) -> dict[str, Any]:
    """Split one original-image annotation into disjoint train and validation sets."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")

    annotation_path = Path(annotation_file).resolve()
    output_path = Path(output_dir).resolve()
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = list(payload["images"])
    image_ids = {int(image["id"]) for image in images}
    if len(image_ids) != len(images):
        raise ValueError("image ids must be unique")

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for image in images:
        groups[group_identity(str(image["file_name"]))].append(int(image["id"]))

    val_ids: set[int] = set()
    membership: dict[str, dict[str, list[str]]] = {
        "train": {"video": [], "image": []},
        "val": {"video": [], "image": []},
    }
    for kind in ("video", "image"):
        identities = sorted(
            (identity for current_kind, identity in groups if current_kind == kind),
            key=lambda identity: group_sort_key(namespace, kind, identity),
        )
        if not identities:
            continue
        val_count = max(1, round(len(identities) * val_fraction))
        for index, identity in enumerate(identities):
            side = "val" if index < val_count else "train"
            membership[side][kind].append(identity)
            if side == "val":
                val_ids.update(groups[(kind, identity)])

    train_ids = image_ids - val_ids
    if not train_ids or not val_ids:
        raise ValueError("split requires non-empty train and validation image sets")
    if train_ids & val_ids:
        raise AssertionError("source split image sets overlap")

    def subset(selected_ids: set[int]) -> dict[str, Any]:
        return {
            "images": [image for image in images if int(image["id"]) in selected_ids],
            "annotations": [
                annotation
                for annotation in payload["annotations"]
                if int(annotation["image_id"]) in selected_ids
            ],
            "categories": payload["categories"],
        }

    output_path.mkdir(parents=True, exist_ok=True)
    train_path = output_path / "program_b_train.json"
    val_path = output_path / "program_b_val.json"
    train_path.write_text(json.dumps(subset(train_ids)), encoding="utf-8")
    val_path.write_text(json.dumps(subset(val_ids)), encoding="utf-8")

    source_groups = {
        f"{kind}:{identity}": sorted(ids) for (kind, identity), ids in sorted(groups.items())
    }
    train_groups = {
        f"{kind}:{identity}"
        for kind, identities in membership["train"].items()
        for identity in identities
    }
    val_groups = {
        f"{kind}:{identity}"
        for kind, identities in membership["val"].items()
        for identity in identities
    }
    report = {
        "status": "FROZEN_PROGRAM_B_VALIDATION_SPLIT",
        "namespace": namespace,
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "val_fraction": val_fraction,
        "group_rule": "video = stem without trailing _I<frame>; image = full stem",
        "split_order": "sha256(namespace:kind:identity)",
        "source_groups": source_groups,
        "membership": membership,
        "source_group_overlap": sorted(train_groups & val_groups),
        "counts": {
            "train": {"images": len(train_ids), "annotations": len(subset(train_ids)["annotations"])},
            "val": {"images": len(val_ids), "annotations": len(subset(val_ids)["annotations"])},
        },
        "train_annotation_sha256": sha256_file(train_path),
        "val_annotation_sha256": sha256_file(val_path),
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report_path = output_path / "program_b_split_manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    print(
        json.dumps(
            build_program_b_split(
                args.annotations,
                args.output_dir,
                namespace=args.namespace,
                val_fraction=args.val_fraction,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
