"""Convert frozen Program B COCO originals into baseline-geometry YOLO tiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


TILE_SIZE = 512
TILE_OVERLAP = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tile_starts(length: int) -> list[int]:
    stride = TILE_SIZE - TILE_OVERLAP
    return list(range(0, max(1, length - TILE_SIZE + stride), stride))


def clipped_box(
    bbox: list[float],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[float, float, float, float] | None:
    x, y, width, height = (float(value) for value in bbox)
    source_area = max(width * height, 1e-6)
    left = max(x, x1) - x1
    top = max(y, y1) - y1
    right = min(x + width, x2) - x1
    bottom = min(y + height, y2) - y1
    if right <= left or bottom <= top:
        return None
    visible_area = (right - left) * (bottom - top)
    minimum_visibility = 0.02 if source_area < 64 else 0.05 if source_area < 256 else 0.20
    if visible_area / source_area < minimum_visibility:
        return None
    return left, top, right, bottom


def build_coco_tiles(
    annotation_file: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    *,
    side: str,
) -> dict[str, Any]:
    """Write one baseline-geometry tile corpus and reversible tile manifest."""
    annotation_path = Path(annotation_file).resolve()
    image_root_path = Path(image_root).resolve()
    output_path = Path(output_dir).resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_path}")

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = payload.get("categories", [])
    if len(categories) != 1 or int(categories[0].get("id", -1)) != 1:
        raise ValueError("Program B expects TinyPerson's one-class category_id=1 annotation")

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        if annotation.get("ignore", False) or annotation.get("uncertain", False) or annotation.get("iscrowd", False):
            raise ValueError("Program B tile export requires no ignored/uncertain/crowd annotations")
        if int(annotation["category_id"]) != 1:
            raise ValueError("unexpected category id")
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    images_path = output_path / "images"
    labels_path = output_path / "labels"
    images_path.mkdir(parents=True)
    labels_path.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for image_info in sorted(payload["images"], key=lambda image: int(image["id"])):
        original_id = int(image_info["id"])
        image_path = image_root_path / str(image_info["file_name"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        if (width, height) != (int(image_info["width"]), int(image_info["height"])):
            raise ValueError(f"image metadata mismatch: {image_path}")
        for y_start in tile_starts(height):
            for x_start in tile_starts(width):
                x1 = min(x_start, max(0, width - TILE_SIZE))
                y1 = min(y_start, max(0, height - TILE_SIZE))
                x2 = min(x1 + TILE_SIZE, width)
                y2 = min(y1 + TILE_SIZE, height)
                name = f"{original_id}_x{x1}_y{y1}"
                tile = image.crop((x1, y1, x2, y2))
                tile.save(images_path / f"{name}.jpg", quality=95, subsampling=0)
                labels: list[str] = []
                for annotation in annotations_by_image[original_id]:
                    box = clipped_box(annotation["bbox"], x1, y1, x2, y2)
                    if box is None:
                        continue
                    left, top, right, bottom = box
                    tile_width = x2 - x1
                    tile_height = y2 - y1
                    center_x = (left + right) / 2 / tile_width
                    center_y = (top + bottom) / 2 / tile_height
                    box_width = (right - left) / tile_width
                    box_height = (bottom - top) / tile_height
                    labels.append(
                        f"0 {center_x:.10f} {center_y:.10f} {box_width:.10f} {box_height:.10f}"
                    )
                (labels_path / f"{name}.txt").write_text(
                    "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
                )
                records.append(
                    {
                        "tile_name": f"{name}.jpg",
                        "original_image_id": original_id,
                        "original_file_name": str(image_info["file_name"]),
                        "original_width": width,
                        "original_height": height,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "label_count": len(labels),
                    }
                )

    manifest = {
        "status": "FROZEN_PROGRAM_B_COCO_TILES",
        "side": side,
        "source_annotation_sha256": sha256_file(annotation_path),
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "original_count": len(payload["images"]),
        "tile_count": len(records),
        "tiles": records,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output_path / "tile_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("train", "validation"), required=True)
    args = parser.parse_args()
    report = build_coco_tiles(
        args.annotations,
        args.image_root,
        args.output_dir,
        side=args.side,
    )
    print(json.dumps({key: report[key] for key in report if key != "tiles"}, indent=2))


if __name__ == "__main__":
    main()
