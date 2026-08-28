from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detector_resize_shape(
    height: int, width: int, min_size: int, max_size: int
) -> tuple[int, int]:
    if min(height, width, min_size, max_size) <= 0:
        raise ValueError("image and transform dimensions must be positive")
    scale = min(min_size / min(height, width), max_size / max(height, width))
    # torch.nn.functional.interpolate with scale_factor and
    # recompute_scale_factor=True floors the output spatial dimensions.
    return max(1, math.floor(height * scale)), max(1, math.floor(width * scale))


def _is_valid_positive(annotation: dict[str, Any], image: dict[str, Any]) -> bool:
    if (
        annotation.get("ignore", False)
        or annotation.get("uncertain", False)
        or annotation.get("iscrowd", False)
    ):
        return False
    x, y, width, height = annotation["bbox"]
    if width < 1 or height < 1 or annotation.get("area", width * height) <= 0:
        return False
    intersection_width = max(
        0.0, min(x + width, image["width"]) - max(x, 0.0)
    )
    intersection_height = max(
        0.0, min(y + height, image["height"]) - max(y, 0.0)
    )
    return intersection_width * intersection_height > 0


def fit_schedule(
    annotation_file: Path,
    *,
    split: str,
    min_size: int,
    max_size: int,
    lower_percentile: float,
    upper_percentile: float,
) -> dict[str, Any]:
    if split != "train":
        raise ValueError("scale schedules may be fitted only on split=train")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")

    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in payload["images"]}
    resize_by_image: dict[int, tuple[float, float]] = {}
    shape_counts: dict[str, int] = {}
    for image_id, image in images.items():
        height, width = int(image["height"]), int(image["width"])
        resized_height, resized_width = detector_resize_shape(
            height, width, min_size, max_size
        )
        resize_by_image[image_id] = (
            resized_width / width,
            resized_height / height,
        )
        key = f"{width}x{height}->{resized_width}x{resized_height}"
        shape_counts[key] = shape_counts.get(key, 0) + 1

    scales: list[float] = []
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        image = images[image_id]
        if not _is_valid_positive(annotation, image):
            continue
        _, _, width, height = [float(value) for value in annotation["bbox"]]
        scale_x, scale_y = resize_by_image[image_id]
        scales.append(math.sqrt(width * scale_x * height * scale_y))

    if not scales:
        raise ValueError("training annotation contains no valid positive boxes")
    values = np.asarray(scales, dtype=np.float64)
    lower, upper = np.percentile(
        values, [lower_percentile, upper_percentile], method="linear"
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "TRAIN_ONLY_SCHEDULE_AUDIT",
        "annotation_file": annotation_file.name,
        "annotation_sha256": sha256_file(annotation_file),
        "split": split,
        "image_count": len(images),
        "valid_positive_count": len(scales),
        "coordinate_system": "post_torchvision_generalized_rcnn_transform_pixels",
        "transform": {"min_size": min_size, "max_size": max_size},
        "resize_shape_counts": dict(sorted(shape_counts.items())),
        "percentile_method": "numpy_linear",
        "scale_summary_px": {
            "min": float(values.min()),
            "p10": float(np.percentile(values, 10, method="linear")),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90, method="linear")),
            "max": float(values.max()),
        },
        "schedule_bounds": {
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "s_min": float(lower),
            "s_max": float(upper),
        },
        "restrictions": [
            "training annotations only",
            "ignore uncertain and crowd excluded",
            "no beta or position-weight range selected by this audit",
            "dataset-specific bounds are not transferable without preregistration",
        ],
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["audit_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=800)
    parser.add_argument("--lower-percentile", type=float, default=10)
    parser.add_argument("--upper-percentile", type=float, default=90)
    args = parser.parse_args()
    result = fit_schedule(
        args.annotations.resolve(),
        split=args.split,
        min_size=args.min_size,
        max_size=args.max_size,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
