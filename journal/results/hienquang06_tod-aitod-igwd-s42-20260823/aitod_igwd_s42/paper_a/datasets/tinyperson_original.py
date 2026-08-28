from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image

from paper_a.datasets.coco_original import CocoOriginalDataset


class TinyPersonOriginalDataset(CocoOriginalDataset):
    """TinyPerson binary-all data with explicit ignore/uncertain routing."""

    def __init__(
        self,
        image_root: str | Path,
        annotation_file: str | Path,
        *,
        transform: Callable | None = None,
        drop_empty: bool = False,
        require_corner: bool = True,
    ) -> None:
        annotation_path = Path(annotation_file)
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = payload.get("categories", [])
        if len(categories) != 1 or categories[0].get("id") != 1:
            raise ValueError(
                "TinyPerson Paper A requires the official binary task-all annotation"
            )
        if categories[0].get("name", "").lower() != "person":
            raise ValueError("TinyPerson binary task category must be named person")
        if any(image.get("in_dense_image", False) for image in payload["images"]):
            raise ValueError("TinyPerson dense images are excluded by the official task")
        if require_corner and any("corner" not in image for image in payload["images"]):
            raise ValueError(
                "TinyPerson training requires the official corner annotation"
            )
        super().__init__(
            image_root,
            annotation_path,
            transform=transform,
            drop_empty=drop_empty,
        )

    def _load_image(self, image_info: dict) -> Image.Image:
        image = super()._load_image(image_info)
        corner = image_info.get("corner")
        if corner is None:
            return image
        if len(corner) != 4:
            raise ValueError("TinyPerson corner must contain x1,y1,x2,y2")
        x1, y1, x2, y2 = [int(value) for value in corner]
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError("TinyPerson corner is outside the source image")
        if (x2 - x1, y2 - y1) != (
            int(image_info["width"]),
            int(image_info["height"]),
        ):
            raise ValueError("TinyPerson corner size disagrees with image metadata")
        return image.crop((x1, y1, x2, y2))

    def _filter_annotations(
        self, image: dict, annotations: Iterable[dict]
    ) -> tuple[list[dict], list[dict]]:
        positives: list[dict] = []
        ignored: list[dict] = []
        for annotation in annotations:
            if annotation.get("category_id") != 1:
                raise ValueError("TinyPerson task-all annotations must use category_id=1")
            x, y, width, height = annotation["bbox"]
            intersection_width = max(
                0.0, min(x + width, image["width"]) - max(x, 0.0)
            )
            intersection_height = max(
                0.0, min(y + height, image["height"]) - max(y, 0.0)
            )
            if intersection_width * intersection_height == 0:
                continue
            if annotation.get("area", width * height) <= 0:
                continue
            if width < 1 or height < 1:
                continue
            if (
                annotation.get("ignore", False)
                or annotation.get("uncertain", False)
                or annotation.get("iscrowd", False)
            ):
                ignored.append(annotation)
            else:
                positives.append(annotation)
        return positives, ignored
