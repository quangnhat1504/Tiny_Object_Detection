"""
AI-TOD-v2 Dataset Adapter for PyTorch & Faster R-CNN Pipelines.
Enforces the official 8-class remote sensing tiny object detection protocol:
1: airplane, 2: bridge, 3: storage-tank, 4: ship, 5: swimming-pool, 6: vehicle, 7: person, 8: wind-mill
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Iterable
from PIL import Image

from paper_a.datasets.coco_original import CocoOriginalDataset

AITODV2_CATEGORIES = [
    {"id": 1, "name": "airplane"},
    {"id": 2, "name": "bridge"},
    {"id": 3, "name": "storage-tank"},
    {"id": 4, "name": "ship"},
    {"id": 5, "name": "swimming-pool"},
    {"id": 6, "name": "vehicle"},
    {"id": 7, "name": "person"},
    {"id": 8, "name": "wind-mill"},
]


class AITODv2Dataset(CocoOriginalDataset):
    """Official AI-TOD-v2 8-class remote sensing dataset with scale metrics."""

    def __init__(
        self,
        image_root: str | Path,
        annotation_file: str | Path,
        *,
        transform: Callable | None = None,
        drop_empty: bool = False,
    ) -> None:
        annotation_path = Path(annotation_file)
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = sorted(payload.get("categories", []), key=lambda item: item["id"])
        
        # Validate 8 categories
        if len(categories) != 8:
            raise ValueError(f"AI-TOD-v2 requires exactly 8 categories; got {len(categories)}")
            
        super().__init__(
            image_root,
            annotation_path,
            transform=transform,
            drop_empty=drop_empty,
        )
        
        # Build robust image lookup index across root hierarchy
        search_root = Path(image_root)
        if not search_root.exists() or not any(search_root.iterdir()):
            search_root = search_root.parent
        if not search_root.exists() or not any(search_root.iterdir()):
            search_root = search_root.parent
        self._image_cache: dict[str, Path] = {}
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
            for p in search_root.rglob(ext):
                self._image_cache[p.name] = p
                
    def _load_image(self, image_info: dict) -> Image.Image:
        file_name = image_info["file_name"]
        raw_name = Path(file_name).name
        if hasattr(self, "_image_cache") and raw_name in self._image_cache:
            with Image.open(self._image_cache[raw_name]) as source:
                return source.convert("RGB")
        return super()._load_image(image_info)

    def _filter_annotations(
        self, image: dict, annotations: Iterable[dict]
    ) -> tuple[list[dict], list[dict]]:
        positives: list[dict] = []
        ignored: list[dict] = []
        for ann in annotations:
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            w, h = float(bbox[2]), float(bbox[3])
            # Exclude degenerate boxes (<= 0 area)
            if w <= 0 or h <= 0:
                continue
            if ann.get("ignore", 0) == 1 or ann.get("iscrowd", 0) == 1:
                ignored.append(ann)
            else:
                positives.append(ann)
        return positives, ignored
