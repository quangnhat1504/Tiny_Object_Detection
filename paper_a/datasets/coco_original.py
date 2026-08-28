from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class CocoOriginalDataset(Dataset):
    """COCO detection data with explicit category and ignore contracts."""

    def __init__(
        self,
        image_root: str | Path,
        annotation_file: str | Path,
        *,
        transform: Callable | None = None,
        drop_empty: bool = False,
    ) -> None:
        self.image_root = Path(image_root)
        self.annotation_file = Path(annotation_file)
        self.transform = transform
        payload = json.loads(self.annotation_file.read_text(encoding="utf-8"))
        self.categories = sorted(payload["categories"], key=lambda item: item["id"])
        self.category_id_to_label = {
            category["id"]: index + 1
            for index, category in enumerate(self.categories)
        }
        self.label_to_category_id = {
            label: category_id
            for category_id, label in self.category_id_to_label.items()
        }
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in payload["annotations"]:
            annotations_by_image[annotation["image_id"]].append(annotation)

        self.records: list[dict] = []
        for image in payload["images"]:
            positives, ignored = self._filter_annotations(
                image, annotations_by_image.get(image["id"], [])
            )
            if drop_empty and not positives:
                continue
            self.records.append(
                {"image": image, "positives": positives, "ignored": ignored}
            )

    def _filter_annotations(
        self, image: dict, annotations: Iterable[dict]
    ) -> tuple[list[dict], list[dict]]:
        positives: list[dict] = []
        ignored: list[dict] = []
        for annotation in annotations:
            if annotation.get("ignore", False):
                continue
            category_id = annotation["category_id"]
            if category_id not in self.category_id_to_label:
                continue
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
            if annotation.get("iscrowd", False):
                ignored.append(annotation)
            else:
                positives.append(annotation)
        return positives, ignored

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _boxes(annotations: Sequence[dict]) -> Tensor:
        boxes = [
            [x, y, x + width, y + height]
            for x, y, width, height in (
                annotation["bbox"] for annotation in annotations
            )
        ]
        return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    def _load_image(self, image_info: dict) -> Image.Image:
        file_name = image_info["file_name"]
        raw_name = Path(file_name).name
        candidates = [
            self.image_root / file_name,
            self.image_root / raw_name,
            self.image_root.parent / file_name,
            self.image_root.parent / raw_name,
            self.image_root.parent / "train" / raw_name,
            self.image_root.parent / "val" / raw_name,
            self.image_root.parent / "test" / raw_name,
            self.image_root.parent.parent / "images" / raw_name,
            self.image_root.parent.parent / "images" / "train" / raw_name,
            self.image_root.parent.parent / "images" / "val" / raw_name,
            self.image_root.parent.parent / "images" / "test" / raw_name,
        ]
        for path in candidates:
            if path.is_file():
                with Image.open(path) as source:
                    return source.convert("RGB")
                    
        # Robust fallback: build in-memory file index if direct lookup fails
        if not hasattr(self, "_file_index"):
            self._file_index = {}
            for p in self.image_root.rglob("*.*"):
                if p.is_file():
                    self._file_index[p.name] = p
                    clean_k = "".join(c for c in p.name.lower() if c.isalnum())
                    self._file_index[clean_k] = p
            if self.image_root.parent.exists():
                for p in self.image_root.parent.rglob("*.*"):
                    if p.is_file():
                        self._file_index[p.name] = p
                        clean_k = "".join(c for c in p.name.lower() if c.isalnum())
                        self._file_index[clean_k] = p

        if raw_name in self._file_index:
            with Image.open(self._file_index[raw_name]) as source:
                return source.convert("RGB")
                
        clean_target = "".join(c for c in raw_name.lower() if c.isalnum())
        if clean_target in self._file_index:
            with Image.open(self._file_index[clean_target]) as source:
                return source.convert("RGB")

        with Image.open(candidates[0]) as source:
            return source.convert("RGB")

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        record = self.records[index]
        image_info = record["image"]
        image = self._load_image(image_info)
        positives = record["positives"]
        ignored = record["ignored"]
        boxes = self._boxes(positives)
        labels = torch.tensor(
            [
                self.category_id_to_label[annotation["category_id"]]
                for annotation in positives
            ],
            dtype=torch.int64,
        )
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(image_info["id"], dtype=torch.int64),
            "area": torch.tensor(
                [annotation.get("area", 0.0) for annotation in positives],
                dtype=torch.float32,
            ),
            "iscrowd": torch.zeros(len(positives), dtype=torch.int64),
            "annotation_ids": torch.tensor(
                [annotation["id"] for annotation in positives],
                dtype=torch.int64,
            ),
            "category_ids": torch.tensor(
                [annotation["category_id"] for annotation in positives],
                dtype=torch.int64,
            ),
            "ignore_boxes": self._boxes(ignored),
            "ignore_category_ids": torch.tensor(
                [annotation["category_id"] for annotation in ignored],
                dtype=torch.int64,
            ),
        }
        if self.transform is None:
            image_tensor = pil_to_tensor(image).to(dtype=torch.float32) / 255.0
            return image_tensor, target
        return self.transform(image, target)

    def predictions_to_coco(
        self,
        predictions: Sequence[dict[str, Tensor]],
        image_ids: Sequence[int],
    ) -> list[dict[str, float | int | list[float]]]:
        if len(predictions) != len(image_ids):
            raise ValueError("predictions and image_ids must have equal length")
        results: list[dict[str, float | int | list[float]]] = []
        for prediction, image_id in zip(predictions, image_ids):
            boxes = prediction["boxes"].detach().cpu()
            scores = prediction["scores"].detach().cpu()
            labels = prediction["labels"].detach().cpu()
            if not (len(boxes) == len(scores) == len(labels)):
                raise ValueError("prediction boxes, scores, and labels disagree")
            for box, score, label_tensor in zip(boxes, scores, labels):
                label = int(label_tensor.item())
                if label not in self.label_to_category_id:
                    raise ValueError(f"unknown contiguous training label: {label}")
                x1, y1, x2, y2 = [float(value) for value in box.tolist()]
                results.append(
                    {
                        "image_id": int(image_id),
                        "category_id": self.label_to_category_id[label],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score.item()),
                    }
                )
        return results
