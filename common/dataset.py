"""YOLOTinyDataset — shared across all experiments."""
from __future__ import annotations
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .config import (
    TILE_SIZE, TILE_OVERLAP, TINY_THRESHOLD_PX,
    USE_COPY_PASTE, COPY_PASTE_PROB, COPY_PASTE_MAX_PER, COPY_PASTE_SCALE_JIT,
    CACHE_IMAGES, TINY_TILE_OVERSAMPLE,
    TRAIN_DIR, VALID_DIR, PATCH_ROOT,
)


class YOLOTinyDataset(Dataset):
    """YOLO-format tiny-object dataset with tiling + cache + copy-paste.

    Args:
        img_dir, lbl_dir: paths
        is_train: applies augmentation if True
        copy_paste_pool: list of (PIL.Image, list of [cls, x1, y1, x2, y2])
                         for copy-paste augmentation.
    """

    def __init__(
        self,
        img_dir: Path,
        lbl_dir: Path,
        is_train: bool = True,
        tile_size: int = TILE_SIZE,
        overlap: int = TILE_OVERLAP,
        copy_paste_pool: Optional[List[Tuple[Image.Image, List[list]]]] = None,
    ):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.is_train = is_train
        self.tile_size = tile_size
        self.stride = tile_size - overlap
        self.img_files = sorted(
            f for f in img_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        self.labels_cache: Dict[int, tuple] = {}
        self.images_cache: Dict[int, Image.Image] = {}
        self.tile_index: List[tuple] = []
        self.tile_has_tiny: List[bool] = []
        self.copy_paste_pool = copy_paste_pool
        self._build_tile_index()
        if CACHE_IMAGES:
            self._preload_images()

    def _preload_images(self):
        n_bytes = 0
        for i, p in enumerate(self.img_files):
            try:
                im = Image.open(p).convert("RGB")
                self.images_cache[i] = im
                n_bytes += len(im.tobytes())
            except Exception:
                pass
        print(f"[Cache] {len(self.images_cache)}/{len(self.img_files)} imgs, "
              f"~{n_bytes/1024/1024:.1f} MB RAM")

    def _build_tile_index(self):
        for img_idx, img_path in enumerate(self.img_files):
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception:
                continue
            boxes = self._parse_yolo_labels(img_path, W, H)
            self.labels_cache[img_idx] = (boxes, W, H)
            for y in range(0, max(1, H - self.tile_size + self.stride), self.stride):
                for x in range(0, max(1, W - self.tile_size + self.stride), self.stride):
                    x1 = min(x, max(0, W - self.tile_size))
                    y1 = min(y, max(0, H - self.tile_size))
                    x2 = min(x1 + self.tile_size, W)
                    y2 = min(y1 + self.tile_size, H)
                    tb = self._clip_boxes(boxes, x1, y1, x2, y2)
                    if self.is_train and len(tb) == 0:
                        continue
                    self.tile_index.append((img_idx, x1, y1, x2, y2))
                    self.tile_has_tiny.append(any(
                        math.sqrt(max((b[3] - b[1]) * (b[4] - b[2]), 0))
                        < TINY_THRESHOLD_PX
                        for b in tb))
        n = sum(self.tile_has_tiny)
        tag = "Train" if self.is_train else "Val"
        print(f"[Dataset] {tag}: {len(self.img_files)} imgs -> "
              f"{len(self.tile_index)} tiles ({n} tiny="
              f"{100*n/max(len(self.tile_index),1):.1f}%)")

    def _parse_yolo_labels(self, img_path: Path, W: int, H: int) -> List[list]:
        lbl = self.lbl_dir / (img_path.stem + ".txt")
        boxes = []
        if not lbl.exists():
            return boxes
        with open(lbl) as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5:
                    continue
                cls_id = int(float(p[0]))
                coords = list(map(float, p[1:]))
                if len(p) == 5:
                    cx, cy, bw, bh = coords
                    x1 = (cx - bw / 2) * W
                    y1 = (cy - bh / 2) * H
                    x2 = (cx + bw / 2) * W
                    y2 = (cy + bh / 2) * H
                else:
                    if len(p) < 7:
                        continue
                    xs = np.array(coords[0::2]) * W
                    ys = np.array(coords[1::2]) * H
                    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
                x1 = float(np.clip(x1, 0, W))
                y1 = float(np.clip(y1, 0, H))
                x2 = float(np.clip(x2, 0, W))
                y2 = float(np.clip(y2, 0, H))
                if x2 > x1 and y2 > y1:
                    boxes.append([cls_id, x1, y1, x2, y2])
        return boxes

    def _clip_boxes(self, boxes, tx1, ty1, tx2, ty2):
        clipped = []
        for cls_id, bx1, by1, bx2, by2 in boxes:
            orig_area = max((bx2 - bx1) * (by2 - by1), 1e-6)
            ix1 = max(bx1, tx1) - tx1
            iy1 = max(by1, ty1) - ty1
            ix2 = min(bx2, tx2) - tx1
            iy2 = min(by2, ty2) - ty1
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            vis = (ix2 - ix1) * (iy2 - iy1) / orig_area
            # Smooth threshold (box càng nhỏ → giữ càng nhiều)
            if orig_area < 64:
                thr = 0.02
            elif orig_area < 256:
                thr = 0.05
            else:
                thr = 0.20
            if vis >= thr:
                clipped.append([cls_id, ix1, iy1, ix2, iy2])
        return clipped

    def _load_tile(self, idx):
        img_idx, tx1, ty1, tx2, ty2 = self.tile_index[idx]
        boxes, W, H = self.labels_cache[img_idx]
        if img_idx in self.images_cache:
            tile = self.images_cache[img_idx].crop((tx1, ty1, tx2, ty2))
        else:
            tile = Image.open(self.img_files[img_idx]).convert("RGB").crop(
                (tx1, ty1, tx2, ty2))
        return tile, self._clip_boxes(boxes, tx1, ty1, tx2, ty2)

    def _copy_paste(self, tile, boxes):
        if not self.copy_paste_pool:
            return tile, boxes
        n_paste = random.randint(1, COPY_PASTE_MAX_PER)
        arr = np.array(tile).copy()
        H, W = arr.shape[:2]
        for _ in range(n_paste):
            src_img, src_boxes = random.choice(self.copy_paste_pool)
            if not src_boxes:
                continue
            cls_id, sx1, sy1, sx2, sy2 = random.choice(src_boxes)
            sw, sh = sx2 - sx1, sy2 - sy1
            if sw < 2 or sh < 2:
                continue
            sc = random.uniform(*COPY_PASTE_SCALE_JIT)
            nw, nh = max(2, int(sw * sc)), max(2, int(sh * sc))
            try:
                patch = src_img.crop(
                    (int(sx1), int(sy1), int(sx2), int(sy2))
                ).resize((nw, nh))
            except Exception:
                continue
            dx = random.randint(0, max(0, W - nw))
            dy = random.randint(0, max(0, H - nh))
            mask = np.array(patch.convert("L"))
            mask = (mask > 10).astype(np.uint8)
            roi = arr[dy:dy + nh, dx:dx + nw]
            pat_arr = np.array(patch)
            blended = roi * (1 - mask[..., None]) + pat_arr * mask[..., None]
            arr[dy:dy + nh, dx:dx + nw] = blended.astype(np.uint8)
            boxes.append([cls_id, float(dx), float(dy),
                          float(dx + nw), float(dy + nh)])
        return Image.fromarray(arr), boxes

    def _augment(self, tile, boxes):
        if USE_COPY_PASTE and random.random() < COPY_PASTE_PROB:
            tile, boxes = self._copy_paste(tile, boxes)
        if random.random() < 0.5:
            tile = TF.hflip(tile)
            W = tile.width
            boxes = [[c, W - x2, y1, W - x1, y2] for c, x1, y1, x2, y2 in boxes]
        if random.random() < 0.5:
            tile = TF.adjust_brightness(tile, random.uniform(0.8, 1.2))
        if random.random() < 0.4:
            tile = TF.adjust_contrast(tile, random.uniform(0.8, 1.2))
        if random.random() < 0.3:
            tile = TF.adjust_saturation(tile, random.uniform(0.8, 1.2))
        return tile, boxes

    def get_sample_weights(self):
        return torch.tensor(
            [float(TINY_TILE_OVERSAMPLE) if h else 1.0
             for h in self.tile_has_tiny],
            dtype=torch.float)

    def __len__(self):
        return len(self.tile_index)

    def __getitem__(self, idx):
        tile, tile_boxes = self._load_tile(idx)
        if self.is_train:
            tile, tile_boxes = self._augment(tile, tile_boxes)
        img_t = TF.to_tensor(tile)
        if tile_boxes:
            boxes_t = torch.tensor(
                [[b[1], b[2], b[3], b[4]] for b in tile_boxes],
                dtype=torch.float32)
            labels_t = torch.tensor(
                [b[0] + 1 for b in tile_boxes], dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        areas = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "area": areas,
            "iscrowd": torch.zeros(len(labels_t), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_copy_paste_pool(train_ds: YOLOTinyDataset,
                          n_images: int = 300) -> List[Tuple]:
    """Collect small boxes from first n_images for copy-paste."""
    if not USE_COPY_PASTE:
        return []
    pool = []
    for i in range(min(n_images, len(train_ds.img_files))):
        try:
            im = Image.open(train_ds.img_files[i]).convert("RGB")
            boxes, _, _ = train_ds.labels_cache.get(i, ([], 0, 0))
            tiny_boxes = [
                b for b in boxes
                if math.sqrt(max((b[3] - b[1]) * (b[4] - b[2]), 0))
                < TINY_THRESHOLD_PX * 2
            ]
            if tiny_boxes:
                pool.append((im, tiny_boxes))
        except Exception:
            continue
    print(f"[Copy-Paste Pool] {len(pool)} images with small boxes")
    return pool


# =============================================================================
# Dataset factory — dễ dàng switch giữa full-image và patch training
# =============================================================================
def build_training_datasets(use_patches: bool = False,
                            is_train: bool = True,
                            copy_paste_pool=None) -> YOLOTinyDataset:
    """Build dataset cho training/evaluation, full-image hoặc patch.

    Args:
        use_patches: True → dùng patch data (Phase 0 output)
        is_train: True → training set (có augmentation)
        copy_paste_pool: pool cho copy-paste augmentation (train only)
    Returns:
        YOLOTinyDataset instance
    """
    if use_patches:
        img_dir = PATCH_ROOT / ("train" if is_train else "valid") / "images"
        lbl_dir = PATCH_ROOT / ("train" if is_train else "valid") / "labels"
    else:
        img_dir = TRAIN_DIR / "images" if is_train else VALID_DIR / "images"
        lbl_dir = TRAIN_DIR / "labels" if is_train else VALID_DIR / "labels"

    return YOLOTinyDataset(
        img_dir=img_dir,
        lbl_dir=lbl_dir,
        is_train=is_train,
        copy_paste_pool=copy_paste_pool if is_train else None,
    )


def compute_reliability_threshold(train_ds: YOLOTinyDataset,
                                   percentile: int = 25,
                                   lo: float = 4.0,
                                   hi: float = 24.0) -> float:
    """Adaptive reliability threshold = P25 of GT sqrt(area)."""
    sizes = []
    for img_idx in train_ds.labels_cache:
        boxes, _, _ = train_ds.labels_cache[img_idx]
        for b in boxes:
            w = max(b[3] - b[1], 1)
            h = max(b[4] - b[2], 1)
            sizes.append(math.sqrt(w * h))
    if not sizes:
        return 16.0
    p25 = float(np.percentile(sizes, percentile))
    print(f"[Adaptive] GT size: min={min(sizes):.1f}, median={np.median(sizes):.1f}, "
          f"P{percentile}={p25:.1f}, max={max(sizes):.1f}")
    return float(np.clip(p25, lo, hi))