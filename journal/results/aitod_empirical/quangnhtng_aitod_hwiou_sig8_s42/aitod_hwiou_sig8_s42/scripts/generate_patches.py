"""
Phase 0: Sinh patch training data cho Faster R-CNN nhanh tinh chinh (Cascaded Uncertainty Routing).

Logic:
1. Doc annotation YOLO polygon tu data/{split}/labels/
2. Chuyen polygon -> bounding box (xyxy)
3. Mo phong Uncertainty Router: voi moi box, crop patch xung quanh voi context padding
4. Remap GT boxes tu toa do full-image -> toa do patch
5. Luu patches + labels dinh dang YOLO (de train Faster R-CNN)

Output:
  data/patches/{split}/images/   -- anh patch
  data/patches/{split}/labels/   -- nhan patch (YOLO bbox format: class xc yc w h)
  data/patches/meta.json         -- thong tin anh goc -> danh sach patch
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "data" / "patches"

SPLITS = ["train", "valid"]
CLASSES = ["dry-person", "wet-swimmer"]

# Patch parameters (seeds tu plan.md, can tune o Phase 4)
CONTEXT_RATIO = 1.5  # mo rong box len 1.5x truoc khi crop
PATCH_SIZE = 512  # resize patch ve kich thuoc nay
MIN_AREA = 16  # bo qua box < 4x4 px (nhieu annotation)
IOU_MERGE_THR = 0.5  # merge cac patch chong lan

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def polygon_to_bbox(coords_norm: list[float], img_w: int, img_h: int) -> tuple:
    """Chuyen YOLO polygon normalized -> bounding box xyxy pixel."""
    xs = coords_norm[0::2]
    ys = coords_norm[1::2]
    x_px = min(xs) * img_w
    y_px = min(ys) * img_h
    x2_px = max(xs) * img_w
    y2_px = max(ys) * img_h
    return x_px, y_px, x2_px, y2_px


def parse_label(path: Path, img_w: int, img_h: int) -> list[dict]:
    """Doc file label YOLO polygon, tra ve list box xyxy."""
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        try:
            cls_id = int(float(toks[0]))
            coords = [float(t) for t in toks[1:]]
        except (ValueError, IndexError):
            continue
        if len(coords) < 4 or len(coords) % 2 != 0:
            continue
        x1, y1, x2, y2 = polygon_to_bbox(coords, img_w, img_h)
        # clamp vao bien anh
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        if (x2 - x1) * (y2 - y1) < MIN_AREA:
            continue
        boxes.append({
            "cls_id": cls_id,
            "class": CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else str(cls_id),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "w": x2 - x1, "h": y2 - y1,
        })
    return boxes


def compute_patch_for_box(box: dict, img_w: int, img_h: int,
                           context_ratio: float, patch_size: int) -> dict:
    """Tinh toa do patch (full-image) tu 1 box va context ratio."""
    cx = (box["x1"] + box["x2"]) / 2
    cy = (box["y1"] + box["y2"]) / 2
    bw = box["w"]
    bh = box["h"]

    # Mo rong box voi context ratio, gioi han toi thieu = patch_size
    pw = max(bw * context_ratio, patch_size)
    ph = max(bh * context_ratio, patch_size)

    # Crop xung quanh tam
    x1 = cx - pw / 2
    y1 = cy - ph / 2
    x2 = cx + pw / 2
    y2 = cy + ph / 2

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": pw, "h": ph}


def merge_overlapping_patches(patches: list[dict], iou_thr: float) -> list[dict]:
    """Gop cac patch chong lan (lay box bao quanh)."""
    if not patches:
        return []
    merged = []
    used = set()
    for i in range(len(patches)):
        if i in used:
            continue
        p = patches[i]
        group = [p]
        used.add(i)
        for j in range(i + 1, len(patches)):
            if j in used:
                continue
            q = patches[j]
            # Tinh IoU giua 2 patch
            x1 = max(p["x1"], q["x1"])
            y1 = max(p["y1"], q["y1"])
            x2 = min(p["x2"], q["x2"])
            y2 = min(p["y2"], q["y2"])
            if x1 < x2 and y1 < y2:
                inter = (x2 - x1) * (y2 - y1)
                union = p["w"] * p["h"] + q["w"] * q["h"] - inter
                iou = inter / union if union > 0 else 0
                if iou > iou_thr:
                    group.append(q)
                    used.add(j)
        # Hop nhat group -> bounding box bao quanh
        x1 = min(g["x1"] for g in group)
        y1 = min(g["y1"] for g in group)
        x2 = max(g["x2"] for g in group)
        y2 = max(g["y2"] for g in group)
        merged.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": x2 - x1, "h": y2 - y1})
    return merged


def process_split(split: str):
    """Xu ly 1 split: sinh patches tu annotation."""
    img_dir = DATA / split / "images"
    lbl_dir = DATA / split / "labels"
    out_img_dir = OUT / split / "images"
    out_lbl_dir = OUT / split / "labels"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"\n[{split}] Processing {len(img_paths)} images...")

    meta = {}  # {original_image: [patch_info]}
    total_patches = 0

    for img_path in img_paths:
        # Doc anh
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Cannot read: {img_path.name}")
            continue
        img_h, img_w = img.shape[:2]

        # Doc label
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        boxes = parse_label(lbl_path, img_w, img_h)

        if not boxes:
            continue

        # Tinh patch cho tung box
        raw_patches = []
        for box in boxes:
            patch = compute_patch_for_box(box, img_w, img_h, CONTEXT_RATIO, PATCH_SIZE)
            raw_patches.append(patch)

        # Merge chong lan
        merged = merge_overlapping_patches(raw_patches, IOU_MERGE_THR)

        # Crop tung patch
        patch_infos = []
        for pi, patch in enumerate(merged):
            # Clamp vao bien anh
            px1 = max(0, int(patch["x1"]))
            py1 = max(0, int(patch["y1"]))
            px2 = min(img_w, int(patch["x2"]))
            py2 = min(img_h, int(patch["y2"]))
            pw = px2 - px1
            ph = py2 - py1
            if pw < 10 or ph < 10:
                continue

            # Crop
            patch_img = img[py1:py2, px1:px2]
            if patch_img.size == 0:
                continue

            # Resize ve PATCH_SIZE
            patch_resized = cv2.resize(patch_img, (PATCH_SIZE, PATCH_SIZE),
                                        interpolation=cv2.INTER_LINEAR)

            # Scale factor cho remap toa do
            scale_x = PATCH_SIZE / pw
            scale_y = PATCH_SIZE / ph

            # Tim GT box trong patch
            patch_boxes = []
            for box in boxes:
                # Tinh overlap giua box va patch
                ox1 = max(box["x1"], px1)
                oy1 = max(box["y1"], py1)
                ox2 = min(box["x2"], px2)
                oy2 = min(box["y2"], py2)
                if ox1 < ox2 and oy1 < oy2:
                    oarea = (ox2 - ox1) * (oy2 - oy1)
                    barea = box["w"] * box["h"]
                    if oarea / barea > 0.3:  # giu box neu overlap > 30%
                        # Remap toa do ve patch
                        rx1 = (ox1 - px1) * scale_x
                        ry1 = (oy1 - py1) * scale_y
                        rx2 = (ox2 - px1) * scale_x
                        ry2 = (oy2 - py1) * scale_y
                        # Chuyen sang YOLO bbox format: class xc yc w h (normalized)
                        rw = rx2 - rx1
                        rh = ry2 - ry1
                        if rw < 2 or rh < 2:
                            continue
                        rcx = (rx1 + rx2) / 2 / PATCH_SIZE
                        rcy = (ry1 + ry2) / 2 / PATCH_SIZE
                        rwn = rw / PATCH_SIZE
                        rhn = rh / PATCH_SIZE
                        patch_boxes.append(f"{box['cls_id']} {rcx:.6f} {rcy:.6f} {rwn:.6f} {rhn:.6f}")

            if not patch_boxes:
                continue

            # Luu anh patch
            patch_name = f"{img_path.stem}_p{pi}.jpg"
            cv2.imwrite(str(out_img_dir / patch_name), patch_resized,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            # Luu label patch
            (out_lbl_dir / (patch_name.rsplit(".", 1)[0] + ".txt")).write_text(
                "\n".join(patch_boxes), encoding="utf-8")

            patch_infos.append({
                "patch_file": patch_name,
                "patch_xyxy": [px1, py1, px2, py2],
                "n_objects": len(patch_boxes),
            })
            total_patches += 1

        if patch_infos:
            meta[str(img_path.name)] = {
                "img_w": img_w,
                "img_h": img_h,
                "n_boxes": len(boxes),
                "n_patches": len(patch_infos),
                "patches": patch_infos,
            }

        if len(meta) % 200 == 0:
            print(f"  [{split}] Processed {len(meta)} images, {total_patches} patches...")

    print(f"[{split}] Done: {len(meta)} images -> {total_patches} patches")
    return meta


def main():
    # Cho phep chay 1 split: python generate_patches.py valid
    splits_to_run = sys.argv[1:] if len(sys.argv) > 1 else SPLITS

    print("=" * 60)
    print("Phase 0: Generate Patch Training Data for Faster R-CNN")
    print("=" * 60)
    print(f"Splits: {splits_to_run}")
    print(f"Context ratio: {CONTEXT_RATIO}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Min area: {MIN_AREA} px^2")

    # Doc meta cu neu co
    all_meta = {}
    if (OUT / "meta.json").exists():
        all_meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
        print(f"Found existing meta with splits: {list(all_meta.keys())}")

    t_start = time.time()
    for split in splits_to_run:
        meta = process_split(split)
        all_meta[split] = meta
        elapsed = time.time() - t_start
        print(f"  Elapsed: {elapsed:.1f}s")

    # Luu meta
    (OUT / "meta.json").write_text(
        json.dumps(all_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Thong ke
    total_patches = sum(
        len(m["patches"]) for split_meta in all_meta.values() for m in split_meta.values()
    )
    total_objects = sum(
        sum(p["n_objects"] for p in m["patches"])
        for split_meta in all_meta.values() for m in split_meta.values()
    )
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output: {OUT}")
    print(f"Total patches: {total_patches}")
    print(f"Total objects in patches: {total_objects}")
    for split in all_meta:
        n_imgs = len(all_meta[split])
        n_patches = sum(len(m["patches"]) for m in all_meta[split].values())
        n_pix = sum(
            (m["patches"][0]["patch_xyxy"][2] - m["patches"][0]["patch_xyxy"][0])
            * (m["patches"][0]["patch_xyxy"][3] - m["patches"][0]["patch_xyxy"][1])
            for m in all_meta[split].values() if m["patches"]
        )
        avg_pix = n_pix / max(n_patches, 1)
        print(f"  {split}: {n_imgs} images -> {n_patches} patches (avg {n_patches/max(n_imgs,1):.1f}/img, ~{avg_pix/1e6:.1f}MP/patch)")


if __name__ == "__main__":
    main()
