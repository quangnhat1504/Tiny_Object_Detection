#!/usr/bin/env python
"""EDA cho dataset TinyPerson (YOLO polygon format, export tu Roboflow).

Quet train/valid/test, chuyen polygon -> bounding box, doc kich thuoc anh,
tinh thong ke tong quan, sinh bieu do truc quan va bao cao Markdown.

Chay:  .venv/Scripts/python.exe eda/eda_tinyperson.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import plots
import report

# ---------------------------------------------------------------------------
# Cau hinh
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "eda"
FIG = OUT / "figures"
SPLITS = ["train", "valid", "test"]
CLASSES = ["dry-person", "wet-swimmer"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Thang kich thuoc tuyet doi (sqrt(area) theo pixel), theo huong TinyPerson.
SIZE_BINS = [0, 8, 12, 20, 32, 96, np.inf]
SIZE_LABELS = ["(0,8)", "[8,12)", "[12,20)", "[20,32)", "[32,96)", ">=96"]


# ---------------------------------------------------------------------------
# Parse nhan
# ---------------------------------------------------------------------------
def parse_label_file(path: Path):
    """Doc 1 file nhan YOLO. Tra ve list (class_id, xc, yc, w, h) normalized.

    Ho tro ca dinh dang polygon (class x1 y1 x2 y2 ...) lan bbox (class xc yc w h).
    """
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        try:
            cls = int(float(toks[0]))
            coords = [float(t) for t in toks[1:]]
        except (ValueError, IndexError):
            continue
        if len(coords) == 4:
            # YOLO bbox chuan: cx, cy, w, h
            xc, yc, w, h = coords
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            # Polygon -> bounding box
            xs = coords[0::2]
            ys = coords[1::2]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            xc = (xmin + xmax) / 2.0
            yc = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin
        else:
            continue
        boxes.append((cls, xc, yc, w, h))
    return boxes


def find_images(split_dir: Path):
    img_dir = split_dir / "images"
    if not img_dir.is_dir():
        return []
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


# ---------------------------------------------------------------------------
# Thu thap du lieu
# ---------------------------------------------------------------------------
def collect():
    instances = []  # 1 dong / object
    images = []      # 1 dong / anh
    for split in SPLITS:
        split_dir = DATA / split
        img_paths = find_images(split_dir)
        lbl_dir = split_dir / "labels"
        for ip in img_paths:
            img = cv2.imread(str(ip))
            if img is None:
                print(f"  [canh bao] khong doc duoc anh: {ip.name}", file=sys.stderr)
                continue
            h_px, w_px = img.shape[:2]
            lp = lbl_dir / (ip.stem + ".txt")
            boxes = parse_label_file(lp)
            images.append({
                "split": split,
                "name": ip.name,
                "img_w": w_px,
                "img_h": h_px,
                "n_obj": len(boxes),
                "has_label_file": lp.exists(),
            })
            for cls, xc, yc, bw, bh in boxes:
                bw_px = bw * w_px
                bh_px = bh * h_px
                area_px = bw_px * bh_px
                instances.append({
                    "split": split,
                    "class_id": cls,
                    "class": CLASSES[cls] if 0 <= cls < len(CLASSES) else str(cls),
                    "cx": xc,
                    "cy": yc,
                    "bw_px": bw_px,
                    "bh_px": bh_px,
                    "area_px": area_px,
                    "sqrt_area": float(np.sqrt(max(area_px, 0.0))),
                    "aspect": (bw_px / bh_px) if bh_px > 0 else 0.0,
                    "area_frac": area_px / (w_px * h_px) if w_px * h_px else 0.0,
                })
        print(f"[{split}] anh={len(img_paths)} object={sum(1 for i in instances if i['split']==split)}")
    return instances, images


# ---------------------------------------------------------------------------
# Tinh thong ke tong hop
# ---------------------------------------------------------------------------
def summarize(instances, images):
    s = {"splits": {}, "classes": CLASSES, "size_labels": SIZE_LABELS}
    sqrt_all = np.array([i["sqrt_area"] for i in instances]) if instances else np.array([])

    # Phan loai theo thang kich thuoc
    if sqrt_all.size:
        cats = np.digitize(sqrt_all, SIZE_BINS[1:-1], right=False)
        size_cat_counts = Counter(int(c) for c in cats)
    else:
        size_cat_counts = Counter()
    s["size_category_counts"] = {SIZE_LABELS[k]: size_cat_counts.get(k, 0)
                                 for k in range(len(SIZE_LABELS))}

    for split in SPLITS:
        imgs = [im for im in images if im["split"] == split]
        insts = [it for it in instances if it["split"] == split]
        n_img = len(imgs)
        n_bg = sum(1 for im in imgs if im["n_obj"] == 0)
        cls_counts = Counter(it["class"] for it in insts)
        nobj = np.array([im["n_obj"] for im in imgs]) if imgs else np.array([])
        sq = np.array([it["sqrt_area"] for it in insts]) if insts else np.array([])
        ar = np.array([it["aspect"] for it in insts]) if insts else np.array([])
        s["splits"][split] = {
            "n_images": n_img,
            "n_background": int(n_bg),
            "n_instances": len(insts),
            "class_counts": {c: int(cls_counts.get(c, 0)) for c in CLASSES},
            "obj_per_img_mean": float(nobj.mean()) if nobj.size else 0.0,
            "obj_per_img_median": float(np.median(nobj)) if nobj.size else 0.0,
            "obj_per_img_max": int(nobj.max()) if nobj.size else 0,
            "sqrt_area_mean": float(sq.mean()) if sq.size else 0.0,
            "sqrt_area_median": float(np.median(sq)) if sq.size else 0.0,
            "sqrt_area_min": float(sq.min()) if sq.size else 0.0,
            "sqrt_area_max": float(sq.max()) if sq.size else 0.0,
            "aspect_median": float(np.median(ar)) if ar.size else 0.0,
            "img_sizes": dict(Counter(f"{im['img_w']}x{im['img_h']}" for im in imgs)),
        }

    # Tong hop toan bo
    s["total"] = {
        "n_images": len(images),
        "n_instances": len(instances),
        "n_background": int(sum(1 for im in images if im["n_obj"] == 0)),
        "class_counts": {c: int(sum(1 for it in instances if it["class"] == c)) for c in CLASSES},
        "sqrt_area_mean": float(sqrt_all.mean()) if sqrt_all.size else 0.0,
        "sqrt_area_median": float(np.median(sqrt_all)) if sqrt_all.size else 0.0,
        "sqrt_area_min": float(sqrt_all.min()) if sqrt_all.size else 0.0,
        "sqrt_area_max": float(sqrt_all.max()) if sqrt_all.size else 0.0,
        "pct_tiny_lt20": float((sqrt_all < 20).mean() * 100) if sqrt_all.size else 0.0,
        "pct_small_lt32": float((sqrt_all < 32).mean() * 100) if sqrt_all.size else 0.0,
    }
    return s


def write_csv(rows, path: Path, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("== Thu thap du lieu ==")
    instances, images = collect()
    if not images:
        print("Khong tim thay anh nao trong data/. Dung lai.", file=sys.stderr)
        sys.exit(1)

    print("== Tinh thong ke ==")
    summary = summarize(instances, images)

    # Luu du lieu tho
    write_csv(instances, OUT / "instances.csv",
              ["split", "class_id", "class", "cx", "cy", "bw_px", "bh_px",
               "area_px", "sqrt_area", "aspect", "area_frac"])
    write_csv(images, OUT / "images.csv",
              ["split", "name", "img_w", "img_h", "n_obj", "has_label_file"])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("== Ve bieu do ==")
    plots.make_all(instances, images, summary, FIG, CLASSES, SPLITS,
                   SIZE_BINS, SIZE_LABELS)

    print("== Sinh bao cao ==")
    report.write_report(summary, OUT / "REPORT.md", FIG, CLASSES, SPLITS)

    print(f"\nXong. Ket qua trong: {OUT}")
    print(f"  - REPORT.md (bao cao tong hop)")
    print(f"  - figures/  ({len(list(FIG.glob('*.png')))} bieu do)")
    print(f"  - summary.json, instances.csv, images.csv")


if __name__ == "__main__":
    main()
