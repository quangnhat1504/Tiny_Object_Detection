"""
Simplified SmartWBF: just scale-adaptive IoU + extent_hull, no adjacency.

Key insight: large objects spanning multiple tiles produce partial
detections with LOW IoU. We need LOWER thresholds for large boxes,
not higher. This avoids the O(n^2) adjacency precompute entirely.

Usage:
    python scripts/tune_smart_wbf.py
"""
from __future__ import annotations
import json, sys, time, itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
import tqdm

from common.config import DEVICE, seed_all, SEED
from common.metrics import get_metric_fn
from common.model import build_model
from common.dataset import YOLOTinyDataset, collate_fn
from common.eval_utils import compute_scale_ap

FRCNN_CKPT = ROOT / "runs/sa_alw_full__la_loss__seed42/best.pt"

GRID = {
    "iou_thr":      [0.30, 0.40, 0.50, 0.55, 0.60],
    "score_thr":    [0.10, 0.20, 0.30],
    "fusion_mode":  ["weighted_avg", "extent_hull"],
    "use_adaptive": [False, True],
}
# 5 * 3 * 2 * 2 = 60 configs


def _weighted_avg(boxes, scores):
    w = scores / scores.sum()
    return (boxes.T @ w).T


def _extent_hull(boxes, scores, iqr_mult=1.5):
    if len(boxes) <= 1:
        return boxes[0] if len(boxes) == 1 else None

    def _robust(v, fn, mult):
        q1, q3 = torch.quantile(v, torch.tensor([0.25, 0.75]))
        iqr = q3 - q1
        bound = q1 - mult * iqr if fn == "min" else q3 + mult * iqr
        valid = v >= bound if fn == "min" else v <= bound
        return (v[valid].min().item() if fn == "min"
                else v[valid].max().item()) if valid.any() else fn(v)

    return torch.tensor([
        _robust(boxes[:, 0], "min", iqr_mult),
        _robust(boxes[:, 1], "min", iqr_mult),
        _robust(boxes[:, 2], "max", iqr_mult),
        _robust(boxes[:, 3], "max", iqr_mult),
    ])


def _adaptive_iou_thr(area, base_thr, use_adaptive):
    """LARGE objects get LOWER threshold (merge partial tile views).
    TINY objects get stock threshold (rarely span tiles).
    """
    if not use_adaptive:
        return base_thr
    if area < 256:
        return base_thr
    if area < 1024:
        return max(base_thr - 0.05, 0.25)
    if area < 4096:
        return max(base_thr - 0.15, 0.20)
    return max(base_thr - 0.25, 0.10)


def fuse_one_image(tile_preds, tile_coords, img_size,
                   iou_thr=0.55, score_thr=0.10,
                   fusion_mode="weighted_avg", use_adaptive=False):
    W, H = img_size
    all_boxes, all_scores, all_labels = [], [], []

    for (tx, ty, tw, th), (boxes, scores, labels) in zip(tile_coords, tile_preds):
        if boxes.numel() == 0:
            continue
        rm = boxes.clone()
        rm[:, 0] = rm[:, 0] * tw / 512 + tx
        rm[:, 1] = rm[:, 1] * th / 512 + ty
        rm[:, 2] = rm[:, 2] * tw / 512 + tx
        rm[:, 3] = rm[:, 3] * th / 512 + ty
        rm[:, 0].clamp_(0, W); rm[:, 1].clamp_(0, H)
        rm[:, 2].clamp_(0, W); rm[:, 3].clamp_(0, H)
        valid = (rm[:, 2] - rm[:, 0] >= 2) & (rm[:, 3] - rm[:, 1] >= 2)
        if valid.any():
            all_boxes.append(rm[valid])
            all_scores.append(scores[valid])
            all_labels.append(labels[valid])

    empty = {
        "boxes": torch.zeros(0, 4),
        "scores": torch.zeros(0),
        "labels": torch.zeros(0, dtype=torch.int64),
    }

    if not all_boxes:
        return empty

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    keep = scores >= score_thr
    if not keep.any():
        return empty

    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    n = len(boxes)

    if n <= 1:
        return {"boxes": boxes, "scores": scores, "labels": labels}

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ious = box_iou(boxes, boxes)

    # Cluster using scale-adaptive IoU
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        xr, yr = find(x), find(y)
        if xr != yr:
            parent[yr] = xr

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            if labels[i] != labels[j]:
                continue
            thr = _adaptive_iou_thr(
                min(areas[i].item(), areas[j].item()), iou_thr, use_adaptive)
            if ious[i, j] > thr:
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    fused_b, fused_s, fused_l = [], [], []
    for indices in groups.values():
        idx = torch.tensor(indices)
        cb, cs, cl = boxes[idx], scores[idx], labels[idx]
        if fusion_mode == "extent_hull":
            fb = _extent_hull(cb, cs)
        else:
            fb = _weighted_avg(cb, cs)
        if fb is not None:
            fused_b.append(fb)
            fused_s.append(cs.mean())
            fused_l.append(torch.mode(cl).values)

    if not fused_b:
        return empty
    return {
        "boxes": torch.stack(fused_b),
        "scores": torch.stack(fused_s),
        "labels": torch.stack(fused_l),
    }


def main():
    seed_all(SEED)
    print(f"Device: {DEVICE}")

    ck = torch.load(FRCNN_CKPT, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    mt = get_metric_fn(cfg.get("metric", "sa_alw_full"))
    model = build_model(mt, "la_loss", cfg.get("reliability_thr", 16.0)).to(DEVICE)
    model.load_state_dict(ck["model"], strict=False)

    td = YOLOTinyDataset(
        img_dir=ROOT / "data/test/images",
        lbl_dir=ROOT / "data/test/labels",
        is_train=False,
    )
    print(f"Test tiles: {len(td)}")

    print("Collecting tile predictions (single pass)...")
    model.eval()
    prev_thr = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = 0.001

    all_tile_preds = []
    loader = DataLoader(td, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=(DEVICE.type == "cuda"))
    for imgs, _ in tqdm.tqdm(loader, desc="FRCNN tiles"):
        imgs = [i.to(DEVICE) for i in imgs]
        with torch.no_grad():
            preds = model(imgs)
        for p in preds:
            all_tile_preds.append(
                (p["boxes"].cpu(), p["scores"].cpu(), p["labels"].cpu()))
    model.roi_heads.score_thresh = prev_thr

    img_groups = {}
    for idx in range(len(td)):
        img_idx = td.tile_index[idx][0]
        if img_idx not in img_groups:
            img_groups[img_idx] = {"tiles": [], "coords": []}
        img_groups[img_idx]["tiles"].append(all_tile_preds[idx])
        img_groups[img_idx]["coords"].append(td.tile_index[idx])
    print(f"Images: {len(img_groups)}")

    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    total = len(combos)
    print(f"Configs: {total}")

    results = []
    t0 = time.time()
    for idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))

        all_preds, all_gts = [], []
        for img_idx, group in sorted(img_groups.items()):
            cache_entry = td.labels_cache[img_idx]
            if len(cache_entry) == 2:
                boxes_raw, (W, H) = cache_entry
            else:
                boxes_raw, W, H = cache_entry

            filtered = []
            for (b, s, l) in group["tiles"]:
                keep = s >= params["score_thr"]
                filtered.append((b[keep], s[keep], l[keep]))

            img_coords = [(tx1, ty1, tx2 - tx1, ty2 - ty1)
                          for (_i, tx1, ty1, tx2, ty2) in group["coords"]]

            fused = fuse_one_image(
                filtered, img_coords, (W, H),
                iou_thr=params["iou_thr"],
                score_thr=params["score_thr"],
                fusion_mode=params["fusion_mode"],
                use_adaptive=params["use_adaptive"],
            )
            all_preds.append(fused)

            gt_boxes = torch.tensor(
                [[b[1], b[2], b[3], b[4]] for b in boxes_raw
                 if b[3] > b[1] and b[4] > b[2]],
                dtype=torch.float32)
            gt_labels = torch.tensor(
                [b[0] + 1 for b in boxes_raw if b[3] > b[1] and b[4] > b[2]],
                dtype=torch.int64)
            areas_gt = ((gt_boxes[:, 2] - gt_boxes[:, 0]) *
                        (gt_boxes[:, 3] - gt_boxes[:, 1])) \
                if gt_boxes.numel() > 0 else torch.zeros(0)
            all_gts.append({
                "boxes": gt_boxes, "labels": gt_labels,
                "area": areas_gt,
                "iscrowd": torch.zeros(len(gt_labels), dtype=torch.int64),
                "image_id": torch.tensor([img_idx], dtype=torch.int64),
            })

        sap = compute_scale_ap(all_preds, all_gts)
        tgt = sum(sap.get(f"n_gt_{s}", 0)
                  for s in ("micro", "tiny", "small", "large"))
        prim = (sum(sap.get(f"AP_{s}", 0) * sap.get(f"n_gt_{s}", 0)
                    for s in ("micro", "tiny", "small", "large"))
                / max(tgt, 1))

        entry = {
            **params,
            "mAP(scale)": round(prim, 4),
            "AP_micro": round(sap.get("AP_micro", 0), 4),
            "AP_tiny": round(sap.get("AP_tiny", 0), 4),
            "AP_small": round(sap.get("AP_small", 0), 4),
            "AP_large": round(sap.get("AP_large", 0), 4),
        }
        results.append(entry)
        elapsed = time.time() - t0
        eta = elapsed / idx * (total - idx) if idx < total else 0
        print(f"  [{idx:>2}/{total}] "
              f"iou={params['iou_thr']:.2f} "
              f"scr={params['score_thr']:.2f} "
              f"{params['fusion_mode']:>13} "
              f"adapt={str(params['use_adaptive']):>5} "
              f"-> mAP={prim:.4f} "
              f"(t={elapsed:.0f}s, eta={eta:.0f}s)")

    results.sort(key=lambda r: r["mAP(scale)"], reverse=True)

    header = (f"{'#':>3} {'iou':>5} {'scr':>5} {'mode':>13} "
              f"{'adapt':>6} {'mAP':>8} {'micro':>8} {'tiny':>8} "
              f"{'small':>8} {'large':>8}")
    print("\n" + "=" * 110)
    print("TOP 10 — Scale-Adaptive IoU (no adjacency)")
    print("=" * 110)
    print(header)
    print("-" * 110)
    for i, r in enumerate(results[:10], 1):
        print(f"{i:>3} {r['iou_thr']:>5.2f} {r['score_thr']:>5.2f} "
              f"{r['fusion_mode']:>13} {str(r['use_adaptive']):>6} "
              f"{r['mAP(scale)']:>8.4f} {r['AP_micro']:>8.4f} "
              f"{r['AP_tiny']:>8.4f} {r['AP_small']:>8.4f} "
              f"{r['AP_large']:>8.4f}")

    out_path = ROOT / "runs/smart_wbf_v3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
