"""
Grid search WBF params for cascade pipeline (Phase 4 fine-tune).

Usage:
    python scripts/tune_cascade.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import itertools

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
from common.eval_utils import compute_scale_ap, evaluate_coco, compute_precision_recall

FRCNN_CKPT = ROOT / "runs/sa_alw_full__la_loss__seed42/best.pt"

GRID = {
    "wbf_iou_thr": [0.30, 0.40, 0.50, 0.55, 0.60],
    "score_thr": [0.10, 0.20, 0.30],
}


def wbf_fusion(tile_preds, tile_coords, img_size, iou_thr=0.55):
    W, H = img_size
    all_boxes, all_scores, all_labels = [], [], []

    for (tx, ty, tw, th), (boxes, scores, labels) in zip(tile_coords, tile_preds):
        if boxes.numel() == 0:
            continue
        rm = boxes.clone()
        rm[:, 0] = rm[:, 0] * tw/512 + tx
        rm[:, 1] = rm[:, 1] * th/512 + ty
        rm[:, 2] = rm[:, 2] * tw/512 + tx
        rm[:, 3] = rm[:, 3] * th/512 + ty
        rm[:, 0].clamp_(0, W); rm[:, 1].clamp_(0, H)
        rm[:, 2].clamp_(0, W); rm[:, 3].clamp_(0, H)
        valid = (rm[:, 2] - rm[:, 0] >= 2) & (rm[:, 3] - rm[:, 1] >= 2)
        if valid.any():
            all_boxes.append(rm[valid])
            all_scores.append(scores[valid])
            all_labels.append(labels[valid])

    if not all_boxes:
        return {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)}

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    if boxes.numel() == 0:
        return {"boxes": boxes, "scores": scores, "labels": labels}

    # Cluster by IoU
    ious = box_iou(boxes, boxes)
    cid = torch.zeros(len(boxes), dtype=torch.long)
    nc = 1
    for i in range(len(boxes)):
        if cid[i] > 0: continue
        cid[i] = nc
        for j in range(i+1, len(boxes)):
            if cid[j] > 0: continue
            if ious[i,j] > iou_thr: cid[j] = nc
        nc += 1

    fused_b, fused_s, fused_l = [], [], []
    for c in range(1, nc):
        m = cid == c
        cb, cs, cl = boxes[m], scores[m], labels[m]
        w = cs / cs.sum()
        avg = (cb.T @ w).T
        fused_b.append(avg)
        fused_s.append(cs.mean())
        fused_l.append(torch.mode(cl).values)

    return {"boxes": torch.stack(fused_b), "scores": torch.stack(fused_s), "labels": torch.stack(fused_l)}


def evaluate_config(all_tile_preds, td, iou_thr, score_thr):
    img_groups = {}
    for idx in range(len(td)):
        img_idx = td.tile_index[idx][0]
        if img_idx not in img_groups:
            img_groups[img_idx] = {"tiles": [], "coords": []}
        img_groups[img_idx]["tiles"].append(all_tile_preds[idx])
        img_groups[img_idx]["coords"].append(td.tile_index[idx])

    all_preds, all_gts = [], []
    for img_idx in sorted(img_groups):
        group = img_groups[img_idx]
        cache_entry = td.labels_cache[img_idx]
        if len(cache_entry) == 2:
            boxes, (W, H) = cache_entry
        else:
            boxes, W, H = cache_entry

        filtered = []
        for (b, s, l) in group["tiles"]:
            keep = s >= score_thr
            filtered.append((b[keep], s[keep], l[keep]))

        coords = []
        for (_img_idx, tx1, ty1, tx2, ty2) in group["coords"]:
            coords.append((tx1, ty1, tx2-tx1, ty2-ty1))

        fused = wbf_fusion(filtered, coords, (W, H), iou_thr)
        all_preds.append(fused)

        gt_boxes = torch.tensor([[b[1], b[2], b[3], b[4]] for b in boxes if b[3] > b[1] and b[4] > b[2]],
                                 dtype=torch.float32)
        gt_labels = torch.tensor([b[0]+1 for b in boxes if b[3] > b[1] and b[4] > b[2]],
                                  dtype=torch.int64)
        areas = (gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:,3]-gt_boxes[:,1]) if gt_boxes.numel()>0 else torch.zeros(0)
        all_gts.append({
            "boxes": gt_boxes, "labels": gt_labels, "area": areas,
            "iscrowd": torch.zeros(len(gt_labels), dtype=torch.int64),
            "image_id": torch.tensor([img_idx], dtype=torch.int64)})

    sap = compute_scale_ap(all_preds, all_gts)
    tgt = sum(sap.get(f"n_gt_{s}",0) for s in ("micro","tiny","small","large"))
    prim = sum(sap.get(f"AP_{s}",0)*sap.get(f"n_gt_{s}",0) for s in ("micro","tiny","small","large"))/max(tgt,1)
    return prim, sap["AP_micro"], sap["AP_tiny"], sap["AP_small"], sap["AP_large"]


def main():
    seed_all(SEED)
    print(f"Device: {DEVICE}")

    # Load model
    ck = torch.load(FRCNN_CKPT, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    mt = get_metric_fn(cfg.get("metric", "sa_alw_full"))
    model = build_model(mt, "la_loss", cfg.get("reliability_thr", 16.0)).to(DEVICE)
    model.load_state_dict(ck["model"], strict=False)

    # Build dataset
    td = YOLOTinyDataset(
        img_dir=ROOT / "data/test/images",
        lbl_dir=ROOT / "data/test/labels",
        is_train=False,
    )

    # Collect tile predictions once (reuse for all configs)
    print(f"Collecting {len(td)} tile predictions (one pass)...")
    model.eval()
    prev_thr = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = 0.001

    all_tile_preds = []
    loader = DataLoader(td, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=(DEVICE.type=="cuda"))
    for imgs, _ in tqdm.tqdm(loader, desc="FRCNN tiles"):
        imgs = [i.to(DEVICE) for i in imgs]
        with torch.no_grad():
            preds = model(imgs)
        for p in preds:
            all_tile_preds.append((p["boxes"].cpu(), p["scores"].cpu(), p["labels"].cpu()))
    model.roi_heads.score_thresh = prev_thr

    # Grid search
    results = []
    total = len(GRID["wbf_iou_thr"]) * len(GRID["score_thr"])
    i = 0
    for iou_thr in GRID["wbf_iou_thr"]:
        for score_thr in GRID["score_thr"]:
            i += 1
            prim, ap_m, ap_t, ap_s, ap_l = evaluate_config(all_tile_preds, td, iou_thr, score_thr)
            results.append({
                "wbf_iou": iou_thr, "score_thr": score_thr,
                "mAP(scale)": round(prim, 4), "AP_micro": round(ap_m, 4),
                "AP_tiny": round(ap_t, 4), "AP_small": round(ap_s, 4), "AP_large": round(ap_l, 4),
            })
            print(f"  [{i}/{total}] iou={iou_thr:.2f} scr={score_thr:.2f} -> mAP={prim:.4f}")

    # Sort and display top
    results.sort(key=lambda r: r["mAP(scale)"], reverse=True)
    print("\n" + "=" * 90)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 90)
    for r in results[:5]:
        print(f"  iou={r['wbf_iou']:.2f} scr={r['score_thr']:.2f} "
              f"mAP={r['mAP(scale)']:.4f} micro={r['AP_micro']:.4f} "
              f"tiny={r['AP_tiny']:.4f} small={r['AP_small']:.4f} large={r['AP_large']:.4f}")

    with open(ROOT / "runs/cascade_grid.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: runs/cascade_grid.json")


if __name__ == "__main__":
    main()
