"""
Cascaded Pipeline (Phase 4):
  FRCNN tile-scan -> WBF (standard or smart) -> full-image evaluation.

Usage:
    python scripts/eval_cascade.py                              # standard WBF (iou=0.55, scr=0.30)
    python scripts/eval_cascade.py --fusion-mode weighted_avg --wbf-iou 0.60 --score-thr 0.10
    python scripts/eval_cascade.py --fusion-mode extent_hull --adaptive-thr
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.config import DEVICE, seed_all, SEED
from common.metrics import get_metric_fn
from common.model import build_model
from common.dataset import YOLOTinyDataset, collate_fn
from common.eval_utils import compute_scale_ap, evaluate_coco, compute_precision_recall
from common.wbf import wbf_fusion_smart

FRCNN_CKPT = ROOT / "runs/sa_alw_full__la_loss__seed42/best.pt"


def wbf_fusion_standard(tile_preds, tile_coords, img_size, iou_thr=0.55):
    """Original standard WBF (weighted_avg, fixed IoU threshold)."""
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

    if boxes.numel() <= 1:
        return {"boxes": boxes, "scores": scores, "labels": labels}

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-mode", choices=["weighted_avg", "extent_hull"], default="weighted_avg",
                        help="Fusion method (default: weighted_avg = standard WBF)")
    parser.add_argument("--wbf-iou", type=float, default=0.55)
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument("--adaptive-thr", action="store_true", default=False,
                        help="Enable scale-adaptive IoU threshold")
    args = parser.parse_args()

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
    tile_count = len(td)
    print(f"Test tiles: {tile_count}")

    model.eval()
    prev_thr = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = 0.001

    all_tile_preds = []
    import tqdm
    loader = DataLoader(td, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=(DEVICE.type=="cuda"))
    t0 = time.time()
    for imgs, _ in tqdm.tqdm(loader, desc="FRCNN tile-scan"):
        imgs = [i.to(DEVICE) for i in imgs]
        with torch.no_grad():
            preds = model(imgs)
        for p in preds:
            all_tile_preds.append((
                p["boxes"].cpu(), p["scores"].cpu(), p["labels"].cpu()))
    model.roi_heads.score_thresh = prev_thr
    print(f"Tile-scan done in {time.time()-t0:.1f}s")

    img_groups = {}
    for idx in range(len(td)):
        img_idx = td.tile_index[idx][0]
        if img_idx not in img_groups:
            img_groups[img_idx] = {"tiles": [], "coords": []}
        img_groups[img_idx]["tiles"].append(all_tile_preds[idx])
        img_groups[img_idx]["coords"].append(td.tile_index[idx])

    print(f"Images: {len(img_groups)}")

    is_smart = args.fusion_mode == "extent_hull" or args.adaptive_thr
    print(f"WBF: mode={args.fusion_mode}, iou={args.wbf_iou}, "
          f"score={args.score_thr}, adaptive={args.adaptive_thr}")

    all_preds, all_gts = [], []
    for img_idx, group in sorted(img_groups.items()):
        cache_entry = td.labels_cache[img_idx]
        if len(cache_entry) == 2:
            boxes, (W, H) = cache_entry
        else:
            boxes, W, H = cache_entry

        filtered = []
        for (b, s, l) in group["tiles"]:
            keep = s >= args.score_thr
            filtered.append((b[keep], s[keep], l[keep]))

        img_coords = []
        for (_img_idx, tx1, ty1, tx2, ty2) in group["coords"]:
            img_coords.append((tx1, ty1, tx2 - tx1, ty2 - ty1))

        if is_smart:
            fused = wbf_fusion_smart(
                filtered, img_coords, (W, H),
                iou_thr=args.wbf_iou,
                fusion_mode=args.fusion_mode,
                adaptive_thr=args.adaptive_thr,
            )
        else:
            fused = wbf_fusion_standard(
                filtered, img_coords, (W, H), iou_thr=args.wbf_iou)

        all_preds.append(fused)

        gt_boxes = torch.tensor([[b[1], b[2], b[3], b[4]] for b in boxes if b[3] > b[1] and b[4] > b[2]],
                                 dtype=torch.float32)
        gt_labels = torch.tensor([b[0]+1 for b in boxes if b[3] > b[1] and b[4] > b[2]],
                                  dtype=torch.int64)
        areas = (gt_boxes[:,2]-gt_boxes[:,0]) * (gt_boxes[:,3]-gt_boxes[:,1]) if gt_boxes.numel() > 0 else torch.zeros(0)
        all_gts.append({
            "boxes": gt_boxes, "labels": gt_labels,
            "area": areas,
            "iscrowd": torch.zeros(len(gt_labels), dtype=torch.int64),
            "image_id": torch.tensor([img_idx], dtype=torch.int64)})

    sap = compute_scale_ap(all_preds, all_gts)
    tgt = sum(sap.get(f"n_gt_{s}",0) for s in ("micro","tiny","small","large"))
    prim = sum(sap.get(f"AP_{s}",0)*sap.get(f"n_gt_{s}",0) for s in ("micro","tiny","small","large"))/max(tgt,1)

    coco = evaluate_coco(all_preds, all_gts)
    pr = compute_precision_recall(all_preds, all_gts)

    print(f"""
CASCADE (FRCNN tiling + WBF):
  mAP(scale): {prim:.4f}
  AP_micro:   {sap.get('AP_micro',0):.4f}  (n={sap.get('n_gt_micro',0)})
  AP_tiny:    {sap.get('AP_tiny',0):.4f}  (n={sap.get('n_gt_tiny',0)})
  AP_small:   {sap.get('AP_small',0):.4f}  (n={sap.get('n_gt_small',0)})
  AP_large:   {sap.get('AP_large',0):.4f}  (n={sap.get('n_gt_large',0)})
  COCO mAP@50:{coco.get('coco_AP50',0):.4f}
  COCO AP@75: {coco.get('coco_AP75',0):.4f}
  Precision:  {pr.get('Precision',0):.4f}
  Recall:     {pr.get('Recall',0):.4f}
""")

    out = {"config": vars(args), "mAP_primary": round(prim,6), **sap, **coco, **pr}
    with open(ROOT / "runs/cascade_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: runs/cascade_results.json")


if __name__ == "__main__":
    main()
