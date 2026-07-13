"""Evaluation utilities — shared across all experiments."""
from __future__ import annotations
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    _HAS_TORCHMETRICS = True
except ImportError:
    _HAS_TORCHMETRICS = False

from .config import (
    DEVICE, CLASS_NAMES, TINY_THRESHOLD_PX,
)


# =============================================================================
# FPS measurement
# =============================================================================
def measure_fps(model, device, batch_size: int = 1,
                img_size: Tuple[int, int] = (640, 800),
                n_warmup: int = 30, n_iters: int = 100) -> float:
    """Measure inference speed (images/second) on a dummy batch.

    Args:
        model: torch model (will be set to eval mode)
        device: torch device
        batch_size: number of images per batch
        img_size: (H, W) input size
        n_warmup: iterations to warm up GPU
        n_iters: iterations to measure
    Returns:
        FPS (images/second)
    """
    model.eval()
    dummy = torch.randn(batch_size, 3, img_size[0], img_size[1]).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    # Measure
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    fps = (n_iters * batch_size) / elapsed
    return fps


# =============================================================================
# Precision & Recall
# =============================================================================
def compute_precision_recall(preds: List[Dict], gts: List[Dict],
                              iou_thresh: float = 0.5,
                              score_thresh: float = 0.05) -> Dict:
    """Compute Precision and Recall at given IoU threshold.

    Args:
        preds: list of {"boxes", "scores", "labels"}
        gts: list of {"boxes", "labels"}
        iou_thresh: IoU threshold for a match
        score_thresh: minimum score to consider a detection
    Returns:
        {"Precision": float, "Recall": float}
    """
    total_gt = 0
    total_tp = 0
    total_fp = 0

    for pred, gt in zip(preds, gts):
        gb = gt["boxes"]
        pb = pred.get("boxes", torch.empty(0, 4))
        ps = pred.get("scores", torch.empty(0))
        total_gt += len(gb)

        if len(gb) == 0 or len(pb) == 0 or len(ps) == 0:
            continue

        # Filter by score
        keep = ps >= score_thresh
        pb = pb[keep]
        ps = ps[keep]

        if len(pb) == 0:
            continue

        ious = box_iou(pb, gb)
        matched = torch.zeros(len(gb), dtype=torch.bool)

        # Sort by score descending
        order = ps.argsort(descending=True)
        for pi in order:
            row = ious[pi].clone()
            row[matched] = 0.0
            best_iou, best_gt = row.max(dim=0)
            if best_iou >= iou_thresh:
                matched[best_gt] = True
                total_tp += 1
            else:
                total_fp += 1

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_gt, 1)

    return {
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "n_gt_total": total_gt,
        "n_tp": total_tp,
    }


# =============================================================================
# Scale-binned AP (micro / tiny / small / large)
# =============================================================================
def compute_scale_ap(preds: List[Dict], gts: List[Dict]) -> Dict:
    """Compute AP per scale bin.

    Bins (sqrt-area based):
      micro: (0, 6]
      tiny:  (6, 16]
      small: (16, 64]
      large: (64, ∞)

    IoU thresholds per bin (lower for tiny, higher for large).
    """
    bins = {
        "micro": (0, 6),
        "tiny":  (6, 16),
        "small": (16, 64),
        "large": (64, 1e9),
    }
    iou_thresh = {"micro": 0.25, "tiny": 0.25,
                  "small": 0.35, "large": 0.50}
    data = {s: {"tp": [], "fp": [], "n_gt": 0} for s in bins}

    for pred, gt in zip(preds, gts):
        gb = gt["boxes"]
        pb = pred.get("boxes", torch.empty(0, 4))
        ps = pred.get("scores", torch.empty(0))
        if len(gb) == 0:
            continue
        gs = ((gb[:, 2] - gb[:, 0]).clamp(0) *
              (gb[:, 3] - gb[:, 1]).clamp(0)).sqrt()
        for gi in range(len(gb)):
            sz = float(gs[gi])
            for sn, (lo, hi) in bins.items():
                if lo <= sz < hi:
                    data[sn]["n_gt"] += 1
                    break
        if len(ps) == 0:
            continue
        order = ps.argsort(descending=True)
        pb = pb[order]
        ps = ps[order]
        ious = box_iou(pb, gb)
        matched = torch.zeros(len(gb), dtype=torch.bool)
        for pi in range(len(pb)):
            row = ious[pi].clone()
            row[matched] = 0.0
            bi = row.max()
            gi2 = row.argmax()
            sz = float(gs[gi2])
            sn = "large"
            for n, (lo, hi) in bins.items():
                if lo <= sz < hi:
                    sn = n
                    break
            if bi >= iou_thresh[sn]:
                matched[gi2] = True
                data[sn]["tp"].append(float(ps[pi]))
            else:
                data[sn]["fp"].append(float(ps[pi]))

    def ap_from(tp, fp, n):
        if n == 0 or (not tp and not fp):
            return 0.0
        ents = sorted([(s, 1) for s in tp] + [(s, 0) for s in fp],
                      key=lambda x: -x[0])
        tc = fc = 0
        pr, re = [], []
        for _, it in ents:
            if it:
                tc += 1
            else:
                fc += 1
            pr.append(tc / (tc + fc))
            re.append(tc / n)
        return sum(
            max((p for p, r in zip(pr, re) if r >= t), default=0.0)
            for t in np.linspace(0, 1, 101)
        ) / 101

    out = {}
    for sn, d in data.items():
        out[f"AP_{sn}"] = round(ap_from(d["tp"], d["fp"], d["n_gt"]), 4)
        out[f"n_gt_{sn}"] = d["n_gt"]
    return out


# =============================================================================
# COCO mAP@50:75 (using torchmetrics if available)
# =============================================================================
def evaluate_coco(preds: List[Dict], gts: List[Dict],
                  iou_thresholds: Optional[List[float]] = None,
                  class_metrics: bool = True) -> Dict:
    """Compute COCO-style mAP over iou_thresholds."""
    if not _HAS_TORCHMETRICS:
        return {"coco_AP50": 0.0, "coco_AP75": 0.0}
    if iou_thresholds is None:
        iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70,
                          0.75, 0.80, 0.85, 0.90, 0.95]
    metric = MeanAveragePrecision(
        iou_type="bbox", iou_thresholds=iou_thresholds,
        max_detection_thresholds=[1, 10, 100],
        class_metrics=class_metrics,
    ).to("cpu")
    for p, g in zip(preds, gts):
        metric.update([p], [g])
    res = metric.compute()

    def s(v, d=0.0):
        x = float(v) if v is not None else 0.0
        return d if x < 0 else x

    out = {
        "coco_AP":   round(s(res.get("map", 0)), 4),
        "coco_AP50": round(s(res.get("map_50", 0)), 4),
        "coco_AP75": round(s(res.get("map_75", 0)), 4),
        "coco_AP_small":  round(s(res.get("map_small", 0)), 4),
        "coco_AP_medium": round(s(res.get("map_medium", 0)), 4),
        "coco_AP_large":  round(s(res.get("map_large", 0)), 4),
        "coco_AR1":   round(s(res.get("mar_1", 0)), 4),
        "coco_AR10":  round(s(res.get("mar_10", 0)), 4),
        "coco_AR100": round(s(res.get("mar_100", 0)), 4),
    }
    if class_metrics and res.get("map_per_class") is not None:
        pc = {}
        for ci, ap in enumerate(res["map_per_class"]):
            v = float(ap)
            pc[CLASS_NAMES.get(ci, f"cls{ci}")] = round(v, 4) if v >= 0 else None
        out["coco_per_class"] = pc
    return out


# =============================================================================
# Cached predictions collection (1 pass for val_loss + predictions)
# =============================================================================
@torch.no_grad()
def collect_predictions(model, loader, device, amp: bool = True
                        ) -> Tuple[float, List[Dict], List[Dict]]:
    """Run model on loader, collect val_loss + predictions + GTs.

    Returns:
        val_loss: float
        preds: list of {"boxes", "scores", "labels"}
        gts:   list of {"boxes", "labels", ...}
    """
    # 1) val_loss in train mode (BN active)
    model.train()
    tvl = 0.0
    nb = 0
    for imgs, targets in loader:
        imgs = [i.to(device) for i in imgs]
        td = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
               for k, v in t.items()} for t in targets]
        try:
            with torch.amp.autocast("cuda", enabled=amp):
                ld = model(imgs, td)
                tvl += sum(v.item() for v in ld.values()
                           if isinstance(v, torch.Tensor) and torch.isfinite(v)) or 0.0
                nb += 1
        except Exception:
            continue
    val_loss = tvl / max(nb, 1)

    # 2) predictions in eval mode — lower score_thresh for tiny objects
    model.eval()
    prev_score = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = 0.001
    preds_all, gts_all = [], []
    for imgs, targets in loader:
        try:
            with torch.amp.autocast("cuda", enabled=amp):
                p = model([i.to(device) for i in imgs])
            for pp, tt in zip(p, targets):
                preds_all.append({k: v.cpu() for k, v in pp.items()})
                gts_all.append({k: v.cpu() if isinstance(v, torch.Tensor) else v
                                for k, v in tt.items()})
        except Exception:
            continue
    model.roi_heads.score_thresh = prev_score
    return val_loss, preds_all, gts_all


# =============================================================================
# Main evaluate() — 1 pass, returns full metrics dict
# =============================================================================
def evaluate(model, loader, device, measure_fps_flag: bool = False,
             fps_img_size: Tuple[int, int] = (640, 800)) -> Dict:
    """Run full evaluation. Returns metrics dict suitable for logging.

    Args:
        model: torch model
        loader: DataLoader for validation
        device: torch device
        measure_fps_flag: if True, measure FPS on dummy data
        fps_img_size: (H, W) for FPS dummy input
    """
    val_loss, preds, gts = collect_predictions(model, loader, device)

    # mAP@50 only
    if _HAS_TORCHMETRICS:
        m50 = MeanAveragePrecision(
            iou_type="bbox", iou_thresholds=[0.50],
            max_detection_thresholds=[1, 10, 500],
            class_metrics=False,
        ).to("cpu")
        for p, g in zip(preds, gts):
            m50.update([p], [g])
        r50 = m50.compute()
        mAP50 = max(float(r50.get("map_50", r50.get("map", 0.0))), 0.0)
    else:
        mAP50 = 0.0

    # Scale AP
    sap = compute_scale_ap(preds, gts)
    tgt = sum(sap.get(f"n_gt_{s}", 0) for s in ("micro", "tiny", "small", "large"))
    prim = (sum(sap.get(f"AP_{s}", 0.0) * sap.get(f"n_gt_{s}", 0)
                for s in ("micro", "tiny", "small", "large")) / tgt
            if tgt > 0 else 0.0)

    # COCO mAP
    coco = evaluate_coco(preds, gts, class_metrics=True)

    # Precision & Recall
    pr = compute_precision_recall(preds, gts, iou_thresh=0.5, score_thresh=0.05)

    # FPS (optional)
    fps = 0.0
    if measure_fps_flag:
        fps = measure_fps(model, device, batch_size=1, img_size=fps_img_size)

    metrics = {
        "val_loss": round(val_loss, 4),
        "mAP_primary": round(prim, 6),
        "mAP_50":      round(mAP50, 6),
        **sap,
        **coco,
        **pr,
    }
    if fps > 0:
        metrics["FPS"] = round(fps, 1)
        metrics["inference_ms"] = round(1000.0 / fps, 2)

    print(f"\n{'='*50}\nEVAL\n{'='*50}")
    print(f"  val_loss    : {metrics['val_loss']:.4f}")
    print(f"  mAP(scale)  : {metrics['mAP_primary']:.4f}")
    print(f"  mAP@50      : {metrics['mAP_50']:.4f}")
    print(f"  AP_micro(n={metrics.get('n_gt_micro',0):4d}): {metrics.get('AP_micro',0):.4f}")
    print(f"  AP_tiny (n={metrics.get('n_gt_tiny',0):4d}): {metrics.get('AP_tiny',0):.4f}")
    print(f"  AP_small(n={metrics.get('n_gt_small',0):4d}): {metrics.get('AP_small',0):.4f}")
    print(f"  AP_large(n={metrics.get('n_gt_large',0):4d}): {metrics.get('AP_large',0):.4f}")
    print(f"\n  COCO AP     : {coco.get('coco_AP', 0):.4f}")
    print(f"  COCO AP@50  : {coco.get('coco_AP50', 0):.4f}")
    print(f"  COCO AP@75  : {coco.get('coco_AP75', 0):.4f}")
    print(f"  COCO AP_small : {coco.get('coco_AP_small', 0):.4f}")
    print(f"  COCO AR@100 : {coco.get('coco_AR100', 0):.4f}")
    print(f"\n  Precision   : {pr.get('Precision', 0):.4f}")
    print(f"  Recall      : {pr.get('Recall', 0):.4f}")
    if fps > 0:
        print(f"  FPS         : {fps:.1f} ({1000.0/fps:.1f} ms/img)")
    return metrics