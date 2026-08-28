"""Cache tile predictions to pickle for fast offline tuning/eval."""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
import tqdm

from common.config import seed_all, SEED
from common.metrics import get_metric_fn
from common.model import build_model
from common.dataset import YOLOTinyDataset, collate_fn


def _infer_model_config(ckpt_path: Path, ck: dict) -> tuple[str, str]:
    cfg = ck.get("config", {})
    run_name = ckpt_path.parent.name
    if run_name.startswith("frcnn_standard"):
        return "iou", "everywhere"
    metric = cfg.get("metric")
    placement = cfg.get("placement")
    if metric and placement:
        return metric, placement
    parts = run_name.split("__")
    if len(parts) >= 2:
        return metric or parts[0], placement or parts[1]
    return metric or "sa_alw_full", placement or "la_loss"


def _parse_args():
    p = argparse.ArgumentParser(description="Cache tile predictions once for fast offline tuning")
    p.add_argument("--split", choices=["valid", "test"], default="valid",
                   help="Dataset split")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None,
                   help="Device override (default: cuda if available)")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Checkpoint path (default: runs/sa_alw_full__la_loss__seed42/best.pt)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output pickle path (default: runs/tile_preds_<split>_seed42.pkl)")
    p.add_argument("--force", action="store_true",
                   help="Re-cache even if file exists")
    return p.parse_args()


def main():
    args = _parse_args()
    seed_all(SEED)

    device = torch.device("cuda" if (args.device or "cuda") == "cuda"
                          and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = args.ckpt or (ROOT / "runs/sa_alw_full__la_loss__seed42/best.pt")
    pred_file = args.out or (ROOT / f"runs/tile_preds_{args.split}_seed42.pkl")
    if not pred_file.is_absolute():
        pred_file = ROOT / pred_file

    if pred_file.exists() and not args.force:
        print(f"Cache exists: {pred_file} (use --force to re-cache)")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    metric, placement = _infer_model_config(ckpt_path, ck)
    mt = None if metric == "iou" else get_metric_fn(metric)
    print(f"Resolved config: metric={metric}, placement={placement}")
    model = build_model(
        mt,
        placement,
        cfg.get("reliability_thr", 16.0),
        box_loss_type=cfg.get("box_loss", "metric"),
        use_quality_score=bool(cfg.get("quality_score", False)),
        quality_loss_weight=float(cfg.get("quality_loss_weight", 0.0) or 0.0),
        use_quality_focal=bool(cfg.get("quality_focal", False)),
        quality_focal_beta=float(cfg.get("quality_focal_beta", 2.0)),
        use_rank_sort=bool(cfg.get("rank_sort", False)),
        rank_sort_delta=float(cfg.get("rank_sort_delta", 0.5)),
        use_double_head=bool(cfg.get("double_head", False)),
        double_head_reg_roi_scale=float(
            cfg.get("double_head_reg_roi_scale", 1.3)),
        double_head_num_convs=int(cfg.get("double_head_num_convs", 4)),
        cbl_refine_steps=int(cfg.get("cbl_refine_steps", 0)),
        cbl_refine_blend=float(cfg.get("cbl_refine_blend", 1.0)),
        cbl_refine_score_threshold=float(
            cfg.get("cbl_refine_score_threshold", 0.0)),
        cbl_refine_extra_min_size_ratio=float(
            cfg.get("cbl_refine_extra_min_size_ratio", 0.0)),
        cbl_refine_train_weight=float(
            cfg.get("cbl_refine_train_weight", 0.0)),
        cbl_refine_train_steps=int(
            cfg.get("cbl_refine_train_steps", 1)),
        cbl_alpha=float(cfg.get("cbl_alpha", 5.0)),
        cbl_num_bins=int(cfg.get("cbl_num_bins", 6)),
        cbl_grid_beta=float(cfg.get("cbl_grid_beta", 1.0)),
        cbl_um_weight=float(cfg.get("cbl_um_weight", 1.0)),
    ).to(device)
    model.load_state_dict(ck["model"], strict=False)

    img_dir = ROOT / "data" / args.split / "images"
    lbl_dir = ROOT / "data" / args.split / "labels"
    td = YOLOTinyDataset(img_dir=img_dir, lbl_dir=lbl_dir, is_train=False)

    model.eval()
    prev_thr = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = 0.001

    all_preds = []
    loader = DataLoader(td, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=(device.type == "cuda"))
    for imgs, _ in tqdm.tqdm(loader, desc="FRCNN tiles"):
        imgs = [i.to(device) for i in imgs]
        with torch.no_grad():
            preds = model(imgs)
        for p in preds:
            all_preds.append({
                "boxes": p["boxes"].cpu(),
                "scores": p["scores"].cpu(),
                "labels": p["labels"].cpu(),
            })
    model.roi_heads.score_thresh = prev_thr

    data = {
        "meta": {
            "checkpoint": str(ckpt_path),
            "split": args.split,
            "ckpt_epoch": ck.get("epoch"),
            "best_mAP50": ck.get("best_mAP50"),
            "metric": metric,
            "placement": placement,
            "config": cfg,
        },
        "preds": all_preds,
        "tile_index": td.tile_index,
        "labels_cache": td.labels_cache,
    }
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(all_preds)} tile predictions -> {pred_file}")


if __name__ == "__main__":
    main()
