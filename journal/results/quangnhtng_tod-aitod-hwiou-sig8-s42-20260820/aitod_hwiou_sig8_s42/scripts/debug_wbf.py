"""Debug: compare standard WBF vs SmartWBF precompute for one image."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from common.config import DEVICE, seed_all, SEED
from common.metrics import get_metric_fn
from common.model import build_model
from common.dataset import YOLOTinyDataset, collate_fn

FRCNN_CKPT = ROOT / "runs/sa_alw_full__la_loss__seed42/best.pt"

seed_all(SEED)
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

model.eval()
prev_thr = model.roi_heads.score_thresh
model.roi_heads.score_thresh = 0.001

all_tile_preds = []
import tqdm
loader = DataLoader(td, batch_size=4, shuffle=False, num_workers=0,
                    collate_fn=collate_fn, pin_memory=(DEVICE.type == "cuda"))
for imgs, _ in tqdm.tqdm(loader, desc="FRCNN tiles"):
    imgs = [i.to(DEVICE) for i in imgs]
    with torch.no_grad():
        preds = model(imgs)
    for p in preds:
        all_tile_preds.append((p["boxes"].cpu(), p["scores"].cpu(), p["labels"].cpu()))
model.roi_heads.score_thresh = prev_thr

# Pick the first image with boxes in multiple tiles
img_groups = {}
for idx in range(len(td)):
    img_idx = td.tile_index[idx][0]
    if img_idx not in img_groups:
        img_groups[img_idx] = {"tiles": [], "coords": []}
    img_groups[img_idx]["tiles"].append(all_tile_preds[idx])
    img_groups[img_idx]["coords"].append(td.tile_index[idx])

# Pick first image with boxes in >1 tile
for img_idx in sorted(img_groups)[:5]:
    group = img_groups[img_idx]
    cache_entry = td.labels_cache[img_idx]
    if len(cache_entry) == 2:
        boxes, (W, H) = cache_entry
    else:
        boxes, W, H = cache_entry
    
    # Count tiles with boxes
    tiles_with_boxes = sum(1 for (b, s, l) in group["tiles"] if b.numel() > 0)
    total_boxes = sum(b.numel() for (b, s, l) in group["tiles"])
    
    if tiles_with_boxes > 1:
        print(f"\nImage {img_idx}: {W}x{H}, {tiles_with_boxes} tiles with boxes, {total_boxes} total boxes")
        
        # Compare standard WBF
        from scripts.eval_cascade import wbf_fusion as standard_wbf
        
        iou_thr = 0.60
        score_thr = 0.10
        
        filtered = []
        for (b, s, l) in group["tiles"]:
            keep = s >= score_thr
            filtered.append((b[keep], s[keep], l[keep]))
        
        img_coords_std = [(tx1, ty1, tx2-tx1, ty2-ty1)
                          for (_i, tx1, ty1, tx2, ty2) in group["coords"]]
        
        fused_std = standard_wbf(filtered, img_coords_std, (W, H), iou_thr=iou_thr)
        print(f"  Standard WBF: {len(fused_std['boxes'])} boxes")
        
        from scripts.tune_smart_wbf import precompute_image_data, smart_fuse_from_precomputed
        
        img_coords_smart = [(tx1, ty1, tx2-tx1, ty2-ty1)
                           for (_i, tx1, ty1, tx2, ty2) in group["coords"]]
        
        pre = precompute_image_data(group["tiles"], img_coords_smart, (W, H))
        if pre:
            print(f"  Precomputed: n={pre['n']} boxes, areas={pre['areas'].tolist()[:5]}...")
            fused_smart = smart_fuse_from_precomputed(
                pre, base_iou_thr=iou_thr, score_thr=score_thr,
                fusion_mode="weighted_avg", adaptive_thr=False)
            print(f"  SmartWBF: {len(fused_smart['boxes'])} boxes")
            
            # Compare area distributions
            if len(fused_std['boxes']) > 0 and len(fused_smart['boxes']) > 0:
                std_areas = ((fused_std['boxes'][:,2]-fused_std['boxes'][:,0]) *
                             (fused_std['boxes'][:,3]-fused_std['boxes'][:,1]))
                smart_areas = ((fused_smart['boxes'][:,2]-fused_smart['boxes'][:,0]) *
                              (fused_smart['boxes'][:,3]-fused_smart['boxes'][:,1]))
                print(f"  Standard area range: {std_areas.min():.0f}-{std_areas.max():.0f}")
                print(f"  Smart area range: {smart_areas.min():.0f}-{smart_areas.max():.0f}")
        break
