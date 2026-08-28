---
title: P2 Feature Implementation - 2026-05-31
type: source
created: 2026-05-31
updated: 2026-05-31
source_file: working/code/9_hard_switch_p2.ipynb
sources: [working/code/9_hard_switch_p2.ipynb]
tags: [p2-features, architecture, micro-objects, implementation]
---

## P2 Feature Implementation - 2026-05-31

## Source

Kaggle notebook `9_hard_switch_p2.ipynb` implementing P2 (stride-4) feature level in FPN to improve detection for micro objects (<8px). Built on top of HARD_SWITCH_NWD_GCD baseline (file 6).

## Motivation

From [[Tiny Object Architecture Improvement - 2026-05-31]], the (0,8)px bin contains **19,196 instances (27% of dataset)** but maps to **sub-pixel resolution** on P3 (stride-8). P2 (stride-4) provides 2× spatial resolution for micro objects.

## Implementation Changes

### Cell 1: Imports
Added P2-specific imports:
```python
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
```

### Cell 2: Config
Memory optimization and anchor updates:
- `METRIC_NAME = "HARD_SWITCH_P2"`
- `BATCH_SIZE = 2` (reduced from 4 for P2 memory)
- `GRAD_ACCUM = 4` (effective batch = 8, reduced from 16)
- `RPN_NUM_PROPOSALS_TRAIN = 2000` (reduced from 3000)
- `RPN_NUM_PROPOSALS_TEST = 1000` (reduced from 1500)
- `DETECTIONS_PER_IMG = 400` (reduced from 500)
- `ANCHOR_SIZES = ((4,6,8), (8,12,16), (24,32,48), (64,96,128), (192,256,384))` — **6 levels (P2-P6)** instead of 5
- `OUTPUT_DIR = "/kaggle/working/runs/rfla_hard_switch_p2"`

### Cell 6: Model Architecture
Custom FPN backbone with P2:
```python
backbone = resnet_fpn_backbone(
    backbone_name='resnet50',
    weights='IMAGENET1K_V1',
    trainable_layers=5,
    returned_layers=[1, 2, 3, 4],  # C2→P2, C3→P3, C4→P4, C5→P5
    extra_blocks=LastLevelMaxPool()  # adds P6
)

roi_pooler = MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],  # P2, P3, P4, P5 (4 levels)
    output_size=7,
    sampling_ratio=2
)
```

**Key differences from baseline**:
- Baseline uses `fasterrcnn_resnet50_fpn()` with default P3-P6 (5 levels)
- P2 version uses custom `resnet_fpn_backbone()` with P2-P6 (6 levels)
- RoI pooler explicitly uses P2-P5 (skips P6 for RoI)
- Anchor generator handles 6 levels instead of 5

### Cells 3-5, 7-9: Unchanged
Metric (HARD_SWITCH), dataset, RFLA assignment, training loop, and evaluation remain identical to baseline.

## Anchor Design for P2

| FPN Level | Stride | Anchor Sizes | Target Objects |
|-----------|--------|--------------|----------------|
| P2 | 4 | (4, 6, 8) | Micro (0-8px) |
| P3 | 8 | (8, 12, 16) | Tiny (8-16px) |
| P4 | 16 | (24, 32, 48) | Small (16-32px) |
| P5 | 32 | (64, 96, 128) | Medium (32-96px) |
| P6 | 64 | (192, 256, 384) | Large (>96px) |

Aspect ratios: `(0.33, 0.5, 1.0)` — optimized for vertical standing pose (81% of objects).

## Memory Optimization Strategy

P2 adds ~30-50% memory overhead (4× spatial resolution vs P3). Mitigations:
1. Reduce batch size: 4 → 2
2. Reduce gradient accumulation effective batch: 16 → 8
3. Reduce RPN proposals: 3000 → 2000 (train), 1500 → 1000 (test)
4. Reduce max detections: 500 → 400
5. PyTorch AMP enabled (`torch.amp.autocast`)
6. Empty cache every 50 steps

## Expected Results

Based on architecture improvement analysis:
- **AP_micro**: +5-10% (from 0.2776 → 0.30-0.35)
- **AP_tiny**: +2-3% (from 0.5721 → 0.59-0.60)
- **AP@75**: +0.01-0.02 (from 0.0428 → 0.05-0.06)
- **mAP(scale)**: +0.02-0.04 (from 0.5770 → 0.60-0.62)
- **Training time**: +20-30% per epoch
- **Memory**: +30-50% GPU usage

## Comparison to Baseline

| Metric | Baseline (file 6) | P2 (file 9) | Change |
|--------|-------------------|-------------|--------|
| FPN levels | P3-P6 (5) | P2-P6 (6) | +P2 |
| Batch size | 4 | 2 | -50% |
| Effective batch | 16 | 8 | -50% |
| RPN proposals (train) | 3000 | 2000 | -33% |
| Anchors per level | 3 | 3 | same |
| Total anchor levels | 5 | 6 | +20% |
| Smallest anchor | 8px | 4px | -50% |

## Kaggle Compliance

Notebook follows Kaggle Notebook Rules:
- ✅ Self-contained (no local `.py` imports)
- ✅ All code inline in cells
- ✅ Paths use `/kaggle/input/` and `/kaggle/working/`
- ✅ GPU check cell included
- ✅ AMP uses `torch.amp.autocast('cuda')` (new API)
- ✅ All 9 cells compile successfully

## Next Steps

1. Upload `9_hard_switch_p2.ipynb` to Kaggle
2. Run 12 epochs on dual T4 GPUs
3. Compare results with HARD_SWITCH baseline (file 6)
4. If AP_micro > 0.30 and AP@75 > 0.05 → proceed to anchor tuning (Step 2)
5. If OOM occurs → reduce batch size to 1 or enable gradient checkpointing

## Related Pages

- [[Tiny Object Architecture Improvement - 2026-05-31]]
- [[SAH-GD Hybrid Metrics Comparison]]
- [[Tiny Object Detection Metrics]]
- [[RFLA]]
