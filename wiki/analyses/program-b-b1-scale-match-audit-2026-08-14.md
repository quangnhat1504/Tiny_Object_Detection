---
title: Program B B1 Scale-Match Audit - 2026-08-14
type: analysis
created: 2026-08-14
status: REVISE_SCALE_AND_ADAPTER_MISMATCH
sources:
  - wiki/analyses/program-b-b1-cbl-pc-protocol-freeze-2026-08-14.md
  - .runtime/local/program_b/b1_scale_match_audit_20260814.json
  - common/config.py
  - common/dataset.py
  - scripts/train_frcnn_metric.py
tags: [program-b, scale-audit, tinyperson, cbl, pre-run, revise]
---

# Program B B1 Scale-Match Audit - 2026-08-14

## Verdict

**REVISE — do not auto-accept B1 or authorize B2 training.**

The Program B source annotation is official TinyPerson and its group-disjoint
split is structurally valid. It is **not scale-matched** to the iterative-CBL
execution baseline under the frozen `640/800` detector transform. In addition,
the frozen iterative-CBL trainer cannot consume this COCO original-image split:
`build_training_datasets()` only constructs `YOLOTinyDataset` from
`train|valid/images` plus YOLO labels, then tiles every image at `512` with
`64` overlap.

This is a protocol/data-interface failure, not a model-performance result.
No training, external evaluation, historical locked-test access, or claim is
permitted from this B1 package.

## Verified transform and data contracts

- The iterative-CBL training path is `data/train` YOLO originals → non-empty
  `512`-pixel tiles with `64` overlap → Faster R-CNN `640/800` transform.
- Program B currently uses un-tiled TinyPerson original images at native sizes.
- The actual torchvision `GeneralizedRCNNTransform(640, 800)` produced:
  - a `1920×1080` original image → `800×450` (scale `0.4167`);
  - a `512×512` tile → `640×640` (scale `1.25`).
- `scripts/train_frcnn_metric.py` has no Program-B COCO data-root/split adapter;
  it calls `build_training_datasets(use_patches=False, ...)`.

## Measured model-input object scale

Scale is `sqrt(clipped_bbox_area)` after the same `640/800` detector resize.
The legacy distribution is measured on all pre-augmentation box occurrences in
non-empty training tiles; its second column applies the existing expected
2× tiny-tile sampler weighting.

| Surface | Objects | P25 | Median | P75 | P90 | ≤8 px | ≤16 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| Program B original-image train | 16,193 | 3.76 | 5.63 | 9.23 | 15.64 | 69.60% | 90.39% |
| Program B original-image validation | 2,240 | 4.38 | 6.54 | 10.23 | 14.80 | 61.34% | 92.10% |
| Iterative-CBL legacy tiled train | 131,836 | 9.45 | 13.92 | 21.79 | 35.39 | 16.07% | 58.56% |
| Iterative-CBL legacy, sampler-weighted | 253,787 | 9.30 | 13.53 | 20.49 | 31.76 | 16.69% | 60.85% |

The Program B train median is **5.63 px**, versus **13.53 px** for the
sampler-weighted iterative-CBL surface: 41.6% of the reference scale (a 2.40×
reduction). Its ≤8 px share is 69.60% versus 16.69%. This is not a small
sampling fluctuation and cannot support a matched baseline/candidate matrix.

## Audit artifact

- JSON: `.runtime/local/program_b/b1_scale_match_audit_20260814.json`
- SHA-256: `d20c75f5fceb8f383c2c2fba77afd048afb42632024b8d7959edbd63331f93d3`
- Reproducibility helper: `.runtime/local/program_b/audit_b1_scale_match_20260814.py`

The artifact uses the actual `YOLOTinyDataset` tile index and labels, filters
TinyPerson ignore/uncertain/crowd annotations as the original-image adapter
does, and uses the verified torchvision resize geometry. It does not load a
checkpoint or produce model predictions.

## Required revision before a new B1 review

1. Build and test a Program B COCO-to-tile adapter that applies the baseline
   `512/64` tiling to only the frozen source-group-disjoint train/validation
   originals, preserves source/original IDs and ignore semantics, and emits the
   exact training surface expected by the current CBL/PC trainer.
2. Reconstruct/deduplicate tile predictions onto the same original-image IDs for
   the pinned official evaluator; never evaluate tiles as independent images.
3. Recompute this audit after tiling and require predeclared scale-match bounds
   before freezing a replacement source bundle, split manifest, and B2 pre-run
   report.
4. Keep the external-test surface unmounted and historical CBL locked test
   excluded throughout the revision.

## Related pages

- [[Program B B1 CBL/PC Protocol Freeze - 2026-08-14]]
- [[Program B CBL Pivot Decision — 2026-08-14]]
- [[Strategic Research Roadmap — 2026-08-14]]
- [[Wiki Overview]]
- [[Wiki Log]]
