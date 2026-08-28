---
title: Program B B1 Tiled Scale Revision Audit - 2026-08-14
type: analysis
created: 2026-08-14
status: SCALE_CONTRACT_PASS_PENDING_EVALUATOR_INTEGRATION
sources:
  - wiki/analyses/program-b-b1-scale-match-audit-2026-08-14.md
  - .runtime/local/program_b/b1_tiled_scale_match_audit_20260814.json
  - scripts/build_program_b_coco_tiles.py
  - scripts/test_build_program_b_coco_tiles.py
tags: [program-b, cbl, tinyperson, tiling, scale-audit, pre-run]
---

# Program B B1 Tiled Scale Revision Audit - 2026-08-14

## Result

The tested COCO-to-tile revision restores the exact **geometric pipeline** of
the iterative-CBL training surface: original images are cropped into `512` tiles
with `64` overlap and then receive the same detector `640/800` transform.

The frozen original-ID split is preserved exactly:

- train: 628 original images → 9,950 materialized tiles;
- validation: 118 original images → 1,684 materialized tiles;
- train/validation original-ID overlap: zero;
- source groups remain inherited from and confined by the frozen split manifest.

This corrects the former 2.40× median scale mismatch. It is **not yet an
execution approval**: B1 still needs actual integration of the tile manifest
with original-image prediction reconstruction in the Program B trainer/evaluator
path.

## Post-transform scale comparison

All values are `sqrt(area)` after the common `640/800` model transform. Both
training rows apply the existing expected `2×` tiny-tile sampler weighting.

| Surface | P25 | Median | P75 | ≤8 px | ≤16 px |
|---|---:|---:|---:|---:|---:|
| Program B tiled train | 10.31 | 15.32 | 24.29 | 12.85% | 52.93% |
| Iterative-CBL tiled train | 9.30 | 13.53 | 20.49 | 16.69% | 60.85% |
| Ratio / difference | 1.109× | 1.133× | 1.185× | -3.85 pp | -7.92 pp |

The revised Program B distribution is slightly larger, but it is no longer in
a different geometric regime. The operational Program B scale contract is now
frozen as: P25/median/P75 ratios must each be within `[0.80, 1.20]`, and the
absolute differences in the ≤8 px and ≤16 px shares must each be at most `10`
percentage points. This repeat audit passes all five conditions:
`1.109/1.133/1.185` ratios and `-3.85/-7.92` percentage-point share deltas.
The rule was frozen before any B2 training or metric exists; it governs every
future rebuild/re-audit of this surface.

## Verified artifacts

- Tile builder: `scripts/build_program_b_coco_tiles.py`
- TDD test: `scripts/test_build_program_b_coco_tiles.py` (RED verified before
  implementation; GREEN pass afterwards)
- Tile manifests:
  - train SHA-256: `6175e8ef8b74a3534c3a8e0227ed40a251a4fe136b902870e6c3112b2604d755`
  - validation SHA-256: `8b61d70256bbd5196b73b342d20e81471608bba0477227f284f4e01cbe4cc985`
- Audit JSON: `.runtime/local/program_b/b1_tiled_scale_match_audit_20260814.json`
- Canonical audit SHA-256: `6659c7c36c94ea32ab5b53238206333b486c525823dc43408106e96556493f5f`
- Adapter/reconstruction fixture suite: 11 tests PASS.

The tile manifest carries each tile's original image ID, original dimensions,
and `x1,y1,x2,y2` offsets. It is sufficient input for the existing
`reconstruct_predictions()` contract, but no Program B inference/evaluation
entry point consumes it yet. Therefore tile-level metrics remain prohibited.

## Required closeout

Before B1 can be reconsidered:

1. Implement and test the Program B validation collector that loads tile records
   from this manifest, reconstructs/deduplicates predictions on original IDs,
   and calls the pinned official evaluator.
2. Freeze a replacement source bundle that includes the tile builder, manifest
   reader, evaluation collector, tests, and exact tiled-data hashes.
3. Issue a replacement B1 report and seek explicit B2 training approval.

No model training, Kaggle activity, external-test access, or historical
locked-test access occurred during this revision.

## Related pages

- [[Program B B1 Scale-Match Audit - 2026-08-14]]
- [[Program B B1 CBL/PC Protocol Freeze - 2026-08-14]]
- [[Wiki Overview]]
- [[Wiki Log]]
