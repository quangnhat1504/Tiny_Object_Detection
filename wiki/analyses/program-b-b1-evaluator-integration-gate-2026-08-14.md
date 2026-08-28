---
title: Program B B1 Evaluator-Integration Gate - 2026-08-14
type: analysis
created: 2026-08-14
status: B1_READY_FOR_OWNER_APPROVAL_NO_TRAINING
sources:
  - paper_a/evaluation/program_b_tiled.py
  - paper_a/evaluation/tile_to_original.py
  - paper_a/evaluation/tinyperson_official.py
  - .runtime/local/program_b/b1_revised_source_20260814/program_b_b1_revised_source_manifest.json
---

# Program B B1 Evaluator-Integration Gate - 2026-08-14

## Decision

B1 technical data/evaluation gates now pass. This authorizes an owner decision
on a separately frozen B2 train-from-scratch protocol; it does not itself start
training or make a performance claim.

## Verified chain

1. `build_tiled_datasets()` constructs explicit, frozen train/validation paths,
   rather than implicitly using the legacy data root.
2. `records_from_tile_manifest()` binds each validation loader filename to its
   immutable tile record, containing original image ID/dimensions and offsets.
   Runtime binding passed: 1,684 tiles and 118 original images.
3. `evaluate_tiled_predictions()` maps tile detections to original coordinates,
   applies class-aware NMS once per original image, converts to COCO detections,
   then dispatches exactly one original-image evaluator call.
4. A real pinned-evaluator integration smoke passed using the Program B
   validation annotation and a manifest-backed tile prediction. It reported
   `tinyperson_official` with evaluator SHA-256
   `222b3173510e7a89bd03d077dce5d4a11e23ea6a7cd22afbbe930817b0886557`.
   This was a synthetic plumbing check only, not model performance evidence.
5. Scale contract remains passed under the baseline `512/64` tile and `640/800`
   transform path.

## Verification

- focused suites: 16 tests PASS;
- revised-source bundle: 13 files hash-verified;
- source archive SHA-256:
  `09c698e1a0e4f3b25335f47ff14e0359e5ca3bb5040813b285e10f81dee4d085`;
- `git diff --check`: PASS.

## B2 boundary

B2 must train baseline and candidates from scratch under one frozen TinyPerson
original-image split, budgets, seeds, model transforms, and manifest-backed
official-validation procedure. It must not reuse historical metrics or access
any locked/external test surface. B2 requires explicit owner authorization
before any training process or Kaggle job is started.
