# Phase 2 Report

Status: `REVISE`
Current derivative decision: `NO_GO_CURRENT_DERIVATIVE`
Paper A final-test material accesses: `1` (AI-TOD-v2 structural audit)
Paper A final-test performance evaluations: `0`

## Completed

- Generated a row-level legacy manifest with source, sequence, augmentation,
  image hash, annotation hash, and split fields.
- Audited all 1,570 processed files and 654 source IDs.
- Verified zero exact source-ID and exact-image-hash overlap across legacy
  splits.
- Detected sequence leakage across every split pair: 30 train-validation, 23
  train-test, and 20 validation-test groups.
- Implemented tile-offset inversion, clipping, class-aware deduplication,
  fixed `maxDets`, empty-image retention, and original-GT construction.
- Passed all nine reconstruction tests, including border-crossing objects and
  original annotation count preservation.
- Acquired and hashed all four official AI-TOD-v2 annotation splits.
- Implemented and tested the AI-TOD-v2 category, ignore/crowd, and prediction
  conversion adapter.
- Locked the official AI-TOD-v2 evaluator and reproduced perfect-box AP, AP50,
  AP75, and tiny AP in a deterministic fixture.
- Implemented the TinyPerson binary task-all adapter with explicit
  ignore/uncertain/crowd routing.
- Loaded the hash-locked official TinyPerson evaluator under NumPy 2 and
  reproduced AP25/AP50/AP75 of `1.0` while an uncertain-region detection is
  ignored through IOD.
- Closed the reused legacy test to all Paper A claims.

## Evidence Created

- `splits/legacy_split_manifest.csv`
- `splits/split_audit.json`
- `evaluation/tile_to_original.py`
- `tests/test_tile_to_original.py`
- `dataset_card.md`
- `test_access_log.md`

The deterministic legacy manifest SHA-256 is
`a1ed88b4a6f588bae9f8e8fe0f514ff396d36b2181e11e22100f4daaab0008f9`.

## Gate Decision

`G2 REVISE`. The reconstruction core is technically validated, but the current
SOD derivative is a submission-data `NO-GO`: sequence groups leak across every
legacy split, the test has prior exposure, and upstream crowd/ignore provenance
is unavailable. Repartitioning these already exposed samples cannot produce a
fresh final test.

## Paper A Pivot

The main evidence path is now:

1. TinyPerson official protocol as D1.
2. AI-TOD-v2 official protocol as D2.
3. The current SOD derivative only for debugging or supplementary development.

No Paper A pilot or benchmark run may start until the selected public dataset
copy, official split, annotation semantics, original-image evaluator, and
train-only schedule coordinate system are hashed and frozen.

## Remaining G2 Work

1. Acquire and hash the AI-TOD image package and official TinyPerson package.
2. Acquire and audit the official TinyPerson task-all annotations and prepared
   erased training images against the tested adapter contract.
3. Freeze train-only scale percentiles in the exact detector-input coordinate
   system.
4. Re-run both official evaluator fixtures inside every Kaggle benchmark
   package before model training.

## Acquisition Update: AI-TOD-v2

All four official AI-TOD-v2 annotation JSON files are now locally acquired and
SHA-256 hashed. Train (`11,214` images) and validation (`2,804`) are filename
disjoint; trainval (`14,018`) is their exact union; official test (`14,018`) is
filename-disjoint from trainval. The shared image package remains pending, so
this does not yet pass G2 and does not authorize any test evaluation.

The structural audit parsed the test annotation file and derived aggregate
counts and size summaries. This material access is now disclosed explicitly in
`test_access_log.md`; no prediction or test metric was computed. AI-TOD-v2 is
therefore performance-locked rather than literally unseen, and all future
schedule/method/checkpoint choices remain train/validation-only.

The original-image adapter and official evaluator fixture now pass locally.
This is technical validation only, not a model experiment or performance
result. The Kaggle copy must rerun the fixture before any assigned package.

## TinyPerson Adapter and Evaluator Update

The binary task-all adapter and official evaluator fixture pass locally. The
fixture locks IoU `0.25/0.50/0.75`, `maxDets=200`, all seven official area
labels, uncertain handling, and IOD matching. This closes the implementation
portion of the TinyPerson evaluator gate, but the official archive is still
absent, so its split/provenance gate remains open and no run is authorized.
