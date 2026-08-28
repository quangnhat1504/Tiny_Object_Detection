---
title: TinyPerson Acquisition and Real-Data Fixture - 2026-08-04
type: analysis
created: 2026-08-04
updated: 2026-08-04
sources: [Paper_A_SA_ALW_Conference_Refinement_Plan.md]
tags: [paper-a, tinyperson, data-acquisition, fixtures]
---

# TinyPerson Acquisition and Real-Data Fixture - 2026-08-04

**Question:** Is the official ScaleMatch TinyPerson package acquired, verified, and ready for Paper A adapter/evaluator work, and does the Paper A fixture chain run on real data?

## Acquisition (completed 2026-08-04)

- The legacy Roboflow derivative in workspace `data/` was verified as
  `NO_GO_CURRENT_DERIVATIVE` and is NOT used for Paper A. See
  [[SA-ALW Paper Resume Checkpoint - 2026-08-02]].
- The official package (source pass `pmcq`, Google Drive) was downloaded and
  restructured to `D:\paper_a_data\TinyPerson\tiny_set` — 4,173 files / 4.14 GB.
- All archive SHA-256 hashes, official counts, and distribution terms are
  recorded in `D:\paper_a_data\TinyPerson\acquisition_manifest.json` and in
  `paper_a/datasets/tinyperson.md` (Acquisition Record).
- Counts match the official statistics table exactly: erase-train 794 images
  (`labeled_images` 717 + `labeled_dense_images` 48 + `pure_bg_images` 29),
  erase-test 816 images.
- Access classes: erase-train material = A1; test material = A3 storage-only
  (hashed, never mounted for evaluation).

## Corner format resolved

The adapter assumed `corner=[x1,y1,x2,y2]` while some notes suggested
`[x1,y1,w,h]`. Real data confirms `[x1,y1,x2,y2]`: e.g. the same source image
`bb_V0032_I0001640.jpg` yields corners `[0,0,640,512]` and `[590,0,1230,512]`,
i.e. overlapping 640x512 crops. All 8,256 train corner-task records satisfy
`(x2-x1, y2-y1) == (width, height)` with zero mismatches.

## Fixture results on real data (2026-08-04)

- Full Paper A test suite: `57 passed, 1 skipped` (system Python 3.13,
  pytest 9.0.3).
- Adapter fixture on
  `erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json`:
  8,256 records, 32,430 positives (matches official docs), 0 ignored
  (uncertain annotations already erased), 3,206 records with positives,
  0 corner-size mismatches. A 100-image decode sample (92 with offset
  corners) passed crop-size and box-in-crop validation.
- Pinned official evaluator smoke (perfect detections on a consistent
  200-image GT subset): AP25/AP50/AP75 `all` = `0.9889`, AP50 `tiny` = `1.0`,
  AR50 `all` = `0.9988`; evaluator SHA-256 `222b3173...` unchanged.
- Report JSON: `.runtime/tinyperson_real_data_fixture.json`.

## Open items

- TinyPerson has NO official validation split; the split policy must be frozen
  in the protocol ledger before any schedule fitting uses a held-out set.
- Train-only P10/P90 scale bounds for TinyPerson remain unfitted (G1).
- AI-TOD-v2 images remain unacquired (G2 still not fully ready).
- TinyPerson redistribution terms restrict Kaggle hosting and paper
  qualitative figures; hosting decision pending.

## Status

`TINYPERSON_FIXTURE_ON_REAL_DATA_PASS; G1_BOUNDS_PENDING; G2_AITOD_IMAGES_PENDING`
