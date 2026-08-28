# Paper A Protocol Ledger

Immutable record of protocol decisions. Each entry is `PROPOSED` until the
user freezes it; frozen entries must never be reopened except by a newer
superseding entry. Every Paper A artifact that depends on a protocol choice
must cite the entry ID.

Entry states: `PROPOSED` → `FROZEN` → (optional) `SUPERSEDED BY <id>`.

## PL-001 TinyPerson validation split (FROZEN, 2026-08-04)

Frozen by the project owner on 2026-08-04 as proposed. Deterministic
artifacts: `splits/tinyperson_validation_split.json` plus the derived
`splits/tinyperson_train_sub.json` and `splits/tinyperson_val.json`
annotation files, produced by `tools/build_tinyperson_validation_split.py`.

Problem: official TinyPerson ships no validation split; the official test is
A3 storage-only. Method selection, checkpoint selection, and G3 pilot gates
still need a validation surface built only from A1 train material.

Observed A1 structure (corner-task train annotation SHA-256 `8474f124...`,
8,256 crops, 32,430 positives):

- 37 video sources (`bb_V*` 34, `youtube_V*` 2, `chitube_V*` 1) contributing
  6,832 crops. Adjacent-frame crops from one video are near-duplicates, so any
  frame- or crop-level split leaks.
- 93 distinct non-video source images (`bing_*`, `google_*`, `baidu_*`, ...)
  contributing 1,424 crops (multiple crops per source image).

Proposed deterministic rule:

1. Group identity: video group = filename stem without the trailing
   `_I<frame>` suffix (e.g. `bb_V0032`); image group = full filename stem.
   Total 130 groups.
2. Order each group class independently by
   `sha256("tinyperson_<video|image>_<identity>")` ascending.
3. First 20 percent (rounded to nearest, min 1) of each ordered class is
   validation: 7 videos (`bb_V0003, bb_V0008, bb_V0023, bb_V0024, bb_V0033,
   bb_V0036, youtube_V0004`) + 19 image groups.
4. Resulting split: train_sub 6,215 crops / 27,711 positives; validation
   2,041 crops / 4,719 positives. Dense images remain excluded everywhere.

Usage rules if frozen:

- Validation serves only run/checkpoint/method selection and local gates. It
  is never a submission or final-test surface.
- Validation evaluation reconstructs original-image predictions through the
  frozen `tile_to_original` path and scores with the hash-pinned official
  TinyPerson evaluator.
- The frozen train-only scale schedule (`tinyperson_train_p10_p90.json`) was
  fitted on the full official train annotation, which includes validation
  crops. This is accepted because bounds are distribution percentiles, not
  metrics tuned against validation outcomes; no schedule value may be
  re-selected using validation results.
- The split is computed by a checked-in deterministic script; membership lists
  are hashed artifacts. Any change requires a new ledger entry.

Restrictions: A3 test material stays unmounted; no validation result may
modify the frozen method specification; this split does not transfer to
AI-TOD-v2, which keeps its official train/val split.

## PL-002 TinyPerson anchor-assignment preflight shapes (FROZEN, 2026-08-04)

The audit tool was generalized with `--target-height/--target-width`
(defaults `640/640`, byte-identical to the frozen AI-TOD-v2 result except the
clarified `selection_rule` text; parity verified on CUDA, seed 42). TinyPerson
crops resize to two orientations, so two preflights were run, both seed 42,
sample 64:

- Dominant orientation (800x640, 2,426 eligible images): full SA-ALW changes
  358 assignments vs ALW, positives 2,442 -> 2,360 (-3.36 percent), GT
  coverage identical at 622/624 across all four variants.
- Alternate orientation (640x800, 775 eligible images): full SA-ALW changes
  493 assignments, positives 3,532 -> 3,323 (-5.92 percent), GT coverage
  identical at 944/944 across all four variants.

The threshold-dominated, coverage-preserving mechanism pattern observed on
AI-TOD-v2 therefore reproduces on TinyPerson. These are mechanism/validation
evidence only; they do not freeze endpoints or authorize runs.

Artifacts: `diagnostics/tinyperson_anchor_assignment_preflight.json`,
`diagnostics/tinyperson_anchor_assignment_by_scale.csv`,
`diagnostics/tinyperson_anchor_assignment_preflight_alt_orientation.json`,
`diagnostics/tinyperson_anchor_assignment_by_scale_alt_orientation.csv`.

## PL-003 WP03 v12 T4 reproducibility re-adjudication (FROZEN, 2026-08-14)

Frozen by the project owner on 2026-08-14. Four diagnostic-only v12 T4 reports
serve as platform reproducibility evidence for
the two immutable WP03-v8 `alw_canonical` checkpoints (seeds 123 and 2024).
The acceptance retains the original `5e-4` primary-endpoint
tolerance, exact saved-detection replay, strict checkpoint loading, frozen
data/split/trainer/checkpoint hashes, and validation-only/no-final-test scope.

This decision does not reinterpret the original local reload failures, retrain a model,
improve any metric, or promote a Paper A claim. If frozen, it only permits the
two ALW v8 rows to enter the validation-evidence ledger with the already
passing SA-ALW rows; A3 still requires a separately approved two-shard seed-42
pre-run report. Evidence and exact owner decision text are in
`paper_a/experiment_reports/wp03_a2_re_adjudication_proposal_2026-08-14.md`.
