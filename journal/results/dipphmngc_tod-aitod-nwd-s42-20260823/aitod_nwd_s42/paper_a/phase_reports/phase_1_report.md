# Phase 1 Report

Status: `REVISE`
Code commit: `113ce299341413bb740c7aeec16319975ac3d14e + uncommitted canonical changes`
Data/split hash: `NOT_FROZEN_G2_PENDING`

## Completed

- Added a canonical namespace without changing historical metric names.
- Implemented squared log-ratio ALW and clipped target-conditioned SA-ALW.
- Separated pairwise assignment similarity from aligned regression distance.
- Added regression-only placement and fixed IoU-NMS Paper A guards.
- Required explicit schedule values and validation-AP checkpoint selection.
- Preserved positional compatibility of the legacy `build_model` API.
- Passed 18 geometry, schedule-form, protocol, assignment, encode/decode,
  CPU/GPU, and AMP
  tests.
- Passed real Faster R-CNN AMP forward/backward for all three placements.
- Passed strict joint state reload and exact baseline parameter parity.
- Re-ran PCGrad and PC-MR/PC-MOC configuration tests without regression.
- Froze the scale-coordinate calculation to post-torchvision-transform pixels
  at `min_size=640,max_size=800` and matched it against the runtime transform.
- Audited AI-TOD-v2 train-only P10/P90 candidate bounds as
  `6.1968/13.8564 px` across 301,494 valid positive boxes.
- Completed a controlled mechanism preflight: beta-only preserves per-target
  rankings, narrows the HLA quality margin by 20 percent for beta 8 to 10, can
  alter cross-scale GT ownership, and remains absent from regression; position
  emphasis can reverse center-versus-shape candidate ordering.
- Added a hash-recorded log-linear interpolation as the single preregistered
  smooth sensitivity alternative; the reference method remains linear.

## Evidence Created

- `common/metrics/sa_alw_canonical.py` -> canonical implementation.
- `method_spec.md` -> formula, numerical contract, and pseudocode.
- `placement_audit.md` -> verified RPN/RoI/NMS call graph.
- `tests/test_alw_saalw.py` -> 15 passing focused tests.
- `diagnostics/canonical_detector_smoke_seed42.json` -> detector-level smoke,
  parameter parity, and strict reload evidence.
- `diagnostics/saalw_mechanism_preflight.json` -> synthetic code-path
  decomposition, not submission mechanism evidence.
- `diagnostics/aitodv2_anchor_assignment_preflight.json` and by-scale CSV ->
  exact-anchor, two-pass HLA audit on 64 seeded AI-TOD-v2 train images.

## Deviations From Plan

- Legacy names were not corrected in place because doing so would silently
  reinterpret historical checkpoints. Canonical Paper A names are separate.
- Final schedule values are not frozen in Phase 1 because the current split has
  not passed G2; accepting the old constants would violate the train-only rule.

## Blockers or Risks

- Assignment-change causes are not yet quantified on repaired training data.
- AI-TOD-v2 anchor effects are quantified on a bounded train sample, but the D1
  TinyPerson full frozen-train artifact is still absent.
- Center/shape gradient ratios are not yet quantified by scale bin.
- TinyPerson train-only schedule bounds and all beta/position endpoints remain
  unfrozen; AI-TOD-v2 bounds cannot be transferred silently.
- Current training data construction emits tiles; it is not evidence that
  original-image evaluation or source-disjoint splitting is correct.

## Claims Enabled or Disabled

- Enabled: C001-C004 and C006.
- Pending: C005 and C007-C015.
- Forbidden claims remain F001-F006.

## Gate Decision

`G1 REVISE`. Canonical math/code and technical execution are now coherent, but
the gate cannot pass until mechanism diagnostics exist and G2 supplies a
train-only frozen schedule.

## Next Exact Actions

1. Audit source identities and current train/validation/test overlap.
2. Acquire/hash TinyPerson and derive its bounds with the frozen coordinate
   calculation.
3. Pre-register beta/position endpoints without using final-test evidence.
4. Run assignment and gradient diagnostics on that frozen train data.
5. Re-run the same 18 tests and detector smoke with the frozen config.
6. Execute the preregistered six-method G3 pilot on Kaggle only after separate
   reports and user assignment.
