# Phase 0 Report

Status: `PASS`
Code commit: `113ce299341413bb740c7aeec16319975ac3d14e`
Data/split hash: `NOT_FROZEN_G2_PENDING`

## Completed

- Frozen Paper A to the ALW base and SA-ALW extension only.
- Fixed primary outcome to original-image COCO AP.
- Fixed matched core seeds to `42`, `123`, and `2024`.
- Registered intended assignment/regression/joint placement without claiming the
  current code already implements the canonical method.
- Classified every pre-refinement result family under the four allowed evidence
  statuses.
- Quarantined the old LaTeX draft and inventoried all result-bearing locations.
- Registered enableable, conditional, and forbidden claims with artifact-level
  evidence requirements.

## Evidence Created

- `scope_contract.md` -> research question, method boundary, outcomes, datasets,
  seeds, and change control are frozen.
- `claims_ledger.csv` -> every planned claim has an explicit evidence bar.
- `evidence_ledger.csv` -> no historical artifact is submission evidence.
- `phase_reports/legacy_manuscript_result_inventory.md` -> old numerical claims
  are prevented from flowing into the new manuscript.
- `tools/validate_phase0.py` -> deterministic integrity check for the ledgers.

## Deviations From Plan

- The old LaTeX files remain in place as immutable research history instead of
  being rewritten during Phase 0. The new `paper_a/` tree is the sole
  submission-facing workspace.
- Existing result files are classified by artifact family because many runs
  emit multiple logs/checkpoints for one protocol decision. The ledger patterns
  cover every current SA-ALW/ALW result family and all out-of-scope research.

## Blockers or Risks

- Legacy `alw_full` is not pure ALW; it includes reliability gating and a
  Charbonnier shape wrapper.
- Legacy SA-ALW also includes those wrappers, so its current ablations do not
  isolate the proposed schedules over canonical ALW.
- The legacy schedules are clipped in code but not in the manuscript equations.
- Dataset source grouping, train-only percentile derivation, and original-image
  reconstruction are not yet audited.
- Legacy LaTeX values drift from `runs/test_results.json`; neither source is
  eligible for new tables.

## Claims Enabled or Disabled

- Enabled for investigation, not yet for manuscript assertion: `C001-C015`.
- Explicitly forbidden: `F001-F006`.
- Performance claims enabled now: none.
- Submission evidence rows enabled now: none.

## Gate Decision

`G0 PASS`. Every anticipated claim has an evidence requirement, all legacy
results have an explicit status, and the new submission tree contains no copied
legacy result. This decision does not imply method, dataset, or performance
validity; those require G1-G4.

## Next Exact Actions

1. Build `method_spec.md` from the plan and canonicalize ALW/SA-ALW code paths.
2. Produce `placement_audit.md` from the real RPN/RoI/NMS call graph.
3. Add geometry, schedule, assignment, loss, and numerical-stability tests.
4. Freeze train-derived schedule bounds only after the split/provenance audit.
5. Do not access test data.

