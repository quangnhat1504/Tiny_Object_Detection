# Paper A: SA-ALW Conference Refinement

This directory is the only submission-facing workspace for:

> SA-ALW: Scale-Adaptive Anisotropic Log-Wasserstein Similarity for Tiny Object Detection

The frozen source plan is
`raw/Paper_A_SA_ALW_Conference_Refinement_Plan.md`. The older files under
`paper/` are retained as diagnostic history. Their tables and numerical claims
must not be copied into this tree.

## Current Gate

- Phase 0 / G0: `PASS`
- Phase 1 / G1: `REVISE` (canonical code/tests pass; final diagnostics and
  train-only schedule freeze pending; synthetic mechanism preflight passes)
- Phase 2 / G2: `REVISE` (`NO_GO_CURRENT_DERIVATIVE`; both adapters and official
  evaluator fixtures pass; AI-TOD images and the TinyPerson package are pending)
- Submission evidence rows: `0`
- Final-test performance evaluation: `0`; prohibited until code, config,
  selection rule, fusion rule, and claims are frozen. One AI-TOD-v2 structural
  annotation access is disclosed separately in `test_access_log.md`.

## Start Here

1. `scope_contract.md` - frozen question, scope, outcomes, and evidence rules.
2. `claims_ledger.csv` - every planned or prohibited claim and its evidence bar.
3. `evidence_ledger.csv` - disposition of all pre-refinement result families.
4. `phase_reports/phase_0_report.md` - Gate G0 decision and exact next actions.
5. `tools/validate_phase0.py` - deterministic ledger validation.
6. `phase_reports/phase_2_report.md` - split/evaluator gate and benchmark pivot.
7. `experiment_execution_policy.md` - local-smoke/Kaggle boundary and separate
   pre-run/post-run reporting rule.
8. `experiments/assignment_board.csv` - team work packages awaiting owner and
   Kaggle account assignment.
9. `results/README.md` - machine-readable result schemas and table-generation
   boundary; accepted validation rows and the A4 negative bootstrap decision
   are preserved, while final-test performance access remains zero.
10. `phase_reports/paper_engineering_checkpoint_2026-08-02.md` - compiled
    manuscript status, remaining blockers, and explicit no-training record.
11. `experiments/baseline_fidelity_matrix.md` - required matched baselines,
    conditional closest-prior candidates, and contextual-prior exclusions.
12. `data_access_policy.md` - separate material/performance test locks and the
    one-pass final-test release gate.
13. `experiments/team_run_shards.csv` - atomic dataset/seed shards awaiting
    balanced team/account assignment.
14. `experiments/pilot_decision_protocol.md` - preregistered six-method G3
    viability/component-selection rule.
15. `schedules/endpoint_protocol.md` - reference effect endpoints and the
    bounded seven-run validation sensitivity grid.
16. `phase_reports/resume_checkpoint_2026-08-02.md` - authoritative end-of-day
    pause state and exact next-day resume order.

## Hard Boundary

Paper A contains ALW only as the base formulation and SA-ALW as the proposed
extension. CBL, ICBL, cascaded routing, refinement, P2, SAC, HFP/SDP, SAH-GD,
PC-MR, PC-MOC, and later research branches are outside the paper.
