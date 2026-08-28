---
title: "Program B CBL Pivot Decision — 2026-08-14"
type: synthesis
created: 2026-08-14
sources:
  - wiki/syntheses/strategic-research-roadmap-2026-08-14.md
  - paper_a/experiment_reports/wp03_a4_no_go_closeout_2026-08-14.md
  - .runtime/kaggle/pc_micro_fair20_seed42/state.json
  - .runtime/kaggle/ra_tb_cbl_fair20_seed42/state.json
  - .runtime/kaggle/ra_tb_cbl_seed31415/state.json
  - .runtime/kaggle/cr_sc_cbl_multiseed/state.json
tags: [decision, program-b, cbl, pc, evidence-gates]
---

# Program B CBL Pivot Decision — 2026-08-14

## Owner decision

The project owner selects **Program B: CBL/PC** as the active research direction.
This is a pivot away from Paper A performance work, not a reinterpretation of
Paper A validation evidence.

## Boundaries that remain frozen

- Paper A remains `NO-GO — CLOSE PAPER A PERFORMANCE WORK`.
- Paper A final-test performance access remains zero. No Paper A final-test,
  external matrix, WP04–WP07 training, or rescue sweep is authorized.
- The legacy Roboflow derivative remains development-only and is not a Program B
  final-test surface or a source of conference performance claims.
- The historical CBL locked-test budget remains closed and diagnostic-only.

## B0 recovery disposition

B0 recovered and audited the known historical CBL/PC artifacts without launching
new Kaggle training. None is promoted into an accepted performance claim:

| Program | Disposition | Consequence |
|---|---|---|
| PC-MOC / PC-MR seed-42 fair-20 | incomplete/unpaired diagnostic | Retain as a lead only; not evidence. |
| RA-TB seed 42 | `diagnostic_gate_failed` | No follow-up promotion. |
| RA-TB seed 31415 | `diagnostic_gate_failed` | No follow-up promotion. |
| CR-SC-CBL fair-20 matrix | artifacts recovered; reload blocked by frozen-source mismatch | No performance claim or matrix promotion. |

This inventory does not reject iterative-CBL as the Program B baseline. It means
that no historical candidate has earned promotion by the current evidence
contract.

## Authorized next package: B1 protocol and baseline freeze

The next work is documentation and local technical validation only. Before any
new Kaggle training, create a new Program B protocol that freezes:

1. the exact iterative-CBL baseline, including all inference-time refinement
   rules and checkpoint selector;
2. a clean training/validation surface with original-image evaluation and
   source/video-disjoint grouping;
3. a separate untouched external-evaluation policy; no historical locked-test
   reuse;
4. seeds `42`, `123`, and `2024`, fixed data order, augmentation, hardware
   class, 20-epoch budget, and artifact/reload ledger;
5. primary AP and AP75; AP50, AR100, class-aware micro/tiny metrics,
   original-image folds, latency, VRAM, parameter count, and train time; and
6. pre-training technical gates: frozen teacher hash, teacher-free inference,
   one real forward/backward batch, PCGrad parameter-scope assertions, finite
   gradients, checkpoint save/reload, and evaluator fixture.

The first B2 candidates, if B1 is owner-approved, are PC-MR and PC-MOC evaluated
separately against the frozen iterative-CBL baseline. A combined PC-MR+PC-MOC
arm is prohibited until both individual candidates have valid positive
full-schedule evidence.

## Not authorized by this decision

This decision does not authorize a Kaggle push, new training run, final-test
access, retrospective selection from historical scores, or a manuscript claim.
Each later training package needs its own approved pre-run report.
