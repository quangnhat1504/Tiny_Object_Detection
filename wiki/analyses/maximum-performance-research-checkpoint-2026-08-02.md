---
title: Maximum-Performance Research Checkpoint - 2026-08-02
type: analysis
created: 2026-08-02
updated: 2026-08-02
sources:
  - runs/pc_moc_fd_seed2718_gate_result.json
  - runs/pc_mr_rpn_r2_seed2718_gate_result.json
  - runs/pc_mhfd_seed1618_gate_result.json
  - runs/pc_mr_moc_gradient_compatibility_seed42.json
tags: [tiny-object, checkpoint, cbl, distillation, pcgrad, kaggle, paper]
---

# Maximum-Performance Research Checkpoint - 2026-08-02

## Checkpoint Identity

- ID: `PERF-R2-2026-08-02`.
- Purpose: durable handoff from the open performance-research goal to the
  next paper-work task.
- Evidence cutoff: `2026-08-02 19:45 +07:00`.
- Goal status: milestone reached, but the maximum-performance goal is not
  complete because five fair-20 jobs are still running.
- Locked test: consumed `1/1` for the iterative-CBL leader and closed. It must
  not be reopened for RA-TB, PC-MR, PC-MOC, their combination, or tuning.

## Frozen Locked-Test Leader

The current paper-eligible locked-test leader remains seed-42, fair-20
SA-ALW plus trainable iterative CBL:

| Method | AP | AP50 | AP75 | AR100 | mAP(scale) |
|---|---:|---:|---:|---:|---:|
| historical SA-ALW | `0.0975` | `0.3058` | `0.0344` | `0.2509` | `0.6014` |
| iterative-CBL fair20 | **`0.1158`** | **`0.3326`** | **`0.0533`** | **`0.2657`** | **`0.6130`** |

This is the only new candidate in this checkpoint with a legitimate locked-test
claim. See [[Iterative CBL Fair-20 Locked-Test Protocol - 2026-08-01]].

## Promoted Validation Candidates

PC-MR-RPN and PC-MOC-FD passed the same frozen seed-2718, two-epoch,
raw/no-EMA performance gate with independent checkpoint reloads and both
original-image folds positive on AP and AP75:

| Candidate delta vs shared baseline | AP | AP50 | AP75 | AR100 | mAP(scale) |
|---|---:|---:|---:|---:|---:|
| PC-MR-RPN | **`+0.0097`** | **`+0.0327`** | `+0.0041` | `+0.0140` | **`+0.0335`** |
| PC-MOC-FD | `+0.0063` | `+0.0208` | **`+0.0057`** | **`+0.0194`** | `+0.0131` |

PC-MR is stronger on AP, AP50, and scale mAP. PC-MOC is stronger on AP75 and
AR100. These are robust short-schedule validation results, not fair-20 or test
results. See [[PC Micro Fair-20 Protocol - 2026-08-02]].

## Rejected Branch

PC-MHFD changed AP/AP50/AP75/AR100/mAP(scale) by
`+0.0028/+0.0037/+0.0035/+0.0279/-0.0029`, but failed the frozen robustness
gate: even-fold AP was `-0.0012`, odd-fold AP75 was `-0.0046`, and class-aware
micro/tiny AP changed `-0.0146/-0.0506`. It is rejected without a sweep,
fair-20 run, or locked-test access. Its RA-TB combination is therefore closed.

## PC-MR Plus PC-MOC Combination

The no-update, seed-42 full compatibility Gate 0 completed all 200 batches and
passed every predeclared condition:

| Metric | Result | Gate |
|---|---:|---:|
| jointly valid batches | `155/200 = 77.50%` | `>=60%` |
| selected micro GT identity | `843/843 = 100%` | `>=99%` |
| disjoint FPN/RPN support | `100%` | `100%` |
| projected PC-MR cosine | `+0.02264` | `>=0` |
| projected PC-MOC cosine | `+0.00576` | `>=0` |
| retained PC-MR/PC-MOC norm | `99.993% / 99.912%` | both `>=95%` |
| final update vs detector cosine | `0.998715` | `>=0.95` |
| final update norm ratio | `1.001551` | `[0.90, 1.20]` |

Peak allocated VRAM was `8.645 GiB`. Artifact:
`runs/pc_mr_moc_gradient_compatibility_seed42.json`.

The shared-teacher dual-PCGrad implementation and exact algebra/configuration
tests are present, but a real CUDA optimizer/reload smoke and a paired Kaggle
performance run have not yet completed. Therefore no combined detector metric
or paper performance claim exists.

## Active Fair-20 Runs

At the checkpoint cutoff, five private kernels were still `RUNNING`:

| Comparison | Arm | Kernel |
|---|---|---|
| RA-TB | baseline | `thyngluthy/tod-icbl-fair20-s42-r2-20260802` |
| RA-TB | candidate | `hienquang06/tod-ra-tb-cbl-fair20-s42-20260802` |
| PC micro | baseline | `ngquangnht/tod-icbl-pcmicro-fair20-s42-20260802` |
| PC micro | PC-MOC | `hngngnguynvn/tod-pcmoc-fd-fair20-s42-20260802` |
| PC micro | PC-MR | `qnhat1504/tod-pcmr-rpn-fair20-s42-20260802` |

The monitor downloads terminal artifacts, and the auditor waits for exact
metrics plus `best.pt` and `last.pt`. `COMPLETE` alone is not a result.

## Paper Claim Boundary

Allowed now:

- iterative-CBL fair20 is the current frozen locked-test leader;
- PC-MR and PC-MOC each have robust short-schedule validation evidence;
- the combination passes a no-update optimization compatibility gate.

Not allowed now:

- claiming RA-TB, PC-MR, or PC-MOC fair-20 gains before artifact audits;
- claiming the PC-MR plus PC-MOC combination improves detector performance;
- reporting a private kernel URL or `RUNNING`/`COMPLETE` status as a metric;
- any further use of the consumed locked test; or
- state-of-the-art or cross-dataset claims.

## Resume Contract

1. Download and independently audit all five fair-20 terminal artifacts.
2. Complete real CUDA optimizer/reload testing for the dual-PCGrad combination.
3. Only if technical integrity passes, launch a same-source paired validation
   Gate 2 for the combination on free Kaggle accounts.
4. Keep the locked test closed and use fresh validation/external data for all
   promotion decisions.
5. Use `paper/checkpoints/performance_research_2026-08-02.md` as the evidence
   entrypoint for the next paper-work task.

## Related Pages

- [[PC-MOC-FD Gates - 2026-08-02]]
- [[PC Micro Fair-20 Protocol - 2026-08-02]]
- [[PC-MHFD Gates - 2026-08-02]]
- [[RA-TB-CBL Fair-20 Protocol - 2026-08-02]]
- [[Wiki Overview]]
- [[Wiki Log]]
