---
title: "Strategic Research Roadmap — 2026-08-14"
type: synthesis
tags: [strategy, paper-a, cbl, pc, evidence-gates]
sources:
  - wiki/overview.md
  - paper_a/experiment_reports/wp02_matched_baselines_postrun.md
  - paper_a/experiment_reports/wp03_matched_proposed_postrun.md
  - paper_a/experiment_reports/wp03_v9_t4_audit_postrun.md
  - paper/checkpoints/performance_research_2026-08-02.md
last_updated: 2026-08-14
---

# Strategic Research Roadmap — 2026-08-14

## 1. Executive Decision

**Outcome update (2026-08-14): `PAPER_A NO-GO; PROGRAM B B0 OPEN`.** A1/A2
passed, the two A3 seed-42 artifacts passed package audit and independent
reload, and A4 completed the matched six-method/three-seed matrix. SA-ALW minus
canonical ALW has mean AP `-0.001286` with paired original-image 95% CI
`[-0.002939,+0.001277]`. This fails the frozen positive-effect criterion.
WP04-WP07 and all Paper A external/final-test performance work are closed; the
next allowed action is the read-only B0 CBL/PC recovery audit.

The project has two independent research programs and must stop treating them
as one queue:

1. **Paper A / SA-ALW** is a narrow conference-evidence program. Its only
   purpose is to establish whether a canonical, scale-adaptive ALW formulation
   has a defensible novelty and performance claim on a clean protocol.
2. **CBL/PC** is the performance-localization program. It studies iterative
   confidence-based localization (CBL), proposal-level micro rescue (PC-MR),
   and micro-object feature distillation (PC-MOC). It is not evidence for
   Paper A and must receive its own research question, validation surface, and
   future manuscript decision.

The immediate objective is **not** to maximize the number of Kaggle jobs. It
is to reach one irreversible decision: Paper A is either sufficiently clean
and promising to complete, or it is recorded as `NO-GO` and compute moves to
CBL/PC. No WP04/WP05 sensitivity or ablation job is allowed before that
decision.

## 2. Facts That Govern the Plan

### 2.1 Paper A evidence is incomplete, not merely delayed

- WP02 is accepted validation evidence: 12 artifacts, four baseline methods,
  three seeds, artifact audits, and independent reloads all pass. RFLA leads
  this matrix at `0.15857 ± 0.00138` selector AP; standard is
  `0.15557 ± 0.00264`.
- WP03 v8 has four downloaded, package-audited proposed-method artifacts.
  SA-ALW reloads pass locally; the two ALW reloads exceed the local
  official-primary tolerance. Four independent v12 replicas on fresh 2×T4
  workers then reproduce both ALW checkpoints within `5e-4`, including exact
  saved-detection replay. This resolves a technical reproducibility concern,
  but the authorized v12 scope is diagnostic-only; it does not itself write a
  ledger row or promote a scientific claim.
- The prospective SA-ALW gain over ALW is small. It must therefore survive a
  strictly matched three-seed analysis and original-image paired bootstrap,
  not be inferred from a pilot ranking.
- The WP01 seed-42 proposed-method outputs and the WP02 matrix do not use the
  same trainer hash. The observed standard difference between those trainers
  demonstrates that this is material. Seed 42 cannot complete a matched
  WP03/Paper-A matrix until ALW and SA-ALW are rerun with the WP02-frozen
  trainer.

### 2.2 A canonicality contradiction is a first-order risk

The frozen Paper A plan forbids a reliability/Charbonnier-wrapped ALW from
serving as the pure ALW denominator. Yet the WP03 pre-run prose describes
`alw_canonical` with reliability and Charbonnier, while the current Paper A
trainer documentation says that `alw_canonical` has no reliability wrapping.
This is a documentation-versus-artifact question, not a basis for guessing.
The run configuration and code hash are authoritative. If the executed method
is wrapped or its actual placement differs from the canonical specification,
the current WP03 results cannot support the intended Paper A comparison.

### 2.3 CBL/PC has stronger performance leads but stale cloud state

The frozen legacy checkpoint records an iterative-CBL locked-test leader, but
that locked-test budget is consumed and closed. PC-MR and PC-MOC each passed a
short, seed-2718 validation gate; their joint PCGrad compatibility gate passed
without proving detector performance. The stored cloud states for the seed-42
PC-MR/PC-MOC/RA-TB and CR-SC-CBL fair-20 runs were last checked on 02–03
August. They are leads to recover, not current results. PC-MHFD and its
RA-TB combination remain rejected and must not be revived through a sweep.

## 3. Program A — Final SA-ALW Decision Gate

### A0. Freeze the boundary now

Do not launch WP04, WP05, a Paper A external matrix, a final-test package, or
any parameter sweep. Preserve current artifacts, source hashes, validator
versions, and downloaded logs. The only allowed work until A3 is deterministic
audit, report writing, and the narrowly defined seed-42 fill run if A1 and A2
pass.

### A1. Canonicality and matching audit — no GPU required

Create a `WP03 canonicality audit` that compares the following four sources
line by line:

- `paper_a/method_spec.md` and `paper_a/placement_audit.md`;
- each WP03 v8 `config.json`, trainer SHA-256, and generated kernel source;
- the WP03 pre-run/post-run reports; and
- the v12 audit manifest, which proves the exact checkpoints and evaluation
  payloads it replayed.

The audit must answer, with an artifact path and hash for every answer:

| Question | Required answer |
|---|---|
| ALW denominator | Is it the unwrapped canonical distance/similarity specified for Paper A? |
| SA-ALW delta | Are beta and position schedules the only intended differences from ALW? |
| Placement | Which of assignment, regression, and NMS actually consumes each schedule? |
| Beta claim | Is beta absent from regression, as the method specification requires? |
| Matching | Do data version, split, transform, augmentation, checkpoint rule, epochs, seed, and frozen trainer match the comparison contract? |
| Reporting | Are labels such as `pure`, `canonical`, and `validation-only` truthful? |

**A1 pass** requires a fully traceable canonical implementation and a clean
within-seed ALW versus SA-ALW comparison. A documentation error may be
corrected and disclosed if the executed artifacts are canonical. If the
artifacts are wrapped or otherwise noncanonical, A1 fails. Record `NO-GO
CURRENT_PAPER_A_FORMULATION`; do not relabel the old outputs as canonical.

### A2. Re-adjudicate reproducibility only after A1 passes

If A1 passes, request a single project-owner decision to create a superseding
protocol entry. That entry must explicitly state that the four downloaded
v12 T4 reports are accepted as the platform reproducibility evidence for the
two immutable WP03-v8 ALW checkpoints. It must retain, rather than relax:

- the original `5e-4` primary-metric tolerance;
- frozen data/split/trainer/checkpoint hashes;
- exact saved-detection replay and strict checkpoint loading; and
- the validation-only, no-final-test boundary.

The entry must not claim that v12 retrained a method or improved a metric.
It merely determines whether the two existing ALW artifacts satisfy the
reproducibility contract on the requested T4 environment. Only after the
entry is approved may the two ALW rows and the already-passing SA-ALW rows
enter the validation ledger. If the owner does not approve this amendment,
the result remains `NO_PROMOTION`, and Paper A proceeds directly to its
no-go closeout.

### A3. Fill the missing matched seed correctly

If A1 and A2 pass, prepare **exactly two** new validation-only Kaggle shards:
`alw_canonical, seed=42` and `sa_alw_full, seed=42`. They must use the WP02
trainer hash, WP02 data version, frozen TinyPerson split, same data order,
same eight-epoch budget, same selector, same original-image evaluator, and
fixed IoU-NMS. They are not a retry of failed work and do not change a method;
their sole purpose is to complete the previously non-homogeneous 42/123/2024
matrix.

Before any push, provide the standard pre-run report: owner/account,
accelerator request, data/code/config hashes, expected outputs, and explicit
`test_access=none`. After terminal status, download the small evidence first,
then the checkpoints deliberately. Accept each shard only after package audit,
strict reload, primary endpoint check, and prediction/evaluator checks pass.
`COMPLETE` alone is never a result.

### A4. Make the Paper A decision from the complete matrix

Build one machine-readable table over standard, RFLA, NWD, IGWD, canonical
ALW, and full SA-ALW at seeds 42/123/2024. Preserve any closest-prior baseline
that cannot be faithfully integrated as an explicit exclusion, not an
approximation. Report per-seed AP/AP50/AP75, mean±std, original-image paired
bootstrap confidence intervals for SA-ALW minus ALW, and per-scale counts.

Use the following decision rule:

| Outcome | Required evidence | Next action |
|---|---|---|
| `NO-GO` | A1 failure; A2 is not authorized; or the SA-ALW-vs-ALW AP interval does not support a positive effect | Close Paper A performance work; preserve negative evidence; pivot to Program B. |
| `CONDITIONAL` | SA-ALW is positive over canonical ALW but does not meet or closely approach strong baselines, or AP75 is mixed | Do one preregistered external diagnostic pilot only; do not run WP04/WP05. |
| `GO_FOR_EXTERNAL` | Canonical matrix passes, at least two of three paired seeds favor SA-ALW, bootstrap AP support is positive, and AP75 has no material regression | Run the external pilot; only then decide whether full D2 and ablations are justified. |

The rule deliberately does not say “publish” after D1. A direct-predecessor
gain that loses badly to RFLA or standard is a weak conference story. The
external pilot is a falsification step, not a rescue sweep.

## 4. Paper A No-Go Closeout

When any `NO-GO` trigger fires, create a concise closeout report containing
the failed criterion, artifact references, affected claims, and an explicit
statement that the Paper A final-test counter remains unchanged. Update the
claims/evidence ledgers, assignment board, result summary, wiki overview, and
scratchpad. Freeze `paper_a/` from new training work; retain its evaluator,
dataset, and protocol infrastructure for later projects only after a separate
scope decision. Do not delete negative results or silently recycle them into a
new paper.

## 5. Program B — CBL/PC Performance and Research Pivot

### B0. Recover historical fair-20 evidence before proposing a new method

The first CBL/PC task is a read-only recovery audit. Inspect credential locks;
then, one account at a time, poll and retrieve the known runs: the PC-MR/PC-MOC
shared-baseline matrix, RA-TB seed-42 pair, RA-TB seed-31415 pair, and
CR-SC-CBL fair-20 run. For every terminal run, preserve its API status, log,
manifest, metrics, `best.pt`, `last.pt`, configuration, teacher hash, epoch
records, and predictions where available.

Classify each as `artifact_ready`, `invalid_artifact`, `incomplete`,
`failure`, or `partial_or_unknown`. A result is usable only if it demonstrates
the requested full schedule, same-source comparison, intended CUDA hardware,
checkpoint identity, and an independent reload. Missing outputs are a recovery
failure, not evidence that a candidate won or lost. This phase must not push a
new kernel, change credentials concurrently, or access the historical locked
test.

### B1. Establish a fresh CBL/PC protocol

The historical maritime locked test is permanently diagnostic. Create a new
Program-B protocol before new training that specifies:

- a training and validation dataset with original-image evaluation and
  source/video-disjoint grouping where relevant;
- a separate, untouched external evaluation protocol; no legacy locked-test
  reuse;
- a frozen iterative-CBL baseline that includes all existing inference rules;
- exact seed set `42/123/2024`, data order, augmentation, checkpoint selector,
  hardware class, and 20-epoch budget;
- primary AP and AP75, plus AP50, AR100, class-aware micro/tiny metrics,
  original-image folds, latency, VRAM, parameter count, and train time; and
- the artifact/reload/claim ledger used by every arm.

The research question should be narrow: **can teacher-bounded proposal rescue
and micro-object feature distillation improve strict micro-object localization
beyond a frozen iterative-CBL baseline without inference-time teacher cost?**
Do not call CBL or distributional box regression novel by itself. Novelty,
if earned, must lie in the validated PC mechanism, its disjoint gradient
scopes, and robust external evidence.

### B2. Candidate order and technical gates

Use an evidence hierarchy rather than reopening every historical branch:

1. **Iterative-CBL baseline** is the single reference implementation.
2. **PC-MR** and **PC-MOC** are first candidates because both passed the
   short robust gate. Recheck their recovered fair-20 artifacts first.
3. **PC-MR + PC-MOC** is eligible only if both individual arms have valid,
   positive full-schedule evidence. The existing compatibility audit is a
   prerequisite, not a performance result.
4. **RA-TB** is retained only if recovered artifacts show a robust gain over
   its matched baseline. **PC-MHFD**, RA-TB+PC-MHFD, and rejected mechanisms
   remain closed.

Before a full run, run a bounded local technical gate: import/compile, frozen
teacher SHA, teacher-free inference graph, one real forward/backward batch,
PCGrad parameter-scope assertions, finite gradients, checkpoint save/reload,
and evaluator fixture. A local performance score is never promotion evidence.

### B3. Validation matrix and promotion rules

Run the smallest matrix that can disprove the hypothesis. First obtain an
audited 20-epoch paired comparison for baseline, PC-MR, and PC-MOC. Promote the
joint model only when each individual method has valid artifacts and neither
causes unacceptable AP75/micro-object regression. Then run baseline versus
PC-MR+PC-MOC over three matched seeds.

An arm passes the Program-B validation gate only when all of the following
hold: artifact contract passes; mean AP is positive versus the exact baseline;
paired bootstrap supports the primary delta; AP75 and class-aware micro/tiny
diagnostics do not show a material contradiction; at least two source/image
folds do not reverse the primary conclusion; and efficiency overhead is
measured. A single high seed, a completion status, or an improvement only in
AR cannot promote the method.

### B4. External validation and manuscript feasibility

Only a validation winner receives an external public-benchmark pilot. Freeze
the integration, teacher policy, threshold, checkpoint rule, and evaluator
before that pilot. If the direction fails externally, frame the work as
domain-limited or stop; do not tune on the external test. If it passes, run a
three-seed external comparison and only then start a CBL/PC manuscript.

Before drafting claims, perform a new primary-source related-work audit
covering C-BBL/UGS, localization distillation, ScaleKD/FGD/CrossKD, and
small-object feature distillation. The manuscript may claim only what the
evidence matrix isolates. Required final additions are component ablation,
teacher-free inference proof, cost table, qualitative failures, and all
negative/rejected branches in the supplement.

## 6. Work Packages, Decision Owners, and Sequence

| Package | Output | Compute | Approval needed before next package |
|---|---|---:|---|
| A1 | WP03 canonicality audit | none | owner accepts or rejects canonical status |
| A2 | T4 re-adjudication amendment | none | owner approves superseding ledger rule |
| A3 | two exact WP02-hash seed-42 shards | 2 T4 jobs | pre-run report with owner/account |
| A4 | D1 matrix and Paper A GO/NO-GO report | none | owner selects external pilot or pivot |
| B0 | CBL/PC artifact recovery report | read-only | owner accepts recovered evidence inventory |
| B1 | Program-B protocol and baseline freeze | local technical only | owner approves data/evaluation boundary |
| B2 | individual PC fair-20 validation | Kaggle | per-package pre-run report |
| B3 | combined PC-MR+PC-MOC three-seed matrix | Kaggle | only after B2 pass |
| B4 | external validation and manuscript decision | Kaggle | owner assigns new external package |

All Kaggle packages remain serialized around credential rotation. Each requires
a specific account, an accessible private dataset, a mount smoke, requested
accelerator verification, a native model-init smoke, a run manifest, and a
post-run downloaded-artifact audit. Account capacity never authorizes a new
hypothesis.

## 7. Risks and Responses

| Risk | Response |
|---|---|
| Documentation and executed artifacts disagree | Treat configuration/hash evidence as authoritative; disclose and correct prose, never rewrite history. |
| Kaggle state is stale or output retrieval fails | Preserve API response, retry bounded small evidence retrieval, and classify unresolved output as unknown. |
| Small gains are noise | Require paired original-image intervals, three seeds, folds, and a fixed baseline before promotion. |
| Historical locked-test result invites retuning | Keep the counter closed; use a new validation/external surface only. |
| CBL/PC scope expands into many variants | Candidate order and explicit rejection list prevent further branch proliferation. |
| A closest prior cannot be faithfully implemented | Document exclusion and narrow the claim; never present an approximation as official. |

## 8. Immediate Next Action

B0 recovery is complete and its historical artifacts have no promoted
performance candidate. B1's tiled scale and manifest-backed original-image
evaluator gates are complete, and the owner-authorized B2 baseline pre-run was
prepared. Its required Kaggle mount/model-init smoke failed before model
initialization because Kaggle assigned an incompatible Tesla P100 (`sm_60`), not
the frozen T4 class. Therefore `b2_baseline_s42` and every candidate remain
blocked; the next action is to obtain a T4-capable assignment or receive an
explicit owner-approved hardware-contract revision. No B2 training, metric,
locked-test, or external-test evidence exists.

## Connections

- [[Paper A SA-ALW Conference Refinement Plan]] — source protocol for Program A.
- [[WP03 A2 T4 Re-adjudication Amendment — 2026-08-14|WP03 v9 independent T4 audit]] — diagnostic reproducibility evidence.
- [[Maximum-Performance Research Checkpoint - 2026-08-02]] — CBL/PC evidence hierarchy.
- [[Iterative CBL Fair-20 Locked-Test Protocol - 2026-08-01]] — closed historical test boundary.
