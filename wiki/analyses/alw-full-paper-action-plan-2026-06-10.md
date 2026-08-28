---
title: ALW Full Paper Action Plan - 2026-06-10
type: analysis
created: 2026-06-10
updated: 2026-07-06
sources: [wiki/sources/alw-main-draft.md, paper/main.tex, runs/test_results.json]
tags: [alw, paper, completed]

## Status: ✅ COMPLETED 2026-07-06
tags: [alw, paper-writing, action-plan, experiments]
---

# ALW Full Paper Action Plan - 2026-06-10

## Diagnosis

The next task is not another architecture experiment. The next task is to turn the current ALW draft into a defensible full paper.

The current draft has a strong core idea and a promising controlled IGWD -> ALW result, but it is still too thin for a full paper because the empirical evidence and ablations are incomplete. The paper should now be treated as a manuscript-building project with experiments serving the claims.

## Main Direction

Lead with the clean ALW story:

1. Prior Gaussian TOD metrics smooth IoU but still use Euclidean shape and isotropic normalization.
2. ALW fixes both with log-ratio shape and per-axis normalization.
3. The clean ALW implementation improves strict localization and recall over IGWD.
4. Public-dataset validation and component ablations establish that the gain is real, not a local artifact.

Do not lead with P2, SAH-GD, or NB14-16. Those are separate follow-up lines unless the ALW evidence collapses.

## Minimum Paper-Completion Experiments

### E1. Byte-Identical Baseline Re-Runs

Run under the same harness, same split, same seed, same schedule, same NMS:

- NWD
- IGWD
- GCD
- ALW (`tod-alw.ipynb`)

Purpose: remove the dagger/footnote weakness from Table III.

Deliverable: one clean table where all methods are directly comparable.

### E2. ALW Component Ablation

Run four variants:

- IGWD-style: isotropic normalization + Euclidean shape
- anisotropic-only: per-axis normalization + Euclidean shape
- log-shape-only: isotropic normalization + log-ratio shape
- full ALW: per-axis normalization + log-ratio shape

Purpose: prove which part of ALW causes the gain.

Deliverable: ablation table with mAP(scale), AP_micro, AP_tiny, AP75, AR@100.

### E3. Public Dataset Validation

Run clean ALW on at least one public TOD dataset:

- AI-TOD / AI-TOD-v2 is the first priority.
- VisDrone is second priority if conversion and compute time allow.

Purpose: make the paper generalizable beyond SOD/TinyPerson-Sea.

Deliverable: public-dataset table against IGWD and preferably GCD.

### E4. Seed Robustness

Run 3 seeds for the most important comparison:

- IGWD vs ALW on SOD/TinyPerson-Sea
- If compute allows: IGWD vs ALW on AI-TOD

Purpose: support mean +/- std and avoid a one-seed paper.

Deliverable: small robustness table.

### E5. Sensitivity / Reproducibility

Low-cost but important:

- beta sweep for ALW assignment similarity: e.g. beta 4, 8, 12
- direct ALW distance loss vs bounded `1 - exp(-beta * ALW)` loss
- optional: ALW-NMS vs IoU-NMS only if the main paper still needs an extra experiment

Purpose: show the method is not a brittle hyperparameter trick.

## Writing Fixes Before Expansion

1. Tighten the claim from "metric" to "distance/similarity" unless triangle inequality is proven.
2. Fix the IGWD scale-invariance discussion so Table I and the text agree.
3. Clarify exactly where ALW is used: assignment and regression in reported runs; NMS is future work unless tested.
4. Move all "to be re-validated" caveats out of the main result table by running E1.
5. Replace placeholder bibliography entries with full metadata.
6. Add a limitations paragraph that is honest but not self-sabotaging after new experiments are done.

## 8-Hour RTX4090 Priority

If only one borrowed 4090 slot is available:

1. First hour: verify environment, data layout, and one-batch smoke test.
2. Run ALW on AI-TOD or AI-TOD-v2 first.
3. If AI-TOD finishes or fails early due to conversion/data issues, run IGWD vs ALW on SOD/TinyPerson-Sea as a controlled rerun.
4. Use remaining time for the component ablation most likely to finish: anisotropic-only and log-shape-only.
5. Do not spend this slot on P2/NB14-16 unless all ALW paper-critical jobs are complete.

## Paper Milestones

### Milestone 1: Evidence Lock

Required:

- E1 complete
- E2 complete
- at least one public dataset result

After this, freeze the claim and tables.

### Milestone 2: Full Draft

Required:

- rewrite experiments section around controlled evidence
- add ablation subsection
- add public benchmark subsection
- revise abstract/conclusion to match the actual validated scope
- fix references and notation

### Milestone 3: Camera-Ready Polish

Required:

- consistent notation across equations/code/tables
- clean figures: metric geometry diagram, method block, result bars
- reproducibility appendix or section
- final grammar and formatting pass

## Decision Rule

If ALW beats IGWD consistently but not GCD, the paper should be framed as:

> ALW is a principled replacement for IGWD that improves strict localization and recall; it is complementary to GCD rather than universally dominant.

If ALW beats IGWD and is competitive with or stronger than GCD on public datasets, the stronger claim is:

> Anisotropic log-ratio geometry is a generally useful box metric for tiny object detection.

The current evidence supports the first claim. The next experiments decide whether the second claim is defensible.
