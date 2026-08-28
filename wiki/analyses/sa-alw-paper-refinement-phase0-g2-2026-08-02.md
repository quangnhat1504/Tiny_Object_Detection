---
title: "SA-ALW Paper Refinement Phase 0-2 - 2026-08-02"
type: synthesis
tags: [sa-alw, paper-a, protocol-audit, original-image-evaluation]
sources: [paper-a-sa-alw-conference-refinement-plan]
last_updated: 2026-08-02
---

## Question

What is the defensible starting state for completing Paper A after auditing the
legacy manuscript, implementation, split, and evaluator?

## Gate State

- G0: `PASS`. Twenty-one claims are registered, forty-seven evidence families
  are classified, and zero historical rows are submission evidence.
- G1: `REVISE`. Canonical math/code, 58 focused tests, and the detector smoke
  pass; train-only schedule bounds and mechanism diagnostics still depend on a
  valid public training split.
- G2: `REVISE`, with `NO_GO_CURRENT_DERIVATIVE`. The reconstruction core passes
  nine tests. AI-TOD-v2 annotations, adapter, evaluator hashes, and the
  perfect-box fixture pass; the image package and TinyPerson path remain
  pending.
- Paper A final-test material access: `1` disclosed AI-TOD-v2 structural audit.
- Paper A final-test performance evaluations: `0`.

## Canonicalization Findings

- Legacy `alw_full` is a reliability-gated Charbonnier wrapper, not pure ALW.
- Legacy ALW-like code uses absolute log ratios rather than the canonical
  squared log-ratio terms.
- The old regression auxiliary builds an all-pairs prediction/GT matrix instead
  of aligned regression pairs.
- The new canonical namespace preserves legacy checkpoint semantics while
  separating pairwise assignment similarity from aligned regression distance.
- Canonical training requires explicit clipped schedule values and validation
  COCO AP checkpoint selection.

## Dataset and Evaluator Findings

- The SOD derivative has 1,570 processed files but only 654 source IDs; train
  contains exactly three variants for each of 458 source images.
- Exact source IDs and image hashes do not cross splits, but sequence groups do:
  30 train-validation, 23 train-test, and 20 validation-test overlaps.
- The legacy test has prior exposure and cannot be reset by repartitioning.
- The previous evaluator treats tiles as independent images. The new
  reconstruction core maps detections to original coordinates, deduplicates by
  class, and builds GT once from original annotations.

## Official Protocol Lock

- TinyPerson official evaluator is pinned to commit
  `bf6b83aa9a149ae15087eed4e9a7283f5cc67603` with ignore/uncertain handling,
  IoU `0.25/0.50/0.75`, and `maxDets=200`.
- AI-TOD-v2 uses the pinned `cocoapi-aitod` commit
  `44a230ae5197cb89bf9e5e62f313cac3ad30c7af`, IoU `0.50:0.05:0.95`, and
  `maxDets=[1,100,1500]`.
- Stock COCO AP and official benchmark results will be generated from the same
  original-image predictions but kept as separately labeled protocols.
- The AI-TOD-v2 wrapper reproduces AP/AP50/AP75/tiny AP of `1.0` on a perfect
  original-image fixture. The same fixture must pass on Kaggle before WP06.
- The TinyPerson binary task-all adapter rejects raw two-class annotations and
  routes `ignore`, `uncertain`, and `iscrowd` boxes away from training
  positives. Its hash-locked official evaluator reproduces AP25/AP50/AP75 of
  `1.0` while IOD ignores a higher-scoring uncertain-region detection. This
  fixture must pass on Kaggle before WP01.
- Schedule fitting is frozen to post-torchvision-transform pixels at
  `min_size=640,max_size=800`. A train-only AI-TOD-v2 audit over 301,494 valid
  positives yields P10/P90 `6.1968/13.8564 px`. These remain D2 candidate
  bounds; TinyPerson bounds and beta/position endpoints are not frozen.
- The AI-TOD-v2 test annotation file was structurally parsed during the package
  audit, including aggregate counts and size summaries. No model prediction or
  test metric was computed. The split is now described as performance-locked,
  not literally unseen; schedule fitting remains train-only.

## Mechanism Preflight and Pilot Decision

- For a fixed GT, beta-only is a monotonic transform and produces zero ranking
  changes in the controlled probes.
- With HLA quality ratio 0.60, beta 8 to 10 narrows the admissible distance
  margin by 20 percent and changes a controlled positive count from four to
  three. Cross-scale ownership can also change.
- Position emphasis can reverse center-error versus shape-error ordering, while
  beta-only regression is exactly canonical ALW regression.
- These are synthetic code-path facts, not C007/C008 submission evidence. The
  public-train assignment-rate and gradient artifacts remain required.
- G3 is now preregistered as a six-method seed-42 pilot: standard, predecessor,
  ALW, beta-only, position-only, and full SA-ALW. The fixed AP/tie rule selects
  or revises the central method before any three-seed matrix.
- On 64 seeded AI-TOD-v2 train images, the exact 306,900-anchor two-pass HLA
  audit finds full SA-ALW changes 593 assignments and reduces positives
  `6899 -> 6643`, but all variants retain identical `1816/1818` GT coverage.
  Only three changes are ownership flips. This is mechanism preflight, not AP.
- Reference schedule effects are now preregistered without performance tuning:
  beta `8 -> 10`, position weight `1 -> 1.5`, and dataset-specific train P10/P90
  scale bounds. Sensitivity is capped at seven one-axis runs, including exactly
  one implemented/tested log-linear alternative; no Cartesian grid is allowed.

## Paper Engineering Update

- Machine-readable result ledgers start with zero accepted rows. Accepted rows
  require a matching run manifest, registered seed, `coco_ap` checkpoint rule,
  artifact audit, and in-range fractional metrics.
- The headline-table generator emits only complete matched groups with seeds
  `42/123/2024`; incomplete groups remain visible only in audit summaries.
- A primary-source audit of NWD, RFLA, SimD, SAFit, GCD, and IGWD removes broad
  priority claims. SimD already has per-axis/train-derived normalization,
  SAFit already conditions a fitness/loss on target area, and GCD already uses
  scale-invariant Gaussian geometry in assignment and regression.
- The extended SWL/MMPW/DILA audit finds that DILA's BGSM center term matches
  ALW's per-axis squared-width/height sums up to a factor of two. C014 is
  therefore enabled only for the exact center-plus-log-shape formulation and
  separately placed schedules, never for center normalization alone.
- IGWD is now verified as Hu, Chen, and Tang, IEEE TMM 2026, DOI
  `10.1109/TMM.2026.3675527`; the anonymous legacy citation is retired.
- The evidence-gated manuscript compiles to a five-page internal draft and a
  one-page supplement. Performance text and tables remain explicitly pending.
- Internal PDF preflight finds no checked identity/path token and all fonts are
  embedded, but the generic A4 article class, one Type-3 font, timestamps, and
  incomplete anonymous code package keep G6 pending.
- Paper A still has zero accepted performance rows and zero final-test
  performance evaluations.

## Decision

Do not run the conference pilot on the current derivative. Acquire and hash the
remaining official images and TinyPerson package, audit them against the
completed adapter/evaluator contracts, then derive the SA-ALW schedule solely
from the frozen training split.
The performance-research Kaggle jobs may continue, but their CBL-family results
remain outside [[Paper A SA-ALW Conference Refinement Plan]].

## Execution Boundary

- Continue paper engineering, dataset/evaluator preparation, manuscript work,
  deterministic tests, and bounded local smokes without launching training.
- Run every Paper A pilot/full experiment on Kaggle.
- Report each Kaggle work package separately before push so the user can assign
  a team member and account, then report downloaded/audited artifacts separately
  after completion.
- Use atomic dataset/seed rows in `paper_a/experiments/team_run_shards.csv` to
  preserve paired comparisons. Balance members by predicted GPU-hours from the
  same smoke, with a target maximum load of team mean plus 15 percent.
- An unassigned shard cannot become `READY_FOR_PUSH`; ordinary shards receive
  no final-test mount.

## Artifacts

- `paper_a/scope_contract.md`
- `paper_a/method_spec.md`
- `paper_a/splits/split_audit.json`
- `paper_a/evaluation/official_evaluator_lock.json`
- `paper_a/phase_reports/phase_2_report.md`
- `paper_a/phase_reports/related_work_audit.md`
- `paper_a/results/README.md`
- `paper_a/manuscript/main.pdf`
- `paper_a/phase_reports/paper_engineering_checkpoint_2026-08-02.md`
- `paper_a/data_access_policy.md`
- `paper_a/experiments/team_run_shards.csv`
- `paper_a/phase_reports/anonymity_preflight_2026-08-02.md`
- `paper_a/diagnostics/saalw_mechanism_preflight.json`
- `paper_a/experiments/pilot_decision_protocol.md`
- `paper_a/phase_reports/aitodv2_anchor_assignment_preflight_2026-08-02.md`
- `paper_a/schedules/endpoint_protocol.md`
