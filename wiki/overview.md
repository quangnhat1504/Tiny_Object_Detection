---
title: Wiki Overview
type: overview
created: 2026-05-09
updated: 2026-08-14
sources: []
tags: [system]
---

## Wiki Overview

This wiki tracks source papers, local experiments, and working conclusions for tiny object detection on SOD-TinyPeopleInSea.

## Current State

- Raw sources live in `raw/` and are treated as read-only.
- Supporting entity, concept, topic, and analysis pages link back through [[Wiki Index]].

### Strategic roadmap (2026-08-14)

- [[Strategic Research Roadmap — 2026-08-14]] freezes a final Paper A
  canonicality/re-adjudication decision gate before any WP04/WP05 or test
  work. A failure to establish canonical, matched novelty is a formal Paper A
  `NO-GO`, not an invitation to sweep more schedules.
- The owner has selected the separate CBL/PC program as the active research
  direction in [[Program B CBL Pivot Decision — 2026-08-14]]. Historical
  fair-20 recovery remains diagnostic-only; the historical CBL locked test is
  closed. [[Program B B1 CBL/PC Protocol Freeze - 2026-08-14]] preserves a
  content-addressed source bundle, group-disjoint original-image split, and
  evaluator bridge. The initial un-tiled scale failure is corrected by
  [[Program B B1 Tiled Scale Revision Audit - 2026-08-14]], which restores the
  `512/64` tile geometry and preserves original IDs; its frozen scale contract
  passes. The manifest-backed original-image evaluator path and focused local
  tests now pass. B2's refreshed private code snapshot is available, but its
  required Kaggle mount smoke received an incompatible Tesla P100 (`sm_60`)
  instead of the frozen T4 and stopped before model initialization. B2 baseline
  training is `BLOCKED_HARDWARE_CONTRACT`; no candidate or test access occurred.
- A1/A2 passed and both owner-approved A3 seed-42 shards on `pptlyn11`
  completed. Canonical ALW and full SA-ALW artifacts both pass package audit
  and independent CUDA reload within the frozen primary tolerance.
- [[WP03 A4 Paper A NO-GO — 2026-08-14]] records the irreversible outcome.
  Across matched seeds `42/123/2024`, SA-ALW minus ALW has mean AP
  `-0.001286` and paired original-image 95% CI
  `[-0.002939,+0.001277]`. The primary criterion fails: Paper A performance
  work, WP04-WP07, external training, and final-test performance access are
  closed. Negative evidence remains in the canonical ledgers; the final-test
  performance-access count remains zero. The next allowed action is Program B
  B0 read-only recovery, not another Paper A run.

### Paper A SA-ALW Refinement (started 2026-08-02)

- [[Paper A SA-ALW Conference Refinement Plan]] now governs the only
  submission-facing workspace, `paper_a/`. CBL and all later research branches
  are outside its scope.
- G0 passes. G1 remains `REVISE` until train-only schedules and mechanism
  diagnostics are frozen. G2 remains `REVISE`; the current SOD derivative is
  `NO_GO_CURRENT_DERIVATIVE` because video sequences overlap every legacy split
  pair and its test is reused.
- Canonical ALW/SA-ALW code is separated from legacy checkpoint names. The
  method, tile reconstruction, both public adapter/evaluator fixtures, and
  result-pipeline test suites pass, together with the detector AMP/reload smoke.
- Official TinyPerson and AI-TOD-v2 evaluator sources are commit/hash pinned.
  The official TinyPerson package is acquired, hashed, and fixture-verified at
  `D:\paper_a_data\TinyPerson\tiny_set`; corner format is confirmed as
  `[x1,y1,x2,y2]`. AI-TOD-v2 annotations/manifest and both adapter/evaluator
  implementations are complete; AI-TOD-v2 images remain the last G2
  acquisition action.
  TinyPerson still lacks an official validation split, so the split policy
  must be frozen in the protocol ledger first.
  See [[TinyPerson Acquisition and Real-Data Fixture - 2026-08-04]].
- The internal Paper A draft and supplement compile from canonical method and
  source-audited related work. Result ledgers now preserve 18 accepted
  validation-only matrix rows and the negative A4 bootstrap decision; no
  final-test performance row exists.
- Schedule coordinates are frozen to torchvision detector-input pixels. The
  AI-TOD-v2 train-only audit gives P10/P90 `6.1968/13.8564 px`; the
  TinyPerson train-only audit gives candidate P10/P90 `7.4328/44.8468 px`
  (much wider distribution), cross-checked independently. Beta/position
  endpoints remain unfrozen. TinyPerson anchor preflights on both crop
  orientations reproduce the threshold-dominated, coverage-preserving
  mechanism pattern. `paper_a/protocol_ledger.md` entry PL-001 freezes a
  deterministic video/source-disjoint 20% TinyPerson validation split (user
  approved; hashed split artifacts checked in).
  See [[TinyPerson G1 Bounds, Mechanism Diagnostics, and Validation Proposal - 2026-08-04]].
- The WP01 TinyPerson G3 pilot is on Kaggle: both private datasets
  (`tinyperson-wp01-a1` data + `paper-a-code-wp01` code/weights) are
  uploaded and replicated per account, a CPU mount-layout smoke kernel
  passed every pinned-hash and import check, and all six T4 training
  kernels (`wp01-pilot-<method>-s42`, seed 42, 8 epochs) run in parallel
  across five pool accounts after a user-approved fan-out (`ngquangnht` =
  standard + igwd; `amongus1504` = alw-canonical; `qnhat1504` =
  sa-alw-beta-only; `thyngluthy` = sa-alw-pos-only; `hienquang06` =
  sa-alw-full). The
  canonical pilot trainer (`paper_a/tools/train_tinyperson_pilot.py`)
  smoke-passed all six frozen methods locally and the perfect-detection
  fixture returns AP = 1.0 on both labeled families. All six pilot runs
  finished and passed every artifact-audit and off-kernel reload check
  (selector AP, seed 42, validation-only: standard `0.16135`, sa_alw_full
  `0.15635`, alw_canonical `0.15461`, sa_alw_beta_only `0.15337`,
  sa_alw_pos_only `0.15315`, igwd `0.14884`). The frozen selection rule
  returned **GO**: full SA-ALW exceeds both references and is the
  formulation for the matched three-seed matrix, with the thin ALW margin,
  the baseline-leading standard result, and the epoch-6/7 convergence
  pattern all disclosed.
  See [[TinyPerson Pilot Harness and Pre-Run Freeze - 2026-08-04]].
- WP02's matched 4-method x 3-seed baseline matrix is complete: all twelve
  downloaded artifacts passed artifact audit and independent CUDA reload of
  their primary official endpoints, and each now has a matching manifest plus
  accepted validation-evidence ledger row. RFLA leads the validation-only
  baseline mean at `0.15857 +/- 0.00138`, followed by standard at
  `0.15557 +/- 0.00264`.
- WP03 is now complete at canonical ALW/full SA-ALW seeds `42/123/2024`.
  PL-003 retains the historical local failures and the v12 T4 reproducibility
  evidence; both A3 seed-42 packages independently pass the frozen contract.
  A4's six-method matrix and 2,000-replicate paired bootstrap trigger
  `PAPER_A PERFORMANCE NO-GO`: mean SA-ALW-minus-ALW AP is `-0.001286` with
  95% CI `[-0.002939,+0.001277]`. See
  `paper_a/experiment_reports/wp03_a4_no_go_closeout_2026-08-14.md`.
- Formula audit verifies IGWD as IEEE TMM 2026 and covers SimD, SAFit, GCD,
  SWL, MMPW, and DILA. DILA's BGSM predates ALW's per-axis center denominator
  up to a factor of two, so only the exact center-plus-log-shape formulation
  and schedule placement remain defensible; no broad first/metric/SOTA claim
  is permitted.
- Controlled mechanism preflight proves beta-only cannot change per-target
  ranking or regression, but can change HLA threshold eligibility and
  cross-scale ownership; position emphasis directly changes center/shape
  ordering. The Kaggle G3 pilot is therefore frozen at six methods, including
  beta-only and position-only, before the multi-seed matrix.
- The 64-image AI-TOD-v2 train-anchor audit confirms a threshold-dominated
  effect: full SA-ALW changes 593 assignments and reduces positive anchors by
  3.71 percent while preserving identical GT coverage. No coverage-preserving
  branch is added before performance evidence.
- The reference linear endpoint effects and a seven-run one-axis sensitivity
  budget are frozen before validation. A log-linear scale interpolation is the
  only smooth alternative; it has a tested/config-hashed code path but no result.
- Paper engineering and bounded smoke tests run locally. All Paper A training
  experiments run on Kaggle only after a separate pre-run report and team/account
  assignment; each returned run gets a separate artifact-audit report.
- Team allocation now uses atomic dataset/seed shards and balances predicted
  GPU-hours, preserving each matched comparison on one account. AI-TOD-v2 has
  one disclosed structural test-annotation access but zero performance
  evaluations; TinyPerson remains unopened.
- See [[SA-ALW Paper Refinement Phase 0-2 - 2026-08-02]].

### Maximum-Performance Research Checkpoint (2026-08-02)

- Checkpoint `PERF-R2-2026-08-02` freezes the current goal state for the next
  paper-work task. See [[Maximum-Performance Research Checkpoint - 2026-08-02]].
- Iterative-CBL fair20 remains the only new locked-test leader at
  AP/AP50/AP75/AR100=`0.1158/0.3326/0.0533/0.2657`; the test budget is closed.
- PC-MR-RPN and PC-MOC-FD independently passed the same seed-2718 robust
  validation gate and are running under one seed-42 fair-20 baseline.
- Their full 200-batch compatibility audit passed with `77.5%` joint validity,
  disjoint FPN/RPN support, final-update cosine `0.998715`, and norm ratio
  `1.001551`. This is mechanism evidence, not combined performance evidence.
- PC-MHFD failed fold and class-aware micro/tiny robustness gates and is
  rejected. Five fair-20 kernels remain `RUNNING` under artifact-first audit.

### SAH-GD Phase (completed — 2026-06-02)
- Ingested papers: [[ALW]], [[NWD]], [[GCD]], [[IGWD Paper]], [[RFLA]].
- Tested NWD, GCD, IGWD, ALW independently — GCD best overall, NWD best for micro.
- Built SAH-GD (hybrid blending), HARD_SWITCH won (mAP=0.5770).
- P2 (stride-4) added, AP_micro +29%.
- Plateaud — blending variants within 1%, AP@75 stuck at 0.02-0.045. See [[SAH-GD Advancement - 2026-06-02]].

### Metric Chain Phase (Phase 2 — completed 2026-07-04)
- Full metric chain: IoU → NWD → IGWD → ALW → SA-ALW (7+ configs) under byte-identical harness.
- [[Phase 2 Metric Chain Ablation - 2026-07-01]]: implementation plan, hyperparameters.
- [[Phase 2-4 Results Summary]]: all validation + test results.
- [[Test-Set Evaluation — Phase 2 Metrics]]: locked test set evaluation (65 images).

### Phase 3-4 (completed 2026-07-04)
- [[Phase 2-4 Results Summary]]: SAALWAssigner (Phase 3) and Cascaded WBF (Phase 4).
- SAALWAssigner (threshold-based): test mAP(scale)=0.5357 vs HLA=0.6005 — HLA dominates.
- Cascaded WBF: best config (0.55/0.20) achieves mAP(scale)=0.6017 but large-object collapse.
- [[WBF Improvement — Root Cause Analysis & Plan]]: root cause — partial-box averaging across tiles.

### Legacy Paper Writing Phase (diagnostic history, 2026-07-06)
- **ALW paper**: `paper/main.tex` + `paper/experiments.tex` - legacy draft only.
- **SA-ALW paper**: `paper/saalw_main.tex` + `paper/saalw_experiments.tex` - legacy draft only.
- Both papers use only COCO metrics (AP, AP50, AP75, AP_S, AP_M, AP_L, AR100) — custom scale-aware metrics removed.
- Cascade section removed from SA-ALW paper (not breakthrough yet, reserved for future).
- Historical numbers were checked against local artifacts, but the underlying
  tile-level/reused-test protocol is invalid for submission claims.
- Papers pushed to GitHub: commit a9ceeb5.
- See [[Paper Rewrite Summary - 2026-07-06]].

### Decoupled Regression Phase (in progress 2026-07-14)
- CIoU/DIoU regression crashes under AMP float16 — see [[CIoU/DIoU Decoupled Regression Training Failure — 2026-07-08]].
- AMP fix deployed: disable autocast + force float32 + filter tiny boxes — see [[CIoU/DIoU AMP Crash Fix for Tiny Boxes]].
- Smooth-L1 AP75 variant (`sa_alw_full__smooth_l1__la_loss__seed42__smooth_l1_ap75`) trained on Kaggle T4 x2; best validation AP75 checkpoint is epoch 8 (`best_ap75.pt`).
- Locked test eval for downloaded Kaggle `best_ap75.pt`: COCO AP=0.0832, AP50=0.2665, AP75=0.0250, AR100=0.2260, mAP(scale)=0.5474. See [[Test-Set Evaluation — Phase 2 Metrics]] and [[Kaggle T4 Run Handoff — 2026-07-13]].
- Checkpoint-selection audit on 2026-07-22 found `smooth_l1_ap75/best.pt` (epoch 7) generalizes better than `best_ap75.pt`: COCO AP=0.0970, AP50=0.3000, AP75=0.0358, AR100=0.2525, mAP(scale)=0.5844.
- Conclusion: validation AP75 selection overfit; `smooth_l1_ap75/best.pt` is a useful candidate but not a new locked-test best. Current Phase 2 checkpoints remain stronger on either AP75 or mAP(scale).
- Prediction-level ensemble audit (`frcnn_standard__patches__seed42` + `smooth_l1_ap75/best.pt`) improved validation AP75 but failed the single frozen test gate: test COCO AP75=0.0354, below patch-seed42 AP75=0.0375 and smooth-L1 best AP75=0.0358.
- Local quality/boundary diagnosis on 2026-07-22 found AP75-quality raw tile candidates still exist (`any_pred_iou75` around 0.21), but boundary-only and micro-box heuristic rescoring did not promote on full validation.
- Quality-score branch launched on 2026-07-22 after local smoke from branch `qscore-experiments-20260722` (`deb9156`). The first six-kernel batch finished training but ended with `KernelWorkerStatus.ERROR` in post-training AP75 analysis because the diagnostics script used a local-only data path; logs still showed validation metrics. Best validation candidates were `q_smooth_l1_w10` (AP75=0.0616) and `q_smooth_l1_w05` (AP75=0.0611), but no checkpoint promotion is valid until outputs are downloaded and re-evaluated locally.
- Quality-score rerun batch launched on 2026-07-23 with artifact-first handling and `--skip-analysis`; all four kernels completed and artifacts were downloaded on 2026-07-24. Local locked-test evaluation did not promote: best qscore AP75 was `q_smooth_l1_w025/best_ap75.pt` at 0.0341, below `smooth_l1_ap75/best.pt` AP75=0.0358 and `frcnn_standard__patches__seed42` AP75=0.0375. Best qscore COCO AP was `q_smooth_l1_w05_seed2024/best_ap75.pt` at 0.0905, still below `smooth_l1_ap75/best.pt` AP=0.0970. See [[Kaggle T4 Run Handoff — 2026-07-13]] and [[Test-Set Evaluation — Phase 2 Metrics]].
- Side-aware Smooth-L1 (`side_smooth_l1`) passed a one-epoch CUDA smoke but failed the local multi-epoch gate on 2026-07-24: epoch 2 slowed to about 10-13 seconds/step in the normal probe, and the no-EMA/no-worker probe hit a native PyTorch/CUDA abort. Do not launch this branch on Kaggle; pivot to QFL/VFL or a cleaner DFL implementation.
- Confidence-driven box localization (`cbl`) completed its 20-epoch private T4 run and artifact audit. The legacy run's raw epoch-5 checkpoint reached validation AP/AP75=`0.1277/0.0554`, then its one frozen locked-test gate reached AP/AP50/AP75/AR100=`0.0987/0.3002/0.0390/0.2486`, a small new standalone AP75 best. The fixed 8-epoch recovery run completed and produced a reloadable EMA epoch-5 checkpoint at validation AP/AP50/AP75/AR100=`0.1409/0.3891/0.0665/0.2947`, weighted/micro class-aware AP=`0.5270/0.3697`. This is the current reloadable CBL validation leader. Do not reopen the locked test for the same CBL family. See [[Confidence-Driven Localization Local Gate - 2026-07-30]] and [[CBL EMA Recovery Audit - 2026-07-30]].
- Paper-faithful UGS entropy minimization (`lambda=0.5`) was tested separately after auditing Eq. 12. It completed two local epochs but underperformed target-entropy matching on AP, AP75, mAP(scale), and AP micro; checkpoint reload confirmed the result. Keep the implementation as an ablation but do not promote it to Kaggle.
- Gated dual-resolution RoI refinement (standard 7x7 plus zero-gated 14x14 residual) raised local epoch-1 AP75 from 0.0454 to 0.0571, but epoch 2 declined to 0.0529, total AP remained below standard CBL at epoch 2, and FPS fell from about 48 to 21. Retain as an AP75 diagnostic; wait for the standard CBL full-budget audit before considering cloud promotion.
- Evaluation caveat found 2026-07-30: legacy `AP_micro/tiny/small/large` ignores class labels during matching. New separately named COCOeval-based `AP_*_class_aware` metrics passed a synthetic wrong-class test. Standard CBL beats gated RoI14 on weighted class-aware scale AP (`0.4938` vs `0.4643`) and micro AP (`0.3515` vs `0.2490`).
- Distributional RPN localization used the paper's `alpha=2`, `beta=1`, 11-logit grid and passed CUDA/target-range checks, but failed the fair raw-weight local gate. Raw epoch 1 reached AP/AP75=`0.1057/0.0445`; raw epoch 2 declined to `0.0971/0.0386`, below standard CBL `0.1200/0.0471`. The much lower initial EMA metric was a harness confound because the baseline had EMA disabled. Do not launch RPN-CBL on Kaggle.
- Entropy-aware CBL score fusion also failed full validation. Multiplying class score by normalized localization confidence reduced AP/AP75 from `0.1200/0.0471` to `0.1156/0.0418` at `gamma=0.1` and `0.1065/0.0341` at `gamma=0.5`. Keep `gamma=0`; target-matched distribution entropy is not a localization-quality score.
- Adversarial RoI uncertainty refinement (`weight=0.5`, `rho=0.5`) produced a small strict-localization signal: reload AP/AP75/AR100=`0.1186/0.0488/0.2775` versus standard CBL `0.1200/0.0471/0.2759`. Total AP/AP50 and class-aware micro regressed, and training needed a clean restart after post-validation slowdown. Retain only as a diagnostic until the standard 20-epoch CBL artifact is audited.
- Official SET code/results were re-verified, then a scoped HBS-RoI Faster R-CNN adaptation was locally gated. Weight `0.5` collapsed by epoch 2; weight `0.1` still reduced weighted/micro class-aware AP to `0.4408/0.2342` at epoch 1. Both settings are negative overall gates and must not be launched on Kaggle.
- Paper-formula QFL was coupled directly to CBL as a joint foreground class-IoU score, without an auxiliary quality head or score multiplication. The two-epoch raw validation gate reached AP/AP75/AR100=`0.0965/0.0418/0.2561`, below standard CBL `0.1200/0.0471/0.2759`; micro class-aware AP also fell from `0.3515` to `0.2317`. Independent reload reproduced AP75=`0.0419`. This is a negative two-stage transfer result; do not launch or tune QFL beta blindly. See [[CBL Quality Focal Loss Local Gate - 2026-07-30]].
- A sampled RoI Rank & Sort adaptation also failed. Best epoch-1 AP/AP75=`0.0953/0.0327`, and epoch 2 declined to `0.0823/0.0293`. Direct score-IoU diagnosis was worse than standard CBL: Pearson/Spearman=`0.4977/0.4786` versus `0.6181/0.5225`, with severe score inflation. Do not launch, sweep `delta`, or use locked test. A future full RS-R-CNN reproduction must be treated as a separate multi-component experiment. See [[CBL Rank and Sort Local Gate - 2026-07-30]].
- Double-Head CBL separated the existing FC classifier from a paper-default four-bottleneck convolutional regression branch using `1.3x` RoIs. The two-epoch gate was negative: reload AP/AP75/AR100=`0.1047/0.0365/0.2513` versus standard CBL `0.1200/0.0471/0.2759`; micro class-aware AP fell from `0.3515` to `0.2788`. Do not launch or sweep it. The experiment also established `TOD_EMPTY_CACHE_EVERY=1` as an opt-in Windows workaround for CUDA cache paging during heavy-head local gates. See [[Double-Head CBL Local Gate - 2026-07-31]].
- Inference-time iterative CBL refinement is the first post-EMA localization win. Reapplying the same CBL regressor once to final detections while preserving labels/scores raised the EMA epoch-5 leader from AP/AP75=`0.1409/0.0665` to `0.1481/0.0746` with a `0.30` score gate; AR100 remained `0.2920` versus `0.2947`. A `0.20` gate reached the best AP75=`0.0747`. The direction repeated on raw epoch-5 and local epoch-2 checkpoints. Use one step, not two; do not reopen the locked CBL test gate. See [[Iterative CBL Refinement Gate - 2026-07-31]].
- Shared-head trainable CBL refinement passed both local and cloud gates. The private 8-epoch EMA run is complete and fully audited; exact EMA epoch-5 reload reaches AP/AP50/AP75/AR100=`0.1486/0.4030/0.0764/0.2949`, versus inference-only CBL `0.1481/0.4038/0.0746/0.2920`. TP@75 increases `1661->1702`, mainly from micro/tiny objects, but background predictions also rise. This is the current reloadable validation leader. Keep threshold `0.30`, use epoch 5 rather than `last.pt`, and do not reopen the CBL-family locked test. Next: a bounded stage-specific second CBL regression head with fixed first-pass class scores. See [[Trainable Iterative CBL Local Gate - 2026-07-31]].
- A stage-specific regression-only CBL refinement head failed to beat weight sharing. Its two-epoch reload AP/AP50/AP75/AR100=`0.1218/0.3420/0.0547/0.2729`, below shared-head refinement `0.1269/0.3612/0.0572/0.2758`; micro class-aware AP also falls to `0.2798`. Do not launch or tune this variant. A future full cascade stage must add refined-proposal re-matching and classification to address the leader's excess background predictions. See [[Stage-Specific CBL Refinement Local Gate - 2026-07-31]].
- The bounded full CBL cascade also failed. Stage-2 IoU-0.60 re-matching, resampling, classification, regression, and score averaging reload at AP/AP50/AP75/AR100=`0.1227/0.3537/0.0527/0.2684`. Preserving stage-1 scores improves AP/AP75 only to `0.1236/0.0549`, still below shared-head refinement. Do not launch or sweep cascade thresholds/weights; both separate-regression and classified-stage variants lose. See [[CBL Cascade Stage-2 Local Gate - 2026-07-31]].
- Reducing the shared-head refinement auxiliary loss from `0.50` to `0.25` is also negative. It starts faster at epoch 1 (AP75=`0.0556`) but declines to `0.0507` at epoch 2; independent reload confirms the epoch-1 checkpoint at AP/AP50/AP75/AR100=`0.1209/0.3414/0.0556/0.2661`. The weight-`0.50` local model remains better at `0.1269/0.3612/0.0572/0.2758`. Keep `0.50`; no Kaggle launch. See [[Trainable Iterative CBL Local Gate - 2026-07-31]].
- Paired refinement analysis on the trainable EMA epoch-5 leader found that self-IoU is not a useful quality score (post-IoU Pearson=`0.0370`; score x stability barely changes IoU75 AUC). Iterative depth is nevertheless positive: three ungated passes are the strict-validation leader at AP/AP50/AP75/AR100=`0.1501/0.4074/0.0774/0.2934`, while a predicted-size extra-pass cutoff equivalent to 12 px is the overall-AP/balanced leader at `0.1504/0.4081/0.0772/0.2946`. The cutoff restores tiny TP75 from `565` to `580` while preserving small TP75. Four passes reduce AP75 to `0.0766`. No self-IoU calibration, further scale/depth sweep, Kaggle rerun, or locked-test look. See [[CBL Refinement Consistency and Depth Gate - 2026-07-31]].
- Three-step unrolled CBL training is negative. The scale-gated variant reloads at AP/AP50/AP75/AR100=`0.1249/0.3591/0.0559/0.2685`; the ungated variant peaks at epoch 1 with `0.1206/0.3453/0.0536/0.2654` and declines at epoch 2. Both trail the one-pass-trained local leader `0.1269/0.3612/0.0572/0.2758`. Recurrent deep supervision shifts the shared head away from a good base-proposal/fixed-point balance; keep one-pass training plus inference-only repetition. No Kaggle, locked test, or unroll sweep. See [[Unrolled Iterative CBL Training Local Gate - 2026-07-31]].
- Full-trajectory analysis found a large GT-oracle ceiling (AP/AP75/AR100=`0.1662/0.0970/0.3233`) but size, self-IoU, direction, and update-growth selectors do not realize it. Damping only the third update to `0.50` gives AP/AP50/AP75/AR100=`0.1505/0.4077/0.0781/0.2943`, the best overall AP/AP75 profile; combining it with the 12 px gate gives `0.1505/0.4072/0.0779/0.2953`, the AR100 leader. Decomposing the last update adds a strict profile: center/size blends `0.25/0.50` reach the new maximum AP75=`0.0787` at AP=`0.1502`. Full-update pass 3 remains best for class-aware scale AP. No Kaggle or locked-test rerun because weights are unchanged. See [[CBL Refinement Trajectory and Damped Final Step - 2026-07-31]].
- Paired horizontal-flip TTA is the first large inference-only gain after CBL refinement. On the scalar-damped base, same-class original/flip matching at IoU `0.50`, score-weighted box averaging, and mean-score calibration reach AP/AP50/AP75/AR100=`0.1561/0.4227/0.0785/0.2961`, weighted/micro class-aware AP=`0.5609/0.3924`. The gain repeats on both validation halves and pair thresholds 0.50-0.70. Applying it to the center/size strict base plus a deterministic tiny-keep-box `<12` px rule reaches the maximum AP75=`0.0795` at AP=`0.1551`; scalar paired fusion remains the AP/AR leader. Union-NMS reaches maximum AR100=`0.3053` with lower AP75. Scalar+strict profile ensembling is negative and should not be promoted. Use TTA for offline/high-accuracy inference and single-view refinement for latency-sensitive deployment. No Kaggle or locked-test rerun because weights are unchanged. See [[CBL Horizontal-Flip TTA Local Gate - 2026-07-31]].
- Transform-scale TTA is the strongest inference-only win so far. Changing the detector transform to `960/1200`, pairing base/scale detections at IoU `0.50`, and keeping unmatched scale detections with score weight `0.75` first reached AP/AP50/AP75/AR100=`0.1653/0.4303/0.0889/0.3140`. Size-aware pair coordinates create the final bounded frontier: cutoff `12` px with scale alpha `0.75/0.40` reaches the maximum overall AP profile `0.1658/0.4314/0.0892/0.3151`; cutoff `16` with alpha `0.85/0.50` reaches maximum AP75=`0.0909` at AP=`0.1654`. The overall profile improves AP and AR on both original-image folds, while the strict profile improves AP75 on both folds. Stop inference calibration here; no Kaggle or locked-test rerun because weights are unchanged. See [[CBL Transform-Scale TTA Local Gate - 2026-07-31]].
- Naive stochastic multi-scale training does not absorb the transform-scale TTA gain. Training the same iterative CBL local gate with shorter-side choices `[640,800,960]` and fixed `640/800` validation peaks at epoch 1 with independently reloaded AP/AP50/AP75/AR100=`0.1141/0.3294/0.0436/0.2590`, below fixed-scale training at `0.1269/0.3612/0.0572/0.2758`; epoch 2 declines further. Legacy small-band AP improves, but micro/tiny AP and every primary COCO metric fall. Do not launch or sweep naive scale tuples. A future SNIP-like route must correctly ignore anchors/proposals around out-of-range objects rather than dropping GT. See [[CBL Stochastic Multi-Scale Training Local Gate - 2026-07-31]].
- Correct RPN/RoI ignored-object supervision does not rescue the equal-step stochastic scale formulation. The SNIP-like epoch-2 checkpoint independently reloads at AP/AP50/AP75/AR100=`0.1052/0.3060/0.0440/0.2490`, below fixed-scale `0.1269/0.3612/0.0572/0.2758`; micro legacy AP falls from `0.3400` to `0.1075`. Do not launch, use locked test, or sweep ranges. This rejects one-scale-per-image stochastic SNIP-like training, not full multi-resolution SNIP/SNIPER. See [[CBL SNIP-Like Scale-Normalized Training Local Gate - 2026-07-31]].
- Full-validation RPN audit on the trainable-CBL EMA epoch-5 leader finds top-1500 proposal recall of `0.8666` at IoU50 but only `0.3193` at IoU75. Micro IoU75 proposal recall is only `0.1552`, versus tiny/small/large `0.3305/0.3463/0.5207`. Increasing the proposal budget cannot recover most strict-localization misses, so the next bounded training direction is detached coarse-to-fine RPN anchor refinement and re-matching, not another scale or proposal-count sweep. See [[CBL RPN Proposal Coverage Audit - 2026-07-31]].
- Reapplying fixed RPN deltas once raises top-1500 proposal IoU75 recall from `0.3193` to `0.3424`, but the full detector falls from AP/AP75/AR100=`0.1486/0.0764/0.2949` to `0.1481/0.0753/0.2914`. A single pre-registered `>=16` px predicted-size gate reaches AP=`0.1492` but still lowers AP75/AR100 to `0.0760/0.2940`. Reject repeat-delta inference and do not sweep its size gate. The next bounded route is learned localization-quality RPN objectness; a full RPN cascade must train residual regression and re-match anchors. See [[CBL Iterative RPN Refinement Gate - 2026-07-31]].
- Learned IoU-aware RPN objectness improves strict proposal ranking but fails the detector gate. At the same two-epoch budget, QFL raises overall top-1500 IoU75 recall `0.3250->0.3475` and top-100 `0.1725->0.2049`, but micro top-1500 falls `0.1541->0.1386`; reload AP/AP75=`0.1173/0.0531` trails the local leader `0.1269/0.0572`. Preserving binary targets below `8/512` restores micro proposal recall to `0.1718` but displaces tiny/small proposals and peaks at only AP/AP75=`0.1192/0.0557`. Do not sweep or launch either target. The next RPN route must separate presence from localization quality or use a trained residual cascade. See [[CBL RPN Quality Objectness Local Gate - 2026-07-31]].
- A CFINet-inspired learned RPN cascade is also negative. A dilation-3 regression-only stage refines anchors, then detached anchors are re-matched for standard objectness and residual regression. Epoch-2 reload reaches AP/AP50/AP75/AR100=`0.1094/0.2914/0.0552/0.2486`, below the local leader `0.1269/0.3612/0.0572/0.2758`, while epoch time rises about `26.6%`. Micro top-1500 proposal IoU75 improves `0.1541->0.1858`, but overall top-100 IoU50 falls `0.5764->0.5228` and small top-1500 IoU75 falls `0.3950->0.3309`; micro class-aware detector AP collapses to `0.1813`. Reject the simplified cascade with no sweep, Kaggle, or locked test. It is not full CFINet because the nine-anchor contract omits adaptive feature offsets and bridging. Next test must keep presence and localization quality as separate RPN signals. See [[CBL Learned RPN Cascade Local Gate - 2026-07-31]].
- Separate RPN presence and positive-only localization-IoU prediction is a negative full-budget result. The initial shared-tower PAA-style head is unstable across two raw epochs and strongly shifts proposals toward large objects. A detached IoU tower plus tempered geometric fusion (`w=0.5`) fixes that instability and completed the same eight-epoch EMA schedule as the trainable leader. Its independently reloaded epoch-4 peak is AP/AP50/AP75/AR100=`0.1460/0.3866/0.0758/0.2923`, close to but below the leader `0.1486/0.4030/0.0764/0.2949`. Proposal audit improves tiny/large top-100 strict recall but lowers overall top-1500 IoU75 `0.3193->0.2976`, micro `0.1552->0.1339`, and small `0.3463->0.3122`. No Kaggle, locked test, or weight sweep. See [[CBL RPN IoU Quality EMA8 Audit - 2026-08-01]].
- The preregistered fair-20 comparison is complete and creates the new single-checkpoint test leader. Trainable iterative CBL ran all 20 epochs with seed 42, EMA, the shared SGD/cosine schedule, and one trainable/inference refinement pass. Frozen `best.pt` was selected only by validation mAP50 at epoch 5; independent reload gave AP/AP50/AP75/AR100=`0.1456/0.3969/0.0711/0.2959`. Its sole locked-test evaluation reached `0.1158/0.3326/0.0533/0.2657` and mAP(scale)=`0.6130`, improving historical SA-ALW by `+18.77%/+8.76%/+54.94%/+5.90%` and `+1.93%`, respectively. The test budget is consumed `1/1`; do not evaluate AP75/COCO-AP checkpoints, TTA, or alternate refinement profiles on test. See [[Iterative CBL Fair-20 Locked-Test Protocol - 2026-08-01]].
- Scale-Consistent CBL Distillation (SC-CBL) is a positive aggregate gate but not cloud-promotion ready. It keeps the student at `640/800`, aligns sampled positive RoIs to a frozen fair20 teacher at `960/1200`, and applies class-specific CBL KL only when teacher GT IoU is at least `0.02` better. Exact raw epoch-2 reload reaches AP/AP50/AP75/AR100=`0.1287/0.3628/0.0586/0.2765`, versus fixed-scale `0.1269/0.3612/0.0572/0.2758`; mAP(scale) is `0.5910` versus `0.5903`. Total TP, precision, recall, and all COCO size APs improve, but legacy micro/tiny AP regresses `0.3400/0.6197 -> 0.3260/0.5938`. Even-image fold AP/AP75 improves `+0.0035/+0.0017`, while odd fold regresses `-0.0019/-0.0014`; AR improves on both. Head-only KL isolation fails at `0.1124/0.3418/0.0443/0.2626`, proving final-predictor updates alone are insufficient. Stop local SC-CBL variants and do not launch Kaggle/locked test until a new method or independent evidence addresses the fold/scale inconsistency. See [[Cross-Scale CBL Localization Distillation Plan - 2026-08-01]].
- Conflict-Aware SC-CBL was rejected before implementation. Its preregistered no-update audit found only `4/200` negative-cosine batches (`2.0%`) versus the `10%` continuation threshold; mean/median cosine was `0.1497/0.1448`, and neither the micro/tiny nor larger diagnostic groups showed a negative batch. Gradient projection would affect too little training to explain the SC-CBL fold inconsistency. See [[Conflict-Aware SC-CBL Plan - 2026-08-01]].
- Coordinate-Reliable SC-CBL replaces whole-RoI distillation gating with detached per-coordinate teacher-advantage and entropy weights. Gates A/B passed, then the fresh seed-123 two-epoch comparison improved independently reloaded AP/AP50/AP75/AR100 from `0.1133/0.3149/0.0540/0.2599` to `0.1203/0.3311/0.0570/0.2644`; mAP(scale) rose `0.5329->0.5441`. AP, AP50, and AR improve on both original-image folds, while odd-fold AP75 is nearly flat at `-0.0006`. The frozen seed `42/123/2024` fair-20 matrix now has five long private kernels running across five accounts; all four new two-T4 smokes passed exact teacher and state-isolation checks. This remains pending downloaded artifacts and independent validation reloads, not a paper checkpoint. See [[Coordinate-Reliable SC-CBL Plan - 2026-08-01]] and [[CR-SC-CBL Multi-Seed Fair-20 Protocol - 2026-08-02]].
- Post-CR-SC-CBL audits reject four plausible variants. Flip-consensus retained `98.7%` of weights and failed its error-separation gate; ordered-W1 produced only `0.0107` weighted auxiliary/detector gradient norm; direct cross-head conflicted on `90.5%` of seed-42 box-head batches. PC-XH-CR-SC-CBL then isolated the auxiliary path from the backbone and applied PCGrad only on the student RoI box head. Its fresh seed-777 reload raised AP75/AR100 by only `+0.0004/+0.0021` while AP/AP50 fell `-0.0012/-0.0068`; even-fold AP fell `-0.0047`. PCGrad activated on only `2.39%/1.04%` of epoch-1/2 batches, so the seed-42 conflict pattern did not transfer. Reject without Kaggle, sweep, or test access. See [[Post-CR-SC-CBL Mechanism Gates - 2026-08-02]].
- RA-CR-SC-CBL aligns high-resolution teacher distillation with the student's detached post-refinement proposal instead of the first sampled RoI. Its technical gate passed, and fresh seed-9001 independent reload improved AP/AP50/AR100/mAP(scale) by `+0.0022/+0.0165/+0.0042/+0.0209`, but AP75 fell `-0.0003`; even-fold AP/AP75 fell `-0.0020/-0.0049`. It therefore fails the frozen robust gate and is rejected without a sweep, fair-20 promotion, or test access. See [[Post-CR-SC-CBL Mechanism Gates - 2026-08-02]].
- RA-TB-CBL keeps refinement alignment but uses the teacher only to select coordinates where the student is worse; selected coordinates optimize the exact ground-truth CBL target rather than teacher KL. Its seed-31415 raw/no-EMA gate passes every preregistered condition: independent AP/AP50/AP75/AR100/mAP(scale) improve from `0.1146/0.3049/0.0591/0.2530/0.5067` to `0.1226/0.3267/0.0646/0.2666/0.5285`; class-aware micro/tiny gain `+0.0421/+0.0441`; AP improves on both folds and odd-fold AP75 stays within guard at `-0.0009`. The same-source seed-42 20-epoch EMA baseline/candidate pair passed exact two-T4 smokes and is now running on Kaggle. No fair-20 result or new paper checkpoint exists until both artifacts and independent reloads pass; the locked test remains closed. See [[Post-CR-SC-CBL Mechanism Gates - 2026-08-02]] and [[RA-TB-CBL Fair-20 Protocol - 2026-08-02]].
- A full-valid cross-scale RPN audit shows that the same fair20 checkpoint at `960/1200` is worse globally than at `800/800`, but is complementary for the `1,927` micro GTs. Direct MR-RPN fails because objectness conflicts on nearly every valid batch. The bounded PC-MR-RPN successor drops objectness and applies weight-`0.005` exact-GT regression with RPN-head PCGrad. Its 200-batch Gate0 and CUDA/reload gates passed. The seed-2718 independent reload then improved AP/AP50/AP75/AR100/mAP(scale) by `+0.0097/+0.0327/+0.0041/+0.0140/+0.0335`; both folds improved AP and AP75. All frozen gates pass, so PC-MR now runs in a same-source seed-42 fair-20 matrix with PC-MOC and one shared baseline. No test access. See [[Post-CR-SC-CBL Mechanism Gates - 2026-08-02]] and [[PC Micro Fair-20 Protocol - 2026-08-02]].
- PC-MOC-FD transfers channel-normalized `7x7` FPN RoI features only for exact-GT micro objects where the high-resolution teacher has a bounded proposal-IoU advantage. Its 200-batch FPN-only Gate0 and four-step CUDA/reload gates passed. The seed-2718 independent reload improved AP/AP50/AP75/AR100/mAP(scale) by `+0.0063/+0.0208/+0.0057/+0.0194/+0.0131`; even and odd folds both improved AP and AP75. All frozen gates pass. PC-MOC and PC-MR now share source `e3c1274c...8111`, seed42, 20 epochs, EMA, one baseline, exact two-T4 smokes, and validation-mAP50 checkpoint selection; all three long jobs are running. No fair-20 or test claim yet. See [[PC-MOC-FD Gates - 2026-08-02]] and [[PC Micro Fair-20 Protocol - 2026-08-02]].
- PC-MSDD failed its frozen PCGrad-justification gate. Its high-frequency successor PC-MHFD passed Gate0, CUDA/reload, and two-T4 smokes, then improved seed1618 aggregate AP/AP75/AR100 by `+0.0028/+0.0035/+0.0279`. It nevertheless failed robustness: even-fold AP `-0.0012`, odd-fold AP75 `-0.0046`, class-aware micro/tiny `-0.0146/-0.0506`, and mAP(scale) `-0.0029`. Reject without sweep, fair-20, combination, or test access. See [[PC-MHFD Gates - 2026-08-02]].
- RA-TB plus PC-MHFD passed a frozen 200-batch FPN compatibility audit and a shared-teacher CUDA/reload contract before performance results. The independent PC-MHFD gate then failed, so the prerequisite is not met and no combination cloud run is launched. Compatibility is retained as technical evidence only. See [[RA-TB plus PC-MHFD Combination Gates - 2026-08-02]].
- CIoU variant (seed 42) currently 11/20 epochs with resumed training after AMP fix.

## Key Results (COCO Metrics, Test Set, Seed 42)

### ALW Paper — Main Results

| Metric | AP | AP50 | AP75 | AP_S | AP_M | AP_L | AR100 |
|--------|-----|------|------|------|------|------|-------|
| IoU full-image | 0.0871 | 0.2670 | 0.0325 | 0.0699 | 0.2353 | 0.4856 | 0.2280 |
| IoU patches | 0.0958 | 0.2921 | 0.0375 | 0.0830 | 0.1938 | 0.4953 | 0.2470 |
| NWD | **0.0932** | **0.2975** | 0.0286 | **0.0882** | 0.1821 | 0.0985 | **0.2456** |
| IGWD | 0.0874 | 0.2642 | 0.0371 | 0.0751 | 0.2048 | 0.4773 | 0.2318 |
| ALW (wrapped) | 0.0738 | 0.2210 | 0.0326 | 0.0654 | 0.1701 | 0.3878 | 0.2058 |

**Key findings:**
- NWD: best small-object AP but large-object collapse (AP_L=0.0985).
- IGWD: best strict localization (AP75=0.0371) and medium-object AP.
- IGWD+aniso: improves AP over IGWD (+0.0034) via better small-object coverage — anisotropy generalizes.
- IGWD+log-shape: preserves AP_L (0.4733) but lowest overall AP.
- ALW wrapped: overfits validation→test (Δ=-0.126), pure formulation training pending.

### SA-ALW Paper — Main Results

| Metric | AP | AP50 | AP75 | AP_S | AP_M | AP_L | AR100 |
|--------|-----|------|------|------|------|------|-------|
| IGWD | 0.0874 | 0.2642 | 0.0371 | 0.0751 | 0.2048 | 0.4773 | 0.2318 |
| ALW (wrapped) | 0.0738 | 0.2210 | 0.0326 | 0.0654 | 0.1701 | 0.3878 | 0.2058 |
| SA-ALW β-only | 0.0917 | 0.2894 | 0.0324 | 0.0802 | 0.1930 | 0.4639 | 0.2462 |
| SA-ALW w_pos-only | 0.0968 | 0.2975 | **0.0353** | 0.0851 | 0.1949 | **0.4867** | 0.2517 |
| **SA-ALW full** | **0.0978** | **0.3059** | 0.0345 | **0.0866** | **0.1924** | 0.4247 | **0.2524** |

**Key findings:**
- SA-ALW full: best AP (0.0978) and AP50 (0.3059), exceeding IGWD by +11.9%.
- β(s): improves AR100 by +0.0404 over wrapped ALW, boosts AP_L.
- w_pos(s): best AP_L (0.4867) and AP75 (0.0353) among Gaussian metrics.
- Both mechanisms contribute independently with some overlap.
- HLA dominates threshold-based assignment by +0.0236 AP.
- Metric-NMS negligible (|Δ|<0.0005).

## Next Steps

1. ~~Phase 1-4 — Baselines, metric ablation, SAALWAssigner, cascaded WBF.~~ ✅ All completed.
2. ~~Checkpoint-selection audit for downloaded Kaggle runs.~~ Done 2026-07-22; `smooth_l1_ap75/best.pt` is the only useful candidate, while `cp_light`, `os1`, and `os125` underperform on locked test.
3. ~~Prediction-level ensemble/WBF between patch42 and `smooth_l1_ap75/best.pt`.~~ Tried 2026-07-22; validation AP75 improved but frozen test did not promote.
4. ~~Quality-score rerun locked-test gate.~~ Completed 2026-07-24; no standalone qscore checkpoint promoted. Treat learned quality scoring as validation-overfit unless paired with a stronger localization/regularization change.
5. **Confidence-driven localization** — completed. Use the reloadable recovery EMA epoch-5 checkpoint from [[CBL EMA Recovery Audit - 2026-07-30]] as the validation leader; preserve the single prior CBL locked-test result without another test-set look.
6. Train pure ALW (without Charbonnier wrapper) for clean test evaluation.
7. Multi-seed experiments (seeds 123, 2024) for ALW, SA-ALW, IGWD.
8. Cross-dataset validation: AI-TOD, AI-TOD-v2.
9. Cascade breakthrough — scale-aware WBF + extent hull fusion.
10. Submit papers to IEEE TIP.
