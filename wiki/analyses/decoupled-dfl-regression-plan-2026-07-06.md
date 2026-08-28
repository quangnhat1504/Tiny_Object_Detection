---
title: Decoupled DFL Regression Plan - 2026-07-06
type: analysis
created: 2026-07-06
updated: 2026-07-06
sources: [common/model.py, common/config.py, common/metrics/sa_alw.py, scripts/train_frcnn_metric.py, wiki/research/deep-research-tiny-od-breakthroughs-2026-07-06.md]
tags: [ap75, regression-loss, dfl, gflv2, decoupled, action-plan, paper2]
---

# Decoupled DFL Regression Plan — 2026-07-06

## Question
How do we break the universal AP@75 = 0.02–0.045 ceiling without discarding
SA-ALW, on a single RTX 5070 Ti 16GB, in a way that is publishable as a second
paper?

## Diagnosis (settled)
See [[Deep Research: Tiny-OD Breakthroughs 2024–2026 & the AP@75 Diagnosis]].
Vanilla RFLA (our assigner) reaches AP75=18.8 on AI-TOD-v2 using **standard
regression**; our AP@75 ≈ 0.03 because we overloaded the Gaussian metric onto the
box-regression loss. The fix is to **decouple**: SA-ALW stays in assignment;
regression gets a proper strict-localization signal.

Exact hook: `common/model.py` → `MetricRoIHeads.compute_loss` line 242:
`box_loss = (1.0 - sim).mean() * self.metric_loss_weight`. This one line is the
bottleneck. `pred_boxes` are already decoded to xyxy at line 228, so an IoU-based
or distribution-based loss can be dropped in directly.

## Thesis for the paper (paper 2)
"Gaussian/Wasserstein metrics are optimal for *assignment* but structurally unfit
as *regression* losses for tiny objects (IoU-insensitive, saturate before tight).
We decouple the two roles: SA-ALW drives label assignment, and a distribution-based
RoI regression head with an IoU-quality branch supplies strict localization. On
tiny maritime objects this lifts AP@75 from ~0.03 to X and roughly doubles COCO AP,
with no backbone change." DFL applied to a **two-stage RoI head for tiny objects**
is an identified literature gap.

## Execution order (one variable at a time — do NOT stack)

### Step 0 — Diagnostic confirmation (0.5 day, cheapest, decisive)
Goal: prove regression is the culprit before building anything.
- Edit `MetricRoIHeads.compute_loss`: keep SA-ALW similarity as an auxiliary term
  but make the **primary** box loss a CIoU/DIoU on the decoded `pred_boxes` vs
  `targets_pos` (both already available at line 228/236).
- `box_loss = ciou_loss(pred_boxes, targets_pos_xyxy) + λ_aux·(1 − sim).mean()`,
  with `λ_aux` small (0.25) and a warmup: pure `(1−sim)` for epochs 0–2, then ramp
  CIoU in over epochs 3–6 (the notebook-10 lesson: no full-strength-from-epoch-1).
- Keep everything else identical to the SA-ALW `la_loss` run (seed 42, 20 epochs).
- **Success gate:** test AP@75 jumps from ~0.03 to > 0.08 AND AP_micro does not drop
  more than 0.02. If yes → diagnosis confirmed, proceed to Step 1. If AP_micro
  collapses → the CIoU is pruning micro recall; reduce ramp / lower CIoU weight.

### Step 1 — DFL RoI regression head (core contribution, 2–3 days)
Replace the 4-scalar delta regressor with a **distribution regressor**:
- Box head predicts, per RoI per side (l,t,r,b relative to RoI, in a normalized
  range), a softmax over `n_bins` (start n_bins=8, reg_max=8 in RoI-normalized
  units). Predicted offset = Σ p_i · bin_i (expectation).
- Loss = **DFL** (cross-entropy pulling mass onto the two bins bracketing the
  continuous target) + a **CIoU** on the integrated box (GFL-style joint).
- Integrate into `MetricRoIHeads`: this changes `box_predictor` output dim
  (4·n_bins instead of 4·num_classes deltas) — class-agnostic regression keeps it
  small. Add a new `DFLFastRCNNPredictor` in `common/model.py` and a
  `dfl_bbox_loss` helper; gate it behind a config flag `USE_DFL_HEAD`.
- SA-ALW still does RPN label assignment (MetricRPN unchanged).
- **Why it fixes saturation:** the CE-to-bins gradient stays strong as overlap → 1
  (unlike `β·exp(−β·D_H)` which vanishes), and the distribution encodes the
  boundary ambiguity of 2–3px micro boxes instead of forcing a false-precise point.

### Step 2 — GFLV2 IoU-quality branch (0.5–1 day, +~2 AP historically)
- A light 2-FC subnet takes the top-k (k=4) values + mean of each side's
  distribution → predicts localization quality (IoU); multiply into the class score
  at inference so tight boxes rank above loose ones. Cheap, no retraining of
  backbone.

### Step 3 — Ablation & write-up
- Test-set (locked, 65 img) table: SA-ALW baseline vs +Step0 vs +DFL vs +DFL+IoU-branch.
- Report AP@75, AP_micro, AP_tiny, AP50, mAP(scale), det/img, FPS.
- Multi-seed (42/123/2024) for the winner → mean±std (closes a known paper gap).

## Optional orthogonal add-ons (only after Steps 0–3 land)
- **SET** (CVPR'25) as a training-only feature enhancer. Re-verification on
  2026-07-30 found official AI-TOD FCOS gains of `+2.2 AP` and `+1.8 AP75`.
  The released code combines convolutional HBS, loss-gradient API, and an
  auxiliary dense-head pass. A Faster R-CNN version is non-paper-faithful and
  needs an isolated validation gate before any full-budget run.
- **SimD / BCDet** as *assignment* baselines in the related-work comparison table.

## Explicitly deferred / rejected for now
- DQ-DETR / DETR pivot: highest ceiling but authors needed batch=1 on 24GB →
  infeasible-to-risky on 16GB, large rewrite. Revisit only if the decoupled head
  plateaus. Track as potential paper 3.
- More metric-formula blending: plateaued (~1% spread), see [[SAH-GD Advancement - 2026-06-02]].
- Metric-NMS: proven no effect (|Δ|<0.0005).

## Compute budget (RTX 5070 Ti 16GB)
All steps keep ResNet50-FPN + P2, no backbone change. DFL head adds ~4·n_bins
outputs on the RoI predictor + a light branch → negligible VRAM. Same 20-epoch
schedule as prior runs. Fully feasible on 16GB.

## Success definition
Primary: test **AP@75 > 0.10** (from ~0.03) with AP_micro ≥ current 0.51.
Stretch: COCO AP > 0.15 (from 0.098). Any of these is a clean paper-2 result.

## Related Pages
- [[Deep Research: Tiny-OD Breakthroughs 2024–2026 & the AP@75 Diagnosis]]
- [[Scale-Adaptive Anisotropic Log-Wasserstein Distance (SA-ALW)]]
- [[Phase 2-4 Results Summary]]
- [[SAH-GD Advancement - 2026-06-02]]
