---
title: "Deep Research: Tiny-OD Breakthroughs 2024–2026 & the AP@75 Diagnosis"
type: research
created: 2026-07-06
updated: 2026-07-06
sources: [arxiv:2404.03507, arxiv:2406.05755, arxiv:2407.02394, arxiv:2301.10051, arxiv:2006.04388, CVPR2025-SET, mdpi:2072-4292/18/3/396]
tags: [deep-research, tiny-object-detection, ap75, regression-loss, dfl, label-assignment, detr, sota]
---

# Deep Research: Tiny-OD Breakthroughs 2024–2026 & the AP@75 Diagnosis

> Conducted 2026-07-06 to find a breakthrough direction beyond the SA-ALW plateau.
> The deep-research workflow gathered 62 claims from 24 primary sources but its
> automated verify phase crashed (StructuredOutput tool bug → all votes 0-0 →
> falsely marked "refuted"). **No claim was actually refuted.** The facts below
> were **re-verified manually via direct WebFetch** on the primary papers.

## The Key Diagnosis (most important finding)

On **AI-TOD-v2** (verified from DQ-DETR paper, arxiv:2404.03507):

| Model (AI-TOD-v2) | Backbone | AP | AP50 | **AP75** |
|---|---|---|---|---|
| **RFLA** | ResNet50-FPN | 25.7 | 58.9 | **18.8** |
| DINO-DETR | ResNet50 | 25.9 | 61.3 | 17.5 |
| DQ-DETR (SOTA) | ResNet50 | 30.2 | 68.6 | 22.3 |
| **Our project** (SOD-TinyPeopleInSea) | ResNet50-FPN | 0.098 | 0.306 | **0.02–0.045** |

**RFLA — the exact assigner our project uses — reaches AP75 = 18.8.** Our project
uses RFLA for assignment yet AP@75 is ~0.03, i.e. **4–8× lower than vanilla RFLA**.
Because RFLA does **not** touch box regression (it gets 18.8 with standard
smooth-L1), the deficit must come from **overloading the Gaussian metric onto the
box-regression loss** as well as assignment.

**Conclusion:** The AP@75 collapse is NOT a tiny-object limit and NOT a flaw in
SA-ALW-as-assigner. It is caused by using an IoU-insensitive Gaussian similarity
(`1 − exp(−β·D_H)`) as the *regression* loss, which saturates before boxes are
pixel-tight. Culprit is `MetricRoIHeads.compute_loss` at `common/model.py:242`
(`box_loss = (1 − sim).mean()`). Fixing regression alone could ~double COCO AP,
since the entire AP50=0.31 → AP75=0.03 gap is the story.

Caveat: SOD-TinyPeopleInSea ≠ AI-TOD-v2, so absolute numbers differ; the
mechanism argument and the same-family RFLA comparison are what hold.

## Lane 1 — Regression loss / strict localization (PRIMARY)

- **Distribution Focal Loss (DFL) / Generalized Focal Loss** (arxiv:2006.04388,
  verified): represents each box side as a discrete probability distribution over
  bins instead of a Dirac-delta point estimate, letting the head express boundary
  uncertainty (exactly the 2–3px annotation noise of micro objects). GFLV2 uses the
  distribution's top-k+mean statistics to predict IoU/localization-quality → +~2 AP
  with no speed cost.
- **Gap identified (novel opportunity):** DFL is native to one-stage/dense heads
  (YOLO, GFL). Applying distribution-regression to a **two-stage RoI head for tiny
  objects, evaluated on AP75**, is unexplored in the literature searched → strong
  second-paper angle.
- **Wise-IoU** (arxiv:2301.10051): non-monotonic dynamic focusing via outlier
  degree; reduces harmful gradients from low-quality boxes. Reported YOLOv7 AP75
  53.03→54.50 on COCO (claim, not re-verified in full text).
- Caution (verified via search): strict-localization losses can *reduce* small-object
  recall if applied too aggressively (one UAV study: small AP 0.119→0.106). → ramp γ,
  do not apply full strength from epoch 1 (matches our notebook-10 failure).

## Lane 2 — Detector architectures

- **DQ-DETR** (arxiv:2404.03507, verified): DINO-DETR + a categorical counting module
  that picks decoder query count (300/500/900/1500) by image object-count bin.
  AI-TOD-v2 **AP75=22.3**, AP=30.2 (SOTA). BUT authors ran **batch=1 on a 3090 (24GB)**
  "due to memory constraints" → **risky on 16GB**, and a large rewrite. High ceiling,
  high cost. Flag as stretch / paper 3.
- **DNTR** (arxiv:2406.05755, verified): DeNoising-FPN (contrastive, plug-in) +
  Trans R-CNN. +17.4% APvt on AI-TOD, +9.6% AP on VisDrone. DN-FPN is a plug-in;
  Trans R-CNN is heavier than a vanilla RoI head.
- **Cross-DINO / Dome-DETR** (arxiv:2505.21868 / 2505.05741): DETR+MLP-backbone SOTA
  on SODA-D / AI-TOD-v2. Architecture pivots, not plug-ins.

## Lane 3 — Label assignment beyond RFLA

- **SimD** (arxiv:2407.02394, verified partial): adaptive similarity-distance metric
  (location + shape), **no fixed hyperparameters**, non-Gaussian (does not saturate).
  Drop-in for IoU in assignment AND NMS. +1.8 AP / +4.1 very-tiny AP on AI-TOD.
- **BCDet** (mdpi 2072-4292/18/3/396): Normalized Bhattacharyya distance between a 2D
  Gaussian RF and GT box; +9.0% AP on Faster R-CNN over baseline on AI-TOD-v2.
- **DILA / ADAS-GPM** (sciencedirect): Gaussian-distribution assigners; RFLA
  successors, +AP on Faster R-CNN family.
- These are useful **baselines/ablations** for the SA-ALW paper, and confirm the
  assignment lane is crowded → differentiate on regression, not assignment.

## Lane 4 — Data / feature

- **SET: Spectral Enhancement for Tiny OD** (CVPR 2025), re-verified against the
  official paper/code on 2026-07-30. Its released FCOS implementation uses
  scale-adaptive residual convolutions for hierarchical background smoothing
  (HBS), normalized loss gradients for adversarial perturbation injection
  (API), and a second detection pass during training; it does not execute an
  FFT in the detector forward path. On AI-TOD, the official table reports FCOS
  AP/AP75 `12.0/8.0` and FCOS+SET `14.2/9.8` (`+2.2 AP`, `+1.8 AP75`).
  Inference uses the original features, so the enhancement adds no inference
  path. The released detector is single-stage FCOS; any use with this project's
  Faster R-CNN/RFLA stack must be labeled and gated as an adaptation rather
  than a paper-faithful reproduction. Verified code revision:
  `huixinsun/SET@9208fbc4cfe571be4c15dccad8db1665cfdcb9d6`.

## Recommended direction (flagship)

**Decoupled Assignment–Regression:** keep SA-ALW for label assignment; replace the
Gaussian-similarity regression loss with a **distribution-based (DFL-style) RoI box
head + IoU-quality branch (GFLV2-style)**. Directly targets AP@75, novel for
two-stage tiny-OD, cheap on 16GB, and fully compatible with the current stack.
Detailed plan: [[Decoupled DFL Regression Plan - 2026-07-06]].

## Related Pages
- [[Decoupled DFL Regression Plan - 2026-07-06]]
- [[Scale-Adaptive Anisotropic Log-Wasserstein Distance (SA-ALW)]]
- [[SAH-GD Advancement - 2026-06-02]]
- [[Deep Research: Architecture & Training Strategies for Tiny Object Detection]]
