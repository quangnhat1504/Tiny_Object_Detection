---
title: Scale-Adaptive Hybrid Gaussian Distance (SAH-GD)
type: concept
created: 2026-05-31
updated: 2026-05-31
sources: [working/idea/5_sah_gd.md, working/code/README_sah_ablation.md]
tags: [tiny-object-detection, metric, loss, ablation, kaggle]
---

## Scale-Adaptive Hybrid Gaussian Distance (SAH-GD)

## Definition

SAH-GD is a proposed hybrid Gaussian distance for tiny object detection. It blends adaptive NWD and symmetric GCD with a differentiable scale gate:

```text
D_H = (1 - alpha) * D_N + alpha * D_G
```

where:

```text
D_N = sqrt(W2^2 + eps) / C(s_bar)
C(s_bar) = sqrt((lambda * s_bar)^2 + C_min^2)
alpha = sigmoid((s_bar - tau) / k)
```

The preferred future form is FPN-normalized:

```text
alpha_l = sigmoid(((s_bar / stride_l) - tau_f) / k)
```

The current Kaggle ablation notebooks implement the pixel-level fallback because the existing RPN scaffold does not yet pass FPN level ids or strides into the metric function.

## Motivation

The local four-metric experiment showed a scale-dependent pattern:

- [[NWD]] has the best `AP_micro`, but only by `0.0070` over [[GCD]].
- [[GCD]] is strongest overall and has much fewer detections per image.
- All metrics have low `COCO AP@75`, so localization and proposal quality remain bottlenecks.

SAH-GD tries to keep NWD's micro-object sensitivity while retaining GCD's stronger tiny/small/general behavior.

## Mathematical Claims

- Non-negative because it is a convex combination of non-negative distances.
- Bounded between adaptive NWD and GCD:

```text
min(D_N, D_G) <= D_H <= max(D_N, D_G)
```

- Symmetric if `s_bar`, `C(s_bar)`, symmetric GCD, and one shared FPN level/stride are used.
- Controlled micro-object gradient because `C(s_bar) >= C_min`.
- Similarity is bounded:

```text
0 < exp(-beta * D_H) <= 1
```

## Implementation Scope

The first implementation round keeps only the core ablations:

- adaptive NWD only,
- hard switch between adaptive NWD and GCD,
- SAH-GD soft blend,
- SAH-GD soft blend plus scale-adaptive RFLA top-k.

Density-aware weighting and duplicate suppression are intentionally treated as future work because they add hyperparameters and are harder to ablate on limited GPU budget.

## Related Pages

- [[Tiny Object Metric Ablation Plan - 2026-05-31]]
- [[Tiny Object Detection Metrics]]
- [[NWD]]
- [[GCD]]
- [[RFLA]]
