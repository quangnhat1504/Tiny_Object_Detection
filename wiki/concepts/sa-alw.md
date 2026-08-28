---
title: Scale-Adaptive Anisotropic Log-Wasserstein Distance (SA-ALW)
type: concept
created: 2026-07-01
updated: 2026-08-02
sources: [raw/detail_implement.md, raw/plan.md, paper-a-sa-alw-conference-refinement-plan]
tags: [tiny-object-detection, metric, loss, scale-adaptive, proposal]
---

## Scale-Adaptive Anisotropic Log-Wasserstein Distance (SA-ALW)

## Definition

SA-ALW is the terminal metric in the chain `IoU → NWD → IGWD → ALW → SA-ALW`. It extends ALW (Anisotropic Log-Wasserstein) with two independent scale-adaptive mechanisms.

## Metric Chain

```
IoU: fails for tiny objects (overlap-based, drops to 0 at small offsets)
  ↓
NWD: Gaussian Wasserstein, smooth but not scale-invariant
  ↓
IGWD: area-normalized Wasserstein, scale-invariant but isotropic + Euclidean shape
  ↓
ALW: anisotropic normalization + log-ratio shape (fixes IGWD's two flaws)
  ↓
SA-ALW: ALW + Scale-Adaptive β(s) + Scale-Adaptive position weight w_pos(s)
```

## SA-ALW Formula

```text
ALW²(s) = w_pos(s) · [ (Δx)²/Sx + (Δy)²/Sy ] + [ (ln wa/wg)² + (ln ha/hg)² ]
SA_ALW_sim = exp(-β(s) · √(ALW²))
```

where:
- **Anisotropic**: `Sx = (wa²+wg²)/2, Sy = (ha²+hg²)/2` (separate per axis)
- **Log-ratio shape**: scale-invariant size comparison
- **beta(s)**: smaller targets may receive a larger fixed scheduled value.
- **w_pos(s)**: smaller targets may receive a larger fixed scheduled value.
- Schedule bounds are clipped and must be fitted only from the eventual frozen
  training split. Historical values `5.6/28.7` are not Paper A constants.

## Status

`[Canonical code tested; performance evidence pending]`. SA-ALW has no added
learnable parameters, but its target-scale conditioning makes it asymmetric and
not jointly scale-invariant. It is described as a Wasserstein-inspired
similarity/loss, not as a proven mathematical metric.

Primary-source audit also forbids a broad first scale-adaptive claim. SAFit
already blends IoU and NWD with a target-area sigmoid, SimD derives global
axis normalizers from train data, and GCD applies scale-invariant Gaussian
geometry to assignment plus regression. SA-ALW's remaining boundary is its
exact two-schedule extension and the separately audited roles of beta and
position emphasis.

## Related Pages

- [[Anisotropic Log-Wasserstein Distance (ALW)]]
- [[IGWD]]
- [[Cascaded Uncertainty Routing]]
- [[Paper A SA-ALW Conference Refinement Plan]]
- [[SA-ALW Paper Refinement Phase 0-2 - 2026-08-02]]
