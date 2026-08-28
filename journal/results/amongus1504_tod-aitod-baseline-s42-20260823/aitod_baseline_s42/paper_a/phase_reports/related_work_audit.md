# Related-Work and Novelty Audit

Status: `CLOSEST_FORMULA_AUDIT_COMPLETE; BASELINE_FEASIBILITY_REVISE`
Checked: `2026-08-02`

## Decision

Paper A must not claim the first scale-adaptive Gaussian box measure, the first
axis-normalized box distance, or the first scale-invariant Gaussian distance
used for assignment and regression. Primary sources already establish each of
those broader ideas.

The defensible method boundary is narrower:

> ALW uses the exact combination of per-axis mean-square center normalization
> and squared log-ratio shape terms. SA-ALW adds two separate, clipped
> target-scale schedules: similarity decay rate for assignment and position
> emphasis for assignment plus aligned regression. It adds no learnable
> geometry parameters and retains fixed IoU-NMS in Paper A.

This is a formulation and placement claim, not a priority claim. Its empirical
value remains conditional on the matched public-benchmark matrix.

## Closest-Method Matrix

| Work | Geometry | Scale handling | Placement | Overlap with Paper A | Required distinction |
|---|---|---|---|---|---|
| NWD (2021/2022) | Gaussian Wasserstein center and linear shape difference | Global dataset constant | Assignment, loss, NMS | Non-overlap signal and Gaussian-box basis | ALW uses per-axis pair-dependent normalization and log-relative shape |
| RFLA (ECCV 2022) | Receptive-field distance plus HLA | Hierarchical assignment quotas | Assignment | Paper A retains the HLA skeleton | Attribute gains only to the changed similarity/loss, not HLA |
| SimD (IROS 2024) | Per-axis location and shape normalization by width/height sums | Global `m,n` estimated on train | Assignment and NMS | Axis-specific normalization and train-derived adaptation predate ALW | Distinguish mean-square denominators, log-ratio shape, target-conditioned schedules, and fixed IoU-NMS |
| SAFit (TPAMI 2025) | Size-aware sigmoid blend of IoU and NWD | Target GT area with fixed transition constant `C` | Evaluation and loss | Target-size-adaptive box fitness/loss predates SA-ALW | Distinguish two schedules inside one geometry and audited assignment/regression roles |
| GCD (GRSL 2025) | Symmetric inverse-variance center and relative shape terms | Intrinsic affine/scale normalization | Assignment and regression | Axis-aware scale invariance and dual placement predate ALW | Distinguish mean-square rather than inverse-variance form and squared log-ratio shape |
| IGWD (TMM 2026) | Gaussian Wasserstein distance divided by summed box area | Pair-dependent isotropic area normalization; fixed beta mapping | Assignment and loss | Verified direct predecessor and scale-invariant Gaussian formulation | ALW changes isotropic area normalization to axis-specific denominators and changes Euclidean shape to log-ratio shape |
| SWIN-TOD/SWL (TGRS 2024) | Piecewise first-/second-order Wasserstein offset norm | Switches by offset magnitude threshold, not target size | RPN and head regression | Smooth piecewise Wasserstein regression predates Paper A | Distinguish target-conditioned schedules and assignment role; do not call piecewise regression new |
| MMPW (Remote Sensing 2024) | Weighted normalized Wasserstein similarity plus MPD-adjusted IoU | Dataset mean target scale `C`; fixed mixture weights | Assignment, loss, and NMS | Dataset-scale normalization and broad multi-placement predate Paper A | Distinguish target-conditioned beta/position schedules and fixed IoU-NMS |
| DILA/BGSM (Applied Soft Computing 2024) | KFIoU plus per-axis normalized center similarity | Dynamic receptive-field priors and fixed BGSM mixture | RPN label assignment in released config | Its center term has the same per-axis squared-width/height sums as ALW up to a factor of two | Novelty can reside only in the full center-plus-log-shape formulation and schedule placement, not the center denominator alone |

## Formula-Level Findings

### IGWD

The official accepted manuscript defines summed box area
`S = w_p h_p + w_t h_t`, normalizes Gaussian Wasserstein distance by `S`, and
maps it to similarity with either an exponential or rational form controlled
by a fixed beta. This confirms IGWD as the direct formula predecessor, but it
does not contain SA-ALW's target-scale beta or position-weight schedules.

Local source SHA-256:
`7268ad1ad5fe5cab058138af8dfc4a081a621da6b0ee57fc82fed1a6b25186e1`.

### SimD

SimD normalizes x/y location and width/height shape differences separately by
the corresponding sums of box widths/heights. Its global `m,n` factors are
computed from the training set. Therefore neither axis normalization nor
train-derived adaptation can be presented as new by itself.

### SAFit

SAFit uses a sigmoid of target GT area to switch continuously between IoU and
NWD and defines a corresponding loss. Therefore Paper A cannot use a broad
"first target-scale-adaptive similarity/loss" claim.

### GCD

GCD symmetrizes inverse predicted/target variance terms for both center and
shape, proves scale/affine invariance, and applies the distance to label
assignment and regression. ALW differs in its mean-square center denominator
and exact log-ratio shape representation, not in the general ideas of
scale-invariant Gaussian geometry or dual placement.

### SWIN-TOD / SWL

SWL uses the L1 norm of the four-dimensional box offset vector when that norm
exceeds a threshold and the squared L2 norm otherwise. It replaces regression
loss in the RPN and detector head. The switch is conditioned on error magnitude,
not target scale, and the released repository references `SmoothWassLoss` in
configs but does not contain its registered implementation at audited commit
`e8bc10f0a20ceae6852ac4a9753774b04f4f82e5`. A matched run would therefore be
an independent formula reproduction, not an official-code baseline.

### MMPW

MMPW mixes an exponential normalized Wasserstein term with an MPD-adjusted IoU
term using fixed weights and a dataset-average scale constant `C`. The paper
uses it for label assignment, regression loss, and NMS, so a one-component
substitution under Paper A's fixed IoU-NMS would not reproduce MMPW-Net. No
official implementation repository is linked by the primary article.

### DILA / BGSM

The released BGSM code contains the center term
`dx^2/(w_p^2+w_t^2) + dy^2/(h_p^2+h_t^2)`. Canonical ALW's position term is
exactly twice this quantity. DILA therefore predates the ALW center
normalization form up to a constant factor. Its full method also changes
dynamic receptive-field priors and features; the released Faster R-CNN config
uses BGSM for RPN assignment while retaining L1 regression and IoU assignment
in the RoI head.

## Source Locks

| Source | Publication metadata | Local SHA-256 |
|---|---|---|
| IGWD | IEEE TMM, 2026, DOI `10.1109/TMM.2026.3675527` | `7268ad1ad5fe5cab058138af8dfc4a081a621da6b0ee57fc82fed1a6b25186e1` |
| SimD | IROS 2024, DOI `10.1109/IROS58592.2024.10801448` | `e40ebdaf664cea94dfb9e1f40e2c6503446045981bb3fd2417ada16f9b52613d` |
| SAFit | IEEE TPAMI, 2025, DOI `10.1109/TPAMI.2025.3544621` | `a1f75e9b121c290b4f9295eb7886b7c5d969dfd5054c7995699dc6531e5ea29e` |
| GCD | IEEE GRSL, 2025, DOI `10.1109/LGRS.2025.3531970` | `1ffb8ef13b94982cf90fc14ea99a2412664f01a65a8e2dc060f4efcf39820b95` |
| SWIN-TOD / SWL | IEEE TGRS, 2024, DOI `10.1109/TGRS.2024.3452010` | `e50532ce076be1fc9bfc0cba2b64f567160f8bc0f794bc52b86fd16484b06f7a` |
| MMPW | Remote Sensing, 2024, DOI `10.3390/rs16234485` | `2c605d8e12fc67f217c0968b4f5499f0ac69530e8d2a6f7b51aab520c15ef08f` |
| DILA code | Official repository commit `103b743b66b3cdf2e370b4a45c44cb96d07fbf65` | BGSM Git blob `918b0098fd3c6e9a12bb3824e010126a786e70ac2360bb714e15473e95cdfeac` |

Primary pages:

- https://arxiv.org/abs/2110.13389
- https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3138_ECCV_2022_paper.php
- https://arxiv.org/abs/2407.02394
- https://arxiv.org/abs/2406.14482
- https://arxiv.org/abs/2510.27649
- https://livrepository.liverpool.ac.uk/3186214/
- https://www.mdpi.com/2072-4292/16/23/4485
- https://www.sciencedirect.com/science/article/pii/S1568494624007543
- https://github.com/chnu-cpl/DILA

## Baseline Feasibility Decision

The required matched core remains standard IoU/Smooth-L1, RFLA, NWD, and the
verified direct predecessor IGWD. SimD and SAFit remain the first candidates
for an extended closest-prior matrix after their exact code paths pass an
isolation audit. SWL, MMPW, and full DILA are cited contextual priors rather
than automatic core runs because their released methods change placement,
assignment, NMS, architecture, or multiple components at once; SWL and MMPW
also lack a complete official drop-in implementation for this harness.

This audit supports only a narrow formulation-and-placement novelty statement.
It cannot support a broad first, SOTA, or universal-superiority claim.
