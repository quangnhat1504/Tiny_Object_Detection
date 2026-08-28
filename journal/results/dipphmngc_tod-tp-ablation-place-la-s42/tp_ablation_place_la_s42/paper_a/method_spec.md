# Canonical ALW / SA-ALW Method Specification

Version: `1.0-draft`
Status: `IMPLEMENTATION_AND_DETECTOR_SMOKE_PASS; SYNTHETIC_MECHANISM_PREFLIGHT_PASS; PUBLIC_TRAIN_DIAGNOSTICS_PENDING`
Code namespace: `common.metrics.sa_alw_canonical`

## Inputs and Domain

A predicted box and target box use center-size form:

```text
p = (x_p, y_p, w_p, h_p)
t = (x_t, y_t, w_t, h_t)
```

The mathematical domain requires positive widths and heights. The numerical
implementation clamps widths and heights to `1e-6` before division and logs.
Properties below refer to the mathematical formulation before this clamp.

## Canonical ALW

```text
S_x = (w_p^2 + w_t^2) / 2
S_y = (h_p^2 + h_t^2) / 2

D_pos = (x_p - x_t)^2 / S_x + (y_p - y_t)^2 / S_y
D_shape = log(w_p / w_t)^2 + log(h_p / h_t)^2
D_ALW = D_pos + D_shape
d_ALW = sqrt(D_ALW)
K_ALW = exp(-beta * d_ALW)
```

`S_x` and `S_y` are per-axis mean-square scales. They are not RMS
denominators. The shape terms are squared log ratios, not absolute log ratios.

## Canonical SA-ALW

Target scale and clipped interpolation:

```text
s = sqrt(w_t * h_t)
u(s) = clip((s_max - s) / (s_max - s_min), 0, 1)
beta(s) = beta_min + (beta_max - beta_min) * u(s)
w_pos(s) = w_min + (w_max - w_min) * u(s)
```

The reference method uses this linear interpolation. One preregistered
validation-only alternative uses

```text
u_log(s) = clip((log(s_max) - log(s)) /
                (log(s_max) - log(s_min)), 0, 1)
```

which preserves the same clipped endpoints and places its midpoint at the
geometric mean. It is a sensitivity candidate, not part of the reference claim
unless the frozen validation selection rule promotes it before core runs.

Target-conditioned distance and similarity:

```text
D_SA = w_pos(s) * D_pos + D_shape
d_SA = sqrt(D_SA)
K_SA(p, t) = exp(-beta(s) * d_SA)
```

Because beta multiplies distance in the exponent, it is a similarity decay
rate (equivalently an inverse temperature), not a temperature. Larger beta
produces faster decay.

The implementation uses `sqrt(D + eps^2) - eps` with `eps=1e-6` to retain a
zero identity value and finite derivative at identity. This is a numerical
stabilization, not a learned or scale-adaptive component.

## Assignment Definition

Assignment consumes a pairwise similarity matrix `K(candidate_i, target_j)`.
Paper A uses the existing HLA skeleton under the same detector harness for all
metric comparisons:

```text
for each image:
    compute pairwise target-conditioned similarity
    pass 1: retain per-GT top-k candidates and quality-ratio extras
    resolve a candidate claimed by multiple GTs using its best similarity
    pass 2: contract unmatched candidate fields and repeat for unfilled quotas
```

For one fixed target, `exp(-beta(s) d)` is monotonic in `d`; beta therefore
cannot change within-GT distance ranking. It can change:

- whether non-base candidates exceed the relative quality threshold;
- best-GT ownership when candidate similarities use different target betas;
- pass-2 eligibility through the same two paths.

These effects remain claims only after
`diagnostics/assignment_change_by_scale.csv` is populated.

For the current HLA quality ratio `r=0.60`, ignoring the absolute similarity
floor, a candidate is an extra relative-threshold positive when

```text
d_i - d_best <= -log(r) / beta(s)
```

Thus changing beta from 8 to 10 narrows the admissible distance margin by 20
percent. This is a threshold calibration effect, not a ranking effect.

## Regression Definition

RoI regression decodes the positive class-specific box for each sampled
positive proposal, then compares only the aligned `(prediction_i, target_i)`
pair:

```text
L_box = mean_i d_method(prediction_i, target_i)
```

It must never construct and average an all-pairs `N x N` matrix. For canonical
SA-ALW, `w_pos(s)` affects regression but `beta(s)` does not occur in the
distance loss. Therefore:

- beta-only SA-ALW regression is exactly canonical ALW regression;
- position-only and full SA-ALW regression use the same target-conditioned
  position weight;
- beta effects may not be credited to regression.

## NMS Definition

Paper A fixes standard class-aware IoU-NMS. Canonical run guards reject
`la_loss_nms`; metric-NMS results belong only to the diagnostic ledger.

## Canonical Variants

| Registry name | Assignment | Regression |
|---|---|---|
| `alw_canonical` | fixed-beta canonical ALW similarity | aligned canonical ALW distance |
| `sa_alw_canonical_beta_only` | adaptive beta, fixed position weight | aligned canonical ALW distance |
| `sa_alw_canonical_pos_only` | fixed beta, adaptive position weight | aligned position-adaptive distance |
| `sa_alw_canonical` | adaptive beta and position weight | aligned position-adaptive distance |

The legacy names `alw_full`, `alw_original`, `sa_alw_full`,
`sa_alw_beta_only`, and `sa_alw_pos_only` are not canonical Paper A methods.
They are retained only to reconstruct historical checkpoints.

## Required Explicit Configuration

Canonical SA-ALW construction fails unless all six schedule values are supplied:

```text
s_min, s_max, beta_min, beta_max, w_min, w_max
```

The run metadata additionally records `schedule_form=linear|log_linear`; the
reference pilot freezes `linear`.

The training CLI records them under `config.metric_config` with
`schedule_source=explicit_frozen_train_config`. Values may be frozen only from
the repaired train split. Current legacy constants are not accepted as Paper A
evidence merely because they exist in `common/config.py`.

Canonical runs additionally require:

- placement in `la`, `loss`, or `la_loss`;
- box loss `metric`;
- checkpoint selector `coco_ap`;
- no CBL, refinement, distillation, quality, or PC auxiliary component;
- fixed IoU-NMS.

## Properties and Boundaries

| Property | Canonical ALW | SA-ALW |
|---|---:|---:|
| Non-negative squared distance | yes | yes |
| Identity | yes | yes |
| Symmetric | yes | no, target-conditioned |
| Dimensionless | yes | yes for fixed dimensionless schedules |
| Joint scale invariance | yes | no, schedule uses absolute target scale |
| Axis-specific normalization | yes | yes |
| Relative squared-log shape | yes | yes |
| Additional learnable parameters | none in geometry | none in geometry |

No triangle-inequality claim is made. The manuscript must call SA-ALW a
Wasserstein-inspired target-conditioned box similarity/loss, not a mathematical
metric.

## Verification

`paper_a/tests/test_alw_saalw.py` currently covers identity, non-negativity,
squared log-ratio exactness, ALW symmetry and scale invariance, intentional
SA-ALW asymmetry, schedule clipping, finite tiny-box gradients, CPU/GPU/AMP
consistency, aligned regression, assignment conflict behavior, box-code round
trip, explicit schedule enforcement, and Paper A protocol guards.

The real CUDA detector smoke passes AMP forward/backward for assignment-only,
regression-only, and joint placement. Strict reload passes and all placements
have exact parameter-count parity with the baseline. The schedule coordinate
system is now frozen to post-`GeneralizedRCNNTransform` pixels at
`min_size=640,max_size=800`; its calculation is checked directly against
torchvision. AI-TOD-v2 train-only P10/P90 candidate bounds are audited, but G1
remains open until TinyPerson bounds, beta/position endpoints, assignment-change
diagnostics, and center/shape gradient diagnostics are frozen.

## Controlled Mechanism Preflight

`diagnostics/saalw_mechanism_preflight.json` is synthetic technical evidence,
not a substitute for public-train diagnostics. It proves five code-level facts:

1. Beta-only produces zero within-target rank changes across controlled target
   scales and preserves each top-3 set.
2. At quality ratio 0.60, beta 8 versus 10 changes a controlled HLA positive
   count from four to three through the relative threshold.
3. Target-specific beta can change ownership across differently scaled GTs.
4. Position emphasis can reverse center-error versus shape-error ranking.
5. Beta-only regression is exactly canonical ALW regression.

Accordingly, the G3 pilot includes beta-only and position-only controls before
the full multi-seed matrix. C007 and C008 remain pending until the same causes
and gradients are measured on the frozen public training split.

An additional AI-TOD-v2 train-anchor preflight runs the exact 306,900-anchor,
two-pass HLA contract on 64 seeded training images. Full SA-ALW changes 593
anchor assignments relative to ALW and reduces positives from 6,899 to 6,643,
while every variant covers the same 1,816 of 1,818 GTs. Only three changes are
ownership flips. This supports a real, threshold-dominated assignment effect
without implying whether the greater selectivity improves AP.
