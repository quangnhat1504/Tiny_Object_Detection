# Paper A: Comprehensive Results Summary

**Status**: WP01 COMPLETE | WP02 COMPLETE | WP03 A1/A2/A3/A4 COMPLETE |
`PAPER_A PERFORMANCE NO-GO; FINAL-TEST PERFORMANCE ACCESS = 0`

**Date**: 2026-08-14

---

## WP01: Pilot (seed 42 only)

Six methods tested on TinyPerson validation set (seed 42, 8 epochs):

| method           | best ep | selector AP | AP50 official | AP75 official |
|------------------|---------|-------------|---------------|---------------|
| **standard**     | 4       | **0.16135** | **0.4535**    | **0.0768**    |
| sa_alw_full      | 6       | 0.15635     | 0.4351        | 0.0708        |
| alw_canonical    | 6       | 0.15461     | 0.4382        | 0.0682        |
| sa_alw_beta_only | 6       | 0.15337     | 0.4308        | 0.0715        |
| sa_alw_pos_only  | 6       | 0.15315     | 0.4306        | 0.0694        |
| igwd             | 7       | 0.14884     | 0.4280        | 0.0676        |

**Gate decision**: GO for full SA-ALW (sa_alw_full exceeds both components
by >0.001, beats igwd by +0.00751, beats canonical ALW by +0.00174).

---

## WP02: Matched Baselines (4 methods x 3 seeds = 12 kernels)

Full matched matrix across seeds 42/123/2024:

| method   | seed | best ep | selector AP | AP50 official | AP75 official |
|----------|------|---------|-------------|---------------|---------------|
| standard | 42   | 4       | 0.15862     | 0.4450        | 0.0733        |
| standard | 123  | 7       | 0.15416     | 0.4307        | 0.0703        |
| standard | 2024 | 7       | 0.15394     | 0.4403        | 0.0696        |
| rfla     | 42   | 4       | 0.15908     | 0.4534        | 0.0730        |
| rfla     | 123  | 5       | 0.15961     | 0.4434        | 0.0776        |
| rfla     | 2024 | 7       | 0.15701     | 0.4410        | 0.0741        |
| nwd      | 42   | 7       | 0.14594     | 0.4123        | 0.0669        |
| nwd      | 123  | 8       | 0.14870     | 0.4152        | 0.0672        |
| nwd      | 2024 | 8       | 0.14937     | 0.4095        | 0.0698        |
| igwd     | 42   | 5       | 0.14913     | 0.4286        | 0.0683        |
| igwd     | 123  | 8       | 0.14891     | 0.4238        | 0.0663        |
| igwd     | 2024 | 8       | 0.15036     | 0.4234        | 0.0653        |

### Summary Statistics (mean +/- std across 3 seeds)

| method   | mean AP    | std AP   | mean AP50 | mean AP75 |
|----------|------------|----------|-----------|-----------|
| **rfla** | **0.15857**| 0.00138  | **0.4459**| **0.0749**|
| standard | 0.15557    | 0.00264  | 0.4387    | 0.0711    |
| igwd     | 0.14947    | 0.00078  | 0.4252    | 0.0666    |
| nwd      | 0.14800    | 0.00182  | 0.4123    | 0.0680    |

**Key findings**:
- RFLA leads all baselines with mean AP 0.15857 +/- 0.00138
- RFLA has the lowest seed variance (most stable)
- Standard is second with mean AP 0.15557 +/- 0.00264
- IGWD and NWD are below standard
- All methods show low seed variance (std < 0.003)
- All 12 downloaded WP02 artifacts passed the artifact audit and independent
  CUDA checkpoint reload. The primary official endpoints are within the
  declared tolerance for every run; only secondary low-count buckets show the
  disclosed cuDNN evaluation drift.

---

## WP03/A4: Complete Matched Proposed-Method Matrix and Decision

A1 verified canonical execution, A2 froze PL-003, and both owner-approved A3
seed-42 artifacts passed package audit plus independent CUDA reload. Combined
with the four accepted v8 rows, canonical ALW and full SA-ALW now each have the
matched seeds `42/123/2024` under the WP02 trainer.

Canonical ALW reaches AP/AP50/AP75 mean `0.156406/0.430988/0.069519`; full
SA-ALW reaches `0.155120/0.426332/0.071202`. The direct SA-ALW-minus-ALW mean
deltas are therefore `-0.001286/-0.004656/+0.001684`.

The 2,000-replicate paired original-image bootstrap gives AP 95% CI
`[-0.002939,+0.001277]`, AP50 `[-0.009807,+0.000785]`, and AP75
`[-0.001778,+0.004946]`. The primary AP mean is negative and its interval does
not support a positive effect, directly triggering the preregistered A4
`NO-GO`. Performance claims C009/C010 are disabled; WP04-WP07 and external
Paper A training are closed. Negative evidence is preserved in
`wp03_a4_no_go_closeout_2026-08-14.md` and the result ledgers. Final-test
performance access remains zero.

---

## Combined Paper A Results (WP01 + WP02 seed 42 comparison)

For seed 42 (the only seed with both WP01 and WP02 data):

| method   | WP01 AP   | WP02 AP   | Delta   |
|----------|-----------|-----------|---------|
| standard | 0.16135   | 0.15862   | -0.00273|
| igwd     | 0.14884   | 0.14913   | +0.00029|

The small delta for standard (-0.00273) is attributable to the extended
trainer (WP02 trainer includes RFLA/NWD method configs, hash `7c05831c...`
vs WP01 trainer hash `38a89023...`). The igwd delta is negligible
(+0.00029), confirming training fidelity.

---

## Next Steps

1. Freeze Paper A from new performance training; retain all accepted and
   negative artifacts, evaluator code, and protocol history.
2. Start only Program B B0: read-only recovery of stale CBL/PC fair-20 cloud
   artifacts, one Kaggle credential/account at a time and without new pushes.
3. Require a separate owner-approved Program B protocol before any new
   training or external evaluation.

---

## Technical Notes

- **Trainer hash**: `7c05831cbc544b84926694ecdd85159a9ac85ee557a7dc6894bebcfaed2b5d03`
- **Dataset**: TinyPerson WP01 A1 (pinned)
- **Code dataset**: Paper A Code WP02 (extended trainer)
- **GPU**: exact training mounts were T4-verified; the eight pool health
  smokes reported T4 x2
- **Epochs**: 8 per kernel
- **Batch size**: 4
- **Internet**: Off (offline training)
- **Accounts exercised**: 12 total: four v8 training accounts plus eight
  independently verified health-smoke accounts
