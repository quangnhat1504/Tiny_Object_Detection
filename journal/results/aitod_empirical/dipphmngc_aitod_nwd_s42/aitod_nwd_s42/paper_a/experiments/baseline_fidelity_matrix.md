# Baseline Fidelity Matrix

Status: `PREREGISTRATION_DRAFT; NO_RUN_AUTHORIZED`

## Required Matched Core

| Method | Role | Integration rule | Seeds |
|---|---|---|---|
| Standard IoU/Smooth-L1 | Strong standard baseline | Frozen detector and conventional assignment/regression | 42/123/2024 |
| RFLA | Assignment-system baseline | Faithful receptive-field distance and HLA | 42/123/2024 |
| NWD | Established Gaussian baseline | Published global constant; fixed IoU-NMS | 42/123/2024 |
| IGWD | Direct formula predecessor | Verified formula, assignment and aligned loss; fixed IoU-NMS | 42/123/2024 |

These twelve runs are WP02. They share dataset copy, backbone, augmentation,
optimizer, epoch budget, checkpoint selector, reconstruction, evaluator, and
seed set. Only the registered metric/assignment/loss component may differ.

## Extended Closest-Prior Candidates

| Method | Relevance | Gate before any run |
|---|---|---|
| SimD | Per-axis and train-derived normalization | Reproduce official formula and isolate its assignment role without silently retaining metric-NMS |
| SAFit | Target-area-aware box fitness/loss | Verify the exact target-area switch and detector placement in a compatible code path |

If either implementation audit passes, its run package receives its own
pre-run report and team/account assignment. A seed-42 validation pilot precedes
any three-seed promotion; these jobs are not included in WP02's current count.

## Contextual Priors, Not Default Core Runs

| Method | Reason |
|---|---|
| SWL | Regression-only piecewise Wasserstein loss; released config references an implementation absent from the audited repository |
| MMPW | Jointly changes assignment, loss, NMS, and network; no official drop-in code is linked |
| DILA | Full method combines dynamic priors, BGSM assignment, and feature imitation; BGSM-only would be an ablation rather than full DILA |

Published results may be discussed with their original protocol labels, but
must never be inserted into the matched result ledger. Implementing any of
these methods later creates a separately assigned Kaggle work package.
