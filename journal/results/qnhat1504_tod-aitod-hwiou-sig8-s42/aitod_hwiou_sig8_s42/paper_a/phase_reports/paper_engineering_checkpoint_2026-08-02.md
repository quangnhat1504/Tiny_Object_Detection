# Paper Engineering Checkpoint - 2026-08-02

Status: `IN_PROGRESS; NO_TRAINING_LAUNCHED`
Code commit: `UNCOMMITTED_WORKTREE`
Paper A final-test material accesses: `1` (AI-TOD-v2 structural audit)
Paper A final-test performance evaluations: `0`
Paper A accepted result rows: `0`
Paper A local tests: `58/58 PASS`

## Completed

- Canonical ALW/SA-ALW implementation, placement audit, detector smoke, and
  protocol guards.
- Original-image reconstruction plus both public-dataset adapters and
  hash-locked official evaluator fixtures.
- TinyPerson binary task-all validation, official corner-crop handling, and
  explicit ignore/uncertain/crowd routing, including an IOD fixture under
  NumPy 2.
- Torchvision-matched detector-coordinate schedule fitter and an AI-TOD-v2
  train-only P10/P90 audit over 301,494 valid positive boxes.
- Official AI-TOD-v2 annotation acquisition and immutable split manifest.
- Result ledger schemas, accepted-row validator, three-seed headline-table
  gate, and generated CSV/LaTeX rows.
- Primary-source formula audit for NWD, RFLA, SimD, SAFit, GCD, IGWD, SWL,
  MMPW, and DILA/BGSM.
- Verified IGWD authors/venue/DOI; removed the legacy anonymous citation.
- Evidence-gated internal manuscript and supplement that compile without
  citation or cross-reference errors.
- Baseline fidelity matrix that keeps the 12-run matched core separate from
  conditional or multi-component contextual priors.
- Explicit A0-A4 data-access policy and a team sharding board that preserves
  matched dataset/seed groups while balancing predicted GPU-hours.
- Internal anonymity/PDF preflight: no checked identity/path leak, all fonts
  embedded, and clean references; venue template and Type-3-font repair remain.
- Controlled SA-ALW mechanism preflight: beta-only preserves within-target
  ranking, changes only threshold/ownership paths in assignment, and is absent
  from regression; position emphasis can change center/shape ordering.
- Six-method G3 pilot rule preregistered before launch, including beta-only and
  position-only controls and a fixed AP/tie decision rule.
- Exact AI-TOD-v2 train-anchor preflight on 64 seeded images: full SA-ALW changes
  593 assignments and reduces positives by 3.71 percent without reducing GT
  coverage; no AP conclusion is drawn.
- Reference endpoint effects and a seven-run one-axis sensitivity budget are
  preregistered; one log-linear schedule form is implemented and tested while
  linear remains the pilot default.

## Evidence Created

- `evaluation/aitodv2_official.py`
- `tests/test_aitod_official_evaluator.py`
- `evaluation/tinyperson_official.py`
- `tests/test_tinyperson_official_evaluator.py`
- `datasets/tinyperson_original.py`
- `tests/test_tinyperson_dataset_adapter.py`
- `tools/fit_train_scale_schedule.py`
- `schedules/aitodv2_train_p10_p90.json`
- `results/*.csv`
- `tools/validate_result_ledgers.py`
- `tools/build_result_tables.py`
- `phase_reports/related_work_audit.md`
- `experiments/baseline_fidelity_matrix.md`
- `manuscript/main.tex`
- `manuscript/supplementary.tex`
- `manuscript/main.pdf`
- `manuscript/supplementary.pdf`

## Experiment Boundary

No Paper A model training, pilot, benchmark, or final-test evaluation was run
or pushed during this paper-engineering checkpoint. Local work was limited to
deterministic/unit tests, synthetic evaluator fixtures, artifact validation,
and LaTeX compilation.

Every future Kaggle work package remains subject to a separate pre-run report
and explicit team-member/account assignment. A returned run is not accepted
until its post-run artifact report and independent reload audit pass.

The AI-TOD-v2 annotation package was structurally parsed before this checkpoint,
including the public test annotation file. This is one disclosed material-access
event but zero performance evaluations; no schedule or method decision may use
its statistics. TinyPerson remains literally unopened.

## Claims Enabled or Restricted

- Method and ALW property claims C001-C004 and no-extra-learnable-parameters
  claim C006 remain enabled by canonical tests and detector construction.
- Performance, mechanism, placement, generalization, and dataset claims remain
  pending.
- Broad novelty claims are further restricted: prior work already contains
  axis-specific normalization, train-derived normalization, target-size-aware
  fitness/loss, scale-invariant Gaussian assignment/regression, and ALW's
  per-axis center denominator form up to a constant factor.
- The remaining defensible novelty boundary is the exact ALW geometry plus the
  separately placed SA-ALW schedules, subject to broader source audit and
  public-benchmark evidence.
- C014 is enabled only for this narrow formulation-and-placement boundary;
  component-level priority and broad first claims remain forbidden.

## Blockers

1. AI-TOD image package is not acquired.
2. TinyPerson image/annotation package is not acquired or hashed.
3. TinyPerson train-only scale bounds and beta/position endpoints are not
   frozen; the coordinate system and AI-TOD-v2 candidate bounds are complete.
4. Mechanism diagnostics require a valid public training split.
5. No experiment owner/account has been assigned.
6. Venue template migration and the complete anonymous code audit are pending.

## Next Exact Actions

1. Finish public image acquisition/provenance without opening final tests.
2. Acquire and audit the official TinyPerson binary task-all package against
   the completed adapter/evaluator contracts.
3. Derive TinyPerson percentiles in the already frozen detector coordinate
   system, then pre-register beta/position endpoints.
4. Prepare the two matched WP01 pilot shards as self-contained Kaggle packages
   and send separate pre-run reports for user assignment; do not push before
   approval.
