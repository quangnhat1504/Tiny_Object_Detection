# Paper A Resume Checkpoint - 2026-08-02

Status: `PAUSED_CLEANLY; READY_TO_RESUME`
Worktree: `UNCOMMITTED_WITH_PREEXISTING_AND_PAPER_A_CHANGES`
Paper A training runs launched: `0`
Paper A accepted result rows: `0`
Final-test material accesses: `1` disclosed AI-TOD-v2 structural audit
Final-test performance evaluations: `0`

## Verified Gate State

- G0: `PASS`; 21 claims and 47 evidence families after this checkpoint.
- G1: `REVISE`; canonical code, mechanism preflights, endpoint protocol, and
  local detector smokes pass, but TinyPerson train diagnostics/bounds remain.
- G2: `REVISE`; both dataset adapters and official evaluator fixtures pass.
  AI-TOD-v2 images and the TinyPerson package remain unacquired.
- G3-G6: not opened.
- Local Paper A suite: `58/58 PASS`.
- Official evaluator source locks: `4/4 PASS`.
- Team shard validator: `15` rows pass; `14` training shards are unassigned.
- Result ledger/table pipeline: pass with zero accepted rows.

## Method Checkpoint

1. Canonical ALW/SA-ALW math, aligned regression, and HLA placement are frozen.
2. Beta is correctly described as a decay rate/inverse temperature. It cannot
   change within-GT ranking and is absent from regression.
3. Synthetic preflight shows beta affects threshold eligibility and rare
   cross-scale ownership; position emphasis can change center/shape ranking.
4. On 64 seeded AI-TOD-v2 train images (1,818 GT, 306,900 anchors/image), full
   SA-ALW changes 593 assignments and reduces positives `6,899 -> 6,643` while
   preserving identical `1,816/1,818` GT coverage. This is not AP evidence.
5. Reference pilot schedule is preregistered as train P10/P90, beta `8 -> 10`,
   position weight `1 -> 1.5`, linear interpolation.
6. Sensitivity is capped at seven one-axis validation runs. Log-linear is the
   only smooth alternative; it passes formula tests, AMP forward/backward for
   assignment/regression/joint placement, parameter parity, and strict reload.

## Frozen Experiment Handoff

G3 uses six reduced-schedule seed-42 methods:

- standard IoU/Smooth-L1;
- verified direct predecessor;
- pure canonical ALW;
- beta-only SA-ALW;
- position-only SA-ALW;
- full SA-ALW.

They are split into `PILOT-D1-S42` and `PILOT-COMP-D1-S42`, but both must use
the same owner/account/hardware/hash contract. Neither shard is ready or
assigned. Before any push, send its separate pre-run report to the user; after
completion, send a separate downloaded-artifact/reload report.

## Exact Resume Order

1. Acquire only the allowed TinyPerson train/validation material and record
   archive, license, split, and image hashes without opening final-test metrics.
2. Run the existing adapter/evaluator fixtures against that package.
3. Derive TinyPerson train-only P10/P90 bounds in detector coordinates.
4. Repeat the exact anchor-assignment audit and add center/shape gradient
   diagnostics on TinyPerson train data.
5. Freeze G1/G2 configs, build self-contained Kaggle pilot notebooks, run local
   technical smokes, and prepare the two separate pre-run reports.
6. Wait for explicit user owner/account assignment before pushing either shard.

## Resume Artifacts

- `method_spec.md`
- `schedules/endpoint_protocol.md`
- `diagnostics/saalw_mechanism_preflight.json`
- `diagnostics/aitodv2_anchor_assignment_preflight.json`
- `phase_reports/aitodv2_anchor_assignment_preflight_2026-08-02.md`
- `diagnostics/canonical_detector_smoke_log_linear_seed42.json`
- `experiments/pilot_decision_protocol.md`
- `experiments/team_run_shards.csv`
- `data_access_policy.md`
- `test_access_log.md`

## Hard Boundaries on Resume

- Do not use legacy SOD test results as Paper A evidence.
- Do not mount final-test material in pilot/core/sensitivity packages.
- Do not launch a Paper A training run without its separate pre-run report and
  explicit team-member/account assignment.
- Do not infer AP from anchor counts or synthetic mechanism probes.
- Do not expand the sensitivity grid beyond the seven preregistered one-axis
  runs without returning to the protocol ledger first.
