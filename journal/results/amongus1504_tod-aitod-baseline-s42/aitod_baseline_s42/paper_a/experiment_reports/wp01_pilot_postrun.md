# Paper A Post-Run Report: `WP01`

Status: `GATE_DECISION_FILED` (all six kernels COMPLETE and audited)

Pre-run counterpart: `wp01_pilot_prerun.md` (decision `KERNELS_RUNNING`,
with the multi-account fan-out amendment).

## Execution

- Shard ID and source work package: `PILOT-D1-S42` + `PILOT-COMP-D1-S42`
  (WP01 TinyPerson G3 reduced pilot).
- Owner/account/kernel: Qoder-Leader / five pool accounts per the fan-out
  amendment (`ngquangnht` = standard + igwd; `amongus1504` = alw_canonical;
  `qnhat1504` = sa_alw_beta_only; `thyngluthy` = sa_alw_pos_only;
  `hienquang06` = sa_alw_full) / six private kernels
  `wp01-pilot-<method>-s42`, T4, internet off, seed 42, 8 epochs, batch 4.
- Start/end time: ngquangnht kernels push ~18:50 local, COMPLETE by
  23:01 local (2026-08-04); the four foreign kernels re-pushed as version 2
  ~21:18 local after dataset `ready`; sa_alw_beta_only and sa_alw_pos_only
  COMPLETE by 2026-08-05 00:02 local; alw_canonical and sa_alw_full
  COMPLETE by 2026-08-05 07:20 local.
- Requested/observed GPU and actual GPU-hours: T4 x1 per kernel (kernels
  guard `"T4" in device name`); observed hours pending from kernel logs.
- Kaggle terminal status: `COMPLETE` all six kernels.
- Downloaded artifact directory: `.runtime/kaggle/wp01/outputs/<method>/`.
- Mount smoke predecessor: `wp01-smoke-mount` (COMPLETE, all checks pass,
  `.runtime/kaggle/wp01/smoke_output/smoke_report.json`).

## Artifact Audit

Per method, run (after `kaggle kernels output ... -p outputs/<method>`):

```powershell
python .runtime/kaggle/wp01/audit_kernel_output.py --method <method>
python .runtime/kaggle/wp01/offkernel_reload_check.py --method <method>
```

- Protocol/config hash agreement: `PASS` for all six runs — split shas
  (`5bea11d2...` train, `31d67f94...` val), trainer sha `38a89023...`,
  schedule bounds `7.4328/44.8468` (SA-ALW and IGWD runs), seed/epochs/
  batch all verified by `audit_kernel_output.py` (20/20 checks each).
- Metrics present: `PASS` for all six runs (`metrics.csv` 8 rows,
  `results.json`, `detections_best.json`, `best.pt`,
  `val_gt_paper_primary.json`).
- Failure artifact: kernel log + partial `metrics.csv` if any run dies;
  reported, never silently restarted with changed config.
- Checkpoint inventory: `PASS` for all six runs (`best.pt` sha in
  `audit.json`).
- Independent validation reload agreement: all six runs reloaded
  off-kernel with strict state-dict match and trainer sha triple-match.
  Selector AP deltas: standard `3.9e-5`, igwd `9.6e-6`, sa_alw_beta_only
  `1.5e-4`, sa_alw_pos_only `3.7e-5`, alw_canonical `6.4e-5`,
  sa_alw_full `1.0e-4` (all within 5e-4).
  Primary official endpoints (`AP25/AP50/AP75_all`) reproduce within
  `1.4e-4`. A small number of low-count secondary buckets exceed the 5e-4
  tolerance (max observed `2.6e-3` on `AR75_reasonable` standard), which is
  the disclosed cuDNN non-determinism at tight IoU match boundaries; the
  strict boolean is `official_within_tolerance=false` for these runs and
  is disclosed here rather than treated as an artifact-integrity failure.
- Final-test material-access count: `0`.
- Final-test performance-evaluation count: `0`.

## Results

- Primary metric: validation `paper_primary_coco` AP (best checkpoint,
  independent reload).
- Secondary metrics: `benchmark_official` AP25/AP50/AP75 + tiny bins
  (pinned TinyPerson evaluator, sha lock per `tinyperson_official.py`).
- Per-seed rows: seed 42 only (pilot scope; matched matrix seeds
  42/123/2024 belong to WP02/WP03, not this report).
- Mean/std or paired CI: not applicable at pilot scale; the decision uses
  the frozen selection rule, not CIs.

| method | best ep | selector AP | AP50 official | AP75 official |
|--------|---------|-------------|---------------|---------------|
| standard | 4 | 0.16135 | 0.4535 | 0.0768 |
| igwd | 7 | 0.14884 | 0.4280 | 0.0676 |
| alw_canonical | 6 | 0.15461 | 0.4382 | 0.0682 |
| sa_alw_beta_only | 6 | 0.15337 | 0.4308 | 0.0715 |
| sa_alw_pos_only | 6 | 0.15315 | 0.4306 | 0.0694 |
| sa_alw_full | 6 | 0.15635 | 0.4351 | 0.0708 |

## Gate Decision

Applied the frozen rule in `paper_a/experiments/pilot_decision_protocol.md`
to the audited selector AP column (independently reloaded checkpoints):

1. Full vs components: full `0.15635` vs pos-only `0.15315` (+0.00320) and
   beta-only `0.15337` (+0.00298) — full exceeds both components by more
   than `0.001`, so the two-schedule method is retained as the SA
   representative.
2. No component exceeds full, so the return-to-G0/G1 branch is not
   triggered.
3. G3 `GO` test: the selected variant (full) must exceed BOTH the direct
   predecessor (igwd `0.14884`) and canonical ALW (`0.15461`). Full beats
   igwd by `+0.00751` and canonical ALW by `+0.00174`; training was finite
   and the mechanism effect is audited and nonzero (rule 1 margins).

- Decision: **`GO` (WP01 pilot gate passed)** at seed-42 reduced budget —
  the two-schedule full SA-ALW exceeds both references, so it is the
  formulation selected for the matched three-seed matrix. Note the margin
  over canonical ALW is thin (`+0.00174`, below the pilot's own `0.001`
  component-separation threshold only in magnitude terms; the frozen rule
  sets no minimum margin for the reference comparison), and the plain
  standard baseline still leads every ALW-family variant
  (`0.16135` vs `0.15635`); both facts are disclosed and belong to the
  matched-matrix and paper framing, not to this gate rule.
- Exact failed/passed criteria: rule 1 passed (full retained over both
  components by > 0.001); rule 2 not triggered; rule 3 passed (full beats
  igwd and alw_canonical). Observed ranking: standard `0.16135` >
  sa_alw_full `0.15635` > alw_canonical `0.15461` > sa_alw_beta_only
  `0.15337` > sa_alw_pos_only `0.15315` > igwd `0.14884`.
- Converge caveat (disclosure, not a rule override): every ALW-family
  variant selected epochs 6-7 of 8, i.e. still improving at budget end,
  while standard peaked at epoch 4; the reduced 8-epoch budget may
  understate methods with slower warmup. This observation is recorded for
  governance review only — the preregistered rule decides as written.
- Claims enabled/disabled: no performance claim enabled by the pilot
  alone; the gate unblocks the matched WP02-WP05 matrix (each work package
  still requires its own pre-run report). The matched matrix must include
  the standard baseline so the baseline-leading observation is resolved
  with three seeds rather than one.
- Ledger/table files updated: pending acceptance bookkeeping; the six
  audited rows qualify as accepted pilot evidence (NO-GO branch not
  taken), to be recorded in `evidence_ledger.csv` under the pilot shard
  IDs once the user confirms this gate disposition.
