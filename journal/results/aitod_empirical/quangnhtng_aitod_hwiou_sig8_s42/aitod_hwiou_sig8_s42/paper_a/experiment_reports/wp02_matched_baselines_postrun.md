# Paper A Post-Run Report: `WP02` Matched Baselines

Status: `VALIDATION_EVIDENCE_ACCEPTED` (12/12 artifacts and independent
reloads audited; comparative formulation decision remains pending WP03)

Pre-run counterpart: `wp02_matched_baselines_prerun.md` (decision `APPROVED`).

## Execution

- Shard ID and source work package: `MATCHED-D1` (WP02 TinyPerson matched
  baselines matrix).
- Owner/account/kernel: Qoder-Leader / four pool accounts
  (`ngquangnht` = standard s42 + rfla s42; `amongus1504` = nwd s42 + igwd s42
  + standard s123 + rfla s2024; `thyngluthy` = nwd s123 + igwd s123 + nwd
  s2024 + igwd s2024; `hienquang06` = rfla s123 + standard s2024) / twelve
  private kernels `wp02-pilot-<method>-s<seed>`, T4, internet off, 8 epochs,
  batch 4.
- Start/end time: kernels pushed 2026-08-05 ~23:00 local; first 6 COMPLETE
  by 2026-08-06 00:12; 6 re-pushed (P100 → T4) as version 2; all 12
  COMPLETE by 2026-08-06 09:45.
- Requested/observed GPU: T4 x1 per kernel (kernels guard `"T4" in device
  name`); 6 initial kernels got P100 and were re-pushed.
- Kaggle terminal status: `COMPLETE` all twelve kernels.
- Downloaded artifact directory: `.runtime/kaggle/wp02/outputs/<method>_<seed>/`.

## Artifact Audit

Per method-seed, run (after `kaggle kernels output ... -p outputs/<name>`):

```powershell
python .runtime/kaggle/wp01/audit_kernel_output.py --method <method>
python .runtime/kaggle/wp01/offkernel_reload_check.py --method <method>
```

- Protocol/config hash agreement: `PASS` for all twelve runs. The accepted
  rows and run manifests record the frozen PL-001 split hash, per-run config
  hash, trainer SHA-256, and best-checkpoint SHA-256.
- Metrics present: `PASS` for all twelve runs (`metrics.csv` 8 rows,
  `results.json`, `detections_best.json`, `best.pt`,
  `val_gt_paper_primary.json`).
- Checkpoint inventory: `PASS` for all twelve runs.
- Independent validation reload: `PASS` for all twelve runs. The primary
  official endpoints are within tolerance; secondary low-count bucket drift
  is retained as a cuDNN disclosure rather than an integrity failure.
- Final-test material-access count: `0`.
- Final-test performance-evaluation count: `0`.
- Result ledger: 12 `ACCEPTED` validation-evidence rows in
  `paper_a/results/main_results.csv`, each with a matching
  `paper_a/runs/manifest.jsonl` entry. This is not final-test or submission
  evidence.

## Results

- Primary metric: validation `paper_primary_coco` AP (best checkpoint,
  independent reload).
- Secondary metrics: `benchmark_official` AP25/AP50/AP75 + tiny bins
  (pinned TinyPerson evaluator).
- Per-seed rows: seeds 42/123/2024 (full matched matrix).

### Full 12-Kernel Matched Baseline Table

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

### Summary Statistics (mean ± std across 3 seeds)

| method   | mean AP    | std AP   | mean AP50 | mean AP75 |
|----------|------------|----------|-----------|-----------|
| **rfla** | **0.15857**| 0.00138  | **0.4459**| **0.0749**|
| standard | 0.15557    | 0.00264  | 0.4387    | 0.0711    |
| igwd     | 0.14947    | 0.00078  | 0.4252    | 0.0666    |
| nwd      | 0.14800    | 0.00182  | 0.4123    | 0.0680    |

## Key Observations

1. **RFLA leads** all baselines with mean AP 0.15857 ± 0.00138, exceeding
   standard by +0.00300. RFLA also has the lowest variance across seeds.
2. **Standard** is second with mean AP 0.15557 ± 0.00264.
3. **IGWD** (third, 0.14947) and **NWD** (fourth, 0.14800) are below
   standard, with NWD being the weakest baseline.
4. **Seed variance**: All methods show low seed variance (std < 0.003),
   confirming training stability.
5. **Best epoch trends**: RFLA peaks early (ep 4-5), while NWD/IGWD peak
   late (ep 7-8). Standard peaks at ep 4-7.

## Comparison with WP01 Pilot (seed 42 only)

| method   | WP01 AP   | WP02 AP   | Delta   |
|----------|-----------|-----------|---------|
| standard | 0.16135   | 0.15862   | -0.00273|
| igwd     | 0.14884   | 0.14913   | +0.00029|

The small delta for standard (-0.00273) is attributable to the extended
trainer (WP02 trainer includes RFLA/NWD method configs, hash `7c05831c...`
vs WP01 trainer hash `38a89023...`). The igwd delta is negligible
(+0.00029), confirming fidelity.

## Gate Decision

The WP02 baseline rows are accepted as **validation evidence** and will serve
as the matched reference for WP03. No SA-ALW formulation or final-test decision
is made here; that gate remains pending the four valid WP03 v8 artifacts and
their independent reloads.

## Next Actions

1. Complete WP03 execution (mount-validated v8 proposed-method matrix).
2. Download/audit each WP03 artifact and run independent reload.
3. File the matched formulation gate after all four valid WP03 results are
   available.
