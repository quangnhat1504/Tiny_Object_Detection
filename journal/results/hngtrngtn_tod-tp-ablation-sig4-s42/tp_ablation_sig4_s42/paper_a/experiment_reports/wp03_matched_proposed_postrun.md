# Paper A Post-Run Report: `WP03` Matched Proposed-Method Matrix

Status: `REQUIRES_TRIAGE; NO_ACCEPTED_WP03_ROWS`

Pre-run counterpart: `wp03_matched_proposed_prerun.md` (`APPROVED`). This
report is a gate scaffold only: no metric, comparison, or claim may be entered
until a downloaded v8 artifact and its independent reload both pass.

## Execution

- Shard ID and source work package: `CORE-D1-S123`, `CORE-D1-S2024`; WP03.
- Owner/account/kernel:

  | account | method | seed | kernel |
  |---|---|---:|---|
  | ngquangnht | alw_canonical | 123 | `wp03-v8-alw-canonical-s123` |
  | amongus1504 | alw_canonical | 2024 | `wp03-v8-alw-canonical-s2024` |
  | thyngluthy | sa_alw_full | 123 | `wp03-v8-sa-alw-full-s123` |
  | hienquang06 | sa_alw_full | 2024 | `wp03-v8-sa-alw-full-s2024` |

- Requested GPU: `NvidiaTeslaT4`; exact account/data/code mount smoke was
  downloaded and verified before these v8 training pushes.
- Kaggle terminal status: all four v8 kernels were `COMPLETE` at monitor poll
  11 (`2026-08-10T00:37:09+07:00`).
- Downloaded artifact directories:
  `.runtime/kaggle/wp03/v8/outputs/<method>_s<seed>/`.
- Final-test material: not mounted; expected access count remains `0`.

## Artifact Audit Gate

Every row requires all of the following before it can be registered:

1. Kaggle terminal state is `COMPLETE` or `ERROR` has been triaged with its
   downloaded log; terminal state alone is not acceptance.
2. `python .runtime/kaggle/wp03/audit_v8_output.py --method <method> --seed
   <seed> --output <downloaded-output>` passes the v8 package, frozen trainer,
   split, eight-epoch, metrics, checkpoint, detection, and validation-only
   checks.
3. `.venv-cuda\Scripts\python.exe
   .runtime/kaggle/wp03/offkernel_v8_reload_check.py --method <method> --seed
   <seed> --output <downloaded-output>` passes strict checkpoint reload,
   selector-AP tolerance, trainer hash, and primary official endpoint
   tolerance.
4. The downloaded `config.json`, `results.json`, checkpoint SHA-256, audit,
   reload record, and manifest/ledger row agree. Secondary low-count cuDNN
   drift is disclosed separately; it cannot replace the primary endpoint gate.

| method | seed | API status | artifact audit | off-kernel reload | ledger row |
|---|---:|---|---|---|---|
| alw_canonical | 123 | COMPLETE | pass | **fail**: primary delta `0.001845`; repeat `0.001811` | withheld |
| alw_canonical | 2024 | COMPLETE | pass | **fail**: primary delta `0.002059`; repeat `0.002091` | withheld |
| sa_alw_full | 123 | COMPLETE | pass | pass; primary delta `0.000058` | withheld pending matched gate |
| sa_alw_full | 2024 | COMPLETE | pass | pass; primary delta `0.000097` | withheld pending matched gate |

## Results

- Primary metric: validation `paper_primary_coco` AP from the independently
  reloaded best checkpoint. The two SA-ALW rows whose reloads passed report
  AP=`0.155938` (seed 123) and AP=`0.156958` (seed 2024). These are audit
  observations, not an accepted matched-method claim.
- The two ALW artifacts have matching trainer SHA-256, strict checkpoint
  reload, and selector-AP tolerance, but both fail the primary official
  endpoint gate. The failure repeats on a fresh CUDA reload and is driven by
  `AP50_all` drift (seed 2024 repeat delta `0.002091`), not by a transient
  download, mount, or checkpoint mismatch.
- Secondary low-count drift remains disclosure-only. It cannot waive the
  failed primary endpoint gate.
- Seed 42 is audited WP01 companion evidence; do not copy it into this WP03
  ledger. No mean/std, paired comparison, or promotion statement is valid.

## Gate Decision

- Decision: `REQUIRES_ALW_PRIMARY_RELOAD_TRIAGE; NO_PROMOTION`.
- Claims enabled: none.
- Claims disabled: any statement that full SA-ALW beats canonical ALW or the
  matched baselines; any final-test or submission claim.
- Ledger/table action: do not append WP03 rows or regenerate matched Paper A
  tables until the two ALW primary-official reproducibility failures are
  explained and pass the locked gate on fresh evidence.

## A2 Re-adjudication Addendum (FROZEN, 2026-08-14)

The project owner accepted `PL-003`, a narrowly scoped superseding
reproducibility disposition. The four downloaded v12 Tesla-T4 diagnostics are
accepted as platform evidence for the two immutable ALW v8 checkpoints:
seed 123 manifest `b1fe988f...b28cd2` and seed 2024 manifest
`806bffbe...acd122`. Every saved-detection replay is exact; regenerated
primary deltas are at most `5.0622038e-7`, within the unchanged `5e-4` gate.

This addendum does not erase the original local failures, alter a metric,
retrain a model, or make a final-test/submission claim. It authorizes the four
v8 proposed-method rows as validation evidence only and unblocks the separate
two-shard seed-42 fill protocol. See
`wp03_a2_re_adjudication_proposal_2026-08-14.md` and `PL-003`.
