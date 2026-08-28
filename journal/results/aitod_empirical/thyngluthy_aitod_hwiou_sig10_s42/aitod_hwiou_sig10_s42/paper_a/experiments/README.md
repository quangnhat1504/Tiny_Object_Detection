# Paper A Experiment Packages

`assignment_board.csv` is the workload-level handoff for the team. It is not a
result ledger and does not authorize a run by itself.

State transitions:

```text
BLOCKED -> PREP -> SMOKE_PASS -> REPORTED_FOR_ASSIGNMENT -> READY_FOR_PUSH
        -> RUNNING -> ARTIFACT_AUDIT -> ACCEPTED | REJECTED | FAILED
```

The user assigns `owner` and `kaggle_account`. Before `READY_FOR_PUSH`, create a
pre-run report from `../experiment_reports/templates/pre_run.md` and report it
separately to the user. After completion, use the post-run template and update
`../runs/manifest.jsonl` plus the evidence/result ledgers.

`team_run_shards.csv` is the assignment-level board. It keeps every matched
dataset/seed comparison atomic and balances people by predicted GPU-hours after
the local smoke. `team_run_sharding.md` defines the rules. The broader
`assignment_board.csv` remains the scientific work-package map.

Validate the assignment board before every state transition:

```powershell
python paper_a/tools/validate_team_shards.py
```

Once an assignment wave is fully populated, add `--enforce-balance`.
