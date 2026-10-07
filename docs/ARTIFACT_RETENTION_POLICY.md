# Artifact Retention Policy

## Governing rules

1. Preserve the Seed-42-only experiment policy and keep the official final test
   closed.
2. Preserve Git-tracked source, raw data, locked manifests, ledgers, and user
   edits during automated cleanup.
3. Distinguish Kaggle API state, downloaded-artifact state, and verification
   state. A `COMPLETE` kernel is not accepted evidence by itself.
4. Never use a broad wildcard to delete model weights.
5. Hygiene scans must prune `paper_a/`, `journal/test_raw/`, `.runtime/`, and
   `.worktrees/` before walking them. Do not inspect, hash, list, move, or
   delete their contents as part of routine repository cleanup.

## Retention classes

### Keep canonical

- Current source and tests.
- Locked run/data/evaluator/source hashes.
- `run_config.json`, `metrics.json`, `detections_best.json`, execution log,
  independent replay report, and acceptance/post-run report.
- Before acceptance: the complete required output bundle, including raw and
  structured best/final checkpoint pairs.
- After acceptance: at least the verified selected raw and structured
  checkpoints. Final/resume checkpoints may be removed only after the post-run
  report records their hashes and a recoverable remote copy is confirmed.

### Archive

Archive small historical notebooks, code snapshots, and documents only when
they retain explanatory or recovery value. Every archive batch needs an index
with its date, origin, reason, and evidence status. `.archive/` is local and
ignored; durable lessons belong in `journal/audits/` or `journal/wiki/`.

### Delete

- Zero-byte or structurally corrupt checkpoints after recording the failure.
- Redundant `last.pt` and `epoch_N.pt` files only when a selected checkpoint in
  the same run survives structural validation.
- Downloadable torchvision backbone caches copied into artifact bundles.
- Python bytecode, test cache, empty generated logs, and interrupted checkpoint
  temporary files.
- Obsolete staging/data replicas whose canonical source is retained.
- Diagnostic-only smoke weights after compact config, metrics, and lesson
  records have been preserved.

Execution logs supporting accepted evidence are not generic cleanup targets.
Generated LaTeX logs are disposable; result logs are evidence until an audit
explicitly says otherwise.

## Required workflow

Run a dry audit first, inspect representative targets, then apply one category
at a time:

```powershell
.\.venv-cuda\Scripts\python.exe scripts\audit_repository_hygiene.py `
  --report .runtime\repository_hygiene_dry_run.json

.\.venv-cuda\Scripts\python.exe scripts\audit_repository_hygiene.py `
  --apply --category checkpoints `
  --report journal\audits\repository_hygiene_checkpoints.json
```

Supported categories are `checkpoints`, `caches`, and `logs`. `.runtime/` is
excluded because it can contain credentials, active workloads, and private
data. `.worktrees/` is also excluded because it can contain user-owned changes.
Review a runtime replica only through a dedicated, path-specific audit.
After cleanup, dry mode must return zero allow-listed candidates. Run the
contract audit defined by the active checkout and record any unavailable gate;
do not report an unavailable check as passed.
