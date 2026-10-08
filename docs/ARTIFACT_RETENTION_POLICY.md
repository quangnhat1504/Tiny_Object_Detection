# Artifact Retention Policy

## Protected material

Keep EH-WIoU work on Seed 42 and treat official final-test surfaces as closed.
Preserve tracked source, private data, locked manifests, accepted evidence, and
user edits. Review untracked or user-owned generated files before applying
cleanup. The generic hygiene scan must prune these paths before traversing them:

- `.git/`, `.archive/`, `.venv*/`, `paper_a/`, `data/`, and `raw/`;
- `.runtime/` and `.worktrees/`;
- `journal/test_raw/`, `journal/audits/`, `journal/results/`,
  `journal/manuscript/`, and `journal/wiki/`;
- `archive/` and `reproducibility/H-WIoU/`.

The tool also skips symlinks and filesystem reparse points during discovery and
checks candidates again before applying cleanup. Do not inspect protected
contents as part of routine cleanup.

## Retention classes

### Keep

- Current source, tests, locked manifests, run configuration, metrics,
  detections, execution logs, independent replay, and acceptance records.
- Every checkpoint until a selected checkpoint from the same run is verified.
- Historical records that explain a result or preserve recovery information.

### Archive

Archive small historical code, notebooks, and documents only when they have
recovery or explanatory value. Each batch needs a dated index with its origin,
reason, and evidence status. Keep durable lessons in the audit log or wiki;
`.archive/` is local and ignored.

### Delete candidates

The tool may identify bytecode, test caches, empty generated logs, interrupted
temporary files, zero-byte checkpoints, and redundant `last.pt` or epoch
snapshots. Apply mode validates checkpoint structure and removes a checkpoint
group only when a structurally valid survivor from that run remains. It does not
deserialize legacy checkpoint formats that cannot be checked safely. Never
remove execution logs that support accepted evidence.

## Required workflow

Start with the read-only scan and review every candidate. Supported categories
are `all`, `checkpoints`, `caches`, and `logs`. Apply cleanup only after
explicit authorization for the specific targets and category.

```powershell
python scripts\maintenance\audit_repository_hygiene.py `
  --report .runtime\repository_hygiene_dry_run_YYYYMMDDTHHMMSSZ.json
```

Use a fresh report path. Reports are create-only when a run begins. Keep them
inside the repository and outside protected paths; `.runtime/` is the preferred
receipt location. Do not write reports under `journal/audits/`, which the tool
protects. Apply mode reserves an `IN_PROGRESS` receipt before scanning and
records each candidate as `PENDING`, `IN_PROGRESS`, and its final outcome.
Review any interrupted path manually; the tool does not resume cleanup.

A dry-run after cleanup should show no remaining allow-listed candidates.
Never use a broad wildcard to delete model weights. A cleanup receipt records
progress but does not make deletion reversible.
