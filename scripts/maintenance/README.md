# Repository maintenance

This folder holds local repository-maintenance tools.

## Hygiene audit

The hygiene auditor lists only allow-listed cache, log, and checkpoint
candidates. Its default mode is read-only. Read the
[artifact retention policy](../../docs/ARTIFACT_RETENTION_POLICY.md) before
using `--apply`; review every candidate and use a fresh report path. Apply
reports retain per-path progress so interrupted operations can be reviewed.
Write reports under `.runtime/`; the tool rejects report destinations inside
protected evidence and data paths.

Replace the timestamp placeholder with a fresh UTC timestamp, then run a
read-only scan:

```powershell
python scripts\maintenance\audit_repository_hygiene.py `
  --report .runtime\repository_hygiene_dry_run_YYYYMMDDTHHMMSSZ.json
```

The former command path
[scripts/audit_repository_hygiene.py](../audit_repository_hygiene.py)
remains a compatibility entry point. The tool verifies the Git top-level path
and supports linked worktrees, where `.git` is a file; it refuses nested paths
that are not the repository root. New documentation should use the grouped
implementation path.
