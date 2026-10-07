# Wiki and Repository Hygiene Audit — 2026-10-07

## Branch scope

This update is based on `origin/main` at `983e87b`. That tree had no top-level
`wiki/` directory and two Journal wiki files. The former overview listed
source paths that are not present on this branch, so it was moved unchanged to
`archive/legacy-journal-wiki-20261007/`. The replacement overview states the
branch and evidence limits without carrying forward unsupported results.

The wider local wiki snapshot was not copied here. Its source metadata refers
to code, audits, figures, and results that are not present in this branch.
Copying only the pages would leave unsupported claims and broken provenance.
The full research-state refresh needs reconciliation against a branch carrying
the matching source and audit records.

## Cleanup controls

- Added `scripts/audit_repository_hygiene.py`; its default mode is read-only.
- The walker prunes protected and private paths before traversal. The cleanup
  allow-list is limited to reproducible caches, safe generated logs, and
  redundant checkpoints with a validated selected checkpoint beside them.
- Added `docs/ARTIFACT_RETENTION_POLICY.md` to define retention and cleanup
  gates.
- No training, inference, dataset access, or cloud operation was performed
  for this branch update.

## Verification and limits

- The added Python auditor compiled in the authoring checkout.
- Wiki structure and links were checked on this branch after replacing the
  stale overview and adding the index.
- Semantic wiki lint could not run because `litellm` is unavailable.
- The authoring checkout still has pre-existing unclassified evidence and
  build scratch. Those files were preserved, so that checkout is not clean.
