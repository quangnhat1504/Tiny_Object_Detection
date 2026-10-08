---
title: "Repository Hygiene"
type: "concept"
created: "2026-10-08"
updated: "2026-10-08"
sources:
  - "docs/ARTIFACT_RETENTION_POLICY.md"
  - "scripts/maintenance/README.md"
  - "journal/audits/repository_hygiene_hardening_20261008.md"
tags:
  - "repository"
  - "maintenance"
---

# Repository Hygiene

The hygiene auditor defaults to a read-only scan. It lists only allow-listed
cache, log, and checkpoint candidates. It skips tracked files and protected
paths, and it rejects symbolic links and filesystem reparse points.

Apply mode is a destructive operation. It requires a new receipt before
scanning and records progress for every candidate. Review the complete report
and obtain authorization for the specific cleanup before using it. An
interrupted receipt needs manual review; the tool does not resume automatically.

Use the grouped command in `scripts/maintenance/`. The old root command stays
as a compatibility entry point. The full retention rules are in
[the artifact policy](../../docs/ARTIFACT_RETENTION_POLICY.md), and the branch
scope is in [the hardening audit](../audits/repository_hygiene_hardening_20261008.md).

See [[EvidenceBoundaries]] for protected paths and [[overview]] for the
evidence limits of this branch.
