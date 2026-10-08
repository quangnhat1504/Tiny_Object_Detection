---
title: "Evidence Boundaries"
type: "concept"
created: "2026-10-08"
updated: "2026-10-08"
sources:
  - "docs/ARTIFACT_RETENTION_POLICY.md"
  - "journal/audits/repository_wiki_cleanup_20261007.md"
  - "journal/audits/repository_hygiene_hardening_20261008.md"
tags:
  - "evidence"
  - "scope"
---

# Evidence Boundaries

The auditor skips tracked files and prunes runtime, worktree, dataset,
sealed-project, result, manuscript, audit, archive, and submodule paths before
scanning. It does not open those contents to classify candidates. Review other
untracked or user-owned files before applying cleanup.

Keep accepted evidence and locked manifests. The tool protects the listed
evidence roots; a candidate outside those roots still needs human review.

A cleanup receipt records actions; it does not prove experiment acceptance or
make deletion reversible. Reports belong outside protected evidence paths, with
`.runtime/` as the preferred receipt location. `journal/audits/` is protected.

This publication branch contains only the hygiene update, not the wider local
experiment source and audit tree. Use [[RepositoryHygiene]] for the command
workflow and [[overview]] for the branch's evidence limits.
