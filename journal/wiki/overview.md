---
title: "Journal Branch Status and Evidence Limits"
type: "overview"
created: "2026-10-07"
updated: "2026-10-08"
sources:
  - "journal/audits/repository_wiki_cleanup_20261007.md"
  - "journal/audits/repository_hygiene_hardening_20261008.md"
tags:
  - "journal"
  - "scope"
  - "evidence"
---

# Journal Branch Status and Evidence Limits

This branch is based on `origin/main` at `983e87b`. Its manuscript and
experiment records predate the October 7 local review. The wiki and hygiene
updates add no experiment results.

The previous overview and research log are preserved in
`archive/legacy-journal-wiki-20261007/`. Their source and result references
describe a broader repository tree and are not current acceptance records. The
archive README records the log hash. Use only evidence whose source, selector,
artifact, replay, and audit are present in the active branch.

The repository hygiene tool defaults to a read-only scan. It excludes protected
and private paths, rejects links and reparse points, and records per-path
outcomes before applying any authorized cleanup. Read the retention policy and
its hardening audit before using the tool.

Official final-test surfaces remain closed. A read-only cleanup scan was run;
no cleanup was applied. No test inference, training, dataset access, or cloud
operation was performed.
