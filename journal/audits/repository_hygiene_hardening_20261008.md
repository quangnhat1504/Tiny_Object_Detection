# Repository hygiene hardening — 2026-10-08

## Scope

This publication branch is based on `origin/publish/repository-hygiene-20261007`
(commit `477a861`). It contains the earlier hygiene policy and auditor. The
wider October 8 script tree is absent, so this update carries only the
self-contained hygiene tool and its supporting documentation. It does not
import experiment audits or result records that refer to missing source files.

## Changes

- The implementation is grouped at
  `scripts/maintenance/audit_repository_hygiene.py`; the former
  `scripts/audit_repository_hygiene.py` path remains a compatibility command.
- The scanner prunes protected paths before walking and excludes tracked files.
  It skips symbolic links and filesystem reparse points, including Windows
  junctions, and checks the path again before applying a candidate.
- Apply mode requires a new receipt before scanning. It records the complete
  candidate list and per-path `PENDING`, `IN_PROGRESS`, and terminal outcomes.
  Interrupted operations remain visible for manual review.
- The retention policy directs reports away from protected evidence paths.
  `.runtime/` is the preferred receipt location; `journal/audits/` is rejected.

The implementation SHA-256 is
`d0a3f583517f5327c28dfa5da1dddfefb83266e172ff918b7861c3f9a88f3a8c`.
The compatibility wrapper SHA-256 is
`fc6416c1ac6aa30cfdc9daa208ae5067e2fa258c69bba0608d51b84212b4e94c`.

## Validation

- Static AST parsing passed for the implementation.
- A read-only dry run against this linked worktree completed with zero
  candidates. No apply mode ran.
- Wiki Health scanned three pages with no empty pages, index mismatch, or log
  coverage gap. Deterministic lint found no orphan pages, broken wikilinks,
  missing entity pages, or sparse pages.
- Semantic LLM lint and unit tests were not run. No cleanup was applied.
  No inference, training, dataset access, or cloud operation was performed.

## Preserve the stale branch research log

The prior 155,172-byte `journal/wiki/log.md` described a wider repository tree
than this publication branch. In the bounded navigation-file review, only the
handover linked to the active log; the log was not an experiment-evidence
source. It was moved byte-for-byte to
`archive/legacy-journal-wiki-20261007/log-20260927.md`. SHA-256
`27e18f003974ab1e02f9e14842180db0d082902d2114012822ab91dac15664ef` matched
before and after the move. The active log now records only branch-level wiki
and hygiene changes. The original remains recoverable from the archive and Git
history.
