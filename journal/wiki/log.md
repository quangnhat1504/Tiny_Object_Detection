# Journal Project Activity Log

> This log tracks the wiki and repository guidance present on this branch.
> Earlier research activity is preserved in the dated archive; it is not a
> current experiment-evidence ledger.

## [2026-10-07] update | Add scoped wiki index and safe cleanup policy

* Added an index for the wiki pages available on this branch.
* Archived the previous overview because its cited source paths were absent.
* Added a read-only-by-default repository hygiene auditor and an artifact
  retention policy.

## [2026-10-08] update | Harden the hygiene workflow

* Grouped the hygiene implementation under `scripts/maintenance/` and kept the
  former command path as a compatibility entry point.
* Added path, link, report, and per-candidate receipt safeguards, including
  support for linked worktrees.
* Corrected the retention guidance to keep receipts outside protected audit
  and result paths.
* The historical research log was preserved byte-for-byte at
  `archive/legacy-journal-wiki-20261007/log-20260927.md` with SHA-256
  `27e18f003974ab1e02f9e14842180db0d082902d2114012822ab91dac15664ef`.
