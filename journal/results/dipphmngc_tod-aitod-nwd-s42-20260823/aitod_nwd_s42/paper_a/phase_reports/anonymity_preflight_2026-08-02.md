# Paper A Anonymity and PDF Preflight - 2026-08-02

Status: `PASS_INTERNAL; VENUE_PACKAGE_PENDING`

## Checked Artifacts

- `manuscript/main.tex`
- `manuscript/main.pdf`
- `manuscript/supplementary.tex`
- `manuscript/supplementary.pdf`

## Passing Checks

- Main author field is `Anonymous submission`.
- Main PDF metadata contains no Author, Subject, Keywords, local path, username,
  Kaggle account, GitHub URL, or executable JavaScript.
- Extracted main-PDF text contains none of the project owner/account/path tokens
  checked in this preflight.
- Main paper compiles to five pages and supplement to one page.
- The final build has no undefined citations, undefined references, multiply
  defined labels, or overfull boxes.
- All listed main-PDF fonts are embedded.

## Open Submission Risks

- The internal draft uses generic two-column A4 `article`, not the frozen WACV
  submission template.
- One embedded font is Type 3. It must be eliminated or accepted by the venue's
  PDF checker after template migration.
- PDF creation/modification timestamps remain in metadata.
- `anonymous_code/` is not yet a complete reproducibility package, so code,
  notebook output, Git identity, path, and checkpoint-metadata scanning remains
  pending.
- Performance placeholders are intentional and must not appear in the final
  submission.

## Decision

The current PDFs are suitable for internal review only. Do not label G6 or the
paper submission-ready until the venue template, complete anonymous code
package, final result artifacts, and independent coauthor red-team review pass.
