---
title: Program B B2 Baseline v4 Recovery Audit - 2026-08-17
type: audit
created: 2026-08-17
status: DIAGNOSTIC_ONLY_PRE_B2_PROTOCOL
sources:
  - wiki/analyses/program-b-b2-train-from-scratch-protocol-2026-08-14.md
  - C:/tmp/tod-b2-baseline-s42-20260814-recovery/
---

# Program B B2 Baseline v4 Recovery Audit

## Scope

This audit recovered terminal artifacts from Kaggle kernel
`ngquangnht/tod-program-b-b2-baseline-s42-20260814`. The kernel reports
`KernelWorkerStatus.COMPLETE`; completion is only an artifact-retrieval state.
The recovered files are staged under the deliberately short local root
`C:/tmp/tod-b2-baseline-s42-20260814-recovery/`.

## Recovered artifacts

- `metrics.csv`: 20 epoch rows;
- `best.pt`, `best_ap75.pt`, `best_coco_ap.pt`, and `last.pt`;
- source tree and `source_manifest.json`;
- Kaggle worker log.

The selected historical `best.pt` SHA-256 is
`f4c85ad2474be94d738bafa23ec1789f76c740063f793c7c226fae634dcd8aad`.
The historical source archive SHA-256 recorded in its manifest is
`86faa10e9b71e420365c34367fade6d0a34fad0c1cddec3403f2f12a55568aae`.

## Eligibility decision

**`DIAGNOSTIC_ONLY_PRE_B2_PROTOCOL` — do not reload, ledger, rank, or use as a
teacher.**

The artifact predates the current B2 execution surface. It does not bind to the
B2 evaluator-integrated source snapshot v2 (whose downloaded manifest SHA-256
is `9b7f351accf613a29c4f6de53bce2e4845e4a6b885c64e774c0b7ecd3b0491ab`) or
the frozen `512/64` manifest-backed original-image evaluation contract. Its
recovered epoch ledger contains legacy tile/validation metric columns
(`mAP_primary`, `coco_AP`, `coco_AP50`, `coco_AP75`) rather than required
per-epoch original-image official-evaluator rows and deterministic
reconstruction inputs.

The canonical-source gate therefore fails before independent checkpoint reload.
A local reload against the changed checkout would be diagnostic only and cannot
repair the provenance/evaluator mismatch. The current B2 baseline remains
`BLOCKED_HARDWARE_CONTRACT`; the later required mount/model-init smoke stopped
on an incompatible P100 before model initialization, so no refreshed B2
baseline or candidate has training evidence.

## Consequences

- No B2 result ledger, claim, or performance comparison is updated.
- No PC-MR, PC-MOC, or combined run may use this checkpoint as its teacher.
- Retry the current frozen baseline only after a T4-capable assignment or an
  explicit owner-approved hardware-contract revision.
