# TRKH 5-Class Pair-Surface DDF v2 Pre-Lock Incident Independent-Review Erratum - 2026-07-29

## Status

This document preserves the independent review of
`TRKH_5CLASS_PAIR_SURFACE_DDF_V2_PRELOCK_PROCESS_INCIDENT_20260729.md`.
It does not replace or rewrite the pushed incident record. The corrected
classification is:

> CONFIRMED PRE-LOCK ACCESS-CONTROL INCIDENT; LIVE TRAIN TARGET/GEOMETRY AND
> CROSS-SPLIT IDENTITY METADATA ACCESSED WITHOUT AUTHORITY; ENTIRE 55/55 RESULT
> WITHDRAWN; NO DEMONSTRATED LIVE CANDIDATE/CHECKPOINT/RAW-PIXEL/RAW-LABEL/
> VALIDATION-TEST-METRIC OR GPU COMPUTE; NO SCIENTIFIC OUTPUT OR QUOTA
> ACCEPTED; IMPACT SOURCE-BOUNDED, NOT TELEMETRY-COMPLETE; REMEDIATION OPEN;
> ALL AUTHORITY ZERO.

## Exact invocation and source-identity limit

The invocation is retained as one literal argv line, rather than using a
backslash as if it were a PowerShell continuation character:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m pytest -q tests/test_pair_surface_ddf_v2_engine.py tests/test_pair_surface_ddf_v2_scientific.py tests/test_audit_pair_surface_ddf_v2_geometry_metadata.py tests/test_build_trkh_pair_surface_ddf_v2_fold_manifest.py
```

The exact scientific test/module blob executed by that process cannot now be
reconstructed. At the pushed sequencing revision, those files were untracked.
The three reconstructible tracked files account statically for 41 cases; the
reported total therefore implies 14 cases from the unavailable scientific
test blob. The later scientific test committed at `9ea25ea` has a different
case set and must not be substituted retrospectively.

## Corrected execution scope

Source inspection confirms that collection activated the live train-cache
geometry regression and the real fold-manifest fixture. The process opened the
cohort and validity containers, decoded train sample indices, targets,
source/image identity metadata, model boxes and validity geometry, and scanned
cross-split manifest identity metadata without the required S2 claim,
member-level ledger or S2A-to-S2B transition.

The original incident wording was too broad in two places:

- Synthetic engine tests did execute synthetic forwards, losses, gradients
  and an optimizer update, and scientific tests fitted synthetic calibrators.
  The supported statement is that no **live/formal candidate** execution or
  real-data calibration occurred.
- Whole-container SHA-256 checks physically read the opaque compressed bytes
  of the cohort archive, including bytes encoding the keeper-probability
  member. That member was not deserialized, decoded, inspected or used as a
  score array.

Source-bounded reconstruction found no raw-pixel or raw-label open, checkpoint
load, validation/test metric, TensorRT execution or GPU compute. ONNX Runtime
was explicitly CPU-only. Because no contemporaneous complete import closure,
member ledger or OS access receipt exists, these are bounded source findings,
not exhaustive telemetry claims.

The live-data-dependent pytest regression decision did occur and is withdrawn.
No scientific candidate promotion, architecture selection, retained efficacy
decision, formal/replay quota or machine-lock observation was accepted. No
authoritative scientific output from the process may be imported into S1, S2,
the machine lock, A0, XAI, integration or a paper result.

## Independent disposition

Withdrawal of the complete 55/55 result is scientifically sufficient only if
every per-node result, intermediate value and derived hash from that process
remains excluded permanently. Remediation stays open until a prospectively
committed test authority pins the exact source/import closure and a bootstrap
tripwire is active before local test imports and collection.

The later runtime-guard manifest invalidation is a separate developmental
test-attestation problem with zero demonstrated live-data access. It is
recorded in
`TRKH_5CLASS_PAIR_SURFACE_DDF_V2_RUNTIME_GUARD_TEST_ATTESTATION_INVALIDATION_20260729.md`.
