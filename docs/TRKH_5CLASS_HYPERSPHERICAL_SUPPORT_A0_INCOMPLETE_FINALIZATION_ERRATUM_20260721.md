# TRKH Hyperspherical Support A0 Incomplete Finalization Erratum - 2026-07-21

## Status

Prospectively locked after the first and only formal readout process stopped,
but before any generated prediction, fold metric, geometry value, or candidate
gate was opened. Only the exception text, payload names, byte sizes, and
SHA-256 values listed below were inspected.

This erratum does not change the locked protocol, equations, folds, seeds,
readout roles, candidate probabilities, controls, thresholds, gates, or
authorization rules. It forbids a second readout run.

## Failure Boundary

The formal process completed the fixed source-held readouts and wrote six
pre-summary payloads. It then stopped while calculating the immutable keeper's
in-sample reference metrics:

```text
ValueError: Probabilities are not normalized
```

The cache loader prospectively accepts the persisted float32 keeper
probabilities with row-sum tolerance `2e-4`. The generic metric helper used a
stricter `1e-9` tolerance intended for newly produced float64 readout
probabilities. Applying that helper directly to the historical float32 keeper
reference is internally inconsistent. The keeper reference does not enter any
candidate metric or promotion gate.

No `summary.json` or `artifact_manifest.json` was written. The Python process
exited and no Python/TensorRT/FFmpeg process remained.

## Locked Incomplete Payloads

Directory:
`runs/audit_hyperspherical_support_a0_20260721`.

Exact payloads, all required and immutable:

| Name | Bytes | SHA-256 |
|---|---:|---|
| `fold_assignment.json` | 4,246 | `162f75c959957ce89c72a43b44a916ebf56b458b7044fc016bb6991727ef97fc` |
| `fold_metrics.csv` | 3,008 | `79ccb358096a893d94a6f42a1e7a307052e9ccd06bf814659560e3da0c6e662c` |
| `fold_protocol.json` | 25,721 | `b45852a53dc015740fb4b945be459e91d76b50fbaa34851c8e133728b8a51966` |
| `predictions.csv` | 6,221,489 | `ec86e45c8d539e48d1bcb6801e09476acb80226f82f4839f8eb48eecc467da20` |
| `synthetic_geometry.csv` | 51,637 | `051f454e47c133fc6ba97b21c67b5eb60c6baa4ce42c87fbda44827888577134` |
| `synthetic_geometry.json` | 119,813 | `9ed3d9241718b49931de26c8640a95f4f97ca7e0ac4207318e67e1ade7fad796` |

The finalizer must reject any missing, additional, renamed, resized, or
hash-changed pre-summary payload. It must also reject an existing summary or
manifest. It may not delete, rewrite, regenerate, or append to any of the six
payloads.

## Sole Authorized Repair

1. Add an explicit `FinalizeIncomplete` launcher/module phase.
2. Require the original protocol hash, this erratum hash, clean pushed git
   state, protocol ancestry, every original input hash, and all six incomplete
   payload hashes before reading any payload content.
3. Reload the immutable train cache only to verify labels/folds and to compute
   the non-gating keeper reference. Normalize each keeper probability row by
   its observed float64 row sum before passing it to the strict metric helper.
   Persist the pre-normalization maximum absolute row-sum error. Do not alter
   the cache or any generated readout probability.
4. Reconstruct all candidate/control metrics, transitions, folds, and gates
   only from the locked `predictions.csv`; do not call support generation or
   logistic fitting.
5. Read the locked fold/geometry payloads only for structural/convergence
   checks. Require the existing fold assignment to equal the independently
   reloaded CIDT assignment exactly and require all original geometry gates to
   pass under the unchanged `1e-10` tolerances.
6. Write only the previously absent `summary.json` and
   `artifact_manifest.json`. The summary must state that original formal
   compute runtime/peak RSS were not recovered because the reporting exception
   occurred before summary serialization; record finalization runtime
   separately and do not invent the missing values.
7. Run the original canonical replay in a second process and require maximum
   numeric difference `<=1e-12` plus complete manifest verification.

If any lock or structural check fails, stop without writing summary/manifest.
No alternative tolerance, metric, seed, readout, rerun, output directory, or
manual result reconstruction is authorized.

## Decision Rule

After successful finalization, accept or reject the route only by the original
protocol gates at SHA-256
`f9485b7bb0ed22b8ff5d6b4cc2f79af0f766b409d43eef2be3f70518410b280c`.
The reporting repair cannot authorize validation/test, probe/full train, or a
current-best command update. A complete original A0 pass can authorize only
the separately locked default-off representation smoke described by the
original protocol.
