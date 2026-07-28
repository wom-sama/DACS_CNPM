# TRKH Pair-Surface DDF v2R2 Fold Provenance Erratum — 2026-07-29

## Status

This prospective erratum supersedes only the provenance disclosure and write
semantics of
`TRKH_5CLASS_PAIR_SURFACE_DDF_V2_FOLD_MANIFEST_20260729.json` (r1, SHA-256
`8a084b171b12c4d8a4e85d09d748d30cdf46191ea7160748cdd03519b38f924b`).
R1 remains byte-preserved as lineage. Zero v1/v2 formal run, replay, candidate
score, validation result, test result, or GPU result was consumed before this
correction.

The effective fold artifact for every future v2 engine, machine lock,
authorization, auditor and replay is
`TRKH_5CLASS_PAIR_SURFACE_DDF_V2R2_FOLD_MANIFEST_20260729.json`, SHA-256
`81a404bedc25dc7b3e7dc3dbb8d79b4ca5e97d74374083c79e0686ececb7fb2d`.
Its SHA sidecar has SHA-256
`d3521190cf8e706b67dcba9e05221a2ca532e8c23f0e9aed69f0288004b73fd0`.

## What was corrected

R1 stated that only four cohort-array keys were accessed and that validation
and test were not accessed. The assignment implementation did deserialize only
the four intended metadata arrays, and no forbidden array affected a fold, but
the statement was too broad:

- the complete NPZ container bytes were hashed for provenance, so bytes that
  contain other members were physically read even though those members were
  not decoded or used as features;
- the complete dataset-manifest CSV was scanned to prove that a related
  `leakage_group` did not cross output splits, so validation/test *identity
  metadata* was read even though no validation/test pixel, label file, score,
  checkpoint or model output was opened;
- explicit `--write` used per-file atomic `fsync` plus `os.replace`, but it did
  not require both frozen outputs to be absent. Two per-file replaces are not a
  transactional two-file commit.

R2 reports those boundaries directly. The full NPZ SHA remains an opaque
lineage/provenance gate and is never an assignment feature. Only these four
members are deserialized for assignment, each with locked dtype, shape and
canonical array SHA-256:

| Member | dtype | shape | SHA-256 |
|---|---|---:|---|
| `sample_indices` | `int64` | 763 | `ad51a9bdbf6acc7449ad8d8f65b3dc69971c318d7f64e2effd814bac8f77fe05` |
| `targets` | `int64` | 763 | `15c43ecc7335c7a7febc4e0fbf622df7ad593fc52e5d9f14e5cc27acc16fc000` |
| `source_stems` | `<U11` | 763 | `b127d4d6ef8d1ae6b7f6137f8b7ff461c105fd85de414afb2abdf678a3065b21` |
| `image_paths` | `<U61` | 763 | `6c42e9e7d5d7a34ecffee72dd96edf5240c54eed291d1a6f86c712016cdfb002` |

The canonical selected-member-set SHA-256 is
`d341f4e6171dd54dcc01c8f89639e1597f8317a24b9565b926d1184d0eb529f3`.
The member-name schema is checked before any member is deserialized. Legacy
`folds`, `keeper_probabilities`, `model_boxes`, `crop_boxes` and `label_paths`
are not deserialized by the r2 assignment builder.

## Scientific identity with r1

R2 independently reconstructs the same 763 rows and the same scientific fold;
only disclosure/write behavior changed. All of the following are byte-identical
to r1:

- canonical metadata SHA-256
  `f0fa476fd4be3765b1ed8c780e09d52037a0215554a7579dbd259d00cab02b60`;
- edge-set SHA-256
  `0b4791ce986f98d6d36d2be0f4ba3cdf7cce3f0354850aa84e54e5c2de0dbc61`;
- component-set/order SHA-256
  `be6cece69ec9d6ea00ad7c953e49af3fb359ff0bc57c4f09bfca7c2895f51c37` /
  `e10e1045e6ede63ade0cb436b684be7a6ab377d0d27f74480ac8721756045ccb`;
- sample-to-fold mapping SHA-256
  `c0726d114df69290d68d9f9e1a40857074ccb1a66e5ca41883caa004f2b0ea40`;
- component-assignment SHA-256
  `1ce9837be1a7e60d6c6cef0768d68eb29c22f2fa326f3558b8d0395e3dbd1819`;
- fold row counts `153/153/153/152/152`, with exactly two target-4 rows per
  fold and zero cross-fold relation/component overlap.

The scientific protocol
`TRKH_5CLASS_PAIR_SURFACE_DDF_A0_V2_PROTOCOL_20260729.md` otherwise remains
unchanged. Its statements about group limitations, masked validity, optimizer,
calibration, causal controls, gates, replay and new-holdout requirements still
apply.

## Immutable write and use contract

The r2 builder is check-only by default. Its explicit write path refuses if
either the JSON or SHA sidecar already exists. Each file is individually
written through `fsync` and atomic replacement, but the pair is explicitly
reported as non-transactional; any crash-created partial pair fails closed and
requires a separately reviewed recovery action. R2 becomes an effective
boundary only when builder, tests, artifact and this erratum are committed and
pushed together.

Every later v2 machine lock and authorization must bind the exact r2 manifest,
builder, original v2 protocol and this erratum. R1 may be used only to reproduce
lineage and prove mapping identity. Neither r1 nor r2 authorizes formal/GPU
execution.
