# TRKH 5-Class Pair-Surface DDF v2 Pre-Lock Process Incident - 2026-07-29

## Status

This is a prospective-process deviation record, not scientific evidence and
not an execution authorization. It must be bound by every later registry,
pre-lock contract and machine lock for Pair-Surface DDF v2.

The deviation was discovered at `2026-07-29T04:31:48+07:00`, after the
pre-lock sequencing erratum had been committed and pushed. No S1, S2, machine
lock or formal authorization existed.

## Exact initiating command

The main process intended to rerun a combined source-test suite:

```text
D:\DataAI\.venv\Scripts\python.exe -m pytest -q \
  tests/test_pair_surface_ddf_v2_engine.py \
  tests/test_pair_surface_ddf_v2_scientific.py \
  tests/test_audit_pair_surface_ddf_v2_geometry_metadata.py \
  tests/test_build_trkh_pair_surface_ddf_v2_fold_manifest.py
```

The command reported `55 passed` with three ONNX warnings. That result is
withdrawn as pre-lock evidence because two selected test files automatically
activated live-data integration paths when their local inputs existed.
Repository HEAD during the command was the pushed sequencing-erratum revision
`12da367c556710db9db4b43ee092cb87cf0d8013`; the scientific/guard prototypes
under test were untracked and had no execution authority.

## Unauthorized read scope

`test_locked_train_table_reproduces_protocol_geometry_numbers` opened the
retained train-only CCR cache and validity mask:

- `runs/audit_cross_colour_ratio_surface_a0_materialized_20260725/cohort_arrays.npz`;
- `runs/audit_cross_colour_ratio_surface_a0_materialized_20260725/image_valid_masks_packbits.npy`.

It deserialized `sample_indices`, `targets` and `model_boxes` and propagated
the validity masks. It recomputed the already frozen geometry checks in memory.

The module-scoped `generated` fixture in the fold-manifest tests also opened
and hashed the cohort container, deserialized only `sample_indices`, `targets`,
`source_stems` and `image_paths`, and read the YOLO manifest plus pinned CVAT
exporter/provenance documents. Its manifest audit scanned cross-split identity
metadata, including validation/test metadata, for an already frozen
group-contamination veto. It did not open validation/test pixels, labels,
scores or checkpoints.

These reads were not protected by an S2 claim, archive-member ledger or
phase-specific allowlist. The fact that the same metadata and conclusions had
been frozen previously does not make this access compliant with the new
sequencing erratum.

## What was not observed or changed

- no candidate forward, loss, gradient, optimizer step or calibration;
- no keeper/candidate/external checkpoint or score array;
- no raw image or raw label file;
- no GPU or TensorRT execution;
- no validation/test pixel, label, prediction or metric;
- no retained S1, S2, formal, replay, XAI, integration or production output;
- no model-selection, architecture, threshold or gate decision from a newly
  exposed row value; and
- no authorization claim or quota consumption, because none existed.

Pytest temporary writes are operational scratch only. The locked fold and
geometry artifacts were not overwritten.

## Consequence

The `55/55` combined result cannot support S2, a machine lock, formal
authorization or a reproducibility claim. No value produced in that process
may be imported into a future S1/S2 evidence bundle. A future S2 must still
derive and replay every required artifact from its exact committed contract in
fresh authorized processes.

This incident does not by itself expose a candidate observation or consume the
future one-formal/one-replay scientific quota. Whether the v2 route may proceed
under a corrected pre-lock boundary must be decided by independent review of
this record; it cannot be assumed merely because the observed metadata was
historical.

## Corrective controls

1. Before S2, run only explicitly enumerated synthetic/tmp-fixture tests. Never
   select a whole test file whose `skipif` or module fixture becomes live when
   a local cache exists.
2. Classify every pinned test as `synthetic_safe`, `S1_only`, `S2_only`,
   `formal_only` or `forbidden_before_new_holdout`. The contract fails closed
   on an unclassified test.
3. Install a pre-test cache-open tripwire before collection/import, not only
   inside the scientific runner.
4. Keep fold-manifest and live geometry integration tests outside the
   pre-lock synthetic suite. Their earlier committed evidence remains lineage;
   any fresh reproduction belongs only to an authorized S2.
5. Bind this incident document and its SHA-256 into the next registry and all
   S1/S2/machine-lock boundaries.
6. Independently review both the incident classification and the corrected
   test manifest before any S1 authorization is written.

## Current authority

Current authority remains zero for S1, S2, machine-lock construction, formal
A0, replay, validation/test, conditional XAI, integration and full train. This
record grants no recovery run and no implicit retry.
