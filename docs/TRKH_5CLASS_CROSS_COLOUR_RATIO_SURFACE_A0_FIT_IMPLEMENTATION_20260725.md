# TRKH Cross Colour Ratio Surface A0 Fit Implementation

Date: 2026-07-25

## Scope

This boundary implements the prospectively locked, train-only Cross Colour
Ratio (CCR) information gate. It does not edit the image trainer, raw data,
validation/test paths, production inference graph, current-best command, or
checkpoint.

The runner consumes only:

- the exact replayed 735-image/763-row sRGB and valid-mask cache;
- the locked cohort arrays;
- the immutable CIDT clean train prediction CSV;
- the prospective lock, materializer evidence, and a separate fit
  authorization.

It never constructs `MangoYOLOCropDataset`, decodes raw dataset images, runs
the keeper, or loads external/pretrained weights.

## Scientific Roles

Five source-held folds train exactly four 3,004-parameter scratch roles for
20 epochs:

1. plain Colour Ratio derivative control;
2. Cross Colour Ratio candidate;
3. independent-seed CCR repeat;
4. trained spatially dephased CCR control.

The candidate and repeat are also scored once with their own frozen weights on
the same spatially dephased held maps. Their clean calibration thresholds are
reused without refit or recalibration.

The implementation verifies the locked fold partitions, source separation,
all 20 epoch-order hashes, dephase-offset hash, primary-role initial-state
identity, finite first-update gradients, every-parameter updates, FP32 AdamW
state, requested/effective workers `4/4`, and the exact parameter count.

## Data And Resource Controls

A process-wide Python audit hook allowlists only the retained materializer
files and immutable CIDT clean CSV. It blocks undeclared run/dataset inputs,
write-like access, and complete `val`, `valid`, `validation`, or `test` path
components.

Only one descriptor family exists at a time. Descriptor and reliability
scratch files are deleted in `finally`; the retained result contains only
compact state, held outputs/maps, training records, metrics, ledger, summary,
and exact-file-set manifests.

The formal run fails closed above:

- 20 minutes clean wall time;
- 2 GiB peak CUDA allocation for extraction/head fitting;
- 10 GiB process RSS;
- 0.40 GiB source-plus-temporary cache;
- 0.10 GiB retained artifacts.

Foreign GPU compute processes, missing CUDA BF16 support, dirty repository
state outside the three protected user paths, or any hash mismatch block the
formal run before candidate fitting.

After every role and fold finishes, the runner shuts down persistent
DataLoader workers and records a second process snapshot. Any surviving
Python/TensorRT child fails `orphan_compute_children_zero` before atomic
publication.

## Replay And Stop Rule

Formal output is built in a hidden sibling and atomically renamed only after
all engineering checks pass. Replay must run in a different process and
repeats the complete clean fit. It compares every state and output array,
including pair maps, scores, thresholds, and actions, within `1e-7`; metrics,
scientific training records, descriptor hashes, and the access-ledger hash
must match.

Failure of any clean scientific gate labels the method
`negative-but-reusable` and closes shifted conditions, XAI finalization,
deployment benchmarking, validation/test, trainer integration, full train,
and current-best command updates. A clean pass remains `mechanism-only` and
only opens a separately controlled shifted/XAI/deployment stage; it is not a
production promotion.

## Commands

```powershell
.\scripts\run_trkh_cross_colour_ratio_surface_a0.ps1 -Mode Preflight
.\scripts\run_trkh_cross_colour_ratio_surface_a0.ps1 -Mode Formal
.\scripts\run_trkh_cross_colour_ratio_surface_a0.ps1 -Mode Replay
```

The launcher invokes Python directly and checks `$LASTEXITCODE`; it does not
pipe native stderr through the PowerShell object pipeline.

## Formal Attempt 1 Harness Failure

The first authorized formal attempt stopped before CCR head fitting or any
metric. Plain colour-ratio descriptor extraction and its five control fits had
completed only in temporary memory. Enabling deterministic algorithms during
those fits caused the subsequent CCR resize matmul to fail because cuBLAS
workspace determinism had not been configured before CUDA initialization.

The same exception retained NumPy memmap frames long enough for immediate
Windows cleanup to report a sharing violation. The final output remained
absent; after the process exited, the two partial scratch files were removed
and no state, score, threshold, action, or metric was retained.

The audit-only recovery sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing
Torch and in the launcher, and explicitly closes every descriptor/source
memmap in a `finally` block. It does not change equations, cached input,
partitions, role seeds, epoch orders, model parameters, optimizer, threshold,
or promotion gates. The original fit authorization is consumed and cannot be
reused.

The recovery runner accepts only state
`fit_recovery_authorized_train_only_no_validation_test`. Its authorization
must pin the SHA-256 sidecar-verified failure evidence and explicitly
supersede consumed authorization SHA
`3ecb6a82d14980e25cd947007561294fa39cfe5419032cb3961389a30d1697b5`.
The runner also verifies that the failure record names that same authorization
as consumed and forbids reuse. The launcher now defaults to the separate
recovery-authorization filename, so the initial authorization fails both the
state and implementation-hash checks.
