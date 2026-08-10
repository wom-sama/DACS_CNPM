# TRKH CAP A0 XAI Harness Erratum

Date: 2026-07-24
Protocol ID: `trkh_cap_integral_region_context_a0_20260724`
Scope: audit-only runtime correction; scientific lock unchanged

## Failure Evidence

The first post-implementation formal process completed all six trainable roles
for all five source-held folds, then failed before creating any formal artifact.
The failure occurred while taking the gradient of the candidate logit with
respect to one frozen block-2 feature tensor:

`RuntimeError: cudnn RNN backward can only be called in training mode`

The candidate model was correctly in evaluation mode. cuDNN's LSTM evaluation
forward does not retain the reserve buffer required for backward attribution.
This is an XAI backend limitation, not a model-training, metric, split, or
equation failure.

The failed output directory contained zero files. No `summary.json`,
prediction array, threshold, metric, contact sheet, or candidate decision was
available before this correction. The immutable failure logs are retained at:

- `runs/evidence_cap_a0_xai_harness_failure_20260724/formal_stdout.log`,
  SHA-256
  `630b5cec96cc3f4d32b781dff371305ff50327f7975fd0531785f7276ebab887`;
- `runs/evidence_cap_a0_xai_harness_failure_20260724/formal_stderr.log`,
  SHA-256
  `4893082958f85c9c330f1432c49d69bb33375fa61e330692115f388326d2a7bb`.

## Narrow Correction

The formal candidate and spatial-control models remain in `eval()` mode.
Only the candidate forward and backward used to construct
`abs(gradient * frozen_feature)` now run inside:

`torch.backends.cudnn.flags(enabled=False)`

This selects PyTorch's native RNN implementation for the XAI operation and
restores the prior cuDNN state on exit. It does not:

- change the CAP engine or any trained state;
- put the LSTM or model into training mode;
- change folds, rows, labels, seeds, orders, epochs, optimizer, loss, region
  geometry, threshold, gate, or visual anchor;
- open validation/test or modify raw data;
- change training/scoring, which still use the locked CUDA/cuDNN path.

The regression test requires CUDA, reproduces an eval-mode CAP LSTM gradient,
checks finite nonzero attribution, verifies that model/LSTM mode and global
cuDNN state do not drift, and bounds the native-RNN logit difference from the
locked eval forward by `1e-5`.

## Verification

- CAP-focused tests pass `26/26`.
- The complete repository suite passes `1925/1925` in `93.86 s`
  (`296` warnings, no failures).
- Research-process report revision 13 passes visual QA on `14/14` rendered
  pages and accessibility audit at `high=0`, `medium=0`, `low=0`.

## Rerun Rule

One rerun of the exact locked formal is allowed only after this erratum,
implementation, focused regression, and complete repository suite are committed
and pushed. The failed zero-artifact invocation is not a candidate trial and
cannot be used to alter any scientific choice. Formal must still start from the
same protocol/lock and a clean pushed commit; replay and all 20 manual XAI rows
remain mandatory.
