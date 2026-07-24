# TRKH CAP Integral-Region Context A0 Implementation Boundary

Date: 2026-07-24
Protocol ID: `trkh_cap_integral_region_context_a0_20260724`
State: implementation complete before candidate metric

## Prospective Lineage

- Protocol/lock commit: `05a9ced`.
- Independent engine commit: `ea90703`.
- Protocol SHA-256: `c6d307bf15096b801ff6462c63c4d989160de64f0bdd691f79774a361a1a1e32`.
- Machine-lock SHA-256:
  `4aaba9ef9f37ed2c16de85a7aef62d0abbf6be01b47dabf67c750fbe3b6c0d87`.
- No CAP candidate score, calibration threshold, validation/test access,
  trainer integration, smoke, probe, full train, or current-best command
  update existed when this implementation contract was written.

## Independent Engine

`trkh/tools/cap_integral_region_context_a0_engine.py` implements:

1. exact rectangular valid-support crop and bilinear restoration to `16x16`;
2. paper-equation pixel context with `gamma=0` initialization;
3. the locked 27 integral regions in exact official order;
4. `42x42 -> region crop -> 7x7` bilinear pooling;
5. additive within-image region attention without the script-only sigmoid,
   squeeze-excitation, or spectral normalization;
6. one-layer ordered LSTM with hidden size 128;
7. residual-less NetVLAD with 32 clusters and official two-stage L2
   normalization;
8. one binary class-1 logit trained by unweighted BCE.

All CAP roles instantiate the same 1,090,979-parameter module. Primary-seed
self-only, candidate, spatial-deranged, and cross-sample roles have
byte-identical initial state. Seed repeat uses only the locked `+100000`
offset. The self-only role retains the same parameter budget but replaces
region context with the exact identity.

The runtime uses the linear identity:

`GAP(alpha @ flattened_regions) == alpha @ GAP(regions)`

This avoids materializing a second `[B,27,256,7,7]` tensor. The test suite
compares it with explicit full region construction.

## Fixed Auditor Semantics

`trkh/tools/audit_cap_integral_region_context_a0.py` fixes the following before
formal metrics:

- geometry-cache probability is authoritative for cohort keeper margin,
  keeper top-1, and highest non-class-1 replacement;
- CIDT clean rows are authoritative for the unchanged 9,215-row baseline;
- the two probability caches differ by at most `0.0015103519`, have identical
  top-1 on all 763 cohort rows, and differ on one near-tied rival argmax;
- each outer fold fits three folds, calibrates on `(outer+1)%5`, and scores
  held once at epoch 30;
- calibration threshold is the locked TP boundary; score equality is retained;
- suppression can only replace a keeper class-1 prediction with its geometry-
  cache highest non-class-1 rival; class-1 FN cannot be rescued;
- full metrics change only the exact cohort sample indices selected by held
  suppression actions;
- threshold-free AUROC, calibrated actions, five-fold stability, class-1
  precision/recall/F1, nonfocus safety, both causal controls, and seed repeat
  are conjunctive.

The process-wide open ledger allowlists locked scientific inputs and exactly
20 train-only visual paths derived from the locked CIDT table. It blocks
writes plus complete `val`, `valid`, `validation`, and `test` path components
before open. Formal and replay must have identical ordered ledger events.
The auditor also recomputes and checks the complete cohort Sattolo-permutation
hash and the ordered 20-anchor hash directly against the machine lock.

## Fixed XAI Definition

For each locked visual anchor:

- feature attribution is
  `abs(gradient(candidate_logit) * frozen_block2_feature)`, summed over
  channels and normalized by mass;
- CAP region attention is mean received key attention, distributed uniformly
  by each integral region's area, resized to `16x16`, and projected only into
  the exact valid support;
- the spatial control map uses its sample-index Sattolo region order;
- the sheet contains RGB+bbox, CAP region map, feature attribution, spatial
  control, absolute contrast, score, threshold, action, valid mass, bbox mass,
  and candidate-control score deltas;
- every anchor must differ from both spatial and cross-sample controls by more
  than `1e-7`;
- automatic XAI cannot replace manual review of all 20 rows.

## Replay And Resource Boundary

- Formal and replay are separate Python processes. Formal stores its PID;
  replay fails if the PID is equal.
- Replay rebuilds all scientific evidence and requires exact discrete arrays,
  ordered ledger events, and maximum numeric error `<=1e-7`.
- Formal and replay each write an exact-file-set manifest. Verification rejects
  missing files, unlisted files, duplicate paths, absolute paths, traversal,
  and hash/size drift. Manual finalization is one-shot and requires both
  manifests to remain intact.
- Synthetic worst-role batch-32 preflight passes for FP32 and BF16.
- Peak synthetic CUDA allocation is `0.934431 GiB` FP32 and `0.890468 GiB`
  BF16; peak process RSS is `1.668243 GiB`.
- Every trainable role has finite logits, loss, and gradients.
- A synthetic candidate 30-epoch cycle produced the exact trace epochs,
  calibration/held scores, optimizer count, and state hash without reading the
  scientific cohort.

## Verification Before Formal

- Engine NumPy-oracle maximum equation error: `2.55e-15`.
- CAP lock/engine/auditor focused tests: `25/25`.
- Complete repository suite: `1924/1924` with 295 existing warnings in
  `76.14 s`.
- PowerShell launcher parses without errors and uses `$LASTEXITCODE`, not a
  native stderr object pipeline.
- The launcher runs formal and replay as separate Python invocations.
- Research-process report revision 12 renders cleanly across all 13 pages and
  passes accessibility at `0/0/0`.

Formal A0 remains prohibited until this implementation stage, its tests, the
living report, and the full repository suite are committed and pushed.
