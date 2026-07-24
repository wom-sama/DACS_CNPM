# TRKH CAP Integral-Region Context A0 Closure

Date: 2026-07-24
Protocol ID: `trkh_cap_integral_region_context_a0_20260724`
Final decision: `reject_exact_cap_a0`
Value verdict: `negative-but-reusable`

## Prospective Boundary

- Protocol/lock SHAs:
  `c6d307bf15096b801ff6462c63c4d989160de64f0bdd691f79774a361a1a1e32`
  and
  `4aaba9ef9f37ed2c16de85a7aef62d0abbf6be01b47dabf67c750fbe3b6c0d87`.
- The paper equation, 27-region order, five source-held folds, all 30 epoch
  orders, seven roles, thresholds, controls, replay tolerances, resource
  ceilings, and 20 visual anchors were fixed before candidate metrics.
- The same-lock formal and replay ran from clean pushed commit
  `9191d8a91d23e3b5722c3d52da53bb73a9a82bba`.
- The auditor opened no validation or test path. The dynamic ledger recorded
  88 ordered events over 32 unique paths, with zero blocked attempts and
  digest
  `20910d4be6a24d0e026158433cacdb8e06c80102d98fa7014b9ebb2c1313e41c`.

The first formal attempt at commit `abc9963` completed all 30 fold-role jobs
but failed before writing an artifact because cuDNN cannot backpropagate
through an evaluation-mode LSTM. Its zero-artifact stdout/stderr SHAs are
`630b5cec96cc3f4d32b781dff371305ff50327f7975fd0531785f7276ebab887`
and
`4893082958f85c9c330f1432c49d69bb33375fa61e330692115f388326d2a7bb`.
Only the attribution backend was corrected: the model stayed in evaluation
mode while that forward/backward used the native PyTorch RNN. No scientific
setting changed.

## Formal Result

The same-lock formal completed in `6150.613 s` and failed the prospective
performance conjunction.

| Measure | Keeper baseline | CAP candidate | Keeper-margin control |
|---|---:|---:|---:|
| TP-vs-restricted-FP AUROC | N/A | `0.560111` | `0.836149` |
| FN-vs-restricted-FP AUROC | N/A | `0.412682` | `0.000000` |
| Restricted FP removed | N/A | `5/222` | `40/222` |
| Class-1 TP broken | N/A | `14/528` | `18/528` |
| Net corrections | N/A | `-9` | `+18` |
| Full macro-F1 | `0.939876` | `0.937491` | `0.942588` |
| Class-1 precision | `0.700265` | `0.699320` | `0.732759` |
| Class-1 recall | `0.975970` | `0.950092` | `0.942699` |
| Class-1 F1 | `0.815444` | `0.805643` | `0.824576` |

The candidate suppressed 19 rows: five corrections and 14 harms. Its
class-1 precision gain was `-0.000946`, recall drop was `0.025878`, class-1
F1 gain was `-0.009801`, and macro-F1 gain was `-0.002385`. Fold AUROCs were
`0.552181`, `0.590476`, `0.604167`, `0.615575`, and `0.551265`; folds 0 and
2 rejected no restricted false positive.

The independent seed repeat was not stable enough. Its AUROC was `0.541163`,
an absolute difference of `0.018948` above the locked `0.01` limit. It
removed eight restricted false positives, broke 18 true positives, and
produced net `-12` corrections.

The causal interpretation also failed:

- spatial derangement AUROC: `0.560452`;
- cross-sample context derangement AUROC: `0.565904`;
- GAP-linear control AUROC: `0.565145`;
- integral self-only control AUROC: `0.598809`.

CAP did not beat its derangements or the simpler self-only control. The
existing keeper-margin signal was substantially stronger than every CAP role.

## Replay, XAI, And Manual Review

Fresh-process replay completed in `5185.448 s`. Every numeric error was
`0.0`, every discrete array/check was exact, the formal process ID differed,
the ordered ledger reproduced exactly, and validation/test open counts stayed
zero.

Automatic XAI passed its locked numerical checks, but manual review rejected
the mechanism:

- all 20 locked rows were inspected at original detail;
- the cohort contained 10 keeper class-1 false positives, five false
  negatives, and five true positives;
- eight attribution maps had bbox mass below `0.75`, and seven were below
  `0.70`;
- 11 candidate-spatial score deltas had absolute value below `0.01`, and 17
  were below `0.05`;
- no reviewed row was suppressed, and no reviewed false-positive score fell
  below its fold threshold;
- region maps were coarse and often close to the spatial control, while
  feature gradients were diffuse or point-like.

The maps therefore do not establish a causal class-1 precision mechanism.

## Resource And Inference Boundary

The formal resource monitor passed its locked mechanism-only ceilings:

- peak CUDA allocation: `0.281028 GiB`;
- peak process RSS: `1.868843 GiB`;
- unexpected compute processes: zero;
- trainable CAP parameters per role: `1,090,979`.

These values are not a production inference contract. CAP A0 did not measure
matched batch-1 mean/p95 latency, batch throughput, end-to-end peak VRAM,
standard-op ONNX parity, or TensorRT feasibility. Even a scientific pass
would therefore have authorized only a separately locked inference smoke,
not integration. Because the scientific conjunction failed, no such smoke is
authorized.

The AAAI-2021 reference used standard transfer learning, pretrained weights
for faster convergence, and a 150-epoch recipe on a Titan V. It reports that
random initialization took nearly twice as many iterations and gives a
4.1-ms per-image figure for its own ResNet-50-based setup. Those numbers are
not transferable evidence for scratch TRKH on an RTX 4060 and do not relax
the local 30-epoch or deployment gates.

## Immutable Evidence

- Formal summary SHA:
  `824549d2683a1a462f66f235c8915e637ec2c9eb6404ee4a2aa589839419f6ee`.
- Formal manifest SHA:
  `a699fda3dab94e3d9805e81f2913dc2f9a1e632e33a2c980931b03d8b46e518e`.
- Replay summary/manifest SHAs:
  `e9f21c5932294cfcc2256b2ec1e2ba07b96d6e9a8dbde2317679b66c4ef447b4`
  and
  `de0c9aa69acf4dabcc598da8b75f83ff2a43818ad392fdfd4f963881c37f364c`.
- Manual review SHA:
  `197ca4920b0f62c0a5a7fb873e84854fd5ce6235486f1f3ab4b9b22851cfd760`.
- Final decision SHA:
  `00263f233eb6373fa2a69e6fdbbba2c2d8faba04ed6a3f56b8ada47d2ba2533b`.
- XAI contact-sheet SHA:
  `5f333cc4c955e9b1c36e606013794d1a477afb5e46a431592569c696d24e6f25`.
- Artifact-set files digest/manifest SHA:
  `ffc55117f10b63968f74ea8826dc95d27d0cc7b1ab0de3262bb146c07bd87b48`
  and
  `77469b559c95846b0de4d04a777e91bfc2b765eaf8bc8b2bff7d114f14c4978a`.

The complete retained artifact root is
`runs/audit_cap_integral_region_context_a0_20260724`; process logs are in
`runs/evidence_cap_a0_same_lock_rerun_20260724`.

## Closure Verification

- CAP-focused closure tests pass `28/28`.
- The complete repository suite passes `1925/1925` with 296 existing
  warnings in `114.75 s`.
- Research-process report revision 14 renders cleanly across all 14 pages,
  passes accessibility at `0/0/0`, and preserves exact table geometry.
- The post-closure read-only retention audit covers 860 run directories and
  all 51 valid object-schema compaction manifests. All 14 protected artifact
  checks exist, all 220 compacted originals remain absent, and there are zero
  blockers or deletions. Free space on the runs volume was `93.084 GiB`.
- Retention summary SHA:
  `99d1835a85246d6d9ad06768e4eab2d530b7bc06e8e3f6f686b67c080604a636`.
- Revision-14 JSON/DOCX/builder SHAs:
  `0b82a449de3335fbd579a1fa0f1f64f0ae1d14c82d68ce112bef9206ad7a869c`,
  `ae1efaf303be68ccf4920203bf5295dce92d6908e89085cefa01832f61f98ade`,
  and
  `7a0e81e4ebdacb0a1f7304c3e7936d76165363fc83ae65a8cefd99118d71002e`.

## Decision And No-Repeat Boundary

Reject the exact CAP A0. All downstream permissions are false: trainer
integration, validation smoke, validation, test, probe, full train, and
current-best command update.

Do not sweep the locked region count/order, captured layer, support resize,
pixel/region widths, LSTM width/order, NetVLAD clusters, optimizer, epoch,
seed, fold, threshold, derangement, or neighboring CAP readouts. Do not add
the script-only squeeze-excitation, spectral normalization, or sigmoid after
observing these results. Reopen this family only if a new equation-distinct
mechanism supplies local class-conditional surface evidence and is locked
prospectively against the keeper margin, TP-retention, inference, replay, and
XAI gates.

The current keeper and full-train command remain unchanged.
