# TRKH 5-Class Global Response Normalization A0 Closure - 2026-07-17

## Decision

CVPR-2023 ConvNeXt-V2 Global Response Normalization A0 is closed before OOF
readout fitting or trainer integration. The locked full/object/outside protocol
has two independent structural failures:

1. `13/607` fixed cohort rows have no outside patch under the prospectively
   locked rule that every `16x16` patch cell intersecting the transformed bbox
   belongs to the object region.
2. The active matched block-2 FFN has median inference runtime ratio
   `1.353753x`, above the locked `1.15x` limit.

Do not change the mask to patch-center membership, erode the bbox, exclude the
13 rows, relax the runtime gate, or sweep GRN layer/norm/epsilon/initialization/
optimizer/threshold/condition variants. No five-epoch pair, validation, test,
full train, current-best command update, or GRN combination is authorized.

## Authority And Locked Inputs

- Accepted paper: Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets
  with Masked Autoencoders," CVPR 2023.
- Official repository commit/tree:
  `2553895753323c6fe0b2bf390683f5ea358a42b9` /
  `0b23579ac3ded0c671f592def6f7b66300a80799`.
- Only the MIT GRN software equation was reimplemented. No CC BY-NC weight,
  FCMAE checkpoint, or ConvNeXt model was used.
- Prospective protocol SHA-256:
  `912a8f101dd41545439a8a235690c3eca29d0d89c5ea7a8a6c65bf92141d86cf`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Frozen declaration SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Implementation commit: `ed5c15e512b8af3231f2b160f7471e47a8df05e4`.
- Failure-preservation commit: `81b1605cb4a0fa5ad2394f84cc426e9d917354c9`.

User-provided research reports remained hypothesis sources. The accepted paper,
official source/license, and direct TRKH replay controlled the decision.

## Engineering Results

The source-grounded implementation itself was correct:

- official output error: `0`;
- independent float64 equation error: `4.44e-16`;
- maximum input/gamma/beta gradient error: `7.11e-15`;
- finite-difference error: `2.35e-10`;
- BF16 maximum error: `0.003837 < 0.01`;
- zero-initialized GRN identity error: `0`;
- added parameters: exactly `2,048`;
- peak-memory ratio: `1.000037x < 1.10x`;
- ONNX Runtime error: `2.98e-7 < 1e-5` with standard domains;
- TensorRT parse and serialized-engine construction: pass.

The first functional implementation used a full tensor concat and measured
`2.998x/1.300x` runtime/memory during development. A single pre-formal
engineering correction used the same equation in-place on the disposable GELU
output at inference. It reduced formal memory to `1.000037x` and runtime to
`1.353753x`, but runtime still failed materially. No additional optimization or
gate sweep was attempted.

## Mask Geometry Failure

The clean geometry replay covered all `607` rows in exact declaration order.
Object support ranged from `64` to `256` cells; `16` rows had at least `240`
object cells and `13` had exactly `256`, leaving zero outside cells. The 13 rows
contain `12` protected class-1 TP and one restricted class-0 FP, with fold
counts `3/2/5/3` for folds `1/2/3/4`. Their ordered-index SHA-256 is
`0a9f70dd3776b502b35309895820a39969702e44c80ca59b072641370e1b846b`.

This is not a missing-label or bbox parser error. The transformed bboxes are
large enough to touch every coarse patch cell under the locked intersection
rule. Because `n_out` is undefined for those rows, the exact 607-row
object-versus-outside causal control cannot be computed without changing the
prospective method after seeing data. The stop rule therefore applies before
feature extraction and OOF fitting.

## Failure Preservation

The first formal attempt at implementation commit `ed5c15e` stopped on the
first clean descriptor batch with:

`ValueError: Every bbox must leave at least one outside patch cell.`

It created an empty output directory but wrote no artifact. That directory was
not overwritten. Commit `81b1605` added a pre-OOF geometry scan and compact
failure writer without changing the mask, cohort, or gate. The recovery replay
is retained at:

`runs/evidence_grn_a0_mask_structural_failure_20260717`

Exact retained hashes:

- `mask_geometry.json`:
  `087f1a9989e1c359a051ad9e7db1816858d364257397656aa290c05fd7a62caf`;
- `summary.json`:
  `c49274b580139b5d197d8f244387f0a53cca73140099110dc9e674e1f077cbfa`;
- `report.md`:
  `77afb61111adc05c3542dac781602adcaaaa6fa497dca22543ab97a2a809aeda`;
- `artifact_manifest.json`:
  `05251283894055baa77efc52c5a511c934bfba45103311ed5aa7b1f2436c8811`.

The retained evidence contains three compact JSON/Markdown payloads totaling
`28,344` bytes plus the manifest. It contains no checkpoint, ONNX, engine,
feature cache, NumPy cache, or image copy.

## Why OOF And XAI Did Not Run

OOF scores, condition metrics, sample-3657 declaration sensitivity, and contact
sheets are intentionally absent. They depend on the full/object/outside
descriptor contract, which is structurally undefined for 13 locked rows. A
partial-row OOF or a substituted outside mask would be a new post-hoc protocol,
not completion of this A0. Equation, deployment, and geometry evidence are
enough to reject trainer integration.

No raw dataset file was modified. Fold 0, official validation, and test were
not loaded. No image-model epoch or readout optimizer step ran.

## No-Repeat Rule

Do not revisit on the current keeper:

- GRN at another block or on another token subset;
- object/outside mask erosion, center-cell membership, fallback border rings,
  row exclusion, or response imputation;
- L1/other response norms, epsilon, gamma/beta initialization, LayerScale, or
  FCMAE/ConvNeXt variants;
- LR, optimizer, epochs, fold, threshold, condition, seed, or class-weight
  changes;
- compile/Triton/custom-kernel rescue or a relaxed runtime gate;
- combinations with closed activation, style, spatial-mixing, routing, edge,
  distillation, or loss families.

The next route must come from a distinct accepted primary source with licensed
official code and must validate all geometry assumptions before any expensive
feature/model pass.

## Verification And Retention

Current-best command/history SHA-256 values remain
`36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
`39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
They remain at three revisions and two actual updates.

The empty first-attempt directory was removed only after the recovery summary
recorded it. Cleanup manifest SHA-256 is
`675b6c16463248e42b46e7d15716b91c336091b74b0c00fe5e2245b9727a46ba`;
it records zero files, zero bytes, and verified deletion.

Compileall and pyflakes passed. Five relevant PowerShell wrappers parsed, the
focused GRN suite passed `11/11`, and full pytest passed `1364/1364`. Read-only
retention `runs/audit_trkh_artifact_retention_post_grn_20260717` passed over
`742` directories and `48` compaction manifests: all `210` compacted originals
are absent, `blockers=[]`, raw data was untouched, and free space is
`70.272 GB`. Retention summary SHA-256 is
`7f36d1f8cbfaabe5cac6e0998d7f15723d2d124c360a875c3845b28d3a0aa839`.
