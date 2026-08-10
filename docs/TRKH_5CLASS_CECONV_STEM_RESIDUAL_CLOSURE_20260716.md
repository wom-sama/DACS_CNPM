# TRKH 5-Class CEConv Stem Residual Closure

Date: 2026-07-16
Decision: reject the locked first-block CEConv residual; Stage B denied

## Scope and provenance

- The immutable protocol is
  `docs/TRKH_5CLASS_CECONV_STEM_RESIDUAL_READINESS_PROTOCOL_20260716.md`,
  SHA-256
  `9c5af4e9886ec2068f5e88899f25f54c9b06e870538b31f6158cce370da57082`.
- Primary evidence is the NeurIPS-2023 CEConv paper and official repository
  commit `8f46c78c3a7cf91ad905d0255b756d13cd1e0c94`. The paper, official
  `ceconv2d.py`, pooling source, and license SHAs were precommitted before the
  formal run.
- Only source-disjoint `yolo_f/train` fit/holdout rows `7372/1843` were used.
  The balanced fit cohort contained all `432` fit class-1 rows plus the exact
  `186` restricted `0/2/4 -> 1` hard negatives. Validation and test were never
  constructed.
- Candidate and identity control wrapped only the keeper's first stem block.
  The native path remained intact and received a zero-initialized, per-channel
  gated residual. The candidate used three fixed `0/120/240`-degree RGB group
  rotations; the control repeated identity three times. Their 960 trainable
  adapter parameters and initial logits were bit-exact.
- Both variants used the same 60 balanced batches, AdamW `lr=5e-4`, weight
  decay `0.01`, five-class CE, no augmentation, deterministic ordering, and
  frozen keeper parameters.

## Structural and training audit

- Official filter-equation replay had maximum absolute error `0`. The RGB
  rotation matrix had determinant `1`, orthogonality error `1.11e-16`, and
  three-cycle error `5.00e-16`.
- Candidate hue-cycle equivariance error was `1.788e-7`; the identity control
  error was `0.559132`. Candidate/control state differed initially only in the
  fixed rotation buffer, while all trainable tensors were bit-exact.
- Zero-gate raw/candidate/control stem outputs and initial logits were
  bit-exact. The prior keeper argmax also matched exactly.
- Candidate loss changed `1.331440 -> 1.302057`; control loss changed
  `1.331440 -> 1.300438`. All trainable gradients were finite and observed,
  all adapter tensors moved, and every frozen keeper tensor remained bit-exact.
  Candidate gate min/mean/max was
  `-0.029276/-0.000449/+0.030342`.

## Locked train-holdout result

| Condition | Variant | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | Restricted FP |
|---|---|---:|---:|---:|---:|---:|
| clean | raw keeper | 0.949323 | 0.748252 | 0.981651 | 0.849206 | 36 |
| clean | CEConv | 0.947069 | 0.732877 | 0.981651 | 0.839216 | 39 |
| dim | raw keeper | 0.825093 | 0.535354 | 0.486239 | 0.509615 | 46 |
| dim | CEConv | 0.825231 | 0.495726 | 0.532110 | 0.513274 | 58 |
| bright | raw keeper | 0.872779 | 0.565517 | 0.752294 | 0.645669 | 62 |
| bright | CEConv | 0.865458 | 0.525316 | 0.761468 | 0.621723 | 72 |
| low contrast | raw keeper | 0.852525 | 0.538462 | 0.577982 | 0.557522 | 53 |
| low contrast | CEConv | 0.846413 | 0.496183 | 0.596330 | 0.541667 | 64 |

- On clean holdout, CEConv changed macro/class-1 F1 by
  `-0.002254/-0.009991`, lowered class-1 precision `-0.015375`, kept recall
  unchanged, and created three net restricted false positives. Its 11 changed
  decisions contained only `4` corrections, `6` harms, and `1` neutral change.
- The harmful class-1 expansion included three previously correct class-2
  rows becoming class 1 and one correct class-4 row becoming class 1. No clean
  class-1 true positive was broken, but this is not useful recall protection:
  support widened into non-class-1 fruit.
- Against the parameter-matched identity control, CEConv still lost macro/
  class-1 F1 `-0.000368/-0.003304`, lost class-1 precision `-0.005054`, and
  added one restricted false positive. The color-group mechanism therefore
  did not provide benefit beyond learning the same residual adapter.
- Under dim/bright/low contrast, class-1 precision fell
  `-0.039627/-0.040201/-0.042278` versus raw and restricted false positives
  increased by `12/10/11`. The small dim recall/F1 gain came from broader
  class-1 support and fails the agricultural precision gate.

## XAI and deployability

- Deterministic CPU FP32 stem Grad-CAM covered every clean correction and harm
  selected by the protocol. Raw, identity-control, and CEConv Grad-CAM maps
  were nearly identical, so the residual did not create a new selective
  object-localization mechanism.
- The three hue-group energy maps differed, but the winning group followed
  both fruit surface and nearby background/border structure. This agrees with
  the observed `2/4 -> 1` harms and illumination-dependent false-positive
  expansion.
- Candidate backward time was `0.263142 s` versus raw `0.070426 s`, ratio
  `3.73645x`, above the locked `1.35x` limit. Peak VRAM was `3.38082 GiB`,
  above the `3.25 GiB` gate.
- Isolated and full static-batch-one ONNX exports succeeded. Maximum absolute
  ONNX Runtime errors were `2.861e-6` and `6.296e-7`, and both argmax checks
  passed. Exportability does not override the behavioral/resource rejection.

## Replay, artifacts, and engineering closure

- Independent CSV replay reproduced all `4 x 1843 = 7372` predictions,
  comparisons, confusion-derived metrics, and transitions exactly.
- Evidence is under
  `runs/audit_ceconv_stem_residual_readiness_20260716`. Its nine manifest
  payloads total `35,273,011` bytes. Summary/prediction/artifact-manifest/XAI-
  manifest SHAs are respectively
  `8177f9769aa3385dbb399e5b42100d0b46d8145d30550306361a8f22f63bca7a`,
  `0bf22a6415501ab39f5c6374976fac44cbb83eb1783141c1f742dca9d018c967`,
  `96fc5719cabf239cf6c20a7ac0afe3cad4f8f4403965dc9b6078dd8e6e1f5972`,
  and
  `20472285119e4df24f5b65ec66460c5226b86d77b83be6c935ff1f6f04d6c8b7`.
- Manifest inventory, byte sizes, and payload hashes were recomputed exactly.
  No checkpoint, TensorRT engine, validation/test prediction, trainable model
  binary, or raw-data write was produced.
- Two earlier formal invocations failed after decision-independent audit code
  reached deterministic adaptive-pooling backward paths. Their partial output
  roots were inspected and removed. Fixes affected only matched resource
  benchmarking/evaluation reuse and isolated CPU Grad-CAM; they did not change
  the locked data, model, initialization, training, conditions, metrics, or
  gates. No failed-run metric was used to alter the protocol.
- Compilation passed; focused tests passed `6/6`; full pytest passed
  `1182/1182` with `257` warnings in `47.53 s`. PowerShell parse and real
  preflight passed without creating a new output directory.
- Read-only retention audit
  `runs/artifact_retention_audit_ceconv_stem_residual_closure_20260716`
  passed over `701` run directories with `blockers=[]`, deleted nothing, and
  reported `72.191 GiB` free. Summary SHA-256 is
  `ae5266f5d42c97436745bb27fa24a6c381f010f3563dd7c27a8ad525f773f426`.
- Keeper, scratch complement, command packet, and command-history hashes remain
  exact. Command tracking remains three revisions and two updates after the
  initial revision; no CEConv command revision occurred.

## Decision and no-repeat boundary

Stage B, validation, test, probe, full train, shared-trainer integration, and
current-best command promotion are denied. Do not sweep rotations, group size,
gate initialization, adapter width, branch normalization, pooling, insertion
block, learning rate, weight decay, steps, batch balance, fold, seed, or nearby
color-group residual fusions on this keeper. The clean and illumination result
shows that this mechanism broadens class-1 support when the required direction
is conservative false-positive control.

This is a rejection of the exact first-block residual transfer and nearby
parameter sweeps, not of every possible CEConv architecture. Reopening a full
CEConv backbone would require a new primary-source protocol that independently
resolves the paper's much longer training/resource assumptions, scratch
`<=30`-epoch convergence, and class-1 precision/TP protection. The next screen
must use a genuinely different supervision or representation mechanism rather
than another color-invariance residual.
