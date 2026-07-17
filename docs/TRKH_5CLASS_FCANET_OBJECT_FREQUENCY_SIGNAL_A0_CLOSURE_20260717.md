# TRKH 5-Class FcaNet Object-Frequency Signal A0 Closure - 2026-07-17

## Decision

ICCV-2021 FcaNet object-aligned top-16 frequency signal A0 is closed before
trainer/model/config integration. The isolated implementation is correct and
deployment-safe, but the locked fit-only signal gate fails materially:

- clean `object_top16` AUROC is `0.578896`, below the `0.65` gate and below
  `object_gap=0.629377` by `-0.050481`;
- clean TP retention is only `0.726841`, versus `0.852732` for object GAP;
- only fold 4 improves over object GAP; the four deltas are
  `-0.109325/-0.081250/-0.056169/+0.034779`;
- all three shifted-condition AUROC deltas are negative, and candidate TP
  retention falls as low as `0.593824` under low contrast;
- clean restricted-FP rejection rises by `+0.112903`, but precision changes
  only `0.723790 -> 0.725118` because `53` additional protected class-1 TP are
  rejected. Clean F1 falls `0.782988 -> 0.725979`.

No short pair, validation, test, full train, nearby sweep, or current-best
command update is authorized.

## Authority And Locked Inputs

- Accepted paper: Qin et al., "FcaNet: Frequency Channel Attention Networks,"
  ICCV 2021.
- Official MIT repository commit/tree:
  `aa5fb63505575bb4e4e094613565379c3f6ada33` /
  `85aa7989ca7d957e1ab87991c65eefdb7fb53a8d`.
- Accepted-paper/source/license SHA-256 values:
  `13c707b575722cdb81003ac2e1bcd6650eb2ef5984265666d495662766f69d55` /
  `9b4337558604958dca1257f1b05e92fbee8f546e536604de5cff83f2a3d0200e` /
  `31e61e165ea1409c53836aadf965a601a0c225aa7c1aaa4eb0c01e18559e4cf1`.
- Prospective protocol SHA-256:
  `b28f4330c3cbaddf82cca9c0da222103a4199b7083b0cfde914b4607cc726283`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Frozen declaration SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Auditor/tests/PowerShell implementation commit: `8aea4f7`.

No FcaNet, ImageNet, ResNet, or other pretrained weight was used. User-provided
research reports remained hypothesis sources; the accepted paper, official
source/license, and direct TRKH replay controlled the decision.

## Geometry And Engineering Results

The CPU-first geometry preflight passed over the exact `607 = 421 TP + 186 FP`
cohort before model inference. Bbox bytes matched
`e9b2143c9dbc7c6f5bf3a38f80a43483bd436e3b5ce99116c3d5989f79892f6b`.
At `16x16`, object/context center supports remained `28..210/46..228`; at
`32x32` they remained `116..900/124..908`, with no empty region.

Every structural and deployment check passed:

- official frequency indices, official output, official DCT weights, and
  gradients are exact;
- independent float64 output error is `8.88e-16`, finite-difference error is
  `1.87e-11`, and GAP proportionality error is `0`;
- project bbox-geometry-control parity error is `0`;
- the precision-safe BF16 path keeps bbox coordinates in FP32 and has maximum
  descriptor error `0.019581 < 0.05`, with finite nonzero gradients;
- standard ONNX Runtime error is `3.58e-7`; TensorRT parse/build passes and no
  ONNX or engine binary is retained;
- full normal-evaluator runtime/memory ratios are
  `1.114320x/1.000726x`, below `1.15x/1.10x`;
- the side descriptor executes at `[32,256]` while normal logits and
  predictions remain exactly unchanged;
- canonical batch-64 declaration replay reproduces only the known sample
  `3657` near-tie and no other mismatch.

Development diagnostics exposed and fixed three auditor issues before formal
execution: source-order FP32 filter construction, BF16 coordinate/accumulation
precision, and use of the established metadata-aware normal evaluator. The
runtime sample count was increased from three to seven, still within the
prospective "at least three" rule, after an isolated extractor measurement
showed the first three-sample ratio was dominated by timing noise. The sole
formal measurement above used five warm-ups and seven samples.

## OOF Signal Failure

All readouts use the same four source-disjoint folds, clean-fit
standardization, `lbfgs`, `C=0.05`, and clean-fit recall-constrained threshold.
The independent CSV replay reproduces all summary values exactly over `2,428`
condition rows with maximum numeric difference `0`.

| Condition | object GAP AUROC | object top16 AUROC | Delta | top16 TP retention | top16 FP rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | `0.629377` | `0.578896` | `-0.050481` | `0.726841` | `0.376344` |
| dim | `0.600375` | `0.578819` | `-0.021556` | `0.679335` | `0.397849` |
| bright | `0.610362` | `0.544262` | `-0.066100` | `0.812352` | `0.241935` |
| low contrast | `0.586737` | `0.567926` | `-0.018811` | `0.593824` | `0.494624` |

Object top16 does beat the bbox-geometry control by `+0.088729` and all 16
frequency groups are finite/nondegenerate. That is insufficient: it loses to
both object GAP and full top16, fails three of four folds, fails every lighting
delta, and converts FP rejection into broad class-1 suppression rather than a
precision-safe boundary signal.

## Visual Review

All four hash-locked contact sheets were reviewed: eight TP and eight
restricted FP rows at each positive/negative normalized advantage extreme,
`32/32` rows total. Source images, transformed boxes, and aligned crops are
complete and correctly ordered.

Every selected row reports strongest raw DCT group `0`, frequency `(0,0)`.
The positive-advantage TP and FP pages share green, spotted, rough, and bright
reflection patterns. The negative-advantage TP page contains valid yellow or
smoother class-1 fruit that top16 strongly suppresses; the negative-FP page
contains visually similar smooth/green non-class1 fruit. Standardized OOF
readouts can still use non-DC channels, but the contact sheets provide no
class-conditional visual evidence that could override the failed numeric gate.

The pre-review summary SHA was
`21b11ed90739b7acccf216e007039ee77709e75108cdfe0b4ec3a29a71c831cf`.
Hash-locked manual review was finalized as `fail`; final summary SHA is
`54330c3d0f82813c839fbb19425893c1651bebd2bb913756c17a02e979b15b62`.

## Retained Evidence

The complete compact evidence remains at
`runs/audit_fcanet_object_frequency_signal_a0_20260717`:

- final `summary.json`:
  `54330c3d0f82813c839fbb19425893c1651bebd2bb913756c17a02e979b15b62`;
- `oof_scores.csv`:
  `135c7f1802485f3a6e6c1e182f1015c02f9bee120901fd1969d4d163c15cfae6`;
- `visual_review/selection.json`:
  `34159657e05601052cda9e5f3ff067a89c618a153b962198058f73cae21aced2`;
- final artifact manifest:
  `e7cc278b1027d8fd8e5e5c94396d43f47f3f547b99ea3a5930a9009bb1ebdc84`.

The manifest payload is `3,807,872` bytes and contains no checkpoint, feature
cache, ONNX, engine, copied raw image, or trainable artifact. Nothing in this
formal directory is large or obsolete enough to justify deletion.

No raw dataset file was modified. Fold 0, official validation, and test were
not loaded. No image-model epoch ran.

## No-Repeat Rule

Do not revisit on the current keeper:

- FcaNet top1/top-k frequency-count, frequency-family, channel grouping,
  frequency selection, layer, crop, alignment, residual, or attention sweeps;
- post-hoc removal/downweighting of group 0, DC subtraction, high-pass-only
  readouts, or selection of visually attractive frequency groups;
- readout C/solver/class weight, threshold/recall target, fold, seed, condition,
  normalization, bbox policy, or cohort changes;
- adding FcaNet attention to the trainer and hoping end-to-end learning rescues
  a representation that failed the prospectively required pre-training gate;
- combinations with closed FFT/Gabor/wavelet/PDC/texture, activation, GRN,
  localization, token-routing, loss, distillation, or post-hoc families.

The next route must be distinct accepted primary work with licensed official
code and must prove a precision-safe class-1 boundary signal before trainer
integration. It cannot win only by rejecting more class-1 predictions.

## Verification And Retention

Current-best command/history SHA-256 values remain
`36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
`39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
They remain at three revisions and two actual updates.

Focused tests passed `8/8`; full pytest passed `1372/1372`. Pycompile,
pyflakes, PowerShell parsing, VS Code-safe preflight, official equations,
ONNX/TensorRT, independent replay, and manual visual finalization all passed
their infrastructure checks.

Read-only all-manifest retention at
`runs/audit_trkh_artifact_retention_post_fcanet_all_manifests_20260717`
passed over `745` directories and `48` compaction manifests. All `210`
compacted originals are absent, `blockers=[]`, no deletion/raw-data touch
occurred, and free space is `103.339 GB`. Retention summary SHA-256 is
`b93c0adcab80a72c1af5ff86704f7b05a5c73ae93488c4815a1bd8eb7cd85d55`.
