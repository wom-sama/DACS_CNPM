# B6 train-only readiness: class-conditional DeepSets over DINOv3 patches

Protocol: `TRKH_PRETRAINED_CLASSF_B6_DEEPSETS_MIL_READINESS_20260731`

This document is locked before any B6 feature extraction or head training. B6
is a diagnostic readiness experiment, not a validation probe, full train, or
deployment claim. It may construct only the canonical `class_f/train` dataset.
Validation and test datasets must never be constructed or enumerated.

## Question and nearest closed controls

B3a showed that fixed summaries after projecting every patch to five global
class logits add no useful information. B3b kept the raw 384-dimensional patch
tokens and found a small but insufficient signal: mean pair AUROC increased by
`0.002238`, `13/15` pair-fold deltas were positive, aggregate class-1 TP rose
`1365 -> 1376`, and rival-to-class-1 FP fell `262 -> 245`. B5 then showed that
global partial-order supervision can reduce `2 -> 1`, but only by losing true
class-1 positives.

B6 asks one narrower question: can a tiny nonlinear, pair-conditioned,
permutation-invariant distribution readout extract information from frozen raw
DINOv3 patch tokens that an exactly capacity-matched pooled-token readout
cannot?

B6 is materially different from the rejected patch router family. It has no
top-k or max-instance selector, attention gate, crop-bbox prior, patch-logit
margin, zero-initialized residual, global confidence router, teacher target, or
threshold sweep.

## Immutable data and representation contract

- Dataset view: `configs/class_f_5class_dev.yaml`, train split only.
- Required support: exactly `8,278` crops and at least `7,000` normalized
  source groups.
- Source group: case-folded filename stem after removing `_boxNNN`.
- Focus class: `1`; rivals: `0`, `2`, and `4`; class 1 is positive.
- Frozen representation: B2 tempered-p0.5 EMA best checkpoint, epoch 4,
  `vit_small_patch16_dinov3.lvd1689m`, average pool, 256 patch tokens of width
  384, checkpoint SHA-256
  `4d3205e7029fa25c313ac622f3ea1abbe0bbf4103d804ee9ac87f527759352a6`.
- Data-YAML SHA-256:
  `fa9581d1a595134bd366105c3999170d3553712099e7222cf67f2339f84da156`.
- DINO extraction is FP32. Raw patch tokens may be persisted only as an
  output-directory NumPy float16 memory map with a manifest containing shape,
  dtype, paths, labels, source groups, checkpoint hash, and quantization error.
- Before quantization, the mean of raw patch tokens must match the deployed
  average-pooled pre-logits with maximum absolute error `<= 1e-6`.
- Cache and all outputs must be outside the canonical image tree.

The B2 encoder was trained on the complete train split. Therefore this audit is
a representation-conditional matched comparison, not an unbiased estimate of
new-source generalization. A pass licenses only the next implementation step;
it cannot be reported as final model performance. A later scientific claim
requires a fold-specific-encoder audit or an independently frozen foundation
representation replication.

## Locked candidate and matched control

Both readouts have the same `10,577` trainable parameters, initialization,
optimizer, example order, loss, folds, and global logits. Layer normalization
is non-affine.

For pair `r-1`, with raw patch token `x_i` and a learned 24-dimensional pair
embedding `e_r`:

```text
u_i = GELU(Linear_384x24(LayerNorm(x_i)) + e_r)
bag = concat(mean_i(u_i), variance_i(u_i), frozen_global_logits_5, e_r)
pair_logit = Linear_16x1(GELU(Linear_77x16(bag)))
```

- Candidate: the bag contains all 256 raw patch tokens.
- Matched pooled control: the bag contains 256 repetitions of the deployed
  average-pooled patch token.
- The 384x24 projection is computed once and shared across all three pair
  conditions. The additional matrix-multiply budget is at most `2.6M` MACs per
  image and the head budget is at most `12,000` parameters.
- A class-1 row contributes one example to each of the three pair tasks. A
  rival row contributes only to its corresponding task.
- BCE is weighted so that each of the six `(pair, binary_target)` strata has
  equal total weight. There is no focal, margin, ranking, distillation, or
  auxiliary loss.

## Locked OOF and optimization contract

- One global five-class `StratifiedGroupKFold`, five folds, shuffle enabled,
  seed `20260731`; the exact global assignment is reused for every pair and
  every readout.
- Candidate and control use identical seed `20260731 + fold` and the same
  expanded-row order.
- Head training is FP32 for exactly 20 epochs, AdamW `lr=1e-3`,
  `weight_decay=1e-4`, batch size 128 expanded pair rows, gradient clip 1.0,
  no dropout, no scheduler, no early stopping, and final epoch only.
- Prediction threshold is fixed at `0.5`. No architecture, width, optimizer,
  epoch, seed, threshold, fold, pair, or cache-precision sweep is permitted.
- One falsification readout uses candidate tokens but independently permutes
  fit labels within each pair using seed `20260731 + fold`; holdout labels are
  never permuted and the permutation cannot affect candidate/control training.
- Holdout rows and source groups never participate in optimization, threshold
  selection, or model selection.

## Readiness gate

Implementation permission requires every check below. Thresholds cannot be
relaxed after reading B6 output.

Integrity and falsification:

- exactly 8,278 train rows, at least 7,000 groups, five complete global folds,
  exactly 15 candidate/control pair-fold outputs, zero source-group overlap,
  finite cache/features/logits/probabilities, and deployed-pool parity;
- token permutation changes a B6 logit by at most `1e-6`;
- candidate and control logits match within `1e-6` when every patch equals the
  pooled token;
- candidate and control parameter counts are equal and `<= 12,000`;
- estimated B6 head MACs are `<= 2.6M` per image;
- mean falsification AUROC is `<= 0.55`.

Incremental evidence over the matched pooled control:

- mean pair AUROC gain `>= 0.010`;
- positive AUROC direction in at least `10/15` pair-fold cells;
- AUROC gain `>= 0.010` in at least `2/3` pairs and no pair below `-0.005`;
- class-1 F1 gain `>= 0.010` in at least `2/3` pairs;
- no pair loses more than `0.010` class-1 recall;
- aggregate class-1 TP retention `>= 0.980`;
- aggregate rival-to-class-1 FP reduction `>= 10%`;
- critical `2-1` pair: AUROC gain `>= 0.010`, FP reduction `>= 10%`, and
  class-1 recall loss no worse than `0.010`.

A pass permits implementation of one default-off production specialist plus a
fold-safe bounded smoke protocol. It does not permit validation, full training,
test inference, or a mobile claim. A failure closes this exact DeepSets route
without a nearby width/loss/threshold sweep.

## Artifacts

The tool must retain `preflight.json`, `cache_manifest.json`,
`global_fold_assignments.csv`, `pair_fold_metrics.csv`, `pair_metrics.csv`,
`train_oof_pair_predictions.csv`, and `summary.json`. Every JSON summary must
state `train_split_used=true`, `validation_split_used=false`, and
`test_split_used=false`.

## Commands

The launcher is intentionally inert unless `-Run` is supplied. Its default
action is preflight only and it never has a validation/test/full mode.

```powershell
& scripts\run_trkh_pretrained_classf_b6_deepsets_mil_readiness.ps1
```

After reviewing preflight and disk capacity, the bounded train-only readiness
audit is explicitly launched with:

```powershell
& scripts\run_trkh_pretrained_classf_b6_deepsets_mil_readiness.ps1 -Run
```
