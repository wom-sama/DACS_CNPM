# B7 train-only readiness: pair-conditioned DINOv3 prefix residuals

Protocol: `TRKH_PRETRAINED_CLASSF_B7_PREFIX_RESIDUAL_READINESS_20260731`

This protocol is locked before B7 feature extraction or head training. B7 is a
train-only, source-grouped readiness experiment. It never constructs or
enumerates validation or test data, and a pass does not constitute a model
performance claim.

## Scientific question and scope

B6 tests a nonlinear distribution readout over all 256 patch tokens. B7 asks a
different, cheaper question: after subtracting the deployed patch mean, do the
DINOv3 CLS token and four register tokens retain pair-specific information
about class 1 that is absent from the global logits and from a zero-residual
matched control?

The representation comes from the exact B2 tempered-p0.5 EMA best checkpoint
(epoch 4), SHA-256
`4d3205e7029fa25c313ac622f3ea1abbe0bbf4103d804ee9ac87f527759352a6`.
The only data view is `configs/class_f_5class_dev.yaml`, SHA-256
`fa9581d1a595134bd366105c3999170d3553712099e7222cf67f2339f84da156`,
with exactly 8,278 train crops and 7,751 normalized source groups. Class 1 is
positive and rivals are classes 0, 2, and 4. The encoder was fitted on this
complete train split, so B7 measures representation-conditional incremental
alignment evidence rather than causal evidence or unbiased new-source
generalization.

## Locked representation and cache

- DINOv3 output must contain exactly five prefix tokens (CLS followed by four
  registers), 256 patch tokens, and width 384.
- Preflight fails closed on timm `1.0.27`, runtime class
  `timm.models.eva.Eva`, `cls_token=[1,1,384]`,
  `reg_token=[1,4,384]`, and a synthetic-zero FP32 forward output of
  `[1,261,384]`. This architecture probe uses no dataset image.
- Extraction is FP32. The arithmetic mean of the 256 patch tokens must match
  the deployed average-pool pre-logits within maximum absolute error `1e-6`.
- Cache only the five prefix tokens and patch mean as NumPy float16, global
  five-class logits as float32, and labels as int64. Full patch tokens are not
  persisted.
- Cache-manifest schema 2 locks data/checkpoint hashes, array shapes/dtypes and
  file hashes, ordered image-path hash, label-content hash, normalized
  source-group hash/count, prefix order, pool parity, and quantization error.
- Cache and outputs must be outside the canonical image tree.

## Locked matched readouts

All three readouts have exactly 7,345 trainable parameters and identical
initialization, pair-row order, folds, labels, loss, optimizer, and seed.
LayerNorm is non-affine. For prefix token `t_j`, patch mean `m`, and pair
embedding `e_r` (all learned widths are 16):

```text
z_j = Linear_384x16(LayerNorm_no_affine(t_j - m))
descriptor_r = concat(z_CLS,
                      mean(z_register_1..4),
                      variance(z_register_1..4),
                      frozen_global_logits_5,
                      e_r)                         # width 69
pair_logit = Linear_16x1(GELU(Linear_69x16(descriptor_r)))
```

- Candidate uses aligned CLS/register residuals.
- Matched control replaces all five prefixes by that sample's patch mean, so
  all pre-normalization residuals are exactly zero.
- Deranged falsification keeps each sample's global logits and patch mean
  aligned, but substitutes the complete five-token residual block from one
  different normalized source. This preserves joint CLS/register structure
  (including register variance) while breaking its alignment to the target
  sample. Fit and holdout use independent, label-blind, source-disjoint
  bijections; no residual crosses a fold boundary and each map has zero fixed
  points.
- The shared 384-to-16 projection is evaluated once per image. Projecting five
  prefixes plus evaluating all three pair heads costs 34,080 linear MACs per
  image, locked below 50,000.

## Locked folds, loss, and optimization

- One five-fold `StratifiedGroupKFold`, shuffle enabled, seed `20260731`, is
  assigned globally over all five classes and reused for all pairs/readouts.
- A class-1 sample expands to all three pair tasks; a rival sample expands only
  to its corresponding task. The six `(pair, binary target)` strata have equal
  total BCE weight.
- Each readout uses seed `20260731 + fold`, FP32, exactly 20 epochs, AdamW
  `lr=1e-3`, `weight_decay=1e-4`, batch 128, clip norm 1.0, no scheduler,
  dropout, early stopping, or sweep. Threshold is fixed at 0.5.
- In addition to weighted BCE, all three readouts use the same fixed
  train-only augmented primal-dual soft-recall constraint. For every pair `r`
  represented by positive rows in a minibatch:

```text
Rsoft_r = mean(sigmoid(logit_i) | pair_i=r, y_i=1)
g_r     = 0.95 - Rsoft_r
L       = weighted_BCE
          + mean_r[lambda_r * g_r + 0.5 * 1.0 * relu(g_r)^2]
lambda_r <- clip(lambda_r + 0.05 * stop_gradient(g_r), 0, 10)
```

  Multipliers start at zero at each fold. These constants and the formula are
  immutable; there is no recall-target, penalty, dual-rate, threshold, seed,
  width, or epoch sweep.

## Predeclared readiness gate

Every integrity and provenance check must pass: exact data/checkpoint hashes,
8,278 rows, 7,751 groups, exact cache schema/shapes/dtypes/hashes, five complete
group-disjoint folds, 15 pair-fold cells, finite data/loss/logits/probabilities,
ten fold/partition block-derangement contracts with zero fixed points, equal 7,345
parameter counts, and at most 50,000 all-pair head MACs.

Incremental candidate evidence over control must satisfy all:

- mean pair AUROC gain `>= 0.010`;
- positive AUROC gain in at least `10/15` pair-fold cells;
- AUROC gain `>= 0.010` in at least `2/3` pairs and no pair below `-0.005`;
- class-1 F1 gain `>= 0.010` in at least `2/3` pairs;
- aggregate class-1 TP retention `>= 0.990`;
- every pair class-1 recall delta `>= -0.005`;
- aggregate rival-to-class-1 FP reduction `>= 10%`;
- critical pair `2-1`: AUROC gain `>= 0.010`, FP reduction `>= 10%`, and
  recall delta `>= -0.005`.

Falsification must also show candidate-minus-deranged mean pair AUROC
`>= 0.008` and candidate AUROC wins in at least `10/15` pair-fold cells.
Thresholds cannot be relaxed after output is read. Passing only licenses a
separate default-off implementation protocol; it grants no validation, full
training, test, or deployment permission. Failure closes this exact B7 route
without a nearby hyperparameter sweep.

## Artifacts and commands

The run retains `preflight.json`, schema-2 `cache_manifest.json`,
`global_fold_assignments.csv`, `pair_fold_metrics.csv`, `pair_metrics.csv`,
`train_oof_pair_predictions.csv`, fold readout checkpoints, and `summary.json`.
Every summary states that train was used and validation/test were not used or
constructed.

The launcher is intentionally inert unless `-Run` is supplied:

```powershell
& scripts\run_trkh_pretrained_classf_b7_prefix_residual_readiness.ps1
```

Only after reviewing preflight may the bounded train-only audit be launched:

```powershell
& scripts\run_trkh_pretrained_classf_b7_prefix_residual_readiness.ps1 -Run
```
