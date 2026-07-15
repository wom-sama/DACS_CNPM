# TRKH 5-Class High-Resolution Shifted-Window Bridge Readiness Protocol - 2026-07-15

## Status

Precommitted before diagnostic implementation. This document locks one
train-only information test. Passing it authorizes implementation of one exact
default-off shifted-window bridge; it does not authorize validation, test,
smoke training, a full train, or a parameter sweep.

## Research Question

Does the current stride-8 CNN stem contain source-aligned local surface evidence
that is lost when the learned `2x2` patch projection reduces its `32x32` map to
the current `16x16` Transformer grid?

The proposed architecture is not another high-resolution input resize, image
tile ensemble, post-hoc classifier, or mature-token adapter. If the diagnostic
passes, the sole candidate will refine the stride-8 feature map with one regular
and one shifted local-window block before the existing patch merge. The current
global TRKH Transformer, registers, branch tokens, pruning, bbox priors, heads,
and XAI path remain intact.

## Primary-Source Lock

- Swin paper: Liu et al., "Swin Transformer: Hierarchical Vision Transformer
  using Shifted Windows", ICCV 2021, `https://arxiv.org/abs/2103.14030`.
- Paper SHA-256:
  `dfdc7631fe2a35bb5892080b9c0eeadc4432d55cf1856f84bc88802efa877a7c`.
- Official repository: `https://github.com/microsoft/Swin-Transformer`.
- Locked official commit:
  `f82860bfb5225915aca09c3227159ee9e1df874d`.
- Reviewed official file: `models/swin_transformer.py`.
- Reviewed file SHA-256:
  `1e91f71a737c8c8ff0874b65e5a5a4d6b50c65e98250a0a634e75ced12a2ff8f`.
- Official license SHA-256:
  `9906940f61b1f0b533fa7d99baf55178b2808fbe113ea51dfbfad8572ccd5f2b`.
- Conv-stem optimization reference: Xiao et al., "Early Convolutions Help
  Transformers See Better", `https://arxiv.org/abs/2106.14881`; paper SHA-256
  `76e8ab2d946a4e215e87fa8370ac2ca23b8d47abab43956a6c803f9dab2529c7`.

The official Swin implementation alternates non-overlapping window attention
and half-window shifts, then uses patch merging between resolutions. Its default
training recipe is 300 epochs with 20 warmup epochs, mixup, and CutMix. That is
adverse transfer evidence for the local no-pretrain, at-most-30-epoch constraint;
therefore no architecture code or GPU smoke is allowed without the A0 signal.
The current TRKH already has a three-block convolutional stem, so the hypothesis
is specifically about preserving its stride-8 detail, not adding a conv stem.

## Local No-Repeat Boundary

The following nearby routes are already closed and must not be repeated:

- image-size 320/384 training or inference and simple 256+320 blending;
- original-resolution surface tiles and local zoom/image experts;
- checkpoint-adapter Shifted Patch Tokenization over diagonal shifted views;
- high-frequency, Gabor, SRM, Deep-TEN, fixed first/second-order, and compact
  bilinear heads;
- local patch mixers, LeFF, GPSA, MogaNet, InceptionNeXt, StarNet, OctConv,
  CoAtNet, MobileViT, EdgeNeXt, and stock Swin-T;
- verifier/router/calibration, class suppression, or a new post-hoc threshold.

Those failures do not directly test trainable local attention on the native
`32x32` stem map before the lossy `2x2` merge. A0 tests only whether aligned
within-cell detail adds information beyond the exact existing merge output.

## Locked Inputs

- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`.
- Data YAML SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Reused train-only fold declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`.
- Fold CSV SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Companion summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.

Use exactly the `9215` clean rows, source groups, labels, keeper probabilities,
and fold assignments in that CSV. Expected fold counts are
`1843/1830/1828/1851/1863`, with `8064` source groups and zero source overlap
between each fold and its complement. Expected class counts are
`1941/541/1920/2520/2293`. Recomputed clean keeper argmax must match all locked
rows; its maximum probability error is recorded as a transform-parity audit.
The frozen keeper was trained on these
rows; only the readout is source-group held out. This is a readiness diagnostic,
not OOF backbone evidence, a paper metric, or a model-promotion result.

## A0 Descriptor Lock

Run the checkpoint's exact clean evaluation transform on `yolo_f/train` only.
Do not construct validation or test datasets. Extraction is locked to CUDA
FP32, batch size `96`, four data-loader workers, seed `20260715`, deterministic
row order, cuDNN benchmark off, and TF32 off.

1. Capture the legacy stem output, required to be `[B,256,32,32]`, and the
   current learned patch projection output, required to be `[B,256,256]` or a
   `16x16` grid of 256-dimensional tokens.
2. Layer-normalize every patch token and apply one fixed QR-orthogonal
   `256 -> 32` channel projection with seed `20260715`.
3. Layer-normalize every stride-8 stem location and apply one independent fixed
   QR-orthogonal `256 -> 16` projection with seed `20260716`.
4. Partition the `32x32` projected stem into the exact non-overlapping `2x2`
   cells consumed by the patch projection. For each cell compute three
   orthonormal signed Haar bands, each divided by `2`:
   horizontal `(z00-z01+z10-z11)`, vertical `(z00+z01-z10-z11)`, and diagonal
   `(z00-z01-z10+z11)`. The discarded low-pass band is not a candidate input.
5. Build a valid crop-bbox interior mask on the `16x16` grid with fixed erosion
   ratio `0.08`. For both low-resolution and detail maps, concatenate full-mask
   mean, full-mask standard deviation, and means in the four fixed image-grid
   quadrants. Add the five shared region-coverage ratios.
6. The shared descriptor is exactly `197` values: `32 * (2 + 4)` pooled
   low-resolution values plus five coverages. The detail descriptor is exactly
   `288` values: `(3 * 16) * (2 + 4)`. Every readout input is exactly `485`
   values.

Predeclared variants:

- `control`: shared descriptor plus 288 exact zeros;
- `candidate`: shared descriptor plus the aligned Haar-detail descriptor;
- `placebo`: shared descriptor plus the same detail rows deterministically
  permuted within the current fit partition and, separately, within its holdout
  partition. The permutation must have zero same-source fixed points and may
  never cross fit/holdout boundaries.

There is no descriptor, projection-rank, erosion, pooling, fold, or permutation
sweep.

## A0 Readout Lock

For each of the five predeclared CIDT folds:

- fit on the other four folds and predict only the held-out fold;
- standardize each input column using fit rows only; constant columns use scale
  one;
- start one 5-class residual linear head at exact zeros for every variant;
- compute `softmax(log(keeper_probability) + 0.10 * residual)`;
- optimize natural-frequency cross entropy plus `1e-3 * mean(weight^2)` using
  deterministic CPU FP64 PyTorch LBFGS, `max_iter=200`, `history_size=20`, and
  `line_search_fn="strong_wolfe"`;
- do not use class weights, balanced sampling, augmentation, validation, test,
  threshold fitting, early selection, or candidate-specific optimization.

All three roles have exactly the same trainable parameter count and receive the
same keeper offsets, shared features, labels, row order, and fold boundaries.
Aggregate only source-group-held-out predictions. Persist summary JSON,
per-row prediction CSV, per-fold metric CSV, descriptor diagnostics, hashes,
and a nonbinary artifact manifest. Do not write a checkpoint or ONNX model.

## A0 Gates

All structural checks must pass:

1. every locked source/input SHA matches;
2. exact row, class, source-group, and fold counts match the declaration;
3. dataset sample index, label, source stem, and image path align one-to-one;
4. every fold has zero fit/holdout source overlap;
5. exact stem/grid/projection/descriptor dimensions match this protocol;
6. all descriptors, standardized inputs, logits, and probabilities are finite;
7. detail effective rank is at least `12`, every Haar band has nonzero standard
   deviation, and aligned detail is not bit-equal to placebo detail;
8. every placebo permutation has zero same-source fixed points and zero
   fit/holdout crossing;
9. every optimizer converges before reaching the 200-iteration cap and has a
   finite objective;
10. raw data is read-only and validation/test loaders are never constructed.

All candidate-versus-control OOF decision gates must pass:

- macro F1 delta `>=+0.002`;
- class-1 F1 delta `>=+0.005`;
- class-1 precision delta `>=+0.005`;
- class-1 recall delta `>=-0.005`;
- restricted `0/2/4 -> 1` false positives decrease by at least `2`;
- corrections `>` harms;
- class-1 FN rescues `>=` class-1 TP breaks;
- maximum non-focus per-class F1 drop `<=0.010`;
- at least four of five folds have nonnegative class-1 F1 delta and at least
  four have nonnegative class-1 precision delta.

The candidate must also beat the parameter-matched placebo by macro F1
`>=+0.001`, class-1 F1 `>=+0.003`, and class-1 precision `>=+0.003`.
The control must remain within `0.010` macro F1 and `0.020` class-1 F1 of the
unmodified keeper OOF-row decisions so that a candidate gain cannot rely on a
collapsed comparator.

## Permission And Stop Rules

- Passing every A0 gate authorizes implementation of one default-off bridge:
  one regular and one half-window-shifted local-attention block on the native
  stride-8 map, followed by the unchanged `2x2` patch projection and global
  TRKH. Exact window size, heads, residual initialization, Stage-A budget,
  export, resource, and XAI gates must be locked in a second protocol before
  code is written.
- Failing any material A0 gate closes this exact Haar-supported shifted-window
  route. Do not sweep descriptor/readout settings or reinterpret a placebo win.
- A0 cannot update the current-best full-train command or its revision history.
- Raw datasets remain unchanged throughout.
