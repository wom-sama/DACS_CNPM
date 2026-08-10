# TRKH Validity Partial-Conv Stage-B Operationalization

Date: 2026-07-20

Status: locked after the complete train-only Stage-A pass and before either
matched smoke model is trained or any candidate validation prediction is
created. This document only makes the already prospective Stage-B protocol
operational; it does not change a metric threshold or authorize test access.

## Stage-A Authorization

- Formal summary SHA-256:
  `683aab0f08d22cbbce9708a046a42836d26a7b111edfc68a95ad83717e14bc42`.
- Formal manifest SHA-256:
  `53ef670cc5ad84189a972424ffb69060b781ec3bd632800f6eb6bc487e474dba`.
- Repository HEAD: `571ba2a84532125fee1b527650d731685e35bffc`.
- All 32 Stage-A checks pass, external gate replay is exact, validation/test
  are unused, and `smoke_permission=true` while `full_train_permission=false`.
- Candidate/control parameter count is identical at `7,245,590`; full-keeper
  ONNX dynamic-batch replay error is `1.862645e-7`; candidate/control AMP
  runtime ratio is `1.110663`; peak candidate VRAM is `2.520991 GiB`.

## Matched Training Pair

Both roles are generated from the keeper's frozen `launcher_args.json`. The
launcher replaces the unavailable historical resume source with the current
keeper, resets epoch/optimizer/scheduler/scaler, and changes these common
diagnostic fields for both roles:

- epochs `2`, scheduler horizon `10`, patience `2`;
- maximum train batches `120`, full validation, no final test;
- seed `42`, batch size `32`, gradient accumulation `2`, LR `8e-5`;
- train/eval workers `4/2`, BF16 AMP, Windows multiprocessing enabled, pin
  memory and persistent workers disabled according to the frozen benchmark;
- data cartography enabled only to persist the ordered
  `sample_index:target_index` SHA-256 for each epoch;
- architecture trace enabled with one deterministic train sample per class.

Apart from run/artifact paths, the sole role difference is:

- control: `stem_convolution=standard`;
- candidate: `stem_convolution=validity_partial`.

The pair audit must verify identical two-epoch occurrence hashes and learning-
rate sequences before accepting any metric gate.

## Source-Group Definition

The protocol phrase "source groups with adequate support" is fixed as five
deterministic source-disjoint folds made by
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` over the
ordered 2,606 validation object rows. The target is the class index and the
group is the case-folded source-image stem. A fold has adequate support only if
it contains at least 25 true class-1 rows and at least 300 rows from restricted
negative classes `{0,2,4}`.

A pre-training geometry-only census found fold class-1 supports
`[27,31,29,28,36]`, restricted-negative supports
`[363,342,348,349,341]`, and zero source overlap. Therefore all five folds are
eligible and the original requirement remains candidate class-1 precision
non-worse than control in at least four.

## Fixed Mask-Shape Placebo

The audit reconstructs the exact checkpoint-compatible validation transform
and records each transformed `256x256` boolean image-valid mask. Within each
class, rows are sorted by `(invalid_pixel_count, source_stem, sample_index)`
and split with `numpy.array_split` into four rank bins. Within every class/bin,
a deterministic minimum-cost bipartite assignment selects a donor that:

1. has the same class and rank bin;
2. has a different source-image stem;
3. has a different exact mask SHA-256;
4. preserves a one-to-one mapping;
5. minimizes absolute invalid-pixel-count difference, with sample order as a
   deterministic tie break.

The donor mask replaces only `image_valid_mask`. Pixels, target, bbox,
crop-bbox, checkpoint, model weights, and all other inference inputs remain
aligned. The mapping and every actual/donor mask identity are persisted.

The geometry-only census inspected no model output. It found 105-250 unique
mask shapes per class and a zero-violation perfect matching in all 20
class/rank-bin partitions. No candidate checkpoint or validation metric existed
when this rule was locked.

## Independent Audit And Replay

- The standard evaluator independently reloads each selected checkpoint and
  writes aligned full-validation predictions.
- The pair auditor independently reloads both checkpoints again, reconstructs
  all 2,606 rows, and replays full control/aligned-candidate probabilities.
  Argmax must be exact and maximum probability error must be at most `5e-4`.
- The same candidate model produces placebo probabilities with the fixed donor
  masks. No threshold, temperature, or post-hoc router is fitted.
- The auditor persists confusion, calibration, transitions, source folds,
  changed cases, placebo predictions, trace/config/scheduler/data-order checks,
  raw-dataset identity, gate inputs, internal replay, external-process replay,
  and SHA-256 manifests.
- Existing clean gates in the prospective protocol remain conjunctive and
  unchanged. A pass authorizes only the fixed post-smoke robustness/XAI audit;
  it does not authorize a probe, full train, test access, or current-best
  command update.
