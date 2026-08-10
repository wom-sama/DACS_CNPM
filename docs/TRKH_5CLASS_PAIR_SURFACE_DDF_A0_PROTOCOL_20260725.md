# TRKH Pair-Surface DDF A0 Locked Protocol

Date: 2026-07-25

State: prospective protocol; no candidate implementation, image fit, metric,
validation/test access, production integration, or command promotion exists at
this boundary

## Question

The current scratch TRKH keeper identifies most true class-1 objects, but it
also predicts class 1 for 158 class-0, 54 class-2, and 10 class-4 train
objects. Fixed colour, texture, frequency, morphology, covariance, part,
prototype, context, and Cross Colour Ratio (CCR) signals did not separate
those false positives without breaking true class-1 predictions.

This A0 asks one narrower question:

> Can a lightweight position- and sample-specific local filter produce new
> bbox-grounded surface evidence that separates true class 1 from each of
> classes 0, 2, and 4, beyond the keeper margin and matched static or causally
> disabled controls?

The experiment is an isolated train-only information gate. A positive result
would authorize a separate production-integration protocol. It would not
itself update the keeper, validation/test metrics, full-train command, or
deployment package.

## Primary Source And Provenance

The selected mechanism is the multiplicative Decoupled Dynamic Filter (DDF)
from Zhou et al., CVPR 2021:

- paper:
  <https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Decoupled_Dynamic_Filter_Networks_CVPR_2021_paper.html>;
- author project:
  <https://thefoxofsky.github.io/project_pages/ddf>;
- official MIT repository:
  <https://github.com/theFoxofSky/ddfnet>;
- local paper:
  `D:\DataAI\external_sources\papers\zhou2021_decoupled_dynamic_filter_cvpr.pdf`;
- local official repository:
  `D:\DataAI\external_sources\repositories\ddfnet`;
- pinned repository commit:
  `4e21b4ad55ffe039b075be4722eaa8968894408c`.

The machine lock must pin the paper, license, official DDF module, and
DDF-ResNet adapter hashes. The candidate is initialized and trained from
scratch. It may adapt the published equation and MIT source, but it must not
load any official, ImageNet, AIDT, Top-5, or other external parameter.

The paper's ImageNet evidence is relevant but not sufficient for TRKH. Its
reported DDF-ResNet training uses 120 epochs, while this A0 is restricted to
20 epochs on a small hard-boundary cohort. The paper's custom CUDA extension
is also not permitted in the candidate. The experiment must use an
independently tested standard-operator implementation.

## Why This Is Not A Reopened Closed Family

This protocol does not reopen CondConv or ODConv. Those methods generate
sample-conditioned weights that remain shared over spatial positions. DDF
factorizes the local depthwise kernel into:

- a spatial filter that varies by sample and pixel; and
- a channel filter that varies by sample and channel.

For input feature `F`, output channel `r`, output location `i`, and local
neighbor `j`, the locked multiplicative equation is:

```text
F'(r, i) = sum_j D_sp(i, i-j) * D_ch(r, i-j) * F(r, j)
```

This also differs from the previously rejected Involution screen. Involution
was measured as a replacement at the 256-channel third-stem geometry and
failed runtime and memory limits. Pair-Surface DDF is a dedicated
`3 -> 16 -> 32` sidecar at a fixed `64x64` analysis resolution. Its
standard-op shift/multiply/reduce implementation was screened before this
lock only for engineering feasibility, without reading cohort pixels or
labels:

- 8,687-parameter three-map prototype;
- batch-1 FP16 mean/p95 `1.806/2.208 ms`;
- batch-32 FP16 mean/p95 `1.616/1.903 ms`;
- peak CUDA allocation `1,699,328/52,781,056` bytes;
- ONNX Runtime maximum absolute error `1.1920929e-7`;
- standard ONNX domains only;
- TensorRT 10.7 parser and serialized-engine build passed.

These pilot values are not candidate metrics and cannot authorize training.
The final four-map implementation must repeat every engineering measurement.

## Immutable Inputs

The only image tensor source is the retained, exact-replay CCR materialized
cache:

`runs/audit_cross_colour_ratio_surface_a0_materialized_20260725`

Required payloads:

- `model_srgb_uint8.npy`: `[763,3,256,256]` uint8 exact keeper-evaluation
  images;
- `image_valid_masks_packbits.npy`: packed `[763,256,256]` validity masks;
- `cohort_arrays.npz`: sample indices, labels, source folds, keeper
  probabilities, model/crop boxes, and paths;
- `summary.json` and `artifact_set_manifest.json`: materializer provenance.

The cache contains:

- 541 class-1 rows: 528 keeper TP and 13 keeper FN;
- 158 class `0 -> 1` restricted FP;
- 54 class `2 -> 1` restricted FP;
- 10 class `4 -> 1` restricted FP;
- 763 object rows from 735 source images;
- five already locked source-disjoint folds.

The complete train prediction table
`runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`
may be read only to reconstruct full-train diagnostic metrics after OOF
actions. Only its `clean` rows are admissible. It may not provide a target,
teacher feature, sample weight, or learned representation.

Raw `class_f` and `yolo_f`, all validation/test paths, the keeper checkpoint
forward path, and every external pretrained model are forbidden during the
formal and replay.

## Locked Input Transform

For every cached row:

1. convert the exact uint8 sRGB tensor to float in `[0,1]`;
2. normalize by the keeper's fixed ImageNet mean/std;
3. resize once to `64x64` with bilinear interpolation,
   `align_corners=false`, and no antialias;
4. do not crop, recolour, sharpen, augment, synthesize, or use a
   candidate-specific background transform.

The bbox mask is rasterized from normalized `cx,cy,w,h` in the locked
`model_boxes` at the final `16x16` map resolution. Bounds are clipped to
`[0,1]`; starts use `floor(bound*16)`, exclusive stops use
`ceil(bound*16)`, and each axis is clamped to retain at least one cell. It is
used only for a training localization loss and for evidence audit. The
network receives only the normalized RGB image. No bbox, mask, label, keeper
probability, or source identifier is an inference input.

## Candidate Architecture

The isolated sidecar is fixed as follows:

```text
RGB 64x64
  -> Conv3x3 s2, p1, 3->16, bias=false, BatchNorm, SiLU
  -> residual multiplicative DDF, C=16, k=3, squeeze=max(4,floor(0.2*C))
  -> BatchNorm, SiLU
  -> Conv3x3 s2, p1, 16->32, bias=false, BatchNorm, SiLU
  -> residual multiplicative DDF, C=32, k=3, squeeze=max(4,floor(0.2*C))
  -> BatchNorm, SiLU
  -> separate 1x1 evidence and attention projections, 32->4 each
  -> spatial softmax attention
  -> four weighted evidence scores
```

The score order is:

```text
union, class1_vs_0, class1_vs_2, class1_vs_4
```

For every DDF block:

- the residual equation is `output = input + ddf(input)`, with no learned
  residual scale;
- the spatial branch is one bias-enabled `1x1` convolution producing nine
  values per pixel;
- the channel branch is global average pooling, `1x1`, ReLU, and `1x1`,
  with bias enabled on both convolutions, producing nine values per channel;
- the squeeze widths are exactly `4` at `C=16` and `6` at `C=32`;
- spatial filter normalization subtracts the nine-tap mean, divides by
  PyTorch's sample standard deviation (`correction=1`) plus `1e-10`, and
  multiplies by `sqrt(2)/3`;
- channel filter normalization performs the same centering and sample-
  standardization, then multiplies elementwise by a learned `C*9` scale
  initialized from `Normal(0,sqrt(2)/3)`, matching the pinned source;
- the spatial and channel filters combine multiplicatively;
- the application is a fixed row-major nine-shift stack with one-pixel zero
  padding, followed by elementwise multiplication and reduction;
- no custom CUDA, deformable sampling, grid sampling, loop dependent on
  image content, or third-party runtime operator is allowed.

The attention projection is pair-specific, but the DDF trunk is shared. The
attention-weighted evidence maps are the required explicit local evidence.
The sidecar must expose the four evidence maps, attention maps, spatial
filters, channel filters, and pooled scores in its audit trace.

The exact trainable-parameter oracle for `ddf_full`, including BatchNorm
affine tensors and channel-filter scales but excluding non-trainable buffers,
is `9,380`.

## Matched Roles

Exactly five trained roles are allowed:

1. `ddf_full`: the candidate above;
2. `static_matched`: replace each DDF block with a parameter-matched static
   depthwise/pointwise residual block while preserving every other component;
3. `ddf_spatial_only`: train the spatial dynamic branch while fixing channel
   factors to the multiplicative-neutral all-ones tensor;
4. `ddf_channel_only`: train the channel dynamic branch while fixing spatial
   factors to the multiplicative-neutral all-ones tensor;
5. `ddf_full_repeat`: repeat `ddf_full` with the locked repeat seed.

The neutral factor is exactly `1.0` at every kernel tap and is not passed
through `FilterNorm`; it removes one multiplicative conditioning branch
without changing the remaining branch or collapsing the local support to the
center tap.

Each static replacement is:

```text
output = input
       + PW_out(SiLU(PW_in(DWConv3x3(input))))
```

`DWConv3x3` has padding one and no bias. Both pointwise convolutions have
bias. The intermediate widths are exactly `28` at `C=16` and `39` at
`C=32`. The resulting `static_matched` oracle is `9,435` trainable
parameters, `55` (`0.586%`) more than `ddf_full`.

Shared convolution, normalization, projection, and head tensors for the first
four roles must start bit-identical where shapes match. No role may receive a
different image, target, loss, batch order, epoch count, optimizer family, or
localization weight.

Same-weight causal evaluations of each held `ddf_full` model are also
mandatory:

- `cross_sample_filters`: apply spatial and channel filters generated by the
  locked Sattolo partner row; partners differ in both row and source stem;
- `spatial_location_dephase`: apply the locked nonzero cyclic `(dy,dx)`
  offset to each `32x32` and `16x16` spatial filter field with `torch.roll`;
- `spatial_fixed_neutral`: replace the spatial filter with the exact
  multiplicative-neutral all-ones tensor;
- `channel_fixed_neutral`: replace the channel filter with the exact
  multiplicative-neutral all-ones tensor.

The Sattolo partners and spatial offsets must be generated and hashed in the
machine lock before candidate code exists. They may not be changed after any
score is observed.

## Loss And Optimization

All roles use:

- 20 epochs;
- batch size 64;
- AdamW, learning rate `0.003`, weight decay `0.0001`;
- cosine decay to zero, no warmup;
- FP32 parameters, optimizer, loss, and training;
- deterministic CUDA algorithms and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- in-memory tensors with `num_workers=0`;
- CPU numeric libraries restricted to one thread;
- no augmentation, dropout, EMA, early stopping, scheduler sweep, or
  checkpoint selection.

Initialization is independent of module-construction order. Each trainable
tensor receives a CPU generator seed derived from SHA-256 of
`role_initialization_seed:canonical_parameter_name`. Convolution weights use
PyTorch Kaiming uniform with `a=sqrt(5)`; convolution biases use the
corresponding fan-in uniform bound; BatchNorm weight/bias are exactly
one/zero; and DDF channel-filter scales use
`Normal(0,sqrt(2)/3)`. The primary `ddf_full`, `static_matched`,
`ddf_spatial_only`, and `ddf_channel_only` roles share one fold seed and copy
all canonical tensors with matching names/shapes bit-exactly.
`ddf_full_repeat` uses its separately locked seed.

The four binary targets are:

- `union`: class 1 versus the union of restricted classes 0, 2, and 4;
- `class1_vs_r`: class 1 versus rival `r`, ignoring the other rivals.

For a binary task with `N_pos`, `N_neg`, and `N_fit` rows in the fit
partition, every positive has fixed weight `N_fit/(2*N_pos)`, every negative
has fixed weight `N_fit/(2*N_neg)`, and an ignored pair row has weight zero.
The binary term uses BCE-with-logits. The batch loss is the sum of weighted
per-row BCE divided by the actual batch size. Averaged over the complete
once-per-epoch order, this is exactly the mean of the positive and negative
BCE means, so class frequency cannot dominate and a mini-batch with no
class-4 negative remains defined. The locked total is:

```text
L = L_union + 0.5 * mean(L_1v0, L_1v2, L_1v4)
    + 0.05 * L_bbox_attention
```

`L_bbox_attention` is the mean negative log of attention mass inside the
locked bbox mask, averaged over the four maps. Mass is clamped below at
`1e-8` before `log`. Empty masks are invalid.
Class-1 rows are not globally oversampled, no row is duplicated, and every
fit row appears once per epoch.

## Source-Held Protocol

For outer held fold `h`:

- held fold: `h`;
- calibration fold: the fixed mapping
  `h=0->2, h=1->3, h=2->4, h=3->2, h=4->3`;
- fit folds: the remaining three folds.

This mapping is selected before candidate code or scores solely to avoid the
known zero-class-4 support in fold 1. Every calibration partition has exactly
three class-4 negatives and every fit partition has at least four. Reuse of a
calibration fold across independent outer fits is allowed; every row is still
held exactly once for pooled OOF measurement.

The model is optimized only on fit folds. The calibration fold may fit only
the fixed downstream calibrators and action threshold described below.
Neither model state nor normalization state may update from calibration or
held pixels.

The primary and matched controls use the same locked epoch permutations.
The repeat role uses its own locked permutations. Source overlap among fit,
calibration, and held partitions must be zero.

## Fixed Information Readout

Report raw union and pair score AUROC/AUPRC first.

For incremental-information measurement, fit fixed balanced L2 logistic
calibrators on each calibration fold. Every calibrator is the deterministic
pipeline `StandardScaler` followed by
`LogisticRegression(C=0.1, class_weight="balanced", solver="lbfgs",
max_iter=1000, tol=1e-8, random_state=20260725)`. The scaler and classifier
are fit only on the calibration rows for that outer fold:

- keeper-only control: `log(p1)-log(max(p0,p2,p4))`, with rival ties
  resolved to the lower class index;
- each image role's union readout: keeper class-1 log margin, the role's raw
  union score, and the raw pair score corresponding to the keeper's strongest
  rival among `0/2/4`;
- keeper-only pair readout for rival `r`: keeper `1-vs-r` log-probability
  margin on rows whose target is `1` or `r`;
- each image role's pair readout for rival `r`: that keeper pair margin plus
  the role's raw `class1_vs_r` score on the same rows.

Apply each calibrator once to the matching held rows. The pooled union and
pair AUROC/AUPRC values in gates 1-5 are calculated from these held
calibrator probabilities concatenated across the five outer folds. There is
no C, feature, weight, temperature, or model-selection sweep. Calibrators are
information-audit tools and are not production artifacts.

For action measurement, use the image role's calibrated union probability.
Select one suppression threshold from the sorted unique finite calibration
probabilities plus a no-action sentinel. Suppress when probability is less
than or equal to the threshold. Select the threshold that maximizes
restricted-FP rejection subject to:

- class-1 TP retention at least `0.97`;
- corrections divided by all changed decisions at least `2/3`.

Ties choose the threshold with higher TP retention, then the higher threshold,
then the lexicographically smaller canonical hexadecimal float
representation. If no non-sentinel threshold is feasible, select no action.
Apply it only to held keeper predictions equal to class 1, replacing them
with the keeper's highest-probability non-class-1 rival. Tied rival
probabilities choose the lower class index. No class-1 rescue or other reroute
is allowed in A0. Full-train diagnostic metrics replace only the locked cohort
rows in the 9,215-row clean CIDT table; every other keeper prediction remains
unchanged.

## Conjunctive Scientific Gates

All gates are prospective and conjunctive. `ddf_full` passes clean A0 only if:

1. the pooled five-fold held AUROC improves by at least `0.015` over the
   keeper-only calibrated margin;
2. pooled held AUPRC improves by at least `0.005` over keeper-only;
3. held AUROC improves by at least `0.015` over `static_matched` and at least
   `0.010` over both trained single-branch controls;
4. candidate AUROC exceeds keeper-only and `static_matched` in at least four
   of five held folds;
5. each of `1-vs-0`, `1-vs-2`, and `1-vs-4` pooled held AUROC is at least
   keeper-only minus `0.005`, and all three fused pair AUROCs improve over
   their corresponding static-control values;
6. cross-sample filter derangement reduces raw union AUROC by at least `0.020`
   and spatial-location dephasing reduces it by at least `0.015`;
7. the full candidate beats both same-weight neutral-factor ablations;
8. OOF suppression retains at least `0.97` of the 528 keeper class-1 TP,
   rejects at least 45 of 222 restricted FP, and has at least twice as many
   corrections as harms;
9. full-train diagnostic class-1 precision improves by at least `0.030`,
   class-1 F1 improves by at least `0.015`, class-1 recall falls by no more
   than `0.030`, and macro-F1 improves by at least `0.003`;
10. repeat pooled AUROC differs by at most `0.010`, clean action agreement is
    at least `0.97`, and repeat action metrics independently pass gate 8;
11. mean attention bbox mass is at least `0.85`, at least 90% of rows have
    bbox mass at least `0.70`, and no pair head collapses to a constant map;
12. every output, loss, gradient, state, action, and metric is finite.

Missing class-4 support in an individual fold is reported and excluded only
from that fold's pair AUROC. The pooled ten class-4 negatives remain
mandatory; no row may be moved between folds.

Failure of any gate closes the exact A0 and all nearby kernel, width, squeeze,
normalization, map-count, attention, loss-weight, LR, epoch, seed, fold,
calibrator, and threshold sweeps.

## Engineering And Deployment Gates

Before any real cached image is read, synthetic checks must prove:

- FP64 PyTorch DDF output matches an independently written NumPy loop within
  `1e-10`;
- official-equation filter normalization matches the pinned source behavior;
- batch shapes `1/2/32`, gradients, and all role paths are finite;
- all intended candidate parameter groups receive nonzero gradients and
  change after an optimizer step;
- static-control parameter count is within the locked interval;
- standard-domain ONNX opset-17 export and checker pass;
- ONNX Runtime maximum absolute output error is at most `1e-5`;
- TensorRT parser and serialized-engine build pass without a plugin;
- candidate sidecar parameters are at most 15,000;
- batch-1 FP16 mean/p95 are at most `2.0/2.5 ms`;
- batch-32 FP16 mean/p95 are at most `2.5/3.0 ms`;
- branch peak CUDA allocation is at most 128 MiB.

If clean scientific gates pass, a separate matched full-keeper integration
protocol must still enforce:

- batch-1 mean and p95 each at most `1.15x` keeper;
- batch throughput at least `0.85x` keeper;
- end-to-end peak CUDA at most `1.15x` keeper and below 6 GiB;
- complete ONNX/TensorRT probability and argmax parity.

A0 cannot waive those future requirements.

## XAI And Shifted Conditions

The fixed visual rows are the 30 sample indices already selected before CCR:

```text
163,2370,856,338,58,2563,1939,2698,341,2004,
1066,313,2,2906,345,923,195,2411,791,363,
3624,3934,4012,4040,4613,4626,5379,5381,5732,5740
```

They cover every fold, TP/FN, class-0 FP, class-2 FP, and all ten class-4 FP.

Only if every clean scientific gate passes may the runner generate:

- fixed evidence/attention/filter contact sheets;
- input-gradient and integrated-gradient maps for the union and relevant pair
  score;
- bbox/border mass, map entropy, filter diversity, and pair-map correlation;
- deterministic brightness, gamma, contrast, desaturation, and one-pixel
  translation conditions derived only from cached train images.

Manual XAI must confirm that gains are supported by fruit-surface evidence
rather than padding, crop border, bbox geometry, or one isolated artifact.
Shifted-condition TP retention and FP rejection must remain within 5
percentage points of clean. XAI or robustness failure closes integration.

## Reproducibility And Access Ledger

Formal execution requires:

- clean pushed repository state except the protected user-owned untracked
  paths;
- a separate, hash-locked execution authorization committed and pushed after
  the candidate auditor exists;
- no unrelated `python.exe`, `pythonw.exe`, or `trtexec.exe`;
- a process-wide dynamic file-open ledger installed before input access;
- an exact allowlist for the lock, authorization, retained cache, clean CIDT
  prediction table, and output root;
- write-like and `val/valid/validation/test` path-component rejection;
- zero raw-dataset opens;
- one formal run and one fresh-process replay only.

The formal must retain model states, raw scores, maps needed for locked XAI,
training records, calibrators, actions, metrics, ordered access ledger, and
artifact manifest. Replay must reproduce all numeric arrays exactly or within
the tolerances below:

- model states, optimizer states, epoch losses, raw scores, calibrated
  probabilities, thresholds, maps, metrics, and every other numeric array:
  byte-identical, maximum absolute difference `0`;
- discrete decisions, row order, access events, artifact paths, and hashes:
  exact.

Any replay difference makes the run `waste/invalid`; it cannot be relaxed
after observing the difference.

Resource ceilings per formal or replay:

- wall time: 1,800 seconds;
- peak CUDA allocation: 2 GiB;
- peak process RSS: 8 GiB;
- maximum temporary storage: 1 GiB;
- maximum retained output: 256 MiB;
- no surviving child process.

## Stop And Promotion Rules

Before a complete clean pass, this protocol authorizes no:

- production model/trainer/launcher integration;
- validation or test evaluation;
- full model smoke, probe, or full train;
- current-best command/history edit;
- ONNX/engine/video package change;
- raw-dataset edit or deletion.

If A0 fails, preserve a compact evidence set, label it
`negative-but-reusable` or `waste/invalid`, record the exact no-repeat
boundary, update the journal/TODO/skill/report, and clean only disposable
artifacts through a retention manifest.

If A0 passes, the next step is a separately committed production-integration
protocol with zero-init residual fusion, full scratch training, complete XAI,
validation-only promotion, and matched inference gates. Test remains sealed
until final promotion.
