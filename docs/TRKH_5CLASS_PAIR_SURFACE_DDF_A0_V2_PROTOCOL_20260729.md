# TRKH Pair-Surface DDF A0 v2 Protocol — 2026-07-29

## Prospective state and supersession

This document is a prospective replacement for the formal-information stage of
`TRKH_5CLASS_PAIR_SURFACE_DDF_A0_PROTOCOL_20260725.md`. The v1 protocol, lock,
engine and synthetic evidence remain immutable lineage. V1 consumed zero formal
runs and zero replays, but its folds are invalid for an OOF claim because they
split both dataset `leakage_group` values and numeric filename neighborhoods.
The exact evidence is frozen in
`TRKH_5CLASS_PAIR_SURFACE_DDF_A0_V1_SUPERSESSION_20260729.{md,json}`.

No v2 candidate image score, fit, metric, validation result, test result or GPU
result existed when this protocol was written. This document becomes an
effective boundary only after it, the v2 fold manifest, the no-repeat registry,
the implementation tests and a machine lock are committed and pushed. A later
authorization may permit exactly one formal run and one fresh-process replay.

## Defensible contribution claim

DDF is not a new operator in this project. Decoupled Dynamic Filter Networks
introduced spatial/channel factorized dynamic depthwise filtering at CVPR 2021:

<https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Decoupled_Dynamic_Filter_Networks_CVPR_2021_paper.html>

Pairwise learning and pairwise attention also predate this work, including
Pairwise Confusion (ECCV 2018) and Dual Cross-Attention Learning (CVPR 2022):

<https://openaccess.thecvf.com/content_ECCV_2018/html/Abhimanyu_Dubey_Improving_Fine-Grained_Visual_ECCV_2018_paper.html>

<https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Dual_Cross-Attention_Learning_for_Fine-Grained_Visual_Categorization_and_Object_Re-Identification_CVPR_2022_paper.html>

The prospective claim is therefore limited to a **scratch DDF sidecar with
pairwise evidence heads for TRKH**, a validity-aware surface input, explicit
pair-attribution controls, and an unusually strict causal/replay protocol for
the mango maturity boundary. Operator novelty is low; composition/application
novelty may be moderate only if the pair contribution and empirical gates pass.
Protocol/reproducibility is a separate contribution. No new-model claim exists
before matched integration and a newly sealed confirmatory cohort.

## Scientific question

> On manifest-group- and numeric-neighborhood-disjoint train-only rows, does a small
> scratch DDF sidecar provide sample- and location-conditional fruit-surface
> evidence that adds information beyond the keeper margin, a parameter-matched
> static sidecar, trained single-factor DDF controls, and a generic DDF union
> score, while protecting class-1 precision and recall?

This is an information/mechanism gate, not a paper accuracy estimate and not a
five-class calibrator. The 763-row cohort contains only targets 0, 1, 2 and 4;
the 1-vs-4 result has only ten class-4 rows. It is a mandatory descriptive
safety veto, never a positive efficacy/generalization endpoint.

## Inputs and forbidden information

The future machine lock must pin exact bytes and SHA-256 for:

- the replayed train-only CCR materialization and its metadata arrays;
- `D:\DataAI\AIEx\newdataset\yolo_f\manifest.csv`;
- the clean 9,215-row CIDT train table;
- the frozen cohort `keeper_probabilities` and clean-table keeper predictions;
- the train-table geometry preflight
  `runs/audit_pair_surface_ddf_v2_geometry_metadata_20260729/summary.json`
  (6,865 bytes, SHA-256
  `4b80126e58e7853b8082977b05b6f7bfc6ae67270df0ecb1ad3ba50419039c22`);
- the historical keeper checkpoint/config as lineage hashes only, never as a
  permitted forward-pass input;
- the v2 fold manifest and no-repeat registry;
- this protocol, the v1 supersession, v2 engine, auditor, replay, tests and
  launcher;
- Python, NumPy, PyTorch, SciPy, scikit-learn, CUDA, cuDNN, ONNX, ONNX Runtime
  and TensorRT versions used by the formal environment.

V2 may read only the retained train-only cache (including its already frozen
keeper probabilities), its manifests, the clean CIDT train table, the dataset
manifest and the immutable protocol/authorization files. It may not open raw
dataset images/labels, validation, test, manual-review labels, external
pretrained predictions/features, a keeper checkpoint for a forward pass, or a
new architecture score. It may not write a relabel, sample-weight, soft-target
or production-routing manifest.

The already viewed project test is permanently legacy/development evidence. No
process ledger can make it confirmatory again. A paper-confirmatory result
requires a new source/time/site-held cohort that is sealed before model choice.

In this document, a held/component-disjoint score means only that the v2
sidecar and its calibrator did not fit that held component. The frozen keeper
probabilities and the error-defined cohort membership came from historical
keeper evaluation on the training table; they are not keeper OOF predictions.
Consequently K/KU/KUP and action results are conditional mechanism diagnostics,
not unbiased end-to-end generalization estimates.

## Group and fold construction

The old `folds` NPZ member is lineage only. Every v2 consumer must reject it as
an assignment source.

Canonical rows are sorted by `sample_index` and contain only:

```text
sample_index, source_stem.casefold(), target, parsed Image_N,
manifest leakage_group, train-relative image identity
```

The group is the transitive connected component of these relations:

1. casefold-exact source stem;
2. exact non-empty YOLO-manifest `leakage_group`;
3. every unordered row pair whose parsed `Image_N` values have absolute
   numeric gap `<=3`;
4. a verified fruit/session identifier, if trustworthy acquisition provenance
   is later recovered before the fold lock.

The current `leakage_group` semantics are pinned to clean CVAT exporter commit
`0c7fd8b873d34d18c6f6f6588b772996528ba0ec`, Git blob
`d405bf3522f8e7809488b4842f9ffb988a069195`, file
`src/cvat_nhai/yolo_editor.py` SHA-256
`65e6056dac1793eaa9775b458a448828bfd0debd7adc4d9acb58807b7b74e4b5`.
That code takes the transitive union of normalized source-family identity and
near-visual matches (64-bit dHash Hamming distance at most 3 plus its quantized
16x16 colour-thumbnail threshold). The exact current YOLO manifest bytes are
canonical. Its 16-hex group IDs hash absolute resolved source paths, so the IDs
are not promised to regenerate byte-identically after moving the dataset root;
membership must be compared by canonical source identities instead.

The current fallback produces 158 components. It is deliberately conservative,
but it does **not** prove fruit/session independence: no trustworthy
fruit/session identifier is presently available in the exporter/manifest. If such provenance is found
before machine authorization, this fold manifest must be superseded and rebuilt;
it cannot be appended after scores exist. All rows in one component must remain
in one fold. Five-fold assignment uses only canonical metadata and a
deterministic lexicographic feasibility procedure over SciPy `milp`; candidate
scores and keeper metrics are absent. Hard per-fold bounds are:

| Quantity | Required per fold |
|---|---:|
| rows | 152–153 |
| target 0 | 31–32 |
| target 1 | 108–109 |
| target 2 | 10–11 |
| target 3 | 0 |
| target 4 | exactly 2 |

The builder must pin its canonical row hash, component-set hash, solver/runtime
identity, lexicographic mapping algorithm, full sample/component-to-fold
mapping and mapping SHA. Replay must obtain the same mapping. Required leakage
counts across folds are zero for exact stems, manifest groups, complete
components, numeric window 1 and numeric window 3.

For outer held fold `h`, calibration fold is `(h+1) mod 5`; the remaining three
folds are fit. Each fold is held and calibration exactly once. Fit contains
457–459 rows and six class-4 rows; calibration and held each contain two
class-4 rows. Source/component overlap among fit, calibration and held is zero.

The one-calibration-fold design is retained because the same fitted image model
produces both its calibration and held scores without training on either.
Pooling inner models that trained on the outer-held fold is forbidden. The
small class-4 support and corresponding uncertainty must be reported.

## V2 sidecar and validity-aware surface input

The role set remains:

1. `ddf_full`;
2. `static_matched`;
3. `ddf_spatial_only`;
4. `ddf_channel_only`;
5. `ddf_full_repeat`.

The v1 3→16→32 trunk, two residual blocks, four evidence maps and four
attention maps are lineage, but v2 requires a new engine boundary because
validity handling and the bbox objective change. Parameter targets remain
9,380 for DDF and 9,435 for the static control; any difference requires a new
prospective protocol.

The DDF application equation, filter normalization/gain, squeeze widths,
residual/static block equations, evidence-head ordering and standard-operator
restrictions inherit exactly from v1 protocol SHA-256
`724f63ef7a0701f4a250cb9aabc04bf2d474ffe9929b9680f05e14fb2dc0fc42`
unless this document explicitly replaces them. V1 folds, calibration mapping,
orders, Sattolo/roll controls, loss reduction, unmasked normalization and bbox
supervision do not carry forward.

Every v1 BatchNorm site is replaced by a validity-aware masked BatchNorm with
the same per-channel affine tensors and buffers, so trainable parameter counts
do not change. For training tensor `x[B,C,H,W]` and binary `m[B,1,H,W]`, let
`M=sum(m)`. `M<=1` fails. Per channel:

```text
mu_c  = sum(x_c*m)/M
var_c = sum((x_c-mu_c)^2*m)/M
y_c   = gamma_c*(x_c-mu_c)/sqrt(var_c+1e-5) + beta_c
y_c   = where(m, y_c, literal +0.0)
```

Optimizer-step forwards use these batch statistics but do not momentum-update
running mean/variance; `num_batches_tracked` only counts the 160 locked training
calls. After the final optimizer step, weights freeze and the four masked-BN
sites are recalibrated in execution order using fit rows only. For each site,
all earlier sites use their already frozen eval statistics; one complete
no-grad fit pass streams that site's pre-normalization activations and mask in
locked row order, accumulating FP64 channel `sum`, `sumsq` and valid count.
Running mean becomes `sum/M` and running variance becomes the population value
`max(sumsq/M-mean^2,0)`. Then that site freezes before the next site receives
its own pass. Calibration/held rows, optimizer updates and augmentation are
forbidden. This prevents the final 9-11-row batch from receiving the same
running-stat momentum as a full batch. Evaluation uses only the recalibrated
statistics and applies the mask again. The static role uses the identical
masked normalization/recalibration. Standard BatchNorm or GroupNorm over
invalid zeros is forbidden because it would retain padding-fraction batch
coupling. Eval-mode ONNX must express this using only standard-domain
arithmetic and `Where` operators.

Inputs are the exact cached `[B,3,256,256]` sRGB uint8 tensor and its unpacked
binary valid mask. The cache was written by
`np.packbits(...,bitorder="little")`; every consumer must therefore use
`np.unpackbits(...,bitorder="little")` explicitly and assert the locked
`valid64/valid32/valid16` hashes. NumPy's default `"big"` unpacking is
forbidden. It reproduces the former erroneous all-row mean
`q=0.3498597859836392` while preserving row bit counts and even the same 12
bbox-unusable rows, so count-only validation cannot catch this error. V2
prepares `[B,3,64,64]` by:

1. converting sRGB to FP32 `[0,1]` and ImageNet normalization;
2. applying `where(valid, normalized, literal +0.0)` before any reduction
   (plain multiplication is forbidden because signed zero is not byte-stable);
3. applying fixed 4×4 average pooling to numerator and mask denominator;
4. dividing numerator by denominator where denominator is positive;
5. writing literal `+0.0` where denominator is zero.

Thus invalid fill is mathematically removed rather than merely blurred into
the input. `valid64` is `pool4x4(valid256)>0`; `valid32` and `valid16` are
propagated by 3x3, stride-2, padding-1 max pooling, exactly matching the two
strided convolution receptive-field supports. Each DDF block receives its
matching mask. Channel pooling is
`sum(where(mask,feature,+0.0))/sum(mask)`, not unmasked GAP. Intermediate
features are set to literal `+0.0` outside the mask after every
convolution/BatchNorm/activation and again after every residual block. Thus
conv1 receives masked `valid64` input; DDF block 1 receives masked `valid32`;
conv2 receives the masked block-1 output; DDF block 2 receives masked
`valid16`; and the evidence/attention projections receive masked final
features. DDF shift stacks also read literal-zero invalid neighbors. Spatial
filter fields are ignored and set to neutral outside their matching validity
mask.
Attention softmax excludes invalid cells and assigns them exact `+0.0` mass.
The final evidence score is the masked attention-weighted sum. Empty validity
is invalid. A metadata-only precheck found 12 rows whose rasterized
bbox/`valid16` intersection is empty (targets 0/1/4: 2/9/1); their sorted
`sample_index,target` record SHA-256 is
`cbe5d09bc0ef6305a6610297ddc1dd747e1e018fd10cdc87d2ed87d224b91534`.
The future lock must reproduce that list before scores. Those rows retain
classification weight but have exact zero bbox weight; they are not silently
dropped.

The four tasks are union, 1-vs-0, 1-vs-2 and 1-vs-4. Union is active on every
row, with `y_union=1` iff target is 1 and zero otherwise. All three pair tasks
are positive and active for target 1. A `1-vs-r` task is negative and active
only for target `r in {0,2,4}`; the other two pair labels and weights on that
rival row are exact zero/inactive. Inactive pair tasks have exactly zero
classification and bbox weight.

## Fixed loss and mini-batch semantics

For every outer fold and role, class-balanced task weights are calculated once
from the complete fit partition and then sliced by the locked batch order. Let
`N_fit` be the complete fit size and `N_pos_t/N_neg_t` the active binary support
for task `t`. A positive weight is `N_fit/(2*N_pos_t)`, a negative weight is
`N_fit/(2*N_neg_t)`, and an inactive pair-row weight is exact zero. Missing
positive or negative support closes the run. For every batch, including the
last short batch:

```text
BCE_t = sum_i w_it * BCEWithLogits(z_it, y_it) / 64
```

The mean pair term is always the arithmetic mean of the three named pair
tasks, including a zero batch contribution when a task has no active row.

The normalized `cx,cy,w,h` model box is clipped to `[0,1]` and rasterized on
16x16 with left/top `floor(16*x)` and right/bottom `ceil(16*x)`, clamped to the
grid with at least one cell, exactly as in v1 lineage. V2 intersects that mask
with `valid16`. With masked attention `a_ituv`,
`bbox_mass_it=sum_uv a_ituv*b_iuv`. For each task, bbox weights are calculated
once from its usable active fit rows: `N_fit/(2*N_bbox_pos_t)` for usable
positives, `N_fit/(2*N_bbox_neg_t)` for usable negatives, and zero otherwise.
Missing usable positive/negative support closes the run. For task `t`:

```text
B_t = sum_i w_it * -log(max(bbox_mass_it, 1e-8)) / 64
```

The batch objective is:

```text
L = BCE_union + 0.5 * mean(BCE_pairs)
    + 0.05 * (B_union + 0.5 * mean(B_pairs)) / 1.5
```

Every batch, including the final short batch, uses the fixed denominator 64.
This prevents the last 9–11 rows from receiving an optimizer step scaled as if
they were a full batch. Mini-batch SGD remains order-dependent because weights
change between steps; the protocol does not claim that training is exactly a
single full-empirical-objective update. Exact batch order, per-step loss scale,
gradient/state canonical hashes and finite/norm telemetry are retained for
replay; full tensors are retained only for the locked initial and final states.

## Optimization and matched exposure

Each role uses 20 epochs, batch size 64, eight optimizer steps per epoch,
AdamW with `betas=(0.9,0.999)`, `eps=1e-8`, weight decay `0.0001`,
`amsgrad=False`, `maximize=False`, `foreach=False`, `fused=False`,
`capturable=False`, and `differentiable=False`. There is exactly one parameter
group containing every `requires_grad=True` tensor in canonical
`named_parameters()` order. Weight decay `0.0001` applies without exclusion to
convolution biases, masked-BatchNorm affine tensors and DDF channel-scale
tensors as well as weights. No gradient clipping, gradient accumulation or
second optimizer exists; every batch begins with
`optimizer.zero_grad(set_to_none=True)`, then one backward and one step. Epoch
`e in [0,19]` uses the
constant-within-epoch learning rate
`0.003*0.5*(1+cos(pi*e/20))`; all 20 epochs therefore have positive LR, and
the recorded terminal post-training LR at `e=20` is zero without an optimizer
step. There is no warmup. There
is no augmentation, dropout, EMA, early stopping, mixed precision, scheduler
sweep or duplicate exposure. Every row appears once per epoch in a locked
permutation. `num_workers=0`; training/evaluation uses FP32.

Before importing NumPy/SciPy/scikit-learn/PyTorch or initializing CUDA, the
launcher sets `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `BLIS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The entire
formal/replay body additionally runs inside
`threadpoolctl.threadpool_limits(limits=1)`. Immediately after scientific
imports and again before finalization, every entry returned by
`threadpoolctl.threadpool_info()` must report `num_threads=1`; library names,
versions, architectures and thread counts are retained, and any mismatch
fails before a score is accepted. This covers the OpenBLAS/OpenMP pools used by
NumPy, SciPy and scikit-learn, which are not controlled by PyTorch thread
setters.

The launcher enables
`torch.use_deterministic_algorithms(True)`, disables cuDNN benchmarking and
TF32 (`torch.backends.cuda.matmul.allow_tf32=False` and
`torch.backends.cudnn.allow_tf32=False`), pins cuDNN deterministic mode, and
sets intra-op/inter-op CPU threads to one. The lock records the exact GPU UUID,
driver/runtime/library versions and all determinism flags. Any unavailable
deterministic kernel closes the run; it is not replaced after observing a
score.

The v2 primary seed is `20260729`; the repeat offset is `100000`. For fold `h`,
primary initialization seed is `20260729+100*h` and repeat seed adds the offset;
the corresponding order seed is initialization seed plus one. Exactly one
`numpy.random.Generator(PCG64(order_seed))` is instantiated per role/fold, then
its state advances through exactly 20 successive `permutation(fit_indices)`
calls, one per epoch; `fit_indices` is the complete fit partition's
`sample_index` vector in ascending numeric order, and successive 64-row slices
of each returned permutation are the eight batches. It is not re-seeded each
epoch.
Tensor initialization uses a CPU generator seeded by the first 63 bits of
`SHA256("seed:canonical_parameter_name")`: convolution weights use Kaiming
uniform with `a=sqrt(5)`, biases use the fan-in uniform bound, masked-BatchNorm
affine weight/bias and running mean/variance start at `1/0/0/1`, and
channel-filter scales use `Normal(0,sqrt(2)/3)`. Masked BatchNorm uses the
exact formula, `eps=1e-5`, no momentum update, affine tensors and final
fit-only population-stat buffers specified above.

Primary DDF, static, spatial-only and channel-only roles copy every matching
named/shape tensor bit-exactly and use the same row order. The repeat role uses
its separately locked seed/order. The fold manifest change requires new
epoch-order hashes, donor mappings and non-wrap offsets; v1 hashes cannot be
copied.

Every recalibration pass records row-order, pre-normalization tensor, mask,
FP64 sufficient-statistic and final-buffer hashes. Four sites times eight fit
batches add 32 no-grad forward batches per role/fold after the 160 optimizer
steps.

## Raw scores, calibrators and pair attribution

Report raw union and pair AUROC/AUPRC before calibration. Probability logs use
`clip(p,1e-12,1.0)` in FP64. Rival ties choose the lower class index.

Calibrators are fixed FP64 pipelines:

```text
StandardScaler(with_mean=True, with_std=True)
LogisticRegression(C=0.1, class_weight="balanced", solver="lbfgs",
                   max_iter=1000, tol=1e-8, random_state=20260729)
```

Zero-variance features, non-convergence, warnings promoted to failure, or
non-finite state close the run. No C, feature, solver, threshold or temperature
sweep is permitted.

For every outer fold, fit the following separately on its calibration fold and
apply once to held rows:

- `K`: keeper `log(p1)-log(max(p0,p2,p4))`;
- `KU`: `[K, raw_union]`;
- `KUH`: `[K, raw_union, I(r*=2), I(r*=4)]`, where `r*` is the keeper's
  strongest rival and rival 0 is the one-hot reference;
- `KUP`: `[K, raw_union, I(r*=2), I(r*=4), s0*I(r*=0),
  s2*I(r*=2), s4*I(r*=4)]`, where `sr` is the role's raw `1-vs-r` score;
- pair `K_r`: keeper `log(p1)-log(pr)` on targets `{1,r}`;
- pair `KP_r`: `K_r` plus raw pair score `1-vs-r`.

The main candidate information readout is `KUP`. `KU` controls generic DDF
union information; `KUH` additionally controls the identity/intercept of the
selected head, so `KUP-KUH` attributes numerical pair evidence rather than
head scale alone. Head selection uses only keeper probabilities. Report
selected-head counts and, for diagnosis only, label-known out-of-task counts;
the frozen cohort currently has 17/222 rival rows where `r*` differs from the
true rival (target 0/2/4: 6/8/3). No target enters a calibrator feature, head
selection or replacement rule. Frozen
calibration-fold targets are used only for supervised logistic fitting and
threshold feasibility/selection; held targets are read only after those states
freeze, to evaluate the held metrics and actions. All calibrated held
probabilities and scaler/logistic states are retained.

AUROC is scikit-learn `roc_auc_score`; AUPRC is
`average_precision_score`, not trapezoidal PR AUC. Precision, recall and F1 use
`precision_recall_fscore_support(..., zero_division=0)`. Float computations are
FP64. A one-class fold pair metric is null and excluded from that fold only;
pooled support is mandatory. Class 3 is not claimed by this A0.

## Locked action rule

For each role and outer fold, use its `KUP` calibration probabilities. Candidate
thresholds are the sorted unique finite FP64 calibration probabilities plus a
distinct JSON sentinel string `NO_ACTION`. Numeric thresholds are retained as
Python `float.hex()` strings and suppress when probability is `<= threshold`.

Evaluate feasibility and the following objective on calibration-fold rows
only. Among numeric thresholds, maximize restricted-FP rejection subject to:

- changed rows greater than zero and restricted-FP rejection greater than zero;
- class-1 TP retention at least 0.97;
- corrections at least twice the sum of harms and neutral changes.

Ties choose higher TP retention, then higher numeric threshold, then the
lexicographically smaller float-hex string. If no numeric threshold is
feasible, choose `NO_ACTION`. For that sentinel, changed/corrections/harms are
all zero, neutral changes are zero, and the correction ratio is recorded as
null, never as one. A zero-change numeric threshold is infeasible rather than a
second spelling of `NO_ACTION`. The frozen calibration threshold is then
applied once to held rows without held-label access.

Only held keeper predictions equal to class 1 may be suppressed. Replacement
is the keeper's highest-probability non-class-1 rival, with lower-index tie
break. Correction means wrong→correct; harm means correct→wrong; neutral change
means wrong→different wrong. Thus every feasible numeric threshold also has
`corrections/changed >= 2/3`; a large neutral set cannot make an action appear
safe. A restricted false positive is a row with target in `{0,2,4}` and keeper
prediction 1. Restricted-FP rejection count is the number of those rows changed
away from class 1; its rate uses the corresponding held or pooled keeper
restricted-FP count as denominator. Both count and denominator are retained.
The 9,215-row CIDT calculation is a **train-table
diagnostic**: only pooled held-component cohort rows change and every other prediction is
untouched. It is not full-train or generalization performance.

## Group-aware uncertainty

For pooled union comparisons, take the manifest's 158 unique
`component_sha256` values in ascending lexical order; their locked order SHA is
`e10e1045e6ede63ade0cb436b684be7a6ab377d0d27f74480ac8721756045ccb`.
Create exactly one `numpy.random.Generator(numpy.random.PCG64(20260729))`, then
make exactly one draw call:

```text
draws = rng.integers(0, 158, size=(2000, 158),
                     dtype=np.int64, endpoint=False)
```

The future lock writes the complete canonical NPY draw matrix and its SHA-256
before candidate score arrays are opened. For each row of `draws`, concatenate
all rows from the indexed components, including repeats. Every replicate must
contain both union classes; otherwise the formal fails rather than silently
discarding replicates. The prospective metadata precheck has 0/2,000 one-class
replicates and the formal must reproduce that count. Compute paired
candidate-minus-control metrics on the identical replicate. The 95% interval
is `numpy.quantile(...,[0.025,0.975],method="linear")`.

Pair 1-vs-4 confidence intervals are descriptive only because ten negatives do
not support a confirmatory claim; its non-inferiority safety veto remains
mandatory.

## Same-weight causal controls

All controls evaluate the exact trained `ddf_full` weights; none is refit.

### Cross-fold capacity-balanced filter substitution

For outer held fold `h`, donors come only from its locked calibration fold
`(h+1) mod 5`. Assign every held victim to exactly one donor. Each donor is used
between `floor(n_held/n_cal)` and `ceil(n_held/n_cal)` times and total usage is
exactly `n_held`; thus equal-size folds yield a bijection, a 153-vs-152 case
duplicates exactly one donor, and a 152-vs-153 case leaves exactly one donor
unused. Held and calibration folds are component-disjoint, so same-component
and fixed-point assignments must both audit to zero. A candidate-independent
integer cost penalizes, in descending order:

1. valid-mask XOR counts at 32x32 and 16x16;
2. 16x16 bbox-and-valid-mask XOR count;
3. focus-vs-rival target-status mismatch;
4. exact target mismatch;
5. keeper strongest-rival mismatch;
6. bbox-area rank distance;
7. valid-fraction rank distance.

These are soft costs, never hard target-capacity constraints. The future
machine lock must state the exact integer coefficients, a rectangular
min-cost-flow/assignment construction, donor capacities and lexicographic
optimal-assignment tie break. Only frozen metadata and historical keeper
probabilities may construct it; candidate scores are forbidden. The
mapping, donor-use histogram and covariate mismatch counts are frozen before
model scores. Under the same trained weights, filters generated from the
assigned calibration donor in its clean FP32 evaluation pass replace both DDF
blocks' filters for the held victim sample. Both blocks use the same donor;
the victim feature field and validity mask remain in force. On a victim-valid,
donor-invalid spatial location, the injected spatial factor is the exact
all-ones multiplicative neutral. A mandatory `support_matched_self` control
uses the victim's own spatial filter on victim/donor-valid overlap and the same
neutral fill elsewhere, while retaining the victim channel factor. This
separates filter identity from support mismatch. Per-block XOR mass, bbox-mask
mismatch and neutral-fill mass are retained. No donor label or pixel changes
the victim input. This is a transductive cross-sample causal diagnostic, not a
deployable inference route.

### Non-wrap spatial displacement

The ordered offset list is
`[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]`. Before any
candidate output exists, separately for each outer fold `h` and zero-based DDF
block `b in {0,1}`, sort held rows by
`(SHA256(UTF8("20260729|h|b|sample_index")), sample_index)`. Row at sorted
position `j` receives `offsets[(j + h + 2*b) mod 8]`. Consequently every
offset is nonzero and its within-fold/block use count differs from every other
offset by at most one. The complete sample-index/block/offset mapping and its
canonical JSON SHA-256 are written into the future machine lock before scores;
candidate outputs, targets and keeper probabilities do not enter the mapping.
The identical mapping is used by every reported same-weight displacement
control.

Spatial filter fields move without cyclic wrapping. Newly exposed positions
use the exact all-ones multiplicative spatial factor; invalid output positions
remain masked. No `torch.roll` wrap artifact is allowed.

### Neutral factors

`spatial_neutral` replaces the spatial factor with the exact all-ones
multiplicative factor and preserves the learned channel factor. `channel_neutral` replaces the
channel factor with its exact neutral multiplicative value and preserves the
learned spatial factor. Raw union metrics use no refitted calibrator.

### Invalid-fill invariance

For every held row, evaluate two deterministic invalid-pixel fills (all-zero
uint8 and locked PCG64 uint8 noise) with the identical valid mask. Prepared
inputs, scores and valid-cell maps must be exactly identical (`max_abs=0`). This
is a clean gate, not postponed XAI.

### Exact map and filter statistics

All map statistics use FP64, population variance (`ddof=0`) and only the
row's `valid16` cells. They are computed per row/task first; medians/fractions
then aggregate those row values. Let `V=sum(valid16)`,
`q=sum(bbox16 AND valid16)/V`, and `m=bbox_mass`. On the 751 bbox-usable rows,
geometry-normalized localization lift is `(m-q)/max(1-q,1e-8)`. The
locked metadata audit gives mean/median
`q=0.3585639994345054/0.34615384615384615`. Across all 763 rows, including the
12 bbox-unusable rows with `q=0`, mean `q=0.35292472290342536`. The audit uses
the denominator `V=sum(valid16)`, not bbox area or all 256 grid cells, and its
independent NumPy/Torch validity pipelines match exactly. Uniform attention
therefore has lift zero on every bbox-usable row.

Normalized attention entropy is
`-sum(a*log(max(a,1e-12)))/log(V)`; `V<=1` fails. Attention coefficient of
variation is `std(a)/(mean(a)+1e-12)`. Evidence dispersion is
`std(e)/sqrt(mean(e^2)+1e-12)`. Pair-attention cosine uses the two flattened
valid vectors without centering. Evidence Pearson subtracts each valid-vector
mean before cosine normalization. A zero/non-finite norm or variance fails; it
is never replaced by zero correlation.

For sample-conditioned filter dispersion in one outer fold, a spatial factor
first becomes a per-sample nine-vector by validity-masked spatial mean; a
channel factor is flattened over channel and tap. For resulting matrix
`F[sample,j]`, dispersion is
`mean_j(var_sample(F[:,j],ddof=0))/(mean(F^2)+1e-12)`. It is computed separately
for both blocks and both factors of primary `ddf_full` in every outer fold.

## Conjunctive clean scientific gates

Every gate is mandatory. A reviewer-quality 0/1/2 rubric may annotate strength
but cannot override failure.

1. `KUP` pooled union AUROC exceeds `K` by at least 0.015, and the paired
   v2-component bootstrap lower 95% bound is greater than zero.
2. `KUP` pooled union AUPRC exceeds `K` by at least 0.005, and its paired
   bootstrap lower 95% bound is nonnegative.
3. Pooled-held `ddf_full KUP` AUROC exceeds pooled-held `static_matched KUP` by
   at least 0.015 and each pooled-held trained single-factor control by at least
   0.010 under matched exposure.
4. Per-fold `ddf_full KUP` AUROC exceeds both keeper `K` and
   `static_matched KUP` in at least four of five folds; the median fold delta
   over each named control is positive.
5. For rivals 0 and 2, pooled calibrated `ddf_full KP_r` AUROC is at least
   keeper `K_r` minus 0.005 and exceeds `static_matched KP_r`. For rival 4,
   `ddf_full KP_4` AUROC must be at least `K_4` minus 0.005 as a mandatory
   safety veto, but no superiority or
   generalization claim is allowed.
6. Pair attribution: `KUP` exceeds `KUH` in pooled AUROC by at least 0.005, is
   nonnegative in at least four folds, and has positive median fold delta.
   `KUH-KU` is reported separately and cannot satisfy this gate.
7. Pooled-held `support_matched_self` raw union AUROC is within 0.005 absolute
   of pooled-held clean, and pooled-held cross-fold capacity-balanced filter
   substitution is at least 0.020 below `support_matched_self`; pooled-held
   non-wrap spatial displacement is at least 0.015 below clean raw union AUROC; invalid-fill
   invariance is exact.
8. Pooled-held clean raw union AUROC exceeds each pooled-held same-weight
   neutral-factor ablation by at least 0.010. This defines the formerly
   ambiguous “beats” gate.
9. Pooled held-component action retains at least 0.97 of 528 keeper class-1 TP,
   rejects at least 45 of 222 restricted FP (rate at least `45/222`), and has
   corrections at least twice the sum of harms and neutral changes.
10. The 9,215-row train-table diagnostic improves class-1 precision by at least
    0.030, class-1 F1 by at least 0.015 and macro F1 by at least 0.003, while
    class-1 recall falls by at most 0.030.
11. `ddf_full_repeat KUP` pooled AUROC differs from `ddf_full KUP` by at most
    0.010 and repeat independently passes gate 9. On exactly the 750 pooled
    action-eligible cohort rows whose keeper prediction is 1 (528 TP and 222
    restricted FP), primary/repeat binary suppression-decision agreement is at
    least 0.97 and their suppressed-row-set Jaccard is at least 0.80. The other
    13 cohort rows and the untouched 8,452 CIDT rows cannot inflate agreement.
    A separate “final-action agreement on the union changed set” is forbidden
    as redundant because replacement is a deterministic keeper rival.
12. On bbox-usable active task-row maps, mean bbox mass is at least 0.85, at
    least 90% have bbox mass at least 0.70, mean geometry-normalized lift is at
    least 0.50, and at least 90% have lift at least 0.20. Every pair head has median normalized
    attention entropy at most 0.995, at least 90% of active rows have attention
    coefficient of variation at least 0.05, and at least 90% have normalized
    evidence dispersion `std/RMS` at least 0.01. On class-1 rows, no pair of
    pair-attention maps has median cosine similarity above 0.995 and no pair of
    spatially centered evidence maps has median absolute Pearson correlation
    above 0.995. Invalid attention mass is exactly zero.
13. For each DDF block and factor, held sample dispersion
    `mean(var_over_samples)/(mean(square)+1e-12)` is at least `1e-4`; filters
    cannot collapse to a static tensor.
14. Every input, output, loss, gradient, optimizer/model state, calibrator,
    score, action, metric, bootstrap replicate and artifact is finite; the
    fresh-process replay is exact.

Failure of any gate closes this exact v2 A0 and all nearby kernel, width,
squeeze, normalization, map-count, attention, loss-weight, LR, epoch, seed,
fold, calibrator and threshold sweeps. Harness failures before any candidate
metric may receive a new authorization only after a committed diagnosis and
correction. Scientific failure cannot be retried.

## Synthetic engineering gates

Before cached input access, v2 must independently prove:

- FP64 equation agreement with a NumPy loop within 1e-10;
- validity-weighted channel pooling and masked softmax oracles;
- masked-BatchNorm train/eval/running-stat NumPy oracles;
- exact invalid-fill invariance;
- batch shapes 1/2/32, finite gradients and parameter updates for all intended
  groups;
- DDF/static parameter counts 9,380/9,435 and sidecar parameters <=15,000;
- ONNX opset-17 standard-domain export/checker;
- ONNX Runtime maximum absolute error <=1e-5;
- TensorRT parser and serialized-engine build without a plugin;
- batch-1 FP16 mean/p95 <=2.0/2.5 ms;
- batch-32 FP16 mean/p95 <=2.5/3.0 ms;
- sidecar peak CUDA allocation <=128 MiB.

V1 synthetic evidence does not waive new v2 validity/loss/causal tests.

## Conditional XAI and shifted conditions

Only after all clean gates pass may automatic and manual XAI run on the 30
preselected train-only rows from v1.

Input gradients target the raw union score and the `r*` selected-pair score;
maps are the channel-summed absolute gradient on the prepared 64x64 tensor.
Integrated gradients use an exact-zero normalized baseline with the same valid
mask, `alpha=k/32` for 33 endpoints, and trapezoidal gradient integration.
Each 64x64 nonnegative map is reduced to 16x16 by non-overlapping 4x4 **sum**
pooling, masked by `valid16`, then normalized to unit valid-cell mass; zero-mass
maps fail. No bilinear resize is permitted for region-mass statistics.

At 16×16, valid border is `valid AND NOT binary_erode(valid,3x3,one iteration)`;
bbox border is `bbox XOR binary_erode(bbox,3x3,one iteration)`. Report fruit
bbox, valid border, bbox border, invalid, entropy, pair-map correlation and
filter-diversity mass/statistics with exact formulas from the clean gates.
The frozen 30-row set contains two bbox-unusable rows (sample indices 58 and
4012). They remain visible with `bbox_supervision_valid=false`; bbox/bbox-border
mass is null and may never be fabricated from a fallback box.

Every deterministic condition starts independently from cached 256x256 sRGB
FP32 `[0,1]`, never from another condition. On valid pixels, brightness adds
`±0.08`; gamma is `x**0.85` or `x**1.15`; contrast is
`(x-channel_mean)*{0.90,1.10}+channel_mean` using the per-channel valid-pixel
mean; and desaturation is `0.5*x+0.5*Y` with fixed Rec.709
`Y=0.2126R+0.7152G+0.0722B`. Invalid RGB is literal zero. Clip once, quantize
with NumPy round-to-nearest-even `rint(255*x).astype(uint8)`, and retain arrays.

The four translations move cached RGB and validity mask together by exactly
four pixels left/right/up/down (one 64x64 sidecar-input cell), with no wrapping
and zero/false fill. When region overlays are computed, bbox geometry moves by
the same displacement. Shifted scores use the clean fold's already fitted
calibrator and action threshold; recalibration/rethresholding is forbidden.
Each condition's TP retention and FP rejection must remain within five
percentage points of clean.

Manual review uses blinded row order and labels each union/matching-pair overlay
as fruit-surface, fruit-boundary, stem, background, padding, crop-border,
bbox-geometry or diffuse. Promotion requires at least 24/30 surface-or-boundary
dominant rows, at most two background-or-padding dominant rows, at most three
crop-border-or-bbox-geometry dominant rows, and at least 8/10 class-4 rows with
surface-or-boundary evidence. An independent second reviewer must reproduce at
least 80% of the categorical decisions; disagreements remain visible.

XAI/shift/manual failure closes production integration even if clean A0 passes.

## Access ledger, replay and resources

The process-wide file-open ledger is installed before any candidate/cache
input is opened. Only the interpreter/standard-library bootstrap and reading
the committed protocol/authorization needed to install that ledger may precede
it, under their own fixed bootstrap allowlist. The authorization provides an
exact runtime allowlist. Read and write-like modes, ordered paths, counts and
hashes are retained. Any raw-data, validation/test, external model, unlisted
path, child process or write-like input access fails closed.

The formal output is atomic and must include fold/group mappings, orders,
donor assignments, offsets, per-step hashes/telemetry, initial/final role states, raw scores,
calibrators, action decisions, bootstrap arrays, the preselected XAI row IDs,
ledger and an artifact manifest. It must not precompute conditional XAI maps.
Both a completed PASS and a completed scientific FAIL are finalized atomically
and retained with their full evidence. Only incomplete temporary payloads from
a crash/harness exception are removed, after an atomic failure tombstone with
the authorization lineage, ledger, exception and surviving hashes is written.

One fresh process replays the formal with the same authorization lineage. All
numeric arrays, discrete arrays, state tensors, metrics, decisions and ordered
ledger events must be byte-exact or have declared `max_abs=0`; path prefixes
are canonically replaced before comparison. Retained numeric/state payloads use
canonical NPY/JSON encodings with sorted manifest keys rather than relying on
pickle/ZIP container byte stability, and their hashes must match. Replay may
not reuse in-memory state.

Per formal or replay limits are 1,800 seconds wall time, 2 GiB peak CUDA, 8 GiB
peak RSS, 1 GiB temporary storage and 512 MiB retained output. No child process
is allowed. Before launch, record estimated 25 role-fold fits × 20 epochs × 8
optimizer steps plus 25 × 32 sequential normalization-recalibration forward
batches, expected runtime, peak resources, evidence gap closed and stop rule. If
the estimate exceeds a limit, no authorization is consumed.

## Progression after A0

Passing A0 establishes a bounded train-only mechanism, not a new TRKH model.
The next protocol must separately freeze zero-init residual integration,
parameter/compute-matched static control, scratch smoke, full-validation probe,
class-1 precision/recall protection, full roles, XAI, robustness,
ONNX/TensorRT parity and artifact retention. Current-best commands cannot change
until that matched sequence wins.

If v2 fails scientifically, do not mine another nearby RGB operator on this
cohort. The preferred new-information route is a prospective controlled
dual-illumination or cross-polarized paired-view dataset with immutable
fruit/session identity, fixed capture/color-chart protocol, fruit/session split
before review, and an unseen later-time/site holdout. Without new data, retain
the negative DDF result and rigorous baseline comparison rather than claiming a
new winning architecture.
