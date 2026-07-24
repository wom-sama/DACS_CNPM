# TRKH 5-Class Cross-Colour-Ratio Surface A0 Protocol

Date: 2026-07-25

Protocol ID: `trkh_cross_colour_ratio_surface_a0_20260725`

State: prospective; no CCR candidate implementation, fit, candidate metric,
validation/test access, trainer integration, or command promotion exists

## Decision Scope

This A0 asks one narrow train-only question:

> Does a scratch-trained local class-evidence head on physics-derived
> Cross Colour Ratio (CCR) maps distinguish true class 1 from the current
> restricted `0/2/4 -> 1` false positives while preserving class-1 true
> positives and supporting the current class-1 false negatives?

The gate may authorize one separately locked, default-off image-model smoke
only if every equation, information, pair, causal, robustness, replay, XAI,
resource, ONNX, and TensorRT-feasibility gate passes. It does not authorize:

- validation or test access during A0;
- a production threshold or post-hoc router;
- changing the immutable raw `class_f` or `yolo_f` data;
- pretrained or external classifier, teacher, router, or inference weights;
- a probe, full train, export package, video test, keeper replacement, or
  current-best command/history update;
- claiming that every class has reached F1 greater than 0.98.

The protocol and machine lock must be committed and pushed before candidate
engine code or candidate metrics exist. Failure closes this exact CCR view,
head, folds, controls, optimization, thresholds, and nearby descriptor
variants.

Lock write mode is allowed only while both HEAD and branch upstream equal the
declared parent commit. After the prospective commit, `--check-only` verifies
the frozen lock-file SHA and the complete regenerated payload while treating
the lock-time HEAD/upstream fields as historical; it also requires the parent
commit to remain an ancestor of current HEAD and upstream.

## Primary Sources And Equation Authority

Primary equation sources:

1. Theo Gevers and Arnold W. M. Smeulders, *Color-based object
   recognition*, Pattern Recognition 32(3), 1999:
   <https://staff.fnwi.uva.nl/th.gevers/pub/GeversPR99.pdf>
2. Anil S. Baslamisli and Theo Gevers, *Invariant Descriptors for Intrinsic
   Reflectance Optimization*, JOSA A 38(6), 2021:
   <https://arxiv.org/abs/2204.04076>
3. Partha Das et al., *IDTransformer: Transformer for Intrinsic Image
   Decomposition*, ICCVW 2023:
   <https://openaccess.thecvf.com/content/ICCV2023W/NIVT/html/Das_IDTransformer_Transformer_for_Intrinsic_Image_Decomposition_ICCVW_2023_paper.html>

Pinned local papers:

- `D:\DataAI\external_sources\papers\Gevers_Smeulders_Color_Based_Object_Recognition_PR_1999.pdf`,
  SHA-256
  `f8d03a11ad0ca4970165a75749d058ef3f9f19f40a661bb75d60b3f6f5286678`;
- `D:\DataAI\external_sources\papers\Baslamisli_Invariant_Descriptors_JOSAA_2021.pdf`,
  SHA-256
  `38f0d8781b413c5466a70d86357f8edbe6c3be13d81ba3deb9a806729c719513`;
- `D:\DataAI\external_sources\papers\Das_IDTransformer_ICCVW_2023.pdf`,
  SHA-256
  `471717bbf6dfd7722f33d85b505275e67f46ff530af888d60cd523900a5d110f`.

The authors' IDTransformer project page still labels code as coming soon.
No external implementation is copied, imported, or executed. TRKH implements
the published equations independently with ordinary PyTorch operations.

## Physical Equation And Adverse Assumptions

For neighboring pixels `x1` and `x2`, the three cross-colour ratios are:

`M_RG = (R_x1 * G_x2) / (R_x2 * G_x1)`

`M_RB = (R_x1 * B_x2) / (R_x2 * B_x1)`

`M_GB = (G_x1 * B_x2) / (G_x2 * B_x1)`

In log space:

`log M_RG = log(R/G)_x1 - log(R/G)_x2`

and analogously for `R/B` and `G/B`. The original paper therefore computes
Gaussian derivatives of `log(R/G)`, `log(R/B)`, and `log(G/B)`. Under the
paper's local image-formation assumptions, common geometry/shading terms and
locally constant per-channel illumination cancel.

This is not an unconditional invariant. The derivation assumes approximately
linear narrow-band RGB sensing, locally constant illumination spectra, and a
predominantly diffuse/matte surface. JPEG tone curves, auto white balance,
clipping, sensor noise, wet/specular highlights, and mixed illumination can
break it. Mango class labels also contain firmness, sweetness, sourness, and
transport-risk traits that RGB cannot always observe.

Those limitations are part of the gate. A performance gain cannot be credited
to CCR unless it exceeds a same-size Colour Ratio control and collapses under
an alignment-destroying placebo. The RGB keeper remains present in any later
architecture; CCR may only become a complementary surface branch.

## Local No-Repeat Boundary

- CIConv-W is closed on the current keeper. It exposed edges but produced
  class-1 TP destruction and error-direction AUROC `0.55879`. CCR is admitted
  only because it tests the exact cross-channel log-ratio cancellation rather
  than CIConv's W invariant.
- Context Gray-Edge, learned/context illuminant estimation, diagonal colour
  constancy, generic illumination consistency, Retinex-like correction,
  AugSelf colour adapters, quaternion colour rotation, chromatic topology,
  Gabor/LBP, morphology, frequency, CutPaste, NMF, local MIL, prototypes, and
  CAP remain closed.
- The matched Colour Ratio control is not a new candidate. It is required to
  prove that any gain comes from cancelling geometry/shading rather than from
  another log-colour derivative view.
- No descriptor scale, Gaussian sigma, crop margin, epsilon, clipping bound,
  head width, loss, class weight, fold, seed, threshold-retention target, or
  fusion weight may be swept after candidate outputs exist.

## Immutable Inputs

Only these scientific inputs are allowed:

- `runs/audit_attention_maxsep_prototype_a0_20260721/cohort_geometry.npz`;
- `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`;
- the 763 ordered `yolo_f/images/train` object-row image references selected
  by that geometry/CIDT cohort;
- the corresponding unique `yolo_f/labels/train` files needed to reproduce
  object crops;
- `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`;
- the frozen keeper checkpoint and resolved configuration for input semantics
  and immutable baseline provenance;
- the three pinned papers;
- the current-best command/history files for byte-identity protection;
- this protocol and its generated machine lock.

The lock records exact input-file hashes, ordered path/content-hash digests,
cohort arrays, source folds, object boxes, descriptor rolls, visual anchors,
runtime versions, and protected state. A process-wide file ledger must block:

- any complete `val`, `valid`, `validation`, or `test` path component;
- any write below `class_f` or `yolo_f`;
- any unlisted scientific input;
- any external checkpoint or feature file.

## Locked Cohort And Source Folds

The cohort is exactly 763 train objects:

- class-1 positives: 541, comprising 528 keeper TP and 13 keeper FN;
- restricted false positives: 222;
- target 0: 158;
- target 2: 54;
- target 4: 10;
- legacy ordered sample-index SHA-256:
  `59d2146617f2642f99c081f49a9aa5812eae4206b27887a1fa5476f6f61c85b2`.
- dtype/shape-prefixed machine array SHA-256:
  `ad51a9bdbf6acc7449ad8d8f65b3dc69971c318d7f64e2effd814bac8f77fe05`.

The immutable CIDT source folds are reused byte-for-byte. For outer fold `f`:

- held fold: `f`;
- calibration fold: `(f + 1) mod 5`;
- fit folds: the remaining three.

Fit, calibration, and held source stems must be disjoint. Fit rows update
weights. Calibration rows select one suppression threshold. Held rows are
scored once after epoch 20 and cannot select epochs, roles, hyperparameters,
or thresholds.

Target-4 evidence is sparse: the held fold counts are `1/0/3/3/3`. Therefore
target-4 gates are aggregate and exact-count based; no unsupported per-fold
AUROC is invented.

## Frozen Input View

Reproduce the keeper's ordinary classification object crop and evaluation
semantics:

- `yolo_f/train`;
- primary-object crop margin `0.05`;
- image size `256`;
- resize mode `pad`;
- evaluation illumination normalization and strength from the frozen keeper;
- no augmentation, source-context synthesis, background suppression,
  foreground crop, surface amplification, or label-dependent preprocessing.

The model-normalized tensor is inverted with the keeper's exact input
mean/std to recover the deployable post-transform sRGB view. This guarantees
that a later branch can run from the same tensor without a second decoder.
The image valid mask removes letterbox padding. The transformed object bbox is
retained for audit/XAI only and is not an input channel.

Downsample valid sRGB to `96x96` with area interpolation, convert standard
sRGB to linear RGB, and apply a fixed separable Gaussian with sigma `1.0` and
truncate radius `3`. Values are clamped below by `1/255` only for the log.
Pixels are reliable only when both derivative-support neighborhoods are valid
and all channels lie strictly between `1/255` and `254/255`. Unreliable
responses are zero. Empty reliable support is fatal.

## Candidate And Controls

### CCR candidate

Construct three log-chromaticity planes:

`C = [log R - log G, log R - log B, log G - log B]`.

Apply the fixed horizontal and vertical Gaussian-derivative kernels to obtain
six signed channels. Clamp only nonfinite protection to `[-8,8]`; record the
clip fraction. No magnitude, histogram, Canny non-maximum suppression,
threshold, learned colour correction, or label-conditioned transform is used.

### Matched Colour Ratio control

Apply the same horizontal/vertical derivatives to:

`CR = [log R, log G, log B]`.

This also yields six channels and uses byte-identical masks, head structure,
initialization, training orders, and optimization. It retains shading/geometry
variation and asks whether cross-channel cancellation adds information.

### Spatially dephased CCR control

Independently toroid-roll each of the six CCR channels in both spatial axes.
Each nonzero offset is derived only from
`NumPy SeedSequence([seed, sample_index, channel])` and is fixed before
fitting. This
preserves each channel's complete value multiset while destroying co-located
cross-channel transitions. Fit, calibration, and held maps receive their own
row's same fixed dephasing.

The trained candidate is also evaluated once on dephased held maps without
refit or threshold change. A causal CCR mechanism must degrade.

### Fixed keeper control

Use the frozen keeper margin:

`log(p_class1) - log(max(p_class0, p_class2, p_class4))`.

Class 3 is excluded because it is not in the restricted cohort. No baseline
probability enters any learned head.

## Scratch Class-Evidence Head

Every trainable role has exactly 3,004 trainable parameters:

1. `Conv2d(6,24,3,padding=1,bias=False)`;
2. `GroupNorm(6,24)`, GELU, `AvgPool2d(2)`;
3. depthwise `Conv2d(24,24,3,padding=1,groups=24,bias=False)`;
4. `Conv2d(24,48,1,bias=False)`;
5. `GroupNorm(8,48)`, GELU, `AvgPool2d(2)`;
6. `Conv2d(48,4,1,bias=True)` class-evidence maps.

The four output classes are ordered `[0,1,2,4]`. Logits are exact
reliability-mask-aware spatial means of the four maps. The three faithful
pair maps are `A_1-A_0`, `A_1-A_2`, and `A_1-A_4`.

There is no pretrained state, keeper feature, attention, token selector,
threshold inside the model, bbox channel, source ID, fold ID, image number,
class/sample weight, oversampling, distillation, or external feature.

## Locked Optimization

Train exactly these roles:

- `colour_ratio_control`;
- `cross_colour_ratio_candidate`;
- `cross_colour_ratio_seed_repeat`;
- `cross_colour_ratio_spatial_dephased_control`.

Settings:

- device: CUDA;
- deterministic algorithms enabled; TF32 disabled;
- primary seed `20260725`; fold seed `20260725 + fold`;
- repeat seed offset `+100000`;
- 20 epochs, no early stopping or epoch selection;
- batch size `64`, drop-last false;
- natural-frequency rows and one deterministic shuffle per epoch;
- unweighted four-class cross entropy;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- betas `(0.9,0.999)`, epsilon `1e-8`;
- gradient-norm clip `1.0`;
- two-epoch linear warmup, then cosine decay to zero;
- FP32 optimizer state and BF16 CUDA autocast;
- requested/effective data workers exactly `4/4`;
- no augmentation, synthetic image, sampler, class weight, threshold loss,
  keeper-logit feature, or hyperparameter sweep.

Primary candidate/control roles start from byte-identical states in each fold
and receive identical fit orders. Every parameter must receive a finite
gradient on the first update and differ from initialization after fitting.

## Fold-Safe Action And Pair Scores

The conservative class-1 evidence score is:

`s = logit_1 - logsumexp(logit_0, logit_2, logit_4)`.

For every fold and role, choose the highest calibration threshold that retains
at least 97% of calibration class-1 rows. Ties move toward the larger
threshold only when retention remains valid. Apply it once to the held fold.

A held keeper class-1 prediction is suppressed only when `s` is below the
threshold; replacement is the keeper's highest-probability non-class-1 class.
The action rule never becomes a validation or production threshold.

Report:

- aggregate and fold AUROC/AUPRC for class 1 versus all restricted FP;
- AUROC/AUPRC for `1-vs-0`, `1-vs-2`, and aggregate `1-vs-4`;
- pair margins `logit_1-logit_rival`;
- keeper-TP retention, keeper-FN support, and restricted-FP rejection;
- target-specific rejection for 0, 2, and 4;
- corrections, TP harms, FN supports, and every transition;
- full 9,215-row train-only classification effect when held suppressions are
  inserted into the locked CIDT clean predictions.

## Clean Promotion Gates

All gates are conjunctive:

- candidate class1-vs-restricted-FP AUROC `>= 0.86` and AUPRC `>= 0.92`;
- candidate aggregate AUROC gain over keeper margin `>= 0.02`;
- candidate aggregate AUROC gain over Colour Ratio `>= 0.02`;
- candidate aggregate AUROC gain over trained dephasing `>= 0.04`;
- same-weight dephasing lowers candidate AUROC by at least `0.03`;
- candidate beats Colour Ratio and trained dephasing in at least four of five
  folds;
- pair AUROC `1-vs-0 >= 0.82`, `1-vs-2 >= 0.80`, and aggregate
  `1-vs-4 >= 0.75`;
- all-class1 and keeper-TP retention each `>= 0.95`;
- restricted-FP rejection `>= 0.25`, with target-0 and target-2 rejection
  each `>= 0.20` and at least `2/10` target-4 FP rejected;
- at least `8/13` keeper class-1 FN rows remain positively supported at their
  fold threshold;
- corrections plus FN supports are at least twice TP harms;
- full train-only macro F1 gain `>= 0.002`;
- full train-only class-1 precision/F1 gains `>= 0.03/0.01`;
- full train-only class-1 recall drop `<= 0.03`;
- every nonfocus-class F1 drop `<= 0.003`;
- seed-repeat AUROC difference `<= 0.01`, action agreement `>= 0.95`, and
  identical pass/fail decision;
- exact fresh-process replay of all states, maps, scores, thresholds,
  actions, metrics, and manifests within `1e-7`.

Failure of any clean gate closes shifted conditions, production integration,
validation/test, image-model smoke, and command promotion.

## Structural, Robustness, XAI, And Deployment Gates

Before metrics:

- NumPy FP64 and PyTorch FP64 direct-ratio/log-difference equations agree
  within `1e-12`;
- common per-pixel shading and spatially constant per-channel gains change
  reliable unclipped CCR values by at most `1e-6`;
- swapping neighbor order negates signed CCR within `1e-12`;
- Colour Ratio and CCR inputs have identical shapes, masks, and finite
  support;
- all role parameter counts, initial states, orders, gradients, and update
  counts match the lock;
- no unknown Python/TensorRT process or foreign GPU workload overlaps formal
  execution.

Only after every clean automatic gate passes, score the frozen fold heads
without refit under fixed `dim`, `bright`, and `low_contrast` transforms.
Each condition requires AUROC `>= 0.82`, keeper-TP retention `>= 0.92`,
restricted-FP rejection `>= 0.20`, and candidate AUROC at least `0.02` above
trained dephasing.

Build one fixed 30-row train-only contact sheet:

- one TP, FN, `0->1`, and `2->1` row per fold;
- all ten `4->1` rows.

Show source image+bbox, model crop, reliability mask, the three candidate pair
maps, Colour Ratio contrast, trained-dephased contrast, score, threshold, and
action. Require finite nonconstant maps, mean pair-map attribution mass inside
the transformed bbox `>= 0.75`, border mass `<= 0.25`, and manual evidence on
fruit colour transition, lesion, wrinkle, or surface structure. Padding,
crop border, hand, background, or broad silhouette shortcuts fail.

If clean science and manual XAI pass, benchmark the keeper and a wrapper that
returns keeper logits plus the CCR head on the same input:

- candidate head parameters exactly 3,004 and total parameter ratio
  `<= 1.01`;
- batch-1 mean and p95 latency each `<= 1.15x` keeper;
- batch throughput `>= 0.85x` keeper;
- peak end-to-end CUDA allocation `<= 1.15x` keeper and below `6.0 GiB`;
- formal process RSS below `10.0 GiB`;
- ONNX Runtime CPU probability error `<= 1e-5` with exact actions;
- standard ONNX operators only, no custom operator;
- TensorRT FP32 build succeeds and probability error is `<= 1e-4`, or a
  predeclared unsupported standard operator is recorded as a failed gate.

Accuracy cannot override a failed inference gate.

## Resource And Stop Rules

Prospective ceilings:

- new image reads: exactly the locked 763 train object rows per condition;
- clean temporary CR+CCR cache: at most `0.40 GiB`;
- retained A0 payload: at most `0.10 GiB`;
- peak CUDA allocation: below `2.0 GiB` for descriptor-head fitting;
- complete clean formal wall time: at most 20 minutes;
- no validation/test read, no keeper forward, and no full image-model train;
- stop immediately on a hash, source-fold, file-ledger, equation, finite,
  gradient, memory, or process-isolation failure.

Temporary maps and failed export scratch are deleted in a verified `finally`
path after compact predictions, states, XAI arrays/sheet, summaries, replay,
and manifests are hashed. The result receives exactly one value label:
`promotable`, `mechanism-only`, `negative-but-reusable`, or `waste/invalid`.

## Downstream Authorization

Only a complete conjunction may authorize:

1. one default-off CCR surface branch fused with the scratch RGB
   CNN/Transformer model;
2. one separately prospectively locked matched validation smoke;
3. complete validation metrics, robustness, architecture trace, XAI, batch-1
   mean/p95, throughput, VRAM, ONNX parity, and TensorRT feasibility review.

A smoke winner must still meet the repository class-1 precision/recall/F1 and
macro-F1 promotion milestone before a full run. Any full run remains limited
to at most 30 epochs with patience 3. The test split remains final-report-only.

If A0 fails, do not sweep signed versus magnitude CCR, gamma/linear RGB,
epsilon, sigma, derivative kernel, crop/context margin, resolution, head
width/depth, normalization, optimizer, class weighting, seed, fold,
threshold, or fusion. Preserve compact negative evidence, update the journal,
TODO, living report and TRKH skill, then select only an equation-distinct
successor.
