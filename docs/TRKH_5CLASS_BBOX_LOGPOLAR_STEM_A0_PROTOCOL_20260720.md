# TRKH 5-Class BBox-Centered Log-Polar Stem A0 Protocol

Date: 2026-07-20

Status: prospectively locked before auditor implementation, descriptor extraction,
or candidate metric inspection.

## Question

Can direct object-centered log-polar sampling of the mango RGB support **before**
the frozen TRKH CNN stem expose a spatially grounded class-conditional cue that
separates true class 1 from restricted `0/2/4 -> 1` false positives, while
preserving class-1 recall and outperforming matched Cartesian feature pooling,
linear-polar sampling, and a source-deranged placebo?

This is an information/readiness audit. It is not a validation result, a model
change, or permission to report a new best checkpoint.

## Why This Route Is Distinct

- C4 probability consensus was applied after the complete classifier and failed
  because rotation averaging suppressed true class-1 probability more strongly
  than restricted false-positive probability. This protocol does not average
  rotations or impose a prediction-consensus rule.
- Axial color topology summarized fixed Cartesian Lab/chromaticity profiles and
  became a broad class-1 suppressor. This protocol resamples RGB before learned
  convolution and compares it directly with feature-space polar pooling.
- Existing context, bbox, objectness, color-statistic, texture, morphology,
  topology, frequency, part, MIL, and second-order branches did not establish a
  selective true-class1-versus-FP direction. The present mechanism changes the
  coordinate system seen by learned local filters while retaining the current
  bbox-centered object support.
- The proposal is not generic rotation invariance. The required gate is
  class-conditional precision selectivity under a fixed object-centered spatial
  representation.

## Primary Sources And Frozen Provenance

### Direct log-polar sampling

Ebel et al., *Beyond Cartesian Representations for Local Descriptors*, accepted
at ICCV 2019, compare sampling the image directly on a log-polar grid with
pooling features computed on a Cartesian grid. Their transform oversamples the
center, undersamples the periphery, and uses bilinear interpolation.

- Accepted CVF paper:
  `D:\DataAI\external_sources\papers\beyond_cartesian_logpolar_iccv2019.pdf`
- Paper SHA-256:
  `19718844bc02faaf20f89e595cb81af46cfa221f96a3726e5bc67908f1418e21`
- Official Apache-2.0 repository:
  `https://github.com/cvlab-epfl/log-polar-descriptors`
- Local pinned repository:
  `D:\DataAI\external_sources\official\log-polar-descriptors-iccv2019`
- Commit/tree:
  `45d0a922dcd58e6e64e6cba158e3dad3f62ccfe8` /
  `527738e303157dee4c60997ed73ea3aeb7d20091`
- License SHA-256:
  `3ddf9be5c28fe27dad143a5dc76eea25222ad1dd68934a047064e56ed2fa40c5`

### Polar transformer and angular wrap

Esteves et al., *Polar Transformer Networks*, accepted at ICLR 2018, make
rotation and dilation translations in polar coordinates and apply wrap-around
padding on the periodic angular axis.

- Paper:
  `D:\DataAI\external_sources\papers\polar_transformer_networks_iclr2018.pdf`
- Paper SHA-256:
  `a49644afa3676cf2eb1219892b327fc26f4ac014565821d9ddee4e458d52e304`
- Official MIT repository:
  `https://github.com/daniilidis-group/polar-transformer-networks`
- Local pinned repository:
  `D:\DataAI\external_sources\official\polar-transformer-networks-iclr2018`
- Commit/tree:
  `c6a4ad613bb2feb7a5ba8b25307449bde931b7a3` /
  `a0b328bb45804625cfec436a169b3c92ceaa0cdc`
- License SHA-256:
  `2d212fb778efe3c9a60e3d54f8718dedaf86909ac1a06fd8b773dcfcc8d205d3`

No external dependency, source file, or pretrained weight will be copied into
TRKH. The auditor will implement the fixed equations with existing PyTorch
`grid_sample` and reuse only the frozen keeper stem weights.

## Immutable TRKH Inputs

- Dataset specification: `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`
  - SHA-256: `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`
- Split: train only, exactly `9,215` object rows.
- Class counts: `[1941, 541, 1920, 2520, 2293]`.
- Frozen keeper:
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt`
  - SHA-256: `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Clean/lighting keeper probabilities, source identities, and five fixed folds:
  `runs\audit_cidt_readiness_full_train_20260714`
  - summary SHA-256:
    `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`
  - predictions/fold declaration SHA-256:
    `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`
  - fold row counts: `[1843, 1830, 1828, 1851, 1863]`
  - source overlap: zero.
- Current-best commands SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`
- Command-history SHA-256:
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`

Validation and test paths, labels, images, predictions, and metrics are forbidden.
Raw dataset files are read-only and must have identical path/size/mtime identity
hashes before and after extraction.

## Fixed Image Geometry

The keeper evaluation transform produces normalized `256 x 256` tensors and a
normalized `crop_bbox=(cx, cy, width, height)`. The auditor first de-normalizes
RGB with the checkpoint mean/std.

For every row, construct one Cartesian support patch with bbox scale `1.10`:

1. center is exactly the supplied bbox center;
2. support width/height are `1.10 * bbox width/height`;
3. sample the rectangular support into `256 x 256` RGB with bilinear
   interpolation, `padding_mode=border`, and `align_corners=False`;
4. clamp source coordinates only through `grid_sample` border semantics;
5. no segmentation mask, foreground estimate, learned origin, random rotation,
   test-time augmentation, or image-dependent parameter is allowed.

This normalizes bbox aspect while retaining the narrow surrounding context
available in the keeper crop.

## Fixed Polar Equations

Let output height be angle and output width be radius. For pixel-center indices
`i,j in {0,...,255}`:

```text
theta_i = 2*pi*(i + 0.5)/256 - pi
u_j     = (j + 0.5)/256
R       = 128

rho_log(j) = (exp(u_j * log(1 + R)) - 1) / R
rho_lin(j) = u_j

x = 0.5 + 0.5 * rho * cos(theta)
y = 0.5 + 0.5 * rho * sin(theta)
```

Both polar views use bilinear `grid_sample`, `padding_mode=border`, and
`align_corners=False`. These constants, axis order, angular origin, radius,
support scale, interpolation, and padding are locked.

## Frozen Stem Paths

The exact keeper `HybridConvStem` weights and BatchNorm statistics remain in
evaluation mode. No parameter is fitted or updated during descriptor extraction.

Four roles are compared:

1. **Cartesian post-feature control**: Cartesian support patch -> native frozen
   stem -> the same log-polar grid applied to the `32 x 32` stem map.
2. **Direct log-polar candidate**: Cartesian support patch -> RGB log-polar
   sampler -> frozen stem with periodic padding on the angular/height axis and
   zero padding on the radial/width axis before each `3 x 3` convolution.
3. **Direct linear-polar control**: identical to the candidate except
   `rho_lin` replaces `rho_log`.
4. **Source-deranged log-polar placebo**: the candidate descriptor is replaced
   by a deterministic different-source descriptor within each fit or holdout
   partition. Targets, keeper probabilities, and row order are unchanged.

The periodic convolution path must reuse the exact keeper convolution,
BatchNorm, activation, and pooling parameters. A unit test must prove equality
with the native stem away from padding boundaries and prove angular roll
equivariance within the declared pooling tolerance.

## Fixed Descriptor

Each role yields a `256 x 32 x 32` feature map whose height is angle and width
is transformed radius. Divide the radius axis into four contiguous eight-column
bands. For each band and channel record:

- mean over angle and radial cells;
- population standard deviation over the same cells.

The descriptor dimension is exactly `4 * 2 * 256 = 2,048`. No global color,
handcrafted texture, class-specific mask, coefficient selection, PCA, learned
projection, or extra geometry scalar is appended.

Structural diagnostics must report finite values, per-role effective rank,
pairwise descriptor correlation, descriptor norms, and candidate-minus-control
non-identity.

## Readout And Source-Safe Evaluation

- Seed: `20260720`.
- Use the immutable five CIDT source-disjoint folds; never regenerate folds.
- For each fold and role, fit the same StandardScaler on the four-fold fit
  partition only.
- Fit one multinomial residual logistic readout with fixed `C=0.30`, L2 penalty,
  `lbfgs`, no class weights, and maximum `2,000` iterations.
- The fixed keeper log-probabilities are the offset. The fitted descriptor logits
  are added to that offset before softmax.
- A role with missing classes must expand probabilities to the fixed five-class
  order through `model.classes_`.
- Clean data alone fits scaler/readout. The same fold model is replayed without
  refitting on `lighting_dim=(brightness 0.70, contrast 0.90)`,
  `lighting_bright=(1.25, 1.10)`, and
  `low_contrast=(1.00, 0.65)`.
- Source derangement is deterministic, stays within fit or holdout partition,
  and must map every row to a different source.
- Persist scaler/readout parameters or sufficient exact equations for replay.

The four conditions require `4 * 9,215` transformed rows, but only clean rows
fit models. No threshold, routing rule, class bias, blend weight, temperature,
or post-hoc calibration may be selected.

## Required Metrics And Audits

For keeper, Cartesian control, candidate, linear control, and source placebo,
report per condition:

- accuracy, macro F1, per-class precision/recall/F1/support;
- NLL, Brier score, ECE, and predicted support;
- full confusion matrix and transition table;
- corrections, harms, class-1 FN rescues, class-1 TP breaks;
- restricted `0/2/4 -> 1` FP removals and creations;
- five clean fold metrics.

For candidate minus Cartesian control, report the class-1 probability-change
distribution on:

- true class-1 keeper TP;
- true class-1 keeper FN;
- restricted keeper FP into class 1;
- remaining restricted negatives.

The direction AUROC labels all true class-1 keeper TP/FN rows positive and all
restricted keeper FP rows negative, using candidate-minus-control `delta_p1` as
the score. Report the same AUROC for linear-minus-control and
source-placebo-minus-control.

Always render a fixed transform/activation contact sheet for the first train row
of each class, sample indices `[1, 2, 3, 46, 0]`, in class order `0..4`. Each row
must show Cartesian support, linear-polar RGB, log-polar RGB, Cartesian
post-feature activation norm, and direct log-polar activation norm. This sheet
validates geometry only and cannot change the locked gate.

## Automatic Pass Gate

Every structural check and every metric check is conjunctive.

### Structural

- every provenance hash, official commit/tree, and license hash is exact;
- both official Git checkouts are clean;
- exactly `9,215` rows and the locked class/fold counts are observed in every
  condition;
- all fold source overlaps are zero and all derangements change source;
- every role has exactly `2,048` finite descriptor values per row;
- all 20 fold-role readouts converge before 2,000 iterations;
- candidate descriptor effective rank is at least `24`;
- candidate is not numerically identical to Cartesian or linear control;
- dataset metadata identity is unchanged;
- validation/test use and checkpoint/model writes are false;
- replay reconstructs probabilities, metrics, transitions, cohort statistics,
  and gates with maximum absolute difference `<=1e-12` from persisted FP64
  readout equations (`<=1e-7` only for explicitly persisted FP32 descriptor
  round trips).

### Clean candidate versus Cartesian post-feature control

- macro F1 delta `>= +0.003`;
- class-1 F1 delta `>= +0.010`;
- class-1 precision delta `>= +0.015`;
- class-1 recall delta `>= -0.005`;
- at least ten net restricted FP removals;
- corrections are not fewer than harms;
- class-1 FN rescues are not fewer than class-1 TP breaks;
- class-1 F1 and precision deltas are positive in at least four of five folds;
- worst-fold class-1 F1 delta `>= -0.010`;
- candidate-minus-control direction AUROC `>=0.62`.

### Representation specificity

- candidate exceeds direct linear-polar control by macro F1 `>=0.002`, class-1
  F1 `>=0.005`, and class-1 precision `>=0.005`;
- candidate exceeds source-deranged placebo by the same three margins;
- candidate direction AUROC exceeds each control/placebo direction AUROC by
  `>=0.03`.

### Lighting robustness

- candidate-minus-Cartesian class-1 precision is nonnegative in all three shifted
  conditions and positive in at least two;
- worst shifted class-1 F1 delta `>=-0.005`;
- worst shifted class-1 recall delta `>=-0.015`;
- candidate has positive total net restricted-FP removal across all three shifts;
- no shifted macro F1 delta is below `-0.005`.

## Decision Rule

- If any automatic gate fails, set status
  `rejected_before_model_integration`; do not open validation/test, modify the
  model/trainer, run smoke/probe/full train, or update current-best commands.
- If every automatic gate passes, perform a manual geometry/activation review.
  A pass authorizes one prospectively locked dual Cartesian/log-polar model
  integration and matched smoke. It does not authorize validation test reuse or
  a full train by itself.
- Any later full train remains subject to the established smoke/probe,
  full-validation, robustness, XAI, loader, and promotion gates, with at most 30
  epochs and measured train/eval workers `4/2` unless re-benchmarked.

## No-Nearby-Sweep Rule

A failure closes this exact bbox center, `1.10` support, `R=128`, raw sampling,
periodic-padding, four-band mean/std descriptor, readout `C`, fold, condition,
and seed configuration. Do not sweep polar origins, radii, support scales,
angle starts, interpolation modes, padding modes, ring counts, descriptor
moments, readout penalties, thresholds, class weights, folds, or seeds on the
current keeper. A future polar route requires a new primary-source mechanism or
independent evidence, not a nearby parameter search.

