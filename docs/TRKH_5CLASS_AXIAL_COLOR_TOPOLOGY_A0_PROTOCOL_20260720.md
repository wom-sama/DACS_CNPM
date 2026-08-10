# TRKH 5-Class Axial Color Topology A0 Protocol - 2026-07-20

## Status

Precommitted before implementation and before any axial-descriptor metric is
computed. This document authorizes one train-only information audit. It does
not authorize validation, test, trainer integration, a checkpoint, a model
smoke, a parameter sweep, or a full train.

## Research Question

Can low-frequency chromatic topology along the fruit axis distinguish true
class-1 examples from restricted `0/2/4 -> 1` false positives better than the
same ROI's global color statistics, while preserving existing class-1 true
positives under clean, dim, bright, and low-contrast conditions?

The candidate is intentionally not another texture-energy, blob-count,
foreground-localization, generic quadrant-pooling, or output-threshold route.
It asks whether *where* a smooth color transition occurs across the fruit adds
information beyond *which colors* are globally present.

## Primary Evidence and Limits

- Nordey et al., Journal of Plant Physiology 2014,
  `https://doi.org/10.1016/j.jplph.2014.07.009`, measured peel and flesh color
  over maturity stages and reported significant spatial and temporal variation
  in all analyzed quality indicators. This supports testing spatial color, not
  assuming a deterministic RGB maturity rule.
- Ngamchuachit et al., Journal of the Science of Food and Agriculture 2016,
  `https://doi.org/10.1002/jsfa.7597`, sampled the stem-scar-to-blossom-end
  axis and reported that the stem end ripened faster than the blossom end.
  The local author-hosted PDF is
  `D:/DataAI/external_sources/papers/mango_spatial_variance_jsfa_2016.pdf`,
  SHA-256 `ef97ba28784e8864e6c8a61bd1b349fa9de5238da13a20c490d57da49acb1f4d`.
- The 2016 result is primarily flesh physiology. It motivates this external
  peel-image hypothesis but does not prove that the five TRKH labels are
  visually identifiable. The audit must therefore fail closed.
- Direct review of the eight 17-model unanimous class-1 residuals confirms
  foreground-dominant, visually adjacent maturity grades. It does not permit
  using validation labels, sequence identifiers, or neighboring frames.

All implementation is original TRKH code. No third-party implementation is
copied into the repository.

## Candidate Screening and No-Repeat Boundary

The following candidates are rejected before code:

- Deep Residual Pooling uses a pretrained DenseNet-161, `448` input, and a
  `300`-epoch notebook recipe. Its `1x1 - sigmoid(identity)` residual followed
  by global average pooling overlaps the rejected Deep-TEN, bilinear, and
  second-order families.
- Graph Texture Network freezes a pretrained ConvNeXt in its official recipe,
  graphs channels rather than explicit image offsets, and its released hard
  threshold prevents gradients to the advertised masking matrix. It overlaps
  the closed ViG graph family.
- HEX has no repository license, warns that optimization from initialization
  is difficult, and targets removal of superficial statistics rather than the
  required class-1 precision direction.
- Co-Occurrence Layer gives essentially no CIFAR accuracy gain over its source
  convolution in the primary paper and has no official reusable code. Deep
  Co-occurrence uses pretrained ResNet-152 and an unlicensed MATLAB release.
- Morphology NAS is GPL-3.0, requires architecture search plus multi-day CIFAR
  training, and has no prospective class-1 direction. Fixed surface-blob
  morphology already failed its TRKH source-safe gate.

Also closed locally: global/diffuse color readouts, Gray-Edge, photometric
invariants, LBP/HOG/Gabor/wavelet/SRM, surface blobs, lacunarity, Haar detail,
quadrant pooling, high-resolution tiles, Deep-TEN, covariance/bilinear pooling,
part pooling, graph mixing, and generic foreground suppression. A0 may reuse a
global color descriptor only as the matched control; it may not reinterpret
that closed control as a candidate.

## Locked Provenance

- Data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256 `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Train-only CIDT declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`,
  SHA-256 `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- CIDT summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.
- Current-best commands SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`.
- Command-history SHA-256:
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

Every hash must match before image access. The current-best command and history
must remain byte-identical throughout A0.

## Locked Data Scope

- Use only immutable `yolo_f/train=9215`, with class counts
  `1941/541/1920/2520/2293` and `8064` source groups.
- Reuse the exact CIDT five-fold source assignment. Fold counts are
  `1843/1830/1828/1851/1863`; every fit/holdout source overlap is zero.
- Construct neither validation nor test datasets. The existing validation/test
  paths may only be rejected by guard code; pixels, labels, predictions, and
  metrics may not be read.
- Reuse the CIDT condition definitions exactly:
  `clean=(1.0,1.0)`, `lighting_dim=(0.7,0.9)`,
  `lighting_bright=(1.25,1.1)`, and `low_contrast=(1.0,0.65)`, where the pair is
  PIL brightness then contrast.
- Apply each condition before the keeper's exact clean evaluation transform.
  Dataset order, sample index, path, target, source stem, and condition-specific
  keeper probabilities must align exactly with the CIDT declaration.

## Locked ROI and Color Channels

For every transformed object row:

1. Denormalize with the checkpoint input mean/std.
2. ROI-align the metadata `crop_bbox`, eroded by exactly `0.05`, to
   `128 x 128`, `sampling_ratio=2`, `aligned=True`.
3. Define pixel-center coordinates `x,y` in `[-1,1]`. Keep the fixed ellipse
   `x^2 + y^2 <= 0.92^2`.
4. Reuse the documented achromatic-highlight exclusion proxy: remove the top
   5% of `HSV_V * (1 - HSV_S)` inside the ellipse, subject to value being at
   least the ellipse median. This mask is nuisance control only and contributes
   no predictive scalar.
5. Build five chromatic maps: CIE-Lab `a/128`, CIE-Lab `b/128`,
   `R/(R+G+B+1e-6)`, `G/(R+G+B+1e-6)`, and HSV saturation.
6. Smooth each map with one fixed Gaussian `sigma=2.0`, reflect boundary.
7. Within the valid mask, robust-standardize each map by median and
   `max(1.4826*MAD, 1e-3)`, then clip to `[-4,4]`.

No segmentation model, GrabCut, learned mask, sequence metadata, class label,
keeper prediction, or condition-specific threshold may alter the ROI or maps.

## Locked Descriptor Equation

The matched global control is the existing 25-dimensional diffuse-color
descriptor over the same ROI and valid mask: RGB and Lab mean/std, circular
hue mean/concentration, saturation/value mean/std, and luminance/chroma
`0.10/0.50/0.90` quantiles.

The candidate-only axial descriptor is fixed as follows:

1. Use four unsigned axes `theta in {0,45,90,135}` degrees.
2. For each axis compute `t=x*cos(theta)+y*sin(theta)` and
   `u=-x*sin(theta)+y*cos(theta)`. Keep valid pixels with `|u| <= 0.70`.
3. Split those pixels into five equal-support bins by fixed coordinate quantiles
   of the ellipse/strip geometry. Compute five robust-standardized channel means
   `q0..q4` for each of the five channels.
4. Emit seven reflection-invariant values per axis/channel:
   `(q0+q4)/2`, `(q1+q3)/2`, `q2`, `|q4-q0|`, `|q3-q1|`,
   `|(q4-q0)+(q3-q1)|/2`, and
   `sum_i |q(i+1)-q(i)| / 4`.
5. For each channel/value pair, sort the four axis responses ascending before
   flattening. This makes axis sign irrelevant and makes 45-degree axis
   permutations representation-invariant.

Dimension is exactly `5 channels * 7 values * 4 sorted axes = 140`.

All three readout roles have exactly 165 input columns and no bias:

- `control = [global25, zeros140]`;
- `candidate = [global25, axial140]`;
- `placebo = [global25, source-deranged axial140]`.

For every fold, derange placebo rows independently inside fit and holdout,
never cross partitions, and allow no row to receive an axial descriptor from
the same source group. Reuse the same holdout derangement for all four
conditions. There is no channel, axis, sigma, mask, bin, feature, or dimension
sweep.

## Locked Readout and Replay

For each of the five source folds:

- fit scaler statistics on clean fit rows only and reuse them for every
  condition of that fold;
- fit zero-initialized multinomial offset residuals on clean fit rows only;
- use the exact condition-specific keeper log probabilities as offsets;
- use `C=0.3`, L-BFGS-B, `max_iter=300`, no bias, class weights, balanced
  sampling, threshold fitting, or candidate-specific optimization;
- evaluate only the held-out source fold, then aggregate five OOF partitions.

Persist one summary JSON, all-condition OOF predictions CSV, per-fold metrics,
descriptor diagnostics, conditional contact sheets if authorized, and a
nonbinary SHA-256 artifact manifest. Independent replay must reconstruct every
metric, transition, direction score, gate, path binding, and payload hash from
the retained CSV/JSON artifacts. Write no model, checkpoint, ONNX, engine, or
test output.

## Automatic Gates

All structural gates must pass:

- exact provenance hashes and current-best hashes;
- exact row/class/source/fold/condition counts and zero source overlap;
- exact ROI, mask, channel, descriptor, and role dimensions;
- finite descriptors, scalers, weights, logits, and probabilities;
- minimum per-axis/bin valid support `>=128` pixels for every row;
- axial effective rank `>=12`;
- all 15 readouts converge before the iteration cap;
- every placebo permutation is source-deranged and partition-local;
- validation/test data unused and raw dataset unchanged.

Clean candidate-minus-control gates must all pass:

- macro F1 delta `>= +0.002`;
- class-1 F1 delta `>= +0.005`;
- class-1 precision delta `>= +0.010`;
- class-1 recall delta `>= -0.005`;
- restricted `0/2/4 -> 1` FP reduction `>=4`;
- corrections greater than harms;
- class-1 FN rescues at least class-1 TP breaks;
- maximum non-focus per-class F1 drop `<=0.010`;
- candidate exceeds placebo by macro F1 `>=0.002` and class-1 F1 `>=0.005`;
- candidate-minus-control class-1 probability direction AUROC, with keeper FN
  positive and restricted keeper FP negative, `>=0.60`.

Robustness gates must all pass:

- class-1 precision delta is nonnegative in all three shifted conditions;
- worst shifted class-1 F1 delta `>=-0.005`;
- worst shifted class-1 recall delta `>=-0.015`;
- shifted restricted FP removals are at least creations in every condition and
  aggregate net reduction is `>=6`;
- FN rescues are at least TP breaks in every condition;
- maximum shifted macro F1 drop `<=0.005`;
- median shifted FN-versus-restricted-FP direction AUROC `>=0.58`.

## Conditional Visual Gate

Only a complete automatic pass may generate four contact sheets. Each sheet
must reserve available FP removals, FN rescues, TP breaks, and FP creations and
show the RGB ROI, valid mask, five axial bins for the dominant axis, and the
five channel profiles for control/candidate decisions. Manual review passes
only if beneficial actions correspond to fruit-local chromatic topology and
not background, hand, glare, crop edge, blur, or global exposure. Any failed
page closes A0.

## Stop and Promotion Rules

- If any automatic or visual gate fails, reject axial color topology before
  trainer integration. Do not sweep axes, bins, channels, sigma, erosion,
  ellipse, highlight quantile, scaler, residual C, fold, seed, threshold, or
  combine it post hoc with closed color/texture routes.
- A complete A0 pass authorizes one default-off differentiable axial-moment
  branch and focused tests, followed by one bounded no-test smoke/probe. It does
  not directly authorize a full train.
- A full train remains authorized only after the integrated candidate passes
  validation, robustness, XAI, resource, export, and independent-reload gates.
  Select epochs and patience from measured convergence, never exceed 30 epochs,
  and use the measured Windows loader baseline `train/eval workers=4/2` unless
  a fresh benchmark proves a better stable setting.
- Current-best commands and command history change only after a true locked
  validation winner. Test remains final and untouched until promotion.
