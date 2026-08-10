# TRKH 5-Class Chromatic Cubical Persistence A0 Protocol - 2026-07-20

## Status

Prospectively locked before implementation and before any persistence feature
is computed from TRKH data. This document authorizes one train-only information
audit. It does not authorize validation, test, trainer/model integration, a
checkpoint, a smoke, a parameter sweep, or a full train.

## Research Question

Can connected components and holes that persist across chromatic thresholds on
the mango surface separate class-1 positives from restricted `0/2/4 -> 1`
false positives while preserving existing class-1 true positives under clean,
dim, bright, and low-contrast conditions?

This candidate is not another single-threshold blob count, local texture
energy, global color histogram, axial gradient, foreground localizer, or
post-hoc confidence threshold. Cubical persistent homology summarizes how
spatial structures are born and merged across all filtration thresholds. A0
tests whether that information is present before adding any neural branch.

## Primary Evidence and Licensing

- Peng et al., *PHG-Net: Persistent Homology Guided Medical Image
  Classification*, WACV 2024, is the accepted primary architecture evidence.
  It represents cubical persistence diagrams with a neural encoder and uses
  those features to refine CNN or Transformer features. The accepted CVF PDF is
  `D:/DataAI/external_sources/papers/phg_net_wacv_2024.pdf`, SHA-256
  `41c862cb346b482f15f7a4a731909f41adcece65f974bb78096d23ac39af1b96`.
- The paper-linked repository is
  `https://github.com/yaoppeng/TopoClassification`, locked at commit
  `6daa5f7dba556e9882611eb4e2e1c89a67f0d2c5`. It has no declared software
  license. Its source may be inspected for provenance but must not be copied or
  imported.
- Cubical persistence is computed only through GUDHI's documented MIT-licensed
  cubical-complex API. Pin `gudhi==3.11.0` and the local CPython-3.9 Windows
  wheel `D:/DataAI/external_sources/wheels/gudhi-3.11.0-cp39-cp39-win_amd64.whl`,
  SHA-256
  `846ac5807d6a72ebc79a29b9405d12a68cfb8dad896ff5aa6a24fde343a8581f`.
- All TRKH descriptor, control, replay, and visualization code must be original.
  No pretrained model or external image is used.

PHG-Net reports that topology alone is weaker than topology-guided visual
fusion. Therefore A0 asks only whether topology adds a class-1 direction beyond
keeper probabilities and global color. It must not claim that a frozen readout
is the final architecture.

## No-Repeat Boundary

Already closed locally: fixed LoG surface blobs, LBP/HOG/Gabor/SRM/wavelet and
scattering descriptors, multiscale lacunarity, global/diffuse color, Gray-Edge
and photometric invariants, axial color topology, DCT/Fourier branches,
Deep-TEN, covariance/bilinear pooling, patch/part prototypes, WILDCAT negative
evidence, and shallow surface-stat fusion.

The closest controls are deliberately retained but cannot be promoted:

- the existing 25D global color descriptor is the matched non-spatial control;
- deterministic within-row pixel permutation preserves each chromatic
  histogram but destroys topology;
- partition-local source derangement preserves the topology-feature
  distribution but destroys row identity.

If the real topology does not beat both placebos and the global control, the
route is closed. Do not combine weak precision suppression from axial color
with weak recall expansion from another rejected descriptor.

## Locked Provenance

- Data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Train-only CIDT declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`,
  SHA-256
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- CIDT summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.
- Current-best commands SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`.
- Command-history SHA-256:
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

Every hash must match before image access. The tracked TRKH tree must be clean
at the exact pushed branch commit. Current-best command files must remain
byte-identical throughout A0.

## Locked Data Scope

Use only immutable `yolo_f/train` rows declared by CIDT and source folds
`1,2,3,4`. Fold 0, validation, test, manual review labels, sequence neighbors,
and raw-data modification are forbidden.

The clean cohort is fixed before feature extraction:

- all class-1 rows in folds `1..4`: `432`, comprising `421` keeper TP and
  `11` keeper FN;
- restricted hard negatives: `186` rows whose target is in `{0,2,4}` and whose
  clean keeper prediction is class 1;
- total rows: `618`;
- fold `(positive, TP, FN, restricted-FP)` counts:
  `1=(115,112,3,45)`, `2=(104,100,4,48)`,
  `3=(103,101,2,52)`, `4=(110,108,2,41)`;
- ordered sample-index SHA-256:
  `98b6e447ace4e3dc4d9a3fcd5402c25662b018697636c5e9d53b9f4d7e91b2dd`.

Reuse CIDT conditions exactly: `clean=(1.0,1.0)`,
`lighting_dim=(0.7,0.9)`, `lighting_bright=(1.25,1.1)`, and
`low_contrast=(1.0,0.65)`, where each pair is PIL brightness then contrast.
Condition transforms occur before the keeper's exact clean evaluation
transform. Path, sample index, source, target, fold, bbox metadata, and keeper
probabilities must match the declaration.

## Locked ROI and Chromatic Maps

For each transformed row:

1. Denormalize with checkpoint input mean/std.
2. ROI-align `crop_bbox`, eroded by exactly `0.15`, to `48 x 48`, using
   `sampling_ratio=2` and `aligned=True`.
3. Build the same fixed ellipse used by prior surface audits,
   `x^2 + y^2 <= 0.92^2`, from pixel-center coordinates in `[-1,1]`.
4. Exclude achromatic highlights at the strict 95th percentile of
   `HSV_V * (1-HSV_S)` inside the ellipse, subject to value being at least the
   ellipse median. Percentile ties remain valid so a uniform ROI is not erased.
5. Build exactly three maps: CIE-Lab `a`, CIE-Lab `b`, and HSV saturation.
6. Fill excluded pixels by nearest valid-pixel propagation, then Gaussian
   smooth once with `sigma=0.8` and reflect padding.
7. Replace each map by deterministic average ranks over valid pixels, scaled to
   `[0,1]`; fill outside-ellipse pixels with the valid median rank `0.5`.

The rank map removes global monotonic intensity and histogram scale. No mask,
channel, ROI, erosion, sigma, threshold, or resolution is selected from labels
or results.

## Locked Cubical Descriptor

For each of the three rank maps, compute two filtrations:

- sublevel: `f`;
- superlevel: `1-f`.

For each filtration, construct a GUDHI `CubicalComplex` from top-dimensional
cells and compute dimensions `H0` and `H1` over coefficient field `2`.
Discard non-finite intervals and intervals with persistence `<=1/32`.

Vectorize each of the `3 channels x 2 polarities x 2 dimensions = 12`
diagrams into exactly 21 values:

- a Betti curve at thresholds `t_j=j/17`, `j=1..16`, where
  `beta(t)=count(birth <= t < death)`;
- interval count, persistence sum, persistence maximum, and persistence
  quantiles `0.50` and `0.90`.

The topology descriptor dimension is exactly `12 x 21 = 252`. Values must be
finite. The descriptor is invariant to diagram interval order. A synthetic
constant field must produce no finite interval above the persistence floor;
synthetic isolated components and a ring must produce the expected H0/H1
signals, verified independently from binary connected-component and Euler
calculations.

The pixel-permutation placebo uses a per-row, per-channel permutation seeded by
`SHA256("ccp-a0|sample_index|channel")`. The same permutation is reused across
all four conditions and its complement supplies the superlevel map. It must
preserve the sorted map values exactly and alter spatial adjacency on every
non-uniform row.

## Locked Readout Roles

Every role has exactly 282 columns:

- base columns: five condition-specific keeper log-probabilities plus the
  existing 25D global color descriptor;
- `control = [base30, zeros252]`;
- `candidate = [base30, real_topology252]`;
- `pixel_placebo = [base30, pixel_permuted_topology252]`;
- `source_placebo = [base30, partition-local source-deranged_topology252]`.

Source derangement is independent inside every outer fit/holdout partition,
never crosses a partition, and never assigns a descriptor from the same source
group. The held-row derangement is reused across all conditions.

The binary target is one for every true class-1 row and zero for every
restricted hard negative. For each outer held fold:

1. fit `StandardScaler` on clean rows from the other three folds only;
2. obtain inner OOF scores over those three folds by fitting on two folds and
   predicting the third;
3. select the largest observed score threshold whose inner-OOF class-1
   retention is at least `0.98`; ties maximize restricted-FP rejection and then
   choose the larger threshold;
4. fit one `LogisticRegression(C=0.1, solver="lbfgs", max_iter=300,
   tol=1e-8, fit_intercept=True, class_weight=None, random_state=42)` on all
   three outer-fit folds;
5. apply that clean-fitted scaler, readout, and threshold unchanged to the held
   fold in all four conditions.

There is no C, scaler, class weight, fold, seed, threshold, target, feature, or
placebo sweep. All 64 readouts (16 outer fits plus 48 inner fits required for
threshold selection) must converge before the iteration cap. Persist
coefficients and thresholds for independent replay; do not persist a
deployable classifier.

For decision accounting only, a below-threshold score suppresses class 1 when
the condition-specific keeper predicts class 1 and replaces it with that
keeper row's highest-probability non-class-1 label. It never changes a keeper
prediction that is already non-class-1. TP retention, FN acceptance, and
restricted-FP rejection gates are computed directly from the fixed cohort's
scores, independent of whether a lighting shift already changed the keeper
argmax. The replacement path is a diagnostic used to count corrections and
harms, not an authorized production router.

## Automatic Gates

All structural gates must pass:

- exact provenance, package version, wheel, command, cohort, fold, and path
  hashes;
- exact ROI/map/filtration/descriptor/role dimensions and finite values;
- no source overlap or source-self-assignment in any fold/placebo;
- exact rank-histogram preservation by the pixel placebo;
- synthetic H0/H1 oracle checks and deterministic two-pass diagram replay;
- all readouts converge and every outer held row appears once per condition;
- raw data, model state, validation, test, and current-best files remain
  untouched.

Clean candidate gates are conjunctive:

- positive-vs-restricted-FP ROC AUROC `>=0.68`;
- candidate-minus-control AUROC `>=0.03`;
- candidate-minus-pixel-placebo AUROC `>=0.03`;
- candidate-minus-source-placebo AUROC `>=0.03`;
- class-1 TP retention `>=0.98`;
- at least `9/11` keeper FN rows remain above the candidate threshold;
- restricted-FP rejection `>=0.15` and at least `28/186` removals;
- candidate removes at least nine more restricted FP than control;
- candidate restricted-FP removals minus TP breaks `>=20`;
- every fold removes at least two restricted FP and breaks at most three TP;
- candidate action corrections exceed harms.

Robustness gates are conjunctive:

- candidate AUROC `>=0.62` in each shifted condition and four-condition mean
  `>=0.66`;
- candidate-minus-control AUROC is nonnegative in every shift and mean
  `>=0.02`;
- candidate exceeds both placebos by mean AUROC `>=0.02`;
- shifted TP retention is `>=0.95` in every condition;
- at least `8/11` keeper FN rows stay above threshold in every shift;
- shifted restricted-FP rejection is `>=0.10` in every condition;
- shifted removals exceed TP breaks in every condition and by at least `30`
  in aggregate;
- per-row candidate score Spearman correlation with clean is `>=0.65` for each
  shift.

## Conditional Visual and Engineering Gates

Only a complete automatic pass may generate four contact sheets. Each page
must include retained TP, removed FP, broken TP, accepted FN, and missed FP
examples when available. Show RGB ROI, all three rank maps, representative
sublevel/superlevel threshold masks, H0/H1 barcodes, and candidate/control/
placebo scores.

Manual review passes only if beneficial topology lies on fruit-local
maturity/risk-surface patterns and not bbox corners, outside-ellipse fill,
glare, stem, hand, pad, text, background, or generic silhouette. Any failed
page closes A0.

Record CPU runtime, memory, diagram counts, and GUDHI version. Direct GUDHI
inference is not an ONNX/TensorRT path. A complete A0 pass authorizes only a
separate, default-off, standard-operator topological-surrogate protocol with
strict native/ONNX/TensorRT parity. It does not authorize a CPU side-channel in
the production video path.

## Replay and Artifacts

The sole formal may write only:

- `summary.json`;
- all-condition OOF score/action CSV;
- compressed descriptor/readout payloads;
- conditional contact sheets and review manifest if authorized;
- launcher transcript and a nonbinary SHA-256 artifact manifest.

Independent replay must reconstruct every count, metric, threshold, action,
gate, source binding, and payload hash without reading image pixels or fitting
a new hyperparameter. No validation/test artifact, checkpoint, model binary,
ONNX file, TensorRT engine, or raw-data derivative is allowed.

## Stop and Promotion Rules

- Any structural, clean, robustness, visual, replay, integrity, or engineering
  failure rejects chromatic cubical persistence before trainer integration.
- On failure, do not sweep channel, polarity, homology dimension, persistence
  floor, rank rule, resolution, ROI, mask, erosion, fill, sigma, Betti grid,
  readout C, threshold, fold, seed, or fuse it with closed blob/color/texture
  suppressors.
- A complete A0 pass authorizes one separately locked exportable surrogate,
  focused tests, then one bounded no-test smoke/probe. It does not directly
  authorize a full train.
- A full train remains authorized only after the integrated candidate passes
  validation, robustness, XAI, resource, export, and independent-reload gates.
  Use measured Windows workers (`train/eval=4/2` unless re-benchmarking proves
  better), early stopping, and at most 30 epochs.
- Current-best commands/history change only after a true locked validation
  winner. Test remains final and untouched until promotion.
