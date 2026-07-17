# TRKH 5-Class EfficientTrain Low-Frequency Curriculum A0 Protocol

Date: 2026-07-17

Status: prospectively locked before candidate inference, trainer integration,
or an image epoch.

## Question

Can the official EfficientTrain low-frequency crop preserve true class-1
support while exposing a direction that suppresses restricted class-1 false
positives under clean, dim, bright, and low-contrast conditions?

Class 1 is the dataset grade `Xoai_Song_ChuaNhe_CoNguyCo`, not a pure lesion
class. In this protocol, "surface signal" means the joint maturity, color, and
risk-surface evidence that distinguishes this grade from classes 0, 2, and 4.
A generic foreground, edge, or lesion response is insufficient.

## Primary Authority

- Accepted paper: Wang et al., *EfficientTrain: Exploring Generalized
  Curriculum Learning for Training Visual Backbones*, ICCV 2023.
- Accepted paper SHA256:
  `3191ac4650bfea859492360ca916c83b487906e7c5eae98af803d83bdb251eae`.
- Supplemental SHA256:
  `aa542b054afdd2cfdc53fb4bb0cb8f7df932c37a126400d822939aba0b79c92c`.
- Paper-linked MIT repository:
  `https://github.com/LeapLabTHU/EfficientTrain`.
- Locked ICCV-2023 implementation commit/tree:
  `bdefd277c71ba3bfc2a88c13768b215f15301350` /
  `5c1362c15be3695eba64d636a8019944c026ac61`.
- Locked source SHA256 values:
  - `utils.py`: `64f42a102eee738f45383dbdc9be36fdb0fc960810caebf9b5b3c9ae0c42a21a`
  - `engine.py`: `2aab50e607f519aa59892acd1086fb6d33594e854f9798d4fd33be2cc6c55058`
  - `ET_training.py`: `75c0eb344d577814b06e2539df8b77d441f3111103a2d1249d19d32f1682ba7c`
  - `README.md`: `b0a5cc82e7d9e28afd0172d9e93948400def13491d3f2b4341f7bddb459d0861`
  - `LICENSE`: `63e8210e6bf3e8c032dc0c69b1d1d2e3ab72c14b02cabcc0dada2618bb188b97`

The paper equation, its supplemental proof, and the locked source are the
authority. No pretrained weight or ImageNet data is used.

## Prior Screen

Curriculum By Smoothing (NeurIPS 2020) is rejected before implementation for
this route. It smooths every CNN feature map to reduce random-initialization
noise, does not provide a class-1-specific information signal, and its official
README requires tuning `std`, `std_factor`, and `epoch`. Its 200-epoch source
schedule cannot be transferred faithfully to the locked 30-epoch TRKH ceiling
without an arbitrary sweep. Do not reopen CBS sigma, decay, insertion, or
schedule variants on this keeper.

EfficientTrain is equation-distinct from the closed FcaNet feature-DCT gate,
same-class Fourier-amplitude mixing, fixed wavelet/scattering/Gabor routes,
LPD/APS/BlurPool downsampling, and post-hoc TTA. It changes only the training
view and leaves the deployed model unchanged.

## Locked Inputs

- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`
  at SHA256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher args SHA256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- Dataset declaration:
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml` at SHA256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- CIDT train-only summary/predictions SHA256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad` /
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Current command/history SHA256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

Use only the established source-fold cohort from `yolo_f/train`:

- 607 rows total;
- 421 keeper class-1 true positives;
- 186 keeper restricted false positives with targets in `{0,2,4}`;
- folds `1..4` with counts `112/45`, `100/48`, `101/52`, `108/41`
  for TP/FP;
- ordered sample-index SHA256
  `a2689d1be8579386eea9ef02a826822a5e7d78e2d29439e8947c1731ccc738bd`.

Validation, test, fold 0 holdout, manual labels, and raw-data changes are
forbidden. Formal execution requires a clean tracked TRKH tree at the exact
pushed branch commit and a clean locked official-source worktree.

## Locked Transform

Apply the exact official `freq_crop` operation after the normal evaluation
transform and lighting shift:

1. Compute channel-wise `fft2` on a real `NCHW` FP32 tensor of edge length
   `H=256`.
2. Copy the four `B/2 x B/2` corner blocks into a `B x B` complex spectrum.
3. Multiply by `(B/H)^2`.
4. Compute `ifft2` and take the real component.
5. Detach the transformed input exactly as in official training code.

The three immutable views are:

- `early_b176`: `B=176`, obtained by nearest-patch scaling of
  `256 * 160 / 224`;
- `middle_b224`: `B=224`, obtained by nearest-patch scaling of
  `256 * 192 / 224`;
- `native_b256`: no crop and the exact keeper control.

The patch multiple is 16, so the resulting grids are `11x11`, `14x14`, and
`16x16`. No bandwidth, crop shape, interpolation, phase, padding, color space,
normalization, or schedule sweep is allowed.

The four immutable conditions are:

- `clean`: brightness `1.00`, contrast `1.00`;
- `lighting_dim`: brightness `0.70`, contrast `0.90`;
- `lighting_bright`: brightness `1.25`, contrast `1.10`;
- `low_contrast`: brightness `1.00`, contrast `0.65`.

## A0 Checks

### Source and Equation

- Exact paper/source/keeper/data/command hashes and source commit/tree.
- Official implementation versus an independent `fftshift -> center crop ->
  ifftshift -> ifft2` oracle, maximum error `<=1e-6` in FP32.
- Exact spectral-corner and scale-factor replay.
- Finite outputs and deterministic two-pass replay.
- `B=256` control must be byte-exact to the unmodified input and keeper logits.
- No parameter, buffer, RNG, or raw file may be mutated.

### Dynamic Geometry

- Standard keeper forward must accept all three grids without changing model
  state or parameter count.
- Positional interpolation, prefix count, pruning indices, and finite logits
  must pass for every view.
- The clean `native_b256` predictions must reproduce all 607 locked keeper
  declarations exactly before any candidate metric is read.

### Precision and Positive-Support Signal

For each cropped view define the locked suppression score

`s_B = p1(native_b256) - p1(B)`.

Higher values must indicate a restricted false positive more often than a true
class-1 positive. For each held fold, fit only one threshold from TP rows in
the other three folds: the empirical 97th percentile of clean `s_B`. Apply
that threshold unchanged to held-fold clean and all three shifted conditions.

`early_b176` passes only if every check below passes:

- clean FP-vs-TP suppression AUROC `>=0.65`;
- each shifted AUROC `>=0.60` and four-condition mean `>=0.65`;
- clean OOF TP retention `>=0.97` and restricted-FP rejection `>=0.10`;
- every shifted TP retention `>=0.93` and restricted-FP rejection `>=0.05`;
- at least three of four folds have positive clean restricted-FP rejection;
- clean OOF restricted-FP removals exceed TP breaks by at least five;
- `early_b176` clean AUROC and FP rejection are not below `middle_b224`;
- direct hard-decision corrections must exceed harms and cannot break more
  than two clean class-1 TP rows.

These thresholds are an information diagnostic only. They do not authorize a
post-hoc inference filter.

### Object Versus Background Mechanism

Construct a same-size `256x256` low-frequency reconstruction by zero-padding
the locked cropped spectrum, and define the removed high-frequency residual.
Use transformed `crop_bbox` geometry without changing it after inspection.

- Every row must have nonempty object and outside regions.
- FP-vs-TP AUROC for object residual energy must be `>=0.60`.
- Object residual AUROC must exceed outside residual AUROC by `>=0.03`.
- In at least three conditions, the FP-minus-TP object residual gap must be
  positive and exceed the corresponding outside gap.
- Manual review of every locked contact-sheet page must show that the ranked
  response is fruit-surface/maturity evidence, not leaves, bbox borders,
  padding, glare alone, or generic high-frequency clutter.

### Cost and Deployment

Benchmark official FP32 crop at batch 32 after five warmups and three repeats.
The transform must add no deployed parameters or inference operation. Record
training-view runtime and peak allocation, but do not relax an information
failure because the transform is cheap. Standard inference remains the exact
`native_b256` model and existing export commands remain unchanged.

### Replay and Artifacts

Write only a compact formal directory containing:

- `summary.json`;
- `predictions_all_conditions.csv`;
- `frequency_mechanism.csv`;
- `contact_sheet_*.png` plus review manifest;
- launcher transcript and `artifact_manifest.json`.

An independent replay must reconstruct row counts, suppression AUROC,
threshold decisions, TP/FP counts, hard transitions, and mechanism summaries
from CSV artifacts. No checkpoint, ONNX, TensorRT engine, cache, validation,
test, or raw-data artifact is permitted.

## Stop Rule

All gates are conjunctive. Any source, equation, geometry, information,
positive-support, mechanism, visual, replay, or integrity failure rejects
EfficientTrain A0 before trainer integration and before an image epoch.

A complete A0 pass authorizes only a new, separately hash-locked matched
scratch-control protocol. It does not authorize validation, test, a full train,
or a current-best command update.

On failure, do not sweep `B`, stage count, epoch fractions, augmentation
magnitude, FFT shape, filter geometry, crop order, normalization, seed, fold,
optimizer, learning rate, or combine this route with closed frequency,
pooling, TTA, or curriculum variants. Preserve the compact evidence, update
the research journal/TODO/skill, rerun retention, and keep current-best
commands unchanged.
