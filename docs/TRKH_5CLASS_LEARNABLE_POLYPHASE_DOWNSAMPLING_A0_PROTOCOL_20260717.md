# TRKH 5-Class Learnable Polyphase Downsampling A0 Protocol - 2026-07-17

## Decision scope

This protocol locks one train-only, no-epoch gate before production model,
trainer, configuration, or launcher integration. It asks two questions:

1. Does the current keeper have material class-1 decision instability under
   deployment-plausible one-pixel translations?
2. Can a source-grounded Learnable Polyphase Downsampling (LPD) replacement
   for the three legacy max-pool operations satisfy equation, gradient,
   precision, export, and resource gates without changing raw data?

Only a complete A0 pass may authorize one separately locked matched scratch
pair. A0 cannot authorize validation, test, a full train, a current-best
command update, or any raw dataset change.

## Primary authority and provenance

- Accepted paper: Rojas-Gomez et al., *Learnable Polyphase Sampling for
  Shift Invariant and Equivariant Convolutional Networks*, NeurIPS 2022:
  `https://proceedings.neurips.cc/paper_files/paper/2022/hash/e87b1e06be8c3594c810e8991e77ea40-Abstract-Conference.html`.
- Author project page:
  `https://raymond-yeh.com/learnable_polyphase_sampling/`.
- Official repository:
  `https://github.com/raymondyeh07/learnable_polyphase_sampling`.
- Local official source:
  `D:\DataAI\external_sources\official\learnable_polyphase_sampling`.
- Locked commit/tree:
  `ef28ff29aa058c5f3fdf05b9a096f196ee1d0f30` /
  `74cda577ec36cc9288a0f2cded289b9b3b3b2269`.
- Official source SHA-256 values:
  - `learn_poly_sampling/layers/lps_utils.py`:
    `f8e28327ad0df66afcf9e5166624c06d59ef8c0555c3f16c8d0564cf05bd3f5e`;
  - `learn_poly_sampling/layers/lps_logit_layers.py`:
    `003a20c75ccbaa4c704f9ec0f63a2dc7a520f458650f27af7ab339fc91cc354a`;
  - `learn_poly_sampling/layers/polydown.py`:
    `21e32cac7226b4bd5bdaf33dd45d916e8415a6a3b66ba4193b373b55e1b0b898`;
  - `learn_poly_sampling/configs/logits_channels/resnet18_imagenet.json`:
    `426163ab52ea97c2dc10dd6ca3334f9634d9b0b9d3bb3666a36251b731790113`;
  - MIT `LICENSE`:
    `f5a41d36e1883bc5f0eb5eb520dd2d11c4b3cc225a60c9dfb2bf25ffafa31842`.
- Local accepted-paper PDF:
  `D:\DataAI\external_sources\papers\RojasGomez_Learnable_Polyphase_Sampling_NeurIPS2022.pdf`.
- Paper SHA-256:
  `cd13170aa45bc0b2f7afdd62e723bb2dc1674345012f26d45994137c4e3689dc`.
- GitHub observation on 2026-07-17: 13 stars, 1 fork, MIT, not archived.
  Popularity is disclosed but is not efficacy evidence; the accepted paper,
  equations, licensed source, and direct TRKH measurements control decisions.
- No LPS, ImageNet, ResNet, or other pretrained weight is permitted.

The paper reports that replacing a trained LPD with Adaptive Polyphase
Sampling (APS) can reduce ImageNet top-1 from `78.8%` to `0.1%`. Therefore an
inference-only pool substitution is scientifically invalid. A future efficacy
pair must co-train LPD from the common scratch initialization.

## Locked interpretation and adaptation

For each stride-2 downsampling site, LPD splits an even feature map into four
polyphase components. One shared selector processes every component with two
`3x3` convolutions and ReLU, averages over channel and space, and emits one
logit per phase. Training uses a Gumbel-Softmax convex combination; evaluation
uses the hard argmax phase. Shared selector weights and circular selector
padding are required for phase-logit permutation equivariance.

The phase order is locked to the official current V2 implementation:
`[(even row,even col), (even row,odd col), (odd row,even col),
(odd row,odd col)]`, abbreviated `[00,01,10,11]`.

The current TRKH stem uses three `MaxPool2d(2,2)` layers after feature maps of
32, 64, and 256 channels. The A0 candidate converts each legacy pool to:

1. right/bottom zero pad by one pixel;
2. dense `MaxPool2d(kernel_size=2, stride=1)`;
3. four stride-2 polyphase components;
4. one LPD selector and selected phase.

For an even input, fixed phase 0 is bit-identical to the legacy max pool. This
is a mandatory matched-control invariant. The three selector hidden widths are
locked to `[8,16,32]`, matching the official stage-wise reduced-width config.
Selector convolutions have bias, ReLU only between convolutions, circular
padding, no normalization, no skip connection, and no antialias filter.

The local implementation may replace the official advanced-index hard select
with an exactly equivalent one-hot multiply-and-sum for ONNX/TensorRT support.
It may not change phase order, logits, averaging axes, training relaxation, or
hard evaluation semantics. A0 uses `tau=1.0`; future scheduling is not opened
by this protocol.

## No-repeat screen

This route is equation-distinct from closed TRKH families:

- `SoftPool2d` and `max_soft` combine values inside each fixed `2x2` window;
  LPD chooses one complete sampling lattice using a learned, shared selector.
- Shifted Patch Tokenization constructs directional residual channels after
  fixed sampling; LPD changes the sampling phase itself.
- OctConv, FFT/DCT/Gabor/wavelet, PDC, frequency-selective pooling, GRN,
  large-kernel, XCA, deformable attention, token pruning, Cropr, and local-
  attention routes remain closed at their recorded settings.
- APS/max-norm phase selection, BlurPool, antialias filter size, selector
  width, selector family, padding, layer subset, temperature, hard/soft mode,
  shift size, shift fill, crop, seed, fold, and threshold sweeps are forbidden.

The Adobe BlurPool reference is CC BY-NC-SA and the inspected APS repository
has no reusable license declaration. No code from either source is copied.

## Locked data and class-1 cohort

- Dataset declaration:
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`.
- Dataset declaration SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Split access: `train=9215` only. Validation and test are forbidden.
- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher declaration SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- Cohort declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`.
- CIDT train-only summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.
- CIDT prediction SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Source-disjoint folds: `1,2,3,4`; fold 0 is not used by A0.
- Positive rows: clean keeper target and prediction are class 1.
- Negative rows: clean keeper predicts class 1 and target is in `{0,2,4}`.
- Locked counts: `607 = 421 TP + 186 restricted FP`.
- Locked fold counts:
  - fold 1: `112 TP / 45 FP`;
  - fold 2: `100 TP / 48 FP`;
  - fold 3: `101 TP / 52 FP`;
  - fold 4: `108 TP / 41 FP`.
- Ordered cohort-index SHA-256:
  `a2689d1be8579386eea9ef02a826822a5e7d78e2d29439e8947c1731ccc738bd`.
- Locked transformed `crop_bbox` byte SHA-256:
  `e9b2143c9dbc7c6f5bf3a38f80a43483bd436e3b5ce99116c3d5989f79892f6b`.

The normal deterministic `256x256` pad-resize transform, input mean/std,
image-valid mask, bbox metadata, standard classification forward, batch 64,
and row order are unchanged. Clean replay must preserve the previously locked
sample-3657 near-tie as the only full-train declaration exception. Any other
clean mismatch closes A0 before shift scoring.

Sample 3657 has target 2 and the old declaration predicts class 1, while the
locked current forward predicts class 2 at a near-tied class-1/class-2 margin.
It remains in all 607 probability-span and numeric-replay calculations but is
excluded prospectively from clean-class-1 exit counts. Therefore decision-exit
denominators are `606 = 421 TP + 185 restricted FP`; no other row may be
excluded or relabeled.

## Locked one-pixel translations

Eight translations are evaluated in this exact order:

`(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)`.

`dx>0` moves image content right; `dy>0` moves it down. Translation occurs on
the already normalized `256x256` tensor. Pad one pixel on every side with
replication, crop back to `256x256`, and never interpolate. Translate the
boolean image-valid mask with false fill. Add `(dx/256,dy/256)` to both bbox
centers passed to the model and the transformed crop bbox, then clamp centers
to `[0,1]`; width and height remain unchanged. Record exact contiguous bytes
and hashes for every condition. No image, label, bbox, or split file is
written.

These are standard finite-boundary translations, not the paper's circular
theoretical setting. Circular rolls are used only by isolated equation tests
and cannot be counted as data evidence.

For each cohort row, record clean and shifted probabilities, argmax, class-1
probability span, target-probability span, margin span, phase-sensitive stem
diagnostics, and whether any shift exits class 1. Report TP exits and
restricted-FP exits separately by fold and direction.

## Shift-signal gates

All gates are conjunctive. The current keeper must show:

1. exact clean declaration replay and exact cohort/bbox hashes;
2. at least `18/606` replay-stable rows whose class-1 clean decision changes under at least
   one shift;
3. at least four TP exits and at least four restricted-FP exits;
4. at least three of four folds with two or more class-1 exit events;
5. median class-1 probability span `>=0.010` and 90th percentile span
   `>=0.030` across the eight shifts plus clean;
6. at least two cardinal and two diagonal directions each produce a class-1
   exit event;
7. all logits/probabilities are finite and independent CSV replay reproduces
   all row, transition, quantile, fold, and direction values within `1e-12`.

These gates establish only that sampling-phase instability is material. They
do not establish that LPD improves classification.

## Isolated equation and gradient gates

Before any keeper shift inference, the isolated candidate must pass:

- exact paper/source/license/commit/tree/protocol/input hashes and a clean
  official-source worktree;
- phase split and phase order equality to both the official source and an
  independent slice oracle within `1e-12` in float64;
- official selector-logit and downsample output error `<=1e-6` after exact
  state transfer;
- selector-logit permutation error `<=1e-6` for circular one-pixel rolls;
- hard-selected global-mean invariance error `<=1e-6` for circular rolls;
- fixed phase-0 output bit-identical to legacy max pool for all three locked
  even feature-map shapes;
- finite analytic input and selector gradients, official-gradient error
  `<=1e-7`, and central finite-difference error `<=1e-4`;
- one Gumbel-Softmax training step at `tau=1.0` moves both selector
  convolutions with finite nonzero gradients;
- FP32/BF16 max output error `<=0.02`, finite BF16 output/gradients, and
  unchanged phase tensor shapes;
- deterministic eval logits/phases across three identical forwards;
- nonconstant selector logits on synthetic and real stem features. No phase-
  balance efficacy claim or phase-frequency gate is imposed before training.

The official source is loaded read-only from the locked clone. Local source
must contain an MIT attribution comment. No official file may be edited.

## Deployment and resource gates

The standalone three-stage stem replacement must also pass:

- standard ONNX ops only, no custom domain, ONNX Runtime max error `<=1e-5`,
  finite output, and matching selected phases;
- TensorRT parser/build success for fixed batch 32 and FP32;
- candidate parameter increase `<=2.5%` over the exact keeper architecture;
- full random-initialized TRKH FP32 batch-32 forward runtime ratio `<=1.15`
  and peak allocated-memory ratio `<=1.10` after warm-up and seven repeats;
- output shape and all post-stem token shapes identical to the control;
- no parameter-name collision, no custom CUDA kernel, and no dependency on the
  official source tree at runtime;
- no timing decision while an unrelated Python/TensorRT process is active.
  Record process ancestry and `nvidia-smi`; never terminate an unknown process.

The resource candidate is an isolated deep copy or wrapper around a random-
initialized exact keeper architecture. It is not saved and cannot be scored
for classification efficacy. A runtime/resource failure closes A0 even if the
shift signal is strong.

## Diagnostic visuals

After numeric scoring, render deterministic contact sheets for the eight TP
and eight restricted-FP rows with the largest class-1 probability span, plus
all decision-changing rows up to 32 per category. Each panel contains clean,
the most adverse shift, bbox overlays, target, all argmax values, class-1
probabilities, margins, fold, and shift direction. Visuals diagnose border or
metadata artifacts but cannot override a failed numeric gate. Missing,
misaligned, or unreadable panels fail formal completeness.

## Stop and promotion rules

- Any provenance, source, split, clean replay, equation, gradient, precision,
  shift-signal, export, TensorRT, runtime, memory, replay, or visual
  completeness failure closes A0.
- On failure: do not integrate LPD into production model/trainer/config code;
  do not run an image epoch, validation, test, smoke, probe, or full train; do
  not sweep any nearby LPD or shift setting; and do not update current-best
  commands/history.
- On full pass: write and hash a separate matched scratch train-only protocol
  before production integration. It must compare legacy max pool against the
  exact locked three-stage LPD under common initialization, source-disjoint
  fit/holdout groups, equal update count, replayed augmentations, raw argmax,
  clean plus illumination and one-pixel shifts, class-1 TP/FP transitions,
  phase diagnostics, XAI, export, and resource gates. At most one short pair
  is authorized.
- A later candidate can update the current-best command/history only after an
  independently locked validation win and all promotion audits. A0 can never
  update them.

Current-best command/history lock-time SHA-256 values remain
`36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
`39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
