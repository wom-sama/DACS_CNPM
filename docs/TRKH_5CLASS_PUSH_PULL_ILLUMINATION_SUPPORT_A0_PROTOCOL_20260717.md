# TRKH 5-Class Push-Pull Illumination Support A0 Protocol - 2026-07-17

## Status

Prospectively locked before implementation or any push-pull measurement. This
document authorizes one fit-only frozen-signal gate. It does not authorize a
production model/trainer/config change, an image-training epoch, the 1,843-row
train holdout, official validation/test, a probe/full train, or a current-best
command revision.

## Research Question

Can the paper-defined first-layer push-pull inhibition operator preserve the
current class-1 positive support under dim, bright, and low-contrast conditions
while exposing a stronger object-level separation between current class-1 true
positives and restricted `{0,2,4} -> 1` false positives than the same native
TRKH first convolution?

This gate is deliberately narrower than a claim that inhibition generally
improves robustness. The previous spatial worst-shift route removed false
positives by contracting class-1 support and breaking true positives. A useful
new signal must reject false positives without repeating that failure.

## Primary-Source Lock

- Peer-reviewed journal article: Strisciuglio, Lopez-Antequera, and Petkov,
  "Enhanced robustness of convolutional networks with a push-pull inhibition
  layer," Neural Computing and Applications 32, 17957-17971 (2020).
- DOI/publisher page:
  `https://doi.org/10.1007/s00521-020-04751-8`
- Author-institution publication page:
  `https://research.utwente.nl/en/publications/enhanced-robustness-of-convolutional-networks-with-a-push-pull-in/`
- Accepted paper PDF:
  `https://research.utwente.nl/files/248043819/Strisciuglio2020enhanced.pdf`
- Locked local paper:
  `D:/DataAI/external_sources/official/push-pull-cnn-layer/papers/Strisciuglio2020_Article_EnhancedRobustnessOfConvolutio.pdf`
- Paper SHA-256:
  `04cf18f3857a7ddede02cc83851b5e99b0f25c5e9c4e36793bbe34bd6ebb1580`
- Official author repository:
  `https://github.com/nicstrisc/Push-Pull-CNN-layer`
- Locked commit/tree:
  `c340f329368e60c888ebdca8e30c373cdad9d29f` /
  `e9c984b6a71dd2f67aeb56388b9e8e84f1d5e5b7`
- Official `pushpull/PPmodule2d.py` SHA-256:
  `1de24434c557967f8c2a58a905e199f65757152ba23483481d60483000235b6a`
- Official README SHA-256:
  `f7e394b176f16fb7a759ad4d96f960b30919d9a1c0bfc70832623b61e93ea4b7`
- Official MIT license SHA-256:
  `0d4cced2ae017bd8484f69877b72d45114c2c4990689458dc21d1087e31c0564`
- Repository popularity when checked: 24 GitHub stars and 2 forks.

The low repository popularity is disclosed and is not treated as efficacy
evidence. Authority comes from the peer-reviewed article, exact equation,
author-maintained source, and MIT license. The article is CC BY 4.0.

The paper replaces only the first convolution, trains the modified networks
from scratch, and warns that replacing the first layer of an already trained
model requires fine-tuning because its response maps change. It fixes
upsampling factor `h=2` and inhibition strength `alpha=1` for the main
experiments. Learning `alpha` made optimization slower and more complicated.
Therefore this A0 neither learns nor sweeps either value.

The source evidence is not uniformly favorable for the exact TRKH problem.
The paper reports average corruption improvements, but its CIFAR-C table shows
that brightness and contrast errors improve for some architectures and worsen
for others. It also uses 200-350 training epochs for its deeper models. These
limitations are why illumination support, convergence, and resource behavior
must be measured prospectively rather than inferred from the paper.

## Source Equation And Locked Adaptation

For input `I`, learned push kernel `k`, half-wave rectifier `H=ReLU`, fixed
inhibition strength `alpha`, and bilinearly upsampled/inverted pull kernel, the
paper and official source define:

`P_h(I) = ReLU(conv(I, k)) - alpha * ReLU(conv(I, -up_h(k)))`.

For a push-kernel width `s`, the pull width is:

`s_pull = floor(s*h) + 1 - (floor(s*h) mod 2)`.

TRKH uses an RGB `3x3`, stride-1, padding-1, bias-free first convolution. The
locked candidate therefore uses a `7x7` pull kernel, padding 3,
`align_corners=True`, `h=2`, and `alpha=1`. Only the first convolution is
adapted; its existing BatchNorm, GELU, max pool, the other stem blocks,
Transformer, heads, and bbox logic remain unchanged.

The official implementation computes the push size from tensor dimension 1,
which is input channels rather than spatial width. This is a latent general
implementation defect, but both values are exactly 3 for the locked TRKH first
layer, so it still constructs the paper-required `7x7` pull. The local
implementation must derive size from the spatial dimensions and must match the
official code for this locked RGB `3x3` case.

Two roles are fixed before measurement:

1. `native_identity_h1`: use the same equation with `h=1`, `alpha=1`. Because
   `ReLU(z) - ReLU(-z) = z`, this must be numerically identical to the native
   convolution before and after the unchanged first block.
2. `push_pull_h2`: use the paper configuration `h=2`, `alpha=1` with the same
   learned keeper push weight.

`h=1` is a causal identity control, not a tuned candidate. No other scale,
inhibition strength, insertion depth, normalization, kernel rescaling, learned
alpha, dual output, or activation is permitted.

## Local No-Repeat Boundary

This route is equation-distinct from the closed families:

- PiDiNet PDC uses linear central/angular/radial difference conversions in all
  three stem blocks; it has no nonlinear opposite-polarity, broader-support
  inhibition.
- RSC suppresses target-gradient-selected pooled channels only during training;
  it leaves inference unchanged.
- Gabor/LBP/SRM/frequency routes add fixed texture or frequency measurements
  rather than a learned antiphase pull derived from the first kernel.
- CIConv computes a fixed photometric-invariant branch; CEConv uses color-group
  equivariance. Neither performs broad half-wave antiphase inhibition.
- MixStyle, SRM normalization, AugSelf, FACT, Retinex-like normalization,
  consistency, worst-augmentation, pooling, routing, loss-reweighting,
  threshold, and post-hoc routes are already closed and may not be combined
  with this A0.

ICML-2021 OOCS and NeurIPS-2024 YOLA were also checked as illumination-facing
alternatives. Their official repositories lacked a license at screening time,
and YOLA's zero-mean kernel construction overlaps the already rejected linear
difference family. No code or equation from either route is used here.

Failure closes the exact first-layer `h=2, alpha=1` push-pull family on the
current keeper, including nearby scale/strength/depth/normalization/activation
and learned-alpha rescues. It does not reject unrelated inhibition mechanisms
from a separately accepted and licensed source.

## Locked Inputs And Cohort

- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Keeper launcher arguments SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`
- Data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`
- Data YAML SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`
- Ordered train/fold declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`
- Declaration SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`
- Companion summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`

Use only clean declaration rows whose fold is `1,2,3,4`. Fold `0` is the
forbidden 1,843-row train holdout. The fit-only binary information cohort is:

- positive: target `1`, keeper prediction `1` (`421` rows);
- negative: target in `{0,2,4}`, keeper prediction `1` (`186` rows).

Per-fold positive/negative counts are locked to `112/45`, `100/48`, `101/52`,
and `108/41`. Ordered cohort index SHA-256 is
`a2689d1be8579386eea9ef02a826822a5e7d78e2d29439e8947c1731ccc738bd`.
No other train row may enter the readout, and no validation/test row may be
opened.

## Exact Frozen Signal And Descriptor

1. Load the keeper with deployment semantics and require the standard clean
   forward to reproduce all 607 locked declarations at batch 64.
2. Clone the first stem block and its convolution weight, BatchNorm affine and
   running state, GELU, and max pool. Freeze every parameter.
3. Run `native_identity_h1` and `push_pull_h2` through both the pre-BatchNorm
   operator output and complete first-block output. The normal keeper itself is
   never mutated.
4. Map the object bbox to each resolution. Define core by shrinking each bbox
   edge inward by `20%`; boundary is bbox minus core; outside is the complement
   of bbox. Every region must be nonempty for every row and stage before model
   inference begins.
5. For each stage and region record signed mean, mean absolute response, and RMS
   response. Add bounded energy contrasts
   `(boundary-core)/(boundary+core+eps)` and
   `(core-outside)/(core+outside+eps)` per stage.
6. Object-only features contain core/boundary statistics plus the first
   contrast from both stages (`14` values). Context-complete features add
   outside statistics and the second contrast (`22` values).
7. For mechanism audit only, record push, pull, and inhibited object/boundary/
   outside RMS, plus `pull_rms/(push_rms+pull_rms+eps)`. These values may not be
   added to or selected for the OOF readout.
8. Extract the same fixed features for clean `(1.00,1.00)`, dim
   `(0.70,0.90)`, bright `(1.25,1.10)`, and low contrast `(1.00,0.65)`
   brightness/contrast conditions. Cohort, folds, bboxes, model state, and order
   remain fixed.

The frozen `push_pull_h2` full-model logits are recorded only as a finite,
noncollapse diagnostic. They are not an efficacy gate because the paper states
that a replaced pretrained first layer needs fine-tuning. The `h=1` control and
the untouched keeper must remain declaration-identical.

## Fixed OOF Readout

Run four source-fold OOF splits. For each held-out fold:

- fit `StandardScaler` on the other three clean folds only;
- fit `LogisticRegression` with `solver=liblinear`, `C=0.1`, balanced class
  weights, intercept enabled, `max_iter=1000`, tolerance `1e-6`, seed `42`;
- score the held-out clean rows and the same rows under all three shifts with
  that clean-fitted scaler/model.

Run this recipe independently for both roles and for object-only/context-
complete features. No C/solver/feature/fold/threshold/seed sweep is allowed.

For each role, define its clean protection threshold as the lower empirical
`3%` quantile of clean OOF class-1-TP scores using NumPy method `lower`. Freeze
that role-specific threshold and report TP retention and restricted-FP
rejection under every condition. The threshold is diagnostic only and may not
become a runtime classifier or post-hoc prediction gate.

## Provenance, Equation, And Deployment Gates

All must pass before the information gate can authorize training:

1. Every source/input/command hash and official commit/tree matches; official
   and tracked TRKH worktrees are clean. Protected untracked reports are
   ignored.
2. Preflight creates no output and opens no dataset, holdout, validation, or
   test.
3. Local `h=2` output matches both an independent equation oracle and the
   official module on deterministic random input and real keeper weight with
   maximum error `<=1e-6`.
4. Local/operator-oracle FP32 output, input-gradient, and weight-gradient errors
   are `<=1e-6`; central finite-difference error is `<=1e-4`.
5. `h=1` pre-block and complete-block outputs match the native first block with
   maximum error `<=1e-6`; full keeper logits/predictions and all 607 clean
   declarations match exactly.
6. `h=2` is nondegenerate versus `h=1`, all outputs/gradients are finite, and
   BF16 output error versus its FP32 reference is `<=0.02`.
7. Ordered cohort hash, folds, counts, bboxes, masks, descriptors, OOF values,
   and frozen diagnostics match the lock and are finite. Every fit converges.
8. A static-batch-1 first-block candidate exports at ONNX opset 17 with only
   standard domains; ONNX Runtime maximum error is `<=1e-5`, shapes match, and
   TensorRT parses and builds an engine. ONNX/engine artifacts are ephemeral.
9. Candidate and control have identical trainable parameter counts. Using the
   normal metadata-aware evaluator path, BF16 autocast, five warmups, fixed
   batch 32, and three timed repeats, median complete-model CUDA runtime is
   `<=1.20x` native and peak allocated memory is `<=1.10x` native under an
   isolated GPU timing check.

## Information, Precision, And Support Gates

All gates use OOF predictions on the 607 fit-only rows. The object-only
`push_pull_h2` role must satisfy all of the following:

- clean TP-vs-restricted-FP AUROC `>=0.64`;
- clean AUROC improvement over `native_identity_h1` `>=0.025`;
- minimum shifted AUROC `>=0.58`;
- candidate AUROC is no worse than native in at least two of three shifts and
  its worst shifted delta is `>=-0.01`;
- clean TP retention `>=0.97` and clean restricted-FP rejection `>=0.20`;
- clean restricted-FP rejection improvement over native `>=0.08`;
- every shifted TP retention `>=0.93` and is no more than `0.01` below native;
- every shifted restricted-FP rejection `>=0.10`, and aggregate shifted
  candidate FP rejection exceeds native;
- median candidate TP score is strictly greater than FP score in every
  condition.

To prove the readout is not background-driven and the mechanism does not merely
contract all support:

- clean object-only candidate AUROC is no more than `0.02` below its context-
  complete AUROC;
- object-only candidate still beats object-only native by the locked clean
  margin;
- median candidate bbox response energy exceeds outside energy for both TP and
  FP cohorts on clean images and in at least two shifted conditions;
- median object-region pull-to-push inhibition ratio is at least `0.01` greater
  for restricted FP than TP on clean images and in at least two shifted
  conditions.

Render one fixed contact sheet per condition. Each sheet contains the four
highest and four lowest candidate OOF scores from each cohort. Every panel must
show source image, bbox/core overlay, push map, pull map, inhibited map, native
and candidate scores, fold, target, and keeper prediction. Manual review must
confirm alignment/readability and that any apparent FP suppression is not
consistently caused by outside/background response. Rendering is an audit, not
a selection input.

## Stop Rule And Next Boundary

Any failed provenance, equation, identity, mapping, deployment, resource,
information, precision, support, or visual gate rejects A0. Preserve compact
quantitative/contact-sheet evidence, independently replay the CSV and summary,
update journal/TODO/skill, verify retention, and close nearby scale/alpha/depth/
activation/normalization/kernel-rescaling/readout/threshold/fold/seed variants.
Do not run holdout, validation, test, an image epoch, or production XAI on a
rejected frozen signal.

Only a complete A0 pass authorizes a separately precommitted, default-off
first-layer implementation and one source-disjoint matched five-epoch scratch
pair. That pair must compare equal updates from random initialization, retain
the current full audit/XAI/deployment suite, and pass train-only gates before
validation. A full run remains capped at 30 epochs and requires the established
class-1 validation milestone; the source paper's longer schedule is not adopted.

Current-best commands remain locked at SHA-256
`36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`;
command history remains locked at SHA-256
`39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
