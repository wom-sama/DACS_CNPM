# TRKH 5-Class Validity Partial-Conv Stem Protocol

Date: 2026-07-20

Status: prospectively locked before implementation, candidate training, or
candidate-output inspection. This protocol authorizes one train-only Stage-A
engineering/readiness audit and, only after every Stage-A gate passes, one
matched two-epoch/120-batch full-validation smoke. It does not authorize test
access, a hyperparameter sweep, a probe, a full train, or current-best command
promotion.

## Research Question

Can the exact `image_valid_mask` prevent synthetic square-padding bands from
attenuating and distorting the three-block CNN stem, while preserving real
wide context and class-1 surface evidence? The required outcome is higher
class-1 precision and fewer restricted `{0,2,4}->1` false positives without a
material class-1 recall or macro-F1 loss.

This is distinct from the rejected validity-aware attention A0. That frozen
intervention removed invalid Transformer keys after the CNN stem had already
formed its feature map and produced nonselective attention renormalization.
This route changes feature formation before tokenization and must be trained
with its own equation.

## Primary-Source Lock

### Partial-convolution padding

Liu et al., *Partial Convolution based Padding*, arXiv:1811.11718, treat padded
support as holes and renormalize convolution by the available valid fraction.
They report scratch-trained classification and segmentation comparisons and
faster convergence than zero padding. They also report a critical cross-test:
ResNet-50 trained with zero padding and switched to partial convolution only at
inference loses `16.357` top-1 points on average. Therefore a frozen-keeper
metric intervention is explicitly forbidden here; the candidate must receive
matched training adaptation.

- Paper: `https://arxiv.org/abs/1811.11718`
- Frozen PDF:
  `D:/DataAI/external_sources/papers/partial_convolution_padding_arxiv1811.11718.pdf`
- PDF SHA-256:
  `6b6d1ceee5bf69e7b0a6b93ef26b6681c8d479601ba70e16c9968bbf82ccc93f`

### Irregular-mask partial convolution

Liu et al., *Image Inpainting for Irregular Holes Using Partial Convolutions*,
ECCV 2018, define masked, renormalized convolution and automatic mask update.
This supplies the equation for TRKH's internal square-padding bands; it does
not supply a mango classifier, pretrained weights, or evidence that class 1
will improve.

- Accepted-paper page:
  `https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.php`
- Frozen PDF:
  `D:/DataAI/external_sources/papers/partial_convolution_inpainting_eccv2018.pdf`
- PDF SHA-256:
  `9d902b164c4563c97cf9ca9f0a7d5508bb292ae4204ce4173cf3f8056009c73b`

### Official implementation

- Repository: `https://github.com/NVIDIA/partialconv` (`1.3k` stars when
  checked; popularity is provenance context, not performance evidence).
- Locked commit/tree:
  `610d373f35257887d45adae84c86d0ce7ad808ec` /
  `39e37de8bdce67efd5ceee3008fe0c94bdd9cd5f`.
- Frozen clean checkout:
  `D:/DataAI/external_sources/official/NVIDIA_partialconv`.
- Reviewed source SHA-256:
  `models/partialconv2d.py=ca92d642523b7b18e56c981d6198e29e6b7cdbfa7469b3a07f5f93e741435116`.
- README SHA-256:
  `7578fea2adcbaab9ac35ad4f1f19b16296a40d868d60243523c262d5ad077dd0`.
- BSD-3-Clause license SHA-256:
  `8dc73b75a37967ada9d2a4b7643b155fad5b17e409319f3aa7c5d6092bf0d3d0`.

No external source code or weights enter TRKH. The local implementation must be
an independent, tested expression of the published equation. The official
checkout must remain clean.

## Immutable TRKH Inputs

- Pre-protocol HEAD/upstream:
  `87b90ef0135b0b446ec8c3c9fdfca4287d33e9d3`.
- Dataset: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher arguments SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- Pre-implementation source SHA-256:
  - `trkh/models/model.py=91468f0417954b98e2db6f0bec6d966d70695f5680b8bb9d495bfb22a8289ec7`;
  - `trkh/core/config.py=f383d733c1da5a59e780a6052c76a5fe0fe0ce0f711b49a47e822c793374b0cb`;
  - `trkh/training/train.py=d01fee45c60dd7bf4ef9ec36e5cb68ddab2e3a88b428d045aed9476fa4306585`;
  - `scripts/run_trkh_5class_attention_views_v8.ps1=61f785a954fa21d242003debfb6ea34ef027b09ec9195e564df72aa4aa1e0e79`.
- Current-best command/history SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

Raw images, labels, YAMLs, and split membership are immutable. Stage A may use
only `yolo_f/train`; Stage B may train on train and evaluate the existing full
validation split. Test paths, pixels, labels, predictions, and metrics remain
forbidden.

## Locked Candidate Equation

Add model option `stem_convolution` with exactly two values:

- `standard` (default): existing `Conv-BN-GELU-MaxPool` blocks;
- `validity_partial`: all three 3x3 convolutions consume and update the exact
  single-channel image-valid mask.

The option is valid only for `stem_architecture=conv_pool`. It adds no trainable
parameter and keeps the same convolution weight keys and tensor shapes, so the
keeper loads strictly into both roles.

For a block input `X`, binary validity mask `M`, 3x3 bias-free convolution
`C`, and epsilon `1e-6`, compute:

```text
M_border = pad(M, one pixel on every tensor edge, value=1)
S        = conv2d(M_border, ones[1,1,3,3], padding=0)
U        = (S > 0)
R        = 9 / (S + 1e-6) * U
Y        = C(X * M) * R
Z        = GELU(BN(Y))
P        = max_pool2d(where(U, Z, dtype_min), kernel=2, stride=2)
M_next   = max_pool2d(float(U), kernel=2, stride=2) > 0
X_next   = where(M_next, P, 0)
```

The outside tensor border is deliberately padded as valid in `M_border`. This
TRKH boundary condition preserves the keeper's existing zero-padding equation
when `M` is all valid and isolates only synthetic padding *inside* the 256x256
canvas. It is not a claim that this boundary choice appears in the paper.

Mask update `U` follows the accepted irregular-hole mechanism. Masked max pool
uses `dtype_min` so an invalid zero cannot beat a negative valid GELU response.
For FP16/BF16 stability use the official README's recommended `1e-6` epsilon.
No mask threshold, soft mask, dilation, erosion, learned gate, layer subset,
pool variant, ratio, or padding-fill variant is permitted.

If `image_valid_mask` is omitted, the model uses an all-valid mask. With an
all-valid mask, candidate and standard stem outputs must be numerically
identical. The normal train/eval/export paths must pass the actual mask.

## Stage-A Readiness Audit

Stage A may open one deterministic five-class train cohort and one resource
batch. It must persist source/config/checkpoint hashes, selected rows, state
schemas, numerical oracle results, per-block mask counts, gradients, runtime,
peak memory, ONNX preflight, a fixed one-row-per-class stem mechanism sheet,
summary, replay data, and artifact manifest.

Every gate is conjunctive:

1. All frozen hashes, source commit/tree/license, and official checkout status
   are exact and clean; current-best files are unchanged.
2. Explicit `standard` is bit-identical to the omitted default in state and
   logits. Control and candidate have identical state keys, values, trainable
   parameter count, and all configuration except `stem_convolution`.
3. Exactly three keeper stem convolutions use the candidate equation; no
   Transformer, patch, head, loss, augmentation, or evaluator module changes.
4. A padding-free random oracle matches the reviewed NVIDIA equation within
   `1e-6`; the TRKH all-valid boundary oracle matches standard convolution,
   complete stem output, logits, and gradients within `1e-6`.
5. Changing every masked input pixel to two distinct finite fills changes
   candidate stem/logits by at most `1e-6`, while the matched standard stem
   changes by more than `1e-6` on the same nondegenerate oracle.
6. Per-block updated-mask counts are monotonic under the locked update equation,
   finite, nonempty, and geometrically aligned. The five-class sheet confirms
   input padding, masks, standard/partial activation energy, and absolute
   differences without changing any gate.
7. Strict keeper load succeeds with no missing/unexpected keys; forward/backward
   gradients are finite and nonzero through all three convolution weights and
   the patch projection.
8. FP32 and CUDA AMP outputs are finite, candidate runtime is at most `1.35x`
   control, peak CUDA allocation is at most `7.5 GiB`, and the normal dynamic-
   batch ONNX/export preflight includes `image_valid_mask` without unsupported
   custom operators.
9. Validation/test remain unopened, no checkpoint is written, raw data are
   unchanged, replay is exact, pycompile/pyflakes/focused/full tests pass, the
   launcher parses in PowerShell, and the implementation is committed/pushed
   before Stage B.

Any failed Stage-A gate closes this exact route before training.

## Matched Adaptation Smoke

Only a complete Stage-A pass authorizes one control/candidate pair. Both roles
strictly resume the current keeper SHA above, reset epoch/optimizer/scheduler/
scaler, use CLI configuration, and share the keeper's v8 data, seed 42, batch
order, augmentation, loss, optimizer, batch size 32, gradient accumulation 2,
learning rate `8e-5`, workers `4/2`, two epochs, 120 train batches per epoch,
full 2,606-row validation, architecture trace, and no final test. Use scheduler
horizon 10 so the two-epoch diagnostic does not collapse LR prematurely.

The sole model difference is:

```text
control:   stem_convolution=standard
candidate: stem_convolution=validity_partial
```

The pair audit must independently reload both selected checkpoints, verify
exact row/source alignment and provenance, compute confusion/calibration/
transition/source-group metrics, and inspect all five architecture traces.
It must also compare the trained candidate under its aligned mask and a fixed
same-class validation mask-shape derangement. The derangement preserves class
and invalid-pixel-count rank bins while changing source and spatial alignment;
it is diagnostic only and cannot be selected or tuned.

Clean precision-first gates, all required:

- validation support is exactly 2,606 for both roles and no test output exists;
- candidate-minus-control macro F1 `>=-0.002`;
- class-1 F1 `>=+0.008` and precision `>=+0.015`;
- class-1 recall `>=-0.012`;
- at least five net restricted `{0,2,4}->1` FP removals;
- corrections are not fewer than harms;
- class-1 TP breaks do not exceed FN rescues plus two and are at most five;
- worst non-class1 F1 delta `>=-0.012`;
- class-1 precision is non-worse in at least four source groups with adequate
  support;
- aligned candidate exceeds its mask-shape placebo by at least `+0.005`
  class-1 F1, `+0.008` precision, and three net restricted-FP removals;
- independent reload, config-difference, scheduler, trace, artifact-hash,
  runtime `<=1.35x`, and data-isolation gates all pass.

After any smoke, inspect the complete clean audit and all traces before making
a decision. If and only if every clean gate passes, run the prospectively fixed
dim/bright/low-contrast replay, padding-fill invariance audit, changed-case
Grad-CAM/attention XAI, background sensitivity, and calibration audit. Required
post-smoke gates are nonnegative class-1 precision under all shifts, worst
class-1 recall delta `>=-0.02`, worst macro-F1 delta `>=-0.005`, aggregate net
restricted-FP removal at least 12, foreground attribution non-worse, and exact
padding-fill invariance for the candidate.

## Escalation And Stop Rules

A complete smoke plus robustness/XAI pass authorizes one five-epoch probe. Only
a probe that beats the locked validation promotion gate, converges cleanly,
passes independent reload, and remains precision/recall/robustness/XAI safe may
authorize one autonomous full train. That full train uses at most 30 epochs,
early-stopping patience 3, and measured stable workers `4/2` unless a fresh
benchmark proves another setting faster. Test remains isolated until the
existing promotion gate grants it.

Any material failure closes `validity_partial` and nearby epsilon/mask-update/
pool/layer/threshold/fill/normalization/fusion/LR/seed variants on this keeper.
Do not rescue it with a threshold, router, post-hoc blend, class-specific mask,
or data edit. Current-best command/history files change only after a locked
validation winner.
