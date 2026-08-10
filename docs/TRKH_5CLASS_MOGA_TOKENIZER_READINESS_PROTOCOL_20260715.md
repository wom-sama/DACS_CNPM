# TRKH 5-Class Moga Surface Tokenizer Protocol (2026-07-15)

## Research question

Can a no-pretrain MogaNet-XT stage-1-to-3 surface tokenizer learn
multi-scale mango damage and maturity evidence before the existing TRKH
Transformer, improving class-1 precision without discarding true class-1
examples?

Primary sources:

- *Efficient Multi-order Gated Aggregation Network*, ICLR 2024:
  https://openreview.net/forum?id=XhYWgjqCrV
- Paper HTML/PDF source:
  https://arxiv.org/abs/2211.03295
- Official PyTorch repository:
  https://github.com/Westlake-AI/MogaNet
- Reviewed repository commit:
  `c83e328b513289fddd2921e17a9143b0f966c0a1`.
- Reviewed `models/moganet.py` SHA-256:
  `1ac13dfb57db813ab310b581c434d3240d875f0997bf955550c92abf49c79797`.

The paper argues that ordinary ConvNets and ViTs can overemphasize extreme
local texture or global shape. MogaNet instead splits channels across
multi-order depthwise convolutions, gates the aggregated context, and
decomposes redundant spatial/channel components. The official ablation
attributes the gain to the complete combination of feature decomposition,
multi-order context, gating, and channel aggregation rather than a large
kernel alone.

## Distinction from closed local routes

This is not another CoAtNet stem transplant. The rejected
`coatnet_mbconv` route used only two early MBConv stages and then expected the
unchanged TRKH Transformer to recover CoAtNet's later representation. The
candidate here runs the complete first three MogaNet-XT stages, including all
`3 + 3 + 10` Moga blocks, until the exact `16x16` Transformer token grid. It
therefore tests MogaNet's core multi-order spatial and channel aggregation,
not only an embedding stem.

It is also distinct from closed single-scale local patch mixers, persistent
local-global coupling, frequency/high-pass experts, EdgeNeXt substitution,
LeFF, large-kernel context, part discovery, and VCA. No auxiliary head,
frequency transform, part loss, output rule, teacher, or new class loss is
introduced. The existing eight Transformer blocks remain the global-context
half of the hybrid.

## Locked inputs and constraints

- Data: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, immutable raw files.
- Source training protocol:
  `runs/full_v8_yolof_randominit_30e_20260714_105524/launcher_args.json`,
  SHA-256
  `a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6`.
- Candidate and control are initialized from scratch with seed `42`; no
  pretrained weights and no checkpoint resume.
- Image size `256`, final patch grid `16x16`, TRKH embedding `256`, depth `8`,
  heads `8`, and the existing branch/register tokens remain unchanged.
- Validation is opened only for the one matched smoke after Stage A. Test
  remains closed. Current-best commands cannot change at readiness or smoke.

## Locked candidate architecture

Add default-off `stem_architecture=moganet_xt_tokenizer`:

1. Official stacked first embedding: two `3x3`, stride-2 convolutions,
   `3 -> 16 -> 32`, with BN and GELU, producing `64x64` features.
2. Moga stage 1: width `32`, depth `3`, channel FFN ratio `8`.
3. Official `3x3`, stride-2 embedding to width `64`, then stage 2 depth `3`,
   FFN ratio `8`, producing `32x32` features.
4. Official `3x3`, stride-2 embedding to width `96`, then stage 3 depth `10`,
   FFN ratio `4`, producing `16x16` features.
5. Preserve the existing TRKH `PatchEmbedding`; because the stem stride is
   now `16`, it becomes a learned `1x1` projection from width `96` to `256`.
   All branch-token, position, pruning, Transformer, pooling, and classifier
   code remains unchanged. The optional existing CNN residual classifier
   consumes the actual stem width (`96` for Moga, `256` for control); this is
   a dimensional compatibility rule, not an added expert or auxiliary loss.

Every Moga block is locked to the official form:

- BN pre-norm, residual layer scales initialized to `1e-5`;
- feature-decomposition scale initialized to `1e-5`;
- channel split `1:3:4`;
- depthwise paths `5x5 d1`, `5x5 d2`, and `7x7 d3`;
- SiLU gated spatial aggregation;
- GELU channel aggregation with depthwise `3x3` and learned decomposition;
- dropout and internal drop path `0`; gating runs in the active tensor dtype;
- official convolution/normalization initialization adapted without importing
  external MogaNet runtime code.

No depth/width/kernel/dilation/split/activation/layer-scale/drop-path or
projection sweep is permitted in this method.

## Stage A: train-only functional and resource gate

Before validation or image-model smoke, require all checks:

1. Default `conv_pool` construction and checkpoint schema remain bit-identical.
2. Candidate maps `B x 3 x 256 x 256` to `B x 96 x 16 x 16`, yields exactly
   `256` patch tokens and finite five-class logits, and round-trips through a
   checkpoint without pretrained access.
3. All `16` Moga blocks and expected `3/3/10` stage depths are present. The
   full TRKH candidate has at most `10M` parameters and at most `3M` extra
   parameters over control.
4. FP32 and CUDA AMP forward/backward are finite. Every stage embedding,
   low/middle/high-order depthwise branch, gate, value/projection,
   spatial-decomposition scale, channel FFN, channel-decomposition scale, and
   final TRKH patch projection receives a finite nonzero gradient.
5. Per-stage activation variance is nonzero; all three order branches have
   nonzero output energy and are not numerically identical. A balanced
   one-example-per-class real train batch has nonzero input/logit sensitivity
   for every class.
6. Batch-32 AMP forward/backward peak allocation remains below `7.75 GiB` and
   runtime is at most `1.75x` the matched control. Validation/test loaders are
   not constructed; the audit records dataset/checkpoint/source hashes.

Passing Stage A authorizes exactly one matched smoke. Functional correctness
alone is not promotion evidence.

## Stage B: one matched precision smoke

Run scratch control `conv_pool` and candidate `moganet_xt_tokenizer` with the
same source arguments and these fixed overrides:

- seed `42`, batch `32`, gradient accumulation `2`;
- `120` train batches per epoch, `2` epochs, scheduler horizon `15`, warmup `1`;
- AdamW LR `5e-4`, minimum LR `1e-6`, weight decay `0.05`, patience `3`;
- existing balanced sampler, LDAM-focal, teacher-focus-binary, metric and
  pairwise-margin recipe;
- existing token pruning and bbox prior unchanged;
- attention-view crop/drop supervision disabled in both variants to isolate
  tokenizer representation and keep the bounded smoke efficient;
- full `2606` validation rows, final test skipped.

Candidate advances only if every gate passes:

- exact support and finite normalized probabilities;
- macro F1 no worse than control by more than `0.003`;
- class-1 F1 gain at least `0.010`;
- class-1 precision gain at least `0.025`;
- class-1 recall at least `control - 0.020`;
- total `0/2/4->1` false positives decrease by at least `5`;
- class-1 TP breaks are no more than FN rescues plus `3`;
- corrections exceed harms and new `3->2` harms are at most `5`;
- no nonfocus class F1 falls by more than `0.020`;
- runtime is at most `1.50x` control and peak VRAM remains below `7.75 GiB`.

After independent reload, always run confusion/transitions, calibration,
architecture trace, native attention, rollout, Grad-CAM, changed-case paired
XAI, and clean/dim/bright/low-contrast/center-occlusion robustness. A metric
pass advances only if at least three of five robustness conditions preserve or
improve macro F1, no condition creates a class-1 recall collapse greater than
`0.05`, Moga stage Grad-CAM remains fruit/surface-focused, and attribution
provenance contains no silent fallback.

## Advancement boundary

A complete Stage-B pass authorizes one fixed five-epoch continuation. That
continuation must clear macro/class-1 F1 `0.887/0.70`, class-1 precision `0.65`,
recall `0.72`, the same TP/FP/transition gates, and at least four favorable
source bootstrap folds before any full train is considered.

Failure at any stage closes this exact Moga tokenizer without depth, stage,
width, dilation, channel-split, normalization, LR, seed, loss, pruning,
augmentation, checkpoint, or run-length sweeps. No smoke, five-epoch result,
oracle, ensemble, or test-informed result may update the current-best command
file.

## Execution outcome (2026-07-15)

Stage A is closed as a resource-gate rejection. The provenance-correct rerun
is `runs/audit_moga_tokenizer_stage_a_20260715_v2/summary.json`, SHA-256
`98c27d5ef784aa9e4ca590fa615894413b78cc62d48e22aa04141e886fe4f406`.
It loaded only all `9215` train objects; validation and test were not loaded.

- Default control schema/state stayed bit-identical. Candidate initialization
  and strict state round-trip were deterministic.
- The candidate had `8,503,022` parameters, adding `1,257,432`; all `16`
  Moga blocks, the `3/3/10` depths, `16x16` grid, and `256` patch tokens were
  present.
- Every declared FP32 and BF16 gradient family was finite and nonzero. Stage
  activations and all three order branches were finite, nonzero, and distinct.
  Balanced real-train input/logit sensitivity passed for all five classes.
- Batch-32 BF16 peak allocation was safe at `4.2665 GiB`, but median matched
  forward/backward time was `0.298064 s` versus control `0.149159 s`, a
  `1.998304x` ratio above the locked `1.75x` gate.
- A metric-free `channels_last` implementation probe was slower
  (`0.335850 s` versus `0.300960 s`) and was not integrated.

The first audit also exposed a pre-existing baseline-wide XAI defect: Lab
chroma used `sqrt(a^2+b^2)` at exactly zero, producing NaN input gradients in
both control and candidate. The implementation now clamps the radicand to
`1e-8` and has a zero-chroma regression test. This repair made the five-class
sensitivity probe valid; it does not change the Moga runtime rejection.

No matched smoke, validation prediction, five-epoch continuation, test,
architecture sweep, or current-best command update is authorized.
