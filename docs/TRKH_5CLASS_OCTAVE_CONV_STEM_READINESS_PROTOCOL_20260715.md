# TRKH 5-Class Octave Convolution Stem Protocol (2026-07-15)

## Research question

Can a parameter-matched Octave Convolution stem preserve fine mango-surface
detail while separately processing slowly varying color/shape context, thereby
raising class-1 precision without discarding true class-1 samples?

The experiment changes only the three convolutions in the current TRKH CNN
stem. It keeps the immutable `yolo_f` split, input size, crop/bbox metadata,
downsample factor, patch grid, eight Transformer blocks, token pruning, head,
loss, sampler, augmentation, optimizer, and no-pretrain policy unchanged.

## Primary sources and source lock

- Chen et al., *Drop an Octave: Reducing Spatial Redundancy in Convolutional
  Neural Networks with Octave Convolution*, ICCV 2019:
  https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Drop_an_Octave_Reducing_Spatial_Redundancy_in_Convolutional_Neural_Networks_ICCV_2019_paper.html
- Paper PDF:
  https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Drop_an_Octave_Reducing_Spatial_Redundancy_in_Convolutional_Neural_Networks_ICCV_2019_paper.pdf
- Official archived implementation:
  https://github.com/facebookresearch/OctConv
- Reviewed official commit:
  `87c44f79162f3a2ef316bf924ad3697b8957a463`
- Reviewed official operator source:
  `utils/gluon/utils/octconv.py`, SHA-256
  `06e60037b8fd5d4cd9e063ab1d439b25ddc98c4a109f770cbc0adcb9d7295675`

The paper defines the factorized feature representation
`X = {X_H, X_L}` and the four paths:

```text
Y_H = conv(X_H, W_HH) + upsample(conv(X_L, W_LH), 2)
Y_L = conv(X_L, W_LL) + conv(avgpool(X_H, 2), W_HL)
```

Nearest-neighbor upsampling and average-pool downsampling follow the paper and
official source. The first operator uses `alpha_in=0`, the middle operator uses
`alpha_in=alpha_out`, and the last uses `alpha_out=0`. The paper reports peak
recognition accuracy commonly at `alpha=0.125` or `0.25`, with ResNet-50 best
at `0.125`; this protocol locks the sole value `alpha=0.125` to preserve
`87.5%` high-frequency channels. There is no alpha sweep.

## Local no-repeat decision

This route is admissible for one experiment because it is not any closed route:

- It is not fixed Gabor, wavelet scattering, high-pass statistics, or an FFT
  expert. Both frequency groups and all four communication kernels are learned
  jointly from class supervision; no handcrafted frequency response is added.
- It is not another MogaNet, InceptionNeXt, StarNet, MobileViT, EdgeNeXt, or
  CoAtNet catalogue substitution. Stem widths, three-block topology,
  downsampling, output width, patch grid, and Transformer remain those of the
  current `conv_pool` TRKH.
- It is not a background/context route. Candidate and control consume exactly
  the same object-level `yolo_f` tensor and bbox/valid-mask metadata.
- It is not a logit gate, teacher, prototype, threshold, label rewrite, or
  class-1 recall-pressure loss. The intervention is representation-only.

HorNet/gated high-order convolution is not selected because its recursive
multiplicative interaction substantially overlaps the closed MogaNet and
StarNet mechanisms. Selective Kernel is not selected because normalized object
crops reduce its scale-selection motivation, while local evidence identifies
surface/color separation rather than object scale as the bottleneck.

## Locked implementation

Add default-off `stem_architecture=octave_conv` with exactly three blocks:

| Block | Input channels | Output channels | `alpha_in` | `alpha_out` |
| --- | ---: | ---: | ---: | ---: |
| first | 3 | 32 | 0 | 0.125 |
| middle | 32 | 64 | 0.125 | 0.125 |
| final | 64 | 256 | 0.125 | 0 |

Each operator uses `3x3`, stride 1, padding 1, no bias. Each available output
frequency receives its own BatchNorm and GELU, followed by the same `2x2`
max-pool used by the control stem. The final result is one
`[B, 256, 32, 32]` tensor for 256px inputs; unchanged patch embedding produces
the same `16x16`/256-token grid.

The four OctConv path parameter counts must sum exactly to the corresponding
vanilla convolution parameter count. BatchNorm affine/state channels must also
sum exactly. Candidate and control full-model trainable parameter counts must
be identical. For a matched scratch comparison, each candidate block is
initialized by drawing the same virtual full `[C_out,C_in,3,3]` Kaiming kernel
as the control and partitioning it into `HH/HL/LL/LH`; candidate construction
and initialization must consume the same global RNG stream as `conv_pool`.
Do not add residual scales, gates, auxiliary losses, branch attention, extra
depth, or a classifier head.

The implementation must expose a diagnostic trace for first/middle/final
high/low tensors and every applicable `HH/HL/LL/LH` path. The normal forward
must return only the final tensor and must not retain graph tensors for logging.

## Stage A: train-only readiness

Stage A may read `yolo_f/train` only. It must not construct a validation or test
dataset/loader and must record `validation_loaded=false`, `test_loaded=false`.
It must pass all of the following before any validation smoke:

1. Exact source/data/operator commit and SHA locks pass.
2. Explicit `conv_pool` remains checkpoint- and RNG-identical to the implicit
   default. Control, candidate, and repeated-candidate construction leave the
   same complete CPU/CUDA RNG state after model initialization.
3. Candidate/control trainable parameters and the three convolution-kernel
   parameter totals are exactly equal.
4. Shapes are exact through every frequency group, final stem output, patch
   embedding, token pruning, logits, and architecture trace.
5. Every applicable `HH/HL/LL/LH` kernel has finite nonzero FP32 and AMP/BF16
   gradients. Input gradients are finite. All five train classes affect logits.
6. Each cross-frequency path is material: same-weight ablation changes the
   final stem output and logits by at least `1e-6` maximum absolute value.
7. The middle `HH` versus upsampled `LH` and `LL` versus `HL` contributions do
   not collapse (`abs cosine < 0.9995`). High/low spatial-energy maps are also
   nonconstant and not identical (`abs cosine < 0.9995`).
8. The low-frequency branch has lower normalized spatial total variation than
   the high-frequency branch on the locked train cohort, confirming the
   intended octave behavior rather than relying on names alone.
9. Checkpoint/config/CLI round-trip preserves `stem_architecture=octave_conv`;
   a legacy checkpoint remains loadable and unknown stem names fail closed.
10. ONNX Runtime FP32 export has max absolute error no greater than `1e-5` and
    zero output-shape mismatch.
11. Median batch-32 AMP training-step runtime is no greater than `1.50x`
    control and peak allocated CUDA memory is no greater than `3.25 GiB` on the
    local RTX 4060 Laptop GPU.

Failure of any Stage-A item closes this exact implementation without a smoke.
Do not tune alpha, pooling, interpolation, widths, kernels, norms, activation,
or stem depth to repair a failed gate.

## Sole matched smoke

Only after Stage A passes, run one source-hash-locked control/candidate pair:

- immutable `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`;
- source recipe
  `runs/full_v8_yolof_randominit_30e_20260714_105524/launcher_args.json`,
  SHA-256
  `a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6`;
- seed 42, scratch/no-pretrain, strict balanced sampling, batch 32,
  accumulation 2, 120 train batches per epoch, two epochs, scheduler horizon
  15, full validation `2606`, and no test;
- control `stem_architecture=conv_pool`;
- candidate `stem_architecture=octave_conv`;
- no attention-view loss/crop/drop in either arm.

The candidate receives continuation permission only if every gate passes:

- macro F1 improves by at least `0.003`;
- class-1 F1 improves by at least `0.005`;
- class-1 precision improves by at least `0.010`;
- class-1 recall falls by no more than `0.010`;
- class-1 true-positive breaks do not exceed rescues;
- locked `0/2/4 -> 1` false positives removed minus created is at least `4`;
- total corrections exceed harms;
- new `3 -> 2` harms do not exceed `3`;
- all Stage-A runtime, memory, provenance, and no-test constraints remain true.

No single metric may override a failed TP-preservation or precision gate.

## Mandatory post-smoke audit

Run these audits after the smoke whether it passes or fails:

1. independent FP32 and matched-AMP reload with prediction reconciliation;
2. full confusion, classwise P/R/F1, calibration, changed transitions, source
   groups, crop/bbox/image statistics, and class-1 FP/FN/TP accounting;
3. five fixed robustness conditions: clean, dim, bright, low contrast, and
   center occlusion;
4. architecture trace and OctConv-specific branch/path energy, total variation,
   contribution cosine, and same-weight path-ablation diagnostics;
5. exact changed-case cohort with native Transformer attention, rollout,
   gradient rollout, Grad-CAM on final stem output, object desaturation,
   background gray, and background blur; no silent fallback is allowed;
6. visual review of every important class-1 rescue, TP break, FP removal, FP
   creation, and new `3 -> 2` harm, capped only after deterministic ranking is
   recorded.

Do not run a five-epoch continuation, probe, full train, final test, TensorRT
benchmark, or video test unless the complete continuation gate passes. Do not
update the current-best VS Code command packet unless an independently reloaded
single checkpoint beats the keeper's macro/class-1 F1 and raises class-1
precision while preserving its recall/TP floor.

## Stop rule

Failure at Stage A or the sole matched smoke closes this exact OctConv stem.
Do not sweep alpha, branch paths, pooling, interpolation, channels, depth,
kernel, norm, activation, optimizer, LR, seed, loss, sampler, augmentation,
checkpoint, or run length. Preserve compact evidence, update the journal and
TRKH skill, rerun retention/protected hashes, and leave current-best commands
unchanged.
