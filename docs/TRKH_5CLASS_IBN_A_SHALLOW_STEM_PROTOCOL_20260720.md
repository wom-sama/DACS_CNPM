# TRKH 5-Class IBN-a Shallow-Stem Protocol - 2026-07-20

## Status

Prospectively locked before implementation or measurement. This protocol
authorizes one train-only engineering/readiness audit and, only if every Stage-A
gate passes, one matched 120-batch/two-epoch validation smoke. It does not
authorize test access, a setting sweep, a full train, or current-best command
promotion.

## Research Question

Can a conservative IBN-a split in the first TRKH convolutional stem block
reduce illumination/style sensitivity while preserving the batch-normalized
channels that carry mango color and maturity information? The target is higher
class-1 precision without sacrificing its recall or the other four classes.

## Primary-Source Lock

- Accepted paper: Pan et al., "Two at Once: Enhancing Learning and
  Generalization Capacities via IBN-Net," ECCV 2018:
  `https://openaccess.thecvf.com/content_ECCV_2018/html/Xingang_Pan_Two_at_Once_ECCV_2018_paper.html`.
- Local paper:
  `D:/DataAI/external_sources/papers/ibn_net_eccv2018.pdf`.
- Paper SHA-256:
  `411fd8d8afd08e0d57f295821919f286c7206e7fe1ce982cedc0f06045de1b36`.
- Official MIT repository: `https://github.com/XingangPan/IBN-Net`.
- Locked commit/tree:
  `d1673389b36c1180cf9bc35ea8260d84046da915` /
  `e113673c2aa64dc761a67f1585fd168a95d6ab4b`.
- Reviewed implementation SHAs:
  `ibnnet/modules.py=1578ca15ba7fbe6349d3e590c684c16236a2c0ccabf737a4762f6fed8a6b499c` and
  `ibnnet/resnet_ibn.py=32af52f5f638bb0528b537b2e352da4ed65c02ccbf39b84fc0c5e06b35edd0f9`.
- MIT license SHA-256:
  `4e2e849faed41630d067a8789edd12e4da452e8cd52b326c44e54b911b064532`.

The official IBN-a layer splits channels at ratio 0.5, applies affine instance
normalization to the first group and batch normalization to the second, then
concatenates them. The paper motivates IN as suppressing appearance variance
and BN as retaining content-discriminative information. It applies IBN only in
earlier residual stages. TRKH imports this equation and placement principle,
not pretrained weights, reported ImageNet accuracy, or repository code.

## Local No-Repeat Boundary

- MixStyle, illumination normalization, patch-style recalibration, SRM,
  background suppression, color-stat fusion, axial-color descriptors, and
  multiple color/texture experts are already closed. This experiment changes
  only feature normalization inside the existing shallow convolutional stem;
  it adds no augmentation, style mixing, descriptor branch, teacher, router,
  threshold, or loss.
- IBN-b, full-IN, layer/group/switchable normalization, a learnable IN ratio,
  channel reordering, and IBN in blocks 2-3 are excluded. Color is label
  evidence in this dataset, so the one allowed candidate is the conservative
  50/50 IBN-a split in stem block 1 only.
- Raw images, labels, split files, and dataset YAMLs are immutable.

## Locked Inputs

- Pre-protocol HEAD/upstream:
  `7e209be25c522da0ece22617fa62d4e4527fc637`.
- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher arguments SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- Raw `yolo_f/data.yaml` SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Pre-implementation source SHAs:
  `model.py=57942c7e0457e947d21b77d20ad789dc6694b31e5d457a2b65dc247efe430f7f`,
  `config.py=a0b79826ac35f226824701ae064ab78a23ff91797d55cecb27ec1dde9167335b`,
  `train.py=80b39fc8cde7c2192ce31f198936246dd757e5de78175cb7d4d8b614addfebc8`,
  and launcher
  `c78752c1113c314851ba39b70c43398a7cb63f9bed7d285e48ed301d59a7292a`.
- Current-best command/history SHAs remain
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

## Locked Candidate

Add model option `stem_normalization` with exactly two values:

- `batch` (default): existing three-block `Conv-BN-GELU-MaxPool` stem;
- `ibn_a_first`: only block 1 uses a 0.5-channel affine IN/BN split; blocks 2
  and 3 remain ordinary BN.

The IBN module must retain the original BatchNorm state-dict schema and tensor
shapes (`weight`, `bias`, `running_mean`, `running_var`, and
`num_batches_tracked`). The first half uses the corresponding affine values
with per-instance spatial statistics; the second half uses the corresponding
BN affine values and running statistics. This permits strict loading of the
keeper and gives the control and candidate bit-identical parameters at time
zero. Parameter count, stem output `[B,256,32,32]`, patch/token geometry, and
all non-normalization behavior must remain identical.

## Stage-A Readiness

Stage A may open only `yolo_f/train` and uses one deterministic five-class
cohort plus one resource batch. It must persist exact source/config/checkpoint
hashes, selected paths, module/state schemas, numerical oracles, gradient and
resource telemetry, and an artifact manifest.

Required gates:

1. Explicit `stem_normalization=batch` is bit-identical to the omitted default
   in state and logits.
2. Control and candidate state-dict keys, values, parameter count, and all
   non-module configuration fields are identical before a forward pass.
3. Exactly block 1 is IBN-a; blocks 2-3 remain BatchNorm2d; channel split is
   `16/16` for the keeper's 32-channel first block.
4. Candidate output matches an independent direct IBN-a equation within
   `1e-6`; its IN half has per-instance spatial mean within `1e-5` of zero and
   variance within `2e-4` of one on a nondegenerate FP32 oracle.
5. Under positive per-instance affine appearance transforms of the block-1
   pre-normalization activation, candidate IN-half error is at most 5% of the
   control BN-half error, while the candidate BN half is bit-identical to the
   control BN half for identical input.
6. Strict keeper checkpoint load succeeds with no missing/unexpected keys;
   forward/backward gradients are finite and nonzero through both channel
   groups and the patch projection.
7. FP32 and CUDA AMP outputs are finite, peak CUDA allocation is at most
   7.5 GiB, candidate runtime is at most 1.30 times control, and export
   preflight uses only standard deployable normalization operators.
8. Selected paths resolve under the training image root; validation and test
   are not opened. Focused tests, full pytest, compile checks, PowerShell parse,
   and `git diff --check` pass on a committed and pushed implementation.

Any failed Stage-A gate closes this exact candidate before training.

## Matched Smoke

Only a passing Stage A permits one pair derived from the locked keeper
`launcher_args.json`. Both roles retain the same v8 data, checkpoint resume,
seed 42, augmentations, losses, optimizer, scheduler, batch size 32, gradient
accumulation 2, `num_workers=4`, `eval_num_workers=2`, two epochs, 120 train
batches per epoch, full validation, architecture trace, no pretrained weights,
and no final test. The sole candidate difference is
`stem_normalization=batch -> ibn_a_first`.

Precision-first smoke gates, all required:

- full validation support `2606` and exact row alignment;
- macro F1 delta at least `0.000`;
- class-1 F1 gain at least `+0.010`;
- class-1 precision gain at least `+0.020`;
- class-1 recall delta at least `-0.015`;
- at least four fewer `{0,2,4}->1` false positives;
- corrections are at least harms and class-1 TP breaks do not exceed class-1
  FN rescues;
- maximum non-focus-class F1 loss is at most `0.015`;
- runtime ratio is at most `1.30`, traces/configs prove the locked difference,
  and neither run contains test output.

## Escalation And Stop Rules

A smoke pass authorizes changed-case XAI plus fixed dim/bright/low-contrast
robustness replay. Only if clean metrics, shifted class-1 precision/recall,
foreground attribution, background sensitivity, calibration, throughput, and
strict checkpoint reload all pass may one five-epoch probe be run. A probe that
beats the locked validation promotion gate and remains prospectively clean may
authorize one autonomous full train with at most 30 epochs, early-stopping
patience 3, and measured workers `4/2`. Final test remains isolated until the
existing independent-reload promotion gate grants it.

Any material gate failure closes `ibn_a_first`; do not sweep ratio, placement,
channel order, normalization family, seed, augmentation, loss, threshold, or
nearby settings. The current-best command/history files remain unchanged unless
a later candidate passes the complete locked promotion process.
