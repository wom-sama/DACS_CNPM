# TRKH 5-Class Diverse-Branch Stem A0 Readiness Protocol - 2026-07-16

## Status

Precommitted before implementation. This document authorizes one exact,
train-only, scratch-initialized control/candidate experiment after every
engineering preflight gate passes on a committed and pushed implementation.
It does not authorize official validation or test access, a full 30-epoch
train, a hyperparameter sweep, or a current-best command update.

## Research Question

Can CVPR-2021 Diverse Branch Block (DBB) training-time structural
re-parameterization improve the three-layer CNN tokenizer of scratch TRKH by
learning complementary local paths while preserving the exact single-3x3-conv
deployment structure, thereby increasing class-1 precision without losing
class-1 support or illumination robustness?

The candidate changes only the three `Conv2d(3x3)-BatchNorm` pairs in the
current `HybridConvStem`. GELU, max pooling, the 16x16 patch grid, patch
projection, all eight Transformer blocks, registers, pruning, bbox spatial
fusion, pooling, heads, losses, and every unrelated training setting remain
shared with the control. Each candidate block is converted once after training
to one bias-enabled 3x3 convolution before deployment evaluation and export.

## Primary-Source Lock

The accepted paper and official Apache-2.0 repository define the method.
User-provided and secondary reports may motivate screening but do not define
equations, settings, or gates.

- Accepted paper: Ding, Zhang, Han, and Ding, "Diverse Branch Block: Building
  a Convolution as an Inception-Like Unit," CVPR 2021:
  `https://openaccess.thecvf.com/content/CVPR2021/html/Ding_Diverse_Branch_Block_Building_a_Convolution_as_an_Inception-Like_Unit_CVPR_2021_paper.html`.
- Local primary paper:
  `D:/DataAI/external_sources/papers/dbb_cvpr2021.pdf`.
- Paper SHA-256:
  `ba1c0f907f69dea3a6e84634f4198ff4cb9799569633078a9ca5a013e3b0d991`.
- Official repository: `https://github.com/DingXiaoH/DiverseBranchBlock`
  (352 stars observed on 2026-07-16; popularity is context, not a gate).
- Locked official commit/tree:
  `8d2b16b6aee45a33236b2d11685be6857f9ba929` /
  `b038d8e014664312f8979d0b7917466e70877b30`.
- Reviewed implementation `diversebranchblock.py` SHA-256:
  `5f67f20081ffef0a9f7ad6f9f5cc1a7bb4e59902dc31c86a8e59c37fd477ccb0`.
- Reviewed transformations `dbb_transforms.py` SHA-256:
  `1424673513c4883ffb2e88b24f81a41cb69569f2621aa18f76a8fe341c92b11f`.
- Reviewed README SHA-256:
  `a8966d48d0eb0f37923bb9e5863f1409bd6f4daaeceefbbb07316e9589456267`.
- Apache-2.0 license SHA-256:
  `1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6`.

The paper replaces every eligible KxK conv plus following BN, trains the
multi-branch structure, and then algebraically fuses it to the original conv.
The representative DBB has four paths: KxK, 1x1, 1x1-KxK, and 1x1-average
pooling. The 1x1-KxK internal width equals the input width and its first 1x1
kernel starts as identity. A BN follows every convolution or average-pooling
path. The paper reports that diverse branches and branch-specific BN, rather
than converted initialization alone, drive the gain. Its ImageNet recipe uses
120 epochs, so its schedule is not imported into the fixed TRKH `<=30e`
contract.

## Local No-Repeat Boundary

This route does not reopen the following closed families:

- InceptionNeXt, MogaNet, StarNet, OctConv, CEConv, and other tokenizer routes
  permanently change inference-time operators. DBB is an optimization-time
  parameterization whose deployed candidate is exactly the original three
  3x3-convolution macrostructure.
- RepVGG is not selected. The official DBB FAQ reports that RepVGG-style
  re-parameterization offers almost no improvement on a non-plain ResNet-50,
  while DBB was designed as the universal replacement for non-plain models.
- The DBB average path is an exactly fusable fixed local operator. It is not a
  persistent low-frequency stream, Fourier/wavelet descriptor, deformable
  sampler, local/global attention route, graph mixer, or input augmentation.
- No channel gate, prompt, capsule, prototype, verifier, router, threshold,
  ensemble, teacher, distillation, oversampling, bbox loss, new data view, or
  raw-data modification is allowed.
- NeurIPS-2021 FENet remains deferred, not combined: its official repository
  has no displayed license, only 21 observed stars, and paper results initialize
  the backbone from ImageNet. Fractal encoding requires a separate prospective
  protocol if DBB closes.

DBB is authorized only as one test of a previously unmeasured mechanism:
whether branch-specific optimization geometry can learn a better local surface
tokenizer under the same scratch budget while compiling away exactly.

## Locked Inputs And Provenance

- Repository pre-protocol HEAD/upstream:
  `ecc7dcd269f60e27c0ed39575a48b115707f0f2d`.
- Raw data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Frozen train-only declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`,
  SHA-256
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Generated fold YAML/summary SHAs:
  `4195ba89995d4eee483cae31626717381df910f56657086e7116322ca971388e` /
  `2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763`.
- Keeper/scratch-complement checkpoint SHAs:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677` /
  `f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549`.
- Current-best command/history SHAs:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

Reuse only `runs/yolof_cidt_fold0_trainonly_20260716`:

- fit: 7,372 object rows, 6,452 source stems, class counts
  `[1561,432,1527,2017,1835]`;
- holdout: 1,843 object rows, 1,612 source stems, class counts
  `[380,109,393,503,458]`;
- fit/holdout source overlap: zero;
- fit/holdout sample-index SHAs:
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d` /
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`.

The generated `test` path mirrors holdout only for loader compatibility. Every
launcher must use `SkipFinalTest`; no test loader or test metric may be
constructed. Raw dataset files remain read-only.

## Locked Candidate Architecture

Add default-off `stem_architecture=dbb_conv_pool`. For each of the three
legacy stem blocks (`3->32`, `32->64`, `64->256`), replace only the 3x3
conv-BN sequence with the official representative dense DBB:

- KxK origin: 3x3 conv, stride 1, padding 1, bias false, then BN;
- multi-scale path: 1x1 conv, stride 1, bias false, then BN;
- sequential path: identity-initialized 1x1 with internal width equal to input
  channels, BN-and-one-pixel bias padding, 3x3 conv without padding, then BN;
- average path: 1x1 conv, BN-and-one-pixel bias padding, 3x3 average pooling
  with stride 1 and no padding, then BN;
- branch merge: addition only; no activation inside the DBB;
- official default initialization: terminal BN gamma is one on every branch;
- unchanged parent tail: one GELU followed by the existing 2x2 max pool.

The implementation must retain Apache attribution and independently reproduce
the official equations. No external source is imported at runtime.

For deployment, fuse every conv-BN, sequential path, average path, smaller
kernel, and branch sum into one bias-enabled 3x3 conv. Conversion is destructive
only on a model copy used for deployment checks/export. The training checkpoint
retains all DBB branches. A converted candidate must expose exactly three stem
convolutions and no DBB branch BN, average pool, or sequential module.

For a seed-42 pair, the candidate origin conv/BN and all non-stem tensors must
be copied from the control and be bit-exact. Candidate-only branch state is
initialized from a dedicated restored RNG fork, and construction must not alter
the downstream control/candidate RNG contract. DBB-only state remains excluded
from the common-state equality check.

## Equation And Engineering Preflight

Before formal training, every check below must pass from a committed and pushed
implementation:

1. every locked input/source SHA, official commit/tree, protocol SHA,
   HEAD/upstream, and tracked-worktree status matches;
2. FP32 train/eval output, equivalent kernel/bias, input gradient, and every
   remapped parameter gradient reproduce the official implementation within
   `1e-6` on fixed random and real fit tensors;
3. each official transform has an independent tensor-equation replay, including
   BN fusion, branch addition, 1x1-to-3x3 padding, sequential fusion, average
   pooling conversion, and BN-aware border padding;
4. the three candidate origin conv/BN pairs and every non-stem tensor are
   bit-exact to control at seed 42, candidate-only state is finite, and CPU/CUDA
   RNG state after paired construction is unchanged;
5. all four paths in every DBB have finite nonzero output norm, all branch
   parameter families receive finite nonzero gradients on a real fit batch,
   and no branch output is numerically identical to another;
6. FP32 and CUDA BF16 forward/backward are finite;
7. train-structure eval versus converted-deploy stem/model maximum absolute
   output and logit errors are `<=1e-5` in FP32 and `<=2e-3` in BF16, with zero
   argmax mismatch;
8. converted candidate stem parameter count equals a control stem with BN
   folded into its three convolutions, and no candidate-only module remains;
9. static batch-1 converted-model ONNX replay has maximum logit error `<=1e-4`
   and matching argmax;
10. converted inference runtime is at most `1.05x` folded control, DBB training
    step runtime is at most `1.50x` control, and peak allocated training VRAM is
    at most `7.5 GiB` and `1.25x` control;
11. the exact generated fold rows/classes/sources/hashes and zero source overlap
    pass, and no official validation/test path is opened;
12. preflight creates a complete source/equation/state/resource/export manifest
    and no checkpoint or behavior result.

One engineering smoke may use at most 16 fit batches and one complete train-only
holdout pass only to prove execution and artifact contracts. It cannot select
the method and must be removed after its diagnostics are recorded.

## Locked Five-Epoch Train-Only Pair

Run exactly one control and one candidate from random initialization, seed 42.
Neither may resume a checkpoint or use pretrained weights. Both use the
generated fold and identical ordered sample occurrences, transforms, optimizer,
scheduler, losses, EMA, stopping, and checkpoint metric.

Shared model recipe:

- current `vit_registers`, image size 256, `embed_dim=256`, depth 8, heads 8,
  four registers;
- control stem `conv_pool`; candidate stem `dbb_conv_pool`;
- bbox spatial fusion, CNN feature fusion, fine-grained pooling, two branch
  tokens, detail patch enhancement, and token pruning at layers `2,5` with
  keep rates `0.85,0.65` remain enabled;
- every unrelated architecture experiment remains disabled.

Shared training recipe:

- epochs `5`, batch `32`, accumulation `2`, AdamW LR `2.5e-4`, weight decay
  `0.05`, warmup `1`, cosine horizon `5`, minimum LR `1e-6`, gradient clip
  `0.7`, patience `3`, EMA `0.995`;
- natural-frequency epoch sampling only; balanced/weighted sampling,
  rare-class repeat, and global class-1 oversampling disabled;
- `ldam_focal`, label smoothing `0.02`, LDAM max margin `0.3`, scale `18`,
  focal gamma `1.0`, focal mix `0.1`;
- pairwise-margin and metric-learning losses remain `0.04/0.04` with the same
  boundary pairs/sources as the scratch record;
- ordinary scratch-record train augmentation remains identical;
- attention-view loss/drop, teacher focus/cache, distillation, target/sample
  manifests, mixup, CutMix, copy-paste, threshold, router, TTA, and final test
  remain disabled.

Persist arguments, resolved configs, paired common-state manifest, ordered
sample-occurrence hashes per epoch, train/holdout metrics, best/last checkpoints,
resource telemetry, architecture traces, and per-branch telemetry. The
train-only holdout metric selects checkpoints for both roles and is not an
official validation result.

## Frozen Conditions And Mechanism Audit

Evaluate both best checkpoints on all 1,843 train-only holdout rows under
clean, dim (`brightness=0.70`, `contrast=0.90`), bright
(`brightness=1.25`, `contrast=1.10`), and low-contrast
(`brightness=1.00`, `contrast=0.65`) conditions.

For candidate training and best checkpoints, report per block/path:

- terminal BN gamma magnitude and movement from initialization;
- path output RMS and share of summed pre-GELU RMS;
- equivalent-kernel norm, center/skeleton mass, and pairwise kernel cosine;
- single-path ablation logit and decision effects on the frozen cohort;
- train-structure versus converted-deploy parity on every audited row.

The frozen selectivity cohort is inherited from the fold-0 CIDT keeper:
positives are class-1 rows predicted class 1 by the keeper; hard negatives have
target in `{0,2,4}` and keeper prediction 1. Report AUROC of
`logit_1-max(logit_0,logit_2,logit_4)` for each role and condition.

## A0 Promotion Gates

Every engineering, clean behavior, condition, mechanism, deployment, and XAI
gate must pass. Compare candidate with matched control on all 1,843 holdout
rows, prioritizing precision:

- clean macro F1 delta `>=+0.003`;
- clean class-1 F1 delta `>=+0.015` and candidate class-1 F1 `>=0.60`;
- clean class-1 precision delta `>=+0.025`;
- clean class-1 recall delta `>=-0.010`;
- restricted `{0,2,4}->1` false positives decrease by at least four;
- corrections exceed harms, class-1 FN rescues are at least TP breaks, and
  maximum non-focus per-class F1 drop is `<=0.010`;
- clean frozen-cohort AUROC is `>=0.65` and improves by `>=+0.020`;
- candidate class-1 precision is no lower than control under any shifted
  condition and improves by at least `0.010` in two of three;
- shifted class-1 recall is never more than `0.020` below control, restricted
  false positives never increase, and no class-1 F1 delta is below `-0.010`;
- all 12 DBB paths remain active, at least two path families per block have
  mean terminal gamma `>=0.05`, and no two fused path kernels have absolute
  cosine above `0.995` in every block;
- converted deployment matches train-structure decisions on every condition,
  passes ONNX/resource gates, and contains no DBB-only operator;
- stem Grad-CAM foreground mass is no more than `0.05` below control, object
  desaturation/blur remains more causal than far-background perturbation, and
  every class-1 transition plus every restricted-FP change is included in the
  XAI manifest/contact sheets and visually reviewed.

## Permission And Stop Rules

- Passing every A0 gate authorizes a separate prospective Stage-B protocol for
  at most ten total train-only epochs. It does not authorize official
  validation, test, 30 epochs, or current-best command promotion by itself.
- Failing any material gate closes this exact DBB stem route. Do not sweep
  branch subsets, internal width, BN gamma initialization, block count,
  insertion location, activation, pooling, optimizer, LR, weight decay, epoch,
  augmentation, fold, seed, threshold, router, or combine DBB with RepVGG,
  FENet, ACB, OREPA, or RepOptimizer on the current keeper.
- Validation and test remain inaccessible during A0. Test remains final audit
  only after a later validation promotion.
- A0 cannot change current-best command/history/pipeline files. Preserve the
  keeper and scratch-complement hashes. Compact rejected artifacts only after
  journal/closure, independent replay, and retention verification.
