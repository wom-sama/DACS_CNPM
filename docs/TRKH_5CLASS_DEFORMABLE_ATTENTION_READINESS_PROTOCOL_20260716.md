# TRKH 5-Class Deformable Attention A1 Readiness Protocol - 2026-07-16

## Status

Precommitted before implementation. This document authorizes one exact,
train-only, scratch-initialized control/candidate experiment. It does not
authorize official validation or test access, a full 30-epoch train, a
hyperparameter sweep, or a current-best command update.

## Research Question

Can one CVPR-2022 DAT-style deformable self-attention block improve the scratch
TRKH representation of subtle mango surface/maturity boundaries by moving a
small shared set of key/value sampling points toward image-dependent fruit
evidence, thereby increasing class-1 precision without suppressing true class-1
support?

The candidate is not a stock DAT backbone. It changes only Transformer block 2
of the current eight-block TRKH. Block 1 remains standard global MHSA. In block
2, prefix queries retain standard global attention while patch queries attend
to spatially deformed patch keys/values. Token pruning remains immediately
after block 2 and consumes an exact bilinear-scatter attention proxy. Blocks
3-8, the CNN stem, patch projection, prefix tokens, pruning schedule, bbox
spatial fusion, pooling, heads, and every unrelated training setting are shared
with the control.

## Primary-Source Lock

The accepted paper and its official Apache-2.0 repository define the method.
User-provided and secondary research reports may motivate screening but do not
define equations, settings, or gates.

- Accepted paper: Xia, Pan, Song, Li, and Huang, "Vision Transformer with
  Deformable Attention," CVPR 2022:
  `https://openaccess.thecvf.com/content/CVPR2022/html/Xia_Vision_Transformer_With_Deformable_Attention_CVPR_2022_paper.html`.
- Local primary arXiv manuscript:
  `D:/DataAI/external_sources/papers/xia2022_dat.pdf`.
- Local paper SHA-256:
  `92c3f6bba2aac7c1039ee4ed386743d345db8088e88279311eb7397630cb8a60`.
- Official repository: `https://github.com/LeapLabTHU/DAT` (938 stars observed
  on 2026-07-16; star count is context, not a reproducibility gate).
- Locked official tag/commit:
  `CVPR2022` / `566a593daf96efc3df58a1542e60b847b8b6f4ff`.
- Locked tree: `4346f051456b3270e9161839410123260db82508`.
- Reviewed attention source: `models/dat_blocks.py`, SHA-256
  `f73185275dfddc6d5f1530ab5e10d3b5a85102b72ae7c73a9d85f64d8b7470ea`.
- Reviewed architecture source: `models/dat.py`, SHA-256
  `cea9a503e5b9c456802011cf84e585e3642b30140f4f2aa889b47272312d8be5`.
- Reviewed `dat_tiny` recipe: `configs/dat_tiny.yaml`, SHA-256
  `9427c5621e3636f9a6781dd1f8988451c8e6348498fdb1f3efd6841a9e119316`.
- Official license SHA-256:
  `1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6`.

The paper samples keys/values at data-dependent positions generated from query
features, uses bilinear interpolation, grouped offsets, and continuous
deformable relative-position bias. The official ImageNet recipe is 300 epochs
and DAT-T is about 29M parameters; neither a stock model nor its recipe is
compatible with the fixed scratch/`<=30e` TRKH contract.

The paper's stage ablation is binding context. Deformable attention in only the
last two spatial stages performed best, while adding it to earlier high-
resolution stages slightly reduced accuracy. TRKH has one `16x16` patch grid
and prunes after block 2. Block 2 is therefore the sole authorized location: it
still sees the complete dense grid, whose resolution is close to DAT stage 3
(`14x14`), and no later TRKH block has a complete rectangular grid. Block 1 and
multi-layer variants are excluded prospectively.

## Local No-Repeat Boundary

This route is distinct from, and must not reopen, nearby closed work:

- FAA used fixed local and pooled positions with one two-route competition;
  DAT predicts continuous spatial key/value positions from each image and has
  no local-versus-pooled route gate.
- ViG selected feature-space k-nearest neighbors and added max-relative graph
  mixing after standard MHSA; DAT replaces patch MHSA key/value sampling using
  grouped spatial offsets inside the attention equation.
- Deformable convolution changes CNN kernel sampling per output location; DAT
  shares each offset group's sampled key/value set across all patch queries.
- VCA, GPSA, local windows, dynamic graph mixing, local zoom, tiling, paired
  `class_f+yolo_f` views, bbox/objectness forcing, context fusion, ConvGLU/LeFF,
  and stock architecture replacement remain closed.
- No verifier, router, threshold, ensemble, pretrained teacher, distillation,
  class-1 oversampling, test tuning, or raw-data modification is allowed.

DAT is authorized only because it tests a previously unmeasured mechanism:
whether image-conditioned continuous sampling can move sparse attention
evidence toward the labeled fruit while retaining natural multiclass support.

## Locked Inputs And Provenance

- Repository pre-protocol HEAD and upstream:
  `ad77e5da9e7ea874b44e05e312a8aa65ccf17fdc`.
- Data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Frozen train-only fold declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`,
  SHA-256
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Declaration companion summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.
- Scratch launcher/resolved-config SHAs:
  `a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6` /
  `c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97`.
- Keeper diagnostic checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Current-best command/history SHAs:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
- FAA closure SHA-256:
  `3cb722008f59e2558b66eee47efdee86def884225f882ab3e2be40079070a896`.

Reuse only the existing generated fold at
`runs/yolof_cidt_fold0_trainonly_20260716`:

- fit: 7,372 object rows, 6,452 source stems, class counts
  `[1561,432,1527,2017,1835]`;
- holdout: 1,843 object rows, 1,612 source stems, class counts
  `[380,109,393,503,458]`;
- fit/holdout source overlap: zero;
- fit/holdout sample-index SHAs:
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d` /
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`;
- fold summary SHA-256:
  `2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763`.

The generated `test` path mirrors holdout only for loader compatibility. Every
launcher must use `SkipFinalTest`; no test loader or test metric may be
constructed. Raw dataset files remain read-only.

## Locked Candidate Architecture

The control uses the current standard block-2 `MultiHeadSelfAttention`. The
candidate replaces only that module with `DeformableSpatialAttention`:

- layer: block 2 only, before the existing first token-pruning operation;
- dense patch grid: exactly `16x16`, identity-ordered patch indices;
- embedding/heads/head dimension: `256/8/32`;
- offset groups: `2`, preserving the official ratio of four heads per group;
- sampled grid: stride `1`, exactly `16x16` positions per group;
- offset network: shared `5x5` depthwise convolution, channel LayerNorm, GELU,
  and bias-free `1x1` two-channel projection;
- reference points: official half-cell centers normalized to `[-1,+1]`;
- offset range: official `2 * tanh(offset) * [1/H,1/W]`;
- sampling: bilinear `grid_sample`, zero padding, `align_corners=True`;
- position bias: one trainable `[8,31,31]` table sampled from continuous
  query-to-deformed-key displacement exactly as in official DAT;
- patch attention: standard scaled dot-product softmax over 256 deformed
  key/value samples;
- prefix attention: unchanged global scaled dot-product attention over every
  prefix and patch token; patch queries consume no prefix values in block 2;
- block 2 output projection, residual, FFN, pruning, and blocks 3-8: unchanged.

The candidate retains public `qkv` and `proj` modules. Construction must first
instantiate standard attention, initialize candidate-only parameters inside a
restored RNG fork, and copy common `qkv/proj` state. Separately constructed
seed-42 control/candidate models must therefore have bit-exact common state;
candidate-only initialization may not perturb downstream shared parameters or
CPU/CUDA RNG state.

For `return_attention=True`, prefix rows expose exact global attention. Patch
attention is scattered through the same bilinear weights used by sampling into
the original 256 patch columns, zero-outside contribution is omitted, and each
nonempty patch row is renormalized only for the pruning/XAI proxy. Patch-to-
prefix mass is zero. This proxy must be labeled
`deformable_bilinear_sample_proxy`; it is not represented as native dense MHSA.
Exact sampled attention, positions, references, offsets, valid interpolation
mass, and per-group traces remain separately available.

## Equation And Engineering Preflight

Before formal training, all checks below must pass from a committed and pushed
implementation:

1. every locked input/source SHA, official tag/tree, protocol SHA, HEAD/upstream,
   and tracked-worktree status matches;
2. an independent patch-only FP32 replay against official
   `DAttentionBaseline` matches output, positions, references, input gradients,
   and every remapped parameter gradient within `1e-6`;
3. candidate prefix outputs match the common-state standard MHSA prefix outputs
   within `1e-6` in eval FP32;
4. common control/candidate state is bit-exact at seed 42, candidate-only state
   is finite, and constructor CPU/CUDA RNG states are unchanged;
5. offset depthwise, offset normalization, offset pointwise, relative-bias,
   qkv, and output-projection families receive finite nonzero gradients on a
   real fit batch;
6. FP32 and CUDA BF16 forward/backward outputs, attention, positions, offsets,
   and gradients are finite;
7. returned proxy has exact `[B,8,T,T]` shape, nonnegative normalized rows,
   zero patch-to-prefix mass, and matches an independent bilinear scatter replay
   within `1e-6`;
8. sampled/reference positions are finite; every group has nonzero offset RMS,
   inter-group position RMS is nonzero, and valid interpolation mass is nonzero;
9. candidate parameters are at most control plus `50,000`;
10. static batch-1 ONNX export/replay has maximum logit error `<=1e-4` and
    matching argmax;
11. median inference runtime is at most `1.35x` control and peak allocated VRAM
    is at most `7.5 GiB` and `1.25x` control;
12. the generated fold has exact rows/classes/sources/hashes and zero overlap,
    and no official validation/test path is opened.

No preflight gate requires random offsets to be foreground-selective. That is a
learned paper claim and is measured after the formal pair. Unlike FAA, DAT has
no two-route mass requirement.

An engineering smoke may use at most 32 fit batches and one complete train-only
holdout pass only to verify execution/artifacts. It cannot select the method and
must be deleted after its diagnostics are recorded.

## Locked Five-Epoch Train-Only Pair

Run exactly one control and one candidate from random initialization, seed 42.
Neither may resume a checkpoint or use pretrained weights. Both use the
generated fold and identical ordered sample occurrences, transforms, optimizer,
scheduler, losses, EMA, stopping, and checkpoint metric.

Shared model recipe:

- current `vit_registers`, image size 256, conv-pool CNN stem, `embed_dim=256`,
  depth 8, heads 8, four registers;
- bbox spatial fusion, CNN feature fusion, fine-grained pooling, two branch
  tokens, detail patch enhancement, and token pruning at layers `2,5` with keep
  rates `0.85,0.65` remain enabled;
- every unrelated architecture experiment remains disabled.

Shared training recipe:

- epochs `5`, batch `32`, accumulation `2`, AdamW LR `2.5e-4`, weight decay
  `0.05`, warmup `1`, cosine horizon `5`, min LR `1e-6`, gradient clip `0.7`,
  patience `3`, EMA `0.995`;
- natural-frequency epoch sampling only; balanced/weighted sampling, rare-class
  repeat, and global class-1 oversampling disabled;
- `ldam_focal`, label smoothing `0.02`, LDAM max margin `0.3`, scale `18`,
  focal gamma `1.0`, focal mix `0.1`;
- pairwise-margin and metric-learning losses remain `0.04/0.04` with the same
  boundary pairs/sources as the scratch record;
- ordinary scratch-record image augmentation remains identical, including crop/
  pad, mild affine/color jitter, illumination normalization, background
  suppression, local exposure, obstacle, and horizontal flip;
- attention-view loss/drop, teacher focus, teacher cache, distillation, target
  manifest, sample weighting, mixup/CutMix/copy-paste, threshold, router, TTA,
  and final test disabled.

Persist arguments, resolved configs, initial/common-state manifest, ordered
sample-occurrence hashes per epoch, train/holdout metrics, best/last checkpoints,
resource telemetry, architecture traces, and deformable traces. The train-only
holdout metric selects checkpoints for both roles and is not a paper validation
result.

## Frozen Selectivity Cohort And Conditions

Use the fold-0 CIDT keeper decisions to define the same model-independent cohort
before reading control/candidate outcomes:

- positives: target class 1 predicted class 1 by the locked keeper;
- hard negatives: target in `{0,2,4}` predicted class 1 by the keeper;
- margin: `logit_1 - max(logit_0,logit_2,logit_4)`;
- report AUROC separating positives from hard negatives for each role/condition.

Conditions are the existing deterministic train-only transforms: clean,
brightness `0.70`/contrast `0.90`, brightness `1.25`/contrast `1.10`, and
brightness `1.00`/contrast `0.65`.

Use each YOLO object row's own bbox only for audit. For each candidate row,
measure the fraction of deformed positions inside its bbox, the same fraction
for uniform reference positions, mean distance of outside positions to the
bbox, normalized offset RMS, group diversity, valid interpolation mass, and
attention-proxy bbox mass. Bboxes do not create a new loss, sample weight,
target, crop, or model input.

## A1 Promotion Gates

Every engineering, training, behavior, condition, spatial-mechanism, and XAI
gate must pass. Candidate is compared with the matched control best checkpoint
on all 1,843 train-only holdout rows.

Clean behavior gates, with precision intentionally prioritized:

- macro F1 delta `>=+0.003`;
- class-1 F1 delta `>=+0.015` and candidate class-1 F1 `>=0.60`;
- class-1 precision delta `>=+0.025`;
- class-1 recall delta `>=-0.010`;
- restricted `{0,2,4}->1` false positives decrease by at least four;
- corrections exceed harms;
- class-1 FN rescues are at least class-1 TP breaks;
- maximum non-focus per-class F1 drop `<=0.010`.

Selectivity and condition gates:

- clean frozen-cohort AUROC `>=0.65` and delta over control `>=+0.020`;
- every shifted AUROC `>=0.60`, none below control, and candidate clean-to-
  condition AUROC drop `<=0.10`;
- candidate class-1 precision is no lower than control in any shifted condition
  and improves by `>=0.010` in at least two of three;
- candidate class-1 recall is no more than `0.020` below control in any shifted
  condition;
- restricted false positives do not increase in any shifted condition and
  decrease in aggregate;
- class-1 F1 is nonnegative versus control in at least three of four conditions
  and no condition delta is below `-0.010`.

Spatial-mechanism gates:

- offset RMS is finite, nonzero, and below `0.20` normalized coordinates in
  every condition; valid interpolation mass is at least `0.95` on average;
- mean inter-group position RMS is at least `0.005` and no group is identical;
- clean deformed-position bbox-hit fraction exceeds the same-row uniform
  reference by at least `0.020`, and at least `55%` of rows have positive hit-
  fraction delta;
- mean outside-position distance to bbox decreases by at least `5%` versus the
  uniform reference;
- candidate block-2 attention-proxy bbox mass is no lower than control and
  improves by at least `0.010` on the frozen cohort;
- no shifted condition reverses both bbox-hit and outside-distance gains.

XAI gates:

- candidate stem and block-2 Grad-CAM foreground mass is not more than `0.05`
  below control on the required event set;
- object desaturation/blur remains more causal than far-background perturbation
  on average;
- every changed class-1 event, restricted-FP removal/creation, TP break, and FN
  rescue appears in the manifest and XAI/position-overlay pages;
- close, wide, partial, edge, tiny-object, dim, bright, and low-contrast cases
  are visually inspected, including whether deformed positions enter the target
  fruit rather than neighboring fruit/background.

## Permission And Stop Rules

- Passing every A1 gate authorizes a new prospective Stage-B protocol for at
  most ten total train-only epochs. It does not authorize validation, test, 30
  epochs, or current-best command promotion by itself.
- Failing any material gate closes this exact block-2 DAT route. Do not sweep
  groups, kernel, stride, offset range, position bias, padding/alignment,
  prefix policy, layer, layer count, proxy, loss, LR, augmentation, epoch, fold,
  seed, threshold, router, bbox supervision, or pretrained/stock variant.
- Validation and test remain inaccessible during A1. Test remains a final audit
  only after a later validation promotion.
- A1 cannot change current-best command/history/pipeline files. Preserve keeper
  and scratch hashes. Compact rejected artifacts only after journal/closure and
  retention verification.
