# TRKH 5-Class Bi-Level Routing Attention A1 Readiness Protocol - 2026-07-16

## Status

Precommitted before implementation. This document authorizes a default-off
implementation and one formal train-only preflight. Only a complete preflight
pass authorizes one exact five-epoch scratch control/candidate pair. It does not
authorize official validation or test access, a full 30-epoch train, a sweep,
or a current-best command update.

## Research Question

Can one CVPR-2023 BiFormer-style Bi-Level Routing Attention (BRA) block improve
class-1 precision by filtering query-specific irrelevant context before fine-
grained patch attention, while retaining true class-1 surface evidence?

This is not a stock BiFormer backbone. The CNN stem, patch projection, seven
prefix tokens, blocks 1 and 3-8, token pruning, bbox spatial fusion, pooling,
heads, and losses remain unchanged. Only Transformer block 2 is eligible
because it is the last block that sees the complete `16x16` patch grid before
the first pruning operation.

## Primary-Source Lock

User-supplied and secondary reports remain hypothesis sources only. The
accepted paper, official supplement, and official licensed repository define
the method and this protocol.

- Accepted paper: Zhu, Wang, Ke, Zhang, and Lau, "BiFormer: Vision Transformer
  With Bi-Level Routing Attention," CVPR 2023:
  `https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_BiFormer_Vision_Transformer_With_Bi-Level_Routing_Attention_CVPR_2023_paper.html`.
- Official supplement:
  `https://openaccess.thecvf.com/content/CVPR2023/supplemental/Zhu_BiFormer_Vision_Transformer_CVPR_2023_supplemental.pdf`.
- Local accepted paper:
  `D:/DataAI/external_sources/official/biformer-cvpr2023/BiFormer_CVPR2023_paper.pdf`.
- Local paper SHA-256:
  `d7415feb19a0818b39b9250b2a048aae2fb311079cfbabf14705362a8a353a5c`.
- Official repository: `https://github.com/rayleizhu/BiFormer` (581 stars and
  41 forks observed on 2026-07-16; popularity is context, not a gate).
- Locked branch/commit:
  `public_release` / `1697bbbeafb8680524898f1dcaac10defd0604be`.
- Locked tree: `313af0f24b31141cdde68fd775e75105278f8e52`.
- Reviewed NCHW BRA source: `ops/bra_nchw.py`, SHA-256
  `5b7b35316bb3b2d3f8494785bafc381200285ecb08086b95d5a61f4a5eab388e`.
- Reviewed regional gather source: `ops/torch/rrsda.py`, SHA-256
  `93d291582f7eb087a6091e247053e6425bbaacc3603383187c2be3669ff22ff4`.
- Reviewed NCHW architecture source: `models/biformer_stl_nchw.py`, SHA-256
  `97a253c8e06bac251be79ec0e298796169f31c49fb99028f64ffe18551f34456`.
- Reviewed training recipe: `configs/train_args.yaml`, SHA-256
  `e583aaeccb1949ef55b96b11859be2df07ca5311ca02a381a20b644e3940b0d2`.
- Official MIT license SHA-256:
  `63e8210e6bf3e8c032dc0c69b1d1d2e3ab72c14b02cabcc0dada2618bb188b97`.

The paper partitions a feature map into regions, averages projected queries and
keys per region, constructs a directed region-affinity graph, keeps a row-wise
top-k set, gathers all key/value tokens from those regions, and applies dense
token attention only inside the gathered set. The supplement reports that
increasing attended tokens can hurt accuracy and interprets explicit sparsity
as regularization against background distraction.

The official ImageNet recipe uses 300 epochs, batch 1024, strong MixUp/CutMix,
and a complete 13M-57M pyramid backbone. It is incompatible with the scratch
TRKH `<=30e` contract. The supplement also reports roughly 30%/40% training/
inference throughput loss against Swin-T due to routing and gather overhead.
Those facts require a one-block adaptation and a strict resource gate.

## Candidate Selection And No-Repeat Boundary

The same primary-source screen rejected the nearby candidates before code:

- CVPR-2023 SCConv targets generic spatial/channel redundancy. Its official
  repository exposed no license, and SRU/CRU overlaps closed spatial/channel
  gates, SRM, XCA, efficient convolutions, and tokenizer work.
- DilateFormer uses content-agnostic multi-scale dilated local attention in
  shallow stages. It overlaps closed local-window, MaxViT, FAA, and fixed
  sparse mixing, and its official repository exposed no license.
- ICCV-2023 LSKNet targets remote-sensing detection with large selective
  kernels. It overlaps the closed large-kernel/LKA, Selective Kernel,
  InceptionNeXt, Moga, StarNet, and OctConv families; the repository license
  metadata was not sufficiently clear for a direct implementation.

BRA remains genuinely distinct:

- DAT shares one image-conditioned deformed key/value grid across patch
  queries; BRA chooses a different discrete region set for each query region.
- ViG adds feature-space k-nearest-neighbor graph mixing after MHSA; BRA
  replaces patch-to-patch MHSA connectivity with a coarse-to-fine spatial
  routing graph.
- Soft-MoE routes tokens to experts; BRA routes each spatial query region to
  key/value regions and keeps no expert bank.
- Existing token pruning selects one image-level patch subset shared by later
  queries; BRA keeps the token grid and selects key/value regions per query.
- FAA uses fixed local and pooled routes; BRA routes globally from content.

No DAT/FAA/ViG/Soft-MoE combination, verifier, router, threshold, ensemble,
teacher, pretraining, global class-1 oversampling, test tuning, or raw-data
modification is allowed.

## Locked Inputs And Provenance

- Repository pre-protocol HEAD/upstream:
  `42a9e690ae3b86415d8058c3c022a5b0c048adc4`.
- Data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Frozen train-only declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`,
  SHA-256
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Declaration summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.
- Current scratch launcher/resolved-config SHAs:
  `dac9977b13249ffb71ffd74339d3c8980e504f2652788c45e677c5aeeca587fe` /
  `c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97`.
- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Current-best command/history SHAs:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
- DBB closure SHA-256:
  `c24367649e6786b3d05af9454691d610f6c01ea506c830da4a469c5973d31504`.

Reuse only `runs/yolof_cidt_fold0_trainonly_20260716`:

- fit: 7,372 rows, 6,452 source stems, class counts
  `[1561,432,1527,2017,1835]`;
- holdout: 1,843 rows, 1,612 source stems, class counts
  `[380,109,393,503,458]`;
- source overlap: zero;
- fit/holdout sample-index SHAs:
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d` /
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`;
- fold-summary SHA-256:
  `2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763`.

The generated `test` path mirrors holdout only for loader compatibility. Every
launcher must use `SkipFinalTest`; no official validation or test loader may be
constructed. Raw images and labels remain read-only.

## Locked Architecture And Causal Control

Feature name: `bi_level_routing_attention`, default off.

Shared candidate/control construction:

- block 2 only, before first token pruning;
- exact dense patch grid `16x16` with identity patch indices;
- embedding/heads/head dimension `256/8/32`;
- `S=4` regions per axis, exactly 16 regions of `4x4` patch tokens;
- one shared route graph across heads, as in the official NCHW source;
- region queries/keys are full-channel averages of projected patch Q/K;
- routing Q/K are detached exactly as in the official source;
- no forced self-region, positional route bias, soft routing, or route loss;
- token-attention scale is the official `qk_scale` override
  `head_dim**-0.5`, matching current TRKH rather than changing temperature;
- every patch query attends all seven prefix tokens plus gathered patch tokens;
- every prefix query retains standard global attention over all tokens;
- official local context enhancement is a shared depthwise `5x5` convolution
  over patch V, added before the common output projection; prefixes receive no
  LCE residual;
- current QKV/output projections, dropout, residual, FFN, and block ordering
  remain unchanged.

The sole causal difference is routing sparsity:

- matched control: `topk=16`, all regions, equivalent to full patch attention;
- candidate: `topk=4`, exactly 64 of 256 patch keys per patch query plus all
  seven prefixes.

Control and candidate have identical tensors and parameter count. The
`topk=16` path must use deterministic dense patch order so it can serve both as
the causal all-region control and an exact standard-MHSA parity oracle when LCE
is disabled.

For `return_attention=True`, return an exact dense `[B,8,T,T]` proxy. Prefix
rows contain global attention. Patch rows scatter their native prefix and
gathered-patch probabilities to original token columns, with exact zeros for
unrouted patch regions. Route indices, region affinities, selected/excluded
affinity margin, query-region coordinates, route distances, LCE norm, and
native sparse attention remain separately traceable. Standard and trace
forwards must execute the same output computation.

The module must reject pruned, non-identity, non-`16x16`, or incompatible
prefix layouts. It may not silently fall back to dense attention.

## Equation, Deployment, And Resource Preflight

Before any epoch, a committed and pushed implementation must pass all checks:

1. every locked source/input hash, official commit/tree, protocol hash,
   worktree scope, and fold contract matches;
2. an independent patch-only FP32 replay against official `nchwBRA` plus
   `regional_routing_attention_torch` matches route indices, sparse attention,
   output, input gradients, and remapped parameter gradients within `1e-6`;
3. with LCE disabled and `topk=16`, output, dense attention, input gradients,
   and QKV/projection gradients match current standard MHSA within `1e-6`;
4. separately constructed seed-42 topk-16/topk-4 models have bit-exact shared
   state, identical parameter inventory, finite state, and unchanged CPU/CUDA
   constructor RNG checkpoints;
5. Q, K, V, output projection, and LCE families receive finite nonzero
   gradients on one real fit batch; route indices remain explicitly detached;
6. FP32 and CUDA BF16 forward/backward tensors are finite. FP32/BF16 route
   mean Jaccard is at least `0.98`, exact route-set agreement is at least
   `0.95`, and all disagreements plus top-k boundary margins are reported;
7. standard and trace forwards have maximum logit error `<=1e-6` in FP32 and
   BF16, zero argmax mismatch, and identical native output tensors;
8. sparse/dense attention rows are finite, nonnegative, normalized within
   `1e-6`, and independent scatter replay error is `<=1e-6`;
9. candidate/control parameters are identical and candidate is at most current
   TRKH plus `7,000` parameters;
10. static batch-1 ONNX output has maximum logit error `<=1e-4`, matching
    argmax, and contains the expected TopK/Gather graph without Python fallback;
11. candidate median inference runtime is at most `1.35x` current MHSA,
    training runtime is at most `1.50x`, and peak allocated VRAM is at most
    `7.5 GiB` and `1.25x` current MHSA;
12. fold rows/classes/sources/hashes are exact and no validation/test path is
    opened.

## Pre-Training Train-Only Selectivity Gate

Before the five-epoch pair, load the keeper's common state into matched topk-16
and topk-4 models and audit all 1,843 source-disjoint holdout rows without
updating a parameter. Use clean, dim (`brightness=0.70`, `contrast=0.90`),
bright (`1.25`, `1.10`), and low-contrast (`1.00`, `0.65`) inputs.

YOLO bboxes are audit-only. Token centers define object membership. An object
query region is weighted by its count of object token centers. Far background
means outside the bbox expanded by one patch in each direction.

Every gate below must pass:

- clean object-query routed foreground fraction exceeds the all-region
  area-matched fraction by at least `0.020` on average;
- at least `55%` of clean rows have positive foreground-route gain;
- clean far-background routed fraction is at least `0.020` below the
  all-region fraction;
- every shifted condition retains positive mean foreground gain and does not
  increase far-background fraction above the all-region control;
- each image has at least four distinct route sets on average, mean pairwise
  route-set Jaccard is at most `0.85`, and at least `10%` of route edges are
  nonlocal beyond one neighboring region;
- mean clean-to-condition route-set Jaccard is at least `0.65` for every shift;
- topk-4 versus topk-16 mean absolute probability change is at least `1e-4`,
  while outputs and routing remain finite;
- route overlays for class-1 keeper TP, class-1 FN, and restricted
  `{0,2,4}->1` FP include close, wide, partial, edge, tiny, dim, bright, and
  low-contrast cases and pass explicit visual review.

Failure closes this BRA route before training. These gates test whether the
trained TRKH Q/K space can support query-adaptive foreground selection; they do
not claim that keeper-adapted logits are a promoted model.

## Locked Five-Epoch Train-Only Pair

Run exactly one topk-16 control and one topk-4 candidate from random
initialization, seed 42. Neither may resume or use pretrained weights. Both use
the generated fold and identical ordered occurrences, transforms, optimizer,
scheduler, losses, EMA, stopping, and checkpoint metric.

Shared model recipe:

- current `vit_registers`, image 256, conv-pool stem, embed 256, depth 8,
  heads 8, four registers;
- bbox spatial fusion, CNN feature fusion, fine-grained pooling, color/edge
  branch tokens, detail enhancement, and pruning at layers `2,5` with keep
  rates `0.85,0.65` remain enabled;
- every unrelated experiment remains disabled.

Shared training recipe:

- epochs `5`, batch `32`, accumulation `2`, AdamW LR `2.5e-4`, weight decay
  `0.05`, warmup `1`, cosine horizon `5`, min LR `1e-6`, clip `0.7`, patience
  `3`, EMA `0.995`;
- natural-frequency sampling; balanced/weighted sampling, repeats, and global
  class-1 oversampling disabled;
- `ldam_focal`, smoothing `0.02`, LDAM margin `0.3`, scale `18`, focal gamma
  `1.0`, focal mix `0.1`;
- pairwise-margin and metric losses remain `0.04/0.04` with the scratch-record
  boundary pairs/sources;
- scratch-record crop/pad, mild affine/color jitter, illumination
  normalization, background suppression, local exposure, obstacle, and
  horizontal flip remain identical;
- attention-view loss/drop, teacher focus/cache, distillation, manifests,
  sample weighting, MixUp/CutMix/copy-paste, thresholds, routers, TTA, and
  final test remain disabled.

Persist exact arguments/configs, initial-state manifest, occurrence hashes,
train/holdout metrics, checkpoints, resource telemetry, architecture traces,
native sparse attention, dense scatter, route graphs, and overlays. The
train-only holdout selects checkpoints and is not official validation.

## A1 Promotion Gates

Candidate is compared with the matched topk-16 control on all 1,843 holdout
rows. Every engineering, behavior, condition, routing, and XAI gate must pass.

Clean behavior, with precision prioritized:

- macro F1 delta `>=+0.003`;
- class-1 F1 delta `>=+0.015` and absolute class-1 F1 `>=0.60`;
- class-1 precision delta `>=+0.025`;
- class-1 recall delta `>=-0.010`;
- restricted `{0,2,4}->1` false positives decrease by at least four;
- corrections exceed harms;
- class-1 FN rescues are at least class-1 TP breaks;
- maximum nonfocus per-class F1 drop is `<=0.010`.

Condition safety:

- candidate class-1 precision is no lower in any shift and improves by at
  least `0.010` in two of three shifts;
- class-1 recall is no more than `0.020` below control in any shift;
- restricted FP never increases and decreases in aggregate;
- class-1 F1 is nonnegative in at least three of four conditions and no delta
  is below `-0.010`;
- macro F1 is nonnegative in at least three conditions and no delta is below
  `-0.010`.

Learned routing and XAI:

- all pre-training route gates still pass after training;
- clean foreground-route gain improves by at least `+0.010` over initialization
  and no shift reverses it;
- selected/excluded affinity margin is positive and routes remain query-
  adaptive rather than collapsing to one common set or only self-regions;
- candidate block-2 attention bbox mass is at least control plus `0.010` on
  the frozen class-1 TP/restricted-FP cohort;
- far-background blur/gray changes candidate class-1 probability no more than
  control, while object desaturation/blur remains more causal;
- stem and block-2 Grad-CAM foreground mass is not more than `0.05` below
  control;
- every changed decision, FP removal/creation, TP break, and FN rescue is in
  the event manifest and route/XAI pages;
- close, wide, partial, edge, tiny, dim, bright, low-contrast, neighboring-
  fruit, hand, leaf, shadow, and border cases receive visual review.

Behavior must come from standard deployment inference. Trace-only forward is
used solely for routing/XAI evidence, with parity already established.

## Permission And Stop Rules

- Passing all A1 gates authorizes a new prospective Stage-B protocol for at
  most ten total train-only epochs. It does not authorize validation, test,
  30 epochs, or command promotion by itself.
- Failing any material preflight/selectivity gate closes before training.
- Failing any pair gate closes block-2 BRA after the complete audit.
- Do not sweep `S`, top-k, LCE kernel, scale, detach policy, prefix policy,
  layer, layer count, route bias, soft routing, optimizer, LR, losses,
  augmentation, epoch count, fold, seed, threshold, router, bbox supervision,
  stock backbone, or pretrained variant.
- Current-best command/history/pipeline files remain unchanged unless a later
  candidate wins the locked official-validation promotion gate.
- Preserve keeper/current-command hashes. Compact rejected binaries only after
  journal/closure, exact manifests, and retention verification.
