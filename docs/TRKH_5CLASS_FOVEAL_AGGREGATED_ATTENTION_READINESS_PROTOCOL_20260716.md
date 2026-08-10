# TRKH 5-Class Foveal Aggregated Attention Readiness Protocol - 2026-07-16

## Status

Precommitted before implementation. This document authorizes one exact,
train-only, scratch-initialized control/candidate experiment. It does not
authorize access to the official validation or test splits, a full 30-epoch
train, a hyperparameter sweep, or a current-best command update.

## Research Question

Can one early TransNeXt-style Aggregated Attention block improve the scratch
TRKH representation of subtle mango surface and maturity boundaries by making
each patch jointly compete over exact local detail and pooled global fruit
context, while preserving class-1 true positives and reducing false positives?

The candidate is not a stock TransNeXt backbone. It changes only block 1 of the
current eight-block TRKH Transformer. Prefix tokens retain global attention;
patch queries use one normalized competition over a `3x3` local neighborhood
and a fixed `4x4` pooled patch grid. Blocks 2-8, the CNN stem, patch projection,
register and branch tokens, bbox spatial fusion, pruning, pooling, heads, and
all non-attention training settings are shared with the control.

## Primary-Source Lock

The method authority is the accepted CVPR 2024 paper and its official
Apache-2.0 implementation. User-provided or secondary research reports may
motivate a check but do not define this experiment.

- Paper: Shi, Ye, Zhang, and Li, "TransNeXt: Robust Foveal Visual Perception
  for Vision Transformers", CVPR 2024:
  `https://openaccess.thecvf.com/content/CVPR2024/html/Shi_TransNeXt_Robust_Foveal_Visual_Perception_for_Vision_Transformers_CVPR_2024_paper.html`.
- Local paper:
  `D:/DataAI/external_sources/papers/shi2024_transnext.pdf`.
- Paper SHA-256:
  `0d1a5d07b747eb7c02ed66a840f2f0242252697d883077325862509ad28e1c5e`.
- Official repository: `https://github.com/DaiShiResearch/TransNeXt`.
- Locked commit:
  `c8a99743b60ac94ac8d2bf66ffe164a440dcfe21`.
- Locked tree:
  `d1ce1c6cf62f5d0ebe1290dfe64b213328d91d6a`.
- Reviewed native attention source:
  `classification/attention_native.py`, SHA-256
  `f70bc20818c9b0456d6eaaf58669561f0ede8fa5aba3c3a557cb5094557d86fe`.
- Reviewed architecture source:
  `classification/transnext.py`, SHA-256
  `c71cccd5d64b7ba04506992ac2d19d29237d28d171c72665a25247a2d6a28bec`.
- Official `transnext_micro` recipe:
  `classification/configs/transnext_micro.py`, SHA-256
  `d898df1de6faf640ad1344ecbf6a60be98fcbce42e1cb8f27ac2bcb78dd0ded1`.
- Official license SHA-256:
  `68341bcf5aea46bf8f6c63f5c382e74dc960feaa31cd391e01a4d764d6121824`.

The official native implementation uses `nn.Unfold` for local keys/values,
adaptive pooling for global keys/values, length-scaled cosine attention,
continuous relative position bias, learnable local tokens, and one shared
softmax over local and pooled logits. The stock official classification recipe
uses 300 epochs, so importing a complete TransNeXt model would violate the
fixed scratch/`<=30e` contract. Only the bounded attention mechanism proceeds.
ConvGLU is explicitly excluded because locally enhanced FFN/LeFF is already a
closed TRKH route.

FasterViT/HAT was also checked against NVIDIA's official ICLR 2024 paper and
repository. Its smallest published model and long official recipe are not a
resource-matched replacement for the 7.25M-parameter, at-most-30-epoch TRKH,
so no FasterViT implementation is authorized by this protocol.

## Local No-Repeat Boundary

The following nearby routes are already closed and must not be reintroduced:

- GPSA/gated positional mixing, local-window mixers, VCA, XCA, Soft-MoE,
  dynamic graph mixing, MogaNet, CoAtNet, MobileViT, EdgeNeXt, stock Swin, and
  stock architecture replacement;
- LeFF/ConvGLU, persistent Conformer-style local/global coupling, shifted
  high-resolution bridges, local zoom, image tiling, and frequency experts;
- full-frame context, wider YOLO margins, `class_f+yolo_f` paired views,
  source-context fusion, bbox/objectness forcing, and background suppression
  as a new method claim;
- a verifier, router, threshold, ensemble, pretrained teacher, distillation,
  class-1 oversampling, or raw-data modification.

This candidate is distinct only in one bounded respect: every block-1 patch
query performs one dynamic softmax competition between its exact local
neighborhood and a compact global patch summary. It does not add another image
view, a second classifier, a post-hoc decision rule, or a ConvGLU branch.

## Locked Inputs And Provenance

- Repository pre-protocol HEAD:
  `9ee5f1737baba2c0eb9ecad6379a76f5555f511a`.
- Data YAML:
  `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Frozen train-only fold declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`,
  SHA-256
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Fold companion summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.
- Scratch-recipe launcher record:
  `runs/full_v8_yolof_randominit_30e_20260714_105524/launcher_args.json`,
  SHA-256
  `a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6`.
- Scratch-recipe resolved config SHA-256:
  `c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97`.
- Keeper checkpoint used only to define the frozen diagnostic cohort:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Current-best command SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`.
- Current-best history SHA-256:
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

Use only the `9,215` `condition=clean` rows in the fold declaration. Fold 0 is
the immutable holdout:

- fit: `7,372` object rows, `6,452` source stems, class counts
  `[1561,432,1527,2017,1835]`;
- holdout: `1,843` object rows, `1,612` source stems, class counts
  `[380,109,393,503,458]`;
- fit/holdout source-stem overlap: exactly zero;
- fit sample-index SHA-256:
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d`;
- holdout sample-index SHA-256:
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`.

A reproducible generated YOLO view may hardlink source images and labels into
`runs/yolof_cidt_fold0_trainonly_20260716`. Its `train` split contains only fit
sources. Its `val` split contains only holdout sources. Its generated `test`
path may mirror holdout solely for loader compatibility, but every launcher
must use `SkipFinalTest`; no test loader or test metric may be constructed.
Raw dataset files are read-only.

## Locked Candidate Architecture

The control uses the current standard block-1 `MultiHeadSelfAttention`. The
candidate replaces only that module with `FovealAggregatedAttention`:

- layer: block 1 only;
- input patch grid: exactly `16x16`, before any pruning;
- embedding/heads/head dimension: `256/8/32`;
- local window: `3x3`, stride one, zero-padded validity mask;
- pooled grid: fixed `4x4` after official-style `1x1 Conv -> GELU -> adaptive
  average pool -> LayerNorm`;
- patch-query attention: L2-normalized query/key, learnable query embedding,
  positive softplus temperature initialized to `1/0.24`, sequence-length log
  scaling, local relative bias, continuous pooled relative position bias, and
  one softmax over `9+16` positions;
- local aggregation: the official learnable-local-token and learnable-bias
  terms are retained;
- prefix queries: unchanged scaled dot-product global attention over every
  prefix and patch token; patch queries do not consume prefix values in block 1;
- blocks 2-8: current standard TRKH attention unchanged;
- ConvGLU and every other optional architecture experiment: disabled.

The candidate must retain public `qkv` and `proj` modules so checkpoint,
instrumentation, and shared-block paths remain explicit. Construction must
always instantiate the standard control attention first. Candidate-only
parameters are then created inside a restored RNG fork, and common `qkv/proj`
weights are copied from the standard module. Therefore every common state key
in separately constructed seed-42 control and candidate models must be
bit-exact, and candidate-only initialization may not perturb downstream shared
weights.

For `return_attention=True`, prefix rows expose exact global attention. Patch
local mass is scattered to its exact patch columns; each pooled-cell mass is
distributed uniformly over its corresponding `4x4` grid bin. The reconstructed
dense map must be finite, nonnegative, and sum to one per row in eval mode. It
is an audit proxy for pooled values, not a claim that the pooled `1x1` transform
is identical to raw patch values; exact-gradient Grad-CAM and perturbation
audits remain mandatory.

## Equation And Engineering Preflight

Before formal training, all checks below must pass from a committed and pushed
implementation:

1. every locked source and input SHA matches;
2. an independent patch-only FP32 replay matches the locked official native
   `AggregatedAttention` output and gradients within `1e-6` after exact state
   remapping;
3. common control/candidate state keys are bit-exact at seed 42, and all
   candidate-only parameters are finite;
4. candidate-only query, local, pooled, CPB, and output-projection parameters
   receive finite nonzero gradients on a real train batch;
5. FP32 and CUDA BF16 forward/backward are finite;
6. returned dense attention has the exact `[B,8,T,T]` shape, normalized rows,
   valid local support, and no patch-to-prefix mass;
7. candidate parameter count is at most control plus `100,000`;
8. static batch-1 ONNX export/replay has maximum absolute logit error
   `<=1e-4`;
9. median inference runtime is at most `1.35x` control and peak allocated VRAM
   is at most `7.5 GiB` and `1.25x` control;
10. the generated fold has exact row/class/source/hash counts and zero source
    overlap, and no official validation/test path is opened.

An engineering smoke may use at most 32 fit batches and one complete train-only
holdout evaluation only to verify execution and artifacts. It cannot select or
reject the method and must be deleted after its diagnostics are recorded.

## Locked Five-Epoch Train-Only Pair

Run exactly one control and one candidate from random initialization, seed 42.
Neither run may resume a checkpoint or use pretrained weights. Both use the
generated fold-0 YAML and identical ordered samples, transforms, optimizer,
scheduler, losses, EMA, and stopping rule.

Shared model recipe:

- current `vit_registers`, image size 256, conv-pool CNN stem, `embed_dim=256`,
  depth 8, heads 8, four registers;
- bbox spatial fusion, CNN feature fusion, fine-grained pooling, two branch
  tokens, detail patch enhancement, and token pruning at layers `2,5` with
  keep rates `0.85,0.65` remain enabled as in the scratch record;
- every unrelated experimental architecture flag remains disabled.

Shared training recipe:

- epochs `5`, batch `32`, gradient accumulation `2`, AdamW learning rate
  `2.5e-4`, weight decay `0.05`, warmup `1`, cosine horizon `5`, minimum LR
  `1e-6`, gradient clip `0.7`, patience `3`, EMA `0.995`;
- natural-frequency epoch sampling only; balanced epoch sampling, weighted
  sampling, rare-class repeat, and global class-1 oversampling are disabled;
- `ldam_focal` with label smoothing `0.02`, LDAM max margin `0.3`, scale `18`,
  focal gamma `1.0`, focal mix `0.1`;
- pairwise-margin loss `0.04` and metric-learning loss `0.04` retain the same
  boundary pairs and sources as the scratch record;
- the scratch record's ordinary image augmentation remains shared, including
  crop/pad, mild affine/color jitter, illumination normalization, foreground
  background suppression, local exposure, obstacle, and horizontal flip;
- attention-view loss/drop, teacher-focus loss, every teacher cache,
  distillation, target manifest, sample weighting, mixup/CutMix/copy-paste,
  threshold, router, TTA, and final test are disabled to isolate architecture
  and prevent a full-train teacher cache from leaking into the holdout.

Persist launch arguments, resolved configs, initial/common-state manifest,
ordered occurrence hashes, per-epoch train and holdout metrics, best and last
checkpoints, resource telemetry, and architecture traces. Checkpoint selection
uses the same declared train-only holdout metric for both roles and is not a
paper validation result.

## Frozen Selectivity Cohort

Use the fold-0 CIDT keeper decisions to define a model-independent cohort
before reading control/candidate outcomes:

- positive rows: target class 1 predicted class 1 by the locked keeper;
- hard-negative rows: target class in `{0,2,4}` predicted class 1 by the locked
  keeper.

For every role and condition, compute `margin = logit_1 - max(logit_0,
logit_2,logit_4)` and the AUROC separating positive from hard-negative rows.
The cohort, labels, and membership never change with candidate decisions.

Conditions use the existing locked train-only transforms:

- `clean`: brightness `1.00`, contrast `1.00`;
- `lighting_dim`: brightness `0.70`, contrast `0.90`;
- `lighting_bright`: brightness `1.25`, contrast `1.10`;
- `low_contrast`: brightness `1.00`, contrast `0.65`.

## A1 Promotion Gates

Every structural, equation, resource, training, behavior, condition, and XAI
gate must pass. Candidate is compared with the matched control best checkpoint
on all `1,843` holdout rows.

Clean behavior gates:

- macro F1 delta `>=+0.003`;
- class-1 F1 delta `>=+0.015` and candidate class-1 F1 `>=0.60`;
- class-1 precision delta `>=+0.020`;
- class-1 recall delta `>=-0.010`;
- restricted `{0,2,4}->1` false positives decrease by at least three;
- corrections exceed harms;
- class-1 false-negative rescues are at least class-1 true-positive breaks;
- maximum non-focus per-class F1 drop `<=0.010`.

Selectivity and condition gates:

- clean frozen-cohort margin AUROC is `>=0.65` and improves over control by
  `>=0.020`;
- every shifted-condition AUROC is `>=0.60`, none is below control, and the
  candidate clean-to-condition AUROC drop is at most `0.10`;
- candidate class-1 precision is no lower than control under any shifted
  condition and improves by at least `0.010` in at least two of three;
- candidate class-1 recall is no more than `0.020` below control under any
  shifted condition;
- restricted false positives do not increase in any shifted condition and
  decrease in aggregate;
- class-1 F1 is nonnegative versus control in at least three of four conditions
  and has no condition delta below `-0.010`.

Mechanism/XAI gates:

- local and pooled attention mass are both finite, nonzero, and noncollapsed;
- at least `95%` of patch queries allocate at least `0.05` mass to each route;
- candidate stem and block-1 Grad-CAM foreground mass is not more than `0.05`
  below control on the required event set;
- object desaturation/blur perturbation remains more causal than far-background
  perturbation on average;
- every candidate/control changed class-1 event, restricted-FP removal/
  creation, TP break, and FN rescue is represented in the audit manifest and
  XAI pages; representative close, wide, partial, edge, dim, bright, and
  low-contrast cases are visually inspected before a decision.

## Permission And Stop Rules

- Passing every A1 gate authorizes a new prospective Stage-B protocol for at
  most ten train-only epochs or one official-validation smoke. It does not
  automatically authorize test, 30 epochs, or command promotion.
- Failing any material gate closes this exact block-1 Aggregated Attention
  route. Do not sweep window size, pool size, CPB width, temperature, local
  tokens, prefix policy, insertion layer, number of FAA layers, loss, LR,
  dropout, augmentation, epochs, fold, seed, threshold, router, or pretrained
  variant.
- Official validation and test remain inaccessible during A1. Test remains a
  final audit only after a later validation promotion.
- A1 cannot change
  `docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt`,
  `scripts/run_trkh_current_best_full_pipeline.ps1`, or
  `docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt`.
- Preserve the keeper and random-scratch checkpoint hashes. Delete only an
  obsolete engineering smoke after its metrics, traces, and XAI conclusion are
  recorded and retention checks pass.
