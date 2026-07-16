# TRKH 5-Class Cropr Token Selector A0 Readiness Protocol - 2026-07-16

## Status

Precommitted before implementation. This document authorizes one default-off
Cropr-style scorer implementation, one engineering preflight, and, only after a
complete preflight pass, one exact five-epoch scratch control/candidate pair on
the locked train-only fold. It does not authorize official validation or test
access, a full train, a selector/keep-rate/layer sweep, or a current-best
command update.

## Research Question

Can task-supervised token relevance improve the precision of TRKH class 1 more
reliably than its current heuristic mixture of prefix attention and a fixed
foreground prior?

TRKH currently prunes after blocks 2 and 5. Its score averages attention from
seven prefix tokens to the spatial patches and adds the existing foreground
prior. That score is only indirectly supervised by the final classifier.
Cropr instead trains a learnable query through an auxiliary classification
head, then uses the query-patch score to retain task-relevant patches.

The causal control and candidate both own and train the same Cropr modules and
receive the same auxiliary losses. The control continues to route patches with
the native TRKH score. The candidate changes only the routing source to the
learned Cropr score. This prevents a gain from being attributed merely to extra
parameters or auxiliary supervision.

## Primary-Source Lock

User-supplied and secondary reports are hypothesis sources only. The accepted
paper and official licensed repository define the method.

- Accepted paper: Bergner, Lippert, and Mahendran, "Token Cropr: Faster ViTs
  for Quite a Few Tasks," CVPR 2025, pages 9740-9750:
  `https://openaccess.thecvf.com/content/CVPR2025/html/Bergner_Token_Cropr_Faster_ViTs_for_Quite_a_Few_Tasks_CVPR_2025_paper.html`.
- Official paper PDF:
  `https://openaccess.thecvf.com/content/CVPR2025/papers/Bergner_Token_Cropr_Faster_ViTs_for_Quite_a_Few_Tasks_CVPR_2025_paper.pdf`.
- Local paper:
  `D:/DataAI/external_sources/official/cropr-cvpr2025-paper.pdf`.
- Paper SHA-256:
  `7525a21f05f1ddd982fea1d7d5777b1e87175aeb52d60729eedcb0e448f1a97d`.
- Official repository: `https://github.com/benbergner/cropr` (29 stars and two
  forks observed on 2026-07-16; acceptance and source provenance are the
  authority, while popularity is supporting context only).
- Locked commit/tree:
  `fa259e9030f5fddf4721ac75cdd18561524de6f9` /
  `4a83993890026c85cfd81b559b08c341fd86851d`.
- Reviewed classification scorer `cls/cropr.py`, SHA-256:
  `3f47ab92dc2509eafcf3a5e247b317637daf5d6fae177a20ba6de83b563402bd`.
- Reviewed integration `cls/vision_transformer.py`, SHA-256:
  `6ba20db35c770f95d9fcbd57c0ea2f7c57be1ae3a0407097f452bb316c176355`.
- Reviewed loss loop `cls/engine.py`, SHA-256:
  `e23c80bc52aeb83e1f9494db05f5f57d0301a28ce7b097b474d91abf6d3aef42`.
- Reviewed classification recipe `cls/CLASSIFICATION.md`, SHA-256:
  `7e160dc61a1a611df3a512902d211214eb2a0d161313996a4fcb8cca29e5fb1b`.
- Official MIT license SHA-256:
  `49bae98540619fac9cf9546aee90df20d77ff754f2fddfb38059fe404d9b2596`.

For one query and one head, the official classification scorer computes

`s = q X^T`,

`a = softmax(s / sqrt(D))`,

`z = a X`,

then optionally applies a residual MLP to `z` and an auxiliary classification
head. The source detaches `X` before this training path so the auxiliary loss
trains the scorer without back-propagating into the backbone. At inference the
aggregation, MLP, and head are omitted; only `s` and Top-K remain.

The paper's classification evidence is conservative for this project. It
fine-tunes MAE/EVA-02 pretrained ViT-L/H models, normally for 50 epochs, and
reports a small accuracy cost relative to unpruned models. It also identifies
smaller models and lower-resolution inputs as harder cases. It therefore
supports the mechanism, not a claim that scratch TRKH will improve in five or
30 epochs.

## No-Repeat Screen

This A0 is not a repeat of the following closed paths:

- `AdaptivePartTokenLearner` learns four foreground-biased part queries but
  only produces a residual classifier logit; it never controls the backbone's
  token set. Its bbox-prior probe was rejected.
- Micro-detail Top-K, TopK reassessment, layer-token fusion, TransFG/FFVT,
  TokenLearner, and PDiscoFormer variants changed heads or pooled descriptors;
  they did not train the actual pruning score with a detached task head.
- Native bbox/foreground forcing, higher keep rates, and no-prune runs changed
  priors or token quantity without learning task relevance and were rejected.
- EViT inattentive-token fusion retained dropped context after the existing
  selector; it did not replace the selector and is now closed.
- DynamicViT/SPViT-style differentiable masks require broader sparsity targets,
  distillation, or latency objectives and would confound this narrower first
  test of task-supervised routing.

Last Layer Fusion is forbidden because TRKH is a classification model and the
nearby dropped-token recovery family is closed. A0 cannot change keep rates,
prune layers, image resolution, native foreground weight, token count, or any
classifier/pooling component.

## Locked Inputs And Provenance

- Repository pre-protocol HEAD/upstream:
  `98e96b5ecf9caab545d52ee4056d6266ced63b87`.
- Data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Declaration CSV SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Current-best command/history SHAs:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
- EViT A0 closure SHA-256:
  `de929023c06781a5104edb72601cffd9595d5b2ca8efed001445a44e23e92a71`.

Reuse only `runs/yolof_cidt_fold0_trainonly_20260716`:

- fit: 7,372 rows, 6,452 source stems, class counts
  `[1561,432,1527,2017,1835]`;
- holdout: 1,843 rows, 1,612 source stems, class counts
  `[380,109,393,503,458]`;
- source overlap: zero;
- fit/holdout sample-index SHAs:
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d` /
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`;
- fold data/summary SHAs:
  `4195ba89995d4eee483cae31626717381df910f56657086e7116322ca971388e` /
  `2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763`.

The generated `test` path mirrors holdout only for loader compatibility. Every
launcher must use `SkipFinalTest`; no official validation or test loader may be
constructed. Raw images and labels remain read-only.

The keeper may be used only for engineering parity and resource checks. It was
trained on the complete original train split, so it is forbidden as a fitted
backbone for a source-disjoint selector-quality claim.

## Locked A0 Adaptation

Feature name: `cropr_token_selector`, default off.

Create one scorer at each existing prune point, after blocks 2 and 5. Each
scorer has:

- embedding dimension `256`;
- one learned query, one head;
- no query/key/value/output projection;
- no pre-attention normalization;
- residual MLP ratio `4`, GELU, and no dropout;
- LayerNorm plus a five-class linear auxiliary head;
- official query initialization with standard deviation `D^-0.5`;
- no LLF, score temperature, learned scale, bbox loss, or sparsity loss.

The scorer receives only the decision-eligible spatial patch tokens. TRKH has
seven always-retained prefix tokens rather than Cropr's single CLS token.
Allowing those prefixes into the auxiliary aggregation would create a shortcut
through tokens that can never be pruned. Excluding them is an explicit bounded
adaptation; the query-patch equation, detached auxiliary path, and inference
scorer remain the official mechanism.

At training time, for each prune layer:

1. detach the post-block spatial patches before the Cropr aggregation path;
2. compute raw query-patch scores and scaled softmax aggregation exactly as
   above;
3. apply the official residual MLP and auxiliary classification head;
4. add one natural-frequency cross-entropy loss per scorer with weight `1.0`;
5. report each auxiliary loss, macro/class F1, query gradient, head gradient,
   score entropy, and selected-set statistics.

The main loss still back-propagates through the selected spatial patches. The
Top-K decision is discrete, so scorer parameters receive their learning signal
only from the detached auxiliary path.

Both causal roles instantiate and train bit-identical scorer modules. Their
only difference is:

- control routing score:
  `normalize(mean_prefix_attention) + 0.35 * normalize(foreground_prior)`;
- candidate routing score:
  `normalize(cropr_raw_score) + 0.35 * normalize(foreground_prior)`.

Both use the unchanged target counts `218` then `167`, spatially sort retained
original indices, preserve all seven prefixes, and expose the same public token
layout. Candidate trace additionally reports raw/scaled Cropr scores,
attention, auxiliary logits, score entropy, native-score comparison, and exact
kept/dropped indices. Evaluation and export execute only the query-patch scorer
and Top-K; auxiliary aggregation/head tensors must be absent from the deployed
graph.

No auxiliary class weighting, label smoothing, pair-specific scorer, scorer
warmup, stochastic sampling, threshold, router, post-hoc correction, or
selector interpolation is allowed.

## Engineering Preflight

Before any data epoch, a committed and pushed implementation must pass every
check below:

1. source/input/protocol/fold/protected hashes and explicit worktree scope
   match, and official commit/tree/license metadata are exact;
2. native scorer output, raw scores, scaled attention, residual MLP output,
   auxiliary logits, selected indices, input gradients, and every scorer
   parameter gradient match an independent replay of the locked official code
   within `1e-6` in FP32;
3. a separate direct equation replay matches scores, softmax aggregation, and
   all parameter gradients within `1e-6`, preventing common-code agreement
   from being the only oracle;
4. the detached isolated auxiliary loss gives finite nonzero gradients to the
   query, MLP, and head, and exactly zero gradient to its input patch tensor;
5. a fixed synthetic task in which the target-relevant token is known reduces
   auxiliary CE by at least `50%`, reaches at least `95%` relevant-token Top-K
   hit rate, and does not use any repository image or label;
6. scorer-disabled construction is bit-exact with the current architecture's
   logits, traces, selected indices, parameter names, and public shapes;
7. separately constructed seed-42 control/candidate models have identical
   state dictionaries, parameter counts, finite state, and unchanged CPU/CUDA
   constructor RNG checkpoints; the routing boolean is not stored in state;
8. control routing is bit-exact to current native pruning even while Cropr
   auxiliary logits are collected; candidate partitions are complete,
   disjoint, target-sized, and spatially sorted at both layers;
9. standard and trace forwards have maximum logit error `<=1e-6`, zero argmax
   mismatch, and identical candidate selected indices in FP32 and CUDA BF16;
10. FP32/BF16 forward and backward are finite, every scorer parameter receives
    a finite nonzero gradient, and BF16/FP32 candidate selected-index mean
    Jaccard is at least `0.98`;
11. static batch-1 ONNX output has maximum logit error `<=1e-4`, matching
    argmax, dynamic TopK/Gather operators, and no auxiliary MLP/head nodes;
12. candidate median inference/training runtime is at most `1.08x/1.20x`
    control; peak allocated VRAM is at most `7.5 GiB` and `1.15x` control;
13. architecture traces cover one fit example per class and prove the exact
    `256 -> 218 -> 167` spatial sequence with no prefix/index corruption;
14. the formal tool refuses overwrite, official validation/test paths,
    mismatched/unpushed HEAD, a dirty tracked worktree, and any pair before the
    preflight summary grants permission.

Any material failure closes A0 before training. A correction after complete
artifacts requires a preserved failure manifest, a new output directory, and
byte-identical replay of unaffected outputs.

## Sole Five-Epoch Train-Only Pair

Only a complete engineering preflight authorizes one control/candidate pair
from random initialization, seed 42, on the exact locked `7372/1843` fold.
Neither role may resume or use pretrained weights. Both receive identical
ordered samples, transforms, optimizer/scheduler steps, losses, EMA handling,
and stopping. The control runs first; the candidate must replay the control's
per-epoch sample-occurrence hashes exactly.

Shared model recipe is the current scratch record: image 256, conv-pool stem,
embed/depth/heads `256/8/8`, four registers, color/edge branch tokens,
detail enhancement, bbox/CNN/fine-grained fusion, and pruning `2,5` at
`0.85,0.65`. Every unrelated experimental module remains disabled.

Shared training recipe: five epochs, batch 32, accumulation 2, AdamW
`2.5e-4`, weight decay `0.05`, warmup 1, cosine horizon 5, min LR `1e-6`, clip
`0.7`, patience 3, EMA `0.995`, and natural-frequency sampling. Main loss is
the unchanged `ldam_focal` recipe with label smoothing `0.02`, LDAM
margin/scale `0.3/18`, focal gamma/mix `1.0/0.1`, and pairwise-margin/
metric-learning losses `0.04/0.04`. Each role also sums the two locked Cropr
natural-CE losses at weight `1.0`.

Crop/pad, mild affine/color jitter, illumination normalization, background
suppression, local exposure, obstacle, and horizontal flip remain shared.
Attention-view loss/drop, teacher focus/cache, distillation, sample/target
manifests, paired views, MixUp/CutMix/copy-paste, thresholds, routers, TTA, and
final test are disabled.

## Pair Audit And Promotion Gate

Evaluate both last checkpoints through the standard deployment forward on all
1,843 train-only holdout rows under clean, dim (`0.70/0.90` brightness/
contrast), bright (`1.25/1.10`), and low-contrast (`1.00/0.65`) conditions.
The last checkpoint is locked before holdout metrics; holdout cannot select an
epoch. Independently replay every prediction, metric, confusion matrix,
selected set, and Cropr trace.

The candidate advances only if every condition below passes:

- clean macro F1 delta `>=0.000`;
- clean class-1 F1 delta `>=+0.015`;
- clean class-1 precision delta `>=+0.025`;
- class-1 TP count is no more than two below control and recall delta is no
  worse than `-0.010`;
- restricted `{0,2,4}->1` false positives fall by at least four and `10%`;
- corrections exceed harms, and no non-class-1 class loses more than `0.015`
  F1;
- mean class-1 precision across all four conditions improves by at least
  `0.015`, and no condition loses more than `0.010` class-1 F1 or `0.015`
  macro F1;
- block-2 and block-5 Cropr auxiliary macro F1 are each at least `0.60`,
  block-5 auxiliary class-1 F1 is at least `0.40`, and neither scorer has
  collapsed score variance or normalized entropy above `0.995` on more than
  `5%` of clean rows;
- candidate clean Cropr softmax mass inside transformed bboxes exceeds the
  matched native attention mass by at least `0.020` at one prune layer and is
  no worse than `-0.010` at the other; selected object-patch recall is no worse
  than `-0.010` at either layer;
- clean-to-shift selected-set Jaccard is at least `0.65` for every condition,
  and no shift reverses the foreground-mass gain by more than `0.020`;
- an auxiliary class-1 margin separating true class 1 from restricted hard
  negatives has AUROC at least `0.65` on clean holdout and at least `0.58` in
  every shifted condition;
- standard-forward perturbation audit shows object suppression changes class-1
  margin more than matched far-background suppression, with no increase in
  restricted false positives under either perturbation;
- all changed decisions, class-1 TP breaks/rescues, restricted FP
  removals/additions, close/wide/partial/tiny/edge cases, and every condition
  are represented in scorer overlays plus stem/block-2/block-5 XAI, and a
  separate hash-checked visual review passes.

Failure closes this exact route. Passing authorizes a separately precommitted
official-validation comparison; it does not itself update current-best files.

## Stop Rules

- No raw-data modification, official validation/test access, pretrained
  teacher, ensemble, calibration, post-hoc threshold, or class-1 oversampling.
- No epoch before every engineering gate passes.
- No second A0 pair, query/head/projection/normalization/aux-weight/keep-rate/
  layer/loss/optimizer/epoch/seed/condition sweep, LLF, or combination with a
  closed route.
- No current-best command/history update unless a later locked official
  validation comparison replaces the keeper.
- Preserve summaries, manifests, hashes, predictions, and XAI for a rejected
  route. Remove only reproducible bulky payloads through a verified cleanup
  manifest, rerun retention, update journal/TODO/skill, and then continue the
  next non-repeated primary-source screen.
