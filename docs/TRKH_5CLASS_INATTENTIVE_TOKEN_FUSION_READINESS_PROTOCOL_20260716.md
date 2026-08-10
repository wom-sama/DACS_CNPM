# TRKH 5-Class Inattentive-Token Fusion A0 Readiness Protocol - 2026-07-16

## Status

Precommitted before implementation. This document authorizes one default-off,
parameter-free implementation and one formal train-only preflight. A complete
preflight pass is required before one exact five-epoch scratch control/candidate
pair. It does not authorize official validation or test access, a 30-epoch
train, a keep-rate/layer sweep, or a current-best command update.

## Research Question

Can an EViT-style summary of patches rejected by the existing TRKH selector
retain useful object/context evidence, especially for small or edge-touching
fruit, while improving class-1 false-positive control?

The causal control is the current hard top-k pruning path. The candidate keeps
the same first-prune spatial patches and adds only one attention-weighted
global context token. Because that token then participates in self-attention,
the later top-k set may evolve even though the selector formula is unchanged.
No raw image, annotation, split, augmentation, loss, head, keep-rate, or
selector-score formula changes.

## Primary-Source Lock

User-supplied and secondary research reports are hypothesis sources only. The
accepted paper and official licensed repository define the source mechanism.

- Accepted paper: Liang et al., "Not All Patches are What You Need: Expediting
  Vision Transformers via Token Reorganizations," ICLR 2022 Spotlight:
  `https://iclr.cc/virtual/2022/spotlight/6169`.
- Author paper page: `https://arxiv.org/abs/2202.07800`.
- Local paper:
  `D:/DataAI/external_sources/official/evit-iclr2022/paper_arxiv_2202.07800v2.pdf`.
- Paper SHA-256:
  `c6d98d2ac474608218014af73ef9a1e4f4eae5db496e6d038c5d8e451959f1a7`.
- Official repository: `https://github.com/youweiliang/evit` (200 stars and
  24 forks observed on 2026-07-16; popularity is context, not a correctness
  gate).
- Locked commit/tree:
  `97e58f610c51d4b74a070341739e41647dced32c` /
  `7d907e81f4a764972baa96a639a16164af6f08a6`.
- Reviewed source `evit.py`, SHA-256:
  `84b164a630d3c8fad2925813bb894e2a796608f029540f57b619a08f7b388b45`.
- Reviewed complement helper `helpers.py`, SHA-256:
  `f7298ec0744ec315852627391f32c7d1270ad49c8a6a0645214026bb45e528e4`.
- Official Apache-2.0 license SHA-256:
  `84f2795e9d1a3be7f3672fad8ae5bc391393ff9904d56e80667c2f90a320e860`.

The paper averages class-token attention over heads, preserves the top-k image
tokens, and computes one fused token

`x_fused = sum(i in dropped) attention_cls_i * x_i`.

The fused token is appended for subsequent Transformer layers. It introduces
no learned parameters and allows gradients to reach rejected tokens. The paper
also reports that fusion usually improves hard removal only slightly. Its main
scratch evidence uses ImageNet, 300 epochs, a cosine keep-rate warmup, and much
larger compute. These results do not establish effectiveness under TRKH's
`<=30e` contract and therefore cannot replace the gates below.

## No-Repeat Screen

The same primary-source screen did not select the nearby alternatives:

- TNT uses inner pixel-word Transformers and outer patch Transformers, but its
  fine-grained transfer evidence relies on long ImageNet pretraining and
  overlaps the rejected high-resolution/local-detail family.
- NesT uses nested local blocks and aggregation, overlapping rejected local
  window and high-resolution routes without directly addressing discarded
  context.
- PaCaViT and Evo-ViT add broader clustering or slow/fast token evolution;
  both change more than the exact unresolved hard-drop operation and have less
  direct causal isolation in the current architecture.
- Keeping more patches or disabling pruning is already rejected on the keeper:
  class-1 F1 fell from `0.6860` at `0.85,0.65` to `0.6514-0.6554`.

EViT-style fusion is distinct because it preserves one aggregate without
restoring every noisy background token. This protocol does not authorize
higher resolution, different keep rates, different prune layers, or a nearby
fusion-weight sweep.

## Locked Inputs And Provenance

- Repository pre-protocol HEAD/upstream:
  `4e7a0e4b5472341ba7245bf8f017c6c822e08f47`.
- Data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Current-best command/history SHAs:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
- BRA closure SHA-256:
  `ab74dc74d7d25b0d1062db14d12654c11e3febfd498a1a473d80d3a4625766e4`.

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

## Locked A0 Adaptation

Feature name: `inattentive_token_fusion`, default off.

Shared control/candidate behavior:

- current `16x16` patch grid and seven base prefix tokens;
- pruning after layers `2,5`, keep rates `0.85,0.65`;
- unchanged TRKH selection score: mean attention from the seven base prefixes
  plus the existing foreground prior;
- unchanged spatial sort and exact original `patch_indices`;
- unchanged CNN stem, patch/detail embeddings, bbox fusion, Transformer
  parameters, pooling, heads, losses, and deployment outputs.

Candidate-only behavior:

1. After a current TRKH prune block has completed, compute class-token
   attention averaged across heads for every spatial patch.
2. Gather the exact complement of the unchanged top-k set.
3. On the first prune, compute the official parameter-free weighted sum over
   dropped post-block patch tokens.
4. Insert the result after the seven base prefixes and before spatial patches.
   It is a global context token, has no positional index, and may participate
   in later self-attention.
5. At the second prune, fold the existing global context into the replacement
   context with its current class-token attention, then add the weighted sum of
   newly dropped spatial patches.
6. Exclude the context token from spatial priors, patch selection, local
   modules, fine-grained patch pooling, and public `patch_indices`. Expose it
   separately in features/trace and remove it from the legacy public token
   layout before downstream consumers.

This is a bounded TRKH adaptation, not a claim of stock EViT equivalence. EViT
reorganizes between MHSA and FFN and may treat a previous fused token as a
later selectable image token. TRKH's existing hard prune hook is after the
whole encoder block and requires exact spatial indices for later modules and
XAI. A0 therefore preserves the existing prune point and keeps one explicitly
non-spatial context token. Changing block ordering would confound the causal
hard-drop comparison and is forbidden here.

No normalization, learned scale, residual gate, context projection, dropout,
class-specific routing, bbox weighting of fusion, selector change, or context
pooling into the classifier is allowed. The candidate and control must have
identical parameter inventories and common state.

## Engineering Preflight

Before any epoch, a committed and pushed implementation must pass all checks:

1. source/input hashes, official commit/tree, protocol hash, fold contract,
   protected artifacts, and explicit worktree scope match;
2. the first-prune fusion tensor, dropped indices, weights, and gradients match
   an independent replay of the official weighted-sum equation within `1e-6`;
3. fusion-off reproduces the current model logits, traces, selected indices,
   and public feature shapes exactly;
4. fusion-on and fusion-off select bit-exact spatial indices at the first
   prune; at the second prune the unchanged selector must retain mean Jaccard
   `>=0.98`, every changed row must be reported, and every selected/dropped set
   must be a disjoint complete partition;
5. context has shape `[B,1,256]`, no spatial sentinel is introduced, public
   patch count equals `patch_indices` count, and the legacy seven-prefix layout
   is restored in exported public tokens;
6. every context attention weight is finite/nonnegative, its reported mass
   equals the independent sum within `1e-6`, and context gradients reach at
   least one dropped token at each prune layer;
7. separately constructed seed-42 control/candidate models have identical
   state dictionaries, parameter counts, finite state, and unchanged CPU/CUDA
   constructor RNG checkpoints;
8. standard and trace forwards of each arm have maximum logit error `<=1e-6`,
   zero argmax mismatch, and identical selected/dropped indices in FP32 and
   CUDA BF16;
9. FP32 and BF16 forward/backward tensors are finite, and BF16 versus FP32
   selected-index mean Jaccard is at least `0.98`;
10. static batch-1 ONNX output has maximum logit error `<=1e-4`, matching
    argmax, and retains dynamic gather/reduction operators without Python
    fallback;
11. candidate median inference/training runtime is at most `1.05x/1.10x`
    control; peak allocated VRAM is at most `7.5 GiB` and `1.10x` control;
12. all 1,843 audit rows come only from the locked holdout and no official
    validation/test path is opened.

## Pre-Training Train-Only Gate

Load the keeper's common state into control and candidate and audit all 1,843
source-disjoint holdout rows without updating a parameter. Use clean, dim
(`brightness=0.70`, `contrast=0.90`), bright (`1.25`, `1.10`), and
low-contrast (`1.00`, `0.65`) inputs. Bboxes are audit-only.

The locked `tiny/edge-object cohort` is defined before model inference from
clean transformed holdout metadata only. A row is `tiny` when its normalized
bbox area is at or below the 25th percentile over all 1,843 valid clean
holdout bboxes, computed with the linear quantile method. A row is `edge` when
`min(x_min, y_min, 1-x_max, 1-y_max) <= 1/16`; negative gaps remain edge rows.
The cohort is the union of those predicates. The auditor must export the
resolved area threshold, all member sample indices, predicate counts, and the
ordered cohort-index SHA-256 before scoring any model output. Rows with a
missing or nonfinite bbox are hard audit failures rather than silently omitted.

Every gate below must pass:

- first-prune spatial indices are identical for control/candidate on every row
  and condition; second-prune mean Jaccard is at least `0.98` in every
  condition and every changed row is exported;
- clean candidate-control mean absolute probability change is at least `1e-5`
  but at most `0.05`, with zero nonfinite outputs;
- clean macro F1, class-1 F1, and class-1 precision deltas are no worse than
  `-0.0025`, `-0.0050`, and `-0.0050`; class-1 TP retention is at least `0.98`;
- no shifted condition loses more than `0.010` class-1 F1 or `0.010`
  class-1 precision versus its matched control;
- at least `95%` of rows have positive finite fused attention mass at both
  pruning stages;
- at least `20%` of the locked tiny/edge-object cohort has nonzero dropped
  object-attention mass, proving that the context can preserve object evidence
  that hard removal loses;
- attention-weighted object fraction in the fused context exceeds the raw
  dropped-object area fraction by at least `0.010` on clean data, while at
  least `20%` of total clean context mass remains outside the bbox, proving the
  token is neither an unweighted background average nor a duplicate object
  crop;
- clean-to-shift context cosine similarity is at least `0.65` for every shift,
  and the candidate does not increase restricted `{0,2,4}->1` false positives
  by more than two under any condition;
- context/dropped-patch overlays cover class-1 TP/FN and restricted FP cases,
  including close, wide, partial, tiny, edge, dim, bright, and low-contrast
  examples, and pass explicit visual review.

Failure closes A0 before training. These permissive behavior thresholds do not
promote a counterfactual keeper; they only verify that the parameter-free path
is active, spatially valid, and not already unsafe before adaptation.

This cohort definition was added prospectively after the default-off runtime
path and unit test existed, but before the formal auditor, any full-holdout
output, visual page, or gate result existed. It resolves an underspecified
term; it does not change the fusion equation or any numeric acceptance gate.

## Sole Five-Epoch Train-Only Pair

Only a complete preflight pass authorizes one control/fusion pair from random
initialization, seed 42, on the exact locked `7372/1843` fold. Neither model may
resume or use pretrained weights. Both must receive identical ordered samples,
transforms, optimizer/scheduler steps, losses, EMA handling, and stopping.

Shared model recipe is the current scratch record: image 256, conv-pool stem,
embed/depth/heads `256/8/8`, four registers, color/edge branch tokens, detail
enhancement, bbox/CNN/fine-grained fusion, and pruning `2,5` at `0.85,0.65`.
Every unrelated experimental module remains disabled.

Shared training recipe: five epochs, batch 32, accumulation 2, AdamW
`2.5e-4`, weight decay `0.05`, warmup 1, cosine horizon 5, min LR `1e-6`, clip
`0.7`, patience 3, EMA `0.995`, and natural-frequency sampling. `ldam_focal`,
label smoothing `0.02`, LDAM margin/scale `0.3/18`, focal gamma/mix `1.0/0.1`,
and pairwise-margin/metric-learning losses `0.04/0.04` retain the same pairs
and sources as the scratch record. Crop/pad, mild affine/color jitter,
illumination normalization, background suppression, local exposure, obstacle,
and horizontal flip remain shared. Attention-view loss/drop, teacher focus or
cache, distillation, manifests, sample weighting, MixUp/CutMix/copy-paste,
thresholds, routers, TTA, and final test are disabled to isolate the
architecture and prevent holdout leakage. There is no EViT keep-rate warmup in
A0 because both arms must preserve the current TRKH pruning control.

The candidate advances only if all conditions pass on the train-only holdout:

- macro F1 delta `>=0.000`;
- class-1 F1 delta `>=+0.005`;
- class-1 precision delta `>=+0.015`;
- class-1 TP count is no more than two below control;
- restricted `{0,2,4}->1` false positives fall by at least four and `10%`;
- corrections exceed harms, with no non-class-1 class losing more than
  `0.015` F1;
- mean class-1 precision across clean/dim/bright/low-contrast improves by at
  least `0.010`, and no condition loses more than `0.010` class-1 F1;
- tiny/edge cohort macro and class-1 F1 do not regress, context selectivity
  remains valid, and matched XAI review passes.

Failure closes this exact route and forbids nearby scales, normalization,
keep-rate, layer, or selector sweeps. Passing authorizes a separately
precommitted official-validation gate; it does not itself update the current
best commands.

## Stop Rules

- No raw-data modification, official validation/test access, pretrained
  teacher, ensemble, calibration, post-hoc threshold, or global class-1
  oversampling.
- No epoch before every engineering and pre-training gate passes.
- No second A0 run, nearby hyperparameter sweep, or retrospective threshold
  change.
- No current-best command/history update unless a later locked official
  validation comparison replaces the keeper.
- Preserve summaries/manifests/hashes/XAI for a rejected route, remove only
  reproducible bulky payloads through a verified cleanup manifest, rerun
  retention, update journal/TODO/skill, then continue the next non-repeated
  primary-source screen.
