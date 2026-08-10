# TRKH 5-Class Attention-MaxSep Prototype A0 Protocol - 2026-07-21

## Question and decision boundary

Can attention-aware, class-specific local prototypes learn lesion or ripeness
evidence that separates true class 1 mangoes from the keeper's restricted
`0/2/4 -> 1` false positives, while preserving class-1 recall?

This is a prospective train-only information gate. It may extract frozen
features from the locked keeper and train isolated heads. It must not read
validation or test, modify raw data, edit the production model/trainer, run an
image-model smoke/probe/full train, or update the current-best commands.

All architecture, optimization, descriptor, threshold, control, robustness,
and XAI choices below are immutable once this file is committed. A clean-gate
failure closes the exact family without a nearby sweep.

## Primary sources and provenance

The selected equation is from Krishna Kanth Nakka and Mathieu Salzmann,
*Towards Robust Fine-grained Recognition by Maximal Separation of
Discriminative Features*, ACCV 2020:

- accepted paper:
  `https://openaccess.thecvf.com/content/ACCV2020/html/Nakka_Towards_Robust_Fine-grained_Recognition_by_Maximal_Separation_of_Discriminative_Features_ACCV_2020_paper.html`;
- paper SHA-256:
  `68a9c61c166762e42042968673de75fbfd0b2e930a4f492818a8fc199c96b4ff`;
- supplement SHA-256:
  `4e360302ed62a6f90b70685a464ac4c393bfa7e97a36db5fbf8590996d5f8f75`;
- authors' repository commit/tree:
  `20939384e02e7941790e7a8fa4498ae9fd0fe431` /
  `5d26bc4ea1b1a40dc288a841fedadde2093e949b`;
- authors' README/model/pooling/train/config SHAs:
  `77a106d5...8ff7ce8`, `c69df11d...cde6fa8`,
  `b2430fbb...dd6ab`, `f90995c5...9cd8d`, and
  `fe92bfb2...741ff`.

The authors' repository has no software license. It is pinned only as
provenance and must never be imported, copied, translated, or adapted. The
implementation must be independent from paper equations and checked by a
separately written NumPy/FP64 oracle.

The permissive architecture reference is Chaofan Chen et al., *This Looks Like
That: Deep Learning for Interpretable Image Recognition*, NeurIPS 2019:

- paper SHA-256:
  `d0ba7d150cfd179d2c7d1ab5a7f0e9b100bc4c70f9473d7b272ce757bb3b7ee3`;
- authors' MIT repository commit/tree:
  `81bf2b70cb60e4f36e25e8be386eb616b7459321` /
  `b6275c576911838e5dac07a6d481c1e79872ae01`;
- license/model/train SHAs:
  `f57a109e...2aaa3`, `448b24ba...eb23b`, and
  `6b183629...3fe3`.

No external source module is imported by the auditor.

## No-repeat screen

Three plausible alternatives are rejected before code:

- Wang et al.'s DFL-CNN, CVPR 2018, paper SHA
  `7108553d...fea5c`, uses positive class-grouped `1x1` filters followed by
  spatial GMP. The paper initializes those filters from ImageNet-pretrained
  VGG features at 448 px. The available community implementation reports 120
  epochs on four Titan V GPUs and is not an authors' implementation. Its
  positive-only local-map mechanism substantially overlaps the closed
  WILDCAT, patch-MIL, sparse class-evidence, and part-prototype families and
  offers no explicit false-positive separation term.
- Sun et al.'s MAMC, ECCV 2018, paper SHA `add72cf6...1c68c`, combines OSME
  attention with N-pair constraints. Its released recipe is 448 px, 60 epochs,
  and pretrained ResNet/VGG. The unique objective is still a multi-attention
  metric-learning instance of the already closed SupCon, attention-diversity,
  and multi-granularity contrastive families.
- Rao et al.'s CAL, ICCV 2021, remains rejected by the existing no-repeat lock.
  Its counterfactual-logit term is bundled with closed BAP, crop/drop, center,
  pretrained, and long-schedule mechanisms and does not directly supervise a
  class-1 surface prototype.

The selected MaxSep equation is materially different from the previous frozen
patch-prototype histogram diagnostic. It jointly learns a local projector,
class-specific prototypes, an attention branch, own-class attraction, and
wrong-class separation. It is also different from pooled prototypes,
hyperspherical supports, WILDCAT maps, sparse evidence maps, and covariance
readouts. The A0 controls below must prove that this distinction is causal.

## Locked inputs and isolation

- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher/config SHAs:
  `908a05cf...9b7eff` / `e9c4f489...54674`.
- Dataset descriptor:
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, SHA
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- CIDT summary/prediction SHAs:
  `d4891edf...d7ad` / `2e099375...b21c`.
- Current command/history SHAs, which must remain unchanged:
  `36b9aa1a...0faf` / `39bd2879...8f53`.

Use all 9,215 clean `yolo_f/train` object rows with natural class counts
`[1941,541,1920,2520,2293]`. The immutable source-disjoint five-fold holdouts
contain `[1843,1830,1828,1851,1863]` rows and have ordered train-index SHA
`d41d14b04dfb70b60b5058f30d1c4cfe0189aef3b0efc0a22741f5672c1473c8`.
No source stem may occur in both fit and held rows.

The decision cohort is the existing 763-row train-only CIDT cohort:

- 528 keeper class-1 true positives;
- 13 keeper class-1 false negatives;
- 222 restricted `0/2/4 -> 1` false positives, with target counts
  `{0:158, 2:54, 4:10}`;
- fold counts `145/160/152/155/151`;
- ordered cohort-index SHA
  `59d2146617f2642f99c081f49a9aa5812eae4206b27887a1fa5476f6f61c85b2`.

The keeper runs once in deterministic evaluation mode at its native 256 px
input. Capture the patch tokens after transformer block index 1, remove the
seven prefix tokens, and reshape the remaining tensor to `[B,256,16,16]`.
The hook must leave ordinary probabilities and keeper state unchanged. Padding
is excluded using the downsampled image-valid mask. Bboxes are recorded only
for controls and explanation statistics; they never gate candidate tokens.

Requested/effective Windows workers are locked to `4/4`, with pin memory and
persistent workers only when the existing opt-in environment flags make them
effective. Cache clean block-2 features as temporary float16. The cache must be
deleted in `finally`; retained artifacts must not include full-train features.

## Independent model equation

For every role, project frozen local feature `X` through two `1x1` layers:

`Z = sigmoid(W2 relu(W1 X))`, with widths `256 -> 64 -> 32`.

There are five prototypes per class, 25 total, each in `[0,1]^32`. Prototype
distance and similarity at spatial location `t` are:

`d_lt = ||z_t - p_l||_2^2`

`s_lt = log((d_lt + 1) / (d_lt + 1e-5))`.

The attention branch applies one class-agnostic and five class-specific `1x1`
filters to `Z`. Attention logits are the valid-spatial mean of their products.
The nonnegative location map is the maximum over classes of `relu(ca * cs_k)`,
zeroed outside valid support, then divided by its valid maximum plus `1e-6`.
Prototype evidence is `max_t(A_t s_lt)` over valid locations. A fixed linear
classifier connects each prototype to its own class with `+1` and every other
class with `-0.5`, without bias.

For target `y`, define nearest own-class and wrong-class distances at each
location. The candidate implements paper Eq. (3) as a batch-mean cross-sample
attention map, normalized over valid sample-location pairs:

`Lreg = 100 * mean(Abar_t min_{l in P_y} d_lt)
        - 0.08 * mean(Abar_t min_{l not in P_y} d_lt)`.

Here `Abar` is the coordinate-wise mean of every valid-normalized sample
attention map in the current batch, then intersected with each sample's valid
support. Attention values remain differentiable in the loss. A separately
written NumPy oracle must disambiguate and verify this equation before formal
work.

The total joint loss is prototype CE plus attention CE plus `Lreg`. No class
weights, oversampling, focal term, label smoothing, pretrained weight, image
augmentation, bbox loss, keeper distillation, or post-hoc target is allowed.

## Locked roles and causal controls

Four independently trained roles start byte-identically per fold and receive
the same natural batches and occurrence order:

1. `cross_sample_maxsep`: the candidate above.
2. `self_attention_maxsep`: replace `Abar` in `Lreg` by each sample's own `A`;
   all other equations and weights stay identical.
3. `cross_sample_cluster_only`: keep cross-sample attention but set the
   wrong-class separation coefficient to exactly zero.
4. `cross_sample_channel_dephased`: apply deterministic, per-channel toroidal
   shifts to frozen block-2 maps before the projector, preserving every channel
   marginal while destroying aligned local channel conjunctions; otherwise use
   the exact candidate equation.

Two same-weight interventions are computed from every held candidate:

- `cross_sample_same_weight_attention_rolled`: deterministically roll `A`
  relative to prototype similarity maps before spatial max;
- `cross_sample_same_weight_class_cycled`: cyclically reassign each prototype
  class identity for descriptor/class-score construction without retraining.

Additional non-neural controls are:

- `keeper_logprob`;
- `bbox_geometry_plus_keeper`, using valid/bbox area, aspect, center, margin,
  and log keeper probability only;
- `bbox_geometry_only`, using the same geometry without keeper probability;
- `block2_gap_plus_keeper`, using valid block-2 channel mean/std plus log keeper
  probability;
- `cross_sample_maxsep_without_keeper`.

No role, seed, width, prototype count, coefficient, feature layer, pooling,
descriptor, or threshold may be replaced after metric access.

## Locked optimization

- seed `42`, deterministic algorithms, TF32 disabled, and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- five source-held heads per trained role;
- batch size `64`, natural deterministic shuffle, no repeat sampler;
- Adam with default betas and no weight decay;
- epochs `1-5`: head warm-up at `3e-4`, prototype CE plus attention CE only;
- epochs `6-15`: joint loss at `3e-3`;
- epochs `16-20`: joint loss at `3e-4`;
- exactly 20 epochs, no early stopping, prototype push, pruning, final-layer
  fit, continuation, retry, or seed selection.

All projection, attention, and prototype parameters must receive finite nonzero
joint-stage gradients and change. Fixed classifier connections must remain
exact. Every role must process exactly the same row occurrences.

## Locked descriptors and OOF readout

For every learned or same-weight role, build the same descriptor from held
rows only:

- five attention logits;
- five prototype logits;
- per-class maximum and mean prototype evidence, ten values;
- per-class minimum prototype distance, five values;
- attention valid mass, entropy, peak, bbox mass, and outside-bbox mass;
- attention, prototype-logit, and nearest-distance class-1-versus-best-rival
  margins;
- five clipped log keeper probabilities, except in the no-keeper role.

Every value must be finite. Fit one `StandardScaler` plus binary L2 logistic
readout with `C=0.1`, `lbfgs`, tolerance `1e-8`, and at most 2,000 iterations
inside each source fold. Binary target is `target == 1`. The action threshold is
the lowest fit-positive score that retains at least 97% fit positives. Apply
the frozen fold readout and threshold to held rows only. There is no threshold
sweep and no validation/test selection.

## Engineering gates

All of these must pass before behavioral evidence can authorize anything:

- pinned input/source hashes and clean external source worktrees;
- no validation/test path opened and no raw file modified;
- exact row counts, order, targets, source disjointness, and fold occurrence
  equality across roles;
- hooked versus ordinary keeper probability error `<=1e-6`, exact argmax, and
  exact keeper state;
- independent NumPy/FP64 errors for projection, attention, distance,
  similarity, Eq. (1), Eq. (2), Eq. (3), descriptor, and gradients `<=1e-10`;
- finite FP32/BF16 output and gradients; FP32-versus-FP64 score error `<=1e-5`,
  BF16-versus-FP32 score error `<=0.02`, and exact non-near-tie actions;
- all trainable parameter families live and changed, fixed connections exact;
- static ONNX contains no custom domain, has max output error `<=1e-4`, and
  preserves non-near-tie actions;
- candidate head has at most 25,000 trainable parameters, adds at most 3.0 ms
  median latency at batch 32 on the RTX 4060, and uses at most 0.25 GiB extra
  peak CUDA memory;
- exact second-process reconstruction of states, descriptors, scores,
  thresholds, actions, and clean analysis with max error `<=1e-7`;
- compile, pyflakes, PowerShell AST parse, focused tests, full pytest, and
  `git diff --check` pass;
- temporary full-train feature caches are absent after both success and error.

## Clean mechanism gates

Every conjunctive gate below must pass. Passing only an aggregate metric is
insufficient.

- candidate AUROC `>=0.85` and AUPRC `>=0.90`;
- keeper-TP retention `>=0.95`, at most 26 keeper TP broken;
- at least 8 of 13 keeper FN supported;
- restricted-FP rejection `>=0.25` and accepted precision `>=0.75`;
- candidate AUROC exceeds keeper, bbox-plus-keeper, and GAP-plus-keeper by at
  least `0.02`, `0.015`, and `0.01`, respectively;
- candidate AUROC exceeds self-attention, cluster-only, and separately trained
  channel-dephased controls by at least `0.01`, `0.01`, and `0.02`;
- same-weight attention rolling and class cycling each reduce AUROC by at least
  `0.015` and `0.025`, respectively;
- candidate precision is at least `0.01` above every trained control and it
  rejects at least ten more restricted FP without breaking more keeper TP;
- candidate wins AUROC and restricted-FP rejection against keeper and every
  trained control in at least four of five folds;
- candidate without keeper reaches AUROC `>=0.72` and exceeds the bbox-only
  descriptor by at least `0.02`;
- class-1 correct-prototype-versus-best-rival margin has AUROC `>=0.75` for
  class-1 positives versus restricted FP;
- projected-feature effective rank is `>=16/32`;
- at least 20/25 prototypes are selected on held rows, with at least 3/5 used
  in every class and normalized class-1 prototype-use entropy `>=0.70`;
- mean nearest cross-class prototype distance is `>=0.10`, all prototype
  vectors are distinct, and nearest held-patch class purity is `>=0.60`;
- no attention map is all-zero/non-finite, fewer than 1% have valid peak below
  `1e-4`, and padding leakage is exactly zero;
- the candidate's direction, control margins, TP retention, and FP rejection
  all pass separately in clean source folds rather than being driven by one
  fold.

## Conditional robustness

Do not extract shifted features unless every engineering and clean mechanism
gate passes. If opened, freeze all heads, readouts, thresholds, and controls and
run `dim (brightness=0.70, contrast=0.90)`, `bright (1.25,1.10)`, and
`low_contrast (1.00,0.65)` over the same train-only cohort.

Each condition must retain candidate AUROC `>=0.80`, keeper-TP retention
`>=0.90`, restricted-FP rejection `>=0.18`, and accepted precision `>=0.70`.
The candidate must still beat every trained and same-weight control in at least
two of the three conditions and may not reverse the clean precision direction.

## Fixed XAI and manual gate

Render exactly 15 prospectively selected cohort rows: for each fold, the
lowest-sample-index keeper TP, keeper FN, and restricted FP available in that
fold. The sheet must show the input/bbox, candidate attention, own-class best
prototype similarity, best-rival prototype similarity, their signed margin,
self-attention, cluster-only, trained dephasing, same-weight rolled attention,
and input-gradient saliency for the frozen fold readout.

Automatic XAI must be finite, deterministic across two passes, reconstruct the
score within `1e-5`, leak zero mass into padding, and cover every fixed row.
Manual review passes only if the candidate repeatedly isolates plausible peel
lesion/ripeness evidence, the own-versus-rival margin is semantically different
between TP/FN and restricted FP, and dephasing/rolling visibly disrupts that
evidence. Broad yellow/green response, fruit silhouette, bbox edge, stem, hand,
table, padding, or generic foreground focus is a failure.

## Stop and promotion rules

- Any structural, clean mechanism, replay, export, resource, or XAI failure
  rejects this exact A0 and closes robustness, trainer integration, smoke,
  probe, full train, validation, test, and command promotion.
- Do not repair a failed gate by changing prototype count, width, lambda,
  attention normalization, layer, fold, seed, epoch, readout, threshold,
  descriptor, tolerance, or XAI row after metric access.
- A complete A0 pass authorizes only a checkpoint-safe, default-off TRKH
  integration followed by preflight and one smoke. It does not itself authorize
  test or current-best promotion.
- Only a later full-validation smoke/probe that reaches the next class-1
  milestone and passes all audits can authorize an autonomous full train under
  30 epochs. Current-best commands/history change only after a locked
  validation promotion and independent reload gate.
