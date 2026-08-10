# TRKH 5-Class Mutual-Channel Patch Readiness Protocol

Date: 2026-07-16
Status: immutable before implementation or metric access

## Decision question

Can class-aligned, spatially diverse channel groups add conservative positive
evidence for the ambiguous mango classes without moving the scratch keeper's
natural five-class geometry toward class-1 suppression or expansion?

This is a train-only A0 information gate. It cannot authorize validation,
test, shared-trainer integration, a probe/full train, or a current-best command
revision unless every gate below passes.

## Primary-source basis and limits

- Chang et al., *The Devil is in the Channels: Mutual-Channel Loss for
  Fine-Grained Image Classification*, IEEE TIP 2020 / arXiv 2002.04264:
  `https://arxiv.org/abs/2002.04264`.
- Paper SHA-256:
  `4b1111d23d427e4219cea72763dfaec1f1866418ffb725418bfea95eab82f7d2`.
- Official repository: `https://github.com/PRIS-CV/Mutual-Channel-Loss`, commit
  `befb3692cd0d5382eb32fa4e093226247f609fd9`.
- Official `CUB-200-2011_ResNet18.py`, `my_pooling.py`, `README.md`, and
  `LICENSE` SHAs are respectively
  `cf52b28b9a92cb2f2623f69576297f786521232a03f350a0e945360b13a92c3f`,
  `b2d526efa446a205d272610e06ed2ac117d3d6084042ec6fe38b7f6504bdaa54`,
  `072043e1663ddfb1ddab5ad355ffb2cf0b351a94ea4225848cd59954cd53b26c`,
  and
  `4f8a89636b5e3ce81d19cdfbfdca6c21aa2cf63cb216d93d15f69c61012cd109`.
- The repository carries an MIT license. Local implementation must still be
  equation-traceable and must not import its Python-1.2-era training script.
- The paper groups `xi=3` channels per class. Its discriminality term randomly
  drops one channel per group, max-pools the remaining group, and applies CE.
  Its diversity term spatially softmaxes every channel, max-pools within each
  class group, and rewards complementary spatial coverage.
- Official scratch code uses `alpha=1.5`, `beta=20`, SGD `lr=0.1`, momentum
  `0.9`, weight decay `5e-4`, and 300 epochs. The paper text reports a different
  diversity weight in one ablation. This A0 transfers the official code values
  once but trains only a frozen-feature adapter for 10 epochs. It is not a
  reproduction and cannot justify a 300-epoch TRKH schedule.

## Why this is not a repeated route

- RSC masks the largest true-logit-gradient channels and was tested as a
  feature-challenging objective. MCL instead creates fixed class-aligned groups
  and explicitly rewards within-group spatial complementarity.
- Deep prompts, semantic parts, Soft-MoE, and part prototypes assign tokens or
  samples to learned experts/parts. This A0 keeps token routing unchanged and
  groups output channels only.
- Register diversity and attention erasing act on prefix/attention behavior.
  MCL supervises the final patch feature map directly and exposes its maps as
  the mechanism audit.
- The candidate is compared against an identically parameterized natural-CE
  adapter. A win over raw alone is insufficient because residual adaptation
  can recalibrate class support without learning complementary evidence.

## Immutable data contract

- Dataset: `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, train paths only,
  SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Reuse exact source-disjoint fold 0: `7372` fit rows (`fold != 0`) and `1843`
  holdout rows (`fold == 0`), source overlap zero. Index SHAs are
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d`
  and
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`.
- Fit uses the natural five-class frequency distribution, not a class-1/hard-
  negative cohort. There is no oversampling, class weighting, or augmentation.
- Ten sequential NumPy `default_rng(42)` permutations produce exactly `73,720`
  fit occurrences. Ordered-index SHA-256 is
  `d56903f7806c12c19dddc4dc639054a98ba84a806f9568871157359be31512d4`.
- Batch size is 32 with the final short batch retained each epoch. Candidate
  and control consume the same order.
- Evaluate the exact 1843-row holdout under clean, dim `(0.70,0.90)`, bright
  `(1.25,1.10)`, and low-contrast `(1.00,0.65)` conditions.
- No validation/test path or prediction may be constructed or read. No raw
  image, label, YAML, or split may be modified.

## Frozen feature map and adapter

- Keep all 7,245,590 keeper parameters frozen in evaluation mode.
- Use the keeper's final normalized post-pruning patch tokens and their original
  patch indices. A preimplementation real-batch structural replay established
  exactly `167` retained 256D tokens. The model applies each keep rate to the
  original 256-token grid, so the final `0.65` rate gives
  `ceil(256 * 0.65) = 167`; multiplying sequential rates to infer 142 was
  incorrect. Scatter the 167 tokens into the native `16x16` grid and maintain
  an exact boolean valid-token mask. Missing cells never participate in spatial
  softmax, max pooling, or averaging.
- Candidate and control each add a `1x1 Conv2d(256,15,bias=True)` projection.
  The 15 output channels are five contiguous groups of exactly three channels.
- Masked spatial max pooling over all 15 maps feeds
  `Linear(15,5,bias=True)`. Its weight and bias are exactly zero, so deployed
  logits initially equal raw keeper logits bit-for-bit. Final logits are
  `raw_logits + 0.05 * residual_logits`.
- Projection initialization uses PyTorch Kaiming-uniform under an isolated
  seed-42 RNG scope. Candidate and control initial state, optimizer state,
  input features, batches, and masks must be bit-exact.
- Exact trainable count is `3935`: projection `3840+15`, residual head `75+5`.
  No keeper/head/token/pruning parameter is trainable.

## Equation-traceable MCL adaptation

- For each class group and occurrence, drop exactly one of three channels.
  The dropped index is SHA-256-derived from
  `mcl-mask|42|epoch|batch|sample_index|class`, modulo three. The complete mask
  occurrence SHA is
  `0b15865764091d839872ac840cdc6a37965662a4ef6b9e3e8edc95a7e2c5afb7`.
- Discriminality logits zero the dropped channel, max over the three channels,
  then average only valid spatial cells. `L_dis` is five-class CE.
- For diversity, spatial-softmax each of the 15 channels over valid cells,
  max the three normalized maps in each class group, and sum over valid cells.
  With `xi=3`, `L_div = 1 - mean(group_coverage)/3`.
- Candidate loss is
  `CE(raw + 0.05*residual, target) + 1.5*L_dis + 20*L_div`.
- Control loss is only `CE(raw + 0.05*residual, target)`. MCL values are logged
  under `detach()` and contribute exactly zero control gradient.
- Both use SGD `lr=0.1`, momentum `0.9`, weight decay `5e-4`, no scheduler,
  gradient clipping, AMP, or retry. Adapter optimization uses cached FP32 patch
  features; image extraction may use the keeper's locked BF16 inference path.
- Independent reference equations must match grouped logits, diversity loss,
  masks, and gradients before behavior is considered.

## Required audits

1. Paper/repository/license/source hashes, clean official worktree, exact
   dataset/split/order/mask hashes, parameter schema, isolated initialization,
   initial raw equivalence, finite gradients, movement, and frozen keeper.
2. Raw/control/candidate clean and three-condition predictions on all 1843
   holdout rows. Independently replay every metric, confusion matrix,
   transition, class support, and restricted `0/2/4 -> 1` false positive.
3. MCL mechanism telemetry: train/holdout `L_dis`, `L_div`, group coverage,
   channel-map cosine, spatial entropy, effective rank, per-class grouped-logit
   accuracy, and candidate-minus-control residual-margin AUROC for class 1
   versus restricted classes.
4. Runtime/VRAM against raw, plus isolated adapter and full static-batch-one
   ONNX opset-17 parity. No auxiliary training-only mask is required at
   inference.
5. XAI must include every clean class-1 TP break/rescue and restricted FP
   creation/removal, then deterministic corrections/harms until at least 12
   rows when available. Render original image, valid patch mask, all three
   class-1 maps for control/candidate, grouped max map, residual margin, and
   input saliency on a deterministic stratified subset. Selection and maps must
   be manifest-replayable.

## All-or-nothing Stage-A gates

### Structural and mechanism gates

- Every provenance/data/equation/order/mask/gradient/frozen-state/replay check
  passes; both variants execute 10 epochs and 73,720 occurrences.
- Every trainable parameter receives a finite nonzero gradient and changes.
- Candidate holdout `L_dis` is at most `0.98x` control and candidate diversity
  coverage is at least control; class-1 group-map spatial entropy remains at
  least `0.20` and maximum within-group cosine is at most `0.98`.
- Candidate class-1 residual-margin AUROC against classes `0/2/4` is at least
  `0.70` and no lower than control.

### Clean decision gates

Versus raw keeper, all must pass:

- macro F1 delta at least `-0.002`;
- class-1 F1 gain at least `+0.005`;
- class-1 precision gain at least `+0.005`;
- class-1 recall delta at least `-0.005`, at most one TP break, and FN rescues
  at least TP breaks;
- at least two fewer restricted false positives;
- corrections strictly exceed harms;
- maximum non-class-1 per-class F1 drop at most `0.010`.

Versus matched natural-CE control, all must pass:

- macro F1 delta nonnegative;
- class-1 F1 and precision gains each at least `+0.003`;
- class-1 recall delta at least `-0.005`;
- at least one fewer restricted false positive;
- corrections strictly exceed harms.

### Illumination, deployment, and XAI gates

- Candidate class-1 F1 and precision are nonnegative versus raw in at least two
  of dim/bright/low contrast, and separately nonnegative versus control in at
  least two of three.
- Worst class-1 recall delta is at least `-0.020` versus both comparators; net
  TP loss is at most two per condition; aggregate restricted-FP removals exceed
  creations versus raw and separately versus control.
- Runtime ratio is at most `1.10x` raw and peak allocation at most `0.75 GiB`.
- Both ONNX exports are finite, argmax-exact, and have maximum absolute error
  at most `1e-5`.
- Required XAI coverage is exact and every map/saliency tensor is finite and
  nonzero where defined.

## Decision rule and no-repeat boundary

- If any gate fails, reject this exact final-patch MCL residual A0. Do not sweep
  channels per class, insertion feature, projection/head shape, residual scale,
  alpha/beta, optimizer, LR, weight decay, epochs, fold, seed, mask rule, class
  balance, or add a post-hoc threshold/router.
- Failure does not reject MCL universally. Reopening requires a genuinely
  different primary-sourced representation and a new prospective protocol,
  not a neighboring MCL recipe.
- Only if every gate passes may Stage B add the exact default-off adapter/loss
  to shared code and run one short no-test validation probe. Current-best
  commands still require a locked validation promotion; test remains final.

## Artifact contract

- Write summary JSON, all-condition prediction CSV, training curve, mechanism
  CSV/JSON, XAI manifest/contact sheets, ONNX files, report, and a SHA-256
  artifact manifest.
- Do not write a `.pt`, `.pth`, `.ckpt`, TensorRT engine, trainable manifest,
  validation/test prediction, or current-best command revision.
- On closure, rerun compile/focused/full tests, PowerShell parse/preflight,
  protected hashes, independent replay, retention, journal/TODO, and skill.
  Never delete formal evidence or user-owned untracked paths.
