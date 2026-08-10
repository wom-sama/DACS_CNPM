# TRKH 5-Class WILDCAT Negative-Evidence Readiness Protocol

Date: 2026-07-17
Status: immutable before implementation or behavioral metric access

## Decision question

Can a class-specific top-plus-bottom spatial evidence head extract a new,
precision-safe class-1 signal from the keeper's last dense Transformer map,
before token pruning removes context, without repeating generic saliency,
channel-diversity, post-hoc suppression, or threshold routing?

This is a train-only A0 information gate. It cannot authorize validation,
test, production-model integration, an image epoch, a probe/full train, or a
current-best command revision unless every gate below passes.

## Primary-source basis and limits

- Durand et al., *WILDCAT: Weakly Supervised Learning of Deep ConvNets for
  Image Classification, Pointwise Localization and Segmentation*, CVPR 2017:
  `https://openaccess.thecvf.com/content_cvpr_2017/html/Durand_WILDCAT_Weakly_Supervised_CVPR_2017_paper.html`.
- Paper SHA-256:
  `2209957a67f669aef37911294ea8bcf52da5c93fbecdaac20798af8297323995`.
- Author repository: `https://github.com/durandtibo/wildcat.pytorch`, MIT
  commit/tree `c7d355049a8fe34e98f49d1ff4ae91c996bd3825` /
  `2b777ae5058f1e71812f7201ddda2dc64cbb127a`.
- On 2026-07-17 the repository page reported 269 stars and 60 forks. Popularity
  is supporting metadata, not behavioral evidence.
- Official `wildcat/pooling.py`, `wildcat/models.py`, `README.md`, and
  `LICENSE` SHAs are respectively
  `d4a5fa61638acd5e39bc9cddf7b67aa67140c28b8c91505f18f7781fdbf504d5`,
  `be9c5bba1244175a21844fec15543e6f4b674c56a9b4f86b7880f71ba23cdbc5`,
  `ec817152e3199287f7ca2bbf23f03fb60d8423468c4de034844a80f053d68fd5`,
  and
  `90a9749c681f6d7f840925c81f8d04aba7f86f05f00bbccb543b443ad206064a`.
- The official code is a legacy PyTorch custom-`Function` implementation and
  contains `is not 0`. Do not import or copy that API. Reimplement the equation
  with current tensor operations, then compare it against source-extracted and
  independent scalar oracles.
- WILDCAT averages `M` maps for each class, averages the largest `k+` spatial
  values, and adds `alpha` times the average of the smallest `k-` values. The
  official source divides the combined score by two when the minimum branch is
  active.
- The paper reports that `alpha > 0` beats top-only pooling, with best values
  around 0.6/0.7/0.8 on VOC07/VOC12 Action/MIT67. Its fixed ablation shows
  complementary gains for `alpha=0.7` and `M=4`; larger `M` eventually
  overfits. The official VOC demo uses 20 epochs, `k=0.2`, `M=8`, and
  `alpha=0.7` with a pretrained ResNet-101. TRKH transfers only the equation,
  `k=0.2`, `alpha=0.7`, the paper's lower-risk `M=4` ablation, and the 20-epoch
  ceiling to a frozen-feature head. It does not use pretraining or claim a
  reproduction.

## Why this is not a repeated route

- The rejected final-patch MCL A0 grouped three channels per class, optimized
  channel dropout/diversity, and operated on 167 sparse post-pruning tokens.
  It never used class-specific bottom spatial evidence. This route instead
  uses the dense 256-token map before the first prune and has no MCL loss,
  channel dropout, diversity term, residual calibrator, or sparse scatter.
- Fine-grained pooling, class prompts, semantic parts, prototypes, query
  diagnostics, and token routers aggregate positive attention or route tokens.
  WILDCAT directly puts the lowest class-map responses into the classification
  equation and is causally checked by removing only that term from the same
  trained head.
- Push-pull and foreground/background filters act on image or convolutional
  support independent of class. WILDCAT's negative evidence is class-specific
  and cannot pass from foreground, boundary, or context energy alone.
- The head is evaluated as an auxiliary classifier. A0 does not tune a fusion
  weight, threshold, router, class bias, or residual scale around the keeper.
  A pass would authorize only one separately locked matched scratch pair using
  the branch as an auxiliary training objective.

## Immutable data contract

- Dataset: `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, train paths only,
  SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Reuse exact source-disjoint fold 0: 7,372 fit rows (`fold != 0`) and 1,843
  holdout rows (`fold == 0`), source overlap zero. Index SHAs are
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d`
  and
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`.
- Fit/holdout class counts are `[1561,432,1527,2017,1835]` and
  `[380,109,393,503,458]`. Use natural frequency with no oversampling, class
  weighting, sample weighting, augmentation, or hard-negative filtering.
- Twenty sequential NumPy `default_rng(42)` permutations produce exactly
  147,440 fit occurrences. Ordered-index SHA-256 is
  `bc02b9a8e0a36c96a7d407b654208a87d558beb907a8c7fa48174d1b5b2a7cb5`.
  Batch size 32 retains the final short batch: 231 updates per epoch and 4,620
  per head. All heads consume the exact same order.
- Evaluate the exact 1,843-row holdout under clean, dim
  `(brightness=0.70, contrast=0.90)`, bright `(1.25,1.10)`, and low-contrast
  `(1.00,0.65)` conditions. Apply brightness then contrast with the established
  TRKH condition implementation.
- No validation/test path or prediction may be constructed or read. No raw
  image, label, YAML, split, or hardlink tree may be modified.

## Frozen dense feature map

- Keep all 7,245,590 keeper parameters frozen in evaluation mode. Use normal
  token pruning and normal keeper inference; the branch must not alter keeper
  logits, token selection, or model state.
- Capture output from `model.blocks[1]` before the caller applies the first
  prune. A preimplementation synthetic replay established output
  `[1,263,256]`, prefix count 7, dense patch suffix `[1,256,256]`, and keeper
  logits `[1,5]`. Reshape the row-major patch suffix to `[B,256,16,16]`.
- This location is prospectively fixed because block 2 is the latest dense
  semantic map before `token_prune_layers=2,5`. Do not sweep patch embedding,
  block number, final sparse patches, stem features, normalization, or a mixed
  layer.
- Extract fit features once under the keeper's locked CUDA BF16 inference
  path, store a temporary float16 disk memmap, and convert each head-training
  batch to FP32. Record cache shape/content hash. Every head must read the same
  immutable cache. Remove the temporary cache after success or failure; it is
  not a formal artifact.

## Matched heads and exact equations

- Instantiate three independent `Conv2d(256,20,kernel_size=1,bias=True)` heads:
  `gap`, `top_only`, and `wildcat`. The 20 channels are five contiguous groups
  of `M=4`; average each group before spatial pooling. Each head has exactly
  5,140 trainable parameters and no other trainable state.
- Initialize one Kaiming-uniform prototype plus bias exactly as PyTorch
  `Conv2d.reset_parameters` under an isolated seed-42 RNG scope, then deep-copy
  it. Initial parameters must be bit-exact and the global RNG must be restored.
- `gap`: mean each class map over all 256 cells.
- `top_only`: sort each class map descending and average the largest
  `round(0.2*256)=51` values. This is the official `alpha=0` causal control.
- `wildcat`: use the same 51 largest and 51 smallest values and output
  `(mean(top51) + 0.7*mean(bottom51)) / 2`, matching the official source.
- For a same-weight causal ablation, evaluate the trained WILDCAT maps with
  `mean(top51)/2`. The common division preserves argmax scale while removing
  only negative evidence; it is not a separately trained fourth head.
- Train each head for exactly 20 epochs with five-class CE, SGD `lr=0.01`,
  momentum `0.9`, weight decay `1e-4`, no scheduler, warmup, clipping, AMP,
  retry, early stopping, checkpoint selection, calibration, or threshold.
  Epoch 20 is final regardless of fit loss.

## Required audits

1. Paper/repository/license/source hashes, clean official worktree, exact
   dataset/split/order hashes, tracked-clean TRKH state, and unchanged
   current-command/history hashes
   `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
   `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
2. Source-extracted and independent scalar equation replay for class-wise,
   `k`, top-only, and top-plus-bottom pooling; FP64/FP32/BF16, analytic
   gradient, finite-difference, ties, non-square-map, and invalid-argument
   checks.
3. Exact feature-hook call count/shape/order, raw-logit identity with and
   without capture, cache hash/cleanup, parameter schema, isolated identical
   initialization, optimizer parity, finite nonzero gradients, movement of
   every head parameter, and bit-exact frozen keeper state.
4. Fit loss/logit scale and clean plus three-condition predictions for raw,
   GAP, top-only, WILDCAT, and same-weight no-bottom ablation. Independently
   replay metrics, confusion, class support, transitions, and restricted
   `0/2/4 -> 1` false positives from CSV.
5. Mechanism telemetry: per-class top/bottom means, selected-cell masks,
   class-1-versus-restricted directional AUROC, bottom-margin AUROC, bbox top/
   bottom fractions, full-versus-no-bottom TP/FP transitions, map entropy,
   effective rank, and lighting stability.
6. Native full-forward runtime/peak allocation against raw while the branch is
   captured and evaluated, isolated-head and full-wrapper ONNX opset-17 parity,
   and TensorRT build/parity when the locked local tool is available. Raw
   keeper logits must remain unchanged.
7. XAI must include every clean WILDCAT-versus-top-only and full-versus-no-
   bottom class-1 TP break/rescue and restricted-FP creation/removal, followed
   by deterministic corrections/harms until at least 16 rows when available.
   Render image+bbox, averaged class-1 map, top mask, bottom mask, signed bottom
   contribution, and input saliency. Selection/maps must be manifest-replayable
   and every contact sheet must be manually reviewed.

## All-or-nothing Stage-A gates

### Structural and mechanism gates

- Every provenance/data/equation/order/cache/gradient/frozen-state/replay check
  passes; each head executes 20 epochs, 4,620 updates, and 147,440 occurrences.
- Every trainable parameter receives a finite nonzero gradient and changes;
  all keeper parameters/buffers and raw logits remain exact.
- Clean WILDCAT class-1-versus-restricted directional AUROC is at least 0.70
  and at least `+0.005` above both separately trained controls.
- Candidate bottom class-margin AUROC for class-1 TP versus restricted rows is
  at least 0.65. Same-weight bottom evidence removes at least two restricted
  false positives, breaks at most one class-1 TP, and produces more corrections
  than harms versus the no-bottom ablation.
- Class-1 top cells are object-aligned above bbox-area chance by at least 0.10
  on TP rows. Geometry alone cannot satisfy any information gate.

### Clean precision and support gates

Against both GAP and top-only controls, all must pass:

- macro F1 delta at least `-0.002`;
- class-1 precision gain at least `+0.005` and candidate precision at least
  0.75;
- class-1 F1 gain at least `+0.003` and candidate F1 at least 0.70;
- class-1 recall delta at least `-0.020`, candidate recall at least 0.70, and
  no more than two additional TP breaks;
- at least two fewer restricted false positives;
- corrections strictly exceed harms;
- maximum non-class-1 per-class F1 drop at most 0.010.

### Illumination, deployment, and XAI gates

- WILDCAT class-1 precision, F1, and directional AUROC are each nonnegative
  versus both trained controls in at least two of dim/bright/low contrast.
- Worst class-1 recall delta is at least `-0.030` versus both controls; aggregate
  restricted-FP removals exceed creations; same-weight bottom ablation is TP-
  safe in at least two conditions.
- Full-forward runtime ratio is at most `1.10x` raw and peak allocation is at
  most `1.10x` raw or `+0.25 GiB`, whichever is more permissive.
- Required ONNX/TensorRT outputs are finite and argmax-exact with maximum
  absolute error at most `1e-5` for ONNX and `2e-3` for BF16 TensorRT.
- XAI coverage/manifest replay is exact; every defined map/saliency tensor is
  finite and nonzero. Manual review must find class-specific TP/FP behavior,
  not only generic foreground or boundary activation.

## Decision rule and no-repeat boundary

- If any gate fails, reject this exact dense-block2 WILDCAT A0. Do not sweep
  insertion layer, `M`, `k`, `alpha`, normalization, head depth, optimizer,
  LR, weight decay, epochs, fold, seed, class balance, augmentation, fusion,
  threshold, or router.
- Failure does not reject every possible signed spatial classifier. Reopening
  requires a different accepted primary equation and a new prospective
  protocol, not a neighboring WILDCAT recipe.
- Only a complete pass may authorize one separately locked matched no-pretrain
  scratch pair with the exact branch as an auxiliary objective. Validation,
  test, and current-best promotion remain separate gates.

## Artifact contract

- Write summary JSON, all-condition prediction CSV, training/mechanism CSVs,
  XAI manifests/contact sheets, ONNX/TensorRT audit outputs, report, and a
  SHA-256 artifact manifest.
- Do not write a `.pt`, `.pth`, `.ckpt`, validation/test prediction, production
  checkpoint, or current-best command revision. Temporary feature caches must
  be absent at terminal status.
- On closure, rerun compile/static/focused/full tests, PowerShell parse and
  no-output preflight, protected hashes, independent replay, retention,
  journal/TODO, and skill updates. Never delete formal evidence or user-owned
  untracked paths.
