# TRKH 5-Class Soft-MoE Patch Adapter Readiness Protocol

Date locked: 2026-07-15

This protocol is immutable after the first formal Stage-A run. Any later
change to architecture, optimizer, fold, budget, gate, or locked source hash
is a new experiment, not a retry.

## 1. Research question

Can a small, fully differentiable patch-token mixture of experts model the
broad intra-class surface variation of mango class 1 while increasing class-1
precision without suppressing true class-1 positives?

The experiment preserves the current no-external-pretrain TRKH keeper, raw
dataset, CNN stem, native spatial MHSA, dense FFN, token pruning, classifier,
and XAI paths. It is a conditional feature transform inside the Transformer,
not another classifier, class prompt, post-hoc router, prototype, detector, or
data-editing policy.

## 2. Primary-source lock

### Soft MoE

- Paper: *From Sparse to Soft Mixtures of Experts*, ICLR 2024,
  `https://arxiv.org/abs/2308.00951`.
- Reviewed PDF SHA-256:
  `0d35f584a24f81b9e2736ca32548630747e9859a8e754d8e448899bbf5eae549`.
- Official repository: `https://github.com/google-research/vmoe`.
- Reviewed commit: `1030c713e7c0db5e12bb1a8cd1164d2c63b7b401`.
- `vmoe/projects/soft_moe/router.py` SHA-256:
  `d50d3894886e259c2adc274e4c852afb676ffa8ff0712982f25378970c75c720`.
- `vmoe/projects/soft_moe/configs/common.py` SHA-256:
  `9c1f9a56660432e43d0a2742b09106ed247942afd148b512801dc423b3137531`.
- Source behavior used here: L2-normalized tokens and slot vectors, one shared
  token-slot logit matrix, token-axis softmax for dispatch, expert/slot-axis
  softmax for combine, fully differentiable routing, and no token dropping.
  The official paper and recipe support one slot per expert.

### Moderate-scale vision MoE caution

- Paper: *Mixture of Experts for Image Classification: What's the Sweet
  Spot?*, `https://arxiv.org/abs/2411.18322v2`.
- Reviewed PDF SHA-256:
  `73a5c6bd80e1818cd2e159399e2e8b909ee1a4535f4e32a25f0ea418b36a2bcb`.
- Relevant findings: vision MoE gains are data- and capacity-sensitive,
  simple linear routing is strongest among tested routers, and moderate
  expert counts and sparse placement are safer than excessive capacity.

Soft MoE's strongest reported results use billion-scale data and very long
training. They do not establish success for this 9,215-row, no-pretrain,
at-most-30-epoch setting. This protocol therefore authorizes only one bounded,
identity-preserving train-only experiment.

## 3. Local no-repeat boundary

Do not reopen these closed routes:

- class prompts, class-query pooling, part tokens, objectness heads, branch
  tokens, post-hoc expert routers, classifier-head capacity, prototypes, and
  margin geometry;
- ViG max-relative patch graphs, XCA, visual-contrast attention, local/global
  concurrent branches, LeFF/local mixers, and patch-style SRM;
- raw Sobel/HOG/LBP/Gabor/wavelet/morphology/Deep-TEN texture injection;
- another CNN stem/tokenizer or global/focus teacher distillation;
- any class-1 support suppression policy that raises precision by breaking
  more true positives than it rescues.

This route is distinct because each patch receives a learned convex mixture
of four expert-slot outputs inside selected Transformer blocks. It changes
features before the existing classifier and retains all native spatial MHSA.

## 4. Locked TRKH adaptation

- Feature: `soft_moe_patch_adapter`, default off.
- Transformer blocks: exactly `2,5` (one-based), immediately after the native
  spatial-MHSA/local residual path and before the existing dense FFN.
- Input: patch tokens only. CLS, register, and branch prefixes bypass exactly.
- Experts: exactly `4` per selected block.
- Slots: exactly one slot per expert.
- Embedding/hidden dimensions: `256/64`.
- Router: one bias-free parameter matrix `Phi[256,4]` and one trainable scalar.
- Router initialization scale: `10.0`, matching the official Soft-MoE
  configuration option used to avoid near-uniform short-horizon routing.
- Expert: independent `Linear(256,64) -> GELU -> Linear(64,256)` MLP.
- Residual: `patch + 0.10 * soft_moe(patch)`.
- Identity initialization: every expert output projection is exactly zero.
- Initialization: isolated deterministic seed `20260715 + block_index`, with
  every visible CUDA generator forked and restored.
- No hard/top-k routing, token dropping, load-balancing loss, router noise,
  class-specific expert, bbox-biased routing, extra classifier, threshold,
  new data transform, or auxiliary loss.

For patch tokens `X[B,N,D]` and slot vectors `Phi[D,E]`:

1. `L = l2_norm(X) @ (scale * l2_norm(Phi))`.
2. `D_dispatch = softmax(L, dim=token)`.
3. `S = D_dispatch^T @ X`.
4. `Y_e = Expert_e(S_e)` for each of four experts.
5. `C_combine = softmax(L, dim=expert)`.
6. `Y = C_combine @ Y_e`.

Expected added parameters per selected block:

- router slots/scalar: `256 * 4 + 1 = 1,025`;
- experts: `4 * ((256*64+64) + (64*256+256)) = 132,352`;
- total per block: `133,377`;
- two-block total: exactly `266,754`.

## 5. Locked data and provenance

- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher-args SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- Dataset: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`.
- Data-spec SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Stage A may construct only `yolo_f/train`.
- Fold: source-disjoint fold `0`; expected fit/holdout rows `7372/1843`,
  with zero normalized source overlap.
- Validation and test loaders, predictions, thresholds, and metrics are
  forbidden during Stage A.

## 6. Matched Stage-A adaptation

Raw keeper, control, and candidate use the same adapter-enabled architecture,
initialized extension tensors, keeper state, deterministic fit order,
holdout order, transform, and forward RNG. Before adaptation, zero output
projections must make all three logits bit-exact to the keeper.

- Control trainable tensors: the eight expert output projections and biases in
  each selected block; router slots/scales and expert input projections frozen.
- Candidate trainable tensors: all Soft-MoE adapter tensors.
- Every original keeper tensor remains frozen and bit-exact.
- Optimizer: SGD, learning rate `0.005`, momentum `0.9`, weight decay `0.001`.
- Batch size: `32`.
- Batches: exactly `60`, hence `1,920` deterministic fit rows.
- Seed: `42`.
- Input: keeper evaluation geometry/normalization, no stochastic augmentation.
- Precision: BF16 adaptation/holdout; FP32 equation, attribution, and export.

The fixed-router control separates learning a generic zero-init residual
projection from learning Soft-MoE routing and expert representations.

## 7. Mandatory functional and mechanism gates

All must pass:

1. Exactly `266,754` added parameters and the exact expected state additions.
2. Constructor CPU/CUDA RNG equality with legacy keeper construction.
3. Every keeper tensor bit-exact after load and both adaptations.
4. Initial logits bit-exact; prefixes bit-exact at adapter boundaries; pruning
   remains after layers `2,5`; all eight native MHSA maps retain their schema.
5. Dispatch sums to one over patch tokens and combine sums to one over experts
   within `1e-5` in FP32 and BF16.
6. After the first warm update, every candidate router, input projection, and
   output projection has finite nonzero gradient; all candidate groups move.
   Control changes only its output projections.
7. Both layers are active: ablating either changes logits by at least `1e-4`.
8. No routing collapse on a balanced real cohort: each expert has at least
   `0.05` mean combine mass, no expert exceeds `0.70`, and normalized combine
   entropy is in `[0.35,0.995)`.
9. Class-1 true positives and restricted `0/2/4 -> 1` false positives have
   mean routing-signature L1 separation at least `0.01` in at least one layer.
10. Candidate residual-norm bbox mass is no more than `0.01` below control and
    candidate outside-bbox context mass increases by at most `0.02`.
11. Static-batch-1 three-input ONNX Runtime max logit error is at most `1e-5`
    with identical argmax.
12. Candidate/keeper median inference runtime is at most `1.25x`; peak
    candidate training allocation is at most `3.25 GiB`.

## 8. Mandatory train-only decision gates

Candidate must pass the following against both raw keeper and matched control
on all `1843` fold-0 holdout rows:

- macro F1 delta at least `0.000`;
- class-1 F1 delta at least `+0.005`;
- class-1 precision delta at least `+0.005`;
- class-1 recall delta at least `-0.005`;
- at least two correct removals of restricted `0/2/4 -> 1` false positives;
- class-1 FN rescues at least class-1 TP breaks;
- corrections strictly greater than harms;
- maximum F1 drop over classes `0/2/3/4` at most `0.005`.

The same raw/control/candidate models are evaluated on fixed dim, bright, and
low-contrast holdout views. Candidate-minus-both-reference class-1 F1 deltas
must all be at least `-0.005`, worst class-1 recall delta at least `-0.010`,
and at least two of three precision deltas must be nonnegative.

Any precision gain caused by class-1 support collapse or by breaking more true
positives than are rescued is an automatic failure.

## 9. Stage-B permission

Only a complete Stage-A pass authorizes one keeper-initialized matched
`120-batch x 2-epoch`, full-validation, no-test smoke. Stage B must then run
independent reload, transition accounting, clean/dim/bright/low-contrast and
occlusion robustness, architecture trace, all native attention, Grad-CAM,
rollout, perturbation audit, Soft-MoE routing/residual maps, expert ablations,
and changed-case contact-sheet review before continuation.

Failure closes this exact route without expert/layer/slot/hidden/scale/
residual/init/LR/optimizer/seed/fold/budget/loss/augmentation/run-length
sweeps. It does not authorize a hard router, class-specific expert, output
ensemble, post-hoc router, or test access.

The keeper, scratch complement, optional precision package, current full-train
command, and command-update history remain unchanged unless a later single
checkpoint passes the existing validation promotion gate.
