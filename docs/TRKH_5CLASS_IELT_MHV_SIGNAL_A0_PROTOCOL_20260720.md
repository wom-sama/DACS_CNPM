# TRKH 5-Class IELT Multi-Head Voting Signal A0 Protocol - 2026-07-20

## Question and boundary

Does the exact spatial voting idea in IELT's Multi-Head Voting (MHV) module
expose a class-1-positive surface signal that distinguishes true class-1
predictions from restricted `0/2/4 -> 1` false positives in the current TRKH
keeper?

This is a source-disjoint, train-only information audit. It does not modify the
raw dataset, model, trainer, checkpoint, validation predictions, test split, or
current-best commands. A passing A0 can authorize one default-off engineering
implementation and one matched short smoke. It cannot authorize a probe, full
train, test opening, or command promotion by itself.

## Primary source and adverse evidence

- Accepted article: Qin Xu, Jiahui Wang, Bo Jiang, and Bin Luo, *Fine-Grained
  Visual Classification via Internal Ensemble Learning Transformer*, IEEE
  Transactions on Multimedia 25 (2023), DOI
  `10.1109/TMM.2023.3244340`.
- Official MIT repository: `https://github.com/mobulan/IELT`, pinned at commit
  `b185111782d4a5f1e835060377022a7355bd5eec`, tree
  `713ed85495c3719a06f3a87fd2c4fd83dccfd050`.
- Official source hashes:
  - `models/IELT.py`:
    `cab0b8b28caab9f315ec9b214be820f82fd482bbc76cb175f1223e47020ef7de`;
  - `models/build.py`:
    `5d0cec610a1baec85d75c1b091c7a889d457882774297a478aeb91d235c6d042`;
  - `configs/cub.yaml`:
    `0ab2f5c4eddd57e0d8792cd388c01669d357d6b3349743a2711b19c46dab8a81`;
  - `README.md`:
    `e7a1ef914498835516aec066c02700dfd48d3dbc92d618006e13475000c13cea`;
  - `LICENSE`:
    `0dd3b7c901d75b63f24f43955d0b121a4edc59ae78be7be753d355be492f9069`.

The source evidence is not transferred uncritically. The released CUB recipe
uses ViT-B/16, input `448`, batch `14`, 50 epochs, an ImageNet-21k pretrained
`ViT-B_16.npz`, `vote_perhead=24`, `total_num=126`, and a dynamic-selection
warm-up of ten epochs. `build_models` loads the pretrained NumPy checkpoint
unconditionally for IELT. This is not evidence that full IELT will converge
from scratch at `256` pixels within TRKH's 30-epoch ceiling.

IELT's CLR and DS modules are not candidates in A0. Cross-layer token fusion,
TransFG/FFVT-style selection, learned Cropr routing, auxiliary refinement, and
dynamic pruning neighbors have already failed on the current keeper. The exact
MHV equation remains narrowly distinct because it treats attention heads as
weak learners, counts per-head top-k votes, then applies a fixed spatial
neighborhood kernel before token selection.

Prior TRKH DeepViT diagnostics are also adverse evidence: adjacent-layer map
cosines are only `0.398..0.645`, late inter-head full-map cosines are
`0.113..0.185`, and late effective ranks are `7.52..7.74/8`. MHV therefore
must empirically prove useful top-k consensus; generic head diversity is not a
pass.

## Locked inputs

- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`, SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher arguments: same run's `launcher_args.json`, SHA-256
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- Dataset declaration: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- CIDT train-only source folds and keeper declarations:
  `runs/audit_cidt_readiness_full_train_20260714/summary.json` and
  `predictions_all_conditions.csv`, SHA-256
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`
  and `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.

The clean cohort contains exactly 607 train objects from CIDT folds `1..4` for
which the keeper predicts class 1:

- 421 true class-1 positives;
- 186 restricted false positives with targets in classes `0`, `2`, or `4`;
- fold TP/FP counts `112/45`, `100/48`, `101/52`, and `108/41`;
- ordered sample-index SHA-256
  `a2689d1be8579386eea9ef02a826822a5e7d78e2d29439e8947c1731ccc738bd`.

Every image path must resolve under `yolo_f/images/train`; validation and test
paths are fatal. Dataset row order, class order, source stem, object index,
target, keeper prediction, probabilities, crop semantics, image-valid mask,
and bbox metadata must align exactly.

## Exact normal-forward capture

Run the keeper in evaluation mode through its ordinary metadata-aware forward
with token pruning enabled. Do not call the no-pruning XAI path. Register
read-only hooks on each block's post-softmax attention dropout and block output,
then call `forward_features(..., return_trace=True)` only to obtain the native
patch-index lineage. Hooks must be removed after every extraction.

For each of eight blocks, persist:

- post-softmax per-head attention `A_l`;
- block-output patch tokens before any pruning immediately following that
  block;
- active original 16x16 patch indices entering the block;
- normal logits and probabilities.

The hook path must reproduce the ordinary forward probabilities within
`1e-6`, preserve exact argmax except the already locked CIDT near-tie
declaration for sample `3657`, leave every state tensor bit-identical, and
retain the native patch counts `256/218/167` implied by pruning after blocks 2
and 5.

## Locked MHV adaptation

The official source uses CLS-to-patch attention, per-head top-24 voting over
784 patches, an unnormalized fixed kernel

```text
1 2 1
2 4 2
1 2 1
```

and descending vote-score selection. A0 preserves those equations while
scaling only counts required by the fixed TRKH geometry:

1. At a layer with `P` active patches, each of eight heads votes for
   `max(1, round(24 * P / 784))` patches. This yields `8/7/5` votes per head
   for `P=256/218/167`.
2. Votes are scattered to original 16x16 patch coordinates and convolved with
   the official fixed kernel using zero padding.
3. The official CUB layer quota vector
   `[16,14,12,10,8,6,8,10,12,14,16]` is linearly resampled to eight blocks.
   Its total is scaled by both layer count and patch area:
   `round(126 * 8/11 * 256/784) = 30` selected token-layer instances.
   Largest-remainder allocation is fixed at `[5,4,3,3,3,3,4,5]`.
4. Ties use ascending original patch index. No random tie break, learned score,
   bbox/foreground blend, threshold, or class-dependent quota is permitted.

An independent NumPy oracle must match top-k votes, bincount, convolution,
tie ordering, and selected original indices exactly on synthetic and real
inputs. The official source itself must remain clean and hash-exact.

## Matched roles and descriptor

All roles select the same locked number of tokens per block and use the same
post-block tokens:

- `cls_mean_smooth`: average CLS attention across heads, scatter to the 16x16
  grid, apply the same fixed kernel, then select;
- `vote_raw`: official per-head top-k vote counts without spatial smoothing;
- `vote_dephased_smooth`: roll each head's binary vote map by its fixed,
  distinct nonzero `(dy,dx)` offset before summation and smoothing. Head order
  uses `[(-3,-1),(-2,2),(-1,-3),(1,3),(2,-2),(3,1),(-3,3),(3,-3)]` with
  circular 16x16 rolls; this preserves per-head top-k mass while destroying
  genuine consensus;
- `mhv_smooth`: the candidate official vote count plus spatial smoothing.

For every selected token, apply parameter-free per-token LayerNorm with
`eps=1e-6`, project `256 -> 32` through a fixed QR-orthonormal Gaussian matrix
with seed `20260720 + block_index`, and concatenate selected-token mean and max
for all eight blocks. Every role is therefore exactly 512 dimensions. Append
the same five clipped keeper log probabilities to each role. `base_only` uses
only those five values.

No descriptor, projection, class weight, readout strength, threshold, role,
layer, quota, head subset, smoothing kernel, or seed may be selected after
observing metrics.

## Source-disjoint readout and actions

For each held CIDT fold in `1..4`, fit only on the other three folds:

- StandardScaler fit on the fit folds only;
- binary LogisticRegression with `C=0.1`, L2 penalty,
  `class_weight=balanced`, `max_iter=4000`, tolerance `1e-9`, seed `20260720`;
- fixed positive label: keeper prediction is a true class-1 positive;
- fixed negative label: keeper prediction is a restricted class-1 false
  positive.

The fit-fold action threshold is the lowest score that retains at least 97% of
fit-fold TP. Apply it once to the held fold. Persist fit/held source lists,
scaler/readout coefficients, convergence state, thresholds, scores, actions,
fold metrics, and concatenated OOF rows. Missing-class probability columns
must be expanded through `model.classes_` before fixed-shape metrics.

Report OOF AUROC, TP retention, FP rejection, per-fold AUROC and actions,
descriptor effective rank, role cosine/correlation, corrections/harms relative
to `base_only`, vote entropy, top-score ties, inter-head top-k Jaccard, selected
bbox/foreground mass, and per-target restricted-FP behavior.

Selector alignment is fixed as the arithmetic mean of the selected-token mean
bbox prior and selected-token mean foreground prior, averaged over all rows and
blocks. The candidate must exceed the dephased placebo on this scalar; neither
prior is an input to a selector or readout.

## Gates and stop rules

All structural, provenance, replay, resource, and clean information gates are
conjunctive. `mhv_smooth` must satisfy all of the following:

1. exact source/model/data/protocol hashes, clean tracked worktree, protected
   untracked hashes, train-only paths, exact 607-row cohort, and bit-identical
   keeper state;
2. normal-forward probability error `<=1e-6`, exact non-near-tie argmax,
   expected token lineage, exact independent MHV equation replay, finite FP32
   and BF16 arithmetic, and deterministic rerun;
3. OOF direction AUROC `>=0.65`;
4. AUROC gains of at least `+0.03` over `base_only`, `+0.02` over
   `cls_mean_smooth`, `+0.01` over `vote_raw`, and `+0.03` over
   `vote_dephased_smooth`;
5. held-fold action TP retention `>=0.95` and restricted-FP rejection
   `>=0.20` in aggregate;
6. candidate action breaks no more TP than every matched role and rejects at
   least ten more FP than every matched role;
7. candidate AUROC exceeds every matched role in at least three of four held
   folds, with no fold TP retention below `0.90`;
8. candidate descriptor effective rank `>=16`, finite nonzero variance in all
   block groups, and no majority top-score tie;
9. candidate beats the dephased placebo in selector alignment and information,
   proving that preserved head consensus rather than smoothing alone matters;
10. peak CUDA allocation `<=3.5 GiB`, no leaked hook, requested/effective
    workers recorded, and no owned or unknown competing compute process during
    formal timing.

Always render one hash-locked clean contact sheet with fixed TP and target
`0/2/4` FP rows, showing input, `cls_mean_smooth`, `vote_raw`,
`vote_dephased_smooth`, and `mhv_smooth` overlays. Manual review cannot rescue
an automatic rejection.

If any clean information gate fails, stop before lighting-condition extraction,
model/trainer integration, validation, test, smoke, probe, or full train. Close
MHV quota/head/kernel/layer/selector/CLR/DS neighbors on this keeper.

Only a complete clean pass authorizes the already fixed dim, bright, and
low-contrast replay. Every condition reuses the scaler, logistic readout, and
action threshold fit on the corresponding clean fit folds; no condition data
may be used for fitting. Those conditions must each retain candidate AUROC
`>=0.60`, TP retention `>=0.92`, and a positive candidate-minus-best-control
AUROC delta. A complete condition and visual pass authorizes one default-off
MHV-plus-light-refinement implementation and one matched, at-most
`120 batches x 2 epochs` full-validation smoke/probe pair. It still does not
authorize full training.

## Promotion boundary

A later matched image-model experiment must independently reach at least:

- validation macro F1 `>=0.884675`;
- class-1 precision `>=0.64`;
- class-1 recall `>=0.75`;
- class-1 F1 `>=0.70`;
- net restricted-FP removal, corrections not below harms, and no net class-1
  TP loss;
- complete robustness, architecture trace, native-attention, Grad-CAM,
  perturbation, calibration, reload, ONNX, runtime, and retention gates.

Only then may a justified autonomous full train be considered, with no more
than 30 epochs, patience 3 unless evidence supports a different value, and
measured train/eval workers selected for the current machine. The current-best
command/history files remain byte-identical until a locked independent-reload
single-model validation winner exists.
