# TRKH 5-Class Balanced BCE Frozen-Embedding A0 Protocol

Date: 2026-07-24
Protocol ID: `trkh_balanced_bce_frozen_embedding_a0_20260724`
State: prospective; no Bal-BCE candidate metric has been observed

## Decision Scope

This A0 asks one narrow question:

> On the frozen 9,215-row `yolo_f/train` keeper representation, does the
> official LiVT Balanced BCE equation improve class-1 precision and remove
> restricted false positives without paying for those gains by breaking
> class-1 true positives, recall, macro F1, or nonfocus classes?

This is a source-disjoint train-only readout gate. It is not:

- an end-to-end LiVT reproduction;
- masked generative pretraining;
- a production-trainer integration;
- a validation, final-test, smoke, probe, or full-train result;
- evidence that TRKH has reached per-class F1 greater than 0.98.

The formal protocol and machine lock must be committed and pushed before any
candidate implementation, training, or metric is allowed. A failed gate closes
the exact equation, role, seed, fold, epoch, learning-rate, and decision
neighborhood without a nearby sweep.

## Primary Sources And License

Accepted primary sources:

1. Zhengzhuo Xu et al., *Learning Imbalanced Data With Vision Transformers*,
   CVPR 2023:
   <https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Learning_Imbalanced_Data_With_Vision_Transformers_CVPR_2023_paper.html>
2. Official LiVT repository:
   <https://github.com/XuZhengzhuo/LiVT>

Pinned local evidence:

- paper:
  `D:\DataAI\external_sources\papers\Xu_Learning_Imbalanced_Data_With_Vision_Transformers_CVPR_2023.pdf`;
- paper SHA-256:
  `79d1afec19528d639209661f4694c8b6506aed1e4e9d6a3ee54ce5be89c505ba`;
- repository:
  `D:\DataAI\external_sources\official\LiVT`;
- commit:
  `68546ef189c486caa271066d8bfa25ec214192df`;
- tree:
  `ec1fb1784566fd5f73f0f9710b2e9de06cb0e50c`;
- root license: MIT;
- license SHA-256:
  `ebed29105302a78fecb80e3f2c174d03ffba136b664596ce3e41d5fc20002e32`;
- `util/loss.py` SHA-256:
  `d73c21ba3a9ca8afbc5b230b48df97555dd57a4d91551fae89167d6f5baf8246`.

The implementation will be equation-first and local. It will not import LiVT
or its old runtime. The pinned source is used to verify provenance, license,
the exact bias direction, one-hot BCE behavior, and class-count scaling.

## Why This Route Is Eligible

The current frozen representation has a recurrent class-1 precision problem.
CE in the RN-LISDA train-only audit reached macro/class-1 F1 about
`0.940600/0.823853`, class-1 precision/recall about
`0.817851/0.829945`, and 96 restricted false positives. Fixed class-wise ISDA
increased precision but broke 25 class-1 true positives and reduced class-1
F1. Natural sampling previously increased precision by `0.171753` but reduced
recall by `0.516557`. Historical Balanced Softmax `tau=0.25` reached only
macro/class-1 F1 `0.8747/0.6558` with class-1 precision/recall about
`0.555/0.8013`.

LiVT is relevant because it derives a binary prior correction for ViTs trained
from scratch and reports fast fine-tuning convergence. It is not accepted on
authority alone. Bal-BCE can over-expand a rare class, and the LiVT paper
explicitly recognizes many-shot trade-offs in long-tailed rebalancing.
Therefore TRKH requires a precision-and-TP conjunction rather than tail
accuracy alone.

LiVT masked generative pretraining is outside this A0. Its published
800-epoch stage is incompatible with the current no-pretrain, at-most-30-epoch
TRKH claim and would confound the loss-level question.

## Immutable Inputs

Only these repository inputs are eligible:

- `runs/diagnostic_reslt_embedding_cache_keeper_yolof_20260712/train_embeddings.npz`;
- `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`;
- current keeper checkpoint and resolved config for immutable provenance only;
- `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml` for dataset-root identity only;
- this protocol and its machine-readable lock;
- current-best command and history files for byte-identity protection only.

The embedding cache contains:

- embeddings `[9215, 256]`, `float32`;
- keeper probabilities `[9215, 5]`, `float32`;
- labels `[9215]`, `int64`;
- keeper predictions `[9215]`, `int64`;
- image paths `[9215]`, all under `yolo_f/images/train`;
- sample indices `[9215]`, exactly `0..9214`.

Class counts are `[1941, 541, 1920, 2520, 2293]`. The five already locked
CIDT outer folds have held counts `[1843, 1830, 1828, 1851, 1863]` and zero
source overlap. The A0 must reuse those assignments byte-for-byte.

Validation and test pixels, paths, labels, predictions, embeddings,
checkpoints, metrics, or statistics are forbidden. `class_f` is not an input
to this loss-level A0.

## Raw And Derived Data Policy

Raw `class_f` and `yolo_f` data are immutable. Preprocessing, augmentation,
and synthetic children from existing data remain generally authorized only
under the project anti-leak policy, but this A0 creates no derived image
corpus and applies no image augmentation. It trains linear heads only on the
frozen train embeddings.

If a later stage uses augmentation or synthetic data:

- every child inherits the union of all parent source, split, and fold IDs;
- generator, prompt, filter, quality model, threshold, and ratio are fitted or
  cross-fitted using only the fitting train partition;
- the complete corpus is frozen before a separate cross-split duplicate audit;
- a validation/test match invalidates the complete corpus or method and cannot
  select, delete, replace, replenish, or tune individual children.

## Dynamic Data-Access Boundary

The formal process must install a process-wide `sys.addaudithook` open-event
ledger before the first candidate input is opened. It must:

- allow only the exact immutable files above, the pinned LiVT source files
  used for hash verification, and the active formal output directory;
- block any path with a complete `val`, `valid`, `validation`, or `test`
  component before open;
- block all writes under dataset roots;
- record normalized path, mode, open count, blocked-attempt count, and an
  ordered ledger digest;
- observe a named installation probe before input loading;
- reproduce the access digest during fresh-process replay.

The cached train image paths may be validated as strings. Formal scoring does
not open their pixels. Only the fixed visual-evidence builder may open the 20
prospectively locked `yolo_f/images/train` anchors after all probabilities and
actions are frozen.

## Outer-Fold Rule

For each outer fold `f`:

1. held rows are the already locked CIDT rows with `fold == f`;
2. fit rows are every other train row;
3. priors and class counts are computed only from fit rows;
4. all six roles train only on fit rows;
5. all roles score the held rows once at epoch 30;
6. held rows never affect a loss, prior, order, state, threshold, epoch, or
   role selection.

Fold assignment, fit/held index arrays, source strings, class counts, priors,
logit biases, initialization states, and all 30 epoch orders are hashed in the
machine lock before candidate code exists.

## Equations

Let `C=5`, raw head logits be `z`, hard class label be `y`, and fit-fold prior
for class `c` be:

`pi_c = n_c / sum_j(n_j)`.

All logarithms use `float64` to construct locked constants, then constants are
cast to `float32` for training.

### CE Control

`L_CE(z,y) = -log softmax(z)_y`.

### Plain BCE Control

Convert `y` to an exact one-hot vector `t`. No smoothing is allowed:

`L_BCE(z,t) = C * BCEWithLogits(z,t,reduction="mean")`.

Multiplication by `C` matches the official LiVT implementation and converts
the class-element mean to a per-sample class sum.

### Exact Bal-BCE Candidate

The official Theorem-2 bias is:

`b_c = log(pi_c) - log(1-pi_c)`.

Training loss:

`L_BalBCE(z,t) = C * BCEWithLogits(z+b,t,reduction="mean")`.

`tau` is fixed to `1.0`. There is no sweep, clipping, class weight,
`pos_weight`, label smoothing, or test-prior term. Held inference and every
promotion metric use raw `z`, not `z+b`, matching the official loss-only
calibration path.

### Balanced-Softmax Historical Control

The locked historical control uses:

`L_BalSoftmax(z,y) = CE(z + 0.25*log(pi), y)`.

Held inference uses raw `z`. `tau=0.25` is fixed from the prior TRKH
experiment and cannot be tuned here.

### Reversed-Prior BCE Control

The sign control uses:

`L_Reverse(z,t) = C * BCEWithLogits(z-b,t,reduction="mean")`.

It tests whether any result is specific to the official prior direction rather
than a generic large intercept perturbation.

## Roles

Exactly six roles are trained:

1. `ce_control`;
2. `plain_bce_control`;
3. `balanced_bce_candidate`;
4. `balanced_bce_seed_repeat`;
5. `balanced_softmax_tau025_control`;
6. `reversed_prior_bce_control`.

The five primary roles share identical head initialization and epoch orders
within each fold. The seed repeat uses a separately locked initialization and
orders with seed offset `+100000`. It changes no equation, fold, epoch,
optimizer, or metric.

## Head And Optimization Lock

Each role trains one `torch.nn.Linear(256,5,bias=True)` head in `float32`.
There is no hidden layer, dropout, normalization, feature update, feature
standardization, class weighting, resampling, or global class-1 oversampling.

Fixed optimization:

- device: CUDA;
- deterministic algorithms: enabled;
- TF32: disabled;
- seed: `20260724`;
- epochs: 30;
- batch size: 64;
- drop-last: false;
- optimizer: SGD;
- initial learning rate: `0.03`;
- momentum: `0.9`;
- dampening: `0`;
- weight decay: `0`;
- Nesterov: false;
- epochs 0-4: linear warmup to `0.03`;
- epochs 5-29: cosine decay over the fixed 30-epoch horizon;
- no early stopping;
- no best-epoch selection;
- score epoch 30 only.

Fit indices are sorted by `sample_index`, then permuted by one locked
NumPy `default_rng` stream per fold and seed. Every primary role consumes the
same ordered indices for all 30 epochs. The repeat consumes its own locked
stream. Every fit row is used once per epoch.

## Equation And Runtime Oracles

Before formal training, tests and preflight must prove:

- manual NumPy CE equals PyTorch CE;
- manual sigmoid BCE equals PyTorch one-hot BCE after class-count scaling;
- candidate bias equals `log(pi)-log1p(-pi)` for every fold and class;
- candidate and reversed controls apply exact opposite biases;
- Balanced-Softmax applies only `0.25*log(pi)`;
- all training losses are finite for extreme oracle logits;
- held inference uses raw logits for every role;
- primary initial states and epoch orders are identical across roles;
- repeat states and orders match their separately locked hashes;
- no held index appears in a fit order or prior count.

The official LiVT runtime is not installed. The repository's existing PyTorch
runtime is the only execution runtime.

## Metrics

From the 9,215 concatenated OOF held predictions, report for every role:

- confusion matrix;
- per-class precision, recall, F1, support, TP, FP, and FN;
- macro and weighted F1;
- accuracy;
- multiclass NLL from raw softmax logits;
- 15-bin equal-width ECE from raw softmax logits;
- class-1 TP, FP, FN;
- restricted class-1 FP from true classes `0`, `2`, and `4`;
- prediction transitions, corrections, harms, class-1 TP breaks/rescues, and
  restricted-FP removals/creations versus CE;
- all metrics per outer fold.

No threshold calibration is allowed. Prediction is `argmax(raw_logits)`.

Training traces record each role/fold at epochs `1`, `5`, `10`, `20`, and
`30`, but traces cannot select an epoch.

## Prospective Gates

All gates are conjunctive.

### Structure And Compatibility

- every immutable hash and external-source hash matches;
- cache shape, order, class counts, train-only paths, and CIDT folds match;
- source overlap is zero in every fold;
- equation/runtime oracles pass;
- data-access ledger is installed before input and reports zero blocked,
  validation, and test opens;
- CE macro F1 is at least `0.935`;
- CE class-1 F1 is at least `0.800`.

### Candidate Versus CE

- macro-F1 gain at least `0.001`;
- class-1 precision gain at least `0.015`;
- class-1 F1 gain at least `0.005`;
- class-1 recall loss at most `0.010`;
- class-1 TP net at least `-3`;
- restricted-FP net removal at least `10`;
- total corrections are at least total harms;
- maximum F1 loss for any nonfocus class at most `0.005`;
- NLL increase at most `0.010`;
- ECE increase at most `0.010`.

### Candidate Versus Plain BCE

- macro-F1 loss at most `0.001`;
- class-1 precision gain at least `0.005`;
- class-1 F1 gain at least `0.005`;
- class-1 TP net at least `-2`;
- restricted-FP net removal at least `5`.

### Candidate Versus Historical And Sign Controls

- macro-F1 gain versus Balanced-Softmax at least `0.010`;
- class-1 F1 gain versus Balanced-Softmax at least `0.010`;
- macro-F1 gain versus reversed-prior BCE at least `0.005`;
- class-1 F1 gain versus reversed-prior BCE at least `0.010`;
- candidate mean raw class-1 logit differs from plain BCE by at least `0.10`;
- candidate and reversed-prior prediction arrays are not identical.

### Fold Stability

- class-1 precision is nonworse than CE in at least four of five folds;
- class-1 F1 improves over CE in at least four of five folds;
- per-fold class-1 recall loss is at most `0.030`;
- per-fold class-1 TP net is at least `-2`;
- no fold loses more than `0.010` macro F1.

### Seed Repeat

- the repeat independently passes every candidate-versus-CE safety gate;
- prediction agreement with the primary candidate is at least `0.970`;
- absolute macro-F1 difference at most `0.010`;
- absolute class-1 F1 difference at most `0.010`;
- absolute class-1 precision and recall differences at most `0.020`.

### Resources And Replay

- start available physical RAM at least `4.0 GiB`;
- peak process RSS at most `3.0 GiB`;
- peak CUDA allocation at most `1.0 GiB`;
- formal elapsed time at most `15 minutes`;
- no unrelated Python/TensorRT compute process;
- fresh-process replay matches logits/probabilities/states within `1e-7`,
  nested metrics within `1e-10`, integer actions exactly, and the ordered
  access ledger exactly.

## Fixed Visual And Manual Review

The fixed metric-independent anchors are the 20 ordered sample indices already
locked before RN-LISDA:

`[4002,3195,3284,4613,3217,2127,2707,5213,4616,4770,3955,4012,4624,3248,2718,5732,5327,6113,4916,3624]`.

Their ordered-index SHA-256 is
`cf6bfe611947a72075d5d85525e74c4a74d103cdfb276a5f7ee9b2d27f8a901e`.

After formal probabilities and actions are frozen, produce one contact sheet
with the original train image, true class, fold, CE/plain-BCE/Bal-BCE/
Balanced-Softmax/reversed-prior prediction and class-1 probability, plus the
raw class-1 feature contribution and intercept. The sheet is diagnostic, not
a selection surface.

Manual review must record:

- whether every class-1 TP break is visually defensible;
- whether class-1 FP removals preserve plausible class semantics;
- whether new class-1 predictions are driven only by a prior intercept despite
  contradictory visible fruit state;
- whether any displayed source is not train-only;
- an explicit accept or reject decision.

No pixel-level XAI claim is made because the encoder is frozen and unchanged.
If A0 passes, the separately locked production smoke must run the complete
TRKH XAI/audit stack before any promotion.

## Artifacts And Finalization

Formal output must include:

- `summary.json`;
- `fold_protocol.json`;
- `fold_metrics.csv`;
- `oof_probabilities.npz`;
- `oof_rows.csv`;
- `role_states.npz`;
- `training_trace.jsonl`;
- `data_access_ledger.json`;
- `fixed_visual_arrays.npz`;
- `fixed_visual_metadata.json`;
- `fixed_visual_contact_sheet.png`;
- `artifact_manifest.json`;
- `replay.json`;
- `formal_visual_review.json`;
- `final_decision.json`;
- `final_manifest.json`.

The finalizer authorizes the next stage only when every structural,
compatibility, performance, fold, seed, mechanism, resource, replay, and
manual gate passes. It must not average away a failed gate.

## Pass And Failure Policy

Pass opens only:

1. a default-off production Bal-BCE loss implementation; and
2. a new prospective matched CE-versus-Bal-BCE validation smoke capped at
   30 epochs, followed by the full audit and XAI stack.

Pass does not authorize final test, probe, full train, or current-best update.

Failure closes this exact A0 and forbids:

- tuning `tau`, prior clipping, smoothing, `pos_weight`, class weights,
  threshold, epoch, seed, batch size, LR, or nearby loss mixtures;
- changing folds after seeing metrics;
- integrating the loss into the production trainer;
- opening validation/test;
- launching smoke, probe, or full train;
- updating current-best commands or history.

Current-best commands may change only after a separately locked candidate wins
validation, replay, audit, and XAI gates. Raw data remains immutable in every
outcome.
