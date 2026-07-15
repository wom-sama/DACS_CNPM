# TRKH 5-Class Class-1 Boundary-Stratified GEM Readiness Protocol

Date locked: 2026-07-15
Status: locked before auditor implementation or candidate computation
Scope: `yolo_f/train` only; validation and test are forbidden

## 1. Question

Did one-reference A-GEM hide finite-step damage to vulnerable class-1 examples
by averaging every class-1 row into one constraint?

This A0 compares an aggregate-margin A-GEM control with an original-GEM-style
multi-constraint projection. Both use the same hard-negative gradient, the
same four class-1 margin-gradient memories, the same initial keeper, and the
same normalized macrostep. Only the way the four constraints are combined may
differ.

## 2. Primary-source lock

- Paper: Lopez-Paz and Ranzato, *Gradient Episodic Memory for Continual
  Learning*, NeurIPS 2017.
- Paper URL:
  `https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html`.
- Downloaded paper SHA-256:
  `9aa6a5f73220449cbe8b2c9bd4bdc3f5407ac49295c4ca1f242a36e79e34b763`.
- Official source: `https://github.com/facebookresearch/GradientEpisodicMemory`.
- Official commit:
  `34c6b8e9a0607db7567301c48b727430d20bee7e`.
- Official `model/gem.py` SHA-256:
  `28e7d895165fd4c12044505309845db0c48bdf123acfff11a35d598295c1f123`.
- Official license SHA-256:
  `864df6b58bf3660c2944072253e854a0f51f47e26e251e1cab6e911e1e8b057e`.

Paper Equations 7-8 require every memory gradient `g_k` to satisfy
`g_projected^T g_k >= 0` and select the closest feasible gradient in squared
L2 norm. Equation 11 solves the dual QP over the number of memory tasks rather
than the number of model parameters.

The official `project2cone2` implements this dual with a `1e-3` diagonal term
and a lower-bound margin of `0.5`. The paper describes a nonnegative optional
gamma as a beneficial-backward-transfer bias. This A0 fixes gamma to zero and
uses the unregularized paper QP because its sole claim is no first-order harm;
no margin or ridge is tuned.

## 3. Distinction from closed routes

- Closed class-1-reference A-GEM used one row-mean multiclass-CE reference
  gradient. It removed eight restricted false positives but broke six keeper
  class-1 true positives.
- Closed PCGrad used localization/objectness tasks, not class-1 decision
  memories.
- Closed focal, OVA, margin, pAUC, Tversky, and contrastive routes blended a
  surrogate into ordinary minibatch optimization; they did not solve a
  multi-constraint projection.
- This A0 does not alter sampling, loss weights, raw data, logits after
  inference, thresholds, architecture, optimizer state, or deployment code.

The adaptation treats four precommitted class-1 boundary strata as GEM memory
tasks. Their loss is the row-mean binary decision-margin loss
`softplus(max_nonfocus_logit - class1_logit)`. The current-task loss remains
row-mean five-class CE on verified restricted head-class negatives.

## 4. Immutable sources and prior comparator

- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher-args SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- `yolo_f/data.yaml` SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- CIDT summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.
- CIDT prediction SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Closed A-GEM summary SHA-256:
  `a74deb9b4d235ced3c860ec00d21e65baa5efdecbdf6841b67552c0ecfae1732`.
- Closed A-GEM prediction SHA-256:
  `bc31de906c4f179487c5104a7d42710bda2a91be88e137aaef53f86a4af1d1d2`.
- Closed A-GEM manifest SHA-256:
  `417c3585604ed65f747e0c2265ef28acdb5b1a26810844f367780219f8feee75`.

The prior prediction file supplies immutable raw, unprojected CE-control, and
one-reference CE-A-GEM rows for clean/dim/bright/low contrast. The new auditor
must verify all prior hashes, row counts, targets, sample indices, and raw
CIDT argmax before using these comparators.

## 5. Split and boundary strata

- Dataset rows/source groups: `9215/8064`.
- Source-disjoint fold: hold out fold `0`; fit/holdout rows `7372/1843`.
- Fit/holdout source overlap must be zero.
- Hard-negative current task: the same `186` fit rows whose true class is
  `0`, `2`, or `4` and whose clean keeper prediction is class `1`.
- Class-1 memory: all `432` fit true-class-1 rows.

For each class-1 fit row, derive the exact keeper decision margin from CIDT
probabilities as
`log(p_class1) - log(max(p_class0,p_class2,p_class3,p_class4))`.
Sort ascending by `(margin, sample_index)` and split into four consecutive
strata of exactly `108` rows. No probability threshold or candidate output is
used.

Locked ordered-index SHA-256 values:

1. stratum 0: `8c1e751941bca6e4bcba7420aaa7edeb20cb4d939b0ef22676b4198299ee1c1f`;
2. stratum 1: `265d7a45af94fe7509f73500e9bb2afb7f4d24f3b70e0cec39cbd42c464b34d9`;
3. stratum 2: `2b9e364969b21cf9b4c303035c768eb22818761fd236d804002ac4eef8526ac3`;
4. stratum 3: `cccd701d5e54a4491066f7f0748f39978cb647747714a652aee81095197b7868`.

Stratum 0 must contain all `11` fit class-1 keeper errors (`10 -> 0`,
`1 -> 4`) and `97` keeper class-1 true positives. Strata 1-3 must each contain
`108` true positives. These are descriptive integrity checks, not selectors.

## 6. Exact A0 computation

1. Use deterministic keeper preprocessing, FP32, TF32 disabled, deterministic
   CUDA algorithms, batch `32`, workers `4`, seed `42`, model eval mode.
2. Compute all-parameter row-mean five-class CE gradient `g_hard` on the 186
   hard negatives.
3. Compute four all-parameter row-mean class-1 decision-margin gradients
   `g_0..g_3`, one for each locked stratum. Accumulate in FP64 and materialize
   in FP32. Missing per-parameter gradients are explicit zeros.
4. Aggregate-margin A-GEM control: set
   `g_avg=(g_0+g_1+g_2+g_3)/4` and apply exact A-GEM Equation 11 to
   `(g_hard,g_avg)`.
5. Boundary-stratified GEM candidate: solve the exact gamma-zero dual of GEM
   Equation 11 for all four constraints and recover the nearest feasible
   gradient. Enumerate all 16 active sets deterministically, solve each active
   Gram system in FP64, enforce KKT conditions, and select the feasible minimum
   primal-distance solution. Lexicographic active-set order breaks exact ties.
6. Compare the tensor-list result with a direct flattened reference; compare
   the active-set solution with SciPy bounded minimization for audit only.
7. Create bit-identical control/candidate models. Apply one explicit descent
   step normalized independently to exactly `1e-4` of the same initial
   trainable-parameter L2 norm.
8. Use no optimizer, clipping, decay, scheduler, EMA, augmentation, teacher,
   auxiliary loss, checkpoint save, or binary export.
9. Evaluate only the new aggregate control and GEM candidate on all 1843
   holdout rows under clean, dim, bright, and low contrast. Reuse hash-locked
   prior raw and CE-A-GEM predictions for comparisons.

The `1e-4` scale is retained to isolate multi-constraint geometry. It is not a
continuation or a sweep of the rejected A-GEM step.

## 7. Structural and QP gates

All checks must pass:

- all source, prior-artifact, commit, argument, cohort, order, and path hashes
  are exact; the official GEM worktree is clean;
- no validation/test dataset or raw-data write occurs;
- prior raw predictions match CIDT on `1843/1843` clean rows and every reused
  condition has exactly 1843 unique aligned rows;
- all logits, probabilities, losses, gradients, Gram values, dual variables,
  and parameter updates are finite;
- all four memory gradients and the hard gradient are nonzero;
- the Gram matrix is symmetric and positive semidefinite within `1e-8`;
- at least two raw hard-gradient constraints are negative, including stratum
  0, so multi-constraint projection is necessary;
- dual variables are at least `-1e-10`, active-set stationarity residual is at
  most `1e-8`, inactive dual-gradient values are at least `-1e-8`, and every
  projected primal dot is at least `-1e-7`;
- tensor-list/flattened GEM maximum absolute error is at most `1e-7` and the
  active-set/SciPy projected-direction relative error is at most `1e-7`;
- GEM retains at least `20%` of the hard-gradient norm and differs from the
  aggregate-margin A-GEM direction by at least `1e-6` relative L2;
- initial state and parameter schema hashes are identical;
- requested and actual step ratios are within `1e-8` of `1e-4`, with actual
  control/candidate step-norm mismatch at most `1e-7` relative;
- first-order loss change is non-increasing for every GEM stratum;
- no checkpoint, ONNX, engine, or other binary model payload is written.

## 8. Clean holdout behavior gates

GEM candidate versus raw keeper must satisfy every gate:

- macro F1 delta `>= 0.000`;
- class-1 F1 delta `>= +0.005`;
- class-1 precision delta `>= +0.005`;
- class-1 recall delta `>= -0.005`;
- restricted `0/2/4 -> 1` false-positive reduction `>= 4`;
- class-1 FN rescues are at least class-1 TP breaks;
- corrections are greater than harms;
- maximum non-class-1 F1 drop `<= 0.010`;
- predicted class-1 support is at least `95%` of raw support.

GEM candidate must also exceed the aggregate-margin A-GEM control in class-1
F1 and recall, have no more TP breaks relative to raw, and exceed the prior
CE-A-GEM class-1 F1 by at least `0.005`. These comparator gates prevent a win
caused by another collapsed control.

## 9. Illumination safety gates

For dim, bright, and low contrast, candidate versus raw must satisfy:

- every class-1 F1 delta `>= -0.010`;
- every class-1 recall delta `>= -0.015`;
- at least two conditions have nonnegative class-1 precision delta;
- aggregate class-1 TP breaks do not exceed FN rescues;
- no condition creates more restricted class-1 false positives than it
  removes.

The candidate must also improve low-contrast class-1 F1 by at least `0.020`
versus the prior CE-A-GEM candidate, because low contrast was its clearest
robustness failure.

## 10. Decision and stop rule

Passing every gate authorizes only a separately precommitted train-only
multi-macrostep Stage B. It does not authorize validation, test, trainer
integration, full train, or command promotion.

Any failed gate closes this exact boundary-stratified margin-GEM A0. Do not
sweep the number or definition of strata, loss, gamma, ridge, QP solver,
normalized step, fold, seed, cohort, parameter set, batch size, conditions, or
number of macrosteps. Do not reinterpret a comparator-only gain as promotion.
Current-best command tracking remains three revisions and two updates after
the initial revision unless a later independently locked validation promotion
wins.
