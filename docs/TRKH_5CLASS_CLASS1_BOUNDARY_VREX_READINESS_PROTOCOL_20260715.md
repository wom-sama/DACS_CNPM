# TRKH 5-Class Class-1 Boundary V-REx Readiness Protocol

Date locked: 2026-07-15
Status: locked before auditor implementation or candidate computation
Scope: `yolo_f/train` only; validation and test are forbidden

## 1. Question

Can Variance Risk Extrapolation preserve the clean precision gain exposed by
aggregate-margin A-GEM while preventing its dim, bright, and low-contrast
failures?

This A0 compares a boundary-balanced environment-mean ERM macrostep with a
V-REx macrostep. Both use the same keeper, rows, four deterministic lighting
environments, all trainable parameters, and normalized step. Only the
population risk-variance term may differ.

## 2. Primary-source lock

- Paper: Krueger et al., *Out-of-Distribution Generalization via Risk
  Extrapolation (REx)*, ICML 2021.
- Paper URL:
  `https://proceedings.mlr.press/v139/krueger21a.html`.
- Downloaded paper SHA-256:
  `99886a5074c06c277511418d1588d635f22b6096551efd334b253911a6656df1`.
- Reference implementation: `https://github.com/facebookresearch/DomainBed`.
- Locked DomainBed commit:
  `b93c22a1cfc3b2428398272c1a116c8de1f4139e`.
- `domainbed/algorithms.py` SHA-256:
  `6c2db72489f15d2ce155ab0490b14268bab16d793446d1ea5ebf2c7f22c4847b`.
- `domainbed/hparams_registry.py` SHA-256:
  `3a647bc462decebeccb0e15b3aa792afd34a5cfddac2136ae7fa668403d336ca`.
- License SHA-256:
  `454f8bb89faa42f38408ef1301982d51e62a1aad411ea5cb594dff57f15a26e5`.

Paper Equation 8 defines V-REx as the sum of environment risks plus a
population variance penalty. DomainBed computes the population mean and
`mean((losses-mean)^2)`. Its pre-anneal penalty weight is exactly `1.0`; the
default later weight is `10` after 500 updates. This one-macrostep A0 fixes
`beta=1.0`, corresponding to the official initial phase. Neither beta nor an
anneal point is selected from local behavior.

The paper warns that equalizing risk can be harmful when inherent label noise
differs across environments. Therefore this A0 does not use the rejected
quality/cartography groups. Every environment contains the same sample indices
and labels; only a deterministic lighting transform changes.

## 3. Distinction from closed routes

- Closed GroupDRO used train-prediction quality/cartography groups and worsened
  class-1 false positives. V-REx here uses paired lighting environments and a
  risk-variance penalty, not adversarial group weights or sample rebalancing.
- Closed A-GEM/GEM projected a clean hard-negative gradient against class-1
  reference constraints. V-REx has no constraint, QP, memory task, gamma,
  ridge, or projected gradient.
- Closed illumination/JSD/chroma consistency matched predictions or features
  between generic augmentations. V-REx minimizes supervised CE risk in every
  predefined environment and penalizes only environment-level risk variance.
- Closed Gray-Edge directly canonicalized color. V-REx preserves RGB inputs
  and uses no illuminant estimator.

## 4. Immutable local evidence

- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher-args SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- `yolo_f/data.yaml` SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- CIDT summary/prediction SHAs:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`
  and
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Boundary-GEM summary/prediction/manifest SHAs:
  `1d2bdd20837ee1300698fd144601dffa7a2cd0a3b3a7df17c7f801cb994a3e0c`,
  `5e66813cb0e2cbe238d7eaccfe94202a3d8362550139a75236baffa02c67cfb3`,
  and
  `960bbc3957f7a0eca4ee01379831f024923e5a72510c1dcc94513ea39e0f4183`.

The locked GEM prediction file supplies immutable raw and aggregate-margin
A-GEM comparators on clean, dim, bright, and low contrast. The new auditor
must verify every hash, target, sample index, path, probability argmax, and raw
CIDT decision before reuse.

## 5. Split, rows, and environments

- Dataset rows/source groups: `9215/8064`.
- Source-disjoint fold: hold out fold `0`; fit/holdout rows `7372/1843`.
- Fit/holdout source overlap must be zero.
- Fit hard-negative cohort: the same `186` clean keeper restricted
  `0/2/4 -> 1` false positives.
- Fit class-1 cohort: all `432` true-class-1 rows.
- Holdout diagnostic cohorts: `36` raw restricted false positives and all
  `109` true-class-1 rows. They are used only after the direction is fixed.
- Environments, applied to the exact same rows and labels:
  `clean=(1.00,1.00)`, `lighting_dim=(0.70,0.90)`,
  `lighting_bright=(1.25,1.10)`, and
  `low_contrast=(1.00,0.65)`, where each pair is brightness/contrast.

No random photometric operation, hue/saturation change, source group,
quality bucket, candidate prediction, validation row, or test row defines an
environment.

## 6. Boundary-balanced V-REx equation

For environment `e`, compute two row-mean five-class CE losses from the same
keeper in eval mode:

`H_e = mean CE` on the 186 restricted hard negatives;

`P_e = mean CE` on the 432 true class-1 rows;

`R_e = 0.5 * H_e + 0.5 * P_e`.

The equal weighting is fixed before computation so neither precision pressure
nor class-1 recall protection can dominate through cohort size. Let `m=4`:

`R_mean = (1/m) * sum_e R_e`;

`V = (1/m) * sum_e (R_e - R_mean)^2`;

`g_erm = (1/m) * sum_e grad(R_e)`;

`g_var = (2/m) * sum_e (R_e - R_mean) * grad(R_e)`;

`g_vrex = g_erm + 1.0 * g_var`.

This is the population-variance equation used by DomainBed. Compute losses and
gradients in FP64 accumulation and materialize parameter gradients in FP32.
Missing gradients are explicit zeros.

## 7. Exact A0 computation

1. Use deterministic keeper preprocessing, FP32, TF32 disabled, deterministic
   CUDA algorithms, batch `32`, workers `4`, seed `42`, model eval mode.
2. Compute `H_e`, `P_e`, and their all-parameter gradients separately for all
   four environments. Compose each `R_e` gradient with exact `0.5/0.5`
   weights.
3. Compose `g_erm`, `g_var`, and `g_vrex` from the locked equations. Compare
   tensor-list results with independently flattened NumPy/FP64 equations.
4. Create bit-identical ERM-control and V-REx-candidate models. Apply one
   explicit descent step, normalized independently to exactly `1e-4` of the
   same initial trainable-parameter L2 norm.
5. Use no optimizer, optimizer state, clipping, decay, scheduler, EMA,
   stochastic augmentation, teacher, auxiliary head, checkpoint save, or
   binary export.
6. Evaluate both new models on all 1843 source-disjoint holdout rows under all
   four environments. Reuse only hash-locked raw and aggregate-margin
   predictions as external comparators.
7. On the fixed 36 hard-negative plus 109 class-1 holdout rows, recompute the
   same `0.5/0.5` boundary risk for raw, aggregate-margin, ERM, and V-REx.

## 8. Structural and mechanism gates

Every check must pass:

- all source, paper, commit, code, argument, cohort, path, and artifact hashes
  are exact; the official DomainBed worktree is clean;
- no validation/test dataset or raw-data write occurs;
- all four environments contain the same exact unique rows and labels;
- reused raw/margin probabilities are normalized, finite, and have zero
  stored-prediction/argmax mismatch; raw clean has zero CIDT mismatch;
- every loss, gradient, risk, variance, parameter update, logit, and
  probability is finite;
- all eight cohort/environment gradients and all four environment-risk
  gradients have nonzero norm;
- environment-risk population variance is at least `1e-8`;
- the variance-gradient norm is nonzero, and V-REx differs from ERM by at
  least `1e-4` relative L2;
- tensor-list/flattened errors for `R_e`, mean, variance, ERM, and V-REx
  gradients are at most `1e-7`;
- the first-order V-REx objective change under the candidate descent direction
  is negative, and the first-order variance change is nonpositive;
- initial state/schema hashes are identical, every trainable parameter is
  included, both requested/actual steps are within `1e-8` of `1e-4`, and
  control/candidate update norms differ by at most `1e-7` relative;
- no checkpoint, ONNX, engine, or other binary model payload is written.

## 9. Clean behavior gates

V-REx candidate versus raw keeper must satisfy all:

- macro F1 delta `>= 0.000`;
- class-1 F1 delta `>= +0.005`;
- class-1 precision delta `>= +0.005`;
- class-1 recall delta `>= -0.005`;
- restricted `0/2/4 -> 1` false-positive reduction `>= 4`;
- class-1 FN rescues are at least TP breaks;
- corrections are greater than harms;
- maximum non-class-1 F1 drop `<= 0.010`;
- predicted class-1 support is at least `95%` of raw support.

V-REx must also exceed matched environment-mean ERM class-1 F1 by at least
`0.002`, have no more TP breaks than ERM, and reach aggregate-margin A-GEM
clean class-1 F1 minus at most `0.002`. These prevent promotion from a weak or
collapsed comparator.

## 10. Multi-condition behavior gates

For V-REx versus raw under dim, bright, and low contrast:

- every class-1 F1 delta `>= -0.010`;
- every class-1 recall delta `>= -0.015`;
- at least two conditions have nonnegative class-1 precision delta;
- aggregate class-1 TP breaks do not exceed FN rescues;
- no condition increases restricted class-1 false positives;
- worst-condition class-1 F1 improves by at least `0.010`.

Relative to matched ERM, V-REx must improve worst-condition class-1 F1 by at
least `0.005`. Its four-condition holdout boundary-risk variance must be at
most `90%` of ERM, while mean boundary risk may exceed ERM by at most `0.005`.

## 11. Decision and stop rule

Passing every gate authorizes only a separately precommitted train-only
multi-macrostep Stage B. It does not authorize validation, test, trainer
integration, probe, full train, or command promotion.

Any failed gate closes this exact boundary-balanced V-REx A0. Do not sweep
beta, anneal, environment transforms, cohort weighting, loss, step, fold,
seed, batch size, parameter subset, optimizer, number of macrosteps, or gate
thresholds. Do not reinterpret a corruption-only win, risk-variance reduction,
or comparator-only gain as a promoted model. Current-best command tracking
remains three revisions and two updates after the initial revision unless a
future independently locked validation candidate wins.
