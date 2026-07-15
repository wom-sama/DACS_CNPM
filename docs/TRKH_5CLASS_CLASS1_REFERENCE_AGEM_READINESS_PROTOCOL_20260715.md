# TRKH 5-Class Class-1-Reference A-GEM Readiness Protocol

Date locked: 2026-07-15  
Status: locked before auditor implementation or model update  
Scope: `yolo_f/train` only; validation and test are forbidden

## 1. Question

Can a class-1 reference gradient prevent the precision-oriented update on
verified `0/2/4 -> 1` hard negatives from destroying true class-1 evidence?

This is a one-macrostep information gate. It does not add an inference module,
change raw data, train a model, or authorize validation by itself.

## 2. Primary-source lock

- Paper: Chaudhry, Ranzato, Rohrbach, and Elhoseiny, *Efficient Lifelong
  Learning with A-GEM*, ICLR 2019.
- Paper URL: `https://arxiv.org/abs/1812.00420`.
- Downloaded paper SHA-256:
  `79372cc7f08e80acf257e7d60e0357b883a5927f6fb94e260bd32017e270c1a9`.
- Official source: `https://github.com/facebookresearch/agem`.
- Official commit:
  `45421499483b28935491251e9e821c55e8b3c089`.
- Official `model/model.py` SHA-256:
  `60ff8ad5e3b35db1700a803c0701269fad12f1e0364a5d2a517e14197391f2c8`.
- License: MIT; license SHA-256:
  `5d6579f2902a45aaeab167ee9e5b16d1a177c4135b833fc3be17545b35d5775f`.

For current-task gradient `g` and reference-memory gradient `g_ref`, A-GEM
keeps `g` when `g^T g_ref >= 0`; otherwise it applies Equation 11:

`g_projected = g - (g^T g_ref / (g_ref^T g_ref)) * g_ref`.

The official source implements this at lines 1156-1186 and applies the
projected gradient through its optimizer. The TRKH A0 below instead uses an
explicit normalized SGD macrostep so AdamW preconditioning cannot invalidate
the measured gradient geometry.

## 3. Adaptation and no-repeat boundary

This is an equation-traceable adaptation, not a reproduction of continual
learning. The current task is supervised correction of verified restricted
head-class negatives; the reference memory is every class-1 fit row.

It is distinct from the closed routes:

- joint spatial PCGrad used objectness/localization gradients and failed after
  a normalized image-model macrostep;
- focus AUC/pAUC, Tversky, margins, OVA, and focus-binary losses blended a
  surrogate into ordinary minibatch training;
- hard mining and rebalancing changed exposure or scalar loss weight;
- post-hoc filters changed decisions without changing representation.

This A0 uses no localization objective, no loss weight, no threshold sweep,
no sampler change, and no optimizer state. It asks only whether the exact
hard-negative direction contains a transferable component that is first-order
non-destructive to all available class-1 fit rows.

## 4. Immutable sources and split

- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher-args SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- `yolo_f/data.yaml` SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- CIDT summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.
- CIDT four-condition prediction SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Dataset rows/source groups: `9215/8064`.
- Source-disjoint fold: hold out fold `0`; fit/holdout rows `7372/1843`.
- Fit/holdout source overlap must be zero.
- Validation and test paths must never be constructed.

The clean fit cohort is fixed before implementation:

- reference memory: all `432` rows with true class `1` in folds `1-4`;
- current hard-negative task: all `186` rows in folds `1-4` whose true class is
  `0`, `2`, or `4` and whose clean keeper prediction is `1`;
- hard-negative class counts: `135/42/9` for classes `0/2/4`;
- holdout restricted false positives: `36` in fold `0`.

No probability cutoff, top-k, class balancing, duplicate exposure, random
subset, or candidate-probability selection is permitted.

## 5. Exact A0 computation

1. Load the keeper strictly and reproduce every clean fold-0 CIDT argmax.
2. Use deterministic keeper evaluation preprocessing, FP32, TF32 disabled,
   deterministic CUDA algorithms, batch `32`, workers `4`, seed `42`.
3. Put the model in evaluation mode and compute row-mean multiclass CE
   gradients over all `186` hard negatives and all `432` class-1 reference
   rows. Accumulate gradients in FP64 and materialize them in FP32.
4. Include every existing trainable model parameter. Missing per-parameter
   gradients are explicit zeros; parameter order and schema are hash-locked.
5. Compute the raw dot product, cosine, norms, A-GEM violation flag, projection
   coefficient, projected dot product, and retained projected norm.
6. Create three bit-identical models: raw keeper, control, and candidate.
7. Control direction is the unprojected hard-negative gradient. Candidate
   direction is the exact A-GEM result. Apply one explicit descent step to each.
8. Normalize each update independently to exactly `1e-4` of the same initial
   trainable-parameter L2 norm. No optimizer, clipping, decay, scheduler, EMA,
   augmentation, teacher, auxiliary objective, or stochastic forward is used.
9. Evaluate raw/control/candidate on all `1843` fold-0 rows under clean, dim,
   bright, and low-contrast transforms.

The macrostep ratio `1e-4` is inherited from the prior normalized spatial
gradient audit. It is locked here only as a local information scale and must
not be swept.

## 6. Structural and equation gates

All checks must pass:

- every source/hash/commit/argument is exact and the official worktree is clean;
- raw clean predictions exactly match CIDT on `1843/1843` rows;
- source-fold integrity, row order, class counts, and cohort counts are exact;
- no validation/test/raw-data write occurs;
- all gradients, logits, probabilities, norms, and losses are finite;
- direct flattened Equation-11 reference and tensor-list implementation agree
  within `1e-7` absolute error;
- `g_ref^T g_ref > 0` and the hard-negative direction is nonzero;
- A-GEM must detect an actual conflict (`g^T g_ref < 0`); an inactive
  projection closes the route as unnecessary;
- projected dot is at least `-1e-8` and projected conflict cosine absolute
  value is at most `1e-6`;
- projected retained norm ratio is at least `0.20`;
- control/candidate initial state hashes and parameter schemas are identical;
- requested/actual update ratios are within `1e-8` of `1e-4`, and update-norm
  mismatch between control and candidate is at most `1e-8` relative;
- candidate first-order class-1 reference loss change is non-increasing, while
  the raw control direction is recorded without reinterpretation.

## 7. Clean fold-0 behavior gates

Candidate must satisfy every gate against the raw keeper, not merely beat a
collapsed control:

- macro F1 delta `>= 0.000`;
- class-1 F1 delta `>= +0.005`;
- class-1 precision delta `>= +0.005`;
- class-1 recall delta `>= -0.005`;
- restricted `0/2/4 -> 1` false-positive reduction `>= 2`;
- class-1 FN rescues are at least class-1 TP breaks;
- total corrections are greater than harms;
- maximum non-class-1 per-class F1 drop `<= 0.010`;
- predicted class-1 support is at least `95%` of raw support.

Candidate versus control must also have class-1 F1 no lower than control and
class-1 TP breaks no greater than control. Precision gained only by reducing
class-1 support or recall is an automatic failure.

## 8. Illumination safety gates

For dim, bright, and low contrast, candidate versus raw keeper must satisfy:

- every class-1 F1 delta `>= -0.010`;
- every class-1 recall delta `>= -0.015`;
- at least two conditions have nonnegative class-1 precision delta;
- aggregate class-1 TP breaks do not exceed FN rescues;
- no condition creates more restricted class-1 false positives than it removes.

## 9. Decision and stop rule

Passing every gate authorizes only a separate, precommitted train-only
multi-macrostep Stage B. It does not authorize validation, test, trainer wiring,
or a full train.

Any failed gate closes this exact class-1-reference A-GEM macrostep. Do not
sweep the step ratio, fold, seed, cohort definition, objective, parameter set,
projection threshold, batch size, optimizer, or number of steps. Do not turn
the result into another post-hoc threshold/router or reuse the prior spatial
PCGrad smoke. The current-best command packet and its three-revision history
remain unchanged unless a later candidate wins the locked validation promotion
protocol.
