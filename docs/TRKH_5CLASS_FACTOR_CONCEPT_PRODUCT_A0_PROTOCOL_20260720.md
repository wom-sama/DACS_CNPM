# TRKH 5-Class Factor-Concept Product A0 Protocol - 2026-07-20

## Status

Prospectively locked before auditor implementation or inspection of any new
candidate metric. This protocol authorizes one source-disjoint, train-only
information audit. It does not authorize validation/test access, image-model
training, a checkpoint, a trainable manifest, a threshold or coefficient
sweep, full training, or current-best command promotion.

## Research Question

Can the five agricultural classes be represented as the conjunction of two
observable factors while preserving true class-1 support?

- maturity state: `{0,1} | {2,3} | {4}`;
- transport/damage state: `{0,2} | {1} | {3,4}`.

The two codes uniquely identify all five classes. In particular, class 1 is
the sole valid conjunction of maturity state 0 and transport/damage state 1.
The proposed mechanism therefore requires both positive factors and permits
either factor to provide class-conditional negative evidence. This differs
from the rejected semantic-attribute loss, which applied grouped log-sum-exp
losses to one shared five-class logit vector and did not create independent
factor representations or a conjunctive decision.

The fixed surface signal is the signed difference between frozen keeper
interior-core and bbox-boundary-ring second-order descriptors. A positive A0
would justify one later two-stream patch-attention concept head. A negative A0
closes this factor code and signed descriptor on the current keeper before any
trainer or model change.

## Primary-Source Lock

- Accepted paper: Koh et al., "Concept Bottleneck Models," ICML 2020:
  `https://proceedings.mlr.press/v119/koh20a.html`.
- Local accepted paper:
  `D:/DataAI/external_sources/papers/concept_bottleneck_icml2020.pdf`.
- Paper SHA-256:
  `250cf7d6fe86575c9a3100983441134e7f407e32d064a166d674b852a48e9b4b`.
- Official authors' repository:
  `https://github.com/yewsiang/ConceptBottleneck`.
- Locked commit/tree:
  `d6353f270702b92feb5b084a6fd065f891d583f8` /
  `d93ca72552f8d6e405a8c44d8b9bdf70d9af4eb9`.
- Repository MIT license SHA-256:
  `dbd16fda64f9c246a6f33a9aa449c9313806f1cfa26329c1ce51d703fcc4ccde`.
- Reviewed model/training-file SHA-256:
  `fa9939dde51fc5e6eec1cc5aea6ac357cf8d92e3eff1c388b4eeee3316e76d84` /
  `ad71ce78fa318e58358e4c4bf661baad0b17872b5a994db6b886fd41522d1f26`.

The paper and official source establish the X-to-concept-to-label bottleneck,
separate concept predictors, sequential/joint training choices, and concept
intervention semantics. TRKH does not copy the old Inception implementation,
weights, CUB attributes, optimizer recipe, or benchmark numbers. The signed
surface descriptor, two mango factor codes, and deterministic product decoder
are TRKH-specific hypotheses implemented with current local dependencies.

## No-Repeat Boundary

- The 2026-07-02 semantic-attribute smoke at weight `0.02` reached only
  validation macro/class-1 F1 `0.8796/0.6685` and widened class-1 FP. It used
  grouped losses on the existing logits; it did not test independent factor
  heads, spatially separated evidence, or a product decoder.
- Ordinal, CORN, NBDT, class-axis energy, GMM, prototype, metric, pairwise,
  WILDCAT/MCL/MIL, class-query, interior-only second-order, and generic
  hierarchy routes are closed. A0 may not tune or combine any of them.
- The retained second-order cache showed that standalone core descriptors and
  their five-class concatenation were weak. A0 does not reopen their rank,
  erosion, covariance, or classifier settings. It tests only the predeclared
  signed core-minus-ring statistic under a distinct factor-product objective.
- Raw images, labels, split files, YAML files, and retained evidence are
  immutable.

## Locked Inputs

- Pre-protocol HEAD/upstream: `acd9fa5`.
- Train descriptor cache:
  `runs/diagnostic_interior_secondorder_rank24_erode020_full_train_val_20260711/train_interior_second_order_descriptors.npz`.
- Train cache SHA-256:
  `917bdbd5cd83861f458df551f5aa9dbba4f88a13eacb8282e1769427a16c1c4e`.
- Parent summary SHA-256:
  `71437ec00935dbd64c77d287d13a67417afc4d7fbff816e6c373657401cfaaf7`.
- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Raw `yolo_f/data.yaml` SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Current-best command/history SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

The auditor must verify every lock before opening the NPZ. It may read only
the train cache. It must reject any cache path or embedded image path that
contains a validation or test split marker. No image pixels are needed.

## Locked Cohort And Folds

- All `9,215` ordered `yolo_f/train` object rows.
- Exact class counts: `[1941, 541, 1920, 2520, 2293]`.
- Expected source groups: `8,064`, from lower-cased source stems already stored
  in the immutable cache.
- Five `StratifiedGroupKFold` folds, shuffle enabled, seed `20260720`.
- Every fit/holdout pair must have zero source overlap and all five classes.
- Candidate/control OOF rows must cover every sample exactly once and retain
  original sample order.

## Fixed Equations

For each fold, fit independent `StandardScaler` objects on fit rows only:

`h = scale(head)`

`s = scale(core_second_order - ring_second_order)`

No absolute value, concatenated core/ring, context term, interaction expansion,
PCA, feature selection, or nonlinear kernel is allowed.

All logistic regressions use scikit-learn multinomial cross entropy,
`solver=lbfgs`, `C=0.3`, `max_iter=1000`, balanced class weights, and seed
`20260720`. No convergence retry or hyperparameter fallback is allowed.

### Matched five-class control

Fit one five-class readout on `[h,s]`. Its OOF probability is `p_control`.

### Fixed factor candidate

Fit one three-state maturity head on `h` and one three-state transport/damage
head on `s`. For class `c`, define:

`z_factor[c] = log p_maturity[m(c)] + log p_transport[t(c)]`.

Apply one softmax over the five valid class codes to obtain `p_factor`. No
class prior, residual class head, keeper blend, temperature, threshold, or
post-hoc calibration is allowed.

### Surface-specific placebos

Fit transport/damage heads with the same recipe on `scale(core + ring)` and
`scale(context_second_order)`. These heads are information placebos only; they
cannot contribute to the candidate prediction. They test whether signed
interior-versus-boundary evidence, rather than generic second-order energy or
context, supplies the factor signal.

## Required Measurements

Persist one OOF row per sample with target, source, fold, keeper/control/factor
predictions and probabilities, both factor posterior vectors, both placebo
transport posterior vectors, class-1 deltas, transport log odds, and all
transition flags.

Report:

1. keeper/control/factor accuracy, macro F1, per-class precision/recall/F1,
   predicted support, confusion matrix, NLL, Brier score, and ECE;
2. factor versus control and factor versus keeper corrections/harms, class-1
   FN rescue/TP break, and restricted `{0,2,4} -> 1` FP removal/creation;
3. all metrics and transition counts per held source fold;
4. maturity/transport confusion and per-state metrics;
5. transport-state-1 log-odds distributions for keeper class-1 TP/FN,
   restricted class-1 FP, and other rows;
6. AUROC/AP for separating keeper class-1 TP from restricted FP with signed
   transport log odds, and AUROC for separating keeper FN from restricted FP
   with `p_factor[1] - p_control[1]`;
7. the same TP-versus-FP AUROC for sum/context placebos, scaler/fold hashes,
   convergence iterations, runtime, peak memory, and artifact hashes.

## Conjunctive A0 Gates

All structural and mechanism gates are required.

Structural gates:

- all source/input/current-best hashes match;
- exact row order, support, class order, dimensions `256/324/324/324`, finite
  normalized probabilities, and complete five-fold OOF coverage;
- zero fit/holdout source overlap in every fold;
- every fixed readout converges without retry;
- validation/test pixels, labels, predictions, and metrics remain unopened;
- compile, focused tests, exact CSV replay, `git diff --check`, and artifact
  manifest verification pass.

Mechanism gates:

- factor macro F1 is no more than `0.002` below matched control;
- factor class-1 F1 gains at least `0.010` over control and is at least `0.80`;
- factor class-1 precision gains at least `0.015` over control;
- factor class-1 recall loses no more than `0.005` versus control;
- factor removes at least `10` net restricted FP versus control;
- factor corrections are at least harms, and class-1 TP breaks do not exceed
  FN rescues;
- maximum F1 loss among classes `0,2,3,4` is at most `0.010`;
- signed transport TP-versus-restricted-FP AUROC is at least `0.65` and at
  least `0.02` above both sum and context placebos;
- FN-versus-restricted-FP direction AUROC from factor-minus-control class-1
  probability is at least `0.62`;
- at least four folds have non-worse class-1 precision, at least four have
  positive restricted-FP net removal, and no fold loses more than two
  class-1 true positives net.

No aggregate gain can compensate for a failed gate. Precision obtained by
broad class-1 contraction is a failure.

## Escalation And Stop Rules

If any A0 gate fails, close this factor code, signed descriptor, product
decoder, concept-prior/temperature/threshold variants, and nearby factor-head
sweeps on the current keeper. Do not inspect validation, generate XAI, modify
the model/trainer, run a smoke/probe/full train, or update current-best
commands.

Only a complete A0 pass authorizes one zero-initialized two-stream patch
concept head with separate maturity and transport/damage attention maps. That
implementation must preserve the default model bit-exactly, strict-load the
keeper, expose factor-specific XAI, pass FP32/BF16/export/resource checks, and
receive one matched two-epoch/120-batch full-validation smoke. A passing smoke
may authorize one short probe; only a prospectively passing probe may authorize
one autonomous full train of at most 30 epochs with patience 3 and freshly
verified workers `4/2`.

Current-best checkpoint, command packet, and update history remain unchanged
unless that complete promotion sequence produces a genuine winner.
