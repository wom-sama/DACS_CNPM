# TRKH Bidirectional Confusion-Aware Spectral Readiness

Date: 2026-07-14

## Scope

This is a train-only gradient-readiness gate for a precision-first continuation
of the 2026-07-14 random-initialized scratch complement. It does not modify raw
data, load validation/test predictions, alter the frozen two-member rule, or
authorize a full train.

CIDT found dense scratch/keeper complementarity, but the dim and low-contrast
views retained only `0.5133/0.5852` of keeper clean class-1 true positives and
therefore failed the predeclared label-preservation gate. The next method must
act on the clean boundary and penalize both directions of class-1 confusion,
not distill destructive counterfactual views.

## Research Basis

- CAR, CVPR 2026 Oral: https://arxiv.org/abs/2603.16732
- CVF paper page: https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_Confusion-Aware_Spectral_Regularizer_for_Long-Tailed_Recognition_CVPR_2026_paper.html
- Decoupled long-tail representation/classifier analysis:
  https://arxiv.org/abs/1910.09217
- Balanced Softmax and prior-shift analysis:
  https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html

CAR defines a differentiable off-diagonal confusion matrix, smooths it across
mini-batches with EMA, frequency-weights its true-class columns, and minimizes
its spectral norm. The paper's advertised GitHub URL returned `404` during this
audit, so equations 7-9 were transcribed from the accepted CVPR PDF and covered
by direct unit tests rather than approximated from a secondary implementation.

## Locked Formulation

For logits `z`, true class `j`, competitor class `i`, and batch class count
`m_j`, use the paper's differentiable confusion entry:

```text
C[i,j] = mean_{y=j} sigmoid(gamma + z_i - z_j)
                         * softmax_{k != j}(z_k - z_j)[i]
C[j,j] = 0
```

The EMA is `C_hat_t = beta*C_hat_(t-1) + (1-beta)*C_t`; history is detached and
only the current batch carries gradients. Dataset-frequency weights are
`lambda_j = (frequency_j + r0)^(-1/2)`.

- Paper CAR: `||C_hat * Lambda||_2`.
- Proposed precision-first BiCAR: `||Lambda * C_hat * Lambda||_2`.

The additional left weight makes false positives *into* a rare predicted class
costly, while the paper's right weight continues to protect false negatives
*out of* that true class. This is the only proposed extension; no feature,
sampler, threshold, architecture, or dataset change is part of the gate.

Locked values are the paper ablation defaults/optima: `beta=0.5`, `r0=0.2`,
`alpha=0.5`, `gamma=0.1`. Use batch size `32`, strict balanced epoch sampler,
seed `42`, one sampler epoch, five source-group folds, and natural dataset class
counts `[1941,541,1920,2520,2293]`.

## Locked Inputs

- CIDT full summary:
  `runs/audit_cidt_readiness_full_train_20260714/summary.json`
- CIDT summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`
- The summary must report split `train`, `test_data_used=false`, and
  `validation_predictions_used=false`.
- Only `clean` rows from its complete prediction CSV are used.
- Candidate checkpoint SHA-256:
  `f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549`.

Because every CAR term depends only on logit differences, `log(probability)` is
an exact softmax-equivalent reconstruction for this audit.

## Readiness Measurements

For both keeper and candidate and for paper CAR/BiCAR, record every sampled
occurrence, batch, and source fold. For a gradient-descent direction `d=-g`,
measure the exact local derivative of focus probability:

```text
d p1 = p1 * (d_1 - sum_k p_k*d_k)
```

Report class-1 TP/FN, non-class1 FP into class 1, and other non-class1 cohorts,
plus pair-specific directions, CAR/CE gradient norm ratio, and cosine to CE.

## Full Readiness Gates

All gates are fixed before the formal audit:

1. Exact CIDT summary hash, `9215` ordered clean train rows, exact class counts,
   finite normalized probabilities, complete source folds, and zero fold-source
   overlap.
2. The strict sampler has relative class-exposure gap at most `0.001`, and every
   batch contains all five classes.
3. Candidate BiCAR raises class-1 probability on at least `99%` of class-1
   occurrences and lowers it on at least `99%` of non-class1 occurrences.
4. Mean suppression magnitude on candidate class-1 false positives is at least
   `2.25x` that on other non-class1 rows.
5. Candidate class-1-FN boost magnitude is at least `0.80x` TP-protection
   magnitude.
6. Relative to paper CAR, BiCAR provides at least `3.0x` false-positive
   suppression magnitude and at least `1.15x` false-negative boost magnitude.
7. At paper weight `0.5`, BiCAR gradient norm is between `0.15x` and `0.30x` CE;
   cosine to CE is in `[0.90,0.99]`. This rejects a vanishing, dominating, or
   numerically duplicate regularizer.
8. All five candidate BiCAR source folds pass both `99%` sign checks and have
   false-positive focus ratio at least `1.75`.

## Single Authorized Smoke

Passing every gate authorizes one matched experiment from the exact scratch
candidate checkpoint: one control and one BiCAR run, same seed/order/config,
`40` train batches, one epoch, full `2606`-row validation, and no test. BiCAR
uses only the locked `alpha/beta/r0/gamma` values.

The BiCAR smoke must beat its matched control by class-1 precision `>=0.02`,
keep class-1 recall `>=0.78`, keep macro F1 within `0.002`, create fewer class-1
false positives than it removes, break at most six class-1 true positives, and
have corrections not below harms. It may proceed to one fixed 120-batch/two-
epoch probe only if it also reaches macro/class-1 F1 `>=0.880/0.680` and class-1
precision `>=0.60`.

Failure closes this exact CAR/BiCAR continuation. It does not authorize a sweep
of weight, margin, EMA, smoothing, sampler, seed, LR, checkpoint, or run length.

## Formal Audit Result

The first formal execution exposed an infrastructure defect rather than a
scientific gate failure. `StrictBalancedBatchSampler` rotated a batch's two
remainder slots from `batch_index`, producing epoch exposure
`[1843,1844,1844,1843,1842]` and relative gap `0.0010846`, just above the
predeclared `0.001` limit. The gate was not relaxed. The sampler now rotates
from `batch_index * remainder`, which distributes all epoch remainder slots
sequentially and is regression-tested to keep global exposure within one.

The unchanged protocol was rerun at
`runs/audit_bicar_gradient_readiness_full_train_v2_20260714`. Summary SHA-256 is
`7998a85c9aaea140b4330eb7a4ccdcea493e0303e63a753052ee35a2708602c8`.
All structural and readiness gates passed:

- Exact train-only scope: `9215` rows, no validation/test predictions, exact
  CIDT/checkpoint hashes, five source folds, and zero source overlap.
- Sampler exposure is `[1844,1843,1843,1843,1843]`, relative gap
  `0.0005423`, with all five classes in every batch.
- Candidate BiCAR false-positive focus ratio is `2.4582`; FN/TP protection
  ratio is `0.8838`; all occurrence and fold sign gates pass.
- Relative to paper CAR, BiCAR FP suppression magnitude is `3.7235x` and FN
  boost magnitude is `1.2767x`.
- At paper weight `0.5`, BiCAR gradient norm is `0.2330x` CE and its cosine to
  CE is `0.9545`.

This pass authorizes only the single matched control/BiCAR smoke described
above. It does not promote BiCAR, update current-best commands, or authorize a
parameter sweep.

## Matched Smoke Result

The first launcher attempt failed before data loading because the new config
guard referenced nonexistent `args.use_sam` instead of parser field
`args.sam`. The guard is fixed and regression-tested for both SAM-off and
SAM-on cases. The PowerShell launcher also preserves native Python error
records in `train_native_errors.txt` on failure without restoring the old
native-stderr object pipeline.

The repaired locked run is
`runs/smoke_bicar_locked_40b_fullval_v3_20260714`. It completed exactly 40
state updates, saved a finite `5x5` EMA confusion matrix, completed the five-
class architecture trace, and did not open test. Independent checkpoint
reloads, rather than the in-memory epoch result, are the promotion evidence:

- matched control macro/class-1 F1: `0.875134/0.654639`; class-1 P/R
  `0.535865/0.841060`;
- BiCAR macro/class-1 F1: `0.875354/0.656331`; class-1 P/R
  `0.538136/0.841060`;
- precision gain `0.002271`, below the locked `0.02` requirement;
- only `2/2606` decisions changed: one `2->1` false positive was corrected to
  class 2, while one correct class-3 decision changed to class 2;
- corrections/harms `1/1`, class-1 FP removed/created `1/0`, and class-1
  FN-rescued/TP-broken `0/0`.

The absolute probe gates also fail: macro/class-1 F1 are below `0.880/0.680`
and class-1 precision is below `0.60`. Transition summary SHA-256 is
`ae2b155cbebd73dd0eb69bf4b523ec07843551f64b5d728ca6969ac66c5fe583`.

Paired all-method XAI plus robustness probes were run on both changed sample
indices `54,172`. Attention and rollout barely moved, and both cases remained
near-ties. Candidate Grad-CAM foreground mass fell from `0.9700` to `0.4786`
while border mass rose from `0.1310` to `0.6069`; the harmful `3->2` case
visibly shifted saliency toward upper background/border structure. Object
desaturation probability drop remained about `0.0754`, versus target drops
near zero for background gray/blur. Control/candidate XAI summary hashes are
`a38a0f7b...a16385` and `791ad7cc...e4d50`.

Decision: the exact CAR/BiCAR continuation is closed. Do not sweep loss weight,
margin, EMA, smoothing, sampler, seed, LR, checkpoint, or run length. Do not
run the 120-batch probe or test. The current-best full-train commands remain on
the keeper because this candidate did not pass the locked validation gate.
