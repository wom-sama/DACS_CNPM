# B6 class-conditional DeepSets MIL closure — 2026-07-31

Protocol: `TRKH_PRETRAINED_CLASSF_B6_DEEPSETS_MIL_READINESS_20260731`

Run: `runs/precheck_classf_b6_dinov3_classconditional_deepsets_20260731_run2`

## Decision

B6 is rejected. It must not be implemented in the production model, promoted to validation/full training, or followed by a nearby width, loss, epoch, threshold or pair sweep.

The locked train-only comparison completed all five source-grouped folds and failed eight predeclared readiness checks. The main result is not a runtime artifact error: post-validation reconstructed all 8,278 canonical train rows, 7,751 source groups and 7,192 unique OOF pair predictions, then recomputed 591 metric values with maximum absolute error `0.0`.

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Mean pair AUROC gain | >= 0.010 | -0.001037 | fail |
| Positive pair-fold directions | >= 10/15 | 8/15 | fail |
| Pairs with AUROC gain >= 0.010 | >= 2/3 | 0/3 | fail |
| Pairs with class-1 F1 gain >= 0.010 | >= 2/3 | 1/3 | fail |
| Worst class-1 recall delta | >= -0.010 | -0.028169 | fail |
| Aggregate class-1 TP retention | >= 0.980 | 0.981604 | pass |
| Aggregate rival FP reduction | >= 10% | 12.992% | pass |
| Critical 2-1 AUROC gain | >= 0.010 | -0.000534 | fail |
| Critical 2-1 FP reduction | >= 10% | 7.018% | fail |
| Critical 2-1 recall delta | >= -0.010 | -0.028169 | fail |
| Mean placebo AUROC | <= 0.55 | 0.495516 | pass |

Pair-level effects:

| Pair | AUROC delta | Class-1 F1 delta | Class-1 recall delta | Rival FP reduction |
|---|---:|---:|---:|---:|
| 0-1 | -0.002813 | +0.011308 | -0.008048 | 18.095% |
| 2-1 | -0.000534 | -0.009408 | -0.028169 | 7.018% |
| 4-1 | +0.000238 | -0.001746 | -0.014085 | 17.143% |

## Integrity boundary

- Cache arrays remained byte-identical while the manifest was upgraded from schema 1 to schema 2 with source-group count/hash metadata.
- Cached labels equal the canonical dataset labels; every source group belongs to exactly one holdout fold.
- Exactly three unique pairs and 15 pair-fold cells are present; OOF paths, labels, folds, source groups and probabilities passed exact coverage checks.
- Validation and test datasets were never constructed or evaluated.
- The B2 encoder was fitted on the full train split, so this remains matched mechanism evidence, not an unbiased generalization estimate.

## Mechanism finding

The nonlinear patch-distribution readout suppresses some false positives, but does so by shrinking the class-1 region and losing 25 aggregate pair-wise true positives (`1359 -> 1334`). The critical 2-1 boundary shows the same failure already seen in B5: fewer false positives are purchased with an unacceptable class-1 recall loss. Raw patch-distribution statistics therefore do not expose a sufficiently separable recall-safe signal for this tiny specialist.

Keep B2 tempered-p0.5 as the current development winner. The next distinct train-only hypothesis may inspect DINOv3 prefix-memory discrepancy (CLS/register tokens versus deployed patch mean) with an equal-capacity pooled control and explicit recall protection; it must be locked as a new protocol before feature extraction.
