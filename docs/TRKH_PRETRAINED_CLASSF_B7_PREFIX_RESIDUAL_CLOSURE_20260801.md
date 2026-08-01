# B7 prefix-residual alignment closure — 2026-08-01

Protocol: `TRKH_PRETRAINED_CLASSF_B7_PREFIX_RESIDUAL_READINESS_20260731`

Run: `runs/precheck_classf_b7_dinov3_prefix_residual_20260801_run4`

Source: `dd312a48710889ecc6d9422dc54834a047149931`

## Decision

B7 is rejected at its locked readiness gate. The exact 7,345-parameter
prefix-residual readout must not be promoted to implementation, validation,
smoke, full training or test, and must not be rescued by a nearby epoch,
threshold, width or loss sweep.

The run itself is valid. Independent post-validation reconstructed all 8,278
train assignments, 7,751 source groups, 7,192 unique OOF pair keys, 15
pair-fold cells and 729 reported metric values with maximum absolute error
`0.0`. Validation and test were neither constructed nor read.

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Mean pair AUROC gain | >= 0.010 | -0.001462 | fail |
| Positive pair-fold directions | >= 10/15 | 3/15 | fail |
| Pairs with AUROC gain >= 0.010 | >= 2/3 | 0/3 | fail |
| Worst pair AUROC delta | >= -0.005 | -0.011974 | fail |
| Pairs with class-1 F1 gain >= 0.010 | >= 2/3 | 3/3 | pass |
| Aggregate class-1 TP retention | >= 0.990 | 0.907216 | fail |
| Worst class-1 recall delta | >= -0.005 | -0.098592 | fail |
| Aggregate rival FP reduction | >= 10% | 68.389% | pass |
| Critical 2-1 AUROC gain | >= 0.010 | +0.009809 | fail |
| Critical 2-1 FP reduction | >= 10% | 70.528% | pass |
| Critical 2-1 recall delta | >= -0.005 | -0.098592 | fail |
| Candidate-minus-deranged mean AUROC | >= 0.008 | +0.176795 | pass |
| Candidate beats deranged folds | >= 10/15 | 15/15 | pass |

Pair-level effects:

| Pair | AUROC delta | Class-1 F1 delta | Class-1 recall delta | Rival FP reduction | Candidate−deranged AUROC |
|---|---:|---:|---:|---:|---:|
| 0-1 | -0.011974 | +0.142306 | -0.084507 | 65.545% | +0.183282 |
| 2-1 | +0.009809 | +0.149450 | -0.098592 | 70.528% | +0.241456 |
| 4-1 | -0.002222 | +0.040505 | -0.088531 | 71.034% | +0.105648 |

## Mechanism finding

The causal falsification is strongly positive: aligned CLS/register residuals
beat source-disjoint whole-block derangement for every pair-fold. Prefix memory
therefore contains real sample-aligned class-boundary information that global
average patch pooling discards.

The exact B7 decision rule cannot use that information safely. It reduces rival
false positives from 1,142 to 361 and raises class-1 F1 through precision, but
also removes 135 class-1 true positives (`1455 -> 1320`). The differentiable
soft-recall constraint does not guarantee hard recall at threshold `0.5`; the
failure is a calibration/decision-region problem rather than absence of prefix
signal.

## Research consequence

Keep B2 tempered-p0.5 as the development winner. A distinct successor may
reuse the demonstrated prefix signal only through a predeclared grouped,
fit-only risk-control mechanism (for example a Neyman–Pearson/conformal
selective correction with an outer-fold hard class-1 error budget). That would
be a new decision-theoretic route, not a threshold sweep of B7. It must prove
feasibility on train-only nested OOF before any validation access and remain a
tiny distillable/mobile component.

Post-validation artifact:
`runs/precheck_classf_b7_dinov3_prefix_residual_20260801_run4/postvalidation.json`
(`46501bf60467c408119040009b9c02a8bfa899b612abe69a134d8dacf073eab5`).
