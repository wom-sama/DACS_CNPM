# TRKH Counterfactual Illumination Disagreement Transfer Readiness

Date: 2026-07-14

## Scope

This is a no-fit, no-validation, no-test readiness gate for a possible new
late-member supervision signal. It does not change raw data, thresholds,
ensemble weights, the selected keeper, or the current full-train command.

The hypothesis is narrow: the 2026-07-14 scratch model has useful class-1
recall under controlled illumination views, but unsafe clean precision. Before
training another branch, measure whether same-view keeper/scratch disagreements
on `yolo_f/train` create a dense, source-diverse class-1 rescue cohort together
with explicit keeper-TP and non-class1 false-positive protection cohorts.

## Research basis

- Positive-Congruent Training treats regressions on predictions that the old
  model got right as a first-class objective rather than applying ordinary
  global distillation: https://arxiv.org/abs/2011.09161
- Consistent teacher/student views are important for distillation to behave as
  function matching: https://openaccess.thecvf.com/content/CVPR2022/papers/Beyer_Knowledge_Distillation_A_Good_Teacher_Is_Patient_and_Consistent_CVPR_2022_paper.pdf
- Augmented-only transfer is not automatically helpful, and excessive
  augmentation can hurt the student: https://openaccess.thecvf.com/content/ACCV2022/html/Li_What_Role_Does_Data_Augmentation_Play_in_Knowledge_Distillation_ACCV_2022_paper.html
- Loss-level augmentation consistency can be preferable to indiscriminate
  feature matching under distribution shift: https://openreview.net/forum?id=a1meaRy1bN

CIDT is not a reproduction of those methods. They justify same-view transfer,
negative-flip protection, and a readiness gate before optimization. The local
method remains label-directional: hard train labels decide which teacher action
is rescue, protection, or harm. No model is trusted globally.

## Locked inputs

- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Scratch complement SHA-256:
  `f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549`
- Data: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, split `train` only.
- Focus class: `1`.
- Five `StratifiedGroupKFold` source folds, seed `20260714`; source key is the
  case-folded original image stem, so multi-object rows stay together.
- FP32 inference, identical transformed tensor and metadata for both models.

The auditor may scan validation/test filenames only to prove zero source-stem
overlap. It must not load their labels, pixels, predictions, or metrics.

## Locked counterfactual views

Use the already audited deterministic PIL-space conditions before the normal
keeper evaluation transform:

| Name | Brightness | Contrast |
| --- | ---: | ---: |
| `clean` | `1.00` | `1.00` |
| `lighting_dim` | `0.70` | `0.90` |
| `lighting_bright` | `1.25` | `1.10` |
| `low_contrast` | `1.00` | `0.65` |

Do not sweep factors after inspecting results. The normal object crop, bbox,
valid mask, resize, illumination normalization, and background treatment are
identical between members and must match the packaged eval semantics.

## Directional events

For target `y`, keeper prediction `k`, scratch prediction `s`, and focus class
`c=1`, record all probabilities plus these nonexclusive events:

- `focus_fn_rescue`: `y=c`, `k!=c`, `s=c`.
- `focus_tp_break`: `y=c`, `k=c`, `s!=c`.
- `focus_fp_remove_correct`: `y!=c`, `k=c`, `s=y`.
- `focus_fp_create`: `y!=c`, `k=y`, `s=c`.
- `candidate_correction`: `k!=y`, `s=y`.
- `candidate_harm`: `k=y`, `s!=y`.

The future loss, if authorized, may imitate only label-correct rescue actions.
Keeper class-1 TP and keeper-correct non-class1 rows become preservation
constraints. Wrong-to-wrong changes never become positive targets.

## Full-readiness gates

All gates are declared before the full audit:

1. Exactly `9215` ordered train object rows, four conditions, finite normalized
   probabilities, complete row identity, and no validation/test source overlap.
2. Keeper and scratch class order/evaluation semantics match exactly; all input
   checkpoint and dataset-identity hashes are persisted.
3. For each non-clean condition, keeper clean-correct retention is at least
   `0.80` and keeper clean class-1-TP retention is at least `0.65`. This rejects
   transformations that mostly destroy label evidence.
4. Across non-clean conditions there are at least `30` class-1 rescue events,
   at least `20` unique rescue rows, at least `15` rescue source groups, and no
   one rescue source contributes more than `20%` of rescue events.
5. Non-clean unique rescue rows exceed clean unique rescue rows by at least
   `10`; otherwise the transform does not densify the closed sparse clean route.
6. At least four of five source folds contain class-1 rescue evidence and at
   least three folds have more rescue than TP-break events.
7. The explicit protection data are nontrivial: at least `10` unique
   `focus_tp_break` rows and at least `30` unique `focus_fp_create` rows across
   non-clean conditions. Sparse harm cohorts cannot support a directional loss.

Passing these gates authorizes one fixed trainer smoke only; it does not prove
generalization or permit validation/test tuning.

## Single authorized follow-up

If every full gate passes, add one default-off fork-after-6 CIDT loss with three
terms on the exact same transformed view:

1. true-class1 rescue toward the scratch member only for train-label-correct
   `focus_fn_rescue` rows;
2. focal positive-congruent preservation toward the keeper on class-1 TP;
3. keeper/hard-label preservation on non-class1 rows where scratch creates a
   class-1 false positive.

Run one finite-gradient preflight, then one fixed 40-batch/full-validation/no-
test smoke. No transform, threshold, loss weight, seed, LR, fork, or run-length
sweep is authorized. Failure closes CIDT on these members.

## Full Audit Result

The full audit is retained at
`runs/audit_cidt_readiness_full_train_20260714`; summary SHA-256 is
`d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.
It covered all `9215` train rows under all four conditions, used no validation
or test predictions, found zero split-source overlap, and passed every
structural gate.

The disagreement signal was dense and source-diverse: non-clean conditions had
`379` class-1 rescue events over `285` unique rows/`276` sources, `75` TP-break
events, and `557` newly created class-1 false positives. Rescue exceeded break
in all five source folds and maximum rescue-source share was only `0.0132`.

CIDT nevertheless failed its predeclared transform-label-retention gate. Keeper
clean class-1-TP retention was `0.5133` for dim and `0.5852` for low contrast,
below `0.65`; bright passed at `0.7746`. Clean-correct retention remained
`0.8978/0.9307/0.9178`. Candidate robustness is real, but dim/low-contrast views
destroy too much keeper-positive evidence to be trusted as same-label
distillation targets.

Decision: close this exact CIDT formulation without a trainer smoke and without
changing transform factors after inspection. Preserve the artifact as negative
evidence; do not turn its cohorts into targets, weights, thresholds, or a new
router.
