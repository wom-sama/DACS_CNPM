# B8 train-only readiness: frozen-DINO risk-controlled `2 -> 1` correction

Protocol: `TRKH_PRETRAINED_CLASSF_B8_FROZEN_DINO_NP_2TO1_20260801`

This protocol is locked before B8 feature extraction. B8 is one bounded,
train-only mechanism audit. It does not construct or read validation/test data,
does not select a deployment model, and cannot be rescued by changing its
score, threshold, variance floor, pair set, fold seed, or pass criteria after
the result is known.

## Scientific question

B7 proved that DINOv3 CLS/register memory is sample-aligned, but its learned
binary rule reduced rival false positives by shrinking the class-1 decision
region: class-1 TP retention was only `0.907216`. B8 asks a narrower question:
can an independently fixed representation identify actual B2 `2 -> 1` errors
while a predeclared source-group tolerance rule protects class 1?

B8 targets only rival class 2. Pairs `0-1` and `4-1` are out of scope because
the train-only B7 evidence and recall-constrained oracle check did not support
them. B8 is a new selective-correction mechanism, not a B7 threshold sweep.

## Independent frozen representation

- Backbone: untouched `vit_small_patch16_dinov3.lvd1689m` from timm/Hugging
  Face revision `3bf4720a82ec2066db88137180ff1f83a675cef0`.
- Required local weight SHA-256:
  `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`.
- The file is hash-verified before and after an offline strict load. All
  backbone parameters remain frozen and inference is FP32/eval mode.
- Input preprocessing is the already fixed B2 evaluation transform at
  `256x256`; there is no train augmentation, learned preprocessing, PCA,
  projection, or descriptor choice.
- The encoder must expose one CLS token, four register tokens, 256 patch
  tokens, and width 384. Let `m` be the arithmetic patch-token mean and let
  `LN` be per-token LayerNorm without affine parameters, `eps=1e-5`:

```text
z_cls = LN(cls - m)
z_j   = LN(register_j - m), j=1..4
d     = concat(z_cls, mean_j(z_j), variance_j(z_j))  # 1,152 values
```

  Register variance is population variance. The descriptor contains no B2
  logit or feature: B2 was fine-tuned on all train groups and therefore cannot
  enter the score or its calibration.

## Locked outer-fold score and calibration

- Data view: the exact canonical train split in
  `configs/class_f_5class_dev.yaml`: 8,278 crops and 7,751 normalized source
  groups. Validation/test datasets are never built or enumerated.
- Reuse the existing five global `StratifiedGroupKFold` assignments, seed
  `20260731`, after verifying row paths, labels, groups, assignments, B2 logits,
  checkpoint and cache hashes.
- In each outer fold, fit a diagonal Gaussian using only outer-fit class-2
  crops. Each source group has total weight one. Exclude every class-2 source
  group also present among the fold's class-1 calibration groups.
- Mean, variance, and normalization are fit only on those class-2 rows. Lock
  every variance denominator to `max(population_variance, 1e-4)`.
- For descriptor `d`, the fixed class-2 conformity score is

```text
s(d) = -mean_k((d_k - mu_k)^2 / max(var_k, 1e-4))
```

  Larger values mean more class-2-like.
- Calibration uses every outer-fit class-1 source group and never updates the
  score model. Its group score is the maximum crop score. The threshold is the
  maximum calibration-group score. The action comparison is strictly `s > t`.
- The correction is identity except when all three conditions hold: B2 top-1
  is class 1, B2 runner-up is class 2, and `s > t`. Then and only then is the
  prediction changed from class 1 to class 2. B2 logits are used only for this
  action mask and effect accounting.

With at least 373 independent calibration source groups, the maximum order
statistic supports the per-fixed-rule tolerance statement
`Pr(R_source > 0.008) <= 0.05` under source exchangeability and no distribution
shift. The claim is per fold/rule, not simultaneous across five folds. Crop
recall and source-group any-veto risk are reported separately. Because B2
itself was fit on all train rows, correction benefit remains train-conditional
mechanism evidence, not an unbiased generalization estimate.

## Locked controls

Both controls repeat the same folds, class-2-only fitting, exclusions,
calibration, strict threshold, and action rule.

1. Zero residual replaces all 1,152 descriptor values by zero.
2. Whole-block derangement substitutes one complete descriptor from another
   normalized source within the same label and fit/hold partition. The mapping
   is bijective, has no fixed row/source assignment, preserves descriptor
   marginals, and never crosses an outer-fold boundary. Its construction does
   not inspect B2 correctness or score values.

The aligned fit and threshold artifacts are serialized and hashed before the
outer holdout action/effect metrics are computed. Controls are calibrated
independently; an identity result remains a valid control result.

## Readiness gate

All integrity checks must pass: exact data/model/B2-cache provenance, complete
row/group/fold coverage, original-DINO runtime token contract, finite FP32
features/scores, class-2 overlap exclusions, at least 373 class-1 calibration
groups in every fold, immutable fit/threshold hashes, source-disjoint control
maps, and explicit evidence that validation/test were not constructed or read.

The aligned route must additionally satisfy all of the following:

- class-1 recall delta `>= -0.005` in every outer fold;
- aggregate class-1 TP retention `>= 0.990`;
- correct at least `14/137` actual B2 `2 -> 1` train errors;
- correct at least one actual B2 `2 -> 1` error in at least `4/5` folds;
- aggregate `2 -> 1` false-positive reduction exceeds each control by at least
  `0.05` absolute;
- aligned reduction exceeds each control in at least `4/5` folds;
- no new `0 -> 1` or `4 -> 1` error and no change outside the locked `1 -> 2`
  action domain.

Passing licenses only a separate default-off teacher/high-accuracy prototype.
The extra DINO forward is not mobile-ready. Sharing or distilling its score
invalidates the calibration claim and requires a new protocol plus independent
calibration evidence. Failure closes this exact prefix risk-routing family:
do not relax `alpha`, select a lower order statistic, flip the energy sign,
alter the variance floor, add other class pairs, or sweep nearby variants.

## Planned artifacts and commands

The inert launcher first writes only a preflight report:

```powershell
& scripts\run_trkh_pretrained_classf_b8_frozen_dino_np_2to1_readiness.ps1
```

Only after preflight review may the one bounded train-only extraction/audit be
run explicitly:

```powershell
& scripts\run_trkh_pretrained_classf_b8_frozen_dino_np_2to1_readiness.ps1 -Run
```

The run must retain the preflight, descriptor cache manifest, fold fit and
threshold hashes, row-level OOF actions, fold/aggregate metrics, controls,
summary, and an independent post-validation artifact.
