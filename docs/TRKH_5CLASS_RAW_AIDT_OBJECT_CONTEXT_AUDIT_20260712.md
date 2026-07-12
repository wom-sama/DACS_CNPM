# TRKH Raw AIDT Object + Wide-Context Fusion Audit

Date: 2026-07-12

## Question

Can the user's proposed two-view strategy improve the pretrained CNN+ViT
representation: use `class_f` as the clean object view, use `yolo_f` to retain
wide near-background and object position, then learn a paired classifier that
distinguishes useful context from the fruit itself?

This is a frozen-representation readiness audit, not a new deployed model. It
does not edit either dataset, open test, tune a context margin, or write a model
or checkpoint.

## Fixed Geometry and Provenance

- Object view: the retained strict `class_f -> yolo_f sample_index` feature
  mapping from the preceding raw-AIDT audit.
- Context view: the exact YOLO object bbox expanded by a fixed margin ratio
  `0.50` on every side before clipping, then black SquarePad and resize to
  `224x224`. Unclipped crop width/height can therefore reach twice the bbox.
- One margin was declared and run. There is no margin, crop, normalization, or
  fusion-weight sweep.
- Both views use the same frozen pretrained encoders and branch-specific timm
  normalization:
  - ResNet SHA-256
    `68faebeca4205c83e1a2c40477f8e96214edebb5c1fa4dbd6974ca08c299ac59`;
  - ViT SHA-256
    `b1b1a872addf729a42019884ba710fba3ebe4639298c3d3e6c4a46ad2a1e90fd`.
- FP32 extraction, batch 128, four workers. Full context features cover
  train/validation `9215/2606`, source groups `8064/2577`, class counts
  `1941/541/1920/2520/2293` and `549/151/544/712/650`, and source overlap `0`.
- Context rows match the object cache exactly on every sample index, label,
  case-normalized source stem, object index, and class order.
- Strict paired cache order is `[object 2816D | context 2816D]`, total 5632D.
  Its input SHA-256 is
  `025e33e2d5e5a9b3f3d2e6039e1ab8fb6f36ab86d102a7736d69d6a9f2d21a5f`
  for train and
  `8a365e43ff55d7e1ccbbf3063e7fc0e1f4a37db15aeb07950c8884751e7f35e5`
  for validation.
- Keeper validation cache SHA-256:
  `5a7be3a0959c9aa2021f90608e7f9af04c2f1f9ec676df99b910e931deb5f8b7`.

A ten-row balanced geometry review confirmed that margin 0.50 genuinely adds
hands, floor/table, foliage, shadows, neighboring fruit, source scale, and
object position while retaining the selected object. It is not a duplicate of
the `class_f` crop.

Implementation:

- `trkh/tools/extract_raw_aidt_backbone_features.py`
- `trkh/tools/build_paired_aidt_object_context_features.py`
- `trkh/tools/audit_raw_aidt_fusion_readiness.py`
- `tests/test_extract_raw_aidt_backbone_features.py`
- `tests/test_build_paired_aidt_object_context_features.py`
- `tests/test_audit_raw_aidt_fusion_readiness.py`

Retained evidence:

- `runs/diagnostic_raw_aidt_object_context_fusion_readiness_full_fp32_20260712`
- `runs/diagnostic_raw_aidt_object_context_changed_case_review_20260712`

The full audit contains `upstream_feature_provenance.json`, which preserves
all extraction/remap/pairing summaries, input/output hashes, encoder hashes,
and generating code hashes before the reproducible feature matrices were
deleted.

## Locked Readout Audit

The same source-grouped protocol as the object-only audit is reused without
changing any optimization value:

- three branches: object-only, context-only, paired object+context;
- the local AIDT nonlinear head `D -> 1024 -> 512 -> 5` with BatchNorm, ReLU,
  and dropout `0.3/0.2`;
- AdamW LR `3e-4`, weight decay `0.05`, two warmup epochs then cosine,
  label smoothing `0.05`, sqrt-inverse class weights, batch 512, 30 epochs;
- five fixed `StratifiedGroupKFold` source folds;
- the control is selected only from object/context train OOF; validation does
  not select it;
- direct keeper macro/class-1/recall and transition safety remain mandatory.

The object branch is bit-exact to the preceding raw-AIDT fusion candidate on
all train OOF and validation probabilities. This independently validates the
paired runner and proves that any change comes from the context input.

## Results

Object is the OOF-selected control.

| Split/method | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 |
| --- | ---: | ---: | ---: | ---: |
| train OOF object | 0.878306 | 0.601942 | 0.573013 | 0.587121 |
| train OOF context | 0.875488 | 0.574818 | 0.582255 | 0.578512 |
| train OOF paired | 0.878969 | 0.596958 | 0.580407 | 0.588566 |
| validation object | 0.901745 | 0.693333 | 0.688742 | 0.691030 |
| validation context | 0.890328 | 0.686567 | 0.609272 | 0.645614 |
| validation paired | 0.899883 | 0.701389 | 0.668874 | 0.684746 |
| validation keeper | 0.884073 | 0.608247 | 0.781457 | 0.684058 |

Context alone is materially weaker than the object view. Paired fusion gives
only OOF `+0.000662/+0.001445` macro/class-1 F1 over object and improves
class-1 in only two of five folds. Validation reverses the tiny gain:
`-0.001862/-0.006284`, with class-1 recall falling by `0.019868` versus object
and by `0.112583` versus keeper.

| Comparison | Changed | Corrections/harms | FP removed/created | FN rescued/TP broken |
| --- | ---: | ---: | ---: | ---: |
| OOF paired vs object | 207 | 96/93 | 37/44 | 33/29 |
| val paired vs object | 59 | 27/29 | 14/11 | 9/12 |
| val paired vs keeper | 174 | 111/57 | 48/15 | 12/29 |

Paired-minus-object class-1 probability direction looks better on validation
than OOF: FN-versus-FP AUROC `0.582811 -> 0.702590`. OOF remains below the
predeclared 0.60 minimum. Raw object/context cosine dissimilarity is not a
reliability signal at all: FN-versus-FP AUROC is approximately `0.504/0.496`
on train/validation. A validation-only p1 delta therefore cannot become a
router, threshold, sample weight, or distillation target.

## Visual Review

The retained review selects the eight largest paired-minus-object class-1
probability changes in each of four action groups. All 32 rows render the
`class_f` crop, full source+bbox, and margin-0.50 context crop; label mismatch
count is zero.

Hands, floor/table texture, foliage, shadows, neighboring fruit, and object
scale occur in both corrected and harmed groups. Rescued and broken true
class-1 fruit remain visually overlapping in pale-green/yellow surface state.
In several multi-object or extreme-scale examples, context changes confidence
by exposing scene/acquisition cues or another fruit rather than new maturity
evidence. This matches the numeric failure to preserve class-1 recall.

## Decision

Reject fixed margin-0.50 object+context fusion before image-model smoke.
Eleven locked checks fail: fold consistency, required OOF and validation gains,
validation corrections/harms, OOF direction, class-1 `+0.01`, class-1 `0.70`,
keeper recall, and keeper FN-rescue safety. `fold_safe_teacher_permission=false`.

Do not sweep context margin, full-frame layout, branch normalization, feature
scale, fusion weight, head capacity, optimizer, or post-hoc routing on these
caches. This experiment directly tests the proposed `yolo_f`-context plus
`class_f`-object strategy; its failure is specifically that scene/scale cues do
not become transferable class-1-positive evidence.

Retained artifacts:

- full audit: 10 payloads, `5902718` bytes, SHA
  `68775466c0048ca8d35298fcb87726fd96836486a2a93d0e0282527cff6e976c`;
- changed-case review: 39 payloads, `5218363` bytes, SHA
  `fd25ae236992682f62867b7ddff1d03c53fd9d2bf09d3328cff8d20d42aeaab4`.

After provenance preservation, guarded cleanup removed eight cache/preflight
roots, 55 files and `495809819` bytes; observed free gain was `495398912`
bytes. Cleanup manifest:
`runs/cleanup_manifest_20260712_raw_aidt_object_context_cache_compaction.json`.

No test, raw-data edit, model, checkpoint, trainable manifest, current-best
command, or deployment pointer was created or changed.

Closure passed py-compile, compileall, focused tests `23/23`, full pytest
`777/777`, bit-exact object-branch reproduction, independent CSV/NPZ metric
reconstruction, full payload/cleanup checks, and retention over 595 run
directories with `blockers=[]`. Keeper and command hashes remain unchanged.
