# TRKH Raw-Pretrained AIDT Fusion Readiness Audit

Date: 2026-07-12

## Question

Do the two original pretrained AIDT encoders provide a source-safe CNN+ViT
representation that is strong enough to supervise or replace the current
no-pretrain TRKH keeper, without editing raw data, opening test, or training a
new image model?

Primary implementation references:

- ResNet-50 A1 model card:
  <https://huggingface.co/timm/resnet50.a1_in1k>
- ViT-B/16 AugReg model card:
  <https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k>
- Official timm implementation: <https://github.com/huggingface/pytorch-image-models>
- Local competitor source: `D:\DataAI\AIDT\model.py`, `data.py`, and `train.py`.

The local AIDT model concatenates a 2048D ResNet feature and a 768D ViT
feature, then applies `2816 -> 1024 -> 512 -> 5` with BatchNorm, ReLU, and
dropout `0.3/0.2`. This audit reproduces that feature geometry and head, but it
is deliberately a frozen-encoder readiness test. It is not a claim to
reproduce AIDT's end-to-end fine-tuning, batch-8 augmentation, or backbone LR.

## Locked Protocol

- Immutable source view: `class_f/train,val`; test unopened.
- Exact AIDT evaluation pixels: black `SquarePad`, resize to `224x224`, and
  each timm backbone's own resolved normalization.
- Raw pretrained encoders, loaded from the local cache:
  - ResNet state SHA-256:
    `68faebeca4205c83e1a2c40477f8e96214edebb5c1fa4dbd6974ca08c299ac59`.
  - ViT state SHA-256:
    `b1b1a872addf729a42019884ba710fba3ebe4639298c3d3e6c4a46ad2a1e90fd`.
- Extraction uses FP32, batch 128, four workers. A bounded benchmark rejected
  BF16 because changing batch `32 -> 128` changed ViT features by mean absolute
  `0.00475` and maximum `0.26994`; FP32 reduced the corresponding all-feature
  difference to mean `6.0e-6` and maximum below `4.65e-4`.
- Full `class_f` feature rows: train `9215`, validation `2606`; dimensions
  `2816`; class counts `1941/541/1920/2520/2293` and
  `549/151/544/712/650`; source groups `8064/2577`; overlap `0`.
- Strict `class_f crop key -> yolo_f sample_index` remap has complete
  `9215/9215` and `2606/2606` coverage, zero missing rows, and zero duplicate
  teacher keys.
- Remapped input SHA-256:
  - train `4f6d0a8e6acd333c222542a2f951e9e49d2ad12df1f81a884a7fa398325b5ce0`;
  - validation `84f0d0a231026d098a8dcad2d7bc3c5da94fe6dacf70e1097f1e6c1c220a049f`.
- Keeper validation cache SHA-256:
  `5a7be3a0959c9aa2021f90608e7f9af04c2f1f9ec676df99b910e931deb5f8b7`.
- Three matched readouts: ResNet-only, ViT-only, and raw concatenation. All use
  the local AIDT nonlinear head, AdamW, LR `3e-4`, weight decay `0.05`, two
  warmup epochs then cosine, label smoothing `0.05`, sqrt-inverse class
  weights, natural-frequency batches of 512, and 30 epochs.
- Five fixed `StratifiedGroupKFold` folds grouped by case-normalized source
  stem. The single component control is selected only by train OOF
  `0.5*macro_f1 + 0.5*class1_f1`; validation does not select it.
- Every head is RAM-only and discarded. No checkpoint, model, trainable
  manifest, test input, raw-data write, threshold, router, ensemble weight, or
  validation hyperparameter sweep is allowed.

Implementation and retained evidence:

- `trkh/tools/extract_raw_aidt_backbone_features.py`
- `trkh/tools/audit_raw_aidt_fusion_readiness.py`
- `tests/test_extract_raw_aidt_backbone_features.py`
- `tests/test_audit_raw_aidt_fusion_readiness.py`
- `runs/diagnostic_raw_aidt_fusion_readiness_full_fp32_20260712`
- `runs/diagnostic_raw_aidt_blockl2_fusion_readiness_full_fp32_20260712`
- `runs/diagnostic_raw_aidt_fusion_changed_case_review_20260712`

The full feature cache and strict remap are temporarily retained as declared
inputs for the next `class_f object + yolo_f wide-context` audit. They are not
candidate checkpoints or deploy artifacts.

## Protocol Checks

All features and probabilities are finite; sample indices are unique;
probability sums differ from one by at most `2.39e-7`. All five fit/hold source
overlaps and train/validation source overlap are zero. Every training batch
contains every class, with minimum support 14, and every full/fold loss curve
decreases.

Independent reconstruction from both prediction CSVs and the diagnostic NPZ
reproduced every reported macro/class-1 F1 and transition count. All 9/9
payload hashes pass for each full audit. Neither manifest contains a model,
checkpoint, or test payload.

## Raw Fusion Results

Train OOF selects ViT as the component control.

| Split/method | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 |
| --- | ---: | ---: | ---: | ---: |
| train OOF ResNet | 0.866006 | 0.565392 | 0.519409 | 0.541426 |
| train OOF ViT | 0.879585 | 0.601113 | 0.598891 | 0.600000 |
| train OOF raw fusion | 0.878306 | 0.601942 | 0.573013 | 0.587121 |
| validation ResNet | 0.873262 | 0.669291 | 0.562914 | 0.611511 |
| validation ViT | 0.894439 | 0.680556 | 0.649007 | 0.664407 |
| validation raw fusion | 0.901745 | 0.693333 | 0.688742 | 0.691030 |
| validation direct keeper | 0.884073 | 0.608247 | 0.781457 | 0.684058 |

The validation result is attractive in isolation: raw fusion gains
`+0.017672` macro and `+0.006972` class-1 F1 over the keeper. It is not a
promotion result. Fusion loses to the ViT control in class-1 F1 in all five
source folds; aggregate OOF changes are `87/75` corrections/harms and class-1
FN-rescue/TP-break is only `18/32`. Candidate-minus-control FN/FP AUROC changes
from `0.430115` OOF to `0.618950` validation.

| Comparison | Changed | Corrections/harms | FP removed/created | FN rescued/TP broken |
| --- | ---: | ---: | ---: | ---: |
| OOF raw fusion vs ViT | 175 | 87/75 | 38/28 | 18/32 |
| val raw fusion vs ViT | 55 | 32/23 | 9/9 | 12/6 |
| val raw fusion vs keeper | 169 | 109/53 | 46/16 | 13/27 |

Against the keeper, fusion is mainly a high-precision class-1 suppressor. It
removes 46 false positives, but class-1 recall falls by `0.092715`; 27 keeper
true positives are broken while only 13 false negatives are rescued.

Eight locked checks fail: three-fold consistency, OOF macro/class-1 gains,
OOF direction, keeper class-1 gain `+0.01`, class-1 F1 `0.70`, keeper recall,
and FN-rescue safety. `fold_safe_teacher_permission=false`.

## Block-L2 Correction

Raw feature diagnostics found median ResNet/ViT norms `5.32/34.07` and mean
per-dimension standard deviations `0.0602/0.9548`. One mechanism-based
correction was therefore predeclared: independently L2-normalize each branch,
then concatenate them. Every other protocol value remained fixed; no nearby
normalization or fusion weight was tested.

Block-L2 did not repair source transfer. OOF ViT/fusion macro-class1 became
`0.881527/0.606679 -> 0.874501/0.568982`, and fusion again lost class-1 in all
five folds. Validation ViT/fusion became
`0.893018/0.662162 -> 0.898699/0.684211`; fusion recall stayed `0.688742`.
Nine checks fail, including OOF corrections/harms `113/122`. This closes
feature-scale normalization as the explanation for the raw OOF/validation
reversal.

## Visual Review

The retained review selects the eight largest absolute class-1 probability
changes in each of four groups: keeper false positives removed, false positives
created, false negatives rescued, and true positives broken. All 32 rows were
rendered with the exact `class_f` crop plus `yolo_f` source frame and bbox;
label mismatch count is zero.

Pale green color, weak yellow transition, fine spots, and smooth-to-mottled
surface occur in both rescued and broken true class-1 examples. The same cues
also appear in class-0 false positives. Backgrounds vary across orchard foliage,
soil, hands, tables, and neutral floors without defining an action group. This
agrees with the numeric result: pretrained fusion has useful conservative
surface evidence, but not a stable class-1-positive recall signal.

## Decision

Reject raw and block-L2 AIDT fusion as a fold-safe teacher or deployable TRKH
replacement. Do not tune normalization family, branch scale, fusion weight,
head width/dropout, optimizer, folds, thresholds, or a post-hoc router on these
cached features. Do not use the isolated validation gain for KD or sample
weighting; the OOF evidence and direct keeper recall both fail.

This result does not reject a genuinely new paired-view representation. The
next bounded hypothesis is fixed before seeing its output: extract the same
frozen pretrained CNN+ViT representation from a wider YOLO bbox-context crop,
then compare object-only, context-only, and object+context fusion under the same
source-grouped, no-test gate. It directly tests the user's proposed use of
`yolo_f` context while keeping `class_f` as the object view.

Retained manifests:

- raw audit: 9 payloads, `5836068` bytes, SHA
  `e0709072adf600b8973146dbaf552906830bb950146fb70c068e3f2e9acd6626`;
- block-L2 audit: 9 payloads, `5834683` bytes, SHA
  `a6db9f102583f299cf4e8a3b30e1e93151fed3ec0ad186356a9799fd2ecd23db`;
- visual review: 39 payloads, `5057756` bytes, SHA
  `9056b345e59e72af1024825fcc1afac3a01523edf373a4ba16b3e762c647a3bb`.

Guarded cleanup removed eight superseded roots, 95 files and `26639806`
bytes, with observed free gain `26865664` bytes. Manifest:
`runs/cleanup_manifest_20260712_raw_aidt_preflights_and_superseded_review.json`.

Keeper checkpoint and current-best command remain unchanged. Command SHA-256:
`3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.
Closure passed py-compile, compileall, focused tests `16/16`, full pytest
`769/769`, independent CSV/NPZ reconstruction, recursive review and audit
payload hashes, cleanup verification, and retention over 594 run directories
with `blockers=[]`.
