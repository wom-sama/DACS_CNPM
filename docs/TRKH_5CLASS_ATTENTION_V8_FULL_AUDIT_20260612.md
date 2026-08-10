# TRKH 5-Class Attention V8 Full Audit - 2026-06-12

## Run

- Run: `runs/mango_cls_256_5class_attention_views_bounded_v8_30e`
- Dataset: `D:\DataAI\AIEx\newdataset\class_f`
- Scratch/no external pretrained; resumed TRKH v4 weights with optimizer/scheduler reset.
- Batch `32`, gradient accumulation `2`, workers `4/2`, BF16.
- Maximum `30` epochs, early stopping patience `3`.
- Completed normally with launcher exit code `0`.
- Best epoch `13`; stopped at epoch `16`.
- Total training time: `3353.8 s` (`55.9 min`).

The run did not hang. GPU utilization drops seen around worker startup, validation,
checkpoint writing and architecture tracing, while steady-state training reached
roughly `74%-91%`.

## Test comparison

| Run | Accuracy | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 |
|---|---:|---:|---:|---:|---:|
| v4 hard-negative/maskfix | 0.9234 | 0.8823 | 0.5870 | 0.7397 | 0.6545 |
| v7 offline distillation | 0.9219 | 0.8801 | 0.5745 | 0.7397 | 0.6467 |
| v8 bounded attention views | **0.9274** | **0.8869** | **0.6000** | 0.7397 | **0.6626** |
| Top-5 pretrained TTA ensemble | about 0.95 | 0.9248 | - | - | 0.7786 |

V8 is the best single TRKH checkpoint so far, but the gain is small. Class 1
recall is unchanged; the improvement comes from reducing false positives.

## Class 1 confusion

Corrected audit:

`runs/mango_cls_256_5class_attention_views_bounded_v8_30e/class1_confusion_audit_test_v2`

- TP `54`, FP `36`, FN `19`.
- `0 -> 1`: `21`.
- `2 -> 1`: `9`.
- `1 -> 2`: `10`.
- `1 -> 0`: `6`.
- `1 -> 4`: `3`.

The first audit attempt used the compact baseline `predictions.csv`. That CSV
contains numeric indices in a sorted baseline order, while `data.yaml` contains
the canonical class order. `audit_class_confusions` incorrectly preferred the
numeric columns over class names and reported class 2 as class 1. The tool now
prefers dataset-backed name mapping, has regression tests, and the valid audit
uses `predictions_detailed.csv`.

## Manual sample review

Representative reviewed samples show real boundary ambiguity:

- Class 1 predicted as class 2: several fruits are already strongly yellow/ripe.
- Class 1 predicted as class 0: several fruits remain uniformly green with very
  little visible risk signal.
- Class 0 predicted as class 1: wrinkling, yellow tips or local discoloration
  visually resemble the class-1 definition.
- Bright sunlight, deep shadows, leaves, dirt and partial fruit views add error.
- Some class 3/4 examples visually resemble defect-heavy neighboring classes,
  so a label-policy audit is required before assuming every error is learnable.

No test image is reused for training or hard-sample mining.

## XAI and background audit

Output:

`runs/mango_cls_256_5class_attention_views_bounded_v8_30e/xai_audit_test_v2`

Audit over `36` difficult/low-margin cases:

| Method | Foreground mass | Background mass | Border mass |
|---|---:|---:|---:|
| Raw/fine-grained attention | 0.7221 | 0.2779 | 0.3619 |
| Attention rollout | 0.8901 | 0.1099 | 0.2058 |
| Grad-rollout | **0.9164** | **0.0836** | **0.1828** |
| Stem Grad-CAM | 0.8664 | 0.1336 | 0.2210 |

Robustness probability drop:

- Background blur: `0.0004`.
- Background gray: `0.0001`.
- Center occlusion: `0.0164`.
- Object desaturation: `0.1157`.

The model is not materially dependent on the background. Its dominant signal is
fruit color/texture. Raw attention is diffuse and often follows borders, while
class-specific Grad-rollout/Grad-CAM localizes fruit surface much better.

## Architecture trace

The automatic trace completed for one random train sample per class:

`runs/mango_cls_256_5class_attention_views_bounded_v8_30e/architecture_trace`

It includes original/preprocessed images, foreground mask, stem activation,
patch/detail maps, token shapes after each block and pruning stage, and bounded
attention crop/drop. Drop area is approximately `6%` as configured.

Visual inspection found that the current view score can still peak on image
borders or green background with fruit-like color. Therefore the next
controlled change is a configurable surface-detail score that uses local
high-frequency/edge evidence gated by the foreground prior, instead of relying
on raw fine-grained attention alone.

## Train loss above validation loss

This remains expected, not evidence of a broken validation loop:

- Train loss includes primary classification, attention-view CE, metric loss,
  foreground consistency and pairwise losses.
- Train images use harder augmentation and attention crop/drop.
- Validation uses the clean primary view/loss.

After attention views start at epoch 2, train loss jumps from `1.025` to `1.817`
and then decreases to `1.284`; validation loss decreases from `1.056` to about
`1.000`. The curves are consistent with the configured objectives.

## Decision

- Keep v8 as the best scratch TRKH checkpoint so far.
- Do not increase class-1 oversampling globally.
- Do not strengthen background suppression blindly; perturbing the background
  has negligible effect.
- Replace the attention-view score source with an auditable surface-detail
  option and test it as a controlled v9 ablation.
- Separately create a train/validation-only ambiguous-boundary review manifest.
  Test labels remain audit-only and must not drive training changes.
