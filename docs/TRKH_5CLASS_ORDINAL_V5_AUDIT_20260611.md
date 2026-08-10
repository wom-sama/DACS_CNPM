# TRKH 5-Class Ordinal V5 Audit - 2026-06-11

## Scope

- Main project: `D:\DataAI\AIEx\TRKH`.
- Dataset: `D:\DataAI\AIEx\newdataset\class_f`, 5 classes.
- Target: improve macro F1 and class 1 without unconditional class-1 oversampling.
- Train/val/test remain isolated; no test metric is used to fit or select the ordinal probe.

## V4 bottleneck

- Test macro F1: `0.882251`.
- Class 1 F1: `0.654545`.
- Class 1 precision/recall: `0.5870 / 0.7397`.
- Main confusions: `0->1`, `2->1`, `1->0`, `1->2`.
- XAI showed foreground focus is already useful; remaining errors are mostly maturity boundary, illumination and defect ambiguity.

## Changes

1. Added one ordinal maturity score for classes `0<1<2<3`.
2. Reduced pairwise head to `4-rest`; class 4 remains a separate defect axis.
3. Added `--ordinal-maturity-loss-weight`.
4. Added `--train-scale-crop-probability`.
5. V5 uses scale-crop `0.88-1.0` on 35% of train samples instead of `0.8-1.0` on every sample.
6. Architecture trace now exports the pseudo foreground-mask overlay.

## Leakage-safe ordinal probe

Artifact:

`runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\ordinal_probe_20260611`

Protocol:

- ridge ordinal direction fitted on train embeddings only;
- residual logit scale selected on validation only;
- test evaluated once after selection.

Result:

| Metric | V4 baseline | Ordinal probe |
|---|---:|---:|
| Test macro F1 | 0.8823 | 0.8907 |
| Test class 1 F1 | 0.6545 | 0.6800 |
| Test class 2 F1 | 0.8973 | 0.9077 |
| Test class 3 F1 | 0.9579 | 0.9657 |

This does not prove the full v5 target, but it confirms that the embedding contains a useful common maturity direction.

## Foreground-mask audit

Artifact:

`runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\architecture_trace_mask_audit_20260611`

The mask generally retains the fruit, but can include green cloth/foliage when foreground and background have similar color. A stricter heuristic was not applied because it could remove partial fruits or dark defect regions. V5 keeps token foreground consistency/pruning and audits the mask visually after training.

## Smoke

Run:

`runs\smoke_trkh_ordinal_v5_20260611`

- 2 train batches and 2 validation batches completed.
- Ordinal loss: finite (`0.5586` mean).
- Peak allocated/reserved VRAM: about `5.44 / 6.32 GB`.
- Workers `4/2`, Windows safe mode, no persistent workers or pinned memory.
- Automatic architecture trace completed for 5 classes.
- Smoke validation metrics are not scientifically meaningful because only two sequential validation batches were used.

## Full run

Run name:

`mango_cls_256_5class_ordinal_maskaudit_v5_30e`

Canonical command is stored in:

`D:\DataAI\AIEx\image_baseline_experiments\Train.md`

Early stopping is strict: `patience=3` on the configured `fair_macro_f1` selection metric. Three consecutive non-improving epochs stop the run; `checkpoints/best.pt` remains the evaluation checkpoint.

After completion, export final-test predictions, class-1 confusion audit, XAI/background audit, architecture trace, and a v4-v5 comparison.
