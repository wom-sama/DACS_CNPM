# TRKH 5-Class ELR V10 Audit - 2026-06-12

## Change

V10 adds optional Early-Learning Regularization (ELR) for classification-only
training.

- CLI flags: `--elr-loss-weight`, `--elr-beta`, `--elr-start-epoch`.
- Default is off: `--elr-loss-weight 0`.
- When enabled, the train dataset is wrapped by `IndexedSampleDataset` after
  train-only rare/hard repeat wrappers and before the sampler.
- The ELR target history is updated only from train batches. Validation and test
  samples are never used to update target history.
- `train_elr_loss` is written to `history.csv`.

Intent: reduce memorization of ambiguous/boundary labels around class 0/1 and
1/2 without increasing class-1 oversampling.

## Code Validation

- `py_compile` passed for `train.py`, `dataset.py`, `config.py`, and the test file.
- Focused ELR/index tests passed.
- Mini one-batch train with ELR enabled passed.
- Full `tests/test_detection_calibration.py`: `98 passed`.

## Smoke

- Run: `runs/smoke_mango_cls_256_5class_surface_detail_elr_v10_20260612`
- Exit code: `0`
- Trace completed with one sample per class.
- ELR state shape: `[9524, 5]`, train-only after hard-sample repeat.
- `history.csv` recorded `train_elr_loss = -0.2485`.

## Probe

- Run: `runs/probe_mango_cls_256_5class_surface_detail_elr_v10_120b_6e_20260612`
- Setup: resume V8 best, surface-detail attention views, ELR weight `0.10`,
  beta `0.70`, start epoch `2`, `120` train batches per epoch, full validation,
  patience `3`.
- Exit code: `0`.
- Stop reason: early stopping at epoch `4`; best epoch `1`.
- Total time: `517.8s`.

Best validation:

| Metric | Value |
| --- | ---: |
| Accuracy | `0.9229` |
| Macro F1 | `0.8862` |
| Class 1 precision | `0.6073` |
| Class 1 recall | `0.7682` |
| Class 1 F1 | `0.6784` |

Final test from best checkpoint:

| Metric | Value |
| --- | ---: |
| Accuracy | `0.9313` |
| Macro F1 | `0.8923` |
| Class 1 precision | `0.6279` |
| Class 1 recall | `0.7397` |
| Class 1 F1 | `0.6792` |

Class-1 confusion audit:

- Output: `runs/probe_mango_cls_256_5class_surface_detail_elr_v10_120b_6e_20260612/class1_confusion_audit_test`
- Class-1 TP/FP/FN: `54 / 32 / 19`.
- Main false positives: `0->1` = `20`, `2->1` = `8`.
- Main false negatives: `1->2` = `10`, `1->0` = `6`, `1->4` = `3`.

## Decision

Do not run full V10.

Reason: the current full-train gate is class-1 validation F1 `>=0.70`. V10
reached only `0.6784`, almost identical to V9 (`0.6783`). ELR is implemented and
kept as an ablation, but this setting does not justify a full 30-epoch run.

## Next Direction

The repeated pattern is not solved by more regularization alone:

- Class 1 is still over-predicted for class 0/2 boundary samples.
- Class 1 false negatives still go mostly to class 2 and class 0.
- Background suppression appears stable; remaining error is fruit-surface
  maturity/label boundary.

Next high-value work:

- Review copied class-1 audit images and create a train-only
  `ambiguous_boundary.csv` for samples that are label-ambiguous.
- Try a train-only boundary-aware loss or sample weighting based on audited
  ambiguous groups, not global class-1 oversampling.
- If staying no-pretrain, prioritize a stronger fine-grained architecture block
  that compares paired local surface/color statistics for `0/1`, `1/2`, and
  `2/3` boundaries.
