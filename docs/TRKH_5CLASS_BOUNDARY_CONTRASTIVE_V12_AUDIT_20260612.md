# TRKH 5-Class Boundary Contrastive V12 Audit - 2026-06-12

## Context

The previous no-pretrain probes plateaued below the class-1 validation gate:

| Run | Val Macro F1 | Val Class 1 F1 | Test Macro F1 | Test Class 1 F1 |
| --- | ---: | ---: | ---: | ---: |
| V9 surface detail | `0.8857` | `0.6783` | `0.8894` | `0.6708` |
| V10 ELR | `0.8862` | `0.6784` | `0.8923` | `0.6792` |
| V11 boundary sample weight | `0.8856` | `0.6745` | `0.8923` | `0.6792` |

The repeated error pattern is still boundary/label/surface-detail dominated:

- false positives into class 1 mainly `0->1`;
- false negatives from class 1 mainly `1->2` and `1->0`;
- visual review shows dark dirt/sap marks, strong shadows, yellow maturity shift,
  and partial/dirty fruit; this is not solved by stronger background filtering alone.

## Research Used

- Pairwise Confusion for FGVC: https://arxiv.org/abs/1705.08016
  - Supports explicitly handling inter-class similarity in fine-grained data.
- API-Net attentive pairwise interaction: https://arxiv.org/abs/2002.10191
  - Supports comparing highly confused classes to expose contrastive clues.
- PMG multi-granularity: https://arxiv.org/abs/2003.03836
  - Supports local/multi-granularity patch features for subtle surface cues.
- WS-DAN: https://arxiv.org/abs/1901.09891
  - Supports attention crop/drop, but only if the attention map focuses on object
    surface instead of uncontrolled background.
- AugMix: https://arxiv.org/abs/1912.02781
  - Supports robustness to brightness/corruption shifts.
- Random Erasing: https://arxiv.org/abs/1708.04896
  - Supports robustness to occlusion, but this project keeps erasing low/off
    because fruit damage cues can be small and easy to erase accidentally.

## V11 Change: Train-Only Boundary Sample Weighting

Implemented:

- `trkh.tools.build_boundary_sample_weights`
- `SampleWeightDataset`
- weighted classification loss path with per-sample loss support
- `--sample-weight-manifest`, `--sample-weight-factor`, `--sample-weight-max`
- `train_sample_weight_mean` in history
- wrapper `scripts/run_trkh_5class_boundary_weight_v11.ps1`

Manifest:

- `runs/boundary_sample_weights_v11_train_only_20260612/sample_weights_train_only.csv`
- built from train-only detailed predictions:
  `runs/mango_cls_256_5class_defectstat_v3_30e/train_maskfix_for_hard_mining/predictions_detailed.csv`
- `459` weighted train samples
- `skipped_non_train_rows=0`
- boundary pair counts: `0-1=233`, `1-2=70`, `2-3=98`, `4-rest=58`

V11 probe:

- run: `runs/probe_mango_cls_256_5class_boundary_weight_v11_120b_6e_20260612`
- exit code: `0`
- early stopped at epoch `4`, best epoch `1`
- val macro F1 `0.8856`, val class-1 F1 `0.6745`
- test macro F1 `0.8923`, test class-1 F1 `0.6792`
- decision: no full train, below class-1 val F1 gate `0.70`

## V12 Change: In-Batch Boundary Contrastive Loss

Implemented:

- `--boundary-contrastive-loss-weight`
- `--boundary-contrastive-pairs`
- `--boundary-contrastive-sources`
- `--boundary-contrastive-margin`
- `--boundary-contrastive-temperature`
- `--boundary-contrastive-max-pairs`
- `train_boundary_contrastive_loss`
- `train_boundary_contrastive_terms`
- wrapper `scripts/run_trkh_5class_boundary_contrastive_v12.ps1`

Design:

- no offline pair dataset is created;
- uses only current train batch embeddings;
- default pairs: `0-1,1-2,2-3,4-rest`;
- default sources: `head,patch`;
- for each boundary pair, each anchor compares same-class positives against the
  hardest opposite-side negative;
- `max_pairs` caps anchor terms per source/pair, preventing pair explosion.

This directly addresses the previous slow/hang risk from pair generation.

## Validation

Commands run:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m py_compile trkh\training\train.py trkh\core\config.py trkh\data\dataset.py trkh\training\losses.py trkh\tools\build_boundary_sample_weights.py tests\test_detection_calibration.py
D:\DataAI\.venv\Scripts\python.exe -m pytest tests\test_detection_calibration.py -k "boundary_contrastive or sample_weight or early_learning"
D:\DataAI\.venv\Scripts\python.exe -m pytest tests\test_detection_calibration.py
powershell -ExecutionPolicy Bypass -File scripts\run_trkh_5class_boundary_contrastive_v12.ps1 -PreflightOnly
powershell -ExecutionPolicy Bypass -File scripts\run_trkh_5class_boundary_contrastive_v12.ps1 -Smoke -RunName smoke_mango_cls_256_5class_boundary_contrastive_v12_20260612
```

Results:

- compile passed
- focused tests: `3 passed`
- full focused test file: `100 passed`
- preflight passed
- smoke passed
- architecture trace completed:
  `runs/smoke_mango_cls_256_5class_boundary_contrastive_v12_20260612/architecture_trace`
- smoke history confirmed:
  - `train_boundary_contrastive_loss=0.8235`
  - `train_boundary_contrastive_terms=143.0`
  - `train_sample_weight_mean=1.0586`

## V12 Probe

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_trkh_5class_boundary_contrastive_v12.ps1 -Probe -RunName probe_mango_cls_256_5class_boundary_contrastive_v12_120b_6e_20260612
```

Run:

- exit code: `0`
- total seconds: `769.2`
- last epoch: `6`
- stop reason: `early_stopping`
- selection best epoch: `3`
- architecture trace completed:
  `runs/probe_mango_cls_256_5class_boundary_contrastive_v12_120b_6e_20260612/architecture_trace`

Best validation by selection metric:

| Metric | Value |
| --- | ---: |
| Macro F1 | `0.8866` |
| Class 1 precision | `0.6203` |
| Class 1 recall | `0.7682` |
| Class 1 F1 | `0.6864` |

Final test from best checkpoint:

| Metric | Value |
| --- | ---: |
| Accuracy | `0.9321` |
| Macro F1 | `0.8943` |
| Class 1 precision | `0.6429` |
| Class 1 recall | `0.7397` |
| Class 1 F1 | `0.6879` |

Detailed output:

- predictions:
  `runs/probe_mango_cls_256_5class_boundary_contrastive_v12_120b_6e_20260612/eval_test_detailed/predictions_detailed.csv`
- class-1 audit:
  `runs/probe_mango_cls_256_5class_boundary_contrastive_v12_120b_6e_20260612/class1_confusion_audit_test`

Class-1 confusion audit:

- TP/FP/FN: `54 / 30 / 19`
- class-1 precision improved versus V10/V11 due to fewer false positives
- main remaining false positives:
  - `0->1 = 20`
  - `2->1 = 6`
  - `4->1 = 3`
- main remaining false negatives:
  - `1->2 = 10`
  - `1->0 = 6`
  - `1->4 = 3`

## Decision

Do not run full 30-epoch V12.

Reason: V12 improves the best no-pretrain probe slightly, but class-1 validation
F1 is still `0.6864`, below the required gate `0.70`. The next full-train gate
remains unchanged.

## Interpretation

Boundary contrastive helps precision but does not recover class-1 recall.
The unchanged `1->2` and `1->0` errors show that head/patch embedding separation
alone is not extracting enough local maturity/damage cues.

The next no-pretrain work should not increase class-1 oversampling or run another
full train. Higher-value directions:

1. Validation-only boundary calibration, tuned on validation and applied once to
   test/inference, to check how much of class-1 F1 is recoverable by decision
   logic instead of representation.
2. Train-only label-boundary audit for class `0/1/2`, especially high-confidence
   `0->1`, `1->0`, and `1->2` images.
3. Local surface/color comparator branch:
   - foreground-only HSV/Lab percentiles;
   - green-yellow ratio, yellow-brown ratio;
   - dark defect density and connected dark streak area;
   - over/under-exposure fraction;
   - combine as small MLP logits and optional boundary logits for `0-1` and `1-2`.
4. PMG-lite local patch branch:
   - no full PMG multi-step optimizer;
   - only 2-3 deterministic foreground crops/multi-scale patches per image;
   - cap extra forward cost and audit trace.

