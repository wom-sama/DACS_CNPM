# Baseline Recheck 2026-05-29

Context: q12 baseline checkpoint was re-evaluated on the current environment before any further training.

Checkpoint:

```text
D:\DataAI\AIEx\TRKH\runs\mango_detr_224_q12_objectness_composite_max3_v2\checkpoints\best.pt
```

Dataset:

```text
D:\DataAI\AIEx\dataset\data.yaml
```

## Direct Recheck

Baseline eval config:

- score mode: `foreground`
- NMS IoU: `0.3`
- max detections per image: `3`
- threshold: auto checkpoint calibration, best curve threshold was `0.08`

| Split | Macro F1 | Bbox mean IoU | Detection F1@50 | Precision@50 | Recall@50 | TP / GT | Output |
|---|---:|---:|---:|---:|---:|---:|---|
| val | 0.981653 | 0.798880 | 0.773032 | 0.753270 | 0.793860 | 2534 / 3192 | `eval_val_recheck` |
| test | 0.979963 | 0.798499 | 0.748283 | 0.727220 | 0.770603 | 1253 / 1626 | `eval_test_recheck` |

The earlier note expected test detection F1 around `0.759`; on the current dataset/environment the same baseline command rechecked at `0.748283`.

## Val Post-Processing Sweep

Output root:

```text
D:\DataAI\AIEx\TRKH\runs\mango_detr_224_q12_objectness_composite_max3_v2\sweep_val_20260529
```

| Rank | Config | Detection F1@50 | Precision@50 | Recall@50 | Threshold |
|---:|---|---:|---:|---:|---:|
| 1 | `class_sqrt_objectness`, NMS `0.2`, max_det `3` | 0.780060 | 0.753209 | 0.808897 | 0.26 |
| 2 | `foreground`, NMS `0.2`, max_det `3` | 0.777260 | 0.747469 | 0.809524 | 0.07 |
| 3 | `foreground_minus_background`, NMS `0.3`, max_det `3` | 0.773032 | 0.753270 | 0.793860 | 0.08 |
| 4 | `foreground`, NMS `0.4`, max_det `3` | 0.770417 | 0.744737 | 0.797932 | 0.08 |
| 5 | `class_sqrt_objectness`, NMS `0.3`, max_det `4` | 0.762133 | 0.730195 | 0.796992 | 0.28 |
| 6 | `foreground`, NMS `0.3`, max_det `4` | 0.760626 | 0.725875 | 0.798872 | 0.08 |
| 7 | `foreground`, NMS `0.3`, max_det `6` | 0.741492 | 0.689469 | 0.802005 | 0.08 |
| 8 | `foreground`, NMS `0.3`, max_det `12` | 0.722370 | 0.654620 | 0.805764 | 0.08 |

## Test Confirmation

Output root:

```text
D:\DataAI\AIEx\TRKH\runs\mango_detr_224_q12_objectness_composite_max3_v2\sweep_test_confirm_20260529
```

| Config | Detection F1@50 | Precision@50 | Recall@50 | Threshold | Delta vs baseline test |
|---|---:|---:|---:|---:|---:|
| `class_sqrt_objectness`, NMS `0.2`, max_det `3` | 0.751958 | 0.711697 | 0.797048 | 0.24 | +0.003675 |
| `foreground`, NMS `0.2`, max_det `3` | 0.750376 | 0.735103 | 0.766298 | 0.08 | +0.002093 |

## Current Recommendation

For deployment/inference, the safest default remains the original baseline (`foreground`, NMS `0.3`, max_det `3`) unless recall matters more than precision.

If using the post-processing sweep result, prefer:

```text
score_mode=class_sqrt_objectness
nms_iou_threshold=0.2
max_detections_per_image=3
confidence_threshold=0.24
```

This improves test recall from `0.770603` to `0.797048`, but precision drops from `0.727220` to `0.711697`; net test F1 gain is small.

Next useful work: inspect false positives/false negatives for the `class_sqrt_objectness + NMS 0.2` test output before starting a new training run.
