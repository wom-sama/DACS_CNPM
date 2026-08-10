# TRKH 5-Class Surface Detail V9 Audit - 2026-06-12

## Change

V9 adds configurable attention-view score sources:

- `learned_attention`: previous V8 behavior.
- `surface_detail`: high-frequency/detail map gated by pseudo foreground.
- `hybrid`: weighted mix of learned attention and surface detail.

The goal is to make attention-guided crop/drop focus on fruit surface details
instead of noisy raw attention around border/background.

## Validation

Code/tests:

- `py_compile` passed for changed Python files.
- Focused attention tests passed.
- Full pytest passed: `104 passed`.
- Preflight passed with dataset `D:\DataAI\AIEx\newdataset\class_f`.

Smoke:

- Run: `runs/smoke_mango_cls_256_5class_surface_detail_v9_20260612`
- Exit code: `0`
- Architecture trace completed.
- No leftover Python process after smoke.

Trace review:

- Class 1 sample: `Image_2274_box000.jpg`.
- Foreground mask kept the fruit but also included some same-color cloth/background.
- `surface_detail` score peaked mostly on the central depression/spots on the fruit surface.
- One remaining hot area existed at the left border, so the foreground prior is still imperfect.
- Crop view retained the fruit surface and local details.
- Drop view blurred the discriminative central surface region as intended.

Probe:

- Run: `runs/probe_mango_cls_256_5class_surface_detail_v9_80b_4e_20260612`
- Setup: resume V8 best, 4 epochs, `80` train batches per epoch, full validation, patience `3`.
- Best epoch: `1`.
- Stop reason: early stopping.

Best validation:

| Metric | Value |
| --- | ---: |
| Accuracy | `0.9221` |
| Macro F1 | `0.8857` |
| Class 1 precision | `0.6031` |
| Class 1 recall | `0.7748` |
| Class 1 F1 | `0.6783` |

Final test from best checkpoint:

| Metric | Value |
| --- | ---: |
| Accuracy | `0.9290` |
| Macro F1 | `0.8894` |
| Class 1 precision | `0.6136` |
| Class 1 recall | `0.7397` |
| Class 1 F1 | `0.6708` |

## Decision

Do not run V9 full 30 epochs yet.

Reason: the next full-train gate from V8 is class-1 validation F1 `>=0.70`.
V9 reached only `0.6783`. The gain over V8 is small and mostly from the same
precision/recall balance, so a full train would not be an efficient use of time.

## Next Direction

The current error pattern is class-1 over-prediction on adjacent classes:

- Validation class 0 -> class 1: `48`.
- Validation class 2 -> class 1: `12`.
- Test class 0 -> class 1: `20`.
- Test class 2 -> class 1: `8`.

The next high-value direction is leakage-safe calibration/selector or a boundary
label audit, not stronger class-1 oversampling.
