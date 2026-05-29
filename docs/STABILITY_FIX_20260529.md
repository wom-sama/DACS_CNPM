# Stability Fix 2026-05-29

Priority rule: no external pretraining. Do not use pretrained backbones, external weights, or foundation-model initialization. Resuming a checkpoint produced by this project from scratch is allowed only when the intent is to continue that same scratch experiment.

## Why This Was Needed

Run:

```text
D:\DataAI\AIEx\TRKH\runs\mango_detr_416_q34_bf16_quality_aux_count_phase1_w2_clean_inf_b16_v3
```

failed by repeatedly skipping optimizer steps with non-finite gradient norm at epoch 21. The run used bf16 autocast, so AMP GradScaler was disabled. The warning text said AMP, but there was no scaler to recover from bad gradients. The likely cause is gradient blow-up from 416px q34 with quality/count/auxiliary losses and high LR, not raw dataset corruption.

## Code Changes

Changed files:

- `loss.py`
- `train.py`
- `config.py`
- `utils.py`
- `dataset.py`
- `tests/test_detection_calibration.py`

Main fixes:

1. Detection loss now sanitizes logits/boxes and computes loss math in fp32 even when model outputs are bf16.
2. Training now aborts after repeated non-finite gradient optimizer steps instead of skipping forever.
3. Non-finite gradient warnings include the first parameter name and invalid gradient count.
4. Stage 1 detection warmup now disables objectness, cardinality, count, quality, auxiliary, and count-objectness consistency losses, not only bbox weights.
5. Added configurable image cache: `--train-image-cache-mb`, `--eval-image-cache-mb`, `TRKH_TRAIN_IMAGE_CACHE_MB`, `TRKH_VAL_IMAGE_CACHE_MB`, `TRKH_TEST_IMAGE_CACHE_MB`.
6. Dataset startup now indexes image stems once instead of checking each extension for every label file.
7. Added `--skip-final-test` for fast smoke/debug runs only.
8. Added unit tests for bf16-safe detection loss, stage loss gating, and repeated non-finite gradient abort.

## Validation

Passed:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m compileall train.py loss.py utils.py config.py dataset.py tests\test_detection_calibration.py
```

Passed:

```powershell
$env:PYTHONPATH='D:\DataAI\AIEx\TRKH'
D:\DataAI\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Result: `33` tests passed.

Extra checks:

- Old 416 run checkpoints `best.pt`, `last.pt`, and `interrupt.pt` all have finite model parameters.
- Prefer resuming from `best.pt`, not `last.pt`, because `last.pt` was saved after repeated skipped gradient steps.
- 416 bf16 quality/count/aux smoke passed without non-finite gradients:
  - `runs/smoke_416_bf16_quality_aux_count_after_stability_fix_20260529`
  - `runs/smoke_416_q34_original_weights_after_stability_fix_20260529`
- Overfit-one-batch debug passed:
  - `runs/debug_overfit_after_stability_fix_20260529`
  - loss decreased from about `5.99` to `5.15` over 8 iterations.
- Dataset train split initialization after image-stem indexing measured about `3.10s`.

## 416 Checkpoint Sweep

Sweep output:

```text
D:\DataAI\AIEx\TRKH\runs\mango_detr_416_q34_bf16_quality_aux_count_phase1_w2_clean_inf_b16_v3\postfix_sweep_val_20260529
```

Best val result from the old 416 checkpoint:

| Config | Macro F1 | Bbox IoU | Detection F1@50 |
|---|---:|---:|---:|
| `class_sqrt_objectness_quality`, NMS `0.2`, max_det `3` | 0.924749 | 0.661359 | 0.635828 |
| `class_sqrt_objectness`, NMS `0.2`, adaptive count margin `1` | 0.924749 | 0.661359 | 0.594063 |
| `class_sqrt_objectness`, NMS `0.2`, max_det `3` | 0.924749 | 0.661359 | 0.590959 |
| `foreground`, NMS `0.3`, max_det `3` | 0.924749 | 0.661358 | 0.580106 |

Conclusion: do not resume this 416 run as the main path to the target. It is stable enough to inspect, but it is far behind the q12 224 baseline.

## Breakthrough Ranking

1. bf16-safe fp32 loss and non-finite gradient abort: 9/10. This directly fixes the failure mode in the screenshot.
2. Dataset image-stem index and cache controls: 8/10. This reduces Windows startup and gives explicit RAM/speed control.
3. Stage loss gating correctness: 7/10. Prevents hidden auxiliary/objectness losses during a supposed warmup stage.
4. 416 post-processing sweep: 3/10. It improved clarity but not model quality enough.
5. `--skip-final-test` debug flag: 2/10. Useful for fast smoke runs, not a quality improvement.

## Current Limitations

- No long full training run has been completed after the fix.
- The target `detection_f1 > 0.98` and `macro_f1 > 0.98` is not supported by current evidence. Macro F1 is near the target on the 224 q12 baseline, but detection F1 is still around `0.75`.
- The old 416 q34 run underperforms badly: val detection F1 around `0.636` even after post-processing.
- The dataset has images with up to 33 objects, but the best q12 deployment config caps detections at 3. This is good for average F1 in current evaluation, but it is a known recall tradeoff for crowded images.
- bf16 smoke tests prove numeric stability on short runs, not convergence over 20+ epochs.

## Next Training Direction

Keep q12 224 as the production baseline.

For a new scratch-only experiment, prefer a conservative 416 run instead of resuming the failed run:

- start from random initialization
- no pretrained backbone
- lower LR: `3e-5` to `5e-5`
- use q16 or q24 before q34
- keep quality/auxiliary weights small at first
- use `class_sqrt_objectness` or `foreground` for eval before trusting quality score
- use `max_det=3` or adaptive count for evaluation, not raw max_det=34
- use `--num-workers 0` with train/eval image cache on Windows unless explicitly benchmarking workers

If the next 416 run shows non-finite gradients again, stop immediately and resume from `best.pt` with:

- `--resume-use-cli-config`
- `--resume-reset-optimizer`
- `--resume-reset-scheduler`
- `--resume-reset-scaler`
- lower LR
- lower `quality_loss_weight`, `auxiliary_loss_weight`, and `count_loss_weight`
