# Handover

Last updated: 2026-05-31.

## Active Goal

Train a scratch-only mango DETR/ViT-Registers model toward both:

- `macro_f1 > 0.98`
- `detection_f1@50 > 0.98`

No pretraining is allowed. This includes ImageNet, timm, torchvision pretrained weights, external detector weights, external classifier weights, and foundation-model initialization.

## Current Technical State

- Active model path: `vit_registers_hybrid` DETR-style multi-object detection.
- Dataset format: YOLO `data.yaml` with image/label splits outside git.
- Local dataset default: `D:\DataAI\AIEx\dataset\data.yaml`.
- Run outputs: `runs/<run_name>`, ignored by git.
- Checkpoints are not committed.
- Cloud/downloaded run artifacts are ignored by git under `Cloud/`.

## Stability Work Completed

The failed run:

```text
runs\mango_detr_416_q34_bf16_quality_aux_count_phase1_w2_clean_inf_b16_v3
```

showed repeated non-finite gradient skips at epoch 21.

Fixes now in code:

- bf16 detection loss math is sanitized and computed in fp32.
- repeated non-finite optimizer steps abort with bad-parameter context.
- stage 1 detection warmup disables objectness/cardinality/count/quality/auxiliary/count-objectness losses.
- dataset indexing avoids repeated image-extension probing.
- train/eval image cache limits can be configured independently.

Details: `docs/STABILITY_FIX_20260529.md`.

## Validation Completed

Passed locally after the stability fix:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m compileall train.py loss.py utils.py config.py dataset.py render_history_artifacts.py tests\test_detection_calibration.py
$env:PYTHONPATH='D:\DataAI\AIEx\TRKH'
D:\DataAI\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Result: `34` tests passed.

Extra checks:

- 416 bf16 quality/count/aux smoke passed.
- q34 original-weight smoke passed.
- one-batch overfit debug loss decreased from about `5.99` to `5.15`.
- failed run checkpoints had finite parameters, but `best.pt` is preferred over `last.pt`.

## Metrics Snapshot

q12 224 baseline test:

- `macro_f1=0.979963`
- `bbox_iou=0.798499`
- `detection_f1@50=0.748283`

Best q12 post-processing test:

- `class_sqrt_objectness`
- NMS `0.2`
- max detections `3`
- `detection_f1@50=0.751958`

Old q34 416 val sweep:

- best `detection_f1@50=0.635828`
- not recommended as the main resume path

Lightning T4 q16 batch16 scratch phase 1:

- run: `mango_detr_416_q16_scratch_batch16_w4_pat80_t4_v1`
- best val calibrated `detection_f1@50=0.851810` at epoch 104
- test `macro_f1=0.992270`
- test calibrated `detection_f1@50=0.828399`
- test best-threshold `detection_f1@50=0.828683`
- root training plots can be regenerated from `history.csv` with `render_history_artifacts.py`

Details: `docs/BASELINE_RECHECK_20260529.md`.

## Next Recommended Work

1. Pull latest code in Lightning and run compile/unit tests.
2. Keep using `/teamspace/studios/this_studio/datasets/dataset/data.yaml`; do not point training at `canbang.yaml`.
3. Resume only project-produced scratch checkpoints.
4. For a new phase from phase 1 best, use `--resume-reset-epoch` with `--resume-reset-scheduler` so LR warmup restarts correctly.
5. Inspect false positives/false negatives from the T4 q16 phase 1 best checkpoint before trying larger q24/q34 variants.
6. If a cloud run is missing root plots, run `python render_history_artifacts.py --run-dir runs/<run_name>`.

## Files To Read First

- `README.md`
- `LIGHTNING_TRAINING.md`
- `TRAINING_640_GUIDE.md`
- `docs/STABILITY_FIX_20260529.md`
- `docs/BASELINE_RECHECK_20260529.md`
