# Handover

Last updated: 2026-05-29.

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
D:\DataAI\.venv\Scripts\python.exe -m compileall train.py loss.py utils.py config.py dataset.py tests\test_detection_calibration.py
$env:PYTHONPATH='D:\DataAI\AIEx\TRKH'
D:\DataAI\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Result: `33` tests passed.

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

Details: `docs/BASELINE_RECHECK_20260529.md`.

## Next Recommended Work

1. Push code to GitHub and clone in Lightning Studio.
2. Upload/mount dataset separately; do not commit data.
3. Run compile/unit tests in Lightning.
4. Run short smoke with `--skip-final-test`.
5. Start conservative scratch 416 q16 run from `TRAINING_640_GUIDE.md` or `LIGHTNING_TRAINING.md`.
6. If stable and detection F1 improves, scale to q24 before q34.
7. Before any bigger run, inspect false positives/false negatives from q12 post-processing result.

## Files To Read First

- `README.md`
- `LIGHTNING_TRAINING.md`
- `TRAINING_640_GUIDE.md`
- `docs/STABILITY_FIX_20260529.md`
- `docs/BASELINE_RECHECK_20260529.md`
