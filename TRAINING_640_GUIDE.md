# Training Guide

This file replaces the older 640px classification guide. The active project target is scratch-only DETR-style mango detection and ripeness classification.

## Non-Negotiable Rule

Do not use pretraining:

- no pretrained backbone
- no ImageNet/timm/torchvision pretrained weights
- no external checkpoint from another project
- no detector/classifier teacher weights

Allowed resume source: checkpoints produced by this repository from random initialization.

## Before A Long Run

Local Windows:

```powershell
$env:PYTHONPATH='D:\DataAI\AIEx\TRKH'
D:\DataAI\.venv\Scripts\python.exe -m compileall train.py loss.py utils.py config.py dataset.py tests\test_detection_calibration.py
D:\DataAI\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Lightning/Linux:

```bash
export PYTHONPATH="$PWD"
python -m compileall train.py loss.py utils.py config.py dataset.py tests/test_detection_calibration.py
python -m unittest discover -s tests -p "test_*.py" -v
```

Current expected test result: `33` tests passed.

## Recommended Next Run

Use q16/q24 before q34. The previous 416 q34 run became numerically unstable and its old checkpoint was not competitive after post-processing.

Windows:

```powershell
D:\DataAI\.venv\Scripts\python.exe D:\DataAI\AIEx\TRKH\train.py ^
  --data D:\DataAI\AIEx\dataset\data.yaml ^
  --run-name mango_detr_416_q16_scratch_v1 ^
  --image-size 416 ^
  --batch-size 1 ^
  --grad-accum-steps 8 ^
  --epochs 120 ^
  --patience 18 ^
  --num-workers 0 ^
  --eval-num-workers 0 ^
  --train-image-cache-mb 4096 ^
  --eval-image-cache-mb 2048 ^
  --model-type vit_registers_hybrid ^
  --num-queries 16 ^
  --full-image-detection ^
  --learning-rate 5e-5 ^
  --min-learning-rate 1e-6 ^
  --warmup-epochs 10 ^
  --grad-clip-norm 0.5 ^
  --max-nonfinite-grad-steps 4 ^
  --quality-head ^
  --quality-loss-weight 0.05 ^
  --count-head ^
  --count-loss-weight 0.05 ^
  --auxiliary-decoder-loss ^
  --auxiliary-loss-weight 0.10 ^
  --objectness-loss-weight 5.0 ^
  --bbox-l1-loss-weight 1.0 ^
  --bbox-giou-loss-weight 0.5 ^
  --best-metric macro_detection_hmean ^
  --eval-detection-score-mode class_sqrt_objectness ^
  --eval-detection-nms-iou-threshold 0.2 ^
  --eval-max-detections-per-image 3
```

Lightning/Linux uses the same flags with Linux paths:

```bash
python train.py \
  --data /path/to/dataset/data.yaml \
  --run-name mango_detr_416_q16_scratch_lightning_v1 \
  --image-size 416 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --epochs 120 \
  --patience 18 \
  --num-workers 2 \
  --eval-num-workers 2 \
  --train-image-cache-mb 0 \
  --eval-image-cache-mb 0 \
  --model-type vit_registers_hybrid \
  --num-queries 16 \
  --full-image-detection \
  --learning-rate 5e-5 \
  --min-learning-rate 1e-6 \
  --warmup-epochs 10 \
  --grad-clip-norm 0.5 \
  --max-nonfinite-grad-steps 4 \
  --quality-head \
  --quality-loss-weight 0.05 \
  --count-head \
  --count-loss-weight 0.05 \
  --auxiliary-decoder-loss \
  --auxiliary-loss-weight 0.10 \
  --objectness-loss-weight 5.0 \
  --bbox-l1-loss-weight 1.0 \
  --bbox-giou-loss-weight 0.5 \
  --best-metric macro_detection_hmean \
  --eval-detection-score-mode class_sqrt_objectness \
  --eval-detection-nms-iou-threshold 0.2 \
  --eval-max-detections-per-image 3
```

## Smoke Test

Use this only to check stability after cloning or changing hardware:

```bash
python train.py \
  --data /path/to/dataset/data.yaml \
  --run-name smoke_416_bf16 \
  --image-size 416 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --epochs 1 \
  --max-train-batches 16 \
  --max-val-batches 8 \
  --num-workers 2 \
  --eval-num-workers 2 \
  --train-image-cache-mb 0 \
  --eval-image-cache-mb 0 \
  --model-type vit_registers_hybrid \
  --num-queries 16 \
  --full-image-detection \
  --quality-head \
  --count-head \
  --auxiliary-decoder-loss \
  --learning-rate 5e-5 \
  --best-metric macro_detection_hmean \
  --skip-final-test
```

Never use `--skip-final-test` for a real metric report.

## Evaluation

```bash
python evaluate.py \
  --checkpoint runs/mango_detr_416_q16_scratch_lightning_v1/checkpoints/best.pt \
  --data /path/to/dataset/data.yaml \
  --split test \
  --batch-size 1 \
  --num-workers 2 \
  --detection-score-mode class_sqrt_objectness \
  --detection-nms-iou-threshold 0.2 \
  --max-detections-per-image 3
```

## Phase Resume Rule

When starting a new phase from a scratch checkpoint, keep the no-pretrain rule
and resume only project-produced weights. Use:

```bash
--resume-use-cli-config \
--resume-reset-optimizer \
--resume-reset-scheduler \
--resume-reset-scaler \
--resume-reset-epoch
```

`--resume-reset-epoch` makes the new run start at epoch 1 so the scheduler
warmup/decay is computed for the new phase instead of the source checkpoint's
old epoch number.

## Current Evidence

- q12 224 baseline test: `macro_f1=0.979963`, `detection_f1@50=0.748283`.
- Best q12 post-processing test: `detection_f1@50=0.751958`.
- Old q34 416 checkpoint after sweep: val `macro_f1=0.924749`, `detection_f1@50=0.635828`.

The `0.98/0.98` target is not yet supported by evidence. The next useful work is stable scratch training plus false-positive/false-negative analysis, not bigger q34 runs by default.
