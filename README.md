# DACS_CNPM / TRKH Mango DETR

Scratch-only mango ripeness detection project built around a ViT-Registers DETR-style model. The active target is multi-object detection plus ripeness classification, with `macro_f1` and `detection_f1@50` optimized together.

## Hard Rule

No pretraining is allowed:

- no pretrained backbone
- no ImageNet/timm/torchvision pretrained weights
- no external detector/classifier checkpoint
- no foundation-model initialization

Only checkpoints created by this repository from random initialization may be resumed.

## Current Evidence

- Lightning T4 q16 batch16 scratch run reached test `macro_f1=0.992270`, calibrated `detection_f1@50=0.828399`.
- Local 5-class imbalanced q40 copy-paste v2 was stopped at epoch 30 for analysis. Best val `selection_metric=0.688553` at epoch 18, `macro_f1=0.952560`, `best_detection_f1@50=0.539130`, `bbox_iou=0.522800`.
- The same v2 `best.pt` on test reached `macro_f1=0.958867`, calibrated `macro_f1=0.968724`, `best_detection_f1@50=0.549635`, `bbox_iou=0.520143`.
- The v2 bottleneck is detection precision/false positives, not classification. The next run should train one-shot from scratch with full detection enabled from epoch 1 and adaptive detection-loss boosting.

The `0.98/0.98` target is not yet supported by evidence. Keep the no-pretrain rule first.

## Project Layout

```text
trkh/
  core/          config, utilities, AMP/checkpoint/artifact helpers
  data/          YOLO dataset, transforms, mosaic/cutmix/copy-paste
  models/        ViT-Registers / DETR model code
  training/      train loop, matcher, losses, stability/debug tools
  evaluation/    evaluate, metrics, plots, robustness/attention analysis
  inference/     image/video/TensorRT inference and deployment helpers
  tools/         dataset builders and split tools
scripts/         thin CLI wrappers
tests/           unit and stability tests
```

Root `train.py` and `evaluate.py` are compatibility wrappers. Preferred commands use module paths, for example `python -m trkh.training.train`.

## Setup

```powershell
D:\DataAI\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH='D:\DataAI\AIEx\TRKH'
```

## Dataset

Keep datasets outside git and pass `--data` explicitly. Expected YOLO layout:

```text
dataset/
  data.yaml
  canbang.yaml
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

Build a dataset from separate image/label folders:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m trkh.tools.build_dataset `
  --images-dir D:\DataAI\Resize `
  --labels-dir D:\DataAI\Nhan `
  --output-dir D:\DataAI\AIEx\dataset `
  --audit-dir D:\DataAI\AIEx\QuanSat `
  --train-ratio 0.7 `
  --val-ratio 0.2 `
  --seed 42 `
  --overwrite
```

For custom/non-4-class datasets use `--class-name-mode raw --expected-num-classes <N>` while training.

## Required Checks

Run before long training:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m compileall train.py evaluate.py scripts trkh tests
D:\DataAI\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Current expected result: `36` tests passed.

## One-Shot Local Training Candidate

This is the next local RTX 4060 8GB candidate for the current 5-class imbalanced dataset. It is scratch-only, has no manual phase resume, uses `--epochs 0` plus `--patience 100`, and lets adaptive guards change loss pressure from validation metrics.

```powershell
cd D:\DataAI\AIEx\TRKH
$env:PYTHONPATH='D:\DataAI\AIEx\TRKH'
$env:TRKH_AMP_DTYPE='bf16'
$env:OMP_NUM_THREADS='6'
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING='1'
$env:TRKH_ALLOW_WINDOWS_PIN_MEMORY='1'
$env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS='1'

D:\DataAI\.venv\Scripts\python.exe -m trkh.training.train `
  --data D:\DataAI\AIEx\dataset\data.yaml `
  --class-name-mode raw --expected-num-classes 5 `
  --run-name mango_detr_416_q40_5cls_one_shot_adaptive_v1 `
  --output-dir runs --disable-resume --seed 42 `
  --model-type vit_registers_hybrid --image-size 416 --patch-size 16 `
  --embed-dim 256 --depth 8 --num-heads 8 --num-registers 4 --drop-path-rate 0.08 `
  --num-queries 40 --decoder-depth 4 --decoder-num-heads 8 --decoder-ffn-dim 1024 `
  --full-image-detection --quality-head --count-head --auxiliary-decoder-loss `
  --batch-size 10 --grad-accum-steps 2 --epochs 0 --scheduler-total-epochs 160 --patience 100 `
  --learning-rate 2.5e-5 --min-learning-rate 8e-7 --weight-decay 0.04 --warmup-epochs 10 `
  --grad-clip-norm 0.30 --max-nonfinite-grad-steps 4 `
  --num-workers 2 --eval-num-workers 2 --train-image-cache-mb 0 --eval-image-cache-mb 0 `
  --class-weight-mode sqrt_inverse --weighted-sampler --weighted-sampler-epoch-multiplier 1.15 `
  --focal-loss-gamma 2.0 --focal-loss-mix 0.25 --label-smoothing 0.02 `
  --stage1-epochs 0 `
  --classification-guard-macro-f1-threshold 0.94 --classification-guard-detection-gap 0.20 --classification-guard-min-cls-weight 0.35 `
  --adaptive-detection-macro-f1-threshold 0.93 --adaptive-detection-f1-target 0.90 --adaptive-detection-gap-threshold 0.20 `
  --adaptive-detection-bbox-iou-target 0.70 --adaptive-detection-max-multiplier 2.20 `
  --bbox-l1-loss-weight 1.3 --bbox-giou-loss-weight 0.85 `
  --background-loss-weight 0.55 --objectness-loss-weight 8.0 `
  --objectness-focal-alpha 0.85 --objectness-focal-gamma 1.25 --matcher-objectness-cost 2.0 `
  --cardinality-loss-weight 0.18 --count-objectness-consistency-weight 0.08 `
  --quality-loss-weight 0.02 --count-loss-weight 0.04 --auxiliary-loss-weight 0.05 `
  --best-metric macro_detection_hmean `
  --eval-detection-score-mode class_sqrt_objectness --eval-detection-nms-iou-threshold 0.18 `
  --eval-adaptive-max-detections --eval-adaptive-count-source auto --eval-adaptive-count-margin 1 --eval-adaptive-min-detections 1 `
  --resize-mode pad --brightness 0.10 --contrast 0.10 --saturation 0.03 --hue 0.01 `
  --random-erasing-probability 0.0 --random-affine-degrees 4 --random-affine-translate 0.03 --random-affine-scale-min 0.94 `
  --horizontal-flip-probability 0.5 --vertical-flip-probability 0.02 --rotate90-probability 0.06 --lighting-probability 0.06 `
  --mosaic-probability 0.06 --cutmix-probability 0.05 --copy-paste-probability 0.12 --copy-paste-max-objects 2 `
  --cutmix-alpha 1.0 --mixup-probability 0.0
```

Monitor:

```powershell
Get-Content runs\mango_detr_416_q40_5cls_one_shot_adaptive_v1\history.csv -Tail 5 -Wait
```

## Artifact Recovery

If a run has `history.csv` but missing root-level plots:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m trkh.evaluation.render_history_artifacts `
  --run-dir D:\DataAI\AIEx\TRKH\runs\<run_name>
```

## Output Policy

Training outputs go under `runs/<run_name>` and are ignored by git. Checkpoints, ONNX/TensorRT files, datasets, logs, and cache folders are intentionally ignored.
