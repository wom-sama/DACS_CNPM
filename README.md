# DACS_CNPM / TRKH Mango DETR

Scratch-only mango ripeness detection project built around a ViT-Registers DETR-style model. The current priority is multi-object detection plus ripeness classification, with `macro_f1` and `detection_f1@50` tracked together.

## Hard Rule

No pretraining is allowed:

- no pretrained backbone
- no ImageNet/timm/torchvision pretrained weights
- no external detector/classifier checkpoint
- no foundation-model initialization

Only checkpoints created by this repository from random initialization may be resumed.

The model code rejects common pretrained config keys, and training docs keep this rule explicit.

## Current Status

- q12 224 baseline recheck: test `macro_f1=0.979963`, `detection_f1@50=0.748283`.
- Best small post-process gain: `class_sqrt_objectness`, NMS `0.2`, max detections `3`, test `detection_f1@50=0.751958`.
- Failed 416 q34 run was stabilized in code, but the old checkpoint is not a good main path: val `detection_f1@50` stayed around `0.636`.
- Lightning T4 q16 batch16 scratch phase 1 improved detection substantially: best val calibrated `detection_f1@50=0.851810`; test `macro_f1=0.992270`, calibrated `detection_f1@50=0.828399`.
- Main next experiment should continue scratch-only phase training from the phase 1 best checkpoint with scheduler/epoch reset before trying q34 again.

See:

- `docs/BASELINE_RECHECK_20260529.md`
- `docs/STABILITY_FIX_20260529.md`
- `LIGHTNING_TRAINING.md`

## Main Files

- `train.py`: training loop, resume logic, bf16/AMP handling, staged detection loss, cache controls.
- `evaluate.py`: validation/test metrics, calibration, NMS, detection score modes.
- `inference.py`: single-image inference.
- `model.py`: ViT-Registers hybrid DETR model and pretrained-option rejection.
- `loss.py`: hybrid detection/classification loss with fp32 sanitization for bf16 stability.
- `dataset.py`: YOLO dataset loader, crop/full-image modes, image-stem indexing, cache support.
- `metrics.py`: classification and detection metrics.
- `utils.py`: dataloaders, optimizer groups, AMP helpers, checkpointing, artifact plotting.
- `render_history_artifacts.py`: regenerate run-level plots from `history.csv` after interrupted/cloud runs.
- `tests/test_detection_calibration.py`: calibration, loss stability, stage gating, and non-finite gradient tests.

Older classification utilities (`ablation.py`, `deploy.py`, `robustness_eval.py`, `attention_viz.py`, stream scripts) remain in the repo for compatibility, but the active target is DETR-style detection.

## Setup

Recommended local interpreter:

```powershell
D:\DataAI\.venv\Scripts\python.exe
```

Install dependencies:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set import root for tests:

```powershell
$env:PYTHONPATH='D:\DataAI\AIEx\TRKH'
```

## Dataset

Do not commit datasets to git. Keep YOLO data outside this repo and pass `--data` explicitly.

Expected layout:

```text
dataset/
  data.yaml
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

Current local default:

```text
D:\DataAI\AIEx\dataset\data.yaml
```

## Required Checks

Run before any long training job:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m compileall train.py loss.py utils.py config.py dataset.py render_history_artifacts.py tests\test_detection_calibration.py
D:\DataAI\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Expected current result: `34` tests passed.

## Conservative Scratch 416 Candidate

Use this as the next Lightning/local long-run candidate. It avoids the unstable q34/high-LR setup that produced non-finite gradients.

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

On Linux/Lightning, use the same flags but replace the interpreter and dataset paths. See `LIGHTNING_TRAINING.md`.

## Smoke Command

Use only to verify stability, not to report final quality:

```powershell
D:\DataAI\.venv\Scripts\python.exe D:\DataAI\AIEx\TRKH\train.py ^
  --data D:\DataAI\AIEx\dataset\data.yaml ^
  --run-name smoke_416_bf16_after_clone ^
  --image-size 416 ^
  --batch-size 1 ^
  --grad-accum-steps 8 ^
  --epochs 1 ^
  --max-train-batches 16 ^
  --max-val-batches 8 ^
  --num-workers 0 ^
  --eval-num-workers 0 ^
  --train-image-cache-mb 512 ^
  --eval-image-cache-mb 512 ^
  --model-type vit_registers_hybrid ^
  --num-queries 16 ^
  --full-image-detection ^
  --quality-head ^
  --count-head ^
  --auxiliary-decoder-loss ^
  --learning-rate 5e-5 ^
  --best-metric macro_detection_hmean ^
  --skip-final-test
```

## Evaluation

```powershell
D:\DataAI\.venv\Scripts\python.exe D:\DataAI\AIEx\TRKH\evaluate.py ^
  --checkpoint D:\DataAI\AIEx\TRKH\runs\mango_detr_416_q16_scratch_v1\checkpoints\best.pt ^
  --data D:\DataAI\AIEx\dataset\data.yaml ^
  --split test ^
  --batch-size 1 ^
  --num-workers 0 ^
  --detection-score-mode class_sqrt_objectness ^
  --detection-nms-iou-threshold 0.2 ^
  --max-detections-per-image 3
```

## Artifact Recovery

If a cloud job is interrupted or killed before `train.py` finishes its post-run cleanup, checkpoints and `history.csv` may exist while root-level plots are missing. Regenerate them from the downloaded run directory:

```powershell
D:\DataAI\.venv\Scripts\python.exe D:\DataAI\AIEx\TRKH\render_history_artifacts.py ^
  --run-dir D:\DataAI\AIEx\TRKH\runs\<run_name>
```

This writes `training_curves.png`, `results.png`, `all_training_metrics.png`, `per_class_training_metrics.png`, `detection_training_metrics.png`, `validation_convergence.png`, and `history_summary.json`.

## Output Policy

Training outputs go under `runs/<run_name>` and are ignored by git. Checkpoints, ONNX/TensorRT files, datasets, logs, and cache folders are intentionally ignored to keep GitHub usable.
