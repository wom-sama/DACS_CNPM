# DACS_CNPM / TRKH Mango Classification

Scratch-only mango ripeness/quality classification branch built around ViT-Registers. The active target on this branch is single-object crop classification only; detection, objectness, bbox regression, count head, quality head, mosaic/cutmix/copy-paste, and DETR stage scheduling stay on `main`.

## Hard Rule

No pretraining is allowed:

- no pretrained backbone
- no ImageNet/timm/torchvision pretrained weights
- no external detector/classifier checkpoint
- no foundation-model initialization

Only checkpoints created by this repository from random initialization may be resumed.

## Current Evidence

- Historical `runs/mango_hybrid_224` used `vit_registers_hybrid`, image size 224, 4 classes, and detection-style bbox auxiliary logic. Best validation `macro_f1=0.970986` at epoch 109.
- That run had `valid_object_count=9392`, `selected_sample_count=7618`, and `ignored_object_count=1774` on train. The new classification-only branch fixes this by converting every valid bbox in a multi-object image into its own crop sample.
- New classification-only runs should use `model_type=vit_registers`, object-level crops, `best_metric=macro_f1`, and no detection metrics.

Keep the no-pretrain rule first.

## Project Layout

```text
trkh/
  core/          config, utilities, AMP/checkpoint/artifact helpers
  data/          YOLO labels, object-crop dataset, classification transforms
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

Current expected result: `43` tests passed.

## Classification-Only Experiment

Use this for the classification paper track. The key is `--model-type vit_registers` and not passing `--full-image-detection`. In classification mode the dataset now turns every valid bbox into an object-level crop sample by default, so multi-object images are used without mosaic/cutmix/copy-paste.

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
  --run-name mango_cls_224_cnnstem_vitreg_5cls_objectcrops_local_v1 `
  --output-dir runs --disable-resume --seed 42 `
  --model-type vit_registers --image-size 224 --patch-size 16 `
  --stem-channels 32 --head-pooling cls_register_mean `
  --embed-dim 256 --depth 8 --num-heads 8 --num-registers 4 `
  --dropout 0.10 --drop-path-rate 0.10 `
  --batch-size 64 --grad-accum-steps 1 --epochs 0 --scheduler-total-epochs 140 --patience 70 `
  --learning-rate 3e-4 --min-learning-rate 1e-6 --weight-decay 0.05 --warmup-epochs 8 `
  --grad-clip-norm 0.75 --max-nonfinite-grad-steps 4 `
  --num-workers 8 --eval-num-workers 4 --train-image-cache-mb 0 --eval-image-cache-mb 0 `
  --class-weight-mode sqrt_inverse --focal-loss-gamma 2.0 --focal-loss-mix 0.20 `
  --label-smoothing 0.015 --ldam-scale 18.0 --best-metric macro_f1 `
  --resize-mode pad --crop-margin-ratio 0.08 `
  --brightness 0.0 --contrast 0.0 --saturation 0.0 --hue 0.0 --lighting-probability 0.0 `
  --random-erasing-probability 0.0 `
  --random-affine-degrees 4 --random-affine-translate 0.03 --random-affine-scale-min 0.95 `
  --horizontal-flip-probability 0.5 --vertical-flip-probability 0.01 --rotate90-probability 0.03 `
  --batch-mix-probability 0.0 --mosaic-probability 0.0 --mixup-probability 0.0 `
  --cutmix-probability 0.0 --copy-paste-probability 0.0
```

Classification-only automatically disables batch composition methods that blur class labels: batch mix, mixup, mosaic, cutmix, and copy-paste. To reproduce the old primary-object-only behavior for an ablation, add `--disable-classification-object-crops`.

Monitor:

```powershell
Get-Content runs\mango_cls_224_cnnstem_vitreg_5cls_objectcrops_local_v1\history.csv -Tail 5 -Wait
```

## Artifact Recovery

If a run has `history.csv` but missing root-level plots:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m trkh.evaluation.render_history_artifacts `
  --run-dir D:\DataAI\AIEx\TRKH\runs\<run_name>
```

## Output Policy

Training outputs go under `runs/<run_name>` and are ignored by git. Checkpoints, ONNX/TensorRT files, datasets, logs, and cache folders are intentionally ignored.
