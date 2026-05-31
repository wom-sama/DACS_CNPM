# Lightning Training Handoff

This project is scratch-only. Do not download pretrained backbones, external checkpoints, foundation-model weights, or ImageNet weights. Only checkpoints produced by this repository from random initialization may be resumed.

## 1. Clone

```bash
git clone https://github.com/wom-sama/DACS_CNPM.git
cd DACS_CNPM
export PYTHONPATH="$PWD"
```

## 2. Environment

Lightning Studio usually already has CUDA PyTorch. If it does not:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Dataset

Keep the YOLO dataset outside git. Upload or mount it in Studio, then pass the absolute `data.yaml` path.

For custom/non-4-class data:

```bash
--class-name-mode raw --expected-num-classes <N>
```

## 4. Required Tests

```bash
python -m compileall train.py evaluate.py scripts trkh tests
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected result after the package refactor: `36` tests passed.

## 5. One-Shot Scratch Training

For a T4, start from the local one-shot command in `README.md` and adjust only hardware paths/workers:

- interpreter: `python`
- data path: `/teamspace/studios/this_studio/datasets/dataset/data.yaml`
- workers: `--num-workers 4 --eval-num-workers 4`
- batch: keep `--batch-size 16 --grad-accum-steps 1` only if T4 VRAM stays stable; otherwise use `--batch-size 10 --grad-accum-steps 2`
- use `--disable-resume` for a clean one-shot run
- use `--epochs 0 --scheduler-total-epochs 160 --patience 100`

The important flags are:

```bash
python -m trkh.training.train \
  --data /teamspace/studios/this_studio/datasets/dataset/data.yaml \
  --class-name-mode raw --expected-num-classes 5 \
  --run-name mango_detr_416_q40_5cls_one_shot_adaptive_t4_v1 \
  --output-dir runs --disable-resume --seed 42 \
  --model-type vit_registers_hybrid --image-size 416 --patch-size 16 \
  --embed-dim 256 --depth 8 --num-heads 8 --num-registers 4 --drop-path-rate 0.08 \
  --num-queries 40 --decoder-depth 4 --decoder-num-heads 8 --decoder-ffn-dim 1024 \
  --full-image-detection --quality-head --count-head --auxiliary-decoder-loss \
  --batch-size 16 --grad-accum-steps 1 --epochs 0 --scheduler-total-epochs 160 --patience 100 \
  --learning-rate 2.5e-5 --min-learning-rate 8e-7 --weight-decay 0.04 --warmup-epochs 10 \
  --grad-clip-norm 0.30 --max-nonfinite-grad-steps 4 \
  --num-workers 4 --eval-num-workers 4 --train-image-cache-mb 0 --eval-image-cache-mb 0 \
  --class-weight-mode sqrt_inverse --weighted-sampler --weighted-sampler-epoch-multiplier 1.15 \
  --focal-loss-gamma 2.0 --focal-loss-mix 0.25 --label-smoothing 0.02 \
  --stage1-epochs 0 \
  --classification-guard-macro-f1-threshold 0.94 --classification-guard-detection-gap 0.20 --classification-guard-min-cls-weight 0.35 \
  --adaptive-detection-macro-f1-threshold 0.93 --adaptive-detection-f1-target 0.90 --adaptive-detection-gap-threshold 0.20 \
  --adaptive-detection-bbox-iou-target 0.70 --adaptive-detection-max-multiplier 2.20 \
  --bbox-l1-loss-weight 1.3 --bbox-giou-loss-weight 0.85 \
  --background-loss-weight 0.55 --objectness-loss-weight 8.0 \
  --objectness-focal-alpha 0.85 --objectness-focal-gamma 1.25 --matcher-objectness-cost 2.0 \
  --cardinality-loss-weight 0.18 --count-objectness-consistency-weight 0.08 \
  --quality-loss-weight 0.02 --count-loss-weight 0.04 --auxiliary-loss-weight 0.05 \
  --best-metric macro_detection_hmean \
  --eval-detection-score-mode class_sqrt_objectness --eval-detection-nms-iou-threshold 0.18 \
  --eval-adaptive-max-detections --eval-adaptive-count-source auto --eval-adaptive-count-margin 1 --eval-adaptive-min-detections 1 \
  --resize-mode pad --brightness 0.10 --contrast 0.10 --saturation 0.03 --hue 0.01 \
  --random-erasing-probability 0.0 --random-affine-degrees 4 --random-affine-translate 0.03 --random-affine-scale-min 0.94 \
  --horizontal-flip-probability 0.5 --vertical-flip-probability 0.02 --rotate90-probability 0.06 --lighting-probability 0.06 \
  --mosaic-probability 0.06 --cutmix-probability 0.05 --copy-paste-probability 0.12 --copy-paste-max-objects 2 \
  --cutmix-alpha 1.0 --mixup-probability 0.0
```

## 6. Evaluation

```bash
python -m trkh.evaluation.evaluate \
  --checkpoint runs/<run_name>/checkpoints/best.pt \
  --data /teamspace/studios/this_studio/datasets/dataset/data.yaml \
  --split test \
  --batch-size 16 \
  --num-workers 4 \
  --detection-score-mode class_sqrt_objectness \
  --detection-nms-iou-threshold 0.18 \
  --adaptive-max-detections
```

## 7. Recover Missing Run Plots

```bash
python -m trkh.evaluation.render_history_artifacts \
  --run-dir runs/<run_name>
```

## 8. Resume Rules

Prefer one-shot clean runs. Resume only repository-produced scratch checkpoints. For a true continuation of the same run, use `--auto-resume`. For a new experiment, use `--disable-resume` and a new `--run-name`.
