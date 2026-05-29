# Lightning Training Handoff

This project is scratch-only. Do not download pretrained backbones, external checkpoints, foundation-model weights, or ImageNet weights. Only checkpoints produced by this repository from random initialization may be resumed.

## 1. Clone

```bash
git clone https://github.com/wom-sama/DACS_CNPM.git
cd DACS_CNPM
```

## 2. Environment

Lightning Studio usually already has CUDA PyTorch. If it does not, install from `requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set the repo as import root:

```bash
export PYTHONPATH="$PWD"
```

## 3. Dataset

Keep the YOLO dataset outside git. Upload or mount it in the Studio, then pass the absolute `data.yaml` path to every train/eval command.

Expected layout:

```text
dataset/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Use raw class names for custom/non-4-class data:

```bash
--class-name-mode raw --expected-num-classes <N>
```

For the current mango 4-class dataset, `auto` is fine.

## 4. Required Tests Before Training

Run these before any long job:

```bash
python -m compileall train.py loss.py utils.py config.py dataset.py tests/test_detection_calibration.py
python -m unittest discover -s tests -p "test_*.py" -v
```

Optional short stability smoke:

```bash
python train.py \
  --data /path/to/dataset/data.yaml \
  --run-name smoke_416_bf16_lightning \
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

`--skip-final-test` is only for smoke/debug. Do not use it for a real run.

## 5. Conservative Scratch Training Candidate

This is safer than resuming the failed q34 run:

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

If this stays stable and improves detection F1, scale to q24 before q34. Do not jump straight to q34 with high LR and full auxiliary weights.

## 6. Evaluation

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

## 7. Resume Rules

Resume only repository-produced scratch checkpoints:

```bash
python train.py \
  --data /path/to/dataset/data.yaml \
  --resume runs/<run_name>/checkpoints/best.pt \
  --resume-use-cli-config \
  --resume-reset-optimizer \
  --resume-reset-scheduler \
  --resume-reset-scaler \
  --run-name <new_run_name> \
  <same architecture/training flags with lower LR if needed>
```

Prefer `best.pt` over `last.pt` after any non-finite-gradient event.
