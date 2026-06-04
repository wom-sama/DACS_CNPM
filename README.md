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

Current expected result: `69` tests passed for `tests/test_detection_calibration.py`; run the full suite before paper-final training.

## Classification-Only Experiment

Use this for the classification paper track. The key is `--model-type vit_registers` and not passing `--full-image-detection`. Keep all checkpoints scratch-only.

For paper-grade evaluation, prefer the grouped/sequence-safe classification-folder crop dataset at `D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_grouped_seqsafe_w3`. It is rebuilt from the same `cls_crops` images without adding data, but keeps same-source crops, byte-identical duplicates, and same-class near-neighbor `Image_N` sequences in one split. Current recommendation after the v17/v18 probes is random class sampling with a soft Balanced Softmax prior, fine-grained patch pooling, a very small VFF-like supervised contrastive term over `head,cnn` embeddings, and a light foreground-consistency loss that discourages patch-token energy on crop borders/background. Do not combine strict balanced batches with Balanced Softmax on this split; v17 over-corrected class 1 and produced many class-1 false positives. The checkpoint selector uses `loss_aware_fair_macro_f1`, so it does not choose a late overfit checkpoint only because the rarest class F1 improved slightly.

Leak-audit note: raw `cls_crops` has no exact cross-split duplicate and no same `Image_N_boxK` source-stem across splits, but it does contain many same-class near-neighbor `Image_N` IDs across train/val/test. Use `cls_crops_grouped_seqsafe_w3` for final claims, and rerun any baseline models on the same grouped split for fair comparison.

```powershell
cd D:\DataAI\AIEx\TRKH
$env:PYTHONPATH='D:\DataAI\AIEx\TRKH'
$env:TRKH_AMP_DTYPE='bf16'
$env:OMP_NUM_THREADS='4'
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING='1'
$env:TRKH_ALLOW_WINDOWS_PIN_MEMORY='1'
$env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS='1'

D:\DataAI\.venv\Scripts\python.exe -m trkh.training.train `
  --data D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_grouped_seqsafe_w3\data.yaml `
  --class-name-mode raw --expected-num-classes 5 `
  --run-name mango_cls_224_clscrops_grouped_fgpool_midprior_norepeat_v18_full `
  --output-dir runs --disable-resume --seed 44 `
  --model-type vit_registers --image-size 224 --patch-size 16 --stem-channels 32 `
  --cnn-feature-fusion --cnn-fusion-dropout 0.26 `
  --fine-grained-pooling --fine-grained-pooling-dropout 0.12 `
  --embed-dim 256 --depth 8 --num-heads 8 --num-registers 4 --register-positional-embedding `
  --head-pooling cls `
  --dropout 0.22 --attention-dropout 0.08 --drop-path-rate 0.18 `
  --batch-size 64 --grad-accum-steps 1 --epochs 0 --scheduler-total-epochs 70 --patience 24 `
  --learning-rate 2.0e-5 --min-learning-rate 5e-7 --warmup-epochs 3 `
  --weight-decay 0.18 --grad-clip-norm 0.60 --max-nonfinite-grad-steps 8 `
  --num-workers 6 --eval-num-workers 4 --train-image-cache-mb 0 --eval-image-cache-mb 0 `
  --best-metric loss_aware_fair_macro_f1 --fair-f1-gap-target 0.35 --fair-f1-gap-penalty 0.25 `
  --fair-f1-min-weight 0.18 --fair-f1-loss-weight 0.12 `
  --classification-loss balanced_softmax --balanced-softmax-tau 0.18 --disable-class-weights `
  --focal-loss-gamma 1.0 --focal-loss-mix 0.0 --label-smoothing 0.05 `
  --metric-learning-loss-weight 0.002 --metric-learning-temperature 0.28 `
  --metric-learning-sources head,cnn `
  --foreground-consistency-loss-weight 0.008 --foreground-consistency-margin 0.07 `
  --balance-auto-max-repeat-factor 1.0 `
  --disable-rare-class-recall-guard `
  --resize-mode pad --brightness 0.03 --contrast 0.03 --saturation 0.015 --hue 0 `
  --random-erasing-probability 0.02 --random-affine-degrees 2 --random-affine-translate 0.015 `
  --random-affine-scale-min 0.97 --horizontal-flip-probability 0.5 --vertical-flip-probability 0 `
  --rotate90-probability 0 --lighting-probability 0 `
  --batch-mix-probability 0 --mosaic-probability 0 --mixup-probability 0 `
  --cutmix-probability 0 --copy-paste-probability 0 --targeted-copy-paste-probability 0
```

This v18 command intentionally does not use SAM, strict balanced sampling, or rare-class repeat. `--balance-auto-max-repeat-factor 1.0` is intentional: it neutralizes the `canbang.yaml` auto-repeat path so class-1 correction is controlled only by the soft Balanced Softmax prior. On the current Windows machine, the measured loader sweet spot is `batch-size=64`, `num-workers=6`, `eval-num-workers=4` when the three `TRKH_ALLOW_WINDOWS_*` environment variables above are enabled.

The `predictions_detailed.csv` artifact keeps the `data.yaml` class ID order. The shorter `predictions.csv` is baseline-compatible and may remap integer IDs to the comparison table class order; use the name columns or `metrics.json/classes` when reading it.

## XAI Audit

Run this after a checkpoint is available to inspect fail cases, low-confidence cases, and major confusion pairs. It is evaluation-only and does not create training data, so it does not violate the no-pretrain/no-leak rule.

```powershell
D:\DataAI\.venv\Scripts\python.exe -m trkh.evaluation.xai_audit `
  --checkpoint runs\mango_cls_224_clscrops_grouped_fgpool_midprior_norepeat_v18_full\checkpoints\best.pt `
  --data D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_grouped_seqsafe_w3\data.yaml `
  --class-name-mode raw --expected-num-classes 5 --split test `
  --output-dir runs\mango_cls_224_clscrops_grouped_fgpool_midprior_norepeat_v18_full\xai_audit_test_v2 `
  --max-cases 56 --mistake-cases 24 --low-confidence-cases 8 --close-margin-cases 12 `
  --per-class-cases 2 --batch-size 64 --num-workers 4 `
  --method all --feature-source stem_last --query-tokens cls --rollout-start-layer 1 `
  --robustness-probes --review-high-confidence 0.45 `
  --review-background-threshold 0.12 --review-border-threshold 0.25 `
  --shortcut-drop-threshold 0.15 --top-k 5
```

The v18 attention heatmap uses `--query-tokens cls` because the classifier head pools CLS while register tokens act as context/sink tokens. `--method all --feature-source stem_last --robustness-probes` exports raw attention, attention rollout, gradient-weighted rollout, CNN-stem Grad-CAM, register diagnostics, robustness probes, `xai_metrics.json`, and `review_manifest.csv`.

## Build Grouped Split

Use this to rebuild the grouped split from raw `cls_crops`:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m trkh.tools.build_classification_grouped_split `
  --data D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops\data.yaml `
  --class-name-mode raw --expected-num-classes 5 `
  --output-dir D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_grouped_seqsafe_w3 `
  --near-id-window 3 --train-ratio 0.70 --val-ratio 0.20 `
  --seed 42 --link-mode hardlink --overwrite
```

The split uses hardlinks by default to save disk space on the same drive; if hardlinking fails, the tool falls back to copying.

## Split Leak Audit

Run this before publishing metrics from a classification folder split:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m trkh.tools.audit_classification_split `
  --data D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_grouped_seqsafe_w3\data.yaml `
  --class-name-mode raw --expected-num-classes 5 `
  --output-dir runs\dataset_leak_audit_cls_crops `
  --near-id-window 3 --max-examples 50
```

The older online YOLO object-crop command is still useful for an ablation because it can include wider crop margins from original images:

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
  --run-name mango_cls_224_cnnstem_vitreg_5cls_objectcrops_rarecrop_local_v3 `
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
  --class-crop-margin-scale-threshold 1.5 --class-crop-margin-max-ratio 0.16 `
  --brightness 0.0 --contrast 0.0 --saturation 0.0 --hue 0.0 --lighting-probability 0.0 `
  --random-erasing-probability 0.0 `
  --random-affine-degrees 3 --random-affine-translate 0.02 --random-affine-scale-min 0.96 `
  --horizontal-flip-probability 0.5 --vertical-flip-probability 0.0 --rotate90-probability 0.02 `
  --batch-mix-probability 0.0 --mosaic-probability 0.0 --mixup-probability 0.0 `
  --cutmix-probability 0.0 --copy-paste-probability 0.0
```

Classification-only automatically disables batch composition methods that blur class labels: batch mix, mixup, mosaic, cutmix, and copy-paste. It also reads class scale from `canbang.yaml`: any class with scale at least `--class-crop-margin-scale-threshold` gets a wider object crop up to `--class-crop-margin-max-ratio`, so rare-class handling is automatic rather than hard-coded to one class. To reproduce the old primary-object-only behavior for an ablation, add `--disable-classification-object-crops`.

Monitor:

```powershell
Get-Content runs\mango_cls_224_cnnstem_vitreg_5cls_objectcrops_rarecrop_local_v3\history.csv -Tail 5 -Wait
```

## Artifact Recovery

If a run has `history.csv` but missing root-level plots:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m trkh.evaluation.render_history_artifacts `
  --run-dir D:\DataAI\AIEx\TRKH\runs\<run_name>
```

## Output Policy

Training outputs go under `runs/<run_name>` and are ignored by git. Checkpoints, ONNX/TensorRT files, datasets, logs, and cache folders are intentionally ignored.
