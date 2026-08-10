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
- New classification-only runs should use `model_type=vit_registers`, object-level crops, a validation-only classification selection metric such as `loss_aware_fair_macro_f1`, and no detection metrics.

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

Current expected result: full unittest suite passes (`78` tests as of 2026-06-06); run the full suite before paper-final training.

## Classification-Only Experiment

Use this for the classification paper track. The key is `--model-type vit_registers` and not passing `--full-image-detection`. Keep all checkpoints scratch-only.

For paper-grade evaluation, prefer the grouped/sequence-safe classification-folder crop dataset at `D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_grouped_seqsafe_w3`. It is rebuilt from the same `cls_crops` images without adding data, but keeps same-source crops, byte-identical duplicates, and same-class near-neighbor `Image_N` sequences in one split. Current recommendation after the v25/v26 probes is random sampling, LDAM-focal without class weights, light geometric augmentation, capped rare repeat, and a dynamic raw-factor rare recall guard. Do not combine strict balanced batches with Balanced Softmax on this split; v17 over-corrected class 1 and produced many class-1 false positives. Do not use the v26 fine-grained pooling probe as the main command; it learned slower and underperformed v25 by epoch 6.

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
  --run-name mango_cls_224_clscrops_grouped_rawguard_bf16_v25_full `
  --output-dir runs --disable-resume --seed 42 `
  --model-type vit_registers --image-size 224 --patch-size 16 --stem-channels 32 `
  --cnn-feature-fusion --cnn-fusion-dropout 0.30 `
  --embed-dim 256 --depth 8 --num-heads 8 --num-registers 4 --register-positional-embedding `
  --head-pooling cls_register_mean `
  --dropout 0.24 --attention-dropout 0.08 --drop-path-rate 0.22 `
  --batch-size 64 --grad-accum-steps 1 --epochs 0 --scheduler-total-epochs 80 --patience 35 `
  --learning-rate 2.2e-5 --min-learning-rate 5e-7 --warmup-epochs 4 `
  --weight-decay 0.16 --grad-clip-norm 0.45 --max-nonfinite-grad-steps 4 `
  --num-workers 6 --eval-num-workers 4 --train-image-cache-mb 0 --eval-image-cache-mb 0 `
  --imbalance-auto-tune --disable-class-weights --class-weight-mode sqrt_inverse `
  --balance-auto-max-repeat-factor 1.25 `
  --best-metric loss_aware_fair_macro_f1 --fair-f1-gap-target 0.05 --fair-f1-gap-penalty 0.8 `
  --fair-f1-min-weight 0.50 --fair-f1-loss-weight 0.16 `
  --classification-loss ldam_focal --ldam-max-margin 0.18 --ldam-scale 10.0 `
  --focal-loss-gamma 1.5 --focal-loss-mix 0.03 --label-smoothing 0.04 `
  --metric-learning-loss-weight 0.006 --metric-learning-temperature 0.22 `
  --metric-learning-sources head,cnn `
  --foreground-consistency-loss-weight 0.015 --foreground-consistency-margin 0.07 `
  --rare-class-recall-target 0.55 --rare-class-recall-guard-scale-threshold 1.5 `
  --rare-class-recall-guard-max-multiplier 1.8 --rare-class-recall-guard-min-precision 0.05 `
  --resize-mode crop --brightness 0.025 --contrast 0.025 --saturation 0.01 --hue 0 `
  --random-erasing-probability 0.0 --random-affine-degrees 4 --random-affine-translate 0.025 `
  --random-affine-scale-min 0.96 --horizontal-flip-probability 0.5 --vertical-flip-probability 0 `
  --rotate90-probability 0.03 --lighting-probability 0 `
  --batch-mix-probability 0 --mosaic-probability 0 --mixup-probability 0 `
  --cutmix-probability 0 --copy-paste-probability 0 --targeted-copy-paste-probability 0
```

This command intentionally does not use SAM, strict balanced sampling, mosaic/cutmix/copy-paste, or external data. The rare-class guard is dynamic: it uses raw imbalance factors from `canbang.yaml`, so any class at least `1.5x` underrepresented can be targeted without hard-coding class ID 1. On the current Windows machine, the measured loader sweet spot is `batch-size=64`, `num-workers=6`, `eval-num-workers=4` when the three `TRKH_ALLOW_WINDOWS_*` environment variables above are enabled.

After training, evaluate both `best.pt` and `last.pt`. On v25, `best.pt` was selected at epoch 5 but `last.pt` scored better on the grouped test split (`macro_f1=0.688339`, class `Xoai_Song_ChuaNhe_CoNguyCo` F1 `0.194`). The grouped validation class-1 support is small, so checkpoint selection is noisy; do not report only `best.pt` without checking `last.pt`.

The `predictions_detailed.csv` artifact keeps the `data.yaml` class ID order. The shorter `predictions.csv` is baseline-compatible and may remap integer IDs to the comparison table class order; use the name columns or `metrics.json/classes` when reading it.

## XAI Audit

Run this after a checkpoint is available to inspect fail cases, low-confidence cases, and major confusion pairs. It is evaluation-only and does not create training data, so it does not violate the no-pretrain/no-leak rule.

```powershell
D:\DataAI\.venv\Scripts\python.exe -m trkh.evaluation.xai_audit `
  --checkpoint runs\mango_cls_224_clscrops_grouped_rawguard_bf16_v25_full\checkpoints\last.pt `
  --data D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_grouped_seqsafe_w3\data.yaml `
  --class-name-mode raw --expected-num-classes 5 --split test `
  --output-dir runs\mango_cls_224_clscrops_grouped_rawguard_bf16_v25_full\xai_audit_test_last_v2 `
  --max-cases 56 --mistake-cases 24 --low-confidence-cases 8 --close-margin-cases 12 `
  --per-class-cases 2 --batch-size 64 --num-workers 4 `
  --method all --feature-source stem_last --query-tokens cls_register_mean --rollout-start-layer 1 `
  --robustness-probes --review-high-confidence 0.45 `
  --review-background-threshold 0.12 --review-border-threshold 0.25 `
  --shortcut-drop-threshold 0.15 --top-k 5
```

Use `--query-tokens cls_register_mean` for the v25 command because the classifier head pools CLS plus register tokens. `--method all --feature-source stem_last --robustness-probes` exports raw attention, attention rollout, gradient-weighted rollout, CNN-stem Grad-CAM, register diagnostics, robustness probes, `xai_metrics.json`, `xai_audit_summary.json`, and `review_manifest.csv`.

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

## Four-Class Merge Ablation

Use this ablation only to test the hypothesis that original class `1` should be merged into original class `0`. This does not add data and does not use pretraining.

First build the direct merged dataset:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m trkh.tools.build_classification_merged_classes `
  --data D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_grouped_seqsafe_w3\data.yaml `
  --class-name-mode raw --expected-num-classes 5 `
  --merge-classes 0,1 `
  --merged-name Xoai_Song_AnDuoc_GomChuaVaChuaNhe `
  --output-dir D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_grouped_seqsafe_w3_merge01_4cls `
  --link-mode hardlink --overwrite
```

Then rebuild a grouped/sequence-safe split from the merged dataset. Do not train on the direct merged split for final claims, because preserving the old split creates new near-ID cross-split pairs after classes 0 and 1 become the same label.

```powershell
D:\DataAI\.venv\Scripts\python.exe -m trkh.tools.build_classification_grouped_split `
  --data D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_grouped_seqsafe_w3_merge01_4cls\data.yaml `
  --class-name-mode raw --expected-num-classes 4 `
  --output-dir D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_merge01_4cls_grouped_seqsafe_w3 `
  --near-id-window 3 --train-ratio 0.70 --val-ratio 0.20 `
  --seed 42 --link-mode hardlink --overwrite
```

Leak audit for the regrouped 4-class split:

- exact cross-split groups: `0`
- source-stem cross-split groups: `0`
- near-ID cross-split pairs: `0`
- perceptual aHash cross-split groups: `113`

Current 4-class result is useful as an ablation but not close to the paper target. The stopped retrain `mango_cls_224_merge01_4cls_grouped_ldam_lossaware_v3_retrain` selected epoch `29`, validation macro F1 `0.8183`, final test accuracy `0.7604`, and final test macro F1 `0.7549`. Its validation loss reached the minimum at epoch `15` and then oscillated while macro F1 continued improving, so this is calibration/generalization overfit rather than failed optimization.

Optional model EMA is now supported with `--model-ema --model-ema-decay 0.995`. EMA uses only weights learned from scratch in the current run; it is not pretraining. Probe `mango_cls_224_merge01_4cls_grouped_ldam_ema_v30_probe20`, continued to epoch `35`, selected epoch `26`, validation macro F1 `0.8142`, final test accuracy `0.7604`, and final test macro F1 `0.7564`. Late-stage validation-loss standard deviation fell from `0.0475` to `0.0292`; test class-1 F1 improved from `0.6435` to `0.6649` and class-2 F1 from `0.7412` to `0.7523`. Use EMA when stability and minority/confusable-class balance are preferred, with `--patience 10` to stop after the plateau.

Early-convergence screening on 2026-06-06 found a faster 256px configuration. `mango_cls_256_merge01_4cls_pad_batch32_decay_v42_probe20` reached validation macro F1 `0.8341` with validation loss `0.6400` at epoch 4, satisfying the five-epoch screening target. It stopped at epoch 16, selected epoch 10, and scored test accuracy `0.7629`, test macro F1 `0.7563` on the canonical regrouped split. Ordinal loss and stronger color jitter were rejected because they reduced canonical test macro F1 to about `0.741` and `0.739`.

Dataset identity is critical. The canonical comparison dataset is `cls_crops_merge01_4cls_grouped_seqsafe_w3`. The similarly named direct-merge dataset `cls_crops_grouped_seqsafe_w3_merge01_4cls` has the same split counts but only `914/1615` matching test filenames. `evaluate.py` now warns when the requested `data.yaml` path differs from the one stored in the checkpoint.

Multi-branch token fusion was implemented for research probes without pretrained weights. It adds optional color-stat, edge/texture, and CNN-stem tokens before the patch tokens, while XAI prefix accounting keeps heatmaps aligned to real patches. Unit tests passed `75/75` on 2026-06-05. Initial probes did not beat the v3 baseline on the grouped test split:

- `mango_cls_224_merge01_4cls_grouped_multibranch_v27_probe12`: best val macro F1 `0.7897`, final test accuracy `0.7102`, final test macro F1 `0.7042`.
- `mango_cls_224_merge01_4cls_grouped_multibranch_min_v28_probe10`: best val macro F1 `0.7708`, final test accuracy `0.7300`, final test macro F1 `0.7260`.

Therefore, keep the v3 LDAM settings and add EMA as the current stability-oriented command. Use `--multi-branch-fusion` only as an ablation until a longer run shows better grouped test metrics. The complete command is maintained in `D:\DataAI\AIEx\image_baseline_experiments\Train.md`.

Recheck on 2026-06-06 after `D:\DataAI\AIEx\dataset\data.yaml` was adjusted to 4 raw classes showed that the old `mango_hybrid_224` checkpoint no longer reproduces its historical validation score on the current data: test macro F1 is `0.8129` with primary-image evaluation and `0.8069` with object-crop evaluation. A new scratch classifier-only probe on the adjusted YOLO object-crop data, `mango_cls_256_yolo4_objectcrops_v49_probe8`, reached test accuracy `0.9370`, macro F1 `0.9345`, and weighted F1 `0.9369` after 8 epochs. Class 1 remains the weakest class at F1 `0.8896`; use this run as the current object-crop ablation baseline, not as a grouped/sequence-safe paper result.

For a classification-folder ablation that does not depend on online YOLO cropping and removes crops from source images with multiple objects, use `D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_merge01_4cls_grouped_seqsafe_w3_singleobject`. It was built from the canonical grouped split by keeping only `source_id` values that appear once in `manifest.csv`: `12,304/16,149` crops kept, `3,845` removed. Hard leak checks remain clean: exact duplicate `0`, same source stem `0`, near-ID `0`; average-hash overlap remains a soft warning.

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
