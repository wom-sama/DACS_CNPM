# Handover

Last updated: 2026-05-31.

## Active Goal

Train a scratch-only mango DETR/ViT-Registers model toward both:

- `macro_f1 > 0.98`
- `detection_f1@50 > 0.98`

No pretraining is allowed. This includes ImageNet, timm, torchvision pretrained weights, external detector weights, external classifier weights, and foundation-model initialization.

## Current Code State

- Main package: `trkh/`
- Training entrypoint: `python -m trkh.training.train`
- Evaluation entrypoint: `python -m trkh.evaluation.evaluate`
- Dataset builder replacing old `soan3.py`: `python -m trkh.tools.build_dataset`
- Compatibility wrappers kept: root `train.py`, root `evaluate.py`
- Tests expected: `36` passed

## Current Dataset State

- Local data path: `D:\DataAI\AIEx\dataset\data.yaml`
- Current dataset is 5-class raw mode.
- `data.yaml` should contain `nc: 5`, `class_name_mode: raw`, and `canbang_yaml: canbang.yaml`.
- Datasets, run outputs, checkpoints, logs, and downloaded cloud artifacts stay out of git.

## Latest Run Analysis

Stopped local run for analysis:

```text
runs\mango_detr_416_q40_5cls_imbalance_copypaste_local_v2
```

Key result:

- stopped at epoch 30
- best validation selection metric at epoch 18: `0.688553`
- epoch 18 `macro_f1=0.952560`
- epoch 18 `best_detection_f1@50=0.539130`
- epoch 18 `bbox_iou=0.522800`
- test `macro_f1=0.958867`
- test calibrated `macro_f1=0.968724`
- test `best_detection_f1@50=0.549635`
- test `bbox_iou=0.520143`
- no non-finite/OOM issue observed

Conclusion: classification is not the main blocker. Detection precision/false positives and localization are the bottleneck.

## Changes Made For Next Run

- Refactored flat root Python files into package directories.
- Added bbox-aware copy-paste augmentation.
- Added adaptive detection-loss boosting:
  - activates after macro F1 is good enough
  - increases bbox/objectness/cardinality/count/aux pressure when detection lags
  - records `adaptive_detection_loss_multiplier` and active state in `history.csv`
- Added robust dataset organizer at `trkh/tools/build_dataset.py`.
- Updated README, training guide, and Lightning handoff.

## Next Recommended Run

Use `README.md` section `One-Shot Local Training Candidate`.

Important flags:

- `--disable-resume`
- `--stage1-epochs 0`
- `--epochs 0`
- `--scheduler-total-epochs 160`
- `--patience 100`
- `--best-metric macro_detection_hmean`
- `--adaptive-detection-max-multiplier 2.20`

This avoids manual phase1/phase2 and lets the loop adjust loss pressure from validation metrics.

## Files To Read First

- `README.md`
- `TRAINING_640_GUIDE.md`
- `LIGHTNING_TRAINING.md`
- `docs/STABILITY_FIX_20260529.md`
- `docs/BASELINE_RECHECK_20260529.md`
