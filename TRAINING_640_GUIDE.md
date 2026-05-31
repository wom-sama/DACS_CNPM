# Training Guide

The active project target is scratch-only DETR-style mango detection and ripeness classification.

## Non-Negotiable Rule

Do not use pretraining:

- no pretrained backbone
- no ImageNet/timm/torchvision pretrained weights
- no external checkpoint from another project
- no detector/classifier teacher weights

Allowed resume source: checkpoints produced by this repository from random initialization.

## Code Layout

Training code now lives under `trkh/`:

- `trkh.training.train`: main training entrypoint
- `trkh.evaluation.evaluate`: validation/test entrypoint
- `trkh.data.dataset`: YOLO dataset and bbox-aware composition
- `trkh.tools.build_dataset`: dataset organizer replacing the old `soan3.py`

Root `train.py` and `evaluate.py` are compatibility wrappers.

## Required Tests

Windows:

```powershell
$env:PYTHONPATH='D:\DataAI\AIEx\TRKH'
D:\DataAI\.venv\Scripts\python.exe -m compileall train.py evaluate.py scripts trkh tests
D:\DataAI\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Linux/Lightning:

```bash
export PYTHONPATH="$PWD"
python -m compileall train.py evaluate.py scripts trkh tests
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected result after the refactor: `36` tests passed.

## Dataset Builder

```bash
python -m trkh.tools.build_dataset \
  --images-dir /path/to/images \
  --labels-dir /path/to/labels \
  --output-dir /path/to/dataset \
  --audit-dir /path/to/audit_images \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --seed 42 \
  --overwrite
```

It writes `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`, and `canbang.yaml`.

## One-Shot Training Rule

For the current imbalanced 5-class dataset, prefer one long run from scratch instead of manual phase 1/phase 2 resumes:

- `--stage1-epochs 5` lets classification stabilize first, saves `stage1_best.pt` and `stage1.pt`, then continues into full detection in the same run.
- `--matcher-class-cost 0.0` keeps stage 2 Hungarian assignment object-first: box/objectness decide which query owns which object, then classification learns the matched object's label. Stage 1 still uses class-aware matching for the cls-only warmup.
- `--eval-detection-score-mode objectness` ranks/filter detections by objectness; the metric still counts true positives only when the final class label is also correct.
- `--epochs 0 --scheduler-total-epochs 160 --patience 100` trains until early stopping.
- classification guard reduces classification weight only when classification is ahead of detection.
- adaptive detection loss boosts bbox/objectness/cardinality/count pressure when validation says detection lags.
- checkpoints are still scratch-only; do not use external weights.

## Recommended Local Command

Use the command in `README.md` section `One-Shot Local Training Candidate`.

## Detection Augmentation

For datasets with few crowded images:

- `--mosaic-probability` creates 4-image composites.
- `--cutmix-probability` pastes rectangular regions and updates/clips boxes.
- `--copy-paste-probability` pastes real labeled objects into other images and drops heavily occluded base boxes.

Keep these light. The stopped v2 run showed that heavy synthetic composition can increase recall but still leave precision low.

## Current Evidence

- Lightning T4 q16 batch16 scratch: test `macro_f1=0.992270`, calibrated `detection_f1@50=0.828399`.
- Local q40 5-class copy-paste v2 stopped at epoch 30: best val epoch 18, `macro_f1=0.952560`, `best_detection_f1@50=0.539130`, `bbox_iou=0.522800`.
- Local q40 5-class copy-paste v2 `best.pt` test: `macro_f1=0.958867`, calibrated `macro_f1=0.968724`, `best_detection_f1@50=0.549635`, `bbox_iou=0.520143`.
- Local q40 one-shot no-stage1 v1 stopped at epoch 72: best val epoch 60, `macro_f1=0.891489`, `best_detection_f1@50=0.636304`, `bbox_iou=0.654993`.
- The next bottleneck is class 1 precision plus detection precision. The current command removes strict balanced sampling, relies on `canbang.yaml` rare-class repeat, restores a short classification-only stage, keeps detection matching independent from class confidence, and keeps adaptive detection-loss boosting for stage 2.

## Artifact Recovery

```bash
python -m trkh.evaluation.render_history_artifacts \
  --run-dir runs/<run_name>
```

This creates `training_curves.png`, `results.png`, `all_training_metrics.png`, `per_class_training_metrics.png`, `detection_training_metrics.png`, `validation_convergence.png`, and `history_summary.json`.
