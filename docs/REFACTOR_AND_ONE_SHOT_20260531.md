# Refactor And One-Shot Training Handoff - 2026-05-31

## What Changed

- Moved large root Python modules into `trkh/` subpackages.
- Kept root `train.py` and `evaluate.py` as compatibility wrappers.
- Added `trkh.tools.build_dataset` as the tracked replacement for the old local `soan3.py`.
- Added adaptive detection-loss boosting for one-shot training.
- Added test coverage for the adaptive detection-loss behavior.

## Validation

```powershell
D:\DataAI\.venv\Scripts\python.exe -m compileall train.py evaluate.py scripts trkh tests
D:\DataAI\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Result: `36` tests passed.

Smoke train through the new module entrypoint completed and wrote artifacts:

```text
runs\smoke_refactor_416_q40_one_shot
```

## Stopped Run Summary

Stopped for analysis:

```text
runs\mango_detr_416_q40_5cls_imbalance_copypaste_local_v2
```

Best observed validation row:

- epoch: `18`
- selection metric: `0.688553`
- macro F1: `0.952560`
- best detection F1@50: `0.539130`
- bbox IoU: `0.522800`

Test result from `best.pt`:

- macro F1: `0.958867`
- calibrated macro F1: `0.968724`
- best detection F1@50: `0.549635`
- bbox IoU: `0.520143`

The run was stable, but detection precision/localization lagged too far behind classification.

## Next Run

Use the one-shot command in `README.md`.

Critical differences from v2:

- no manual phase: `--stage1-epochs 0`
- indefinite with early stop: `--epochs 0 --scheduler-total-epochs 160 --patience 100`
- lower LR: `2.5e-5`
- stronger precision pressure: higher background/objectness/cardinality/matcher-objectness
- lighter synthetic composition: lower mosaic/cutmix/copy-paste
- adaptive detection-loss boost enabled

No pretraining is allowed.
