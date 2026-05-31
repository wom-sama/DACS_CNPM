# Notes

## Pretraining Policy

Do not add or enable `timm`, torchvision pretrained weights, YOLO pretrained detectors, external checkpoints, or any model weights not created by this repository from random initialization.

`trkh.models.model` rejects common pretrained config keys. Keep that behavior.

## Package Layout

Use module entrypoints for new work:

```bash
python -m trkh.training.train
python -m trkh.evaluation.evaluate
python -m trkh.tools.build_dataset
```

Root `train.py` and `evaluate.py` exist only as compatibility wrappers.

## Dataset Split

Local manifest, if present:

```text
D:\DataAI\AIEx\dataset\test_split_manifest.json
```

Do not commit dataset files or split copies. Commit only code and documentation.

## Dataset Builder

The old local `soan3.py` workflow is now a tracked CLI tool:

```bash
python -m trkh.tools.build_dataset --help
```

It writes YOLO split folders plus `data.yaml` and `canbang.yaml`.
