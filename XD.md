# Notes

## Pretraining Policy

Do not add or enable `timm`, torchvision pretrained weights, YOLO pretrained detectors, external checkpoints, or any model weights not created by this repository from random initialization.

`model.py` rejects common pretrained config keys. Keep that behavior.

## Dataset Split

The local test split was created from the original validation data with `prepare_test_split.py` using seed `42` and ratio `0.4`.

Local manifest:

```text
D:\DataAI\AIEx\dataset\test_split_manifest.json
```

Do not commit dataset files or split copies. Commit only code and documentation.

## Compatibility Scripts

Some older utility scripts remain for analysis/deploy experiments. The active training target is now DETR-style detection through `train.py` and `evaluate.py`.

If an old script assumes classifier-only behavior, update it before using it for model selection.
