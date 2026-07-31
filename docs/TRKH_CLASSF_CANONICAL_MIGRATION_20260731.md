# Canonical `class_f` migration

Effective 2026-07-31, the only development dataset is:

`D:\DataAI\AIEx\newdataset\class_f\data.yaml`

## Locked contract

| Item | Value |
|---|---|
| `data.yaml` SHA-256 | `312eb376e89023f42e9df24fea8723b64a6b885c15c24f85d8e319d1ff4f1fa8` |
| `manifest.csv` SHA-256 | `59cf846b69f73c72bc415feaa3ce19fae13ca99f2c58ddb1119c0aaacc77a870` |
| `stats.json` SHA-256 | `17d39f7de0e84b578a2ba77c1958ed3de9c73fe9715bc7ebd6c19b465e0a8273` |
| Ordered image-tree SHA-256 | `70a1b7d2b4c6f80e28fe3f0f714f1ba3e8ab654a90ce50b1e9cae8e4dac4a503` |
| Train / val / test | `8278 / 2479 / 1262` |
| Train class counts | `[1987, 497, 1326, 2080, 2388]` |
| Val class counts | `[558, 158, 380, 494, 889]` |

Run the fail-closed validator before training:

```powershell
python -m trkh.tools.validate_canonical_classf `
  --data D:\DataAI\AIEx\newdataset\class_f\data.yaml
```

## Semantic reset

Against the former `yolo_f` labels, `1924/12019` targets changed. Class 1 agrees
in only `299/738` cases. Therefore all old checkpoints, teacher CSV/cache,
thresholds and metrics are historical references only. They must not initialize,
distil, calibrate or select a canonical `class_f` model.

The clean control is DINOv3 initialization plus a new five-class head, trained
from the canonical hard labels. Hybrid transfer is closed until a class-f-native
adaptation protocol explicitly resets incompatible heads and proves its value.

## Input and evaluation contract

- Exact class order from `data.yaml`; mismatch fails closed.
- `256x256`, pad resize, bicubic interpolation, ImageNet normalization.
- Full validation has 2,479 samples; no partial-validation promotion.
- Train/audit commands use `configs/class_f_5class_dev.yaml`, which contains no
  test key. The full canonical YAML is used only for immutable package
  fingerprint/inventory validation. That static check hashes all image bytes,
  including test, but performs no model inference and no test metric enters
  development.
- The first mobile student is hard-label-only; KD opens only after a canonical
  teacher cache is regenerated and audited.
