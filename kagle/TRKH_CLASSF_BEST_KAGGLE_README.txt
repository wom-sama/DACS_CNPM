TRKH Kaggle recipe - current version: TRKH_PRETRAINED_CLASSF_B0_20260731

File to run:
  TRKH_CLASSF_BEST_KAGGLE.ipynb

Attach 3 extracted Kaggle datasets:
  1) the complete canonical class_f folder;
  2) a complete clean archive of TRKH_pretrained commit
     73c96f3d8f42e80c62ab2c4e3c0691ff81f45b77 from branch
     research/pretrained-hybrid-v1, containing trkh/, configs/ and tests/;
  3) the real DINOv3-S/16 model.safetensors file (not a symlink).

Locked hashes:
  class_f/data.yaml:
  312eb376e89023f42e9df24fea8723b64a6b885c15c24f85d8e319d1ff4f1fa8
  DINOv3 model.safetensors:
  2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040
  normalized TRKH source tree:
  d7de5ed0dcd5d9c4eb6e0eeaaef2939505749a6831c7b0885cde172ec822003a

Attach exactly one copy of each input. Run all cells from top to bottom with GPU
enabled. A bounded 5-epoch probe must reach macro-F1 >= 0.55 and class-1 F1
>= 0.30. The notebook then stops intentionally. Review the probe, set
CONFIRM_FULL=True in Cell 1 and rerun the training cell to open the 30-epoch
full run.

Set AUTO_RESUME=True only after restoring the same full-run directory under
/kaggle/working/runs; this automatically skips the probe and verifies last.pt
against the data, class order, architecture and DINO hash. With AUTO_RESUME=False
an existing run directory is rejected instead of overwritten.

The canonical validator hashes the ordered paths and bytes of all 12,019 images,
including test, for static integrity only. It performs no model inference on
test and reads no test metric. The development YAML physically removes test;
train, selection, XAI, robustness, export and reported metrics use train/val.

Do not attach old yolo_f checkpoints or teacher caches: their label semantics,
especially class 1, are incompatible with canonical class_f. B0 is the best
currently executable clean pretrained configuration, not a claimed winning
model until the full validation run finishes.

For offline Kaggle, attach the complete dependency wheel closure. Direct locks
include timm 1.0.27, safetensors 0.7.0, PyYAML 6.0.3, Pillow 11.3.0,
matplotlib 3.9.4 and pytest 8.4.2; the final manifest records the full runtime.
ONNX export is optional and cannot prevent the final manifest/ZIP from being
created. The final cell creates the ZIP in /kaggle/working.

When a better configuration passes the same canonical validation/audit contract,
replace this notebook and update the protocol/date in both files.
