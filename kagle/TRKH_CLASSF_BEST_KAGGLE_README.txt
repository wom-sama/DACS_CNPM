TRKH Kaggle recipe - current version: TRKH_PRETRAINED_CLASSF_B0_20260731

File:
  TRKH_CLASSF_BEST_KAGGLE.ipynb

Attach 3 extracted Kaggle datasets:
  1) one classification-folder dataset with data.yaml, train/ and val/;
  2) a complete clean archive of TRKH_pretrained commit
     73c96f3d8f42e80c62ab2c4e3c0691ff81f45b77 from branch
     research/pretrained-hybrid-v1, including trkh/, configs/, tests/ and a
     root SOURCE_COMMIT.txt containing that exact 40-hex commit;
  3) the real DINOv3-S/16 model.safetensors file (not a symlink).

The uploaded image dataset may differ from local class_f in file hashes, paths
and sample counts. It must preserve this exact ordered five-class schema:
  0 Xoai_Song_Chua_KhoDap
  1 Xoai_Song_ChuaNhe_CoNguyCo
  2 Xoai_Chin_NgotThanh_DeDap
  3 Xoai_ChinGia_NgotGat_KhongVanChuyen
  4 Xoai_Hu_KhongAnDuoc

If Kaggle contains more than one compatible data.yaml, set DATA_YAML_OVERRIDE
in the first cell. Set DATA_ROOT_OVERRIDE only when train/val cannot be resolved
relative to data.yaml.

Only model assets are release-locked:
  DINOv3 model.safetensors:
  2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040
  normalized TRKH source tree:
  d7de5ed0dcd5d9c4eb6e0eeaaef2939505749a6831c7b0885cde172ec822003a

Before uploading an extracted source folder, create its archive-only marker:
  Set-Content <source-folder>\SOURCE_COMMIT.txt 73c96f3d8f42e80c62ab2c4e3c0691ff81f45b77 -NoNewline
SOURCE_COMMIT.txt is provenance metadata for the Kaggle archive; it is not part
of the normalized trkh/configs source-tree digest.

The notebook does not compare the uploaded data with local class_f hashes or
counts. It computes new train/val provenance hashes, dynamic class counts and
validation support for each run, verifies image decoding, and rejects exact
train/val byte duplicates. It creates a development YAML with no test split.
Test image content is never opened or hashed, and no test inference, metric or
selection is performed.

Run all cells from top to bottom with GPU enabled. The official local B0
build_train_args function supplies the model, optimizer, loss, augmentation,
scheduler and effective batch-48 configuration, so Kaggle differs only in its
uploaded data identity and GPU-sized micro-batch.

A bounded 5-epoch probe must reach macro-F1 >= 0.55 and class-1 F1 >= 0.30.
The notebook then stops intentionally. Review the probe, set CONFIRM_FULL=True
in the first cell, rerun that cell and then rerun the training cell to open the
30-epoch full run. These probe thresholds are engineering guards, not a claim
that two different datasets are directly comparable.

Set AUTO_RESUME=True only after restoring the same full-run directory under
/kaggle/working/runs. Resume verifies data-tree hash, development YAML, class
order, source tree, architecture and DINO hash. With AUTO_RESUME=False an
existing run directory is rejected instead of overwritten.

All selection, full evaluation, confusion analysis, XAI and robustness use the
complete uploaded validation split. The expected support is derived from that
split rather than hard-coded to local class_f.

Do not attach a checkpoint or teacher trained on another label contract. B0 is
the exact executable pretrained control configuration; it is not a winning
claim until full validation and matched comparisons finish.

For offline Kaggle, attach the complete dependency wheel closure. Direct locks
include timm 1.0.27, safetensors 0.7.0, PyYAML 6.0.3, Pillow 11.3.0,
matplotlib 3.9.4 and pytest 8.4.2. ONNX export is optional. The final cell writes
an honest provenance/audit manifest and a ZIP under /kaggle/working.

When a better configuration passes the same validation/audit contract, replace
this notebook and update the protocol/date in both files.
