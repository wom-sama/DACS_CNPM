TRKH Kaggle recipe - current version: TRKH_PRETRAINED_CLASSF_B2_TEMPERED_P05_20260731

Run:
  TRKH_CLASSF_BEST_KAGGLE.ipynb

Recommended upload: exactly two ZIP files

1) TRKH_KAGGLE_B2_UPLOAD_BUNDLE_<version>.zip

Required root layout:
  TRKH_CLASSF_BEST_KAGGLE.ipynb
  TRKH_KAGGLE_B2_UPLOAD_MANIFEST.json
  TRKH_pretrained/SOURCE_COMMIT.txt
  TRKH_pretrained/trkh/...
  TRKH_pretrained/configs/...
  TRKH_pretrained/tests/...
  weights/model.safetensors

SOURCE_COMMIT.txt must contain:
  f1d79def090e347c0586efae4003b2579d368a28

Locked model assets:
  normalized TRKH source tree:
  61b860a5549ac172d06c64ee1c5f9fe035cc9853074312f2fe5a2de80390e747
  DINOv3-S/16 model.safetensors:
  2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040

The asset manifest is a JSON object with notebook_contract set to
TRKH_KAGGLE_TWO_ZIP_B2_V1_20260731, protocol set to
TRKH_PRETRAINED_CLASSF_B2_TEMPERED_P05_20260731, experiment set to
b2-tempered-p05, plus source_commit, source_tree_sha256, dino_sha256 and a
files array. Every extracted regular file except the root
manifest itself must appear exactly once in files; no missing or extra file is
accepted. Each entry has:
  {"path": "relative/posix/path", "size_bytes": 123, "sha256": "64-hex"}

2) TRKH_CLASSF_DEV_DATASET_<version>.zip

It may have one wrapper directory but must contain exactly one compatible
data.yaml plus train/ and val/. It must contain no path component named "test"
(case-insensitive), and data.yaml must not have a test key.

Required ordered class schema:
  0 Xoai_Song_Chua_KhoDap
  1 Xoai_Song_ChuaNhe_CoNguyCo
  2 Xoai_Chin_NgotThanh_DeDap
  3 Xoai_ChinGia_NgotGat_KhongVanChuyen
  4 Xoai_Hu_KhongAnDuoc

The uploaded dataset may differ from local class_f in paths, hashes and sample
counts. The notebook computes new train/val provenance, verifies image decoding,
rejects exact train/val byte duplicates and derives full-validation support from
the uploaded val split. It never opens or hashes test content.

Cell 2 discovers at most one ZIP matching each exact prefix under
/kaggle/input and extracts it into a SHA-specific directory under
/kaggle/working. Extraction rejects absolute/traversal paths, duplicate paths,
symlinks, special/encrypted entries, oversized payloads and insufficient disk.
It never uses extractall and never overwrites a partial extraction. Restart the
Kaggle session if a failed extraction directory remains.

Already-extracted mode remains supported: attach no matching ZIP and provide
exactly one source tree, DINO weight and compatible data.yaml under
/kaggle/input. Mixed/ambiguous duplicate inputs fail closed. DATA_YAML_OVERRIDE
and DATA_ROOT_OVERRIDE in the first cell can select an already-extracted path
or a path inside an extracted ZIP.

Enable a Kaggle GPU and Internet for the locked dependency installation. Direct
locks include timm 1.0.27, safetensors 0.7.0, PyYAML 6.0.3, Pillow 11.3.0,
matplotlib 3.9.4 and pytest 8.4.2.

The official local build_train_args function is called with
experiment="b2-tempered-p05". It supplies the DINOv3 model, deterministic
tempered class sampler q_c proportional to n_c**0.5, LDAM-Focal loss,
augmentation, optimizer, scheduler and effective batch-48 configuration.
Kaggle changes only the uploaded data identity and GPU-sized micro-batch.

Run all cells from top to bottom. The 5-epoch probe must reach macro-F1 >= 0.55
and class-1 F1 >= 0.30, then the notebook stops. Review it, set
CONFIRM_FULL=True in the first cell, rerun that cell and the training cell to
open the 30-epoch full run. The probe guard is not a cross-dataset metric claim.

For AUTO_RESUME, restore the same full-run directory under
/kaggle/working/runs. Resume verifies the data tree, development YAML, source,
architecture, DINO and complete train contract. Existing runs are otherwise
never overwritten.

All model selection, evaluation, confusion analysis, XAI and robustness use
train/val only. ONNX export is optional. The final cell writes an evidence
manifest and ZIP under /kaggle/working.
