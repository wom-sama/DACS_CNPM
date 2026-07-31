TRKH Kaggle B2/T4 offline release - 2026-07-31
================================================

Notebook
  TRKH_CLASSF_BEST_KAGGLE.ipynb

Attach exactly these two ZIP files unchanged
  1. TRKH_KAGGLE_B2_T4_UPLOAD_BUNDLE_20260731.zip
  2. TRKH_CLASSF_DEV_DATASET_20260731.zip

Required Kaggle configuration
  - Accelerator: NVIDIA T4 (the Kaggle UI normally shows T4 x2).
  - Do not select P100.  Kaggle GPU image v170 uses Torch CUDA 12.8 and its
    default build cannot safely execute Pascal sm_60 kernels even when
    torch.cuda.is_available() returns True.
  - Internet may remain OFF.  This release does not call pip and does not
    downgrade or replace Kaggle's Torch, torchvision, NumPy, Pillow,
    Matplotlib, pytest, or ONNX packages.
  - The current training code deliberately uses cuda:0 only.  A second T4 may
    be visible but this release does not claim multi-GPU training.

Recommended upload procedure
  1. Upload the asset ZIP and dataset ZIP as private Kaggle Datasets, or attach
     them directly as notebook inputs.  Keep each ZIP unchanged.
  2. Import/open TRKH_CLASSF_BEST_KAGGLE.ipynb.
  3. In Settings, choose the T4 GPU accelerator.
  4. Leave Internet disabled and run all cells from top to bottom.

What the first Run All proves before the long run
  - safe extraction with traversal, symlink, duplicate, size, CRC and SHA-256
    checks;
  - exact TRKH source commit/tree, DINOv3 weight and DINOv3 Agreement;
  - Python 3.12, Torch 2.10, torchvision 0.25 and CUDA 12.8 runtime family;
  - exact vendored pure-Python timm 1.0.27 without modifying site-packages;
  - real CUDA FP16 matrix multiplication and torchvision CUDA NMS;
  - test-locked train/validation dataset provenance and image decoding;
  - focused source tests;
  - a one-epoch engineering smoke limited to 4 train and 2 validation batches;
  - FP16 GradScaler init scale 1024, at least one real optimizer update, zero
    non-finite/skipped optimizer steps, and non-empty AdamW checkpoint state;
  - independent checkpoint reload and one validation-batch inference;
  - the locked five-epoch B2 tempered-p0.5 probe.

Default scientific gate
  CONFIRM_FULL is False by default.  The notebook completes the smoke and
  five-epoch probe, writes a PROBE_READINESS.json file, and skips full-only
  evaluation/audit cells without throwing an intentional error.

  Review the probe metrics.  If they pass, set CONFIRM_FULL=True in cell 0 and
  rerun from cell 0 downward.  The verified smoke/probe markers are reused and
  the locked 30-epoch full run starts.  Set CONFIRM_FULL=True before the first
  Run All only when you explicitly want smoke -> probe -> full in one session.

Runtime fail-closed behavior
  - P100, non-Linux, Python other than 3.12, or a Torch/torchvision/CUDA family
    different from the v170 target stops before training with an actionable
    message.  Do not repair that by reinstalling Torch inside the notebook.
  - Training/evaluation workers start at 2/2.  If Kaggle reports a worker-start,
    shared-memory or bus error, change NUM_WORKERS and EVAL_NUM_WORKERS to 0 in
    the hardware cell, choose a new RUN_TAG in cell 0, and rerun.  The notebook
    never auto-retries or overwrites a possibly partial scientific run.
  - AUTO_RESUME is fail-closed and expects the matching full-run directory to
    remain under /kaggle/working/runs in the same retained session.

Dataset contract
  ZIP 2 may have one wrapper directory but must contain exactly one compatible
  data.yaml and train/ plus val/.  It must not contain a folder component named
  test and data.yaml must not have a test key.  The uploaded paths, hashes and
  sample counts may differ from the local class_f dataset, but the ordered
  class schema must be exactly:

    0 Xoai_Song_Chua_KhoDap
    1 Xoai_Song_ChuaNhe_CoNguyCo
    2 Xoai_Chin_NgotThanh_DeDap
    3 Xoai_ChinGia_NgotGat_KhongVanChuyen
    4 Xoai_Hu_KhongAnDuoc

Model and evidence
  The model recipe is the exact local B2 winner at source commit
  7f7f0883cbb71b6a5620fee86c15b996c400a813.  This commit preserves the B2
  architecture/recipe and adds the audited FP16 scale plus optimizer-update
  telemetry.  It uses DINOv3-S/16 at 256 px,
  effective batch 48, the deterministic n_c**0.5 tempered sampler, LDAM-Focal,
  EMA and the locked optimizer/scheduler configuration.  Kaggle FP16 is a
  documented runtime port; it is not claimed to be bitwise identical to the
  local Torch 2.6/bfloat16 run.

  After full training, the notebook runs complete validation, class-1
  confusion forensics, XAI, robustness and architecture tracing.  ONNX export
  is optional and cannot invalidate the PyTorch checkpoint/audits.  The final
  evidence ZIP and SHA inventory are written under /kaggle/working.

Third-party notice
  The asset ZIP redistributes the locked DINOv3 checkpoint together with the
  complete DINOv3 Agreement and provenance file.  Use and redistribution remain
  subject to that Agreement, including the research acknowledgment provision.

Official runtime references
  https://github.com/Kaggle/docker-python/releases/tag/v170-GPU-bdf9e0538555f90453619adefb49ba40cfa136db44a9c9be7a42ea715c0aa068
  https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#kaggle-kernels-push
  https://github.com/Kaggle/docker-python/wiki/Missing-Packages
