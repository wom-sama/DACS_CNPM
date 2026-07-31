# TRKH Kaggle B2/T4 local clean-room report

Status: **passed locally; real Kaggle T4 acceptance is still pending**.

- Canonical bundle: `2063da3b2c730e3cb2227e48fd66e794f236ac2e958d8a7b06be2a46efd9e874`, 86,493,012 bytes.
- ZIP CRC, 685 manifest files and all per-file hashes passed.
- External and bundled notebook bytes match; all 12 code cells compile.
- Source commit `7f7f0883cbb71b6a5620fee86c15b996c400a813` and tree `7c8752efe6acb728ed913abe5db165b218c94177e3ef13f5b37ce1c62b19d965` passed.
- Vendored timm 1.0.27 loaded in both parent and child processes; 48 focused tests passed.
- Dataset: 8,278 train and 2,479 validation images, all decodable, zero exact cross-split duplicates. Test was absent and was not opened, hashed, inferred or evaluated.
- Model preflight: 21,588,869 parameters, output shape `[1, 5]`, zero no-decay violations.
- FP16 smoke: exactly 4 train and 2 validation batches, GradScaler 1,024, two optimizer attempts and two successful updates, with all non-finite/skip counters equal to zero.
- `last.pt` contained 164 optimizer-state entries. A separate process reloaded `best.pt` and wrote 32 predictions from one validation batch.

The clean-room used real RTX 4060 CUDA FP16 and torchvision CUDA NMS. Its in-memory host adapter only redirected Kaggle paths, removed Kaggle/Linux/T4 identity checks and reduced Windows DataLoader workers from 2/2 to 0/0. The release notebook itself remains unchanged and requests 2/2 workers on Kaggle Linux.

This report is local engineering evidence, not a claim that Kaggle has accepted the run. The final gate is a top-to-bottom run of the unmodified notebook on Kaggle v170 with `NvidiaTeslaT4`.
