# B3b train-only OOF: raw DINOv3 patch moments

## Why this is distinct from rejected B3a

B3a projected every 384D patch token through the five-class head before
summarizing it. It failed its locked gate and closes that local-logit route.
An independent code audit established before reading B3a metrics that the B2
checkpoint uses exact average pooling over 256 raw patch tokens. Consequently,
standard deviation and coarse spatial moments may still contain information
that the five-logit projection removed.

B3b tests that raw-token hypothesis once. It is not a threshold or loss sweep.
Validation and test remain closed.

## Locked experiment

- B2 EMA best checkpoint, epoch 4, TIMM
  `vit_small_patch16_dinov3.lvd1689m`.
- FP32 extraction only.
- One global five-class `StratifiedGroupKFold`, seed `20260731`.
- Group unit: source image name after removing `_boxNNN`.
- Boundaries: `0-1`, `2-1`, `4-1`; class 1 is positive.
- Control: deployed average-pooled descriptor, 384D.
- Candidate: pooled 384D + per-channel patch standard deviation 384D +
  2x2 spatial patch means 1536D = 2304D.
- Capacity match inside every fold:
  `StandardScaler -> whitened randomized PCA-128 -> balanced logistic`,
  fixed `C=0.30`, threshold `0.5`.
- No descriptor, PCA, C, fold, threshold, pair, or AMP sweep.

Implementation permission requires all of:

- 8,278 train crops, at least 7,000 source groups, complete 15 pair-fold
  predictions, and zero group overlap;
- exact deployed-pool parity (`max abs <= 1e-6`);
- mean pair AUROC gain `>= 0.010`;
- positive AUROC direction in at least `10/15` pair-folds;
- AUROC gain `>= 0.010` in at least `2/3` pairs and no pair below `-0.005`;
- class-1 F1 gain `>= 0.010` in at least `2/3` pairs;
- no pair recall loss beyond `0.020`;
- aggregate class-1 TP retention `>= 0.970`;
- aggregate rival-to-class-1 FP reduction `>= 10%`.

Pass licenses design of one default-off residual specialist plus an equal-
capacity pooled control. It does not license smoke/full training. Failure closes
raw patch moments without relaxing the gate.

## Command

```powershell
$env:PYTHONPATH = "D:\DataAI\AIEx\TRKH_pretrained"

D:\DataAI\.venv\Scripts\python.exe -m trkh.tools.precheck_dinov3_pair_patchstats_train_oof `
  --data D:\DataAI\AIEx\TRKH_pretrained\configs\class_f_5class_dev.yaml `
  --checkpoint D:\DataAI\AIEx\TRKH_pretrained\runs\pretrained_dinov3_classf_tempered_p05_probe_20260731_local_b2_tempered_p05\checkpoints\best.pt `
  --output-dir D:\DataAI\AIEx\TRKH_pretrained\runs\precheck_classf_b3b_dinov3_raw_patch_moments_20260731 `
  --batch-size 32 `
  --workers 0 `
  --torch-threads 4
```
