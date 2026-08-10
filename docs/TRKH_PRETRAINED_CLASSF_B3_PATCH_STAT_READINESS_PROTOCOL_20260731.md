# B3 train-only readiness: DINOv3 pairwise patch statistics

## Question

Does the spatial distribution of the current B2 DINOv3 patch evidence add
class-boundary information that is absent from its global average-pooled
representation?

This is a diagnostic gate, not a train run. It opens only `class_f/train`.
Validation and test remain closed.

## Locked comparison

- Checkpoint: B2 tempered-p0.5 best checkpoint, epoch 4.
- Focus class: `1`.
- Rival boundaries: `0-1`, `2-1`, and `4-1`.
- Control features: DINOv3 average-pooled pre-logits plus the five global logits.
- Candidate features: the exact control plus fixed local patch-logit summary
  statistics from all 256 patch tokens.
- Readout: balanced binary logistic regression, `C=0.10`.
- Validation of the readout: five matched `StratifiedGroupKFold` folds, seed 42.
- Group unit: normalized source image (`*_boxNNN` removed).
- No feature/readout/threshold/fold/pair sweep.

The candidate is permitted for implementation only if all checks pass:

- exactly 8,278 canonical train crops and at least 7,000 source groups;
- zero source-group overlap in every fold;
- finite features;
- mean balanced-accuracy gain at least `0.005`;
- mean AUROC gain at least `0.005`;
- mean rival-specificity gain at least `0.010`;
- no pair loses more than `0.020` class-1 recall;
- the critical class-2 boundary gains at least `0.005` balanced accuracy and
  `0.020` rival specificity.

Passing authorizes implementation of one default-off residual patch specialist
and its pooled-capacity control. It does not authorize smoke/full training.
Failure rejects this representation route without relaxing the thresholds.

## Reproducible command

```powershell
$env:PYTHONPATH = "D:\DataAI\AIEx\TRKH_pretrained"
$env:TRKH_AMP_DTYPE = "bf16"

D:\DataAI\.venv\Scripts\python.exe -m trkh.tools.audit_dinov3_pair_patch_stat_readiness `
  --data D:\DataAI\AIEx\TRKH_pretrained\configs\class_f_5class_dev.yaml `
  --checkpoint D:\DataAI\AIEx\TRKH_pretrained\runs\pretrained_dinov3_classf_tempered_p05_probe_20260731_local_b2_tempered_p05\checkpoints\best.pt `
  --output-dir D:\DataAI\AIEx\TRKH_pretrained\runs\audit_classf_b3_dinov3_pair_patch_stat_20260731 `
  --batch-size 64 `
  --workers 4 `
  --amp `
  --torch-threads 4
```
