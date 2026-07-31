# TRKH pretrained `class_f` B1 natural-sampling protocol

Protocol: `TRKH_PRETRAINED_CLASSF_B1_NATURAL_20260731`

## Causal question

The matched B0 probe reached accuracy `0.876160`, macro-F1 `0.835473` and
class-1 P/R/F1 `0.511905/0.816456/0.629268`. Its full validation confusion
contains `76` class-`2 -> 1` errors and `119` restricted `0/2/4 -> 1` false
positives.

B0's strict-balanced sampler exposes every class equally. In each 120-batch
probe epoch this gives class 1 about 576 samples instead of the natural
expectation of about 173. B1 tests whether this 3.33x exposure expands the
class-1 decision region.

## Locked comparison

B1 starts independently from the same DINOv3 weights and seed as B0. The only
semantic delta is `--disable-balanced-epoch-sampling`. Model, hard labels,
LDAM-focal loss, augmentations, optimizer, EMA, schedule, 120 train batches,
complete validation and test lock remain identical. B1 never resumes B0.

Open a longer B1 run only if the five-epoch probe satisfies all:

- accuracy `>= 0.8760`;
- macro-F1 `>= 0.8350`;
- class-1 F1 `>= 0.6500` and recall `>= 0.7000`;
- class `2 -> 1 <= 61`, a reduction of at least 20% from B0.

Otherwise reject sampler-only B1 and run the separately locked LDAM-margin
ablation; do not combine both changes before their individual effects are
known.

## Probe command

```powershell
$env:PYTHONPATH = 'D:\DataAI\AIEx\TRKH_pretrained'
$env:TRKH_AMP_DTYPE = 'bf16'
$Python = 'D:\DataAI\.venv\Scripts\python.exe'
$Data = 'D:\DataAI\AIEx\newdataset\class_f\data.yaml'
$DevData = 'D:\DataAI\AIEx\TRKH_pretrained\configs\class_f_5class_dev.yaml'
$Dino = 'C:\Users\ADMIN\.cache\huggingface\hub\models--timm--vit_small_patch16_dinov3.lvd1689m\snapshots\3bf4720a82ec2066db88137180ff1f83a675cef0\model.safetensors'

& $Python -m trkh.recipes.pretrained_classf_b0 `
  --experiment b1-natural --mode probe `
  --data $Data --train-data $DevData --dino-checkpoint $Dino `
  --output-dir runs --run-tag 20260731_local_b1_natural `
  --batch-size 24 --num-workers 0 --eval-num-workers 0
```

Workers remain `0/0` for this matched quality comparison because B0 was run
with Windows safe-mode workers `0/0`. Multiprocessing throughput is evaluated
separately and must not confound this probe.
