# TRKH pretrained `class_f` B1 LDAM-margin ablation

Protocol: `TRKH_PRETRAINED_CLASSF_B1_MARGIN0_20260731`

B0 over-predicts class 1: P/R/F1 `0.511905/0.816456/0.629268`, with 76
class-`2 -> 1` errors. The matched natural-sampling arm proved sampler exposure
is causal, but overcorrected to P/R/F1 `0.750000/0.474684/0.581395`.

This sibling arm starts again from the same DINOv3 weights and seed. It keeps
strict-balanced sampling and changes exactly one numeric value:
`--ldam-max-margin 0.3 -> 0.0`. LDAM remains enabled, scale stays `18`, focal
gamma/mix stay `1.0/0.1`, and all model, data, augmentation, optimizer,
schedule, EMA, full-validation and test-lock settings remain B0-identical.
It never resumes another checkpoint.

Open a longer run only if the five-epoch probe satisfies all:

- accuracy `>= 0.8760` and macro-F1 `>= 0.8350`;
- class-1 precision `>= 0.5800`, F1 `>= 0.6500`, recall `>= 0.7000`;
- class `2 -> 1 <= 61`.

If it fails, do not combine margin removal with natural sampling. The next
sampler hypothesis must use a single predeclared intermediate exposure between
the two already measured endpoints.

```powershell
$env:PYTHONPATH = 'D:\DataAI\AIEx\TRKH_pretrained'
$env:TRKH_AMP_DTYPE = 'bf16'
$Python = 'D:\DataAI\.venv\Scripts\python.exe'
$Data = 'D:\DataAI\AIEx\newdataset\class_f\data.yaml'
$DevData = 'D:\DataAI\AIEx\TRKH_pretrained\configs\class_f_5class_dev.yaml'
$Dino = 'C:\Users\ADMIN\.cache\huggingface\hub\models--timm--vit_small_patch16_dinov3.lvd1689m\snapshots\3bf4720a82ec2066db88137180ff1f83a675cef0\model.safetensors'

& $Python -m trkh.recipes.pretrained_classf_b0 `
  --experiment b1-margin0 --mode probe `
  --data $Data --train-data $DevData --dino-checkpoint $Dino `
  --output-dir runs --run-tag 20260731_local_b1_margin0 `
  --batch-size 24 --num-workers 0 --eval-num-workers 0
```
