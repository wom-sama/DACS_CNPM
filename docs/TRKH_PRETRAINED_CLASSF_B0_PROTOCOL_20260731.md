# TRKH pretrained `class_f` B0 protocol

Protocol: `TRKH_PRETRAINED_CLASSF_B0_20260731`

B0 is the clean, executable pretrained control: DINOv3-S/16 with a new
five-class head, hard labels only, no old checkpoint, no old teacher and no
test metrics or test-driven selection. The test split is inspected only as part
of the immutable dataset inventory, including its image-content fingerprint.
B0 is not a claimed winner until full validation is complete.

## Commands

```powershell
$env:PYTHONPATH = 'D:\DataAI\AIEx\TRKH_pretrained'
$Python = 'D:\DataAI\.venv\Scripts\python.exe'
$Data = 'D:\DataAI\AIEx\newdataset\class_f\data.yaml'
$DevData = 'D:\DataAI\AIEx\TRKH_pretrained\configs\class_f_5class_dev.yaml'
$Dino = 'C:\Users\ADMIN\.cache\huggingface\hub\models--timm--vit_small_patch16_dinov3.lvd1689m\snapshots\3bf4720a82ec2066db88137180ff1f83a675cef0\model.safetensors'
```

Preflight:

```powershell
& $Python -m trkh.recipes.pretrained_classf_b0 --mode preflight `
  --data $Data --train-data $DevData --dino-checkpoint $Dino --output-dir runs `
  --run-tag 20260731
```

Five-epoch probe:

```powershell
& $Python -m trkh.recipes.pretrained_classf_b0 --mode probe `
  --data $Data --train-data $DevData --dino-checkpoint $Dino --output-dir runs `
  --run-tag 20260731_probe --batch-size 24
```

Full training is allowed only from a clean pretrained branch after reviewing
the probe:

```powershell
& $Python -m trkh.recipes.pretrained_classf_b0 --mode full `
  --data $Data --train-data $DevData --dino-checkpoint $Dino --output-dir runs `
  --run-tag 20260731_full --batch-size 24 --confirm-full
```

Use `--auto-resume` with the same run tag to resume. Effective batch size is
always 48 (`24x2`, `16x3`, `12x4`, or `8x6`). All stages write a provenance
manifest and keep final test disabled.

The maintained Kaggle version is
`D:\DataAI\AIEx\TRKH\kagle\TRKH_CLASSF_BEST_KAGGLE.ipynb`.
