# TRKH 5-Class Final-Fit Commands 2026-06-27

> **Historical workflow only.** This file merges `train+val` and is not the
> development/selection protocol for a new scientific candidate. Use
> `docs/TRKH_SCIENTIFIC_FULL_TRAIN_AIDT_COMMANDS_20260728.md` for the current
> test-locked train, audit, pretrained-inference, and comparison workflow.

Muc tieu: dung validation da co de chon top-5 expert va so epoch, sau do
train lai tren `train+val` voi checkpoint cuoi cung. Che do nay khong dung
validation de chon checkpoint nua, vi validation da nam trong train.

Khong them TTA percentile-stretch vao full run vi validation MobileNetV3 giam
tu macro/class1 `0.9040/0.7059` xuong `0.9018/0.7048`.

## 1. Final-fit top-5 expert

Chay tu PowerShell:

```powershell
cd D:\DataAI\AIEx\TRKH

$py = 'D:\DataAI\.venv\Scripts\python.exe'
$train = 'D:\DataAI\AIEx\image_baseline_experiments\scripts\02_train_timm_classifier.py'
$data = 'D:\DataAI\AIEx\newdataset\class_f'
$outRoot = 'D:\DataAI\AIEx\image_baseline_experiments\outputs_finalfit_trainval_20260627'

$runs = @(
  @{name='mobilenetv3';      model='mobilenetv3_large_100.ra_in1k';       epochs=22; family='Mobile CNN'},
  @{name='efficientnet_b3';  model='efficientnet_b3.ra2_in1k';            epochs=9;  family='EfficientNet'},
  @{name='efficientnetv2_s'; model='tf_efficientnetv2_s.in21k_ft_in1k';   epochs=25; family='EfficientNet'},
  @{name='densenet121';      model='densenet121.tv_in1k';                 epochs=29; family='DenseNet'},
  @{name='convnext_tiny';    model='convnext_tiny.fb_in22k_ft_in1k';      epochs=24; family='ConvNeXt'}
)

foreach ($r in $runs) {
  & $py $train `
    --data $data `
    --model $r.model `
    --paper-name "FinalFit-$($r.name)" `
    --family $r.family `
    --outdir "$outRoot\$($r.name)" `
    --epochs $r.epochs `
    --batch-size 16 `
    --workers 4 `
    --lr 3e-4 `
    --weight-decay 0.05 `
    --amp `
    --train-splits train,val `
    --final-fit `
    --focus-class-name Xoai_Song_ChuaNhe_CoNguyCo
}
```

## 2. Export TTA test probabilities

```powershell
cd D:\DataAI\AIEx\TRKH

$py = 'D:\DataAI\.venv\Scripts\python.exe'
$data = 'D:\DataAI\AIEx\newdataset\class_f'
$outRoot = 'D:\DataAI\AIEx\image_baseline_experiments\outputs_finalfit_trainval_20260627'
$predRoot = 'D:\DataAI\AIEx\TRKH\runs\finalfit_top5_trainval_tta_5class_20260627'

$names = @('mobilenetv3','efficientnet_b3','efficientnetv2_s','densenet121','convnext_tiny')
foreach ($name in $names) {
  & $py trkh\tools\export_timm_predictions.py `
    --checkpoint "$outRoot\$name\best.pt" `
    --data $data `
    --split test `
    --output-dir "$predRoot\$name" `
    --batch-size 64 `
    --workers 4 `
    --amp `
    --tta-horizontal-flip `
    --tta-brightness-deltas=-0.06,0.06 `
    --tta-contrast-scales=0.92,1.08
}
```

## 3. Build ensemble test audit

```powershell
cd D:\DataAI\AIEx\TRKH

$py = 'D:\DataAI\.venv\Scripts\python.exe'
$predRoot = 'D:\DataAI\AIEx\TRKH\runs\finalfit_top5_trainval_tta_5class_20260627'

& $py -m trkh.tools.build_ensemble_teacher_cache `
  --data configs\class_f_5class.yaml `
  --split test `
  --input "mobilenetv3=$predRoot\mobilenetv3\predictions_test.csv" `
  --input "efficientnet_b3=$predRoot\efficientnet_b3\predictions_test.csv" `
  --input "efficientnetv2_s=$predRoot\efficientnetv2_s\predictions_test.csv" `
  --input "densenet121=$predRoot\densenet121\predictions_test.csv" `
  --input "convnext_tiny=$predRoot\convnext_tiny\predictions_test.csv" `
  --output-dir "$predRoot\ensemble"
```

Doc ket qua chinh trong:

- `runs\finalfit_top5_trainval_tta_5class_20260627\ensemble\teacher_probs_test_metrics.json`
- tung expert: `runs\finalfit_top5_trainval_tta_5class_20260627\<expert>\metrics_test.json`
