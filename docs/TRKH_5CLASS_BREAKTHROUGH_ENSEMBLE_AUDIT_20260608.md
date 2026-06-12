# TRKH 5-Class Breakthrough Ensemble Audit - 2026-06-08

## Ket luan

Neu muc tieu la vuot cac doi thu pretrained, huong no-pretrain TRKH don le khong con la duong ngan nhat. XAI v4 cho thay loc nen da kha dung:

- attention foreground mass: `0.8479`;
- grad-rollout foreground mass: `0.9263`;
- background blur lam giam original prediction trung binh chi `0.00035`.

Nut that con lai la ranh gioi class mong va prior feature. Vi vay huong dot pha la dung multi-expert ensemble/selector.

## Ket qua hien tai

| System | Accuracy | Macro F1 | Class 1 F1 | Ghi chu |
|---|---:|---:|---:|---|
| TRKH v4 hardneg maskfix | 0.9234 | 0.8823 | 0.6545 | no-pretrain, single model |
| MobileNetV3 | 0.9408 | 0.9050 | 0.7218 | pretrained single model |
| Top-5 pretrained majority vote | 0.9526 | 0.9226 | 0.7692 | vuot tat ca single baseline hien co |
| Top-3 pretrained + TRKH v4 vote | 0.9463 | 0.9151 | 0.7536 | TRKH tang recall class 1 nhung chua tang macro bang top-5 |
| Top-5 pretrained selector, val-fit | 0.9448 | 0.9034 | n/a | validation-only logistic selector, thap hon majority vote |

Top-5 pretrained majority vote gom:

- MobileNetV3;
- EfficientNet-B3;
- EfficientNetV2-S;
- DenseNet121;
- ConvNeXt-Tiny.

Artifact:

- `runs\ensemble_top5_pretrained_vote_5class`
- `runs\ensemble_top3_pretrained_plus_trkh_v4_vote_5class`

## Upper bound

Oracle tren cung prediction CSV:

| Pool | Oracle Accuracy | Oracle Macro F1 | Oracle Class 1 F1 |
|---|---:|---:|---:|
| Top-5 pretrained | 0.9692 | 0.9524 | 0.8657 |
| Top-5 pretrained + TRKH v4 | 0.9771 | 0.9654 | 0.9051 |

Day khong phai metric deploy duoc, vi oracle biet nhan that. Nhung no chung minh rang cac expert co loi khac nhau; neu train duoc selector bang validation/train-only thi muc tieu macro F1 `0.95` va class 1 F1 `>0.85` la co co so.

Muc tieu moi khi chap nhan pretrained la macro/overall F1 gan `0.99` va class 1 F1 `0.96`. Pool hien tai chua du: ngay ca oracle top-5 pretrained + TRKH v4 moi dat macro F1 `0.9654`, class 1 F1 `0.9051`. Vi vay can them expert khac that su bo sung loi, uu tien AIDT `ResNet50 + ViT-B/16`, ViT/DeiT pretrained va label audit cho nhom common-fail.

## Tool da them

`trkh/tools/ensemble_predictions.py`

Tool nhan nhieu `predictions.csv`, align theo path, vote theo majority/weighted vote va xuat:

- `metrics.json`;
- `predictions.csv`;
- `confusion_matrix.csv`.

Canh bao: chon thanh vien/weight bang test la leakage. Ket qua top-5 o tren la audit nhanh tren output hien co; de bao cao chinh thuc can chon member bang validation hoac cau hinh khoa truoc khi xem test.

`trkh/tools/train_prediction_selector.py`

Tool fit logistic selector tren validation predictions/probabilities va apply mot lan len test. Ket qua thu ngay 2026-06-08:

- top-5 pretrained, unweighted selector: test accuracy `0.9448`, macro F1 `0.9034`;
- top-5 pretrained + TRKH v4, unweighted selector: test accuracy `0.9408`, macro F1 `0.8963`;
- balanced selector lam val macro F1 cao hon nhung test thap hon.

Ket luan: selector don gian chua tot bang majority vote. Khong nen tiep tuc tune selector tren test; can them validation-safe expert moi.

## Lenh tai lap top-5 vote

```powershell
cd D:\DataAI\AIEx\TRKH
$env:PYTHONPATH='D:\DataAI\AIEx\TRKH'

D:\DataAI\.venv\Scripts\python.exe -m trkh.tools.ensemble_predictions `
  --input mobilenetv3=D:\DataAI\AIEx\image_baseline_experiments\outputs\mobilenetv3\predictions.csv `
  --input efficientnet_b3=D:\DataAI\AIEx\image_baseline_experiments\outputs\efficientnet_b3\predictions.csv `
  --input efficientnetv2_s=D:\DataAI\AIEx\image_baseline_experiments\outputs\efficientnetv2_s\predictions.csv `
  --input densenet121=D:\DataAI\AIEx\image_baseline_experiments\outputs\densenet121\predictions.csv `
  --input convnext_tiny=D:\DataAI\AIEx\image_baseline_experiments\outputs\convnext_tiny\predictions.csv `
  --output-dir runs\ensemble_top5_pretrained_vote_5class
```

## Vi sao TRKH v4 chua bat kip

So sanh tung anh voi top-5 ensemble:

- Ensemble sua `55` loi cua TRKH.
- TRKH chi thang lai ensemble `18` anh.
- Con `38` anh ca TRKH, ensemble va MobileNetV3 deu sai.

Top confusion cua TRKH:

- `Xoai_ChinGia_NgotGat_KhongVanChuyen -> Xoai_Chin_NgotThanh_DeDap`: 23.
- `Xoai_Song_Chua_KhoDap -> Xoai_Song_ChuaNhe_CoNguyCo`: 23.
- `Xoai_Chin_NgotThanh_DeDap -> Xoai_Song_ChuaNhe_CoNguyCo`: 10.
- `Xoai_Song_ChuaNhe_CoNguyCo -> Xoai_Chin_NgotThanh_DeDap`: 9.

Artifact:

`runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\comparison_audit_v4_vs_ensemble`

## Huong tiep theo de vuot 0.95/0.85

1. Train expert selector bang validation-safe predictions.

   Can co prediction/probability tren `val` va `test` cho tung expert. Selector chi duoc fit tren `val`, sau do apply mot lan tren `test`. Neu chi co hard label, dung logistic/decision tree tren one-hot vote va disagreement pattern; neu co probability, dung stacking logistic regression.

2. Them AIDT/class_f lam expert.

   AIDT hien la pretrained `ResNet50 + ViT-B/16`. Da them wrapper:

   ```powershell
   powershell -ExecutionPolicy Bypass -File D:\DataAI\AIDT\run_train_class_f_5class.ps1 -UsePretrained true -Epochs 30 -BatchSize 6 -GradAccum 4 -Workers 4
   ```

   Dung `-UsePretrained false` cho ablation scratch. Sau khi train xong, export predictions va them vao ensemble pool.

3. Train TRKH-pretrained ablation neu chap nhan bo rao no-pretrain.

   Baseline pretrained dang thang vi feature prior. Neu TRKH van bi cam pretrained, no chi nen duoc so sanh voi no-pretrain models. Neu muc tieu san pham la metric cao nhat, can cho phep pretrained backbone/teacher distillation.

4. Label audit 38 mau ca ba he deu sai.

   Nhung mau nay kha nang cao la label boundary/noise, anh qua sang/toi, bi che, do, hoac chi thay mot phan qua. Khong nen sua bang prediction test; chi dung de tao review list cho relabel thu cong roi split lai.
