# TRKH 5-Class Run - 2026-06-08

## Run hien tai

- Dataset: `D:\DataAI\AIEx\newdataset\class_f`
- Run: `runs\mango_cls_256_5class_defectstat_v3_30e`
- Checkpoint: `runs\mango_cls_256_5class_defectstat_v3_30e\checkpoints\best.pt`
- Gioi han: 30 epoch
- Best epoch: 27
- Stop reason: completed
- Model: TRKH ViT 256, depth 8, 7.27M tham so
- Token pruning: block 2 va 5, keep rates `0.75,0.50`
- Balanced train exposure: `[1843, 1843, 1843, 1844, 1843]`
- Checkpoint selection: `fair_macro_f1_min_class_gap_penalty`
- Tong thoi gian train: 4506.44 giay, khoang 75 phut

## Test sau mask/prior fix

Output: `runs\mango_cls_256_5class_defectstat_v3_30e\final_test_maskfix`

| Metric | Gia tri |
|---|---:|
| Accuracy | 0.9179 |
| Macro precision | 0.8684 |
| Macro recall | 0.8857 |
| Macro F1 | 0.8751 |
| Weighted F1 | 0.9201 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Xoai_ChinGia_NgotGat_KhongVanChuyen | 0.9761 | 0.9185 | 0.9465 | 356 |
| Xoai_Chin_NgotThanh_DeDap | 0.8727 | 0.9339 | 0.9023 | 257 |
| Xoai_Hu_KhongAnDuoc | 0.9719 | 0.9780 | 0.9749 | 318 |
| Xoai_Song_ChuaNhe_CoNguyCo | 0.5778 | 0.7123 | 0.6380 | 73 |
| Xoai_Song_Chua_KhoDap | 0.9433 | 0.8859 | 0.9137 | 263 |

Confusion matrix theo thu tu class trong `metrics.json`/baseline alphabetical:

```text
[[327, 25,  1,  3,  0],
 [  8,240,  3,  5,  1],
 [  0,  1,311,  3,  3],
 [  0,  8,  3, 52, 10],
 [  0,  1,  2, 27,233]]
```

## XAI audit

Output: `runs\mango_cls_256_5class_defectstat_v3_30e\xai_audit_test_defectstat_v3`

- Selected cases: 75
- Attention foreground mass: 0.6825
- GradCAM foreground mass: 0.7906
- GradCAM background mass: 0.2094
- Top confusions: class 3 -> 2, class 0 -> 1, class 1 -> 0, class 1 -> 2.

Trace sau mask fix:

`runs\mango_cls_256_5class_defectstat_v3_30e\architecture_trace_checkpoint_preprocess_maskfix`

## Ket luan

Class 1 van la nut that, nhung khong phai do thieu recall. Precision class 1 thap vi class 0 bi day sang class 1 qua nhieu. Huong tiep theo la hard-negative training train-only, khong tang class 1 toan cuc.

Hard manifest:

`runs\mango_cls_256_5class_defectstat_v3_30e\hard_mining_train_only\hard_samples_train_only.csv`

Audit va command run tiep theo:

`docs\TRKH_5CLASS_OPTIMIZATION_AUDIT_20260608.md`
