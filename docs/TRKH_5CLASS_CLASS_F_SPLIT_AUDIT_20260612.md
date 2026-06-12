# TRKH class_f Split Audit - 2026-06-12

## Pham vi

Dataset:

`D:\DataAI\AIEx\newdataset\class_f`

Tong `13088` anh:

- train: `9215`;
- validation: `2606`;
- test: `1267`.

Artifact:

`runs/class_f_split_leak_audit_20260612/leak_audit.json`

## Ket qua tu dong

- exact SHA1 cross-split: `0` group, `0` file;
- same source stem cross-split: `0`;
- same average-hash cross-split: `144` group, `343` file;
- numeric ID cach nhau toi da 3 qua split: `13606` pair, lien quan `8548` file.

Khong co file byte-identical khong co nghia split doc lap theo vat the.

## Kiem tra truc quan

Chuoi class 3:

- train `Image_1076`;
- validation `Image_1077`, `Image_1087`;
- test `Image_1089`, `Image_1092`;
- train `Image_1091`.

Nhung anh nay cho thay cung mot qua, cung tay/nen va cung phien chup, chi thay
goc nhin nho. Chuoi cua mot vat the da bi rai vao train, validation va test.

Day la source-sequence leakage. Model co the hoc fruit identity, nen, camera va
phien chup thay vi chi hoc maturity/quality. Metric hien tai co the optimistic
va confidence interval tinh theo anh doc lap la khong hop le.

## Doi chieu ViT 1.0

Bang local:

`D:\DataAI\AIEx\image_baseline_experiments\tables\comparison_table.md`

khong co ViT macro F1 `1.0` tren dung protocol `class_f`. Baseline single-model
manh nhat duoc ghi local tren split nay la pretrained MobileNetV3 macro F1
khoang `0.9050`.

Anh chup ViT `1.0` khong kem dataset, split hash, group policy hoac class-wise
metric. Khong du co so de xem day la doi thu truc tiep. Neu no dung random
image split tren chuoi lien tiep, ket qua co the duoc huong loi tu cung leakage.

## Yeu cau sua dataset

1. Tao `group_id` theo qua goc/phien chup, khong theo crop file.
2. Tat ca goc chup cua mot `group_id` chi nam trong mot split.
3. Stratify group theo class; khong di chuyen tung anh rieng le.
4. Audit exact hash, perceptual hash va numeric sequence sau khi split lai.
5. Dong bang test group truoc moi tuning.
6. Bao cao macro F1, per-class F1 va bootstrap CI theo group, khong bootstrap
   tung frame.

## Anh huong den muc tieu

Muc tieu `macro F1=0.99`, class-1 F1 `0.96` khong nen duoc toi uu tiep tren
split hien tai nhu mot bang chung tong quat. Truoc het can:

- group-clean split;
- review label boundary class 0/1/2;
- ghi ro anh qua sang/toi, che khuat, ban va partial-fruit;
- danh gia lai V12/V15 va cac baseline tren cung protocol.

Metric tren split cu van huu ich de so sanh ablation noi bo, nhung khong du de
khang dinh kha nang generalization.
