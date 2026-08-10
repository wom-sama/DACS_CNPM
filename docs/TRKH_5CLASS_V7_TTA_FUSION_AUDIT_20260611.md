# TRKH 5-Class V7 / TTA / Fusion Audit 2026-06-11

## Muc tieu

- Tiep tuc huong dot pha sau v4/v5: distill TRKH bang pretrained ensemble, thu TTA/fusion de tang macro F1 va class 1.
- Dataset bat buoc: `D:\DataAI\AIEx\newdataset\class_f`.
- Gioi han train: 30 epochs, early stop neu 3 epochs khong cai thien.

## V7 offline ensemble distillation

Run:

```text
runs\mango_cls_256_5class_v4_top5_offline_distill_v7_30e
```

Ket qua:

- Process khong treo; early-stopping dung o epoch 4, `exit_code=0`.
- Best epoch: 1.
- Validation macro F1: `0.877408`.
- Validation class 1 F1: `0.654867`.
- Final test macro F1: `0.880090`.
- Final test class 1 F1: `0.646707`.
- `architecture_trace` completed trong run dir.

Ket luan: offline soft-label distillation tu top-5 teacher khong vuot v4. Khong nen tiep tuc train student cung cau hinh nay.

## Teacher/fusion experiments

### Top-5 teacher cache

Best teacher-level result truoc TTA:

- `runs\ensemble_top5_teacher_cache_5class_v1`
- Test macro F1: `0.924146`.
- Test class 1 F1: `0.778626`.

Temperature 2 cache:

- `runs\ensemble_top5_teacher_cache_5class_t2_v1`
- Validation macro F1: `0.916615`.
- Test macro F1: `0.923434`.
- Test class 1 F1: `0.778626`.

### Top-10 teacher cache

Exported val/test probabilities cho 5 backbone con lai:

- `efficientnet_b0`
- `resnet50`
- `mobilenetv2`
- `inceptionv3`
- `vgg16`

Top-10 average:

- `runs\ensemble_top10_teacher_cache_5class_v1`
- Validation macro F1: `0.914816`.
- Test macro F1: `0.911726`.

Ket luan: them model yeu vao average lam giam tong the. Khong chon top-10 average.

### Top-10 focus-class specialist

Run:

```text
runs\focus_class1_specialist_top10_pretrained_oof_quality_20260611
```

Ket qua:

- Selected candidate: `extra_trees_balanced_leaf2`.
- OOF macro F1: `0.917537`.
- OOF class 1 F1: `0.754967`.
- Test macro truoc/sau: `0.918328 -> 0.918328`.
- Test class 1 truoc/sau: `0.759690 -> 0.759690`.

Ket luan: specialist khong tao thay doi huu ich tren test.

### TTA export

Da them TTA vao `trkh.tools.export_timm_predictions`:

- `--tta-horizontal-flip`
- `--tta-brightness-deltas=-0.06,0.06`
- `--tta-contrast-scales 0.92,1.08`

Top-5 TTA cache:

```text
runs\ensemble_top5_tta_teacher_cache_5class_v1
```

Ket qua:

- Validation macro F1: `0.913897`.
- Validation class 1 F1: `0.738255`.
- Test macro F1: `0.924826`.
- Test class 1 F1: `0.778626`.

Ket luan: TTA tang nhe macro test so voi top-5 non-TTA (`0.924826` vs `0.924146`) nhung khong cai thien class 1. Co the giu lam inference/fusion artifact, nhung khong giai quyet muc tieu class 1.

## Class 1 audit

Audit:

```text
runs\ensemble_top5_tta_teacher_cache_5class_v1\class1_confusion_audit_test
```

Class 1 metrics:

- TP: `51`
- FP: `7`
- FN: `22`
- Precision: `0.879310`
- Recall: `0.698630`
- F1: `0.778626`

Main confusions:

- `1 -> 0`: 11 mau.
- `1 -> 2`: 9 mau.
- `1 -> 4`: 2 mau.
- `0 -> 1`: 4 mau.
- `2 -> 1`: 2 mau.

Nhan xet mau anh:

- Nhieu mau `1 -> 2` nhin rat giong class 2 ve mau vang/chin.
- Nhieu mau `1 -> 0` van con xanh va chi co vet vang/texture nhe.
- False positive `0 -> 1` thuong co vung vang nhe o day qua.
- Vung phan biet class 1 chu yeu la texture/mau cuc bo tren vo, khong phai nen.

## Oracle ceiling

Dung 10 pretrained experts da export:

- Validation class 1 support: 151.
- Validation any-expert class 1 recall ceiling: `133/151 = 0.8808`.
- Test class 1 support: 73.
- Test any-expert class 1 recall ceiling: `58/73 = 0.7945`.
- Test oracle macro F1 neu chon dung khi bat ky expert dung: `0.956689`.
- Test oracle class 1 F1: `0.872180`.

Ket luan: muc tieu class 1 F1 `0.96` khong kha thi voi tap expert hien tai neu khong co label audit, du lieu bo sung, hoac feature/branch manh hon rat nhieu. Mot so mau class 1 khong duoc bat ky expert nao du doan la class 1.

## XAI/background audit

Run:

```text
runs\mango_cls_256_5class_v4_top5_offline_distill_v7_30e\xai_audit_test_small
```

Tom tat 12 case:

- Foreground mass mean: `0.7748`.
- Background mass mean: `0.2252`.
- Border mass mean: `0.1719`.
- `attention_background_attention`: 8/12.
- `attention_border_attention`: 5/12.
- `object_color_sensitive`: 11/12.
- Background gray/blur lam giam prediction probability gan nhu 0 (`~0.001`).
- Object desaturate lam giam prediction probability manh (`~0.186`).

Ket luan: van co attention ra nen/vien trong mot so case, nhung prediction khong phu thuoc nhieu vao nen khi probe gray/blur. Shortcut chinh la mau/texture cua qua. Huong loc nen nen giu, nhung can tap trung vao surface detail va label boundary.

## AIDT check

Thu export AIDT probabilities tu checkpoint:

```text
D:\DataAI\AIDT\runs\resnet50_vit_b16_class_f_5class_pretrained\best.pt
```

Ket qua:

- Export inline bi timeout sau 15 phut, khong ghi duoc CSV/metrics.
- Hai process Python con sot da duoc dung.
- AIDT qua nang so voi muc tieu train nhanh; khong uu tien lam nhanh trong vong nay.

## De xuat tiep theo

1. Review label bang tay cac anh trong `class1_confusion_audit_test`, dac biet `1->0`, `1->2`, `0->1`, `2->1`.
2. Neu nhan label co nhieu mau ambiguous, tao danh sach `ambiguous_boundary.csv` va loai khoi metric chinh hoac gan nhan lai theo tieu chi ro hon.
3. Neu tiep tuc model, uu tien pretrained feature branch/strong ViT/ConvNeXt feature-level fusion hon la train TRKH scratch/distillation tiep.
4. Thu crop/segmentation object-tight hon cho expert/TTA chi sau khi label audit xac nhan loi do nen/che khuat.
5. Dung top-5 TTA teacher cache lam best current inference artifact neu can baseline manh nhat tam thoi.
