# TRKH 5-Class Distillation V6 Audit 2026-06-11

## Muc tieu

- Dataset bat buoc: `D:\DataAI\AIEx\newdataset\class_f`.
- Muc tieu chat luong: macro F1 gan `0.99`, class 1 (`Xoai_Song_ChuaNhe_CoNguyCo`) gan `0.96`.
- Gioi han train: toi da 30 epochs, dung som neu 3 epochs khong cai thien.
- Khong tiep tuc tang oversampling class 1 vo dieu kien; can uu tien loc nen, tang khac biet dac trung va tan dung pretrained neu co loi.

## Ket qua audit truoc V6

- TRKH v4 `mango_cls_256_5class_hardneg_maskfix_v4_30e`: test macro F1 `0.8823`, class 1 F1 `0.6545`.
- V5 ordinal scratch `mango_cls_256_5class_ordinal_maskaudit_v5_30e`: early-stop epoch 8, best epoch 5, test macro F1 `0.8095`, class 1 F1 `0.5376`; reject. Nguyen nhan chinh: train tu scratch lam mat loi the cua checkpoint v4.
- Logit bias calibration chi tang nhe: macro `0.8823 -> 0.8866`, class 1 `0.6545 -> 0.6582`; reject lam huong chinh.
- OOF CV selector va focus-class specialist deu khong vuot top-5 pretrained majority vote tren test; reject.
- Top-5 pretrained vote ensemble dang la doi thu noi bo tot hon TRKH don le: macro F1 khoang `0.9226`, class 1 F1 khoang `0.7692`.

## Huong V6

V6 fine-tune tu checkpoint v4 best va them teacher distillation tu MobileNetV3 baseline pretrained:

- Student: TRKH v4 architecture, resume `runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\checkpoints\best.pt`.
- Teacher: `D:\DataAI\AIEx\image_baseline_experiments\outputs\mobilenetv3\best.pt`.
- Teacher model: `mobilenetv3_large_100.ra_in1k`.
- Teacher class order: alphabetic baseline.
- TRKH target raw class order: `[Song_Chua_KhoDap, Song_ChuaNhe_CoNguyCo, Chin_NgotThanh_DeDap, ChinGia_NgotGat_KhongVanChuyen, Hu_KhongAnDuoc]`.
- Mapping target-to-teacher: `[4, 3, 1, 0, 2]`.
- Teacher parameters: `4,208,437`, all frozen.

## Thay doi code

- `trkh/training/train.py`
  - Them `--pretrained-distillation` va `--no-pretrained-distillation`.
  - Them teacher checkpoint, distillation weight, temperature, focus class index/weight.
  - Them `_build_pretrained_distillation_teacher(...)` cho checkpoint TIMM baseline.
  - Them `_distillation_loss(...)` dung KL divergence voi temperature scaling.
  - Ghi `train_distillation_loss` vao history.
  - Ghi summary distillation vao `resolved_config.json`.
- `trkh/core/config.py`
  - Them cac field distillation vao `TrainConfig`.
- `scripts/run_trkh_5class_distill_v6.ps1`
  - Launcher co tham so `BatchSize`, `NumWorkers`, `EvalNumWorkers`, `PretrainedDistillation`, `DistillationWeight`, `Smoke`, `MaxTrainBatches`, `MaxValBatches`, `TraceArchitecture`.
  - Mac dinh batch `48`, workers `4/2`, distillation weight `0.08`, patience `3`.
- `tests/test_pretrained_distillation.py`
  - Unit test KL distillation loss finite va backpropagate.

## Smoke da chay

Run: `runs\smoke_trkh_v4_mobilenet_distill_v6_script_20260611`

- Command: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_trkh_5class_distill_v6.ps1 -Smoke -RunName smoke_trkh_v4_mobilenet_distill_v6_script_20260611`
- Ket qua: exit code `0`.
- Resume v4 best thanh cong, optimizer/scheduler/scaler reset.
- DataLoader Windows safe mode:
  - Train requested/effective workers: `4/4`.
  - Val requested/effective workers: `2/2`.
  - Pin memory: disabled.
  - Persistent workers: disabled.
- Balanced epoch exposure: `[1910, 1910, 1911, 1911, 1910]`, chenh lech duoi 10%.
- Distillation summary:
  - loss weight `0.08`
  - temperature `2.0`
  - focus class index `1`
  - focus class weight `1.5`
- Smoke history:
  - `train_loss=1.2061`
  - `train_distillation_loss=4.625` truoc nhan weight, dong gop thuc te khoang `0.37`.
  - `val_macro_f1=0.1947` khong dung de danh gia chat luong vi `--max-val-batches=2`.
- Architecture trace completed tai:
  - `runs\smoke_trkh_v4_mobilenet_distill_v6_script_20260611\architecture_trace`

## Lenh full train de tiep tuc

```powershell
cd D:\DataAI\AIEx\TRKH
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_trkh_5class_distill_v6.ps1
```

Neu can tat teacher distillation:

```powershell
cd D:\DataAI\AIEx\TRKH
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_trkh_5class_distill_v6.ps1 -PretrainedDistillation:$false
```

## Viec can lam sau full run

- Doc `runs\mango_cls_256_5class_v4_mobilenet_distill_v6_30e\final_test_detailed\metrics.json`.
- Chay class 1 confusion audit tren `predictions_detailed.csv`.
- Chay XAI/background audit de xac nhan token/foreground mask co tap trung vao qua, khong bam nen.
- So sanh v6 voi v4, v5 va top-5 pretrained vote.
- Neu v6 khong vuot top-5 vote, thu weight `0.04/0.12` hoac tao teacher ensemble logits tu top-5 pretrained thay vi mot MobileNetV3 teacher.
