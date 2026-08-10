# TRKH 5-Class Attention V8 Smoke Audit 2026-06-12

## Pham vi

- Project: `D:\DataAI\AIEx\TRKH`
- Dataset: `D:\DataAI\AIEx\newdataset\class_f`
- Run smoke:
  `runs/smoke_mango_cls_256_5class_attention_views_bounded_v8b`
- Resume tu checkpoint scratch v4; khong dung external pretrained hay distillation.
- Batch `32`, workers train/val `4/2`, AMP `bfloat16`.

## Thay doi attention drop sau audit anh

Trace dau tien cho thay threshold `0.72` chi blur `0.78%-2.19%` dien tich anh.
Vung nay qua nho de ep model tim cue phu.

Da thay bang bounded attention dropping:

- Chon pixel co attention cao.
- Dilation truoc khi gioi han dien tich.
- Dam bao dien tich blur nam trong khoang cau hinh, mac dinh `6%-16%`.
- Ghi `train_attention_drop_area_fraction` vao `history.csv`.
- Architecture trace doc cau hinh tu checkpoint va xuat crop/drop thuc te.

Flags moi:

```text
--attention-drop-dilation-kernel
--attention-drop-min-area-ratio
--attention-drop-max-area-ratio
```

Audit 1 anh moi class dat `0.0600` dien tich drop cho ca 5 class. Crop bam be
mat qua; drop xoa cue noi bat cuc bo hoac cue nen ma model dang su dung, nhung
khong pha vo hinh dang tong the. Artifact nam tai:

```text
runs/smoke_mango_cls_256_5class_attention_views_bounded_v8b/architecture_trace
```

## Regression va preflight

- Full test suite: `101 passed`.
- Dataset counts: train `9215`, val `2606`, test `1267`.
- Strict balanced train exposure:
  `[1907, 1908, 1908, 1907, 1906]`, gap `0.10%`.
- Preflight xac nhan dung Python venv, CUDA, checkpoint va 5 class.
- Smoke 10 train batch + 4 val batch ket thuc exit code `0`.
- Architecture trace 5 class ket thuc thanh cong.

## Throughput va GPU

- Train 10 batch: `22.98s`, gom worker startup; cac batch sau warm-up khoang
  `0.6-0.9s/batch`.
- Validation 4 batch: `8.74s`, gom worker startup va metric.
- Peak allocated VRAM: `4366.5 MiB`.
- Peak reserved VRAM: `5136 MiB`.
- GPU monitor trong train on dinh dat `74%-91%`.
- Cac doan GPU thap trung voi khoi tao DataLoader worker, validation,
  checkpoint/artifact va subprocess architecture trace.

Khong co bang chung deadlock hay CUDA hang. Khong bat persistent workers tren
Windows trong run nay de tranh tang rui ro process worker khong thoat. Batch
`32` + gradient accumulation `2` cho effective batch `64` la diem can bang
giua VRAM, throughput va do on dinh.

## Vi sao train loss cao hon val loss

Smoke:

- `train_loss = 1.7766`
- `train_cls_loss = 1.0400`
- `train_attention_view_loss = 1.7504`
- attention loss contribution xap xi `0.35 * 1.7504 = 0.6126`
- `val_loss = 0.5251`

Train loss la tong classification tren anh augment, attention crop/drop,
metric-learning, foreground consistency va pairwise margin. Val loss dung anh
sach va loss chinh. Vi vay `train_loss > val_loss` la ky vong voi cau hinh nay,
khong phai dau hieu data leak hay tinh loss sai. Chi can canh bao neu
`train_cls_loss` tren anh goc tang dai han trong khi validation suy giam.

## Quyet dinh full run

- Full attention view bat tu epoch `2`; epoch 1 warm-up attention.
- Max `30` epoch, early stopping patience `3`.
- Giu batch `32`, grad accumulation `2`, workers `4/2`.
- Sau run bat buoc xuat class-1 confusion, per-class F1, XAI/background audit
  va so sanh voi v4 `0.8823/0.6545` cung top-5 TTA `0.9248/0.7786`.
