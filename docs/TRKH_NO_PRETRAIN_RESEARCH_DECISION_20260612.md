# TRKH No-Pretrain Research Decision 2026-06-12

## Van de da duoc xac nhan

- TRKH scratch v4 chi dat test macro F1 `0.8823`, class 1 F1 `0.6545`.
- Top-5 pretrained TTA ensemble dat `0.9248/0.7786`, van sai nhieu o bien `1->0`
  va `1->2`.
- XAI cho thay background perturbation anh huong nho, nhung object desaturation
  lam probability giam manh. Nut that la surface color/texture cuc bo va boundary
  label, khong chi la nen.
- Kien truc hien tai da co conv stem, detail patch enhancement, color/edge branch,
  fine-grained pooling, foreground prior va hard token pruning. Them head tuy y
  khong con la huong uu tien.

## Nguon nghien cuu

### WS-DAN

- Paper: https://arxiv.org/abs/1901.09891
- Code: https://github.com/GuYuc/WS-DAN.PyTorch

WS-DAN dung attention cropping de phong to vung phan biet va attention dropping
de che vung noi bat, buoc model tim them dau hieu. Ma goc chay raw, crop va drop
thanh ba forward moi batch.

### Pairwise Confusion

- Paper:
  https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Abhimanyu_Dubey_Improving_Fine-Grained_Visual_ECCV_2018_paper.pdf
- Code: https://github.com/abhimanyudubey/confusion

Loss goc ghep hai nua batch va toi thieu hoa L2 feature distance. TRKH da co
pairwise margin head nham truc tiep `0-1`, `2-3`, `4-rest`. Pairwise Confusion
chi hop ly neu co sampler tao cap dung; them loss ngay co nguy co keo cac class
gan nhau lai qua muc.

### Early Learning Regularization

- Paper:
  https://proceedings.neurips.cc/paper_files/paper/2020/hash/ea89621bee7c88b2c5be6681c8ef4906-Abstract.html
- Code: https://github.com/shengliu66/ELR

ELR luu EMA prediction theo sample index va ngan model memorization nhan sai.
No phu hop sau khi audit xac nhan label noise. Voi class 1 hien tai, prediction
som thuong lech sang class 0/2, nen bat ELR khong warm-up co the khoa lech nay.

### PMG

- Code: https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training

PMG hoc feature da ti le bang jigsaw va ba nhanh feature. Ma goc chay bon
optimizer step moi batch va 200 epoch. Chi phi khong phu hop gioi han 30 epoch.

### Small-data transformer from scratch

- DHVT:
  https://proceedings.neurips.cc/paper_files/paper/2022/file/5e0b46975d1bfe6030b1687b0ada1b85-Paper-Conference.pdf
- Efficient Training of Visual Transformers with Small Datasets:
  https://proceedings.neurips.cc/paper_files/paper/2021/file/c81e155d85dae5430a8cee6f2242e82c-Paper.pdf
- ViTAE:
  https://proceedings.neurips.cc/paper/2021/file/efb76cff97aaf057654ef2f38cd77d73-Paper.pdf

Cac cong trinh nay cung ung ho local/convolutional inductive bias khi train ViT
tu dau tren du lieu nho. TRKH da co conv stem va local detail branch, nen can
cai thien supervision cho vung cuc bo truoc khi them mot backbone moi.

## Ma tran quyet dinh

| Huong | Loi ich ky vong | Chi phi | Rui ro | Quyet dinh |
|---|---:|---:|---:|---|
| Attention crop/drop co xac suat | Cao | Trung binh | Crop sai khi attention chua hoc | Trien khai |
| ELR co warm-up | Trung binh-cao neu co label noise | Thap | Khoa bias class 1 som | Cho label audit |
| Pairwise Confusion goc | Thap-trung binh | Thap | Trung muc tieu voi pairwise head | Chua dung |
| PMG day du | Trung binh | Rat cao | Khong hoi tu nhanh trong 30 epoch | Loai |
| Backbone scratch moi | Khong chac chan | Cao | Tang do phuc tap, audit lai | Chua dung |

## Trien khai da chon

Attention-guided view training duoc them vao classification path, mac dinh tat:

- Lay `fine_grained_attention` neu fine-grained pooling dang bat.
- Fallback sang patch-token norm neu khong co learned pooling attention.
- Anh xa patch score ve grid goc bang `patch_indices`, nen ho tro token pruning.
- Tron learned score voi foreground prior/pseudo foreground.
- Chon toi da mot view phu cho moi sample:
  - crop vung salient va resize ve input size;
  - hoac blur vung salient de model tim dau hieu phu.
- Chi forward cac sample duoc chon, khong nhan ba toan batch.
- Co warm-up theo epoch de tranh crop theo attention ngau nhien.
- Chi ap dung classification; detection/inference khong doi.

Flags:

```text
--attention-view-loss-weight
--attention-crop-probability
--attention-drop-probability
--attention-view-start-epoch
--attention-crop-threshold
--attention-drop-threshold
--attention-crop-padding-ratio
--attention-crop-min-area-ratio
--attention-view-foreground-weight
--attention-drop-blur-kernel
```

Audit trong `history.csv`:

- `train_attention_view_loss`
- `train_attention_view_fraction`
- `train_attention_crop_fraction`
- `train_attention_drop_fraction`

## Cau hinh thu dau tien

```text
--fine-grained-pooling
--attention-view-loss-weight 0.35
--attention-crop-probability 0.40
--attention-drop-probability 0.20
--attention-view-start-epoch 2
--attention-crop-threshold 0.55
--attention-drop-threshold 0.72
--attention-crop-padding-ratio 0.08
--attention-crop-min-area-ratio 0.25
--attention-view-foreground-weight 0.40
--attention-drop-blur-kernel 15
```

Chi phi forward ky vong khoang `1.6x` neu view fraction dat `0.60`, thap hon
WS-DAN goc `3x`. Smoke phai kiem tra GPU utilization, batch time, view fraction
va gradient truoc full train.

## Gate de tiep tuc

1. Unit/regression test pass.
2. Preflight dataset/leak/config pass.
3. Smoke 1-2 batch pass, khong treo va khong non-finite.
4. Smoke co ghi dung crop/drop fraction.
5. Chi full train neu throughput chap nhan duoc.
6. Full run dung patience `3`; reject neu khong vuot v4 tren validation.
7. Sau run phai xuat class-1 confusion audit va review attention crop/drop mau.
