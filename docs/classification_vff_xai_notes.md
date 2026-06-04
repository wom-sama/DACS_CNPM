# Classification VFF/XAI Notes

Ngay cap nhat: 2026-06-04

## Ket qua v11

Run `runs/mango_cls_224_clscrops_balsoftmax_supcon_v11` dung `cls_crops`, scratch-only, Balanced Softmax va SupCon tren head embedding.

- Best validation epoch: `86`.
- Best validation macro F1: `0.897712`.
- Final test macro F1: `0.887993`.
- Final test class 1 F1: `0.601626`.
- Top test confusion:
  - true `1` -> pred `0`: `20` mau.
  - true `1` -> pred `2`: `9` mau.
  - true `0` -> pred `1`: `9` mau.

Nhan xet: class 1 la nut that that su. Loi khong chi den tu threshold vi nhieu mau sai co confidence rat cao. `high_norm_patch_fraction` khoang `0.033-0.035`, gan muc tu nhien cua nguong `mean + 2*std`, nen chua nen xem day la bang chung register tokens hong. Van de lon hon la ranh gioi fine-grained giua class 1 voi class 0/2 va crop co nhieu texture/context gay nhieu.

## Ket qua v12 va chan doan

Run `runs/mango_cls_224_clscrops_vffsupcon_xai_v12` them SupCon da nguon feature va XAI audit. Run nay bi dung tai epoch `46`, best validation o epoch `40`.

- Best validation macro F1: `0.879000`.
- Best validation class 1 F1: `0.627737`.
- Final test macro F1: `0.876534`.
- Final test class 1 F1: `0.571429`.
- Top XAI confusion tren test: true `1` -> pred `0`: `23` mau, true `2` -> pred `1`: `13` mau, true `0` -> pred `1`: `12` mau.

Nhan xet: v12 khong tot hon v11. SupCon da nguon `head,cnn,patch,registers` va repeat tu `canbang.yaml` day recall class 1 len nhung lam precision roi manh. XAI audit cho thay mot so loi confidence cao tap trung vao ria crop/nen/bong thay vi be mat qua, nen can dieu khien patch attention nhe thay vi tang oversampling tiep.

## Huong v13

Da them co cau an toan hon:

- `--balance-auto-max-repeat-factor`: chan auto repeat/augmentation tu `canbang.yaml`, tranh class hiem bi lap qua manh khi no giong class khac.
- `--foreground-consistency-loss-weight`: phat patch-token energy nam ngoai pseudo foreground trong classification crop, giam hoc nen/ria anh.
- `train_foreground_consistency_loss`: ghi vao `history.csv` de xem regularizer co hoat dong hay khong.

Probe `runs/mango_cls_224_clscrops_vffsupcon_fgcap_v13_probe12` chay 12 epoch, khong dung pretrain va khong co non-finite gradient.

- Best validation macro F1 trong 12 epoch: `0.827638`.
- Best validation class 1 F1 trong 12 epoch: `0.459120`.
- `high_norm_patch_fraction` epoch 12: `0.000234`, giam ro so voi khoang `0.03` cua v11/v12.

Probe nay chua du dai de ket luan se vuot v11, nhung no sua dung dau hieu XAI/high-norm. Lenh day du hien khuyen nghi la `mango_cls_224_clscrops_vffsupcon_fgcap_v13_full` trong `README.md`.

## Huong lay tu tai lieu VFF/Hybrid

Khong thay toan bo training bang Forward-Forward. Huong an toan nhat cho nhanh hien tai la ap dung phan tuong thich:

- Cosine Similarity Contrastive Learning tren embedding, tuong tu tinh than CSCL.
- Layer grouping nhe bang cach tinh SupCon tren nhieu nguon feature: `head`, `cnn`, `patch`, `registers`.
- Giu CNN stem vi no giu texture cuc bo quan trong cho tap du lieu nho.
- Tang regularization vua phai vi v11 co dau hieu train loss giam nhung val loss khong giam tuong ung.

Da them co:

- `--metric-learning-sources head,cnn,patch,registers`
- `--metric-learning-loss-weight`
- `--metric-learning-temperature`

Khuyen nghi v12: xem `README.md`, run name `mango_cls_224_clscrops_vffsupcon_xai_v12`.
Khuyen nghi hien tai: xem `README.md`, run name `mango_cls_224_clscrops_vffsupcon_fgcap_v13_full`.

## Huong XAI da tich hop

Da them `trkh.evaluation.xai_audit`:

- Chon high-confidence mistakes.
- Chon cac cap confusion lon nhat.
- Chon low-confidence va close-margin cases.
- Xuat `xai_audit_summary.json`, `xai_cases.csv`, `xai_audit.md`, va anh overlay attention/Grad-CAM cho tung case.

Da sua attention heatmap de ho tro:

- `--query-tokens cls`
- `--query-tokens registers`
- `--query-tokens cls_register_mean`

Mac dinh audit dung `cls_register_mean` de khop voi `head_pooling=cls_register_mean`.

## Danh gia cac huong XAI tiep theo

- Attention Rollout: nen lam tiep. Chi can forward attention qua cac block, chi phi thap va hop voi ViT-register hon raw attention mot lop.
- Grad-weighted Rollout: nen de sau rollout. Co tinh class-specific, huu ich cho fail cases, nhung can backward moi mau.
- Grad-CAM feature source cho `stem_last`: nen lam tiep. V11 audit cho thay Grad-CAM o patch embedding co the tap trung vao bien/background; stem-level Grad-CAM se tot hon cho vet, dom va texture cuc bo.
- XAI metric subset: nen lam sau khi co rollout/stem Grad-CAM. Metric deletion/insertion patch-level co the dung 100-300 mau, khong nen dua vao train chinh luc nay.
- XAI regularization trong training: chi bat ban nhe `foreground_consistency_loss`. Khong dung label/heatmap tu ben ngoai va khong tao du lieu moi, nen khong vi pham scratch-only/no-pretrain. Neu class 1 recall giam manh, giam weight tu `0.020` xuong `0.012`.

## Lenh audit mau

```powershell
D:\DataAI\.venv\Scripts\python.exe -m trkh.evaluation.xai_audit `
  --checkpoint runs\mango_cls_224_clscrops_balsoftmax_supcon_v11\checkpoints\best.pt `
  --data D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops\data.yaml `
  --class-name-mode raw --expected-num-classes 5 --split test `
  --output-dir runs\mango_cls_224_clscrops_balsoftmax_supcon_v11\xai_audit_test_v2 `
  --max-cases 16 --mistake-cases 5 --low-confidence-cases 3 --close-margin-cases 3 `
  --per-class-cases 2 --batch-size 64 --num-workers 0 `
  --method both --query-tokens cls_register_mean --top-k 5
```
