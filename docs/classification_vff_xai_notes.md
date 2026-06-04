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
Khuyen nghi hien tai: xem `README.md`, run name `mango_cls_224_clscrops_grouped_ldam_sam_xaiv2_v16`.

## Ket qua v13 full va chan doan moi

Run `runs/mango_cls_224_clscrops_vffsupcon_fgcap_v13_full` bi dung tai epoch `59`.

- Best validation epoch: `51`.
- Best validation macro F1: `0.889011`.
- Best validation class 1 F1: `0.654275`.
- Final test macro F1: `0.879163`.
- Final test class 1 F1: `0.575758`.
- `high_norm_patch_fraction` ve `0.0` tu khoang epoch 30, nen register token/patch norm da on hon v11/v12.

Nhan xet: v13 sua duoc van de high-norm patch, nhung chua giai quyet duoc class 1. Sau epoch 51, train loss tiep tuc giam trong khi val loss dao dong/tang, nen run co overfit cuoi chu ky. Cac loi class 1 tren test chu yeu la `1 -> 0` va mot phan `1 -> 4`, khong phai loi threshold don gian vi nhieu mau sai co confidence rat cao.

## XAI audit v2

Da bo sung:

- Attention Rollout cho ViT/register tokens.
- Grad-CAM tai `stem_last` de nhin texture/vet cuc bo cua CNN stem.
- Dinh luong `foreground_mass`, `background_mass`, `border_mass`, `entropy` cho tung heatmap.

Audit `runs/mango_cls_224_clscrops_vffsupcon_fgcap_v13_full/xai_audit_test_v2` tren 30 case:

- Attention foreground/background trung binh: `0.9757 / 0.0243`.
- Grad-CAM foreground/background trung binh: `0.9799 / 0.0201`.
- Rollout foreground/background trung binh: `0.9795 / 0.0205`.
- Border mass van khoang `0.1569-0.1913`, nen van can canh giac voi crop sat ria, tay/nen, va object phu.

Ket luan XAI: model khong con chu y nen mot cach tong quat. Loi lon hon la crop nhieu object, tay/nen nam trong vung trung tam, va ranh gioi nhan giua class 1 voi class 0/4. Vi du high-confidence `1 -> 0` co object phu ben phai mang vet den dai, khien model co ly do hinh anh de chon lop khac.

## Leak audit cls_crops

Tool moi `trkh.tools.audit_classification_split` cho ket qua tren `cls_crops`:

- Exact SHA1 duplicate across splits: `0`.
- Same source stem across splits: `0`.
- Same average-hash bucket across splits: `397` bucket / `969` file.
- Same-class near numeric image ID across splits, window `3`: `33333` pair / `14383` file.

Ket luan leak: khong co hard duplicate leak, nhung co rui ro sequence leakage cao do nhieu `Image_N` lien tiep cung class bi chia qua train/val/test. Neu viet bai bao, can tao split grouped/sequence-safe hoac it nhat bao cao audit nay; neu khong, val/test co the lac quan hon thuc te.

## Huong v14

V14 nen chay tren split moi `D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_grouped_seqsafe_w3` de phuc vu bai bao. Split nay duoc tao tu cung anh `cls_crops`, khong them du lieu va khong dung pretrain.

Audit split moi:

- Total images: `16149`.
- Target split counts: train `11304`, val `3230`, test `1615`.
- Class 1 counts: train `464`, val `133`, test `66`.
- Exact SHA1 duplicate across splits: `0`.
- Same source stem across splits: `0`.
- Same-class near numeric image ID across splits, window `3`: `0`.
- Average-hash overlap con `166` bucket, chi la tin hieu mem vi nhieu crop xoai nhin tuong tu nhau.

Cac thay doi cau hinh v14:

- Tang regularization nhe: dropout/drop-path/weight-decay cao hon v13 va scheduler ngan hon de giam overfit sau epoch 51.
- Tang fairness selector: `fair-f1-gap-penalty=1.8`, `fair-f1-min-weight=0.50`.
- Tang rare-class exposure dong o muc vua phai: cap `2.0`, khong quay lai muc raw `2.63` de tranh precision sap.
- Tang nhe metric learning va foreground consistency, nhung van khong dung pretrain hay du lieu ngoai.

## Ket qua v14 grouped va dieu chinh v15

Run `runs/mango_cls_224_clscrops_vffsupcon_fgcap_v14_regfair` da bi dung tai epoch `32`.

- Dataset: `cls_crops_grouped_seqsafe_w3`, khong dung pretrain va khong them du lieu.
- Best theo `fair_macro_f1`: epoch `29`.
- Best validation macro F1 ghi trong summary: `0.682289`.
- Checkpoint best epoch `29`: validation macro F1 `0.656542`, class 1 F1 `0.263852`, val loss `1.251230`.
- Final test: accuracy `0.734985`, macro F1 `0.645249`, class 1 F1 `0.162679`, val/test loss van cao.
- XAI audit test: attention foreground mass `0.9747`, rollout foreground mass `0.9670`, Grad-CAM foreground mass `0.8755`; van con border/background signal nhe nhung khong phai loi chinh.
- Top confusion test cua XAI: true `3` -> pred `2`, true `0` -> pred `1`, true `1` -> pred `2`.

Chan doan:

- Split grouped/sequence-safe kho hon raw `cls_crops` nen diem giam la hop ly, khong nen so truc tiep voi v11/v13 raw split.
- `fair_macro_f1` v14 qua gat voi class gap va min-class, nen co the chon epoch muon co `val_loss` xau.
- Class 1 dang bi over-correction: nhieu mau class 0 bi day sang class 1 voi confidence cao. Vi vay khong nen tiep tuc tang repeat/recall guard cho rare class.
- `train_loss` giam trong khi `val_loss` tang ve sau, nen can checkpoint selector co penalty theo `val_loss` va giam cac loss phu gay co cum qua manh.

Da them trong code:

- `--best-metric loss_aware_fair_macro_f1`.
- `--fair-f1-loss-weight`.
- Selector moi: `macro_f1 + min_weight * min_class_f1 - gap_penalty * max(0, gap - gap_target) - loss_weight * val_loss`.

Huong v15:

- Dung `loss_aware_fair_macro_f1` de tranh chon checkpoint overfit.
- De xuat hien tai: `fair_f1_loss_weight=0.10`; tren history v14 no chon epoch `13` thay vi epoch `29`, trong khi khong qua nang ve epoch rat som.
- Giam rare-class repeat/class-aware cap tu `2.0` xuong `1.6`.
- Tat `rare_class_recall_guard` trong lenh khuyen nghi vi no dang day class 1 qua manh khi precision thap.
- Giam Balanced Softmax tau tu `0.70` xuong `0.45`.
- Giam SupCon tu `0.012` xuong `0.006` va foreground consistency tu `0.025` xuong `0.015`.
- Giu augmentation hinh hoc nhe, khong bat brightness/contrast/saturation/hue/cutmix/mosaic/copy-paste cho nhanh classification-only.

## Ket qua v15 va XAI audit v2

Run `runs/mango_cls_224_clscrops_grouped_lossaware_v15` da bi dung tai epoch `33`.

- Best selection epoch: `21`.
- Best validation macro F1 trong summary: `0.682987`.
- Best metrics file: validation macro F1 `0.679967`, accuracy `0.750774`, class 1 F1 `0.278215`.
- Final test: accuracy `0.725077`, macro F1 `0.637897`, weighted F1 `0.751239`, class 1 F1 `0.117647`.
- Test confusion class 1: predicted class 1 qua nhieu. Cot pred `1` co `189` mau, trong do chi `15` dung class 1.
- Val loss tot nhat o epoch `11`: `0.758571`; ve cuoi run train loss giam den `0.227` nhung val loss dao dong `0.97-1.07`, nen van overfit.

XAI audit v2 da chay tai `runs/mango_cls_224_clscrops_grouped_lossaware_v15/xai_audit_test_v2`.

- Da them `grad_rollout`, register diagnostics, robustness probes va review manifest.
- Top confusions tren test: true `3` -> pred `2` (`96`), true `0` -> pred `1` (`88`), true `2` -> pred `1` (`53`), true `4` -> pred `1` (`31`).
- Heatmap foreground mass van cao: attention `0.9636`, rollout `0.9635`, grad_rollout `0.9566`, Grad-CAM `0.9612`.
- Background probe khong lam giam confidence dang ke: background blur drop `0.0056`, background gray drop `-0.0018`.
- Object desaturate drop trung binh `0.5873`, nen model rat nhay voi mau/texture tren object. Dieu nay dung mot phan voi bai toan do chin, nhung cung giai thich vi sao ranh class 0/1/2 de sai confidence cao.
- Register diagnostics: `cls_register_heatmap_similarity=0.9994`, `register_attention_entropy=0.9409`. Register dang gan nhu nhin giong CLS va kha phan tan, chua tao signal khac biet ro.
- Review flags: `high_confidence_misclassification=17`, `object_color_sensitive=27`, `grad_rollout_border_attention=19`, `register_attention_diffuse=9`.

Ket luan v15:

- Loi chinh khong phai shortcut nen ro ret.
- Can giam over-correction class 1, khong tang rare repeat/rare recall guard nua.
- Can lam register tokens bot trung lap voi CLS: thu `--register-positional-embedding`.
- Can giam overfit cuoi run: thu SAM nhe, weight decay/dropout/drop-path cao hon va scheduler ngan hon.
- Can photometric jitter rat nhe, khong dung hue/cutmix/mosaic/copy-paste. Muc tieu la tang do ben mau nhe chu khong pha tin hieu mau.

Huong v16:

- Chuyen tu Balanced Softmax tau `0.45` sang `ldam_focal` nhe: `ldam_max_margin=0.18`, `ldam_scale=12`.
- Tat class weights trong CE/LDAM de tranh nhan doi bias rare class.
- Giam cap dynamic rare repeat/class-aware tu `1.6` xuong `1.25`.
- Giu `--disable-rare-class-recall-guard`.
- Bat `--register-positional-embedding`.
- Bat `--sam --sam-rho 0.03`.
- Giam LR `2.8e-5`, tang regularization: dropout `0.24`, drop-path `0.22`, weight decay `0.16`.
- Audit sau train bat buoc dung `--method all --robustness-probes`, doc `xai_metrics.json` va `review_manifest.csv` truoc khi quyet dinh tiep.

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
