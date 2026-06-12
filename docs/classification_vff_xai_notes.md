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
Khuyen nghi hien tai: xem `README.md`, run name `mango_cls_224_clscrops_grouped_fgpool_midprior_norepeat_v18_full`.

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

## Ket qua v16c va huong v17

Run `runs/mango_cls_224_clscrops_grouped_ldam_xaiv2_fast_v16c` da hoan thanh den early stopping tai epoch `62`.

- Dataset: `cls_crops_grouped_seqsafe_w3`, khong dung pretrain va khong them du lieu.
- Best epoch: `32`, best validation macro F1 `0.686863`, selection metric `0.496424`.
- Final test: accuracy `0.749226`, macro F1 `0.655481`, weighted F1 `0.773473`.
- Final test class 1: precision `0.065476`, recall `0.166667`, F1 `0.094017`.
- Test confusion class 1: true class 1 co `66` mau, chi `11` duoc doan dung; nhieu mau class 0/2/4 bi day sang pred class 1.
- Val loss tang ve cuoi run trong khi train loss tiep tuc giam, nen v16c van overfit muon.

XAI audit da xuat tai `runs/mango_cls_224_clscrops_grouped_ldam_xaiv2_fast_v16c/xai_audit_test_v2`.

- Top confusion tren test: true `0` -> pred `1` (`79`), true `3` -> pred `2` (`77`), true `4` -> pred `1` (`56`), true `2` -> pred `1` (`38`), true `1` -> pred `2` (`28`).
- Heatmap foreground mass cao: attention `0.9866`, grad rollout `0.9899`, rollout `0.9861`, Grad-CAM `0.9796`.
- Background probe gan nhu khong lam giam confidence: background blur drop `-0.0004`, background gray drop `-0.0015`.
- Object desaturate drop `0.0926`, cho thay tin hieu mau/texture tren object van quan trong nhung khong qua cuc doan nhu v15.
- Review flags dang chu y: near top-2 tie `22`, object color sensitive `18`, Grad-CAM border attention `15`.
- Register diagnostics: `cls_register_heatmap_similarity=0.99998`, `register_attention_entropy=0.9058`. Register token da on dinh nhung van nhin gan giong CLS, chua tao nguon bang chung doc lap ro.

Chan doan:

- Loi chinh cua v16c khong phai shortcut nen. XAI cho thay foreground focus cao va background perturbation khong anh huong dang ke.
- Class 1 khong chi thieu recall; no la ranh gioi fine-grained voi class 0/2/4 va dang bi over-correction, lam precision sap rat manh.
- LDAM + rare repeat/class-aware cap nhe van khong giai quyet duoc, va SAM lam toc do train giam nhieu so voi loi ich quan sat duoc.
- Vi grouped split kho hon raw split, muc macro F1 `0.65-0.69` la canh bao that, khong nen so truc tiep voi cac run raw `cls_crops` truoc day.

Huong v17:

- Bo SAM de lay lai toc do train va giam chi phi lap.
- Bo rare-class repeat trong lenh chinh, dung `--weighted-sampler` de kich hoat strict balanced batch thay cho repeat mau le. Cach nay giup moi batch co ti le lop on dinh hon cho SupCon ma khong nhan doi oversampling.
- Doi `ldam_focal` sang `balanced_softmax` tau thap `0.30`, tat class weights, focal mix rat nhe `0.02`.
- Giam metric-learning loss xuong `0.005`, chi dung nguon `head,cnn`.
- Tang regularization vua phai: dropout `0.28`, attention dropout `0.10`, drop-path `0.25`, weight decay `0.20`, scheduler `70` epoch, patience `22`.
- Giu foreground consistency nhe `0.012` va random erasing `0.03` de giam border/crop-edge cue nhung khong pha tin hieu mau.
- Benchmark local cho thay `batch-size=64`, train `num-workers=6`, eval `num-workers=4` la diem tot nhat hien tai khi bat cac bien moi truong multiprocessing tren Windows.

## Ket qua v17 va probe v18

Run `runs/mango_cls_224_clscrops_grouped_strictbal_bsoftmax_xaiv2_v17` bi dung tai epoch `21`.

- Best epoch: `8`, best validation macro F1 `0.666079`, validation loss `0.988528`.
- Final test: accuracy `0.658204`, macro F1 `0.596498`, weighted F1 `0.699739`.
- Final test class 1: precision `0.093407`, recall `0.515152`, F1 `0.158140`.
- Best validation confusion cho thay pred class 1 bi day qua cao: `523` mau duoc doan la class 1 trong khi support class 1 chi `133`.
- Test confusion class 1 cung bi qua muc: `364` mau pred class 1 trong khi true class 1 chi `66`.

XAI audit `runs/mango_cls_224_clscrops_grouped_strictbal_bsoftmax_xaiv2_v17/xai_audit_test_v2`:

- Top confusion: true `0` -> pred `1` (`159`), true `2` -> pred `1` (`87`), true `4` -> pred `1` (`72`), true `2` -> pred `3` (`70`), true `3` -> pred `2` (`60`).
- Foreground mass van cao: attention `0.9780`, grad rollout `0.9720`, Grad-CAM `0.9780`, rollout `0.9792`.
- Border mass con dang chu y: grad rollout `0.2660`, attention `0.2210`.
- Review flags: `register_attention_diffuse=45/45`, `object_color_sensitive=39/45`, `high_confidence_misclassification=27/45`.

Chan doan v17:

- Loi chinh la over-correction class 1, khong phai shortcut nen tong quat.
- Strict balanced sampler lam prior trong batch gan uniform; Balanced Softmax lai tiep tuc sua theo prior dataset, nen class hiem bi day qua manh.
- Khong nen lap lai cau hinh `--weighted-sampler` + `--classification-loss balanced_softmax` tren grouped split, tru khi chi lam ablation.

Da them trong code:

- `--fine-grained-pooling`.
- `--fine-grained-pooling-dropout`.
- Khoi `FineGrainedPatchPooling`: attention-pooling patch tokens theo global CLS feature roi residual-fuse vao classifier head. Lop fuse cuoi duoc zero-init, nen co the bat khoi nay khi resume checkpoint cu ma logit ban dau khong doi. Day la khoi chung, khong hard-code class 1.

Probe v18 scratch-only, moi probe `8` epoch, validation full, khong final test:

| Run | Cau hinh chinh | Best macro F1 | Val loss | Class 1 F1 | Nhan xet |
| --- | --- | ---: | ---: | ---: | --- |
| `mango_cls_224_clscrops_grouped_fgpool_softprior_v18_probe8` | random sampler, BalancedSoftmax tau `0.08`, repeat cap `1.10` | `0.649387` | `0.826592` | `0.085561` | under-predict class 1 |
| `mango_cls_224_clscrops_grouped_fgpool_midprior_norepeat_v18b_probe8` | random sampler, BalancedSoftmax tau `0.18`, no repeat | `0.669946` | `0.783026` | `0.133829` | on dinh nhat, pred class 1 `136` gan support `133` |
| `mango_cls_224_clscrops_grouped_fgpool_strict_ce_v18c_probe8` | strict balanced sampler, CE/label smoothing, no prior | `0.667974` | `0.923003` | `0.225397` | tang recall class 1 nhung pred class 1 `497`, val loss xau |

Huong v18 hien tai:

- Chon v18b lam lenh full vi val loss tot nhat, macro F1 tot nhat trong probe ngan, va khong sap vao loi over-predict class 1 nhu v17/v18c.
- Khong dung strict balanced sampler trong lenh chinh.
- Khong dung rare-class repeat; dat `--balance-auto-max-repeat-factor 1.0` de chan auto repeat tu `canbang.yaml`.
- Dung Balanced Softmax tau vua phai `0.18`, metric-learning rat nhe `0.002`, foreground consistency nhe `0.008`.
- Dung `head-pooling cls`; register tokens van co trong transformer nhu context/sink tokens, nhung khong trung binh truc tiep vao classifier head.
- Class 1 van la nut that fine-grained that su. V18b chua dat muc dot pha, nhung la huong on dinh hon de chay full truoc khi them co che phuc tap tiep.

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

Audit can dung query token khop voi head pooling cua run. V17 dung `cls_register_mean`; v18 hien khuyen nghi dung `cls`.

## Danh gia cac huong XAI tiep theo

- Attention Rollout: nen lam tiep. Chi can forward attention qua cac block, chi phi thap va hop voi ViT-register hon raw attention mot lop.
- Grad-weighted Rollout: nen de sau rollout. Co tinh class-specific, huu ich cho fail cases, nhung can backward moi mau.
- Grad-CAM feature source cho `stem_last`: nen lam tiep. V11 audit cho thay Grad-CAM o patch embedding co the tap trung vao bien/background; stem-level Grad-CAM se tot hon cho vet, dom va texture cuc bo.
- XAI metric subset: nen lam sau khi co rollout/stem Grad-CAM. Metric deletion/insertion patch-level co the dung 100-300 mau, khong nen dua vao train chinh luc nay.
- XAI regularization trong training: chi bat ban nhe `foreground_consistency_loss`. Khong dung label/heatmap tu ben ngoai va khong tao du lieu moi, nen khong vi pham scratch-only/no-pretrain. Neu class 1 recall giam manh, giam weight tu `0.020` xuong `0.012`.

## Lenh audit mau

```powershell
D:\DataAI\.venv\Scripts\python.exe -m trkh.evaluation.xai_audit `
  --checkpoint runs\mango_cls_224_clscrops_grouped_fgpool_midprior_norepeat_v18_full\checkpoints\best.pt `
  --data D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_grouped_seqsafe_w3\data.yaml `
  --class-name-mode raw --expected-num-classes 5 --split test `
  --output-dir runs\mango_cls_224_clscrops_grouped_fgpool_midprior_norepeat_v18_full\xai_audit_test_v2 `
  --max-cases 56 --mistake-cases 24 --low-confidence-cases 8 --close-margin-cases 12 `
  --per-class-cases 2 --batch-size 64 --num-workers 4 `
  --method all --query-tokens cls --feature-source stem_last --rollout-start-layer 1 `
  --robustness-probes --review-high-confidence 0.45 --top-k 5
```

## Cap nhat v25/v26 - 2026-06-05

Nguon tai lieu moi da doc:

- `deep-research-report (9).md`: phan loai mau can than voi RGB do nhay anh sang/camera; HSV/Lab/histogram co ich nhung khong nen pha hue/saturation qua manh.
- `deep-research-report (10).md`: bai toan fine-grained + long-tail can group-safe split, pair confusion, macro/minority metric, hard-pair audit, va regularization nhe. Khong dung pretrain.

Thay doi code da them:

- `color_audit.png` va `color_audit.json` duoc xuat sau khi load dataset. Artifact nay ve mean RGB/HSV/Lab train/val theo class de xem lech mau va giai thich confusion.
- Rare-class recall guard dung ca `raw_auto_repeat_factors` tu `canbang.yaml`, khong chi dung repeat factor da bi cap. Nhu vay class nao co raw scale >= threshold moi duoc target dong, khong hard-code class 1.
- Them `--dual-patch-norm` va `ColorStatisticFusion` mo rong Lab/HSV histogram, nhung chua khuyen nghi bat trong lenh chinh vi probe mau/dual patch truoc do chua tao dot pha tren grouped split.

Probe v25: `runs/mango_cls_224_clscrops_grouped_rawguard_bf16_fast_v25_probe20`

- Dataset: `cls_crops_grouped_seqsafe_w3`, scratch-only, khong them du lieu.
- Windows loader dung worker that: train `6`, val/eval `4`, bf16. Epoch sau warm-up khoang `56s`, GPU util co luc 80-90%.
- Guard da bat dung sau epoch 2: target class co raw scale `2.6303`, multiplier khoang `1.5-1.7`.
- Best validation macro F1 trong history: `0.673372` o epoch `12`.
- Best class 1 validation F1: `0.176471` o epoch `5`.
- `best.pt` bi chon epoch `5` theo `loss_aware_fair_macro_f1`; test macro F1 chi `0.642724`.
- `last.pt` test tot hon ro: accuracy `0.792570`, macro F1 `0.688339`, weighted F1 `0.792291`, class `Xoai_Song_ChuaNhe_CoNguyCo` F1 `0.1940`.
- So voi v24b last, v25 tang class 1 test F1 tu `0.056` len `0.194` va macro F1 tu khoang `0.651` len `0.688`.

Per-class test v25 `last.pt`:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `Xoai_ChinGia_NgotGat_KhongVanChuyen` | 0.943 | 0.763 | 0.844 | 414 |
| `Xoai_Chin_NgotThanh_DeDap` | 0.687 | 0.679 | 0.683 | 336 |
| `Xoai_Hu_KhongAnDuoc` | 0.808 | 0.928 | 0.864 | 459 |
| `Xoai_Song_ChuaNhe_CoNguyCo` | 0.191 | 0.197 | 0.194 | 66 |
| `Xoai_Song_Chua_KhoDap` | 0.841 | 0.874 | 0.857 | 340 |

XAI v25 `last.pt`: `runs/mango_cls_224_clscrops_grouped_rawguard_bf16_fast_v25_probe20/xai_audit_test_last_v2`

- Top confusion: true `3` -> pred `2` (`80`), true `2` -> pred `1` (`40`), true `1` -> pred `0` (`27`), true `2` -> pred `4` (`26`), true `0` -> pred `4` (`24`).
- Foreground mass cao: attention `0.9646`, grad rollout `0.9700`, Grad-CAM `0.9501`, rollout `0.9669`.
- Background mass thap: khoang `0.03-0.05`; background blur/gray gan nhu khong lam giam confidence.
- Border mass van cao: attention `0.3399`, Grad-CAM `0.5146`, rollout `0.3166`.
- Review flags: `attention_border_attention=38/40`, `gradcam_border_attention=34/40`, `near_tie_top2=15/40`, `object_color_sensitive=19/40`, `register_attention_diffuse=13/40`.
- Object desaturate drop `0.0967`, thap hon cac run color-heavy, nen van dung mau nhung khong phu thuoc cuc doan.

Chan doan v25:

- Van de chinh khong phai background shortcut. Model nhin dung object nhung con bam bien/crop-edge va ranh gioi mau/texture giua class 0/1/2/4 qua mong.
- Guard raw-factor co loi ich that cho class 1, nhung validation class 1 support thap lam checkpoint selection nhieu nhieu. Can evaluate ca `best.pt` va `last.pt`.
- Khong nen tang guard manh hon; v25 da dao dong. Huong tiep theo neu test them la giu v25, thu selection/evaluate protocol tot hon, hoac them hard-pair mining tu train-only confusion neu co manifest train loi.

Probe v26: `runs/mango_cls_224_clscrops_grouped_fgpool_rawguard_bf16_v26_probe20`

- Them `--fine-grained-pooling`, giam repeat cap xuong `1.18`, giam guard target/max multiplier.
- Bi dung som tai epoch `6` vi kem hon v25: best val macro F1 `0.615075`, best class 1 F1 `0.066038`.
- Ket luan: fine-grained pooling cau hinh nay lam hoc cham/kho hon, chua giam duoc loi class 1. Khong dung v26 lam lenh chinh.

Lenh chinh hien tai nen dua tren v25:

- `ldam_focal`, tat class weights, no SAM, no strict balanced sampler.
- `--balance-auto-max-repeat-factor 1.25`.
- Guard raw-factor: target `0.55`, threshold `1.5`, max multiplier `1.8`, min precision `0.05`.
- `head-pooling cls_register_mean`, `register-positional-embedding`, `cnn-feature-fusion`.
- Augmentation hinh hoc nhe, photometric cuc nhe, khong hue, khong mosaic/cutmix/copy-paste/mixup.
- Luon xuat `color_audit`, `evaluate best`, `evaluate last`, va `xai_audit` cho checkpoint duoc bao cao.

## Cap nhat gop class 0+1 thanh 4 class - 2026-06-05

Muc tieu: test gia thuyet original class `1` nen duoc gop vao original class `0`, tao bai toan 4 class de giam nham lan fine-grained giua hai muc song/chua nhe. Quy tac van giu nguyen: khong pretrain, khong them anh, khong dung mosaic/cutmix/copy-paste/mixup.

Dataset da tao:

- Direct merge: `D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_grouped_seqsafe_w3_merge01_4cls`.
- Split dung de train/evaluate: `D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_merge01_4cls_grouped_seqsafe_w3`.
- Mapping: old `0,1 -> new 0`; old `2 -> new 1`; old `3 -> new 2`; old `4 -> new 3`.
- Tong mau: `16149`; train/val/test: `11304/3230/1615`.

Ly do phai regroup:

- Direct merge giu split cu tao `408` near-ID cross-split pairs, vi old class 0 va 1 gan nhau theo sequence sau khi gop thanh cung label.
- Ban regroup sequence-safe co exact/source/near-ID leakage bang `0`.
- Van con `113` perceptual aHash cross-split groups, nen test split van kho va can bao cao leak audit kem metric.

Ket qua probe/full:

| Run | Epoch | Selection | Val macro F1 | Test macro F1 | Test accuracy | Nhan xet |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `mango_cls_224_merge01_4cls_grouped_bf16_v1_probe15` | 15 | macro_f1 | `0.8110` | `0.7359` | `0.7412` | LDAM nhe, test gap lon |
| `mango_cls_224_merge01_4cls_grouped_colorce_v2_probe12` | 12 | loss-aware fair | `0.7807` | `0.7037` best / `0.7200` last | `0.7108` best / `0.7257` last | color-stat + CE nhe kem hon |
| `mango_cls_224_merge01_4cls_grouped_ldam_lossaware_v3_50` | 47 stop | loss-aware fair | `0.8180` | `0.7571` final | `0.7628` final | tot hon v1 tren test, nhung xa 0.98 |

XAI tren v1 test:

- Foreground mass rat cao: attention `0.9796`, grad rollout `0.9854`, Grad-CAM `0.9832`, rollout `0.9829`.
- Background mass thap: khoang `0.0146-0.0204`.
- Object desaturate lam giam predicted probability `0.1172`, nen mau van la tin hieu quan trong.
- Loi chinh khong phai background shortcut; la boundary mau/texture giua new class `1` va `2`, them confusion `0 -> 3` va `3 -> 0`.

Ket luan tam thoi:

- Gop `0+1` lam bai toan hop ly hon ve mat nhan, nhung khong tao dot pha du lon.
- Tren split grouped/sequence-safe, muc `accuracy/macro_f1 > 0.98 trong 50 epoch` khong thuc te neu khong thay doi kien truc/split/nhan. Neu dat 0.98 bang split de hon hoac leak thi khong nen dung cho bai bao.
- Huong tiep theo dang gia la multi-branch feature extraction nhe, nhung phai gioi han tham so de tranh overfit.

## Multi-branch hybrid da trien khai va probe - 2026-06-05

Y tuong da duoc trien khai o muc nhe: cho anh di qua nhieu nhanh dac trung roi hoi tu thanh token/fusion feature, sau do dua vao ViT-register aggregator. Tat ca nhanh khoi tao random, khong dung pretrain.

Kien truc hien co:

- Branch A - color-stat token: RGB/Lab/HSV histogram, moment mau, center-border delta.
- Branch B - edge/texture token: Sobel grayscale, global/center/border edge stats, 4x4 pooled edge map.
- Branch C - CNN-stem token: global pooled feature tu CNN stem hien co.
- Fusion: project tung nhanh ve `embed_dim`, them branch type embedding, concat thanh prefix token.
- ViT aggregator: token order `CLS + registers + branch_tokens + patches`; `num_prefix_tokens` duoc cap nhat de XAI khong nham branch thanh patch.
- Head pooling moi: `cls_branch_register_mean`.
- CLI flags: `--multi-branch-fusion`, `--branch-color-tokens`, `--branch-edge-tokens`, `--branch-cnn-tokens`, `--branch-token-dropout`.

Kiem thu:

- `python -m compileall trkh tests`: pass.
- `python -m unittest discover -s tests -p "test_*.py" -v`: pass `75/75`.

Ket qua probe 4-class grouped:

| Run | Epoch | Val macro F1 | Val class 1 F1 | Test macro F1 | Test accuracy | Nhan xet |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mango_cls_224_merge01_4cls_grouped_ldam_lossaware_v3_50` | 47 stop | `0.8180` | `0.7745` | `0.7571` | `0.7628` | baseline 4-class tot nhat hien tai |
| `mango_cls_224_merge01_4cls_grouped_multibranch_v27_probe12` | 12 | `0.7897` | `0.7021` | `0.7042` | `0.7102` | qua nhieu head phu, test gap xau |
| `mango_cls_224_merge01_4cls_grouped_multibranch_min_v28_probe10` | 10 | `0.7708` | `0.6817` | `0.7260` | `0.7300` | nhanh hon v27 nhung van kem v3 |

Ket luan: multi-branch da san sang lam ablation, nhung chua nen thay lenh chinh. Dau hieu hien tai la branch/token phu tang kha nang hoc train/val som nhung lam generalization tren grouped test kem hon. Neu tiep tuc, nen thu branch nho hon nua hoac chi bat edge token; khong tang len 5 full backbone vi se tang overfit va thoi gian train.

## Danh gia overfit retrain va EMA - 2026-06-05

Run bi dung:

- `mango_cls_224_merge01_4cls_grouped_ldam_lossaware_v3_retrain`
- Best epoch `29`, val macro F1 `0.818315`, test macro F1 `0.754891`, test accuracy `0.760372`.
- Minimum val loss o epoch `15`: `0.756546`.
- Epoch `29`: train loss `0.291051`, val loss `0.809641`.
- Overfit bat dau ro theo loss sau epoch `15`, nhung F1 van tang den epoch `29`. Day la overfit calibration/generalization, khong phai model khong hoc duoc.
- XAI: attention foreground `0.9821`, background `0.0179`; loi chinh la confusion `2 -> 1` va `0 -> 3`, khong phai background shortcut.

Probe CE regularization:

- `mango_cls_224_merge01_4cls_grouped_ce_reg_v29_probe20`
- Early stop epoch `18`, best epoch `10`.
- Val macro F1 `0.788942`, test macro F1 `0.706843`.
- Ket luan: bo LDAM va tang label smoothing lam calibration som tot hon nhung tong quat hoa kem hon; khong dung lam lenh chinh.

EMA da trien khai:

- CLI: `--model-ema --model-ema-decay 0.995`.
- Validation va `best.pt` dung EMA weights.
- `last.pt`/`interrupt.pt` giu train weights va `ema_model_state` de resume dung optimizer va EMA.
- Tat ca EMA weights deu sinh ra tu run scratch hien tai, khong co pretrain.
- Unit tests pass `76/76`; smoke checkpoint va resume EMA pass.

Ket qua `mango_cls_224_merge01_4cls_grouped_ldam_ema_v30_probe20` sau khi resume den epoch `35`:

| Metric | Retrain khong EMA | EMA |
| --- | ---: | ---: |
| Best epoch | `29` | `26` |
| Val macro F1 | `0.818315` | `0.814180` |
| Test accuracy | `0.760372` | `0.760372` |
| Test macro F1 | `0.754891` | `0.756398` |
| Late val-loss std | `0.0475` | `0.0292` |
| Test class 1 F1 | `0.6435` | `0.6649` |
| Test class 2 F1 | `0.7412` | `0.7523` |

XAI EMA:

- Attention foreground `0.9805`, background `0.0195`.
- Grad-CAM border mass giam `0.4148 -> 0.3668`.
- Rollout-border flags giam `15 -> 7`.
- Object desaturate probability drop tang `0.1020 -> 0.1128`, xac nhan model van phu thuoc manh vao mau.
- Register-CLS heatmap similarity van gan `1.0`; register token chua tao focus rieng ro.

Ket luan:

- EMA giam dao dong va cai thien hai class nham lan nhieu nhat, nhung khong giai quyet domain/color shift.
- Cau hinh chinh moi giu nguyen LDAM v3, them EMA decay `0.995`, `epochs 45`, `patience 10`.
- Khong tang them branch/backbone trong run chinh. Muc `0.98` van khong kha thi tren grouped split neu khong thay doi chat luong nhan hoac thu thap them du lieu.

## Sang loc hoi tu som 256px - 2026-06-06

Tieu chi sang loc: trong 5 epoch dau phai co it nhat mot epoch `val_macro_f1 >= 0.82`, `val_loss <= 0.70`; loss giam ro trong warm-up va khong dung pretrained weight.

Ket qua:

| Run | Ket qua som | Test canonical | Ket luan |
| --- | --- | --- | --- |
| `mango_cls_256_merge01_4cls_pad_batch32_decay_v42_probe20` | epoch 4: F1 `0.8341`, loss `0.6400` | accuracy `0.7629`, macro F1 `0.7563` | tot nhat, giu |
| `mango_cls_256_merge01_4cls_pad_ordinal_v43b_probe5` | epoch 4: F1 `0.8373`, loss `0.6366` | macro F1 `0.7411` | ordinal khong tong quat hoa, loai |
| `mango_cls_256_merge01_4cls_domainjitter_v46_probe5` | best F1 `0.8193`, loss `0.6824` | macro F1 `0.7388` | jitter mau manh hon, loai |

Luu y du lieu:

- Canonical: `cls_crops_merge01_4cls_grouped_seqsafe_w3`.
- `cls_crops_grouped_seqsafe_w3_merge01_4cls` la split khac: chi trung `914/1615` ten anh test.
- Mot lan evaluate nham split thu hai cho macro F1 `0.8016`; ket qua nay khong duoc dung de so sanh voi baseline tren canonical.
- `evaluate.py` da co warning khi `--data` khac `data_yaml` luu trong checkpoint.

Ket luan:

- Cau hinh 256px, batch 32, LR `1.65e-4`, warm-up 4, cosine horizon 20 dat hoi tu som tot nhat.
- `epochs=50` chi la tran. Dung `patience=6`; khong ep loss giam co dinh `0.05-0.1` sau khi da vao plateau.
- Muc `0.92` tren canonical chua dat. Tran test lap lai hien tai quanh `0.756`; validation cao hon test la domain/sequence shift, khong the sua chi bang tang epoch.
- Lenh chinh moi nam trong `D:\DataAI\AIEx\image_baseline_experiments\Train.md`.

## Audit lai run lich su `mango_hybrid_224` - 2026-06-06

Muc dich: chi tham khao ly do run cu co validation cao, khong quay lai cau truc `vit_registers_hybrid` va khong dung pretrained weight.

Ket qua doc artifact:

- Run: `D:\DataAI\AIEx\TRKH\runs\mango_hybrid_224`.
- Data luu trong run: `D:\DataAI\AIEx\dataset\data.yaml`, khong phai bo `cls_crops*_grouped_seqsafe*` hien tai.
- Checkpoint/run config la 4 class, trong khi `D:\DataAI\AIEx\dataset\data.yaml` hien tai da la 5 class. Vi vay `evaluate.py` hien tai tu choi evaluate checkpoint cu bang data.yaml moi do mismatch class count.
- `best_metrics.json` la validation, khong phai test: support `[674, 541, 366, 594]` trung voi `val_class_counts`.
- Best validation epoch `109`: accuracy `0.971034`, macro F1 `0.970986`, val loss `0.944368`, bbox IoU `0.814573`.
- Train report: `valid_object_count=9392`, `selected_sample_count=7618`, `ignored_object_count=1774`, `multi_object_image_count=844`.
- Val report: `valid_object_count=2751`, `selected_sample_count=2175`, `ignored_object_count=576`.
- Test report: `valid_object_count=1292`, `selected_sample_count=1093`, `ignored_object_count=199`.

Ly do run cu nhin hieu qua hon:

- Bai toan khac: 4 class lich su, khong phai 5 class raw hoac 4-class merge grouped/sequence-safe hien tai.
- Split khac va co kha nang de hon: khong co grouped/sequence-safe audit nhu bo hien tai, nen khong duoc dung lam bang chung cho muc tieu paper tren canonical grouped test.
- Dataset cu chi chon mot object chinh moi anh; nhieu object hop le bi bo qua. Dieu nay lam bai toan phan loai sach/de hon nhung khong phan anh day du phan bo object.
- Model cu co bbox auxiliary loss (`bbox_l1=5.0`, `bbox_giou=2.0`), nen nhan duoc tin hieu dinh vi vung doi tuong. Day la inductive bias huu ich, nhung khong phu hop neu bai bao chi muon classification-only sach.
- Augmentation cu rat manh (`batch_mix=0.6`, `mosaic=0.25`, `mixup=0.45`, `cutmix=0.45`). Cac phep nay co the regularize validation cu, nhung da bi loai khoi lenh classification-only vi tron mau/nhan va khong sach cho thuc nghiem phan loai do chin.
- LR cu cao hon nhieu (`1e-3`) va train dai (`139` epoch). No hoc nhanh tren validation cu: epoch 5 da dat macro F1 `0.9036`, epoch 109 dat `0.9710`.

Dieu co the hoc theo ma khong copy lai cau truc cu:

- Giu inductive bias object-centered: dung crop doi tuong chat luong cao, margin hop ly, va XAI/foreground audit de tranh focus vao nen.
- Neu can thu tiep, chi nen them regularization/auxiliary train-only nhe dua tren foreground/part consistency; khong bat DETR decoder, objectness, bbox regression lam metric chinh.
- Luon evaluate tren dung `data.yaml` luu trong checkpoint va tren canonical grouped split; khong so sanh validation cu voi grouped test hien tai.
- Muc `0.97` cua run cu la moc tham khao ve split cu, khong phai bang chung rang current grouped classification-only co the dat `0.98` chi bang doi hyperparameter.

## Recheck sau khi `D:\DataAI\AIEx\dataset\data.yaml` duoc chinh lai 4 class - 2026-06-06

Dataset YOLO hien tai da duoc chinh thanh 4 class raw:

- `0`: `Xoai_Song_Chua_KhoDap`
- `1`: `Xoai_Chin_NgotThanh_DeDap`
- `2`: `Xoai_ChinGia_NgotGat_KhongVanChuyen`
- `3`: `Xoai_Hu_KhongAnDuoc`

Da bo sung compatibility loader de doc lai checkpoint legacy `mango_hybrid_224`: checkpoint nay luu `model_type=vit_registers_hybrid`, co `head.*` va `bbox_head.*`, nhung khong co DETR decoder/query/objectness. Loader moi normalize ve `vit_registers` khi evaluate classification, bo qua `bbox_head.*` va `classification_head.*` legacy. `evaluate.py` cung xac dinh detection/classification mode sau khi build model de khong chon sai criterion.

Ket qua evaluate lai run cu tren data 4-class hien tai:

| Run/checkpoint | Dataset mode | Split | Accuracy | Macro F1 | Ghi chu |
| --- | --- | --- | ---: | ---: | --- |
| `mango_hybrid_224/best.pt` | primary image only | test | `0.8115` | `0.8129` | khong con dat muc validation cu |
| `mango_hybrid_224/best.pt` | object crops | test | `0.8097` | `0.8069` | object-level cung khong tot hon |
| `mango_hybrid_224/best.pt` | primary image only | val | `0.8028` | `0.8031` | validation hien tai khac artifact cu |

Probe moi train tu dau tren YOLO object-crop 4-class:

| Run | Epochs | Test accuracy | Test macro F1 | Test weighted F1 | Ket luan |
| --- | ---: | ---: | ---: | ---: | --- |
| `mango_cls_256_yolo4_objectcrops_v49_probe8` | `8` | `0.9370` | `0.9345` | `0.9369` | vuot run cu ro rang, chua overfit trong 8 epoch |

Per-class F1 cua v49 tren test:

- class 0: `0.9549`
- class 1: `0.8896`
- class 2: `0.9500`
- class 3: `0.9434`

Nhan dinh:

- Loi the that cua huong moi khong phai do copy lai `vit_registers_hybrid`, ma do dung object-level crops day du, khong bo qua object trong anh nhieu nhan, va classifier-only ViT-register + CNN stem/fusion duoc regularize nhe hon.
- `mango_hybrid_224` tung co validation cao tren artifact cu, nhung khi data.yaml da chinh lai va evaluate lai dung test hien tai thi chi quanh `0.81` macro F1.
- v49 van yeu nhat o class 1, nham lan chinh la class 1 -> class 2/3 va class 2/3 -> class 1. Huong tiep theo nen keo dai v49 len 20-30 epoch, audit XAI tren class 1, va chi tang regularization/metric loss dong theo class gap thay vi hard-code class 1.

Kiem tra code:

- `python -m compileall trkh tests`: pass.
- `python -m unittest discover -s tests -v`: pass `80/80`.

## Single-object-only ablation cho canonical 4-class split - 2026-06-06

Theo yeu cau tach khoi YOLO online de de so sanh voi cac baseline classification-folder, da tao bo:

`D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops_merge01_4cls_grouped_seqsafe_w3_singleobject`

Cach tao:

- Doc `manifest.csv` cua `cls_crops_merge01_4cls_grouped_seqsafe_w3`.
- Dem `source_id`.
- Chi giu cac row co `source_id` xuat hien dung 1 lan.
- Loai toan bo crop tu source co nhieu box, vi do la anh goc nhieu object.
- Dung hardlink, khong copy them data va khong dung pretrain.

Thong ke:

| Split | Class 0 | Class 1 | Class 2 | Class 3 | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | `1997` | `1893` | `2763` | `1481` | `8134` |
| val | `699` | `655` | `770` | `593` | `2717` |
| test | `383` | `331` | `411` | `328` | `1453` |

Tong quan:

- Source rows: `16149`.
- Kept rows: `12304`.
- Removed rows: `3845`.
- Removed multi-source IDs: `1243`.
- Hard leak audit: exact duplicate `0`, source stem cross-split `0`, near-ID cross-split `0`.
- Average-hash cross-split: `111` buckets / `265` files, tiep tuc xem la canh bao mem do xoai crop rat giong nhau.

Tool moi:

- `trkh.tools.filter_classification_single_source`
- Unit test: `test_filter_classification_single_source_removes_multi_box_sources`.
- Full tests sau khi them tool: `python -m unittest discover -s tests -v` pass `81/81`.

Lenh train day du duoc ghi trong `D:\DataAI\AIEx\image_baseline_experiments\Train.md`, muc `TRKH 4-Class Single-Object-Only Ablation`.

## Audit provenance va semantic label cua bo merge01 - 2026-06-06

Da truy nguoc day du duong tao dataset:

`AIEx\dataset` 5 class -> `cls_crops` -> grouped 5 class -> merge old `0+1` -> regroup canonical 4 class -> loc single-source.

Ket qua doi chieu:

- `16149` manifest rows.
- `0` missing YOLO label.
- `0` mismatch giua old-to-new mapping va nhan 4 class hien tai.
- Mapping da dung: `0->0`, `1->0`, `2->1`, `3->2`, `4->3`.

Do do khong co loi copy file hoac remap class ID trong pipeline. Rui ro nam o quy tac semantic: new class `0` gom ca `Xoai_Song_Chua_KhoDap` va `Xoai_Song_ChuaNhe_CoNguyCo`. Chuoi `Image_7002..7011` thuoc old class `1`, sau merge nam o new class `0`, co nhieu dom be mat va bi checkpoint v3 du doan new class `3` tren ca chuoi. Can human review theo dinh nghia nghiep vu truoc khi relabel; khong tu dong doi nhan val/test bang prediction cua model.

Run `mango_cls_256_merge01_4cls_grouped_stable_v3_30e`:

- best epoch `5`;
- best validation macro F1 `0.8407`;
- test accuracy `0.7418`;
- test macro F1 `0.7357`;
- test class `0 -> 3`: `88`;
- test class `3 -> 0`: `36`.

Lenh v4 trong `Train.md` da bo cac co che cu cua bai toan 5 class: imbalance auto-tune, class-aware augmentation, rare repeat/recall guard, LDAM, focal mixing, fair-loss va metric auxiliary. Cau hinh moi dung CE tuong duong, batch `48`, learning rate `1e-4` va regularization vua phai. Tuy nhien cau hinh khong the sua semantic label; label policy van la blocker lon nhat.
