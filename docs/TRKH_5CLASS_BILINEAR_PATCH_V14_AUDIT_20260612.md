# TRKH 5-Class Bilinear Patch V14 Audit - 2026-06-12

## Muc tieu

V14 thu mot huong learned fine-grained thay cho heuristic mau V13: gom tuong
tac bac hai giua cac patch token sau token pruning. Gia thuyet la texture, dom,
vet bam va chuyen mau nho co the duoc mo ta tot hon bang co-occurrence giua cac
kenh feature thay vi chi first-order attention pooling.

Khong dung pretrained. Khong dung test split de chon hyperparameter.

## Co so nghien cuu

- Bilinear CNN gom outer product cua local feature va cho thay hieu qua tren
  fine-grained recognition.
- Compact Bilinear Pooling cho thay bilinear descriptor co the nen manh ma van
  backprop end-to-end.
- Low-rank Bilinear Pooling giam compute/parameter va chi can category labels.

Tai lieu:

- https://arxiv.org/abs/1504.07889
- https://arxiv.org/abs/1511.06062
- https://arxiv.org/abs/1611.05109

## Kien truc

`CompactBilinearPatchFusion` nhan:

- patch tokens sau pruning `[B, N, 256]`;
- fine-grained attention `[B, N]`;
- valid-patch mask neu co.

Xu ly voi rank `R=32`:

1. `LayerNorm(256)`;
2. hai projection doc lap `256 -> R`;
3. GELU va L2-normalize tung local vector;
4. weighted outer product `sum_n w_n left_n outer right_n`;
5. signed square-root va L2 normalization;
6. noi left mean, right mean va bilinear descriptor.

Descriptor:

`R + R + R*R = 32 + 32 + 1024 = 1088`

Classifier residual:

`LayerNorm(1088) -> Linear(1088,128) -> GELU -> Dropout -> Linear(128,5)`

Linear cuoi zero-init de checkpoint V8 co prediction ban dau khong doi. Model va
EMA resume chi cho phep missing key duoi prefix
`bilinear_patch_fusion_head.*`.

V14 cung sua `FineGrainedPatchPooling.attention_weights` de invalid/padding
patch khong nhan attention; neu mot sample co mask rong thi fallback an toan.

CLI:

- `--bilinear-patch-fusion`;
- `--bilinear-patch-rank`;
- `--bilinear-patch-dropout`.

Launcher:

`scripts/run_trkh_5class_bilinear_patch_v14.ps1`

## Kiem tra

- compile: pass;
- focused bilinear/padding tests: `3 passed`;
- full regression: `105 passed`;
- preflight dataset/GPU: pass;
- final smoke: exit `0`;
- model va EMA partial resume: pass;
- architecture trace: `5` sample, moi class mot sample.

Smoke:

`runs/smoke_mango_cls_256_5class_bilinear_patch_v14_20260612`

Trace them:

- `09g_bilinear_patch_attention.png`;
- `bilinear_patch_descriptor_shape`;
- `bilinear_patch_attention_shape`.

Class-1 smoke trace:

- descriptor `[1,1088]`;
- attention `[1,167]` sau pruning.

## Probe

Run:

`runs/probe_mango_cls_256_5class_bilinear_patch_v14_120b_6e_20260612`

Cau hinh:

- resume V8 best;
- V12 sample weighting + boundary contrastive;
- rank `32`, dropout `0.08`;
- `120` train batch/epoch, full validation;
- toi da `6` epoch, patience `3`;
- batch `32`, accumulation `2`, workers `4/2`.

Run exit `0`, early-stop epoch `4`, best epoch `1`, tong `520.55s`.

Validation:

| Metric | V12 | V14 |
| --- | ---: | ---: |
| macro F1 | 0.8866 | 0.8859 |
| class-1 precision | 0.6203 | 0.6062 |
| class-1 recall | 0.7682 | 0.7748 |
| class-1 F1 | 0.6864 | 0.6802 |

Test audit:

| Metric | V12 | V14 |
| --- | ---: | ---: |
| accuracy | 0.9321 | 0.9305 |
| macro F1 | 0.8943 | 0.8912 |
| class-1 precision | 0.6429 | 0.6207 |
| class-1 recall | 0.7397 | 0.7397 |
| class-1 F1 | 0.6879 | 0.6750 |

Class-1 TP/FP/FN: `54/33/19`.

Confusion chinh:

- `0 -> 1 = 20`;
- `1 -> 2 = 10`;
- `2 -> 1 = 8`;
- `1 -> 0 = 6`;
- `1 -> 4 = 3`;
- `4 -> 1 = 3`.

Detailed output:

- `eval_test_detailed/predictions_detailed.csv`;
- `class1_confusion_audit_test`.

## Trace review

Heatmap `09g` sau probe van co mot so diem nong manh o bien trai/phai cua qua.
Do V14 dung fine-grained attention hien co lam pooling weight, branch bilinear
hoc channel interaction nhung khong tu sua duoc object localization.

Token pruning da loai nhieu padding, nhung cac patch bi giu van gom bien qua,
bong do va mot phan background. Vi vay bac hai feature co the khuech dai texture
o bien/nen cung nhu texture tren vo qua.

## Quyet dinh

Khong full train V14:

- validation class-1 F1 `0.6802 < 0.70`;
- thap hon V12 `0.6864`;
- precision class 1 giam;
- attention localization khong cai thien.

Giu nhanh bilinear o dang optional de phuc vu ablation sau nay. Khong tang rank,
dropout hoac full-train khi localization input chua duoc sua.

## Huong tiep theo

1. Audit exact/near duplicate va source sequence giua train/val/test cua
   `class_f`; doi chieu fairness cua ket qua ViT `1.0`.
2. Chay validation-only boundary calibration tren checkpoint V12 de do phan
   loi do decision boundary, khong dung test de tune.
3. Tao train-only label-boundary review set cho `0/1/2`.
4. Neu split sach va label dung, learned object-tight crop/mask phai duoc hoc
   truoc khi thu lai bilinear/local comparator.
