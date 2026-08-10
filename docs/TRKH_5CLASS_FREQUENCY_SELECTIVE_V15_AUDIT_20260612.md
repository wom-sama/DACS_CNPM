# TRKH 5-Class Frequency-Selective V15 Audit - 2026-06-12

## Muc tieu

V15 thu frequency-selective aggregation theo huong LaSt-ViT de giam lazy
aggregation va background shortcut. Nhanh nay khong dung pretrained va chi la
residual tren V12, khong thay token-pruning path.

Tai lieu:

- https://arxiv.org/abs/2602.22394
- https://github.com/ChengShiest/LAST-ViT

## Kien truc

Input:

- patch token sau pruning `[B,N,256]`;
- valid-patch mask `[B,N]`;
- pseudo foreground prior `[B,N]`.

Xu ly:

1. FFT moi patch token tren chieu embedding.
2. Loc tan so bang Gaussian kernel voi `sigma=sqrt(256)`.
3. IFFT ve feature da loc.
4. Tinh stability theo tung feature dimension:

   `stability = x / abs(filtered - x)`

5. Chi cho patch co foreground prior vuot threshold tham gia neu con du ung
   vien; fallback ve valid patch khi mask qua chat.
6. Moi feature dimension chon top-k patch, sau do gather va average.
7. Tron feature moi voi pooled feature cu:

   `pooled = (1-blend)*pooled_old + blend*frequency_selected`

Probe dung `top_k=1`, `foreground_threshold=0.35`, `blend=0.25`. Blend nho giu
checkpoint V8/V12 on dinh; smoke voi `blend=1.0` co train loss `8.03`, trong
khi masked residual smoke con `1.262`.

CLI:

- `--frequency-selective-pooling`;
- `--frequency-selective-top-k`;
- `--frequency-selective-blend`;
- `--frequency-selective-foreground-threshold`.

Launcher:

`scripts/run_trkh_5class_frequency_selective_v15.ps1`

## Kiem tra

- full regression: `116 passed`;
- preflight RTX 4060 8 GB: pass;
- batch `32`, accumulation `2`, workers `4/2`;
- smoke masked: exit `0`;
- architecture trace: `5` sample, mot sample moi class;
- probe: exit `0`, `6` epoch, `120` train batch/epoch, full validation;
- tong probe: `772.5s`.

Trace them:

- `09h_frequency_selective_votes.png`;
- vote tensor shape/sum;
- foreground-prior weighted mean;
- foreground vote mass.

## Ket qua

Validation checkpoint duoc chon o epoch 3:

| Metric | V12 | V15 |
| --- | ---: | ---: |
| macro F1 | 0.8866 | 0.8872 |
| class-1 precision | 0.6203 | 0.6257 |
| class-1 recall | 0.7682 | 0.7417 |
| class-1 F1 | 0.6864 | 0.6788 |

Test audit sau khi chon checkpoint bang validation:

| Metric | V12 | V15 |
| --- | ---: | ---: |
| accuracy | 0.9321 | 0.9353 |
| macro F1 | 0.8943 | 0.8975 |
| class-1 precision | 0.6429 | 0.6506 |
| class-1 recall | 0.7397 | 0.7397 |
| class-1 F1 | 0.6879 | 0.6923 |

Test tang nhe nhung khong duoc dung de dao nguoc quyet dinh validation. V15
khong qua gate class-1 validation F1 `>=0.70`.

Artifacts:

- `runs/probe_mango_cls_256_5class_frequency_selective_v15_120b_6e_20260612`;
- `eval_test_detailed/predictions_detailed.csv`;
- `class1_confusion_audit_test`;
- `architecture_trace`.

## Audit localization

Foreground vote mass tai 5 trace sau probe:

| Class | Foreground vote mass |
| --- | ---: |
| 0 | 0.5273 |
| 1 | 0.8906 |
| 2 | 0.5273 |
| 3 | 0.5859 |
| 4 | 0.4062 |

Gating loai padding, nhung pseudo-mask van coi co/la xanh va nen co mau gan qua
la foreground. Class 4 la truong hop xau nhat. Frequency stability tu no khong
phan biet duoc texture vo qua voi texture nen cung mau.

Class-1 test audit:

- TP/FP/FN: `54/29/19`;
- `0->1=19`;
- `1->2=10`;
- `1->0=6`;
- `2->1=6`;
- `1->4=3`;
- `4->1=3`.

Review anh cho thay nhieu `1->2` da vang/chin ro, trong khi nhieu `0->1` co
nhan vo, vet benh, bong do hoac anh sang lam bien nhan mo ho. Day khong phai
mot van de co the giai quyet bang tang class 1 toan cuc.

## Quyet dinh

Khong full-train V15:

- validation class-1 F1 `0.6788 < 0.70`;
- thap hon V12 `0.6864`;
- foreground localization chua sach tren nen co mau gan qua.

Giu nhanh frequency-selective o dang optional de ablation sau khi co
object-tight mask/crop tot hon. Uu tien tiep theo:

1. group-split theo chuoi/qua goc va label-boundary audit;
2. learned object-tight localization hoac mask co supervision;
3. pairwise specialist/router chi kich hoat khi top-2 thuoc boundary da khai
   bao, thay vi logit adjustment toan cuc.
