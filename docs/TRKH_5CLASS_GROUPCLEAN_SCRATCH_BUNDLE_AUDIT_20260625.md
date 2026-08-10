# TRKH 5-Class Group-Clean Scratch Bundle Audit 2026-06-25

## Muc tieu

- Kiem tra tiep cac huong co kha nang dot pha tren split group-clean `D:\DataAI\AIEx\newdataset\class_f_groupclean_v1`.
- Khong dung test de tune.
- Khong full train neu class-1 validation F1 chua dat gate `>= 0.70`.
- Uu tien no-pretrain, train tu scratch, giu balanced input exposure.

## Leakage guard

- Checkpoint cu `runs/mango_cls_256_5class_attention_views_bounded_v8_30e/checkpoints/best.pt` bi overlap source-sequence voi group-clean val/test, nen chi dung lam diagnostic.
- Cac run trong audit nay deu set `-ResumeCheckpoint " "` va `-SampleWeightManifest " " -HardSampleManifest " "`.
- Targeted-margin manifest dung train-only:
  `runs/targeted_margin_groupclean_v26_scratch_train_only_20260625/targeted_margin_train_only.csv`.

## Code thay doi

- Them `--class-loss-multipliers` vao train config/CLI de giam loss contribution theo class ma khong doi sampler.
- Them `trkh.tools.fit_class1_reroute_calibration` de fit rule reroute class 1 tren train/val va apply split khac.
- Mo rong launcher V8/V26:
  - `ImageSize` cho V26.
  - `HardSampleManifest/HardSampleRepeatFactor` co the tat.
  - expose loss knobs: `ClassificationLoss`, focal/LDAM, `ClassLossMultipliers`.
  - expose bundle knobs: `MixStyle`, `ForegroundSurfaceFusion`, `Sam`, `LocalExposure`, `Obstacle`, `RandAugment`.
- Luu y PowerShell: khi can pass bool typed params nhu `-MixStyle:$true`, nen dung direct call hoac nested `powershell -Command`; `powershell -File -MixStyle $true` co the bien bool thanh string.

## Ket qua probe

| Run | Mo ta | Val macro F1 | Val class 1 F1 | Class 1 P/R | Ket luan |
|---|---:|---:|---:|---:|---|
| `probe_groupclean_targeted_margin_v26_scratch_120b_6e_20260625` | scratch V26 baseline | 0.7057 | 0.3360 | 0.242 / 0.549 | over-predict class 1 |
| `probe_groupclean_targeted_margin_v26_scratch_balsoft_120b_6e_20260625` | balanced softmax | 0.6121 | 0.2766 | 0.167 / 0.797 | reject, FP class 1 tang manh |
| `probe_groupclean_targeted_margin_v26_scratch_c1loss065_120b_6e_20260625` | class1 loss multiplier 0.65 | 0.7278 | 0.3780 | 0.316 / 0.471 | best clean scratch hien tai, chua qua gate |
| `probe_groupclean_v26_style_surface_cf_aug_120b_6e_20260625` | MixStyle + foreground surface + background CF + stronger aug | 0.7081 | 0.3379 | 0.290 / 0.405 | reject, over-augmentation/underfit |
| `probe_groupclean_v26_scratch_c1loss065_180b_12e_20260625` | longer schedule, 180 batches | 0.7131 | 0.3557 | 0.294 / 0.451 | reject, them epoch khong giup |
| `probe_groupclean_v26_c1loss065_img320_80b_6e_20260625` | image size 320 | 0.7136 | 0.3478 | 0.390 / 0.314 | reject, precision tang nhung recall mat |

## Calibration va specialist diagnostic

- Reroute train->val tren c1loss065:
  - output `runs/class1_reroute_probe_groupclean_v26_scratch_c1loss065_train_to_val_20260625`
  - class 1 `0.3780 -> 0.3930`, macro `0.7279 -> 0.7314`
  - lift nho, khong du lam huong chinh.
- Train-fit logit bias train->val:
  - output `runs/trainfit_logit_bias_probe_groupclean_v26_scratch_c1loss065_to_val_20260625`
  - class 1 `0.3780 -> 0.3458`, macro giam
  - reject vi overfit train.
- Deterministic prior correction theo train counts:
  - alpha 0.25 da lam class1 recall val xuong `0.085`, F1 `0.1512`
  - reject, logit class1 qua sat ranh nhung khong du separation.
- Focus-class specialist train->val voi image features:
  - output `runs/focus_class1_specialist_train_to_val_groupclean_c1loss065_20260625`
  - OOF train class1 `0.6847`, nhung val class1 `0.3065`
  - reject, feature/router overfit group train va khong generalize.

## Forensic audit

Run:
`runs/forensics_val_groupclean_c1loss065_20260625`

Chinh:

- Class 1: TP `72`, FP `156`, FN `81`, F1 `0.3780`.
- ECE rat cao `0.4569`, confidence thap va miscalibrated.
- Margin loi rat thap: median error margin `0.041`, median correct margin `0.099`.
- Top loi:
  - `3->2`: 90
  - `2->3`: 82
  - `3->4`: 66
  - `1->0`: 63
  - `0->1`: 61
  - `4->1`: 48
  - `2->1`: 40
- `background_heavy` chi 48 mau, 12 loi; nen khong phai nguyen nhan chinh.
- `highlight_heavy` co 506 mau, 128 loi; anh sang co tac dong nhung khong giai thich du class-1 collapse.
- Contact sheet: `runs/forensics_val_groupclean_c1loss065_20260625/review_images/contact_sheet_top_errors.jpg`.

Nhan dinh thu cong tu contact sheet:

- Nhieu `0->1` la qua xanh voi khac biet cuc nho ve dom/vet; label boundary rat mong.
- Mot so `background_heavy` la crop mong/qua bi cat, nen pseudo foreground khong du tin cay.
- Nhieu low-margin case la partial, bi do, hoac chuyen tiep maturity/damage giua 2 lop gan nhau.

## Research tiep theo

Cac huong phu hop hon voi evidence hien tai:

- Dataset Cartography: dung confidence/variability theo epoch de tach easy/ambiguous/hard-label-error; phu hop vi loi chu yeu low-margin va nghi label-boundary.
- Group DRO / worst-group optimization: tao train-only quality groups tu lighting/background/partial va toi uu worst group, thay vi tang augmentation dong loat.
- AugMix-style consistency: co ich cho robustness shift, nhung probe augmentation manh da lam yeu model; neu thu lai thi can consistency co kiem soat va gate nho.
- SupCon/self-supervised pretrain noi bo tren train images: no external pretrain, co the tang representation truoc khi fine-tune class 1.

Nguon da doi chieu:

- Group DRO: https://arxiv.org/abs/1911.08731
- Dataset Cartography: https://arxiv.org/abs/2009.10795
- AugMix: https://arxiv.org/abs/1912.02781
- Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362

## Ket luan

- Chua dat gate full train. Best clean no-pretrain group-clean hien tai: `macro 0.7278`, class1 `0.3780`.
- Cac huong hau xu ly/calibration/specialist khong du; representation va data-boundary moi la bottleneck.
- Khong nen full train cac bundle tren.
- Buoc tiep theo nen la:
  1. Them training-dynamics/data-cartography logger tren train/val boundary samples.
  2. Tao train-only quality groups va thu group-aware loss/sampler nhe.
  3. Neu van thap, chuyen sang self-supervised pretrain noi bo tren `class_f_groupclean_v1/train`, sau do fine-tune TRKH.
