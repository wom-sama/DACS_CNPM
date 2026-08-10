# TRKH 5-Class Expert Diversity And TTA Audit 2026-06-27

## Muc tieu

Tiep tuc tim huong dot pha cho `class_f` 5 class, uu tien cai thien
`Xoai_Song_ChuaNhe_CoNguyCo` ma khong sua nhan, khong bo sung data va khong
tune tren test. Baseline tam thoi van la:

- `runs/ensemble_top5_tta_teacher_cache_5class_v1`
- Val macro/class1 F1: `0.913897 / 0.738255`
- Test macro/class1 F1: `0.924826 / 0.778626`

## Oracle/gap audit

Da tong hop 16 experts hien co: 10 TIMM non-TTA, 5 top-5 TTA va AIDT.

- Val: top-5 TTA co `41` false-negative class 1; co `25/41` mau duoc it
  nhat mot expert khac doan la class 1.
- Test: top-5 TTA co `22` false-negative class 1; chi `7/22` mau duoc it
  nhat mot expert khac doan la class 1.
- Test con `15/22` false-negative class 1 ma khong expert hien co nao bat
  duoc. Vi vay selector/router tren expert hien co khong the dua class1 F1 den
  `0.96`; can expert/representation moi that su.

Artifact:

- `runs/expert_oracle_gap_audit_20260627/summary.json`

## AIDT exporter dung pipeline

Them `trkh.tools.export_aidt_predictions` de export AIDT bang dung pipeline
train-time: SquarePad + Resize shared RGB, normalize rieng cho ResNet/ViT branch.

Ket qua:

| Run | Split | Macro F1 | Class1 F1 | Ket luan |
| --- | --- | ---: | ---: | --- |
| AIDT | val | `0.909308` | `0.716418` | thap hon top-5 TTA |
| Top5 TTA + AIDT | val | `0.918847` | `0.756579` | validation tang |
| AIDT | test | `0.885293` | `0.626667` | thap |
| Top5 TTA + AIDT | test | `0.920922` | `0.766917` | thap hon top-5 TTA |

Ket luan: AIDT co diversity tren validation nhung khong generalize test; reject
artifact chinh.

## Boundary focus specialists

Mo rong `trkh.tools.train_focus_binary_specialist`:

- `--train-include-class-name` de train chi tren subset class bien.
- `--balanced-train-sampler` de can bang positive/negative trong specialist,
  chi tren train split.
- `--eval-only-checkpoint` de export val/test tu checkpoint da chon bang val,
  khong train lai.

Them `trkh.tools.fuse_focus_pairwise_specialists`:

- Sweep rule tren validation.
- Apply lai rule da chon sang test bang `--rules-json`.
- Router chi can thiep khi teacher dang o ranh focus/negative class.

Ket qua:

| Run | Split | Macro F1 | Class1 F1 | Ghi chu |
| --- | --- | ---: | ---: | --- |
| Top5 TTA base | val | `0.913897` | `0.738255` | base |
| Pairwise 0/1 + 1/2 fusion | val | `0.918208` | `0.755700` | promote `9` mau |
| Top5 TTA base | test | `0.924826` | `0.778626` | base |
| Apply val rule pairwise | test | `0.922349` | `0.770370` | giam, reject |

Ket luan: pairwise external specialists overfit validation. Khong full train,
khong promote artifact.

Artifacts:

- `runs/probe_pair01_focus_vs_raw_10e_20260627`
- `runs/probe_pair12_focus_vs_ripe_10e_20260627`
- `runs/pairwise_focus_specialist_fusion_val_coarse_20260627`
- `runs/pairwise_focus_specialist_fusion_test_apply_valrule_20260627`

## Expert backbone probes

Thu them cac pretrained expert da dang hon top-5 CNN hien tai, chi probe ngan,
full validation, skip test.

| Expert | Probe | Best val macro F1 | Best val class1 F1 | Ket luan |
| --- | --- | ---: | ---: | --- |
| CaFormer-S18 | 80 batch x 8e | `0.892227` | `0.649254` | reject |
| MobileNetV4 Conv Medium | 100 batch x 8e | `0.859756` | `0.570423` | reject |
| RepViT-M2 | 100 batch x 8e | `0.892479` | `0.652330` | reject |

CaFormer duoc export val de thu blend nhe:

- Best top5+CaFormer validation weight `0.35`: macro/class1
  `0.916610 / 0.747475`.
- Van thap hon pairwise val va khong du ly do export test.

Ket luan: tiep tuc nem backbone pretrained ngau nhien khong kinh te; cac
backbone nay khong tao expert bat class 1 manh hon teacher hien tai.

## Spatial crop TTA

Mo rong `trkh.tools.export_timm_predictions`:

- `--tta-spatial-crop-fractions`
- Moi fraction tao 5 crop resized: top-left, top-right, center,
  bottom-left, bottom-right.

Probe validation crop fraction `0.86`:

| View | Val macro F1 | Class1 F1 | Ket luan |
| --- | ---: | ---: | --- |
| MobileNetV3 spatial | `0.897990` | `0.690323` | thap hon TTA cu |
| EfficientNet-B3 spatial | `0.895723` | `0.673139` | reject |
| EfficientNetV2-S spatial | `0.906575` | `0.721088` | view tot nhat nhung van thap |
| DenseNet121 spatial | `0.897097` | `0.684932` | reject |
| ConvNeXt-Tiny spatial | `0.894374` | `0.673684` | reject |
| Top5 spatial-only | `0.908700` | lower than base | reject |
| Top5 TTA + spatial weight `0.10..0.50` | max `0.913848` | max `0.736486` | reject |

Ket luan: multi-crop lam mat global context, khong giup class 1 tren split nay.

## Robust augmentation probe

Mo rong `D:\DataAI\AIEx\image_baseline_experiments\scripts\02_train_timm_classifier.py`:

- `--color-jitter`
- `--random-erasing`
- `--randaugment-ops`
- `--randaugment-magnitude`
- safe probe flags da co: `--max-train-batches`, `--patience`,
  `--best-metric`, `--focus-class-name`, `--skip-test`

Smoke MobileNetV3 robust augment pass. Probe `120` train-batch x `10` epoch:

| Run | Best epoch | Val macro F1 | Focus class F1 | Ket luan |
| --- | ---: | ---: | ---: | --- |
| `outputs/probe_mobilenetv3_robust_aug_120b_10e_20260627` | `8` | `0.883228` | `0.651466` | thap hon top-5 TTA, reject |

Ket luan: photometric/erasing/RandAugment train-time lam model robust hon tren
ly thuyet nhung khong tao expert moi bat boundary class 1. Khong export test.

## Percentile-stretch TTA

Mo rong `trkh.tools.export_timm_predictions`:

- `--tta-channel-stretch-percentiles`
- `--tta-luma-stretch-percentiles`

Muc tieu la xu ly anh qua sang/qua toi bang inference-time contrast stretch.
Smoke pass tren 16 mau MobileNetV3. Full validation MobileNetV3 voi TTA goc
cong stretch:

| View | Val macro F1 | Focus class F1 | Ket luan |
| --- | ---: | ---: | --- |
| MobileNetV3 TTA goc | `0.904011` | `0.705882` | base |
| MobileNetV3 TTA + stretch | `0.901845` | `0.704762` | giam, reject |

Ket luan: stretch lam recall focus tang nhe nhung precision/macro giam; khong
nen dua vao full export top-5.

## Top-5 logit-bias calibration

Sua `trkh.tools.calibrate_classification_logits` de doc duoc teacher-cache cu
chi co `true_name` bang cach infer `target_index` tu sidecar metrics JSON.

Validation-only calibration tren top-5 TTA:

| Split | Base macro/class1 | Calibrated macro/class1 | Ket luan |
| --- | ---: | ---: | --- |
| val | `0.913897 / 0.738255` | `0.918116 / 0.750000` | tang validation |
| test | `0.924826 / 0.778626` | `0.922473 / 0.772727` | giam test, reject |

Selected bias: `[-0.08, 0.00, -0.27, 0.35, 0.00]` theo class order cua
teacher-cache. Ket luan: calibration boundary tiep tuc overfit validation,
khong promote artifact.

## Final-fit train+val implementation

Them ho tro vao baseline TIMM script:

- `--train-splits train,val`
- `--final-fit`
- `optimizer_steps` trong history de audit AMP skip.
- Ha `GradScaler(init_scale)` xuong `1024` de batch dau khong bi skip khi AMP
  trong smoke/probe.

Smoke final-fit MobileNetV3 tren `train,val`, `max_train_batches=1`, `--amp`
pass va co `optimizer_steps=1`.

Luu lenh full final-fit va export audit tai:

- `docs/TRKH_5CLASS_FINALFIT_COMMANDS_20260627.md`

Day la huong full-run hop ly tiep theo neu chap nhan benchmark split cu:
hyperparameter/epoch da chon bang validation truoc do, final-fit khong dung
validation de chon checkpoint nua. Chua chay full trong phien nay.

## Ket luan ky thuat

1. CNN thuon/pretrained manh hon TRKH scratch vi pretrained representation va
   split cu co source-sequence leakage. No-pretrain/hybrid nhieu nhanh da dung
   background/texture/attention nhung khong co supervision moi nen khong bat
   duoc boundary class 1.
2. Background khong phai nut that chinh: cac audit truoc cho thay object
   desaturation/surface moi lam prediction doi ro, con background blur/gray
   anh huong nho.
3. De len class1 F1 `0.96` tren current test, can expert moi bat duoc nhom
   `15/22` false-negative ma tat ca expert hien tai deu khong du doan class 1.
4. Khong nen full train tu cac huong tren: khong huong nao vuot base test hoac
   tao validation gate du manh de xung dang full train.

## Unseen FN review

Tao audit rieng cho `15` mau test class 1 ma top-5 TTA doan sai va khong co
expert nao trong 16 experts doan la class 1.

Artifacts:

- `runs/unseen_class1_fn_audit_20260627/unseen_class1_fn.csv`
- `runs/unseen_class1_fn_audit_20260627/review.html`
- `runs/unseen_class1_fn_audit_20260627/contact_sheet.jpg`

Quan sat nhanh tu contact sheet:

- Nhieu mau class 1 trong nhom nay nhin bang mat rat gan class 0 xanh/chua
  hoac class 2 vang/chin; dau hieu `song chua nhe/co nguy co` khong ro.
- Co mau bi do sang manh, bong nang, vet ban/hu va qua bi cat mot phan.
- Day la failure cua semantic boundary/quality condition hon la failure loc nen.
  Tat ca expert pretrained da nhin sai cung huong, nen can train signal moi
  cho boundary nay; routing tren expert cu khong du.

## Huong tiep theo

- Neu khong duoc sua nhan/bo sung data: uu tien representation-level method co
  kha nang hoc boundary moi, khong phai routing tren expert cu:
  - OOF teacher probabilities cho clean data-cartography train-only.
  - Expert train tu hard train-only boundary bucket voi loss calibrated theo
    precision, khong all-vs-rest.
  - Patch-level contrastive/pretext dai hon neu chap nhan chi phi.
- Neu chap moi loi the benchmark split cu: co the train mot expert manh hon voi
  full schedule chi khi probe validation vuot top-5 TTA ro rang. Cac probe
  CaFormer/MobileNetV4/RepViT khong dat dieu kien.
