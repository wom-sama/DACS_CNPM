# TRKH 5-Class V75-V79 SSL/Amplification Audit 2026-06-26

## Muc tieu

- Tiep tuc tim huong vuot diem nghen class 1 tren `class_f_groupclean_v1`.
- Uu tien no-pretrain, train-only SSL, khuech dai dac trung cuc bo nhung phai kiem soat nhiem nen/ambiguity.
- Khong dung test split de tune. Khong full train neu validation class-1 F1 chua dat gate `0.70`.

## Nguon nghien cuu da ap dung

- MAE: masked patch reconstruction de hoc representation noi bo tu train-only data.
  Source: https://arxiv.org/abs/2111.06377
- DINO: self-distillation teacher/student de hoc object/part evidence khong can label ngoai.
  Source: https://arxiv.org/abs/2104.14294
- Fine-grained background suppression/refinement: HERBS goi y multi-scale refinement + background suppression.
  Source: https://arxiv.org/abs/2303.06442
- Discriminative region discovery: FBSD goi y boost/suppress salient regions de tim complementary evidence, nhung can kiem soat false-positive.
  Source: https://arxiv.org/abs/2103.02782

## Thay doi code

- Mo rong `trkh.tools.pretrain_internal_barlow` thanh internal SSL `barlow|vicreg|mae|dino`.
- Them MAE train-only:
  - foreground/detail weighted patch masking;
  - decoder nho tren patch tokens;
  - token pruning tat rieng trong pretrain de giu mapping patch 1-1.
- Them DINO train-only:
  - student model + `DinoProjector`;
  - teacher model/projector EMA;
  - center update va entropy logging;
  - checkpoint kind `internal_dino_pretrain`, fine-tune chi dung student `model_state`.
- Them tests:
  - `tests/test_internal_ssl_mae.py`;
  - `tests/test_internal_ssl_dino.py`;
  - `tests/test_internal_ssl_vicreg.py`.
- Sua regression test sample-weight de nhan API moi tra them `per_sample_classification_loss`.

Focused tests:

```text
pytest tests\test_detection_calibration.py::DetectionCalibrationTests::test_sample_weight_dataset_and_weighted_loss tests\test_generalized_cross_entropy_loss.py tests\test_internal_ssl_mae.py tests\test_internal_ssl_vicreg.py -q
11 passed

pytest tests\test_internal_ssl_dino.py tests\test_internal_ssl_mae.py tests\test_internal_ssl_vicreg.py -q
9 passed
```

## Ket qua probe

| Run | Huong | Val macro F1 | Class-1 P/R/F1 | Ket luan |
| --- | --- | ---: | ---: | --- |
| V64 | Best clean single: high-frequency texture + V64 config | `0.7088` | `0.3532/0.5033/0.4151` | Baseline hien tai |
| V75 | MAE pretrain 80b x 3e + V64 fine-tune | `0.7134` | `0.2885/0.4771/0.3596` | Macro tang nhe, class 1 giam manh |
| V76b | Local zoom + micro-detail + high-frequency | `0.7037` | `0.3222/0.5033/0.3929` | Tang false-positive class 1 |
| V77 | LDAM + GCE robust loss | `0.4812` | `0.1623/0.9608/0.2776` | Over-recall class 1, reject |
| V78 | Cumulative ordinal head | `0.7046` | `0.2950/0.5033/0.3720` | Precision class 1 giam |
| V79b | DINO soft-temp pretrain 80b x 3e + V64 fine-tune | `0.7256` | `0.3431/0.4575/0.3922` | Macro tot nhat nhom nay, class 1 van giam |

V79b training note:

- Tool timeout cat run sau 4 epoch, truoc khi ghi `summary.json`/trace.
- `best_metrics.json` va `checkpoints/best.pt` da co; best epoch ghi trong history la epoch 4.
- Val export rieng: `runs\probe_groupclean_v79b_dino_highfreq_80b_6e_20260626\val_detailed_eval`.
- Eval summary: accuracy `0.7830`, macro F1 `0.7256`, calibrated macro F1 `0.7507`, class-1 F1 `0.3922`.

## Ensemble/router diagnostic V64 + V79b

Average probabilities tren validation:

| V79 weight | Macro F1 | Class-1 F1 |
| ---: | ---: | ---: |
| `0.0` | `0.7088` | `0.4151` |
| `0.5` | `0.7183` | `0.4034` |
| `0.7` | `0.7236` | `0.4160` |
| `1.0` | `0.7256` | `0.3922` |

Router diagnostic:

- Dung V64 neu top-2 cua bat ky model co class 1: macro `0.7219`, class-1 F1 `0.4151`.
- Dung V64 neu bat ky model predict class 1: macro `0.7234`, class-1 F1 `0.4151`.

Ket luan: V79b bo sung tin hieu cho class 0/2/3/4, nhung khong tao them evidence dung cho class 1. Ensemble chi nang macro, khong dat gate class 1.

## Train-only review artifact moi

- Export V79b train predictions:
  `runs\probe_groupclean_v79b_dino_highfreq_80b_6e_20260626\train_detailed_eval`
  - train accuracy `0.8918`;
  - train macro F1 `0.8271`;
  - calibrated train macro F1 `0.8435`.
- Tao boundary review train-only:
  `runs\boundary_review_groupclean_v79b_train_20260626`
  - selected rows `300`;
  - reasons: `focus_false_negative=100`, `focus_false_positive=100`,
    `low_margin_error=100`;
  - boundary pairs: `0-1=120`, `1-2=89`, `4-rest=76`, `2-3=15`;
  - notable buckets: `1->0=95`, `4->1=50`, `2->1=29`, `0->1=20`,
    `over_bright_or_glare=62`, `center_border_lighting_gap=30`;
  - HTML: `boundary_review_report.html`.

Visual spot-check:

- `1->0` sample has many surface speckles/marks but remains very green and
  backlit; this is a label-boundary plus illumination case.
- `0->1` sample is clean/green overall but has small speckles and leaf/shadow
  occlusion near the stem; class-1 evidence is weak and easy to over-amplify.
- `2->1` sample is yellow-green with slight spot/bruise; maturity boundary is
  visually plausible.
- `4->1` sample has yellow/green mixed color, wrinkled/damaged surface and a
  small dark wound; quality class and risk class share the same local evidence.

## Chan doan

- Khuech dai dac trung cuc bo toan cuc dang khuech dai ca dau hieu gay nham class 1:
  - cham/vet nhe tren class 0;
  - texture/surface cua class 2;
  - anh class 4 chat luong thap.
- SSL noi bo Barlow/VICReg/MAE/DINO khong du de tach class 1 khi label boundary mo.
- XAI/forensic V64 truoc do cho thay model chu yeu nhin tren qua, background blur/gray gan nhu khong doi prediction; loi con lai khong phai background filtering don thuan.
- V79b cho thay representation chung co the tot hon, nhung class 1 can label-policy/part-level supervision ro hon.

## Quyet dinh

- Khong full train V75-V79b: khong run nao dat class-1 validation F1 `>=0.70`.
- Khong tiep tuc cac bien the tang amplification/weighting tu dong neu khong co supervision moi; cac probe da lap lai pattern recall/FP xau.
- Huong tiep theo phai la data/annotation-aware:
  1. Dien train-only boundary review cho `0-1`, `1-2`, `2-3`, `4-rest`.
  2. Tao manifest `clean/ambiguous/ignore/soft-target` tu nhan xet train-only.
  3. Neu can tiep tuc model-side, dung annotation do de train specialist/router hoac loss chi tren nhom da xac nhan, khong suy dien tu val/test.
