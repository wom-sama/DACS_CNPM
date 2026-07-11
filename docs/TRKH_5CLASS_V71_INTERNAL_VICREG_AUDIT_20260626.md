# TRKH 5-Class V71 Internal VICReg Audit 2026-06-26

## Muc tieu

- Thu mot huong representation-learning noi bo manh hon Barlow thuan.
- Split dung cho gate: `D:\DataAI\AIEx\newdataset\class_f_groupclean_v1`.
- Pretrain chi dung train split, khong dung val/test va khong dung external
  pretrained.
- Gate full train van la class-1 validation F1 `>=0.70`.

## Thay doi code

- Mo rong `trkh.tools.pretrain_internal_barlow` thanh internal SSL tool co
  `--ssl-method barlow|vicreg`.
- Them `vicreg_loss` voi 3 thanh phan:
  - invariance MSE giua 2 view;
  - variance hinge de chong collapse;
  - covariance off-diagonal decorrelation.
- Them checkpoint kind `internal_vicreg_pretrain`; train resume van doc duoc
  `model_state`.
- Neu resume tu method khac, best SSL loss duoc reset vi thang loss khac nhau.
- Them test `tests/test_internal_ssl_vicreg.py`.

## Kiem chung

- `python -m py_compile trkh\tools\pretrain_internal_barlow.py trkh\training\train.py trkh\core\config.py`: pass.
- `pytest tests\test_internal_ssl_vicreg.py tests\test_high_frequency_texture_expert.py tests\test_surface_amplified_supervised_loss.py -q`: `9 passed`.
- Dry-run:
  `runs\dryrun_groupclean_v71_internal_vicreg_20260626`.
- Smoke 2 batch:
  `runs\smoke_groupclean_v71_internal_vicreg_2b_20260626`.

## Pretrain V71

Run:

`runs\pretrain_groupclean_v71_internal_vicreg_supcon_80b_3e_20260626`

Config chinh:

- method `vicreg`;
- batch `32`;
- max train batches `80`;
- epochs `3`;
- learning rate `1e-4`;
- VICReg coeffs `25/25/1`;
- SupCon train-label-only weight `0.03`;
- balanced train sampler;
- no val/test.

Ket qua SSL:

| Epoch | SSL loss | VICReg loss | Invariance | Variance | Covariance | SupCon |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 34.5593 | 34.4357 | 0.0847 | 1.1314 | 4.0323 | 4.1191 |
| 2 | 32.3241 | 32.1942 | 0.0673 | 1.0228 | 4.9418 | 4.3276 |
| 3 | 31.4353 | 31.3025 | 0.0662 | 0.9713 | 5.3671 | 4.4278 |

Checkpoint:

`runs\pretrain_groupclean_v71_internal_vicreg_supcon_80b_3e_20260626\checkpoints\best.pt`

## Fine-tune V71

Run:

`runs\probe_groupclean_v71_vicreg_highfreq_80b_6e_20260626`

Config khac V64 duy nhat o checkpoint resume:

- V64 resume Barlow V34.
- V71 resume VICReg+SupCon V71.
- Con lai giu V64: high-frequency texture expert, multi-granularity aux,
  attention views, boundary contrastive, class-1 loss multiplier `0.80`.

Ket qua:

| Metric | Value |
| --- | ---: |
| Best epoch | 2 |
| Val accuracy | 0.7532 |
| Val macro F1 | 0.6972 |
| Val weighted F1 | 0.7641 |
| Class-1 precision | 0.2589 |
| Class-1 recall | 0.4771 |
| Class-1 F1 | 0.3356 |
| Early stop | epoch 5 |

Detailed eval bang checkpoint best:

- output: `runs\probe_groupclean_v71_vicreg_highfreq_80b_6e_20260626\val_detailed_eval`;
- macro F1 `0.6963`;
- calibrated macro F1 `0.7170`.

Class-1 audit:

- output:
  `runs\probe_groupclean_v71_vicreg_highfreq_80b_6e_20260626\class1_confusion_audit_val`;
- TP `72`, FP `209`, FN `81`;
- precision `0.2562`, recall `0.4706`, F1 `0.3318`.

FP class 1:

- `0->1=85`;
- `2->1=66`;
- `4->1=47`;
- `3->1=11`.

FN class 1:

- `1->0=65`;
- `1->2=13`;
- `1->4=3`.

## Ket luan

- VICReg+SupCon train-only hoc duoc SSL objective, nhung representation sau
  fine-tune lam class 1 qua rong: FP tang `141` o V64 len `209` o V71.
- Huong nay khong dat gate va thap hon V64 (`0.4151` class-1 F1).
- Khong full train.
- Khong nen lap lai VICReg ngan chi bang cach doi nhe he so; can bang chung moi
  hoac method khac co object/part-level signal ro hon.

## Buoc tiep theo

- Tao audit review hop nhat confusion + quality/cartography + XAI/trace cho
  boundary cases, uu tien xac dinh label ambiguity va dieu kien anh sang/dirty/
  partial.
- Neu quay lai SSL, uu tien DINO/MAE noi bo hoac pretext patch/object-level
  thay vi Barlow/VICReg global invariance.
