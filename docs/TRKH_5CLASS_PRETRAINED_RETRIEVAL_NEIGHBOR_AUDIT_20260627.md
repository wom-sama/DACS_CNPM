# TRKH 5-Class Pretrained, Retrieval, Cleanlab, Neighbor Audit - 2026-06-27

## Muc tieu

- Kiem tra vi sao TRKH/hybrid thua cac baseline CNN/pretrained trong `D:\DataAI\AIEx\image_baseline_experiments`.
- Thu cac huong co kha nang tao buoc nhay lon ma khong chay full train dai khi chua qua gate.
- Tach ro hai protocol:
  - `class_f` split hien tai: co source-sequence leakage, dung de doi chieu benchmark local.
  - `class_f_groupclean_v1`: honest generalization, hien tai no-pretrain van rat thap.

## Nguon tham khao da doi chieu

- Confident Learning / cleanlab: https://arxiv.org/abs/1911.00068
- Dataset Cartography: https://arxiv.org/abs/2009.10795
- Generalized Cross Entropy / noisy-label loss: https://arxiv.org/abs/1805.07836
- DINO self-supervised ViT: https://arxiv.org/abs/2104.14294
- DINOv2 robust visual features: https://arxiv.org/abs/2304.07193

Ket luan tu tham khao va audit noi bo: khi loi chinh la boundary label/illumination/partial fruit, viec them head/loss/augmentation toan cuc de lam manh class 1 de gay false positive. Can co decision layer co guard, label-policy train-only, hoac expert diversity that su moi co co hoi vuot gate.

## Thay doi/code moi trong dot nay

- Them `trkh/tools/evaluate_embedding_retrieval.py`
  - TIMM pretrained/fine-tuned feature extraction.
  - kNN/prototype/logistic decision layer.
  - Val-only hyperparameter selection, optional test final audit.
  - Cache feature `.npz` va smoke caps `--max-samples-per-class`, `--max-samples-per-split`.
- Them `trkh/tools/audit_label_issues_cleanlab.py`
  - Train-only cleanlab/low-confidence review manifest.
  - Guard khong dung val/test lam train signal.
- Sua `trkh/tools/train_prediction_selector.py`
  - Doc duoc `teacher_pred_name`, `selector_pred_name`, `teacher_pred_index`.
- Them `cleanlab==2.7.1` vao `requirements.txt`.
- Sua ngoai repo git `D:\DataAI\AIEx\image_baseline_experiments\scripts\02_train_timm_classifier.py`
  - Them `--max-train-batches`, `--max-eval-batches`, `--patience`, `--skip-test`.
  - Them `--best-metric objective|macro_f1|class1_f1`.
  - Them `--focus-class-name` de tranh loi class order cua `ImageFolder` khac `data.yaml`.
  - Mac dinh khong doi hanh vi full train cu.

## Ket qua chinh tren `class_f`

Best current system truoc dot nay:

| System | Val Macro F1 | Val Class-1 F1 | Test Macro F1 | Test Class-1 F1 | Quyet dinh |
|---|---:|---:|---:|---:|---|
| Top-5 pretrained TTA uniform teacher | 0.9139 | 0.7383 | 0.9248 | 0.7786 | Best current |

### Frozen/retrieval probes

| Huong | Run | Val Macro/Class1 | Test Macro/Class1 | Quyet dinh |
|---|---|---:|---:|---|
| DINOv2 small raw kNN | `runs/embedding_retrieval_dinov2_small_classf_20260627` | 0.8918 / 0.6715 | 0.8751 / 0.6047 | Reject |
| MobileNetV3 fine-tuned feature kNN | `runs/embedding_retrieval_mobilenetv3_ft_classf_20260627` | 0.9022 / 0.7042 | 0.8985 / 0.6720 | Reject |
| EfficientNet-B3 fine-tuned feature | `runs/embedding_retrieval_efficientnet_b3_ft_valonly_20260627` | 0.8922 / 0.6688 | skipped | Reject |
| EfficientNetV2-S fine-tuned feature | `runs/embedding_retrieval_efficientnetv2_s_ft_valonly_20260627` | 0.9105 / 0.7273 | 0.9009 / 0.6870 | Reject |
| ConvNeXt-Tiny fine-tuned feature | `runs/embedding_retrieval_convnext_tiny_ft_valonly_20260627` | 0.9011 / 0.6920 | skipped | Reject |
| DenseNet121 fine-tuned feature | `runs/embedding_retrieval_densenet121_ft_valonly_20260627` | 0.8975 / 0.6951 | skipped | Reject |
| CLIP ConvNeXt base raw | `runs/embedding_retrieval_convnext_clip_base_valonly_20260627` | 0.8828 / 0.6308 | skipped | Reject |

Blend top-5 TTA + EfficientNetV2-S feature kNN duoc chon tren validation voi alpha `0.30`, tang val rat nhe `0.9139/0.7383 -> 0.9143/0.7400`, nhung test giam `0.9248/0.7786 -> 0.9169/0.7538`. Reject.

### Selector/meta-stacking

Run `runs/selector_top5tta_plus_effv2knn_valselect_20260627`:

- Val macro F1 `0.9136`, thap hon top-5 TTA.
- Test macro F1 `0.9107`, accuracy `0.9455`.
- Reject.

### Cleanlab train audit

Run `runs/cleanlab_top5_teacher_train_audit_20260627`:

- Input la top-5 teacher train probabilities.
- `cleanlab_issue_count=0`, do train probabilities la in-sample/over-confident.
- Sau khi sort low-confidence, top rows van huu ich de review bang mat, nhung chua du tin cay de auto-relabel/downweight.

Ket luan: muon dung cleanlab dung cach can OOF train probabilities, khong dung in-sample teacher cache.

### TIMM pretrained classifier probes

| Huong | Run | Probe result | Quyet dinh |
|---|---|---:|---|
| ViT base augreg pretrained | `probe_classf_timm_vit_base_augreg_pretrained_ce_80b_5e_20260627` | val macro/class1 khoang `0.8855/0.6354` | Reject gate |
| ConvNeXt base pretrained | `probe_classf_timm_convnext_base_pretrained_ce_40b_4e_20260627` | best val macro `0.7678` | Reject |
| EfficientNetV2-M pretrained, preprocess fixed | `probe_classf_timm_effv2m_preprocessfix_ce_384_80b_4e_20260627` | best val macro `0.7624` | Reject |
| Swin-T pretrained baseline-like | `probe_classf_timm_swin_tiny_pretrained_baselinelike_80b_6e_20260627` | val macro/class1 `0.8343/0.4519` | Reject |

### Baseline-script expert probes sau khi them safe flags

| Huong | Run | Val Macro/Class `Xoai_Song_ChuaNhe_CoNguyCo` F1 | Quyet dinh |
|---|---|---:|---|
| ConvNeXtV2-Tiny FCMAE pretrained | `D:\DataAI\AIEx\image_baseline_experiments\outputs\probe_convnextv2_tiny_120b_6e_20260627` | `0.8570 / 0.5360` | Reject gate |
| ViT-B/16 augreg2 pretrained | `D:\DataAI\AIEx\image_baseline_experiments\outputs\probe_vit_b16_120b_8e_20260627` | `0.7791 / 0.3303` | Reject gate |

Ghi chu: cac run nay dung full validation nhung chi train `120` batches/epoch va `--skip-test`, nen la probe xu huong, khong phai ket qua bao cao chinh thuc.

Chua co pretrained probe nao qua gate class-1 validation F1 `>=0.70` de xung dang full train.

## Neighbor/source refinement prototype

Do `class_f` co source-sequence leakage, da thu decision layer ngoai model:

- Train labels only.
- Neighbor signals:
  - nearest `Image_<id>` sequence.
  - cheap perceptual hash.
  - RGB histogram + exposure/detail stats.
- Teacher: top-5 TTA uniform.
- Selection: validation-only grid search.

Artifacts:

- `runs/neighbor_teacher_refine_proto_20260627/summary_fast.json`
- `runs/neighbor_teacher_refine_proto_20260627/candidate_rows_fast.csv`
- `runs/neighbor_teacher_refine_proto_20260627/predictions_test_fast.csv`

Ket qua:

| System | Val Macro/Class1 | Test Macro/Class1 | Ghi chu |
|---|---:|---:|---|
| Top-5 TTA uniform | 0.9139 / 0.7383 | 0.9248 / 0.7786 | baseline |
| Neighbor-selected `id_k3_t1 + blend_all alpha=0.3` | 0.9209 / 0.7609 | 0.9230 / 0.7634 | val tang, test giam |

Ket luan: neighbor/source signal co tao uplift tren validation nhung khong generalize sang test. Khong duoc dung lam he chinh.

## Chan doan vi sao CNN baseline cao hon TRKH

1. Cac CNN baseline trong bang local khong phai scratch. Chung dung pretrained ImageNet/timm va train du 20-30 epoch.
2. `class_f` split cu co source-sequence leakage, nen pretrained CNN co the hoc visual fingerprint/source sequence rat tot.
3. TRKH scratch/hybrid du co nhieu branch nhung representation ban dau yeu hon pretrained CNN. Them branch khuech dai detail da nhieu lan lam class 1 rong hon, tang false positive.
4. XAI/forensic truoc do cho thay nen khong phai nut that chinh: object desaturation lam drop manh, background blur/gray drop nho.
5. Class 1 la label-boundary hiem, nam giua class 0 va 2; illumination/dirty/partial fruit lam feature khac biet rat nho va de mau thuan.

## Quyet dinh hien tai

- Best current deploy/benchmark artifact van la `runs/ensemble_top5_tta_teacher_cache_5class_v1`.
- Khong chay full train moi tu cac probe 2026-06-27 vi khong qua gate hoac uplift chi co tren validation va giam tren test.
- Khong tiep tuc them loss/head toan cuc cho class 1 khi chua co supervision moi; cac run V64-V79 da cho thay amplification lam tang false positive.

## Huong tiep theo co kha nang dot pha hon

1. Tao OOF train probabilities cho top experts de cleanlab/data cartography dung cach.
2. Tao train-only boundary label-policy manifest:
   - `clean`
   - `ambiguous`
   - `ignore`
   - `soft 0/1` hoac `soft 1/2`
   - `quality_lighting`, `quality_dirty_obstacle`, `quality_partial_fruit`
3. Neu duoc phep “chấp mọi lợi thế” tren split cu, thu expert-diversity thay vi TRKH scratch:
   - train/export them pretrained expert co backbone khac va TTA dong nhat.
   - chon ensemble bang validation, khong tune test.
4. Neu muc tieu la honest generalization, tiep tuc tren `class_f_groupclean_v1`; luc nay can label-policy/manual review hoac data moi, vi best clean single probe moi chi quanh class-1 F1 `0.4151`.
