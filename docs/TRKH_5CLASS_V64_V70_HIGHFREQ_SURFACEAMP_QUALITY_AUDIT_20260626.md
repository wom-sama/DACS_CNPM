# TRKH 5-Class V64-V70 High-Frequency / Surface Amplification / Quality Audit 2026-06-26

## Muc tieu

- Split dung cho gate: `D:\DataAI\AIEx\newdataset\class_f_groupclean_v1`.
- Khong dung test de tune. Tat ca so lieu ben duoi la validation group-clean.
- Gate full train hien tai: class-1 validation F1 `>= 0.70`.
- Uu tien no-external-pretrain. Internal SSL checkpoint duoc phep vi chi dung train split:
  `runs\pretrain_groupclean_v34_internal_barlow_80b_3e_20260625\checkpoints\best.pt`.

## Thay doi da them

### High-Frequency Texture Expert

- Them `HighFrequencyTextureExpert` trong `trkh.models.model`.
- Input: anh normalized `[B,3,H,W]`.
- Xu ly: unnormalize, resize analysis size, tinh high-pass RGB, gradient, Laplacian,
  local contrast/foreground-detail, roi tao descriptor texture.
- Output:
  - `high_frequency_texture_logits [B,5]`;
  - residual logit adjustment co route theo low-margin boundary;
  - trace maps `09l` den `09p`.
- Loss train-only tuy chon:
  - `high_frequency_texture_aux_loss`;
  - `high_frequency_texture_pairwise_loss`.

### Surface-Detail Amplified Supervised View

- Them train-only loss trong `trkh.training.train`.
- Input: batch image normalized `[B,3,H,W]`, hard targets va model hien tai.
- Xu ly:
  1. Chon mot phan batch theo `surface_amplified_probability`.
  2. Tao view khuech dai surface detail bang `_surface_counterfactual_images`.
  3. Forward lai cung model tren view moi.
  4. Tinh supervised CE/LDAM tren view moi va optional boundary margin loss
     cho cac pair `0-1,1-2,2-3,1-4,4-rest`.
- Output history:
  - `train_surface_amplified_supervised_loss`;
  - `train_surface_amplified_boundary_margin_loss`;
  - `train_surface_amplified_fraction`;
  - `train_surface_amplified_boundary_terms`.
- Mac dinh tat, chi bat bang CLI. Day khong phai inference branch.

### Quality Sample Weight Guard

- Them tool `trkh.tools.build_quality_sample_weights`.
- Input: quality/cartography manifest train-only.
- Guard: fail neu path nam trong `/val/` hoac `/test/`.
- Output: CSV `image_path,target_index,target_name,sample_weight,quality_bucket,cartography_bucket,quality_group_name`
  va `summary.json`.
- Train loop dung `SampleWeightDataset` de nhan per-sample classification loss.
- Mac dinh tat, khong thay doi inference path.

## Kiem chung

- `python -m py_compile trkh\training\train.py trkh\core\config.py`: pass.
- `pytest tests\test_surface_amplified_supervised_loss.py tests\test_surface_detail_amplification.py tests\test_high_frequency_texture_expert.py -q`: `10 passed`.
- `python -m py_compile trkh\tools\build_quality_sample_weights.py`: pass.
- V69 smoke sau fix target/history:
  - `train_surface_amplified_supervised_loss=6.78125`;
  - `train_surface_amplified_boundary_margin_loss=0.25390625`;
  - `train_surface_amplified_fraction=0.3125`;
  - `train_surface_amplified_boundary_terms=10.0`.
- V69/V70 architecture trace completed:
  - `runs\probe_groupclean_v69_surface_amp_supervised_80b_6e_20260626\architecture_trace`;
  - `runs\probe_groupclean_v70_quality_weight_guard_80b_6e_20260626\architecture_trace`.

## Ket qua probe

| Run | Huong | Best val macro F1 | Class-1 F1 tai best macro | Best class-1 F1 | P1/R1 tai best class-1 | Ket luan |
| --- | --- | ---: | ---: | ---: | --- | --- |
| V64 | High-frequency texture expert | 0.7088 | 0.4151 | 0.4151 | 0.3532 / 0.5033 | Best clean single probe hien tai |
| V65 | V64 + high-frequency pairwise aux | 0.7090 | 0.4140 | 0.4140 | 0.3516 / 0.5033 | Reject |
| V66 | V64 + margin sharpening | 0.7052 | 0.4053 | 0.4053 | 0.3423 / 0.4967 | Reject |
| V67 | Slow fine-tune + class1 precision bias | 0.7146 | 0.3647 | 0.3869 | 0.3882 / 0.3856 | Reject |
| V68 | V64 + quality GroupDRO | 0.7067 | 0.3938 | 0.3938 | 0.3262 / 0.4967 | Reject |
| V69 | Surface amplified supervised view | 0.7063 | 0.3700 | 0.3821 | 0.3080 / 0.5033 | Reject |
| V70 | Quality/cartography sample weights | 0.6962 | 0.3912 | 0.3912 | 0.3125 / 0.5229 | Reject |

## V69 detailed audit

- Detailed eval tai:
  `runs\probe_groupclean_v69_surface_amp_supervised_80b_6e_20260626\val_detailed_eval`.
- Class-1 confusion audit tai:
  `runs\probe_groupclean_v69_surface_amp_supervised_80b_6e_20260626\class1_confusion_audit_val`.
- Class-1 metrics: TP `74`, FP `173`, FN `79`, precision `0.2996`,
  recall `0.4837`, F1 `0.3700`.
- FP class1 theo target:
  - `0->1=89`;
  - `2->1=49`;
  - `4->1=31`;
  - `3->1=4`.
- FN class1 theo prediction:
  - `1->0=65`;
  - `1->2=12`;
  - `1->4=2`.

Ket luan V69: loss khuech dai surface co hoat dong, nhung no lam tang false
positive class 1. Dau hieu can tach class 1 khong phai chi tiet cuc bo don le;
global amplification dang khuech dai ca vet/anh sang/texture cua class 0, 2 va 4.

## V70 sample weight audit

- Quality manifest train-only:
  `runs\quality_group_v64_train_20260626\quality_groups_train_only.csv`.
- Output sample weights:
  `runs\sample_weights_quality_guard_v70_train_20260626\sample_weights_quality_guard_train_only.csv`.
- Summary:
  - rows `9161`, all train;
  - raw mean `0.9098`, normalized mean `1.0`;
  - min/max `0.6045/1.3739`;
  - cartography: `hard_low_self=531`, `medium=6319`, `ambiguous_boundary=2311`;
  - quality: `dark_dirty_detail=5968`, `normal=2375`, `overbright=469`,
    `partial_or_border=182`, `low_contrast=149`, `underdark=18`.

Ket luan V70: heuristic weighting tang class-1 recall len `0.5229` nhung
precision chi `0.3125` va macro giam. Quality/cartography buckets hien tai
chua du chinh xac de tu dong sua ranh class 1.

## Quyet dinh

- Khong gui/khong chay full train. Best clean class-1 validation F1 van
  `0.4151`, thap hon gate `0.70`.
- Dung lap lai cac huong khuech dai/weighting toan cuc neu khong co bang chung
  moi. Chung dang day class 1 qua rong, khong lam boundary ro hon.
- Background filtering khong phai bottleneck chinh trong nhom probe nay; loi
  lon van la low-margin label boundary/domain shift quanh `0/1`, `1/2`,
  `2/3` va `4->1`.

## Huong tiep theo

1. Tao audit review hop nhat XAI + confusion + quality/cartography cho boundary
   samples, uu tien train-only de phat hien label noise/ambiguous rule.
2. Neu tiep tuc model-side, can representation learning noi bo manh hon Barlow:
   thu VICReg/DINO/MAE train-only, roi fine-tune bang V64 config.
3. Khong full train cho den khi short probe tren group-clean dat class-1
   validation F1 `>=0.70`, hoac co override ro rang.
