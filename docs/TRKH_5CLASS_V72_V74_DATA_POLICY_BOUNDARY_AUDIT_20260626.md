# TRKH 5-Class V72-V74 Data-Policy/Boundary Audit 2026-06-26

## Muc tieu

- Tiep tuc tu V64 best clean single probe tren group-clean split.
- Kiem tra cac huong co the khu khuech/kiem soat dac trung boundary class 1 ma
  khong tang oversampling class 1 va khong dung val/test de tune.
- Chi chay probe ngan; full train van bi gate boi class-1 validation F1 `>=0.70`.

## Evidence truoc khi probe

- XAI V64 cho thay heatmap chu yeu nam tren qua/surface; background blur/gray gan
  nhu khong doi prediction.
- V69 surface amplification lam class-1 false positive tang, nen viec khuech dai
  surface toan cuc la nguy hiem.
- V71 VICReg+SupCon train-only giam SSL loss nhung lam class 1 rong hon, F1 giam.

## Research note

Huong tiep theo duoc doi chieu voi cac phuong phap:

- VICReg: https://arxiv.org/abs/2105.04906
- Dataset Cartography: https://arxiv.org/abs/2009.10795
- MAE: https://arxiv.org/abs/2111.06377
- DINO: https://arxiv.org/abs/2104.14294

Ket qua hien tai ung ho Dataset Cartography/data-policy o muc audit, nhung manifest
tu dong chua du chinh xac. Neu tiep tuc model-side, can patch/object-level SSL nhu
MAE/DINO noi bo thay vi global feature agreement.

## V72: ambiguous soft target tu V64 train predictions

Artifact:

- Manifest: `runs\ambiguous_soft_targets_v72_v64train_20260626\ambiguous_soft_targets_train_only.csv`
- Probe: `runs\probe_groupclean_v72_ambiguous_v64train_80b_6e_20260626`

Manifest:

- Train-only rows: `700`.
- Matched train samples: `700`.
- By soft pair:
  `0->1=161`, `1->0=151`, `2->1=100`, `4->1=132`,
  `3->2=48`, `2->3=66`, `1->2=21`, `1->4=21`.

Metric best epoch 2:

- Accuracy `0.7609`.
- Macro F1 `0.7088`.
- Class 1 precision/recall/F1 `0.3551/0.4967/0.4142`.
- V64 class 1 was `0.4151`, so V72 is not an improvement.

Decision:

- Reject. Soft target generated only from model errors mainly copies the same
  uncertainty back into training.

## V73: boundary-center + high-frequency bundle

Artifact:

- Smoke: `runs\smoke_groupclean_v73_boundary_center_highfreq_2b_20260626`
- Probe: `runs\probe_groupclean_v73_boundary_center_highfreq_80b_6e_20260626`

Smoke:

- `train_boundary_center_loss=0.6305`
- `train_boundary_center_terms=157.5`
- `train_boundary_contrastive_loss=2.4356`
- `train_high_frequency_texture_aux_loss=5.1875`

Metric best epoch 3:

- Accuracy `0.7464`.
- Macro F1 `0.6974`.
- Class 1 precision/recall/F1 `0.3060/0.5359/0.3895`.
- Confusion: class-1 recall tang nhe nhung precision sap do false-positive,
  dac biet `0->1`, `2->1`, `4->1`.

Decision:

- Reject. Boundary-center lam cum chua tach duoc class 1; no keo them mau gan
  boundary vao class 1 va lam precision giam.

## V74: train-only ambiguous boundary downweight

Artifact:

- Manifest: `runs\sample_weights_v74_v64train_downweight_20260626`
- Smoke: `runs\smoke_groupclean_v74_downweight_2b_20260626`
- Probe: `runs\probe_groupclean_v74_downweight_80b_6e_20260626`

Manifest:

- Source prediction: `runs\selector_inputs_groupclean_v67diag_20260626\v64_train\predictions_detailed.csv`
- Prediction rows: `9161`.
- `skipped_non_train=0`.
- Candidate issues: `1783`.
- Selected train-only issues: `412`.
- Reasons: `low_self_confidence_error=152`, `low_margin_boundary=260`.
- Boundary pairs: `0-1=131`, `1-2=73`, `1-4=48`, `2-3=50`, `4-rest=110`.
- Weight range: min `0.5`, max `0.8`, mean `0.6893`.
- Smoke matched samples: `412`, observed mean weight `0.9860`.

Metric best epoch 2:

- Accuracy `0.7548`.
- Macro F1 `0.7025`.
- Class 1 precision/recall/F1 `0.3348/0.5033/0.4021`.

Decision:

- Reject. Downweighting ambiguous rows improves nothing vs V64. The selection is
  probably too noisy without manual label/quality review, and affects only a small
  fraction of the balanced exposure.

## Comparison

| Run | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| V64 high-frequency baseline | `0.7088` | `0.3532` | `0.5033` | `0.4151` | best clean single |
| V72 ambiguous soft target | `0.7088` | `0.3551` | `0.4967` | `0.4142` | reject |
| V73 boundary-center bundle | `0.6974` | `0.3060` | `0.5359` | `0.3895` | reject |
| V74 downweight ambiguous | `0.7025` | `0.3348` | `0.5033` | `0.4021` | reject |

## Ket luan

- Chua co full-train command du dieu kien.
- V64 van la best clean single probe tren group-clean split.
- Cac bien the tu dong quanh label policy khong du de vuot nut that.
- De tao dot pha, hai huong co kha nang nhat:
  1. Manual train-only boundary review roi tao `clean/ambiguous/ignore/soft-target`
     manifest co chat luong that.
  2. Patch/object-level self-supervised pretrain noi bo, uu tien MAE/DINO-style
     local/part objective, vi global VICReg/Barlow da khong du.
