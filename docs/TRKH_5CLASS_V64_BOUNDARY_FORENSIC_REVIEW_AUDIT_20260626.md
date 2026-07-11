# TRKH 5-Class V64 Boundary Forensic Review Audit 2026-06-26

## Muc tieu

- Dung best clean single probe hien tai V64 de xem diem nghen co phai background,
  thieu texture, hay label-boundary.
- Khong dung test. Val chi dung de chan doan; train-only manifest moi duoc phep
  dung cho data cleaning/quyet dinh train tiep.

## Artifact da tao

### V64 detailed validation

- `runs\probe_groupclean_v64_highfreq_texture_80b_6e_20260626\val_detailed_eval`
- Non-TTA metrics:
  - accuracy `0.7605`;
  - macro F1 `0.7088`;
  - class-1 precision/recall/F1 `0.3532/0.5033/0.4151`.

### Boundary review manifests

- Train-only:
  `runs\boundary_review_groupclean_v64_train_20260626`
  - selected rows `300`;
  - reasons: `focus_false_negative=100`, `focus_false_positive=100`,
    `low_margin_error=100`;
  - boundary pairs: `0-1=120`, `1-2=99`, `4-rest=51`, `2-3=30`;
  - copied review images `120`;
  - HTML: `boundary_review_report.html`.
- Validation diagnostic:
  `runs\boundary_review_groupclean_v64_val_20260626`
  - selected rows `240`;
  - reasons: `focus_false_negative=76`, `focus_false_positive=80`,
    `low_margin_error=80`, `low_margin_correct_boundary=4`;
  - boundary pairs: `0-1=100`, `1-2=87`, `4-rest=16`, `2-3=37`;
  - copied review images `100`;
  - HTML: `boundary_review_report.html`.

Tool moi:

- `trkh.tools.render_boundary_review_html`
- Test: `tests/test_render_boundary_review_html.py`

### XAI audit

- `runs\xai_audit_groupclean_v64_val_small_20260626`
- Selected cases: `30`.
- Top confusions:
  - `3->2=126`;
  - `2->3=89`;
  - `3->4=70`;
  - `0->1=63`;
  - `1->0=61`.
- Heatmap foreground mass:
  - attention `0.9674`;
  - grad-rollout `0.9089`;
  - gradcam `0.9571`;
  - rollout `0.9587`.
- Background robustness:
  - background blur pred-prob drop `0.0000`;
  - background gray pred-prob drop `-0.0001`;
  - object desaturate pred-prob drop `0.0604`.
- Review flags:
  - `near_tie_top2=10`;
  - `grad_rollout_background_attention=11`;
  - `register_attention_diffuse=30`.

### TTA diagnostic

- `runs\probe_groupclean_v64_highfreq_texture_80b_6e_20260626\val_tta_eval`
- TTA metrics:
  - macro F1 `0.7014`;
  - class-1 precision/recall/F1 `0.3277/0.5033/0.3969`.
- Non-TTA V64 is better: macro/class1 `0.7088/0.4151`.
- Decision: reject simple TTA/consistency as breakthrough direction.

## Visual review notes

Reviewed samples:

- `1->0` sample:
  `runs\boundary_review_groupclean_v64_val_20260626\review_images\val\0-1\focus_false_negative\val_00001_t1_p0_m0.131.jpg`
  - fruit is bright green and visually close to class 0;
  - no strong background shortcut;
  - likely label-boundary/definition issue.
- `0->1` sample:
  `runs\xai_audit_groupclean_v64_val_small_20260626\case_019_t0_p1_confusion_0_to_1\crop.png`
  - fruit has small dark speckles and surface marks;
  - prediction class 1 is plausible if class 1 means slight risk/surface issue;
  - explains why surface amplification V69 increased class-1 FP.
- XAI overlay for same `0->1`:
  `grad_rollout_overlay.png`
  - attention lies mostly on fruit/surface and edges;
  - background is not the dominant error source.

## Diagnosis

- More background filtering is unlikely to close the gap alone:
  XAI foreground mass is high and background blur/gray barely changes predictions.
- Global feature amplification is risky:
  V69 amplified surface marks that also exist in class 0/2/4 and increased
  class-1 FP.
- Class-1 bottleneck is a label-policy/low-margin surface boundary:
  class 1 shares visual evidence with class 0 and class 2, while some class 4
  low-quality cases also look like risky immature fruit.
- Current best clean single probe remains V64 with class-1 F1 `0.4151`;
  no full train gate.

## Next actions

1. Fill manual status columns in the train-only review CSV:
   `manual_label_status`, `manual_expected_class`, `quality_lighting`,
   `quality_dirty_obstacle`, `quality_partial_fruit`, `quality_background_mask`.
2. Build train-only `clean/ambiguous/ignore/soft-target` manifest from reviewed
   rows. Do not use val/test rows for this.
3. If model-side research continues before manual review, avoid global
   invariance/amplification and prioritize patch/object-level SSL such as MAE or
   DINO-style local view consistency trained only on `train`.
