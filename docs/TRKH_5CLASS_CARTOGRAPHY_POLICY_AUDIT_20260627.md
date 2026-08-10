# TRKH 5-Class Cartography Policy Audit - 2026-06-27

## Scope

- Branch: `classification-only-research`.
- Dataset: `D:\DataAI\AIEx\newdataset\class_f`.
- Goal: use train-only training dynamics to expose ambiguous/hard samples for class-1 boundary handling without touching val/test labels.
- Literature checked:
  - Dataset Cartography: training dynamics can separate easy, ambiguous, and hard/mislabeled regions: https://arxiv.org/abs/2009.10795
  - Generalized Cross Entropy: robust-loss option for noisy labels: https://arxiv.org/abs/1805.07836
  - DivideMix: clean/noisy split framing for noisy labels: https://arxiv.org/abs/2002.07394

## Code Changes

- Added `--data-cartography` and `--data-cartography-output` to training.
- `data_cartography_train.csv` is train-only and includes:
  - `target_index`, `target_name`;
  - `seen_count`, confidence/correctness/loss means;
  - `prediction_index`, `prediction_name`, top-2 margin;
  - `prob_0..prob_N` mean probabilities so existing manifest tools can consume it.
- `build_ambiguous_soft_targets.py` and `build_targeted_margin_manifest.py` now skip cartography rows with `seen_count <= 0`.
  - This matters because unseen rows have zero probabilities and otherwise look like false low-margin samples.
- Launcher `scripts/run_trkh_5class_attention_views_v8.ps1` now supports `DataCartography`.
- Fixed launcher handling for empty `ClassLossMultipliers`.

## Probe 1: Cartography Only

Run:

`runs\probe_v8_cartography_120b_2e_20260627b`

Config:

- resume: `runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\checkpoints\best.pt`
- epochs: `2`
- max train batches: `120`
- full validation
- skip final test
- architecture trace completed

Validation result:

| Metric | Value |
|---|---:|
| Macro F1 | `0.8809` |
| Class 1 precision | `0.5871` |
| Class 1 recall | `0.7815` |
| Class 1 F1 | `0.6705` |

Confusion highlights:

- `0->1`: `52`
- `1->0`: `16`
- `1->2`: `10`
- `2->1`: `13`
- `4->1`: `14`

Decision: does not pass the full-train gate `class-1 val F1 >= 0.70`.

## Train-Only Manifests From Probe 1

Output:

`runs\cartography_policy_probe_v8_20260627`

Ambiguous soft targets:

- rows: `600`
- skipped unseen cartography rows: `4950`
- reasons before cap: `low_margin_pair=518`, `error_pair=259`
- main selected pairs: `0->1=272`, `1->0=130`, `2->1=93`, `3->2=120`, `2->3=108`

Targeted margin:

- rows: `191`
- skipped unseen cartography rows: `4950`
- reasons: `focus_false_positive=154`, `focus_false_negative=37`
- main pairs: `0->1=114`, `2->1=36`, `1->0=28`

## Probe 2: Cartography Policy

Run:

`runs\probe_v8_cartography_policy_120b_2e_20260627`

Config:

- same as probe 1;
- adds `ambiguous_soft_targets_train_only.csv`;
- adds `targeted_margin_train_only.csv`;
- targeted margin loss weight: `0.05`;
- skip final test;
- architecture trace completed.

Validation result:

| Metric | Value |
|---|---:|
| Macro F1 | `0.8786` |
| Class 1 precision | `0.5764` |
| Class 1 recall | `0.7748` |
| Class 1 F1 | `0.6610` |

Confusion highlights:

- `0->1`: `55`
- `1->0`: `16`
- `1->2`: `11`
- `2->1`: `12`
- `4->1`: `15`

Decision: reject this manifest policy for full train. It reduced class-1 precision and F1 versus cartography-only probe.

## Interpretation

- The current class-1 limit is not simply under-exposure. The probes still trade recall for false positives around `0/1` and `2/1`.
- Softening/margining from a short cartography probe reinforces noisy boundary regions but does not make the decision boundary cleaner on validation.
- The visual audit remains consistent with this: class 1 overlaps strongly with class 0 and class 2, and some samples have weak label policy separation.

## Next Candidate

Do not run another full TRKH train from these manifests.

Generated train-only review artifact:

`runs\boundary_review_cartography_v8_train_20260627`

- CSV: `boundary_review_manifest.csv`
- HTML: `boundary_review_report.html`
- selected rows: `320`
- skipped unseen cartography rows: `4950`
- copied review images: `258`
- reasons: `focus_false_positive=120`, `focus_false_negative=41`, `low_margin_error=57`, `low_margin_correct_boundary=102`
- main pairs: `0-1=120`, `1-2=115`, `1-4=37`, `2-3=45`

Next higher-value work:

1. Use cartography CSV to render a manual boundary-review sheet for train-only `0->1`, `1->0`, `1->2`, `2->1`, and `4->1`.
2. Create a reviewed `clean/ambiguous/relabel` manifest, then rerun a probe with either:
   - removing hard suspected label errors from loss, or
   - softening only manually confirmed ambiguous rows.
3. Keep test split untouched until a validation probe passes the gate.

## AI Boundary Review Follow-Up

Reviewed output:

`runs\reviewed_boundary_policy_cartography_v8_train_20260627`

Inputs:

- source review artifact: `runs\boundary_review_cartography_v8_train_20260627`;
- AI-reviewed train-only CSV: `boundary_review_manifest_ai_reviewed.csv`;
- manual/visual status summary:
  - `ambiguous`: `184`
  - `correct`: `122`
  - `needs_crop`: `11`
  - `wrong`: `3`

Generated conservative train-only manifests:

- sample weights: `304` deduplicated rows;
- ambiguous soft targets: `145` deduplicated rows;
- targeted margin: `90` deduplicated rows;
- relabel candidates: `3` rows, review-only.

Validation probes:

| Run | Policy | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| `runs\probe_v8_reviewed_policy_120b_2e_20260627` | sample weights + soft targets + targeted margin | `0.8777` | `0.5779` | `0.7616` | `0.6571` | Reject |
| `runs\probe_v8_targeted_c1only_120b_2e_20260627` | targeted margin only for non-1 false positives into class 1 | `0.8782` | `0.5750` | `0.7616` | `0.6553` | Reject |

Both probes are worse than cartography-only class-1 F1 `0.6705`, so no full train was launched.

## Class-1 Reroute Calibration Follow-Up

Predictions exported from the cartography-only checkpoint without touching test:

- train: `runs\cartography_probe_predictions_20260627\train\predictions_detailed.csv`;
- val: `runs\cartography_probe_predictions_20260627\val\predictions_detailed.csv`.

Train-fit then apply-to-val:

| Run | Rule | Val Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| `runs\class1_reroute_train_to_val_cartography_v8_20260627\coarse_precision` | reject class-1 top1 if top1-top2 margin `<0.06` | `0.8811` | `0.6992` | `0.6159` | `0.6549` | Reject |
| `runs\class1_reroute_train_to_val_cartography_v8_20260627\coarse_with_rescue` | same selected rule | `0.8811` | `0.6992` | `0.6159` | `0.6549` | Reject |

Diagnostic val-fit upper bound:

- run: `runs\class1_reroute_train_to_val_cartography_v8_20260627\val_fit_upper_bound`;
- best simple rule: reject class-1 top1 if top1-top2 margin `<0.02`;
- val class-1 F1 only `0.6790`.

Decision: simple confidence/margin rerouting cannot cross the `0.70` class-1 gate and trades recall for precision.

## Cleanlab Review Follow-Up

Code:

- `trkh.tools.audit_label_issues_cleanlab` now accepts both old teacher-cache schema and `evaluate.py` detailed prediction schema:
  - `image_path` -> `path`;
  - `target_name` -> `true_name`;
  - `prob_0_<class_name>` style probability columns.
- Added regression test: `tests/test_cleanlab_label_issue_schema.py`.

Run:

`runs\cleanlab_cartography_probe_train_audit_20260627`

Summary:

- rows used: `9215` train-only;
- `cleanlab_issue_count`: `29`;
- top issue pairs:
  - `0->1`: `11`;
  - `3->2`: `9`;
  - `2->1`: `4`;
  - `4->1`: `3`;
  - `3->1`: `1`;
  - `4->0`: `1`.

Visual contact sheet:

`runs\cleanlab_cartography_probe_train_audit_20260627\cleanlab_issue_contact_sheet_top60.jpg`

Interpretation: top cleanlab rows are useful review candidates, but they are mostly boundary/quality cases rather than high-confidence hard relabels. Do not auto-relabel from this in-sample prediction audit. OOF probabilities are still the better next step if cleanlab is used as training signal.

## Top-5 TTA Blend Diagnostic

Goal: check whether the cartography TRKH checkpoint adds useful probability diversity to the current best teacher-level artifact.

Run:

`runs\blend_top5tta_trkh_cartography_valonly_20260627`

Validation-only selection:

- source teacher: `runs\ensemble_top5_tta_teacher_cache_5class_v1\teacher_probs_val.csv`;
- TRKH source: `runs\cartography_probe_predictions_20260627\val\predictions_detailed.csv`;
- sweep: `blend = (1-alpha) * top5_tta + alpha * trkh_cartography`;
- selected alpha: `0.46`;
- validation macro/class1 improved from `0.9139/0.7383` to `0.9176/0.7533`.

Frozen-alpha test audit:

- teacher baseline test macro/class1: `0.9248/0.7786`;
- blend test macro/class1: `0.9235/0.7786`.

Decision: reject. The validation uplift does not generalize, and class 1 does not improve on test.

## OOF Cleanlab Follow-Up

Goal: replace in-sample cleanlab with train-only out-of-fold probabilities so label-noise and ambiguity mining is not hidden by memorization.

Code:

- Added `trkh.tools.build_classification_oof_folds` to build stratified train-only OOF folds for `classification_folder` data.
- Added `trkh.tools.stitch_timm_oof_predictions` to stitch fold validation exports back to canonical train paths and map TIMM/ImageFolder class order to `data.yaml` order.
- Added regression tests:
  - `tests/test_build_classification_oof_folds.py`;
  - `tests/test_stitch_timm_oof_predictions.py`.

OOF data:

- fold dataset: `runs\classf_train_oof_folds_5x_20260627`;
- stitched prediction CSV: `runs\oof_mobilenetv3_5fold_2e120b_20260627\predictions_train_oof.csv`;
- rows: `9215`;
- fold rows: `1843` each;
- OOF train macro/class-1 F1: `0.8599/0.5403`.

Cleanlab on OOF:

- run: `runs\cleanlab_oof_mobilenetv3_5fold_2e120b_20260627`;
- rows used: `9215`;
- cleanlab issues: `442`;
- main issue pairs:
  - `0->1`: `164`;
  - `1->0`: `112`;
  - `1->2`: `35`;
  - `3->2`: `30`;
  - `2->3`: `26`;
  - `2->1`: `20`.

Interpretation: OOF probabilities expose far more plausible boundary/quality issues than in-sample prediction audits. The contact sheet still shows mostly boundary/quality ambiguity rather than simple hard relabels.

## OOF + Cartography Consensus Policy

Consensus artifact:

`runs\oof_cleanlab_cartography_consensus_20260627`

Summary:

- OOF cleanlab issues: `442`;
- cartography review rows: `320`;
- consensus rows: `88`;
- manual/visual status in consensus:
  - `ambiguous`: `52`;
  - `correct`: `31`;
  - `needs_crop`: `4`;
  - `wrong`: `1`.

Generated conservative policy:

`runs\consensus_policy_oof_cartography_v8_train_20260627`

- sample weights: `88`;
- soft targets: `47`;
- targeted margin rows: `31`;
- relabel candidate rows: `1` review-only.

Probe:

| Run | Policy | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| `runs\probe_v8_consensus_oof_policy_120b_2e_20260627` | sample weights + soft targets + targeted margin from OOF/cartography consensus | `0.8803` | `0.5808` | `0.7616` | `0.6590` | Reject |

Decision: no full train. Consensus policy is narrower than the previous 320-row reviewed policy but still worsens class-1 F1 relative to cartography-only `0.6705`.

## OOF Neighbor Label Policy and Downweight Probe

Code:

- Added `trkh.tools.build_oof_neighbor_label_policy` to intersect OOF cleanlab issues with train-only feature-neighbor consensus.
- Updated `trkh.tools.build_data_centric_sample_weights` to accept cleanlab OOF aliases:
  - `true_index`;
  - `pred_index`;
  - `suggested_index`;
  - `suggested_name`;
  - `top1_confidence`;
  - `self_confidence`.
- Added tests:
  - `tests/test_oof_neighbor_label_policy.py`;
  - cleanlab alias regression in `tests/test_data_centric_sample_weights.py`.

Strict neighbor policy:

`runs\oof_neighbor_label_policy_relaxed_20260627`

- feature banks: EfficientNetV2-S fine-tuned features and MobileNetV3 fine-tuned features;
- settings: top-k `15`, min vote fraction `0.60`, OOF confidence `>=0.75`, self-confidence `<=0.25`, issue rank `<=300`;
- strict rows: `13`;
- strict pairs:
  - `0->1`: `5`;
  - `0->2`: `3`;
  - `2->3`: `4`;
  - `4->0`: `1`.

Interpretation: truly strict relabel/ignore evidence is too sparse to move the model by itself.

OOF cleanlab downweight policy:

`runs\oof_cleanlab_sample_weights_boundary_20260627`

- selected train-only issues: `240`;
- reasons:
  - high-confidence disagreement: `180`;
  - low-self-confidence error: `51`;
  - low-margin boundary: `9`;
- boundary pairs:
  - `0-1`: `107`;
  - `1-2`: `33`;
  - `2-3`: `49`;
  - `4-rest`: `51`;
- weight range: `0.25` to `0.80`, mean `0.3131`.

Probe:

| Run | Policy | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| `runs\probe_v8_oof_cleanlab_weight_policy_120b_2e_20260627` | OOF cleanlab sample downweight only | `0.8804` | `0.5895` | `0.7483` | `0.6570` | Reject |

Decision: no full train. Even OOF-derived downweighting reduces effective boundary signal and does not pass class-1 validation gate.

## No-Hardrepeat Cartography Sanity Probe

Goal: check whether the default V8 hard-sample repeat manifest from older probes is hurting the current cartography-only setup.

Probe:

| Run | Change | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| `runs\probe_v8_no_hardrepeat_cartography_120b_2e_20260627` | `HardSampleManifest=''`, `HardSampleRepeatFactor=1.0`, cartography on | `0.8805` | `0.5846` | `0.7550` | `0.6590` | Reject |

Best class-1 F1 across the two probe epochs was `0.6608`, still below the cartography-only baseline class-1 F1 `0.6705` and far below the `0.70` full-train gate. Architecture trace completed successfully.

Decision: stale hard-repeat samples are not the main bottleneck. Keep hard-repeat optional, but do not spend more runs on toggling it unless paired with a materially new train-only signal.

## Strict OOF Cleanlab Soft-Relabel Probe

Goal: test the stronger hypothesis that the highest-confidence OOF cleanlab issues are true label-policy errors, so they should receive relabel-like soft targets rather than simple downweighting.

Code/artifacts:

- Added `trkh.tools.build_oof_cleanlab_review_manifest` to create a train-only class-1 boundary review package from OOF cleanlab issues.
- Added `trkh.tools.build_oof_cleanlab_strict_soft_targets` to create thresholded train-only soft targets from OOF cleanlab/review manifests.
- Review package: `runs\oof_cleanlab_class1_boundary_review_20260627`
  - selected rows: `203`;
  - hardlinked review images: `203`;
  - selected by direction: `0->1=70`, `1->0=70`, `1->2=35`, `2->1=20`, `4->1=5`, `1->4=3`.
- Strict soft-target policy: `runs\strict_oof_cleanlab_soft_targets_class1_boundary_20260627`
  - thresholds: `top1_confidence >= 0.95`, `self_confidence <= 0.05`, `top2_margin >= 0.80`;
  - soft alpha: `0.85`;
  - selected rows: `93`;
  - selected by direction: `1->0=58`, `1->2=19`, `0->1=11`, `2->1=2`, `1->4=2`, `4->1=1`.

Probe:

| Run | Policy | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| `runs\probe_v8_strict_oof_soft_class1_120b_2e_20260627` | strict high-confidence OOF cleanlab soft targets, alpha `0.85` | `0.8794` | `0.6022` | `0.7219` | `0.6566` | Reject |

Architecture trace completed successfully. Smoke artifact was deleted after validation to save space.

Decision: high-confidence OOF cleanlab soft-relabeling improves class-1 precision slightly but loses too much recall, ending below both cartography-only class-1 F1 `0.6705` and the `0.70` full-train gate. Do not auto-relabel from OOF cleanlab without a manual label-policy pass.

## Metric Learning Ablation

Goal: check whether the current supervised contrastive metric-learning term is amplifying noisy class-1 positives/negatives.

Probe:

| Run | Change | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| `runs\probe_v8_no_metric_learning_cartography_120b_2e_20260627` | `MetricLearningLossWeight=0`, cartography on | `0.8801` | `0.5838` | `0.7616` | `0.6609` | Reject |

Architecture trace completed successfully. Smoke artifact was deleted after validation to save space.

Decision: metric-learning noise is not the dominant bottleneck in this setup. Do not implement a more complex weighted contrastive path until there is a stronger signal, because removing the term entirely did not beat the cartography-only class-1 F1 `0.6705`.

## Focus-Class Head Probe

Goal: test whether an auxiliary binary head for class 1 can improve the ambiguous one-vs-rest boundary without changing the main no-pretrain TRKH backbone.

Probe:

| Run | Change | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| `runs\probe_v8_focus_class1_head_120b_2e_20260627` | `FocusClassHead=true`, class index `1`, aux loss `0.05`, positive weight `1.25` | `0.8817` | `0.5928` | `0.7616` | `0.6667` | Reject |

Architecture trace completed successfully. The smoke artifact should be deleted after validation to save space.

Decision: the focus-class auxiliary head slightly improves macro F1 and class-1 F1 versus the no-metric/no-hardrepeat ablations, but it still does not beat the cartography-only class-1 best (`0.6705`) or open the `0.70` full-train gate. Do not run a full train from this head alone.

## Quality GroupDRO Probe

Goal: test whether class-1 failure is concentrated in train-only image-quality groups rather than only in label-boundary rows.

Code/artifacts:

- Built train-only quality-group manifest: `runs\quality_group_class_quality_cartography_probe_20260628\quality_groups_train_only.csv`
- source split: train only, `9215` rows, `0` invalid images, `9215` cartography matches
- grouping: `class_quality`, `26` groups
- quality buckets: `dark_dirty_detail=6037`, `normal=2324`, `overbright=516`, `partial_or_border=199`, `low_contrast=125`, `underdark=14`
- class-1 groups: `c1__dark_dirty_detail=279`, `c1__normal=196`, `c1__overbright=59`, `c1__low_contrast=7`

Probe:

| Run | Change | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| `runs\probe_v8_quality_groupdro_class_quality_120b_2e_20260628` | `QualityGroupManifest`, `GroupDroLossWeight=0.05`, temperature `0.25`, min samples `2` | `0.8796` | `0.5838` | `0.7616` | `0.6609` | Reject |

Architecture trace completed successfully. Smoke artifact was deleted after validation to save space.

Decision: quality-aware GroupDRO is active but does not improve class 1. It behaves like another robustness weighting signal and lands below both the cartography-only class-1 best (`0.6705`) and the `0.70` gate. Do not run full train from GroupDRO alone.

## YOLO_F Object-Crop Probe

Goal: check whether the alternative candidate dataset `D:\DataAI\AIEx\newdataset\yolo_f` carries better source-image context than the static crop dataset `class_f`.

Dataset audit:

- `class_f` is a `classification_folder` crop dataset with train/val/test object counts `9215/2606/1267`.
- `yolo_f` is a YOLO source-image dataset. Object counts exactly match `class_f`:
  - train: `9215` objects from `8064` images, with `691` multi-object images;
  - val: `2606` objects from `2577` images, with `29` multi-object images;
  - test: `1267` objects from `1267` images.
- No missing labels, malformed labels, or out-of-range boxes were found in the YOLO audit.
- Class-1 contact sheet from YOLO crops still shows the same boundary/lighting/quality ambiguity as `class_f`.

Code fix:

- `trkh.tools.trace_architecture` now instantiates `MangoYOLOCropDataset` when `data.yaml` is YOLO-format instead of always using `ClassificationFolderDataset`.
- Direct trace retry on the YOLO smoke checkpoint completed with `samples: 5`.

Probe:

| Run | Change | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| `runs\probe_v8_yolof_nohard_cartography_120b_2e_20260628` | YOLO object crops, no stale hard-repeat manifest, cartography on | `0.8797` | `0.5897` | `0.7616` | `0.6647` | Reject |

Architecture trace completed successfully. No final test was run.

Decision: `yolo_f` is useful for source-context audit, but the current dynamic object-crop path does not beat the `class_f` cartography-only class-1 best (`0.6705`) and does not open the `0.70` full-train gate.

## Clean-Anchor Balanced Repeat Probe

Goal: test the opposite of hard/noisy boundary emphasis: repeat only high-confidence correct train anchors, balanced per class, so class 1 gets cleaner local support without global class-1 oversampling.

Code/artifacts:

- Added `trkh.tools.build_clean_anchor_manifest`.
- Added regression test `tests/test_clean_anchor_manifest.py`.
- Built train-only artifact: `runs\clean_anchor_cartography_160pc_20260628`.
- Selected `800` clean anchors: exactly `160` per class.
- Source cartography rows: `9524`; selected after thresholding/deduplication from correct anchors by class:
  - class 0: `843` available;
  - class 1: `477` available;
  - class 2: `897` available;
  - class 3: `963` available;
  - class 4: `979` available.
- Hard-repeat summary in smoke/probe:
  - manifest paths: `800`;
  - base samples: `9215`;
  - effective samples: `9703`;
  - extra repeats: `488`;
  - balanced epoch exposure: about `1945-1946` per class.

Focused verification:

- `python -m pytest tests\test_clean_anchor_manifest.py -q`: `1 passed`.
- `python -m py_compile trkh\tools\build_clean_anchor_manifest.py trkh\tools\trace_architecture.py`: passed.

Probe:

| Run | Change | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| `runs\probe_v8_clean_anchor_160pc_120b_2e_20260628` | clean-anchor hard repeat, repeat factor `1.6`, cartography on | `0.8788` | `0.5895` | `0.7417` | `0.6569` | Reject |

Architecture trace completed successfully. No final test was run.

Decision: cleaner repeated anchors reduce class-1 recall and do not improve precision enough. This argues against further simple sample-repeat/weighting probes unless a genuinely new train-only label-policy signal is added.

## YOLO Source-Context Review and Margin Probe

Goal: use `yolo_f` source images as a review signal to check whether class-1 boundary errors come from crop mistakes, missing surrounding context, or true label-policy ambiguity.

Code/artifacts:

- Added `trkh.tools.build_yolo_source_context_review`.
- Added regression test `tests/test_yolo_source_context_review.py`.
- The source-context CSV includes manual review fields compatible with `trkh.tools.build_reviewed_boundary_training_manifests`:
  - `manual_label_status`;
  - `manual_expected_class`;
  - `quality_lighting`;
  - `quality_dirty_obstacle`;
  - `quality_partial_fruit`;
  - `quality_background_mask`;
  - `review_notes`.
- Exposed YOLO crop margin controls in `scripts\run_trkh_5class_attention_views_v8.ps1`:
  - `CropMarginRatio`;
  - `ClassCropMarginScaleThreshold`;
  - `ClassCropMarginMaxRatio`.
- Focused tests:
  - `python -m pytest tests\test_yolo_source_context_review.py tests\test_clean_anchor_manifest.py -q`: `2 passed`.

Review artifact:

`runs\yolo_source_context_oof_class1_boundary_review_20260628`

- input rows from OOF cleanlab class-1 boundary review: `203`;
- rendered train-only source-context rows: `160`;
- skipped missing mapping: `0`;
- class_f label vs YOLO label mismatch: `0`;
- boundary pairs:
  - `0-1`: `109`;
  - `1-2`: `43`;
  - `1-4`: `8`;
- source object-count distribution:
  - single-object rows: `126`;
  - multi-object rows: `34`;
- median YOLO bbox area: about `0.2269`.

Visual interpretation: source-context composites show the same maturity/quality ambiguity as crop-only review. Most selected rows are single-object and map cleanly to YOLO labels, so crop/box mismatch is not the dominant class-1 bottleneck.

Probe:

| Run | Change | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| `runs\probe_v8_yolof_margin020_120b_2e_20260628` | YOLO object crops with global crop margin `0.20`, no stale hard-repeat, cartography on | `0.8562` | `0.5538` | `0.6821` | `0.6113` | Reject |

Architecture trace completed successfully. No final test was run.

Decision: adding wider source context hurts class 1 and macro F1. Do not run full train, and do not continue tuning crop margin without a new visual reason. The source-context artifact is valuable for manual label-policy review, not as an automatic crop-margin training signal.

## YOLO Source-Context Training Probes - 2026-06-30

Goal: answer whether using the wider `yolo_f` source image, including background and object position, lets ViT self-attention separate object evidence from context better than object crops.

Code changes:

- `MangoYOLOCropDataset` can now return source-context classification images without editing data.
- Supported layouts:
  - `full`: full source frame with object spotlight/background suppression;
  - `crop_inset`: object crop with a source-context inset.
- Train/evaluate/launcher/architecture trace paths support the new source-context config.
- Focused regression test: `tests\test_yolo_source_context_dataset.py`.

Probe results:

| Run | Change | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| `runs\probe_v8_yolof_sourcectx_120b_2e_20260630` | full source frame with bbox spotlight/context | `0.7573` | n/a | n/a | `0.4460` | Reject |
| `runs\probe_v8_yolof_sourcectx_inset_120b_2e_20260630` | object crop plus source-context inset | `0.8688` | n/a | n/a | `0.6535` | Reject |
| `runs\probe_v8_yolof_focusboundary_120b_2e_20260630` | YOLO object crop plus focus/boundary head | `0.8803` | `0.5813` | `0.7815` | `0.6667` | Reject |

Interpretation:

- Full-frame context is actively harmful on this split. The model sees more background and non-target fruit without a supervised segmentation signal, so self-attention does not automatically isolate the correct object.
- The crop+inset layout is less harmful but still below the object-crop baseline (`0.6647`) and the `class_f` cartography-only class-1 result (`0.6705`).
- The focus/boundary head gives the best of these three context-adjacent probes, but it still does not cross the `0.70` full-train gate.

## YOLO BBox Spatial Prior Fusion - 2026-06-30

Goal: test whether the normalized YOLO bbox geometry (`x`, `y`, `w`, `h`, border distance, area/aspect, center offset) can provide a lightweight object-position prior without modifying data.

Code changes:

- `MangoYOLOCropDataset` can return classification metadata: `(image_tensor, label, {"bbox": xywh})`.
- `VisionTransformerWithRegisters` has optional `BBoxSpatialPriorFusion`, a zero-init residual MLP that adds small bbox logits only when `--bbox-spatial-fusion` is enabled.
- Training, EMA resume allowlist, evaluate, architecture trace, and launcher now pass bbox metadata through.
- Focused regression test: `tests\test_yolo_bbox_spatial_fusion.py`.

Verification:

- Compile: `py_compile` passed for dataset/model/config/train/evaluate/trace and the bbox test.
- Unit tests: `tests\test_yolo_bbox_spatial_fusion.py` and `tests\test_yolo_source_context_dataset.py` passed.
- Smoke with architecture trace: `runs\smoke_v8_yolof_bboxprior_retry_20260630`, trace status `completed`.

Probe:

| Run | Change | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| `runs\probe_v8_yolof_bboxprior_120b_2e_20260630` epoch 1 | YOLO object crop + bbox spatial prior | `0.8825` | `0.5979` | `0.7682` | `0.6725` | Reject full train |
| `runs\probe_v8_yolof_bboxprior_120b_2e_20260630` epoch 2 | same | `0.8800` | `0.5900` | `0.7815` | `0.6724` | Reject full train |

Confusion at epoch 2:

- `0->1`: `53`
- `1->0`: `16`
- `1->2`: `11`
- `2->1`: `12`
- `4->1`: `13`

Decision: bbox spatial prior is technically working and has a small positive signal versus YOLO object-crop baseline (`0.6647 -> 0.6725`), but it only matches/slightly exceeds `class_f` cartography-only (`0.6705`) and remains below the `0.70` gate. Do not launch full train from this variant.

## Why External/Pretrained Models Still Look Stronger

The local evidence now points away from "missing background context" as the main bottleneck. `yolo_f` full background hurts, crop+inset hurts less but still loses, and bbox position helps only slightly. The stronger AIDT/pretrained/TTA models likely benefit from representation scale and inductive bias, not just dataset layout.

Relevant primary sources checked:

- ViT needs large-scale pretraining to work well compared with convolutional baselines on smaller data: https://arxiv.org/abs/2010.11929
- DeiT improves data-efficient transformer training via strong augmentation/distillation, still relying on training recipe/teacher signal beyond plain scratch ViT: https://arxiv.org/abs/2012.12877
- Shortcut learning/background bias is a known failure mode in visual classifiers when context correlates with labels: https://arxiv.org/abs/2004.07780

Implication: for a reportable no-pretrain TRKH contribution, the next useful step is not more automatic source-context/bbox tweaks. Either add a genuinely new supervised signal (manual train-only boundary policy, object-tight segmentation/localization supervision) or make the paper honest about the no-pretrain limitation and compare against pretrained baselines as stronger external-representation systems.

## Updated Decision

Do not repeat simple soft-target, targeted-margin, reroute, downweight, strict OOF soft-relabel, metric-learning ablation, focus-head-only, quality-GroupDRO-only, YOLO object-crop-only, YOLO wide-margin-only, YOLO source-context-only, bbox-prior-only, or clean-anchor-repeat policies from cartography/cleanlab alone. OOF cleanlab and YOLO source images are valuable as audit/review sources, but neither is sufficient as an automatic training policy for class 1.

Higher-value next directions:

1. Manual label-policy pass on OOF cleanlab boundary images, especially `0/1` and `1/2`, with explicit `correct/ambiguous/wrong/needs_crop` labels.
2. If continuing automation, use OOF cleanlab only to sample review candidates, not to directly relabel/downweight all issues.
3. For model-side work, use a new source of supervision or representation signal; simple weighting/loss/crop/context/bbox variants have repeatedly failed the `class-1 val F1 >= 0.70` gate.

## YOLO Source-Context Auxiliary Localization - 2026-06-30

Goal: test the user's hypothesis that `yolo_f` can teach object/background separation while the classifier still learns from object crops. This keeps raw data untouched and uses YOLO bboxes only as train-time supervision.

Relevant primary sources checked:

- BoxSup shows bounding boxes can supervise object localization/segmentation without pixel masks: https://arxiv.org/abs/1503.01640
- Hide-and-Seek/attention dropping is a weakly supervised localization idea, but it can expose non-object shortcuts if applied without a strong object prior: https://arxiv.org/abs/1704.04232
- TransFG uses transformer attention for fine-grained part selection, supporting the general idea that attention can find discriminative regions, but it does not imply background context will be useful without suitable supervision: https://arxiv.org/abs/2103.07976

Code changes:

- `MangoYOLOCropDataset` can return a train-time auxiliary source-context image plus transformed `source_context_bbox`.
- Training supports:
  - `--classification-source-context-aux`;
  - `--source-context-aux-loss-weight`;
  - optional source-context CE/consistency weights, kept at `0.0` for accepted probes because full-frame CE was noisy.
- Added bbox attention alignment loss over patch-token energy, with history fields:
  - `train_source_context_aux_loss`;
  - `train_source_context_aux_inside_mass`;
  - `train_source_context_aux_outside_mass`;
  - `train_source_context_aux_valid_fraction`.
- Added optional bbox-guided token pruning:
  - `--token-prune-bbox-weight`;
  - `--token-prune-bbox-margin-ratio`.
  This is off by default and only affects pruning when a bbox token prior is explicitly passed to `forward_features`.

Verification:

- Compile passed for `trkh\data\dataset.py`, `trkh\core\config.py`, `trkh\models\model.py`, `trkh\training\train.py`, `trkh\tools\trace_architecture.py`, and focused tests.
- Focused pytest: `tests\test_yolo_bbox_spatial_fusion.py tests\test_yolo_source_context_dataset.py -q` -> `6 passed`.

Probe A: source-context bbox-attention only

Run: `runs\probe_v8_yolof_sourceaux_bboxattn_120b_2e_20260630`

| Epoch | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Aux inside mass | Aux outside mass |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.8743` | `0.5722` | `0.7351` | `0.6435` | `0.4779` | `0.5221` |
| 2 | `0.8761` | `0.5672` | `0.7550` | `0.6477` | `0.4875` | `0.5125` |

XAI audit:

- Output: `runs\probe_v8_yolof_sourceaux_bboxattn_120b_2e_20260630\xai_audit_val_small`
- Selected cases: `20`.
- Top confusions: `3->2=56`, `0->1=53`, `1->0=16`, `4->1=16`, `1->2=15`.
- Background perturbation is still near-zero:
  - background blur prediction drop: `0.00138`;
  - background gray prediction drop: `0.00070`.
- Object desaturation is much stronger:
  - original prediction drop: `0.1278`.

Decision: reject. It is below YOLO bbox-prior (`0.6725` class-1 F1), below `class_f` cartography-only (`0.6705`), and far below the `0.70` full-train gate.

Probe B: bbox spatial prior + source-context bbox-guided token pruning

Run: `runs\probe_v8_yolof_bboxprior_sourceaux_bboxprune_120b_2e_20260630`

Config highlights:

- `--bbox-spatial-fusion`;
- `--classification-source-context-aux`;
- `--source-context-aux-loss-weight 0.04`;
- `--token-prune-bbox-weight 1.2`;
- `--token-prune-bbox-margin-ratio 0.02`;
- no source-context CE or KL consistency.

| Epoch | Macro F1 | Class 1 precision | Class 1 recall | Class 1 F1 | Aux inside mass | Aux outside mass |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.8751` | `0.5781` | `0.7351` | `0.6472` | `0.4964` | `0.5036` |
| 2 | `0.8753` | `0.5604` | `0.7682` | `0.6480` | `0.5072` | `0.4928` |

Best confusion matrix:

- `0->1`: `60`;
- `1->0`: `17`;
- `1->2`: `13`;
- `2->1`: `11`;
- `4->1`: `17`.

XAI audit:

- Output: `runs\probe_v8_yolof_bboxprior_sourceaux_bboxprune_120b_2e_20260630\xai_audit_val_small`
- Selected cases: `20`.
- Review flags: attention background `11`, attention border `8`, gradcam background `8`, gradcam border `10`, object-color sensitive `10`.
- Mean foreground mass:
  - attention: `0.7803`;
  - gradcam: `0.7719`.
- Background perturbation remains near-zero:
  - background blur prediction drop: `0.00140`;
  - background gray prediction drop: `0.00090`.
- Object desaturation remains the dominant perturbation:
  - original prediction drop: `0.1227`.

Decision: reject. Bbox-guided pruning improves source-branch inside mass but worsens class-1 precision and overall class-1 F1. The localization signal is technically active, but it does not resolve the maturity/quality boundary. Do not launch full train from this branch.

Cleanup:

- Deleted failed/tentative artifacts:
  - `runs\probe_v8_yolof_sourceaux_bboxattn_180b_2e_20260630`;
  - `runs\probe_v8_yolof_sourceaux_bboxattn_180b_2e_retry_20260630`;
  - `runs\probe_v8_yolof_sourceaux_bboxattn_180b_2e_retry2_20260630`;
  - `runs\probe_v8_yolof_sourceaux_bboxattn_180b_2e_retry3_20260630`;
  - `runs\smoke_v8_yolof_sourceaux_20260630`;
  - `runs\smoke_v8_yolof_sourceaux_retry_20260630`;
  - `runs\smoke_v8_yolof_sourceaux_bboxprune_20260630`.

Updated interpretation:

- Wider context and train-time bbox localization are useful as audit/supervision signals, but the model's class-1 failures still come from fruit surface color/quality/maturity ambiguity.
- The current evidence says not to keep pushing full-source auxiliary classification, consistency, or bbox localization loss alone.
- A future model-side attempt should use a materially different train signal, such as object-aware token dropout/background invariance on the crop itself, or a longer internal representation-learning run with explicit part-level audit, not another simple source-context/bbox tweak.
