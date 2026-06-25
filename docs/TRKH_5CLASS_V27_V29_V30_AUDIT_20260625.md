# TRKH 5-Class V27/V29/V30 Audit - 2026-06-25

## Scope

- Repo: `D:\DataAI\AIEx\TRKH`.
- Dataset: `D:\DataAI\AIEx\newdataset\class_f`.
- No-pretrain TRKH remains the priority.
- Test split was not used for tuning in this cycle.
- Full train gate remains class-1 validation F1 `>= 0.70`.

Best no-pretrain reference before this cycle is V16:

- Run: `runs/probe_mango_cls_256_5class_routed_pairwise_v16_120b_6e_20260615`.
- Validation macro/class-1 F1: `0.8874 / 0.6866`.
- Class 1 precision/recall: `0.6250 / 0.7616`.

## Audit Evidence Used

Existing XAI/background audits show:

- Foreground heatmap mass is generally high.
- Background blur/gray perturbation has near-zero effect in earlier XAI audits.
- Object desaturation changes predictions much more than background perturbation.
- Class-1 failures are concentrated at `0<->1`, `1<->2`, plus some `2<->1/4->1`, with strong lighting and label-boundary ambiguity.

Manual review of V29 validation review images confirmed this pattern:

- high-confidence `0->1`: fruit fills the image; background is not the primary cue.
- high-confidence `2->1`: heavy shadow/illumination changes object color.
- low-margin `1->0` and `1->2`: dirty/scratched mangoes and maturity boundary are visually ambiguous.

This means more generic background suppression is unlikely to be enough.

## Research Check

The next attempts were grounded in foreground/background and noisy-label literature:

- Foreground/background recomposition can reduce background bias in Transformers: https://arxiv.org/html/2503.09399v1
- Object/background sensitivity is a real classification issue: https://openaccess.thecvf.com/content/CVPR2022/papers/Moayeri_A_Comprehensive_Study_of_Image_Classification_Model_Sensitivity_to_Foregrounds_CVPR_2022_paper.pdf
- Foreground object transformation is useful in fine-grained few-shot settings: https://arxiv.org/abs/2109.05719
- Attention-guided crop/drop is a standard weakly supervised fine-grained strategy: https://doi.org/10.1088/1742-6596/1754/1/012189

Local audit constrained the implementation: use light foreground crop/forensics first, not strong background recomposition, because V22/V25-style counterfactual/background operations did not improve class 1.

## V27 - Data-Centric Sample Weights

Added:

- `trkh.tools.build_data_centric_sample_weights`
- `scripts/run_trkh_5class_data_centric_v27.ps1`
- `tests/test_data_centric_sample_weights.py`

Manifest:

- Output: `runs/data_centric_weights_v27_train_only_20260625`.
- Rows merged into final sample weight manifest: `464`.
- Candidate issues: `391`; selected issues: `279`.
- Selected reasons: high-confidence disagreement `220`, low-self-confidence error `9`, low-margin boundary `50`.

Probe:

- Run: `runs/probe_mango_cls_256_5class_data_centric_v27_120b_6e`.
- Best epoch: `3`.
- Validation macro/class-1 F1: `0.8842 / 0.6746`.
- Forensics class-1 F1: `0.6785`.

Decision:

- Rejected. It underperforms V16.
- Main issue: manifest was mined from older V3 predictions; downweighting stale errors likely reduced useful class-1 precision.

## V29 - High Resolution 384

Change:

- Added `ImageSize` propagation through V8/V12/V16/V27 launchers.
- Ran V16-style probe at `384`, batch `16`, grad accumulation `4`.

Probe:

- Run: `runs/probe_mango_cls_384_5class_routed_pairwise_v29_80b_4e`.
- Best epoch: `4`.
- Validation macro/class-1 F1: `0.8788 / 0.6524`.
- Forensic class-1 F1: `0.6505`.

Compared with V16:

- `0->1`: `47 -> 45`, slightly better.
- `1->0`: `17 -> 22`, worse.
- `1->2`: `13 -> 15`, worse.
- `2->1`: `9 -> 14`, worse.
- `3->2`: unchanged at `54`.

Decision:

- Rejected. Higher resolution alone makes class-1 false negatives worse and does not address `3->2`.

## V30 - Foreground Object Crop

Added:

- Optional transform-level `foreground_crop_mode`: `none|pseudo|grabcut`.
- Train probability and eval deterministic crop support.
- Config/CLI/checkpoint propagation through train, evaluate, XAI, robustness, attention viz, trace.
- Launcher: `scripts/run_trkh_5class_foreground_crop_v30.ps1`.
- Test: `tests/test_foreground_crop_transform.py`.

Rationale:

- V17 background suppression changed the image distribution and hurt F1.
- V30 instead crops around object-like foreground to increase fruit detail while preserving original colors.

Smoke:

- Run: `runs/smoke_mango_cls_256_5class_foreground_crop_v30`.
- `foreground_crop_pseudo` appears in train transform log.
- Architecture trace completed.

Probe:

- Run: `runs/probe_mango_cls_256_5class_foreground_crop_v30_120b_6e`.
- Best epoch: `2`.
- Early stopped at epoch `5`.
- Validation macro/class-1 F1: `0.8851 / 0.6786`.
- Class 1 precision/recall: `0.6162 / 0.7550`.
- Confusion: `0->1=48`, `1->0=18`, `1->2=14`, `2->1=10`, `3->2=50`, `4->1=10`.
- Architecture trace with `foreground_crop_box`:
  `runs/probe_mango_cls_256_5class_foreground_crop_v30_120b_6e/architecture_trace`.

Decision:

- Rejected for full train. It remains below V16 class-1 F1 `0.6866` and below gate `0.70`.
- Foreground crop slightly improves `3->2` but not class 1.

## Current Decision

Do not launch full train from V27, V29, or V30.

Current best no-pretrain probe remains:

`runs/probe_mango_cls_256_5class_routed_pairwise_v16_120b_6e_20260615`

with validation macro/class-1 F1 `0.8874 / 0.6866`.

## Next High-Value Direction

The remaining gap is unlikely to be solved by another generic crop/mask/head. Next work should be data/decision focused:

1. Create a train-only and val-only boundary review manifest with buckets: label likely correct, ambiguous, likely wrong, lighting/shadow, dirty/obstacle, partial fruit.
2. Build group-clean split by source sequence before claiming generalization.
3. Try train-only self-supervised or consistency pretraining only if label audit says labels are reliable enough; otherwise it will learn ambiguous boundaries more strongly.
4. If the target remains `0.99/0.96`, report explicitly that the current no-pretrain oracle/ensemble evidence suggests label ambiguity is now the limiting factor.
