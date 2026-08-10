# TRKH Deep Research 11 Action Audit - 2026-06-25

## Scope

Read `D:\deep-research-report-11-clean.md` and mapped it back to the active TRKH 5-class work on:

- repo: `D:\DataAI\AIEx\TRKH`
- dataset: `D:\DataAI\AIEx\newdataset\class_f`
- current no-pretrain gate: no full train until a smoke/probe reaches class-1 validation F1 `>= 0.70`

The report is useful, but it was written against a `classification-only-research` framing and some pair assumptions do not exactly match the current `class_f` V16 validation errors.

## Useful Findings From The Report

The report's strongest point is procedural: stop adding generic heads/losses and first turn failures into structured metrics. This matches the current plateau from V8-V24.

Actionable items already aligned with current work:

- Add sample-level forensics: calibration metrics, top-2 margin, pair metrics, lighting/background buckets.
- Treat the labels as mixed latent factors, not pure ordinal classes.
- Prefer counterfactual prediction consistency over harder pseudo-segmentation masks.
- Use train-only ambiguity handling via soft targets/manifests, not external teachers and not val/test mining.
- Keep all new knobs default-off and gate full training by short probes.

## Local Correction Against `class_f`

The report proposes focusing on `0-1` and `3-4`. Current V16 validation forensics says the active priority is different:

- `3->2`: `54`
- `0->1`: `47`
- `1->0`: `17`
- `1->2`: `13`
- `0->4`: `11`
- `4->1`: `10`
- `2->1`: `9`

So for the current `class_f` run, the priority pairs are:

- primary: `0-1`
- primary: `2-3`
- secondary: `1-2`
- monitor-only: `3-4`

Direct `3<->4` confusion is currently tiny (`3->4=1`, `4->3=1` on V16 val), so optimizing hard for `3-4` now would be poorly targeted.

## V16 Forensics Summary

Input:

`runs/probe_mango_cls_256_5class_routed_pairwise_v16_120b_6e_20260615/eval_val_detailed/predictions_detailed.csv`

Output:

`runs/probe_mango_cls_256_5class_routed_pairwise_v16_120b_6e_20260615/val_forensics_report_20260625`

Key metrics:

- samples: `2606`
- validation accuracy: `0.9229`
- validation macro F1: about `0.8870`
- class-1 precision/recall/F1: `0.6250 / 0.7616 / 0.6866`
- ECE: `0.6174`
- Brier: `0.6158`
- NLL: `1.2120`
- top-2 margin mean on errors: `0.0626`
- top-2 margin median on errors: `0.0601`

Interpretation:

- Errors are mostly low-margin and under-confident, not high-confidence false positives.
- A large part of class-1 weakness is boundary ambiguity plus weak discrimination near neighboring classes.
- Pure confidence calibration is unlikely to solve class 1 by itself.

## Pair Calibration Check

I ran a validation-only grid audit on V16 probabilities. This was not used on test and is only for deciding whether a pair-calibration head is worth implementing.

Baseline V16 validation:

- macro F1: `0.8870`
- class-1 F1: `0.6866`
- class-1 precision/recall: `0.6250 / 0.7616`

Best simple pair edits:

- `0-1` only: no improvement, class-1 F1 remains `0.6866`
- `1-2` only: class-1 F1 `0.6905`, macro F1 `0.8880`
- `2-3` only: macro F1 `0.8916`, class-1 F1 unchanged
- combined `0-1 + 2-3`: macro F1 `0.8916`, class-1 F1 unchanged

Conclusion:

- A simple pairwise logit bias/calibration layer is not enough to cross the `0.70` class-1 gate.
- If a pair-specialist head is attempted later, it must use actual feature evidence, not only top-2 logit adjustment.

## Decision

Do not launch full train.

Do not spend the next cycle on a simple pair-calibration head.

Next high-value work should be one of these, in order:

1. Integrate forensic export into the normal evaluation path, with cheap modes by default and foreground modes behind limits/cache.
2. Build train-only ambiguous-boundary manifests from low-margin/error-prone train predictions and use existing soft-target support.
3. Add background-neutralized consistency as prediction-level KL consistency, not another hard mask.
4. Add local lighting consistency only after background/ambiguity has a clean smoke/probe path.

## Next Probe Candidate

Proposed V25 direction:

- no-pretrain
- start from V16/V12 stable config
- target pairs: `0-1,2-3,1-2`
- add train-only ambiguity manifest support
- add optional background-neutralized consistency with low probability
- keep all new flags default-off
- run smoke first, then a short probe only
- no full train unless class-1 validation F1 reaches `>= 0.70`

