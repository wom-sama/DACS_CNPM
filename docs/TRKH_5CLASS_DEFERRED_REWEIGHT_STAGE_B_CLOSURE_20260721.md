# TRKH 5-Class Deferred Reweight Stage-B Closure - 2026-07-21

## Decision

Reject and close the exact natural-first deferred-reweight recipe. Do not
implement Stage C, sweep the natural-sampling horizon, run a probe/full train,
open test, or update the current-best command from this route.

The random-initialized five-epoch Stage-B smoke used all 9,215 immutable
`yolo_f` train rows, natural shuffled sampling, no class weights, scheduler
horizon 30, seed 42, batch 32, gradient accumulation 2, train/eval workers
4/2, full validation, and no test. The only changes from the locked scratch
recipe were strict-balanced to natural sampling and run horizon 30 to 5 while
preserving scheduler horizon 30.

## Independent Clean Result

The independent FP32 reload covers all 2,606 validation objects and exactly
replays its prediction CSV. Macro/class-1 F1 is
`0.767070/0.291262`; class-1 precision/recall is
`0.545455/0.198675`. The class-1 confusion row is
`[70,30,40,0,11]`: only 30 of 151 true class-1 objects survive.

The low class-1 FP count (`25`) is therefore suppression, not selective
precision. Relative to the locked strict epoch-5 reference, natural sampling
raises precision from `0.373702` to `0.545455` but lowers recall from
`0.715232` to `0.198675` and class-1 F1 from `0.490909` to `0.291262`.
Macro F1 also misses its `0.778520` gate.

All four nonfocus safety checks pass:

| Class | Candidate F1 | Locked reference F1 | Delta |
|---:|---:|---:|---:|
| 0 | `0.877843` | `0.863850` | `+0.013993` |
| 2 | `0.831904` | `0.824593` | `+0.007311` |
| 3 | `0.922198` | `0.905633` | `+0.016565` |
| 4 | `0.912141` | `0.907618` | `+0.004523` |

The result isolates the failure: natural sampling benefits common classes and
shrinks class-1 support too aggressively during representation formation.
Late weighting cannot be authorized from an epoch-5 state that has already
lost 121 of 151 class-1 positives.

## Robustness

The candidate loses to the current keeper on macro F1 in all five matched
full-validation conditions and fails the existing class-1 TP safety rule:

| Condition | Candidate macro F1 | Keeper macro F1 | Delta | Candidate/keeper class-1 TP |
|---|---:|---:|---:|---:|
| clean | `0.767070` | `0.882925` | `-0.115855` | `30/117` |
| center occlusion | `0.759253` | `0.876231` | `-0.116979` | `24/113` |
| dim | `0.722743` | `0.780164` | `-0.057421` | `36/69` |
| bright | `0.659424` | `0.809343` | `-0.149919` | `16/90` |
| low contrast | `0.717767` | `0.801929` | `-0.084162` | `20/86` |

The brightest condition is the worst candidate condition, but the clean TP
collapse is already decisive. This is not a corruption-only weakness.

## XAI Review

The fixed 12-row validation cohort contains four keeper class-1 false
negatives, four keeper class-1 false positives, and four nonfocus `3 -> 2`
boundary errors. Candidate all-method FP32 XAI uses native block-7 MHSA
probabilities; the historical keeper attention column is explicitly labeled
`feature-map fallback`. Grad-CAM and eight-layer grad-rollout are the primary
paired evidence.

The automated XAI safety checks pass. Candidate-minus-keeper foreground-mass
deltas are `+0.020441` for Grad-CAM and `+0.009902` for grad-rollout;
background gray/blur prediction-drop deltas are `-0.011803/-0.012459`.
This confirms that the candidate is not failing because it attends broadly to
the background.

Manual review of all three contact sheets is still negative for the scientific
hypothesis:

- all four class-1 false negatives remain false negatives despite candidate
  heatmaps moving further inside the fruit;
- two of four class-1 false positives are corrected, one remains class 1, and
  one merely changes to class 2;
- only one of four `3 -> 2` boundary errors is corrected;
- the candidate generally localizes on fruit surface/spot regions, but this
  cleaner localization does not restore the missing class-1 decision region.

The first paired contact sheet was preserved as superseded because its column
title called the keeper fallback map native attention. Commit `4cab65d` fixes
the renderer to disclose fallback provenance. The final provenance-fixed
paired summary SHA is
`4d5133b0b04055db35d161aa237ac274356835558a895d408d393d1300f1ecba`.

## Runtime Finding

Epoch 2 took `1678.7 s`, while epochs 3-5 took `202.9/189.8/183.0 s`.
Non-invasive GPU Engine sampling found an unowned `GeometryDash` GUI process
using about 10-15% GPU during the outlier; once it stopped using the GPU,
training returned to historical speed. The trainer remained in CUDA backward,
P0 at 2550 MHz, without paging, memory spill, thermal throttling, or OOM.

For future autonomous full trains, workers 4/2 remain the measured train/eval
choice, but launch must fail closed or defer when a non-workflow process has
material GPU utilization. Never terminate an unknown GUI process.

## Provenance

- Training launcher commit: `1bf7076`.
- Post-smoke audit infrastructure commit: `4edda29`.
- XAI provenance-label fix commit: `4cab65d`.
- Candidate best checkpoint SHA:
  `fad2980474660e2b53f54b5d48f59a7de68d6534f68ed79d940c412c401cf86e`.
- Candidate history SHA:
  `0049c53319ff584d1bf874a2e753ed98251afe600f525b1548df7fb0a3279221`.
- Independent metrics/predictions SHAs:
  `7664bba074f7bc9468d88d8ce31ee2130df922976226b2f4ee62ea5a9f96ecd4` /
  `ac9ab1207b3f52ea91d682fbc4f42f4fae29ecc57a532369651d5e0bca82350e`.
- Robustness summary SHA:
  `480f3835dc0659fd9f26e56e00d74836ff13b4df44afb4a95ae7b0f2249c8f32`.
- Candidate XAI summary SHA:
  `8270f858edcdd5ba04dfc0475beae42ca5b9bf195c33c31a6e35a3ec875db9e7`.
- Final gate summary/manifest SHAs:
  `adbf785a0c8336ef5f1778d549a225e6b87b4c9103f8852d5811c880e551efc2` /
  `076156347f93b0d7c1fea6349bba525984e31647fa89224c02cb214f4f4fe4f2`.
- Architecture trace has exactly one valid sample for each class `0..4`.
- `test_used=false`, `raw_dataset_modified=false`,
  `deferred_reweight_used=false`, `full_train_authorized=false`, and
  `current_best_command_update_authorized=false`.

The current-best checkpoint and all current-best train/export/test/XAI command
files remain unchanged.
