# B8 frozen-DINO risk-controlled `2 -> 1` closure — 2026-08-01

Protocol: `TRKH_PRETRAINED_CLASSF_B8_FROZEN_DINO_NP_2TO1_20260801`

Canonical run: `runs/precheck_classf_b8_frozen_dino_np_2to1_20260801_run4_controlfix`

Source: `b7bb468`

## Decision

B8 is rejected at its locked train-only readiness gate. It must not be
promoted to model integration, validation, smoke/full training, test, or
mobile deployment. The known result must not be rescued by changing the
threshold, risk tolerance, variance floor, descriptor, score direction, class
pair, or control construction.

The canonical run contains all 8,278 train crops and 7,751 normalized source
groups in five source-disjoint outer folds. It uses the original, strictly
loaded and fully frozen DINOv3-S/16 checkpoint
`2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`.
The descriptor cache has shape `[8278, 1152]`, FP32, and SHA-256
`f9a7b8ea8d89792bb600da609830f7e44c303cc753e988420323369eb858003d`.
Validation and test were neither constructed nor read.

| Locked gate | Required | Observed | Result |
|---|---:|---:|---|
| Worst-fold class-1 recall delta | >= -0.005 | -0.010638 | fail |
| Aggregate class-1 TP retention | >= 0.990 | 0.997041 | pass |
| Corrected B2 `2 -> 1` errors | >= 14/137 | 0/137 | fail |
| Positive-effect folds | >= 4/5 | 0/5 | fail |
| Reduction above zero control | >= 0.050 | 0.000000 | fail |
| Reduction above corrected deranged control | >= 0.050 | -0.014599 | fail |
| Folds beating zero control | >= 4/5 | 0/5 | fail |
| Folds beating corrected deranged control | >= 4/5 | 0/5 | fail |
| New `0 -> 1` or `4 -> 1` errors | 0 | 0 | pass |
| Changes outside exact `1 -> 2` domain | 0 | 0 | pass |

Aligned B8 acted once: it changed the correctly classified class-1 crop
`Image_10370_box000.jpg` to class 2. Thus class-1 true positives fell from 338
to 337 and no actual `2 -> 1` error was corrected. The zero control made no
change. The donor-aware, source-disjoint deranged control corrected two of 137
errors without losing a class-1 true positive. The aligned rule therefore did
not demonstrate sample-specific value beyond either control.

## Control correction and canonical evidence

Run 3 established the same negative aligned result exactly, but its deranged
control allowed donor descriptors from cross-label-excluded class-2 sources to
re-enter the Gaussian fit. That control is invalid and is retained only as
debugging provenance. Commit `b7bb468` separated valid and excluded class-2
donor roles and added a fail-closed role-crossing check. Run 4 replayed the
locked computation from the byte-identical descriptor cache; every derangement
reports zero fixed points, same-source assignments, label mismatches, fold
crossings, and class-2 exclusion-role crossings. Run 4 is the sole canonical
B8 result.

## Mechanism finding

The frozen DINO descriptor can separate ordinary class-2 and class-1 crops
moderately, but the 125 route-eligible B2 class-2 errors occupy the atypical,
class-1-like tail. Every one was below its predeclared safe threshold; the
closest fold margins were still negative (`-0.15265`, `-0.19148`, `-0.07744`,
`-0.04291`, `-0.10767`). The only aligned exceedance was a visually ambiguous
yellow-green class-1 crop that resembles the nearest class-2 errors.

The limiting factor is therefore not simply missing global pretrained
features. It is overlap in the observed phenotype/label boundary: a one-class
prototype of typical class 2 cannot recover atypical class-2 examples without
also consuming class-1 true positives. This closes the exact frozen-DINO
selective-router family and argues against more post-hoc threshold sweeps.

## Research consequence

B2 tempered-p0.5 remains only the best observed canonical development probe,
not a protocol-promoted full-train candidate: its class-1 F1 was `0.653409`,
below the locked `0.66` gate. The next model run must preserve that failed gate
in its provenance. A full B2-shaped run may be created only under a separately
predeclared exploratory reference-completion protocol; it cannot retroactively
validate B2. The already-authorized B0 full run remains the matched pretrained
control. Final test stays locked until an explicit post-training audit permits
it.

Independent post-validation artifact:
`runs/precheck_classf_b8_frozen_dino_np_2to1_20260801_run4_controlfix/postvalidation.json`
(SHA-256 `518b2fe3ad667656a1bfb546d0f2f5a85fe285bbd239dc73bfce6e2626580d06`).
