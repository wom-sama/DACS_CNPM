# TRKH 5-Class Pixel-Difference Stem Signal A0 Closure - 2026-07-17

## Decision

Status: rejected before trainer/model integration and before any image epoch.

The prospectively locked ICCV-2021 PiDiNet/PDC frozen-stem gate did not expose
a class-1 precision signal. The exact `CD -> AD -> RD` conversion produced
valid and object-boundary-sensitive responses, but its clean object-only OOF
AUROC was `0.499591`, below the same frozen vanilla stem at `0.575141` by
`-0.075550`. It also lost to native under every locked illumination shift and
rejected only `3.7634%` of restricted false positives at the class-1-protecting
threshold.

No PDC trainer option, checkpoint, epoch, holdout, validation, test, full train,
or current-best command update is authorized. The raw data and keeper were not
changed.

## Primary Sources And Lock

Authority was the accepted paper and official source, not secondary reports:

- Su et al., "Pixel Difference Networks for Efficient Edge Detection," ICCV
  2021 (oral):
  `https://openaccess.thecvf.com/content/ICCV2021/html/Su_Pixel_Difference_Networks_for_Efficient_Edge_Detection_ICCV_2021_paper.html`;
- accepted PDF SHA-256: `7ac63751...a4e09`;
- official repository: `https://github.com/hellozhuo/pidinet`;
- locked official commit/tree:
  `d21aa881ed9c628571636fad39acfe1fad517ebd` /
  `b57c16070137773a76d09356df70dd973632887f`.

The official license combines a research-purpose notice with MIT-style grant
text. This audit remains research-only; any future commercial use still needs
separate license review.

Protocol
`TRKH_5CLASS_PIXEL_DIFFERENCE_STEM_SIGNAL_PROTOCOL_20260717.md` has corrected
SHA-256 `af2a2ba8...57adcaa`. Before implementation or measurement, source
review corrected RPDC wording to follow official `weight[:, :, 1:]`: source
indices `1..8` are mapped and index `0` is unused. No cohort, readout, threshold,
gate, or authorization criterion changed. Auditor implementation commit is
`2f7432dfd29dc02d86371fc12eb1753260269285`.

## Locked Cohort And Replay

The sole formal run is
`runs/audit_pixel_difference_stem_signal_a0_20260717`. It used only the 607
predeclared fit rows:

```text
class-1 true positives                 421
restricted {0,2,4}->1 false positives 186
fold 1 TP/FP                           112 / 45
fold 2 TP/FP                           100 / 48
fold 3 TP/FP                           101 / 52
fold 4 TP/FP                           108 / 41
ordered-index SHA                      a2689d1b...c738bd
```

The keeper deployment replay at batch 64 matched 606/607 declarations. Sample
`3657`, target class 2, changed from declaration class 1 to replay class 2. Its
locked declaration probabilities were `p1=0.23123835` and `p2=0.23106094`, a
near-tie margin of only `0.00017741`. A read-only sensitivity calculation from
the already written CSV excluded that one negative without refitting; PDC
clean object AUROC decreased from `0.499591` to `0.497567`. This mismatch is a
strict structural failure but cannot explain or rescue the information result.

The output CSV has all `4,856 = 607 x 4 conditions x 2 roles` rows. Independent
replay reproduced metrics, response comparisons, and every information-gate
decision exactly.

## Equation And Deployment Audit

The core PDC implementation is faithful and active:

```text
official conversion max error           0
independent-oracle conversion max error  5.9605e-8
forward max error                        1.1921e-7
input-gradient max error                 5.9605e-8
weight-gradient max error                1.7881e-7
finite-difference max error              8.5651e-13
BF16 versus FP32 max error               0.004143
```

All FP32/BF16 outputs and gradients were finite, all three operators were
nondegenerate versus vanilla, and the converted stem used only ordinary
convolutions. CUDA batch-64 median runtime was `53.9392 ms` versus native
`45.0652 ms` (`1.196914x`, pass); peak-memory ratio was `1.000932x` (pass).

ONNX opset 17 exported with only standard `Conv/MaxPool/Erf/Add/Mul/Div`
operators and matching shapes. ORT maximum error was `3.33786e-5`, above the
locked `1e-5`, so the strict ONNX gate failed. The information gate remains
rejected even if this numerical export tolerance were ignored.

## Fit-Only Information Result

The fixed four-source-fold OOF logistic readouts produced:

| Role/view | Clean AUROC | Dim AUROC | Bright AUROC | Low-contrast AUROC |
|---|---:|---:|---:|---:|
| Native object | 0.575141 | 0.567121 | 0.555053 | 0.524519 |
| PDC object | 0.499591 | 0.476643 | 0.515414 | 0.486732 |
| Native context | 0.571233 | 0.555015 | 0.562524 | 0.523357 |
| PDC context | 0.507637 | 0.500051 | 0.500651 | 0.496207 |

At the clean-frozen lower-3% TP threshold, PDC object metrics were:

```text
condition       TP retention   restricted-FP rejection
clean             0.971496          0.037634
dim               0.988124          0.005376
bright            0.897862          0.091398
low contrast      0.990499          0.010753
```

PDC failed 12 of 15 information checks: clean/minimum-shift AUROC, clean gain,
shift superiority and worst-shift safety, clean and shifted FP rejection,
bright TP retention, aggregate FP rejection, and TP-over-FP median ordering.
Its three shifted AUROC deltas versus native were `-0.090478`, `-0.039639`, and
`-0.037788`.

All three PDC blocks did pass the bbox-boundary-versus-outside median response
test for both cohorts, and object-only AUROC stayed within `0.02` of context.
This is useful negative evidence: PDC found the fruit boundary, but that
boundary response did not distinguish class-1 TP from class-0/2/4 false
positives.

## Visual Diagnosis

All four contact sheets were reviewed. Low PDC scores commonly show smooth
green fruit in both TP and FP cohorts; high scores commonly show spots,
lesions, clutter, or rough surface in both cohorts. Block-3 energy often traces
the fruit silhouette and high-frequency surroundings while suppressing smooth
interiors. The mechanism therefore measures edge/roughness strength without a
stable class-1 semantic direction.

Per-target clean PDC object medians reinforce this:

```text
target 0 / 1 / 2 / 4 medians  0.504167 / 0.503282 / 0.501998 / 0.510962
object-context score Spearman  0.786716
score-bbox-area Spearman       0.060985
```

The failure is not far-background area dependence; it is nondiscriminative
surface/boundary energy.

## Artifacts And Retention

Formal manifest SHA-256 is `39988fc3...7b0d05`.
The manifest records eight artifacts totaling `8,292,757` bytes:

- cohort CSV SHA `4b26a783...75c34`;
- four contact-sheet SHAs `5010c9de...f1b8`, `a1418fca...196ce`,
  `1a412681...ce49`, and `301b4d43...baa4`;
- converted ONNX SHA `86e8e3b9...09dd`;
- report SHA `c0ca1f2f...8bde`;
- summary SHA `5f5afa2e...c4ee`.

The directory contains no checkpoint, raw-data copy, or engine and is only
about 8.3 MB, so it is retained intact. Current-best command/history hashes
remain `36b9aa1a...9faf` and `39bd2879...8f53`; tracking stays three revisions
and two promoted updates.

Post-PDC retention audit
`runs/audit_trkh_artifact_retention_post_pdc_20260717` passed over 738 run
directories with `blockers=[]`, deleted nothing, touched no raw data, and
reported `70.288 GB` free. Its summary SHA is `68c6e5ec...50c3ae`.

Closure verification passed Python compileall, `pyflakes`, six PowerShell
parses, focused PDC tests `8/8`, and full pytest `1,345/1,345`. The 273 warnings
were existing dependency deprecations and tracing notices, not failures.

## No-Repeat Rule

- Do not integrate or train the frozen-weight `CD -> AD -> RD` stem route.
- Do not rescue it by sweeping theta, operator order, block count/location,
  region geometry, descriptor choice, readout C/solver, threshold, lighting,
  fold, seed, batch size, ONNX tolerance, or a nearby PDC auxiliary loss.
- Do not interpret boundary-over-outside response as class separability.
- Do not waive information failures because equation, runtime, memory, or
  visualization gates passed.
- This closure rejects the exact learned-kernel-compatible frozen PDC signal
  and nearby TRKH stem variants. It does not claim that the separately trained
  full PiDiNet edge detector is generally invalid.
- Select the next route from a distinct accepted primary source whose
  pre-training gate tests class-conditional evidence, not edge magnitude alone.
