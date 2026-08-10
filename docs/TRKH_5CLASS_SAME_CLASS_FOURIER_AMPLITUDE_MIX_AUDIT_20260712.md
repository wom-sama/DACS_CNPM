# TRKH Same-Class Fourier Amplitude Mix Audit - 2026-07-12

## Decision

Reject the locked source-distinct same-class full-spectrum Fourier amplitude
mix before trainer integration. Smoke, probe, test, and current-best command
promotion remain closed.

The full no-test readiness gate passed `20/26` checks. The six failures are
decisive label-preservation and class-1 safety failures, not implementation or
coverage failures:

- validation clean-correct retention;
- validation class-1 true-positive retention;
- validation FN-rescue versus TP-break safety;
- validation FP-remove versus FP-create safety;
- validation same-view class-1 F1;
- source-fold direction stability.

## Locked Method

The protocol was written before the full run in
`docs/TRKH_5CLASS_SAME_CLASS_FOURIER_AMPLITUDE_MIX_PROTOCOL_20260712.md`.
It follows the released FACT operation in RGB space before normalization:

`A_mix = 0.5 * A_target + 0.5 * A_peer`

The reconstruction keeps only target phase. The target and peer share the hard
class but must have different original YOLO source stems. Validation targets
pair only with train peers. A fixed focus-boundary cross-class peer is a
destructive control and is never a training candidate.

Primary references:

- FACT paper: https://openaccess.thecvf.com/content/CVPR2021/html/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.html
- FACT code: https://github.com/MediaBrain-SJTU/FACT
- FDA paper: https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html
- FDA code: https://github.com/YanchaoYang/FDA

## Safety And Coverage

Artifact:
`runs/diagnostic_same_class_fourier_amplitude_mix_full_20260712`.

| Check | Result |
|---|---:|
| Keeper SHA-256 | `1f49d577...482677` |
| Train targets / source groups | `9215 / 8064` |
| Validation targets / source groups | `2606 / 2577` |
| Train-validation source overlap | `0` |
| Same-label pair coverage | `11821/11821` |
| Same-source pair count | `0` |
| Locked cross-control coverage | `11821/11821` |
| Validation peer split | train only |
| Test split used | no |
| Model fit / checkpoint written | no / no |
| Raw dataset modified | no |
| Identity max RGB error | `8.34465e-7` |
| Max inverse-FFT imaginary residual | `8.40690e-6` |
| Non-finite output | `0` |
| Same-view clipping fraction | `0.004646` |
| Same-view mean RGB MAE | `0.058431` |
| Object / background RGB MAE | `0.061498 / 0.064948` |

The exact clean FP32 validation pass reproduced keeper macro/class-1 F1
`0.882924855/0.678260863`, class-1 precision/recall
`0.603092790/0.774834454`. This rules out a mismatched model, transform,
normalization, bbox prior, or mask path as the cause of rejection.

## Full Metrics

| Split/view | Macro F1 | Class-1 F1 | Class-1 P | Class-1 R |
|---|---:|---:|---:|---:|
| Train clean | `0.939876` | `0.815444` | `0.700265` | `0.975970` |
| Train same-class Fourier | `0.707090` | `0.383481` | `0.261219` | `0.720887` |
| Train cross-class control | `0.488650` | `0.335414` | `0.216434` | `0.744917` |
| Validation clean | `0.882925` | `0.678261` | `0.603093` | `0.774834` |
| Validation same-class Fourier | `0.688662` | `0.358362` | `0.241379` | `0.695364` |
| Validation cross-class control | `0.478679` | `0.296296` | `0.188748` | `0.688742` |

The same-class restriction is meaningfully less destructive than the
cross-class control, but it is not label-preserving. Validation retains only
`1841/2395 = 0.768685` clean-correct decisions and
`89/117 = 0.760684` clean class-1 true positives.

## Class-1 Transitions

| Split | FN rescue | TP break | FP remove | FP create | Corrections | Harms |
|---|---:|---:|---:|---:|---:|---:|
| Train | `3` | `141` | `93` | `658` | `125` | `2087` |
| Validation | `16` | `28` | `27` | `197` | `65` | `554` |

The probability statistic alone looks encouraging but is misleading:

- train FN/FP mean delta-p1: `+0.011081/-0.032042`, AUROC `0.751906`;
- validation FN/FP mean delta-p1: `+0.038316/-0.033494`, AUROC `0.802176`.

It reflects large confidence contraction and broad decision movement rather
than a recall-safe cue. On validation, every FP removal is accompanied by more
than seven new FP creations, and every four FN rescues cost seven true-positive
breaks. The transformed class-1 F1 falls by `0.319899`.

Only three of five source-grouped train folds have both signed mean deltas in
the declared direction. Folds 2 and 4 have negative class-1-FN mean delta-p1,
so the direction is not source-stable even before considering transition harm.

## Visual Audit

The 12-row validation contact sheet includes high-magnitude FN rescues,
TP breaks, FP removals, FP creations, corrections, and harms. Full-spectrum
mixing produces visible purple/green color casts, duplicated contours, global
halos, crop-edge ringing, and peer-background energy over the target fruit.
These artifacts occur for same-class peers as well as cross-class controls.

The failure is consistent with the task physics. FACT treats amplitude as
mostly domain/style information on PACS, but mango color and surface-frequency
statistics are label evidence. More importantly, two independently cropped
fruits have incompatible spatial layouts. Combining peer amplitude with target
phase therefore does not isolate harmless illumination style.

## Independent Verification

An independent CSV pass, separate from the audit summary, verified:

- exact `9215/2606` unique sample indices;
- `8064/2577` source groups and zero split overlap;
- every same-class label and source-distinct condition;
- every confusion matrix, macro F1, class-1 precision/recall/F1;
- train and validation FN/TP/FP transition counts and delta-p1 AUROC;
- all eight payload sizes and SHA-256 hashes;
- aggregate payload manifest SHA-256
  `7a82f434b31b14ddba435f121d2407a0d6df8d3bd977eb5a43375e004675020e`;
- payload size `17,194,976` bytes and no checkpoint/model/test payload.

Focused compile/tests passed `6/6`; complete pytest passed `820/820`. The two
superseded preflights were removed under an exact 18-file hash manifest
(`6,337,363` bytes; observed free-byte gain `6,356,992`). Retention then passed
over `605` run directories and all `28` object compaction manifests with
`blockers=[]`.

## No-Repeat Boundary

Do not integrate or sweep this exact same-class, source-distinct,
full-spectrum, `lambda=0.5` policy. Do not turn its delta-p1 AUROC into a
router, sample weight, margin, threshold, or consistency target: the direct
transition evidence is strongly unsafe.

Nearby lambda, random-lambda, consistency-weight, probability, peer-policy,
or full-spectrum variants are also closed. A low-frequency/localized method
would be a different mechanism, but it must first explain and eliminate the
observed crop-layout ringing with a new predeclared identity/label-retention
gate; it is not authorized as an immediate parameter sweep.

The keeper, deployment wrappers, and
`docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt` remain unchanged.
