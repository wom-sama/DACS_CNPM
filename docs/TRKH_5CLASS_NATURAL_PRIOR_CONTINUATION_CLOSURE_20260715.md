# TRKH 5-Class Natural-Prior Continuation Closure

Date closed: 2026-07-15

## Decision

Reject the exact two-epoch natural-prior continuation. It reduced some class-1
false positives by suppressing class-1 predictions, not by improving the
precision/recall boundary. Do not sweep sampler power, LR, seed, budget, loss,
augmentation, or continuation length around this result.

No test split was used. Raw data, labels, bboxes, split membership, keeper,
scratch checkpoint, and current-best command files were not modified.

## Locked Experiment

- Source checkpoint: scratch random-init full run, selected at epoch 20.
- Both roles: fresh optimizer/scheduler/scaler/epoch state, deterministic seed
  `42`, full-model update, `2 x 60` train batches, batch `32`, accumulation `2`,
  LR `8e-5`, full validation support `2606`, and final test disabled.
- Control retained strict balanced-epoch sampling.
- Candidate changed exactly one factor: natural random sampling via
  `--disable-balanced-epoch-sampling`.
- Locked manifest SHA-256: `de8df0be...c447a0c`; protocol document SHA-256:
  `5922d0c5...76ec20`.

## Clean Validation

Independent FP32 reloads are authoritative:

| Model | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | Class-1 FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Keeper | 0.882925 | 0.603093 | 0.774834 | 0.678261 | 77 |
| Scratch | 0.874172 | 0.535865 | 0.841060 | 0.654639 | 110 |
| Strict control | 0.875714 | 0.538136 | 0.841060 | 0.656331 | 109 |
| Natural candidate | 0.873448 | 0.537778 | 0.801325 | 0.643617 | 104 |

Candidate minus control was `-0.002266` macro F1, `-0.000358` class-1
precision, `-0.039735` recall versus scratch, and `-0.012714` class-1 F1. Of
16 changed decisions, `7` were corrections, `7` harms, and `2` neutral. It
removed `5` restricted class-1 false positives but rescued no class-1 false
negative and broke `6` class-1 true positives.

The locked comparison failed 13 behavioral checks. Probe, full-train, and test
permissions are all false. Comparison summary SHA-256 is
`49cd235f...d4c74`.

## Robustness And XAI

| Condition | Control class-1 P/R/F1 | Candidate class-1 P/R/F1 | FP control -> candidate |
| --- | --- | --- | ---: |
| Clean | 0.538/0.841/0.656 | 0.538/0.801/0.644 | 109 -> 104 |
| Center occlusion | 0.555/0.768/0.644 | 0.557/0.742/0.636 | 93 -> 89 |
| Dim | 0.419/0.715/0.528 | 0.425/0.709/0.531 | 150 -> 145 |
| Bright | 0.421/0.722/0.532 | 0.426/0.662/0.518 | 150 -> 135 |
| Low contrast | 0.460/0.728/0.564 | 0.459/0.695/0.553 | 129 -> 124 |

The 16-case FP32 XAI reconciliation had zero backend prediction drift. Native
attention and rollout foreground mass were nearly unchanged, while Grad-CAM
shifted toward the fruit in both the five false-positive removals and six
true-positive breaks. The contact sheets therefore support a broad
class-boundary/prior shift, not selective foreground reasoning. Bright and
low-contrast recall losses independently confirm suppression risk. Paired XAI
summary SHA-256 is `58856b49...c8e5f9`.

## Retention

- Compact evidence:
  `runs/evidence_natural_prior_continuation_rejected_20260715`.
- `420` payloads and `54,309,197` bytes retained; payload-manifest SHA-256 is
  `8c223f60...613ea9`.
- `1,174` reproducible binary/per-case files and `576,989,850` bytes excluded.
- Three source roots were hash-verified and deleted; cleanup-manifest SHA-256
  is `7ebc3c91...b8295`.
- Retention passed over `682` run directories with `blockers=[]`; summary
  SHA-256 is `46d28c06...9a896`.

Keeper, scratch complement, current-best command packet, and command-update
history remain unchanged. This result does not add a full-train command
revision.
