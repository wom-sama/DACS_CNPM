# B5 POSF Q/V-LoRA closure — 2026-07-31

Protocol: `TRKH_PRETRAINED_CLASSF_B5_POSF_QV_LORA_R4_20260731`

Source commit: `ce3e61b71b4abd528d3aea03470728654bd63ae4`

Run:
`runs/pretrained_dinov3_classf_b5_posf_qv_lora_r4_probe_20260731_local_b5_posf_probe_locked`

## Decision

B5 is rejected and must not be promoted to full training, deployment audit, or
a nearby POSF sweep. The locked fair-selection rule selected epoch 3.

| Gate | Required | B5 selected | Result |
|---|---:|---:|---|
| Validation support | 2,479 | 2,479 | pass |
| Accuracy | >= 0.8879 | 0.890682 | pass |
| Macro-F1 | >= 0.8495 | 0.851923 | pass |
| Class-1 precision | >= 0.610 | 0.601064 | fail |
| Class-1 recall | >= 0.715848 matched-B4 | 0.715190 | fail |
| Class-1 F1 | >= 0.665 | 0.653179 | fail |
| Class-1 TP | >= 113 matched-B4 | 113 | pass |
| Restricted `0/2/4 -> 1` | <= 69 | 73 | fail |
| `2 -> 1` | <= 39 | 40 | fail |
| `0 -> 1` not worse than B4 | <= 16 | 16 | pass |
| `4 -> 1` not worse than B4 | <= 17 | 17 | pass |

Selected confusion matrix:

```text
[531, 16,  2,  1,  8]
[ 29,113, 11,  1,  4]
[  0, 40,323, 13,  4]
[  0,  2, 19,429, 44]
[ 11, 17, 17, 32,812]
```

The semantic loss was active for all three tasks at every epoch. At the
selected epoch its mean was `0.431787`; the run retained exactly `24,576`
trainable Q/V-LoRA parameters. The launcher transcript contains no skipped
non-finite loss or gradient step, and all 360 tensors inspected across the
selected model, EMA, and optimizer states were finite.

Selected checkpoint SHA-256:
`755c85a0f8df976b37b97ef262c3b87d6585c653e1ce73efcbab2732ffdb251f`.

## Mechanism finding

POSF moved the intended boundary in the correct direction, but not far enough
without losing true class-1 cases:

- versus B4, `2 -> 1` fell `45 -> 40` and restricted false positives fell
  `78 -> 73`;
- class-1 precision rose `0.589744 -> 0.601064`;
- class-1 TP fell `115 -> 113`, mainly through `1 -> 2`, so class-1 recall
  fell `0.727848 -> 0.715190`;
- class-1 F1 improved only `0.651558 -> 0.653179`.

Against the current B2 winner, B5 reduced `2 -> 1` by three and restricted
false positives by four, but lost two class-1 true positives and produced a
slightly lower class-1 F1 (`0.653179` versus `0.653409`). This is useful
negative evidence: partial-order supervision can steer the error type, but
global grouped logits still cannot expose enough local surface evidence to
separate class 1 from class 2.

## Research consequence

- Keep B2 tempered-p0.5 as the current development winner.
- Do not change the POSF weight/specs after observing validation.
- Preserve the Q/V-LoRA and POSF implementations as auditable negative
  controls; do not present either as the final proposed model.
- The next bounded hypothesis should first test, on train-only source-grouped
  folds, a class-conditional DeepSets/MIL patch-distribution specialist over
  frozen DINO patch tokens. It must directly protect class-1 positives while
  suppressing `0/2/4 -> 1`, remain tiny enough for mobile deployment, and
  reach a predeclared readiness gate before any validation run.

No test inference or test metric was used.
