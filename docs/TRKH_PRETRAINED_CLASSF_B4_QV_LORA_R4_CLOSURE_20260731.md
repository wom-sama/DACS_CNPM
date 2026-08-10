# B4 Q/V-LoRA rank-4 closure — 2026-07-31

Protocol: `TRKH_PRETRAINED_CLASSF_B4_QV_LORA_R4_20260731`

Source commit: `ba891380bbb8947a7db8e00ec1185e2afbed9dea`

Run:
`runs/pretrained_dinov3_classf_b4_qv_lora_r4_probe_20260731_local_b4_qv_lora_r4_probe_locked`

## Decision

B4 is rejected and must not be promoted to full training. The inherited
`fair_macro_f1_min_class_gap_penalty` rule selected epoch 1. It passed the
global accuracy/macro-F1 and class-1 recall/TP checks, but failed every
predeclared class-1 precision/error check:

| Gate | Required | B4 selected | Result |
|---|---:|---:|---|
| Accuracy | >= 0.8879 | 0.889068 | pass |
| Macro-F1 | >= 0.8495 | 0.850161 | pass |
| Class-1 precision | >= 0.610 | 0.589744 | fail |
| Class-1 recall | >= 0.708 | 0.727848 | pass |
| Class-1 TP | >= 112 | 115 | pass |
| Class-1 F1 | >= 0.665 | 0.651558 | fail |
| Restricted `0/2/4 -> 1` | <= 69 | 78 | fail |
| `2 -> 1` | <= 39 | 45 | fail |

Selected confusion matrix:

```text
[531, 16,  2,  1,  8]
[ 30,115,  8,  1,  4]
[  0, 45,319, 12,  4]
[  0,  2, 21,430, 41]
[ 11, 17, 19, 33,809]
```

Epoch 3 reached a higher raw macro-F1 (`0.851243`) but was not selected by the
locked rule and also had class-1 F1 only `0.651297`. It is not a rescue
checkpoint.

## Mechanism finding

Against the matched B2 FP32 validation export, B4 changed only 12/2,479
argmax predictions: eight corrections and three harms. Six samples moved
*into* class 1 and none moved out; two new `class 2 -> class 1` errors were
added. Mean absolute class-1 probability movement was only `0.000487`
(maximum `0.001879`).

The selected merged Q/V update had aggregate Frobenius norm only about
`0.001021` of its Q/V base weights. This is not a broken-gradient result:
preflight proved zero-init parity, non-zero first-step up gradients and
non-zero down gradients after the first update. The adapter learned a real
but very small global-attention perturbation, and its direction slightly
expanded the class-1 region instead of removing the dominant hard negatives.

This result agrees with B3/B3b: patch evidence contains a weak local signal,
but a generic static readout or a generic Q/V adaptation does not impose the
precision/recall constraint needed by class 1.

## Research consequence

- Keep B2 tempered-p0.5 as the current development winner.
- Do not sweep LoRA rank, alpha, layers, LR or epoch selection.
- Do not spend deployment-audit time on a rejected candidate.
- The next hypothesis must directly optimize the high-score class-1 negative
  tail while preserving low-margin class-1 positives, and must add no
  inference cost if possible.
- Validation has now supported several exploratory choices; any later winner
  remains exploratory until the sealed test or a new prospective holdout is
  opened once under a final protocol.

No test inference or test metric was used.
