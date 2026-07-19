# TRKH 5-Class Schedule-Free AdamW A0 Closure

Date: 2026-07-20

## Decision

Reject Schedule-Free AdamW before Stage B. Do not rerun or tune its LR,
betas, warmup, `r`, weight-LR power, weight decay, fold, seed, update budget,
or BatchNorm calibration on the current keeper. Do not integrate it into the
trainer, run validation/test, launch a scratch smoke/full train, or update the
current-best command.

The candidate raises class-1 recall by expanding class-1 support. It does not
learn the required precision-selective TP-versus-restricted-FP boundary, and
the failure worsens under illumination shifts.

## Locked Evidence

- Formal source commit: `6c9bafc3e4dfd7d1ac150636c1d108fecd3cd7c8`.
- Protocol SHA-256: `455d285dfd4ec559629a353d7621b8991579be8d33c0c005cf99e74b990fe499`.
- Official Meta source commit/tree: `d24878d...9d93c4` / `3aff6c4c...d8782f`.
- Keeper SHA-256: `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Train-only split: 7,372 fit rows and 1,843 source-disjoint fold-0 holdout
  rows; validation predictions and test data were unused.
- Both roles consumed 60 byte-identical augmented batches of 32 rows from
  identical model state. All parameter groups moved; gradients and optimizer
  state were finite.
- Control/candidate PreciseBN each consumed the same 50 clean fit batches
  (1,600 rows) without parameter updates.

## Results

| Condition | Macro F1 control -> candidate | Class-1 P / R / F1 control -> candidate | Restricted FP control -> candidate | Corrections / harms |
| --- | --- | --- | --- | --- |
| Clean | `0.813044 -> 0.837111` | `0.513158/0.357798/0.421622 -> 0.533333/0.513761/0.523364` | `37 -> 49` | `24 / 12` |
| Dim | `0.733371 -> 0.741827` | `0.282609/0.119266/0.167742 -> 0.250000/0.155963/0.192090` | `33 -> 50` | `24 / 21` |
| Bright | `0.731656 -> 0.749686` | `0.350877/0.183486/0.240964 -> 0.373737/0.339450/0.355769` | `36 -> 59` | `23 / 34` |
| Low contrast | `0.634011 -> 0.614827` | `0.129032/0.110092/0.118812 -> 0.131783/0.155963/0.142857` | `76 -> 104` | `25 / 75` |

Clean class-1 F1 improves by `+0.101743`, but the mechanism is recall expansion:
17 false negatives are rescued while restricted false positives increase by
12. Shifted restricted false positives increase by 17, 23, and 28. Dim
class-1 precision falls by `-0.032609`; low-contrast macro F1 falls by
`-0.019183`, true-positive retention is `0.916667`, and maximum nonfocus F1
drop reaches `0.076064`.

Candidate/control median step-time ratio is `0.991206`; both peak at
`3.917105 GiB`. Schedule-Free is computationally viable, but resource parity
cannot override the eight independent metric/robustness failures.

## Audit Erratum

The immutable formal summary also reports two audit-harness defects:

1. The v1 implementation required bit-exact full-model `train/eval/train`
   round trips even though the protocol declared FP32 tolerance. Observed
   `y/x` maximum errors are only `3.7252903e-9` and `7.2759576e-12`, both below
   one FP32 epsilon (`1.1920929e-7`). The post-formal harness now applies that
   declared tolerance while retaining the v1 artifact key names.
2. The shared exporter received CPU inputs while a mutable alias left the
   candidate model on CUDA. ONNX therefore was not measured. The exporter now
   normalizes the model to CPU before export.

These fixes are covered by focused tests and do not authorize a rerun. Even if
all six affected structural/export booleans were treated as passing, the clean
restricted-FP gate and seven illumination gates still reject the method. The
formal summary, replay, and artifact manifest remain byte-unchanged as the
record of what ran.

## Replay And Artifacts

- Independent replay recomputed all 7,372 condition rows, metrics,
  transitions, and original gates with maximum absolute difference `0`.
- Summary SHA-256: `4c2c71f88d6f1749edda7ddfb8ef53845ab8100c307e799d703f4b28294d330a`.
- Replay SHA-256: `2a543eaf413c3060f608ddc123836af146b8ef89a8d01d0a11f1ddcd4d0b2f67`.
- Artifact-manifest SHA-256: `8f3360e3aed432f189da609b1fd2cb296b042c6880efb285182da4c2d62f225e`.
- Retained run payload: 4,344,951 bytes; no checkpoint, ONNX model, engine,
  validation prediction, or test artifact exists.
- XAI was correctly not generated because the prospective protocol made it
  conditional on all automatic gates passing. Therefore no visual-review
  claim is made.

## Consequences

Close nearby Schedule-Free optimizer rescues on this keeper. The useful lesson
is narrow: its evaluation iterate can move the model quickly toward class 1 at
no measured speed or memory cost, but that movement is not class-conditional
enough for the agricultural precision requirement. A future method must add a
new morphology/surface signal that separates true class 1 from classes 0, 2,
and 4 under lighting shifts; another optimizer-only or global ranking change
is not justified.

Current-best command/history stay at three revisions and two actual updates,
with SHA-256 `36b9aa1a...940faf` / `39bd2879...98f53`.
