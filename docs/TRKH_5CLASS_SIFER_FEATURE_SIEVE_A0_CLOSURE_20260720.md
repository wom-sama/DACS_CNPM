# TRKH 5-Class SIFER Feature-Sieve A0 Closure - 2026-07-20

## Decision

Reject the locked SIFER stem feature-sieve route on the current keeper before
trainer integration, a two-epoch smoke, validation, test, full training, or a
current-best command update.

SIFER produces a small clean holdout gain over matched ordinary adaptation,
but it does not deliver the requested precision-safe class-1 behavior. Clean
class-1 F1 improves by `+0.013179`, while precision improves by only
`+0.004153`, no net restricted false positive is removed, and the effect
reverses under dim, bright, and low-contrast perturbations. The official
forgetting mechanism succeeds on only five of twelve forget steps. All four
auxiliary CAM sheets show nearly unchanged generic fruit response rather than
a new class-conditional maturity/surface signal.

## Authority and execution

- Accepted source: Tiwari and Shenoy, *SIFER: Robust Multi-label
  Classification with Feature Sieve*, ICML 2023.
- Official Apache-2.0 Google Research commit/tree:
  `76b0612b5b48acfa53eba6b433e98764805286f4` /
  `890e37473fb42a9272e6b5478c23e70e6d3c0a57`.
- Prospective protocol SHA-256:
  `90588283fd200ad0f39b2622433091dd4db0c900d77d479eab2d7c8292c235bc`.
- Implementation commit: `b732d72add5c6a553b1459b3ccfac7ed7bf21c06`.
- Replay erratum commit: `93eca1ce9deaff5e4fd695f010fd66010c8ac2bb`.
- Formal evidence:
  `runs/audit_sifer_feature_sieve_a0_20260719`.
- Immutable pre-review summary SHA-256:
  `883a87d57eecbfa730cce4183e738e5b7ffa7c52a824890feb95463e958ce65d`.
- Final summary/manifest/replay SHAs:
  `4b23302a6e15e88a4909f1f33356fbe282caafc8296be59fe8dc3853110953db` /
  `4a1519a2015fba59b80c20615adc5848198b267b73f7391ba955fec7ff734613` /
  `784c9e1eb77eb1e568b75411ea0fb091b1d3cd5f3227db05b779e2d8016b1f51`.

Three launcher attempts stopped before model/data access because the locked
three-sample GPU-isolation gate observed background display utilization above
10 percent. No formal directory existed after those stops. The sole completed
formal run started only after the Codex window was temporarily hidden,
Wallpaper Engine was paused through its official control command, the display
queue settled for five seconds, and the launcher observed `0/0/0%`. Both were
restored without terminating a user process.

## Locked experiment

- Dataset access: `yolo_f/train` only; validation and test were forbidden.
- Holdout: all 1,843 source-disjoint fold-0 rows.
- Fit stream: 60 matched logical updates, batch 32, identical train order and
  role-reset main-forward RNG between control and candidate.
- Control: ordinary keeper adaptation at AdamW LR `1e-5`.
- Candidate: the same adaptation plus the official two-block width-256
  auxiliary decoder at the `256x32x32` stem output.
- Identify: auxiliary SGD LR `1e-2` on detached stem features every step.
- Forget: stem-only SGD LR `1e-4` toward uniform auxiliary targets at steps
  `0,5,...,55`.
- All main parameter groups received finite gradients and moved. Auxiliary
  initialization preserved the pre-existing CPU and CUDA RNG states exactly.

## Clean result

| Variant | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | Restricted FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| matched control | 0.907520 | 0.821429 | 0.633028 | 0.715026 | 15 |
| SIFER candidate | 0.910614 | 0.825581 | 0.651376 | 0.728205 | 15 |
| delta | +0.003095 | +0.004153 | +0.018349 | +0.013179 | 0 |

Only four clean decisions change: three corrections and one harm. SIFER
rescues two class-1 false negatives and breaks no clean class-1 true positive,
but one restricted false positive is removed while another is created. The
required precision delta was `>=0.005` and the required restricted-FP
reduction was at least two; both gates fail.

## Illumination result

| Condition | Class-1 P delta | Class-1 R delta | Class-1 F1 delta | Restricted FP control -> candidate | TP rescues / breaks |
| --- | ---: | ---: | ---: | ---: | ---: |
| dim | -0.012653 | 0.000000 | -0.002468 | 18 -> 19 | 0 / 0 |
| bright | -0.032132 | 0.000000 | -0.006005 | 10 -> 12 | 0 / 0 |
| low contrast | -0.000337 | -0.018349 | -0.016386 | 19 -> 18 | 0 / 2 |

Precision declines in all three conditions. Dim and bright create one and two
net restricted false positives. Low contrast removes one false positive only
by breaking two true class-1 positives and crosses both the locked recall and
F1 safety bounds. The clean gain is therefore not robust.

## Mechanism and visual review

- Identify loss decreases on `12/12` designated steps.
- Uniform-target CE is non-increasing on only `5/12` forget steps.
- Auxiliary entropy is non-decreasing on only `5/12` forget steps.
- Stem mean absolute difference is `0.004924196`; mean absolute prediction
  difference is only `0.000488500`.

All four contact sheets were inspected and finalized `fail` from the immutable
pre-review summary. Clean control/candidate CAMs remain nearly identical broad
fruit and peel responses. Dim maps continue to follow silhouettes, edges, and
background. Bright maps create two class-2 to class-1 restricted false
positives on yellow/spotted fruit without a meaningful CAM change. Low-
contrast maps remain generic foreground responses while two true class-1
positives are broken. There is no repeatable evidence that the stem forgot a
spurious cue while preserving the real maturity/surface boundary.

## Replay erratum

The first independent replay stopped after formal completion because the
formal tensor-wise FP32 batch reduction and the mathematically equivalent mean
of serialized per-row FP32 means differed by
`2.008685405507915e-11`. Both map to the same FP32 value; the discrepancy is
only `0.0431` of the `4.656612873077393e-10` FP32 ULP at that magnitude.

The pushed erratum permits `5e-10` only for
`mechanism.stem_mean_absolute_difference`. Every other numeric comparison
remains at `1e-12`, and paths, hashes, rows, decisions, booleans, and sequences
remain exact. Regression tests require a `6e-10` stem mismatch and a `2e-12`
non-stem mismatch to fail. Replay then reproduced all 1,843 clean rows, 5,529
illumination rows, nine automatic failures, the forget sequence, contact-sheet
hashes, and all summary artifact bindings.

## Resource and deployment result

- Median control/candidate update times are `0.305109/0.462728` seconds;
  candidate ratio `1.516599x` passes the locked `1.60x` ceiling but is close to
  it.
- Control/candidate peak allocations are `3.917104/3.961097 GiB`.
- The auxiliary decoder is external and discarded. Production inference
  parameter count and state schema are unchanged.
- Static ONNX opset-17 export passes with exact argmax and maximum absolute
  error `4.7683716e-7`; the 30,535,555-byte temporary ONNX was deleted after
  verification.

Correct deployment behavior does not override the precision, mechanism,
illumination, and visual failures.

## Artifacts and no-repeat rule

Preserve the finalized 12-artifact payload totaling `8,025,440` bytes plus its
artifact manifest. It contains the immutable CSV/history inputs, four contact
sheets, review manifest, independent replay, report, and final summary. No
checkpoint, ONNX, engine, validation prediction, test artifact, or trainable
manifest is retained.

Do not sweep the current SIFER stem attachment, auxiliary width/depth,
identify/forget interval, main/auxiliary/forget LR, optimizer, uniform target,
fold, seed, budget, class mask, or nearby RSC/feature-forgetting variants on
this keeper. The failure is not missing capacity or export support: the
candidate changes the stem but does not produce precision-safe, illumination-
stable class-conditional information.

A future route must be equation-distinct and first prove a train-only signal
that separates class-1 true positives from class `0/2/4` false positives under
clean, dim, bright, and low-contrast conditions. It may proceed to full train
only after a prospectively locked smoke/probe plus complete audit/XAI pass.

## Closure verification

- Focused pytest `13/13`, full pytest `1468/1468`, compileall, pyflakes, four
  PowerShell AST checks, official replay, and staged diff checks pass.
- Read-only retention passes over 764 run directories and all 48 compaction
  manifests. All 210 compacted originals remain absent, all 14 protected
  checks pass, `blockers=[]`, nothing was deleted by retention, and free space
  is `102.62 GiB`. Retention summary SHA-256 is
  `e42037f861dacee3a376fafb59ae890d200d80f5a18168c8d3e20011defd4379`.
- A failed 9.8 MiB sparse source clone and an empty 26 KiB large-LR repository
  were removed from `%TEMP%` after resolved-path guards. The official SIFER
  bare repository and paper remain available.
- Keeper/current-command/history SHAs remain
  `1f49d577...2677` / `36b9aa1a...0faf` / `39bd2879...8f53`.
  Current-best commands remain three revisions and two actual updates.
