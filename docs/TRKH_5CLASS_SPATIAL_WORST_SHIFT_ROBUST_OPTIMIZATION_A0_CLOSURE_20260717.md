# TRKH 5-Class Spatial Worst-Shift Robust Optimization A0 Closure - 2026-07-17

## Decision

Reject label-aware spatial worst-shift robust optimization on the current
keeper before XAI, validation, test, a five-epoch pair, trainer integration,
full training, or a current-best command update.

The official worst-transform signal is active and deployment-neutral, but the
candidate changes only one clean decision relative to the matched random-shift
control. That change is beneficial, yet it is far below the locked advancement
margin and does not survive official-worst or illumination gates. Both adapted
models also trade too much class-1 recall for precision relative to the
immutable keeper.

No checkpoint is retained. Current-best commands remain unchanged.

## Locked authority and execution

- Accepted source: Engstrom et al., *Exploring the Landscape of Spatial
  Robustness*, ICML 2019, PMLR 97.
- Paper-linked official MIT repository commit/tree:
  `a1c9e364d4179d410209ba3a3b06fdcc73ad2ff9` /
  `60fe12b9c388b00d00068e07c2e93d134f126cf3`.
- Protocol SHA-256:
  `eec9e20c1dd11100504a647bdb90e09b24f9f553fb5bf3c7319e1d8f93ca0a54`.
- Pushed implementation commit: `d67f306421367c271f4057b94e441bc2a67f8a79`.
- Formal evidence:
  `runs/audit_spatial_worst_shift_robust_optimization_a0_20260717`.
- Formal summary SHA-256:
  `0e9c80823f9d90b939a1525336867c74e938af65b43aba7fabf516c4eb7af984`.
- Artifact manifest SHA-256:
  `fe65a9241f1dd6f5e4d0c325922738636ad830a20f134b3b56cf4eeccc7e3a56`.

Clean preflight passed every paper/source/license/input/cohort/implementation
hash and created no output. A one-row CPU integration smoke then verified the
real `yolo_f/train` decode, metadata, nine translated deployment forwards, and
finite `(9,1,5)` probabilities without optimizer work or an artifact.

Formal began with no unrelated Python/TensorRT process and GPU at `1%`,
`1845/8188 MiB`, `57 C`. A known Geometry Dash window was minimized and then
closed through `CloseMainWindow` before formal because it held the GPU at
`22-39%` and about `2772 MiB`; no unknown process was force-terminated.

## Cohort, selector, and optimization integrity

- Only `yolo_f/train` was constructed: 9,215 object samples from 8,064 images,
  with zero missing images, invalid boxes, or invalid classes. Validation and
  test were not constructed.
- Whole-source update/probe/holdout cohorts contain `1024/256/1843` rows with
  zero source overlap. Ordered-index SHAs are the locked
  `2279a58a...72bb7`, `6490523a...c5bd`, and `a628686b...e97ae`.
- The probe gate passed. Official selection used all nine transforms, selected
  a non-clean transform for `239/256 = 0.933594` rows, and exposed loss-span
  signal on all 16 clean-correct class-1 rows plus nine restricted false
  positives. Repeated outputs, translated bytes, and independent selector
  indices were exact.
- Both variants started from state SHA `8d718a18...6a46`, replayed identical
  batch bytes and logical RNG, and performed exactly 32 AdamW updates, 288
  no-grad selection forwards, and 32 gradient forwards.
- Every parameter group received finite nonzero gradients and moved. Random
  and worst-shift selection differed on `905/1024 = 0.883789` update rows, so
  the negative result is not an inactive-selector failure.
- Random/worst wall times are `57.209771/55.654201 s`, ratio `0.972809`.
  Both peak at `3.904689 GiB <= 4.5 GiB`.

## Clean train-holdout behavior

On all 1,843 source-disjoint fold-0 train rows:

| Variant | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | TP | Restricted FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| immutable keeper | 0.949323 | 0.748252 | 0.981651 | 0.849206 | 107 | 36 |
| matched random shift | 0.944997 | 0.846847 | 0.862385 | 0.854545 | 94 | 17 |
| official worst shift | 0.946049 | 0.854545 | 0.862385 | 0.858447 | 94 | 16 |

Relative to the matched random control, worst-shift optimization changes one
clean decision: one restricted class-1 false positive becomes correct, no
decision is harmed, precision rises `+0.007699`, class-1 F1 rises `+0.003902`,
macro F1 rises `+0.001053`, TP remains 94, and restricted FP changes `17 ->
16`. These are positive but below the locked `+0.010` precision, `+0.005`
class-1 F1, and two-FP gates.

Relative to the keeper, the candidate removes 20 restricted FP and correctly
repairs 19, but breaks 13 class-1 TP. Corrections/harms are `20/28`; macro F1
falls `-0.003273` despite class-1 precision/F1 gains. This is support
contraction, not a precision-safe boundary improvement.

The matched random control independently shows the same broad adaptation
effect: much higher precision but 13 fewer TP than the keeper. It has no
locked validation authority and cannot be promoted as a candidate.

## Worst-shift and illumination behavior

Under each model's target-aware official worst transform on clean photometry:

| Variant | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | TP | Restricted FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| keeper | 0.938887 | 0.701987 | 0.972477 | 0.815385 | 106 | 45 |
| random shift | 0.934703 | 0.819820 | 0.834862 | 0.827273 | 91 | 20 |
| worst shift | 0.933834 | 0.818182 | 0.825688 | 0.821918 | 90 | 20 |

The candidate now changes two decisions relative to random, with one
correction and one harm. It breaks one class-1 TP, does not reduce restricted
FP, and loses class-1 and macro F1. All-nine shift consistency improves
slightly (`keeper/random/candidate = 0.986978/0.987520/0.988063`), but this
cannot override worse official-worst classification.

Illumination confirms the recall problem:

| Condition/mode | Random P/R/F1, TP, restricted FP | Candidate P/R/F1, TP, restricted FP |
| --- | --- | --- |
| dim clean | 0.6393 / 0.3578 / 0.4588, 39, 22 | 0.6333 / 0.3486 / 0.4497, 38, 22 |
| dim worst | 0.5246 / 0.2936 / 0.3765, 32, 29 | 0.5254 / 0.2844 / 0.3690, 31, 28 |
| bright clean | 0.7059 / 0.4404 / 0.5424, 48, 19 | 0.7188 / 0.4220 / 0.5318, 46, 17 |
| bright worst | 0.6087 / 0.3853 / 0.4719, 42, 26 | 0.6176 / 0.3853 / 0.4746, 42, 25 |
| low-contrast clean | 0.7000 / 0.4495 / 0.5475, 49, 21 | 0.7042 / 0.4587 / 0.5556, 50, 21 |
| low-contrast worst | 0.6087 / 0.3853 / 0.4719, 42, 27 | 0.6029 / 0.3761 / 0.4633, 41, 27 |

Bright precision/FP and low-contrast clean metrics contain useful local gains,
but dim and official-worst safety fail. The candidate does not strictly improve
precision and F1 in two of three lighting conditions.

## Replay, deployment, and artifacts

- Independent replay covers exactly `199,044 = 4 x 3 x 9 x 1843` prediction
  rows. Row count, official indices, mean-probability diagnostic predictions,
  summaries, comparisons, and gates replay with maximum numeric error `0`.
- Standard inference runtime passes: keeper/candidate batch-32 medians are
  `92.5042/91.8081 ms`, ratio `0.992474`.
- Peak inference allocation is `0.620945/0.647891 GiB`, ratio `1.043394 >
  1.02`. The graph/schema/parameter count are unchanged; different dynamic
  activation/token decisions still make this a measured resource failure.
- Static batch-1 standard-domain ONNX passes with exact argmax and maximum
  error `1.49e-7`. The 30,535,555-byte temporary ONNX was deleted.
- No checkpoint, ONNX, engine, XAI image, validation, test, or trainable
  manifest is retained. The compact no-repeat payload has six evidence files
  totaling `40,406,579` bytes; prediction/selection/history SHAs are
  `e8a31037...74ef7`, `4fc4f0e6...27da8`, and `eea7df06...8cdf5`.

## No-repeat rule

Do not rerun or sweep the one-pixel worst-shift family on this keeper through
shift size/order/padding, random-control distribution, update count, LR,
optimizer, weight decay, seed, batch size, fold assignment, class weights,
clean-loss blending, CE/KL/JSD combinations, thresholding, TTA aggregation,
class-1 oversampling, or relaxed metric/resource gates. Do not combine it with
closed LPD/APS/BlurPool/SPT, consistency, AugMix, routing, DART, or post-hoc
filtering routes.

The transferable finding is narrower: low-LR train-only adaptation can remove
many class-1 FP, but current methods contract positive support under dim and
bright conditions. A future route must be equation-distinct and prospectively
protect class-1 TP/illumination recall while retaining the observed FP gains.

## Closure verification

- Compileall, pyflakes, PowerShell AST, focused `10/10`, integration `46/46`,
  and full pytest `1413/1413` pass.
- Read-only retention passes over 753 run directories and all 48 object
  compaction manifests. All 210 compacted originals remain absent,
  `blockers=[]`, no deletion/raw-data touch occurred, and free space is
  `102.994 GiB`. Retention summary SHA-256 is
  `852ba3828286a3d683ca639332e2c3e30cd3b33d2fafcdac33cacb9d1a34ca22`.
- Current-best command/history SHAs remain
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`,
  still three revisions and two actual updates.
