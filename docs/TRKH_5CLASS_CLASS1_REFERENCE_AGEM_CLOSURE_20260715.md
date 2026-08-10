# TRKH 5-Class Class-1-Reference A-GEM Closure

Date: 2026-07-15  
Decision: reject the locked one-reference A-GEM macrostep; Stage B denied

## Scope and provenance

- The immutable protocol is
  `docs/TRKH_5CLASS_CLASS1_REFERENCE_AGEM_READINESS_PROTOCOL_20260715.md`,
  SHA-256
  `4bec6a835ce8a5bfb7ebe6a31ca533cf596906fe60279903e05198d0c9010968`.
- Primary sources were the ICLR-2019 A-GEM paper
  (`https://arxiv.org/abs/1812.00420`) and official repository commit
  `45421499483b28935491251e9e821c55e8b3c089`.
- Preflight verified the official source and license, keeper, launcher args,
  data YAML, CIDT summary/predictions, paper, and protocol hashes. The official
  worktree was clean.
- Only `yolo_f/train` was used. Source-disjoint fit/holdout support was
  `7372/1843`, with zero source overlap. Validation and test were never
  constructed.
- The current task contained all `186` fit restricted `0/2/4 -> 1` false
  positives. The reference memory contained all `432` fit class-1 rows.

## Equation and update audit

- The hard-negative and class-1 reference gradients were strongly conflicting:
  dot product `-73.433973`, cosine `-0.556244`.
- Exact A-GEM projection was active and retained `0.831019` of the current
  gradient norm. The flattened Equation-11 comparison error was
  `2.913459e-8`.
- After required FP32 materialization, projected dot/cosine were
  `-6.046462e-8/-5.511366e-10`. The tiny negative dot missed the locked
  absolute `-1e-8` gate, but is numerically negligible relative to the raw dot
  and cannot reverse the behavioral rejection.
- Control/candidate actual parameter-step ratios were
  `1.0000000898e-4/1.0000000214e-4`; their actual norm mismatch was
  `6.839169e-8` relative. No AdamW, optimizer state, scheduler, clipping,
  decay, augmentation, teacher, or binary model artifact was used.

## Clean fold-0 result

| Variant | Macro F1 | Class-1 F1 | Precision | Recall | Predicted class-1 |
|---|---:|---:|---:|---:|---:|
| raw keeper | 0.949323 | 0.849206 | 0.748252 | 0.981651 | 143 |
| unprojected control | 0.948563 | 0.845815 | 0.813559 | 0.880734 | 118 |
| A-GEM candidate | 0.947871 | 0.848739 | 0.782946 | 0.926606 | 129 |

- Candidate versus raw changed `26` decisions, made `12/14`
  corrections/harms, removed `8` restricted false positives, rescued zero
  class-1 false negatives, and broke `6` true positives.
- Precision improved `+0.034694`, but recall fell `-0.055046`; class-1 and
  macro F1 changed `-0.000467/-0.001452`. Predicted class-1 support fell
  `143 -> 129`, below the locked 95% floor.
- Projection materially protected class 1 relative to the raw gradient:
  control/candidate recall was `0.880734/0.926606`, TP breaks were `11/6`, and
  class-1 F1 improved `+0.002925` candidate versus control. A single average
  class-1 CE constraint is therefore active but insufficient at the finite
  normalized step.

## Illumination result

- Dim improved macro/class-1 F1 `+0.009208/+0.023095`, but created two net
  restricted false positives.
- Bright improved macro/class-1 F1 `+0.012324/+0.027058` and removed 26
  restricted false positives, but class-1 recall fell `-0.073394` with nine
  TP breaks and one rescue.
- Low contrast lost macro/class-1 F1 `-0.014087/-0.044479`, precision/recall
  `-0.050858/-0.036697`, and created seven net restricted false positives.
- Aggregate illumination TP breaks exceeded rescues `15 > 7`. The mechanism
  is not robust enough for an agricultural deployment gate.

## Independent verification and artifacts

- Independent CSV replay found exactly `1843` unique rows in each of clean,
  dim, bright, and low contrast, and reproduced all clean metrics and
  transitions exactly.
- Raw clean predictions matched the immutable CIDT keeper argmax on
  `1843/1843` rows.
- Artifact manifest SHA-256 is
  `417c3585604ed65f747e0c2265ef28acdb5b1a26810844f367780219f8feee75`.
- Summary SHA-256 is
  `a74deb9b4d235ced3c860ec00d21e65baa5efdecbdf6841b67552c0ecfae1732`.
- Prediction and gradient-group SHAs are
  `bc31de906c4f179487c5104a7d42710bda2a91be88e137aaef53f86a4af1d1d2`
  and
  `1360b0db5f69cdb508da68206e7924dc104ffe6808b3eaa28af56fc29d3b5630`.
- Five nonbinary payloads total `2,962,864` bytes. No checkpoint, ONNX,
  TensorRT engine, test prediction, or raw-data write exists.

## Stop rule and next distinct screen

Do not sweep the A-GEM step ratio, fold, seed, cohort, objective, optimizer,
parameter subset, batch size, or number of macrosteps. Do not reinterpret the
precision increase as a win because it came with keeper TP/support loss and
low-contrast failure. Stage B, validation, test, probe, full train, and a
current-best command revision are denied.

The positive candidate-versus-control recall protection justifies only a fresh
primary-source screen of a genuinely multi-constraint method. A possible next
information gate is original GEM-style projection onto several precommitted
class-1 boundary strata using class-1 decision-margin constraints, while
retaining raw keeper and one-reference A-GEM comparators. It must be locked as
a new method and prove finite-step keeper TP/support protection before any
trainer or validation access.
