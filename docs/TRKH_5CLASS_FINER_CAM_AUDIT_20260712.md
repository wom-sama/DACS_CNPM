# TRKH Finer-CAM Class-1 Readiness Audit

Date: 2026-07-12

## Question

Can class-contrastive attribution reveal a class-1-positive interior-surface cue
that ordinary class-1 Grad-CAM hides behind features shared with confusing
classes, and can that cue support a recall-safe classification action?

Primary sources:

- Finer-CAM CVPR 2025 paper:
  <https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Finer-CAM_Spotting_the_Difference_Reveals_Finer_Details_for_Visual_Explanation_CVPR_2025_paper.html>
- Official implementation:
  <https://github.com/Imageomics/Finer-CAM>

The paper explains why subtracting two post-ReLU CAM maps is not equivalent to
Finer-CAM: the logit difference must be differentiated before the final ReLU.
The published experiment uses three references and a default comparison strength
of `0.6`; the current official repository instead exposes a probability-weighted
multi-reference objective with default `alpha=1`. This audit locked the current
official objective before reading the result and did not compare or sweep the two.

## Locked Protocol

- Checkpoint: current no-pretrain keeper, SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Data: immutable `yolo_f`; train `9215`, validation `2606`; test unopened.
- Precision: FP32. Gradient attribution is not mixed with AMP numerics.
- Layer: existing default XAI source `patch_embed.proj`; no layer comparison.
- Target: class 1 for every row, independent of the ground-truth label.
- References: the three non-class-1 logits with the smallest absolute distance
  from the class-1 logit, selected independently for each sample.
- Objective:
  `sum_i p_i * (logit_1 - alpha * logit_i) / sum_i p_i`, with `alpha=1`.
- Control: ordinary class-1 Grad-CAM from the same activation and forward graph.
- Faithfulness action: replace the top 5% of valid pixels with the checkpoint
  mean RGB and measure relative confidence drop
  `(p1-p1_masked) - (weighted_pref-pref_masked)`.
- One fixed classification sanity action: when class 1 is already in the keeper
  top 2, add `RD_finer - RD_gradcam` to `p1`, clamp positive, and renormalize.
- No fit, threshold/alpha/reference/layer/mask sweep, raw-data write, checkpoint,
  trainable manifest, or test access.
- Train keeper outputs are in-sample and are transfer support, not current-keeper
  OOF evidence.

Implementation and retained evidence:

- `trkh/tools/audit_finer_cam_class1_readiness.py`
- `trkh/tools/review_finer_cam_changed_cases.py`
- `tests/test_audit_finer_cam_class1_readiness.py`
- `runs/diagnostic_finer_cam_class1_full_20260712`
- `runs/review_finer_cam_class1_full_changed14_batch32_v2_20260712`

## Results

| Split | Method | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| train | FP32 keeper | 0.939876 | 0.700265 | 0.975970 | 0.815444 |
| train | fixed contrast gain | 0.939627 | 0.702957 | 0.966728 | 0.814008 |
| val | FP32 keeper | 0.882925 | 0.603093 | 0.774834 | 0.678261 |
| val | fixed contrast gain | 0.884183 | 0.609375 | 0.774834 | 0.682216 |

The validation candidate changed 14 rows: 8 corrections and 6 harms. It removed
6 class-1 false positives and created 4, but rescued 2 class-1 false negatives
while breaking 2 true positives. Recall is numerically unchanged, yet there is no
positive recall action and class-1 F1 remains below both the required `+0.005`
gain and the `0.70` milestone. Smoke permission is `false` (`13/17` checks).

The Finer-CAM faithfulness metric improves on average but not as a stable error
separator:

| Split | Grad-CAM RD | Finer-CAM RD | Gain | FN-vs-FP gain AUROC |
| --- | ---: | ---: | ---: | ---: |
| train | 0.005041 | 0.005317 | +0.000276 | 0.574541 |
| val | 0.006346 | 0.006777 | +0.000430 | 0.616501 |

The validation AUROC passes the local `0.60` check, but train does not. The 13
train false negatives are also too sparse to treat the val-only separation as a
new supervision target. Current-keeper OOF probabilities do not exist, so fitting
a threshold/router from this signal would reuse in-sample train evidence and tune
on validation.

## Spatial And Transition Review

Finer-CAM does not move evidence toward the required interior surface:

| Split | Metric | Grad-CAM | Finer-CAM |
| --- | --- | ---: | ---: |
| train | bbox mass | 0.807631 | 0.777175 |
| train | interior mass | 0.355120 | 0.306467 |
| train | border mass | 0.177931 | 0.195930 |
| val | bbox mass | 0.814190 | 0.805513 |
| val | interior mass | 0.358258 | 0.323801 |
| val | border mass | 0.181833 | 0.191826 |

All 14 changed validation rows were recomputed with their original batch-32
windows (`334` context rows). Every target/global/candidate prediction reproduced,
and the maximum absolute relative-drop difference from the full CSV was
`6.56e-7`. The contact sheet shows a mixed and unsafe mechanism:

- useful `1->0` corrections often suppress class-1 evidence concentrated on
  silhouette edges, crop borders, bright endpoints, stems, hands, or padding;
- the two rescued class-1 rows are still border dominated: changed-case interior
  mass falls `0.115296 -> 0.075906`, while border mass rises
  `0.400166 -> 0.412056`;
- the four newly created class-1 false positives include central color bands,
  lesions, stems/hands, and bottom-border evidence;
- the two broken true positives lose class 1 through the same broad boundary
  suppression that removes false positives.

This is a small conservative calibration effect, not a new class-1-positive
surface representation. The visual review agrees with the direct action: false
positive control improves slightly, but recall evidence remains inseparable.

## Baseline Precision Note

The earlier surface-tile audit reported the AMP in-training anchor
`0.884675/0.686046`; this FP32 gradient audit reproduces the locked independent
reload gate `0.882925/0.678261`. Row-wise comparison found identical labels and
paths, six changed predictions, maximum probability difference `0.022944`, and
mean absolute difference `0.000333`. The difference is AMP versus FP32, not a
checkpoint/data mismatch. Finer-CAM is evaluated only against its matched FP32
control.

## Decision

Reject Finer-CAM as a trainer-smoke target on the current keeper. Do not sweep
`alpha`, target layer, reference count/ranks, mask fraction/fill, CAM backend,
residual scale, sign thresholds, or a post-hoc router. In particular, do not turn
the six validation false-positive corrections into another `1->0` filter: that
route was already shown to be a validation trap, and no fold-safe keeper signal
exists here.

Finer-CAM remains useful as a reporting/XAI tool because its relative-drop gain
does expose a modest target-vs-confuser distinction. It is not permission to
train or promote a model. A next representation hypothesis must create direct
class-1 FN rescue with interior evidence and preserve true-positive recall before
GPU smoke.

The full artifact has 8 payloads, `8770683` bytes, aggregate SHA-256
`dfa7a1a584a3b5a7ee2dca190f7acf2dee9c8568715797c35a5a362f8aa940db`.
The exact changed-case review has 4 payloads, `1032981` bytes, aggregate SHA-256
`10f48a651355fc8364621020dbda06f0258915c2bde6926fecd4169ae91b67b9`.
Neither contains a model, checkpoint, engine, or test payload. The keeper and
current-best command remain unchanged.

Two guarded cleanup manifests removed five superseded method roots and one
obsolete retention snapshot: 41 files, `3312956` bytes, with observed free-space
gain `3383296` bytes. Every deleted file had a persisted SHA-256 before removal;
both deletions verified the keeper and retained full/review manifests first.
Final retention passed over 583 run directories with `blockers=[]`.

Closure passed py-compile, compileall, focused tests `13/13`, full pytest
`737/737`, `git diff --check`, artifact/cleanup/retention hash verification, and
explicit protected-path review. The current-best command SHA remains
`3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.
