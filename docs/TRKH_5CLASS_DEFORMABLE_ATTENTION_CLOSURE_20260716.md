# TRKH 5-Class Deformable Attention A1 Closure - 2026-07-16

## Decision

Close the exact block-2 CVPR-2022 DAT A1 route after its sole locked
five-epoch scratch pair. The candidate produced a real clean class-1 precision
signal, but it failed the prospectively locked macro-F1, error-transition,
condition-stability, selectivity, outside-bbox distance, and XAI gates.

Stage B, official validation, test, full training, neighboring DAT sweeps, and
current-best command promotion are denied. This is a behavioral and spatial-
selectivity rejection, not an equation or deployment failure.

## Primary-Source Provenance

- Accepted paper: Xia et al., "Vision Transformer with Deformable Attention,"
  CVPR 2022.
- Official Apache-2.0 repository: `https://github.com/LeapLabTHU/DAT`.
- Repository observation at protocol time: 938 stars.
- Locked tag/commit/tree: `CVPR2022` / `566a593daf96efc3df58a1542e60b847b8b6f4ff`
  / `4346f051456b3270e9161839410123260db82508`.
- Official attention-source SHA-256:
  `f7318527e9de7a440ab341ba0aa60ca3f164714fc97e1c03670f165f6b7470ea`.
- Local primary-paper SHA-256:
  `92c3f6bb2858055d0d6932b2168638394320d4764110b5223651e14a6cb8a60`.
- Immutable protocol SHA-256:
  `6abbd94c55f2e6013d61b728016f57af4730e49c87c3ada6dd1d861c63f73cbc`.
- Pushed implementation HEAD:
  `aa26c09ff44b084514e95fc15200da7e501706ad`.

User-provided and secondary research reports were hypothesis inputs only. The
method, equations, settings, and gates came from the accepted paper and the
official tagged code.

## Exact Route

Stock DAT-T and its 300-epoch recipe were excluded. The sole TRKH route
replaced only block 2, immediately before token pruning, while the complete
`16x16` patch grid was still available:

- 8 attention heads and 2 offset groups;
- shared depthwise `5x5` offset network, stride 1, range factor 2;
- bilinear sampling with `align_corners=True`;
- continuous deformable relative-position bias;
- unchanged global prefix attention;
- bilinear-scatter proxy used only to preserve the existing pruning contract;
- no bbox loss, pretrained weights, teacher, router, threshold, or raw-data
  modification.

The exact source-disjoint `yolo_f/train` fold contained 7,372 fit and 1,843
holdout rows, with zero source overlap. No official validation or test path was
used.

## Engineering Gate

The corrected formal preflight passed all 65 checks:

- official equation parameter-gradient maximum error: `3.7252903e-9`;
- official output and sampled-position maximum error: `1.1920929e-7`;
- independent bilinear-proxy maximum error: `1.8626451e-9`;
- added parameters: 11,528 (`7,245,590 -> 7,257,118`);
- candidate/control median inference ratio: `1.2920`;
- candidate/control peak training-memory ratio: `1.07393`;
- static ONNX maximum logit error: `3.5762787e-7` with matching argmax.

Preflight summary/manifest SHAs are
`4805085daad648a24345947b5b4a14b089185248094c26e5da2257a8ea7456a1`
and `3492aea27d37462f6a1624a40dc72945f29a1b05be4c06af283bedb207ed5687`.

## Formal Pair

Control and candidate used the same seed, five-epoch schedule, source order,
augmentation, optimizer, loss, batch/effective batch, and train-only fold.
The outer orchestration shell timed out after control completion and before the
candidate trained. Six zero-epoch initialization files were hashed and removed,
then the exact candidate argument vector was replayed once on the same pushed
HEAD. The recovery manifest SHA is
`f25d870974be3814090bbd780588c20a56b3647f2b87efadca38a9d8e6c8a61c`.

Independent standard-inference audit of the selected checkpoints gave:

| Metric | Control | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Clean macro F1 | 0.792451 | 0.789104 | -0.003347 |
| Clean class-1 F1 | 0.277778 | 0.293706 | +0.015929 |
| Clean class-1 precision | 0.571429 | 0.617647 | +0.046218 |
| Clean class-1 recall | 0.183486 | 0.192661 | +0.009174 |
| Clean restricted FP | 15 | 13 | -2 |

The class-1 F1/precision/recall delta gates passed, but the absolute class-1 F1
minimum `0.60`, macro delta `+0.003`, restricted-FP reduction of four, and
error-transition gates failed. Candidate corrections/harms were `10/23`; it
rescued four class-1 FN and broke three TP.

Condition behavior was not stable:

| Condition | Macro F1 delta | Class-1 F1 delta | Precision delta | Recall delta | Restricted FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bright | -0.006279 | -0.016450 | +0.092567 | -0.018349 | 15 -> 9 |
| Dim | -0.014112 | -0.008482 | -0.054935 | +0.018349 | 45 -> 61 |
| Low contrast | +0.007373 | +0.036616 | -0.109317 | +0.036697 | 8 -> 16 |

Clean TP-versus-hard-negative AUROC improved only
`0.596833 -> 0.603583` (`+0.006750`), below the locked absolute `0.65` and
delta `+0.020` gates. Bright/dim AUROC stayed below `0.60`.

## Spatial And XAI Result

DAT was active rather than collapsed:

- clean offset RMS: `0.055461`;
- clean inter-group position RMS: `0.076681`;
- valid interpolation mass: `0.990372`;
- bbox-hit delta: `+0.047965`, positive on `0.823114` of rows;
- frozen-cohort block-2 bbox mass: `0.701354 -> 0.772842`.

However, points that stayed outside the bbox moved farther from it on average:
reference/candidate distance was `0.077866 -> 0.089552`. The field remained a
broad near-grid over object and context instead of selectively rejecting crop
background.

All 94 required XAI rows, 87 decision events, eight representative categories,
and 16 contact sheets were inspected. Candidate-minus-control foreground mass
was `-0.061831` at the stem and `-0.179018` at block 2. Sampling density and
saliency repeatedly included hands, shadows, borders, and background. Object
perturbation remained more causal than far-background perturbation, but this
did not rescue the failed selectivity and condition gates. Visual review was
therefore finalized as fail.

## Auditor Correction And Replay

The first post-run audit incorrectly used BF16 `return_attention=True` trace
logits for behavioral scoring. A 64-row diagnostic found maximum trace versus
standard-inference logit differences of `0.072754`. That generated audit was
hashed, declared invalid, and removed; cleanup-manifest SHA is
`df42ffc98e2b864b36a49ed3b7042283ff95200a3b75624ed570fa0b1f72f545`.

The auditor now uses standard deployment inference for all decisions and a
separate trace-only forward for spatial/XAI evidence. A regression test forces
the two paths to disagree and proves that behavior follows standard inference.
The complete audit and all 16 visual pages were regenerated.

Independent replay of 14,744 prediction rows reproduced every confusion matrix
and F1 value with maximum absolute error `1.11e-16`, all 87 events, all clean
spatial means, and the 143-row frozen-cohort bbox masses.

Final pair summary/prediction/visual-review SHAs are
`57f0638e961ece19dc1a543b0844806290f9bcebb34e409e387eb278b35726a6`,
`43788e356f15347ccf9d73961e53481766fa3d44600a69afc71eae0d815a433d`,
and `20a7f49a0c2c9d1066a55dffb8cfa2bb9d70014d167a5025c68f6d9dfcdec667`.

## Stop Boundary

- Do not sweep DAT groups, kernel, range, stride, RPE, insertion block, layer
  count, optimizer, LR, loss, augmentation, epoch count, fold, seed, bbox
  supervision, threshold, or router on this keeper.
- Do not reinterpret clean precision gain as object selectivity: macro F1,
  transitions, illumination, outside-bbox distance, and XAI contradict it.
- Any future deformable route requires a distinct primary source and a new
  mechanism that explicitly proves selective outside-context rejection before
  training.
- Current-best commands remain at three revisions, two updates, and zero
  keeper replacements.

## Evidence Retention

Rejected checkpoints and reproducible ONNX diagnostics totaling 520,134,563
bytes were removed only after exact SHA verification. All summaries, histories,
launcher/config payloads, data cartography, predictions, event manifests, and
XAI pages remain. Cleanup-manifest SHA is
`953cf3095a796ff147bdacf6afa45d02fd3762c0839c30fdfe5198ab0a792454`.

Protected keeper, random-scratch checkpoint, current-best command, and command-
history hashes remain exactly unchanged. Free space after compaction was
71.110 GiB.

Compileall, PowerShell parse, focused DAT tests `19/19`, and full pytest
`1250/1250` passed. The read-only retention audit passed over 719 directories
with `blockers=[]`; retention-summary SHA is
`bae9909d950b94a3b22454cb19658cb102857473c034bf331255f966077c4228`.
