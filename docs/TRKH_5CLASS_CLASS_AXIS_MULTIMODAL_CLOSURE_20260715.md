# TRKH 5-Class Class-Axis and Multimodal Closure - 2026-07-15

## Scope and research integrity

This closure evaluates two new representation-family screens outside the closed
gradient-combination family:

1. a deliberately favorable balanced class-axis energy diagnostic inspired by CCAR;
2. a fixed 10-mode class-conditional density diagnostic inspired by the WACV 2024
   orthonormal-matrix work.

Neither candidate is an exact end-to-end reproduction. The question was whether the
current frozen keeper embedding contains enough aligned signal to justify shared-trainer
integration under the local 30-epoch limit.

An exploratory train-to-validation calculation was inspected before the formal protocol.
It was not used to tune a setting: the class-axis diagnostic reached validation
macro/class-1 F1 `0.7640/0.4036`, while the natural-prior GMM reached
`0.8613/0.6258`. The formal decision below used a separately locked train-only split,
fixed recipes, and no validation or test access. The formal run is prospective relative
to its exact gate, but not blind to the broad family-level weakness.

## Locked formal run

- Fit/holdout: `7372/1843` `yolo_f/train` rows.
- Source groups: `6452/1612`, overlap zero.
- Feature dimension: `256`.
- Raw and aggregate-margin A-GEM predictions were preserved on the same holdout.
- The reusable cache and canonical raw replay differed by one argmax and maximum
  probability `0.0327522`; this reconciliation was declared before the formal run.
- Validation/test/model binaries/raw-data writes: none.
- Protocol SHA-256: `099b797c...9bc776`.
- Precommit: `5bc29cd`.

## Results

| Candidate | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | Class-1 TP | Restricted FP | Direction AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw keeper | 0.949323 | 0.748252 | 0.981651 | 0.849206 | 107 | 36 | - |
| Aggregate-margin A-GEM | 0.951161 | 0.769784 | 0.981651 | 0.862903 | 107 | 32 | - |
| Balanced class axis | 0.807973 | 0.329268 | 0.990826 | 0.494279 | 108 | 206 | 0.250000 |
| Natural-prior 10-mode GMM | 0.929297 | 0.742188 | 0.871560 | 0.801688 | 95 | 33 | 0.652778 |

The class-axis assignment was structurally valid: each class received 51 unique
dimensions and one dimension was unused. It nevertheless expanded class 1 to 328
predictions and added 170 restricted false positives. Its direction AUROC was inverted.

All five GMMs converged in `8..115` iterations. The density score had a useful direction
AUROC, but that ranking did not survive a class decision: it removed only three restricted
false positives, broke 13 raw class-1 true positives, made `14/40` corrections/harms,
and reduced class-1 recall by `0.110092`. Precision obtained with that TP loss is unsafe
for the agricultural objective.

## Decision

- Shared-trainer authorization is denied for both fixed candidates.
- Do not add CCAR-style class-energy blocks, a GMM head, an orthonormal-matrix proxy,
  or a neighboring component/prior/assignment sweep to the current keeper embedding.
- This closes only the current frozen-representation screens. It does not claim that the
  original CCAR preprint or WACV method is universally ineffective.
- The WACV route also overlaps already rejected sub-centroid, nearest-subspace, MCR2,
  prototype, kNN, and soft-centroid evidence. An end-to-end version would require the
  100-300-epoch regime used by the primary work and lacks a local information signal.
- Current-best full-train commands remain at three revisions and two updates after the
  initial revision. No validation winner or checkpoint promotion occurred.

## Evidence

Evidence root: `runs/audit_class_axis_multimodal_readiness_20260715`.

- Summary: SHA-256 `780f3a6e...2fe4a`.
- Holdout predictions: SHA-256 `53d8ed1a...c56f77`.
- Report: SHA-256 `d4091b53...12ebe5`.
- Manifest: SHA-256 `59ba1324...c41e40`.
- Four nonbinary payloads total less than 1 MB; no cleanup is needed.

Independent replay reconstructed all 1,843 probability vectors, argmax values,
confusion matrices, macro/class-1 F1 values, corrections/harms, restricted-FP deltas,
and payload hashes. No checkpoint, ONNX graph, engine, validation payload, or test payload
exists in the evidence root.

Compileall, focused tests `4/4`, full pytest `1165/1165`, PowerShell parsing, and
preflight all passed. Read-only retention passed over 695 run directories with
`blockers=[]` and `72.274 GiB` free; retention summary SHA-256 is
`cae7f1b3...ece7f63`. Keeper, scratch complement, current command, and command-history
hashes remained exact.

## Next boundary

A shared/discriminative feature decomposition is only admissible as a new route if it
first survives a source-grouped train-only adapter gate and remains distinct from the
closed context-fusion, reconstruction, centroid/subspace, and background-suppression
experiments. The GSFL paper is relevant, but its official recipe uses pretrained VGG,
cross-validated class groups, 150+200 training epochs, and test-time model selection; its
repository also has no license and indexes one shared-center loss with `labels[0]` for
every later batch row. These issues must be resolved prospectively without copying source
before any implementation is allowed.
