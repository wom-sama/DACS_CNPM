# TRKH 5-Class VICRegL Crop-Match Audit

Date: 2026-07-12

Status: rejected before any trainer or model change. The current-best checkpoint
and full-train command remain unchanged.

## Question

Does the current TRKH representation lack stable local correspondence across two
overlapping views of the same mango, such that a VICRegL-style local objective
could create class-1-positive interior evidence without widening class-1 false
positives?

This is not another hflip consistency run, global VICReg warm-up, pixel MIM, or
`class_f/yolo_f` feature-consistency experiment. It tests the local mechanism
specific to VICRegL: feature and location matching before adding any projector or
auxiliary loss to the trainer.

## Primary references

- VICRegL, NeurIPS 2022:
  <https://proceedings.neurips.cc/paper_files/paper/2022/hash/39cee562b91611c16ac0b100f0bc1ea1-Abstract-Conference.html>.
- Official implementation:
  <https://github.com/facebookresearch/VICRegL>, inspected at commit
  `803ae4c8cd1649a820f03afb4793763e95317620`.

The official method applies VICReg to global embeddings and to local feature
pairs selected by two symmetric nearest-neighbor criteria: original-image
location and feature-space distance. Its default local match count is 20. The
paper trains for 100-400 epochs with much larger batches than TRKH can support,
so importing the full recipe without a readiness check would be unjustified.

## Locked protocol

- Frozen keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Data: all `9,215/2,606` `yolo_f` train/validation object rows and
  `8,064/2,577` source groups, with zero train/validation source overlap.
- Test is closed; raw data is read-only.
- Starting from the deterministic 256px runtime object view, create two fixed
  240px crops: top-left `(0,0)` and `(16,16)`, each resized back to 256px. Their
  overlap is 224px. This avoids the already rejected hflip consistency route.
- Transform bbox, crop-bbox, and valid-image mask exactly for each crop.
- Use final post-pruning patch tokens plus `patch_indices` to map each token
  center back to original-image coordinates.
- Apply symmetric location nearest-neighbor matching and feature nearest-neighbor
  retrieval, restricted to bbox-interior tokens. Keep the closest 20 location
  matches per direction, matching the official gamma.
- Compare direct keeper, crop A, crop B, and equal crop-average predictions.
- Audit local cosine, feature retrieval top-1, class-1 TP/FN/FP direction,
  prediction transitions, and a location-mismatch contact sheet.

A no-artifact 128-row implementation pilot and a class-order-biased 1,024/512
protocol preflight were allowed only to verify geometry, thresholds, runtime, and
XAI. All gate thresholds were fixed before the full run. Only full support can
authorize or reject a trainer smoke.

## Results

| Validation role | Macro F1 | Class-1 F1 | Class-1 precision | Class-1 recall |
| --- | ---: | ---: | ---: | ---: |
| Direct keeper | 0.884675 | 0.686047 | 0.611399 | 0.781457 |
| Crop A | 0.879363 | 0.670423 | 0.583333 | 0.788079 |
| Crop B | 0.874436 | 0.642659 | 0.552381 | 0.768212 |
| Crop average | 0.881340 | 0.670487 | 0.590909 | 0.774834 |

Crop-average changes 78 validation rows: 35 corrections, 38 harms, and five
neutral changes. It removes/creates class-1 false positives `12/18` and
rescues/breaks class-1 false negatives/true positives `5/6`. The views widen the
class-1 region rather than add a conservative positive cue.

Local matching is already close to saturated:

| Signal | Train | Validation |
| --- | ---: | ---: |
| Location-match cosine | 0.966519 | 0.965089 |
| Feature retrieval top-1 at location match | 0.893692 | 0.895635 |
| FN minus TP mismatch | -0.003113 | -0.001721 |
| FN-vs-FP mismatch AUROC | 0.470264 | 0.411313 |

Every row supplies all 20 required matches. True class-1 validation examples have
location cosine `0.963750`, false negatives `0.965471`, and false positives
`0.961567`. False negatives are therefore slightly more crop-consistent than
true positives, not less. Optimizing this consistency has no class-1-FN-specific
deficit to repair.

## XAI review

The 12-case contact sheet shows base input, both crops, and crop-A mismatch.
Mean mismatch is only `0.010515`. Its mass is more concentrated at the border
than the center (`0.215986` versus `0.149665`) and commonly follows crop edges,
padding, hands, stems, and object boundaries. Central fruit-surface tokens are
mostly stable already.

This matters because a local VICRegL loss would spend capacity enforcing the
remaining edge/context invariance while the actual class-1 FN/FP boundary is not
encoded by mismatch. The official paper also notes that spatial feature vectors
can have receptive fields covering the full image; high local matching does not
prove missing fine-grained maturity semantics.

## Decision

Eleven fixed checks fail and `image_smoke_permission=false`. Do not add a local
VICRegL projector or crop-pair loss, and do not sweep crop geometry, gamma,
location threshold, interior threshold, projector width, local/global weight,
optimizer, or epochs. Do not convert mismatch into a router, sample weight,
targeted margin, or class-1 reliability score.

Reopen local SSL only when a different predeclared view or target exposes a
source-safe class-1 FN deficit that is stable across train/validation and does
not create reciprocal `0/2/4->1` risk.

## Evidence and cleanup

- Full evidence:
  `runs\diagnostic_vicregl_crop_match_full_20260712`, nine payloads,
  `4,791,831` bytes, payload SHA-256
  `f249f51c8302454269ec17ae7253d44864d659ef110bf179e327d512170f6281`.
- It contains no model, checkpoint, feature cache, trainable manifest, or test
  payload.
- Cleanup manifest:
  `runs\cleanup_manifest_20260712_vicregl_crop_preflight_superseded.json`
  removed exactly one preflight root, eight files, and `1,776,284` bytes;
  observed free-space gain was `1,794,048` bytes.
- Retention:
  `runs\artifact_retention_audit_after_vicregl_crop_cleanup_20260712` passed
  over 578 run directories with `blockers=[]`.
- Current-best command SHA-256 remains
  `3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.
