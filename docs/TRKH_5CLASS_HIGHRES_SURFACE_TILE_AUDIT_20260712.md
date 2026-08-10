# TRKH High-Resolution Surface-Tile Readiness Audit

Date: 2026-07-12

## Question

Does the original-resolution object crop contain local class-1 surface evidence
that is diluted by the keeper's single 256 px global crop?

This is narrower than another backbone substitution. FastViT-SA12 was reviewed
but not launched: local `timm 1.0.27` defines three RepMixer stages followed by
one attention stage, with the default RepMixer kernel size 3. After the controlled
MobileViT, EdgeNeXt, CoAtNet, EfficientFormerV2, LeFF, and persistent-coupling
failures, that stock staged hybrid did not provide a new class-1-positive target.
The primary FastViT paper remains relevant architecture background, but it does
not override the existing stock-backbone closure:
<https://openaccess.thecvf.com/content/ICCV2023/html/Vasu_FastViT_A_Fast_Hybrid_Vision_Transformer_Using_Structural_Reparameterization_ICCV_2023_paper.html>.

The retained alternative follows the local-region motivation in NTS-Net and the
multi-granularity motivation in PMG, but first tests the image evidence without
training a navigator, router, or new backbone:

- <https://openaccess.thecvf.com/content_ECCV_2018/html/Ze_Yang_Learning_to_Navigate_ECCV_2018_paper.html>
- <https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3399_ECCV_2020_paper.php>

## Locked Protocol

- Checkpoint: current no-pretrain TRKH keeper, SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Data: immutable `yolo_f`; train `9215`, validation `2606`; test unopened.
- Global control: exact checkpoint object crop, resize/pad transform, source bbox,
  transformed `crop_bbox`, valid mask, and bbox token prior.
- Candidate views: top-left, top-right, bottom-left, bottom-right, and center tiles,
  each fixed at 70% of the original-resolution bbox crop before model resize.
- Each tile keeps the clipped transformed object bbox and the unchanged source
  bbox metadata. Views are forwarded sequentially to keep GPU memory bounded.
- Fixed candidate: `0.50 * global_probability + 0.50 * mean(tile_probability)`.
- No model fitting, threshold/weight sweep, raw-data write, checkpoint output, or
  test access.
- Train keeper probabilities are in-sample. Their view deltas are transfer support,
  not current-keeper OOF evidence.

Implementation and evidence:

- `trkh/tools/audit_highres_surface_tiles_readiness.py`
- `tests/test_audit_highres_surface_tiles_readiness.py`
- `runs/diagnostic_highres_surface_tiles_full_20260712`

## Results

The global path reproduced the keeper exactly, which validates the crop and
metadata contract:

| Split | Method | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| train | global | 0.939831 | 0.699735 | 0.977819 | 0.815729 |
| train | tile mean | 0.914416 | 0.676829 | 0.820702 | 0.741855 |
| train | fixed fusion | 0.938949 | 0.718529 | 0.939002 | 0.814103 |
| val | global | 0.884675 | 0.611399 | 0.781457 | 0.686046 |
| val | tile mean | 0.862581 | 0.569892 | 0.701987 | 0.629080 |
| val | fixed fusion | 0.880824 | 0.601064 | 0.748344 | 0.666667 |

Validation fusion changed 48 decisions: 21 corrections, 24 harms, and 3 neutral
changes. It removed 10 class-1 false positives but created 10, broke 5 true
class-1 predictions, and rescued zero class-1 false negatives. Therefore it failed
macro non-inferiority, class-1 gain, the 0.70 milestone, recall preservation,
correction/harm balance, and positive recall action. Core smoke permission is
`false` (`9/15` checks passed).

## Direction And Visual Audit

The mean tile-minus-global class-1 probability delta had FN-vs-FP AUROC
`0.844347` on train and `0.840000` on validation. The sign also transferred:

- train FN/FP mean delta: `+0.003567 / -0.034688`
- validation FN/FP mean delta: `+0.011426 / -0.029362`

This is not a deployable action. The direct candidate mostly contracts class-1
confidence. A fixed matched binary-readout sanity check using only train labels
confirmed the limitation: global-logit control reached validation macro/class-1
`0.867316/0.633245`; adding tile summaries reached only
`0.869291/0.640212`, both below the unchanged keeper. The small incremental
direction does not permit a router, threshold, or core-model smoke.

The 12-row contact sheet confirms correct tile geometry and shows surface spots,
color transitions, glare, lesions, stems, and crop-adjacent context. It does not
show a consistent tile that distinguishes true class 1 from visually overlapping
class `0/2/4` cases. Only `9.27%` of validation class-1 crops have a raw minimum
side of at least 384 px; false negatives are not concentrated in smaller crops.
The missing class-1 evidence is therefore not primarily global-resize loss.

## Decision

Reject direct original-resolution surface tiling and do not sweep tile fraction,
positions, count, fusion weight, crop threshold, image size, or a post-hoc tile
router on the current keeper. Do not launch a trainer smoke from the high AUROC
alone. A future local-region route must change the spatial target itself, for
example a deployable class-contrastive attribution that suppresses features shared
by class 1 and its confusing neighbors before selecting a region.

The full artifact contains 8 payloads, `8462921` bytes, no model/checkpoint/test
payload, and aggregate payload SHA-256
`010d368e87ab3214a5b1f09f0bcc048032890e5c8663bc901a191a85e0760de2`.
The implementation preflight was removed after full preservation. Its cleanup
manifest honestly records a PowerShell 5.1 provenance warning rather than
inventing unavailable per-file hashes.

Closure passed py-compile, compileall, focused tests `7/7`, full pytest
`724/724`, payload-manifest verification, and retention over 580 run
directories with `blockers=[]`. The keeper and current-best command remain
unchanged.
