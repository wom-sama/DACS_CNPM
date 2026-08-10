# TRKH 5-Class MambaVision-Nano Audit - 2026-07-12

## Decision

Reject the fixed scratch-only MambaVision-Nano candidate at the five-epoch
gate. Do not continue to epochs 10/15/30, open test, update the current-best
command packet, or sweep width/depth/window/input/LR/loss/sampler/teacher
settings. The candidate changed the token mixer but did not learn conservative
class-1 boundaries quickly enough.

The locked protocol is in
`docs/TRKH_5CLASS_MAMBAVISION_NANO_PROTOCOL_20260712.md`. Full compact evidence
is retained at
`runs/evidence_mambavision_nano_yolof_scratch_5e_reject_20260712`.

## Research Basis

The primary sources were the [MambaVision paper](https://arxiv.org/abs/2407.08083),
the [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.pdf),
the [official NVIDIA repository](https://github.com/NVlabs/MambaVision), and the
[NVIDIA Research page](https://research.nvidia.com/publication/2025-06_mambavision-hybrid-mamba-transformer-vision-backbone).
The official smallest released model is much larger than the keeper, so one
fixed 6.172M-parameter scaling was declared before validation:

- convolution stages: depths `1/2`;
- later stages: depths `4/2` with Mamba/attention mixers `2+2` and `1+1`;
- width `48`, input `256`, windows `8/8/16/8`, drop path `0.10`;
- no pretrained weights and no architecture knob exposed through CLI.

This is a bounded representation test, not a claim that scaled MambaVision is
a new architecture. The external package has NVIDIA Source Code License-NC;
no package source was copied into TRKH.

## Implementation And Preflight

- Added checkpoint-safe `model_type=mambavision_nano` with exact structural and
  parameter-count guards.
- Added fixed ImageNet normalization and a clear CUDA-only selective-scan
  preflight.
- Extended the existing compact-hybrid launcher without changing its default.
- Added generic, signature-aware architecture tracing. The old trace path
  incorrectly passed TRKH-only `bbox_token_prior/return_trace` arguments to
  every `forward_features`; focused regressions now cover both APIs.
- Added a fail-closed rejected-run compactor that hashes source files, copies
  non-binary evidence, verifies payloads, persists a pre-delete manifest, and
  only then removes declared source directories.

The exact runtime was PyTorch `2.6.0+cu124`, `mambavision==1.2.0`,
`mamba-ssm==2.2.4`, and `timm==1.0.27`. A BF16 batch-64 forward/backward/AdamW
step was finite at `137.20 img/s`, with peak allocation/reservation
`1150.68/1548.00 MiB`. No dependency was changed despite pre-existing shared
environment pin conflicts.

## Smoke Gate

The one-epoch/20-batch smoke completed train and all `2606` validation rows,
strictly rebuilt the checkpoint, produced boundary/XAI artifacts, and did not
open test. Its very early independent macro/class-1 F1 was
`0.12234/0.13898`; this value was infrastructure-only and was not used for
model selection.

On 12 smoke errors, Grad-CAM foreground/background mass was
`0.96375/0.03625`. Background blur/gray changed predicted probability by only
`0.00554/0.00664`, versus `0.09746` for object desaturation. The pipeline was
therefore healthy enough to permit the predeclared five-epoch run.

## Five-Epoch Result

Trainer full-validation history:

| Epoch | Macro F1 | Class-1 F1 | Class-1 P | Class-1 R |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.68957 | 0.43094 | 0.36967 | 0.51656 |
| 2 | 0.73417 | 0.38994 | 0.28528 | 0.61589 |
| 3 | 0.76558 | 0.43810 | 0.34201 | 0.60927 |
| 4 | 0.75904 | 0.43981 | 0.33808 | 0.62914 |
| 5 | 0.78136 | 0.48095 | 0.37546 | 0.66887 |

The independent checkpoint reload over all `2606` rows was slightly lower:

| Metric | MambaVision-Nano | Epoch-5 gate | Keeper reload |
| --- | ---: | ---: | ---: |
| Macro F1 | 0.78048 | 0.83032 | 0.88292 |
| Class-1 F1 | 0.47733 | 0.55340 | 0.67826 |
| Class-1 precision | 0.37313 | diagnostic | 0.60309 |
| Class-1 recall | 0.66225 | diagnostic | 0.77483 |

Both continuation gates failed. The independent class F1 values were
`0.84550/0.47733/0.81034/0.89443/0.87479`; class 1 was not the only deficit.

Independent confusion was:

```text
[[446, 78,  6,  1, 18],
 [ 33,100, 15,  0,  3],
 [  0, 36,470, 29,  9],
 [  0,  2, 97,610,  3],
 [ 27, 52, 28, 12,531]]
```

The model produced `168` class-1 false positives for only `100` true
positives, versus keeper `77/117`. It also increased class-1 false negatives
from keeper `34` to `51`. The selective-scan candidate therefore lost both
precision and recall rather than trading one safely for the other.

## XAI And Trace

The final XAI set was fixed from independent BF16 predictions and balanced as
two cases each for `1->0`, `1->2`, `1->4`, `0->1`, `2->1`, and `4->1`.
FP32 batch-1 XAI changes a few near-boundary full-split decisions, so its
confusion counts are diagnostic only; gate metrics remain the independent BF16
reload above.

- Grad-CAM foreground/background mass: `0.93987/0.06013`; border `0.15400`.
- Background blur/gray predicted-probability drop: `-0.00048/+0.00030`.
- Center occlusion drop: `0.01873`.
- Object desaturation drop: `0.13313`.
- Six of 12 cases were explicitly flagged object-color sensitive.

The generic `attention` map is a feature-map fallback, not exported
MambaVision attention and not causal evidence. Grad-CAM, perturbations, and
visual review agree that the model mainly looks at the fruit, but uses broad
color/surface regions. Correct class-1 positives and `0/2/4->1` false positives
share pale-green/yellow/mottled cues. Wide background is not the limiting
signal for this candidate.

The five-class architecture trace captured patch/stage feature shapes
`48x64x64 -> 96x32x32 -> 192x16x16 -> 384x8x8 -> 384x8x8` and one fixed train
sample per class. Stage RMS maps become coarse and smooth; they are structural
activation diagnostics, not localization claims.

## Cleanup And Reproducibility

The compact evidence keeps configs, histories, plots, complete validation
predictions, boundary manifests, both XAI audits, five-class trace, and exact
source inventory. After correcting a self-review bug that initially excluded
nested source `summary.json` files from the payload manifest, final retention
contains `314` verified payloads with manifest SHA-256
`2660ab5599d8e1a37462ac8375cee5339efd7d81ed8e2fa922ad0060c6f446a1`.

Cleanup manifest
`runs/cleanup_manifest_20260712_mambavision_nano_rejected.json` records nine
deleted source roots. The compactor copied `312` source files (`48,325,685`
bytes), retained hashes for four excluded checkpoint files (`297,313,032`
bytes), verified deletion, and observed `346,447,872` bytes of free-space gain.
No dataset, test artifact, keeper, current-best command, or protected user path
was modified.

## Verification

- Compileall passed for `trkh` and tests.
- Focused model/trace/compactor/TIMM/XAI regressions passed `21/21`.
- Full pytest passed `799/799`.
- Retention audit
  `runs/artifact_retention_audit_after_mambavision_nano_reject_20260712`
  passed over `601` run directories with `blockers=[]` and confirmed all nine
  compacted source roots are absent.
- Keeper SHA-256 remains
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`;
  current-best command SHA-256 remains
  `3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.
- `pip check` reports only the pre-existing exact-pin conflicts from
  MambaVision plus OpenCV's NumPy requirement. No dependency was changed.

## Closed Variants

Do not sweep MambaVision model size, width, stage depth, Mamba/attention ratio,
window size, input size, drop path, LR, loss, sampler, teacher weight, or
augmentation on this protocol. Do not transplant only the Mamba mixer into the
current keeper as another ungrounded adapter. Reopen state-space mixing only
when a new class-1-positive supervision target can independently protect keeper
true positives and suppress `0/2/4->1` false positives.
