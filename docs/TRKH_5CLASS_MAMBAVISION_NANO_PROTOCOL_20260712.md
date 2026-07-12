# TRKH 5-Class MambaVision-Nano Locked Protocol - 2026-07-12

## Scope

This stage tests one TRKH-controlled, scratch-only hybrid architecture. It is
not a catalogue sweep and is not a current-best promotion. Raw `class_f` and
`yolo_f` files remain immutable, test remains closed, and the existing keeper
and command packet remain protected.

The new mechanism is selective state-space mixing between convolution and
self-attention. This is distinct from the already rejected MobileViT,
EdgeNeXt, CoAtNet, EfficientFormer, pure SSL/frozen-readout, context-fusion,
texture, uncertainty, and post-hoc routing families. It does not by itself
claim a new scientific architecture; it is a bounded representation gate for
whether selective scan is worth integrating more deeply into TRKH.

## Primary Sources

- MambaVision paper: <https://arxiv.org/abs/2407.08083>
- CVPR 2025 paper: <https://openaccess.thecvf.com/content/CVPR2025/papers/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.pdf>
- Official NVIDIA repository: <https://github.com/NVlabs/MambaVision>
- NVIDIA Research publication page: <https://research.nvidia.com/publication/2025-06_mambavision-hybrid-mamba-transformer-vision-backbone>

The official design uses early convolution stages, redesigned Mamba mixers in
later stages, and final self-attention blocks for longer-range spatial
dependencies. The smallest released T model is about 31.2M parameters, so it
is not used directly. One fixed Nano scaling is used to stay near the current
7.246M-parameter keeper.

## Locked Architecture

`model_type=mambavision_nano`, input `256x256`, 5 classes, no pretrained
weights:

| Field | Value |
| --- | --- |
| Base factory | `mamba_vision_T` |
| Stem width / stage width | `in_dim=32`, `dim=48` |
| Stage depths | `[1, 2, 4, 2]` |
| Attention heads | `[2, 4, 8, 16]` |
| Window sizes | `[8, 8, 16, 8]` |
| Drop path | `0.10` |
| Parameters | `6,172,197` |
| Stage 0/1 | `1/2` convolution blocks |
| Stage 2 | `2` MambaVision mixers + `2` attention blocks |
| Stage 3 | `1` MambaVision mixer + `1` attention block |

There are no CLI knobs for width, depth, window, mixer ratio, or input size.
The builder validates the exact block signature and parameter count so package
drift fails closed. Checkpoint reconstruction uses only the dedicated model
type and the fixed code specification.

## Runtime Provenance

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch/CUDA: `2.6.0+cu124` / `12.4`
- `mambavision==1.2.0`, `mamba-ssm==2.2.4`, `timm==1.0.27`
- `einops==0.8.2`, `transformers==4.57.6`
- `mamba_vision.py` SHA-256:
  `b5e18075f4d3be2de06ba9336716bda7d504351aed17cab7977be90a6d9a8136`
- `selective_scan_interface.py` SHA-256:
  `33f8996082a9f94df52f921171a5d13e268cfb133204c70d577accbf14635a0d`
- CUDA selective-scan binary SHA-256:
  `224628f15e61b27bc7029808e53888d88723fa8ba1bfa8f0b27c8fce67e55cfc`

The installed selective-scan kernel is CUDA-only. Training now fails early
with an explicit message when CUDA is unavailable. No package was installed,
removed, or upgraded for this stage; the shared-environment `pip check`
conflicts remain pre-existing and must not be repaired inside this experiment.

The MambaVision package uses the NVIDIA Source Code License-NC. No package
source is vendored into TRKH. This route is research-only unless the license is
reviewed separately for deployment or commercialization.

## Resource Preflight

A real CUDA BF16 forward/backward/AdamW step at batch 64 and image 256 was
finite: `0.46646 s`, `137.20 img/s`, peak allocated/reserved
`1150.68/1548.00 MiB`. Batch 64 is retained to match the controlled compact
hybrid protocol rather than changing architecture and batch geometry together.

## Fixed Training Protocol

- `yolo_f/data.yaml`, classification object crops, full `val=2606`.
- Batch 64, seed 42, 120 train batches/epoch, scheduler horizon 15.
- LDAM-focal, strict balanced sampler, EMA `0.995`.
- Teacher-focus-binary `0.015` and the same mild keeper-style augmentation.
- `--skip-final-test`; no test prediction, threshold, or selection.
- One 20-batch/1-epoch pipeline smoke, then one 5-epoch architecture gate.
- No LR, loss, width, depth, window, input, teacher, sampler, or augmentation
  sweep.

## Predeclared Gates

The 1-epoch smoke is infrastructure-only. It must have finite train/eval,
strict checkpoint reload, all 2,606 validation rows, working boundary/XAI
exports, and no data/test access before the 5-epoch run is permitted.

At epoch 5, continuation is permitted only if independent reload reaches both:

- macro F1 at least `0.83032`; and
- class-1 F1 at least `0.55340`.

These are the controlled CoAtNet-Nano epoch-5 marks, not values chosen after
seeing MambaVision results. Class-1 false positives/false negatives and XAI
must also show no catastrophic background or recall failure.

At epoch 10, a continuation to epoch 15 requires independent macro/class-1 F1
above the strongest rejected scratch hybrid's final marks
`0.86161/0.62637`, with class-1 recall at least `0.70`.

A 30-epoch/full-train allocation requires the existing milestone class-1 F1
`>=0.70`, clear audit evidence, and no gate regression. Current-best commands
may be updated only after a final independent locked-validation win over the
keeper, including macro/class-1/recall preservation, followed by command
preflight and focused tests. Smoke, oracle, in-training, or test-tuned results
cannot promote the model.

## Cleanup Contract

Each rejected smoke/probe is compacted only after metrics, complete validation
predictions, confusion/boundary evidence, XAI, launcher/config provenance, and
payload hashes are retained. Deletion requires a guarded manifest and a fresh
artifact-retention audit. Git staging remains explicit; `BaoCao/` and both
untracked deep-research reports remain outside this stage.
