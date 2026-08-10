# TRKH 5-Class InceptionNeXt-Atto Tokenizer Protocol (2026-07-15)

## Research question

Can the first three official InceptionNeXt-Atto stages provide efficient
multi-direction surface evidence before the existing TRKH Transformer, raising
class-1 precision without sacrificing true class-1 recall?

Primary sources:

- *InceptionNeXt: When Inception Meets ConvNeXt*, CVPR 2024:
  https://openaccess.thecvf.com/content/CVPR2024/html/Yu_InceptionNeXt_When_Inception_Meets_ConvNeXt_CVPR_2024_paper.html
- Accepted paper PDF:
  https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_InceptionNeXt_When_Inception_Meets_ConvNeXt_CVPR_2024_paper.pdf
- Official Apache-2.0 PyTorch repository:
  https://github.com/sail-sg/inceptionnext
- Reviewed repository commit:
  `3f9769c6b3fcf903d1dc2f436eddc6963cacb535`.
- Reviewed `models/inceptionnext.py` SHA-256:
  `aed1b0a9ac410d5d9db042b85afe23de2b799bc70110c349acf851688e9f7b88`.

The paper identifies memory access, rather than FLOPs alone, as the main cost
of large depthwise kernels. Its Inception depthwise mixer splits channels into
parallel identity, square, horizontal-band, and vertical-band branches. The
reported InceptionNeXt-T training throughput is `1.6x` ConvNeXt-T while its
ImageNet top-1 result is slightly higher. This method therefore tests a
surface-oriented mixer designed explicitly for operational efficiency.

## Distinction from closed routes

- This is not a reduced MogaNet retry. The closed Moga tokenizer applies
  serial low/middle/high-order paths, gating, decomposition, and 16 blocks; it
  failed the locked runtime gate at `1.9983x` control. InceptionNeXt-Atto uses
  ten official blocks and parallel channel-split spatial paths.
- This is not FasterNet PConv. Every block retains an identity channel group
  plus three explicit spatial scales/orientations; the hypothesis concerns
  directional mango-surface evidence, not only partial-channel throughput.
- This is not a stock backbone substitution. The candidate ends at the exact
  `16x16` TRKH patch grid and retains all eight existing Transformer blocks,
  prefix/register/branch tokens, pruning, heads, and training losses.
- This does not reopen rejected large-kernel, EdgeNeXt, MBConv, CoAtNet,
  MobileViT, EfficientFormer, local-patch-mixer, frequency-expert, or output-
  rule families. It introduces only the official InceptionNeXt-Atto tokenizer.

## Locked inputs and constraints

- Data: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, immutable raw files,
  SHA-256 `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Scratch config source:
  `runs/full_v8_yolof_randominit_30e_20260714_105524/checkpoints/best.pt`,
  SHA-256
  `f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549`.
- Source launcher:
  `runs/full_v8_yolof_randominit_30e_20260714_105524/launcher_args.json`,
  SHA-256
  `a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6`.
- Candidate and control initialize from scratch at seed `42`; checkpoint
  weights are not loaded and no pretrained access is allowed.
- Image size `256`, patch grid `16x16`, TRKH embedding `256`, depth `8`, heads
  `8`, source prefix tokens, pruning, bbox prior, and source heads remain fixed.
- Validation is opened only for one matched smoke after Stage A. Test and the
  current-best command file remain closed.

## Locked candidate architecture

Add default-off `stem_architecture=inceptionnext_atto_tokenizer` and reproduce
the first three stages of the official `inceptionnext_atto` definition:

1. Stem: `4x4`, stride-4 convolution, `3 -> 40`, followed by BatchNorm.
2. Stage 1: width `40`, depth `2`, no further downsampling, output `64x64`.
3. Stage 2: BatchNorm then `2x2`, stride-2 convolution, `40 -> 80`, depth `2`,
   output `32x32`.
4. Stage 3: BatchNorm then `2x2`, stride-2 convolution, `80 -> 160`, depth `6`,
   output `16x16`.
5. Preserve TRKH `PatchEmbedding`; it becomes a `1x1`, `160 -> 256`
   projection. Existing CNN residual fusion consumes the actual stem width
   `160`; the control remains width `256` and bit-identical.

Every one of the ten blocks is locked to official Atto settings:

- channel split ratio `0.25`, producing equal identity, square, horizontal,
  and vertical groups at widths `40/80/160`;
- depthwise square kernel `3x3`;
- depthwise band kernels `1x9` and `9x1`;
- concatenate the four branches without an extra spatial projection;
- BatchNorm, `1x1` Conv MLP ratio `4`, GELU, `1x1` projection;
- residual layer scale initialized to `1e-6`;
- dropout and drop path `0`;
- official truncated-normal convolution initialization with standard
  deviation `0.02`, zero biases, and default BatchNorm initialization;
- no dependency on the external repository or timm at runtime.

No depth, width, branch ratio, kernel, normalization, activation, MLP ratio,
layer-scale, drop-path, projection, or memory-layout sweep is permitted.

## Stage A: train-only functional and resource gate

Before validation or an image-model smoke, require every check:

1. Explicit and implicit default `conv_pool` models have identical state names
   and values at seed `42`.
2. Candidate state initialization is deterministic and strict checkpoint
   round-trip succeeds without pretrained access.
3. Candidate maps `B x 3 x 256 x 256` to `B x 160 x 16 x 16`, yields exactly
   `256` patch tokens and finite five-class logits.
4. Exact depths `2/2/6`, ten blocks, channel split `10/10/10/10`,
   `20/20/20/20`, and `40/40/40/40`, kernel `3/9/9`, MLP ratio `4`, and layer
   scale `1e-6` are present. Candidate has at most `10M` parameters and adds at
   most `3M` over control.
5. FP32 and CUDA AMP forward/backward are finite. Stem/downsampling,
   square/horizontal/vertical depthwise branches, block normalization, both
   MLP projections, every layer scale, and final TRKH patch projection receive
   finite nonzero gradients.
6. Every stage output has nonzero variance. Identity, square, horizontal, and
   vertical first-block branches have nonzero energy and distinct signatures.
   A balanced one-example-per-class real train batch has finite nonzero
   true-class logit sensitivity for all five classes.
7. Matched batch-32 AMP forward/backward uses three post-warmup repetitions,
   peaks below `7.75 GiB`, and has candidate/control median runtime ratio at
   most `1.75x`. Validation/test loaders are not constructed, and source/data
   hashes plus split provenance are recorded.

Passing Stage A authorizes exactly one matched smoke. It is not promotion or
full-train evidence.

## Stage B: one matched precision smoke

Run scratch control `conv_pool` and candidate
`inceptionnext_atto_tokenizer` from the locked source arguments with:

- seed `42`, batch `32`, gradient accumulation `2`;
- `120` train batches per epoch, `2` epochs, scheduler horizon `15`, warmup `1`;
- AdamW LR `5e-4`, minimum LR `1e-6`, weight decay `0.05`, patience `3`;
- existing balanced sampler, LDAM-focal, teacher-focus-binary, metric,
  pairwise, pruning, bbox-prior, and augmentation recipe unchanged;
- attention-view crop/drop supervision disabled in both variants to isolate
  the tokenizer and bound runtime;
- full `yolo_f/val=2606`; final test skipped.

Candidate advances only if all gates pass:

- exact support and finite normalized probabilities;
- macro F1 no worse than control by more than `0.003`;
- class-1 F1 gain at least `0.010`;
- class-1 precision gain at least `0.025`;
- class-1 recall at least `control - 0.020`;
- total `0/2/4->1` false positives decrease by at least `5`;
- class-1 TP breaks are no more than FN rescues plus `3`;
- corrections exceed harms, new `3->2` harms are at most `5`, and no
  nonfocus class F1 falls by more than `0.020`;
- runtime is at most `1.50x` control and peak VRAM remains below `7.75 GiB`.

After independent reload, always run confusion/transitions, calibration,
architecture trace, native attention, rollout, Grad-CAM, changed-case paired
XAI, and clean/dim/bright/low-contrast/center-occlusion robustness. At least
three of five robustness conditions must preserve or improve macro F1, no
condition may reduce class-1 recall by more than `0.05`, stage Grad-CAM must
remain fruit/surface-focused, and attribution provenance must have no silent
fallback.

## Advancement boundary

A complete Stage-B pass authorizes one fixed five-epoch continuation. It must
reach macro/class-1 F1 `0.887/0.70`, class-1 precision `0.65`, recall `0.72`,
the same transition gates, and at least four favorable source bootstrap folds
before any full train is considered.

Failure at any stage closes this exact InceptionNeXt-Atto tokenizer without a
nearby architecture, optimizer, LR, seed, loss, pruning, augmentation,
checkpoint, or run-length sweep. No readiness, smoke, oracle, ensemble, or
test-informed result may update current-best commands.

## Executed result and closure

Stage A passed every train-only functional and resource check at
`runs/audit_inceptionnext_atto_stage_a_20260715`:

- summary SHA-256
  `8b1b4e6d1485bfa4e5224c432a769f7ccdb91ef5a15a345a4aa9a9f6b6ff41a5`;
- artifact-manifest SHA-256
  `62c80bc99d4dc6dbd58ad6aefa862ecbde48d1d382551735ac984b486ac4108f`;
- control/candidate parameters `7,245,590/8,298,198`, or `1,052,608` added;
- candidate peak `2.0435 GiB` and candidate/control median runtime `0.8710x`;
- exact architecture, deterministic schema/round-trip, FP32/BF16 gradients,
  branch activity, five-class sensitivity, and no validation/test loader all
  passed.

The single authorized matched smoke completed on full `yolo_f/val=2606` at
`runs/audit_inceptionnext_atto_matched_smoke_pair_20260715`. Pair summary
SHA-256 is
`163d4d1c2ba522712a1e7f16f9d89d5b9b82114e0760781ccb8f08694ba0d8bb`.
Independent control/candidate macro F1 was `0.768491/0.732042`; class-1
P/R/F1 was `0.357143/0.629139/0.455635` versus
`0.330049/0.443709/0.378531`. Candidate runtime was `1.02565x` control.

Candidate reduced locked `0/2/4->1` false positives `168 -> 134`, but this was
not selective precision improvement. It changed `277` decisions with
`77/144` corrections/harms, rescued/broke class-1 FN/TP `5/33`, and created
`20` new `3->2` harms. All four nonfocus class F1 values also fell; the largest
drops were class 4 `0.0401` and class 2 `0.0350`. Eight material metric gates
failed, including macro F1, class-1 precision/F1/recall, TP preservation, net
corrections, `3->2`, and nonfocus preservation.

The mandatory post-smoke closure completed at
`runs/audit_inceptionnext_atto_matched_smoke_pair_20260715/postsmoke_audit`.
Summary SHA-256 is
`66e47a4d8a26230fdcb261171748edcfebb6b3ccfd8317db6de64f56877f2841`.
Candidate won `0/5` robustness macro-F1 comparisons. Clean, center occlusion,
dim, bright, and low-contrast macro deltas were
`-0.03644/-0.04020/-0.04542/-0.00388/-0.02313`; worst class-1 recall delta was
`-0.36424` under bright illumination. Under dim illumination candidate recall
rose, but class-1 FP expanded to `281` and precision fell to `0.24259`; under
bright illumination candidate TP collapsed to `45` and recall to `0.29801`.

Paired all-method XAI used `16` locked changed cases, native last-block
`.mhsa_probability`, all eight valid grad-rollout layers, and Stage-3 tokenizer
output Grad-CAM with zero fallback. Paired summary SHA-256 is
`69338aa0f7929f7c3bc1ea37b774456f29bbd985c9d6fec9f9500c015269cae9`.
Candidate attention remained foreground-focused (`0.92169`) and Stage-3
Grad-CAM foreground mass was `0.94754`, but Grad-CAM border mass rose by
`0.07774` versus control. Object-desaturation causal drop was `0.09399` while
maximum background gray/blur drop was only `0.00164`. Manual review found
broader color/surface activation on class-1 TP breaks and new `3->2` harms,
plus healthy-edge/background activation on some severe-damage FP additions.

This exact tokenizer is therefore closed without a five-epoch continuation,
full train, test evaluation, or hyperparameter sweep. Current-best train,
export, video, and audit commands remain unchanged. Its two rejected smoke
roots were compacted into `324` verified nonbinary payloads at manifest SHA
`de69c4649a25eefd3daaadbe83087dc5d59375ff3c2e978a659180902a76e443`;
post-cleanup retention passed over `649` run directories with no blocker.
