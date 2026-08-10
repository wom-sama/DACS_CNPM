# TRKH 5-Class StarNet-S2 Tokenizer Protocol (2026-07-15)

## Research question

Can the first three official StarNet-S2 stages create local multiplicative
surface evidence before the existing TRKH Transformer, increasing class-1
precision while preserving true class-1 recall under illumination changes?

Primary sources:

- *Rewrite the Stars*, CVPR 2024:
  https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Rewrite_the_Stars_CVPR_2024_paper.html
- Accepted paper PDF:
  https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Rewrite_the_Stars_CVPR_2024_paper.pdf
- Official Apache-2.0 PyTorch repository:
  https://github.com/ma-xu/Rewrite-the-Stars
- Reviewed repository commit:
  `c999eb50840a44f9f1d92e8f7d2c22cd645a6d5e`.
- Reviewed `imagenet/starnet.py` SHA-256:
  `4e9eb1f58ea51427eebe17baf9891a3e1d73d000597c032a700176c954c3031a`.

The paper treats elementwise multiplication as an implicit expansion into a
higher-dimensional nonlinear feature space without widening the stored
feature map. Official StarNet-S2 reports `74.8` ImageNet-1k top-1 at `3.7M`
parameters, `547M` FLOPs, and `2.0 ms` P100 latency. Replacing every star with
summation lowers StarNet-S4 top-1 by `3.1` points; the reported contribution is
small in stages 1-2 but `1.6` points in stage 3. The official result uses 300
epochs, so it is architecture evidence only and does not imply that this TRKH
hybrid will meet the required `<=30`-epoch convergence target.

## Distinction from closed routes

- This is not the rejected pooled compact-bilinear head. The star operation is
  applied locally at every spatial position inside nine pre-Transformer
  blocks, before global pooling or token readout.
- This is not a MogaNet retry. It has no multi-order branch split, feature
  decomposition, channel aggregation, SiLU gate, or 16-block serial tokenizer.
- This is not an InceptionNeXt retry. It uses two learned pointwise projections
  and multiplicative interaction between depthwise convolutions rather than
  fixed identity/square/horizontal/vertical channel groups.
- This is not a wide-background suppression method. Existing evidence says
  object color, surface, illumination, and class boundaries dominate; StarNet
  is admitted only as a direct local representation change.
- This is not a stock-backbone replacement. The candidate ends at the exact
  `16x16` TRKH patch grid and retains all eight Transformer blocks, prefix
  tokens, pruning, heads, losses, and the no-pretrain constraint.

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
  `8`, source prefix tokens, pruning, bbox prior, source heads, losses, sampler,
  and augmentations remain fixed.
- Stage A may construct only the train loader. Validation is opened for one
  matched smoke only after every readiness gate passes. Test and the
  current-best command file remain closed.

## Locked candidate architecture

Add default-off `stem_architecture=starnet_s2_tokenizer` and reproduce official
StarNet-S2 stages 1-3 without `timm` or repository runtime dependencies:

1. Stem: `3x3`, stride-2 `Conv2d + BatchNorm`, `3 -> 32`, then `ReLU6`, output
   `128x128`.
2. Stage 1: `3x3`, stride-2 `Conv2d + BatchNorm`, `32 -> 32`, then one star
   block, output `64x64`.
3. Stage 2: `3x3`, stride-2 `Conv2d + BatchNorm`, `32 -> 64`, then two star
   blocks, output `32x32`.
4. Stage 3: `3x3`, stride-2 `Conv2d + BatchNorm`, `64 -> 128`, then six star
   blocks, output `16x16`.
5. Preserve TRKH `PatchEmbedding`; it becomes a `1x1`, `128 -> 256`
   projection. Existing CNN residual fusion consumes width `128`; the default
   control remains width `256` and state-identical.

Every one of the nine blocks is locked to the official S2 implementation:

- `7x7` depthwise `Conv2d + BatchNorm` at input width `C`;
- independent bias-enabled `1x1` projections `f1` and `f2`, without BatchNorm,
  from `C -> 4C`;
- exact interaction `ReLU6(f1(x)) * f2(x)`;
- bias-enabled `1x1` projection `g`, with BatchNorm, from `4C -> C`;
- final bias-enabled `7x7` depthwise convolution without BatchNorm;
- identity residual and drop path `0`;
- actual official runtime initialization: PyTorch `Conv2d.reset_parameters`,
  BatchNorm scale `1`/bias `0`; there is no layer scale or block activation
  after the residual;
- no classifier head, final StarNet BatchNorm, average pool, or pretrained
  checkpoint is imported.

The matched candidate uses star in all three stages. Paper ablation says the
largest contribution is stage 3, but selecting stage-specific sum/star variants
from local validation would be a sweep. The readiness audit may compute a
same-weight sum counterfactual for mechanism verification only; it cannot train
or select a sum model.

No depth, width, expansion, kernel, bias, normalization, activation,
interaction, drop-path, initialization, memory-layout, or stage-selection
sweep is permitted.

## Stage A: train-only functional and resource gate

Before validation or an image-model smoke, require every check:

1. Explicit and implicit default `conv_pool` models have identical ordered
   state names, tensor values, and five-class logits at seed `42`.
2. Candidate initialization is deterministic and strict state-dict round-trip
   succeeds without pretrained or external-repository access.
3. Candidate maps `B x 3 x 256 x 256` to `B x 128 x 16 x 16`, yields exactly
   `256` patch tokens and finite five-class logits.
4. Exact depths `1/2/6`, widths `32/64/128`, nine blocks, expansion `4`, two
   `7x7` depthwise convolutions, `ReLU6`, pointwise BN placement, and residual
   graph are present. Candidate has at most `10M` parameters and adds at most
   `3M` over control.
5. FP32 and CUDA BF16 forward/backward are finite. Stem/downsampling, every
   block's first depthwise path, `f1`, `f2`, `g`, second depthwise path, and the
   final TRKH patch projection all receive finite nonzero gradients.
6. Every stage output has finite nonzero variance. For each stage, `f1`, `f2`,
   activated `f1`, and the star product have nonzero energy; the product is not
   identical to either input branch.
7. With identical weights and block inputs, replacing `*` by `+` changes each
   audited stage output by a finite nontrivial amount. Stage-3 star-product
   gradients reach both `f1` and `f2`. This is mechanistic evidence, not model
   selection.
8. A balanced one-example-per-class real train batch has finite nonzero
   true-class logit sensitivity for all five classes and finite input
   gradients. Only train paths may appear in the provenance report.
9. Matched batch-32 CUDA BF16 forward/backward uses three post-warmup
   repetitions, peaks below `7.75 GiB`, and has candidate/control median runtime
   ratio at most `1.75x`. Validation/test loaders are not constructed; source,
   data, launcher, and checkpoint hashes are recorded.

Passing Stage A authorizes exactly one matched smoke. It is not promotion or
full-train evidence.

## Stage B: one matched precision smoke

Run scratch control `conv_pool` and candidate `starnet_s2_tokenizer` from the
locked source arguments with:

- seed `42`, batch `32`, gradient accumulation `2`;
- `120` train batches per epoch, `2` epochs, scheduler horizon `15`, warmup `1`;
- AdamW LR `5e-4`, minimum LR `1e-6`, weight decay `0.05`, patience `3`;
- existing balanced sampler, LDAM-focal, teacher-focus-binary, metric,
  pairwise, pruning, bbox-prior, and augmentation recipe unchanged;
- attention-view crop/drop supervision disabled in both variants to isolate
  the tokenizer and bound runtime;
- full `yolo_f/val=2606`; final test skipped.

Candidate advances only if all gates pass:

- exact support, exact aligned samples, and finite normalized probabilities;
- macro F1 no worse than control by more than `0.003`;
- class-1 F1 gain at least `0.010`;
- class-1 precision gain at least `0.025`;
- class-1 recall at least `control - 0.020`;
- total `0/2/4->1` false positives decrease by at least `5`;
- class-1 TP breaks are no more than FN rescues plus `3`;
- corrections exceed harms, new `3->2` harms are at most `5`, and no nonfocus
  class F1 falls by more than `0.020`;
- runtime is at most `1.50x` control and peak VRAM remains below `7.75 GiB`.

After independent reload, always run calibration, confusion/transitions,
boundary forensics, architecture trace, native attention, rollout, Stage-3
Grad-CAM, changed-case paired XAI, perturbations, and clean/dim/bright/
low-contrast/center-occlusion robustness. At least three of five robustness
conditions must preserve or improve macro F1; no condition may reduce class-1
recall by more than `0.05`; attribution must remain fruit/surface-focused with
no silent fallback. Precision gains produced mainly by TP removal fail.

## Advancement boundary

A complete Stage-B pass authorizes one fixed five-epoch continuation. It must
reach macro/class-1 F1 `0.887/0.70`, class-1 precision `0.65`, recall `0.72`,
the same transition gates, and at least four favorable source bootstrap folds
before any full train is considered.

Full training remains capped at 30 epochs with patience `3`. A full candidate
may update the current-best command only after independent locked validation
beats the deployed single-checkpoint keeper and all precision/TP/robustness
gates. Smoke, probe, oracle, ensemble, or test-informed evidence cannot promote
a command. Test stays final-only after promotion is already frozen.

Failure at any stage closes this exact StarNet-S2 tokenizer without a nearby
depth, width, operation, stage, LR, optimizer, seed, loss, augmentation,
checkpoint, pruning, initialization, or run-length sweep.

## Stage-A execution

The locked train-only audit passed at
`runs/audit_starnet_s2_stage_a_20260715`:

- summary SHA-256
  `7fdb01920a5d8c03324194efd35d2fd394e516c7e89ca9f4df3233f5624ddcaa`;
- artifact-manifest SHA-256
  `8e8710f9cc138ecfcc44143084065e6e24fd1a2ed7b49d49af98e3621cd633fb`;
- control/candidate parameters `7,245,590/8,345,974`, added `1,100,384`;
- candidate peak CUDA memory `2.12805 GiB`;
- control/candidate median batch-32 BF16 forward/backward time
  `0.1441955/0.1550486 s`, ratio `1.075267x`;
- all nine star blocks had live FP32 and BF16 gradients in both pointwise
  branches and both depthwise paths; minimum same-weight star-vs-sum output
  delta was `0.022263`;
- all five balanced real-train class sensitivities passed; validation and test
  loaders were not constructed.

Stage A grants exactly one matched Stage-B smoke. It does not grant a probe,
full train, test evaluation, or current-best command change.
