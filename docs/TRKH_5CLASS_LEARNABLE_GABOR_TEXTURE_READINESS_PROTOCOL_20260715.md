# TRKH 5-Class Learnable-Gabor Texture Residual Protocol (2026-07-15)

## Research question

Can a compact, constrained learnable-Gabor branch add interior surface evidence
to the existing keeper without replacing its semantic CNN/Transformer path,
thereby reducing class-1 false positives while preserving every keeper class-1
true positive under illumination changes?

Primary sources:

- *Learning Gabor Texture Features for Fine-Grained Recognition*, ICCV 2023:
  https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Learning_Gabor_Texture_Features_for_Fine-Grained_Recognition_ICCV_2023_paper.html
- Accepted paper PDF:
  https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Learning_Gabor_Texture_Features_for_Fine-Grained_Recognition_ICCV_2023_paper.pdf
- Official supplement:
  https://openaccess.thecvf.com/content/ICCV2023/supplemental/Zhu_Learning_Gabor_Texture_Features_for_Fine-Grained_Recognition_ICCV_2023_supplemental.pdf
- Local deep-research reports reviewed before this protocol:
  `D:/deep-research-report (14).md`, SHA-256
  `561ba5a677cca7a2a52ec100ea1aac91851f40760f746a4153238dfe94374d7f`;
  `D:/deep-research-report (15).md`, SHA-256
  `7933984f468c90a394e5642088c56da09518c8b88e66cfa5a64f38aa46a7d4b0`.

The paper learns constrained Gabor parameters, separates low/high-frequency
filters, summarizes each response with a Learnable Histogram Operator (LHO),
and models relations among filters with a Filter Correlation Module (FCM). Its
ablations report losses of `4.3` points for fixed/manual Gabor parameters,
`2.7` without frequency separation, `3.8` without LHO, `1.6` without FCM, and
`2.9` when LHO is replaced by handcrafted mean/variance/max/min statistics.
The supplement reports that `N=32` remains useful and chooses `M=8` histogram
levels. No author-provided implementation was found. This work is therefore an
equation-traceable compact adaptation, not an exact code reproduction.

## Why this route is admissible

- The current single-checkpoint keeper is precision-limited but remains the
  strongest reproducible no-pretrain TRKH model. Independent `yolo_f/val`
  macro/class-1 F1 is `0.882925/0.678261`; class-1 precision/recall is
  `0.603093/0.774834`, corresponding to `117/151` true positives.
- The 2026-07-14 random-initialized full model is a recall complement, not a
  replacement: macro/class-1 F1 `0.874172/0.654639`, class-1 P/R
  `0.535865/0.841060`, and `33` more class-1 false positives than the keeper.
- Repeated XAI shows that object color, surface, illumination, and near-boundary
  evidence dominate. Background blur/gray is weak. The wide `yolo_f` frame is
  still retained by the unchanged semantic path; only the new texture path is
  object/interior weighted.
- This is not the rejected high-frequency logit expert, fixed Gabor/Sobel/LBP/
  wavelet statistic, replacement tokenizer, post-hoc router, or output
  calibrator. The new descriptor is a small residual on an existing edge token
  and is interpreted jointly with wide-context patch tokens by all eight
  existing Transformer blocks.

## Locked inputs and provenance

- Data: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, immutable raw files,
  SHA-256 `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Scratch complement, retained only as a diagnostic reference:
  `runs/full_v8_yolof_randominit_30e_20260714_105524/checkpoints/best.pt`,
  SHA-256
  `f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549`.
- Current VS Code command packet:
  `docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt`, SHA-256
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`.
- Seed `42`, image size `256`, five classes, no pretrained weights, existing
  CNN stem, patch embedding, branch tokens, eight Transformer blocks, pruning,
  bbox prior, heads, sampler, loss, and augmentation recipe remain unchanged.
- Stage A may construct a train loader only. Stage B may open full validation
  exactly once after Stage A passes. Test remains closed.

## Locked compact adaptation

Add default-off `learnable_gabor_texture_residual` after existing branch-token
fusion and before positional dropout/Transformer processing.

### Object-normalized texture input

1. Keep the normalized full image unchanged for the semantic path.
2. For the texture path only, convert the denormalized RGB image to grayscale.
3. Build a differentiable soft mask from normalized `cx,cy,w,h` bbox metadata,
   with an interior margin of `0.06` of bbox width/height and a small soft edge.
   The sigmoid edge softness is exactly one `64x64` analysis pixel. Use the
   transformed `crop_bbox` for `yolo_f`; if bbox metadata is unavailable, use
   a full-frame mask and record fallback.
4. Compute mask-weighted mean and variance, then z-score grayscale values with
   an epsilon floor. Set the normalized outside-mask response to zero. This is
   the locked illumination-control mechanism; no CLAHE or raw-file edit occurs.
5. Resize the texture input and soft mask to `64x64`. This preserves the
   paper supplement's relative analysis scale (`112/448 = 64/256`).

### Constrained Gabor bank

- Use `N=32` complex quadrature filters, split equally into `16` low- and `16`
  high-frequency filters; response is real/imaginary magnitude.
- Use finite support `K=11`. Raw `theta`, `sigma_x`, `sigma_y`, and frequency
  `W` are learned through `lower + (upper-lower)*sigmoid(raw)`.
- Lock `theta` to `[0, pi]`, `sigma_y` to `[5/(2*pi), K/5]`, and
  `sigma_x` to `[5/(2*pi*(1-2W)), K/5]`.
- Lock frequency to `[0, W_max/2]` for the first 16 filters and
  `[W_max/2, W_max]` for the final 16, where
  `W_max=(2*pi*K-25)/(4*pi*K)`. Initialize each half at evenly spaced interior
  quantiles, orientations evenly across `[0, pi)`, and scales inside their
  valid intervals. Normalize every real/imaginary kernel to zero mean and unit
  L2 norm before convolution.
- Pool response magnitudes to a fixed `16x16` grid and apply the resized soft
  mask. No learned convolution may replace the Gabor equations.

### LHO and FCM

- Use `M=8` levels per filter between its masked response minimum and maximum.
  Apply differentiable triangular assignment
  `relu(1 - abs(response-level)/bin_width)` and normalize level counts to sum
  to one.
- For every level, concatenate normalized count and level, project `2 -> 64`,
  add a weighted two-dimensional sine/cosine position descriptor, run one
  four-head self-attention layer across the eight levels, and average levels.
  Following the unnormalized weighted sum in Eq. 8, divide this descriptor
  only by the fixed `16*16` response area for scale control. Do not divide by
  input-dependent per-bin mass: near-empty bins would receive the same weight
  as occupied bins and amplify backend rounding noise.
- Project the four constrained filter parameters `4 -> 64`, add them to each
  LHO filter descriptor, run one four-head self-attention layer across the 32
  filters, and average filters. Project the resulting `64`-D descriptor to the
  existing `256`-D TRKH token width.
- This follows the paper's LHO/FCM equations and ordering while adapting region
  selection, spatial encoding, dimensions, and filter count to this 256-pixel
  task. These departures must be recorded in every audit artifact.

### Residual fusion

- Require the current multi-branch layout to contain at least one edge token.
  Add the texture descriptor only to the first existing edge token, whose
  index is `color_token_count` within `branch_tokens`.
- Use scalar gate `0.10*tanh(raw_gate)`, with `raw_gate=0` at initialization.
  The candidate must therefore reproduce keeper logits before optimization.
- Construct randomized Gabor projections/attention inside a CPU RNG fork. Their
  initialization remains deterministic for the locked seed, but constructing
  the optional branch must restore the caller RNG exactly so the matched control
  and candidate receive the same later dropout, sampler, and augmentation stream.
- Do not add a Gabor classifier, class-1 logit bias, decision router, auxiliary
  label head, extra token, or threshold. The unchanged Transformer/head must
  decide whether the complementary texture descriptor is useful.
- Trace bbox mask/fallback, constrained parameters, low/high response energy,
  LHO counts/entropy, per-filter features, texture descriptor, and gate value.

No filter-count, level-count, crop scale, kernel, mask margin, hidden width,
attention-head, gate scale, target token, normalization, initialization, LR,
seed, loss, augmentation, or run-length sweep is permitted.

## Stage A: train-only readiness gate

Before validation or a smoke, require every check:

1. Verify data, keeper, scratch-reference, and current-command hashes. Construct
   only `yolo_f/train=9215`, with one real sample per class for sensitivity.
2. Explicit and implicit default-off models have identical ordered schema,
   tensor values, and logits. Loading the keeper into the enabled candidate may
   miss only new Gabor-module keys and must have no unexpected key. With the
   same seed, implicit control, explicit control, and enabled candidate must
   leave bit-identical CPU RNG state after construction; repeat candidate
   construction must also preserve that exact state.
3. At zero gate, candidate and independently loaded keeper logits must be
   bit-identical where practical and have max absolute error at most `1e-7`.
   Checkpoint round-trip must preserve schema, outputs, and gate.
4. All 32 filters remain finite and inside constraints with exact `16/16`
   low/high split. Kernels are noncollapsed/diverse; real and imaginary paths
   both change responses and response variance is nonzero.
5. LHO counts sum to one within `1e-5`, level entropy is noncollapsed, and FCM
   outputs are finite/nonconstant. Low-only, high-only, no-LHO-position, and
   no-FCM-parameter-encoding ablations must each materially change the
   descriptor; they are mechanism audits only and cannot select a model.
6. With a temporary small audit gate, FP32 and CUDA BF16 forward/backward are
   finite. Gate, every raw Gabor parameter family, real/imaginary response,
   LHO projection/attention, FCM parameter projection/attention, and final
   projection receive finite nonzero gradients. At the true zero gate, the
   first backward must give a finite nonzero gate gradient.
7. All five balanced real-train classes have finite nonzero descriptor and
   true-class-logit sensitivity. Response foreground mass, measured before
   applying the mask, must exceed the bbox mask area fraction; the bbox-aware
   descriptor must also differ materially from its full-frame counterfactual
   without collapsing.
8. Fixed clean/dim/bright/low-contrast views must retain descriptor cosine
   similarity of at least `0.90` on average, with no class below `0.85`.
9. Added trainable parameters are at most `100,000`. Matched batch-32 CUDA BF16
   forward/backward uses three post-warmup repetitions, peaks below `7.75 GiB`,
   and has candidate/control median runtime ratio at most `1.50x`.
10. Dynamic kernels and a materialized-kernel inference copy agree within
    `1e-6`. ONNX Runtime CPU output error must be at most `1e-5` if the exporter
    supports the graph; otherwise Stage A fails deployability rather than
    silently waiving export. Export a second diagnostic graph that exposes the
    intermediate texture, response, LHO, and FCM tensors so any backend error
    is localized rather than inferred from the final token alone.

Passing Stage A authorizes exactly one matched Stage-B smoke. It is not model
promotion, a probe, or evidence for changing the full-train command.

## Stage B: one keeper-initialized precision smoke

Run one control and one candidate initialized from the locked keeper with:

- seed `42`, batch `32`, gradient accumulation `2`;
- `120` train batches per epoch, `2` epochs, scheduler horizon `15`, warmup `1`;
- AdamW LR `8e-5`, minimum LR `1e-6`, weight decay and all base settings loaded
  from the keeper recipe; no Gabor-specific LR multiplier;
- the exact same sampler, loss, augmentation, pruning, bbox, and supervision
  settings in both runs; the candidate differs only by the enabled residual;
- full `yolo_f/val=2606`, independent reload, and final test skipped.

Candidate advances only if every absolute and matched gate passes:

- exact aligned support, finite normalized probabilities, and no test access;
- macro F1 at least keeper `0.882925` and no worse than matched control;
- class-1 precision at least `0.613093` and at least control `+0.010`;
- class-1 recall at least keeper `0.774834`, with at least `117/151` TP and no
  net keeper TP loss;
- class-1 F1 at least `0.683261` and at least control `+0.005`;
- remove at least four total `0/2/4->1` keeper false positives;
- corrections exceed harms, FN rescues are at least TP breaks, new `3->2`
  harms are at most `3`, and no nonfocus class F1 falls by more than `0.010`;
- candidate runtime is at most `1.50x` control and peak VRAM is below
  `7.75 GiB`.

After independent reload, always run calibration, confusion/transitions,
boundary forensics, architecture trace, native attention, rollout, Grad-CAM,
changed-case paired XAI, and object/background perturbations. Add Gabor-specific
low/high response, bbox/full-frame, LHO entropy, filter-correlation, and
illumination counterfactual reports. At least three of five clean/center-
occlusion/dim/bright/low-contrast conditions must preserve or improve macro F1;
no condition may lose more than one keeper class-1 TP. Attribution fallback
must be explicit and cannot support a positive conclusion.

## Advancement and command boundary

A complete Stage-B pass authorizes one fixed five-epoch continuation only. It
must reach macro/class-1 F1 at least `0.887/0.70`, class-1 precision `0.65`,
recall `0.775`, retain at least `117` class-1 TP, pass the same transition/XAI/
robustness gates, and win at least four of five source-group bootstrap folds
before a full train is considered.

Any full run remains capped at 30 epochs with patience `3`. Update
`scripts/run_trkh_current_best_full_pipeline.ps1` and
`docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt` only after an
independently reloaded single checkpoint beats the current keeper on all locked
validation gates. Smoke, probe, ensemble, oracle, post-hoc, or test-informed
evidence cannot promote the command. Export/video commands remain separate.

Failure at Stage A or B closes this exact compact adaptation without a nearby
architecture, hyperparameter, seed, loss, checkpoint, or run-length sweep. Its
nonbinary evidence may be preserved and rejected checkpoint binaries compacted
only after full audit and hash verification.

## Observed closure 2026-07-15

- The first matched attempt was invalid for causal interpretation: constructing
  the optional randomized branch changed the post-constructor CPU RNG stream.
  Branch construction and the parent model's recursive initialization were
  moved inside RNG forks. The corrected Stage-A run then proved exact control/
  candidate post-constructor RNG equality with zero differing bytes.
- Corrected Stage A passed all 50 locked checks at
  `runs/audit_learnable_gabor_texture_stage_a_rngneutral_20260715`; summary SHA
  is `e53d8c1a...81e8f`. Runtime was `1.05603x`, peak CUDA memory was
  `2.58663 GiB`, and ONNX CPU maximum error was `8.34465e-7`.
- The sole RNG-neutral Stage-B pair completed over all `2606` validation
  objects with no test access. Control/candidate macro F1 was
  `0.885917 -> 0.884195`; class1 precision/recall/F1 was
  `0.613402/0.788079/0.689855 -> 0.600985/0.807947/0.689266`.
  Candidate versus control had `25/30` corrections/harms, removed/created
  `8/15` focus false positives, and created five new `3->2` harms. Pair summary
  SHA is `8adcae23...56bd3`; all five-epoch/probe/full/test permissions are
  false.
- The texture path was numerically starved rather than successfully learned.
  Effective gate magnitude was only `2.00668e-6`, residual norm was
  `3.21004e-5`, FCM normalized attention entropy was `0.99999946`, and maximum
  pairwise filter-feature cosine was `0.99999833`. The post-smoke robustness
  gate won only two of five macro conditions; summary SHA is
  `a6b26aae...edc2`.
- Paired XAI used exact FP32 forwards and native block-7 MHSA on all 16 cases,
  with zero attention or grad-rollout fallback. One near-tie control prediction
  changed between the AMP selection pass and FP32 XAI; the exact sample set was
  retained and reconciled with both predictions and categories preserved.
  Reconciliation/paired summary SHAs are `2593dba2...d0e` and
  `5c858fcd...bbf2`. Candidate Grad-CAM foreground mass fell by `0.039925` and
  border mass rose by `0.056445`; the four new focus-FP cases had border delta
  `+0.070231`.
- This closes the zero-gated, keeper-initialized, low-LR edge-token residual
  exactly as locked. It does not establish that the paper's direct active
  texture-plus-semantic fusion trained from initialization is ineffective.
  Such a route, if considered, requires a new protocol, active-gradient and
  filter-diversity gates, deterministic or replicated matched training, and no
  reuse of this Stage-B permission.
