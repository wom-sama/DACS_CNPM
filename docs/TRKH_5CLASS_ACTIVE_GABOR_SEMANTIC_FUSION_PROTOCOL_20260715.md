# TRKH 5-Class Active Gabor-Semantic Fusion Protocol (2026-07-15)

## Research question

Can an actively trained object-texture branch complement the unchanged
wide-context TRKH semantic path and improve class-1 precision without trading
away class-1 true positives?

Primary sources:

- *Learning Gabor Texture Features for Fine-Grained Recognition*, ICCV 2023:
  https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Learning_Gabor_Texture_Features_for_Fine-Grained_Recognition_ICCV_2023_paper.html
- Accepted paper PDF:
  https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Learning_Gabor_Texture_Features_for_Fine-Grained_Recognition_ICCV_2023_paper.pdf
- Official supplement:
  https://openaccess.thecvf.com/content/ICCV2023/supplemental/Zhu_Learning_Gabor_Texture_Features_for_Fine-Grained_Recognition_ICCV_2023_supplemental.pdf

The paper uses a full-image semantic branch, selects informative regions for a
texture branch, applies constrained learnable Gabor filters followed by LHO and
FCM, and adds all texture features directly to the semantic feature before the
final fully connected layer. Its published training starts both branches
together; it does not insert a zero-initialized scalar residual gate.

No author implementation was found. This protocol therefore locks an
equation-traceable TRKH adaptation rather than claiming an exact reproduction.

## Why this is a new hypothesis

The closed experiment in
`TRKH_5CLASS_LEARNABLE_GABOR_TEXTURE_READINESS_PROTOCOL_20260715.md` loaded the
keeper, multiplied the texture token by a zero-initialized scalar, and injected
the result into one edge token before the Transformer. After two low-LR epochs,
the effective gate was only `2.01e-6`, the residual norm was `3.21e-5`, and
filter features remained nearly identical. That experiment rejects the
zero-gated keeper adapter. It does not test an active branch trained jointly
from initialization and added at the classification feature, as in the paper.

This protocol changes only that scientific variable:

- the semantic path still receives the entire `yolo_f` frame;
- the texture path still uses the transformed object bbox and interior mask;
- the same constrained `32`-filter, `16/16` low/high, eight-level LHO/FCM
  encoder is retained;
- the texture feature is active on the first optimization step and is added
  directly to the final pooled semantic feature before the main classifier;
- control and candidate are both trained from initialization.

This is not another zero-gate, gate-scale, keeper-LR, target-token, texture-logit
expert, post-hoc router, or raw-data experiment.

## Locked inputs and provenance

- Data: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Scratch-reference full-run launcher:
  `runs/full_v8_yolof_randominit_30e_20260714_105524/launcher_args.json`,
  SHA-256
  `a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6`.
- Scratch-reference resolved configuration:
  `runs/full_v8_yolof_randominit_30e_20260714_105524/resolved_config.json`,
  SHA-256
  `c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97`.
- Scratch-reference checkpoint, used only as protected provenance and an
  absolute comparison after validation:
  `runs/full_v8_yolof_randominit_30e_20260714_105524/checkpoints/best.pt`,
  SHA-256
  `f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549`.
- Current keeper checkpoint, used only as an absolute promotion boundary:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Current VS Code command packet:
  `docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt`, SHA-256
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`.
- Seed `42`, five classes, image size `256`, no pretrained weights. Raw images,
  labels, split membership, and bboxes are immutable.
- Stage A may construct `yolo_f/train` only. Stage B may open full validation
  once after Stage A passes. Test remains closed.

## Locked architecture

Add default-off model option `learnable_gabor_texture_semantic_fusion`.

### Semantic path

Keep the scratch V8 CNN stem, patch embedding, branch tokens, all eight
Transformer blocks, token pruning, fine-grained pooling, CNN logit fusion,
bbox-spatial head, pairwise routing, teacher-focus-binary loss, sampler,
augmentation, and all other losses unchanged. This path continues to see the
full padded `yolo_f` crop and therefore retains object/background context.

### Texture path

Reuse the equation-traceable encoder locked by the prior protocol:

- denormalize RGB, convert only the texture path to grayscale, resize to
  `64x64`, and apply a differentiable transformed-bbox interior mask with
  margin `0.06`;
- use mask-weighted mean/variance normalization and explicit full-frame
  fallback when bbox metadata is invalid;
- use `32` complex quadrature Gabor filters with exact `16/16` low/high
  frequency split, constrained `theta/sigma_x/sigma_y/W`, zero-mean unit-L2
  kernels, and `11x11` support;
- pool magnitudes to `16x16`, apply eight-level differentiable LHO with spatial
  encoding, then four-head FCM over filter descriptors;
- project the final `64`-D texture descriptor to the TRKH width `256` and apply
  its existing token normalization.

The active encoder must not contain an unused residual gate. The old
`learnable_gabor_texture_residual` and new active fusion are mutually exclusive.

### Direct semantic fusion

Let `M` be the final TRKH head input after register/branch pooling,
fine-grained pooling, and any enabled late pooling. Let `T` be the normalized
texture feature. The candidate uses exactly:

`H = M + T`

and sends `H` to the unchanged main classification head. There is no scalar
gate, learned blend, fixed scale, concat projection, auxiliary Gabor
classifier, class-1 bias, threshold, or router. Existing CNN/bbox/pairwise
logit adjustments remain after the main head exactly as in the control.

Trace `M`, `T`, `H`, their per-sample norms and cosine, bbox fallback, mask,
low/high responses, LHO counts/entropy, filter features, constrained
parameters, and the final texture descriptor.

Construct every randomized optional layer inside a restored CPU RNG context
and exclude it from the parent's ordinary recursive initialization. With seed
`42`, all shared control/candidate tensors and the complete post-constructor CPU
RNG state must be bit-identical.

No fusion scale, normalization, filter count, histogram level, hidden width,
kernel, crop, bbox margin, color conversion, LR, seed, loss, augmentation, or
run-length sweep is permitted.

## Stage A: train-only readiness

Stage A must pass every check before validation is allowed:

1. Verify every locked input hash and construct exactly `9215` train objects.
   Prove all sampled paths resolve below `yolo_f/images/train`; validation and
   test loaders must not be constructed.
2. Explicit default-off and implicit legacy configs have identical ordered
   state schema, tensor values, post-constructor RNG bytes, and logits.
3. Same-seed scratch control and candidate have bit-identical shared tensors
   without loading one model into the other. Candidate-only keys must all begin
   with `gabor_texture_semantic_encoder.`. Repeated candidate construction and
   checkpoint round-trip must be exact.
4. The active candidate must differ from control immediately: texture and
   fused features are finite/nonconstant, main-head logits change by more than
   `1e-6`, and no scalar gate exists. Mean `||T||/||M||` must be in
   `[0.50, 3.00]`; mean `||M+T||/||M||` must be in `[0.75, 3.00]`.
5. All Gabor constraints, exact low/high split, kernel normalization, response
   activity, LHO count normalization, entropy, bbox concentration, bbox/full-
   frame sensitivity, mechanism ablations, and materialized-kernel equivalence
   from the prior Stage A must still pass.
6. FP32 and CUDA BF16 main classification loss must deliver finite nonzero
   gradients to every Gabor parameter, LHO projection/attention, FCM parameter
   projection/attention, output projection, both real/imaginary responses, the
   main classifier, and at least one shared semantic-path family.
7. Run exactly `16` deterministic FP32 AdamW updates on one fixed real-train
   balanced batch containing one object from each class. Use the full-run LR
   `2.5e-4`, weight decay `0.05`, no augmentation resampling, and update the
   complete candidate. This is a mechanism audit; its weights are discarded.
   Require finite loss, at least `10%` loss reduction, movement in every Gabor
   gradient family, nonzero raw-filter movement, and a material texture
   contribution after the final step. Mean off-diagonal texture-token cosine
   or mean off-diagonal filter-feature cosine must decrease by at least
   `1e-4`; otherwise the branch is still representation-collapsed and Stage A
   fails.
8. All five balanced classes must retain nonzero descriptor and true-class
   logit sensitivity. Fixed dim/bright/low-contrast views must have mean
   descriptor cosine at least `0.90` and no class-condition below `0.85`.
9. Added parameters must be at most `100,000`. Matched batch-32 CUDA BF16
   forward/backward uses three post-warmup repetitions, peaks below
   `7.75 GiB`, and has median candidate/control runtime ratio at most `1.50x`.
10. Dynamic and materialized Gabor paths must agree within `1e-6`. ONNX Runtime
    CPU must reproduce the materialized active texture-plus-semantic sum for
    dynamic batches `1` and `2` within `1e-5`; failure is not waived.

Passing Stage A authorizes exactly one Stage-B pair. It does not authorize a
probe, full run, test, or command promotion.

## Stage B: one deterministic scratch pair

Create control and candidate from the locked scratch launcher arguments. Do not
resume any checkpoint. Both runs use:

- seed `42` plus `--deterministic`;
- batch `32`, gradient accumulation `2`, `120` train batches per epoch;
- `5` epochs, scheduler horizon `30`, warmup `1`, patience `3`;
- AdamW LR `2.5e-4`, minimum LR `1e-6`, weight decay `0.05`;
- transformed bbox source `crop_bbox`;
- full `yolo_f/val=2606` by independent reload and `--skip-final-test`.

The candidate differs only by
`--learnable-gabor-texture-semantic-fusion`. The launcher must compare the
fully normalized argument arrays after removing only that flag and replacing
the run name placeholder. Existing artifacts may never be overwritten.

The candidate advances only if every relative precision gate passes:

- aligned support `2606`, finite normalized probabilities, and no test access;
- macro F1 at least control `+0.005`;
- class-1 precision at least control `+0.020` and at least `0.50`;
- class-1 recall no more than `0.020` below control;
- class-1 F1 at least control `+0.010`;
- remove at least eight control `0/2/4->1` false positives, create fewer such
  false positives than it removes, and retain at least `95%` of control
  class-1 true positives;
- corrections exceed harms, class-1 FN rescues are at least TP breaks, new
  `3->2` harms are at most `3`, and no nonfocus class F1 falls by more than
  `0.015`;
- candidate runtime at most `1.50x` control and peak VRAM below `7.75 GiB`.

These are early-convergence gates, not promotion gates. Even a complete Stage-B
pass cannot replace the current keeper or change the full-train command.

## Mandatory post-smoke audit

After both independent reloads, always run before deciding continuation:

- confusion matrix, per-class calibration, transitions, class-1 boundary and
  source-group forensics;
- architecture trace with one sample per class;
- clean/center-occlusion/dim/bright/low-contrast robustness, requiring at least
  three of five macro wins and no condition with more than two additional
  class-1 TP losses versus control;
- native Transformer attention, ordinary and gradient rollout, stem Grad-CAM,
  exact changed-case paired XAI, and object/background perturbations with
  explicit AMP-to-FP32 reconciliation;
- Gabor-specific parameter movement, low/high response energy, bbox/full-frame
  counterfactual, LHO entropy, FCM attention entropy, texture-token and
  filter-feature diversity, semantic/texture/fused norm ratios, and direct
  texture-logit contribution;
- attribution fallback and backend provenance must be explicit.

If the candidate fails Stage B, complete these audits anyway, record the
mechanism, compact rejected checkpoint binaries only after hash-verified
evidence preservation, and close this exact route without a nearby sweep.

## Continuation and promotion boundary

A complete Stage-B pass authorizes one locked deterministic ten-epoch scratch
probe using the same recipe, not a continuation from validation-selected
weights. The probe must reach all of:

- macro/class-1 F1 at least `0.887/0.70`;
- class-1 precision at least `0.65`, recall at least `0.775`, and at least
  `117/151` class-1 TP;
- the same transition, robustness, XAI, and mechanism gates;
- at least four of five source-group bootstrap wins.

Only then may a capped 30-epoch run with patience `3` be considered. Update
`scripts/run_trkh_current_best_full_pipeline.ps1` and
`docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt` only after an
independently reloaded single checkpoint beats the current keeper on every
locked validation gate. Smoke, probe, ensemble, oracle, post-hoc, or test-aware
evidence cannot promote commands. Export and video commands remain separate.
