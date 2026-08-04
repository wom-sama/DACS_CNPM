# B16 SwiftSurface-XS prospective protocol — 2026-08-05

Protocol ID: `TRKH_PRETRAINED_CLASSF_B16_SWIFTSURFACE_XS_20260805`

This protocol is frozen before asset acquisition, synthetic preflight, TRAIN
access or any B16 metric. The uncommitted implementation draft is under review,
has no execution authority and must conform to this document. The protocol
itself does not authorize execution.

## Question and claim boundary

B16 asks whether a mobile SwiftFormer-XS can retain class-1 surface-boundary
evidence when it is trained against a **spatial, projector-free** relation field
from raw externally pretrained DINOv3-S and preserves a local residual at its
`14x14 -> 7x7` transition. The matched control has the same student source,
relation target/loss, trainable tensors and byte-identical branch initialization,
but replaces the local residual by its spatial mean before fusion. Therefore the
candidate-versus-control contrast tests preservation of spatial residual structure,
not the existence of the teacher, extra capacity or a different auxiliary loss.

The ordinary pretrained SwiftFormer-XS arm has neither branch nor relation loss.
It is a practical anchor only, not a capacity-matched causal control. A B16 pass
does not show that hybrids beat ViTs, that DINO semantics are causal, or that the
model generalizes. It permits only a separately frozen validation/quantization
protocol. Validation and test remain closed throughout B16.

## Immutable sources and TRAIN boundary

- Worktree/branch: `D:\DataAI\AIEx\TRKH_pretrained`,
  `research/pretrained-classf-b1`; formal work requires one clean committed HEAD.
- Student ID is the unambiguous timm/Hugging Face artifact
  `timm/swiftformer_xs.dist_in1k`, ImageNet-1K author-trained, Apache-2.0. Lock HF
  revision `ac0196f198e58c82183a67f5f5c0952421b3e6ac` and
  `model.safetensors`, exactly `13,957,760` bytes, SHA256
  `c0dfb2e99069ece07b691ff470f5dafed98716d90be738369828693c3bf2071c`.
  Runtime is timm `1.0.27`; preflight must hash the exact SwiftFormer source and
  factory used. Load the 1000-class artifact strictly with safetensors before
  resetting both distilled heads inside `torch.random.fork_rng` seed `20260805`
  to `Linear(220,5)`, timm `trunc_normal_(std=0.02)`, bias zero. Copy that exact
  complete initialized student state into every arm and fold. Pickle and
  ambiguous bare model aliases are forbidden.
- Expected timm `1.0.27` runtime source hashes, which preflight must recompute,
  are `swiftformer.py` SHA256
  `e7f79c9bfb3750636ada4cd776c9ab9da41d321ba56ada28201a7dc480b654a7`,
  `_factory.py` SHA256
  `30a6eecdaba750af470cfae3196186fd647052c72338a21c928914aac06163e4`,
  and `_builder.py` SHA256
  `424afd527cb1780d73c50f2302cd0945f3b1589a860427c45221d4f136e29708`.
- The official architecture reference is
  `https://github.com/Amshaker/SwiftFormer` at commit
  `4aa6cd67527baae00390b58134fa8954e06c5431`; the paper is ICCV 2023,
  `https://arxiv.org/abs/2303.15446`.
- Teacher ID is timm/Hugging Face
  `vit_small_patch16_dinov3.lvd1689m`, revision
  `3bf4720a82ec2066db88137180ff1f83a675cef0`, DINOv3 license, cached
  `model.safetensors`, exactly `86,362,376` bytes, SHA256
  `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`.
  It is the raw external B13 source and resolves to timm `Eva`; it is never B9,
  never a `class_f`-fine-tuned checkpoint and never a B9 feature/logit cache.
  Strict-load it with `num_classes=0`, freeze every tensor, set `eval()`, and use
  `no_grad()` FP32 only. Any class-fitted metadata/state key or non-exact weight
  hash fails closed.
- Preflight is offline: it never downloads. Any explicit future asset-acquisition
  step must fetch only the two pinned revisions, verify size/SHA256, record source
  URLs/licenses and finish before accepted preflight. No unpinned fallback exists.
- Canonical data is `D:\DataAI\AIEx\newdataset\class_f\data.yaml`, SHA256
  `312eb376e89023f42e9df24fea8723b64a6b885c15c24f85d8e319d1ff4f1fa8`.
  Formal B16 uses TRAIN only: `8278` ordered rows, counts
  `[1987,497,1326,2080,2388]`, aggregate content SHA256
  `e9319e6fbddd03382050fadc42bcf156195c590926700b3a7287adc38d4373c8`.
- Reuse the exact B13 five source/adjacency/pHash-union folds: assignment CSV
  SHA256 `afde4fec6b344b867d31e0e01b89f16c5f4be7fd551fdc1fdfd0ea34219d725f`,
  assignment vector `fca50be670fac7da20dd23cb382f9096c3c071d1a1e0dbaa1fd123e65c8aeb4b`
  and union-group vector
  `1243b61c91497459fa12036d1b9f63770ba4f1f11bb132ca17fb3b6013f1ade2`.
  No component crosses fit/held folds. Formal start/end must rehash paths and
  image bytes. No validation/test object, path, image, prediction or metric may
  be constructed or opened.

## Exact SwiftSurface graph

At `224x224`, the locked XS stages have channels/spatial sizes
`48@56`, `56@28`, `112@14`, `220@7`. Let `S2 in R^(B,112,14,14)` be the
complete stage-2 output and let

`base = stage3.downsample(S2) in R^(B,220,7,7)`.

The only new deployment branch is

`b = PW_112->220(SiLU(BN(DWConv3x3_s2_g112(S2))))`,

`r = gamma * b`, where `gamma in R^(1,220,1,1)` (state shape
`[1,220,1,1]`) is trainable and initialized to
`1e-3`. `DWConv` uses padding one, bias false; BN uses affine true,
`eps=1e-5`, `momentum=0.1`; `PW` is a `1x1` convolution with bias false.
Candidate fusion is `y=base+r`; the control is
`y=base+mean_hw(r)` broadcast to `7x7`. Both then execute the unchanged
`stage3.blocks`, final norm, average pool and the two reset five-class heads.
Keep `distilled_training=false`; CE and inference consume the arithmetic mean of
the two head logits, never a selected head or an extra distillation target.

The branch has exactly `26,092` trainable parameters: `1,008` depthwise,
`224` BN, `24,640` pointwise and `220` LayerScale. Its convolutional work before
globalization is exactly `1,256,752` MACs at batch one. Candidate and control
must copy one explicitly initialized template, giving byte-identical branch
state and preserving the caller RNG state. Create that template inside
`torch.random.fork_rng` with seed `20260805`; DW/PW weights use Kaiming-normal
`fan_in`, BN weight/bias are one/zero, and `gamma=1e-3`.

Candidate and control are capacity- and **pre-globalization branch-op** matched.
Their full FLOP counts must not be called exact-matched: the control adds a
spatial reduction and broadcast. Report and benchmark both full graphs
separately.

## Projector-free relation supervision

From the raw EXIF-transposed RGB PIL image, sample one RRC crop window and one
horizontal-flip bit per row. Apply that crop and flip once, with no padding,
letterbox or photometric transform. Resize the same resulting crop bicubically
with antialiasing to `256x256` for DINO and `224x224` for SwiftFormer, then apply
each locked ImageNet mean/std transform. DINO executes exactly once per batch.
Take its final post-norm `16x16` patch grid after removing the five prefix tokens;
do not apply a second norm. In FP32, bilinearly resize this **feature grid** to
`14x14` with `align_corners=false` before computing relations. Never compute a
`16x16` relation map and resize that map.

For either the resized DINO grid or student `S2`, now both
`X in R^(B,C,14,14)`, define FP32 channel-normalized
`U=X/max(||X||_2,1e-6)` at every cell and all valid directed-axis edges

`R_right[i,j] = sum_c U[c,i,j] * U[c,i,j+1]`,

`R_down[i,j]  = sum_c U[c,i,j] * U[c,i+1,j]`.

Thus right has shape `[B,14,13]` and down `[B,13,14]`. Flatten each in row-major
order and concatenate right then down to the exact all-edge vector
`R(X) in R^(B,364)`. There is no padded/replicated/self-cosine boundary, spatial
pooling, learned/fixed projector, channel adapter, temperature, threshold, label,
logit or B9 input. The auxiliary loss for candidate and control is identically

`L_rel = F.smooth_l1_loss(R(S2), R(stopgrad(resize_16->14(P))), beta=0.1,
reduction="mean")`,

`L_CE = F.cross_entropy((head+head_dist).float()/2, y, weight=w_fold,
reduction="mean")`,

`L = L_CE + 0.10 * L_rel`.

PyTorch's weighted mean divides the summed weighted per-row CE by the sum of
target weights in that microbatch. The stock arm uses `L_CE` only.

Horizontal directions are defined in the already flipped screen coordinates, so
no second direction remap is applied.
The teacher and relation loss are training-only and must be absent from export
and deployment; the learned residual branch remains. Teacher targets must be
recomputed from the current augmented crop and must never be cached across
epochs or augmentations.

## Locked five-fold screen

- Arms per fold: `stock`, `globalized_control`, `swiftsurface_candidate`. Build
  all three from one strict-loaded student state; reset-head state is bit-equal
  across arms, and candidate/control branch state is bit-equal. Each arm owns
  separate model, optimizer and BN state.
- On every step, all arms consume the same ordered batch, sampled crop/flip and
  labels. Run DINO once and reuse the same detached target for candidate/control.
  Dropout and stochastic depth are zero. Arm execution must not consume another
  arm's RNG stream. The teacher sees fit rows only and is never run on a held-fold
  row.
- Training augmentation is one shared `RandomResizedCrop` geometry with
  `scale=[0.80,1.00]`, `ratio=[0.90,1.10]`, plus horizontal flip `p=0.5`; no
  vertical flip, colour jitter, RandAugment, MixUp, CutMix or label smoothing.
  Held-fold inference uses official SwiftFormer evaluation preprocessing:
  bicubic resize short side `floor(224/0.95)=235`, center crop `224`, antialias
  true.
- Train exactly eight epochs, batch `16`, gradient accumulation `2` (nominal
  effective batch `32`), workers `0`, and `drop_last=false`. Partition each
  epoch's microbatches into consecutive windows of two; if the final window has
  one microbatch, its divisor is one. For a window of `k in {1,2}`, backpropagate
  each arm's microbatch loss as `L/k`, then clip and step once after the window.
  Scheduler step count is `ceil(number_of_microbatches/2)` per epoch. Use
  final-epoch OOF only. No early stopping,
  best-epoch checkpoint, held-fold inspection or metric-driven selection.
- Base seed is `20260805`; fold seed is `20260805 + fold_id`. Require deterministic
  CUDA, BF16 student autocast on this BF16-capable GPU, FP32 teacher and FP32
  relation construction. TF32 is off. A BF16/CPU/FP16 fallback is invalid.
- Natural shuffled fit rows only; no weighted sampler or row duplication. For
  each fold recompute `w_c=n_c^(-1/2)` from fit indices only and normalize the
  five weights to arithmetic mean one. Use these fixed weights for all three
  arms in that fold.
- Separate AdamW groups: pretrained student backbone peak LR `3e-5`; reset heads
  and new branch peak LR `3e-4`; weight decay `0.05`. Bias, BN/norm and LayerScale
  have zero weight decay. Warm up linearly from zero for one epoch, then cosine
  decay each group to `1e-6`; global gradient-norm clip is `0.7` after each
  accumulated BF16 step. No EMA, KD logits, freeze phase or scheduler restart.

## Endpoints, bootstrap and all-or-nothing gate

Train, finalize and hash all `5 folds x 3 arms = 15` final states before any
held-fold inference, logit construction or metric code is entered; persist a
metric-barrier manifest proving that ordering. Only then persist final OOF logits
for every arm and candidate with `r=0` at inference.
Report accuracy, macro-F1, every class P/R/F1/support, confusion matrices,
class-1 TP/FP/FN, exact `1->2`, `2->1` and restricted `0/2/4->1` counts, and
AUROC for `1-vs-{0,2,4}` using logit margin `logit_1-logit_rival`, overall and
per fold. Mean pair AUROC is the unweighted mean of the three pairs.

Use the exact B13 `5000` paired fold-stratified whole-union-component bootstrap
draws, seed `20260803`, draw SHA256
`d8967cbc13b17b78dd225841bd82a1a3074a7f57c3728fc6dd600aba646ec469`.
Never resample rows. Store every finite replicate and linear percentile
`0.025/0.975` interval. These are conditional TRAIN component-stability
intervals after adaptive project reuse, not confirmatory confidence intervals.

Using unrounded values, candidate passes only if all conditions hold:

1. absolute class-1 F1 is at least `0.700`; absolute macro-F1 is at least `0.800`
   (an engineering floor set in the context of raw-DINO B13 `0.802076`, not a
   generalization estimate);
2. class-1 F1 delta versus globalized control is at least `+0.010` and its paired
   bootstrap lower endpoint is strictly greater than zero;
3. class-1-vs-2 AUROC delta versus control is at least `+0.005`, mean pair-AUROC
   delta is at least `+0.003`, and both lower endpoints are strictly above zero;
4. class-1 F1 delta versus stock is at least `+0.005` and its lower endpoint is
   strictly above zero;
5. macro-F1 delta lower endpoint is at least `-0.002` versus both control and
   stock;
6. class-1 recall is no more than `0.005` below either comparator; restricted
   `0/2/4->1` and `2->1` are each at least `10%` below control and no higher
   than stock; `1->2` is no higher than either comparator;
7. candidate class-1 F1 is strictly higher in at least `4/5` folds versus control
   and at least `3/5` folds versus stock;
8. active candidate class-1 F1 exceeds its same-checkpoint `r=0` ablation by at
   least `+0.003`, its paired component-bootstrap lower endpoint is strictly
   greater than zero, and active `2->1` does not increase;
9. every provenance, strict-load, shared-batch/target, finite-gradient, OOF,
   component-isolation, bootstrap, resource and artifact-integrity check passes.

Each `10%` reduction is `(control_count-candidate_count)/control_count >= 0.10`;
a zero control denominator invalidates rather than passes that gate.

Stock gates establish practical net benefit only. They do not change the matched
causal contrast, which remains candidate versus globalized control.

## Absolutely label-free preflight and mobile gates

Before any dataset YAML/path, fold/group vector, label, TRAIN image or prior
prediction is read, a committed offline preflight uses synthetic tensors/images
only and must prove:

- exact dependency, source, revision, file-size/SHA and strict-load contracts;
  the expected XS stage shapes and DINO `[B,261,384]` geometry;
- exact branch count/formula, candidate/control state equality, preserved RNG,
  candidate/globalized formula parity and finite gradients to S2/branch/head,
  with zero teacher gradients; exact five-class parameter totals are `3,035,570`
  for stock and `3,061,662` for each candidate/control;
- direct NumPy/PyTorch relation parity, values in `[-1,1]`, exact 364 all-edge
  ordering with no boundary padding, feature-grid `16->14` interpolation,
  coordinate-level shared crop/flip equivariance, one teacher call shared by two
  losses, and no projector/B9/class-fitted state;
- one complete synthetic batch-16 CUDA step: one raw FP32 DINO forward followed
  by sequential BF16 forward/backward for stock, control and candidate using the
  same synthetic images/targets; all losses/gradients are finite, DINO remains
  gradient-free, and peak CUDA allocated memory is at most `7 GiB`;
- deployment-only opset-17 ONNX for stock, candidate and control at static
  `[1,3,224,224]`, standard `ai.onnx` domains only, no DINO/relation-loss node,
  PyTorch/ORT maximum absolute logit error at most `1e-5` and identical argmax;
- total five-class candidate/control parameters are exactly the values above;
  FP32 ONNX is at most `13.5 MiB` and no more than `0.25 MiB` above stock. Export
  stock, candidate and control with ONNX Runtime `1.19.2`
  `quantize_static(..., quant_format=QuantFormat.QDQ, activation_type=QInt8,
  weight_type=QInt8, per_channel=true, reduce_range=false,
  calibrate_method=CalibrationMethod.MinMax,
  op_types_to_quantize=[Conv,MatMul,Gemm], nodes_to_quantize=null,
  nodes_to_exclude=null, use_external_data_format=false, extra_options={})`.
  The calibration bytes are C-order
  `default_rng(20260805).random((32,3,224,224), dtype=float32)`, followed by
  FP32 channelwise ImageNet normalization with FP32 mean/std. The reader yields
  `x[i:i+1]` in index order, exactly 32 batch-one inputs, and never reads a file.
  Every QDQ graph is at most `6 MiB`, with at least `90%` of eligible
  constant Conv/Gemm/MatMul weight elements covered by QDQ. Eligibility is every
  rank-at-least-two initializer directly consumed as a weight by those operators;
  coverage is covered eligible `numel` divided by total eligible `numel`.
  CPU execution must be finite; QuantizeLinear/DequantizeLinear sets are nonempty,
  and no custom-domain, DINO or relation-loss node is present. Persist Q/DQ node
  counts, eligible/covered weight `numel`, domains/operators, graph SHA/bytes,
  CPU provider, finite output range, FP32-versus-QDQ maximum absolute logit error
  and argmax agreement on a separate 16-image set generated by the identical
  byte recipe with seed `20260806`;
- ONNX Runtime CPU provider, four threads, batch one, 15 warmups, then five
  alternating 100-run trials, benchmark FP32 and QDQ separately. At each precision
  and for **each** candidate/control versus same-precision stock, both median and
  p95 ratios are at most `1.10`, and absolute p95 is at most `12 ms`. Record every
  graph's operators, bytes, trial distributions, median and p95; do not substitute
  theoretical FLOPs.

These are desktop engineering gates, not a phone claim. Even after a scientific
pass, mobile promotion requires a newly sealed named-device protocol with the
deployment graph only: batch one, sustained five-minute p95 at most `30 ms`,
at least `33 FPS`, peak RAM at most `256 MiB`, package at most `16 MiB`, thermal
p95 degradation at most `15%`, and INT8/FP16 drops versus FP32 no more than
`0.005` macro-F1 and `0.010` class-1 F1 on a prospectively sealed holdout.

## Stop, artifacts and code boundary

Formal budget is one three-arm run: five folds, eight epochs, peak CUDA allocated
at most `7 GiB`, monotonic wall time at most `12 h`, retained bytes at most
`1.5 GiB`. Persist commands/config, environment/Git, all source/weight/data hashes,
ordered paths/labels/folds/groups, per-fold class weights and batches, loss curves,
final states, OOF logits/metrics, confusion transitions, bootstrap arrays,
branch/ablation telemetry, resource measurements and explicit
`train=true, validation=false, test=false` flags. Write atomically and retain a
hashed failure manifest.

One identical retry is allowed only for an infrastructure failure before any OOF
metric is constructed. After OOF construction, success or failure is final. Any
gate failure closes this exact XS/S2/right-down-cosine/SmoothL1-0.10/local-residual
route: do not sweep relation weight, edge padding, layer, branch width, gate,
sampler, LR, augmentation, resolution or control after seeing B16. A pass still
does not authorize validation/test; it authorizes writing the next protocol.

DATA-01 remains unresolved: label-boundary review and fruit/session provenance
still require two blinded reviewers plus adjudication or new immutable provenance.
B16 cannot relabel rows or support an independent generalization claim.

B16 implementation must live in dedicated small modules, specifically
`trkh/models/swiftformer_surface_b16.py` and one B16 runner, with focused tests and
only a thin registry hook if required. It must not add another large path to
`model.py` or `train.py`, reuse B9 hybrid code, delete evidence or expand an
unrelated monolith.
