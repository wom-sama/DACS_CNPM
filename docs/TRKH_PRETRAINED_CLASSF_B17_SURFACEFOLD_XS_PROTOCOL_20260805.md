# TRKH B17 SurfaceFold-XS protocol (2026-08-05)

## Status and boundary

Protocol ID: `TRKH_PRETRAINED_CLASSF_B17_SURFACEFOLD_XS_20260805`.

B17 is a new structural-reparameterization route, not a retry or retroactive
pass of B16. B16 stopped before any dataset access because its preregistered
Windows ORT-CPU absolute-latency gate failed for stock, control and candidate.
No B16 image, label or quality metric exists. Exact B16 remains closed.

B17 starts with no data authority. Under the transparently revised `DEPLOY-01`
two-tier policy, only a committed, clean-source, offline, label-free preflight
may authorize one TRAIN-only five-fold causal screen.
Validation and test remain sealed. Passing TRAIN only leaves deployment status
`DEVICE_PENDING`; it does not authorize a mobile-ready or generalization claim.

## Why this successor is materially different

B16 added a nonlinear `DWConv -> BN -> SiLU -> PWConv` residual after the
stage-3 downsample. Its DINO relation loss was computed on the upstream S2 map,
so it supplied no direct gradient to that residual. The branch could only learn
from class CE, reproducing the starvation mechanism already observed in V3.
The nonlinear branch also could not be folded into a stock convolution.

B17 adapts the existing SwiftFormer-XS stage-3 downsample kernel itself and
places relation supervision on its output. Its train-time factors are removed
algebraically at export, leaving the exact stock inference topology.

Related work limits the novelty claim. Spatial/channel convolution adaptation
overlaps LoCA and convolutional LoRA families; fold-at-export overlaps
structural reparameterization such as RepVGG and MobileOne. The proposed
contribution tested by B17 is a matched, raw-DINO relation-guided, foldable
surface adapter and its class-1 boundary evaluation on canonical `class_f`, not
the algebra in isolation. Any contribution claim remains conditional on TRAIN,
a later prospectively sealed holdout and the physical-device gates.

Primary context:

- LoCA: <https://arxiv.org/abs/2607.06918>
- RepVGG: <https://arxiv.org/abs/2101.03697>
- MobileOne: <https://openaccess.thecvf.com/content/CVPR2023/html/Vasu_MobileOne_An_Improved_One_Millisecond_Mobile_Backbone_CVPR_2023_paper.html>
- ORT mobile workflow: <https://onnxruntime.ai/docs/tutorials/mobile/>

## Locked assets and data boundary

- Student: `timm/swiftformer_xs.dist_in1k`, revision
  `ac0196f198e58c82183a67f5f5c0952421b3e6ac`, Apache-2.0,
  `model.safetensors` bytes `13,957,760`, SHA256
  `c0dfb2e99069ece07b691ff470f5dafed98716d90be738369828693c3bf2071c`.
- Training-only teacher: `timm/vit_small_patch16_dinov3.lvd1689m`, revision
  `3bf4720a82ec2066db88137180ff1f83a675cef0`, DINOv3 license,
  bytes `86,362,376`, SHA256
  `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`.
- Canonical immutable YAML is
  `D:\DataAI\AIEx\newdataset\class_f\data.yaml`, SHA256
  `312eb376e89023f42e9df24fea8723b64a6b885c15c24f85d8e319d1ff4f1fa8`.
- TRAIN has `8,278` ordered rows, class counts
  `[1987,497,1326,2080,2388]`, aggregate path/image-content SHA256
  `e9319e6fbddd03382050fadc42bcf156195c590926700b3a7287adc38d4373c8`.
  Class order is exactly `Xoai_Song_Chua_KhoDap`,
  `Xoai_Song_ChuaNhe_CoNguyCo`, `Xoai_Chin_NgotThanh_DeDap`,
  `Xoai_ChinGia_NgotGat_KhongVanChuyen`, `Xoai_Hu_KhongAnDuoc`.
- The fixed label-independent five-fold assignment CSV SHA256 is
  `afde4fec6b344b867d31e0e01b89f16c5f4be7fd551fdc1fdfd0ea34219d725f`.
- Its ordered fold-vector, ordered path/fold and ordered union-group-vector
  SHA256 values are respectively
  `fca50be670fac7da20dd23cb382f9096c3c071d1a1e0dbaa1fd123e65c8aeb4b`,
  `0f3856bf6020ba31d8564a48394e2a6fba7ad6f415e279e0063c5baadff20cb7`
  and `1243b61c91497459fa12036d1b9f63770ba4f1f11bb132ca17fb3b6013f1ade2`.
- Formal TRAIN must hash every ordered path and image byte at start and end and
  reproduce all values above. A mismatch invalidates the run.
- Validation is design-exposed and test is historical/sealed. Neither may be
  constructed, enumerated, read or used by B17 preflight/TRAIN.
- Adjacency/pHash/union components are used only for fold exclusion and paired
  component bootstrap. They are never labels, targets or inference inputs.

## Locked architecture

The five-class SwiftFormer-XS has `3,035,570` deployment parameters. Its
stage-3 downsample is a dense `Conv2d(112,220,3,stride=2,padding=1,bias=True)`
followed by the stock `BatchNorm2d(220)`. Let its immutable pretrained kernel
and bias be `W0` and `b0`.

For input channel `i`, learn one spatial atom `D[i,:,:]`; learn channel mixing
`P[o,i]`. The candidate kernel is

```text
delta_W[o,i,h,w] = P[o,i] * D[i,h,w]
W_eff = W0 + delta_W
```

The globalized control owns the exact same `P,D` tensors and state but replaces
each atom by its spatial mean broadcast over `3x3` before forming `delta_W`.
The stock-relation arm sets `delta_W=0`. `W0` and `b0` are frozen in all three
arms; all other student parameters, including the following BN, may train.
This freeze is required: a trainable full `W0` could absorb the factor update
and destroy the candidate-versus-control causal contrast.

Initialization is deterministic and RNG-preserving. Set every `D[i,h,w]=1/9`.
Draw `P_raw ~ Normal(0,1)` with a private CPU `torch.Generator`, seed `17042`,
device CPU and dtype float64. For each output `o`, rescale the row before casting
to the projection dtype/device so `||P[o,:,None,None]*D||_F =
0.01*||W0[o]||_F`. A zero or nonfinite raw/base row, scale, cast factor or delta
fails closed. Candidate and control are bit-equal at initialization while both
`P` and `D` receive a first-backward gradient. No zero scalar gate is allowed.

Load the pinned 1000-class student state strictly first. Inside
`torch.random.fork_rng`, seed `20260805`, replace `head` followed by `head_dist`
with `Linear(220,5)`, initialize each in that order using timm
`trunc_normal_(std=0.02)` and zero bias, then leave the caller CPU/CUDA RNG state
unchanged. One complete base-state copy supplies all three arms in every fold;
candidate and control then receive the same deterministic factor state. Every
state equality, disjoint storage and RNG invariant is hashed/asserted.

Factor counts are exact: `P=24,640`, `D=1,008`, total `25,648`; candidate and
control train-state totals are `3,061,218`. With `W0,b0` frozen, trainable totals
are candidate/control `2,839,238` and stock `2,813,590`.

At export:

```text
W_fold = W0 + delta_W
b_fold = b0
```

The factors are deleted and the ordinary stock conv is restored. The deployed
model must have exactly `3,035,570` parameters, the stock operator topology and
zero additional inference operator/MAC/storage from SurfaceFold.

## Direct raw-DINO relation supervision

The supervised student map is S3 immediately after the stage-3 downsample BN
and before stage-3 blocks: `[B,220,7,7]`. DINO `forward_features` must return its
final post-norm `[B,261,384]` tokens. Remove exactly the first five prefix tokens,
reshape the remaining 256 tokens to `[B,384,16,16]`, and do not apply a second
normalization layer. Detach/cast this raw map to FP32, then bilinearly resize the
feature map (never its relation map) to `7x7`, `align_corners=False`.

For either map `X`, compute FP32 channel-normalized cells
`U=X/max(||X||_2,1e-6)`. In row-major order concatenate the 42 right edges
`sum_c U[c,i,j]U[c,i,j+1]` followed by the 42 down edges
`sum_c U[c,i,j]U[c,i+1,j]`. Let the resulting `[B,84]` vectors be `R_s,R_t`.
The exact objectives are

```text
L_rel = smooth_l1_loss(R_s, stopgrad(R_t), beta=0.1, reduction="mean")
L_CE  = cross_entropy(((head + head_dist) / 2).float(), y,
                      weight=w_fold, reduction="mean")
L      = L_CE + 0.10 * L_rel
```

There is no projector, second-head CE, separate-head averaging of losses or
logit KD.

All three arms receive identical class CE plus identical relation loss. Thus
candidate versus control isolates spatial atoms, and candidate versus stock
measures the foldable adaptation under the same teacher signal. B17 does not
claim an isolated causal effect for relation distillation itself.

## Arms and TRAIN recipe

Exactly three separately optimized arms per fold:

1. `stock_relation`: frozen `W0,b0`, no factor delta;
2. `surfacefold_mean_control`: factor delta with mean-broadcast atoms;
3. `surfacefold_spatial_candidate`: full spatial atoms.

The five folds use seeds `20260805 + fold_id`. Set `CUBLAS_WORKSPACE_CONFIG` to
`:4096:8` before CUDA initialization; seed Python, NumPy, CPU and CUDA; require
deterministic algorithms, cuDNN deterministic true/benchmark false, TF32 off,
and BF16 student autocast on the locked GPU. CPU/FP16 fallback is invalid.
Dropout and stochastic depth are zero. Each arm owns separate model, BN,
optimizer, scaler-free BF16 state and CPU/CUDA RNG streams; its saved RNG state
is restored before and captured after its execution so one arm cannot advance
another. Shuffle uses its fold generator above. Augmentation uses a second
private CPU generator seeded exactly `30_260_805 + fold_id`; for each row in
batch order it draws up to ten `(target_area, log_aspect)` attempts, then
`top,left`, then the flip bit, matching the locked B16 sampling algorithm. Its
state continues across epochs and is never consumed by shuffle or any arm.

From each EXIF-transposed RGB PIL image, use one shared RandomResizedCrop window
with scale `[0.80,1.00]`, ratio `[0.90,1.10]`, and one horizontal flip bit with
`p=0.5`. Apply the crop/flip once, then bicubic-antialias resize it to `224x224`
for the student and `256x256` for DINO. Both use FP32 ImageNet normalization
mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)`. Vertical flip, colour
jitter, RandAugment, MixUp, CutMix, label smoothing, padding and letterbox are
forbidden. Held-fold inference uses bicubic-antialias short-side resize
`floor(224/0.95)=235`, center crop `224`, then the same normalization.

Natural fit rows only: one persistent private CPU generator seeded by the fold
seed produces one `torch.randperm` per epoch; there is no sampler, duplication
or replay weighting. Every arm consumes the identical ordered batches, crop,
flip, labels and detached teacher target. The teacher runs exactly once per fit
batch under FP32 inference mode and never enters a student checkpoint.

Train exactly eight epochs, batch `16`, workers `0`, `drop_last=false`, gradient
accumulation `2`. Consecutive microbatches form windows of two; a final singleton
window divides loss by one, otherwise each loss is divided by two. Clip global
gradient norm at `0.7` after accumulation and before every update. Scheduler
steps per epoch are exactly `ceil(microbatch_count/2)` and total steps are eight
times that value.

Use fit-fold-only `n_c^(-1/2)` CE weights normalized to arithmetic mean one.
AdamW betas/epsilon are `(0.9,0.999)`/`1e-8`: pretrained backbone peak LR
`3e-5`; reset heads and `P,D` peak LR `3e-4`; weight decay `0.05`, except every
bias, BN/norm and LayerScale tensor has zero decay. For actual optimizer update
index `u=0..T-1`, let `S` be steps/epoch. If `u<S`, LR is
`peak*u/(S-1)`; otherwise it is
`1e-6+(peak-1e-6)*(1+cos(pi*(u-S+1)/(T-S)))/2`. Thus the first, epoch-one last
and final applied LRs are exactly `0`, peak and `1e-6`. No EMA, GradScaler,
logit KD, freeze phase, restart, early stopping, best-epoch selection or
intermediate metric is allowed.

## Label-free preflight gate

Preflight runs offline, uses only synthetic tensors and performs these gates in
order:

1. Require clean committed `research/pretrained-classf-b1`. Exact runtime is
   Python `3.9.11`, torch `2.6.0+cu124`, torchvision `0.21.0+cu124`, CUDA
   runtime `12.4`, cuDNN `90100`, timm `1.0.27`, NumPy `1.26.4`, ONNX `1.19.1`,
   `onnxruntime-gpu 1.19.2`, safetensors `0.7.0`, huggingface-hub `0.36.2`.
   Rehash timm `swiftformer.py`, `_factory.py`, `_builder.py` as respectively
   `e7f79c9bfb3750636ada4cd776c9ab9da41d321ba56ada28201a7dc480b654a7`,
   `30a6eecdaba750af470cfae3196186fd647052c72338a21c928914aac06163e4`,
   `424afd527cb1780d73c50f2302cd0945f3b1589a860427c45221d4f136e29708`.
   Verify offline flags, licenses and both pinned asset byte/hash contracts.
2. Strict 100% asset loads, deterministic five-class heads and exact
   architecture/parameter/freeze/RNG contracts must pass. All focused tests run
   without skip.
3. Qualify stock first. Export only the deployment model with
   `torch.onnx.export`: static batch-one `[1,3,224,224] -> [1,5]`, input `x`,
   output `logits`, opset `17`, constant folding true and no dynamic axes. Only
   standard domains are allowed; FP32 size is at most `13.5 MiB`.
   PyTorch/ORT CPUExecutionProvider parity on 64 fixed synthetic inputs requires
   maximum absolute error `1e-5` and zero argmax mismatch.
4. Run the ORT Mobile prebuilt-package checker and NNAPI/CoreML usability
   heuristic and retain their full logs. Convert stock with ORT
   `OptimizationStyle.Fixed`, target platform `arm`, type reduction true,
   optimized-ONNX save false, conversion failures false, no custom-op library.
   Reload with CPUExecutionProvider; ONNX/ORT-format max error is `1e-5`, zero
   argmax mismatch. `.ort` is at most `14 MiB` and `1.10x` source ONNX bytes.
5. Host sessions use CPUExecutionProvider only, intra/inter-op threads `4/1`,
   `ORT_SEQUENTIAL`, `ORT_ENABLE_ALL`, batch one, 15 warmups and five trials of
   100 paired calls. Alternate order by `(trial+iteration) mod 2`; retain raw
   timings, median and linear p95. ORT-format median and p95 may each be at most
   `1.05x` ONNX. The retired B16 12 ms Windows threshold remains closed and is
   not a mobile claim.
6. Only after stock passes, construct candidate/control and prove identical
   floating base/factor initial state and logits (their persistent mode IDs must
   differ), disjoint storage, cross-mode strict-load rejection, immutable
   `W0,b0` and unchanged caller RNG. First, backpropagate `L_rel` alone: require
   finite nonzero gradients to `P` and candidate centered `D`; control `D`
   gradients must be spatially equal and nonzero. Then on a fresh graph
   backpropagate the exact CE+relation objective and require finite nonzero
   gradients for both heads, post-BN/pre-block S3 and `P,D`. Teacher gradients
   are absent in both checks.
7. Float64 depthwise-plus-pointwise versus expanded-delta oracle maximum error
   is `1e-10`; factor form versus folded pre-BN/features/logits FP32 maximum
   error is `1e-6`; delta-off is bit-exact stock.
8. Fold candidate/control, export them identically to stock and prove no
   factor/teacher/relation identifier; `3,035,570` parameters; identical nested
   graph/operator sequence, domains, attributes, initializer names/shapes/dtypes
   and structural constants. Initializer values may differ. Each artifact is
   within `4 KiB` of stock and passes the same parity/mobile-checker/ORT-format
   contracts; candidate/control host timing is diagnostic only because topology
   identity, not a host absolute budget, is the deployed-cost causal contract.
9. On the exact NVIDIA GeForce RTX 4060 Laptop GPU (capability `8.9`, memory
   `8,585,216,000` bytes), run one complete synthetic batch-16 preflight: one
   raw DINO FP32 forward and target construction, then sequential BF16
   forward/backward/AdamW state materialization for stock, control and candidate
   with one shared synthetic image/label/target exposure. Require finite losses,
   every trainable tensor gradient and optimizer state; explicitly nonzero S3,
   both heads and candidate/control `P,D` gradients; no teacher gradient. Peak
   CUDA allocated memory must be at most `7 GiB`.
10. Final Git/source/runtime/asset rehash equals the start; source and output
    manifests prove zero dataset YAML/path/image/label/prior-prediction access,
    zero validation/test construction and zero network use.

Any failure closes the exact implementation before TRAIN. A pass grants only
the one TRAIN-only screen and records `DEVICE_PENDING`.

## State-before-metrics and promotion gate

Train all 15 final states first. Save each as safetensors and verify exact
state IDs, hashes, sizes, parameter counts, exposure/loss hashes and strict
reload before constructing any held-fold dataset, OOF buffer or metric backend.
Then infer every TRAIN row once from its held fold and retain the unrounded
arithmetic mean of `head` and `head_dist` logits for all three arms plus the
same-checkpoint candidate with `delta_W=0`.

Report accuracy, macro-F1, each class precision/recall/F1/support, confusion
matrix and exact C1 TP/FP/FN, `1->2`, `2->1`, restricted `0/2/4->1`. For rival
`r in {0,2,4}`, pair AUROC uses only true class-1/rival rows and score
`logit_1-logit_r`; mean pair AUROC is the unweighted mean of the three.

Use exactly 5,000 paired, fold-stratified whole-union-component bootstrap draws;
never resample rows. Seed is `20260803` and the serialized draw vector must hash
to `d8967cbc13b17b78dd225841bd82a1a3074a7f57c3728fc6dd600aba646ec469`.
Every replicate is finite and retained; use NumPy linear quantiles
`0.025/0.975`. All deltas below are candidate minus named comparator; the
same-checkpoint effect is active candidate minus delta-off. Use unrounded point
values and replicate values throughout.

Every gate must pass:

- candidate class-1 F1 at least `0.700` and macro-F1 at least `0.800`;
- class-1 F1 delta versus control at least `+0.010`, lower bound `>0`;
- class-1 F1 delta versus stock at least `+0.005`, lower bound `>0`;
- class `2` pair AUROC delta at least `+0.005` and mean pair-AUROC delta at
  least `+0.003` versus control, both lower bounds `>0`;
- macro-F1 lower bounds at least `-0.002` versus control and stock;
- class-1 recall no more than `0.005` below either comparator; restricted
  `0/2/4 -> 1` and `2 -> 1` each reduced by at least `10%` versus control and
  no worse than stock; `1 -> 2` no worse than either;
- at least four of five class-1 fold wins versus control and three versus stock;
- same-checkpoint delta-off loses at least `0.003` class-1 F1 with lower bound
  above zero and does not improve `2 -> 1`;
- folded/unfolded equivalence, finite telemetry, immutable `W0,b0`, exact
  exposure/update/LR/state/hash/resource contracts all pass.

For each required `10%` count reduction use
`(control_count-candidate_count)/control_count >= 0.10`; a zero control
denominator invalidates and fails the gate rather than passing it.

Failure closes exact B17 without rank/layer/width/loss/LR/epoch/backend sweeps.
Passing authorizes drafting a separate validation protocol, not validation
execution or test access.

## Physical-device boundary

No Android ARM64 device/runtime is presently connected, so host preflight
cannot certify mobile latency. TRAIN may only test the locked representation
hypothesis and remains `DEVICE_PENDING`; no validation protocol may execute
before the following physical-device gate. Freeze device/SoC, Android/API, ORT
AAR SHA, thermal/power state, thread count and an application-derived absolute
latency/RSS budget. Measure stock first on CPU and XNNPACK; lock one backend
before candidate timing. If stock fails, close that target contract rather than
switching provider after candidate results. Candidate folded topology must
remain stock-identical and sustained p95 at most `1.03x` stock. Until this
passes, neither mobile-ready, deployable-performance nor promotion claims are
allowed.
