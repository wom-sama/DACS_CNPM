# TRKH B18 TargetMatch SurfaceFold-XS protocol (2026-08-05)

Protocol ID: `TRKH_PRETRAINED_CLASSF_B18_TARGETMATCH_SURFACEFOLD_XS_20260805`.

## Status and scope

B18 is a protocol-only successor to the closed B17 implementation. It is not a
new architecture, a hyperparameter retry or a result. Before this document is
committed, B18 has no authority to enumerate `class_f`, construct a dataset,
read a label, train, validate or test.

The exact B17 formal implementation is preserved at commit `954655c` and tag
`trkh-pretrained-b17-closed-954655c`. Its first formal preflight failed at
`strict_load`; failure artifact SHA256 is
`ad1ded2a42808cf0fd39843b07efe0f197471b8e7ff90fe449a486ac2e672848`.
CUDA had already been initialized, but `torch.manual_seed(20260805)` ran inside
`torch.random.fork_rng(devices=[])`. The context restored CPU RNG only, while
`torch.manual_seed` also changed CUDA RNG. A direct synthetic probe reproduced
CPU preservation with CUDA drift and showed that including CUDA device `0` in
the fork restores both. B17 therefore remains closed and is never retried.

No model, data or quality metric informed the two B18 corrections below. The
only additional deployment evidence was stock-only and label-free: an ARM-
target ORT-format artifact was repeatedly slower when executed on the Windows
AMD64 host. This is a target-confounded measurement, not evidence about
SurfaceFold representation quality.

## Immutable inherited contract

B18 inherits every non-conflicting term of the B17 protocol whose SHA256 is
`56ae72973b844186198dbc78c94851f48cf463b902370d2b3d2078b280169540`.
The following are unchanged:

- canonical immutable `class_f`, TRAIN-only five-fold source/near-component
  assignment, exact YAML/CSV/path/content hashes, class order and sealed
  validation/test boundaries;
- SwiftFormer-XS student, raw DINOv3-S teacher, exact pretrained assets and
  five-class deterministic heads;
- SurfaceFold placement, `P,D` factorization, private FP64 projection seed,
  one-percent delta norm, spatial candidate, equal-state mean-surface control
  and stock arm;
- post-downsample-BN/pre-stage-block S3 relation target, 42 right plus 42 down
  cosine relations, SmoothL1 and exact `CE + 0.1 * relation` objective in all
  three arms;
- eight epochs, batch 16, workers zero, accumulation two, natural fit-only
  exposure, augmentations, per-fold seeds, AdamW groups, LR schedule, gradient
  clipping and resource limits;
- 15-state-before-metrics barrier, one-use authorization, strict reload,
  cross-mode rejection, held-fold OOF construction, candidate delta-off
  inference, 5,000 whole-component bootstrap draws and every B17 statistical
  promotion gate;
- exact fold-to-stock topology and `3,035,570` deployed parameters;
- no validation/test permission and `DEVICE_PENDING` after any TRAIN pass.

The locked model source remains SHA256
`7707ed6153fc4beed91312fc5d962f84a65652468e428f7da680546f899778db`.
B18 source/runtime/asset hashes are recorded by the committed executable
preflight and must match again before metrics and at process end.

## Correction 1: complete RNG isolation

The exact CUDA device contract is established before focused tests or model
construction. Every caller-RNG guard snapshots the CPU generator and every
available CUDA generator. Any scoped call to `torch.manual_seed` uses
`torch.random.fork_rng(devices=list(range(torch.cuda.device_count())))` after
CUDA initialization, or an equivalent explicit save/restore proven bit-exact.
It is insufficient to preserve CPU state alone.

The focused suite has no skip path. On the actual initialized GPU it must first
demonstrate that the retired `devices=[]` pattern changes CUDA RNG, then prove
that B18 strict model construction, deterministic head reset and active-arm
construction leave CPU and every CUDA RNG state bit-identical. The formal TRAIN
runner repeats the same initialized-CUDA head-reset proof. A leak closes B18
before data.

## Correction 2: target-matched ORT-format measurement

Stock is always qualified before candidate or control exists. Export one static
batch-one FP32 ONNX graph exactly as in B17: `[1,3,224,224] -> [1,5]`, names
`x/logits`, opset 17, constant folding, no dynamic axes, standard domains and
at most `13.5 MiB`. PyTorch/ONNX Runtime CPU parity on the same 64 fixed
synthetic inputs requires maximum absolute error at most `1e-5` and zero argmax
mismatch.

From the exact same source ONNX SHA create two distinctly named ORT-format
artifacts with ONNX Runtime `1.19.2`, `OptimizationStyle.Fixed`, type reduction
enabled, optimized-ONNX save disabled, conversion failures disabled and no
custom-op library:

1. `host_native_amd64.ort` with `target_platform=amd64`;
2. `android_arm.ort` with `target_platform=arm`.

Each conversion manifest records the common source ONNX SHA, target, artifact
SHA, required-operator/type config SHA and command settings. Each artifact must
reload through CPUExecutionProvider, match the source ONNX on all 64 fixed
inputs within `1e-5` with zero argmax mismatch, remain at most `14 MiB` and at
most `1.10x` the source ONNX bytes. The ARM package additionally retains the
full ORT Mobile prebuilt-package and NNAPI/CoreML heuristic logs. These logs are
structural diagnostics, not physical-device certification.

Only `host_native_amd64.ort` is compared with ONNX on the Windows AMD64 host.
Use CPUExecutionProvider, intra/inter-op threads `4/1`, `ORT_SEQUENTIAL`,
`ORT_ENABLE_ALL`, batch one, 15 warmups and five trials of 100 paired calls,
alternating order by `(trial+iteration) mod 2`. Retain all 500 timings. Native
ORT-format median and linear p95 may each be at most `1.05x` ONNX. No retry,
thread/provider switch, cache manipulation or threshold change is allowed.

ARM-on-AMD64 latency, if executed, is retained with status
`NOT_APPLICABLE_TARGET_MISMATCH`; it is never an authorization, mobile or
performance gate. B17's observed ARM-on-AMD64 result is not relabelled as a B18
pass.

Only after every stock gate passes may control/candidate be constructed. Their
folded exports must retain B17 factor/oracle/parity/identifier/parameter and
stock-topology equality. Both target packages must satisfy the same artifact
and parity contracts; active-arm host latency is diagnostic because their
deployed topology is required to be stock-identical.

## Preflight order and authorization

The committed offline preflight executes exactly in this order:

1. clean canonical branch, bound source/runtime/timm/assets and offline flags;
2. exact CUDA initialization and guarded focused tests with zero dataset,
   network, deselection or child-process escape;
3. strict student/teacher loads, five-class heads and complete RNG restoration;
4. stock ONNX plus dual-target ORT qualification and native-host gate;
5. active-arm construction, relation/total-objective gradients, FP64 oracle,
   fold parity and folded dual-target artifacts;
6. one synthetic CUDA batch-16 teacher/three-arm forward-backward-AdamW check;
7. final Git/source/runtime/asset rehash and atomic evidence publication.

Any failure closes exact B18 before TRAIN. A pass grants one use of the
unchanged B17 TRAIN-only five-fold/three-arm screen and records
`DEVICE_PENDING`. It grants no validation, test or mobile-ready claim.

## Physical Android boundary

Before validation execution or any deployable-performance claim, commit a
separate physical-device protocol containing device/SoC, Android/API, exact ORT
AAR/build SHA, thermal/power state, thread count, sample schedule and
application-derived absolute p95 and peak-RSS budgets. Measure stock first on
CPU and XNNPACK; choose the eligible backend by a predeclared stock-only rule
before candidate timing. If no stock backend meets both absolute budgets, close
that target contract.

On the locked backend, stock and folded candidate must meet the absolute p95/RSS
budgets, preserve parity and candidate sustained p95 must be at most `1.03x`
stock. A TRAIN statistical pass without this Android pass remains
`DEVICE_PENDING` and cannot open validation. The historical test remains sealed
regardless of either result.

