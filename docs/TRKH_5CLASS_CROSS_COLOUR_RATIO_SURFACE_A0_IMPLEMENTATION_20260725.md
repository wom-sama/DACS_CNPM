# TRKH 5-Class Cross-Colour Ratio Surface A0 Implementation

Date: 2026-07-25
Protocol: `cross_colour_ratio_surface_a0_train_only_v1`
Parent commit: `171a8fbc294adaae1020e12747aeb8c1f52e5c60`
State: `equation_engine_preflight_only`

## Scientific State

This implementation is still prospective:

- no cohort image has been loaded by the CCR implementation;
- no CCR or matched-control weight has been fitted;
- no candidate score, threshold, action, AUROC, AUPRC, F1, validation
  metric, or test metric has been observed;
- no raw `class_f` or `yolo_f` file has been changed;
- no pretrained or external parameter has been loaded;
- no external implementation has been copied or imported;
- the keeper checkpoint and current-best commands remain unchanged.

The engine was implemented only after the protocol, lock, and same-tensor
erratum were committed and pushed. The only cohort artifact read by its
regression tests is the immutable geometry metadata needed to reproduce the
already-locked sample indices, folds, epoch orders, and dephase offsets.

## Implemented Files

| Artifact | SHA-256 |
|---|---|
| `trkh/tools/cross_colour_ratio_surface_a0_engine.py` | `ac610d7c37eda1b96a54257f6814a3e8c7ef5bfc243ec3bb3c7ae13a86d26fe9` |
| `tests/test_cross_colour_ratio_surface_a0_engine.py` | `5a83a138ebef39b290c29d0b2121c3675c89d843ef84366403b30a7a5294c840` |

These hashes cover the equation engine and its independent regression
oracles before any image pass or fit.

## Frozen Tensor Contract

The engine accepts the exact normalized `256 x 256` keeper evaluation tensor
and its image-valid mask. It does not load images or reconstruct the keeper
crop itself. The future direct loader must reproduce the frozen keeper
evaluation tensor exactly as amended by the preimplementation erratum before
calling this engine.

The engine:

1. inverts the frozen ImageNet mean and standard deviation;
2. clamps reconstructed sRGB to `[0, 1]`;
3. downsamples `256 -> 96` with a fixed adaptive-area-equivalent linear
   operator;
4. applies the standard IEC 61966-2-1 inverse sRGB transfer;
5. clamps linear RGB only at the locked log epsilon `1/255`;
6. constructs the three frozen planes;
7. applies fixed signed Gaussian derivative correlations;
8. masks every response without complete valid, nonsaturated `7 x 7`
   support.

The fixed area resize is represented by two matrix multiplications rather
than a backend-specific resize operator. Its FP32 maximum difference from
PyTorch area interpolation in the locked regression is below `5e-7`.

## Frozen Descriptor Equation

The candidate planes are:

```text
P_rg = log(R) - log(G)
P_rb = log(R) - log(B)
P_gb = log(G) - log(B)
```

The matched Colour Ratio control planes are:

```text
P_r = log(R)
P_g = log(G)
P_b = log(B)
```

For both roles, each plane is correlated with the same normalized
Gaussian-x and Gaussian-y derivative kernels using:

```text
sigma = 1
radius = 3
dG_correlation(x) = x * G(x) / sigma^2
```

The exact channel order is:

```text
[rg_x, rg_y, rb_x, rb_y, gb_x, gb_y]
```

The descriptor is signed. Finite protection clamps to `[-8, 8]`; unreliable
responses are zeroed. No magnitude reduction, histogram, threshold, Canny,
learned colour correction, or label-dependent operation is present.

An independent FP64 NumPy convolution oracle matches the PyTorch equation
within `2e-15`. A separate multiplicative-ratio oracle matches the
log-difference equation within `1e-12`, including antisymmetry when the two
samples are exchanged.

## Physical Oracle

The regression suite verifies the property this candidate is intended to
test:

- spatially varying common illumination cancels pointwise from the
  cross-colour planes;
- fixed per-channel gain adds only a spatial constant to each cross-colour
  plane and therefore cancels under the signed derivative;
- the matched raw Colour Ratio control changes under the same common
  illumination field.

This is an equation-level result only. It does not claim that mango class 1
is empirically separable until the locked held-fold experiment is complete.

## Reliability Contract

A view pixel is initially reliable only when:

- its entire area-resize support comes from the keeper-valid image region;
- all three sRGB channels are strictly above `1/255`;
- all three sRGB channels are strictly below `254/255`.

A derivative response is reliable only when all `49` positions in its
`7 x 7` support satisfy those conditions. The same reliability mask is used
for candidate and control. After the two fixed average-pooling stages, its
fractional area is the only spatial weight used by the evidence mean.

For the trained spatial-dephased control, each descriptor channel is rolled
independently with the locked nonzero `[y, x]` offset. Channel value
multisets are preserved while cross-channel spatial alignment is destroyed.
The original reliability mask remains fixed, as specified prospectively.

## Scratch Head

The head is exactly:

```text
Conv2d(6, 24, 3, bias=False)
GroupNorm(6, 24)
GELU
AvgPool2d(2)
DepthwiseConv2d(24, 24, 3, bias=False)
PointwiseConv2d(24, 48, 1, bias=False)
GroupNorm(8, 48)
GELU
AvgPool2d(2)
Conv2d(48, 4, 1, bias=True)
reliability-weighted spatial mean
```

The class order is `[0, 1, 2, 4]`. The parameter oracle is exactly `3,004`
trainable parameters across nine tensors. There is no backbone, embedding,
teacher, pretrained state, keeper probability input, bbox input, or source
metadata input.

Candidate, Colour Ratio control, and trained-dephased control have
byte-identical scratch initialization for a fold. The seed repeat has the
locked `+100000` offset and a different initial state. A regression optimizer
step proves finite gradients and a changed state.

## Reproducibility Contract

The engine reproduces:

- dephase shape `[763, 6, 2]`;
- dephase SHA-256
  `814bd0da6d1781f2d943810b5686f9e50c4edf3d6a0782d178f3075eebba951d`;
- every primary and repeat epoch-order hash for all five locked folds;
- the two-epoch linear warmup and epoch-20 cosine endpoint;
- exact class mapping, evidence score, calibration tie rule, and
  keeper-suppression semantics;
- the production margin-crop integer geometry on independent examples.

The head evidence score excludes class 3 exactly as locked:

```text
s = logit_1 - logsumexp(logit_0, logit_2, logit_4)
```

If a keeper class-1 prediction is suppressed, replacement remains the
keeper's highest-probability non-class-1 class, which may be class 3.

## Deployment Preflight

The complete descriptor-plus-head pipeline exports at ONNX opset 17 with:

- only standard ONNX-domain nodes;
- dynamic batch input and output axes;
- probability and pair-map replay through ONNX Runtime;
- maximum observed replay error below the locked `1e-5` limit;
- output shapes `[B, 4]` and `[B, 3, 24, 24]`.

The export test emits two upstream PyTorch warnings: one internal
GroupNorm shape check and one deprecated boolean-cast symbolic. Neither
creates a custom operator, fixes the batch dimension, nor exceeds the replay
gate. TensorRT timing and parity remain forbidden until every clean
scientific gate passes.

## Verification

Executed with `D:\DataAI\.venv\Scripts\python.exe`:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m py_compile `
  trkh/tools/cross_colour_ratio_surface_a0_engine.py `
  tests/test_cross_colour_ratio_surface_a0_engine.py

D:\DataAI\.venv\Scripts\python.exe -m pyflakes `
  trkh/tools/cross_colour_ratio_surface_a0_engine.py `
  tests/test_cross_colour_ratio_surface_a0_engine.py

D:\DataAI\.venv\Scripts\python.exe -m pytest `
  tests/test_cross_colour_ratio_surface_a0_engine.py -q
```

Result:

```text
8 passed, 2 upstream export warnings
```

## Direct Same-Tensor Materializer Preflight

The direct materializer is implemented in
`trkh/tools/cross_colour_ratio_surface_a0_materializer.py`. It does not
construct `MangoYOLOCropDataset` in the formal path. It reads the locked CIDT
CSV and geometry first, derives the exact 735 image/label pairs, then
authorizes only those paths in a process-wide dynamic file-open ledger.
Validation/test components, undeclared data-domain inputs, and all writes to
data domains are blocked.

For every selected object, the materializer:

1. matches the target and float32 bbox from the one same-stem label file;
2. reproduces the margin-0.05 integer crop and all intersecting objects;
3. applies the exact keeper evaluation transform, including the existing
   illumination normalization and `desaturate_blur`;
4. requires exact model-box, transformed crop-box, 16x16 valid-mask, and
   bbox-mask parity with the immutable geometry cache;
5. inverts mean/std to the exact post-transform 8-bit sRGB tensor and proves
   bit-exact normalization replay;
6. packs the 256x256 valid mask with a fixed little-endian bit order.

The cache contains no descriptor or model state. Its prospective upper bound
is `160,456,704` bytes: `150,011,904` sRGB bytes, `6,250,496` packed-mask
bytes, and a conservative `4 MiB` for headers/identity metadata. This is
below the locked temporary limit `429,496,730` bytes. Keeping the lossless
same-tensor cache, rather than descriptors, lets all 763 geometry rows pass
before any descriptor exists and avoids a second image decoder.

Windows emits two low-level audit events for one `Path.read_bytes()` call.
The ledger preserves both raw events for exact replay and separately collapses
the adjacent `mode='r'`/`mode=None` pair into one logical open. The formal
gate requires one logical open per unique image and label.

Eleven materializer tests pass. They cover bit-exact parity against the
production dataset on multi-object synthetic images, source-code exclusion of
the broad dataset constructor, label filtering, valid-mask packing, forbidden
path blocking, real lock preflight without cohort pixel/label reads, an
end-to-end synthetic formal artifact set, atomic rename, and forced-failure
cleanup. Authorization now pins the full resource-limit block and exact
CPU-only/no-fit execution constraints; invalid authorization creates no output
parent. Resource limits are checked before and during the image loop, and a
failed replay removes partial finalization files. A Windows regression test
also requires NUL-delimited Git porcelain so protected untracked paths with
spaces are compared by literal path rather than display quoting. The combined
lock/erratum/engine/materializer suite passes `29/29`. Focused CCR/report
verification passes `32/32`; the complete repository suite passes
`1957/1957` in `74.10 s` with 373 existing warnings. Revision-21 Word QA
passes all `18/18` rendered pages and accessibility `0/0/0`.

Current implementation hashes are:

- materializer:
  `cc30d9734a2d75c621f56fed5b90b86a6169e071a33ce51aa0c8edb46f24af22`;
- materializer tests:
  `f8b2962f47d0515d27b11806752a3faf98ea0ebd40b8908bbd8470b4cedf5945`.

The real structural preflight reports 763 rows, 735 unique images, 735 unique
labels, nine logical locked-input opens, zero blocked attempts, no cohort
image/label read, and no descriptor, model state, candidate metric, CUDA,
validation, or test use.

## Formal Attempt 1 Failure and Correction

The one attempt authorized by SHA `e76bd56c...07068` was consumed from pushed
commit `c594549...8f2af`. It stopped fail-closed after approximately 25.4
seconds. All 763 rows completed and image/label manifests, model boxes,
16x16 valid masks, bbox masks, and model-tensor round trips were exact. Only
`crop_boxes_exact` failed. No formal output or temporary directory remained,
and no descriptor, state, metric, CUDA, validation, or test artifact existed.

Source comparison and an independent synthetic reproduction isolated the
failure to production float ordering. Production computes integer crop bounds
from the original parsed primary bbox, then converts every iterated object box
through one `torch.float32` tensor before calculating crop-relative boxes.
The direct loader previously used original decimal object boxes in the second
step. The resulting crop-relative coordinates can differ by one ULP
(`2.98e-8` to `5.96e-8`) while all masks remain identical.

The direct loader now mirrors the production float32 conversion at the same
boundary and keeps the original primary bbox for integer crop bounds. A
dedicated regression case uses image size 2938x1150 and bbox
`(0.617804, 0.027735, 0.303894, 0.8)`. Failure evidence is locked at SHA
`11714304...a554e9`. The consumed authorization must not be reused.

## Recovery Materialization and Fresh Replay

Recovery authorization SHA `7081cece...16e42` was committed and pushed at
`3ac7302c554dc5f2023868ec63775516f1263be2`. It pins the correction commit,
failure evidence, exact module/test/lock/erratum/engine hashes, output path,
resource limits, and CPU-only/no-descriptor/no-fit/no-validation/no-test
constraints.

The replacement materialization completed in `22.013373 s` under PID 28104.
All 763 rows and every manifest, model box, crop box, valid mask, bbox mask,
packed-mask replay, and model-tensor round trip passed with maximum numeric
error `0.0`. Peak process RSS was `744,726,528` bytes; the
`156,728,872`-byte temporary cache stayed below the `429,496,730`-byte limit,
and no CUDA context was used.

The fresh-process replay completed in `21.784652 s` under PID 17800 and passed
all 13 checks. It reproduced the authorization, lock, module, transform,
image/label manifests, access ledger, cohort arrays, sRGB cache, packed valid
masks, and parity record exactly.

The dynamic ledger contains 2,958 raw events collapsed to 1,479 unique
logical reads: 735 train images, 735 train labels, and nine immutable inputs.
Every logical path was opened once; blocked, write-like, validation, and test
counts are all zero. The retained local cache contains only uint8 sRGB,
packed uint8 valid masks, and immutable cohort geometry/identity/keeper
probabilities. It contains no candidate descriptor or candidate state.

Compact recovery evidence is locked at SHA
`e992c297c2416976152694dec95df6d99692dd63af6395f00df99b9efa98c3ed`.
It pins the formal/artifact-set manifests, all cache hashes and sizes, array
schemas, ledger digest, resources, replay checks, and the no-fit scientific
boundary. The 150 MB cache remains a local temporary dependency and is not
added to Git.

Revision-22 Word QA passes all `18/18` rendered pages at original detail with
no clipping, overlap, broken table flow, or malformed source links.
Accessibility findings are `0/0/0`.

## Next Authorized Stage

Commit and push the compact materialization/replay evidence first. Then define
and commit a separate prospective descriptor/four-role head-fit runner
boundary that consumes only the exact local cache and immutable lock. It must
pin output, resource, file-set, replay, no-validation/test, and fail-closed
cleanup rules before descriptor extraction or fitting.

Head fitting, descriptors, candidate metrics, validation/test, production
integration, full train, and current-best command changes remain unauthorized
until that new boundary is committed and pushed.
