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

## Next Authorized Stage

The next stage is a direct, fail-closed cohort tensor materializer. It must:

1. use only locked CIDT image identity, target, source, sample identity, and
   geometry;
2. avoid constructing the dataset class that scans all train labels;
3. reproduce the exact keeper crop and frozen evaluation transform;
4. prove row-by-row parity of crop boxes, model boxes, bbox masks, valid
   masks, identities, targets, sources, and manifests;
5. cache only the minimum descriptor tensors within the prospective storage
   budget;
6. stop before fit if any parity, process, disk, CUDA, or provenance gate
   fails.

Only after that materializer is independently tested and committed may the
minimum locked A0 image pass and four-role fold fit begin.
