# TRKH Pair-Surface DDF A0 Engine Implementation

Date: 2026-07-25

State: synthetic engineering implementation complete; no cached/raw image,
label, candidate fit, score, metric, validation/test access, production
integration, full train, or current-command change

## Scope

This stage implements only the hash-locked Pair-Surface DDF equation and its
synthetic engineering checks:

- `trkh/tools/pair_surface_ddf_a0_engine.py`;
- `trkh/tools/preflight_pair_surface_ddf_a0.py`;
- `tests/test_pair_surface_ddf_a0_engine.py`;
- `tests/test_preflight_pair_surface_ddf_a0.py`.

The engine contains no dataset, cache, CIDT, checkpoint, validation, or test
path. It cannot launch a fit. The preflight reads only the committed
prospective lock and synthetic tensors.

## Implemented Contract

- standard-operator row-major nine-shift multiplicative DDF;
- official sample-standard-deviation filter normalization and learned channel
  scale semantics;
- independent FP64 NumPy loop oracle;
- exact 9,380-parameter `ddf_full` and 9,435-parameter `static_matched`;
- spatial-only and channel-only neutral-factor controls;
- independent full-repeat initialization;
- tensor-name-derived deterministic initialization;
- separate four-map evidence and spatial-softmax attention heads;
- same-weight filter override, spatial roll, and neutral-factor causal paths;
- fixed input transform, balanced BCE-with-logits tasks, bbox-attention loss,
  and 20-epoch cosine schedule;
- dynamic-batch standard-domain ONNX opset 17 and ONNX Runtime replay;
- TensorRT 10.7 FP16 parser/build with batch `1..32` profile.

Synthetic tests verify finite role paths, bit-identical shared initialization,
all intended candidate gradient groups, optimizer updates, attention
normalization, loss weights, causal score changes, and static-role rejection
of dynamic overrides.

## Engineering Attempts

### Attempt 1: invalid role comparison and eager timing

Retained summary:
`runs/audit_pair_surface_ddf_a0_engineering_preflight_20260725/summary.json`

SHA-256:
`fefee523c3833511328941c0027fec9e4ed9130649cb4d8fdeb286a480cd468b`

The equation, ONNX, ORT, and TensorRT checks passed. The role contract was
invalid because the harness compared BatchNorm running buffers after each
role had already executed in train mode. Eager batch-1 mean/p95 was
`1.860261/2.601318 ms`; only the locked `2.5 ms` p95 gate failed. No candidate
data was read.

### Attempt 2: corrected role comparison, unstable eager p95

Retained summary:
`runs/audit_pair_surface_ddf_a0_engineering_preflight_recovery_20260725/summary.json`

SHA-256:
`0193dd98a3d9404d6f77fdbdab49ec54b1589ed5709c0ad43e157a76707676b1`

The role contract became bit-exact and every non-timing gate passed. Eager
timing before TensorRT remained unstable under Windows WDDM:

- batch 1 mean/p95 `2.213581/3.966939 ms`;
- batch 32 mean/p95 `2.316314/4.115659 ms`;
- peak CUDA `996,864/31,694,848` bytes.

This ruled out treating one favorable eager p95 as deployment evidence. The
gate and architecture were not relaxed.

### Attempt 3: fixed-shape CUDA Graph branch backend

Retained summary:
`runs/audit_pair_surface_ddf_a0_engineering_preflight_cuda_graph_20260725/summary.json`

SHA-256:
`62e1c4dd4fe5720326501ac397c97e0a4392a633cecfba5c77f56911e6f73bdf`

CUDA Graph removes launch gaps for the fixed batch-1 and batch-32 branch
bindings without changing a parameter, equation, or output. Eager and graph
outputs are bit-exact (`max_abs=0`).

Final measured branch-compute results:

| Check | Observed | Locked limit |
|---|---:|---:|
| FP64 NumPy oracle max abs | `8.8817842e-16` | `1e-10` |
| Batch-1 FP16 mean | `0.182688 ms` | `2.0 ms` |
| Batch-1 FP16 p95 | `0.221491 ms` | `2.5 ms` |
| Batch-32 FP16 mean | `0.515338 ms` | `2.5 ms` |
| Batch-32 FP16 p95 | `0.825498 ms` | `3.0 ms` |
| Peak CUDA batch 1/32 | `997,888/31,695,872 B` | `134,217,728 B` |
| ONNX Runtime max abs | `2.2351742e-8` | `1e-5` |
| TensorRT FP16 engine build | pass, `1,355,748 B` | pass |

The latency scope is sidecar compute on a static CUDA binding. It excludes
host transfer and preprocessing. This does not waive the future matched
end-to-end keeper integration gates, which must include all input handling,
fusion, and output work under the same backend.

## Decision

Engineering status is `passed_mechanism_unproven`. Preserve all three compact
attempt summaries because they distinguish a harness defect and WDDM eager
jitter from the final deployable branch path.

The next allowed stage is implementation of the hash-locked train-only
auditor, dynamic access ledger, exact replay, retention, and authorization
builder. It may not read the retained 763-row cache until that implementation
is committed/pushed and a separate one-formal/one-replay authorization is
also committed/pushed.
