# TRKH 5-Class DOLG Orthogonal Local-Global Signal A0 Closure - 2026-07-17

## Decision

ICCV-2021 DOLG object-conditioned orthogonal local-global signal A0 is closed
before OOF readout, trainer/model/config integration, or any image epoch. The
implementation is equation-faithful and deployment-compatible, but the sole
isolated formal run failed the prospectively locked full-forward runtime gate:

- native median latency: `57.028606 ms` at batch 32 BF16;
- DOLG sidecar median latency: `66.032639 ms`;
- runtime ratio: `1.157886x`, above the locked `1.10x` maximum;
- peak-memory ratio: `1.002584x`, within the `1.10x` maximum;
- normal logits and predictions remain exactly unchanged.

The structural gate passed `29/30` checks. Signal OOF and visual review were
not run by design. No short pair, validation, test, full train, nearby sweep,
or current-best command update is authorized.

## Authority And Locked Inputs

- Accepted paper: Yang et al., "DOLG: Single-Stage Image Retrieval With Deep
  Orthogonal Fusion of Local and Global Features," ICCV 2021.
- Author MIT repository commit/tree:
  `63b117d76f660db9617c2081ec771e3cbdf4bcc8` /
  `c7639a440fbd5b25b911737dae392ddfd1d346a6`.
- Accepted-paper/source/license SHA-256 values:
  `3ca7070e9d749fa84c9606b43dabd159242a518bd902817565b27b2fc22fb679` /
  `83b1984de8ca0be585975622b1dba6e85351b1cd6d17b71fb06c3ce17ab6dd54` /
  `4a9589726b97c388428d37e0526743b77f189f531aef987017af06abb1b450d4`.
- Prospective protocol SHA-256:
  `51b52aaeb2810ea4172ccd93c6238add378fd845149808fa4dea1cfc694e150d`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Frozen train-only declaration SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Implementation/test commits: `7df3a45` and `4ac7ffd`.

No DOLG or other pretrained weight was used. The accepted paper, author source,
license, immutable protocol, and direct TRKH replay controlled the decision.

## Geometry And Engineering Results

The CPU-first preflight passed over the exact train-only
`607 = 421 class-1 TP + 186 restricted FP` cohort. Fold counts remained
`112/45`, `100/48`, `101/52`, and `108/41`; ordered-index SHA-256 remained
`a2689d1b...38bd`. Transformed bbox bytes matched
`e9b2143c9dbc7c6f5bf3a38f80a43483bd436e3b5ce99116c3d5989f79892f6b`.
At `16x16`, object/context center supports remained `28..210/46..228`, with
no empty region.

All non-runtime structural checks passed:

- the official five source assignments were AST-replayed exactly;
- official/oracle output errors were `4.44e-16/2.22e-16`;
- local/global gradient errors were `5.55e-17/2.50e-16`;
- pooling-equivalence, relative-orthogonality, and finite-difference errors
  were `7.63e-17`, `2.20e-16`, and `1.73e-11`;
- BF16 descriptor error was `0.007287 < 0.05`, with finite nonzero gradients;
- standard ONNX Runtime error was `6.98e-9`; TensorRT parse and engine build
  passed, and neither binary was retained;
- the real full-forward descriptor executed at `[32,512]`, hooks fired once,
  and normal logits/predictions had zero difference;
- canonical batch-64 declaration replay reproduced only the locked sample
  `3657` near-tie and no other mismatch.

Formal launch isolation was valid. The wrapper observed `0%` GPU utilization,
`1712 MiB` used, and `58 C` immediately before launch. The resource audit began
at `0%` utilization after loading the keeper. No unrelated Python or TensorRT
process existed. The `+15.7886%` latency therefore cannot be dismissed as the
earlier game/GUI load.

## Fail-Closed Scope

The protocol required runtime to pass before feature extraction and OOF. Once
`1.157886x > 1.10x` failed, the auditor correctly wrote a terminal
`rejected_pre_oof_structural_gate` summary. It did not fit logistic readouts,
inspect validation/test, render contact sheets, train a model, or expose a
behavioral metric that could justify relaxing the gate.

Do not rescue or repeat on the current keeper:

- timing repeat count, warm-up, batch size, precision, process/GPU gate, runtime
  threshold, compiler, custom kernel, CUDA graph, Triton, or TensorRT plugin;
- local/global layer, bbox mask, normalization, pooling, projector, descriptor
  width, readout C/solver, fold, seed, threshold, or lighting-condition sweeps;
- ASPP, learned DOLG attention, GeM, ArcFace, CFCD/GLAM additions, or direct
  concat/orthogonal hybrids around this route;
- trainer integration in the hope that end-to-end learning compensates for a
  prospectively failed deployment gate;
- combinations with closed local-global coupling, token routing, part token,
  second-order, frequency, texture, activation, normalization, loss,
  distillation, or post-hoc families.

A future route must come from distinct accepted primary work with licensed
official code and must pass its own cheap precision-safe signal and deployment
gate before trainer integration.

## Retained Evidence

Compact formal evidence remains at
`runs/audit_dolg_orthogonal_local_global_signal_a0_20260717`:

- terminal `summary.json` SHA-256:
  `d6f0d371c2303e2042d916cada3c02d8dc973dd12661e9a11d0884305435e59d`;
- `report.md` SHA-256:
  `5535da615c0bb99380c0cb3eff2ac255d1d68de93305406cc55df541ae2a0766`;
- artifact-manifest SHA-256:
  `b076914e77859715591482eaf527bd33a750dc2d33d271283d603e8c5fcb093a`.

The manifest payload is `13,554` bytes and contains no checkpoint, feature
cache, ONNX, engine, copied image, or trainable artifact. Nothing is large or
obsolete enough to delete. No raw dataset file was modified; validation and
test were not loaded.

## Verification And Retention

Python compile, pyflakes, PowerShell AST parse, focused tests `11/11`, full
pytest `1383/1383`, clean pushed-commit preflight, real-checkpoint CPU forward,
formal equation/deployment checks, and immutable-output checks passed.

Read-only retention at
`runs/audit_trkh_artifact_retention_post_dolg_20260717` passed over `747`
directories and all `48` compaction manifests. All `210` compacted originals
remain absent, `blockers=[]`, no deletion/raw-data touch occurred, and free
space is `103.074 GB`. Retention summary SHA-256 is
`062c4f226af3c77ad7eebfa5b5a7391b257f6f43b39cf2651551f39aa798757f`.

Current-best command/history SHA-256 values remain
`36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
`39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
They remain at three revisions and two actual updates.
