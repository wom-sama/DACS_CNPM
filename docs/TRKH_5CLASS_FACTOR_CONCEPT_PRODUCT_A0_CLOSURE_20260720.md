# Factor-Concept Product A0 Closure (2026-07-20)

## Decision

Reject the fixed mango factor code, signed core-minus-ring transport descriptor,
and deterministic product decoder on the current no-pretrain keeper. The sole
source-disjoint train-only A0 passes every structural gate but fails 14 of 15
mechanism gates. It does not authorize validation/test access, XAI, model or
trainer integration, a smoke, a probe, full training, or current-best command
promotion.

## Locked Evidence

- Accepted source: Koh et al., Concept Bottleneck Models, ICML 2020.
- Official MIT source: `yewsiang/ConceptBottleneck` at commit/tree
  `d6353f2...d583f8`/`d93ca72...af4eb9`; no source or weights were copied.
- Protocol SHA-256: `bf3cb5ac...9b78b`; implementation commit: `f527029`.
- Auditor/test/launcher SHA-256:
  `58a9cd7c...d0416`/`28a1cc54...d9cd0`/`aa43d6aa...18d96`.
- Formal summary/manifest/prediction SHA-256:
  `438b8f90...8d4e`/`a4682bb3...e32f`/`92c87ab3...ea23`.
- Keeper SHA-256: `1f49d577...2677`.

The formal uses only the immutable 9,215-row `yolo_f/train` descriptor cache.
Five `StratifiedGroupKFold` folds cover every row exactly once with zero source
overlap. Every StandardScaler is fitted on fit rows only. All 25 fixed `lbfgs`
readouts converge in 67-187 iterations without retry. Input, source, code,
paper, checkpoint, data-YAML, and current-command hashes all match. Validation
and test pixels, labels, predictions, and metrics remain unopened.

## Result

| Metric | Matched 5-class control | Factor product | Delta |
|---|---:|---:|---:|
| Accuracy | 0.949864 | 0.882474 | -0.067390 |
| Macro F1 | 0.920172 | 0.825983 | -0.094188 |
| Class-1 precision | 0.725293 | 0.395425 | -0.329868 |
| Class-1 recall | 0.800370 | 0.670980 | -0.129390 |
| Class-1 F1 | 0.760984 | 0.497601 | -0.263383 |
| Predicted class-1 support | 597 | 918 | +321 |

Factor versus control produces only `140` corrections and `761` harms. It
rescues `49` class-1 false negatives but breaks `119` true positives. It removes
`39` restricted `{0,2,4}->1` false positives while creating `430`, a net change
of `-391`. Non-focus F1 losses are `0.12104/0.05032/0.03675/-0.00055` for
classes `0/2/3/4`.

Every fold independently rejects the route. Control-to-factor macro-F1 deltas
are `-0.11135/-0.09215/-0.07936/-0.08761/-0.10252`; class-1 precision falls in
all five folds. Restricted-FP net removals are
`-96/-68/-74/-71/-82`, and class-1 net TP losses are `17/23/4/12/14`.

## Mechanism Diagnosis

The global-head maturity factor is highly identifiable: three-state macro F1
is `0.98316`. The proposed transport/damage factor is not. Signed
`core_second_order - ring_second_order` obtains state-1 precision/recall/F1
`0.32223/0.66174/0.43341`. It maps `627` true transport-state-0 rows to rare
state 1, so the conjunction cannot veto class-1 FP and instead expands them.

The signed descriptor does not distinguish keeper class-1 TP from restricted
FP: AUROC is `0.51915`. The generic `core + ring` and context placebos score
`0.58306` and `0.54585`, so the declared interior-versus-boundary mechanism is
both below chance-level usefulness and worse than generic energy. Although
factor-minus-control class-1 probability separates the tiny keeper-FN cohort
from restricted FP at AUROC `0.64910`, its AP is only `0.07432` and it cannot
compensate for the aggregate, TP-safety, FP, placebo, and all-fold failures.

This is not a convergence, calibration, fold, source-leakage, or runtime
failure. It is an information failure in the transport representation plus a
rare-state over-expansion under the fixed balanced product readout. Close the
exact factor code, signed descriptor, product decoder, concept prior,
temperature, threshold, class-weight, `C`, seed, residual/blend, and nearby
factor-head sweeps on the current keeper. The stronger sum placebo remains far
below its prospective direction gate and is not an escalation candidate.

## Runtime And Retention

The formal takes `15.023 s`, including `12.745 s` for all OOF fits, and peaks
at `0.6425 GiB` sampled RSS with eight BLAS threads. Exact CSV replay rebuilds
all metrics and gates with maximum numerical difference `0.0` and verifies all
five payload hashes. The five payloads total `7,816,780` bytes, so retain
`runs/audit_factor_concept_product_a0_20260720` without compaction.

Focused tests pass `12/12`, and the complete repository suite passes
`1561/1561`. The post-closure read-only retention audit covers `781` run
directories and all `50` valid compaction manifests. All `219` manifest-derived
original directories remain absent, `deleted_anything=false`, and
`blockers=[]`; its summary SHA-256 is `e15c0485...adcf`.

Current-best checkpoint, full-pipeline command, and update-history hashes stay
`1f49d577...2677`, `36b9aa1a...0faf`, and `39bd2879...8f53`.
