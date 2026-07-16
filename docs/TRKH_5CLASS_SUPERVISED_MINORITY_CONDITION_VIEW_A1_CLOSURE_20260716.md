# TRKH 5-Class Supervised-Minority Condition-View A1 Closure

Date: 2026-07-16

Status: rejected before representation training by the immutable train-only
pre-training selectivity gate. Validation, test, shared-trainer integration,
full train, keeper replacement, and current-best command revision are denied.

## Authority And Execution

- Primary authority: Mildenberger et al., CVPR 2025, as served by CVF:
  https://openaccess.thecvf.com/content/CVPR2025/html/Mildenberger_A_Tale_of_Two_Classes_Adapting_Supervised_Contrastive_Learning_to_CVPR_2025_paper.html
- Official MIT implementation: https://github.com/aiforvision/TTC
- Official source commit/tree:
  `0b1e6974254993b074ad27a226c7ce864da7f95c` /
  `707c9b34ece30ac4a23f1de28be9d0ee803710b7`.
- A1 protocol SHA-256:
  `e17f5cb56631b1f24206b6183fdef7b92e17a050097b84c600adb0a0744d9aa2`.
- Protocol precommit: `52f8be8c5a7637f4d7abf7826d4e15898165fec8`.
- Implementation/preflight commit:
  `45e9fda3fdc02bfa8fec056c14e0cc26c5f0e651`.
- Formal output:
  `runs/audit_supervised_minority_condition_view_a1_fp32_20260716`.

No secondary or user-supplied research report defined the method or gate.

## Numeric Contract Passed

- All four clean/dim/bright/low-contrast caches contain the same ordered 9,215
  `yolo_f/train` objects, exact targets/sample indices, finite 256D pooled
  features, and finite five-class logits.
- Every cache states `fp32_autocast_disabled`; extraction batch is 64.
- Full clean CIDT replay has maximum probability error
  `8.940696716308594e-08` against limit `1e-6`, with zero argmax mismatches.
- Immediate 64-row replay has exactly zero pooled/logit error and exact target
  and sample-index tensors.
- Source-disjoint fold-0 cohorts remain exact: 7,372 fit rows/6,452 sources and
  1,843 holdout rows/1,612 sources, with zero source overlap. Holdout fixed
  clean cohorts are 107 class-1 TP and 36 restricted hard FP.
- Cache elapsed times were clean `398.23s`, dim `331.85s`, bright `336.60s`,
  and low contrast `327.61s`; total cache time was about `1,394.30s`. Formal
  wall time was `1,493.3s`. Windows safe loading used zero effective workers;
  observed GPU utilization stayed around 4-5% with about 3.2 GiB VRAM.

The A0 FP16/CIDT mismatch is therefore fully corrected. A1 is a valid method
readiness result rather than an infrastructure failure.

## Pre-Training Representation Result

The fixed k-NN neighborhood contains 369 fit rows. Metrics are computed only
from the source-disjoint train fit/holdout split.

| Condition | Class-1 CAC | Non-class-1 CAC | TP-vs-hard AUROC | TP-hard gap | Clean-to-condition SAA |
|---|---:|---:|---:|---:|---:|
| clean | 0.339450 | 0.994810 | 0.589174 | 0.040188 | 1.000000 |
| dim | 0.000000 | 1.000000 | 0.490784 | -0.006411 | 0.163863 |
| bright | 0.000000 | 1.000000 | 0.501947 | 0.002376 | 0.315247 |
| low contrast | 0.000000 | 1.000000 | 0.538811 | 0.015388 | 0.131850 |

Clean already misses the required TP-vs-hard AUROC `>=0.60`. Every shifted
condition misses AUROC, TP-hard gap, class-1 CAC, and SAA gates. Class-1 CAC
collapsing to zero means none of the class-1 holdout rows receives a majority
class-1 label among its 369 nearest condition-matched fit neighbors. The low
SAA values show that the keeper pooled embedding does not preserve individual
sample identity under these deterministic illumination changes.

These failures are material to the method: forcing paired condition views
together on this frozen representation would not start from selective evidence
that separates true class-1 support from the exact false-positive cohort. The
protocol correctly stops before an adapter can blur that boundary.

## Stop Outcome And Audit Coverage

- `representation_training_started=false` and `probe_training_started=false`;
  no epoch, optimizer, model binary, ONNX, or deployment payload exists.
- XAI is required only after a trained probe under the locked protocol. Because
  selectivity rejected A1 before epoch 1, `xai_required=false`; no XAI claim is
  made and no XAI omission is hidden.
- Validation/test flags are both false. Only source-disjoint `yolo_f/train`
  paths were opened, and the raw dataset was not modified.
- Independent replay recomputed the complete selectivity gate from
  `representation_metrics.json` and matched the stored gate and ordered failed
  checks exactly.

Formal payload hashes:

- `summary.json`:
  `6ae437b2ddb6c87515863c5c4bf207e15183ca08dd43e75582e162a652b38f22`
- `representation_metrics.json`:
  `77f66bb27616b8429e7ff4df8da416e23492c3a936a63e363a1c690b5d2978bf`
- `report.md`:
  `f63e97ffecb643f9b58962f7264fdce65420e335aef0ee64762cb2ee624f6245`
- `artifact_manifest.json`:
  `b84cbbd89d538548f4819a2f1b0484fd0ea52a0e23764e6687ff58f3f07a3e7f`

The formal `report.md` title says A0 due to a cosmetic hard-coded generator
label. Its method field in `summary.json` and `artifact_manifest.json` is the
correct A1 identifier, and every metric/hash is unaffected. The generator is
fixed after the formal run and covered by a regression test; formal evidence
is intentionally not rewritten.

## No-Repeat Boundary

Close the exact frozen-keeper pooled-feature Supervised-Minority condition-view
adapter route. Do not sweep k, CAC threshold, condition strengths/order,
condition pairing, cache dtype/batch, minority ratio, temperature, adapter or
projector width, residual scale, optimizer, learning rate, epochs, fold, seed,
sampling, threshold, or router on this keeper.

This is not a universal rejection of Supervised Minority. A future revisit
would need a fundamentally new scratch image representation whose train-only
features first demonstrate condition-stable class-1 TP-versus-hard-FP
selectivity; it may not be presented as an A1 parameter continuation.

## Engineering Closure

- Focused tests: `14/14`.
- Full pytest: `1217/1217` with 257 warnings.
- Compileall, launcher parse/preflight, independent gate/payload replay, and
  all four protected hashes passed.
- Read-only retention audit
  `runs/artifact_retention_audit_ttc_supmin_a1_closure_20260716` passed over
  709 run directories with `blockers=[]`, no deletion, and 71.273 GiB free.
- Retention summary SHA-256:
  `9e8234e079c211443adeb8c5a9fdd4f862bce93571e235e0ebe0c1727df26f5b`.

Current-best command tracking remains three revisions, two updates after the
initial revision, and zero keeper replacements. The current-best command and
history files are unchanged.
