# TRKH 5-Class Scratch, Inference, And Resource-Value Audit

Date: 2026-07-24
Branch: `classification-only-research`
Decision: tighten the gate before any new long GPU experiment

## Executive Finding

The current research record is scientifically useful but has not yet produced a
new production checkpoint. The current-best command packet has three tracked
revisions, two post-initial updates, and zero selected single-model checkpoint
replacements. The repository contains 68 closure documents, 90 protocol
documents, and 48 audit documents. This is too much process unless each next
run closes a genuinely distinct uncertainty at bounded cost.

The retained `runs` tree currently contains 149,044 files and approximately
11.742 GiB. The largest roots are:

| Root | Size | Files | Current interpretation |
|---|---:|---:|---|
| `yolof_oof_folds_train5_20260704` | 2.488 GiB | 96,774 | High cleanup priority after retention proof |
| `audit_bbox_logpolar_stem_a0_20260720` | 0.806 GiB | 10 | Large compact payload; retain only if manifest requires it |
| `audit_augself_color_adapter_readiness_20260716` | 0.708 GiB | 70 | Closed route; candidate for evidence compaction |
| `yolof_cidt_fold0_trainonly_20260716` | 0.498 GiB | 19,354 | Shared train-only evidence; do not delete before dependency audit |
| current keeper | 0.339 GiB | 460 | Protected |
| random-init full train | 0.285 GiB | 1,860 | Rejected but reportable full-run evidence |

This audit does not delete any path. Cleanup requires a separate dependency and
retention manifest.

## Scratch-Only Provenance

The proposed classifier may borrow architecture or equations, but it must not
load pretrained/external parameters. Pretrained models may remain isolated
comparison baselines. An external pretrained image generator is a separately
disclosed synthetic-data prior and is never part of classifier inference.

The selected keeper checkpoint has SHA-256
`1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
Direct checkpoint inspection records:

- `model_config.pretrained = false`;
- no resume lineage in the keeper checkpoint;
- no distillation teacher checkpoint path;
- the model weight source is its own EMA state;
- the enabled focus-binary signal reads an offline probability CSV rather than
  loading another model at training or inference.

The offline teacher CSV has SHA-256
`a5ca8a45bede61fd5a09fb833b77fd8c18f6f862747e1a30f48196e21df0318d`.
Its retained summary calls it a weighted no-pretrain TRKH ensemble and has
SHA-256
`445e72c15f05715deef3b035c059c5a93b5a4d8dbc25111380af86514e58ac2e`.
The contributing probe directories were compacted by
`cleanup_manifest_20260702_obsolete_smoke_probe.json`, SHA-256
`ce731b7e349118c04c6ee9b28830967760536af36ace60702c0961416eba24b1`.

This is sufficient to show that the current candidate does not load external
pretrained parameters. It is not a complete cryptographic parent-checkpoint
chain because the contributing probe checkpoints were deleted by the legacy
cleanup. Future teacher caches must be rejected unless every source checkpoint,
source config, source hash, scratch flag, split, and generation command is
attested before compaction.

The teacher mixture weights were selected on validation in the historical
experiment. This is ordinary hyperparameter selection rather than pixel
leakage, but its own validation score is optimistic and cannot be treated as an
untouched estimate. The locked final test remains the only independent report
for that keeper. Future teacher construction must be train-OOF or
source-disjoint cross-fit before validation promotion.

## Architectures Beyond CNN And Transformer

The project has already tested or screened several other neural computation
families:

| Family | Local result | Decision |
|---|---|---|
| Selective state-space / Mamba | Scratch MambaVision-Nano reached validation macro/class-1 F1 `0.78048/0.47733` at epoch 5, with 168 class-1 FP | Closed on the fixed compact protocol |
| Dynamic graph neural mixing / ViG | Class-1 precision `+0.01011`, but recall `-0.22936` and F1 `-0.14745` | Closed; similarity neighbors are not reliability evidence |
| Dynamic-routing capsules | Macro/class-1 F1 changed by `-0.01266/-0.00791`; routing stayed near uniform | Closed |
| Recurrent aggregation | Class-1 F1 improved `+0.01470`, but restricted FP rose `12 -> 30`; recurrent and final-only averaging were identical | Mechanism had no recurrent causal value |
| LSTM integral-region context / CAP | Prospective train-only A0; first formal exposed no metric because XAI failed before artifact creation | Same-lock rerun allowed after audit-only fix |
| Nonnegative matrix factorization / Hamburger | FP rejection rose, but 65/528 class-1 TP broke and seed stability failed | Closed |
| Bilinear covariance / DeepBDC | Tested with matched controls, export and runtime gates; did not authorize production integration | Closed |
| Quaternion, topology, morphology, Gabor, log-polar and frequency operators | Multiple locked A0s failed selectivity, TP protection, robustness, or inference gates | Closed exact neighborhoods |

Fresh primary-source screening also covered alternatives not yet worth a local
run:

- Spiking vision models require multiple simulation time steps and commonly
  use long schedules. SpikingResformer reports a 320-epoch ImageNet recipe.
  Its energy claim targets spike operations or neuromorphic assumptions, not
  guaranteed RTX 4060 latency, so it fails the current `<=30` epoch and
  conventional-inference fit.
- Deep equilibrium vision models require iterative fixed-point solvers at
  inference. The official repository recommends four GPUs and notes that
  multiscale DEQ can be slower as image size and solver iterations grow. This
  conflicts with the deployment objective.
- KAN/KAConvNet is conceptually distinct, but the April-2026 paper itself
  identifies B-spline inefficiency and overfitting in prior KANs; the official
  repository had only two stars and three commits when checked. It is retained
  as a low-confidence research lead, not authorization for a GPU run.
- Vision MLP architectures such as Hire-MLP provide an efficient non-attention
  mixer, but the local project has already tested several token/FFN/local-mixer
  families. A new run would need a class-1 surface-selectivity mechanism, not
  merely an MLP replacement.

Primary references:

- MambaVision: <https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html>
- Dynamic routing capsules:
  <https://proceedings.neurips.cc/paper/2017/hash/2cad8fa47bbef282badbb8de5374b894-Abstract.html>
- Vision GNN:
  <https://openreview.net/forum?id=htM1WJZVB2I>
- SpikingResformer:
  <https://openaccess.thecvf.com/content/CVPR2024/html/Shi_SpikingResformer_Bridging_ResNet_and_Vision_Transformer_in_Spiking_Neural_Networks_CVPR_2024_paper.html>
- Deep Equilibrium Models: <https://github.com/locuslab/deq>
- KAConvNet: <https://arxiv.org/abs/2604.23320> and
  <https://github.com/UnicomAI/KAConvNet>
- Hire-MLP:
  <https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Hire-MLP_Vision_MLP_via_Hierarchical_Rearrangement_CVPR_2022_paper.html>

## What Has Real Value

No post-keeper architecture currently has production value. The useful results
are narrower:

1. Repeated perturbation and XAI evidence shows that far background is usually
   not the dominant error source. Object desaturation and local
   surface/boundary changes alter predictions much more.
2. The class-1 failure is two-sided. Methods that lower `0/2/4 -> 1` false
   positives often destroy true class-1 recall. Precision-only improvement is
   not acceptable without a TP-retention budget.
3. Scratch Mamba, graph, capsule, NMF, recurrent averaging, global priors and
   many post-hoc routers have failed under matched controls. This prevents
   repeating attractive but non-causal families.
4. The engineering path now has source-aware split controls, dynamic file-open
   ledgers, exact replay, XAI, ONNX checks, TensorRT feasibility checks,
   loader benchmarking, manifest-based cleanup and VS Code-safe launchers.
5. Worker benchmarking improved the measured training loader from about
   `107.006` to `196.346` images/s on the same host. This is a real efficiency
   gain even though it is not a model-quality gain.

The process failure is also real: the first CAP formal spent roughly 90 minutes
on all 30 fold-role jobs and then lost the result because the end-only XAI step
used cuDNN LSTM evaluation backward. No metric or output artifact existed.
Future long auditors must run the exact XAI/backend path during preflight and
checkpoint sufficient intermediate state to avoid repeating completed work
when the prospective artifact contract permits it.

## Mandatory Gate From This Point

Before a new method can consume a long GPU run, its protocol must lock:

1. a mechanism not already closed by the no-repeat map;
2. the exact evidence gap it can resolve;
3. reused frozen artifacts and why new image passes are unavoidable;
4. wall-time, VRAM and disk ceilings;
5. an early information gate that does not open validation/test;
6. class-1 TP-retention and restricted-FP budgets;
7. matched parameter/runtime controls and causal derangement;
8. a relative inference budget against the same keeper and runtime:
   batch-1 mean/p95 latency, batch throughput, peak VRAM, parameter count,
   ONNX error and TensorRT feasibility;
9. a fail-closed stop rule and post-run cleanup rule;
10. one of four value labels: `promotable`, `mechanism-only`,
    `negative-but-reusable`, or `waste/invalid`.

Accuracy cannot override a failed deployment gate. No full image smoke is
authorized from an embedding/head A0 unless every prospective information,
causal, resource, replay and visual gate passes. No full train is authorized
without a validation-locked winner and convergence evidence under 30 epochs.

## CAP Decision

CAP remains the only active run because its equation, folds, controls, resource
limits, inference checks and visual rows were locked before candidate metrics.
It tests ordered multi-scale region context with an LSTM and NetVLAD, which is
not a cosmetic CNN/Transformer recombination. The failed formal produced zero
metrics, so an audit-only native-RNN attribution correction does not create
hindsight. One same-lock rerun is allowed after focused/full tests, report QA,
clean commit and push. Any additional failure or failed scientific gate closes
CAP; no neighboring sweep is permitted.
