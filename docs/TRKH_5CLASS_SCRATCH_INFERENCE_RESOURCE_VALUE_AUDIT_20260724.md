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
| `yolof_oof_folds_train5_20260704` | 2.488 GiB logical | 96,774 | Protected hardlink fold materialization; low reclaim value |
| `audit_bbox_logpolar_stem_a0_20260720` | 0.806 GiB | 10 | Large compact payload; retain only if manifest requires it |
| `audit_augself_color_adapter_readiness_20260716` | 0.708 GiB | 70 | Closed route; candidate for evidence compaction |
| `yolof_cidt_fold0_trainonly_20260716` | 0.498 GiB | 19,354 | Shared train-only evidence; do not delete before dependency audit |
| current keeper | 0.339 GiB | 460 | Protected |
| random-init full train | 0.285 GiB | 1,860 | Rejected but reportable full-run evidence |

This audit does not delete any path. Cleanup requires a separate dependency and
retention manifest.

The OOF root needs special interpretation. Its summary records 8,064 source
images, 9,215 objects, and `96,768` materialized hardlinks with `0` copies.
The 2.488-GiB traversal total counts the same NTFS file content through many
directory entries; it is not 2.488 GiB of independently reclaimable data.
Those fold paths are also live inputs for retained OOF/CIDT evidence. Do not
prioritize or delete this root for disk reclamation. At most, a future
dependency-complete migration may recover directory metadata after replacing
every consumer with an immutable manifest.

The post-CAP closure retention pass is read-only and covers 860 run
directories plus all 51 valid object-schema compaction manifests. It confirms
all 14 protected artifact checks present, all 220 compacted originals absent,
zero blockers, zero deletion, and `93.084 GiB` free on the runs volume. Seven
future cleanup candidates total only `31.602 MiB`, so no speculative cleanup
is justified. The immutable summary SHA is
`99d1835a85246d6d9ad06768e4eab2d530b7bc06e8e3f6f686b67c080604a636`.

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
| LSTM integral-region context / CAP | Same-lock A0 AUROC `0.560111`; removes 5 FP, breaks 14 TP, net `-9`; macro/class-1 F1 `0.937491/0.805643` | Closed exact A0 after exact replay and failed manual XAI |
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
- I2-HOFI combines CNN features with inter- and intra-region graph interaction,
  but the official TensorFlow/Spektral repository recommends at least 16 GB
  VRAM and still lists its inference script as an upcoming item. It overlaps
  the already failed graph/region-interaction family and has no matched
  deployment evidence for this 8-GB Windows host, so it is a no-run screen.
- Feature Magnitude Regularization explicitly addresses bias in pretrained
  features in a low-data transfer setting. The proposed TRKH classifier must
  be trained from scratch, so its motivating failure is not the current
  scratch representation bottleneck. It is retained as a reference, not an
  experiment.
- AD-Net combines augmented views and self-distillation with zero additional
  inference modules. Its published setting is low-data fine-tuning, while the
  local project has already closed nearby multi-view/self-distillation routes
  without a new class-1 surface signal. A local run would be overlapping
  rather than equation-distinct.

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
- I2-HOFI paper/repository:
  <https://link.springer.com/article/10.1007/s11263-024-02260-y> and
  <https://github.com/Arindam-1991/I2-HOFI>
- Feature Magnitude Regularization:
  <https://arxiv.org/abs/2409.01672>
- AD-Net paper/repository:
  <https://arxiv.org/abs/2406.19814> and
  <https://github.com/demidovd98/fgic_lowd>

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

The corrected same-lock CAP formal and fresh-process replay later completed.
The candidate AUROC was only `0.560111`, statistically and causally no better
than its spatial/cross-sample controls, while the existing keeper-margin
control reached `0.836149`. CAP removed five restricted false positives but
broke 14 class-1 true positives. Exact replay and complete manual XAI review
make this a reusable negative result rather than authorization to sweep.

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

CAP is closed as `negative-but-reusable`. The corrected same-lock run preserves
the prospective equation, folds, controls, 30 orders, thresholds, resource
limits, and visual rows. Formal performance fails: candidate/seed-repeat AUROC
is `0.560111/0.541163`, the candidate yields net `-9` corrections, and full
macro/class-1 F1 falls to `0.937491/0.805643`. Replay reproduces every numeric
value with error `0.0`, but all 20 visual rows fail manual approval because
region maps are coarse/control-like and feature attribution is often diffuse.

The CAP lock covered parameter count plus mechanism-run CUDA/RSS ceilings. It
did not measure matched batch-1 mean/p95 latency, throughput, end-to-end
inference VRAM, ONNX parity, or TensorRT feasibility. Calling those checks
complete would be incorrect. No inference smoke is needed after the scientific
failure, and no integration, validation/test access, probe, full train, or
current-best command update is authorized. Do not sweep neighboring region,
LSTM, NetVLAD, optimizer, threshold, seed, or captured-layer settings.

## Post-CAP Successor Resource Screen

The 2026-07-25 primary-source screen covers Self-ONN, learnable morphological
neurons, PointNet/Deep Sets, DeepEMD/optimal transport, normalizing flows, and
hierarchical local-material recognition. None satisfies all four immediate
requirements: equation distinctness from local closures, new
sample-conditional class1-vs-0/2/4 surface evidence, scratch convergence within
the project budget, and a plausible matched standard-op inference contract.
The detailed decisions and sources are in
`TRKH_5CLASS_POST_CAP_SUCCESSOR_NO_RUN_SCREEN_20260725.md`.

No implementation or experiment is authorized from this screen. Its resource
value is the avoided cost of six overlapping or assumption-mismatched probes:
zero new GPU time, zero validation/test reads, zero checkpoint/ONNX/engine
artifacts, and no command change. A future proposal must first pass a
source-disjoint train-only information and causal-control gate on explicit
pair-specific bbox-valid surface evidence; a generic replacement neuron,
pooling layer, density head, or set aggregator is insufficient.

## CCR Prospective Resource Gate

Cross Colour Ratio is admitted only as a bounded information gate after the
post-CAP screen, not as an architecture promotion. Its log cross-channel
derivatives have an analytic common-shading/per-channel-gain cancellation
under the original diffuse, approximately linear narrow-band assumptions.
That mechanism is cheap and standard-op compatible, but camera tone curves,
auto white balance, clipping, wet fruit, and specular reflection can invalidate
it. The gate therefore compares CCR with an equal-size plain colour-ratio
derivative control and both trained and same-weight spatial dephasing.

The prospective lock fixes 763 train-only rows, 735 unique images, five
source-held folds, a 3,004-parameter scratch evidence head, 20 epochs, batch
64, workers 4, no augmentation/weighting/oversampling, and no validation/test
access. It caps A0 at 20 minutes, fitting VRAM below 2 GiB, temporary disk at
0.40 GiB, and retained output at 0.10 GiB. Clean information, pairwise,
TP-retention, restricted-FP rejection, seed, causal-control, and replay gates
must all pass before robustness, XAI, or matched inference work can run.

Protocol/lock SHAs are `1e408502...f347` and
`6c2a604d...c835`. The lock state is
`prospective_no_candidate_observation`: no metric, GPU run, production model
edit, checkpoint, full train, or current-best command update exists. Commit
and push this boundary before candidate code so a later result cannot alter
the accepted cohort, controls, gates, or resource limits.

The pre-implementation same-tensor review adds one fail-closed erratum at SHA
`f9a901b1...d03b`. The CCR branch must consume the frozen keeper evaluation
tensor, which already includes its selected `desaturate_blur` preprocessing;
it may not add or alter suppression. To keep the read set exact, the auditor
must crop from locked CIDT paths and geometry rather than constructing the
full train dataset index. Crop-box and valid-mask parity are required before
descriptor extraction. This correction occurred before candidate code,
image reads, fitting, metrics, or resource use.

Pre-commit verification passes focused lock/erratum/report tests `13/13` and
complete pytest `1938/1938` in `75.90 s` with 296 existing warnings. The
revision-17 report passes visual QA on all `16/16` pages and accessibility
`0/0/0`. This verification consumed no dataset pixels, GPU candidate run,
checkpoint, validation/test access, or current-best command revision.

## CCR Equation-Engine Resource Result

The pre-fit CCR engine is a bounded engineering result, not a model result.
It adds a fixed descriptor path and a 3,004-parameter scratch head without
touching the production trainer. Equation, physical-invariance, fold/order,
state/gradient, and dynamic-batch ONNX Runtime tests pass `8/8`. The ONNX graph
uses only standard-domain operators and stays below the locked `1e-5` replay
error limit.

No cohort image, CUDA fitting job, validation/test row, checkpoint, or run
directory was consumed to obtain this result. The only cohort dependency read
was immutable geometry metadata used to reproduce hashes that were already
fixed prospectively. Engine/test/implementation SHAs are
`ac610d7c...d26fe9`, `5a83a138...94c840`, and `18c2dbeb...8c804`.

The value label remains `mechanism-only-preflight`. Parameter count and ONNX
compatibility do not establish keeper-relative batch-1 mean/p95 latency,
throughput, end-to-end VRAM, TensorRT parity, or scientific utility. Those
measurements remain closed until all clean train-only information, causal,
TP-retention, restricted-FP, pairwise, seed, and replay gates pass.

The next resource spend is authorized only after a direct cohort tensor
materializer proves exact row identity and crop/model-box/bbox-mask/valid-mask
parity without indexing the broad train-label tree. Any parity, provenance,
process, disk, or memory failure stops before descriptor retention and before
head fitting. This preserves the prospective 20-minute/2-GiB/0.40-GiB/0.10-GiB
wall/fit-VRAM/temp/retained ceilings.

Focused CCR/report verification passes `21/21`; the complete repository suite
passes `1946/1946` in `72.24 s` with 298 warnings. Revision-18 Word QA passes
all `16/16` pages and accessibility `0/0/0`. These checks used no cohort
pixels, candidate GPU fit, validation/test access, checkpoint, or command
revision.

## CCR Direct-Materializer Preflight Result

The direct same-tensor materializer is implemented without constructing the
broad train dataset index. The real structural preflight resolves the locked
763 rows to 735 unique image/label pairs, verifies the exact keeper transform
and source hashes, and records nine logical locked-input opens with zero
blocked attempts. It opens no cohort image or label and creates no descriptor,
model state, candidate score, checkpoint, or run artifact.

The temporary representation is exact post-transform sRGB uint8 plus
little-endian packed 256x256 valid masks. Its conservative upper bound is
`160,456,704` bytes versus the prospective `429,496,730`-byte temporary
ceiling. This avoids retaining two float32 descriptor families and preserves
the ability to prove all geometry rows before descriptor extraction. The
formal process is CPU-only; any initialized CUDA context is a failure.

Eleven tests cover production-parity crops, multi-object labels, exact tensor
round-trip, packed-mask replay, allowlist blocking, no-pixel real preflight,
synthetic formal manifests, Windows logical-open accounting, atomic publish,
forced-failure cleanup, exact resource/CPU-only/no-fit authorization, and
no-write-before-authorization. The repository gate uses NUL-delimited Git
porcelain so protected paths containing spaces cannot be rejected because of
display quoting. Resource limits are checked before and during the image loop;
failed replay finalization removes partial files. Combined CCR focused
verification passes `29/29`. Materializer/test SHAs are
`cc30d973...f24af22` and `f8b2962f...edf5945`.
Focused CCR/report verification passes `32/32`; the full repository suite
passes `1957/1957` in `74.10 s` with 373 existing warnings. Revision-21 Word
QA passes `18/18` pages and accessibility `0/0/0`.

Value remains `mechanism-only-preflight`. The result avoids a broad 9,215-row
dataset index and bounds the only necessary image pass, but it provides no
class-1 information gain, latency, throughput, VRAM, TensorRT, or model metric.
Commit/push and a separate hash-locked execution authorization are required
before one formal materialization plus one fresh-process replay. Descriptor
creation and head fitting remain forbidden at this stage.

The first authorized formal attempt used about 25.4 seconds and failed only
exact crop-box parity; every other reported parity and manifest check passed.
No output or temporary directory remains. The mismatch is a one-ULP
production float-order issue, not a resource failure: production converts
iterated object boxes through float32 after deriving integer crop bounds from
the original primary bbox. The corrected path and regression test pass without
another cohort read. Failure evidence SHA is `11714304...a554e9`; a separate
recovery authorization is mandatory before a replacement image pass.
