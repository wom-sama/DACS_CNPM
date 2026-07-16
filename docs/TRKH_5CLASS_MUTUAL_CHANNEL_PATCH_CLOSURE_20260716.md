# TRKH 5-Class Mutual-Channel Patch Closure

Date: 2026-07-16
Decision: reject the locked final-patch MCL residual; Stage B denied

## Scope and provenance

- The amended pre-metric protocol is
  `docs/TRKH_5CLASS_MUTUAL_CHANNEL_PATCH_READINESS_PROTOCOL_20260716.md`,
  SHA-256
  `5444f20b01fbf1af3b2a36595012a6621912a31d5471e9ae9b94b9287b2680b2`.
  A two-row structural replay corrected the assumed final patch count from
  142 to 167 before training or behavior access. TRKH keep rates are absolute
  to the original 256-token grid, so `ceil(256 * 0.65) = 167`.
- Primary evidence is Chang et al., *The Devil is in the Channels:
  Mutual-Channel Loss for Fine-Grained Image Classification*, TIP 2020, plus
  the MIT-licensed official repository at commit
  `befb3692cd0d5382eb32fa4e093226247f609fd9`. The paper and four official
  source/license files matched their precommitted hashes.
- Only source-disjoint `yolo_f/train` rows were used: 7,372 natural-frequency
  fit rows and 1,843 holdout rows, with zero source overlap. Ten sequential
  NumPy seed-42 permutations produced exactly 73,720 occurrences. Order and
  SHA-derived dropout-mask hashes were
  `d56903f7806c12c19dddc4dc639054a98ba84a806f9568871157359be31512d4`
  and
  `0b15865764091d839872ac840cdc6a37965662a4ef6b9e3e8edc95a7e2c5afb7`.
- Validation/test paths or predictions were never constructed. No raw data,
  shared model/trainer code, checkpoint, or current-best command changed.

## Structural and training audit

- The frozen keeper had exactly 7,245,590 parameters. Candidate and control
  each had exactly 3,935 trainable parameters: a `256 -> 15` one-by-one patch
  projection and a zero-initialized `15 -> 5` residual head. The five class
  groups each contained three contiguous channels. Initial deployed logits
  were bit-exact to raw under residual scale 0.05.
- Candidate/control parameter and optimizer states were bit-exact initially.
  Isolated seed-42 initialization restored the global RNG. Every trainable
  parameter saw a finite nonzero gradient and changed; the keeper remained
  bit-exact.
- Independent grouped-logit, spatial-softmax, coverage, loss, and gradient
  equations matched the local FP32 implementation with maximum error
  `3.576e-7`. Candidate MCL-only projection gradients were nonzero. Detached
  control gradients matched CE-only gradients bit-for-bit.
- Both variants executed 10 epochs, 2,310 batches, and 73,720 occurrences.
  Control CE changed `0.601003 -> 0.171663`. Candidate total loss changed
  `1.380554 -> 0.618719`; its CE/L_dis/L_div ended at
  `0.188975/0.269822/0.001251` with mean coverage `2.996248/3`.

## Mechanism result

| Split | Variant | L_dis | L_div | Coverage | Group accuracy | C1 entropy | Max C1 cosine | Direction AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fit | control | 5.179463 | 0.082150 | 2.753551 | 0.228703 | 0.427409 | 0.999991 | 0.952681 |
| fit | MCL | 0.246548 | 0.001199 | 2.996401 | 0.916305 | 0.272884 | 0.820779 | 0.950840 |
| clean holdout | control | 5.228253 | 0.081825 | 2.754525 | 0.236028 | 0.428172 | 0.999331 | 0.948628 |
| clean holdout | MCL | 0.257927 | 0.001269 | 2.996192 | 0.912100 | 0.270939 | 0.529963 | 0.946437 |

- MCL successfully learned class-aligned, spatially complementary channels.
  Holdout L_dis was only `0.04933x` control, group accuracy rose to 0.912100,
  coverage approached the theoretical maximum, and class-1 channel cosine
  fell to 0.529963.
- Absolute residual-margin AUROC was high at 0.946437, but it was below the
  matched control 0.948628 and failed the prospectively fixed comparator gate.
  Strong rank separation did not produce a safe deployed class-support shift.

## Locked train-holdout behavior

| Condition | Variant | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | Restricted FP |
|---|---|---:|---:|---:|---:|---:|
| clean | raw | 0.948369 | 0.743056 | 0.981651 | 0.845850 | 37 |
| clean | CE control | 0.900524 | 0.804878 | 0.605505 | 0.691099 | 16 |
| clean | MCL | 0.899295 | 0.779070 | 0.614679 | 0.687179 | 19 |
| dim | raw | 0.825963 | 0.535354 | 0.486239 | 0.509615 | 46 |
| dim | CE control | 0.794131 | 0.531646 | 0.385321 | 0.446809 | 37 |
| dim | MCL | 0.767908 | 0.522388 | 0.321101 | 0.397727 | 32 |
| bright | raw | 0.872482 | 0.561644 | 0.752294 | 0.643137 | 63 |
| bright | CE control | 0.828560 | 0.588235 | 0.458716 | 0.515464 | 35 |
| bright | MCL | 0.815217 | 0.494737 | 0.431193 | 0.460784 | 48 |
| low contrast | raw | 0.854165 | 0.547826 | 0.577982 | 0.562500 | 51 |
| low contrast | CE control | 0.815730 | 0.629032 | 0.357798 | 0.456140 | 23 |
| low contrast | MCL | 0.794675 | 0.653061 | 0.293578 | 0.405063 | 17 |

- On clean holdout, MCL changed macro/class-1 F1 by
  `-0.049074/-0.158670` versus raw. Precision rose only `+0.036014` while
  recall fell `-0.366972`: 40 raw class-1 true positives broke and zero false
  negatives were rescued. Restricted FP fell 37 to 19, but 24 corrections
  versus 73 harms makes this unsafe suppression, not agricultural improvement.
- Natural CE control already broke 41 class-1 true positives and reduced
  class-1 F1 by 0.154750. MCL did not beat that control: macro/class-1 F1 fell
  another `-0.001229/-0.003920`, precision fell `-0.025808`, and restricted
  FP increased 16 to 19. Its one-TP recall gain over control was insufficient.
- Dim, bright, and low-contrast class-1 F1 fell by
  `-0.111888/-0.182353/-0.157437` versus raw. MCL produced 23/40/32 TP breaks
  with only 5/5/1 rescues. No illumination safety gate passed.
- A strict raw/CIDT argmax replay check differed on one near-tie row only,
  sample 7240. The prior p0/p1 values were `0.244257/0.243177`; the formal
  BF16 replay gave `0.242732/0.244636`, margin 0.001904. Order, target, path,
  and all within-run replay checks remained exact. This numerical near-tie
  cannot affect the behavioral rejection.

## XAI and deployment audit

- Required XAI selected all 92 clean special events and produced 31 pages.
  Categories included 40 class-1 TP breaks, 22 restricted-FP removals and four
  creations versus raw, plus 16 TP breaks, 17 rescues, five FP removals, and
  eight creations versus control. Selection replay, finite maps, and nonzero
  defined maps all passed.
- Quantitative replay over every selected map found candidate within-group
  mean correlation `-0.011779`, confirming real diversity. Candidate/control
  grouped-map correlation was only 0.103411. Invalid-grid candidate mass was
  exactly zero. Twelve deterministic 4x4 input-occlusion maps were nonzero on
  a mean 98.861% of cells.
- Visual review of pages 0, 5, 10, 20, and 30 showed candidate channels split
  fruit surface, tip/stem, and occasional nearby border/background regions.
  Both removed-FP and broken-TP rows contained close crops and wide context.
  The mechanism localized diverse evidence, but residual margins shifted many
  legitimate subtle class-1 fruits downward. This is a support/calibration
  failure, not absence of channel diversity or a simple background problem.
- Batch-one median runtime was 0.035063 s versus raw 0.037563 s, ratio
  0.933450x; candidate peak VRAM was 0.113485 GiB. Isolated/full ONNX exports
  were finite and argmax-exact with maximum errors `9.537e-7/2.384e-6`.
  Deployment gates passed but cannot override TP and F1 rejection.

## Replay, artifacts, and engineering closure

- Independent CSV replay reproduced all `4 x 1843 = 7372` rows, comparisons,
  confusion-derived metrics, and transitions exactly. All 42 manifest entries
  matched their file sizes and SHA-256 values.
- Evidence is under
  `runs/audit_mutual_channel_patch_readiness_20260716`. Its 42 payloads total
  54,138,676 bytes. Summary/prediction/artifact-manifest/XAI-manifest SHAs are
  respectively
  `cf4a0143afde61ce2a737fca4a859397d19d058ad3ca01c85d624a34e7183b0f`,
  `1feff96cedf182b003163f2c22b14b075481de892ca3acdab878e728c925a1d5`,
  `e4e727a70939cea5292432eeb5f52edce210757195086391b599744a6154a924`,
  and
  `83fb7b24443c9741d742cd903b2aca1e360f8bf64cf2e7bd0d83ff4adcfc9973`.
- No `.pt`, `.pth`, `.ckpt`, TensorRT engine, trainable manifest,
  validation/test output, or raw-data write exists. Current-best command
  tracking remains exactly three revisions and two updates after the first.
- Compile/focused tests passed `8/8`; full pytest passed `1196/1196` with 257
  warnings in 77.47 s. The VS Code-safe launcher parsed and formal preflight
  passed. The initial XAI backward attempt was discarded after CUDA reported
  a deterministic adaptive-pool limitation; commit `2c3c5ed` replaced it with
  deterministic inference-only 4x4 occlusion and the complete run was repeated
  from scratch.
- Read-only retention audit
  `runs/artifact_retention_audit_mutual_channel_patch_closure_20260716`
  passed over 705 run directories with `blockers=[]`, deleted nothing, and
  reported 71.346 GB free. Summary SHA is
  `b473c2f80e98f771a73b3ae1d013ffbe3ee7183d93a934cd793b3ee9d32652de`.
- Keeper, scratch complement, current-best command packet, and command-history
  hashes remain exact.

## Decision and no-repeat boundary

Stage B, validation, test, probe/full train, shared-trainer integration, and
current-best command promotion are denied. Do not sweep channels per class,
final patch source, projection/head shape, residual scale, alpha/beta,
optimizer, learning rate, weight decay, epochs, fold, seed, mask rule, class
sampling, or a post-hoc threshold/router on this keeper.

This closes the exact final-patch MCL residual and neighboring calibration
variants, not MCL universally. A future class-aligned channel method must be a
new sourced representation with a prospectively locked natural-multiclass
TP/support constraint across illumination. It must retain raw, natural-CE, and
this MCL result as comparators; high AUROC or fewer false positives cannot
authorize a route that obtains precision by suppressing true class 1.
