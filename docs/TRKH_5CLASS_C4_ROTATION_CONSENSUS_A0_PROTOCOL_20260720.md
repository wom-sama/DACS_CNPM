# TRKH 5-Class C4 Rotation-Consensus A0 Protocol - 2026-07-20

## Status

Prospectively locked before auditor implementation, new inference, or metric
inspection. This protocol authorizes one clean, train-only information audit
of a fixed four-element rotation orbit. It does not authorize validation or
test access, threshold fitting, a rotation/TTA sweep, model training, a full
train, or current-best command promotion.

## Research Question

Does the frozen keeper contain orientation-complementary evidence that can
selectively suppress restricted `{0,2,4} -> 1` false positives while retaining
true class-1 predictions? A positive result would support one later
weight-shared C4 stem experiment. A negative result closes this exact route
before any architecture or trainer change.

The target is precision, not generic invariance. A four-view average that
improves class-1 precision only by contracting class-1 support and breaking
true positives is a failure.

## Primary-Source Lock

- Accepted paper: Cohen and Welling, "Group Equivariant Convolutional
  Networks," ICML 2016:
  `https://proceedings.mlr.press/v48/cohenc16.html`.
- Local paper:
  `D:/DataAI/external_sources/papers/group_equivariant_cnn_icml2016.pdf`.
- Paper SHA-256:
  `e9b4c64d46f1c99569f6835bc11fc01e4635551eb236f0b9c0b7a466ed0a0b93`.
- Official authors' implementation reference:
  `https://github.com/QUVA-Lab/e2cnn`.
- Locked commit/tree:
  `022d6ca4a78ab666f3c149d1849e324ed1d40238` /
  `7673db4f3950f6aac8bbe27534af5a5d165df6ff`.
- Reviewed `R2Conv` implementation SHA-256:
  `0e72b17230dee09dbde1810395aaf39409c9dd19035ccc729074e02ba1ae9e9a`.
- Repository redistribution-license SHA-256:
  `9d3ddfa2ee02769845dffb422007b836a28eafb9de37e8af3f79a6cd3f28e02c`.

The paper's relevant claim is that group convolutions exploit known discrete
rotation symmetries through additional weight sharing and can reduce sample
complexity without adding parameters. TRKH imports only that design principle.
No e2cnn source, dependency, pretrained weight, or reported benchmark number
is copied into the model. The older e2cnn project is used as an official
equation/implementation cross-check, not as a runtime dependency.

## Local Evidence And No-Repeat Boundary

- Mango maturity labels are invariant to object orientation, while the
  keeper's training recipe uses only `rotate90_probability=0.03`.
- The earlier PDisco diagnostic measured keeper foreground-part rotation
  equivariance cosine `0.71559`. Its trained regularizer improved that value to
  `0.83517` but collapsed foreground/class-1 behavior. This establishes
  orientation instability, not that rotation averaging is beneficial.
- Horizontal-flip TTA widened class-1 support and reduced precision. This
  protocol is not permission to reopen flip TTA, D4, arbitrary angles, test-
  time tuning, PDisco, or nearby equivariance losses.
- Context fusion, paired `class_f+yolo_f`, source-context branches, DETR
  localization, high-resolution tiles, differential attention, and many
  texture/color/morphology branches are already closed. C4 is accepted only as
  an equation-distinct weight-sharing hypothesis.
- Raw images, labels, split files, and dataset YAMLs are immutable.

## Locked Inputs

- Pre-protocol HEAD/upstream:
  `e559776`.
- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher arguments SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- Keeper resolved-config SHA-256:
  `e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674`.
- Raw `yolo_f/data.yaml` SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Existing full-train CIDT prediction CSV SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Existing CIDT summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.
- Current-best command/history SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

The audit must verify every locked file before loading the model or dataset.
It may scan validation/test filenames only to prove source-group disjointness;
it must not open their pixels, labels, predictions, or metrics.

## Locked Cohort And Transform

- Split: all `9,215` rows of `yolo_f/train`, ordered by the dataset loader.
- Source groups: lower-cased image stems, assigned once with the existing
  deterministic five-fold `StratifiedGroupKFold`, seed `20260714`.
- Evaluation semantics: exactly those embedded in the keeper; temporal frames
  must equal one.
- Orbit: `K={0,1,2,3}` with `torch.rot90(image, k=K)` over the two spatial
  dimensions. The transformed `image_mask` is rotated by the same `K`.
- `bbox` remains unchanged because it is a source-image coordinate side
  channel, while the tensor is an object crop. `crop_bbox` is not consumed by
  the frozen evaluation forward. Source-context/paired-view inputs are disabled
  by the keeper semantics.
- Baseline: softmax probability at `K=0`.
- Fixed candidate: arithmetic mean of the four softmax probability vectors.
  Logit averaging, majority vote, confidence weighting, angle selection,
  temperature scaling, thresholds, and routers are excluded.
- Batch size `64`, requested workers `4`, CUDA, deterministic seed `20260714`.

The baseline probabilities, targets, paths, and sample order must reproduce
the existing CIDT clean keeper rows. Maximum absolute probability difference
must be at most `2e-5`; predictions and targets must match exactly. This is a
pipeline-integrity check, not a tunable tolerance.

## Required Measurements

Persist one row per train sample with target, source group, fold, baseline and
C4 predictions/probabilities, all four angle probability vectors, class-1
probability delta/range/standard deviation, top-1 agreement count, mean
Jensen-Shannon divergence, and directional transition flags.

Report:

1. baseline/C4 accuracy, macro F1, per-class precision/recall/F1, support,
   predicted support, confusion matrix, NLL, Brier score, and ECE;
2. class-1 FN rescue, TP break, restricted FP removal/creation, total
   corrections/harms, and transition matrix;
3. the same metrics and directional counts for each source-disjoint fold;
4. class-1 probability deltas and rotation-disagreement statistics separately
   for keeper class-1 TP, keeper class-1 FN, restricted class-1 FP, and all
   remaining rows;
5. AUROC for distinguishing restricted FP from class-1 TP using the fixed
   score `-delta_p1`, plus AUROC using rotation Jensen-Shannon divergence;
6. runtime, throughput, effective loader settings, peak CUDA allocation,
   source overlap, hashes, and a complete artifact manifest.

## Conjunctive A0 Gates

All structural gates and all mechanism gates are required.

Structural gates:

- exact hashes, class order, `9,215` ordered rows, finite normalized
  probabilities, four unique angles, complete five-fold assignment, and zero
  source overlap across train/validation/test;
- exact CIDT target/prediction alignment and probability maximum absolute
  difference at most `2e-5`;
- validation/test pixels, labels, predictions, and metrics remain unopened;
- focused tests, compile, replay, `git diff --check`, and artifact-manifest
  verification pass.

Mechanism gates:

- macro F1 delta at least `0.000`;
- class-1 F1 delta at least `+0.005`;
- class-1 precision delta at least `+0.010`;
- class-1 recall delta at least `-0.005`;
- at least `10` net fewer restricted `{0,2,4} -> 1` false positives;
- total corrections are at least total harms;
- class-1 TP breaks do not exceed class-1 FN rescues;
- maximum F1 loss among classes `0,2,3,4` is at most `0.010`;
- restricted-FP mean `delta_p1` is at least `0.010` more negative than the
  class-1-TP mean, and `AUROC(-delta_p1) >= 0.62`;
- at least four folds have non-worse class-1 precision, at least four have
  positive restricted-FP net removal, and no fold loses more than two class-1
  true positives net;
- peak CUDA allocation is at most `7.5 GiB` and four-view throughput is
  recorded without an out-of-memory retry.

No individual metric can compensate for a failed gate. In particular, lower
restricted FP is not useful if it is achieved by broad class-1 suppression.

## Escalation And Stop Rules

If any A0 gate fails, close C4 probability consensus and the proposed
four-pass stem orbit average on the current keeper. Do not sweep C4/D4,
rotation direction, angle subsets, aggregation space, weights, temperatures,
thresholds, bbox handling, seeds, or nearby regularizers. Do not run
validation, XAI, a smoke, a probe, or a full train.

Only a complete A0 pass authorizes one implementation of a parameter-neutral
C4 stem orbit:

`Phi_C4(x) = (1/4) * sum_k R_-k Phi(R_k x)`.

That implementation must use standard PyTorch operations, share the existing
stem parameters, align feature maps back to the original frame, preserve the
default model bit-exactly, strict-load the keeper, and pass export/resource
preflight. It then receives one matched two-epoch/120-batch validation smoke.
Only a smoke that passes clean precision, recall, restricted-FP, robustness,
changed-case XAI, calibration, throughput, and strict-reload gates can
authorize a five-epoch probe. Only a prospectively passing probe may authorize
one autonomous full train, at most 30 epochs, early-stopping patience 3, and
freshly measured loader workers. Final test remains isolated behind the
existing independent-reload promotion gate.

Current-best checkpoint, command, and update history remain unchanged unless
the complete promotion process produces a genuine winner.
