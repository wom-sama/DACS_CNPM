# TRKH 5-Class Schedule-Free AdamW A0 Protocol 2026-07-20

## Decision status

- Status: prospectively locked before runtime implementation or image-model
  inference.
- Scope: one train-only, source-disjoint, keeper-initialized A0 comparison.
- Forbidden during A0: official validation, official test, raw-data edits,
  current-best command edits, a scratch image epoch, hyperparameter sweeps,
  and production trainer integration.
- A complete A0 pass permits one separately locked scratch matched smoke. It
  does not permit a full train by itself.

## Why this route is still open

The remaining class-1 failure is not insufficient recall. The current scratch
full run predicts class 1 too broadly: its raw validation class-1 precision is
`0.537815` at recall `0.847682`, with `62/23/20` incoming class-1 errors from
classes `0/2/4`. Recent loss-only routes reduced false positives by contracting
true class-1 support at nearly the same rate. A useful optimizer route must
therefore show selective trajectory behavior, not generic confidence decay.

Schedule-Free learning is equation-distinct from the closed post-hoc SWA,
checkpoint soups, EMA, SAM, confidence penalties, and Spectral Decoupling
routes. It evaluates an online average `x_t`, computes gradients at an
interpolated point `y_t`, and maintains a fast iterate `z_t`. The accepted paper
reports strong anytime optimization without a horizon-dependent decay
schedule. The official implementation also states important caveats: AdamW
usually still needs LR selection and warmup, and models with BatchNorm require
statistics recalculation at the evaluation iterate. Those caveats are part of
this protocol rather than being hidden after results are observed.

This is a practical optimization candidate, not a claim that averaging alone
contains a mango-specific feature. Its A0 permission depends on directly
showing that the `x` trajectory distinguishes true class-1 support from
restricted `0/2/4 -> 1` false positives under illumination shifts.

## Primary sources and immutable provenance

1. Accepted NeurIPS 2024 oral paper, *The Road Less Scheduled*:
   <https://openreview.net/pdf?id=0XeNkkENuI>
2. Author arXiv copy used for local hashing:
   <https://arxiv.org/abs/2405.15682>
3. Official Meta Apache-2.0 implementation:
   <https://github.com/facebookresearch/schedule_free>

Locked local provenance:

- paper path:
  `%TEMP%/trkh_schedule_free_arxiv_2405.15682.pdf`
- paper SHA-256:
  `1aef42351ac417eb08ba81919cb49bd9d14d98320d0bd3d47eac72f48dcf20d3`
- official repository path:
  `%TEMP%/trkh_schedule_free_official_20260720`
- commit:
  `d24878d3489bf8ede6148eb6390c9b272b9d93c4`
- tree:
  `3aff6c4c3c69a3c965d8ea406008be6ff2d8782f`
- corrected AdamW implementation blob:
  `e8f5a5d52f497577fa229b3c4341feca251bb6e9`
- paper-compatible AdamW implementation blob:
  `aeb80e58f473181204d109302ceb90680ce03f6a`
- license blob:
  `f49a4e16e68b128803cc2dcea614603632b04eac`
- corrected implementation SHA-256:
  `52afca387c4a8004b669cfa8e58b58b98492d128e54e17d15137b70716202f52`
- license SHA-256:
  `5ad8c213095c573921d7388edf93f6a9f54490b8ebf3b165ae8ef1947cf70846`

The A0 uses the current corrected `AdamWScheduleFree`, not
`AdamWScheduleFreePaper`. The official repository documents that the latter
retains the published simplified bias correction and can be unstable. The
three-sequence method is unchanged; the corrected implementation is the
maintained production choice. No implementation from the user-supplied
research reports defines the equation.

## Rejected alternatives in this screen

- MixMo and MIMO are not selected. Their official image recipes rely on
  member-specific encoders/heads and long training, including 300 epochs on
  CIFAR and 1200 epochs plus batch repetitions on Tiny ImageNet for MixMo.
  Compressing that recipe to the TRKH 30-epoch ceiling would be an unvalidated
  new method rather than a faithful implementation.
- WTConv is not selected. Its official recipe uses 300 epochs, and its wavelet,
  frequency, large-receptive-field hypothesis overlaps closed Kymatio,
  high-frequency, FcaNet, EfficientTrain, OctConv, and large/local mixer
  evidence on this keeper.
- BatchEnsemble and Packed-Ensembles are deferred. They add member capacity but
  do not provide a prospective class-1 TP-versus-FP signal and overlap the
  closed late-member and precision-ensemble records.

## Locked data and model boundary

- Data YAML:
  `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`
- Allowed split: `train` only.
- Expected rows and class counts:
  `9215`, `(1941, 541, 1920, 2520, 2293)`.
- Source-grouped fold: existing immutable fold `0`.
- Fit/holdout rows: `7372/1843`.
- Expected holdout class counts: `(380, 109, 393, 503, 458)`.
- Fit and holdout source overlap must be exactly zero.
- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Evaluation semantics, primary-object crop policy, class order, and input size
  come from that checkpoint and cannot be overridden.
- The existing CIDT train-only cohort is allowed only for immutable row/fold
  provenance. Its saved predictions cannot replace live control/candidate
  inference.

The auditor must fail closed if a validation or test path is opened, if a
source appears in both partitions, or if any protected count/hash differs.

## Locked optimization comparison

Both roles start from bit-identical keeper model state and consume the same
`60 x 32 = 1920` augmented fit rows in the same order. Role-local RNG is reset
before every forward so dropout and stochastic training paths are identical.
Training is FP32 to avoid GradScaler cache ambiguity at the `x/y` transition.

Shared settings:

- objective: ordinary five-class cross entropy;
- label smoothing: `0`;
- class weights and oversampling: disabled;
- LR: `1e-5` for every parameter group;
- betas: `(0.9, 0.999)`;
- epsilon: `1e-8`;
- weight decay: `0.05`;
- gradient clipping: global norm `1.0`;
- warmup: exactly eight optimizer steps;
- warmup factor at step `k`: `min(1, (k + 1) / 8)`;
- no EMA, SAM, external scheduler, checkpoint averaging, or test-time
  augmentation.

Control:

- PyTorch `AdamW`;
- LR is set to `1e-5 * warmup_factor` before each step and remains constant
  after warmup.

Candidate:

- official corrected `AdamWScheduleFree`;
- `r=0`, `weight_lr_power=2`, `inner_momentum=0`, and official supported
  foreach implementation;
- built-in `warmup_steps=8` and no external scheduler;
- `optimizer.train()` is called before the first gradient step;
- `optimizer.eval()` is called before checkpointing, calibration, inference,
  export, or hashing evaluation weights.

The A0 is an adaptation signal test, not the eventual full-train LR claim. If
it passes, a scratch matched smoke must compare the existing 30-epoch cosine
AdamW baseline to Schedule-Free with one separately locked LR derived from the
official `1x-10x` guidance. No LR sweep is authorized here.

## BatchNorm evaluation contract

The current CNN stem contains BatchNorm, so evaluating candidate `x` with
running statistics accumulated at `y` is invalid. Both roles receive the same
calibration treatment after optimization:

1. place the candidate optimizer in eval mode;
2. reset every BatchNorm running mean, variance, and batch counter;
3. temporarily use cumulative BatchNorm statistics;
4. run exactly the first 50 fixed, clean, non-augmented fit batches under
   `model.train()` and `torch.no_grad()` with role-reset RNG;
5. restore each BatchNorm momentum and set `model.eval()`;
6. hash calibration row order and all resulting running statistics.

No parameter update, label access, validation row, or holdout row is allowed
during calibration. Control and candidate calibration orders must be exact.
The auditor also records pre-calibration predictions as diagnostics, but only
the recalibrated predictions can decide promotion.

## Evaluation conditions and mechanism evidence

Live inference covers every one of the 1843 holdout rows under:

- clean: brightness `1.00`, contrast `1.00`;
- dim: brightness `0.72`, contrast `1.00`;
- bright: brightness `1.28`, contrast `1.00`;
- low contrast: brightness `1.00`, contrast `0.65`.

For each condition retain target, source group, control prediction and
probabilities, candidate `x` prediction and probabilities, and candidate fast
iterate `z` diagnostic probabilities. The `z` model is reconstructed directly
from official optimizer state; it is never used to select a threshold or tune
the candidate.

Required mechanism checks:

- official optimized implementation matches the official reference/scalar
  equation on deterministic toy trajectories;
- foreach and non-foreach paths agree within declared FP32 tolerance;
- `train/eval/train` transitions are reversible within FP32 tolerance;
- candidate `x`, training point `y`, and fast iterate `z` are finite and
  measurably distinct after 60 steps;
- every trainable parameter group moves and every optimizer state tensor is
  finite;
- both roles consume byte-identical batch indices and targets;
- BatchNorm recalibration is deterministic and uses exactly 1600 fit rows;
- candidate `x` is not behaviorally identical to control or `z`;
- checkpoint save/load in optimizer eval mode reproduces candidate `x` logits;
- ONNX export of candidate `x` has maximum FP32 logit error `<=1e-5` and
  matching argmax on the locked export batch.

## Automatic promotion gates

All gates are conjunctive. A single failure closes this A0 and forbids the
scratch smoke.

### Clean candidate minus control

- macro F1 delta `>= +0.003`;
- class-1 F1 delta `>= +0.010`;
- class-1 precision delta `>= +0.015`;
- class-1 recall delta `>= -0.010`;
- candidate keeps at least `97%` of control true class-1 predictions;
- net restricted `0/2/4 -> 1` false-positive reduction `>= 4`;
- corrections exceed harms;
- class-1 FN rescues are at least class-1 TP breaks;
- no nonfocus class F1 decreases by more than `0.010`.

### Illumination safety

For dim, bright, and low contrast independently:

- class-1 precision delta `>= 0`;
- class-1 F1 delta `>= -0.005`;
- class-1 TP retention `>= 0.95`;
- restricted class-1 false positives do not increase;
- macro F1 delta `>= -0.005`.

At least two of the three shifted conditions must improve class-1 precision by
`>= +0.010`, so a clean-only fluctuation cannot pass.

### Resource and integrity

- candidate/control timed training ratio `<=1.20`, excluding shared loading,
  calibration, evaluation, and export;
- candidate peak allocated VRAM `<=6.50 GiB` and ratio `<=1.15`;
- all provenance, split, RNG, batch-order, equation, optimizer-state,
  calibration, checkpoint replay, ONNX, finite-value, and no-validation/test
  checks pass;
- independent replay recomputes every confusion, metric, transition, and gate
  from retained CSV/JSON artifacts without importing decision values from the
  summary.

## XAI and manual review gate

If the automatic gates pass, the same formal run must produce four deterministic
changed-case contact sheets for clean, dim, bright, and low contrast. Each page
must reserve available FP removals, FP creations, TP breaks, and FN rescues
before filling with highest class-1 probability changes. Show the input,
control native attention/Grad-CAM, candidate native attention/Grad-CAM, bbox,
target/predictions, and class-1 probability delta.

Manual review passes only if candidate FP removals repeatedly suppress
misleading peel/border/background evidence while retained TP or rescued FN keep
coherent fruit-surface evidence. Broad heat reduction on both FP and TP,
lighting-dependent sign reversal, stronger border/background attention, or
missing transition categories is a failure. The review decision and note are
hash-bound; automatic metrics alone cannot promote the route.

## Stop and promotion policy

1. Implement only the isolated auditor, independent replay, focused tests, and
   a VS Code-safe three-phase PowerShell launcher.
2. Run compile, static checks, focused tests, full pytest, source/hash checks,
   and a preflight that creates no output and opens no dataset split.
3. Commit and push the immutable implementation before formal execution.
4. Run exactly one formal A0 only when owned-process and GPU checks permit
   defensible timing.
5. Read all metrics, transitions, optimizer/BN diagnostics, robustness rows,
   replay output, and every contact-sheet page before deciding.
6. On any failure, close Schedule-Free LR/beta/warmup/r/weighting/weight-decay/
   fold/seed/budget/BN-calibration variants on this keeper. Do not rescue it by
   tuning.
7. On a complete pass, write and hash a separate scratch matched-smoke
   protocol. Only that smoke plus full validation robustness/XAI gates may
   authorize an autonomous full train of at most 30 epochs with measured
   loader workers and evidence-based early stopping.
8. The current-best command/history changes only after an actual locked
   validation winner. Test remains untouched until a candidate is frozen.
