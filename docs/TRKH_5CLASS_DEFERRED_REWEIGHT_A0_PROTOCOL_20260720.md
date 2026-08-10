# TRKH 5-Class Deferred Reweight A0 Protocol - 2026-07-20

## Decision question

Does strict class-balanced sampling from random initialization create a
class-1 prior/support bias large and stable enough to justify one natural-prior
scratch smoke, followed only on success by a deferred class-reweighting fork?

This protocol does not authorize test access, raw-data edits, trainer/model
integration, a probe, a full train, or current-best command promotion.

## Primary source and licensed reference

- Paper: Cao et al., *Learning Imbalanced Datasets with
  Label-Distribution-Aware Margin Loss*, NeurIPS 2019.
- Paper SHA256:
  `f873d07d4bc6fedf6de7222a981331c88a961509c5004287346d444e96e1b1f3`.
- Authors' official implementation:
  `https://github.com/kaidic/LDAM-DRW`, commit
  `2536330f2afdaa65618323cb5a5850efccce762a`, tree
  `c076696ca0ae868e0c41038d72159a439652a146`.
- Official `cifar_train.py` SHA256:
  `f2f2e10c7dbc837c6cd640b9295e97eb945064d965d2f6c19c1301f22d1f6f43`.
- MIT license SHA256:
  `24769cbd6cf77906da1b7c51608079de664b44c4025cf5a2ef99d1f7e2b60bc5`.

The paper and official code keep natural-frequency shuffled sampling and defer
class reweighting. The official 200-epoch CIFAR schedule switches at epoch 160;
that duration and switch point are not transferred to TRKH's <=30-epoch
constraint. The operative hypothesis transferred here is only: learn the
initial representation under the natural prior, then apply a bounded late
class-weight phase.

## Locked local evidence

- Dataset YAML SHA256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Immutable train rows/classes: `9215`, counts
  `[1941, 541, 1920, 2520, 2293]`.
- Scratch full checkpoint SHA256:
  `f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549`.
- Scratch launcher/resolved/history SHA256:
  `a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6`,
  `c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97`,
  `a6eeddd7b1ef29222e78764bad854c9d670b0bc98de38362c01e2373359404e2`.
- Train-only CIDT summary/prediction SHA256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`,
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Current-best command/history SHA256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`,
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

The scratch configuration is the current V8 no-pretrain recipe with
`batch_size=32`, `seed=42`, strict balanced sampling, no class weights,
LDAM-focal, EMA, `epochs=30`, scheduler horizon `30`, and test skipped.

## A0: no-training authorization audit

A0 reads only the locked train metadata, strict sampler equation, scratch
configuration, scratch history metadata, and the already-created CIDT clean
train predictions. It must not load dataset pixels or a checkpoint, run a
model forward, fit a readout, or access validation/test rows or predictions.
The five CIDT partitions are used only to test source-group stability; the
underlying model predictions are in-sample and must not be described as OOF.

Reconstruct strict exposure with `StrictBalancedBatchSampler` at batch 32 and
epoch multiplier 1.0. Reconstruct natural prior from the train labels. For
each of keeper and scratch candidate, calculate clean class-1 predicted-support
ratio, precision, recall, F1, and restricted `0/2/4 -> 1` false positives in
every fixed source partition.

Two fixed negative controls apply Bayes prior factors to the persisted clean
probabilities without fitting:

```text
sqrt control: p_hat_j * sqrt(p_train_j / 0.2)
full control: p_hat_j * (p_train_j / 0.2)
```

They are safety controls only. Neither may become a post-hoc router,
calibrator, threshold, or candidate.

### A0 gates

All gates are conjunctive:

1. Hashes, `9215` unique sample indices, class counts, five source partitions,
   and clean-only row coverage match exactly.
2. Strict exposure totals `9216` rows and differs by at most one row between
   classes.
3. Class-1 strict/natural exposure multiplier is at least `3.0`; no nonfocus
   class multiplier exceeds `1.10`.
4. Scratch class-1 predicted-support/true-support ratio is at least `1.25` in
   all five source partitions.
5. Scratch class-1 recall minus precision is at least `0.20` in all five
   partitions, and aggregate restricted FP is at least `250`.
6. At least one fixed prior control lowers class-1 recall below `0.70`; this
   confirms that post-hoc prior correction is unsafe and cannot substitute for
   natural-first representation learning.
7. No validation/test/model forward/readout/training output exists.

Passing A0 authorizes exactly one Stage-B natural-only scratch smoke. Failure
closes the route before training.

## Stage B: locked natural-only scratch smoke

Use the same V8 scratch recipe, seed, initialization procedure, augmentation,
optimizer, LR schedule, model, data, batch size, workers, and full validation
semantics as the locked full run. Change only:

- strict balanced sampling -> natural shuffled sampling;
- run horizon -> five epochs while preserving scheduler horizon `30`;
- retain class weights disabled;
- retain test skipped and architecture trace enabled.

Historical strict-balanced epoch-5 reference is locked at macro/class-1 F1
`0.798520/0.490909`, class-1 P/R `0.373702/0.715232`.

Stage B passes only if full validation support is `2606` and all conditions
hold:

- macro F1 >= `0.778520`;
- class-1 F1 >= `0.470909`;
- class-1 precision >= `0.403702`;
- class-1 recall >= `0.635232`;
- total class-1 FP is at most `162`, a >=10% reduction from the historical
  epoch-5 total of `181` reconstructed from locked P/R/support;
- no nonfocus class F1 loses more than `0.04` versus epoch 5;
- architecture trace has one valid sample per class;
- XAI/robustness shows no material foreground, lighting, or class-1 TP safety
  regression under the project's existing gates.

If Stage B fails, record/compact it and close natural-first/DRW on this recipe.

## Stage C: deferred fork, only after Stage B passes

Implement a default-off deferred reweight schedule with these invariants:

- natural shuffled sampling for every epoch; never oversample class 1;
- epochs before the locked start use unit class multipliers;
- epochs at/after the start use effective-number weights with existing local
  `beta=0.999`, normalized to mean 1;
- the expected class-1 late weighted mass share is about `0.1166`, below the
  strict sampler's `0.20` and above natural `0.0587`;
- evaluation remains unweighted;
- resolved config, epoch telemetry, history, checkpoint resume, and disabled
  bit-equivalence are mandatory.

Fork the same retained epoch-5 natural checkpoint into a natural-only control
and a deferred candidate through epoch 8. Start reweighting at epoch 6. Use a
second late-phase sampler seed for the deferred repeat; use a start-after-horizon
placebo to prove disabled equivalence. A reweight-from-epoch-1 temporal placebo
is allowed only if the deferred candidate first passes clean gates.

The candidate must beat both the natural fork and historical strict reference
on full validation macro/class-1 F1, improve class-1 precision without reducing
recall below `0.68`, reduce restricted FP, pass every adequate source fold, and
pass full robustness/XAI/replay/resource gates before any longer probe.

No test access is allowed in A0, Stage B, Stage C, or a follow-up probe. A full
train remains subject to the class-1 F1 milestone, <=30 epochs, patience 3,
worker benchmark, independent validation reload, XAI, and current-best command
promotion rules.
