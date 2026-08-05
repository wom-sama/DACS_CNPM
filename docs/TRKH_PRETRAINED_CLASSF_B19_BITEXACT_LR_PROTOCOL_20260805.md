# TRKH B19 bit-exact LR SurfaceFold-XS protocol (2026-08-05)

Protocol ID: `TRKH_PRETRAINED_CLASSF_B19_BITEXACT_LR_SURFACEFOLD_XS_20260805`.

## Status and inherited contract

B19 is a numerical-verification successor to the closed B18 protocol whose
SHA256 is
`a1bd634a8f9e4b8a259d4e8e68881bd85db5cdcf4b86577fcea595a071a7d475`.
It is not an architecture, data, optimization or threshold retry. The exact B18
source is preserved at commit `9bb1232` and tag
`trkh-pretrained-b18-closed-9bb1232`.

B18's label-free preflight passed with artifact SHA256
`b32697dd4911f38ae2fb3ab42bc7aeb635ec92e9ad21a288aa02c763ad73dfda`.
Its single TRAIN authorization then completed all eight fold-0 training epochs
but stopped before serializing any fold state or computing any OOF metric.
`LockedOptimizerStepper.evidence()` compared the applied warmup dictionary with
the literal peak dictionary using exact equality. At the real fold-0 horizon of
`202` updates per epoch, the backbone expression `3e-5 * 201 / 201` evaluated to
`2.9999999999999997e-5`, one ULP (`3.3881317890172e-21`) below the literal
`3e-5`. The mathematical schedule was correct; the executable evidence contract
was not bit-exact. Failure artifact SHA256 is
`75370288a392497b045a977924b0fdf8c3dc168a3849863ca69024c75afbc8ed`.
It records `validation_used=false` and `test_used=false`.

B19 inherits every non-conflicting B18 term unchanged, including the canonical
TRAIN content and five component folds, model and pretrained assets, SurfaceFold
mechanism and equal-state controls, relation objective, augmentations, natural
exposure, AdamW groups and peak rates, eight-epoch linear-warmup/cosine schedule,
gradient clipping, 15-state metric barrier, OOF/bootstrap promotion gates,
deployment qualification, resource limits, one-use authorization and sealed
validation/test boundaries.

## Sole correction: literal scheduler endpoints

For each optimizer group and each locked fold horizon, the B19 LR function is
identical to B18 at every interior update. It returns these endpoints explicitly:

- update `0`: literal `0.0`;
- update `steps_per_epoch - 1`: that group's stored literal `peak_lr`;
- update `total_steps - 1`: literal `MIN_LR = 1e-6`.

The inherited linear formula remains unchanged for other warmup updates and the
inherited cosine formula remains unchanged for other decay updates. No tolerance
is introduced into the evidence gate: applied endpoint dictionaries must remain
bit-equal to their literals. This changes only a one-ULP implementation artifact
at affected endpoints, not the intended schedule, optimizer state or training
horizon.

The focused suite must exercise the exact assignment-derived update counts in
fold order: `202, 206, 212, 207, 210`. For both inherited peak rates `3e-5` and
`3e-4`, it must run every scheduled optimizer step and prove:

- first, warmup-last and final dictionaries exist exactly once;
- their values are bit-equal to `0.0`, `peak_lr` and `1e-6` respectively;
- completed steps equal `8 * steps_per_epoch`;
- the first decay value is below the peak and every subsequent decay value is
  non-increasing, finite and at least `MIN_LR`;
- the previous B18 fold-0 expression is reproduced as unequal to `3e-5`, so the
  regression test cannot pass without exercising the actual failure mode.

Synthetic tests with only two warmup steps are insufficient. Any change to an
interior LR value, peak rate, epoch count, accumulation, fold assignment, loss,
arm or promotion gate closes B19 before TRAIN.

## Execution and stop rule

The B19 offline preflight repeats the complete accepted B18 RNG, source, asset,
stock-first ONNX/dual-target ORT, mechanism, fold, CUDA and isolation gates, with
the B19 scheduler regression included in its no-skip focused suite. A pass grants
one B19 TRAIN-only five-fold/three-arm execution. It grants no validation, test or
mobile-ready claim.

Any B19 preflight or TRAIN failure closes exact B19; it is never relaunched. If
the causal OOF gate passes, fold the candidate to stock SwiftFormer-XS topology
and proceed only to a separately committed physical-Android protocol. If the
mechanism is active but the quality gate fails, do not tune B19; a successor may
open only after a TRAIN-only precondition proves a new surface observable. If the
mechanism itself collapses, the only admissible architectural successor is a
separately preregistered one-factor foldable stem adapter with a matched fixed-
basis control.
