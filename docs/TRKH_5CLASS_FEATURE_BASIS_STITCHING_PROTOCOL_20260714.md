# TRKH 5-Class Feature-Basis Stitching Readiness Protocol (2026-07-14)

## Purpose

Directly copying scratch blocks 7-8 onto keeper block-6 features collapsed
validation macro/class-1 F1 to `0.142509/0.134868`. Output-only sparse,
KL, and focal logit-matching supervision then failed to reproduce the useful
scratch complement. This protocol tests the remaining causal hypothesis:
keeper and scratch contain compatible information in different feature bases,
and a small direct-matching adapter can align those bases before the frozen
scratch tail.

This is a train/validation-only readiness experiment. It does not edit raw
data, use test rows, tune the existing precision rule, or authorize a full
train.

## Locked Evidence And References

- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`
  with SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Scratch complement:
  `runs/full_v8_yolof_randominit_30e_20260714_105524/checkpoints/best.pt`
  with SHA-256
  `f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549`.
- Data: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, train and validation
  only. Expected support is `9215/2606`; train/validation source-stem overlap
  must be zero.
- CKA is descriptive evidence, not the selection objective:
  https://proceedings.mlr.press/v97/kornblith19a.html
- Model stitching evaluates functional interchangeability with a small map:
  https://arxiv.org/abs/2106.07682
- Task-loss stitching can create out-of-distribution receiver activations;
  direct feature matching is therefore mandatory here:
  https://arxiv.org/abs/2412.11299
- Activation mean/variance must be checked because aligned or merged paths can
  still suffer activation-statistic collapse:
  https://arxiv.org/abs/2211.08403

## Fixed Candidate Family

The only candidate forks are after blocks `5`, `6`, and `7`. Block 5 is the
earliest admissible fork because token pruning is locked at blocks `2,5`; the
receiver tail must not silently use a different pruning policy.

For each fork:

1. Freeze both complete models.
2. Pair CLS/register/branch tokens by role and patch tokens only by exact
   original `patch_indices` intersection.
3. Fit one shared full-rank affine token map `xW+b` by direct squared feature
   error. Fit a separate affine map for the keeper CNN pooled vector because
   the copied scratch CNN readout also expects the scratch feature basis.
4. Include an intercept. Use centered sufficient statistics and a fixed
   numerical ridge ratio `1e-4 * trace(X'X)/dimension`; do not sweep it.
5. Apply the token map to every keeper token, retain keeper patch indices, run
   the frozen scratch blocks after the fork plus scratch norm/readout, and feed
   the mapped CNN vector to the frozen scratch CNN fusion readout.
6. Keep bbox metadata unchanged. No labels enter either affine fit.

The audit must also report four functional controls: native keeper, native
scratch, identity splice without adapters, and oracle receiver-tail replay
from native scratch fork tokens. These separate token-basis failure from tail
or readout implementation error.

## Leakage Control And Selection

A runtime preflight may use a deterministic balanced cap of 128 train rows and
two source-grouped folds. It validates hooks, token-index alignment, finite
solves, tail replay, and output provenance only; it cannot choose a fork or
change a threshold.

The full readiness run uses five `StratifiedGroupKFold` folds with seed
`20260714`, grouped by case-folded source image stem. Each affine map is fit on
four train folds and evaluated on the unseen source fold. Fork selection uses
only concatenated train OOF evidence. Validation remains unopened during fork
selection.

A fork is OOF-eligible only if all conditions hold:

- source overlap is zero and OOF assignment is complete;
- mean exact patch-index intersection is at least `0.85` of each model's
  retained patches;
- mapped token R2 is at least `0.50`, prefix-token R2 at least `0.35`, and
  patch-token R2 at least `0.50`;
- mapped CNN pooled R2 is at least `0.50`;
- mapped/receiver target activation standard-deviation ratios for tokens and
  CNN vectors are each in `[0.80, 1.20]`;
- stitched versus native-scratch probability MAE is at most `0.04` and argmax
  agreement is at least `0.75`.

Among eligible forks, select the lowest OOF stitched/native-scratch
probability MAE. An exact tie within `1e-4` selects the later fork to minimize
duplicated depth. If no fork is eligible, stop before validation and close the
linear adapter family.

## Single Validation Gate

Fit the selected fork maps once on all train rows, then evaluate all `2606`
validation rows exactly once. Report direct matching, CKA, activation
statistics, native/oracle/identity/mapped outputs, strict transition counts,
and class metrics. Apply only the already frozen probability rule:

1. `q = 0.60 * keeper + 0.40 * stitched_candidate`;
2. subtract `0.034` from class-1 probability, clamp to `1e-8`, renormalize;
3. take the argmax.

An integrated single-checkpoint adapter is allowed only if validation meets
all conditions:

- stitched/native-scratch probability MAE `<=0.03` and argmax agreement
  `>=0.85`;
- macro F1 `>=0.887`;
- class-1 precision `>=0.65`, recall `>=0.70`, and F1 `>=0.69`;
- class-1 false positives `<=60`;
- corrections are not fewer than harms versus the keeper;
- class-1 TP breaks are at most `10`;
- no direct-match or activation-statistic OOF gate regresses on validation.

Failure closes full-rank linear feature-basis stitching without fork, ridge,
token weighting, role-specific map, nonlinear adapter, rule-weight, margin,
seed, or split sweeps. Passing permits one checkpoint-parity implementation
and one at-most-120-batch/two-epoch frozen-primary probe; it does not promote
current-best commands by itself.

## Artifact And Cleanup Contract

The audit writes hashes, fold assignments, sufficient-statistic summaries,
adapter tensors, OOF/validation prediction CSVs, per-fork metrics, strict
transition reports, and a machine-readable gate decision. Test usage and raw
data modification must both be declared false. Temporary caches and rejected
late-member checkpoints are compacted only through the repository's guarded
manifest workflow after the audit is independently replayed.

## Locked Execution Result

- The full five-fold audit completed on all `9215` train objects and `2606`
  validation objects with `0` train/validation source-group overlap. No test
  rows or raw-data writes were used.
- All three forks passed the train-OOF compatibility gates. Fork 5 was selected
  strictly by the lowest stitched/native-scratch probability MAE:
  `0.009962`, versus `0.010409/0.011149` for forks 6/7. Fork-5 token/prefix/
  patch/CNN R2 was `0.860077/0.876004/0.858384/0.994715`, and argmax
  agreement was `0.957244`.
- The single validation opening rejected the adapter. The fused result improved
  keeper macro/class-1 F1 from `0.882925/0.678261` to
  `0.890221/0.700337` and class-1 precision from `0.603093` to `0.712329`,
  while reducing class-1 false positives from `77` to `42`. It nevertheless
  failed the locked recall and TP-preservation gates: class-1 recall was
  `0.688742` and TP breaks were `15` (limits `0.70` and `10`).
- Direct compatibility itself remained valid on validation: stitched/native-
  scratch probability MAE was `0.010463`, argmax agreement `0.944359`, and
  oracle receiver-tail replay error was zero. The rejection is therefore a
  precision/recall behavior failure, not a numerical stitching failure.
- Decision: do not build an integrated adapter checkpoint, run a probe, open
  test, or change current-best commands. Do not sweep fork, ridge, map family,
  role weighting, fusion weight, class-1 margin, seed, or split around this
  result. Full evidence is retained at
  `runs/audit_feature_basis_stitching_full_5fold_20260714` with summary SHA-256
  `726627e05fde790c149242153302d80d63184bdc9e7d9752a407247931a42335`.
