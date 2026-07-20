# PMG Progressive-Jigsaw Signal A0 Closure (2026-07-20)

## Decision

Reject exact stage-aligned `8 -> 4 -> 2` PMG jigsaw on the current no-pretrain
keeper. The sole source-disjoint train-only A0 passes every structural,
provenance, convergence, visual-geometry, and replay check, but fails 11 of 16
mechanism gates. It improves class-1 recall while lowering class-1 precision
and creating more restricted false positives, contrary to the agricultural
precision objective.

Do not integrate PMG into the model/trainer, run validation or test, launch a
smoke/probe/full train, or update current-best commands from this result.

## Locked Evidence

- Normative sources are Du et al., ECCV 2020, and the authors' MIT repository
  pinned at commit/tree `db7a7d7...2b51a`/`85949974...e99e`. Paper SHA is
  `ae701299...da85` and protocol SHA is `7af3630a...a8c1`.
- Source evidence remains adverse: the released recipe uses pretrained
  convolutional layers, input 448, up to 200 epochs, and four optimizer updates
  per logical batch. The source calls `pretrain=True` despite the README's
  scratch wording.
- Infrastructure and calibrated host-gate commits are `86cd087` and `9b8dcf1`.
  Auditor/test/launcher SHAs are `6610629f...c01ed`/
  `7316224e...fd4b82`/`0b487ca0...a5c45f`.
- The formal uses all 9,215 immutable `yolo_f/train` object rows from 8,064
  source images. Train-to-validation/test source overlap and inter-fold source
  overlap are zero. Dataset identity is `f771d280...2703a`.
- CIDT row, target, path, source, fold, and argmax replay are exact; clean
  keeper probability maximum error is `8.94e-8`. The keeper state hash remains
  bit-exact before/after extraction. Validation/test pixels, labels,
  predictions, and metrics remain unopened.

## OOF Result

All four fixed natural-prior `C=0.05` readouts converge on every source fold.

| Role | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | Effective rank |
|---|---:|---:|---:|---:|---:|
| Clean control | 0.915522 | 0.755859 | 0.715342 | 0.735043 | 11.2959 |
| PMG aligned | 0.916351 | 0.753321 | 0.733826 | 0.743446 | 10.8696 |
| Reverse placebo | 0.913395 | 0.743738 | 0.713494 | 0.728302 | 10.9279 |
| Deepest placebo | 0.917481 | 0.760077 | 0.731978 | 0.745763 | 10.7365 |

Candidate-minus-clean changes macro/class-1 F1 by
`+0.000829/+0.008403`, precision/recall by
`-0.002539/+0.018484`, and changes 210 decisions with `96` corrections versus
`103` harms. It rescues/breaks `40/30` class-1 FN/TP but removes/creates only
`30/36` restricted false positives, a net creation of six.

Predicted class-1 support expands `512 -> 527`. Restricted class-1 FP by true
class `0/2/4` change from `91/24/6` to `93/24/10`. The largest paired changes
are `29` class-1 `0 -> 1` rescues, `25` class-1 `1 -> 0` breaks, `25` true
class-0 `0 -> 1` harms, and `21` true class-0 `1 -> 0` corrections. PMG therefore
trades in both directions around the same class-0/class-1 boundary without a
reliable precision veto.

| Fold | Macro delta | Class-1 F1 delta | Precision delta | Recall delta | Corrections/harms | FP net removal |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.009790 | +0.040533 | -0.013592 | +0.082569 | 27 / 21 | -4 |
| 1 | -0.004049 | -0.015495 | -0.022063 | -0.008696 | 19 / 23 | -3 |
| 2 | +0.003971 | +0.016021 | +0.012574 | +0.019231 | 24 / 21 | +1 |
| 3 | -0.008434 | -0.024333 | +0.009637 | -0.058252 | 10 / 19 | +2 |
| 4 | +0.003032 | +0.025789 | -0.007438 | +0.054545 | 16 / 19 | -2 |

Only two folds have non-worse precision. Fold 3 loses `0.058252` recall, and
folds 0/1/4 create restricted FP on net. The aggregate gain is not stable.

## Mechanism Diagnosis

- Candidate TP-versus-restricted-FP direction AUROC is only `0.535661`, below
  the locked `0.60`; reverse/deepest placebos reach `0.493789/0.511509`, so the
  candidate advantage over the strongest placebo is only `0.024152`, below
  `0.03`.
- Candidate class-1 F1 is `0.002317` below the deepest-only placebo. The best
  aggregate role therefore does not require the paper's progressive
  stage/granularity alignment.
- Every 256-dimensional block is nonconstant, but the candidate's nominal
  1,024-dimensional descriptor has effective rank only `10.8696`, below the
  locked minimum 16 and below the clean control. Jigsaw views add a weak,
  correlated recall signal rather than a new high-rank selective surface cue.
- Eleven gates fail: macro/class-1 F1 and precision magnitude, corrections over
  harms, restricted-FP removal, fold precision, worst-fold recall, deepest
  placebo, both directional-AUROC gates, and effective rank.

Manual review of the fixed five-class sheet passes geometry. `P8/P4/P2` preserve
local pixels with sharp patch boundaries and no resize/interpolation or
label-dependent transform. Gray square-padding patches move with their exact
validity masks, as locked. Thus the rejection is information/selectivity
failure, not a transform implementation error.

Close jigsaw grids, permutation seeds, stage choices, stage/granularity order,
deepest/reverse variants, readout `C`, class weights, fusion weights, thresholds,
routers, jigsaw probability, and exact sequential PMG training on this keeper.

## Runtime, Replay, And Retention

- Extraction takes `106.576 s` at `86.464` source images/s and `345.858`
  forward views/s. Requested/effective workers are `4/4`; RTX 4060 peak CUDA
  allocation is `1.2557 GiB`. Block patch counts are exactly `256/218/167` for
  every view.
- Visual review is finalized against pre-review summary SHA
  `50cb3f77...4f1d`. Final summary/manifest SHAs are
  `13c47a69...d7f44`/`f48b5a9f...be89c`; exact external replay maximum
  difference is `0.0`.
- Prediction/cache/contact-sheet/descriptor/readout/fold SHAs are
  `8027ee06...e29`/`3c59b140...24b0`/`edbeb271...fb3e2`/
  `f185d1e4...523a`/`3296749c...528`/`1b8666c0...1a8f`.
- Retain all seven replay payloads (`9,159,700` bytes including summary after
  visual finalization); there is no checkpoint or optimizer payload to delete.
  The read-only retention audit covers 792 run directories and all 50 valid
  object-schema compaction manifests. All 219 compacted originals remain
  absent, `deleted_anything=false`, and `blockers=[]`; retention summary SHA is
  `4c7133c4...c9e33`.
- Focused tests pass `15/15`; the complete suite passes `1617/1617` with 277
  non-failing warnings. Current-best keeper/command/history hashes remain
  `1f49d577...2677`/`36b9aa1a...0faf`/`39bd2879...8f53`.
