# TRKH 5-Class WILDCAT Negative-Evidence A0 Closure - 2026-07-17

## Decision

Reject dense block-2 WILDCAT negative-evidence pooling on the current keeper
before a matched scratch pair, production integration, validation, test, full
training, or a current-best command update.

The locked WILDCAT head is equation-faithful, reproducible, cheap, and better
than the matched top-only head on clean class-1 F1. It is still materially worse
than the matched GAP head and fails the precision-first objective. Its bottom
term expands class-1 support and creates restricted false positives instead of
selectively rejecting them. The effect worsens under dim, bright, and
low-contrast shifts.

Manual review of all nine XAI pages also fails. Class-1 maps mostly follow
broad yellow/green fruit regions rather than lesion morphology. Bottom maps
alternate between background or bbox edges and valid fruit tissue, so rescue
and harm cases do not expose a stable semantic negative-evidence mechanism.
Stage B and all current-best updates remain unauthorized.

## Authority and immutable recovery

- Accepted paper: Durand et al., *WILDCAT: Weakly Supervised Learning of Deep
  ConvNets for Image Classification, Pointwise Localization and Segmentation*,
  CVPR 2017.
- Authors' MIT repository commit/tree:
  `c7d355049a8fe34e98f49d1ff4ae91c996bd3825` /
  `2b777ae5058f1e71812f7201ddda2dc64cbb127a`.
- Paper/pooling/license SHAs:
  `2209957a...3995` / `d4a5fa61...04d5` / `90a9749c...064a`.
- Prospective protocol SHA:
  `0914b04beb8b27496cf980d08a753e74be330e8b25210278f7eb23187433fb0c`.
- Implementation, erratum, and corrected-recovery commits:
  `32316ff` / `9ad7abb` / `c106aaef953b3c5d15b5e9a53a3714d420bb3785`.
- Erratum SHA:
  `c5f441c8698ab263e8b540617ef2c3aa4d7432902ecf58058a41aed55e51c02a`.
- Final evidence directory:
  `runs/audit_wildcat_negative_evidence_a0_20260717`.
- Final summary/manifest SHAs:
  `a7543a9c4f48d0b07f79cd33ddd1cf66d292dc82346b51be9c8a41636c24800b` /
  `ff0bf89d950e1dbd45168a0f664ea3ea4a37f56eb0c6500f3bbb0c4a4ae641d0`.

The first formal invocation completed feature extraction, all 20 matched head
epochs, four holdout conditions, CSV replay, resource checks, and exports. It
then stopped at input-gradient XAI because CUDA adaptive-average-pool backward
has no strict deterministic implementation. No summary existed and no
behavioral metric was inspected.

Before reading any metric, the five pre-XAI artifacts were preserved under
`runs/audit_wildcat_negative_evidence_a0_interrupted_xai_determinism_20260717`.
Its 38,271,153-byte manifest SHA is
`4cf7cc1805bae15d6f2766d392752fbb7ca01685ad2a85404f82e01f4772f845`.
The pushed erratum authorized exactly one corrected run: strict determinism
remained active outside two independently repeated XAI saliency passes, which
used `warn_only=True` and restored strict state in `finally`.

The corrected run reproduced all five pre-XAI files byte-for-byte before XAI:
training CSV, prediction CSV, mechanism CSV, isolated ONNX, and full ONNX.
`behavioral_metrics_exposed_before_replay=false`. This establishes that the
determinism correction changed only XAI availability, not method behavior.

## Locked train-only experiment

The frozen 7,245,590-parameter keeper produced dense block-2 features with
shape `[B,256,16,16]` before the first token-pruning stage. Matched 5,140-
parameter GAP, top-only, and WILDCAT heads started bit-identically and received
the same fixed SGD batches and order.

- WILDCAT configuration: `M=4`, `k+=k-=0.2`, 51 top and 51 bottom cells,
  `alpha=0.7`, and the official final division by two.
- Source-disjoint train-only split: 7,372 fit and 1,843 holdout rows, with zero
  source overlap.
- Fixed training: 20 head-only epochs and 4,620 updates per head; all gradients
  and final parameters are finite and nonzero/moved.
- Fixed order SHA:
  `bc02b9a8e0a36c96a7d407b654208a87d558beb907a8c7fa48174d1b5b2a7cb5`.
- The temporary 966,262,784-byte float16 feature cache was removed in the
  audited `finally` path.
- No image epoch, validation prediction, test sample, pretrained weight, raw
  data change, or trainable manifest was used.

## Information result

Negative FP reduction means the candidate created more restricted class-1
false positives than the control.

| Condition | Macro F1 | Class-1 P/R/F1 | F1 delta vs GAP | F1 delta vs top-only | Restricted-FP reduction vs GAP/top |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 0.826386 | 0.561644 / 0.376147 / 0.450549 | -0.063417 | +0.174687 | -8 / -16 |
| dim | 0.759207 | 0.326389 / 0.431193 / 0.371542 | -0.030033 | +0.007905 | -3 / -35 |
| bright | 0.798733 | 0.406504 / 0.458716 / 0.431034 | +0.022658 | +0.143363 | -30 / -57 |
| low contrast | 0.760848 | 0.359649 / 0.376147 / 0.367713 | -0.054967 | -0.016125 | -29 / -22 |

On clean data, the candidate has 41 class-1 TP out of 109 and predicts class 1
73 times. Against GAP it changes 62 decisions, with only 20 corrections versus
34 harms, breaks 12 class-1 TP, and loses `0.016319` macro F1. Precision,
recall, and F1 remain far below the locked `0.75/0.70/0.70` thresholds.

The direct same-weight no-bottom ablation proves that the bottom term is active
but unsafe. On clean data it raises class-1 F1 by `0.198894` and makes 112
corrections versus 19 harms, yet creates nine net restricted false positives
and breaks one TP. Net restricted-FP reductions under dim/bright/low-contrast
are `-81/-29/-54`. This is broad class-1 support expansion, not selective
negative evidence.

Clean direction AUROC is `0.928975/0.903778/0.917267` for GAP/top-only/WILDCAT.
WILDCAT therefore improves over top-only but not GAP. Bottom-margin AUROC is
`0.784748`, and class-1 TP top-map bbox alignment is `0.444524` above chance,
but these localization signals do not translate into precision-safe decisions.

## XAI and manual review

The formal XAI contains 68 prospectively selected transition rows over nine
contact sheets. NPZ replay is exact, all values are finite, all required event
indices and page rows are covered, and the two-pass saliency correction passes:

- class-map error `0`, exact top/bottom indices, and zero saliency argmax
  mismatch;
- raw/normalized maximum saliency error
  `7.276e-12 / 1.863e-9`;
- minimum saliency cosine `0.999999762`;
- strict deterministic state restored after every scoped pass.

All pages `001-009` were inspected. Top maps usually select large yellow or
bright regions of the fruit. Bottom maps sometimes cover padding, background,
or bbox edges, but often cover legitimate green fruit tissue. Similar spatial
patterns appear in class-1 rescues, class-0/2 false-positive creations, and
class-4 confusion cases. Input saliency is diffuse and rarely isolates small
lesions. Page 007 also includes an almost featureless geometry-sensitive crop.
The visual mechanism is reproducible but not semantically stable; manual review
is recorded as `fail` in the finalized summary.

## Equation and deployment

- Official-source/oracle pool errors are `3.647e-8` and `2.776e-17`; gradient
  oracle and finite-difference errors are `6.776e-21` and `9.225e-14`.
- Strict BF16 error is `0.001343`; edge cases and selected cells pass.
- Batch-32 candidate runtime and peak-memory ratios are
  `0.947352/1.000005`, both passing.
- Isolated/full ONNX export errors are `1.907e-6/9.537e-7`, with finite output
  and exact argmax.
- TensorRT FP32 parses and builds and retains exact argmax, but maximum absolute
  error is `0.00333977`; strict parity fails. The temporary engine is not
  retained.

Automatic gates fail 23 checks before manual review and 24 after it. The
failures include absolute class-1 precision/recall/F1, GAP comparisons,
restricted-FP reduction, illumination stability, TensorRT parity, and manual
semantic review. Runtime and equation correctness cannot override them.

## Artifacts and no-repeat rule

Preserve the finalized 18-artifact payload totaling 63,751,425 bytes, including
the two ONNX files, exact CSVs, nine contact sheets, replayable XAI NPZ, report,
summary, and manifests. There is no checkpoint, TensorRT engine, or temporary
feature cache. Preserve the five-file interrupted reference because it is the
pre-metric bit-exact recovery authority, not an obsolete duplicate run.

Do not rerun or sweep WILDCAT map count, top/bottom fraction, alpha, scaling,
capture layer, head depth, head learning rate, epochs, optimizer, seed, fold,
lighting strength, bbox rule, class-specific fusion, or export tolerance. Do
not rescue this route with nearby WILDCAT/MCL/MIL pooling, negative-evidence
losses, or a scratch pair. The failure is behavioral and visual, not a missing
hyperparameter search.

A next route must be equation-distinct and first prove that its signal separates
class-1 morphology from chromatically similar class 0/2/4 fruit under dim and
bright shifts. Foreground localization, bbox alignment, broad yellow response,
or generic suppression alone is insufficient.

## Closure verification

- Independent PowerShell replay reconstructed 5x5 confusion matrices directly
  from all 7,372 prediction rows. Twelve condition/variant metric sets match
  the finalized JSON with maximum absolute error `1.110223e-16`.
- Compileall, pyflakes, PowerShell AST parsing, focused pytest `18/18`, full
  pytest `1442/1442`, and `git diff --check` pass.
- The official WILDCAT source worktree is clean. No unknown process was killed;
  the corrected formal exits with no owned Python/TensorRT/ffmpeg process.
- Read-only retention passes over 758 run directories using all 48 compaction
  manifests. All 210 compacted originals remain absent, `blockers=[]`, nothing
  was deleted, and free space is `102.692 GiB`. Retention summary SHA is
  `b036ff312d511e10e59729559eeb6c512e85c5bfd1cd68b3e57c66dd7fdc6d61`.
- Keeper/current-command/history SHAs remain
  `1f49d577...2677` / `36b9aa1a...0faf` / `39bd2879...8f53`.
  Current-best commands remain three revisions and two actual updates.
