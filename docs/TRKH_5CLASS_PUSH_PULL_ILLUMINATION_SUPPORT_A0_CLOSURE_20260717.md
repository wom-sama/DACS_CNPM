# TRKH 5-Class Push-Pull Illumination Support A0 Closure - 2026-07-17

## Decision

Reject first-layer push-pull inhibition on the current keeper before a matched
five-epoch pair, trainer integration, validation, test, full training, or a
current-best command update.

The locked `h=2, alpha=1` operator is equation-faithful, active, and within the
runtime gate. It does not provide the required class-1 support separation. Its
clean object-only AUROC is slightly worse than the exact `h=1` native control,
the dim shift degrades AUROC materially, and its only useful false-positive
gain under bright lighting is purchased by unsafe class-1 TP loss. The pull
response is almost equally strong for TP and restricted FP in every condition.

Peak inference allocation is also `1.908909x`, and strict ONNX parity misses
the prospectively locked tolerance. No automatic or manual route to a short
pair is authorized. Current-best commands remain unchanged.

## Authority and immutable execution

- Peer-reviewed source: Strisciuglio, Lopez-Antequera, and Petkov, *Enhanced
  robustness of convolutional networks with a push-pull inhibition layer*,
  Neural Computing and Applications 2020, DOI `10.1007/s00521-020-04751-8`.
- Authors' MIT repository commit/tree:
  `c340f329368e60c888ebdca8e30c373cdad9d29f` /
  `e9c984b6a71dd2f67aeb56388b9e8e84f1d5e5b7`.
- Official paper/module/license SHAs:
  `04cf18f...1580` / `1de24434...b6a` / `0d4cced2...564`.
- Corrected prospective protocol SHA:
  `111e648ad6c4a37e8839794a61f18055890ca512bd2ec42d5813a9d87ed8c7ae`.
- Implementation/fix commits: `1255229` / `5a8a702647f7a70920d53deaff8628584424be8d`.
- Formal evidence directory:
  `runs/audit_push_pull_illumination_support_a0_20260717`.
- Formal summary/manifest SHAs:
  `e1d3a7ad97d079f791e6925adcb2f35bfee2cf811095253364f750a43793a19e` /
  `c272289ecf8548949a3bf795f8f65049f0dbdb8df15a5b6525e72e19e90c4c10`.

The sole completed formal run used only the exact 607-row train cohort: 421
keeper class-1 TP and 186 restricted `{0,2,4} -> 1` FP. Fold TP/FP counts are
`112/45`, `100/48`, `101/52`, and `108/41`; ordered cohort SHA is
`a2689d1b...38bd`. Four source-fold OOF logistic readouts were fitted on clean
descriptors only. Validation, test, holdout predictions, image training, raw
data changes, pretrained weights, and trainable manifests were not used.

An earlier invocation stopped before equation, resource, or dataset
measurement because candidate introspection reapplied a Conv2d-only assertion.
Its verified empty directory was removed and a regression test was added. It
is an engineering interruption, not a second formal result.

## Information gate

Per-role thresholds are the lower third percentile of clean TP scores, as
locked prospectively.

| Condition | Native AUROC | Push-pull AUROC | Delta | Native/Candidate TP retention | Native/Candidate FP rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 0.606365 | 0.601946 | -0.004419 | 0.971496 / 0.971496 | 0.016129 / 0.037634 |
| dim | 0.581386 | 0.555079 | -0.026307 | 1.000000 / 1.000000 | 0.000000 / 0.000000 |
| bright | 0.592534 | 0.583812 | -0.008722 | 0.912114 / 0.881235 | 0.075269 / 0.161290 |
| low contrast | 0.588805 | 0.590683 | +0.001877 | 0.990499 / 0.995249 | 0.016129 / 0.010753 |

The clean FP-rejection gain is only `+0.021505` versus the locked `+0.08`
requirement. Bright FP rejection gains `+0.086022`, but TP retention falls
`-0.030879` and violates both the absolute and native-relative safety gates.
Dim produces no FP rejection and has the worst AUROC delta. Low contrast is a
small AUROC gain with worse FP rejection.

The information gate fails 13 checks, including clean AUROC level/delta,
shifted AUROC level/count/worst delta, clean and shifted FP rejection, shifted
TP safety, object candidate versus native, and FP-specific mechanism checks.
Candidate object AUROC remains within `0.02` of context AUROC, so the failure
cannot be rescued by claiming that context alone hid a strong object signal.

## Mechanism and manual XAI review

| Condition | TP pull/push | FP pull/push | FP minus TP |
| --- | ---: | ---: | ---: |
| clean | 0.884254 | 0.885618 | +0.001364 |
| dim | 0.869767 | 0.871683 | +0.001915 |
| bright | 0.883275 | 0.882890 | -0.000385 |
| low contrast | 0.881037 | 0.882487 | +0.001451 |

Object energy exceeds outside energy for both cohorts in all four conditions,
so the operator is foreground-responsive. It is not FP-specific: every gap is
far below the locked `0.01`, and bright suppresses the TP fraction slightly
more than the FP fraction.

All four 16-row contact sheets were manually inspected after the formal run,
covering the four lowest/highest TP and FP scores in each condition. Fruit is
usually energetic, but leaf, hand, tray, crop-edge, and textured-background
boundaries are also strongly active. Under shared scaling, pull and inhibited
maps are visually near-indistinguishable across TP and FP, including dim and
bright extremes. The 64 reviewed rows provide no visual evidence of selective
restricted-FP inhibition. Formal `manual_review_status=pending` remains
unchanged because the formal summary is immutable; this closure records the
required human review and its rejecting result.

Contact-sheet SHAs for clean/dim/bright/low-contrast are
`7f9edc12...01ef`, `3f8b5033...42d1`, `1d1342a7...35f0`, and
`aca52bc2...a9c`.

## Equation, identity, and deployment

- Official-source and independent-oracle output, input-gradient, and
  weight-gradient maximum errors are exactly `0`.
- The `h=1` causal control has exact operator and full-model identity: maximum
  logit error `0` and zero prediction mismatch.
- Finite-difference error is `1.0193e-11`; strict BF16 error is `0.013557 <=
  0.02`, with finite nonzero gradients.
- The `h=2` candidate is nondegenerate: finite logits, four predicted classes,
  maximum class share `0.868204`, and 81 prediction differences versus native.
- Cohort mapping is exact, but the independently replayed native declarations
  are not all exact. This remains a structural failure; no post-formal
  exclusion or declaration rewrite is allowed.
- Batch-32 BF16 native/candidate medians are `63.486977/71.980034 ms`, ratio
  `1.133776 <= 1.20`.
- Peak allocations are `406,110,720/775,228,416` bytes, ratio
  `1.908909 > 1.10`.
- Static standard-domain ONNX and TensorRT parse/build pass, but ORT maximum
  absolute error is `1.0490417e-5`, above the locked `1e-5` by `4.904e-7`.
  The temporary ONNX and engine were deleted. The tolerance is not relaxed.

The structural gate therefore fails declaration replay, strict ONNX parity,
and memory. These failures are independent of the 13 information failures.

## Artifacts and no-repeat rule

Preserve the seven-file compact evidence payload totaling `13,482,897` bytes,
plus its manifest. CSV/report SHAs are
`9b290cb9206eeef1b303838c7881eed2ff81eb5b323a388a9b8b77d5dd42f4d6` /
`f7ac8252153ac1b60010528496900f5db05ddbb293ed3a7dd50b794532cbfa8d`.
No checkpoint, ONNX, TensorRT engine, validation prediction, test result, or
production model artifact is retained.

Do not rerun or sweep push-pull scale, `alpha`, learned inhibition, depth,
insertion point, kernel rescaling, bias, activation, normalization, optimizer,
schedule, threshold, fold, lighting strength, seed, or export tolerance on this
keeper. Do not combine this route with closed edge/frequency/localization,
pooling, shift, consistency, routing, post-hoc, or support-contraction methods.

The transferable result is narrower: first-layer foreground energy is easy to
obtain, but class-conditional FP-versus-TP inhibition is absent. A next route
must be equation-distinct, deployment-light, and prove a class-conditional
positive-support signal under dim and bright shifts before any image epoch.

## Closure verification

- Compileall, pyflakes, PowerShell AST, focused `11/11`, related integration
  `23/23`, and full pytest `1424/1424` pass.
- The official source worktree is clean after removing only generated
  `pushpull/__pycache__`; no source file was changed.
- Read-only retention passes over 755 run directories and all 48 object
  compaction manifests. All 210 compacted originals remain absent,
  `blockers=[]`, nothing was deleted, and free space is `102.792 GiB`.
  Retention summary SHA is
  `b9a09fadca363d439248d77190eac7b85efc1dd7da7ff3fb51fe9e1c2291e76b`.
- Current-best command/history SHAs remain
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`,
  still three revisions and two actual updates.
