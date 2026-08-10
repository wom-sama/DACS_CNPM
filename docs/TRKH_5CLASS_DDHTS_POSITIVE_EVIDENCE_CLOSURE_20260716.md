# TRKH 5-Class DDHTS Positive-Evidence Closure - 2026-07-16

## Decision

Close fixed DDHTS-style cross-layer/cross-frequency binary histograms on the
current keeper. Do not integrate the descriptor into the trainer, access
validation/test, launch a probe/full train, or revise current-best commands.

## Why this route was tested

- Li et al.'s 2026 DDHTS-Net is directly relevant to plant texture and is more
  specific than the already rejected fixed wavelet, LBP, Deep-TEN, Gabor, and
  high-frequency routes. Its distinct element is joint sign coding across CNN
  depths and across three IUWT contrast domains.
- The paper's headline accuracy is not an apples-to-apples baseline. It uses
  AlexNet/VGG features, an SVM, random class-stratified `80/20` image splits
  repeated 50 times, and a tuned `C=250`; it does not report source-group
  separation, class F1, SVM settings, backbone weight provenance, or code.
- The prospectively committed protocol transferred only Equations 2-18 to the
  frozen scratch keeper. Protocol commits are `746fe14` and `09a8c5a`; final
  protocol SHA is `bbaec0ff...9e1970`. Implementation commit is `e6b75e1`.

## Locked A0

- Data scope: only `yolo_f/train`, canonical `9215` object rows and `8064`
  source groups. The existing source-disjoint split was reused exactly:
  `7372` fit and `1843` holdout rows with zero source overlap.
- Readout fit cohort: all `432` fit class-1 references plus the exact `186`
  restricted hard negatives. Validation and test were forbidden.
- Descriptor: three keeper CNN stem levels, `C=32`, common `32x32` grid, four
  views `(RGB,D1,D2,D3)`, 24D InterGSU plus 16D IntraHSU, total 40D.
- Positive-evidence rule: one fixed balanced linear SVM (`C=1`); its threshold
  is the minimum clean-fit raw class-1 TP score minus `1e-6`. Only raw class-1
  predictions below that threshold may be changed to their raw runner-up.
- The same fit and threshold were applied unchanged to clean, dim, bright, and
  low-contrast holdout images.

## Result

- Clean raw and candidate were identical: macro/class1 F1 remained
  `0.949323/0.849206`, class1 precision/recall remained
  `0.748252/0.981651`, restricted FP remained `36 -> 36`, and changes were
  `0`. Direction AUROC was only `0.548546`.
- The locked threshold was `-2.419230`. Even a label-assisted clean holdout
  threshold that was allowed to preserve all 107 raw class-1 TP could remove
  `0/36` restricted FP. The weakest TP score (`-1.613407`) was below the
  weakest FP score (`-1.455529`), so the zero-TP-loss failure is representation
  overlap, not an unlucky fit threshold.
- Under dim lighting, one true class-1 prediction was vetoed (`1 -> 0`) while
  no restricted FP was removed. Class1 precision/recall/F1 changed by
  `-0.004741/-0.009174/-0.007200`; macro F1 fell `-0.001664`.
- Bright and low-contrast decisions were unchanged. Direction AUROC was
  `0.558025/0.391304/0.456724` for bright/dim/low contrast. A zero-TP-loss
  oracle removed `0/62`, `0/46`, and only `1/53` restricted FP respectively.
- Standardized descriptor effective rank was only `5.346925` out of 40. The
  same-row score correlation from clean to bright/dim/low contrast was only
  `0.476713/0.490446/0.576362`; maximum absolute shifts reached
  `5.535553/6.781914/3.889473`.
- XAI contact sheets show D1-D3 emphasizing fruit contour and nearby background
  edges together. The dim-broken TP changes score from `-1.092987` clean to
  `-2.556979` dim even though its contrast maps remain visually similar. The
  binary sign readout is unstable around activation zero crossings and does
  not isolate positive class-1 surface evidence.

## Replay and artifacts

- Independent CSV replay refit the fixed scaler/SVM and reproduced all `4 x
  1843` candidate predictions, four confusion matrices, threshold, effective
  rank, AUROC, and every original manifest payload hash. Cross-process LibSVM
  scores differed by at most `1.0371e-6`; decisions were exact and the reviewed
  numerical tolerance was `2e-6`.
- Evidence directory:
  `runs/audit_ddhts_positive_evidence_readiness_20260716`.
- Summary/original-manifest/replay/score-distribution/closure-manifest SHAs:
  `e65e2b65...d6512b`, `7fcb97a6...9cb65`,
  `fb07dd0b...d2d5f4`, `a09abee8...0f79c`, and
  `89d14eb3...efb76`.
- Ten closure payloads occupy `9,328,181` bytes. No checkpoint, binary model,
  validation prediction, test prediction, or raw-data write was produced.

## Engineering closure

- Compilation passed; focused tests passed `6/6`; full pytest passed
  `1176/1176` in `42.85 s`; PowerShell parse and real preflight passed without
  creating a run directory.
- Read-only retention passed over `699` run directories with `blockers=[]`;
  retention summary SHA is `f89fa7a7...26387`. Drive D retained `72.232 GiB`
  free.
- Keeper, scratch complement, current command packet, and command-history
  hashes remain exact. Command tracking remains three revisions and two
  updates after the initial revision.

## No-repeat rule

Do not sweep DDHTS wavelet scales, channel count, selected layers, feature-map
reduction, sign/histogram variants, SVM kernel/C/class weights, threshold,
seed, condition settings, or fusions on this keeper. This result also closes a
nearby revisit of fixed LBP-like deep binary patterns. Reopen only if a newly
sourced trainable representation or supervision signal changes the underlying
class-1 TP versus restricted-FP support before thresholding.
