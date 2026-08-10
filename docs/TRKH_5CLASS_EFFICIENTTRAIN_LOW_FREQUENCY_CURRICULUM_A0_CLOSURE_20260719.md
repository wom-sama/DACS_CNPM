# TRKH 5-Class EfficientTrain Low-Frequency Curriculum A0 Closure - 2026-07-19

## Decision

Reject the locked EfficientTrain low-frequency curriculum route on the current
keeper before a matched scratch pair, trainer integration, image epoch,
validation, test, full training, or current-best command update.

The input transform is equation-faithful and all structural checks pass, but
its class-1 false-positive suppression signal is at or below chance. The B176
view has clean AUROC `0.472212`, removes only one of 186 restricted false
positives at the locked OOF threshold, and breaks 11 of 421 class-1 true
positives. B224 is slightly less harmful but still non-informative at clean
AUROC `0.501762` and removes only six false positives while breaking 13 true
positives. Illumination shifts do not rescue either view.

All four contact sheets fail manual review. The removed high-frequency
residual mainly follows bbox/padding edges, fruit silhouettes, and generic peel
texture in both cohorts. It does not expose a stable class-1 false-positive
mechanism. No subsequent stage is authorized.

## Authority and recovery

- Accepted paper: Wang et al., *EfficientTrain: Exploring Generalized
  Curriculum Learning for Training Visual Backbones*, ICCV 2023.
- Paper-linked MIT source commit/tree:
  `bdefd277c71ba3bfc2a88c13768b215f15301350` /
  `5c1362c15be3695eba64d636a8019944c026ac61`.
- Paper/supplement SHAs:
  `3191ac...1eae` / `aa542b...c92c`.
- Original prospective protocol SHA:
  `c632f92074dd394528830ebeaa5713d47a36e3968d3aa0621008819ac5c1c7a2`.
- Declaration-only erratum protocol SHA:
  `c8cd9bcf65b297c43e7dacc5e588a82f6edccf58c53a1e625f8f8a79ff071be4`.
- Initial implementation and erratum commits:
  `ce9a75f04f2085c299ca87a78d34355c3a128e84` /
  `51efab0105d1e10ba66b3f6a70d9250e783645af`.
- Final evidence directory:
  `runs/audit_efficienttrain_low_frequency_curriculum_a0_20260719`.
- Final summary/manifest/replay SHAs:
  `c7357893662ad029896610b1fdd66f75873d6e59dc5d77d0c2ee360b14edd03e` /
  `8555691ec5dccea9e9d7cb9c76e333b00cc6e99f2e9a2912e508c0c7f90e3109` /
  `70399c41ad8e33ae5950e51fc55c4479f670af0c2dee35a5105ff3ada58ed36a`.

The first formal invocation from `ce9a75f` stopped during native declaration
replay at cohort position 337, sample 3657, before any B176/B224 forward or
candidate metric. It reproduced the established BF16 near tie as target 2,
declared prediction 1, replay prediction 2. Preserve the two-file interrupted
reference under
`runs/audit_efficienttrain_low_frequency_curriculum_a0_interrupted_declaration_20260719`
at manifest/transcript SHAs `1ed3b08b...5e702` / `80d636a9...75ba3`.

The pushed erratum authorized one corrected formal run. It requires exactly
that sole tuple, keeps all 607 rows in the primary analysis, and repeats every
information, hard-decision, and mechanism gate after read-only removal of
sample 3657 without refitting the full-cohort thresholds. Both analyses fail,
so the known near tie cannot create or rescue the decision.

## Locked experiment

- Keeper parameters: `7,245,590`, unchanged before and after the audit.
- Train-only cohort: 607 rows, comprising 421 class-1 true positives and 186
  restricted false positives over four fixed source folds.
- Views: official four-corner FFT crop at B176 and B224, with native B256 as
  the bit-exact control.
- Conditions: clean, lighting dim, lighting bright, and low contrast.
- Decision rule: fold-held-out clean-TP 97th-percentile suppression threshold,
  fitted once on the full cohort.
- Mechanism control: object-region versus outside-region high-frequency
  residual energy.
- No validation prediction, test sample, training update, checkpoint, engine,
  raw-data edit, or trainable manifest was used.

## Information result

| View/condition | FP-vs-TP AUROC | TP retention | Restricted-FP rejection | TP breaks | FP removals |
| --- | ---: | ---: | ---: | ---: | ---: |
| B176 clean | 0.472212 | 0.973872 | 0.005376 | 11 | 1 |
| B176 dim | 0.461970 | 0.995249 | 0.005376 | 2 | 1 |
| B176 bright | 0.426800 | 0.995249 | 0.000000 | 2 | 0 |
| B176 low contrast | 0.445228 | 0.992874 | 0.000000 | 3 | 0 |
| B224 clean | 0.501762 | 0.969121 | 0.032258 | 13 | 6 |
| B224 dim | 0.518747 | 0.976247 | 0.032258 | 10 | 6 |
| B224 bright | 0.473220 | 0.978622 | 0.016129 | 9 | 3 |
| B224 low contrast | 0.507701 | 0.988124 | 0.010753 | 5 | 2 |

Four-condition mean AUROC is `0.451552` for B176 and `0.500358` for B224.
B176 therefore fails the requirement to be no worse than B224. Positive clean
FP rejection occurs in only one fold for B176. Its OOF rule removes one FP
while breaking 11 TP, the opposite of the required precision-safe behavior.

Direct low-frequency hard predictions are active but unsafe. B176 records 38
corrections, 21 harms, 45 restricted-FP removals, and 20 class-1 TP breaks.
B224 records 25 corrections, six harms, 28 restricted-FP removals, and five TP
breaks. Net corrections do not override the prospectively locked TP-safety and
OOF reliability failures.

The unchanged-threshold 606-row sensitivity analysis fails the same ten
information checks and three mechanism checks. It is not an outlier-driven
rejection.

## Mechanism and visual review

For B176, object-residual AUROC is only `0.527354/0.518453/0.531288/0.525847`
on clean/dim/bright/low-contrast. Its advantage over outside-residual AUROC is
`0.019730/0.037136/0.006206/0.015018`, below the required `0.03` in three of
four conditions. The object FP-minus-TP residual gap is tiny and becomes
negative under dim lighting.

The clean, dim, bright, and low-contrast contact sheets were all inspected.
Residual energy repeatedly outlines resized borders, gray padding, fruit
boundaries, stems, cracks, and broad peel texture in both TP and restricted-FP
examples. High and low ranked rows do not reveal a repeatable maturity/color/
risk-surface distinction. Manual review is finalized as `fail` from locked
pre-review summary SHA `0ef62b03...04f2`.

## Structural and resource result

- Official AST-extracted function, independent FFT-shift oracle, and spectral
  replay errors are exactly zero for B176 and B224.
- Maximum low-frequency reconstruction error is `2.3841858e-7`; native B256 is
  bit-exact and returns the same tensor storage.
- Dynamic grids/prefix/pruning geometry are exact at `11x11/7/79`,
  `14x14/7/128`, and `16x16/7/167` for B176/B224/B256.
- Model state, parameter count, CPU RNG, and CUDA RNG are unchanged.
- The transform is training-view-only and adds zero deployed parameters and
  zero deployed inference operations. Its non-isolated batch-32 median was
  `7.6484 ms`, recorded for diagnostics rather than promotion.
- Independent replay reconstructs every metric, threshold, decision, full-607
  gate, and 606-row sensitivity gate within `1e-12`.

## Artifacts and no-repeat rule

Preserve the finalized 11-artifact payload totaling `15,177,238` bytes plus
its artifact manifest, including all four contact sheets, 2,428 prediction
rows, 4,856 mechanism rows, independent replay, report, and final summary.
Preserve the declaration-only interrupted reference as erratum authority.

Do not rerun or sweep B176/B224/B256 widths, stage count, FFT crop convention,
energy scale, curriculum schedule, threshold quantile, fold, lighting strength,
object/outside mask, batch, seed, or nearby blur/frequency curricula on this
keeper. Do not combine this route with closed high-frequency/FcaNet/OctConv/
PDC/LPD/suppression families. The failure is a lack of class-conditional
information, not an implementation or schedule defect.

A next route must be equation-distinct and first show a train-only signal that
separates class-1 TP from restricted FP under all illumination conditions while
retaining true positives. Generic foreground localization, border energy,
frequency magnitude, broad color response, or support contraction is
insufficient.

## Closure verification

- Compileall, pyflakes, PowerShell AST, focused pytest `13/13`, full pytest
  `1455/1455`, and staged diff checks pass.
- The corrected formal ran from pushed commit `51efab0`; tracked HEAD and
  upstream matched and the official source worktree remained clean.
- Read-only retention passes over 761 run directories and all 48 object-form
  compaction manifests. All 210 compacted originals remain absent,
  `blockers=[]`, nothing was deleted, and free space is `102.632 GiB`.
  Retention summary SHA is
  `874e70a4aad39948427dcc8e9c4f880ecc49fa8e8fb0fa528d73bfe6328264b8`.
- Keeper/current-command/history SHAs remain
  `1f49d577...2677` / `36b9aa1a...0faf` / `39bd2879...8f53`.
  Current-best commands remain three revisions and two actual updates.
