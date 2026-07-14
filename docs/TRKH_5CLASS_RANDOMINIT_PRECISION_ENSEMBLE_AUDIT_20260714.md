# TRKH 5-Class Random-Init and Precision-Ensemble Audit - 2026-07-14

## Scope

This audit closes the user-run scratch experiment
`runs/full_v8_yolof_randominit_30e_20260714_105524`. Raw data was not edited.
All model selection and ensemble fitting used `yolo_f/val`; the frozen rule was
applied to test exactly once only after all 14 readiness gates passed.

Primary references:

- Lakshminarayanan et al., *Deep Ensembles*, NeurIPS 2017:
  https://papers.nips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html
- Guo et al., *Calibration of Modern Neural Networks*, ICML 2017:
  https://proceedings.mlr.press/v70/guo17a.html
- Geifman and El-Yaniv, *Selective Classification*, NeurIPS 2017:
  https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html

## Run integrity and convergence

- Scratch initialization, seed 42, image 256, immutable YOLO crops, 7,245,590
  parameters, maximum 30 epochs, and early stop at epoch 23.
- Total training time was 5,093.86 seconds. The fair class-gap-aware selector
  selected epoch 20; epoch 23 had the highest observed raw macro F1.
- Legacy `summary.json` used `best_macro_f1` for the maximum observed macro F1,
  even when `best_epoch` came from a different selector. The corrected audit
  artifact distinguishes these values, and future summaries now emit explicit
  selected-checkpoint and maximum-observed fields.
- `last.pt/model_state` contains raw train weights; validation used
  `last.pt/ema_model_state`. The fail-closed EMA extractor verified all 185
  tensor keys and shapes before making a compact evaluation checkpoint.

## Independent checkpoint results

| Model / split | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|
| Selected keeper / val | 0.882925 | 0.603093 | 0.774834 | 0.678261 | 117 / 77 / 34 |
| Random-init best.pt / val | 0.874172 | 0.535865 | 0.841060 | 0.654639 | 127 / 110 / 24 |
| Random-init epoch-23 EMA / val | 0.877334 | 0.545852 | 0.827815 | 0.657895 | 125 / 104 / 26 |
| Random-init last raw weights / val | 0.862000 | 0.517241 | 0.794702 | 0.626632 | 120 / 112 / 31 |
| Selected keeper / final test | 0.886517 | 0.597826 | 0.753425 | 0.666667 | 55 / 37 / 18 |
| Random-init best.pt / final test | 0.873884 | 0.553398 | 0.780822 | 0.647727 | 57 / 46 / 16 |

The challenger learns a real recall complement, but class-1 precision is too
low for an agricultural deployment: on validation it gains 10 true positives
while creating 33 additional false positives. Its dominant class-1 false-
positive directions are `0->1=62`, `2->1=23`, `4->1=20`, and `3->1=5`.
Neither its epoch-20 checkpoint nor epoch-23 EMA replaces the keeper.

## Frozen precision ensemble

The formal tool aligned all 2,606 validation rows and 2,577 source groups,
used five `StratifiedGroupKFold` folds, and failed closed on test paths,
misalignment, duplicates, group overlap, and nonfinite probabilities.

- Fold winners stayed within candidate weight `0.40..0.45` and class-1 margin
  `0.028..0.038`.
- OOF macro/class-1 F1 was `0.886214/0.683544`; class-1 precision/recall was
  `0.654545/0.715232`.
- The median-locked rule is keeper probability `0.60`, challenger probability
  `0.40`, and class 1 must exceed the strongest non-class-1 probability by
  `0.034`.
- Locked full-validation macro/class-1 F1 was `0.890298/0.696774`; class-1
  precision/recall was `0.679245/0.715232`.
- Versus keeper it changed 62 rows: 36 corrections, 18 harms, 8 neutral;
  class-1 FP remove/create was `27/1`, while FN rescue/TP break was `1/10`.
- All 14 declared validation gates passed. Readiness summary SHA-256 is
  `6c6397d3561ec917d9716ecfd2c6b65a616433e3a07c45031df898578d07a8c8`.
- The hardened apply path now requires that exact summary hash for `--split
  test`; malformed probability vectors and nonempty output roots fail closed.

The frozen rule was then applied to final test once. Macro F1 improved
`0.886517 -> 0.887869`; class-1 precision/F1 improved
`0.597826/0.666667 -> 0.666667/0.675676`, while recall fell
`0.753425 -> 0.684932`. Per-class F1 became
`[0.944551, 0.675676, 0.900763, 0.952518, 0.965839]`.

This is a useful optional precision mode, not a universal replacement: it uses
two forward passes, lowers class-1 recall, slightly lowers class-3/class-4 F1,
and current ONNX/TensorRT/video wrappers do not yet reproduce the dual-model
rule. Test results must not be used to tune another threshold.

### Post-review integrity hardening

- Prediction vectors now fail closed on nonfinite, negative, and zero-sum
  values both at CSV ingestion and direct ensemble application. Every row's
  split is verified from `image_path`, independent of the CSV filename.
- A frozen `--split test` application requires the exact validation-summary
  SHA-256; a missing, malformed, or mismatched hash is rejected before reading
  the prediction inputs. Existing validation artifacts replayed bit-identical
  across all six files, including summary SHA `6c6397d...a8c8`.
- Native attention now fails if `return_attention=True` does not supply a
  prefix-plus-full-grid tensor. It cannot silently return to a square reshape
  of pruned QKV tokens.
- Paired XAI validates exact sample membership, cross-model targets, declared
  cohort target/prediction fields, correctness flags, attention provenance,
  and aggregate hashes for every input `case.json`. The 30-case precision
  cohort replay had known provenance, `0` missing contact-sheet tiles, and
  keeper/candidate case-set SHA `e366cd37...be904` / `f5883c3c...6849e`.
  The five-case native-attention replay also had `0` fallback and `0` missing
  tiles, with case-set SHA `afb1dea1...8d50` / `400b85e0...de9d`.
- Post-review verification passed focused tests `33/33`, full pytest
  `859/859`, compile checks, current full-pipeline/export/video preflights, and
  retention over 614 run directories with `blockers=[]`. These changes do not
  promote the challenger or alter the current-best commands.

## Robustness

The challenger is weaker on clean and center-occlusion validation, but its
class-1 recall complement persists under illumination changes:

| Condition | Keeper macro / class-1 F1 | Challenger macro / class-1 F1 |
|---|---:|---:|
| Clean | 0.882925 / 0.678261 | 0.874172 / 0.654639 |
| Center occlusion | 0.876231 / 0.662757 | 0.867878 / 0.627778 |
| Dim | 0.780164 / 0.435331 | 0.793512 / 0.527316 |
| Bright | 0.809343 / 0.512821 | 0.802956 / 0.524272 |
| Low contrast | 0.801929 / 0.491429 | 0.820226 / 0.546875 |

This supports preserving member diversity, not promoting the noisy member by
itself. The next trainable route should retain separate late decision paths
instead of averaging away the complementary lighting response.

## XAI audit correction and findings

Earlier raw-attention images were structural `feature_map_fallback` maps. The
late QKV hook observed a pruned non-square token set. XAI now performs a
no-grad `forward_features(return_attention=True)` pass, disables pruning for
that pass, verifies the full patch grid, and records the exact last block
source. A five-case candidate/keeper rerun produced 10/10 native sources from
`forward_features.return_attention.blocks[7]` and zero fallback.

- Overall native-attention foreground mass increased
  `0.905981 -> 0.963862`; border mass decreased `0.277641 -> 0.254573`.
- In the larger 30-case paired cohort, candidate Grad-CAM foreground mass
  increased `0.84690 -> 0.92122` and border mass fell `0.28497 -> 0.22793`.
- Removed false positives become more fruit/surface focused, but harms show
  edge/glare/truncated-crop attention and missed spotted-surface evidence.
- Retained false positives remain foreground-dominant. Wide background is not
  the current bottleneck; pale-green/yellow maturity and damage-surface overlap
  are the unresolved features.

The hashed manual review is stored beside the native-attention contact sheets.

## Decision and next boundary

1. Reject the random-init checkpoint and epoch-23 EMA as keeper replacements.
2. Retain the frozen two-checkpoint rule as a precision-oriented research and
   future deployment candidate; do not tune it again on test.
3. Keep current-best train/export/video commands on the existing deployable
   keeper until dual-checkpoint export/video support passes equivalence tests.
4. Do not repeat nearby seed, weight, margin, threshold, checkpoint-soup, SWA,
   or epoch sweeps from this result.
5. The next model change must convert the observed member complement into a
   train-only supervision mechanism with separate late heads or subnetworks,
   direct class-1 TP protection, conservative `0/2/4->1` control, and a locked
   full-validation gate before any test access.

## Closure verification

- Focused tests: `28/28`; full pytest: `850/850`.
- PowerShell parse passed for current full pipeline, export, video, and V8
  launchers. Full pipeline, TensorRT export, and PyTorch video preflights passed
  under direct native invocation with checked exit codes.
- Guarded cleanup removed only 13 superseded files (`148,024,034` declared
  bytes) after inventory and preservation hash checks; observed free-space gain
  was `147,976,192` bytes.
- Retention audit passed over 613 run directories with `blockers=[]`.
