# BBox-Centered Log-Polar Stem A0 Closure (2026-07-20)

## Decision

Reject direct bbox-centered log-polar RGB sampling before the frozen CNN stem
on the current no-pretrain keeper. The sole source-disjoint train-only A0 passes
all provenance, data-isolation, geometry, convergence, and exact-replay checks,
but fails 22 automatic information, precision, TP-safety, specificity, and
lighting gates. It does not authorize model/trainer integration, validation or
test access, a smoke, a probe, full training, or current-best command promotion.

## Locked Evidence

- Accepted sources: Ebel et al., ICCV 2019, and Esteves et al., ICLR 2018.
- Official repositories are pinned outside TRKH at commit/tree
  `45d0a922...cfe8`/`527738e3...091` (Apache-2.0) and
  `c6a4ad61...b7a3`/`a0b328bb...0cdc` (MIT). No source or weights were copied.
- Protocol SHA-256: `dc04f38e...6f76`; protocol/implementation commits:
  `fd69860`/`d0d16d4`.
- Auditor/test/launcher SHA-256:
  `94d651aa...84f0`/`b5583ffb...3e90`/`c24e7cd2...ded6`.
- Formal summary/manifest/prediction/cache/sheet SHA-256:
  `968b53a1...9db`/`84a2fd4c...4ae`/`a876e4d3...75a`/
  `4acc06c0...984`/`235d411a...a81`.
- Keeper SHA-256: `1f49d577...2677`.

The formal uses all 9,215 immutable `yolo_f/train` object rows and the fixed
five CIDT source folds. Every fit/hold source overlap is zero, every source
derangement changes source, all 20 fixed logistic readouts converge in 17-23
iterations, and the raw-data metadata SHA remains
`a7caeccb...5fb6`. Validation and test pixels, labels, predictions, and metrics
remain unopened.

## Clean Result

| Role | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | Predicted class 1 |
|---|---:|---:|---:|---:|---:|
| Keeper probabilities | 0.939876 | 0.700265 | 0.975970 | 0.815444 | 754 |
| Matched Cartesian stem | 0.824906 | 0.857843 | 0.323475 | 0.469799 | 204 |
| Direct log-polar candidate | 0.823356 | 0.855000 | 0.316081 | 0.461538 | 200 |
| Direct linear-polar control | 0.840895 | 0.871560 | 0.351201 | 0.500659 | 218 |
| Source-deranged placebo | 0.933767 | 0.693548 | 0.953789 | 0.803113 | 744 |

Candidate minus matched Cartesian changes macro/class-1 F1 by
`-0.001550/-0.008260`, precision/recall by `-0.002843/-0.007394`, and yields
`56/57` corrections/harms. It rescues 12 class-1 false negatives while breaking
16 true positives. Restricted `{0,2,4}->1` FP are unchanged at `28`: five are
removed and five are created.

The candidate also loses to direct linear-polar by
`-0.017539/-0.039120/-0.016560` in macro F1/class-1 F1/class-1 precision. It
loses to the source-deranged placebo by `-0.110411` macro F1 and `-0.341574`
class-1 F1. Its clean TP-versus-restricted-FP direction AUROC is only
`0.518060`; linear-polar reaches `0.571073`. Effective rank is `13.986852`,
below both the locked minimum 24 and the Cartesian/linear controls
`14.726585/15.175502`.

Only one of five source folds improves class-1 F1 versus Cartesian. Fold
macro/class-1-F1 deltas are
`-0.006509/-0.023258`, `+0.002082/+0.009125`,
`+0.000372/-0.004396`, `+0.000779/-0.003279`, and
`-0.004718/-0.020721`. This is not a hidden aggregate win.

## Robustness And Visual Diagnosis

Under dim, bright, and low-contrast replay, candidate-minus-Cartesian
macro/class-1-F1 deltas are respectively
`+0.006075/+0.015179`, `-0.008316/-0.033613`, and
`+0.005541/+0.008319`. Class-1 precision changes
`+0.005763/+0.014587/-0.009231`; restricted-FP net removals are
`-1/+8/-3`. Bright light contracts true class-1 support strongly, while low
contrast creates more restricted FP. The route therefore fails the locked
worst-condition precision, recall, F1, and macro gates.

Manual review of `fixed_transform_activation_sheet.png` confirms that bbox
centering, radial order, and angular order are correct. The linear and
log-polar images preserve recognizable fruit-surface structure, so the result
is not explained by an axis reversal or an off-object center. However,
log-polar compression allocates little support to the outer fruit region and
the direct-stem maps emphasize silhouette, shadow boundaries, padding/context
transitions, and the angular seam. They do not isolate lesion morphology from
the chromatically similar class 0/2/4 surfaces. The class-1 shadow sample is a
particularly clear nuisance response.

This is an information and selectivity failure, not a loader, convergence,
geometry, or replay failure. Both real residual readouts broadly suppress
class-1 support, while the source-deranged descriptor stays close to keeper
behavior. Direct linear-polar is less damaging than log-polar but remains below
the locked direction gate and far below keeper performance. Close log/linear
radius schedules, center/scale/band variants, stem attachment, seam handling,
readout `C`, fold/seed, threshold/router, polar fusion, and nearby polar-stem
sweeps on this keeper.

## Runtime And Retention

Four-condition extraction takes `392.258 s` inside `447.951 s` total runtime.
Windows workers `4/4`, persistent workers, prefetch 2, no pin memory, and the
RTX 4060 Laptop GPU peak allocation of `2,254.911 MiB` are recorded rather
than inferred. Independent equation replay verifies all manifest hashes with
maximum probability and metric/gate differences both `0.0`.

The ten replayable payloads total `865,702,024` bytes; the compressed descriptor
cache is `839,966,791` bytes. Retain this one formal unchanged because exact
four-condition equation replay depends on that cache. Current-best checkpoint,
full-pipeline command, and update-history hashes stay
`1f49d577...2677`, `36b9aa1a...0faf`, and `39bd2879...8f53`.

Focused tests pass `10/10`, and the complete repository suite passes
`1571/1571` with 276 non-failing warnings. The post-closure read-only retention
audit covers 783 run directories and all 50 valid compaction manifests. All 219
manifest-derived originals remain absent, `deleted_anything=false`, and
`blockers=[]`; its summary SHA-256 is `a6e7dcf4...ca21`.
