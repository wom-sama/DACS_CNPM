# TRKH 5-Class Hyperspherical Support A0 Closure - 2026-07-21

## Decision

Reject the locked empirical-cap hyperspherical support route before model or
trainer integration. Do not run a validation smoke, XAI, probe, full train, or
test audit, and do not update current-best commands.

All structural gates pass, but 9 of 22 mechanism gates fail. The candidate
provides a stable interpolation between natural and duplicate balancing rather
than a selective class-1 surface representation. It removes class-1 false
positives relative to duplicate oversampling only by breaking too many true
class-1 predictions.

## Provenance And Isolation

- Prospective protocol SHA-256:
  `f9485b7bb0ed22b8ff5d6b4cc2f79af0f766b409d43eef2be3f70518410b280c`.
- Protocol commit: `b28d961`.
- Initial auditor/launcher/tests commit: `f2c45cb`.
- Incomplete-finalization erratum SHA-256:
  `1c19383dfde7c6216220a88a8d7833b07ba8df3c6432642d79580b1c6e6d1258`.
- Erratum/finalizer commits: `8f9ba47` / `4102ca0`.
- Focused tests before formal/finalization: `12/12` then `15/15`.
- Full repository pytest before formal: `1681 passed`, `295 warnings`.
- Input rows/classes/folds: `9215`, `[1941,541,1920,2520,2293]`,
  `[1843,1830,1828,1851,1863]`; all source overlaps are zero.
- Validation/test rows, labels, predictions, pixels, and metrics remain
  unopened. No raw-data, model, trainer, checkpoint, or command file changed.

The first formal process completed the fixed readouts and wrote six payloads,
then stopped before summary serialization because the historical float32
keeper probabilities were passed to the strict float64 normalization check.
Before opening any metric, the six payload names/sizes/hashes and a sole
finalization procedure were locked. Finalization did not call support
generation or logistic fitting, did not rewrite the six files, normalized only
the non-gating keeper reference, and replayed the existing predictions with
maximum numeric difference `0.0`.

The persisted NPZ probabilities are float32; the loader promotes them to
float64 before the normalization helper, so the summary's `source_dtype` field
describes the helper input, not the on-disk dtype. The observed pre-normalizing
row-sum error is only `1.1920928955078125e-7`, below the locked `2e-4` cache
tolerance, and the keeper reference never enters a candidate gate.

## Source-Held Train Results

These are source-held readout metrics over frozen embeddings from a keeper
trained on all train rows. They are an information gate, not encoder-OOF or
validation evidence.

| Role | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | Predicted class-1 | NLL |
|---|---:|---:|---:|---:|---:|---:|
| Natural control | 0.935540 | 0.809091 | 0.822551 | 0.815765 | 550 | 0.131364 |
| Duplicate control | 0.935643 | 0.723164 | 0.946396 | 0.819856 | 708 | 0.136741 |
| Cap candidate | 0.936758 | 0.755796 | 0.903882 | 0.823232 | 647 | 0.129019 |
| Cap seed repeat | 0.936659 | 0.755796 | 0.903882 | 0.823232 | 647 | 0.129021 |
| Nearest-center deranged | 0.723816 | 0.067308 | 0.103512 | 0.081573 | 832 | 0.319156 |

Candidate minus duplicate:

- macro F1: `+0.001115`, below the required `+0.002`;
- class-1 precision: `+0.032632`;
- class-1 recall: `-0.042514`;
- class-1 F1: `+0.003376`, below the required `+0.010`;
- class-1 TP: `512 -> 489`, net `-23`, versus the allowed `-2`;
- restricted `{0,2,4}->1` FP: `191 -> 153`, net removal `38`;
- corrections/harms: `41/31`;
- class-1 FN rescues/TP breaks: `0/23`.

Candidate minus natural:

- class-1 TP rises `445 -> 489`, but restricted FP rises `101 -> 153`;
- corrections/harms are `44/49`;
- class-1 precision falls `0.809091 -> 0.755796`.

Thus the candidate is not a precision-preserving representation improvement.
It is an intermediate operating point between the high-precision natural
readout and the high-recall duplicate readout.

## Fold And Seed Evidence

- Precision improves over duplicate in all `5/5` held source folds.
- Restricted FP net removal is positive in all `5/5` folds.
- Class-1 TP net changes by fold are `[-9,-2,-6,-1,-5]`; four folds violate
  the allowed per-fold loss of at most one TP.
- Class-1 F1 changes by fold are approximately
  `[-0.01179,+0.01689,-0.01397,+0.02160,+0.00412]`; only three folds improve.
- Primary/repeat prediction agreement is `0.999457`; macro/class-1 F1
  differences are `0.00009884/0.0`. The stable repeat rules out random support
  draws as an explanation and closes seed neighbors.
- The deranged placebo collapses class-1 F1 to `0.081573`, proving class-center
  conditioning is active. That structural activity does not satisfy TP safety.

## Geometry Evidence

- Every class in every fit fold has empirical 0.99 angular radius above the
  separable half-simplex bound, so all 25 primary supports are clipped to
  `0.9117382909684877 rad`.
- No class cap is non-overlapping with even one rival in any fold:
  `nonoverlap_rival_count=0` for all 25 records.
- Class 1's nearest rival is class 0 in all folds, with center cosine
  `0.685338..0.693481`.
- Class-1 minimum cap-separation margin is
  `-1.018989..-1.007748 rad`, far from separable.
- About `1576..1587` synthetic class-1 vectors are generated per fold from only
  `426..438` real fit rows. The method therefore fills a broad overlapping cap
  instead of adding a new class-1-vs-0 surface cue.
- All numerical geometry checks pass: unit norms, cap bounds, tangent
  orthogonality, reconstructed angles, balanced counts, RNG isolation, and all
  25 logistic fits.

## Failed Gates

The nine failed mechanism gates are:

1. macro F1 gain over duplicate `<0.002`;
2. class-1 F1 gain over duplicate `<0.010`;
3. class-1 precision is more than `0.005` below natural;
4. class-1 recall is more than `0.015` below duplicate;
5. aggregate class-1 TP net is below `-2`;
6. TP breaks exceed FN rescues;
7. per-fold TP safety fails;
8. primary aggregate safety fails;
9. repeat aggregate safety fails.

Every structural gate and the remaining 13 mechanism gates pass. No passing
subset can compensate for a failed conjunctive gate.

## Closure Boundary

Close on the current keeper:

- empirical single-cap generation from final embeddings;
- nearby `alpha`, cap-radius, generated-count, seed, logistic `C`, threshold,
  class-weight, or natural/duplicate blend sweeps;
- head-statistic interpolation added post hoc to the same frozen supports;
- a FeatRecon-like projection/SupCon trainer fork justified by this A0;
- any interpretation of FP reduction without accounting for the 23 broken TP.

This result does not claim that every train-time geometric representation is
impossible. It shows that the current final embedding violates the separable
cap premise and that filling those caps does not supply the missing
class-1-vs-0 evidence. A next route must learn a new conditional surface or
boundary representation with explicit TP protection before another GPU smoke;
it must not be another prototype, subcenter, GMM, neighbor, global SupCon, or
support-expansion variant.

## Artifacts And Retention

- Formal directory:
  `runs/audit_hyperspherical_support_a0_20260721`, 8 files, `6.218 MiB`.
- Summary SHA-256:
  `ff203f2dfb350ffb97827dbff704b3c86b4f44e4c37770aa436ca616452e537e`.
- Artifact-manifest SHA-256:
  `0cefb8c9ef1e6db9722dd0c84d2732ec9ddaa30c7ef5df0b6ec73fe76aa09a8d`.
- Canonical second-process replay: `9215` rows, maximum difference `0.0`, all
  seven manifest payloads verified.
- Post-closure read-only retention:
  `runs/audit_trkh_artifact_retention_post_hyperspherical_support_a0_20260721`.
- Retention summary SHA-256:
  `405c210cf93a70b12e5cb0f47b9c4696515ed5a92a1265ad257379de897f5121`.
- Retention covers `805` run directories and 51 object-schema manifests with
  233 expected-absent compacted names, zero remaining originals,
  `deleted_anything=false`, and `blockers=[]`.

Keep the compact A0 and retention evidence. Delete nothing. Current-best
command/history hashes remain exactly
`36b9aa1a...940faf` / `39bd2879...98f53`; update count remains unchanged.
