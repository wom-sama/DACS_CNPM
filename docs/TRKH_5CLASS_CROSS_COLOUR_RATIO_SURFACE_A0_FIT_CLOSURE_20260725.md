# TRKH Cross Colour Ratio Surface A0 Fit Closure

Date: 2026-07-25
Protocol ID: `trkh_cross_colour_ratio_surface_a0_20260725`
Final decision: `reject_exact_ccr_surface_a0`
Value verdict: `negative-but-reusable`

## Prospective Boundary

- Protocol, lock, equation engine, and same-tensor erratum were fixed before
  candidate metrics. Their SHAs are
  `1e408502...f347`, `6c2a604d...c835`, `ac610d7c...d26fe9`, and
  `f9a901b1...d03b`.
- The exact 763-row/735-image train-only cohort, five source-held folds, four
  scratch roles, 20 epochs, 3,004-parameter head, initialization, order,
  optimizer, calibration rule, controls, thresholds, clean gate, resource
  ceilings, and replay tolerance were locked prospectively.
- The first authorization was consumed by a deterministic-cuBLAS harness
  failure before a candidate metric. Failure evidence SHA is
  `51a58928...983ccf`. The harness-only correction was pushed at
  `0218379d...ef0b71`; the replacement authorization SHA
  `b7cda19b...a36770` was pushed at `54c3e5c3...748df`.
- Formal execution opened only the exact materialized cache, CIDT clean
  train CSV, and locked immutable inputs. It read no raw image/label,
  validation, or test path.

## Formal Result

The replacement formal completed in `190.493 s`. Every engineering,
authorization, resource, access-ledger, and worker-process check passed, but
the complete prospective clean scientific gate failed.

| Measure | Keeper margin | Plain colour ratio | CCR candidate | Trained spatial dephase |
|---|---:|---:|---:|---:|
| AUROC | `0.816697` | `0.528151` | `0.535520` | `0.503797` |
| AUPRC | `0.921070` | `0.755351` | `0.757140` | `0.716772` |
| Restricted FP removed | N/A | `7/222` | `2/222` | `3/222` |
| Class-1 keeper TP retained | N/A | `0.952652` | `0.943182` | `0.962121` |
| Corrections / harms | N/A | `7 / 25` | `2 / 30` | `3 / 20` |
| Full macro-F1 gain | N/A | `-0.004624` | `-0.006738` | `-0.004123` |
| Full class-1 F1 gain | N/A | `-0.018928` | `-0.026845` | `-0.016702` |

CCR improved AUROC over the equal-size plain colour-ratio control by only
`0.007369`, below the locked `0.02` requirement. It remained `0.281178`
below the keeper margin and exceeded the separately trained spatial-dephase
control by only `0.031723`, below the locked `0.04` requirement. Fold AUROCs
were `0.412589`, `0.526957`, `0.538261`, `0.565161`, and `0.642572`; the
candidate beat both learned controls in only `2/5` folds instead of `4/5`.

Pairwise AUROC was also insufficient:

- class 1 versus 0: `0.599839`;
- class 1 versus 2: `0.462723`;
- class 1 versus 4: `0.396858`.

The candidate changed 32 full-train predictions, corrected two restricted
false positives, and harmed 30 class-1 true positives. It rejected only
`1/158`, `1/54`, and `0/10` restricted false positives from targets 0, 2,
and 4. Keeper true-positive retention was `498/528 = 0.943182`; 12 of 13
keeper false negatives had descriptor support, so lack of support does not
explain the failure.

The full-train diagnostic baseline had macro-F1 `0.939876` and class-1
precision/recall/F1 `0.700265/0.975970/0.815444`. Applying CCR actions
reduced these to `0.933138` and
`0.689751/0.920518/0.788599`. This is the opposite of the required
class-1 precision improvement.

## Controls And Interpretation

The independent seed repeat was stable but still scientifically negative:
AUROC `0.537668`, candidate-repeat difference `0.002148`, and action
agreement `0.980341`. It changed 23 rows, produced three corrections and 20
harms, and reduced macro/class-1 F1 by `0.004166/0.016702`.

The same-weight spatial dephase lowered candidate AUROC by `0.036136`, so
the head used spatial layout. That control nevertheless changed 444 rows,
produced 121 corrections and 314 harms, retained only `0.405303` of keeper
class-1 true positives, and reduced macro/class-1 F1 by
`0.071218/0.312506`. Spatial sensitivity therefore does not establish
class-discriminative or action-safe surface evidence.

All four roles trained normally. Every fold-role used 20 epochs and 160
updates with four workers, finite gradients, all 3,004 parameters changed,
and FP32 optimizer state. Candidate mean loss fell from `1.327921` at epoch
0 to `0.799995` at epoch 19. The result is not an optimizer-no-op failure;
the learned representation does not separate the difficult class pairs.

## Replay, XAI, And Downstream Boundary

Fresh-process replay completed in `188.756 s` under a different process ID.
All 180 state arrays and 46 output arrays were exact, maximum numeric error
was `0.0`, and metrics, training science, authorization, descriptor records,
and access ledger reproduced exactly. Validation and test remained unused.

XAI was not run. The prospective protocol authorizes XAI only after the
complete clean information conjunction passes; running it after this failure
would spend resources on a mechanism already barred from integration.
Shifted conditions, deployment, production integration, validation, test,
probe, full train, and current-best command update remain unauthorized.

## Resource And Access Evidence

- Peak CUDA allocation: `366,425,600` bytes, below `2,147,483,648`.
- Peak process RSS: `2,416,246,784` bytes, below `10,737,418,240`.
- Maximum combined temporary storage: `333,692,816` bytes, below
  `429,496,730`.
- Access ledger: 55 observed events, 34 logical opens, nine unique input
  paths, zero blocked/write-like/raw-dataset/validation/test access.
- Unexpected or orphan Python/TensorRT compute children: zero.

These checks establish reproducibility and bounded experiment cost, not a
production inference pass. No matched batch-1 mean/p95 latency, throughput,
end-to-end VRAM, ONNX parity, or TensorRT measurement is authorized because
the clean scientific gate failed.

## Immutable Evidence

- Formal summary/metrics SHAs:
  `86c05016...841be` / `b2617256...d2d7d`.
- Formal/replay manifest SHAs:
  `b358a9d5...d76af` / `e2f8414f...59c39`.
- Replay summary SHA:
  `6a71ed2a...3e22`.
- Held outputs/state/training/ledger SHAs:
  `c2437cf1...eff9`, `e91cd48a...acfd`,
  `4293e18f...35a8`, and `5fafc1c4...2e31`.
- Artifact-set manifest SHA:
  `9f504ab9...a3ba1`.
- Retained artifact root:
  `runs/audit_cross_colour_ratio_surface_a0_20260725`.
- Exact materialized-cache evidence SHA:
  `e992c297...98c3ed`; its local cache remains a protected dependency.

## Decision And No-Repeat Boundary

Reject the exact CCR Surface A0 and keep its evidence because it rules out a
plausible but insufficient signal family. Do not sweep derivative sigma,
log epsilon, channel-ratio order, reliability support, head width/depth,
epochs, learning rate, seed, folds, loss weights, calibration thresholds,
dephase offsets, or direct trainer integration after observing this result.

Reopen only with an equation-distinct, sample-conditional material/surface
mechanism that is prospectively locked to beat the keeper margin and causal
controls for all three `1-vs-0/2/4` boundaries while preserving class-1 true
positives, improving precision and F1, replaying exactly, and exposing a
standard-op inference path.

The current keeper, full-train command, and command history remain unchanged.
