# TRKH Pair-Surface DDF A0 v2 Prospective Erratum — 2026-07-29

## Status and scope

This erratum is prospective. It was written before a v2 engine was frozen and
before any v2 formal run, replay, candidate score, validation/test result or GPU
result existed. It resolves contradictions or unspecified implementation
choices in
`TRKH_5CLASS_PAIR_SURFACE_DDF_A0_V2_PROTOCOL_20260729.md`. Every clause not
explicitly replaced below remains unchanged. Future machine locks and
authorizations must bind the base protocol, the v2R2 fold provenance erratum,
the effective v2R2 fold manifest and this document together.

## 1. Held-label firewall and donor assignment

The cross-fold capacity-balanced donor cost is target-blind. Remove
`focus-vs-rival target-status mismatch` and `exact target mismatch` from the
assignment cost. In descending lexicographic priority, the permitted edge costs
are now only:

1. valid-mask XOR counts at 32x32 and 16x16;
2. 16x16 bbox-and-valid-mask XOR count;
3. keeper strongest-rival mismatch;
4. bbox-area rank distance;
5. valid-fraction rank distance.

The future machine lock must freeze integer coefficients whose dominance is
proved against the maximum possible aggregate lower-priority cost, rectangular
donor capacities, the min-cost construction and its deterministic optimal-tie
break. Candidate scores and every calibration/held target are forbidden inputs.
The complete mapping and its SHA are frozen before any candidate output.

For each outer fold, image-model weights and normalization state first freeze;
calibration scores may then be paired with calibration targets to fit the
declared calibrators and action threshold. All clean and same-weight held
outputs are produced with the already frozen donor/offset mappings and without
opening held targets. Held suppression decisions and deterministic keeper-rival
replacements are then frozen using only the calibrated probability, threshold
and keeper probabilities/prediction. Only after those states, outputs,
calibrators, threshold and label-blind actions are immutable may held targets be
opened for metric attribution, corrections, harms, neutral changes and post-hoc
donor target-mismatch telemetry. That telemetry can never change a donor.

## 2. Exact replay versus operational telemetry

Replay equality is partitioned explicitly.

`exact_scientific_payload` uses byte equality or declared `max_abs=0` and
includes canonical input identities, fold/group/order/donor/offset/noise/
bootstrap arrays, initialized/final model and optimizer state, masked-BN
sufficient statistics and buffers, per-step loss/gradient/state hashes, raw and
calibrated scores, scaler/logistic states, thresholds, causal outputs, map/filter
statistics, metrics, gates, actions, and the exact shared/replay-only ledger
projections defined in Sections 6 and 8.

`structural_resource_telemetry` includes wall-clock durations, timestamps,
process IDs, temporary absolute paths, peak RSS/CUDA observations, utilization,
filesystem mtimes and import timing. Formal and replay must have the same
telemetry schema, environment/machine identity and pass/fail checks, and each
must remain within every resource ceiling; the observed numeric values and PIDs
need not equal. Operational telemetry is excluded from scientific payload
hashes and cannot affect a metric, gate, threshold or action. The replay report
must enumerate every excluded field and its comparison rule; unlisted numeric
or discrete fields default to exact comparison.

## 3. Exact aggregation for clean gates 12 and 13

Gate 12 is evaluated on the primary `ddf_full` clean pooled out-of-fold held
outputs only. For each named task independently (`union`, `1-vs-0`, `1-vs-2`,
`1-vs-4`), take the pooled held rows on which that task is active and whose
`bbox16 AND valid16` is non-empty. Each task independently must satisfy mean
bbox mass at least 0.85, at least 90% of rows with bbox mass at least 0.70, mean
geometry-normalized lift at least 0.50, and at least 90% of rows with lift at
least 0.20. No fold average, task average or duplicated global positive pool may
replace those four task-specific checks.

For each of the three pair heads independently, normalized-attention-entropy,
attention-CV and evidence-dispersion thresholds use all pooled held active rows,
and every such row must have `sum(valid16)>1`; a row at or below one fails the
formal rather than being excluded. Bbox-unusable rows remain included for these
non-bbox statistics. Class-1 pair-attention cosine and centered-evidence Pearson use all
pooled held target-1 rows, separately for each of the three unordered pair-head
combinations. Invalid attention mass must be exact zero for all four tasks on
all primary clean held rows. Static, single-factor and repeat-role map
statistics are descriptive and cannot satisfy gate 12.

Gate 13 uses primary `ddf_full` clean held filters. All 20 combinations of five
outer folds, two DDF blocks and two factors must independently meet dispersion
at least `1e-4`; pooling folds or factors is forbidden. Other roles are reported
descriptively.

## 4. Invalid-fill RNG contract

Invalid-fill noise uses a generator distinct from the bootstrap and every
training/order generator:

```text
rng = numpy.random.Generator(numpy.random.PCG64(20260729))
noise = rng.integers(
    0, 256,
    size=(763, 3, 256, 256),
    dtype=numpy.uint8,
    endpoint=False,
)
```

This is exactly one `integers` call in ascending canonical `sample_index` row
order and NCHW layout. Its canonical dtype+shape+bytes array SHA-256 is
`9c18e8ab55f6bbf15d1372876bf2578cd094f2a3f1a755e89992c086bf365596`.
The complete NPY and file/array hashes are frozen before candidate scores. Valid
pixels always retain their cached RGB; only invalid pixels select either
literal uint8 zero or the corresponding noise value. The two variants then run
independently through the same preparation path and must match exactly.

Every v2 canonical array hash encodes dimensions as explicit little-endian
signed int64 (`<i8`) before contiguous payload bytes; platform-native integer
byte order is not a permitted alternative. Frozen r1/r2 fold hashes remain
immutable and equivalent on their pinned little-endian Windows host; this
clarification governs new engine/lock/formal payloads.

## 5. Non-wrap displacement convention

An offset is `(dy,dx)` in rows/columns and moves source content to its
destination:

```text
destination[y + dy, x + dx] = source[y, x]
```

when both coordinates are in bounds. Thus `dy=-1` moves the field up and
`dx=+1` moves it right. For dimension length `L` and displacement `d`, the
source slice is `[max(0,-d), min(L,L-d))` and the destination slice is that
source slice plus `d`. Every destination cell not written by this copy receives
the exact all-ones multiplicative neutral; there is no wrap, reflection or
interpolation. The victim validity mask remains authoritative after the shift.
The machine lock freezes the complete offset mapping and an executable slice
oracle before scores.

## 6. Seed, input constants, bootstrap and import canonicalization

The parameter seed phrase is
`UTF8(f"{decimal_seed}:{canonical_parameter_name}")`, where `decimal_seed` is
base-10 with no sign, whitespace or leading zero and the parameter name is the
exact canonical `named_parameters()` name.
Take the first eight SHA-256 digest bytes as unsigned big-endian and clear the
top bit with `& ((1<<63)-1)`. No platform-native byte order is permitted.

ImageNet mean/std decimal constants are cast directly to FP32 in RGB order.
Their required IEEE-754 uint32 bit patterns are respectively
`3ef851ec/3ee978d5/3ecfdf3b` and
`3e6a7efa/3e656042/3e666666`.

For each component-bootstrap draw row, visit its 158 draw columns from left to
right; append every selected component's rows in ascending `sample_index`, and
retain repeats as repeated observations. No deduplication or implicit
component/row reweighting is allowed. Candidate and control metrics use the
identical concatenated multiset. The canonical C-contiguous int64 draw matrix
and its NPY/file/array hashes freeze before candidate scores.

The launcher sets all seven numerical thread variables from the base protocol,
plus `PYTHONDONTWRITEBYTECODE=1`, `PYTHONHASHSEED=0` and
`PYTHONNOUSERSITE=1`; it removes `PYTHONPATH` and `PYTHONHOME`, then starts
Python with `-S -B`. `-I`/`-E` are forbidden because they would ignore the
locked hash seed. The interpreter opening its entry script, the exact pinned
launcher/guard sources and their documented stdlib-only bootstrap path precede
the hook and are recorded as a finite unaudited bootstrap boundary; no
implementation may claim the hook observed interpreter startup. No third-party
or candidate module may enter that boundary.

The pinned guard installs the process audit hook before adding exact pinned
environment/repository paths or importing NumPy, SciPy, scikit-learn, PyTorch,
threadpoolctl, ONNX or ONNX Runtime. A lock-time import probe under the same
launch freezes permitted Python dependency files and extension modules. The
formal/replay hook checks `open`/`os.open`, Python import/dynamic-library audit
events, child-process events and filesystem mutations including create/mkdir,
rename/replace, remove/unlink and rmdir. Because a Python audit hook cannot
guarantee observation of every transitive Windows native-loader action, the
lock also freezes canonical path-sorted before/after loaded-module sets with
path, bytes/SHA
where readable, and signer/version identity where hashing is unavailable; any
unapproved delta fails. This is evidence for the pinned non-adversarial process,
not an OS sandbox. The authorization enumerates exact files and exact output
operations, never permissive input roots.

Canonical ledger paths replace pinned prefixes with `$PYTHON`, `$ENV`, `$REPO`,
`$CACHE` and `$RUN_OUT`; formal and replay output prefixes both map to
`$RUN_OUT`. Canonical event order contains path token, access kind and normalized
mode/flags, but not timestamps or PIDs. Formal and replay have distinct role
sets. Their projection onto shared scientific inputs/imports must be exact. The
replay additionally reads, in a separately frozen exact order, a finalized
formal handoff manifest and only the hash-listed formal evidence needed for
comparison; these replay-only events cannot be incorrectly required in the
formal ledger. Every replay-only byte is hash-chained to the original lock and
authorization. Child-process launch and every unlisted read, write or mutation
fail closed.

## 7. Shifted-XAI bbox geometry

This clause applies only after clean A0 passes. Translations start from cached
256x256 RGB and `valid256`; neither `valid64/32/16` nor an already rasterized
bbox mask may be shifted. For `(dy,dx)` in raw pixels, move RGB and validity by
the exact non-wrap slice rule above with zero/false fill. Convert the original
continuous normalized bbox to 256-space corner coordinates, add `dx` to both
x corners and `dy` to both y corners, clip the four corners once to `[0,256]`,
then rasterize on 16x16 with the base protocol's floor-left/top and
ceil-right/bottom rule. Intersect with the validity masks recomputed from the
shifted `valid256`. If clipped continuous area or the final intersection is
empty, record `bbox_supervision_valid=false` and null bbox-region metrics; no
minimum-cell fallback, roll or fabricated box is permitted.

## 8. Atomic failure evidence

The authorization freezes schemas for completed PASS, completed scientific
FAIL, handled harness exception and detected incomplete temporary output. A
handled exception tombstone includes authorization/lock/engine hashes, phase,
exception type, normalized message and traceback hash, final canonical ledger
prefix, and a manifest of surviving owned files. Timestamps/PIDs are structural
telemetry. A hard crash may leave only a temporary directory; the next launch
must detect it, refuse authorization reuse and require a separately reviewed
recovery record. No implementation may claim that two per-file atomic replaces
or an unhandled process death form an atomic multi-file transaction.

On formal finalization, first hash payload files into a canonical payload
manifest and root. Next write `replay_handoff.json`, containing the original
authorization/lock hashes, that payload root, exact reference-file bytes/SHA
records and the ordered replay-only read contract. Finally an outer directory
manifest covers both the payload manifest/files and handoff. The handoff never
embeds or hashes the outer manifest/root, so the chain is acyclic. Replay is
allowed to read formal evidence only through this handoff chain; it cannot
broadly allow a directory from its name. Shared-input ledger projections,
replay-only reference events and output-event projections are compared
separately under their declared exact rules.

## 9. Updated novelty boundary

This targeted literature check is non-exhaustive and cannot prove novelty. It
was refreshed before implementation. Masked and
renormalized convolution over valid pixels is established by Partial
Convolutions (ECCV 2018):

<https://openaccess.thecvf.com/content_ECCV_2018/html/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.html>

Counterfactual intervention on learned attention for fine-grained recognition
is established by Counterfactual Attention Learning (ICCV 2021):

<https://openaccess.thecvf.com/content/ICCV2021/html/Rao_Counterfactual_Attention_Learning_for_Fine-Grained_Visual_Categorization_and_Re-Identification_ICCV_2021_paper.html>

Pair/example selection for fine-grained tasks is also an active prior-art area,
including Task-Aligned Context Selection (CVPR 2026):

<https://openaccess.thecvf.com/content/CVPR2026/html/Guo_Learning_What_Helps_Task-Aligned_Context_Selection_for_Vision_Tasks_CVPR_2026_paper.html>

Therefore v2 claims no novelty for validity masking/renormalization,
counterfactual attention, pair selection, pairwise learning or DDF operators in
isolation. The only prospective model contribution is their specific
scratch-only, pair-evidence, validity-aware composition for the TRKH mango
boundary, plus target-blind filter attribution and the reproducibility protocol.
Even that composition/application claim remains unproven until A0, matched
integration and a newly sealed cohort pass. A negative A0 is a protocol and
mechanism result, not a new-model accuracy claim.

## Authorization state

This erratum resolves a prospective specification boundary only. It does not
authorize cached-input access, a formal run, replay, validation/test access or
GPU execution. The no-repeat registry must be rebuilt or receive a pinned
addendum that includes v2R2 and this erratum before any authorization can become
effective.
