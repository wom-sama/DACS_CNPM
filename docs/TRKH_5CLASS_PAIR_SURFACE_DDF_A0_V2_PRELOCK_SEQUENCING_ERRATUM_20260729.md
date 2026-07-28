# TRKH 5-Class Pair-Surface DDF A0 v2 Pre-Lock Sequencing Erratum - 2026-07-29

## Status and narrow scope

This is an additive prospective erratum to:

- `TRKH_5CLASS_PAIR_SURFACE_DDF_A0_V2_PROTOCOL_20260729.md`;
- `TRKH_5CLASS_PAIR_SURFACE_DDF_A0_V2_PROTOCOL_ERRATUM_20260729.md`;
- `TRKH_5CLASS_PAIR_SURFACE_DDF_V2R2_FOLD_ERRATUM_20260729.md`; and
- the effective v2R2 fold manifest.

It resolves only the circular pre-machine-lock execution sequence. It does not
authorize a formal run, model training, validation, test, conditional XAI,
production integration or a scientific-result retry. Except where explicitly
replaced below, every earlier requirement remains unchanged. Every later
contract, authorization and machine lock must bind all four earlier documents,
the effective v2R2 fold manifest and this document by exact bytes and SHA-256.

## The circularity being corrected

The base protocol forbids GPU work until a machine lock is committed and
pushed, while that machine lock must freeze TensorRT build, FP16 latency and
CUDA-allocation evidence. Those engineering facts require a GPU execution.

The machine lock must also freeze validity, donor, order, random, task-weight
and offset artifacts before candidate scores. Most values can be reconstructed
from frozen code and manifests, but the complete donor mapping cannot. It needs
per-row train-cache validity masks, bbox geometry and historical keeper
probabilities. A SHA-256 commitment verifies bytes that are already present; it
does not reconstruct the committed per-row values.

Therefore neither broad unledgered cache access nor an already complete machine
lock can legitimately create all prerequisites of that machine lock. Two
separate pre-lock phases are required and are the only exceptions introduced by
this erratum:

1. `S1_SYNTHETIC_ONLY_GPU`; then
2. `S2_SCORE_INDEPENDENT_DERIVATION_CPU`.

S2 cannot become effective unless S1 has completed PASS, completed a fresh
replay and had its finalized result identity committed and pushed. Neither
phase is a formal run or a candidate observation.

## Common pre-lock authority boundary

Each phase requires its own exact contract JSON/SHA pair and its own inert
pending authorization JSON/SHA pair. The implementation, guard, launcher,
auditor, replay code, tests, protocol set and no-repeat registry must already be
committed and pushed before the corresponding pending authorization is built.
The pending authorization becomes effective only after a later, separate
commit containing that authorization pair is pushed and all frozen-boundary
checks pass.

The two authorizations are distinct, non-transferable and non-renewable. Each
has exactly one primary process and one fresh-process replay. Claiming a phase
consumes its primary quota before the first GPU or cache operation. Claiming
its replay consumes the replay quota before the first replay-only evidence
read. An authorization ID, claim, output root, handoff or consumption record
from one phase can never satisfy the other phase.

The combined quotas are exactly:

| Operation | S1 | S2 |
|---|---:|---:|
| Synthetic-only primary GPU process | 1 | 0 |
| Synthetic-only fresh replay | 1 | 0 |
| Score-independent primary CPU derivation | 0 | 1 |
| Score-independent fresh CPU replay | 0 | 1 |
| Formal runs | 0 | 0 |
| Candidate training runs | 0 | 0 |
| Validation runs | 0 | 0 |
| Test runs | 0 | 0 |
| Conditional XAI runs | 0 | 0 |

No full-train, formal, validation, test or production command can be inferred
from either pre-lock authorization. A machine lock built from their evidence
still authorizes zero executions.

Both phases use the corrected `-S -B` launcher, fixed pre-import environment,
dependency probe, exact-path access ledger, loaded-module inventory, no-child-
process rule and canonical path projections from the protocol erratum. A
phase-specific allowlist replaces any permissive directory read. Every input,
output and mutation not explicitly listed for that phase fails closed.

## S1 - synthetic-only GPU qualification

S1 exists only to close the synthetic engineering gates on the exact machine
that a later machine lock will name. Its input allowlist contains only the
frozen protocol/errata, source, tests, dependency files, authorization files
and deterministic synthetic arrays generated from locked constants and seeds.
The following are absent from the allowlist and are forbidden:

- every train-cache or cohort artifact;
- every raw image, raw label and dataset split path;
- all train, validation and test metadata;
- all keeper or external checkpoints, features and predictions;
- every candidate or historical score array; and
- every formal, audit-review or production output.

S1 may instantiate the exact scratch v2 roles, execute synthetic forward and
backward equations, perform the protocol's finite synthetic parameter-update
oracle, export/check ONNX, build the TensorRT engine, run the locked FP16 timing
workload and measure CUDA allocation. It may not fit on cohort data, select a
hyperparameter, calibrate a score, choose an action threshold or report AUROC,
AUPRC, precision, recall, F1 or any other candidate metric.

Before either S1 execution, the committed contract prospectively pins the
expected GPU name and UUID, driver, CUDA, cuDNN, TensorRT, Python and complete
dependency-lock identities. It also pins the exact TensorRT builder flags,
workspace/profile shapes, precision policy, tactic/timing-cache policy, warm-up
count, measured-iteration count, synchronization points, timer source, batch
shape, input construction and latency/resource ceilings. A value first learned
from primary telemetry cannot be promoted after the run into an expected
machine or timing contract. Primary and replay must each prove the pinned
environment before creating a TensorRT build or recording a timing sample.

The primary and replay independently prove all synthetic engineering gates in
the base protocol. Deterministic inputs, tensors, gradients, updates, ONNX
bytes, build configuration, timing workload, numerical/parity outputs, gate
decisions and scientific manifests are exact. Timing samples, wall-clock
durations, utilization and peak-resource observations are structural
telemetry: machine/schema/workload identities are exact, every observation
must remain within its frozen ceiling, and the PASS decision must agree, but
the measured values need not be byte-identical. Independently built TensorRT
plan bytes may also differ because backend tactic selection is not claimed to
be byte-deterministic under this protocol. Both plan hashes and byte counts are
retained as explicit backend-artifact identities and excluded from the
scientific replay root; each plan must independently satisfy the identical
locked numerical, parity, workload and PASS checks. No other field receives a
structural exception.

S1 emits a canonical scientific payload manifest, a replay handoff and an
acyclic outer finalization manifest. The S1 evidence root includes exact GPU
name/UUID, driver, CUDA, cuDNN, ONNX, ONNX Runtime and TensorRT versions, all
synthetic workload identities, raw timing vectors, summaries, resource traces,
serialized-engine identity, ledgers and replay comparison. Both primary and
replay must PASS. An S1 FAIL or incomplete authorization blocks S2 and the
machine lock.

## S2 - CPU score-independent derivation

S2 can start only from an exact committed-and-pushed S1 PASS/replay root. S2 is
CPU-only: CUDA initialization, GPU allocation and TensorRT execution are
forbidden. Its initial read allowlist contains only exact committed source and
protocol artifacts, the S1 finalization chain, target-free canonical
projections of the effective v2R2 fold manifest and pinned geometry audit, and
the exact train-only cache containers needed for metadata derivation. The fold
projection contains row `sample_index/component_sha256/fold`, component order,
fold/calibration mapping and source hashes but no target or target count. The
geometry projection contains input identities, mask/bbox definitions and
hashes, and bbox-unusable sample indices but no target. Both projections are
separately committed and bind the full source artifacts; the full
target-bearing artifacts remain deferred until S2B. S2 cannot open raw images,
raw label files, validation, test, a model checkpoint, an external model
artifact or any new or historical candidate score other than the already
frozen keeper probabilities explicitly permitted as a target-blind covariate.

S2 may instantiate a scratch model on CPU only to enumerate canonical
parameter names/shapes and reproduce deterministic initialization identities.
It may not call a candidate forward method, create or step an optimizer,
compute a candidate loss or gradient, load a checkpoint, fit a calibrator, or
produce a candidate prediction or score.

The process audit hook and member-level archive guard are active before the
first train-cache open. Because one NPZ file can contain both permitted
target-blind members and deferred target members, file-open logging alone is
insufficient. The pinned archive reader additionally records and enforces the
exact member name, dtype, shape, uncompressed byte identity, read order and
phase. Merely opening the container does not authorize every member.

### S2A - target-blind derivation and inner finalization

Before any target array or target-bearing metadata member is opened, S2A reads
only the exact sample identities, model-box geometry, validity packbits and
historical keeper probabilities needed below. It derives and retains the full
preimages for:

- `valid64`, `valid32`, `valid16`, `bbox16` and `bbox16 AND valid16`;
- bbox usability and the twelve unusable sample-index records without
  consulting a target value;
- keeper strongest-rival identities;
- bbox-area and valid-fraction ranks;
- all five cross-fold donor cost tensors, coefficients, capacities, mappings,
  per-edge selected costs, histograms and target-blind mismatch telemetry;
- all ten fold/block non-wrap offset mappings and executable slice-oracle
  evidence;
- every role/fold seed, full epoch-order array, parameter-order identity,
  deterministic initial-state identity and matching-primary tensor identity;
- the complete invalid-fill noise and component-bootstrap draw arrays; and
- the score-independent formula schemas, with no XAI row ID exposed.

The target-blind artifacts are written into a one-shot temporary root, hashed
into a canonical inner payload manifest/root and an acyclic
`target_blind_finalization.json`, and then frozen. This is an internal phase
transition, not the replay handoff for the complete S2 result. After this inner
root is finalized, none of its files may be changed, replaced or omitted. The
ledger may expose the separately listed target metadata for S2B only after
verifying that inner root and recording the state transition. Donor, validity,
rank and offset outputs can never depend on, or change after, that transition.

### Exact rank and donor assignment contract

Canonical row lists use ascending `sample_index`, but ranks are constructed
separately inside the held-victim set and calibration-donor set of each outer
fold. For each set define:

- bbox-area ordinal order by ascending
  `(sum(bbox16 AND valid16), sample_index)`; and
- valid-fraction ordinal order by ascending `(sum(valid16), sample_index)`.

The zero-based position in each order is the corresponding rank, so each rank
vector is a permutation of `0..n_set-1`. The sample index is the exact tie
break; no average/dense rank or floating perturbation is permitted. A
bbox-unusable row has `sum(bbox16 AND valid16)=0` and remains in the bbox-rank
order. Division by 256 is not performed for valid-fraction ranking because it
would not change the order. These covariates are recomputed per outer-fold
problem; when recorded per role, every role uses the identical mapping for the
same outer fold.

Keeper strongest rival is `argmax` over classes `(0,2,4)` in that order, so an
exact tie chooses the lower class index. No target enters this value.

For outer held fold `h`, victims are its rows in ascending sample index and
donors are calibration fold `(h+1) mod 5` in ascending sample index. For each
victim/donor edge the six descending lexicographic costs are exactly:

1. Boolean XOR count of `valid32`;
2. Boolean XOR count of `valid16`;
3. Boolean XOR count of `bbox16 AND valid16`;
4. keeper strongest-rival mismatch, in `{0,1}`;
5. absolute ordinal bbox-area-rank distance; and
6. absolute ordinal valid-fraction-rank distance.

The two validity resolutions are separate priorities; they are not summed into
one opaque cost. Let `n_h` be the victim count and `M[h,k]` the maximum
single-edge value of cost `k` in that fold's complete rectangular cost tensor.
Set `A[h,5]=1` and, for `k=4,...,0`, set

```text
A[h,k] = 1 + n_h * sum_{j=k+1..5}(M[h,j] * A[h,j])
```

All arithmetic is exact unbounded integer arithmetic. The retained dominance
proof must show that a one-unit improvement at priority `k` exceeds the maximum
possible aggregate contribution of every lower priority.

If the donor count is `d_h`, each donor usage lies in
`L=floor(n_h/d_h)` through `U=ceil(n_h/d_h)`, and total usage is `n_h`.
Equivalently, create `L` mandatory slots for every donor and, only when
`U>L`, one optional slot per donor; select `n_h` distinct slots and every
mandatory slot. Among feasible assignments, minimize the six aggregate costs
lexicographically. If several assignments have the same aggregate cost vector,
minimize the donor-position vector in victim order lexicographically, where a
donor position is its zero-based index in the ascending calibration-donor
list. An exact equivalent scalar encoding may use base `d_h+1`; floating
perturbations and solver-dependent ties are forbidden.

Every fold artifact retains the victim list, donor list, six complete cost
matrices, maxima, coefficients, dominance proof, numeric lower/upper capacity,
selected donor position and sample index for every victim, each selected
six-cost vector, aggregate vector, donor-use histogram, zero same-component
pairs, zero fixed points and all post-target mismatch fields as null. A digest
without these preimages does not satisfy the lock.

Every offset artifact similarly retains each `(sample_index, outer_fold,
block, dy, dx)` row, not only a mapping hash. Every random/order/weight artifact
retains its canonical NPY or canonical JSON array plus file and canonical-array
hashes. Hash-only placeholders and self-declared `automatic_passed` booleans
cannot replace executable validation of their preimages.

### S2B - target-metadata-only completion

Only after S2A inner finalization may the archive guard expose the exact
train-only target metadata and full target-bearing fold/geometry artifacts
authorized for S2B. S2B may use them only to:

- reproduce role/fold task and bbox-support counts and exact fixed weights;
- prove that the locked 2,000x158 component-bootstrap draw matrix has zero
  one-class replicates;
- attach the frozen bbox-unusable target records already anticipated by the
  geometry audit;
- expose and validate the exact predeclared target-stratified XAI row IDs,
  which remain unavailable to every donor/rank/offset derivation; and
- complete target-bearing schemas needed later for fold-safe metric
  attribution.

S2B cannot recompute, choose or alter a donor, rank, validity mask, offset,
random draw, predeclared XAI row or candidate-independent action/gate formula.
It cannot fit any learned state. Target-member access, derived support records
and the unchanged S2A root are all retained in the ordered ledger.

The task-weight preimage includes `N_fit`, positive/negative support, usable
bbox positive/negative support and all four FP32 weight bit patterns for every
outer fold and task. The validator independently recomputes
`N_fit/(2*N_support)` exactly as cast by the frozen engine; a positive finite
number or formula string alone is insufficient.

S2B additionally emits immutable, per-outer-fold target shards with canonical
row identities and arrays: `fit_targets`, `calibration_targets` and
`held_targets`. It also emits a separately deferred CIDT target/baseline
projection. These are distinct scientific artifacts, not aliases for one
all-row target member. Their manifests bind the source cache/member identities,
sample-index order, fold role, dtype, shape, bytes and canonical-array hash.
No later phase receives general permission to read the original all-row target
member merely because its container or SHA-256 was validated during S2B.

The target-stratified XAI-ID artifact is likewise hash-bound at S2B but remains
content-unreadable to candidate training, normalization recalibration,
calibrator fitting, threshold selection and every label-blind action. Its IDs
may be released only after clean/causal outputs, held-target attribution and
all clean gate decisions are immutable, either for end-of-clean retention or
inside the separately authorized conditional-XAI phase. Its membership cannot
select, tune or retry any clean scientific state.

S2 finalization covers the immutable S2A inner root, every S2B artifact, the
complete input/member ledger, payload manifest, replay handoff and acyclic
outer directory manifest. The fresh S2 replay independently repeats the S2A
then S2B state transitions and must reproduce every scientific NPY/JSON byte,
mapping, order, decision and canonical ledger projection exactly. Structural
telemetry has only the exclusions already listed in the protocol erratum.

## Full-preimage and provenance requirement

The S1 and S2 scientific manifests contain, for every source and output, a
canonical logical path, byte count, SHA-256, access class and the
authorization/contract/source lineage. Every scientific output is named below
the literal logical prefix `$RUN_OUT`; therefore primary and replay can have
different physical output roots while comparing the same canonical payload.
Absolute primary/replay output paths are retained only in an outer operational
telemetry record and are excluded from the scientific replay root. Input paths
remain exact frozen source identities. Canonical array records additionally
contain dtype, shape, explicit little-endian dimension encoding, array byte
count and canonical array hash. The finalized roots retain the data needed to
reproduce and verify a mapping; they do not retain only an unverifiable claimed
digest.

The original cache records in S2 become the authoritative provenance records
for later machine-lock construction. A machine-lock builder validates the S1
and S2 authorization chains, finalization manifests, replay comparisons and
derived preimages. The machine-lock builder, authorization issuer and their
check-only modes do not reopen or rehash the original train cache at all; only
the finalized S2 ledger supplies those prospective identity records. In
particular, runtime structural validation of a pending formal authorization
cannot hash every immutable artifact before installing the formal ledger.

The later formal process receives the exact cache and member identities
recorded by S2. On the first authorized formal container open, the formal
ledger independently checks path, bytes and SHA-256, but this never releases
every member to scientific code. The member-level guard permits only the
current phase and fold. `fit_targets` may open for training that fold.
`calibration_targets` may open only after candidate weights/BN state and the
corresponding raw calibration scores have been frozen. `held_targets` and the
CIDT target/baseline projection may open only after clean/causal outputs,
calibrator, action threshold and all label-blind actions have been frozen. A
phase transition is ledgered before each release. The XAI-ID artifact remains
hash-only until the later gate described above, and the original all-row target
member remains unreadable throughout formal/replay. Lineage-only checkpoints
remain unreadable at runtime; their historical digest is not reverified by
opening the checkpoint.

## Machine-lock transition

A v2 machine lock can be built only after all of the following are separately
committed, reviewed and pushed:

1. the code/protocol/registry boundary;
2. the S1 contract and authorization;
3. S1 primary PASS, fresh replay PASS and finalization root;
4. the S2 contract and authorization; and
5. S2 primary PASS, fresh replay PASS and finalization root.

The machine lock binds exact identities for both contracts, authorizations,
claims, ledgers, handoffs, replay comparisons and final roots. It binds the
complete derived artifact set and the S1 machine/environment identity. It
contains non-null consumed S1/S2 observations and replay evidence. Only the
downstream candidate/formal/replay/validation/test observations are null and
their downstream authorization counts are exactly zero. It cannot be issued
from an S1 or S2 tombstone, scientific FAIL, replay mismatch, incomplete
temporary root, unpushed revision or dirty worktree outside the exact
protected-untracked exceptions.

After the no-authority machine-lock pair is itself committed, reviewed and
pushed, a new and separate pending authorization may be created for exactly one
formal run and one fresh formal replay under the earlier protocol. The pre-lock
claims do not satisfy, enlarge or replenish that later quota.

## Atomic completion, failure and recovery

S1 and S2 use phase-owned temporary and finalized roots with exact mutation
allowlists. A claim is exclusive-created before scientific work. Completed
PASS and completed scientific FAIL are both retained with their full evidence.
A handled exception writes an atomic tombstone containing phase,
authorization/contract/source hashes, normalized exception identity,
traceback hash, final canonical ledger prefix and a manifest of surviving
owned files.

A hard crash or interrupted multi-file handoff may leave an incomplete
temporary root. The next invocation must detect it, refuse authorization reuse
and require a separately committed and reviewed recovery record. Per-file
atomic replace, or even a final directory rename, is not described as an
atomic multi-file transaction. Replay never reads a directory by name; it
starts from the finalized outer manifest and follows only the exact acyclic
hash-listed handoff chain.

A scientific S1 failure closes S1 and forbids S2. A scientific S2 failure
closes this exact derivation and forbids the machine lock. A harness failure
before any forbidden observation may receive a new authorization only after a
committed diagnosis, code/test correction, new source revision and new
contract. Neither phase permits silent retry, cleanup that erases evidence, or
reuse of a consumed claim.

## Scientific interpretation

S1 establishes only machine-specific engineering feasibility on synthetic
inputs. S2 establishes only deterministic score-independent train-metadata
artifacts. Neither phase estimates efficacy, observes a candidate result or
supports a new-model claim. Passing them merely removes the circularity that
otherwise prevents a defensible no-authority machine lock.
