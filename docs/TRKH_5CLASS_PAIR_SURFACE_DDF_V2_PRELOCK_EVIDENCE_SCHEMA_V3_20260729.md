# TRKH 5-Class Pair-Surface DDF V2 Pre-Lock Evidence Schema V3 - 2026-07-29

## Status and scope

This is a prospective producer/verifier schema. It binds the pushed scientific
foundation at `9ea25ea87ae6d497b80dd86c27e59391d21c4a9b`, the independently
reviewed boundary at `c8fd6ca080effcfd78bae8fb9fc389056bd27ad9` and the
fail-closed trusted-runner foundation at
`3942f140efdf84ad1882aed8ecf7185b7868bef3`.

It authorizes zero tests, cache reads, GPU work, S1, S2, formal training,
validation, test, XAI or candidate execution. Its sole purpose is to freeze the
evidence that later S1/S2 producers must emit and the verifier must independently
recompute. No positive production fixture may substitute for finalized S1/S2
evidence.

## Exact document serialization

Every pre-lock document uses the runner's existing outer wrapper:

```json
{
  "schema_version": 2,
  "protocol_id": "trkh_pair_surface_ddf_a0_v2",
  "kind": "<exact artifact role>",
  "state": "<exact state>",
  "payload": {
    "evidence_schema_version": 3
  }
}
```

Bytes are `json.dumps(..., ensure_ascii=True, sort_keys=True,
separators=(",", ":")).encode("utf-8")` with no trailing newline. Extra keys,
path aliases, caller-returned digest sentinels and unbound caller mappings fail.

The eight existing stage roles and their states are exact:

| Role | State |
|---|---|
| `v2_s1_contract` | `s1_contract_v3_frozen` |
| `v2_s1_authorization` | `s1_authorization_v3_pending_unconsumed` |
| `v2_s1_bundle_manifest` | `s1_primary_replay_bundle_v3_finalized` |
| `v2_s1_finalization_root_record` | `s1_outer_root_finalized` |
| `v2_s2_contract` | `s2_contract_v3_frozen` |
| `v2_s2_authorization` | `s2_authorization_v3_pending_unconsumed` |
| `v2_s2_bundle_manifest` | `s2_primary_replay_bundle_v3_finalized` |
| `v2_s2_finalization_root_record` | `s2_outer_root_finalized` |

The pure schema module serializes only the two verifier-owned finalization
roles. It rejects contract, authorization and bundle kinds rather than treating
their payloads as opaque mappings. Those six payloads remain the responsibility
of their stage-specific exact parsers in the contracts/runner closure; until
those parsers are prospectively updated and independently reviewed, activation
remains blocked.

The boundary registry remains wrapper schema 2 with state
`prelock_boundary_registry_v2_finalized`. Its existing payload key set does not
gain an unparsed version field. Instead, its exact `hashes` mapping gains the
fixed role `v2_prelock_evidence_schema_v3`, whose bytes are also bound by every
stage contract. It also gains `v2_prelock_evidence_schema_v3_source`, the pure
executable typed-schema module described below. The role-set parser in runner
commit `3942f14` must therefore receive an explicit prospective update before
any registry can pass; this document does not silently reinterpret that parser.

## Primitive records

`ArtifactRecordV3` has exact keys:

```text
role, logical_path, media_type, access_class, bytes, sha256
```

Inputs retain exact absolute identities. Outputs use `$RUN_OUT/...` logical
paths so primary and replay may use distinct physical roots.

`ArrayRecordV3` adds:

```text
dtype, shape, array_bytes, array_sha256
```

The verifier derives `array_sha256` from ASCII NumPy dtype, little-endian int64
dimensions and C-contiguous array bytes. It does not trust an array hash without
the corresponding canonical NPY payload. `array_bytes` must equal the product
of the frozen dimensions and dtype item size, and the media type is exactly
`application/x-npy`.

`ArchiveMemberEventV3` has exact keys:

```text
event_index, archive_role, archive_expected_bytes, archive_expected_sha256,
member_name, archive_directory_index, compression_method, compressed_bytes,
uncompressed_bytes, uncompressed_sha256, dtype, shape, array_sha256, stage,
outer_fold, target_bearing, transition_sha256, read_count,
snapshot_artifact_role, snapshot_array_sha256
```

NPY decoding uses `allow_pickle=False`. Each decoded member is read exactly
once, snapshot to a canonical artifact, and cross-bound to its archive directory
entry and raw uncompressed member bytes.

An authorized S2A runner may physically read/hash the complete container bytes
to verify the contract's container identity. That event is logged as container
access and does not release a target member. Raw container bytes are never
returned to the producer. Target release means typed member decompression and
decode, which remains forbidden until the S2B transition. This distinction is
the same one required by the sequencing erratum.

`WriteReceiptV3` is runner-owned:

```text
sequence, role, logical_path, bytes, sha256, creator_claim_sha256,
process_identity_sha256, physical_identity, media_type
```

## Common lineage and acyclic contract order

Every stage contract binds:

```text
foundation_commit = 9ea25ea87ae6d497b80dd86c27e59391d21c4a9b
boundary_review_commit = c8fd6ca080effcfd78bae8fb9fc389056bd27ad9
source_commit
branch
upstream_ref
upstream_commit == source_commit
repository_tree_sha256
launcher ArtifactRecordV3
v2_prelock_evidence_schema_v3 ArtifactRecordV3
v2_prelock_evidence_schema_v3_source ArtifactRecordV3
full_local_import_closure {module -> ArtifactRecordV3}
package_files {package -> ArtifactRecordV3}
fixed_protocol_and_incident_hashes
expanded_test_node_registry_sha256
```

The S2 contract additionally contains the exact absolute
`verified_s1_receipt` `ArtifactRecordV3` with role
`v2_s1_finalization_root_record`. The fixed protocol/incident map has the exact
thirteen keys and values in the executable schema; missing or extra entries
fail. The launcher/schema-source/local-closure records use exact
`committed_pushed_local_import_identity`, the schema document uses
`committed_pushed_boundary_document_identity`, package files use
`pinned_package_file_identity`, and the prior S1 receipt uses
`committed_pushed_stage_receipt_identity`. All are absolute inputs, never
`$RUN_OUT` aliases. The schema-source record must also appear byte-identically under
`trkh.tools.pair_surface_ddf_v2_prelock_schema_v3` in the full import closure.

`repository_tree_sha256` is the SHA-256 of canonical JSON mapping every reviewed
logical source path to its exact byte count and SHA-256; it is not a Git SHA-1
renamed as SHA-256.

A stage contract cannot bind an authorization that does not yet exist. The
separate later authorization binds the exact committed/pushed contract record.
Claims, bundles and verifier-owned stage receipts bind both exact records. The
S2 contract additionally binds the already committed/pushed verifier-owned S1
PASS receipt. This order is acyclic:

```text
source/schema -> contract -> authorization -> primary/replay operational roots
              -> verifier-owned stage PASS receipt
```

The local closure is the transitive closure actually used before and during
execution/collection, including package `__init__` files, launchers, tests and
local `conftest.py` files when applicable. A direct-file list is insufficient.

The invalidated manifest SHA
`f36f95e2654a8bf8708f418716a99b6156947864632427176a00baa2d9e4c6af`
fails if it appears as an authority, PASS manifest, source manifest,
predecessor attestation or accepted lineage. It may appear only in the exact
rejection list with reason `incomplete_transitive_import_closure` and zero
scientific/authority effect. The invalidation document SHA
`5b52066708f419ab5ce728f64328585829ab5eaf84954da74c1909e2b58bd558`
is mandatory.

## Executable typed-schema source

Before producer or verifier code, the boundary commits and pushes the pure
stdlib module `trkh.tools.pair_surface_ddf_v2_prelock_schema_v3`. Its hash is
the `v2_prelock_evidence_schema_v3_source` record. The module contains no I/O,
dataset, model, metric or authorization operation. It defines strict immutable
schemas for the wrapper and primitive records; ordered payload keys, artifact
roles, dtypes, shapes and row ordering for all `12/10/5` scientific groups; the
lineage matrix; and both verifier-owned finalization payloads. Producer and
verifier import this one serialization definition but implement derivation and
recomputation separately.

No group manifest may contain an opaque `payload` mapping. Every group entry is
validated against the exact ordered schema module, and every array field is an
`ArrayRecordV3` whose canonical NPY bytes are present in the payload manifest.
Changing that module requires a new reviewed schema boundary before producer or
verifier changes.

Every group child role is scoped as `<group_role>__<field_name>`, uses the exact
path `$RUN_OUT/<group_role>/<field_name>.<json|npy|bin>`, and has a stage-exact
access/replay class. Plan bytes, build logs, timing vectors and CUDA allocation
traces are structural observations; all other group fields are scientific-exact.
Every JSON
registry uses one canonical seven-key tabular envelope:

```text
evidence_schema_version, schema_id, columns, order_by,
count_rule, row_count, rows
```

The executable `RegistrySpecV3` table fixes all row columns/types, unique exact
Cartesian axes/order, a typed exact or source-derived nonempty count rule and
required constant values. `parse_registry_bytes` requires the corresponding
typed group payload and rejects unless the registry bytes and length match that
group's exact `ArtifactRecordV3`. The current pure schema validates the shape of
a source-derived registry but then always raises; no caller-supplied integer can
make it return success. A later stage-specific verifier must derive the row
count internally from its pinned typed artifacts before that path is activated. In
particular, S2A source-member rows require `target_bearing=false`; all 250
calibrator rows require absent fitted state/probabilities/threshold; donor
proofs require dominance; assignment rows require zero same-component pairs and
fixed points; S2B target-member rows require `target_bearing=true`; and all
one-class bootstrap counts are zero. Scientific tensors are individual canonical
NPY artifacts referenced by those rows rather than opaque output archives. The
only opaque binary outputs are the real ONNX model and TensorRT plan. This schema
declares their exact node/binding registries, but activation remains blocked
until the verifier opens the record-bound bytes and re-derives those registries
with the real ONNX/TensorRT parsers. A schema ID string by itself is not evidence.

## S1 synthetic-only evidence

Primary and replay each emit exactly twelve scientific groups inside the S1
bundle manifest:

1. `s1_environment_preimage`
2. `s1_synthetic_inputs`
3. `s1_ddf_fp64_equation_preimage`
4. `s1_validity_pool_softmax_preimage`
5. `s1_masked_bn_preimage`
6. `s1_gradient_update_preimage`
7. `s1_onnx_model_and_node_registry`
8. `s1_ort_reference_outputs`
9. `s1_trt_build_preimage`
10. `s1_trt_plan_identity_and_outputs`
11. `s1_raw_timing_vectors`
12. `s1_raw_cuda_allocation_trace`

Synthetic inputs and masks are regenerated from frozen constants/seeds. The
stored maximum synthetic batch is exactly 32 and shapes 1/2/32 are evaluated as
locked slices. Raw TensorRT timing retains exactly 300 batch-1 and 200 batch-32
FP64 seconds, rather than a caller-selected vector length. The gradient/update
group retains name, shape, before value, gradient, after value
and optimizer-state identity for every intended parameter; a representative
tensor cannot pass. The verifier independently recomputes FP64 equations,
validity pooling, masked softmax, masked-BN, invalid-fill invariance, parameter
counts, finite gradients and complete update coverage.

The ONNX model is real opset-17 bytes with exact domain/node registry. The
verifier runs checker and ONNX Runtime on the exact inputs and recomputes parity.
Each process retains its TensorRT plan bytes/hash/bindings independently; plan
bytes may differ, while workload, build configuration, outputs and decisions
must satisfy the identical frozen checks. Timing and CUDA allocation groups
retain raw vectors/traces from which the verifier recomputes summaries and
ceilings. Producer booleans such as `onnx_passed` or `latency_passed` are not
evidence.

The S1 bundle binds the exact contract and authorization plus both claims,
access ledgers, payload manifests, exact scientific roots, structural telemetry
roots, handoffs and runner-owned operational outer roots. Plan bytes, timing
samples and resource observations are structural; every other scientific
preimage is exact across replay. Only the stage verifier may accept these roots
and emit `v2_s1_finalization_root_record`, a verifier-owned zero-authority S1
PASS receipt. That receipt is committed and pushed before the S2 contract and
authorization are created.

## S2A target-blind evidence

Primary and replay each emit exactly ten S2A scientific groups:

1. `s2a_target_free_source_snapshot`
2. `s2a_validity_bbox_preimages`
3. `s2a_keeper_rival_and_rank_preimages`
4. `s2a_donor_assignment_preimages`
5. `s2a_nonwrap_offset_preimages`
6. `s2a_seed_epoch_order_preimages`
7. `s2a_parameter_schema_and_init_identities`
8. `s2a_bootstrap_draws`
9. `s2a_invalid_fill_noise`
10. `s2a_formula_and_formal_schema`

The source snapshot contains sample indices, exact sample-to-component mapping,
canonical ascending order of all 158 component hashes, held folds, the exact
outer-to-calibration fold map, model boxes, validity packbits, historical keeper
probabilities and exact source archive/member identities. It contains no target,
target count, CIDT target/baseline or XAI ID.

From those raw snapshots and the frozen engine, the verifier recomputes:

- `valid64`, `valid32`, `valid16`, `bbox16`, and `bbox16 AND valid16`;
- the exact twelve bbox-unusable sample indices;
- keeper strongest rivals with tie order `(0,2,4)`;
- per-fold victim/donor ordinal ranks;
- all five-by-six donor cost matrices, maxima, integer coefficients, dominance
  proofs, capacities, assignments, selected costs and histograms;
- zero same-component pairs and zero fixed points;
- the complete per-row/two-block non-wrap mapping, all eight permitted nonzero
  offsets and their executable source/destination slice rules;
- role/fold seeds, complete twenty-epoch orders, parameter schemas,
  initialization identities and matching-primary tensor identities;
- the exact `2000x158` component-bootstrap draws;
- the complete `763x3x256x256 uint8` invalid-fill-noise array; and
- the formal raw-evidence shapes/dtypes, the exact 250 calibrator role/formula/
  schema specifications with zero fitted state/probability/threshold, and the
  fixed 69-record Gate-14 registry. Actual calibrator fitting remains inside
  the later fold-gated formal execution after calibration targets are released.

The retained non-wrap array is the per-row `763x2x2 int64` preimage from which
the ten fold/block seed/rule applications are recomputed. The target-free snapshot also contains
the exact five-element outer-to-calibration fold map.

Rank records cover the exact `(outer_fold, held|calibration)` Cartesian axis and
the locked fold sizes `153/153/153/152/152`. Donor evidence contains five
fold-level `held x calibration x 6` cost tensors, 30 coefficient-dominance rows
and five assignments with exact lower/upper capacities, selected donor IDs,
six-cost vectors and null pre-target mismatch telemetry. The calibrator registry
uses exact `5 roles x 5 folds x 10 named feature sets` order, calibration fold
`(h+1) mod 5`, and the frozen scaler/logistic hyperparameters. Its state,
probability and threshold flags are all false. Gate-14 rows are the exact
ordered `(category,name,comparison_rule)` triples whose registry SHA-256 is
`689730f64f918e3b482c1f2d565cf71be4e03a89cafafc09626c08008cbe2dd6`.

The runner finalizes the producer outputs as `s2a_payload_manifest`,
`s2a_inner_root_sha256` and `target_blind_finalization.json`. Before S2B, it
re-reads and verifies this inner root. No S2A artifact may change afterward.

## S2B target-metadata completion

Only after the immutable S2A root, each process emits exactly five S2B groups:

1. `s2b_target_source_snapshot`
2. `s2b_task_and_bbox_weight_preimages`
3. `s2b_fold_target_shards`
4. `s2b_cidt_projection`
5. `s2b_deferred_xai_ids`

The first target member requires a runner-owned transition binding the consumed
S2A inner root and verified inner-manifest SHA. The verifier recomputes task and
bbox supports, all FP32 weight bit patterns, zero one-class bootstrap
replicates, bbox-unusable target records, and fifteen distinct
fit/calibration/held target shards. It also binds the exact CIDT row/target/
baseline projection over exactly 9,215 train rows and the predeclared 30 XAI
IDs/strata. S2A bytes and roots
must remain unchanged across the transition.

Target-derived support covers exact `5 folds x 4 tasks x 3 partitions` axes for
both task and bbox support, with partition row counts and positive/negative
support retained. Fit weights contain exactly 80 positive/negative
classification/bbox FP32 bit patterns. The bootstrap registry is one exact
`2000/0` replicate/one-class row. The fifteen target shards use
`fold x (fit,calibration,held)` order and locked dependent lengths. The three
CIDT registry rows must repeat the top-level ArrayRecords byte-for-byte and bind
the pushed exact array hashes; a second self-declared array cannot pass.

The original all-row target member is never an input to the later machine-lock
or formal allowlist. Formal execution receives only typed per-fold shards.
After both complete S2 executions, the runner-owned operational roots enter the
S2 bundle. The final verifier recomputes them and emits the verifier-owned
`v2_s2_finalization_root_record` plus the zero-authority prelock receipt; it does
not rewrite an operational root.

## Root order and ownership

Each execution uses this acyclic order:

```text
claim -> raw artifacts -> payload manifest -> exact scientific root
      -> structural telemetry root -> replay handoff -> outer manifest
      -> outer root record
```

Replay reads primary artifacts only through the exact handoff role set.

- Runner-owned: claims, process identities, ordered access/member events,
  transitions, write receipts, physical inventories, handoffs, operational
  execution roots and tombstones.
- Producer-derived: raw engine arrays, states, outputs, timing samples and S2
  snapshots/preimages.
- Verifier-recomputed: every lineage/root identity, S1 gate, S2 derivation,
  replay comparison, verifier-owned S1 PASS receipt and final zero-authority
  prelock receipt.

Producer-reported metrics, actions, gate decisions and PASS booleans are not
trusted inputs.

The verifier-owned `v2_s1_finalization_root_record` payload has exactly:

```text
evidence_schema_version, foundation_commit, boundary_review_commit,
schema_document_sha256, schema_source_sha256, source_commit,
s1_bundle_sha256, s1_contract_sha256, s1_authorization_sha256,
primary_operational_root_sha256, replay_operational_root_sha256,
exact_scientific_root_sha256, primary_structural_root_sha256,
replay_structural_root_sha256, engineering_decision_root_sha256,
fresh_process_replay, s2_contract_may_be_created, downstream_authority,
cache_or_lineage_reopened
```

`fresh_process_replay` and `s2_contract_may_be_created` are exact verifier
booleans `true`; `downstream_authority` contains zeros for formal, candidate,
validation, test and conditional XAI; `cache_or_lineage_reopened` is zero.

The nested `downstream_authority` mapping has exactly:

```text
formal_runs_authorized, candidate_runs_authorized,
validation_runs_authorized, test_runs_authorized,
conditional_xai_runs_authorized
```

Every value is integer zero; booleans are rejected as integers.

The verifier-owned `v2_s2_finalization_root_record` payload has exactly:

```text
evidence_schema_version, foundation_commit, boundary_review_commit,
schema_document_sha256, schema_source_sha256, source_commit,
s1_finalization_record_sha256, s2_bundle_sha256, s2_contract_sha256,
s2_authorization_sha256, primary_operational_root_sha256,
replay_operational_root_sha256, s2a_inner_root_sha256,
exact_scientific_root_sha256, primary_structural_root_sha256,
replay_structural_root_sha256, derivation_decision_root_sha256,
fresh_process_replay, downstream_authority, cache_or_lineage_reopened
```

It contains no self hash. After its create-once bytes exist, the future
machine-lock builder loads and revalidates this record, computes its SHA-256,
and persists the runner's exact `s1_s2_finalized_verified_no_execution_authority`
mapping inside the machine lock. The builder derives that mapping itself and
never accepts a caller-supplied receipt or token.

## Stage-aware production verifier API

The first production entry runs after S1 primary/replay and before S2 exists:

```python
def verify_finalized_s1_v3(
    s1_bundle_path: PathLike,
    expected_s1_bundle_sha256: str,
) -> VerifiedArtifactHandle:
    ...
```

It writes the exact create-once `v2_s1_finalization_root_record` at the output
path fixed by the S1 bundle and returns only its typed path/bytes/hash handle.
The second entry runs only after that record is committed/pushed and S2
primary/replay are finalized:

```python
def verify_finalized_prelock_v3(
    registry_path: PathLike,
    expected_registry_sha256: str,
) -> VerifiedArtifactHandle:
    ...
```

It writes `v2_s2_finalization_root_record` at the registry-fixed create-once
path and returns only its typed path/bytes/hash handle. Each entry follows every
contract and root from its one exact manifest/registry. Neither accepts arrays,
mappings, policies, root overrides or arbitrary output paths from the caller.
There is no public receipt serializer or token/capability factory. The later
machine-lock builder derives and persists the runner's exact receipt with zero
formal, candidate, validation, test and conditional-XAI authority and
`cache_or_lineage_reopened=0`.

## Authorized software-test surface

The future test authority enumerates exactly eight tmp-only nodes:

```text
test_outer_wrapper_exact_and_f36_rejected
test_source_commit_and_full_import_closure_required
test_s1_raw_preimage_schema_and_pure_oracles
test_s1_primary_replay_root_partition_and_fresh_process
test_s2a_target_member_blocked_until_inner_finalization
test_s2_archive_member_snapshot_binding
test_s2_pure_derivation_oracles_and_transition_rejections
test_final_receipt_exact_zero_authority_and_no_reopen
```

The current eight pure-schema developmental tests are not those authority
nodes and cannot be promoted or renamed into test authority after execution.
There is no positive production fixture. Synthetic fixtures may exercise pure
helpers and rejection paths only. `verify_finalized_s1_v3()` rejects without a
real finalized S1 chain, and `verify_finalized_prelock_v3()` rejects until real
finalized S1 and S2 chains exist. Test collection itself requires a prospective
full-import-closure authority and pre-import tripwire.

## Complexity budget and stop rule

- Add no top-level stage role beyond the existing eight.
- Keep S1/S2A/S2B scientific groups at exactly `12/10/5` per execution.
- Expose at most the two verifier functions above.
- Authorize at most eight test nodes.
- Keep schema, verifier and launchers near 3,200 nonblank lines total and tests
  near 900 lines. This one increase from 2,000 is justified only by the reviewed
  constructive fake-preimage path through opaque registry rows; no further
  increase is allowed without another concrete false-PASS reproducer.
- Stop once lineage, target release, raw recomputation and replay are proven.
  Do not build a generic hostile-code sandbox.

Any field, role, test or implementation growth requires a named gate or a
concrete false-PASS path. Protocol complexity is not scientific information
gain.
