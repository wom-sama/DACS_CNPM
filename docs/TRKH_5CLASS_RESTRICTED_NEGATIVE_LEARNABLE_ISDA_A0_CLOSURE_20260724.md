# TRKH 5-Class Restricted-Negative LearnableISDA A0 Closure - 2026-07-24

## Decision

Reject and close the prospectively locked restricted-negative LearnableISDA
(RN-LISDA) A0 on the current frozen `yolo_f/train` evidence. The fixed
class-wise covariance increases class-1 precision and removes restricted
false positives, but the learnable sample-wise covariance collapses to the
fixed control, loses 25 class-1 true positives versus CE, and fails its
causal/mechanism controls.

Do not integrate RN-LISDA into the production trainer, open validation/test,
run a smoke/probe/full train, or update the current-best commands from this
result. This closure applies to the exact locked TRKH adaptation; it does not
invalidate ISDA, MetaSAug, or LearnableISDA in their published settings.

## Locked Scope

- Formal commit:
  `79d0b7185909a32f23560c23f991caca733c239a`.
- Lock SHA-256:
  `ee63db6a9ec515c58b0697d467928c014c9df424b4b2226619fdff3169d03c42`.
- Protocol SHA-256:
  `e9d960d1260e34a2b0f2b456c3c39d3927f48f70795d2ce53f027de8895d1017`.
- Formal evidence:
  `runs/audit_rn_lisda_a0_20260724`.
- Formal summary SHA-256:
  `1bd069aefeda01fbaae7862f6522e2c47466895bb37f6468aa37c3bb745edc7d`.
- Pre-replay artifact-manifest SHA-256:
  `68db8305029ad5e37285b2f8acf139b813000289c44781e8bf4351b100548ff4`.
- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.

The A0 uses all 9,215 immutable train-object rows, five fixed source-disjoint
outer folds, train-only frozen embeddings, and opposite source-disjoint inner
partitions for the meta objective. Validation and test access are forbidden.
The candidate augments only true classes `0/2/4` against rival class 1; true
classes 1 and 3 remain clean.

## Formal Result

| Role | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | TP | Restricted FP |
|---|---:|---:|---:|---:|---:|---:|
| CE control | 0.940600 | 0.817851 | 0.829945 | 0.823853 | 449 | 96 |
| Fixed class-wise ISDA | 0.938291 | 0.848000 | 0.783734 | 0.814601 | 424 | 74 |
| **RN-LISDA candidate** | **0.938291** | **0.848000** | **0.783734** | **0.814601** | **424** | **74** |
| CovNet seed repeat | 0.938291 | 0.848000 | 0.783734 | 0.814601 | 424 | 74 |
| Restricted all-rival meta | 0.939795 | 0.838150 | 0.804067 | 0.820755 | 435 | 81 |
| Full all-class meta | 0.939630 | 0.813406 | 0.829945 | 0.821592 | 449 | 99 |
| Deranged-input control | 0.938291 | 0.848000 | 0.783734 | 0.814601 | 424 | 74 |
| Joint/no-meta control | 0.938291 | 0.848000 | 0.783734 | 0.814601 | 424 | 74 |

Only 17 of 37 performance checks pass. Versus CE, the candidate changes 58
actions, makes 24 corrections and 30 harms, removes 22 restricted false
positives without creating one, but rescues zero class-1 false negatives and
breaks 25 class-1 true positives. Precision therefore improves by suppressing
class 1 rather than learning a selective new signal.

The candidate, fixed covariance, seed repeat, deranged input, and joint/no-meta
roles have exactly the same discrete metrics. Prediction agreement between the
candidate and seed repeat is 1.0. The all-rival controls recover some recall
but do not beat CE class-1 F1 or macro F1.

## Mechanism Result

Twenty-two of 25 mechanism checks pass. The equation oracle, initial
covariance, gradients, updates, state changes, partition provenance, paired
orders, and balanced meta schedules are valid. Three decisive checks fail:

- `samplewise_scale_std`: learned scale varies by only about
  `5.50e-8` across held samples;
- `same_label_input_swap_change`: swapping the covariance input changes the
  output by only about `1.68e-8`;
- `candidate_to_joint_covariance_ratio`: candidate/joint mean covariance is
  only `1.000066`, below the locked separation requirement.

The CovNet is active numerically but effectively outputs a constant scale near
one. It does not learn sample-specific covariance semantics.

## Visual Review

All 20 fixed rows were reviewed at original detail. Five class-1 rows
correctly disable augmentation. On every enabled row, candidate proxy 0
collapses to the deranged-input proxy. Candidate proxy 1 does not preserve a
stable sample-specific lesion, ripeness, or surface meaning. Several
class-0/2/4 rows map to generic class-1-like proxies rather than a defensible
same-sample semantic direction.

This agrees with the failed causal checks and the 25-TP loss. Manual review
therefore rejects the mechanism independently of the performance gate.

## Structural And Resource Result

All structural checks pass. Formal execution takes `898.04 s`, peaks at
`1.3284 GiB` process RSS and `0.1111 GiB` CUDA allocation, and observes no
unexpected compute process. Validation/test open counts are zero. Raw metadata
is unchanged at
`c6c9375d0ab3b097938c6cb1b88119abf5bd987af79e8e9271084b3815592eb5`
over 16,129 files.

Current-best commands, command history, and keeper checkpoint remain
byte-identical at:

- `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`;
- `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`;
- `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.

## Replay And Checker Errata

Fresh-process replay E passes. Replay SHA-256 is
`58c6fe1240e1c1ed9a2cf300227c0515f6b888fe840699017daff02ea3ed4712`.
Probability, state, trace, fold, analysis, performance, mechanism, data-access,
visual metadata, and visual-array comparisons are exact; the text-CSV
probability round-trip error is `5.0000004e-11`, below the locked `1e-7`
tolerance. Data-access ledger error is zero, validation/test opens are zero,
and no unexpected process is observed. Replay runtime is `787.36 s`.

The replay history is retained transparently:

1. An initial launcher check used an invalid one-second command timeout and
   closed pytest stdout, causing `OSError [Errno 22]` before scientific output.
2. Replay B completed computation but failed at `root.path_counts`. Formal had
   rendered the contact sheet and opened 70 unique train images; replay had
   `render_contact=False`. This was a checker read-set mismatch.
3. Replay C used the external erratum wrapper but failed immediately with
   `ModuleNotFoundError: trkh`; no scientific computation or artifact occurred.
4. Replay D matched the visual read-set but the checker compared JSON `bool`
   with `numpy.bool_` using object identity. It falsely reported a difference
   at `candidate_covariance_positive_finite`.
5. Replay E normalized boolean scalar types and rendered the same fixed visual
   evidence to a temporary path. It passed every replay check.

The replay-only wrapper SHA-256 is
`e0551f921c5e0a09bf1709326a4a8ba0d51fc6f13612a79ffa5f5e2cefc7cd81`.
It changes only replay checking, not equations, inputs, training, metrics, or
formal artifacts.

The permanent repository fix now renders replay visual evidence to a temporary
contact sheet while preserving the formal metadata path, and compares
`bool`/`numpy.bool_` by typed logical value. Regression tests prove the formal
contact sheet is not overwritten and mismatched boolean values/types still
fail closed.

## Finalization

- Visual-review SHA-256:
  `8a91ca8f6f55cd0c57297478d6d6633c0558cf5474fefb61e249c43b6b1210aa`.
- Final-decision SHA-256:
  `af0f42a40ade134caa859d662e4143e67cce69d8e18e0057f8dea723b565e61f`.
- Final-manifest SHA-256:
  `349a04559abc8d0b4a683fd697f0e627cfd8e8dfefd3b1d78a48807578e7a14d`.
- Final status: `rejected`.
- Authorization: false for integration, validation smoke, test, probe, full
  train, and current-best update.

## Closed Boundary

Do not sweep the exact restricted classes, rival column, lambda schedule,
CovNet width/depth, meta interval, balanced meta batch, fold schedule,
covariance seed, input derangement, or nearby epoch/LR variants. The candidate
already collapses across independent initializations and causal controls.

The retained lesson is that fixed class-wise semantic variance can trade
class-1 recall for precision, but making its scale learnable does not create a
sample-specific signal on the current representation. The next loss-level
route must be equation-distinct, fitted only on source-held train partitions,
and prospectively require both restricted-FP removal and class-1 TP retention.
Balanced BCE from LiVT is eligible for exactly such an OOF readout gate, with
plain BCE, CE, prior-direction, and historical Balanced Softmax controls. It
must not reach production training unless that independent gate passes.

## Verification

- RN-LISDA focused suite after permanent checker repair: `32/32` passed.
- Formal finalizer suite before repair: `31/31` passed.
- Complete repository suite after closure repair: `1884/1884` passed with
  295 existing warnings in `71.06 s`.
- Research-process report revision 8 renders cleanly across 12/12 pages and
  passes accessibility with `high=0`, `medium=0`, and `low=0`.
- Raw data, production trainer/model/config, validation/test, and current-best
  command/history were not modified by RN-LISDA A0.
