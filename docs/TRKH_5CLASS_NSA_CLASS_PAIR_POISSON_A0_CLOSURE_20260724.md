# TRKH 5-Class NSA Class-Pair Poisson A0 Closure - 2026-07-24

## Decision

Reject and close the prospectively locked NSA source-gradient Poisson
class-pair query adaptation on the current frozen keeper. The geometry is
natural and source-safe, and the query adapter does learn query-dependent
responses, but the candidate does not localize class compatibility and does
not beat its matched clean, no-query, permuted-query, or base-context controls.
Do not integrate this branch, open validation/test, launch a smoke/probe/full
train, or update current-best commands from this result.

This decision applies to the exact independent TRKH adaptation below. It does
not claim that the accepted NSA anomaly-localization method is invalid.

## Locked Scope

- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Final prospective protocol SHA-256:
  `fbd7788c5ddba0e08a8b4cf680c427674f68bb9744198f41c6c1c7e6f146254f`.
- Accepted ECCV-2022 NSA paper and supplement SHAs:
  `fdd14c4e...74e44` and `5804ae02...32da`. The authors' MIT repository is
  pinned at commit/tree
  `919591685307ce030fe27cb77687509dc277189c` /
  `6eebd6e9d7ce9ff77d7d0aa9a2640da7f304cd06`.
- Formal evidence:
  `runs/audit_nsa_class_pair_poisson_a0_formal_v2_20260724`.
- Formal summary SHA-256:
  `576c061fd78d14f1cf848a32d40437a510a3c0222a853aa0aefac9a9cac37e13`.
- Pre-replay artifact-manifest SHA-256:
  `7b443c0d572b14afda1b3538ab8e33527524680247b0645102d5a05cc58f36b1`.
- Fixed 19-row contact-sheet SHA-256:
  `2564d055d014e02f2109478e487b5e977a733c1b4619d5c81cfa81159f1cb9b2`.

The A0 uses all 9,215 immutable `yolo_f/train` object rows only. Five fixed
source-held folds train four capacity-matched adapters for two epochs:
Poisson class-query candidate, clean-query control, no-query control, and
independently permuted-query control. Clean source-held readouts score exactly
750 rows: 528 class-1 true positives and 222 restricted `0/2/4 -> 1` false
positives. A separate 375-row held synthetic panel is diagnostic only.
Validation/test, trainer code, raw data, and current-best command/history are
untouched.

## Formal Result

Only 11 of 21 conjunctive mechanism gates pass.

| Role | AUROC | TP retention | FP rejection | Corrections | Harms |
|---|---:|---:|---:|---:|---:|
| Base context | 0.824887 | 0.975379 | 0.135135 | 26 | 13 |
| Clean query | 0.824333 | 0.975379 | 0.130631 | 25 | 13 |
| No query | 0.824094 | 0.975379 | 0.139640 | 27 | 13 |
| Permuted query | 0.824657 | 0.975379 | 0.126126 | 24 | 13 |
| **Poisson query candidate** | **0.821398** | **0.975379** | **0.139640** | **26** | **13** |

The candidate misses the locked AUROC floor `0.855`, TP-retention floor
`0.98`, every required AUROC margin, four-of-five-fold superiority, and
three-corrections-per-harm requirement. It rejects 31 of 222 restricted false
positives but breaks 13 of 528 class-1 true positives. Rejection by true class
is `0.120253/0.203704/0.100000` for classes `0/2/4`; the aggregate action is
not better than the no-query control.

The adapter is not simply ignoring query identity:

- true-query compatibility is better than false queries on `0.950667` of held
  diagnostic comparisons;
- five-query class accuracy is `0.589333`;
- effective spatial rank is `4.033522`;
- every candidate query parameter has a finite nonzero gradient and update.

Those positive controls do not establish the intended local mechanism. Mean
candidate cross-map mass inside the fixed intensity target is only `0.092528`,
far below `0.75`. Same-support pixel mismatch is low at `0.033731`, showing
stable maps, but they are stably diffuse rather than target-localized.

## Structural And Resource Result

Twenty-nine of 30 structural checks pass. The complete fold/source schedule,
geometry eligibility, update counts, readout convergence, query gradients,
750-row cohort, 375-row diagnostic panel, 19 visual rows, immediate saved
readout replay, and current-best/keeper hashes are exact. Validation/test
access and raw-data changes are zero. There is no unexpected Python/TensorRT
process.

The sole structural failure is the prospectively locked virtual-memory
fraction: observed peak `0.952` versus required `<0.82`. Physical and CUDA
resources are acceptable at `2.2666 GiB` peak RSS and `1.8905 GiB` peak CUDA.
Formal elapsed time is `3382.39 s`. Four DataLoader workers are the measured
winner over two workers and were used throughout.

The virtual-memory failure is retained as a real gate failure. It is not
relaxed after observation and is not the scientific reason for rejection;
the candidate independently fails localization, AUROC, TP protection, causal
control, and action-balance gates.

## Visual Review

All 19 fixed rows were inspected at original detail. The Poisson composites
remain natural: no systematic seam, padding edge, hand, basket, bbox edge, or
background shortcut is visible. The reviewed support correction successfully
keeps the previously problematic finger region out of the target rectangle.

Manual mechanism review still rejects the candidate. Candidate mismatch maps
are diffuse over broad fruit regions, are visually close to clean/no-query/
permuted controls, and do not concentrate on the explicit changed support.
Small targets are particularly weak. Natural synthetic geometry therefore
does not translate into discriminative local class-pair evidence.

## Replay And Finalization

- Fresh-process replay status: `failed`.
- Replay SHA-256:
  `ea6ed9630c1bb84a3fc7fa4c68210942a85f620d17194d2d26c5b8aa6a10fdf4`.
- Maximum replay errors and process/resource result:
  readout scores `2.220446049250313e-16`; metrics, query diagnostics,
  aggregate diagnostics, fold maps, and all 266 visual arrays `0.0`;
  peak RSS/CUDA `1.5769/0.6078 GiB`; peak virtual-memory fraction `0.899`;
  elapsed `11160.98 s`; no unexpected compute process.
- Final visual-review and decision-record SHA-256:
  `93a4da577a7c5e0cd34fd9028398f875d96d60a899926e65a050e0b222e46f89`.
- Final manifest SHA-256:
  `9b978d3abc0ea186a7ccdb1e9881dc7108717709b046c451670a41eb999aabc6`.

Replay fails the unchanged virtual-memory gate and the ordered geometry hash
gate. The latter is a checker-order defect rather than geometry drift: formal
stores each batch as `[all same][all cross0][all cross1]`, while replay hashes
`[same,cross0,cross1]` per sample. Reordering only the already saved formal
records reproduces the replay hash in all five folds. The complete proof and
all ten hashes are in
`docs/TRKH_5CLASS_NSA_REPLAY_GEOMETRY_ORDER_DIAGNOSIS_20260724.md`.
The replay file remains failed; no threshold or prospective contract is
relaxed after observation.

The final authorization remains false for trainer integration, validation,
test, smoke, probe, full train, and current-best command/history update.

## Closed Boundary

Close the exact same-class/cross-class Poisson geometry, hand-like exclusion,
minimum support extent, rectangle retry, source/query schedule, two-epoch
adapter, clean/no-query/permuted controls, readout, seed/fold, and nearby
post-metric threshold/weight/epoch variants. Do not tune the query embedding,
adapter width, support thresholds, Poisson mode, panel composition, readout,
or metric gates after seeing this result.

The useful retained lesson is narrower: structured pixel synthesis can be
geometrically valid while its learned mismatch representation remains
non-causal for fine-grained class separation. The next route should act in
the already informative 256-dimensional semantic feature space, preserve the
observed class-1 precision pressure of fixed class-wise ISDA, and recover
recall through a prospectively locked sample-wise meta mechanism. This leads
to the separate restricted-negative LearnableISDA A0; no NSA artifact or
metric may tune that protocol.

## Retention

The failed first formal attempt was compacted only after the v2 closure was
final. Its 57-file, `161,368,189`-byte source was replaced by 27 compact
evidence files plus inventories/manifests; 30 heavy files totaling
`159,348,276` bytes were excluded and the source deletion was verified.
Cleanup manifest:

`runs/cleanup_manifest_20260724_nsa_failed_v1_compacted.json`

The compactor resolves `output-dir` relative to `runs-root`; therefore the
verified evidence root is intentionally retained at:

`runs/runs/evidence_nsa_class_pair_poisson_a0_failed_v1_20260724`

It is not moved after verification because doing so would invalidate the
cleanup manifest's exact evidence path. The complete v2 formal evidence and
current keeper remain protected.

The post-closure read-only retention audit passes across 847 run directories:
all 36 source directories named by the seven compaction manifests are absent,
there are no blockers, no deletion is performed, and `88.918 GiB` remains
free. Its summary is
`runs/audit_trkh_artifact_retention_post_nsa_20260724/summary.json`, SHA-256
`82800b946962f85254b8f809cd189abbbcc8ca974d4eb8e98755908b02ecc958`.

## Closure Verification

- Focused NSA and retention tests: `29/29` passed.
- Complete repository suite: `1852/1852` passed with 295 existing warnings in
  `122.17 s`.
- Research-process report revision 7: 11 rendered pages reviewed at original
  detail; accessibility findings high/medium/low are `0/0/0`.
- Current-best command/history remain byte-identical at
  `36b9aa1a...940faf` and `39bd2879...98f53`.
