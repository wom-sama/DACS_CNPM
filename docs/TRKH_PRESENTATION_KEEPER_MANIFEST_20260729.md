# TRKH presentation keeper — 2026-07-29

This is the human-readable index for
`docs/TRKH_PRESENTATION_KEEPER_MANIFEST_20260729.json`. It freezes the minimum
complete evidence set before later cleanup or new candidate training.

## What is protected

- Historical deployable TRKH keeper, including checkpoint, resolved config and
  audits.
- Completed July-14 strict-balanced random-init full baseline, including its
  checkpoint, training history, architecture trace and independent val/final
  audits.
- The July-28 pre-fix warm-start plateau run and natural-only scratch run as
  negative evidence; neither may be promoted, resumed or extended.
- The complete validation-only seven-model comparison, including strict remap,
  per-model predictions, confusions, forensics, paired/bootstrap tables and all
  seven checkpoint hashes.
- Exact CCR train-only materialization and Pair-Surface DDF synthetic engineering
  evidence, the v1 fold-leakage supersession audit, and the v2 train-table
  geometry preflight.
- A source-only snapshot of AIDT and TIMM comparison projects, because those two
  directories are not Git repositories. The snapshot contains 28 code/config/doc
  files and explicitly excludes data, outputs, runs, checkpoints, weights and
  caches. A separate environment snapshot supports replay of the current
  validation-inference code; because the requirement files are not fully
  pinned, it does not claim to reconstruct the historical checkpoint-training
  environments.

The ten integrity-locked evidence roots contain 5,007 files and use
`1,495.939 MiB`, while drive D had about `87.388 GB` free at the refreshed
audit. There is no storage justification for pruning any of them now.

The refreshed read-only retention audit at
`runs/artifact_retention_audit_presentation_keeper_20260729` scanned 872 run
directories. All 9 essential-file byte/SHA locks and all 10 complete-root
inventories passed with zero blockers. The inventory covers every one of the
5,007 files, not merely directory existence. Summary SHA-256 is
`2e8e66914cb6fb6bc3240660015358004aa573db2f56e9915e79171d3ba5f23d`;
the exported complete-file inventory SHA-256 is
`00687ac6ac0ddbd313138f7437903437e6cca4795e37ed3c8369c0c4974af846`.

## Current comparison boundary

| Role | Validation macro F1 | Focus-class F1 | Interpretation |
|---|---:|---:|---|
| AIDT pretrained fusion | 0.910267 | 0.718563 | External-pretrained comparison only |
| TRKH historical keeper | 0.882925 | 0.678261 | Deployable historical TRKH anchor |
| TRKH strict-balanced random-init full | 0.874172 | 0.654639 | Completed scratch full baseline |
| Natural-only scratch probe | 0.754579 | 0.246154 | Rejected negative control |

The historical test split was opened in earlier work. The July-29 comparison did
not reopen it. A new confirmatory scientific claim requires an external or
temporal holdout; test results already seen may only be retained as historical
evidence.

## Pair-Surface DDF decision

The 2026-07-25 protocol/lock and synthetic engine evidence remain immutable
historical artifacts. No formal or replay run was consumed. A metadata-only
audit found that exact source stems do not overlap, but the dataset's own
`leakage_group` and conservative numeric filename neighborhoods do. Among 531 manifest groups,
150 cross folds, covering 330 rows; they create 217 cross-fold row pairs (92
same-label and 125 cross-label). Of those 150 groups, 105 contain more than one
label. Their fold-span distribution is 140/7/2/1 groups across 2/3/4/5 folds.

The independent numeric-neighborhood audit gives:

| Window | Cross-fold row pairs | Covered rows | Same-label | Cross-label |
|---:|---:|---:|---:|---:|
| 1 | 393 | 541 | 316 | 77 |
| 3 | 974 | 659 | 748 | 226 |

Therefore v1 is superseded before formal execution. It must not be used to claim
OOF generalization. A prospective v2 must group exact stems, available
near-duplicate families and every connected `Image_N` component at numeric gap
`<=3` before assigning folds. Its runner must reject the old fold array.
This numeric relation is not presented as verified fruit/session provenance.

The manifest `leakage_group` producer is preserved separately as clean CVAT
commit `0c7fd8b873d34d18c6f6f6588b772996528ba0ec`, exporter SHA-256
`65e6056dac1793eaa9775b458a448828bfd0debd7adc4d9acb58807b7b74e4b5`.
It unions normalized source families with dHash/colour-thumbnail near-visual
matches. Because its group ID hashes absolute source paths, the current
manifest bytes are canonical and relocated IDs are not assumed portable.

The prospective v2 fold manifest is now built without candidate scores: 158
connected components map to folds of 153/153/153/152/152 rows, with exactly two
class-4 rows per fold and zero cross-fold component/manifest-group/numeric-window
overlap. Its mapping SHA-256 is
`c0726d114df69290d68d9f9e1a40857074ccb1a66e5ca41883caa004f2b0ea40`.
This is manifest-group and numeric-neighborhood disjointness, not verified
fruit/session independence.

The v2 geometry preflight also exposed and fixed a subtle audit error. The
valid-mask cache was packed with `bitorder="little"`; a previous hand
calculation unpacked with NumPy's default `"big"`, producing the plausible but
wrong mean q `0.349860`. Explicit little-endian unpacking plus independent
NumPy/Torch propagation gives 751 bbox-usable rows, 12 unusable rows, and
usable mean/median q `0.3585639994/0.3461538462`. The wrong-endian path retained
the same row bit counts and the same 12 unusable rows, so the new regression
guard locks the full validity hashes as well as counts.

No v2 engine, machine authorization, formal score, replay or GPU training has
yet been consumed. The next valid step is implementation plus fail-closed
preflight—not a speculative full train. Passing that mechanism gate would
still require scratch integration against the keeper/random-init/static
controls and, for a paper claim, a newly sealed source/time/site holdout.

## Cleanup rule

Do not delete anything until this manifest is committed and pushed, a fresh
retention audit passes, and any replacement winner has its own checkpoint,
resolved config, metrics, audit, XAI/deployment and provenance bundle. Never
stage or delete the user-owned `BaoCao/` or two `deep-research-report` files as
part of this work.
