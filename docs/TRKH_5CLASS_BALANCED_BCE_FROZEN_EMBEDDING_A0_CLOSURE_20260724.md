# TRKH 5-Class Balanced BCE Frozen-Embedding A0 Closure - 2026-07-24

## Decision

Reject and close the prospectively locked Balanced BCE (Bal-BCE)
frozen-embedding A0 on the current `yolo_f/train` evidence. Exact LiVT
Bal-BCE expands rare class 1 and recovers true positives, but it expands the
same boundary into rival classes `0/2/4`: class-1 precision falls from
`0.821494` to `0.759317` and restricted false positives rise from 94 to 151.

Do not integrate Bal-BCE into the production trainer, open validation/test,
run a smoke/probe/full train, or update the current-best commands from this
result. This closure applies to the exact locked TRKH frozen-embedding
adaptation; it does not invalidate LiVT or Balanced BCE in their published
end-to-end long-tailed settings.

## Locked Scope

- Formal commit:
  `f2fcb80fcde4a905ca8a382516041f4cf35110a4`.
- Protocol SHA-256:
  `27c32d2aa7ca0bc421018966a5d261fbcedde301f8e2e2200aba515d82a5c93b`.
- Lock SHA-256:
  `1e28c6864542fdc2cada627bbd919f73b17232f70f9ba60ba347cea6672bc07e`.
- Formal evidence:
  `runs/audit_balanced_bce_frozen_embedding_a0_20260724`.
- Formal summary SHA-256:
  `b501fe60538b5d54010cc4741ce0d3a490341b3ddce3af60ae5ac5e7960451b6`.
- Pre-replay artifact-set digest:
  `084fde3c64fd4c8344baa633c7218b67e2ada77aa713069e441bba8156f0eb51`.
- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.

The A0 uses all 9,215 immutable train-object rows, five fixed source-disjoint
outer folds, frozen train-only embeddings, and priors computed only from each
fold's fitting rows. Six roles share locked initialization, order, optimizer,
and 30-epoch budget: CE, plain BCE, exact Bal-BCE, an independent Bal-BCE
repeat, historical Balanced Softmax at `tau=0.25`, and reversed-prior BCE.
Validation and test access are forbidden.

## Formal Result

| Role | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | TP | Restricted FP |
|---|---:|---:|---:|---:|---:|---:|
| CE control | 0.941361 | 0.821494 | 0.833641 | 0.827523 | 451 | 94 |
| Plain BCE | 0.940091 | 0.819013 | 0.828096 | 0.823529 | 448 | 96 |
| **Bal-BCE candidate** | **0.940560** | **0.759317** | **0.903882** | **0.825316** | **489** | **151** |
| Bal-BCE seed repeat | 0.940132 | 0.756173 | 0.905730 | 0.824222 | 490 | 154 |
| Balanced Softmax `tau=0.25` | 0.941954 | 0.801029 | 0.863216 | 0.830961 | 467 | 111 |
| Reversed-prior BCE | 0.933270 | 0.865217 | 0.735675 | 0.795205 | 398 | 61 |

Only 19 of 34 performance checks pass. Against CE, Bal-BCE changes 116
actions, rescues 38 class-1 false negatives, breaks no class-1 true positive,
and gains 38 TP. That recall gain is not selective: it makes 48 corrections
and 59 harms, removes zero restricted false positives, and creates 57.

Against plain BCE, the candidate gains 41 TP but creates 55 net restricted
false positives. Balanced Softmax is stronger than Bal-BCE in both macro F1
and class-1 F1. The seed repeat reproduces the same failure direction and
creates 60 restricted false positives against CE.

The reversed prior proves the opposite global pressure is also unsuitable. It
removes 33 restricted false positives and raises precision to `0.865217`, but
rescues no false negative and breaks 53 class-1 true positives. A global
class-prior intercept can move the precision-recall operating point, but it
does not learn the local surface/boundary evidence needed to separate class 1
from classes `0/2/4`.

## Visual Review

All 20 fixed train-only anchors were reviewed at original detail. Bal-BCE
systematically raises class-1 confidence on visually rival class-0, class-2,
and class-4 fruit, including already-confident false positives. Several
non-class-1 rows approach class-1 probability 0.99, while some difficult
class-1 misses remain.

The contact sheet confirms the transition counts: the gain comes from broad
rare-class expansion, not selective attention to lesion, ripeness, bruising,
or surface boundaries. Manual review therefore rejects the candidate
independently of the automatic gate.

## Structural, Resource, And Replay Result

Every structural check passes. The formal run takes `205.86 s`, peaks at
`1.1537 GiB` process RSS and `0.0742 GiB` CUDA allocation, starts with
`4.3093 GiB` available RAM, and observes no unexpected compute process.
Validation/test open counts are zero and the raw dataset is unchanged.

Fresh-process replay passes in `205.12 s`. OOF logits/probabilities, model
states, training trace, fold protocol, equations, analysis, performance
gates, data-access ledger, visual arrays, and visual metadata all reproduce
with maximum error `0.0`. Replay SHA-256 is
`8766c26ede66bb7fb53f0478469d58ad822703a4d6439c1f90e31ac30df5ef07`.

Current-best commands, command history, and keeper checkpoint remain
byte-identical at:

- `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`;
- `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`;
- `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.

## Finalization

- Visual-review SHA-256:
  `5523edc82611ce61e1f32bfb890e98384a66ea55911bca11c7228e154c54e4ec`.
- Contact-sheet SHA-256:
  `ba87430089739824d4fd172dd981e110d4ff3b4b409795fced197fe1e6a9e3e0`.
- Final-decision SHA-256:
  `93fc089b914bfdf55253fe47730a53c9909b27d13c4cc7d5d86a706691b025e5`.
- Final artifact-set digest:
  `f972d1f4c399d45f3a0c172c3a4d127fb468616481def506b8a22a7e8bf07b7b`.
- `final_manifest.json` file SHA-256:
  `fdd5628b66bca2ac6c80736eb0339294644eefc02c3c444a0762e4195900675`.
- Final status: `rejected`.
- Authorization: false for integration, validation smoke, test, probe, full
  train, and current-best update.

## Closed Boundary

Do not sweep the exact prior multiplier, BCE scaling, seed, fold assignment,
epoch count, optimizer schedule, or nearby Balanced Softmax temperature using
this result. The two prior directions already expose the full global tradeoff:
positive rare-class pressure creates many rival false positives, while the
reverse pressure removes false positives by sacrificing true positives.
Post-result interpolation would be an unregistered threshold sweep, not new
evidence.

The retained lesson is that class-frequency correction alone cannot solve the
class-1 boundary on the frozen representation. The next candidate must add a
genuinely sample-conditional local representation or supervision signal and
must first pass a source-held train-only gate that simultaneously preserves
class-1 TP and removes restricted false positives. Any derived augmentation
or synthetic child must be generated only from its fitting-train parents,
inherit every parent source/fold identity, and remain outside raw data.

## Verification

- Bal-BCE lock/engine/auditor plus retention focused suite: `17/17` passed.
- Complete repository suite after closure: `1899/1899` passed with 295
  existing warnings in `72.11 s`.
- Read-only retention scanned 850 run directories and all 51 currently
  present object-schema compaction manifests. All 220 named original
  directories are absent, no file was deleted, `blockers=[]`, and
  `88.871 GiB` remains free. Retention-summary SHA-256 is
  `b48ebd8248b913c65039fdf403e19172eab3b7cd94a2a22191eee5ee09705510`.
- Research-process report revision 10 renders cleanly across 12/12 pages and
  passes accessibility with `high=0`, `medium=0`, and `low=0`.
- Formal structure, independent replay, and manual visual review are complete.
- Raw data, production trainer/model/config, validation/test, and current-best
  command/history were not modified by Bal-BCE A0.
