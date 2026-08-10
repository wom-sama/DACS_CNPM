# CutPaste Surface-Response A0 Closure (2026-07-20)

## Decision

Reject the exact frozen-keeper object-interior CutPaste/CutPaste-Scar response
descriptor. It does not expose a precision-safe class-1 false-positive veto
beyond clean keeper logits or matched Cutout. Do not run Stage B lighting,
integrate a CutPaste auxiliary branch, open validation/test, run a smoke/probe/
full train, or update current-best commands from this result.

## Locked Scope

- The accepted source is Li et al., CVPR 2021, DOI
  `10.1109/CVPR46437.2021.00954`. Paper/supplement SHAs are
  `e40ec13...17f4`/`fa89f98...1ea6`; the licensed PyTorch sampler reference is
  torchvision `v0.21.0` at commit/tree
  `7af6987...9eba`/`9188eec...ce0a`.
- Protocol SHA is `31017e37...256d1`. A prospective geometry erratum replaced
  the unsafe crop rectangle with deterministic surface-mask/bbox/image-mask/
  ellipse support and one two-pixel erosion before any keeper forward or
  candidate metric.
- Infrastructure and the operational RAM-guard correction were pushed at
  `5aefb3c` and `bc68585`. The fixed cohort is all 750 train-only keeper-class1
  rows: 528 true class-1 predictions and 222 restricted target-`0/2/4` false
  positives over five source-disjoint folds. Validation and test were unused.

## OOF Result

| Role | AUROC | TP retention | FP rejection | FP rejects | Corrections | Harms |
|---|---:|---:|---:|---:|---:|---:|
| Base only | 0.831900 | 0.965909 | 0.229730 | 51 | 46 | 18 |
| Cutout control | 0.826645 | 0.967803 | 0.238739 | 53 | 49 | 17 |
| CutPaste candidate | 0.824913 | 0.964015 | 0.193694 | 43 | 39 | 19 |
| Paired contrast | 0.833871 | 0.965909 | 0.243243 | 54 | 48 | 18 |
| Source deranged | 0.822772 | 0.964015 | 0.238739 | 53 | 48 | 19 |

Candidate AUROC deltas versus base/Cutout/source-deranged are
`-0.006987/-0.001732/+0.002141`. It rejects eight fewer restricted FP than
base and ten fewer than Cutout. It wins both controls in only two of five
folds. Target-`0/2/4` rejection is `28/13/2` of `158/54/10`; aggregate TP
retention passes, but aggregate FP rejection misses `0.25` at `0.193694`.

Nine of 18 mechanism gates fail: absolute and all required relative AUROC
gates, FP rejection, both extra-reject gates, four-of-five fold wins, and
effective rank. The paired contrast does beat Cutout by `+0.007226`, but its
AUROC is only `0.833871`, just `+0.001971` over base and below the candidate
gate. No partial gain compensates for the failed conjunction.

## Mechanism Diagnosis

- Candidate response effective rank is only `3.781693` despite all 40
  dimensions being nonconstant. Cutout and paired-contrast ranks are
  `4.526601/3.980412`; response energy is strongly redundant.
- The strongest single candidate cue is regular CutPaste class-1-versus-rival
  mean-margin delta with oriented AUROC `0.642421`. True class-1 rows are more
  disrupted (`-0.06427`) than false positives (`-0.00610`), so the transform
  mainly measures fragility of genuine class-1 evidence rather than a
  conservative false-positive veto.
- The best paired cues remain weak: regular mean margin/logit deltas reach
  oriented AUROC `0.598041/0.596907`; scar JS response reaches `0.591993`.
  This explains the small paired-control gain without supporting a new branch.
- The clean probability replay has exact argmax but maximum probability error
  `2.261996e-5`, narrowly above the prospective `2e-5` gate. Do not repair the
  tolerance after seeing metrics. The independent mechanism rejection already
  fixes the decision.
- Geometry is not the cause: all 6,000 records pass support/range/IoU/identity
  checks, second-process geometry replay is exact at SHA
  `4a8871cf...bdbfc`, and the formal 12-row sheet passes manual review. The
  sheet SHA `4ec2c8ed...8d59d6e` exactly matches the prospective final preview.

Close CutPaste/CutPaste-Scar area, aspect, jitter, draw, seed, scar geometry,
Cutout-fill, threshold, readout, descriptor-subset, response-voting, and
test-time-augmentation neighbors on this keeper. Prior failed class-label
SnapMix/counterexample paste routes remain closed as well. A future method must
add a learned class-conditional surface representation, not repackage frozen
perturbation sensitivity.

## Replay And Retention

- Requested/effective workers are `4/4`, effective paired GPU batch is `64`,
  extraction takes `148.240 s`, peak allocation is `1.13453 GiB`, and model
  state is bit-identical. Focused/full tests pass `14/14` and `1644/1644`.
- Internal and external readout replay have zero analysis/state/refit-score
  difference; applied score error is `2.22e-16` with exact actions. Final
  summary/manifest SHAs are `887a5132...742ee`/`8edf28ea...15d3f2`.
- Preserve all ten files (`6.96 MiB`) in
  `runs/audit_cutpaste_surface_response_a0_20260720`; there is no checkpoint,
  optimizer, smoke, or probe artifact to delete.
- Read-only retention covers 796 run directories and all 50 valid object-
  schema compaction manifests. All 219 compacted originals remain absent,
  `deleted_anything=false`, and `blockers=[]` at summary SHA
  `38b3643f...c60216`.
- Keeper/current-best command/history hashes remain
  `1f49d577...2677`/`36b9aa1a...0faf`/`39bd2879...8f53`. Stage B, trainer,
  short pair, full train, and command-promotion flags are all false.
