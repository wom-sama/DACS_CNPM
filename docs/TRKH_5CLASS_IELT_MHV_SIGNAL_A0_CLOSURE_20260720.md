# IELT Multi-Head Voting Signal A0 Closure (2026-07-20)

## Decision

Reject the exact area-scaled IELT Multi-Head Voting signal on the current
no-pretrain keeper. The sole source-disjoint train-only A0 reaches absolute
direction AUROC `0.685299`, but loses `0.136350` AUROC to the keeper-logit-only
control, loses to both raw and dephased voting, retains only `77.67%` of true
class-1 predictions, and produces more harms than corrections.

Do not integrate MHV, CLR, dynamic layer quotas, or a light refinement head;
do not run lighting extraction, validation, test, smoke, probe, or full train;
and do not update current-best commands from this result.

## Locked Source And Scope

- The accepted source is Xu et al., IEEE TMM 2023, DOI
  `10.1109/TMM.2023.3244340`, with the authors' MIT repository pinned at
  commit/tree `b185111...5eec`/`713ed85...fd050`. Official `IELT.py` SHA is
  `cab0b8b...f7de` and protocol SHA is `0e4984a0...e58`.
- The released recipe is adverse evidence for direct adoption: ViT-B/16,
  ImageNet-21k pretraining loaded unconditionally, input 448, 50 epochs, and a
  ten-epoch dynamic-selection warm-up. A0 therefore tests only the distinct
  per-head top-k voting equation, not the full pretrained IELT stack.
- Infrastructure was committed and pushed at `ec2760d` before formal metric
  access. Auditor/test/launcher SHAs are `6117e082...bbccf`/
  `80d074eb...4753f`/`5724e304...3053`.
- The fixed cohort is 607 immutable `yolo_f/train` objects from CIDT folds
  1..4: 421 true class-1 predictions and 186 restricted `0/2/4 -> 1` false
  positives. Ordered sample-index SHA is `a2689d1...38bd`; no validation or
  test row, label, pixel, prediction, or metric is opened.

## Structural Incident

All implementation-sensitive checks pass: ordinary and hooked probabilities
are bit-exact (`0.0` maximum difference), model state SHA remains
`522b95f1...a38c`, bbox bytes match `e9b2143c...2f6b`, token lineages are
exactly `256/218/167`, every one of 14,568 NumPy-oracle row/role/block checks is
exact, all 20 readouts converge, hooks return to zero, and replay is exact.

One prospective structural declaration fails. The batch-32 BF16 formal has two
CIDT argmax mismatches instead of the one locked exception:

- sample 2446, target 0: cached `p1-p0=+0.001412`; formal reaches an exact
  `p0=p1=0.251658` tie and deterministic argmax 0;
- sample 3657, target 2: cached `p1-p2=+0.000177`; formal reaches
  `p1-p2=-0.000902` and argmax 2, the already known exception.

This is batch-shape-sensitive BF16 near-tie drift, not hook corruption: normal
and hooked forwards agree exactly. Do not amend the exception set or rerun
after seeing the result. The mechanism is independently rejected by ten of 15
mechanism gates, so the structural incident cannot rescue or reverse the
decision.

## OOF Result

| Role | AUROC | TP retention | FP rejection | TP breaks | FP rejects |
|---|---:|---:|---:|---:|---:|
| Base logits only | 0.821648 | 0.964371 | 0.215054 | 15 | 40 |
| CLS-mean smooth | 0.670370 | 0.790974 | 0.408602 | 88 | 76 |
| Vote raw | 0.711248 | 0.762470 | 0.500000 | 100 | 93 |
| Vote dephased smooth | 0.713585 | 0.769596 | 0.473118 | 97 | 88 |
| MHV smooth | 0.685299 | 0.776722 | 0.381720 | 94 | 71 |

Candidate AUROC deltas versus base/CLS/raw/dephased are
`-0.136350/+0.014929/-0.025949/-0.028286`. MHV wins zero of four held folds
against every control; fold TP retention is only
`0.7589/0.7800/0.7921/0.7778`. Relative to base-only actions, it changes 152
rows with `52` corrections and `100` harms.

Although MHV rejects 71 restricted false positives, it breaks 94 true
class-1 predictions. Rejection by target `0/2/4` is `49/19/3`; this is broad
class-1 suppression rather than a precision-safe veto. Base logits alone keep
406 of 421 true positives while MHV keeps only 327.

## Mechanism Diagnosis

- Mean inter-head top-k Jaccard is only `0.050005`, confirming that the heads
  do not form the consensus assumed by MHV. Smoothed-vote entropy is high at
  `0.851411`.
- Dephasing improves AUROC from `0.685299` to `0.713585` and selector alignment
  from `0.436358` to `0.459235`. Preserving genuine head spatial agreement is
  therefore not the useful component.
- Raw voting also beats MHV and rejects 22 more FP, showing that the official
  fixed spatial kernel erases useful local differences while not protecting
  true class 1.
- Candidate effective rank is healthy (`65.4298`) and every block group is
  nonconstant. The rejection is selectivity, not descriptor collapse.
- Ten of 15 mechanism gates fail: every required control gain, TP safety,
  FP-over-control margin, fold win, worst-fold retention, and dephased
  alignment gate. Only absolute AUROC, absolute FP rejection, rank,
  nonconstant-block, and top-tie gates pass.

The fixed 12-row sheet is complete and geometrically readable. Candidate maps
usually follow broad fruit surface or boundary hotspots and do not isolate a
class-1-specific lesion cue. Two extremely flat transformed crops correctly
show a narrow valid image band surrounded by padding; this does not explain
the OOF failure. Manual inspection cannot override the automated rejection.

Close vote-count/quota/kernel/layer/head-subset grids, MHV thresholds and
routers, learned/light refinement around this selector, CLR token
reconstruction, dynamic layer selection, and full IELT neighbors on this
keeper. Future attention work must introduce a new class-conditional signal,
not aggregate the same low-overlap CLS heads more aggressively.

## Runtime, Replay, And Retention

- Clean extraction takes `24.069 s` (`25.219` rows/s) with requested/effective
  workers `4/4`; peak RTX 4060 allocation is `0.36496 GiB`. No competing
  Python/TensorRT/FFmpeg process remains afterward.
- Summary/manifest/cache/contact-sheet SHAs are
  `ea1247bb...f91c`/`8a418876...fbe4`/`0e29ba9a...3a89`/
  `d8e70bf7...e01a`. External replay has analysis/state difference `0.0`,
  serialized score difference `2.22e-16`, and exact actions.
- Preserve all eight manifest payloads (about 8.0 MiB plus manifest). There is
  no checkpoint, optimizer, smoke, or probe artifact to delete.
- Pycompile/pyflakes, PowerShell parse, focused tests `13/13`, and full pytest
  `1630/1630` pass. Read-only retention covers 794 run directories and all 50
  valid object-schema manifests; all 219 compacted originals remain absent,
  `deleted_anything=false`, and `blockers=[]` at summary SHA
  `ce5f71fa...dd78e`.
- Keeper/current-best command/history hashes remain
  `1f49d577...2677`/`36b9aa1a...0faf`/`39bd2879...8f53`. Every training and
  command-promotion authorization remains false.
