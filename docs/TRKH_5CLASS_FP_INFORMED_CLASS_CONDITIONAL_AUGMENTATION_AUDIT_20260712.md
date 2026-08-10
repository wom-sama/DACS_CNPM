# TRKH 5-Class FP-Informed Class-Conditional Augmentation Audit - 2026-07-12

## Decision

Reject the locked class-1 augmentation-reduction candidate after the matched
60-batch smoke. Do not open the from-initialization probe, test split, or
current-best command promotion. Keep the implementation default-off for exact
reproduction, but do not sweep class-1 scales, transform subsets, other class
sets, or nearby learning rates on the current keeper.

The locked protocol is
`docs/TRKH_5CLASS_FP_INFORMED_CLASS_CONDITIONAL_AUGMENTATION_PROTOCOL_20260712.md`.
The candidate used train-only scales `1.0,0.5,1.0,1.0,1.0`; validation stayed
at `1.0` for all classes. Raw data was not edited and test was not opened.

## Research And No-Repeat Gate

The experiment was fixed before training from Kireev et al., *Understanding
the Detrimental Class-level Effects of Data Augmentation*, NeurIPS 2023, and
its supplement. The local hypothesis was that spatial/local augmentation was
too destructive for subtle class-1 surface evidence. Only class 1 received a
0.5 probability/magnitude factor for affine, flips, rotate-90, random scale
crop, foreground crop, local exposure, and obstacle. Color photometric
scaling, illumination normalization, background suppression, sampling,
losses, teacher, architecture, and validation transforms were unchanged.

AUM and forgetting-event filtering were excluded before code. They require
ordered per-example epoch trajectories that the current cumulative
`DataCartographyRecorder` does not retain. Approximate filtering would also
repeat already rejected cartography, Cleanlab, GroupDRO, ELR, small-loss,
self-paced, and co-teaching policy families without a new class-1-positive
target.

## Implementation Contract

- Added a default-empty five-class scale policy to config, CLI, YOLO and
  classification-folder train datasets, transforms, and the V8 launcher.
- Empty policy and explicit scale 1.0 preserve legacy RNG behavior.
- Validation and test never consume the train-only scale policy.
- Invalid length, non-finite values, and values outside `[0.1,1.0]` fail
  closed.
- Focused tests, full pytest, PowerShell parse, candidate dry-run, and the
  existing current-best preflight passed before GPU allocation.
- Both traces resolve the same scratch TRKH architecture, seed 42, five train
  samples, and exact `yolo_f` YAML. The only experiment-level difference is
  the declared train augmentation scale.

## Independent Matched Smoke

Both runs resumed the exact keeper for one epoch, 60 train batches, batch 32,
accumulation 2, LR `1e-5`, scheduler horizon 1, warmup 0, patience 3, full
`yolo_f/val=2606`, and `SkipFinalTest`. The deciding values below come from
independent FP32 checkpoint reloads.

| Metric | Keeper | Matched control | Candidate | Candidate gate |
| --- | ---: | ---: | ---: | ---: |
| Macro F1 | 0.882925 | 0.882725 | 0.881795 | >=0.882925 |
| Class-1 F1 | 0.678261 | 0.680233 | 0.676385 | >=0.678261 and control+0.003 |
| Class-1 precision | 0.603093 | 0.606218 | 0.604167 | >=0.603093 |
| Class-1 recall | 0.774834 | 0.774834 | 0.768212 | >=0.761589 |
| Class-1 TP | 117 | 117 | 116 | diagnostic |
| Class-1 FP | 77 | 76 | 76 | diagnostic |

Candidate confusion was:

```text
[[486, 50,  2,  0, 11],
 [ 18,116, 12,  0,  5],
 [  6, 10,513,  8,  7],
 [  0,  4, 53,653,  2],
 [  8, 12,  4,  1,625]]
```

The matched control differs from the candidate on only one validation row:
candidate breaks a correct class-1 prediction into class 2. Candidate versus
the keeper changes 13 rows with `5/7/1` corrections/harms/neutral, removes and
creates class-1 false positives `2/1`, and rescues/breaks class-1 FN/TP `1/2`.
Its class-1 delta direction AUROC versus keeper is `0.605042`, but this is a
validation diagnostic over only 111 keeper class-1 FN/FP rows and does not
authorize routing or threshold fitting.

Five locked checks fail:

1. candidate macro F1 is below the keeper;
2. candidate class-1 F1 is below the keeper;
3. candidate class-1 F1 is below matched control plus 0.003;
4. FN rescues are fewer than TP breaks;
5. corrections are fewer than harms.

Precision, recall floor, FP remove/create direction, support, metadata, and
validation-only leakage checks pass. The failed checks are sufficient to
reject before the predeclared probe.

## Boundary, XAI, And Trace Audit

The candidate boundary scan selected 43 capped validation rows: 12 class-1
FN, 12 class-1 FP, 12 low-margin errors, and 7 low-margin correct boundary
rows. Dominant class-1 buckets remain `1->0`, `1->2`, `0->1`, `2->1`, and
`4->1`, with both dark/shadow and bright/glare cases.

Matched FP32 XAI used the same 13 keeper-relative changed `sample_index` rows
for control and candidate, all methods, bbox prior, and robustness probes.
Candidate aggregate values were:

- attention foreground/background/border mass
  `0.923821/0.076179/0.133131`;
- grad-rollout foreground/background/border mass
  `0.960623/0.039377/0.183943`;
- Grad-CAM foreground/background/border mass
  `0.846915/0.153085/0.271421`;
- rollout foreground/background/border mass
  `0.924872/0.075128/0.267112`;
- object-desaturation original-prediction drop `0.044803`, versus background
  blur/gray `-0.003545/-0.001058` and center occlusion `0.004747`;
- `12/13` rows were top-2 near ties.

Control and candidate maps are visually and numerically almost identical on
most cases. The decisive matched-control harm is sample 1113: candidate turns
class `1->2` and its Grad-CAM moves from fruit surface/boundary support toward
hands and the image edge. The single keeper FN rescue and two FP removals are
also near-tie movements, not a shared new visual cue. Object desaturation
continues to dominate background interventions, so reducing spatial/local
augmentation did not solve the surface/color boundary or make wide context a
new discriminator.

Architecture traces use the same source image for every class and identical
model shapes. The class-1 trace uses `Image_2273.jpg`, foreground/detail mask
fractions `0.747147/0.634506`, token shapes
`256 -> 218 -> 167` patches, and final shape `1x174x256` in both runs. The
attention-view score and layer-5 prune map are visually identical. This rules
out a launcher, architecture, validation-transform, or trace-selection
mismatch as the explanation for the failed metric.

## Artifacts And Cleanup

Guarded compaction preserved all non-binary matched smoke, independent reload,
transition, boundary, XAI, and architecture-trace evidence at
`runs/evidence_classcondaug_c1s05_rejected_20260712` before deleting any source
root. The retained transition payload contains exact input hashes, metadata
equality checks, gate booleans, and both changed-case CSVs.

- Ten source roots contained 640 files. Compaction copied 636 source files
  totaling `72,627,253` bytes and retained 638 hash-verified payloads after
  adding evidence metadata.
- Four `.pt` checkpoints totaling `348,750,240` bytes were inventoried with
  source hashes but intentionally excluded from compact evidence.
- Payload-manifest SHA-256 is
  `6de15fcea9cc70b2af78bfd5236dd2aede469032d1cd356500070482e100ab0d`.
- Cleanup manifest
  `runs/cleanup_manifest_20260712_classcondaug_c1s05_rejected.json` has SHA-256
  `8a7551ad4f4d3b9cb40ae9482f98897d06208577f58fa21e2481d6da9a168ea8`,
  records all ten absolute source roots, and confirms deletion.
- Observed free-space gain was `422,850,560` bytes. An independent manifest
  replay verified all 638 retained hashes with zero failures.
- Retention passed over 607 run directories and all 29 object compaction
  manifests with `blockers=[]`. No final-test payload is present and the
  evidence declares `test_data_used=false`.

The keeper SHA remains
`1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`;
the current-best command SHA remains
`3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.
Both dataset roots and all protected user paths remain present and untouched.

Closure passed compileall, focused regressions `22/22`, full pytest `825/825`,
and PowerShell parsing for the V8, current-best, engine-export, and video-test
wrappers. The current-best wrapper also passed direct `-PreflightOnly`: 30
epochs, effective batch 64, train-side final test skipped, empty
class-conditional policy, exact keeper checkpoint, and no run directory.

## Closed Variants

Do not sweep class-1 scale 0.6/0.7/0.8/0.9, individual transform toggles,
photometric scaling, other class sets, seed, LR, or run length from this
result. Those variants would optimize against one validation split after a
candidate that already failed the matched control, keeper, recall-direction,
and net-correction gates. Reopen class-dependent augmentation only with a
different predeclared mechanism that independently identifies augmentation-
specific class harm and protects keeper class-1 TP before training.
