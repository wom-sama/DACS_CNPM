# TRKH 5-Class FP-Informed Class-Conditional Augmentation Protocol

Date: 2026-07-12

## Decision scope

This is one fixed, no-test, no-raw-data-edit experiment on the current
`yolo_f` TRKH keeper. It asks whether reducing stochastic augmentation only
for class 1 can keep its subtle surface evidence compact enough to reduce
`0/2/4 -> 1` false positives without losing true class-1 recall.

The candidate is not generic weaker augmentation and is not a nearby
RandAugment/color-consistency retry. Classes `0,2,3,4`, the sampler, losses,
teacher, bbox prior, optimizer, seed, batch budget, and validation pipeline
remain identical to the matched control.

## Primary-source basis

- Kireev et al., *Understanding the Detrimental Class-level Effects of Data
  Augmentation*, NeurIPS 2023:
  <https://proceedings.neurips.cc/paper_files/paper/2023/file/38c05a5410a6ab7eeeb26c9dbebbc41b-Paper-Conference.pdf>
- NeurIPS supplement, especially the FP/FN definition and the failed
  fully-trained-checkpoint fine-tuning result:
  <https://proceedings.neurips.cc/paper_files/paper/2023/file/38c05a5410a6ab7eeeb26c9dbebbc41b-Supplemental-Conference.pdf>

The paper selects classes whose false-positive count grows under stronger
augmentation and changes augmentation strength for those classes. Local TRKH
evidence points in the same direction: light RandAugment and hflip instability
both widened class-1 false positives. The paper also reports that applying the
policy only after a model is fully trained did not recover affected classes;
therefore this protocol uses the current early two-epoch keeper only as a
short readiness smoke and requires a from-initialization probe if it passes.

## No-repeat decision for AUM and forgetting events

AUM and example-forgetting are not the candidate in this round:

- AUM averages assigned-class logit minus the largest competing logit over
  training and uses intentionally mislabeled threshold samples to determine a
  removal threshold:
  <https://proceedings.neurips.cc/paper_files/paper/2020/hash/c6102b3727b2a7d8b1bb6981147081ef-Abstract.html>
- Forgetting events require the ordered per-example sequence of correct to
  incorrect transitions:
  <https://www.microsoft.com/en-us/research/publication/an-empirical-study-of-example-forgetting-during-deep-neural-network-learning/>
- Official Dataset Cartography expects per-epoch logits and already includes
  forgetfulness as a selection metric:
  <https://github.com/allenai/cartography>

The current `DataCartographyRecorder` stores cumulative probability,
correctness, and loss aggregates rather than the required ordered epoch
trajectory. More importantly, TRKH has already rejected cartography policies,
Cleanlab policies, GroupDRO, ELR, small-loss selection, self-paced learning,
and co-teaching on this dataset. Reconstructing an approximate AUM and using
it for filtering/downweighting would repeat that closed policy family without
a new class-1-positive target. No AUM run or dataset filtering is permitted by
this protocol.

## Locked candidate

- Data: explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`.
- Focus class: `1`.
- Per-class augmentation scales: `1.0,0.5,1.0,1.0,1.0`.
- The `0.5` factor reduces only stochastic spatial/local transform magnitude
  or probability: affine, horizontal/vertical flip, rotate-90, random scale
  crop, foreground crop, local exposure, and obstacle.
- Illumination normalization and background suppression remain unchanged.
- Photometric color-jitter scaling remains disabled for this candidate.
- RandAugment, random erasing, batch mix, mosaic, mixup, CutMix, copy-paste,
  and foreground/background mix remain disabled exactly as in the keeper.
- The feature is default-off. An empty scale string must preserve old behavior.
- Validation and test transforms never consume the train-only class scale.

No scale, class set, operator set, seed, LR, loss, or threshold sweep is
allowed in this round.

## Contract preflight

Before GPU training:

1. CLI/config/launcher round-trip the exact five scales.
2. Empty policy reproduces legacy transform behavior for identical RNG state.
3. Classes `0,2,3,4` remain identical between control and candidate for the
   same image, target, and RNG state.
4. Class 1 receives scale `0.5`, keeps valid labels/bboxes/pad masks, and shows
   lower transform magnitude/exposure than scale `1.0` in deterministic tests.
5. Invalid length, non-finite values, or values outside `[0.1, 1.0]` fail
   closed for the explicit reduction policy.
6. PowerShell parsing, V8 dry-run, compileall, focused tests, and current-best
   preflight must pass. Current-best command and checkpoint hashes must remain
   unchanged.

## Matched readiness smoke

Run two one-epoch, 60-train-batch continuations from the exact keeper with the
same seed and full `yolo_f/val=2606`:

- control: empty explicit scale policy;
- candidate: `1.0,0.5,1.0,1.0,1.0`.

Both use seed `42`, batch `32`, accumulation `2`, LR `1e-5`, minimum LR
`1e-6`, scheduler horizon `1`, warmup `0`, and patience `3`.

Both runs must use `SkipFinalTest`, full validation, the original
teacher-focus-binary/bbox/pair-route recipe, and independent FP32 checkpoint
reload. Candidate smoke permission requires all of:

- independent macro F1 `>= 0.882925`;
- independent class-1 F1 `>= 0.678261`;
- class-1 F1 at least `+0.003` above the matched control;
- class-1 precision `>= 0.603093`;
- class-1 recall `>= 0.761589`;
- versus the keeper: class-1 FN rescues `>=` TP breaks;
- versus the keeper: class-1 FP removals `>=` FP creations;
- total corrections `>=` harms;
- no source mismatch, non-finite value, test access, or raw-data write.

Every smoke requires full confusion/transition review, boundary audit,
architecture trace, and the same balanced class-1 XAI/robustness suite used for
the keeper. A failed check rejects the method before a probe.

## Probe and promotion

Only if the matched smoke passes, run one fixed from-initialization
`120 batches x 2 epochs` probe with the candidate policy and the otherwise
exact keeper recipe. It must beat the locked independent keeper and reach the
next class-1 validation milestone (`>=0.70`) before any longer train.

The current-best full pipeline, export, engine, and video commands are updated
only after an independently reloaded locked-validation winner. Test remains
closed until the normal promotion gate authorizes the one final audit.
