# TRKH 5-Class Natural-Prior Continuation Protocol

Date locked: 2026-07-15

## Question

The completed scratch run used strict class-balanced exposure on every epoch:

- natural train counts: `[1941, 541, 1920, 2520, 2293]`;
- realized epoch exposure: `[1843, 1844, 1844, 1843, 1842]`;
- class 1 was therefore repeated about `3.41x` per epoch;
- independent validation class-1 precision/recall/F1 was
  `0.535865/0.841060/0.654639`, with `110` false positives.

This protocol tests one narrow hypothesis: after the scratch model has learned
for 20 selected epochs, a short full-model continuation under the natural train
prior may remove class-1 false positives without destroying the useful class-1
recall learned by the balanced phase.

This is not a square-root sampler sweep, a new loss, post-hoc thresholding,
classifier-only cRT, BBN/readout fusion, or a dataset edit.

## Primary Sources

- Kang et al., *Decoupling Representation and Classifier for Long-Tailed
  Recognition*, ICLR 2020:
  https://openreview.net/pdf?id=r1gRTCVFvB
  The paper reports that natural instance-balanced sampling can learn strong
  representations and separates representation learning from class balancing.
- Zhou et al., *BBN: Bilateral-Branch Network with Cumulative Learning for
  Long-Tailed Visual Recognition*, CVPR 2020:
  https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_BBN_Bilateral-Branch_Network_With_Cumulative_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2020_paper.pdf
  The paper reports that rebalancing can improve classifier learning while
  damaging feature representation, and keeps a conventional natural sampler as
  a distinct representation path.

The local experiment does not claim to reproduce either paper. Their findings
only motivate the locked sampler intervention.

## Locked Provenance

- Data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Scratch launcher arguments:
  `runs/full_v8_yolof_randominit_30e_20260714_105524/launcher_args.json`,
  SHA-256
  `a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6`.
- Scratch resolved configuration:
  `runs/full_v8_yolof_randominit_30e_20260714_105524/resolved_config.json`,
  SHA-256
  `c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97`.
- Scratch best checkpoint, selected at epoch 20:
  `runs/full_v8_yolof_randominit_30e_20260714_105524/checkpoints/best.pt`,
  SHA-256
  `f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549`.
- Scratch independent FP32 validation predictions, SHA-256
  `079ff545df0d0272fd2640453715621843da2b16653293d52b4fc1c2688ff0d8`.
- Keeper validation predictions, SHA-256
  `9a3ee6b7ee1ab53ad8c140a864789d0dc29029ac17c78b55bca594c6af66a088`.
- Current command packet, SHA-256
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`.

Raw data, labels, bboxes, split membership, keeper, scratch checkpoint, and
current-best commands are immutable.

## Matched Runs

Both roles replay every source argument and differ only in sampler mode.

- Common: seed `42`, deterministic mode, full-model update, no pretraining,
  fresh optimizer/scheduler/scaler/epoch state, source checkpoint epoch 20,
  `2` adaptation epochs, `60` train batches per epoch, full validation `2606`,
  batch `32`, accumulation `2`, LR `8e-5`, minimum LR `1e-6`, scheduler horizon
  `10`, warmup `1`, patience `2`, and test disabled.
- Control: source strict-balanced sampler remains enabled.
- Candidate: add exactly `--disable-balanced-epoch-sampling`, producing the
  natural random train sampler.

The effective training age is at most `20 + 2 = 22` epochs, below the project
limit of 30. No sampler power, LR, seed, budget, loss, or augmentation sweep is
allowed after validation is read.

## Clean Validation Gates

All checks must pass simultaneously:

1. All four prediction tables align exactly on the locked `2606` validation
   objects and class support `[549, 151, 544, 712, 650]`.
2. Control does not collapse versus raw scratch: macro F1 drop at most `0.005`,
   class-1 F1 drop at most `0.010`, and recall drop at most `0.010`.
3. Candidate class-1 precision improves control by at least `0.020` and raw
   scratch by at least `0.020`.
4. Candidate class-1 F1 improves control and raw scratch by at least `0.010`,
   and reaches the project milestone `0.70`.
5. Candidate class-1 recall is no more than `0.010` below raw scratch.
6. Candidate macro F1 is not below control or raw scratch.
7. Candidate removes at least `8` restricted `0/2/4 -> 1` false positives net,
   corrections exceed harms, and class-1 FN rescues are at least TP breaks.
8. No nonfocus class F1 falls by more than `0.010` versus control.
9. Candidate reaches or exceeds keeper reload macro/class-1 F1
   `0.882925/0.678261`; this is necessary but not sufficient for promotion.
10. Resolved model/loss/augmentation settings match, test summaries are null,
    and sampler telemetry proves `strict_balanced` versus `random`.

Precision gained by suppressing class-1 support is a failure even if false
positives decrease.

## Required Post-Smoke Audit

Regardless of the clean gate result, run:

- class-confusion and boundary forensics for both roles;
- clean plus dim, bright, low-contrast, and occlusion robustness;
- exact changed-case FP32 native attention, rollout, gradient rollout, Grad-CAM,
  and perturbation probes for both checkpoints;
- paired XAI reconciliation and source-group transition accounting;
- architecture trace and artifact-retention audit.

Test remains closed. A failure closes this exact natural-prior continuation and
forbids nearby sampler/LR/budget sweeps. A complete pass permits one longer
no-test validation probe, not immediate full-train or command promotion.
