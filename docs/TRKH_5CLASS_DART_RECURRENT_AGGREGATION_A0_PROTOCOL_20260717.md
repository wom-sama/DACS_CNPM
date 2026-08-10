# TRKH 5-Class DART Recurrent Aggregation A0 Protocol - 2026-07-17

## Decision scope

This protocol locks one train-only, equal-update A0 before any production
trainer, launcher, full train, validation, or test integration. The question is:

> Can recurrent same-basin averaging of four augmentation-specialized TRKH
> branches internalize the precision-safe complement seen between existing
> models better than both one mixed-training trajectory and one final-only
> checkpoint average?

All variants start from the exact current keeper. A0 may authorize one later
matched validation smoke only if every structural, clean holdout, illumination,
loss-barrier, resource, export, and independent-replay gate passes. It cannot
authorize test, a full train, or a current-best command update.

## Primary authority and provenance

- Accepted paper: Jain et al., *DART: Diversify-Aggregate-Repeat Training
  Improves Generalization of Neural Networks*, CVPR 2023, pages 16048-16059:
  `https://openaccess.thecvf.com/content/CVPR2023/html/Jain_DART_Diversify-Aggregate-Repeat_Training_Improves_Generalization_of_Neural_Networks_CVPR_2023_paper.html`.
- Local accepted-paper PDF:
  `D:\DataAI\external_sources\papers\Jain_DART_CVPR2023_accepted.pdf`.
- Paper SHA-256:
  `6d941daf99495df5c1e88a6a4d53054c491ef14f34e259c443ac8db0c681b06d`.
- Author repository: `https://github.com/val-iisc/DART`.
- Local source: `D:\DataAI\external_sources\official\dart-cvpr2023`.
- Locked commit/tree:
  `62274e83d2f08eb416db61d0957476c53fde9361` /
  `83d56bedfc1159eb16ecbf8b4363e014d15c6f75`.
- Official `domainbed/trainer.py` SHA-256:
  `02cfea225325f94d6d68671107f735b708065a57aa9a6ced316be080a8917aa6`.
- Official MIT `LICENSE` SHA-256:
  `3dbd7ffa11dde98c571ecf13468792d03d0185936f225de9cc99240b01dc3fcf`.
- GitHub observation on 2026-07-17: 12 stars, 3 forks, MIT, not archived.
  Popularity is context only; the accepted paper and direct TRKH evidence
  control the decision.
- No DART, ImageNet, ResNet, CLIP, SWAD, EMA, or other pretrained weight is
  permitted.

Algorithm 1 first reaches a common solution, copies it into `M` branches,
trains each branch on a distinct augmentation/domain dataset, averages model
weights every fixed interval, broadcasts the average, and repeats. The paper
attributes the benefit to repeated low-barrier aggregation, not merely a final
model soup. Its fine-grained table reports gains on CUB-200 and Stanford Cars,
but those results do not establish efficacy on mango classification.

The official DomainBed code hard-codes four models, averages model weights at
`inter_freq`, and leaves branch optimizer states unaveraged. It instantiates
the four DG algorithms separately because that path normally uses a common
pretrained backbone. A0 follows the paper's stronger common-point contract by
copying the exact TRKH keeper bit-for-bit into all branches, while preserving
the official optimizer-state behavior.

## Why this is not the closed model-soup route

- The closed `trkh.tools.average_checkpoints` experiment averaged completed
  checkpoints after independent optimization. It did not broadcast an average
  during training and did not keep branches in one repeatedly synchronized
  basin.
- The keeper and 2026-07-14 scratch complement use different initializations.
  Their post-hoc weight and feature-basis combinations exposed permutation and
  basis compatibility problems. DART A0 starts every branch from one exact
  keeper state and synchronizes twice before deployment.
- The `single_final_average` A0 control uses the same common initialization,
  data, augmentations, update count, optimizer, and final averaging equation,
  but omits the intermediate broadcast. DART must beat this control to support
  a claim about recurrence.
- A0 does not reopen output ensembling, calibration, teacher distillation,
  feature stitching, router, SWA/EMA, augmentation sweeps, or any closed model
  architecture family.

## Rejected screen alternatives

- CVPR-2023 DropKey is equation-distinct from R-Drop and ADL, but its main
  effect is smoother, more global attention and better random-occlusion
  robustness. Current TRKH audits repeatedly show negligible far-background
  causal effect, already high foreground mass, and excessive class-1 false
  positives. It is therefore not selected ahead of DART and must not be added
  to this A0.
- NeurIPS-2021 SOAP/AP optimization remains global ranking. Earlier class-1
  AUC and partial-AUC objectives widened false positives and reduced precision.
- SimAM, CAL, INTR, CrossViT, LCR, and noisy-label SNSCL routes are already
  closed, overlap earlier families, lack a reusable official license, or lack
  official source as recorded in the journal.

## Locked data and split

- Dataset declaration:
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`.
- Dataset declaration SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Only `train=9215` may be constructed. Validation and test are forbidden.
- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher declaration SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- Train-only source-fold declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`.
- CIDT summary/prediction SHAs:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad` /
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Fold 0 is the untouched source-disjoint holdout: 1,843 rows.
- Folds 1, 2, 3, and 4 supply the four fit branches. No source stem may occur
  in more than one branch or in holdout.
- For each fit fold, derive a 256-row class quota from that fold's complete
  natural class distribution by largest remainder, with lower class index as
  the deterministic fractional-remainder tie break. This is proportional
  subsampling, not class balancing or oversampling.
- Group rows by exact `source_stem`. Sort groups by the tuple
  `(SHA256("42|update|fold=<fold>|<source_stem>"), source_stem)`. Traverse once
  and accept a complete group only when none of its five class counts exceeds
  the remaining quota. Stop only at the exact 256-row quota; failure to reach
  it fails preflight. Within a group, order rows by integer `sample_index`.
- The locked branch class counts for folds 1 through 4 are respectively
  `[54,16,51,71,64]`, `[55,15,54,69,63]`, `[56,14,53,69,64]`, and
  `[52,15,54,71,64]`. Their ordered sample-index SHA-256 values are
  `dde3e3836cfb1777cfb40acdaf223e53c6a7d73550f539bd791be1799e48cca2`,
  `6dd694b539c77bbba50caa192b2b3be1023833106bea79be973faa5434b8b41f`,
  `6a53d03930a94fca27729160802e548a6e6413a113f72370b5c7fcb8ec63f83f`,
  and `ee75abd58cb502434612bd3186ee538f770627d94a31ad08a3f6a4ee04bc23d1`.
  The concatenated 1,024-row hash is
  `2279a58ace27fbb5840c31a037a3e9871d6173799279327c8729f9298b572bb7`.
- After excluding update and holdout source stems, repeat the same algorithm
  independently per fit fold with namespace `42|probe|fold=<fold>` and a
  natural-distribution quota of 64 rows. The resulting 256-row clean fit probe
  has class counts `[54,16,52,70,64]`, ordered sample-index SHA-256
  `6490523a2de91848dc06e3ace3f78be5cfb1cb1670a28e05468527eff050c5bd`,
  and zero source overlap with update or holdout rows. It is used only for
  aggregation loss barriers.

No raw dataset file, label, bbox, split, or image may be edited. Generated
manifests and run artifacts are allowed only inside the A0 output directory.

## Locked augmentation domains

All roles use image size 256, pad resize, keeper input mean/std, keeper bbox
metadata semantics, illumination normalization enabled at strength 0.35, no
RandAugment, no class-aware scaling, and no mixup/cutmix/mosaic/copy-paste.

1. `context_geometry`, fold 1:
   RRC scale/probability `0.88/0.35`; affine degrees/translate/min-scale
   `3/0.02/0.96`; hflip/rotate90 `0.50/0.03`; all color jitter, lighting,
   local exposure, obstacle, erasing, and background suppression disabled.
2. `illumination_surface`, fold 2:
   no RRC, affine, flip, rotate, obstacle, or erasing; brightness/contrast/
   saturation/hue `0.08/0.08/0.04/0.01`; lighting `0.20`; local exposure
   probability/strength `0.30/0.25`; keeper desaturate-blur background
   suppression probability `0.80`.
3. `occlusion_parts`, fold 3:
   the geometry settings from role 1; all color jitter, lighting, and local
   exposure disabled; random erasing `0.08`; obstacle probability/max-area
   `0.12/0.08`; keeper background suppression probability `0.80`.
4. `keeper_mixed`, fold 4:
   exact keeper augmentation declaration: RRC `0.88/0.35`, color jitter
   `0.04/0.04/0.02/0.01`, affine `3/0.02/0.96`, hflip `0.50`, rotate90
   `0.03`, local exposure `0.15/0.25`, obstacle `0.04/0.08`, and keeper
   background suppression probability `0.80`.

Materialize each role's eight transformed batches once into an immutable CPU
cache before model training. Record per-batch sample-index order and SHA-256 of
contiguous transformed image, bbox, mask, and target bytes, then replay those
exact tensors for all three variants. The fixed clean fit probe uses the normal
pad-resize evaluation transform without augmentation. A byte, order, or target
mismatch fails A0.

## Equal-update training variants

All variants use full-model FP32 AdamW, LR `1e-5`, betas `0.9/0.999`, weight
decay `0.05`, batch 32, deterministic algorithms, plain five-class cross
entropy, no scheduler, no AMP, no clipping, no EMA, and no auxiliary loss.
Every variant receives exactly 32 optimizer updates and the same 1,024
transformed examples.

Before every logical `(role, step)` forward, reset Python, NumPy, CPU Torch,
and CUDA Torch RNGs to the same SHA-256-derived seed for all variants. This
locks dropout and drop-path masks for corresponding batch updates without
forcing different logical steps to share a mask. Record the derived seed map
and initial model-state hash before any update; any mismatch fails A0.

1. `mixed_control`: one keeper copy. For rounds 1 through 8, consume one batch
   from each role in fixed role order, giving 32 interleaved updates.
2. `single_final_average`: four bit-identical keeper copies. Each branch takes
   its eight role-specific updates. Average all floating model-state tensors in
   FP32 once after step 8; nonfloating buffers must be bit-identical. This is
   the matched final-only soup control.
3. `dart_recurrent`: four bit-identical keeper copies. Cycle 1 gives every
   branch its first four updates, averages model weights in FP32, and broadcasts
   the exact average to all branches. Cycle 2 gives every branch its remaining
   four updates and repeats average plus broadcast. Branch AdamW states remain
   branch-local and are not averaged or reset between cycles.

The deployed candidate is the second recurrent average. It has the exact keeper
architecture, parameter names, parameter count, and one-forward inference cost.
No branch ensemble is permitted at evaluation or deployment.

## Aggregation and basin diagnostics

Before every aggregation, record branch state hashes, parameter-group movement,
pairwise parameter-delta cosine, pairwise distance, and fit-probe CE. Average
floating tensors in FP32. Every nonfloating tensor must match bit-for-bit or A0
fails; copying a differing integer buffer is forbidden.

After broadcast, all four model-state hashes must match the aggregate hash.
Optimizer states must remain distinct and retain their step counters. On the
fixed fit probe, define:

`barrier = CE(average weights) - mean(CE(branch endpoint))`.

Both recurrent cycles and the final-only control require finite barrier
`<=0.02`. Final recurrent barrier must be no larger than final-only barrier.
At least one pair of branches must have parameter-delta cosine below `0.995`,
all cosines must exceed `-0.50`, and every required parameter group must receive
finite nonzero gradients and move. These gates require useful diversity without
permutation-scale incompatibility.

## Train-only behavior gates

Evaluate raw argmax predictions on all 1,843 clean fold-0 rows. No threshold,
temperature, bias, calibration, or model selection is fitted. All checks are
conjunctive for `dart_recurrent` versus `mixed_control`:

1. macro F1 delta `>=0`;
2. class-1 F1 delta `>=0.005`;
3. class-1 precision delta `>=0.005`;
4. class-1 recall delta `>=-0.005`;
5. net restricted `{0,2,4}->1` false-positive reduction `>=2`;
6. class-1 FN rescues are at least class-1 TP breaks;
7. total corrections exceed harms;
8. maximum nonfocus per-class F1 drop `<=0.010`;
9. predicted class-1 support is at least 95% of control support.

To establish the Repeat claim, recurrent versus `single_final_average` also
requires class-1 F1 delta `>=0.002`, nonnegative class-1 precision and recall
deltas, no increase in restricted class-1 FP, and corrections at least harms.
If final-only averaging is better, DART is rejected rather than relabeled as a
model-soup success.

## Illumination and deployment gates

With all weights frozen, repeat fold-0 evaluation under the established dim,
bright, and low-contrast transforms. Relative to mixed control:

- every condition has class-1 F1 delta `>=-0.010` and recall delta
  `>=-0.015`;
- at least two conditions have nonnegative class-1 precision delta;
- no condition increases restricted class-1 FP;
- aggregate class-1 FN rescues are at least TP breaks.

Structural and deployment checks additionally require:

- exact paper/source/license/commit/tree/protocol/input hashes and clean
  official-source worktree;
- zero validation/test construction and zero fit/holdout source overlap;
- exact common initialization, equal update counts, batch-byte parity, and
  aggregate broadcast hashes;
- unchanged model schema and parameter count;
- standard ONNX export, no custom domain, ONNX Runtime max error `<=1e-5`,
  finite outputs, and matching argmax;
- candidate/native FP32 batch-32 inference runtime ratio `<=1.02` and peak
  inference-memory ratio `<=1.02` after warm-up;
- peak allocated training memory `<=4.5 GiB` using sequential branch
  execution; no four-model simultaneous CUDA residency;
- no timing decision while an owned Python/TRT process is active. Record the
  process list and `nvidia-smi`; never terminate an unknown process.

An independent CSV replay must reproduce all clean and illumination confusion,
transition, and gate values exactly within `1e-12`.

## Visual audit and stop rules

If and only if all nonvisual gates pass, retain the recurrent checkpoint and
build a separate 16-case train-holdout XAI protocol before opening validation.
The cohort must prioritize class-1 TP breaks, restricted-FP removals/additions,
FN rescues, corrections, and harms. It must compare native attention,
gradient-rollout provenance, stem Grad-CAM, bbox foreground/border mass, and
background blur/gray versus object-desaturation perturbations for mixed control,
final-only average, and recurrent DART.

Any source, split, batch-parity, optimization, aggregation, barrier, behavior,
illumination, export, runtime, memory, replay, or later XAI failure closes the
route. On failure:

- do not run validation, test, a longer A0, a smoke, probe, or full train;
- do not sweep branch count, role/fold mapping, augmentation values, cycle
  count, aggregation interval, LR, optimizer, loss, seed, fold, budget, or
  average weights;
- do not combine DART with DropKey, SWAD, EMA, SAM, pretrained weights, closed
  architecture modules, distillation, output routing, or feature stitching;
- do not update current-best commands or command history.

On a full pass, write and hash a separate matched validation-smoke/XAI protocol
before trainer or launcher integration. Current-best command/history lock-time
SHAs remain
`36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
`39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

## Preflight erratum

The first clean preflight rejected the initial protocol SHA
`a9b2bd96a257bd00aa2a228d22b3072c22a044225c29b6adbc44e5c663a11e55`
before CUDA or output creation. The CIDT prediction SHA transcription omitted
the second `fe` in `...429bfefe1...`. The corrected value above matches the
existing file and all earlier authoritative CIDT auditors. No cohort, row,
augmentation, optimization, aggregation, metric, threshold, or stop rule
changed, and no formal measurement was observed before this erratum.
