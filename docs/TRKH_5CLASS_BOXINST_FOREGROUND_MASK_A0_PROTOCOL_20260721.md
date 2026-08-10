# TRKH 5-Class BoxInst Foreground-Mask A0 Protocol - 2026-07-21

## Status And Decision Boundary

This document prospectively locks one train-only information gate before the
auditor, mask-head implementation, cached feature extraction, or candidate
metric exists. It tests whether a class-agnostic foreground mask learned from
the existing `yolo_f` boxes can expose new class-1 surface evidence without
using mask annotations or modifying either dataset.

Passing A0 does not authorize a full train. It can authorize one default-off
TRKH integration and a matched no-test smoke/probe only after every clean,
causal, robustness, replay, export, resource, and manual-XAI gate passes. A
full train remains limited to at most 30 epochs with patience 3 and requires a
separate validation promotion gate plus a fresh Windows worker/GPU benchmark.

Failure closes the exact family described below. It cannot be repaired by
post-metric changes to mask width, loss weights, color threshold, dilation,
epochs, fold, seed, readout, action threshold, or gate values.

## Research Question

The current keeper already uses wide-context `yolo_f` crops, deterministic bbox
priors, valid padding masks, boundary-band dropout, foreground heuristics, and
CNN/Transformer attention. These mechanisms produce high foreground XAI mass,
but they do not learn a pixelwise object shape. Can the BoxInst projection and
local color-affinity equations learn a tighter, image-dependent foreground
mask that improves class-1 TP/FN versus restricted `0/2/4 -> 1` FP separation
over:

1. the keeper probabilities;
2. uniform valid-image pooling;
3. the rectangular bbox itself;
4. a matched projection-only learned mask;
5. a matched mask trained with spatially dephased LAB affinity; and
6. the same candidate mask shifted away from its image features?

The intended mechanism is not “more foreground attention.” It is causal
alignment between an image-dependent, box-supervised object mask and local
surface features before spatial averaging.

## Primary Source And Provenance

- Zhi Tian, Chunhua Shen, Xinlong Wang, and Hao Chen, *BoxInst:
  High-Performance Instance Segmentation with Box Annotations*, CVPR 2021:
  <https://openaccess.thecvf.com/content/CVPR2021/html/Tian_BoxInst_High-Performance_Instance_Segmentation_With_Box_Annotations_CVPR_2021_paper.html>.
- Accepted paper stored at
  `D:\DataAI\external_sources\papers\Tian_BoxInst_CVPR_2021.pdf`, SHA-256
  `95b01b2bfaa56f522aa84e7817cbf39ccaa72f01f6572723655a7989552f66e0`.
- Authors' official AdelaiDet repository:
  <https://github.com/aim-uofa/AdelaiDet>, approximately 3.5k stars and 655
  forks when inspected. Local clean clone is pinned at commit/tree
  `5e19cb172b8363820b409ed1a2754fb19ad3acb8` /
  `bd7918078c2b7145ca06809b9e5f976f1286a702`.
- The official academic-use license is BSD-2-Clause-like with a separate
  commercial restriction. LICENSE SHA-256 is
  `9515f73d1af71fabae85d2992c4c002d7ac8d4240e1ec02afbeeca13f27aee42`.
  TRKH will not copy or import AdelaiDet code. It will independently implement
  the published equations and compare them with a separately written NumPy/
  FP64 oracle.
- Inspected official references and SHA-256 values:
  - `adet/modeling/condinst/dynamic_mask_head.py`:
    `96bc6c06bdfddac254a289f3d93dc6c425aa5e916cf30fe1ea2fdc29c2d962bf`;
  - `adet/modeling/condinst/condinst.py`:
    `91559ade3036373f907960ac37a8681afb82c78b6a9805879b4f94ffd68fa2d6`;
  - `configs/BoxInst/Base-BoxInst.yaml`:
    `e64a417214d62e59b7c787044f500206fce8fab618810137f96c5180c306e3be`;
  - `adet/config/defaults.py`:
    `df0cd255c616a25d4bf6535258067bbb6ea2b1870ec8d014562b9ec435a048f9`;
  - `README.md`:
    `86a1f7582f1c62ac8b5f3afcf5c6f1986ea23d561710622c182705362b76c9a4`.

The paper predicts a foreground probability map `m`. Its projection loss is
the sum of Dice losses between the maximum x/y projections of `m` and the box
mask. For a local pixel pair `(i,j)`, the same-label probability is

`p_same = m_i*m_j + (1-m_i)*(1-m_j)`.

LAB similarity is `exp(-||c_i-c_j||/2)`. Only edges at or above the fixed
similarity threshold supervise the positive pairwise term. The total mask loss
is projection plus the warmed pairwise term.

## Adverse Evidence And Transfer Limits

- BoxInst is an instance-segmentation method evaluated on COCO/Pascal, not a
  five-class mango classifier. Its reported mask AP cannot be transferred as a
  TRKH accuracy expectation.
- Existing TRKH audits show that distant background perturbations usually have
  near-zero effect and that the remaining class-1 errors are dominated by
  surface, illumination, maturity continuum, and label boundaries. A learned
  mask therefore has a high burden of proof and may add no useful information.
- Positive LAB affinity can smooth away lesions, shadows, and ripeness regions
  whose color differs from the surrounding peel. Projection constraints also
  admit thin or cross-shaped masks. Rectangle, projection-only, affinity-
  dephased, mask-shift, geometry, and manual-XAI controls are mandatory.
- Earlier bbox prior, valid mask, wide-context fusion, paired `class_f/yolo_f`,
  foreground statistics, objectness, partial convolution, attention, part,
  crop, and background augmentation failures do not count as evidence for this
  pixelwise loss. Conversely, BoxInst A0 cannot reopen or sweep those families.

## Locked Inputs

- Branch must be `classification-only-research`; tracked worktree must be
  clean and HEAD must match upstream before formal execution. Protected
  untracked user payloads remain allowed and byte-exact.
- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher/config SHAs:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff` /
  `e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674`.
- Dataset declaration:
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- CIDT train-only summary/predictions SHAs:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad` /
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Current command/history SHAs:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

Raw images, YOLO labels, YAML, splits, and retained artifacts are immutable.
No validation or test path, pixel, label, prediction, or metric may be opened.

## Source-Disjoint Mask Training Population

Use all 9,215 clean `yolo_f/train` object rows only for class-agnostic mask
training. The ordered sample-index SHA-256 is
`d41d14b04dfb70b60b5058f30d1c4cfe0189aef3b0efc0a22741f5672c1473c8`.
Class counts are `[1941,541,1920,2520,2293]`; class labels are never inputs to
the mask loss.

Use the immutable five CIDT source folds. Holdout row counts are
`[1843,1830,1828,1851,1863]`; each fold has zero source overlap. For held fold
`k`, every learned mask role is trained only on the other four folds and is
then frozen before any mask, descriptor, or classifier is computed for fold
`k`. No mask role may see held-fold pixels or boxes during optimization.

## Locked Class-1 Gate Cohort

Classification evidence uses exactly the previously locked 763 clean
train-only rows:

- 528 keeper class-1 TP;
- 13 keeper class-1 FN;
- 222 restricted keeper `0/2/4 -> 1` FP;
- fold `(TP,FN,FP)` counts:
  `0=(107,2,36)`, `1=(112,3,45)`, `2=(100,4,48)`,
  `3=(101,2,52)`, `4=(108,2,41)`;
- ordered-index SHA-256
  `59d2146617f2642f99c081f49a9aa5812eae4206b27887a1fa5476f6f61c85b2`.

True class 1, including keeper FN, is the positive binary class. Restricted
keeper class-1 FP is the negative class. This cohort is used only after the
fold-specific class-agnostic mask head has been frozen.

## Frozen Feature And Geometry Contract

1. Execute the complete metadata-aware keeper forward; do not bypass bbox,
   valid-mask, or classification-only semantics.
2. Hook exactly `model.stem.blocks[1]`. Verify hooked and ordinary keeper
   probabilities under the same process and batch within `1e-6`, exact argmax,
   and bit-exact keeper state before/after extraction.
3. Retain the full `[64,64,64]` block-2 map. Do not crop or resize it. This
   preserves `yolo_f` wide context, bbox position, letterbox geometry, and
   aspect ratio.
4. Downsample the normalized image-valid mask and the transformed bbox to
   `[64,64]` using the existing geometry path. Every bbox must be nonempty,
   lie inside valid support, and reconstruct from recorded coordinates.
5. Downsample the unnormalized RGB image to `[64,64]` by area averaging and
   convert `[0,1]` sRGB to CIE LAB D65 with an independently tested Torch
   equation. Compare against `skimage.color.rgb2lab` in FP64.
6. The temporary clean cache contains FP16 block-2 features and FP16 LAB maps,
   plus boolean valid/bbox masks and immutable metadata. Every array is hash-
   attested, memory-mapped, never copied into a dataset tree, and deleted after
   replayable descriptors, masks, states, geometry, XAI, and manifests exist.

## Mask Head And Losses

Every learned role receives an independently initialized but byte-matched
three-layer pointwise head:

- input: 64 frozen block-2 channels plus bbox-relative `dx,dy` coordinate
  channels;
- `1x1 (66 -> 16)`, GroupNorm with four groups, GELU;
- `1x1 (16 -> 16)`, GELU;
- `1x1 (16 -> 1)` mask logit;
- sigmoid output multiplied by the valid mask only at loss/pooling time.

The coordinate channels are deterministic from the transformed bbox center
and half-size, clipped to `[-2,2]`. They are not labels and do not change the
raw image. All roles use identical architecture, initialization, fit-row order,
batching, update count, optimizer, and coordinate construction.

Locked learned roles:

- `projection_aligned`: projection loss only;
- `boxinst_aligned`: projection plus aligned LAB pairwise affinity, the sole
  candidate;
- `boxinst_affinity_dephased`: projection plus pairwise weights computed after
  a deterministic SHA-derived common toroidal roll of each complete LAB map
  relative to the feature/bbox geometry.

The dephasing roll preserves LAB channels together, map dimensions, value
multiset, periodic local-neighbor affinity multiset, and global RNG state while
destroying image-location alignment. It is a separately trained mechanism
placebo, not augmentation.

Loss constants follow the paper/official configuration without a sweep:

- pairwise neighborhood `3x3`, center removed;
- dilation `2` with periodic-neighbor construction for the dephasing oracle and
  valid-edge masking in the actual image geometry;
- LAB similarity `exp(-||delta||/2)`;
- positive-edge threshold `0.3`;
- projection Dice epsilon `1e-5`;
- total candidate loss `L_proj + warmup*L_pair`;
- pairwise warmup is `ceil(total_fold_updates/9)`, preserving the official
  `10000/90000` one-cycle ratio;
- AdamW, `lr=1e-3`, betas `(0.9,0.999)`, weight decay `1e-4`;
- batch 64, exactly 20 epochs, natural row frequency, seed 42;
- no class weight, class label, resampling, scheduler, AMP optimizer state,
  early stopping, or post-metric refit.

All head parameters and optimizer states stay FP32. Features may be loaded in
FP16 and compute may use BF16 only after FP32/FP64 checks pass.

## Descriptor And Matched Readout

Create one fixed seed-42 orthonormal projection `64 -> 24` and apply it to
the same frozen block-2 map for every role. For each role mask, compute over
valid pixels:

- mask-weighted 24-channel mean and standard deviation;
- unweighted-valid 24-channel mean and standard deviation;
- mask area/valid area, mask area/bbox area, binary entropy, and outside-bbox
  mass share;
- the five clipped keeper log probabilities.

This fixed 105-dimensional ordering is identical for all mask roles. Static
controls replace only the mask:

- `valid_uniform` uses the complete valid mask;
- `bbox_rectangle` uses the transformed rectangular bbox;
- `boxinst_same_weight_mask_rolled` uses the aligned candidate mask after an
  independently SHA-derived toroidal roll relative to the feature map.

Also fit `keeper_logprob` and `boxinst_aligned_without_keeper` controls. For
each held fold, standardization, a deterministic FP64 L2 logistic readout
(`C=0.1`, LBFGS, maximum 2,000 iterations), and the acceptance threshold are
fit only on the other four cohort folds. The threshold is the highest fit
score retaining at least 97% fit positives; ties are deterministic. Apply it
once to the held fold. No C, descriptor subset, threshold, or direction sweep
is permitted.

## Equation And Engineering Gates

All checks below are conjunctive:

1. Torch FP64 projection loss matches an independent NumPy FP64 oracle within
   `1e-10` on random and analytic masks.
2. Torch FP64 same-label probability and pairwise loss match an independent
   NumPy FP64 oracle within `1e-10`.
3. Torch FP64 sRGB-to-LAB matches scikit-image FP64 within `5e-4`; black,
   white, gray, RGB primaries, and random colors are finite.
4. Projection max-axis orientation, Dice symmetry, complement invariance of
   pairwise same-label probability, valid-edge masking, and singleton batches
   pass analytic checks.
5. Dephasing is deterministic, fold/sample dependent, nonzero, value- and
   periodic-affinity-multiset preserving, and global-RNG isolated.
6. Every declared trainable parameter receives a finite nonzero first-step
   gradient and changes in every learned role. Matched roles begin byte-exact
   and consume the same fit occurrence order and update count.
7. Head/optimizer tensors remain FP32 and finite. BF16-versus-FP32 mask/
   descriptor probability error is `<=0.01`; FP32-versus-FP64 is `<=1e-5`,
   with exact non-near-tie action.
8. Hooked/ordinary keeper probability error is `<=1e-6`, argmax is exact, and
   keeper state is bit-exact. Historical CIDT cache drift remains telemetry
   only and cannot replace this check.
9. Requested/effective Windows workers are exactly `4/4`, with pin and
   persistent workers. No unknown process is terminated.
10. Static ONNX mask-head/descriptor export has probability error `<=1e-5`,
    exact stored-threshold actions, and no custom operator domain.
11. Candidate runtime ratio versus the bbox-rectangle descriptor path is
    `<=1.15`, extra peak CUDA allocation is `<=0.50 GiB`, and total measured
    peak is `<=3.5 GiB`.
12. In-process and independent second-process replay reconstruct row order,
    folds, masks, descriptors, standardizers, readout scores, thresholds,
    actions, map statistics, and analysis within `1e-7` with exact actions.

Any structural failure rejects A0 without metric interpretation.

## Clean Mechanism Gates

Let `boxinst_aligned` be the candidate. Every condition below must pass:

1. AUROC `>=0.85` and AUPRC `>=0.90`;
2. all-positive retention `>=0.95` and keeper-TP retention `>=0.95`;
3. at least 8 of 13 keeper FN are supported;
4. restricted-FP rejection `>=0.25`;
5. precision after action `>=0.75`;
6. FP corrections plus FN supports are at least TP breaks;
7. candidate AUROC exceeds keeper-logprob, valid-uniform, and bbox-rectangle
   by at least `0.02` each;
8. candidate AUROC exceeds projection-only by at least `0.01`;
9. candidate AUROC exceeds trained affinity-dephased and same-weight mask-
   rolled controls by at least `0.02` each;
10. candidate AUPRC exceeds bbox-rectangle, projection-only, affinity-dephased,
    and mask-rolled controls by at least `0.01` each;
11. candidate FP-rejection rate exceeds bbox-rectangle and projection-only by
    at least `0.03`, and affinity-dephased/mask-rolled by at least `0.05`;
12. candidate wins AUROC in at least four of five held folds against each of
    bbox-rectangle, projection-only, affinity-dephased, and mask-rolled;
13. candidate-without-keeper AUROC is at least `0.70`;
14. mean two-axis projection loss is `<=0.10`, outside-bbox mass share is
    `<=0.05`, and every row has finite nonempty mask support;
15. mean mask/bbox area ratio is in `[0.35,0.90]`, its cross-row standard
    deviation is at least `0.02`, and no mask is all-zero, all-one, constant,
    or more than 99.5% saturated;
16. aligned candidate pairwise loss on aligned high-similarity edges is lower
    than the trained affinity-dephased role by at least 5%;
17. mask-roll and affinity-dephasing each reduce AUROC by the declared causal
    margin without improving the TP-harm/FP-correction balance.

No one aggregate metric can rescue a failed conjunctive gate. In particular,
a visually tight mask does not authorize integration if class selectivity or
causal controls fail.

## Shifted Robustness Gate

Only a complete clean pass may open fixed train-only conditions `dim`,
`bright`, and `low_contrast`, using the same transformations and strengths as
CIDT. Fold mask heads, standardizers, readouts, thresholds, and dephasing
offsets remain frozen.

For every condition, candidate AUROC must be `>=0.80`, TP retention `>=0.92`,
FP rejection `>=0.15`, and candidate AUROC must exceed bbox-rectangle and
affinity-dephased by at least `0.01`. Maximum AUROC drop from clean is `0.05`;
all actions and mask statistics must replay exactly.

## XAI And Manual Review

After the automatic clean decision, render fixed 15-row sheets selected before
metrics and covering keeper TP, keeper FN, restricted FP, every fold, narrow/
wide geometry, stems, hands, shadows, and visible lesions.

The direct sheet contains RGB+bbox, valid-uniform, bbox-rectangle,
projection-only, aligned BoxInst, affinity-dephased, and same-weight mask-roll
columns under common scales. A second sheet contains input gradients of the
binary score through mask-weighted features for every learned/control role.

Automatic XAI checks require exact row/column counts, finite nonconstant maps,
zero padding leakage, exact score reconstruction, and hash-attested arrays.
Manual pass requires the aligned mask and gradient to follow the fruit surface
and relevant lesion/peel regions more consistently than rectangle,
projection-only, dephased, and rolled controls. Dominant bbox edges, fruit
silhouette alone, padding, hands, stems, shadows, or distant context is a
visual failure. Manual review cannot rescue an automatic failure.

## Artifacts, Cleanup, And Promotion

Persist protocol/source/input hashes, repository state, all fold mask states,
optimizer summaries, occurrence hashes, OOF masks/descriptors/scores/actions,
geometry, exact replay metadata, ONNX, resource snapshots, XAI arrays/sheets,
and an artifact manifest. Delete only the named temporary full-train feature/
LAB caches after their SHA-256 values are recorded and replayable outputs are
verified. Never delete raw data, protected keepers, user payloads, or another
process's files.

On failure:

- keep the compact formal and closure evidence;
- close nearby BoxInst head width/depth, projection/pairwise weights, LAB
  threshold/theta, neighborhood/dilation, warmup, coordinate channels, mask
  area, projection target, pooling statistics, optimizer, epochs, fold, seed,
  readout C, threshold, and fusion variants on this keeper;
- do not edit the production model/trainer, run smoke/probe/full train, open
  validation/test, or update current-best commands.

On complete pass only:

1. add one default-off mask branch and auxiliary loss to TRKH;
2. run matched no-test control/candidate smoke with full validation support;
3. inspect all existing trace, confusion, robustness, resource, and XAI audits;
4. run one short probe only if smoke meets its prospective milestone;
5. run a full train of at most 30 epochs with patience 3 only if the raw
   validation promotion gate passes and a fresh benchmark confirms the worker
   count that maximizes GPU throughput on the current machine;
6. update the one-command VS Code pipeline and command history only after a
   genuine current-best promotion.
