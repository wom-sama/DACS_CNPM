# TRKH 5-Class Sparse Class-Evidence A0 Protocol - 2026-07-21

## Status And Scope

Prospectively locked before auditor implementation, feature extraction, model
fit, or candidate metric inspection. This protocol authorizes one
source-disjoint, train-only information audit of a learned sparse local
class-evidence branch on the frozen current keeper. It does not authorize
validation/test access, production model or trainer edits, an image-model
smoke/probe/full train, or current-best command promotion.

## Research Question

Can a local class-evidence map, trained with an explicit L1 penalty, separate
true class-1 mango evidence from restricted `0/2/4 -> 1` false positives while
preserving current class-1 true positives and supporting the current class-1
false negatives?

The intended later architecture, only if A0 passes, is a small parallel branch
from the second CNN-stem block. It copies and adapts the keeper's third local
stem block, emits a two-channel class-evidence map, and fuses a class-1 margin
with the existing CNN/Transformer path during joint training. A0 is not a
post-hoc router and cannot authorize a thresholded deployment rule. It first
tests whether the sparse local representation contains a causal, recall-safe
signal.

## Accepted Primary Source And Licensed Reference

- Djoumessi et al., *Sparse Activations for Interpretable Disease Grading*,
  MIDL 2023 Oral:
  `https://openreview.net/forum?id=us8BFTsWOq`.
- Local accepted paper:
  `D:/DataAI/external_sources/papers/Djoumessi_Sparse_Activations_MIDL_2023.pdf`,
  SHA-256
  `0fe7a832e20d4fb3438410664a2fc7ebe823189ec3a7a80f8989d2ee1e3a10ca`.
- Authors' repository:
  `https://github.com/kdjoumessi/interpretable-sparse-activation`, locked at
  commit/tree
  `2b3731fdbec09e10cd2aa25b0742bc9164f9a60c` /
  `d9d54544af7392ebe6690fb412a86087540229a5`.
- The repository is MIT licensed. License SHA-256:
  `05e28d9ece454402704f2d6c8a41a89b7a4dd89214aa4edc8977fca32c5fe4e7`.
- Reviewed source hashes:
  - `modules/builder.py`:
    `0b673cc8136d39b60e311ad9f6df5a8de8b28ec1105db8e8cd2a4c6620ec1480`;
  - `train.py`:
    `33dacba60e37c884a4f91bb80d74d43bcad21b09e1d0f7340991fe7d7b2be9a1`;
  - `README.md`:
    `6afef729a89c796e1162c2fd70cd6fd65e0f19fbec0b95c128871ec43875e19e`.

The paper's Eq. (1) swaps a linear classifier and spatial average, producing
an explicit class-evidence map with a `1x1` convolution. Eq. (2) adds the sum
of absolute class-evidence values to cross-entropy. The accepted paper reports
that sparsity raised binary specificity from `0.991` to `1.0`, but lowered
sensitivity from `0.779` to `0.750`; this is relevant adverse evidence and is
why TP protection is conjunctive here. Its multiclass sparse model also lost
accuracy and collapsed severe grades into a neighboring grade.

The official recipe uses ImageNet initialization, 512-pixel images, 100
epochs, and validation-selected lambda values. None transfers to scratch TRKH.
The released training code also evaluates `torch.norm(linear_fts[0], 1)`,
which regularizes only the first batch item. TRKH will independently implement
the paper equations over every batch item and normalize by the number of valid
map elements. No official model code is imported, copied, or executed.

## Local No-Repeat Boundary

- Stock BagNet substitution remains rejected. A0 imports only the accepted
  class-evidence and map-sparsity equations and applies them to the native
  no-pretrain TRKH local stem.
- WILDCAT is closed. It sorts top and bottom spatial cells at inference; this
  candidate uses every valid cell, ordinary masked spatial averaging, and an
  L1 training constraint. It must beat a matched dense head and a global-logit
  L1 control so generic score shrinkage cannot be mislabeled local evidence.
- RSC, SIFER, attention erasing, part selection, patch MIL, Finer-CAM, semantic
  parts, prototypes, covariance, frequency, morphology, Gabor/LBP, Deep-TEN,
  NMF, color mixing, and post-hoc routing remain closed and are absent.
- The candidate does not use a pretrained model, teacher, sample/class weight,
  oversampling, augmentation, external feature, validation threshold, or test
  result.
- BatchEnsemble and Packed-Ensembles are rejected for this stage: they add
  member capacity but do not provide a prospective class-1 TP-versus-FP local
  evidence mechanism. The failed shared-body late-member family is not
  repeated.

## Locked Inputs And Protected State

- Pre-protocol TRKH HEAD/upstream:
  `d1bc8e7ee85835b5a1af1c1358b861546be09b5a`.
- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher arguments, SHA-256
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- Dataset declaration:
  `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- CIDT train-only summary/predictions, SHA-256
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`
  and
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- CIDT fold declaration, SHA-256
  `2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763`.
- Current-best command/history, SHA-256
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`
  and
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

The seven protected untracked payloads must remain untracked and hash-exact:

- `BaoCao/GT.md`: `c7847176...c7b840`;
- `BaoCao/mango_cls_256_merge01_4cls_architecture.dot`:
  `c9aebcb9...528a9d`;
- `BaoCao/mango_cls_256_merge01_4cls_architecture.png`:
  `93d6f9c2...04832d`;
- `BaoCao/mango_cls_256_merge01_4cls_architecture.svg`:
  `2ead05f8...453d05`;
- `BaoCao/mango_cls_256_merge01_4cls_architecture_summary.md`:
  `d43b82be...0e493`;
- `deep-research-report (9).md`: `7e52cc8e...546f8a`;
- `deep-research-report (10).md`: `37f74b5e...c1522a`.

## Locked Train-Only Cohort And Folds

Read only CIDT `clean` train rows satisfying either condition:

1. target class is class 1; or
2. keeper prediction is class 1 and target is in `{0,2,4}`.

The cohort is exactly 763 objects:

- 541 class-1 positives: 528 keeper TP and 13 keeper FN;
- 222 restricted false positives: class counts `0:158`, `2:54`, `4:10`;
- fold `(TP,FN,FP)` counts:
  `0:(107,2,36)`, `1:(112,3,45)`, `2:(100,4,48)`,
  `3:(101,2,52)`, `4:(108,2,41)`;
- ordered sample-index SHA-256
  `59d2146617f2642f99c081f49a9aa5812eae4206b27887a1fa5476f6f61c85b2`.

Each immutable CIDT source fold is held once. Fit and held source stems must be
disjoint. Every image path must resolve below `yolo_f/images/train`; opening a
validation/test path, row, prediction, label, or metric is fatal.

## Frozen Feature Extraction And Local Geometry

Run the keeper's ordinary metadata-aware forward in evaluation mode with its
normal bbox prior and token pruning. Capture only the output of
`model.stem.blocks[1]`, after the second local `3x3 + GELU + pool` block and
before the keeper's third stem block. Expected shape is `[B,64,64,64]`.

The hook path must reproduce the ordinary keeper probabilities within `1e-6`,
preserve every argmax, remove every hook exactly once, and leave every keeper
parameter and buffer bit-identical.

For each row:

1. downsample `image_valid_mask` to the `64x64` captured grid by area
   interpolation;
2. find the minimal rectangle whose mask value is at least `0.5`;
3. crop the complete valid rectangle from the FP32 feature map;
4. resize the crop by bilinear interpolation to exactly `48x48`;
5. persist FP16 features only in a temporary cache, with source rectangle,
   original aspect ratio, feature hash, and one fixed geometry preview per
   fold/category.

This removes square padding but keeps the complete wide object crop and nearby
context. It does not use labels, bbox shape, foreground masks, attention, or a
learned crop. All learned roles consume byte-identical aligned caches.

Clean extraction occurs first. Shifted caches may be opened only after every
clean automatic gate passes. Temporary caches must be deleted in a verified
`finally` path after final payload hashing.

## Sparse Local Evidence Branch

Each learned role has exactly the same architecture and parameter count:

1. a deep copy of the keeper's third `ConvStemBlock` (`64 -> 256`, local
   `3x3`, keeper normalization/activation/pooling), initialized from the
   locked keeper state;
2. a learned `1x1` convolution `256 -> 2` producing class-evidence maps
   `A_neg` and `A_pos` on a `24x24` grid;
3. ordinary spatial mean over every map cell to obtain two binary logits.

The positive logit denotes class 1. The evidence margin map is
`M = A_pos - A_neg`, and its spatial mean is exactly the binary logit margin.
There is no max/top-k pooling, attention, token selection, threshold inside the
model, or connection to keeper logits during fitting.

## Independent Equations And Controls

For each image and class map, independently verify paper Eq. (1):

`logit_c = mean_ij(A_c[i,j])`.

The candidate implements a scale-normalized form of paper Eq. (2):

`loss = CE(binary_logits, target) + lambda_map * mean(abs(A))`.

The mean differs from the paper's unnormalized sum only by a fixed scalar and
prevents batch/grid size from silently changing regularization strength. A
separate NumPy/FP64 oracle must verify logits and the complete objective. The
official first-item-only hook behavior is explicitly prohibited.

Four parameter-matched learned roles start from byte-identical branch state in
each fold and receive the same natural-frequency order:

- `dense_aligned`: CE only;
- `sparse_aligned`: candidate map-L1 objective;
- `logit_l1_aligned`: CE plus L1 on the two spatially averaged logits;
- `sparse_channel_dephased`: candidate objective on a causal spatial placebo.

For each penalized role, lambda is fixed from its first deterministic fit
batch before any update so that its regularizer equals exactly 5% of CE on
that batch:

`lambda = 0.05 * detached_CE / detached_regularizer`.

The value then remains fixed for the fold. Only fit-fold rows participate in
this calibration. A zero/nonfinite regularizer is fatal. This predeclared
ratio replaces the paper's validation-selected lambda sweep; no nearby ratio
or lambda is permitted.

The causal placebo independently toroid-rolls every input channel in both
spatial axes. Offsets are deterministically derived from
`SHA256(seed, fold, sample_index, channel)` and are nonzero whenever the grid
allows. This preserves each channel's complete value multiset, mean, variance,
and histogram while destroying aligned local cross-channel evidence. It is
applied consistently to fit and held rows for the dephased learned role.

The aligned candidate weights are also evaluated once on dephased held maps
using the aligned fit threshold. This same-weight intervention must degrade
the candidate if aligned local evidence is causal. A common spatial
permutation of the final evidence maps must leave logits and map L1 invariant;
that expected Eq. (1)-(2) symmetry is verified but is not used as evidence of
locality.

## Locked Optimization

- five immutable source-disjoint folds;
- seed `42`, with fold-specific RNG derived only as `42 + fold`;
- natural-frequency fit rows, no sampler or class/sample weights;
- batch size `64`;
- 20 fixed epochs, no held-fold early stopping or epoch selection;
- AdamW, learning rate `1e-3`, weight decay `1e-4`, betas `(0.9,0.999)`;
- FP32 optimizer state and BF16 CUDA autocast for branch forward;
- one identical deterministic shuffle per epoch and role;
- no augmentation, teacher, distillation, margin loss, threshold loss, or
  keeper-logit feature.

Every trainable parameter must receive a finite nonzero gradient on the first
update and differ from initialization after training. Optimizer configuration,
update count, batch identities, and initial states must match across roles.

## Fold-Safe Action Rule

For every fold and role, fit one threshold only on the four fit folds. Among
all unique fit scores, select the highest threshold whose class-1 retention is
at least 97%; ties are resolved toward the larger threshold. Predict binary
class 1 for scores greater than or equal to that threshold. Apply it once to
the held fold and concatenate all five held outputs into OOF predictions.

Report separately:

- AUROC and AUPRC over all 763 rows;
- retention over all 541 class-1 rows and over the 528 keeper TP rows;
- support among the 13 keeper class-1 FN rows;
- rejection over the 222 restricted FP rows;
- restricted-FP corrections, keeper-TP breaks, and keeper-FN supports;
- every fold/target/source transition and threshold.

No threshold becomes a production or validation rule.

## Clean Promotion Gates

Every structural/equation/resource check and every condition below is
conjunctive:

- candidate AUROC `>= 0.85` and AUPRC `>= 0.90`;
- all-class1 and keeper-TP retention each `>= 0.95`;
- at least `8/13` keeper class-1 FN rows receive positive support;
- restricted-FP rejection `>= 0.25`;
- `FP corrections + FN supports >= TP breaks`;
- candidate AUROC/AUPRC are no more than `0.005` below dense, while candidate
  restricted-FP rejection exceeds dense by at least `0.03`;
- candidate exceeds global-logit L1 by at least `0.01` AUROC and `0.03`
  restricted-FP rejection;
- candidate exceeds trained channel-dephasing by at least `0.03` AUROC and
  `0.05` restricted-FP rejection, with AUROC wins in at least four folds;
- same-weight channel dephasing lowers candidate AUROC by at least `0.02`;
- candidate class-margin-map Hoyer sparsity and top-10%-absolute-mass share
  each exceed dense by at least `0.10` without a zero/constant map;
- exact replay reconstructs every row, threshold, metric, transition, map
  statistic, and gate from serialized states and CSVs.

Failure of any clean gate closes shifted extraction, production integration,
validation/test, image-model training, and current-best updates. There is no
lambda, layer, grid, width, seed, epoch, optimizer, threshold, or cohort sweep.

## Robustness And Faithful XAI

Regardless of clean pass/fail, generate a fixed 15-row clean contact sheet
selected before candidate outputs: the lowest sample index for one keeper TP,
one keeper FN, and one restricted FP in each fold. Each row must show the RGB
crop plus dense, sparse, logit-L1, trained-dephased, and same-weight-dephased
class-margin maps using one common color scale.

The direct class-margin map is the primary faithful explanation. Also compute
the candidate margin gradient with respect to the captured block-2 feature
input, through the copied third block and evidence head. This feature-input
gradient must not be described as a raw-image saliency map.
Automatic XAI requires exact margin reconstruction, finite maps, no padding
leak, candidate foreground mass `>= 0.60`, border mass `<= 0.30`, and exact
coverage of all 15 fixed rows. Manual review must find stable fruit-surface or
lesion evidence that differs materially from the dephased controls; broad
color fields, silhouette, hands, padding, or image borders fail.

Only after all clean automatic gates pass, repeat frozen fold heads on train
images transformed by the existing fixed `dim`, `bright`, and `low_contrast`
conditions. Each condition must satisfy AUROC `>= 0.80`, all-class1/keeper-TP
retention `>= 0.92`, restricted-FP rejection `>= 0.20`, and candidate AUROC
at least `0.02` above trained dephasing. No shifted refit is allowed.

## Numeric, Export, And Resource Gates

- FP64 PyTorch versus independent NumPy equation/objective error `<= 1e-12`;
- FP32 versus FP64 probability error `<= 1e-5`;
- BF16 versus FP32 probability error `<= 0.01`, with exact non-near-tie
  argmax;
- ordinary versus hooked keeper probability error `<= 1e-6`, exact argmax,
  exact keeper state hash, and zero leaked hooks;
- ONNX Runtime CPU probability error `<= 1e-5`, exact action decisions at the
  stored thresholds, and no custom operator;
- candidate inference runtime no more than `1.10x` dense and extra peak CUDA
  allocation no more than `0.25 GiB`;
- formal requested/effective data workers exactly `4/4`. This is locked from
  the existing same-machine loader benchmark: workers 4 reached
  `196.346 images/s` with `10.85%` data wait, versus `107.006 images/s` and
  `52.48%` wait for workers 2. Effective worker count and extraction
  throughput must be recorded.

## Stage-B Authorization If And Only If A0 Passes

A complete clean, shifted, numeric, export, resource, replay, and manual-XAI
pass authorizes implementation of one default-off parallel sparse-evidence
branch in TRKH and one matched no-test smoke/probe sequence. The first image
run remains limited to five epochs and 120 batches per epoch, uses workers
chosen by a fresh `2` versus `4` benchmark, patience `3`, full validation,
one architecture trace per class, full robustness, and complete XAI.

A full run remains closed until the candidate passes the repository milestone
with validation class-1 F1 `>= 0.70`, class-1 precision `>= 0.65`, recall
`>= 0.72`, macro F1 no lower than the keeper, FP no greater than the keeper,
TP breaks no greater than FN rescues, and corrections no fewer than harms.
Only then may an autonomous full train use at most 30 epochs and patience `3`.

## Stop And Preservation Rules

- Validation and test remain closed during A0.
- Do not modify raw `class_f` or `yolo_f` files.
- Do not update the full-train command, command history, deploy pointer, or
  current-best checkpoint from A0 alone.
- Preserve protocol, provenance, code/tests, exact OOF CSVs, trained branch
  states, replay, 15-row XAI sheet, summary, and manifest before cleanup.
- Delete only verified temporary feature caches and failed export scratch in a
  scoped cleanup record. Never delete the keeper, current command files,
  protected user payloads, or an unclassified run.
- If A0 fails, document the exact failed gates and close this locked sparse
  branch plus nearby lambda/grid/width/layer/seed/threshold variants before
  screening a new equation-distinct route.
