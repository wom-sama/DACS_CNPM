# TRKH 5-Class DeepBDC Stem Joint-Dependence A0 Protocol - 2026-07-21

## Status And Scope

Prospectively locked before auditor implementation, feature extraction, or any
new candidate metric. This protocol authorizes one source-disjoint, train-only
information audit of a learned Brownian distance covariance side head on the
frozen current keeper. It does not authorize validation/test access, a TRKH
model or trainer edit, a checkpoint, an image-model smoke/probe/full train, or
current-best command promotion.

Preflight erratum, recorded before any dataset pixel, feature, or candidate
metric was read: the `stl_deepbdc.py` and `README.md` hashes below each had one
missing/transposed hexadecimal character in the first lock. They are corrected
to independently recomputed file hashes; no scientific setting or gate changed.

## Research Question

Can a small learned projection plus Brownian distance covariance (BDC) expose
nonlinear channel dependence in the current CNN stem that distinguishes true
class-1 predictions from restricted `0/2/4 -> 1` false positives while
explicitly protecting true class-1 recall?

The intended later architecture, if A0 passes, is a CNN-stem BDC auxiliary head
trained jointly with the existing Transformer path. A0 is not a post-hoc
class-1 suppressor and cannot authorize threshold routing. It first asks
whether a genuinely new surface/context representation exists beyond keeper
probabilities and matched first/second-order controls.

## Accepted Primary Source And Licensed Reference

- Xie et al., *Joint Distribution Matters: Deep Brownian Distance Covariance
  for Few-Shot Classification*, CVPR 2022 oral:
  `https://openaccess.thecvf.com/content/CVPR2022/html/Xie_Joint_Distribution_Matters_Deep_Brownian_Distance_Covariance_for_Few-Shot_Classification_CVPR_2022_paper.html`.
- Local paper:
  `D:/DataAI/external_sources/official/deepbdc-cvpr2022/Xie_DeepBDC_CVPR_2022.pdf`,
  SHA-256
  `cdb4a73cf396aaa4be0eeb5414ce5f2971512278eb43a86b2ae9fe3243bc7654`.
- Authors' repository: `https://github.com/Fei-Long121/DeepBDC`, observed at
  185 stars and 27 forks. Locked commit/tree:
  `adfab39f121ec1c17502306eca9e134244f7a83a` /
  `058e8d08cc7cc1c3af8805be061d15c202882cec`.
- The repository uses CC BY-NC 4.0, not a permissive software license. License
  SHA-256:
  `2dd05518b1e6ee64b8f319a0d8ecfa55e02b6261dcabaea2c6ccf979b4bf7ef5`.
- Reviewed source hashes:
  - `methods/bdc_module.py`:
    `27ef7cb7752ff085b3858945c89818383f12a4ce25cbe0e208f38891fab30bf6`;
  - `methods/stl_deepbdc.py`:
    `258602a39f50e94518eee8b8e623d09666322d43e92485065310b42d9f12b37b`;
  - `README.md`:
    `ed4025410d52ca745abb6393fe4153349963e9a2faeedf45a071fc5bd5d0e38d`.

TRKH will not import, copy, or adapt the repository source. The implementation
must be written independently from paper Eq. (4)-(6), checked by a separately
written NumPy/FP64 oracle, and cite the paper. The official source is used only
to verify interpretation and adverse implementation details.

The paper's supervised STL form is relevant: Eq. (9) trains a BDC matrix with
a conventional FC softmax classifier and cross-entropy, so this audit does not
need a prototype or neighbor classifier. The paper also shows BDC outperforming
matched covariance pooling and states that Euclidean channel distances model
nonlinear joint dependence whereas covariance models linear dependence. Its
published results use ResNet backbones, meta-training/pretraining, long recipes,
and optional sequential self-distillation. Those accuracies and schedules do
not transfer to scratch TRKH; A0 imports only the BDC equation and supervised
head structure.

## Local No-Repeat Boundary

- Whole-patch compact bilinear pooling and an interior rank-24 square-root
  covariance readout are closed. BDC is admitted only because its double-
  centered Euclidean channel-distance matrix is equation-distinct from an
  outer product or covariance matrix. A matched covariance head must be beaten.
- Prototype, subcenter, GMM, neighbor, support-generation, contrastive,
  reweighting, calibration, and router families remain closed and are absent.
- WILDCAT, class prompts/queries, MCL, patch MIL, Finer-CAM, part selectors,
  Deep-TEN, lacunarity, NMF, Gabor/wavelet/LBP, spectral branches, and dynamic
  convolution remain closed. A0 uses none of their losses or selectors.
- StarNet, MogaNet, HorNet, and the NeurIPS-2025 higher-order convolution paper
  were screened before this lock. Their multiplicative interactions overlap
  failed local multiplicative stem routes and do not supply an independent
  class-1 TP-protection mechanism.
- FDConv was screened and rejected before code because it composes an ODConv-
  like dynamic kernel with frequency-band modulation. Both operative families
  already failed locally, and it provides no class-1-specific TP protection.
- No raw data, label, split, sample weight, class weight, augmentation, teacher,
  distillation target, loss reweighting, threshold sweep, or ensemble is used.

## Locked Inputs And Protected State

- Pre-protocol TRKH HEAD/upstream:
  `37b7380b7df2a3e23d9ba50d53d776d3a4e35d0b`.
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

## Locked Cohort And Folds

Use only CIDT `clean` train rows whose keeper prediction is class 1 and whose
target is in `{0,1,2,4}`. The cohort is exactly 750 objects:

- 528 true class-1 positives and 222 restricted false positives;
- fold TP/FP counts `107/36`, `112/45`, `100/48`, `101/52`, `108/41`;
- ordered sample-index SHA-256
  `55913ec45265b156a611dc96c779e46b08f28582737af8269105afe6f65694d7`.

Each of five immutable CIDT source folds is held once. Fit and held source
stems must be disjoint. Every path must resolve below `yolo_f/images/train`;
opening any validation/test path, prediction, pixel, label, or metric is fatal.

## Frozen Stem Extraction And Geometry

Run the keeper's ordinary metadata-aware forward in evaluation mode with normal
token pruning and bbox prior. Capture only the output of its existing
`HybridConvStem`, before MixStyle and patch projection. Expected shape is
`[B,256,32,32]`. The capture path must reproduce ordinary probabilities within
`1e-6`, preserve non-near-tie argmax, remove every hook, and leave every keeper
parameter/buffer bit-identical.

For each row:

1. downsample `image_valid_mask` to the stem grid by area interpolation;
2. find the minimal rectangle whose mask value is at least `0.5`;
3. crop that rectangle from the FP32 stem map and resize it by bilinear
   interpolation to exactly `16x16`;
4. persist the source rectangle, valid fraction, feature hash, and one fixed
   geometry preview per class/TP/FP category.

This removes square-padding size as a shortcut while retaining the complete
valid wide crop, including object and nearby context. It does not use labels,
bbox shape, foreground masks, or a learned crop. All roles consume the exact
same cached map. The expected float16 cache is `750x256x16x16`; it is temporary
and must be deleted after artifact hashes are finalized.

## Independent BDC Equation

For projected features `X` with shape `[B,d,n]`, `d=32`, `n=256`:

1. compute `G = X X^T`;
2. compute squared channel distance
   `D2_ij = G_ii + G_jj - 2 G_ij`, clamp at zero;
3. compute `D = sqrt(exp(tau) * D2 + 1e-5)`, with trainable scalar
   `tau` initialized to `log(1/(2*n))` as in the released module;
4. double-center `A = D - row_mean(D) - col_mean(D) + mean(D)`;
5. vectorize the upper triangle and L2-normalize it.

The auditor must compare this implementation against a separately written
NumPy/FP64 pairwise-distance oracle, not the authors' source. The official
implementation's batch-size-sensitive `squeeze` behavior must not be copied.

## Matched Learned Roles

Every learned role begins with an independently initialized but bit-identical
`1x1`, bias-free `256 -> 32` projection, `BatchNorm2d(32)`, and ReLU, matching
the paper's dimension-reduction block. Each role appends the same five
fit-standardized clipped keeper log probabilities before its binary FC head.

- `base_logprob`: logistic readout on only the five standardized log
  probabilities;
- `mean32_base`: projected spatial mean plus base values;
- `cov32_base`: centered `32x32` covariance, upper triangle, signed square-root
  and L2 normalization, plus base values;
- `bdc32_dephased_base`: BDC after a deterministic independent spatial
  permutation for every projected channel, preserving each channel marginal
  while destroying aligned joint dependence, plus base values;
- `bdc32_aligned_base`: the candidate aligned BDC plus base values;
- `bdc32_aligned_only`: candidate BDC without keeper probabilities, reported
  only to prove the descriptor has standalone directional information.

Also evaluate the trained candidate with the same weights after applying the
dephasing operation. This same-weight causal placebo cannot be trained or used
for model selection.

The fixed dephasing permutation for sample `i`, channel `c` uses a local CPU
generator seeded by `20260721 + 1000003*i + 1009*c`. Global Python, NumPy, CPU
Torch, and CUDA RNG states must be restored exactly.

## Locked Training And Action Rule

For each held fold, train on the other four folds only:

- binary target `1`: keeper class-1 prediction is a true class-1 positive;
- binary target `0`: keeper class-1 prediction is a restricted false positive;
- natural occurrence frequency, no class/sample weighting or resampling;
- seed `20260721`, batch `64`, exact shared epoch permutations;
- AdamW, LR `1e-3`, weight decay `1e-4`, 20 fixed epochs, no scheduler,
  warmup, clipping, early stopping, retry, checkpoint selection, or AMP;
- BCE-with-logits; FP32 cache-to-head training; epoch 20 is always evaluated.

Fit-only keeper-log means/stds and BN running state are frozen before held-fold
inference. Persist initial/final parameter hashes, exact occurrence hashes,
loss histories, gradients, optimizer state summaries, fold scores, and model
state needed for independent replay. No image-model checkpoint is written.

For each role and fold, choose the lowest fit score that retains at least 97%
of fit-fold TP. Accept a held row as class 1 only when its score meets that
threshold. This rule is fixed before held scores and explicitly protects TP;
it is an audit action, not an authorized production router.

## Structural And Information Gates

All gates are conjunctive:

1. all source/data/model/external/protocol/protected hashes match, the tracked
   tree is clean and pushed, only train paths open, and the cohort/folds are
   exact with zero source overlap;
2. BDC NumPy/FP64, Torch FP64/FP32/BF16, analytic-gradient, finite-difference,
   symmetry, translation, orthonormal-transform, scale, singleton-batch, and
   invalid-input checks pass;
3. all roles share exact input maps/order and candidate/control initial
   projection tensors; every trainable parameter receives finite nonzero
   gradients and changes, while keeper state/logits remain exact;
4. candidate aggregate OOF AUROC is at least `0.85`, at least `+0.02` above
   `base_logprob`, `+0.02` above `cov32_base`, `+0.015` above
   `mean32_base`, and `+0.03` above both trained and same-weight dephased BDC;
5. aligned-only BDC AUROC is at least `0.65`, proving that any gain is not only
   a reparameterization of keeper probabilities;
6. aggregate held action retains at least `0.95` of TP, rejects at least `0.25`
   of restricted FP, produces more corrections than harms, and no held fold
   has TP retention below `0.90`;
7. candidate rejects at least ten more FP than every matched role while
   breaking no more than two additional TP, and its AUROC beats every role in
   at least four of five held folds;
8. candidate BDC effective rank is at least 16, projected channel variance is
   finite/nonzero, no score or descriptor collapses, and fit loss is finite;
9. clean candidate score and actions independently replay within `1e-7`, with
   exact thresholds/argmax/action rows and stable artifact hashes;
10. extraction plus head runtime is at most `1.15x` ordinary keeper inference,
    peak CUDA allocation is below `3.5 GiB`, requested/effective workers and
    Windows GPU-engine contention are recorded, and no unknown process is
    terminated.

## Robustness, XAI, And Advancement Boundary

Only a complete clean pass opens fixed dim `(0.70,0.90)`, bright
`(1.25,1.10)`, and low-contrast `(1.00,0.65)` train-cohort extraction. Reuse
the clean fold heads, BN state, log standardizers, and thresholds without any
fit. Each condition must retain AUROC `>=0.80`, TP retention `>=0.92`, and a
positive candidate-minus-best-control AUROC delta; aggregate FP removals must
exceed creations.

Always render a fixed 16-row clean contact sheet after the automatic result:
input plus valid rectangle, candidate Grad-CAM/feature-gradient map, covariance
map, and same-weight dephased map for TP/FP accepts/rejects. The sheet and an
independent score/action replay must be hash locked. Manual review cannot
rescue a failed automatic gate.

A complete clean, robustness, XAI, resource, and export pass authorizes only a
separate default-off TRKH BDC side-head readiness implementation and one
prospectively locked matched scratch smoke, at most `120 batches x 2 epochs`
with full validation and final test skipped. It does not authorize a probe or
full train. A later smoke must improve class-1 precision/F1 while respecting
locked TP/recall, robustness, deployment, and XAI gates before the user-granted
autonomous full-train permission can apply.

Any material A0 failure closes `d=32`, stem-final, full-valid-crop BDC with this
projection, normalization, optimizer, epoch count, base fusion, action rule,
and dephasing family on the current keeper. Do not sweep channel dimension,
stem layer, object/context crop, projection, temperature, epsilon, centering,
normalization, epochs, LR, weight decay, batch, seed, fold, threshold, or fuse
it with closed covariance/bilinear/texture/attention routes.
