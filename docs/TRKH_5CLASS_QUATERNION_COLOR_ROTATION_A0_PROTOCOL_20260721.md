# TRKH 5-Class Quaternion Color-Rotation A0 Protocol - 2026-07-21

## Status And Scope

Prospectively locked before auditor implementation, dataset-pixel access, crop
extraction, model fitting, or candidate metrics. This protocol authorizes one
source-disjoint, train-only information audit of the ECCV-2018 quaternion color
rotation equation. It does not authorize validation/test access, a production
TRKH model or trainer edit, a checkpoint, an image-model smoke/probe/full
train, or current-best command promotion.

## Research Question

Can a compact supervised color-surface branch distinguish true class-1 keeper
predictions from restricted `0/2/4 -> 1` false positives by learning aligned
RGB rotations around the achromatic axis, while retaining true class-1 support
and outperforming ordinary convolution, spatial color covariance, and
causally de-registered RGB controls?

The intended later architecture, only if every A0 gate passes, is a small
quaternion color branch fused with the existing CNN+Transformer descriptor.
A0 is not a post-hoc threshold/router and cannot authorize deployment. It first
asks whether the quaternion constraint exposes class-conditional peel evidence
that the keeper and previously closed color/texture families do not contain.

## Accepted Primary Source And Reference Boundary

- Zhu et al., *Quaternion Convolutional Neural Networks*, ECCV 2018:
  `https://openaccess.thecvf.com/content_ECCV_2018/html/Xuanyu_Zhu_Quaternion_Convolutional_Neural_ECCV_2018_paper.html`.
- Local paper:
  `D:/DataAI/external_sources/papers/zhu2018_quaternion_cnn_eccv.pdf`,
  SHA-256
  `0cbf101924ecefcf6d2be1bb67bbad358e7b48d2c84c6ed1d23e6495c26a8cb4`.
- Paper-citing Keras reference repository:
  `https://github.com/XYZ387/QuaternionCNN_Keras`, observed at 33 stars and
  seven forks. Locked commit/tree:
  `8c381f5f2c9e2d53cdd3fe30b0a2dc6994f79401` /
  `82a0545923e8238e1b61063e1511eb7b8703368d`.
- Reviewed reference hashes:
  - `README.md`:
    `903755c19e00468aa248137bd8800bc426a7b101815e0b939366c9486ceb33e6`;
  - `quaternion_layers/conv.py`:
    `91e987496d8ab499411bcfdfc08530fc8827fdb2c2efd459ad23431fe3f2a0c7`;
  - `quaternion_layers/init.py`:
    `d4e62cfa304f95a195bfe2c362439316524cd55fb529d76b6e4c074fe4decc3d`;
  - `cifar10_cnn.py`:
    `6322a5ec396caefc4a641ac0039de22fe578b84d9344a2ddc428e15c018c503f`.

The repository has no LICENSE/COPYING/NOTICE file and its README says that the
layers borrow heavily from another project. It is therefore not a permissive
software dependency and is not treated as proof of author identity. TRKH must
not import, copy, translate, or adapt any source from that repository. The
auditor must be implemented independently from paper Eq. (3)-(8), with a
separately written NumPy/FP64 Rodrigues-rotation oracle. The repository is used
only as adverse provenance and interpretation evidence.

The paper represents an RGB pixel as the pure quaternion `0 + R i + G j + B k`
and constrains every spatial weight to a scale plus a rotation around the unit
gray axis. It reports color-classification improvements but uses CIFAR-10,
RMSProp, augmentation, and schedules that do not transfer to TRKH. A0 imports
only the color-rotation equation and the permitted real-valued classifier
connection, not its architecture, initialization results, or accuracy claims.

## Local No-Repeat Boundary

- CEConv color equivariance, AugSelf color prediction, RGB/HSV/Lab fusion,
  illuminant correction, MixStyle, Gabor/LHO/FCM, Deep-TEN, fixed histograms,
  color topology, covariance/bilinear pooling, and chromatic augmentation are
  closed. Quaternion A0 is admitted only because each spatial tap is a tied
  scale plus a three-channel rotation that preserves the achromatic axis. It
  must beat a parameter-matched real CNN and a spatial covariance readout.
- Quaternion A0 does not use a part selector, attention map, MIL/top-k pooling,
  prototype, contrastive loss, class reweighting, teacher, calibration,
  router, or validation-fitted threshold.
- The exact same crop, RGB bytes, labels, folds, occurrence order, optimizer
  updates, and base keeper values are shared by candidate and controls. No raw
  dataset file, label, split, or annotation is changed.
- A trained channel-dephased role and a same-weight channel-dephased
  intervention are mandatory. Marginal RGB information alone cannot support a
  quaternion claim.

## Locked Inputs And Protected State

- Pre-protocol TRKH HEAD/upstream:
  `7c9b16e13e3a937cbb697fa8b070012db2cfb772`.
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
stems must be disjoint. Every path must resolve below `yolo_f/images/train`.
Opening any validation/test path, prediction, pixel, label, or metric is fatal.

## Locked RGB Crop

Build the keeper's deterministic evaluation transform at `256x256`, invert
only its declared input mean/std, clamp to `[0,1]`, and use `crop_bbox` in the
transformed canvas. No image-model forward is needed.

For every row:

1. convert normalized `xywh` to an integer half-open rectangle with floor on
   the left/top and ceil on the right/bottom;
2. inset each side by exactly 10% of the rectangle width/height, using ceil for
   the new left/top and floor for the new right/bottom;
3. require positive area and at least 98% `image_valid_mask` coverage, with no
   fallback, mask repair, foreground segmentation, or label-dependent crop;
4. bilinearly resize with antialiasing to `64x64`, round `255*x` to nearest
   integer, and cache exact `uint8 RGB` bytes;
5. record original/inset rectangles, valid fraction, per-channel mean/std,
   byte hash, and fixed previews.

The expected temporary cache is `750x3x64x64 uint8`, about 8.79 MiB. Every
role consumes the same clean cache. It must be deleted only after final
feature/state/artifact hashes and XAI have been written; the cache hash remains
in the summary. Geometry failure stops A0 before fitting and cannot be repaired
after observing a metric.

## Independent Quaternion Equation

For each input quaternion group `c`, output group `k`, and spatial tap, learn a
scale `s` and angle `theta`. With the unit gray axis
`u=(1,1,1)/sqrt(3)`, define:

```
f1 = 1/3 + (2/3) cos(theta)
f2 = 1/3 - (2/3) cos(theta - pi/3)
f3 = 1/3 - (2/3) cos(theta + pi/3)

M_gray(s, theta) = s * [[f1, f2, f3],
                         [f3, f1, f2],
                         [f2, f3, f1]]
```

The generated real convolution kernel sums these `3x3` transforms over input
quaternion groups. A separate NumPy/FP64 oracle must instead use Rodrigues'
formula `R(v)=cos(theta)v + (1-cos(theta))(u dot v)u + sin(theta)(u cross v)`
and agree after the explicitly tested orientation convention. Do not derive
the oracle from `f1/f2/f3` or import reference code.

The equation must preserve projection on the gray axis up to scale and become
theta-invariant for grayscale vectors. Test FP64, FP32, BF16, analytic and
finite-difference gradients, cyclic RGB symmetry, multi-group accumulation,
singleton batches, invalid shapes/nonfinite values, and exact materialization
to an ordinary static `Conv2d` kernel.

## Locked Candidate And Controls

All image roles use component-wise ReLU, component-wise `2x2` max pooling
after layer two, global average pooling, and a real binary head. There is no
normalization, dropout, attention, top-k selection, residual path, or image
augmentation.

Quaternion image trunk:

- groups `1 -> 8`, kernel `5`, stride `2`, padding `2`;
- groups `8 -> 8`, kernel `3`, stride `1`, padding `1`;
- component-wise max pool `2`, stride `2`;
- groups `8 -> 16`, kernel `3`, stride `2`, padding `1`;
- final real feature dimension `48`.

At each quaternion layer, initialize scales from
`U[-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out))]`, where
`fan_in=q_in*k*k` and `fan_out=q_out*k*k`; initialize angles from
`U[-pi/2,pi/2]`; initialize component biases to zero. Construction uses local
CPU generators and restores Python/NumPy/Torch CPU/CUDA RNG state exactly.

The learned roles are:

- `base_logprob`: five fit-standardized clipped keeper log probabilities;
- `rgb_grid_covariance_base`: per-cell RGB mean plus upper-triangle covariance
  over a fixed `4x4` grid, the same global statistics (`153` RGB values), and
  five base values;
- `real_cnn_base`: ordinary real CNN channels `3 -> 9 -> 13 -> 18`, with the
  exact candidate kernels/strides/padding/pooling and five base values. Its
  total parameter count must be within 5% of `quaternion_gray_base`;
- `quaternion_red_axis_base`: the same quaternion trunk, initialization, and
  updates, but Rodrigues rotations use fixed axis `(1,0,0)`;
- `quaternion_dephased_base`: the exact gray-axis trunk trained on the locked
  channel-dephased RGB input;
- `quaternion_gray_base`: the candidate gray-axis trunk plus five base values;
- `quaternion_gray_only`: the candidate trunk without keeper values;
- `quaternion_gray_repeat`: the candidate role repeated with seed
  `20260721 + 17`.

The candidate, red-axis, dephased, and gray-only trunks begin with bit-exact
scale/angle/bias tensors. Candidate and gray-only classifier image weights are
also bit-exact at construction; gray-only simply omits the five base weights.

Also evaluate the trained candidate with unchanged learned state under:

- `quaternion_same_weight_dephased`: locked dephased held RGB;
- `quaternion_same_weight_theta_zero`: all materialized angles replaced by
  zero while retaining scales, biases, and classifier state.

These same-weight interventions are never trained or used for selection.

## Locked Channel-Dephasing Placebo

For sample index `i` and RGB component `c in {0,1,2}`, apply a toroidal roll:

```
dy = 3 + ((20260721 + 37*i + 11*c) mod 13)
dx = 3 + ((20260721 + 53*i +  7*c) mod 13)
```

This preserves every channel's exact pixel multiset and toroidal within-channel
spatial adjacency while breaking co-located RGB vectors. It must be exactly
deterministic, alter every channel registration pair, and leave all global RNG
states unchanged. The same equation is used for trained and same-weight
placebos under clean and allowed robustness conditions.

## Locked Training And Action Rule

For each held fold, train on the other four folds only:

- binary target `1`: keeper class-1 prediction is a true class-1 positive;
- binary target `0`: keeper class-1 prediction is a restricted false positive;
- natural occurrence frequency, no class/sample weighting or resampling;
- seed `20260721`, batch `64`, exact shared epoch permutations;
- AdamW, LR `2e-3`, weight decay `1e-4`, 20 fixed epochs, no scheduler,
  warmup, clipping, early stopping, retry, checkpoint selection, or AMP;
- BCE-with-logits; FP32 training from the immutable uint8 cache; epoch 20 is
  always evaluated.

Fit-only base means/stds and covariance-feature means/stds are frozen before
held inference. Persist initial/final parameter hashes, occurrence hashes,
loss histories, first-step gradient/update evidence, optimizer summaries,
fold scores, final pooled features, and model/head state needed for independent
score/action replay. No production checkpoint is written.

For every role and fold, choose the lowest fit score that retains at least 97%
of fit-fold TP. Accept a held row as class 1 only when its score meets that
threshold. This is an information-audit action with explicit TP protection,
not an authorized runtime router.

## Structural And Information Gates

All gates are conjunctive:

1. all source/data/model/external/protocol/protected hashes match, the tracked
   tree is clean and pushed, only train paths open, and cohort/folds have zero
   source overlap;
2. independent Rodrigues/explicit-matrix equations, FP64/FP32/BF16,
   gray-projection, grayscale invariance, cyclic symmetry, finite-difference,
   analytic-gradient, multi-group, singleton, invalid-input, and static-kernel
   materialization checks pass;
3. every declared parameter receives finite nonzero first-step gradients and
   changes; matched quaternion roles start bit-exact, shared occurrences are
   exact, and all global RNG state is restored by construction/dephasing;
4. real-control parameter count differs by at most 5%, clean materialized
   logits differ by at most `1e-6`, ONNX static-branch logits differ by at most
   `1e-5`, and no dynamic quaternion operator is required at deployment;
5. candidate aggregate OOF AUROC is at least `0.85`, at least `+0.03` above
   base, `+0.02` above RGB grid covariance, `+0.015` above matched real CNN,
   `+0.02` above red-axis quaternion, and `+0.03` above both trained and
   same-weight channel-dephased roles;
6. candidate also beats same-weight theta-zero AUROC by at least `0.02`, while
   `quaternion_gray_only` AUROC is at least `0.78`;
7. aggregate candidate actions retain at least 95% TP, reject at least 25% of
   restricted FP, produce more corrections than harms, and no held fold has TP
   retention below 90%;
8. candidate rejects at least ten more FP than each trained control while
   breaking no more than two additional TP, and beats covariance, real CNN,
   red-axis, and trained dephasing AUROC in at least four of five folds;
9. repeat AUROC differs by at most `0.02`, repeat action agreement is at least
   `0.95`, and repeat TP/FP gates also pass;
10. candidate feature effective rank is at least 12, channel variance and
    learned angle/scale diagnostics are finite/noncollapsed, and every fit loss
    is finite;
11. clean pooled features, scores, thresholds, and actions independently
    replay within `1e-7`, with exact row ordering and stable artifact hashes;
12. requested/effective extraction workers are recorded; peak CUDA allocation
    is below 2.0 GiB; candidate branch runtime is at most `2.25x` the matched
    real branch, consistent with the paper's disclosed cost; and no unknown
    process is terminated.

## Robustness, XAI, And Advancement Boundary

Only a complete clean pass opens fixed PIL `ImageEnhance` conditions generated
from the clean `64x64 uint8` cache: dim `(brightness=0.70, contrast=0.90)`,
bright `(1.25,1.10)`, and low contrast `(1.00,0.65)`, always brightness then
contrast. Reuse clean fold states, fit standardizers, and thresholds without
refit. Each condition must retain AUROC `>=0.80`, TP retention `>=0.92`, FP
rejection `>=0.20`, and positive candidate-minus-best-control AUROC; aggregate
FP removals must exceed TP harms.

Always render a fixed 16-row clean sheet: four highest-score TP, four
lowest-score TP, four lowest-score restricted FP, and four highest-score
restricted FP. Show RGB crop, candidate input-gradient/activation map, matched
real map, same-weight dephased map, and theta-zero map. Automatic XAI requires
finite nonconstant maps, candidate border mass below `0.45`, and no numerical
map collapse. Manual review must decide whether evidence lies on peel/lesion
content rather than resize edge or broad silhouette. It cannot rescue a failed
automatic gate.

A complete clean, robustness, XAI, replay, resource, and static-export pass
authorizes only a separate default-off production branch and one prospectively
locked matched random-initialization scratch smoke, at most `120` batches by
`2` epochs, scheduler horizon 30, full validation, final test skipped, and all
existing audits. It does not authorize a probe or full train by itself.

Only a later integrated smoke that improves class-1 precision/F1 while meeting
TP/recall, all-class, robustness, XAI, runtime, ONNX, and TensorRT gates may
open the user's autonomous full-train permission. That full train remains
bounded by 30 epochs, justified early stopping, a pre-run GPU-contention check,
and a measured Windows worker setting. Current-best commands change only after
independent locked validation promotion.

Any material A0 failure closes this exact central-bbox `64x64`, gray-axis
three-layer quaternion family, widths/kernels, real/covariance/red-axis/
dephasing controls, optimizer, epoch count, fold, seed, and TP threshold on the
current keeper. Do not sweep crop inset/size, quaternion widths, axes, angles,
pooling, LR, weight decay, epochs, seed, fold, threshold, or fuse it with closed
color/texture/attention routes.
