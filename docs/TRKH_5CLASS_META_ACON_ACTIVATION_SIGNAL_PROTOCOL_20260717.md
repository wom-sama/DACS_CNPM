# TRKH 5-Class Meta-ACON Activation Signal A0 Protocol - 2026-07-17

## Status and question

This protocol is locked before implementation or measurement. It asks one
narrow question: can a sample-conditioned channel-wise activation create a
train-only, object-derived signal that separates current class-1 true positives
from the restricted `0/2/4 -> 1` false positives better than both the unchanged
stem and static ACON-C?

This is a fit-only readiness gate. It is not trainer integration, a validation
experiment, a test experiment, a new full train, or permission to tune the
current class-1 decision rule. Raw images, labels, bboxes, split membership,
checkpoints, and command packets remain immutable.

## Source policy and screen

User-supplied research reports are hypothesis sources only. Equations and
implementation claims below are cross-checked against accepted primary work
and official repositories.

- Primary paper: Ma et al., *Activate or Not: Learning Customized Activation*,
  CVPR 2021:
  <https://openaccess.thecvf.com/content/CVPR2021/html/Ma_Activate_or_Not_Learning_Customized_Activation_CVPR_2021_paper.html>
- Accepted paper PDF:
  `D:/DataAI/external_sources/papers/acon_cvpr2021.pdf`, SHA-256
  `acae6d8f9100250f9d21c2bbfc8a6f5ae4fbd7d886643e0cbd4a1405a02f9218`.
- Official MIT repository: <https://github.com/nmaac/acon>, observed at about
  `207` stars during the screen.
- Locked official commit/tree:
  `99fd67928a6ffb0543b54614303caada96c756f5` /
  `f269fee6201f0945fc154134f310704b59175837`.
- Official `acon.py` SHA-256:
  `670d4d70e44009fb857bc4e056798dd60fe66ac2ed02c1c849f04c0b44ee4087`.
- Official MIT `LICENSE` SHA-256:
  `e09e657f606b61bf1ab24f59db9b68d61b0532943ff22db2ff8dd0cfc84ece60`.

The paper and source disagree on initialization. Section 3 states
`beta=p1=1, p2=0`, while the official file initializes `p1` and `p2` with
random normal values. The A0 experiment follows the paper initialization and
must report this discrepancy. The official module is still loaded and replayed
with copied state to verify the implemented equation.

Alternatives were screened first:

- ODConv and API-Net remain closed by existing local evidence; reopening them
  would violate the no-repeat ledger.
- Involution was checked against the CVPR 2021 paper and official MIT repository
  (about `1.3k` stars). Its official fast path uses CuPy/custom CUDA. At the
  actual third-stem input geometry, the official-equation `nn.Unfold` path
  (`1x1 64->256` plus 3x3 involution) was `3.640968x`, `7.428328x`, and
  `7.145325x` the matched 3x3 convolution runtime at batches `1/4/8`, with
  peak-memory ratios `7.436997x/8.082161x/8.201651x`. ONNX and TensorRT parsed,
  but this resource result and the custom-op alternative deny integration.
- Focal Modulation was checked against the NeurIPS 2022 paper and official
  Microsoft repository. Its hierarchical depthwise context plus gated
  local/global modulation overlaps the already closed MogaNet, large-kernel,
  local/global coupling, and attention-modulation families.
- Dynamic ReLU has an accepted ECCV 2020 paper, but no author-maintained,
  licensed implementation was identified in this screen.

Meta-ACON is retained because its input-conditioned non-linearity is not a new
loss, sampler, output threshold, token router, fixed texture operator, or
another local/global spatial mixer.

## Locked provenance

- Repository branch: `classification-only-research`.
- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher arguments SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- Dataset YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Frozen train-only fold declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`,
  SHA-256
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Fold companion summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.
- Current best command packet SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`.
- Seed `42`, CUDA only, deterministic kernels, TF32 disabled, batch `64`,
  workers `4`.

Only CIDT folds `1,2,3,4` are used. The exact cohort is every clean train row
for which the keeper predicts class `1` and either:

- target is class `1` (positive, retained class-1 TP), or
- target is class `0`, `2`, or `4` (negative, restricted class-1 FP).

Expected locked support is `607 = 421 TP + 186 FP`, with fold counts:

| Fold | TP | FP |
|---:|---:|---:|
| 1 | 112 | 45 |
| 2 | 100 | 48 |
| 3 | 101 | 52 |
| 4 | 108 | 41 |

The ordered cohort SHA-256 is
`a2689d1be8579386eea9ef02a826822a5e7d78e2d29439e8947c1731ccc738bd`.
No validation or test loader may be constructed.

## Locked operator and controls

Let `x` be the frozen keeper's final `256x32x32` CNN-stem map before patch
embedding. ACON-C is

`f(x) = (p1-p2)x * sigmoid(beta * (p1-p2)x) + p2*x`.

All roles use the same frozen stem map, bbox-derived object mask, and shared
linear-head initialization:

1. `identity`: no new activation; object-masked channel mean then `Linear(256,1)`.
2. `acon_static`: paper-initialized channel-wise `p1=1`, `p2=0`, learnable
   static `beta=1`, then the same object readout.
3. `meta_acon`: paper-initialized `p1=1`, `p2=0`; channel-wise beta is generated
   by the official `GAP -> Conv(256,16) -> BN -> ReLU -> Conv(16,256) -> BN ->
   sigmoid` route, then the same object readout.

The object mask is the complete transformed crop bbox (`core OR boundary`) at
`32x32`; only activated pixels inside it enter the classifier. Meta-ACON's
primary beta uses official full-map GAP. A locked evaluation-only ablation
recomputes beta from object-masked GAP without refitting. This ablation must
remain competitive so a pass cannot be explained solely by far background.

The keeper stem and its batch-normalization state are frozen. Only each role's
activation parameters and one binary head are optimized. There is no class
weight, balanced sampler, oversampling, hard mining, augmentation, teacher,
threshold loss, or sample selection.

## Locked OOF optimization

For each held fold in `1,2,3,4`:

- fit on the other three folds and hold out the complete selected fold;
- preserve natural row frequency and source grouping;
- train on clean maps only for exactly `12` epochs, batch `64`;
- AdamW, learning rate `0.003`, weight decay `0.0001`;
- binary cross entropy with logits, no weighting;
- use the same epoch permutations and identical shared head initialization for
  all three roles;
- never early-stop or select an epoch from held-fold metrics.

After fitting, select one threshold from fit-clean scores only: the highest
observed threshold whose fit TP retention is at least `0.90`. Apply that frozen
fold threshold to held clean and all held corruptions.

The four evaluation conditions are fixed:

- `clean`: brightness `1.00`, contrast `1.00`;
- `lighting_dim`: brightness `0.70`, contrast `0.90`;
- `lighting_bright`: brightness `1.25`, contrast `1.10`;
- `low_contrast`: brightness `1.00`, contrast `0.65`.

## Equation, deployment, and activity checks

Before accepting information metrics, the auditor must verify:

- local output and input/parameter gradients against official `MetaAconC` with
  copied state;
- an independent equation oracle and a finite-difference coordinate;
- paper initialization exactly `p1=1`, `p2=0`;
- FP32/BF16 finiteness and maximum FP32/BF16 error `<=0.02`;
- nonzero gradients for `p1`, `p2`, both beta-generator convolutions, and head;
- active sample conditioning: between-sample beta RMS `>=0.005` and activation
  delta RMS `>=0.01` after OOF fitting;
- static ONNX ops only, ONNX Runtime maximum error `<=1e-5`, TensorRT parser and
  serialized-engine build success;
- full frozen stem plus post-stem Meta-ACON runtime ratio `<=1.25` and peak
  allocated-memory ratio `<=1.25` against the frozen stem at locked geometry.

Preflight validates hashes, source cleanliness, exact cohort/folds, equations,
and locked arguments but creates no output directory. Formal output must contain
no checkpoint, model state, engine, raw image copy, or reusable feature cache.

## Information gate

Every engineering/provenance check above must pass, plus all of these fixed
information checks:

1. clean Meta-ACON OOF AUROC `>=0.65`;
2. clean Meta-ACON AUROC is at least `0.03` above both identity and static ACON;
3. Meta-ACON beats static ACON in at least `3/4` clean folds and no fold is
   worse by more than `0.02`;
4. every shifted Meta-ACON AUROC is `>=0.60` and no lower than static ACON;
5. clean OOF TP retention `>=0.90` and restricted-FP rejection `>=0.20` under
   fit-only fold thresholds;
6. every shift retains TP `>=0.85` and rejects restricted FP `>=0.10`;
7. TP median score exceeds FP median score in all four conditions;
8. object-only-beta clean AUROC is at least static ACON plus `0.02` and no more
   than `0.02` below the full-beta Meta-ACON AUROC.

Only a complete pass authorizes implementation of one post-stem Meta-ACON
scratch control/candidate pair capped at five epochs and still closed to test.

## Stop rule

Any failed gate closes A0 before trainer/model integration. Do not sweep
initialization, beta structure, reduction ratio, insertion layer/count,
activation family, LR, optimizer, epochs, batch, weight decay, class weighting,
folds, thresholds, conditions, bbox masks, or residual scales. Do not reinterpret
a static-ACON gain, context-only gain, beta variance, equation pass, export pass,
or isolated fold gain as smoke permission. A rejected A0 does not update the
current-best full-train, engine, video, audit, or command-history files.
