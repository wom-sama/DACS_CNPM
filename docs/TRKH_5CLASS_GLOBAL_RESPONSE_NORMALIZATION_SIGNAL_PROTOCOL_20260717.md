# TRKH 5-Class Global Response Normalization Signal A0 Protocol - 2026-07-17

## Status And Research Question

Prospectively locked before implementation, dataset access, or measurement.
This protocol asks one narrow question: does the official ConvNeXt V2 Global
Response Normalization (GRN) statistic expose a source-disjoint, illumination-
stable signal that keeps current class-1 true positives while rejecting the
restricted `{0,2,4} -> 1` false positives?

This is a fit-only information gate on frozen keeper features. It is not a
trainer/model integration, image-training epoch, 1,843-row train-holdout run,
official validation/test run, prediction threshold, full train, or current-best
command update. Raw images, labels, bboxes, split membership, checkpoints, and
command packets remain immutable.

## Primary-Source And License Lock

User-provided research reports are hypothesis sources only. The source of truth
for this A0 is the accepted paper, the authors' repository, and direct TRKH
measurement.

- Accepted paper: Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets
  with Masked Autoencoders," CVPR 2023.
- CVF page:
  `https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html`
- Accepted PDF:
  `D:/DataAI/external_sources/official/convnext-v2-cvpr2023-paper.pdf`, SHA-256
  `1c2de7ea1d4f811dcbbafb8fcfac4b9ac506c27fe82cfdb720dec0282890b6a5`.
- Official repository: `https://github.com/facebookresearch/ConvNeXt-V2`.
- Locked official commit/tree:
  `2553895753323c6fe0b2bf390683f5ea358a42b9` /
  `0b23579ac3ded0c671f592def6f7b66300a80799`.
- Official `models/utils.py` SHA-256:
  `0dabefc9489ed74d0fbfd614527edf83d4172fe905297a17697f15b76c268bb5`.
- Official `models/convnextv2.py` SHA-256:
  `c6a65592720aa7ee57ad92f143ee392e29e18249793ee91259642b401e0f5855`.
- Official `LICENSE` SHA-256:
  `a6a3de076198f9cabf4d2c59429fb6cf74f65d0bde2ceb6f0d9ff9dbc257f0eb`.
- The repository had `2,063` stars and `174` forks when checked.

The combined license file starts with the MIT software grant and also includes
CC BY-NC 4.0 terms for separately distributed material. This A0 reimplements
only the four-line GRN source equation and uses no official pretrained weight,
FCMAE checkpoint, image, or model definition. Any later release must retain the
MIT attribution and must not bundle the repository's non-commercial weights.

The paper reports two limitations that constrain interpretation here. GRN was
most effective when co-designed with FCMAE, and adding a newly initialized GRN
only at fine-tuning time reduced ImageNet performance. Therefore this protocol
forbids post-hoc insertion into the trained keeper. A successful A0 would allow
only a separately locked, zero-initialized scratch pair in which GRN is present
from the first optimizer step.

## No-Repeat Screen

- GhostNetV2 DFC was checked against its NeurIPS 2022 paper, the paper-linked
  Apache-2.0 MindSpore implementation, and Apache-2.0 `timm v1.0.27`. The
  practical gate is `AvgPool -> depthwise 1x5 -> depthwise 5x1 -> sigmoid ->
  upsample`; at TRKH geometry it is another axis-factorized local mixer/spatial
  gate and overlaps the closed large-kernel, LKA/LSK, local-global, and dynamic
  spatial-attention families. It is not selected.
- BatchFormer was checked against its CVPR 2022 paper and authors' repository
  (`253` stars). The repository exposes no license, and its batch-axis behavior
  introduces batch-composition dependence already known to move near-tie TRKH
  predictions. It is not selected.
- CrossNorm/SelfNorm was checked against its ICCV 2021 paper and official
  Apache-2.0 Amazon repository. CrossNorm exchanges per-channel style statistics
  and SelfNorm recalibrates them; this overlaps rejected MixStyle, SRM/style
  recalibration, and illumination-normalization routes. It can also suppress the
  object-color cue that current XAI shows is stronger than background. It is not
  selected.
- The historical pretrained ConvNeXtV2-Tiny probe reached only validation
  macro/class-1 F1 `0.8570/0.5360`. This forbids another stock ConvNeXt V2
  launch, but it did not isolate zero-initialized GRN inside the current
  no-pretrain TRKH representation.

GRN remains distinct from closed activation and spatial-mixing routes because
it performs deterministic global L2 response competition across hidden
channels. It has no spatial kernel, token router, teacher, added loss, sampler,
or inference threshold.

## Locked TRKH Inputs And Cohort

- Repository branch: `classification-only-research`.
- Protocol parent commit:
  `805588c080b44b7773a7f47a1d7cc43c9870c07a`.
- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher arguments SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- Data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Frozen train/fold declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`,
  SHA-256
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Fold companion summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.
- Current command/history SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
- Seed `42`, CUDA only, deterministic kernels, TF32 disabled, batch `64`,
  workers `4`.

Only declaration folds `1,2,3,4` may be opened. The exact cohort contains clean
train rows where the keeper predicts class `1` and the target is either:

- class `1`: `421` protected true positives; or
- class `0`, `2`, or `4`: `186` restricted false positives.

Expected support is `607`, with fold TP/FP counts `112/45`, `100/48`, `101/52`,
and `108/41`. Ordered cohort SHA-256 is
`a2689d1be8579386eea9ef02a826822a5e7d78e2d29439e8947c1731ccc738bd`.
Fold `0`, validation, and test loaders must never be constructed.

Prior PDC and Meta-ACON audits already established one batch-context near-tie:
sample `3657` has target `2`, declaration prediction `1`, canonical batch-64
prediction `2`, and declaration `p1-p2=0.00017741`. Keep it in the declaration-
defined FP cohort. Replay must match all other 606 rows and this exact exception;
any additional mismatch is a structural failure. Report every information gate
again after read-only exclusion of sample `3657`, without refitting, but never
use that sensitivity result to rescue a failed full-cohort gate.

## Locked Feature Site And Equation

The keeper has eight Transformer blocks, seven prefix tokens, hidden width
`1024`, and a `16x16` patch grid before its first pruning event. A hook on the
normal deployment forward captures block 2 FFN output immediately after its
first `Linear(256,1024)` and `GELU`, before dropout and projection. This is the
last complete grid immediately before layer-2 pruning and is the direct TRKH
analogue of the paper's dimension-expansion MLP activation. Trace or
`return_attention` logits may not be used.

For patch hidden values `X` with shape `[P,C]`, official GRN is:

`g_c = sqrt(sum_p X[p,c]^2)`

`n_c = g_c / (mean_j(g_j) + 1e-6)`

`Y[p,c] = X[p,c] + gamma_c * X[p,c] * n_c + beta_c`

where `C=1024`, `gamma=0`, and `beta=0` at initialization. The seven prefix
tokens are excluded from response aggregation and left unchanged. This A0 does
not alter keeper logits; it trains only isolated binary readouts over captured
descriptors.

Map each transformed bbox to the `16x16` patch grid. Define `O` as all patch
cells intersecting the bbox and `B` as its complement; both must be nonempty.
For every sample and condition retain only these in-memory descriptors:

- object mean `m_O = mean_{p in O} X[p,:]`;
- full-grid GRN response `n_all`;
- object-only GRN response `n_obj`, recomputed over `O`;
- outside-only response `n_out`, recomputed over `B`.

No hidden token map, image, checkpoint, NumPy cache, or feature cache may be
written to disk.

## Locked OOF Readouts

Run four source-fold OOF fits. For each held fold, fit on the other three clean
folds and evaluate the complete held fold under all conditions. Preserve
natural row frequency and source grouping. There is no weighting, balanced
sampler, oversampling, hard mining, augmentation, teacher, or sample selection.

Two trainable roles use the same deterministic linear-head initialization and
identical epoch permutations:

1. `identity`: `Linear(m_O,1)`.
2. `grn_l2`: `Linear(m_O + gamma*(m_O*n_all) + beta,1)`, with official
   zero-initialized `gamma/beta`.

Fit each role for exactly `12` epochs, batch `64`, unweighted binary cross
entropy with logits, AdamW learning rate `0.003`, and weight decay `0.0001`.
Never early-stop or select an epoch from held-fold behavior.

For each role/fold, choose the highest observed fit-clean threshold retaining
at least `95%` of fit class-1 TP. Freeze it for the held clean and shifted rows.
For `grn_l2`, also score `n_obj` and `n_out` with the already fitted
`gamma/beta/head` and the same threshold; these are evaluation-only controls
and may not be refit.

Conditions are fixed to:

- `clean`: brightness `1.00`, contrast `1.00`;
- `lighting_dim`: brightness `0.70`, contrast `0.90`;
- `lighting_bright`: brightness `1.25`, contrast `1.10`;
- `low_contrast`: brightness `1.00`, contrast `0.65`.

## Equation, Activity, Deployment, And Audit Gates

All checks below must pass before information metrics can authorize trainer
integration:

1. Every source/input/command hash and official commit/tree is exact; official
   and tracked TRKH worktrees are clean; the TRKH commit is pushed.
2. Preflight creates no output directory and constructs no data loader.
3. Local GRN output matches both the official class loaded without bytecode and
   an independent float64 equation oracle at maximum error `<=1e-10`.
4. Input, `gamma`, and `beta` gradients match the oracle at `<=1e-9`; one
   float64 finite-difference coordinate is within `1e-5`.
5. Zero initialization is exact identity. FP32/BF16 outputs and gradients are
   finite and BF16 error is `<=0.01`.
6. Standard keeper inference reproduces all targets and exactly the one locked
   sample-`3657` prediction exception at batch `64`, with no new mismatch; the
   hook captures exactly `[B,263,1024]`, seven prefixes, and 256 pre-pruning
   patches without changing logits.
7. Every bbox/object/outside mask and descriptor is finite and nonempty. Every
   OOF fit has disjoint sources, both binary classes, identical role head
   initialization, and exact natural row counts.
8. The fitted candidate is active: mean OOF `gamma` RMS is `>=0.005`, mean
   GRN-versus-identity object-vector delta RMS is `>=0.005`, and between-sample
   full-response RMS is `>=0.01`.
9. An ephemeral block-2 FFN with patch-only GRN exports at ONNX opset 17 with
   standard operators, matching shape and ONNX Runtime error `<=1e-5`.
   TensorRT parser and serialized-engine construction must pass; neither file is
   retained.
10. At locked `[64,263,256]` geometry, median candidate FFN runtime is
    `<=1.15x` and peak allocated memory is `<=1.10x` the matched FFN control.
11. Formal output contains only compact CSV/JSON/Markdown/PNG evidence. No
    checkpoint, state dict, ONNX, engine, raw image copy, or reusable feature
    cache may be retained.

Before gate finalization, independently replay every OOF metric and fold delta
from the written CSV. Render a fixed contact sheet for the four highest and
four lowest clean candidate scores in each cohort. Each panel shows the source
image only inside the rendered audit page, bbox overlay, hidden RMS map, fitted
GRN modulation RMS map, fold/target/keeper prediction, and all four condition
scores. Rendering is diagnostic and cannot change a threshold or gate.

## Information And Precision Gate

All checks use source-disjoint OOF scores on the 607 fit-only rows:

1. clean `grn_l2` AUROC is `>=0.65` and at least `0.03` above `identity`;
2. `grn_l2` beats identity in at least three of four clean folds, with no fold
   worse by more than `0.02`;
3. every shifted `grn_l2` AUROC is `>=0.60`, no shifted AUROC is below the
   corresponding identity AUROC, and the worst clean-to-shift drop is `<=0.05`;
4. clean TP retention is `>=0.95` and restricted-FP rejection is `>=0.20`;
5. every shift retains TP `>=0.90` and rejects restricted FP `>=0.10`;
6. TP median score is strictly above FP median score in every condition;
7. clean object-only-response AUROC is no more than `0.02` below full-response
   AUROC and remains at least `0.025` above identity;
8. object-only-response clean AUROC is not below outside-only-response AUROC,
   so far background cannot be the sole discriminative response;
9. object-only response keeps clean TP `>=0.93` and rejects restricted FP
   `>=0.18` under the unchanged fitted fold thresholds.

Only a complete structural, deployment, activity, visual, replay, and
information pass authorizes a separately precommitted model/config/trainer
implementation and one matched source-disjoint five-epoch scratch pair. That
pair still requires precision, recall, restricted-FP, illumination, XAI,
runtime, ONNX, and TensorRT gates before validation can be considered.

## Stop Rule

Any failed gate closes A0 before trainer integration. Do not sweep GRN layer,
block count, response norm, epsilon, affine initialization, object/context
mask, LR, optimizer, epochs, batch, weight decay, folds, thresholds, conditions,
losses, FCMAE, LayerScale, or full ConvNeXt variants. Do not reinterpret an
equation pass, active `gamma`, isolated fold gain, feature-map visualization,
or outside-context gain as training permission.

A rejected A0 does not update the current-best full-train, engine, video, XAI,
audit, or command-history files.
