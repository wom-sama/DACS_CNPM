# TRKH 5-Class DOLG Orthogonal Local-Global Signal A0 Protocol - 2026-07-17

## Decision scope

This protocol locks one fit-only compatibility gate before any DOLG model,
trainer, config, or launcher integration. The question is narrow:

> Does the exact DOLG orthogonal projection expose object-local CNN information
> that complements the current keeper's global Transformer descriptor, and does
> it separate current class-1 true positives from restricted `{0,2,4}->1` false
> positives better than the same features joined by direct concatenation?

The full `yolo_f` view remains the normal model input. The bbox is used only to
pool an object-local descriptor from the already-computed dense patch embedding;
no crop image, raw label, or dataset file is changed. A complete pass may
authorize one separately locked short train-only pair. It cannot authorize
validation, test, a full train, or a current-best command update.

## Primary authority and provenance

- Accepted paper: Yang et al., *DOLG: Single-Stage Image Retrieval With Deep
  Orthogonal Fusion of Local and Global Features*, ICCV 2021, pages 11772-11781:
  `https://openaccess.thecvf.com/content/ICCV2021/html/Yang_DOLG_Single-Stage_Image_Retrieval_With_Deep_Orthogonal_Fusion_of_Local_ICCV_2021_paper.html`.
- Local accepted-paper PDF:
  `D:\DataAI\external_sources\papers\Yang_DOLG_ICCV2021_accepted.pdf`.
- Paper SHA-256:
  `3ca7070e9d749fa84c9606b43dabd159242a518bd902817565b27b2fc22fb679`.
- Author repository: `https://github.com/feymanpriv/DOLG`.
- Local source: `D:\DataAI\external_sources\official\dolg-iccv2021`.
- Locked commit/tree:
  `63b117d76f660db9617c2081ec771e3cbdf4bcc8` /
  `c7639a440fbd5b25b911737dae392ddfd1d346a6`.
- Official `core/model/dolg_model.py` SHA-256:
  `83b1984de8ca0be585975622b1dba6e85351b1cd6d17b71fb06c3ce17ab6dd54`.
- Official MIT `LICENSE` SHA-256:
  `4a9589726b97c388428d37e0526743b77f189f531aef987017af06abb1b450d4`.
- GitHub API observation on 2026-07-17: `66` stars, `12` forks, MIT, not
  archived. Popularity is context only, not efficacy evidence.
- No DOLG, ResNet, ImageNet, retrieval, AIDT, or other pretrained weight is
  permitted.

The paper and source jointly establish the locked mechanism. For every local
feature `l_ij` and global feature `g`, DOLG removes the component of `l_ij`
parallel to `g`:

```text
projection(l_ij, g) = ((l_ij dot g) / ||g||^2) * g
orthogonal(l_ij, g) = l_ij - projection(l_ij, g)
```

The paper reports orthogonal fusion above direct concatenation and Hadamard
fusion in its retrieval ablation. It also states that average pooling commutes
with the projection/subtraction, reducing the operation to
`pool(l) - (pool(l) dot g) * g / ||g||^2`. Those claims motivate the gate; they
do not establish efficacy on mango classification.

## No-repeat screen

This A0 is distinct from, and does not reopen, the following closed routes:

- Persistent concurrent local-global coupling exchanged a learned CNN state
  with patch tokens after Transformer blocks 1-8. DOLG A0 performs one late,
  first-order subtractive redundancy removal and does not alter either path.
- Existing `cnn_feature_fusion` pools the CNN stem and adds independent class
  logits. A0 instead measures whether a spatial object-local descriptor has a
  component orthogonal to the actual global head input.
- Fine-grained attention pooling, layer-token fusion, source-context fusion,
  paired-view fusion, bilinear/second-order pooling, patch prototypes, frequency
  branches, local/large-kernel mixers, routing, and post-hoc suppressors remain
  closed at their recorded settings.
- A0 does not implement DOLG's multi-atrous local branch, learned Softplus
  attention, GeM, ArcFace, or retrieval recipe. It tests the orthogonal fusion
  equation against already-trained, dimension-aligned TRKH features before any
  new trainable branch is allowed.
- After A0, do not sweep feature layer, projector, local attention, pooling,
  bbox mask, normalization, epsilon, readout, threshold, fold, seed, lighting,
  or gate values. Do not rescue a failure by adding ASPP, ArcFace, GeM, a
  pretrained backbone, or another closed local/global mechanism.

## Locked data and cohort

- Dataset declaration:
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`.
- Dataset declaration SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Split access: `train` only. Validation and test are forbidden.
- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Launcher declaration SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- Cohort declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`.
- CIDT summary/prediction SHAs:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad` /
  `2e0993752d58d99ea429bfe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Source-disjoint folds: `1,2,3,4`; fold `0` never enters A0.
- Positive rows: clean keeper prediction `1`, target `1`.
- Negative rows: clean keeper prediction `1`, target in `{0,2,4}`.
- Locked counts: `607 = 421 TP + 186 restricted FP`.
- Fold counts: `112/45`, `100/48`, `101/52`, `108/41` TP/FP.
- Ordered cohort-index SHA-256:
  `a2689d1be8579386eea9ef02a826822a5e7d78e2d29439e8947c1731ccc738bd`.

The clean declaration replay uses the normal deployment forward at canonical
batch 64. It must reproduce only the known sample-3657 near-tie declaration
exception; any additional mismatch closes A0.

Before CUDA model loading, the auditor must repeat the existing CPU geometry
preflight on the exact ordered cohort. Normalized `xywh` bbox bytes must hash to
`e9b2143c9dbc7c6f5bf3a38f80a43483bd436e3b5ce99116c3d5989f79892f6b`.
At the `16x16` patch grid, the fixed cell-center rule must produce `28..210`
object cells and `46..228` context cells, with no empty row. No intersection
mask, erosion, dilation, margin, imputation, or row exclusion is permitted.

## Locked feature points

All features are captured by side-effect-free hooks during the standard keeper
deployment forward. The auditor may not call a trace-only or pruning-disabled
model path to create descriptors.

1. `local_tokens`: exact output of `model.patch_embed`, after the keeper CNN
   stem and before position embeddings, prefixes, Transformer blocks, or token
   pruning. Shape is `[B,256,256]`, interpreted as `16x16` local descriptors.
2. Normalize every local token along its 256-channel axis with the official
   `F.normalize(..., p=2, dim=channel, eps=1e-12)` convention. This mirrors the
   normalization in the official DOLG local attention branch; A0 adds no new
   learned attention.
3. `global`: exact input delivered to `model.head` by the standard forward,
   after the keeper's native global prefix pooling and fine-grained pooling.
   Shape is `[B,256]`. It is captured before the existing independent CNN-logit
   addition and pairwise logit adjustment.
4. `full_local`: uniform mean of all 256 normalized local tokens.
5. `object_local`: uniform mean of normalized local tokens whose fixed cell
   centers lie inside the transformed bbox.
6. Projection, pooling, and descriptor assembly run in an explicit FP32 island.
   The keeper's normal autocast policy remains unchanged.

The local and global vectors already share the keeper's 256-dimensional patch
embedding basis. No learned or random projector is introduced. This is the only
feature-point choice authorized in A0.

## Locked roles

For each row and condition, compute exactly seven roles:

1. `global`: the 256D global vector.
2. `full_local`: the 256D full-local mean.
3. `object_local`: the 256D object-local mean.
4. `global_object_concat`: `[global, object_local]`; direct-concatenation
   control with the same 512D size as the candidate.
5. `global_full_orth`: `[global, orthogonal(full_local, global)]`; official
   equation without object conditioning.
6. `global_object_orth`: `[global, orthogonal(object_local, global)]`; sole
   candidate.
7. `bbox_geometry`: the established deterministic 18-value normalized bbox
   descriptor, with no image feature.

No role ensembling, weighting, score calibration, or post-hoc routing is
allowed. The matched concat control isolates whether orthogonal subtraction,
rather than merely exposing local features, adds useful information. The full
orthogonal control isolates whether object conditioning is useful while the
global vector preserves wide context.

## OOF readout and conditions

- Four source-disjoint OOF rounds: hold out one fold from `1..4`, fit on the
  other three.
- Fit descriptors: clean condition only.
- Standardization: clean-fit per-feature mean and standard deviation; standard
  deviations clamp to `1e-6`.
- Readout for every role: scikit-learn logistic regression, `solver=lbfgs`, L2,
  `C=0.05`, intercept enabled, `max_iter=2000`, `tol=1e-9`, no class weighting,
  `random_state=42`.
- A convergence warning or iteration-limit hit fails A0.
- Conditions, with clean-fit normalization/readout frozen:
  - `clean`: brightness `1.00`, contrast `1.00`;
  - `lighting_dim`: brightness `0.70`, contrast `0.90`;
  - `lighting_bright`: brightness `1.25`, contrast `1.10`;
  - `low_contrast`: brightness `1.00`, contrast `0.65`.
- Each fold threshold is selected only on its clean fit rows as the highest
  score retaining at least `90%` fit TP, then frozen for held-out clean and all
  shifted rows.

The artifact must retain row-level scores, fold thresholds, AUROC, TP retention,
restricted-FP rejection, precision, F1, fold/condition deltas, descriptor norms,
projection energy, and an independent CSV replay exact within `1e-12`.

## Signal gates

All checks are conjunctive. `global_object_orth` passes only if it satisfies:

1. clean OOF AUROC `>=0.67`;
2. clean AUROC gain over `global >=0.025`;
3. clean AUROC gain over matched `global_object_concat >=0.015`;
4. clean AUROC gain over `global_full_orth >=0.015`;
5. clean AUROC gain over `bbox_geometry >=0.05`;
6. at least three of four fold AUROC deltas versus concat are positive, with no
   fold delta below `-0.02`;
7. mean shifted-condition AUROC gain versus concat `>=0.01`, with no shifted
   delta below `-0.015`;
8. clean held-out TP retention `>=0.90`;
9. clean restricted-FP rejection `>=0.25` and at least `0.05` above concat;
10. clean precision `>=0.74`, at least `0.01` above concat, and clean F1 is not
    below concat;
11. every shifted condition has TP retention `>=0.88` and FP rejection
    `>=0.18`;
12. all values are finite, every global norm is nonzero, the orthogonal
    descriptor differs from concat, and each candidate feature has nonzero
    clean cohort variance.

These are information gates, not claims that a future learned DOLG branch will
improve classification. A metric miss cannot be overridden by visual review.

## Equation, precision, and deployment gates

Before OOF fitting, the isolated implementation must pass:

- exact paper/source/license/commit/tree/protocol/input hashes and a clean
  tracked official-source worktree;
- official-source equation replay in float64 with output error `<=1e-12`;
- an independently written projection oracle with error `<=1e-12`;
- orthogonality residual `|orthogonal dot global|` relative error `<=1e-10`;
- spatial-project-then-pool versus pool-then-project equivalence `<=1e-12`;
- input/global gradient agreement `<=1e-10` and finite-difference error
  `<=1e-5`;
- BF16-input/FP32-island descriptor max error `<=0.05`, finite gradients, and
  unchanged bbox/mask coordinates;
- standard ONNX export with no custom domain, ONNX Runtime error `<=1e-5`, and
  TensorRT parser/build success without retaining an engine;
- standard-forward hook parity: exact normal logits and predictions, exact
  `[B,256,256]` local and `[B,256]` global shapes, and exactly one capture from
  each locked module per forward;
- full-forward candidate-sidecar runtime ratio `<=1.10` and peak allocated
  memory ratio `<=1.10` at batch 32 after warm-up and at least five repeats;
- no timing decision while another owned train/audit/export process is active.
  High unrelated GUI GPU load must be recorded, and formal timing must wait for
  an isolated enough workload instead of terminating unknown processes.

Any structural or deployment failure closes A0 even if the OOF signal is good.

## Visual audit

After complete scoring, render four deterministic eight-row contact sheets:

- TP rows with the largest positive candidate-minus-concat OOF advantage;
- restricted FP rows with the largest positive advantage;
- TP rows with the largest negative advantage;
- restricted FP rows with the largest negative advantage.

Each row must show the exact transformed image, bbox/object-cell overlay, the
`16x16` per-cell orthogonal-residual energy heatmap, target and keeper
probability, fold threshold, candidate/concat scores, object/global cosine, and
removed parallel-energy ratio. The reviewer must check all 32 rows for mapping
errors and whether the candidate is merely reacting to outside background or
bbox size. Missing or misaligned visuals fail completeness; visuals cannot
rescue a numeric failure.

## Stop and promotion rules

- Any provenance, geometry, equation, gradient, parity, deployment, signal,
  independent replay, or visual-completeness failure closes A0.
- On failure: no DOLG model/trainer/config integration, no image epoch,
  validation, test, full train, command update, or nearby projector/layer/mask/
  attention/readout sweep.
- On full pass: write and hash a separate matched train-only protocol before
  integrating code. At most one source-disjoint five-epoch scratch
  control/candidate pair may be authorized.
- The short pair must later pass full prediction, confusion, condition, XAI,
  normal-forward parity, resource, checkpoint-reload, and visual gates before
  validation is even considered.
- Current-best commands/history may change only after a later candidate wins
  the locked promotion gate. Their lock-time SHAs remain
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` and
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
