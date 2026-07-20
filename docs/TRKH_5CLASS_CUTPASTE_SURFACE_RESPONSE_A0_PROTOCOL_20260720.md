# TRKH 5-Class Object-Interior CutPaste Surface-Response A0 Protocol - 2026-07-20

## Status And Scope

Prospectively locked before auditor implementation or any new candidate metric.
This protocol authorizes one source-disjoint, train-only information audit of
the frozen current keeper. It does not authorize validation/test access, model
or trainer edits, a checkpoint, a smoke/probe/full train, or current-best
command promotion.

## Prospective Geometry Erratum - 2026-07-20

This erratum was locked after a geometry-only 12-row image preview and before
any keeper forward, response descriptor, readout, validation/test access, or
candidate metric. The original `crop_bbox`-rectangle rule was unsafe because
`crop_bbox` describes the valid transformed crop, not a mango segmentation:
the preview exposed placements on a hand, basket, and background. Its SHA-256
is `49454d5f2b619ff2b59b594bdd16ba5b0d4324294aff6dda9b1458f4a7bf664a`.

The exact valid support is prospectively replaced by the intersection of:

- the existing deterministic `_surface_detail_foreground_mask_array` computed
  from denormalized evaluation RGB with `margin=0.08`;
- `crop_bbox` and `image_mask`;
- a centered crop-bbox ellipse with radius factor `0.92`;
- a final binary erosion of two pixels (`5x5` invalid-mask max pool).

The regular-patch area ratio is relative to the final valid-support pixel
count. For each sampled shape, enumerate every fully valid source placement
and every fully valid destination placement satisfying source/destination IoU
`<=0.05`, then sample uniformly from those finite sets with the locked local
generator. Resample the shape/geometry at most ten times. Persist the support
mask hash and both candidate counts for every record.

A simple ellipse-only preview still leaked background, while a GrabCut variant
did not improve the deterministic surface mask. Static review then removed an
unintended duplicate bbox inset so that the implementation performs exactly one
two-pixel erosion after all intersections. The exact final geometry-only
surface-support preview passed practical inspection at SHA-256
`4ec2c8eda5713af947cfc0fc6deb594238844a35d813e6c5c0c24c6df8d59d6e`.
No model was loaded for that correction or preview. These previews selected
geometric validity only and cannot authorize or rescue any mechanism gate.
This erratum supersedes only the conflicting support and placement wording
below; all cohort, seed, descriptor, readout, gate, and stop rules remain
unchanged.

## Research Question

Does the frozen keeper respond to object-interior CutPaste irregularities in a
way that distinguishes true class-1 predictions from restricted
`0/2/4 -> 1` false positives, beyond the information already present in its
clean probabilities and beyond a geometry-matched Cutout response?

Class 1 is the rare conjunction of an unripe mango and a mild damage/risk cue.
The current keeper already localizes the fruit well, while background blur and
gray perturbations are nearly inert and object desaturation is dominant. The
missing signal is therefore expected to be a local surface irregularity, not
another global context or CLS-attention selector.

This A0 is intentionally an information gate. A positive result would justify
one later default-off CutPaste auxiliary branch trained jointly with TRKH. A
negative result closes the exact frozen-keeper response descriptor and prevents
an expensive auxiliary-training experiment without claiming that every
possible anomaly detector is impossible.

## Primary Source And Licensed Equation Reference

- Accepted primary paper: Chun-Liang Li, Kihyuk Sohn, Jinsung Yoon, and Tomas
  Pfister, *CutPaste: Self-Supervised Learning for Anomaly Detection and
  Localization*, CVPR 2021, DOI `10.1109/CVPR46437.2021.00954`.
- Local paper:
  `D:/DataAI/external_sources/official/cutpaste-cvpr2021/Li_CutPaste_CVPR_2021.pdf`,
  SHA-256
  `e40ec13e20ded2a5307cd2cd4ddb04027ad3f657a2b5490bdcbb5892d81617f4`.
- Local supplemental:
  `D:/DataAI/external_sources/official/cutpaste-cvpr2021/Li_CutPaste_CVPR_2021_supplemental.pdf`,
  SHA-256
  `fa89f983a759076082b5be87c92cbe4fdf45ba958175f4b48fed736051f41ea6`.
- The paper says its rectangle sampler closely follows PyTorch
  `RandomErasing`. The licensed equation/code reference is the official
  `pytorch/vision` repository at tag `v0.21.0`, matching installed
  `torchvision 0.21.0+cu124`:
  - commit `7af698794eded568735f9519593603c1ec889eba`;
  - tree `9188eecffeb7a0febd04dea72028a9b59dfece0a`;
  - BSD-3-Clause `LICENSE` SHA-256
    `c06363f9d33627dc5173b13522444cc85fbb4739f63e16d4587d5c0a165b5b1e`;
  - `torchvision/transforms/transforms.py` SHA-256
    `75c1d80d921b60e5bab0ecb01f8d6e3a205df6d3df12039d7c56d2cea38db5f0`.

The paper provides the authoritative CutPaste objective and augmentation
ranges. It does not provide an authors' public implementation. The auditor is
therefore an independent implementation checked against the paper equations
and the official licensed PyTorch rectangle sampler, not a claim of copying an
unavailable official CutPaste codebase.

The primary recipe trains a ResNet-18 from scratch for 65,000 updates, defines
256 updates as one paper epoch, and studies 128-384 such epochs. That schedule
is incompatible with TRKH's at-most-30-dataset-epoch constraint. A0 transfers
only the local-irregularity hypothesis and exact geometry ranges; it does not
transfer the schedule, ResNet, Gaussian density estimator, MVTec assumptions,
or five-model ensemble.

## Literature Screen And No-Repeat Boundary

- CAL was screened and rejected before code. Its exact counterfactual logit
  difference is new, but the released fine-grained recipe is a 32-map bilinear
  attention-pooling model with attention crop/drop, center loss, pretrained
  ResNet-101, 448-pixel inputs, and a 160-epoch configuration. Those components
  overlap closed TRKH attention, pooling, crop/drop, center, and pretrained
  routes, and its maps are not a direct class-1 surface cue.
- DRAEM was screened but not selected. Its official method uses an external
  texture source, a large reconstruction/segmentation stack, and a long anomaly
  recipe, conflicting with the immutable raw-data and <=30-epoch constraints.
- Prior TRKH class-1 intra-class foreground SnapMix and class-1-to-`0/2/4`
  counterexample paste used five-class labels on mixed images. Both widened
  class-1 false positives and were rejected. A0 does not reopen their weights,
  areas, pairs, or classification targets.
- RGB masked reconstruction, Cutout/erasure regularization, local-zoom defect
  heuristics, deterministic surface statistics, local texture descriptors,
  class-specific query/MIL/WILDCAT, Finer-CAM, and attention/token selectors
  are closed. A0 may use Cutout only as a matched response control and may not
  turn either response into a validation-fitted suppressor.
- Raw images, labels, YAML, split declarations, and retained evidence are
  immutable. All transformations are ephemeral tensors and reproducible audit
  artifacts.

## Locked Inputs

- Pre-protocol HEAD/upstream:
  `1a34f8ee0a2ecc8f4632145c240ce527650432c2`.
- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper `launcher_args.json`, SHA-256
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- Dataset declaration: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- CIDT train-only source-fold summary and prediction CSV:
  `runs/audit_cidt_readiness_full_train_20260714/summary.json` and
  `predictions_all_conditions.csv`, SHA-256
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`
  and
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Current-best command/history SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`
  and
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

The auditor must verify all locks and the seven protected untracked payload
hashes before opening image pixels. The TRKH tracked worktree must be clean and
HEAD must equal upstream for the formal run.

## Locked Cohort And Folds

Use only CIDT `clean` train rows whose keeper prediction is class 1 and whose
target is in `{0,1,2,4}`. The cohort is exactly 750 objects:

- 528 true class-1 positives;
- 222 restricted false positives;
- fold TP/FP counts:
  `107/36`, `112/45`, `100/48`, `101/52`, `108/41`;
- restricted-FP target totals by fold:
  `0:[23,36,37,36,26]`, `2:[12,9,8,13,12]`,
  `4:[1,0,3,3,3]`;
- ordered sample-index SHA-256, serialized as comma-joined ASCII integers with
  no trailing delimiter,
  `55913ec45265b156a611dc96c779e46b08f28582737af8269105afe6f65694d7`.

Use the immutable CIDT folds `0..4`. Each OOF readout fits four folds and is
applied once to the held fold. Fit and held source stems must have zero overlap.
Every path must resolve below `yolo_f/images/train`; any validation/test path,
prediction, label, pixel, or metric access is fatal.

## Exact Surface Transform

Start from the keeper's ordinary 256x256 evaluation tensor and metadata. Undo
only the ImageNet channel normalization to obtain clamped RGB `[0,1]`, apply
the synthetic transform, then restore the same normalization. Lighting shifts,
illumination normalization, padding, and background suppression happen before
CutPaste. The keeper receives its unchanged native `bbox` and `image_mask`.

Sampling is limited to the final surface-support mask defined by the prospective
geometry erratum. A sample is fatal if no valid geometry can be drawn after ten
shape attempts. Source and destination masks must be fully valid, come from
uniform draws over their enumerated valid-placement sets, and have
intersection-over-union at most `0.05`.

For each sample, condition, and draw `d in {0,1,2,3}`, seed a local CPU
generator from `20260720 + 1000003*sample_index + 1009*d + condition_id`.
Global Python, NumPy, and Torch RNG state may not be consumed.

### Standard CutPaste

- sample pasted area ratio log-uniformly in `[0.02,0.15]` relative to the
  final valid-support pixel count;
- sample aspect ratio log-uniformly in `[0.3,3.3]`;
- select source and destination uniformly among fully valid placements;
- apply brightness, contrast, saturation, and hue jitter in a deterministic
  randomized order, each with maximum intensity `0.1`;
- paste without alpha blending or rotation.

### CutPaste-Scar

- sample short side uniformly from integer pixels `[2,16]` and long side from
  `[10,25]`;
- select source uniformly among placements fully inside final valid support;
- apply the same maximum-0.1 color jitter;
- rotate uniformly in `[-45,45]` degrees with bilinear RGB and nearest mask;
- select a destination uniformly among placements that keep the full rotated
  mask in final valid support and satisfy the locked IoU, then paste by the
  rotated binary mask.

### Matched Controls

- `cutout_regular` and `cutout_scar` use the exact candidate destination masks
  and fill them with the per-image valid-object RGB mean;
- `identity_repaste` copies an unjittered patch back to its exact source
  location and must be bit-exact;
- candidate and Cutout masks, changed-pixel support, geometry records, and RNG
  lineage are persisted for independent replay.

No patch-size, draw-count, seed, jitter, bbox erosion, rotation, fill, or blend
sweep is allowed after metrics.

## Frozen Forward And Response Descriptor

The keeper remains in evaluation mode and all state tensors must remain
bit-identical. Run the ordinary metadata-aware path for clean, CutPaste,
CutPaste-Scar, and matched Cutout tensors. Persist logits, probabilities, and
the final pooled embedding.

For every transformed view relative to clean, compute exactly five scalars:

1. class-1 logit delta;
2. class-1-versus-maximum-`0/2/4` logit-margin delta;
3. Jensen-Shannon divergence between clean and transformed probabilities;
4. cosine distance between L2-normalized pooled embeddings;
5. L2 distance between L2-normalized pooled embeddings.

For each transform family, aggregate the four draws by mean, standard
deviation with population denominator, minimum, and maximum. Standard
CutPaste plus CutPaste-Scar therefore contribute 40 response dimensions. The
two matched Cutout variants contribute the same 40 dimensions. The paired
contrast is elementwise CutPaste response minus its geometry-matched Cutout
response before aggregation and is also 40 dimensions.

Every role appends the same base vector: five clipped clean log probabilities
plus the clean class-1-versus-maximum-`0/2/4` logit margin.

Locked roles are:

- `base_only`: six base dimensions;
- `cutout_control`: base plus 40 Cutout dimensions;
- `cutpaste_candidate`: base plus 40 CutPaste dimensions;
- `paired_contrast`: base plus 40 paired-difference dimensions;
- `source_deranged`: base plus candidate response cyclically reassigned within
  held fold and clean base-margin quartile, requiring a different source stem.

The deterministic derangement is created before labels are passed to the
readout. No PCA, feature selection, nonlinear expansion, learned image
adapter, anomaly-density estimator, threshold grid, or descriptor subset is
allowed.

## Source-Disjoint Readout And Precision Action

For each held fold and role:

- fit `StandardScaler` on fit rows only;
- fit binary `LogisticRegression` with L2 penalty, `C=0.1`,
  `class_weight=balanced`, `solver=lbfgs`, `max_iter=4000`, `tol=1e-9`, and
  seed `20260720`;
- positive label is a true class-1 keeper prediction;
- negative label is a restricted `0/2/4 -> 1` keeper false positive;
- choose on fit rows the lowest score that retains at least 97% of fit TP;
- apply the scaler, readout, and threshold once to the held fold.

An accepted row keeps the keeper class-1 prediction. A rejected row changes to
the keeper's highest-probability non-class-1 class. This action is diagnostic
only. It may not be exported as a deployment threshold or current-best model.

Persist all scaler/readout coefficients, iteration counts, fit/held sources,
thresholds, raw descriptors, logits/probabilities, geometry, OOF scores,
actions, transitions, and per-fold metrics. An independent replay must rebuild
descriptors, scores, thresholds, decisions, metrics, gates, and payload hashes.

## Conditions And Visual Audit

Stage A extracts clean tensors only. Always render a hash-locked contact sheet
for fixed TP and target-`0/2/4` FP rows with RGB input, source/destination
geometry, CutPaste, CutPaste-Scar, and both Cutout controls. Manual review must
confirm all changed support is on valid mango surface and that no padding,
background, text, hand, label, or crop boundary is used. Manual review cannot
rescue a failed automatic gate.

Only a complete clean automatic and visual pass authorizes Stage B. Stage B
repeats the fixed descriptors under CIDT lighting conditions:

- dim: brightness `0.70`, contrast `0.90`;
- bright: brightness `1.25`, contrast `1.10`;
- low contrast: brightness `1.00`, contrast `0.65`.

Each condition reuses the clean-fold scaler, readout, and threshold. No
condition refit or threshold change is allowed.

## Conjunctive Gates

All structural and mechanism gates are required.

Structural gates:

1. every input/source/current-best/protected hash matches, official torchvision
   worktree is clean at the locked commit/tree, TRKH HEAD equals upstream, and
   tracked worktree is clean;
2. exact 750-row cohort/order/counts, exact five source folds, zero fit/held
   source overlap, train-only paths, and no validation/test use;
3. clean probability maximum error versus CIDT is `<=2e-5` with exact argmax;
4. candidate/control masks are paired exactly, changed pixels are a subset of
   valid object support, all geometry is in range, all standard source/dest
   IoUs are `<=0.05`, identity-repaste maximum error is `<=1e-7`, and a second
   process reproduces geometry hashes;
5. all logits/descriptors/readouts are finite, all readouts converge without
   retry, candidate response effective rank is at least `8`, and keeper state
   is bit-identical before/after extraction;
6. requested/effective workers are recorded, peak CUDA allocation is at most
   `3.5 GiB`, and no competing compute process invalidates formal timing;
7. compile, pyflakes, focused/full tests, PowerShell parse, exact replay,
   artifact-manifest verification, and manual geometry review pass.

Clean mechanism gates for `cutpaste_candidate`:

1. OOF TP-versus-restricted-FP AUROC is at least `0.84`;
2. AUROC gains are at least `+0.02` over `base_only`, `+0.015` over
   `cutout_control`, and `+0.02` over `source_deranged`;
3. `paired_contrast` AUROC is at least `0.65` and beats `cutout_control` by at
   least `0.005`, proving that pasted surface structure contributes beyond
   generic occlusion geometry;
4. held-fold action TP retention is at least `0.95` and restricted-FP
   rejection is at least `0.25` in aggregate;
5. candidate rejects at least ten more restricted FP than both `base_only` and
   `cutout_control`, with corrections at least twice TP harms;
6. candidate AUROC beats both controls in at least four of five folds, no fold
   TP retention is below `0.90`, and no fold creates an action outside the
   locked keep/reject semantics;
7. target-0 and target-2 restricted-FP rejection are each at least `0.15`, and
   at least one of the ten target-4 FP is rejected without using target labels
   in the transform or threshold.

Stage-B condition gates for the clean-fitted candidate:

- every condition AUROC is at least `0.78` and gains at least `+0.01` over
  both clean-fitted controls;
- every condition retains at least `0.92` TP and rejects at least `0.15`
  restricted FP;
- no condition has corrections below harms or a target-0/2 FP-rejection rate
  below `0.08`.

No aggregate gain can compensate for a failed gate. Precision obtained by
broad true-class-1 suppression is a failure.

## Escalation And Stop Rules

If any Stage-A gate fails, close the exact frozen CutPaste/CutPaste-Scar
response descriptors, matched Cutout comparison, draw/geometry/jitter/seed
neighbors, response readouts, and test-time CutPaste voting on this keeper.
Do not run Stage B, integrate the model/trainer, open validation/test, run a
smoke/probe/full train, or update current-best commands.

Only a complete Stage-A and Stage-B pass authorizes one default-off
object-interior 3-way CutPaste auxiliary head. It must use only ephemeral
train-time views, preserve the default model bit-exactly, strict-load the
keeper, expose local anomaly maps in architecture/XAI audits, and pass a
matched at-most `120 batches x 2 epochs` full-validation smoke against an
identical control. A passing smoke still must reach validation macro F1
`>=0.884675`, class-1 precision `>=0.64`, recall `>=0.75`, F1 `>=0.70`, net
restricted-FP removal, corrections not below harms, and all robustness/export
gates before any autonomous full train can be considered.

Any later autonomous full train remains capped at 30 epochs with a justified
early-stopping patience and measured worker choice for the current RTX 4060
laptop. The current-best checkpoint, command packet, and update history remain
byte-identical until a locked independent-reload single-model validation winner
exists.
