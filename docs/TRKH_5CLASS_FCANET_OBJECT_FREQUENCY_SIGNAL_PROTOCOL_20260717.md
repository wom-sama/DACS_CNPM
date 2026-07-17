# TRKH 5-Class FcaNet Object-Frequency Signal A0 Protocol - 2026-07-17

## Decision scope

This protocol locks one fit-only signal gate before any FcaNet trainer or TRKH
runtime integration. The question is narrow:

> Does the official FcaNet top-16 DCT channel descriptor expose a repeatable,
> object-surface signal that separates current keeper class-1 true positives
> from restricted `{0,2,4}->1` false positives better than GAP, while the
> unchanged Transformer path retains the full image context?

The gate may authorize exactly one matched short train-only pair. It cannot
authorize validation, test, a full train, a current-best command update, or raw
dataset changes.

## Primary authority and provenance

- Accepted paper: Qin et al., *FcaNet: Frequency Channel Attention Networks*,
  ICCV 2021, pages 783-792:
  `https://openaccess.thecvf.com/content/ICCV2021/html/Qin_FcaNet_Frequency_Channel_Attention_Networks_ICCV_2021_paper.html`.
- Local accepted-paper PDF:
  `D:\DataAI\external_sources\papers\fcanet_iccv2021.pdf`.
- Paper SHA-256:
  `13c707b575722cdb81003ac2e1bcd6650eb2ef5984265666d495662766f69d55`.
- Official repository: `https://github.com/cfzd/FcaNet`.
- Locked official commit/tree:
  `aa5fb63505575bb4e4e094613565379c3f6ada33` /
  `85aa7989ca7d957e1ab87991c65eefdb7fb53a8d`.
- Official `model/layer.py` SHA-256:
  `9b4337558604958dca1257f1b05e92fbee8f546e536604de5cff83f2a3d0200e`.
- Official MIT `LICENSE` SHA-256:
  `31e61e165ea1409c53836aadf965a601a0c225aa7c1aaa4eb0c01e18559e4cf1`.
- GitHub API observation on 2026-07-17: `602` stars, `105` forks, MIT,
  not archived. Popularity is context only, not efficacy evidence.
- No FcaNet, ResNet, ImageNet, or other pretrained weights are permitted.

The paper establishes that GAP is the zero-frequency DCT component, proposes
fixed multi-spectral DCT compression, and reports top-16 as the best two-step
selection count in its ablation. The formal gate must replay the official
equations rather than reconstructing them from prose.

## No-repeat screen

This route is not permission to reopen adjacent families:

- Frequency-selective V15 FFTs each patch token along the embedding axis and
  selects patches. FcaNet instead applies fixed two-dimensional spatial DCT
  bases to channel maps. V15 remains closed.
- Active Gabor, Fourier/wavelet/high-frequency input experts, PDC, OctConv,
  XCA, SRM, Meta-ACON, GRN, large-kernel, local-attention, token-routing, and
  texture-pooling routes remain closed at their recorded settings.
- Coordinate Attention and Triplet Attention were screened out because their
  principal new signal is spatial/axis localization, while repeated TRKH XAI
  already shows foreground-clean attention and negligible far-background
  causal effect.
- No frequency-count, frequency-family, layer, residual, readout, crop,
  threshold, fold, seed, condition, or budget sweep is permitted after A0.

## Locked data and cohort

- Dataset declaration:
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`.
- Split access: `train` only. Validation and test are forbidden.
- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`.
- Cohort declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`.
- Source-disjoint folds: `1,2,3,4`; fold `0` never enters this A0 cohort.
- Positive rows: clean keeper predicts class 1 and target is class 1.
- Negative rows: clean keeper predicts class 1 and target is in `{0,2,4}`.
- Locked counts: `607 = 421 TP + 186 restricted FP`.
- Locked fold counts:
  - fold 1: `112 TP / 45 FP`;
  - fold 2: `100 TP / 48 FP`;
  - fold 3: `101 TP / 52 FP`;
  - fold 4: `108 TP / 41 FP`.
- Ordered cohort-index SHA-256:
  `a2689d1be8579386eea9ef02a826822a5e7d78e2d29439e8947c1731ccc738bd`.

The clean declaration replay uses the normal deployment forward at canonical
batch 64. It must reproduce the previously declared sample-3657 near-tie and
no other mismatch. A different mismatch closes A0.

## Geometry preflight locked before GPU inference

The exact clean transform and exact 607-row order were replayed on CPU before
this protocol was written. Normalized `xywh` crop bboxes were finite and had:

- width range `0.29296875..1.0`;
- height range `0.15234375..1.0`;
- bbox byte SHA-256:
  `e9b2143c9dbc7c6f5bf3a38f80a43483bd436e3b5ce99116c3d5989f79892f6b`.

Using cell centers on the exact dense grids:

| Grid | Object cells | Context cells | Empty object | Empty context |
| --- | ---: | ---: | ---: | ---: |
| `16x16` | `28..210` | `46..228` | `0` | `0` |
| `32x32` | `116..900` | `124..908` | `0` | `0` |

The formal auditor must repeat this CPU preflight before loading a model onto
CUDA. Any count, order, bbox hash, or nonemptiness change closes A0 before
feature extraction. It must not erode boxes, change intersection/center rules,
impute context, or exclude rows.

## Locked feature point and object alignment

1. Run the keeper CNN stem and patch embedding only, in eval mode, on the
   normal deterministic `256x256` input.
2. Reshape the dense pre-prefix patch embedding from `[B,256,256]` to
   `[B,256,16,16]`. No position embedding, prefix token, pruning, attention,
   classifier, or trace-only forward is used to create descriptors.
3. Preserve this full `16x16` map as the context-retaining representation.
4. Build a fixed `16x16` sampling grid from each transformed bbox and use
   bilinear `grid_sample`, `padding_mode=zeros`, `align_corners=False` to form
   one object-aligned `[B,256,16,16]` map. Grid points are output-cell centers
   within the bbox. The operation is transient; no image or label is written.
5. The future candidate, if authorized, may read this object branch while the
   original full-map Transformer path remains unchanged. A0 does not add that
   branch to the model.

Object alignment fixes the output shape and prevents a linear readout from
winning only through bbox area. A separate bbox-geometry control is still
required.

## Locked FcaNet descriptor

The candidate is the exact official `top16` selection at `16x16`, 256 channels:

- official x indices:
  `[0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2]`;
- official y indices:
  `[0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2]`;
- official scale: `16 // 7 = 2`;
- 16 channels per selected frequency;
- fixed orthonormal DCT-II filters;
- descriptor: elementwise feature/filter product summed over both spatial
  dimensions, producing `[B,256]`.

The GAP control is the official `top1` `(0,0)` DCT descriptor and uses the same
orthonormal scaling. It must be numerically proportional to direct GAP.

## Locked roles

Exactly five clean-trained roles are evaluated:

1. `full_gap`: top1 DCT on the full patch map.
2. `object_gap`: top1 DCT on the object-aligned map.
3. `full_top16`: official top16 DCT on the full patch map.
4. `object_top16`: official top16 DCT on the object-aligned map; sole candidate.
5. `bbox_geometry`: the existing deterministic 18-value normalized bbox
   descriptor, with no image features.

No concatenation of roles is allowed in A0.

## OOF readout and conditions

- Four source-disjoint OOF rounds: hold out one of folds `1..4`, fit on the
  other three.
- Fit data: clean descriptors only.
- Standardization: per-feature mean and standard deviation from the clean fit
  rows only; standard deviations clamp to `1e-6`.
- Readout: deterministic scikit-learn logistic regression, `solver=lbfgs`,
  `C=0.05`, L2 penalty, intercept enabled, `max_iter=2000`, `tol=1e-9`, no
  class weighting, `random_state=42`.
- The same readout protocol is used independently for every role. Convergence
  warnings or iteration-limit hits fail the gate.
- Conditions, with clean-fit normalization/readout frozen:
  - `clean`: brightness `1.00`, contrast `1.00`;
  - `lighting_dim`: brightness `0.70`, contrast `0.90`;
  - `lighting_bright`: brightness `1.25`, contrast `1.10`;
  - `low_contrast`: brightness `1.00`, contrast `0.65`.
- For decision metrics, each fold's threshold is selected on its clean fit
  rows as the highest score retaining at least `90%` fit TP. That threshold is
  frozen for the held-out clean and shifted rows.

The formal artifact must contain row-level role scores, fold thresholds,
AUROC, TP retention, restricted-FP rejection, precision, F1, fold deltas, and
condition deltas. An independent CSV replay must exactly reproduce summary
metrics within `1e-12`.

## Signal gates

All gates are conjunctive. A0 passes only if `object_top16` satisfies:

1. clean OOF AUROC `>= 0.65`;
2. clean AUROC gain over `object_gap >= 0.025`;
3. clean AUROC gain over `full_top16 >= 0.015`;
4. clean AUROC gain over `bbox_geometry >= 0.05`;
5. at least three of four clean fold AUROC deltas versus `object_gap` are
   positive, and no fold delta is below `-0.02`;
6. mean shifted-condition AUROC gain over `object_gap >= 0.02`, with no
   condition delta below `-0.015`;
7. clean held-out TP retention `>= 0.90`;
8. clean held-out restricted-FP rejection `>= 0.20` and at least `0.05` above
   `object_gap`;
9. shifted TP retention is at least `0.88` in every condition;
10. all descriptors/scores are finite, the top16 descriptor differs from top1,
    and every selected frequency group has nonzero cohort variance.

These are information gates, not claims that a trained attention gate will
improve classification.

## Equation, precision, and deployment gates

Before OOF fitting, the isolated extractor must pass:

- exact locked commit/tree/source/license/paper/protocol hashes and clean
  official worktree;
- official source-loader output error `<=1e-6` in float64;
- independent DCT-basis/output error `<=1e-12` in float64;
- input-gradient error `<=1e-8` and finite-difference error `<=1e-5`;
- BF16 versus FP32 descriptor max error `<=0.05`, finite gradients, and no
  object-grid coordinate change;
- `object_gap` proportionality to direct aligned-map GAP within `1e-6`;
- standard ONNX export with no custom domain, ONNX Runtime max error `<=1e-5`,
  and zero non-finite outputs;
- TensorRT parser/build success for the fixed-batch extractor;
- full keeper-forward hook runtime ratio `<=1.15` and peak allocated-memory
  ratio `<=1.10` at batch 32, after warm-up and at least three repeats;
- no standard/trace path substitution and no change to keeper logits or
  predictions during the side-effect-free descriptor benchmark.

A deployment failure closes A0 even if the descriptor is informative.

## Visual audit

After complete OOF scoring, generate fixed contact sheets for:

- eight TP and eight restricted FP rows with the largest positive
  `object_top16 - object_gap` normalized-score advantage;
- eight TP and eight restricted FP rows with the largest negative advantage;
- per row: source image, transformed bbox, object-aligned crop, target,
  keeper probability, both OOF scores, fold threshold, and strongest DCT
  frequency group.

The renderer must be deterministic and hash-locked. Visuals are diagnostic;
they cannot override a failed numeric or engineering gate. Missing/misaligned
images fail formal completeness.

## Stop and promotion rules

- Any geometry, provenance, equation, replay, deployment, signal, or visual
  completeness failure closes A0.
- On failure: no trainer/model/config integration, no five-epoch pair, no
  validation/test/full train, no nearby sweep, and no current-best command
  update.
- On full pass: prospectively lock a separate matched train-only protocol
  before code integration. At most one short pair may be authorized.
- A later candidate can update the current-best command/history only after it
  independently beats the keeper promotion gates. This A0 can never do so.

