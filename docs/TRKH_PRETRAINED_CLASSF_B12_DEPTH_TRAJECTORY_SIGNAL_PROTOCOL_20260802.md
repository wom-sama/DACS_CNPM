# B12 TRAIN-only signal gate: patch-depth trajectory

Protocol: `TRKH_PRETRAINED_CLASSF_B12_DEPTH_TRAJECTORY_SIGNAL_20260802`

Role: representation-readiness audit only. This protocol does not authorize a model, trainer integration, validation, test, full train, or deployment audit.

## Question and no-repeat boundary

B9 classifies only the last DINOv3 state. V2/V3 inject before block 11, A0/A1 operate at the final block or post-norm, and B11 operates after pooling. The older FFVT-lite and multi-granularity routes independently pool or select intermediate tokens and blend/attach CE, KL, or contrastive heads. B12 asks a different question: **does the state transition of the same spatial patch through B9 depth carry a class-1 correction direction that is absent from a single latest snapshot?**

The motivation is bounded by primary evidence, not used as proof: DINOv3 was designed to preserve dense features; Visual Query Tuning reports useful frozen intermediate ViT representations; Perception Encoder reports that useful embeddings can occur before the output; and ConvPass supports small visual adapters parallel to pretrained blocks. See [DINOv3](https://arxiv.org/abs/2508.10104), [VQT](https://openaccess.thecvf.com/content/CVPR2023/html/Tu_Visual_Query_Tuning_Towards_Effective_Usage_of_Intermediate_Representations_for_CVPR_2023_paper.html), [Perception Encoder](https://arxiv.org/abs/2504.13181), and [ConvPass](https://arxiv.org/abs/2207.07039). None establishes the TRKH mechanism; this gate must do so locally.

B12 is new only if all remain true: track aligned patch deltas in depth order; use no top-k, bbox/foreground prior, pooled-layer concatenation, auxiliary CE/KL/SupCon, ordinal target, uncertainty router, or raw-RGB specialist; strict-load and freeze selected-EMA B9; compare against both a latest-state spatial control and a depth-correspondence falsification; and keep all work TRAIN-only under immutable source-union folds. Failure closes the depth-trajectory proposal without changing layers, statistics, regularization, solver, or thresholds.

## Frozen inputs

- Canonical immutable dataset: `D:\DataAI\AIEx\newdataset\class_f`; TRAIN only, `8278` rows, ordered class support `[1987,497,1326,2080,2388]`.
- Selected-EMA B9 `best.pt`: checkpoint SHA256 `52de1b4dd45281d3531f36974f615fd71ba7c108747cca54b4052ef19e506d10`, selected state SHA256 `efe735ccbd70df9d056da8f9576203b326063678e311160b5d700be26fa2f9ad`.
- Reuse and verify the A0 path/label/base-logit cache and immutable fold assignment. Paths, labels, base logits, and assignment SHA256 are respectively `a472608c3d0b1ce1b941a1fa4bfa72b9f3032140d7f4c2792cb9bdd357b1c0fb`, `97a79688410600e6808140a441ddbfaaafddaf2294ffbbf77849adf0cd337a55`, `643cd63059df171a48770f7f7e9b15ccbc9df7001d09da2bb1200db4048d2f22`, and `afde4fec6b344b867d31e0e01b89f16c5f4be7fd551fdc1fdfd0ea34219d725f`.
- Five union-component folds remain unchanged and group every declared group, same crop source, adjacent source ID, and luminance-pHash radius-3 component together. B9 has seen TRAIN, so results are conditional representation evidence rather than independent generalization.

## Locked patch-depth representation

Run the frozen FP32 B9 once and tap zero-based blocks `[2,5,8,11]`. Apply B9's final norm to each state, remove its five prefix tokens, and retain the aligned `16 x 16` patch grid. Block-11 mean-patch logits through the frozen B9 head must match the existing base-logit cache within `1e-6`; otherwise no artifact is valid.

For rivals `j in {0,2,4}`, define the fixed unit direction `d_j=(W_1-W_j)/||W_1-W_j||`, using the frozen B9 classifier. Let `x_l(p)` be patch `p` after block `l` and final norm.

The candidate maps follow the same patch through depth:

`C_25_j(p)=<x_5(p)-x_2(p),d_j>` and `C_58_j(p)=<x_8(p)-x_5(p),d_j>`.

The equal-width latest-state control sees only block 8:

`S_level_j(p)=<x_8(p),d_j>` and `S_local_j(p)=<x_8(p)-N4(x_8)(p),d_j>`, where `N4` is circular four-neighbour averaging on the fixed `16 x 16` grid. This is a strong spatial control, not a zero/repeated-feature placebo.

The correspondence falsification preserves layer marginals but breaks same-patch identity:

`D_25_j(p)=<x_5(p)-roll_x(x_2,8)(p),d_j>` and `D_58_j(p)=<x_8(p)-roll_y(x_5,8)(p),d_j>`.

Each of the six maps per arm is summarized only by population mean, population standard deviation, and deterministic linear-interpolation quantiles `q10/q50/q90`, giving exactly `30` FP32 features per arm. No label, component, source, filename, or neighbour prediction enters a feature.

## Locked source-union OOF probe

For each binary pair `1-vs-j`, use that pair's ten arm features plus the frozen B9 margin `b_1-b_j`: exactly `11` inputs for candidate, latest-state control, and deranged control. Within each of five folds, fit preprocessing only on the other four folds: per-column mean/std with a `1e-8` floor, then deterministic L2 logistic regression (`solver=lbfgs`, `C=1.0`, `tol=1e-9`, `max_iter=2000`, intercept enabled, seed `20260802`). Fit weights are proportional to `n_class^-0.5` and normalized to mean one within the fit pair. The held fold is never used for scaling, fitting, thresholding, or model selection.

Report OOF AUROC and threshold-zero class-1 precision/recall/F1/TP/FP per pair and arm; raw frozen-B9 margin is an unfitted reference. Report all 15 pair-fold deltas, convergence state, feature finiteness, feature rank, coefficients, and same-patch versus deranged feature correlations. Statistical intervals use `5000` fixed-seed paired draws that resample whole union components within fold. No layer, feature statistic, solver, regularizer, class weighting, seed, or gate may be selected from the results.

## Conjunctive readiness gates

The signal gate passes only if every condition holds:

1. Candidate mean pair-AUROC exceeds the latest-state control by at least `+0.0010`, with the 95% component-bootstrap lower bound above zero.
2. Candidate mean pair-AUROC exceeds the deranged control by at least `+0.0010`, also with lower bound above zero.
3. Candidate beats the latest-state control in at least `4/5` folds by mean pair-AUROC; at least `10/15` pair-fold deltas are positive.
4. Candidate AUROC improves over the latest-state control for both critical pairs `1-vs-0` and `1-vs-2` by at least `+0.0002`; no pair is worse by more than `0.0002`.
5. Candidate mean pair-AUROC exceeds the raw B9-margin reference by at least `+0.0002` with bootstrap lower bound above zero. A trajectory that only replaces a weaker probe control does not justify a Hybrid.
6. At threshold zero, aggregate class-1 TP across the three pair tasks is at least `98%` of both B9 and the latest control, while aggregate rival FP is no higher than the latest control. This prevents buying precision by deleting true class-1 support.
7. Block-11/base parity, exact path/label/fold/cache hashes, all `8278` OOF rows, finite features/scores, solver convergence, and TRAIN-only split guards all pass.

Passing authorizes only a separate preregistration and implementation of `B12-DTSA-A0`: a patchwise low-rank diagonal state update over blocks `2/5/8`, injected before block 9 so blocks `9--11` integrate it. That future model must have fewer than `25,000` added parameters, strict B9 inheritance, a bit-exact disabled branch, an equal-parameter block-8-only control, and a depth-deranged falsification. Failure forbids writing that trainer or opening validation/test/mobile audits.

