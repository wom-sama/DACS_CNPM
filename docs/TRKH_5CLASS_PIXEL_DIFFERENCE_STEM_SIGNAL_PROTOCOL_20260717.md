# TRKH 5-Class Pixel-Difference Stem Signal A0 Protocol - 2026-07-17

## Status

Prospectively locked before implementation or PDC measurement. This document
authorizes one fit-only information gate. It does not authorize trainer/model
integration, an image-training epoch, the 1,843-row train holdout, official
validation/test, a probe/full train, or a current-best command revision.

## Research Question

Does the official PiDiNet pixel-difference operator expose a stable fruit
surface/boundary signal that separates current class-1 true positives from
restricted `{0,2,4} -> 1` false positives better than the same frozen vanilla
TRKH stem?

This is deliberately narrower than asking whether another edge detector makes
cleaner maps. The gate must retain class-1 positives, reject false positives,
work from object-core/boundary features without relying on background alone,
and preserve direction under dim, bright, and low-contrast transforms.

## Primary-Source Lock

- Accepted paper: Su et al., "Pixel Difference Networks for Efficient Edge
  Detection," ICCV 2021 (oral).
- CVF page:
  `https://openaccess.thecvf.com/content/ICCV2021/html/Su_Pixel_Difference_Networks_for_Efficient_Edge_Detection_ICCV_2021_paper.html`
- Accepted PDF:
  `https://openaccess.thecvf.com/content/ICCV2021/papers/Su_Pixel_Difference_Networks_for_Efficient_Edge_Detection_ICCV_2021_paper.pdf`
- Locked local PDF:
  `D:/DataAI/external_sources/papers/pidinet_iccv2021.pdf`
- Paper SHA-256:
  `7ac637512d852cee0ba6cf2d475c3bb06ded4c34431be4b1d93cea74e1da4e09`
- Official repository: `https://github.com/hellozhuo/pidinet`
- Locked commit/tree:
  `d21aa881ed9c628571636fad39acfe1fad517ebd` /
  `b57c16070137773a76d09356df70dd973632887f`
- Official `models/ops.py` SHA-256:
  `b71294df26463b3caf075f1cc596e9849b2859f35db230346ed7f02f35a4ba34`
- Official `models/ops_theta.py` SHA-256:
  `8672cc4732c50bb1b8245f53d47a9535d59246f9ec206628022f36a8184ca2cb`
- Official `models/config.py` SHA-256:
  `5790cb9469694f7c3f2e690b8f9964b0fbc5f8e82ac3e18b9d28ad208f01a752`
- Official `models/convert_pidinet.py` SHA-256:
  `72cd01fe9e8e1ba4ffb599e0b1c1c5cf83e626971029b9be37b4aefb9d9ad519`
- Official `models/pidinet.py` SHA-256:
  `ee4243d1537f95a8a9ab795e47380115886e4984415c1586c2abcd913bfb8c7b`
- Official license SHA-256:
  `81ae4ff9ec8a220b015473aba1adbe1ee1c2e9cae844f8dc57fd5848023ba2c1`
- Repository popularity when checked: 614 GitHub stars and 87 forks.

The source license contains both a research-purpose notice and MIT-style grant
text. This A0 is research-only and independently reimplements the published
equations. Any later commercial deployment requires a separate license review.

The paper reports a scratch-trained edge detector, not mango classification.
Its evidence is therefore motivation, not proof. The official CAR-v4 schedule
cycles central, angular, radial, and vanilla operators. TRKH has exactly three
stem convolutions, so this A0 locks the first three source roles only:
`CD -> AD -> RD`. It does not claim to reproduce the full PiDiNet network.

## Source Equations

For a learned `3x3` kernel `W` and source `theta=1`:

1. Central difference (`CD`) subtracts `sum(W)` from the center coefficient.
2. Angular difference (`AD`) subtracts the official clockwise permutation
   `[3,0,1,6,4,2,7,8,5]` from the flattened `3x3` kernel.
3. Radial difference (`RD`) maps the eight non-center coefficients to official
   positive positions `[0,2,4,10,14,20,22,24]` and negative positions
   `[6,7,8,11,13,16,17,18]` in a `5x5` kernel; its center is zero.

The official conversion must produce an ordinary convolution with the same
forward and input/weight gradients. Only the converted ordinary-convolution
form is eligible for later ONNX/TensorRT deployment.

## Local No-Repeat Boundary

AdaFace is not selected. Its accepted CVPR-2022 paper and official MIT source
were checked, but current keeper head-feature norm separated class-1 TP from
restricted FP at AUROC `0.557205` on train and reversed to `0.447959` on
validation. It is also too close to rejected plain angular margin, reliability-
gated center margin, and sub-center proxy routes.

Neighborhood Attention is not selected. The CVPR-2023 paper and official MIT
repository are credible, but the exact operator requires NATTEN. The current
Windows/Python-3.9/Torch-2.6.0+cu124 environment has no NATTEN installation,
the official matching wheel index exposes Linux wheels, and a custom operator
would add ONNX/TensorRT risk. A hand-written approximation would overlap the
already closed shifted-window/FAA/DAT/BRA/local-attention families without
faithfully testing the source implementation.

PDC is distinct from the rejected fixed LBP, Gabor, Sobel/HOG, SRM, Deep-TEN,
frequency, second-order, color-normalization, and attention routes because it
tests a learned-kernel-compatible spatial difference operator with an exact
ordinary-convolution conversion. Failure closes only this exact frozen-weight
CAR stem signal. It does not reject a separately sourced full PiDiNet edge
detector or arbitrary gradient features.

## Locked Inputs And Cohort

- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Keeper launcher arguments SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`
- Data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`
- Data YAML SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`
- Ordered train/fold declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`
- Declaration SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`
- Companion summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`

Use only clean declaration rows whose fold is `1,2,3,4`. Fold `0` is the
forbidden 1,843-row train holdout. The fit-only binary information cohort is:

- positive: target `1`, keeper prediction `1` (`421` rows);
- negative: target in `{0,2,4}`, keeper prediction `1` (`186` rows).

Per-fold positive/negative counts are locked to `112/45`, `100/48`, `101/52`,
and `108/41`. The other fit rows may be used only to verify the declaration and
dataset mapping; they may not enter the readout. The eleven fit class-1 false
negatives are excluded because this A0 asks whether an added signal can protect
current true positives while rejecting current false positives.

## Exact Frozen Stem And Descriptor

1. Load the keeper with deployment semantics and require the standard clean
   forward to reproduce every one of the 607 locked declarations at batch 64.
2. Clone its three `HybridConvStem` convolutions and all BatchNorm affine/
   running state. Native uses `CV -> CV -> CV`; candidate converts the same
   weights to `CD -> AD -> RD`. Both retain the same GELU and max pooling.
3. No parameter is trained, randomized, selected, or changed. Transformer,
   classifier, bbox fusion, heads, and raw datasets remain untouched.
4. Record each complete block output. Map the object bbox to that resolution.
   Define core by shrinking each bbox edge inward by `20%`; boundary is bbox
   minus core; outside is the complement of bbox. Every region must be nonempty.
5. For each of three blocks and each region, record mean absolute response and
   RMS response. Add per-block bounded contrasts
   `(boundary-core)/(boundary+core+eps)` and
   `(core-outside)/(core+outside+eps)`.
6. Object-only features contain core/boundary statistics plus the first
   contrast (`15` values). Context-complete features additionally contain
   outside statistics and the second contrast (`24` values).
7. Extract the same features for clean, dim `(0.70,0.90)`, bright
   `(1.25,1.10)`, and low contrast `(1.00,0.65)` brightness/contrast settings.
   The cohort, folds, bboxes, model state, and ordering remain fixed.

## Fixed OOF Readout

Run four source-fold OOF splits. For each held-out fold:

- fit `StandardScaler` on the other three clean folds only;
- fit `LogisticRegression` with `solver=liblinear`, `C=0.1`, balanced class
  weights, intercept enabled, `max_iter=1000`, tolerance `1e-6`, seed `42`;
- score the held-out clean rows and the same rows under all three shifts with
  that clean-fitted scaler/model.

Run this recipe independently for native/PDC and object-only/context-complete
features. No C/solver/feature/fold/threshold/seed sweep is allowed.

For each role, define the clean protection threshold as the lower empirical
`3%` quantile of clean OOF class-1-TP scores using NumPy method `lower`. Report
TP retention and restricted-FP rejection at this one clean-frozen threshold for
all conditions. This threshold is diagnostic only and may not become a runtime
classifier or post-hoc prediction gate.

## Provenance And Equation Gates

All must pass before the information gate can authorize integration:

1. Every source/input/command hash and official commit/tree matches; official
   and tracked TRKH worktrees are clean. Protected untracked reports are ignored.
2. Preflight creates no output and opens no dataset, holdout, validation, or test.
3. Local CD/AD/RD kernels exactly match an independent equation oracle and the
   official functions on deterministic random and real keeper weights.
4. Direct and converted FP32 output/input-gradient/weight-gradient maximum error
   is `<=1e-6`; finite-difference error is `<=1e-4`.
5. Converted BF16 output error versus its FP32 reference is `<=0.02`, all
   gradients are finite, and every operator is nondegenerate versus vanilla.
6. The standard keeper reproduces all 607 clean target/prediction declarations,
   the ordered index hash, fold counts, and cohort counts exactly.
7. Every bbox/core/boundary/outside mask is valid and every descriptor/readout
   value is finite. Each fold contains both classes and every fit converges.
8. A static-batch-1 converted stem exports at ONNX opset 17 using only standard
   operators; ONNX Runtime FP32 maximum error is `<=1e-5` with matching shapes.
9. Median converted-stem CUDA runtime is `<=1.50x` native and peak allocated
   memory is `<=1.25x` native over three batch-64 repeats.

## Information And Precision Gates

All gates use OOF predictions on the 607 fit-only rows.

Object-only PDC must satisfy:

- clean TP-vs-restricted-FP AUROC `>=0.64`;
- clean AUROC improvement over native `>=0.025`;
- minimum shifted AUROC `>=0.58`;
- PDC AUROC is no worse than native in at least two of three shifts, with worst
  shifted delta `>=-0.015`;
- clean TP retention `>=0.97` and restricted-FP rejection `>=0.20` at the
  clean-frozen threshold;
- clean restricted-FP rejection improvement over native `>=0.08`;
- every shifted TP retention `>=0.90`, every shifted FP rejection `>=0.10`, and
  aggregate shifted PDC FP rejection exceeds native;
- median PDC score for TP is strictly greater than FP in every condition.

To prove the gate is not passed by background alone:

- clean object-only AUROC is no more than `0.02` below context-complete AUROC;
- object-only PDC still beats object-only native by the locked clean margin;
- at least two PDC blocks have median bbox-boundary response greater than
  outside response for both TP and FP cohorts on clean images.

Render a fixed contact sheet for the four highest and four lowest clean PDC OOF
scores in each cohort. Each panel contains the source image, bbox/core overlay,
block-3 PDC energy map, native/PDC object-only scores under all conditions, and
fold/target/keeper prediction. The renderer is an audit, not a selection input.

## Stop Rule And Next Boundary

Any failed provenance, equation, mapping, deployment, resource, or information
gate rejects A0. Still preserve compact quantitative/contact-sheet evidence,
independently replay the CSV/summary, update journal/TODO/skill, verify retention,
and close all nearby theta/operator-order/block-count/region/readout/threshold/
fold/seed variants. Do not run the holdout, validation, test, image training,
or XAI on a rejected frozen signal.

Only a complete A0 pass authorizes a separately precommitted default-off PDC
stem implementation and one matched source-disjoint five-epoch scratch pair.
That later pair must use the full audit/XAI/deployment gate before validation.

Current-best commands remain locked at SHA-256
`36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`;
command history remains locked at SHA-256
`39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
