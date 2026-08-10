# TRKH 5-Class CAP Integral-Region Context A0 Protocol

Date: 2026-07-24
Protocol ID: `trkh_cap_integral_region_context_a0_20260724`
State: prospective; no CAP candidate implementation, training, or metric exists

## Decision Scope

This A0 asks one narrow question:

> Do context-conditioned, multi-scale integral regions from the frozen scratch
> TRKH block-2 map distinguish true class-1 evidence from restricted
> `0/2/4 -> 1` false-positive evidence while retaining class-1 true positives?

The test is source-held and train-only. It may authorize one separately locked,
default-off trainer smoke only if every structural, causal, performance,
replay, resource, and visual gate passes. It is not:

- an end-to-end reproduction of CAP;
- a validation, test, smoke, probe, or full-train result;
- a production classifier or threshold;
- permission to update the current-best command;
- evidence that every class has reached F1 greater than 0.98.

The protocol and machine lock must be committed and pushed before candidate
code exists. A failure closes this exact frozen layer, region geometry,
candidate/control set, seed, optimizer, epoch, and calibration neighborhood.

## Primary Sources And License

Accepted primary sources:

1. Ardhendu Behera et al., *Context-aware Attentional Pooling (CAP) for
   Fine-grained Visual Classification*, AAAI 2021:
   <https://ojs.aaai.org/index.php/AAAI/article/view/16176>
2. Authors' official implementation:
   <https://github.com/ArdhenduBehera/cap>

Pinned local evidence:

- accepted paper:
  `D:\DataAI\external_sources\papers\Behera_Context_Aware_Attentional_Pooling_AAAI2021.pdf`;
- paper SHA-256:
  `2fd8cb834e44cd5819894ddf049e972c2f35d227a27727e0def36ec3f0624cb2`;
- official repository:
  `D:\DataAI\external_sources\cap_official`;
- commit:
  `cae53050af583c2b3fa38081c0abc227199e7dc1`;
- tree:
  `70f47d05ad0f6dafc8c43db32022a4955f4ce14a`;
- root license: MIT;
- license SHA-256:
  `624c3e2c494268065b8d00c99087e24520456eef496637d86558da6326f2e12e`;
- `train_CAP.py` SHA-256:
  `099652789ba95dce375f970a947ccd797bd13102c84b005781d877bf3e994ad9`;
- `SelfAttention.py` SHA-256:
  `ad67bff1c4a3c0a2c89c3edf08a8d657a86553c44d1a99003e601c67c8a2d8db`;
- `SeqAttention.py` SHA-256:
  `e6ab256d596ab8a8bdd59d69a8e988f36f783d3ef41d4243a4877fd7657cd5fa`;
- `RoiPoolingConv.py` SHA-256:
  `cdf12820d27d64354a7c67fb93052ca2d585a8b46936cdc3c4b7ca2882900eed`;
- `loupe_keras.py` SHA-256:
  `22816655f907fb79d7ad42a0d35759dda06ac6cfe559a6f143a60699542b5c1e`.

The TRKH implementation must be independent PyTorch code. It may use the
paper equations and MIT source as equation/provenance oracles, but must not
import TensorFlow/Keras CAP code or preserve its batch-ordering bug.

The accepted paper equation is the A0 authority where the paper and training
script differ. `train_CAP.py` additionally inserts squeeze-excitation, uses
spectral-normalized pixel projections, and passes
`attention_activation='sigmoid'` before the region softmax. Those three
script-only additions are excluded from A0 so that the gate isolates the
published CAP equation on frozen TRKH features. There is no pre-CAP channel
recalibration, no projection spectral normalization, and no activation between
`W_alpha beta + b_alpha` and softmax. These choices are locked before metrics
and cannot be added after a failed result.

## Why This Route Is Distinct

The CAP mechanism is not another top-k patch selector or global pooling head.
For each image it:

1. relates every spatial feature to every other spatial feature;
2. constructs 27 integral regions of different size and aspect ratio;
3. makes each region attend to every region in the same image;
4. encodes the shared spatial order with an LSTM;
5. pools hidden responses with residual-less NetVLAD.

This differs from locally closed routes:

- FFVT, TransFG, Cropr, IELT, WILDCAT, patch MIL, and Attention-MaxSep select
  or aggregate individual patches but do not apply the CAP integral-region
  equation and ordered within-sample context;
- DOLG subtracts a global direction from local features but has no
  region-conditioned sequence;
- DeepBDC and compact covariance pool global second-order statistics;
- Hamburger NMF and sparse class evidence decompose or select a token map;
- PMG changes jigsaw granularity;
- persistent Conformer-style coupling exchanges CNN and token states across
  Transformer layers rather than testing an ordered integral-region head.

PEDTrans is rejected before code because its local enhancement plus
similarity dropout overlaps closed local mixers, attention/token dropping,
RSC, and patch pruning. ViT-CoMer is rejected because TRKH already ran an
eight-layer bidirectional CNN-token coupling gate. ViTree is not selected
because the linked official repository is empty and has no usable licensed
implementation at the pinned revision.

## Immutable Inputs

Only the following scientific inputs are allowed:

- block-2 feature cache:
  `runs/audit_attention_maxsep_prototype_a0_20260721/cohort_block2_features.npy`;
- geometry/probability cache:
  `runs/audit_attention_maxsep_prototype_a0_20260721/cohort_geometry.npz`;
- full train-only CIDT predictions:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`;
- current keeper checkpoint and resolved config for provenance only;
- current-best command/history files for byte-identity protection only;
- pinned CAP paper/source files;
- this protocol and its machine lock.

Locked hashes:

- feature cache:
  `e4a833b20b29af09adb0d88a6eddb0768b1ce85a1e13bfd3cfb6215b78d8de2a`;
- geometry cache:
  `69f581e1cca66bc64136352ea7f11668e1e50cf4715b2f9ba8c49e8e6f66dc72`;
- CIDT predictions:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`;
- keeper checkpoint:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`;
- keeper resolved config:
  `e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674`;
- current-best commands:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`;
- current-best history:
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

The feature cache is `[763,256,16,16]` `float16`. Geometry contains exact
sample indices, targets, five CIDT folds, source stems, keeper probabilities,
valid masks, bbox masks, and model/crop boxes. The locked cohort is:

- class-1 true positives: 528;
- class-1 false negatives: 13;
- restricted false positives: 222;
- `0 -> 1`: 158;
- `2 -> 1`: 54;
- `4 -> 1`: 10.

The full clean CIDT table contains 9,215 train rows. Its keeper metrics are
macro/class-1 F1 `0.9398761497/0.8154440154`, class-1 precision/recall
`0.7002652520/0.9759704251`. These are pre-existing baseline facts, not CAP
candidate observations.

Validation/test images, labels, paths, predictions, features, statistics, or
checkpoints are forbidden. No raw dataset file may be modified.

## Source-Held Split And Calibration

Reuse the five immutable CIDT source-disjoint folds byte-for-byte. For outer
fold `f`:

- held fold: `f`;
- calibration fold: `(f + 1) mod 5`;
- fit folds: the other three folds.

No source stem may occur in two partitions. Fit rows update parameters.
Calibration rows select one suppression threshold only. Held rows are scored
once at epoch 30 and never affect optimization, threshold, model choice,
region order, seed, or role selection.

Each role obtains its own calibration threshold. A row is suppressed only
when the keeper predicts class 1 and its role score is below that threshold.
The replacement is the keeper's highest-probability non-class-1 class.
No A0 role rescues a class-1 false negative; FN scores are a mandatory
readiness diagnostic, preventing apparent precision gains from hiding the
same true-positive suppression failure seen in earlier methods.

The threshold is the largest deterministic midpoint that rejects no more than
`floor(0.03 * N)` calibration class-1 true positives. Scores equal to the
threshold are retained. If ties make that impossible, use the next lower
`float64` value. No target from a held fold enters calibration.

## CAP Equations

Let frozen block-2 features be `x_raw` with shape `[B,256,16,16]`.

### Valid-Support Canonicalization

The cached valid mask is used only to remove model padding. It is not a fruit
bbox or an object label. Every locked mask must be one non-empty rectangle.
For each sample:

1. compute the exact rectangular support from the valid mask;
2. crop `x_raw` to that support;
3. bilinearly resize the crop to `16x16` with `align_corners=False`;
4. call the resulting tensor `x`.

No padded feature enters CAP. This prevents aspect-dependent letterbox padding
from becoming a shortcut while retaining all source context inside the real
image support. The global-GAP control uses the mathematically corresponding
valid-mask average of `x_raw`.

### Pixel Context

Following the paper's non-local self-attention:

`q = W_q x`, `k = W_k x`, `v = W_v x`

`A = softmax(q^T k)` over source positions

`o = x + gamma * v A`

where `q/k` have 32 channels, `v` has 256 channels, and scalar `gamma` is
initialized to zero. The implementation must verify its orientation against
an independent NumPy oracle.

### Integral Regions

Upsample `o` from `16x16` to `42x42` using bilinear interpolation with
`align_corners=False`. Divide each axis into three 14-cell intervals. Enumerate
the exact official `grid_size=3`, `min_grid_size=2` order:

- every valid rectangle aligned to the four grid boundaries;
- retain a rectangle when width or height spans at least two intervals;
- exclude the full image;
- append the full image last.

This yields exactly 27 regions. Resize every region to `7x7` by bilinear
interpolation and flatten it to `f_bar_r` in `R^(7*7*256)`.

### Region-Conditioned Context

For every ordered region pair:

`beta_(r,r') = tanh(W_beta f_bar_r + W_beta_prime f_bar_r' + b_beta)`

`alpha_(r,r') = softmax_r'(W_alpha beta_(r,r') + b_alpha)`

`c_r = sum_r' alpha_(r,r') * f_bar_r'`

The query/key hidden dimension is 32. Global-average-pool each `c_r` to a
256-dimensional region vector. In particular, there is no extra sigmoid before
this softmax; the A0 follows Equation 2 in the accepted paper.

### Spatial Encoding And Learnable Pooling

Feed the 27 vectors in the fixed region order to a one-layer LSTM with hidden
size 128. For hidden state `h_r`, residual-less NetVLAD uses 32 clusters:

`gamma_k(h_r) = softmax_k(W_k^T h_r + b_k)`

`N_v(:,k) = sum_r gamma_k(h_r) * h_r`

L2-normalize the cluster response and flattened descriptor exactly as the
official source. A final linear layer maps it to one binary logit for
`target == 1`. No keeper probability, target identity, bbox mask, source ID,
fold ID, image path, or handcrafted color statistic enters the candidate.

## Roles And Causal Controls

Exactly seven roles are scored:

1. `keeper_margin_control`: fixed `log(p1)-log(max rival probability)`;
2. `global_gap_linear_control`: trainable linear binary head on valid-mask
   global-average block-2 features;
3. `integral_self_only_control`: the candidate architecture with identity
   region context, retaining pixel context, region geometry, LSTM, and
   residual-less NetVLAD;
4. `cap_context_candidate`: the exact candidate above;
5. `cap_context_seed_repeat`: the same equation with seed offset `+100000`;
6. `cap_spatial_deranged_control`: same architecture and parameter budget,
   but each sample receives a deterministic sample-index-seeded Sattolo
   single-cycle permutation of the 27 regions before context/LSTM. It has
   exactly zero fixed positions;
7. `cap_cross_sample_context_control`: own region queries attend to keys from
   a deterministic label-blind, different-source row mapping within the
   current fit, calibration, or held partition. Values remain the sample's own
   regions, preserving its region multiset and final label while corrupting
   the within-image context relation.

Derangements are fixed by immutable sample index before training, shared
across epochs, source-disjoint, partition-preserving, and never selected from
outcomes. A fit row can never draw cross-sample context from calibration or
held rows. The machine lock stores the complete spatial-permutation hash and
the partition-specific cross-sample mapping hashes. Candidate success must
disappear when shared spatial order or within-sample context is destroyed.

## Optimization Lock

All trainable roles use:

- implementation: PyTorch, independent from official Keras runtime;
- device: CUDA;
- arithmetic: float32;
- deterministic algorithms: enabled;
- TF32: disabled;
- primary seed: `20260724`;
- epochs: 30;
- batch size: 32;
- drop-last: false;
- loss: unweighted `BCEWithLogitsLoss`;
- optimizer: AdamW;
- learning rate: `3e-4`;
- betas: `(0.9,0.999)`;
- epsilon: `1e-8`;
- weight decay: `1e-4`;
- gradient-norm clip: `1.0`;
- epochs 0-2: linear warmup to `3e-4`;
- epochs 3-29: cosine decay to zero;
- no early stopping, checkpoint selection, class weighting, oversampling,
  resampling, augmentation, synthetic data, or hyperparameter sweep;
- score epoch 30 only.

Fit rows are sorted by sample index, then permuted by one locked NumPy
`default_rng` stream per fold/role seed. Candidate and causal controls use
identical fit orders for the primary seed.

## Performance Gates

All gates are conjunctive.

Threshold-free evidence:

- candidate true-class-1 versus restricted-FP AUROC `>= 0.82`;
- AUROC gain over keeper margin `>= 0.02`;
- AUROC gain over the strongest non-deranged trainable control `>= 0.01`;
- AUROC gain over each deranged control `>= 0.015`;
- class-1-FN versus restricted-FP AUROC `>= 0.65`;
- candidate FN-versus-FP AUROC gain over keeper margin `>= 0.03`;
- candidate wins the two non-deranged controls in at least four of five folds.

Calibrated action:

- class-1 TP retention `>= 0.97` overall and `>= 0.94` in every fold;
- restricted-FP rejection `>= 0.25` overall and `>= 0.10` in every fold;
- positive FP rejection in every fold;
- correct suppressions at least twice TP harms;
- net corrections minus harms `>= 25`;
- full 9,215-row macro-F1 gain `>= 0.002`;
- full class-1 precision gain `>= 0.03`;
- full class-1 F1 gain `>= 0.01`;
- full class-1 recall drop `<= 0.03`;
- every nonfocus-class F1 drop `<= 0.003`.

Repeatability:

- seed-repeat AUROC absolute difference `<= 0.01`;
- candidate/repeat suppression agreement `>= 0.95`;
- candidate/repeat pass/fail decision identical.

No threshold, seed, fold, epoch, learning rate, region geometry, hidden size,
cluster count, control, or gate may change after candidate metrics exist.

## Structural, Resource, Replay, And XAI Gates

Before formal training:

- independent NumPy/PyTorch oracles must cover pixel attention, 27-region
  enumeration, valid-support canonicalization, bilinear region pooling,
  region attention, Sattolo permutations, cross-sample query/key/value roles,
  LSTM input order, residual-less NetVLAD, threshold ties, and action
  semantics;
- default and every causal role must have finite forward/backward gradients;
- candidate and same-budget controls must expose expected trainable tensors;
- cached features, geometry, keeper probabilities, CIDT rows, folds, source
  stems, and hashes must match the lock exactly;
- a synthetic BF16/FP32 resource check must keep peak CUDA allocation below
  6.0 GiB and process RSS below 10.0 GiB;
- no unknown Python/TensorRT process or unisolated GPU workload may overlap
  formal timing.

The formal process installs a process-wide dynamic file-open ledger before
opening scientific inputs. It allowlists exact locked files and the formal
output root, blocks dataset writes, and blocks any complete
`val/valid/validation/test` path component before open. It records an
installation probe, normalized path/mode/count events, blocked attempts, and
an ordered digest.

After probabilities/actions are frozen, build a fixed 20-row train-only sheet:

`[163,2370,856,4613,58,2563,1939,2698,341,2004,1066,3934,`
`2,2906,345,4040,195,2411,791,3624]`.

For each outer fold this contains one TP1, one FN1, one `0->1` FP, and one
`2/4->1` FP. The sheet shows input+bbox, aggregate CAP region attention,
feature-gradient attribution, spatial-derangement contrast, score, threshold,
and action. Require:

- finite, non-constant maps;
- mean attribution mass inside valid support `>= 0.95`;
- mean attribution mass inside bbox `>= 0.75`;
- candidate context differs from both derangements;
- all 20 rows manually reviewed with no padding/background shortcut;
- local response corresponds to visible surface, lesion, ripeness, contour,
  or stem evidence rather than a sample-index artifact.

A fresh process must reproduce every probability, threshold, action, metric,
fold summary, state hash, ordered ledger digest, and visual numeric array.
Maximum numeric replay error is `1e-7`; discrete artifacts must be exact.

## Decision And Downstream Authorization

Only a complete conjunction may authorize:

1. one default-off CAP side head in production code;
2. one separately prospectively locked matched validation smoke;
3. full XAI/audit review after that smoke.

Any failure means:

- reject CAP A0;
- do not sweep regions, layer, hidden size, clusters, normalization, optimizer,
  learning rate, threshold retention, seed, or nearby pooling variants;
- do not open validation/test, trainer integration, smoke, probe, full train,
  export, video test, or current-best command/history update;
- preserve compact evidence, write closure/journal/TODO/Word/skill updates,
  run retention, verify tests, commit, and push.
