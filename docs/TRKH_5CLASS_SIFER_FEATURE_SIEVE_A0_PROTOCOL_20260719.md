# TRKH 5-Class SIFER Feature-Sieve A0 Protocol

Date locked: 2026-07-19
Status: locked before candidate inference or adaptation
Scope: `yolo_f/train` only; validation and test are forbidden in A0

## 1. Question

Can the alternating identify-and-forget mechanism from SIFER remove an
easy-to-decode early feature that drives `0/2/4 -> 1` errors, while forcing the
unchanged upper CNN-Transformer classifier to retain true class-1 support from
more complex surface evidence?

This is a training-only intervention. The auxiliary network is discarded after
training, so the deployed TRKH graph, parameter schema, and forward equation
remain unchanged.

## 2. Primary-source lock

- Paper: Tiwari and Shenoy, *Overcoming Simplicity Bias in Deep Networks using
  a Feature Sieve*, ICML 2023, PMLR 202.
- Paper URL:
  `https://proceedings.mlr.press/v202/tiwari23a/tiwari23a.pdf`
- Paper SHA-256:
  `49a64b977dcbabfed14473639a211e1385ea71da6ffb75cf19c8065b46ec7959`.
- Official source:
  `https://github.com/google-research/google-research/tree/76b0612b5b48acfa53eba6b433e98764805286f4/sifer`.
- Original SIFER release commit:
  `76b0612b5b48acfa53eba6b433e98764805286f4`.
- SIFER tree object:
  `890e37473fb42a9272e6b5478c23e70e6d3c0a57`.
- `sifer/learning/algorithms.py` blob:
  `43595d17c2ddbca3d50f95bde0189228b92da1aa`.
- `sifer/models/networks.py` blob:
  `cf59f5414e1168edd7c45214545eb91c6c52e3cf`.
- `sifer/params.py` blob:
  `5d3aefd418843cb6ef4d61905e26933427170a96`.
- Repository license blob:
  `d645695673349e3947e8e5ae42332d0ac3164cd7` (Apache-2.0).

The paper and code use an auxiliary classifier at an intermediate feature.
Each iteration first trains the auxiliary classifier on the real label. At a
fixed interval, auxiliary parameters are not stepped while the lower main
network is updated toward a uniform auxiliary posterior. The ordinary main
classification update then runs. The official default interval is five
iterations, with a two-block auxiliary network of width 256.

The paper's real-image experiments use ImageNet-pretrained ResNet-18 and tune
configuration using validation performance. That evidence is not efficacy
evidence for scratch TRKH. A0 therefore tests one fixed keeper adaptation on a
source-disjoint train holdout and forbids validation access.

## 3. No-repeat boundary

SIFER is not reopened RSC, adversarial input augmentation, loss reweighting,
gradient surgery, feature masking, branch diversity, or a post-hoc router:

- RSC masks high-gradient channels only in the current forward pass; SIFER
  changes lower-network weights so an independently trained auxiliary decoder
  becomes uncertain.
- A-GEM, GEM, V-REx, CAGrad, and worst-shift routes constrain or combine task
  gradients; SIFER alternates a supervised decoder update and a uniform-target
  lower-network update.
- RSC, attention drop, token pruning, and dropout remove activations; SIFER
  leaves the deployed graph unchanged and attempts to recode its features.
- The auxiliary head is never used as a teacher, confidence score, router, or
  inference ensemble.

Do not reinterpret a precision increase caused by reduced class-1 support as
evidence that the sieve learned a better feature.

## 4. Exact TRKH adaptation

Let `M_d(x)` be the output of the unchanged three-block `HybridConvStem`; for
the locked `256x256` input it has shape `B x 256 x 32 x 32`. Let `A` be the
training-only auxiliary network and `M` the complete unchanged TRKH model.

The auxiliary network is the official two-block topology:

1. BasicBlock 1 has `3x3 256->256`, BatchNorm, ReLU, `3x3 256->256`,
   BatchNorm, no residual addition, then ReLU.
2. BasicBlock 2 has the same two convolutions and BatchNorm layers, with an
   identity residual addition, then ReLU.
3. Adaptive average pooling and a `Linear(256,5)` layer produce auxiliary
   logits.
4. The exact trainable auxiliary parameter count must be `2,362,629`.

For zero-based logical step `k` and one immutable image batch `(x,y)`:

1. **Identify:** compute `z = stop_gradient(M_d(x))`, then update only `A`
   from `CE(A(z), y)`.
2. **Forget:** when `k mod 5 == 0`, hold the auxiliary optimizer fixed and
   update only the three stem blocks from
   `CE(A(M_d(x)), [1/5,1/5,1/5,1/5,1/5])`.
3. **Main:** update every existing TRKH parameter from ordinary
   `CE(M(x), y)`.

The official source keeps the auxiliary module in train mode during forgetting;
A0 preserves that lifecycle, including BatchNorm running-stat updates, while
only the stem optimizer steps. Auxiliary and non-stem tensors must remain
unchanged across the forget optimizer step except auxiliary BatchNorm buffers.

No class mask, OOF pseudo-label, sample weight, teacher, extra augmentation,
threshold, or validation-derived choice is allowed. Class-1 protection is
enforced by the prospective behavior gates, not by hiding its examples from
the equation.

## 5. Data and optimization lock

- Dataset: only `D:\DataAI\AIEx\newdataset\yolo_f\train`.
- Dataset YAML SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher-args SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- Immutable CIDT summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.
- Immutable CIDT predictions SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Source-disjoint fold: fold 0; fit/holdout rows `7372/1843`, holdout class
  counts `[380,109,393,503,458]`.
- Fit order: NumPy PCG64 seed `42`, first `60 x 32 = 1920` fit rows.
- Transform: deterministic keeper evaluation transform, with no augmentation.
- Variants: keeper-initialized matched control and one SIFER candidate.
- Main optimizer for both variants: AdamW, LR `1e-5`, weight decay `0.05`,
  betas `(0.9,0.999)`.
- Auxiliary optimizer: SGD, LR `1e-2`, weight decay `1e-4`, no momentum.
- Forget optimizer: SGD over stem parameters only, LR `1e-4`, no momentum or
  weight decay. This preserves the official 10:1 forget/main LR ratio while
  retaining the established keeper-adaptation main LR.
- Auxiliary initialization seed: `42042`, isolated from all CPU/CUDA
  main-model RNG and verified by before/after RNG-state hashes.
- Main seed: `42`; deterministic algorithms enabled; TF32 disabled.
- No scheduler, EMA, model loss stack, sample balancing, or augmentation.

The candidate and control receive the same materialized sample order. Before
each main forward, both restore the same role-specific CPU/CUDA RNG state so
auxiliary forwards cannot shift dropout or stochastic-depth masks.

No auxiliary position/depth/width, interval, optimizer, LR, fold, seed, batch,
budget, target, transform, or loss sweep is allowed.

## 6. Structural and equation gates

All checks must pass:

- checkpoint/data/CIDT/paper/protocol hashes and official Git commit/tree/blob
  objects are exact;
- no validation or test dataset, pixel, prediction, metric, or path is loaded;
- checkpoint stem is exactly the vanilla three-block `HybridConvStem` with
  output channels 256 and downsample factor 8;
- control and candidate main models start byte-identical with identical schemas;
- candidate inference parameter count and state schema match control because
  `A` is external and discarded;
- official auxiliary topology and parameter count are exact;
- direct reference CE and uniform-target equations match the implementation;
- identify updates auxiliary parameters and not the detached stem;
- every forget step updates stem parameters, does not update non-stem
  parameters, and does not step auxiliary parameters;
- exactly 12 forget steps occur at `0,5,...,55`;
- main-forward RNG summaries match between variants at all 60 steps;
- all losses, logits, final parameters/buffers, and gradients are finite;
- every main parameter family receives a nonzero finite gradient and moves;
- candidate peak allocated VRAM is at most `6.50 GiB`;
- median candidate step time excluding five warmup steps is at most `1.60x`
  matched control after local GPU isolation is confirmed;
- candidate main model exports through the unchanged static batch-1 ONNX graph
  with maximum error `<=1e-5` and contains no auxiliary node or parameter.

If unrelated GUI GPU utilization prevents isolated timing, the formal must not
run. Timing is not converted to a warning after results are visible.
Candidate timing includes the algorithmic detached-stem identify forward,
auxiliary update, optional forget update, and ordinary main update. Audit-only
before/after diagnostics, state hashing, evaluation, and rendering are excluded.

## 7. Mechanism gates

Across the 12 forget steps:

- auxiliary logits, losses, entropy, and measured accuracy are finite;
- identify loss decreases on at least 8/12 measured forget-step batches;
- immediate uniform CE does not increase after the stem forget update on at
  least 8/12 steps;
- auxiliary posterior entropy does not decrease after forgetting on at least
  8/12 steps;
- mean absolute candidate-control stem-feature difference on holdout is
  `>=1e-4`;
- mean absolute candidate-control clean class probability difference is
  `>=1e-4`.

These establish that the candidate actually identifies and suppresses an early
feature. They do not substitute for precision and TP-safety gates.

## 8. Clean train-only behavior gates

On all 1843 fold-0 holdout rows, candidate versus matched control must satisfy
every condition:

- macro F1 delta `>= 0.000`;
- class-1 F1 delta `>= +0.005`;
- class-1 precision delta `>= +0.005`;
- class-1 recall delta `>= -0.005`;
- restricted `0/2/4 -> 1` false-positive reduction `>= 2`;
- class-1 FN rescues are at least class-1 TP breaks;
- total corrections are greater than harms;
- maximum non-class-1 per-class F1 drop `<= 0.010`;
- candidate predicted class-1 support is at least `95%` of control support.

## 9. Illumination and visual gates

Evaluate the identical holdout under fixed dim, bright, and low-contrast
transforms. Candidate versus control must satisfy:

- every condition has class-1 F1 delta `>= -0.010`;
- every condition has class-1 recall delta `>= -0.015`;
- at least two conditions have nonnegative class-1 precision delta;
- aggregate FN rescues are at least TP breaks;
- no condition creates more restricted class-1 FP than it removes.

Render four locked contact sheets from clean changed cases and the three
lighting conditions. Each sheet must prioritize restricted-FP removals,
class-1 TP breaks, class-1 FN rescues, and newly created restricted FP, and must
show the image, target, control/candidate probabilities, transition, and
auxiliary confidence/entropy. Manual review must find a repeatable fruit-surface
or maturity mechanism rather than generic color suppression, border response,
background removal, or global class-1 contraction.
The selector reserves one row for every available safety category above before
filling the remaining eight-row budget by category priority and class-1
probability shift, so abundant FP removals cannot hide a TP break.

## 10. Decision

A0 passes only if structural, equation, mechanism, clean, illumination,
resource, export, independent replay, and manual visual checks all pass.
The replay process must bind every input path to the immutable summary artifact
and exit nonzero whenever any replay check fails.

Passing authorizes exactly one separately locked `120 batches x 2 epochs`
full-validation, no-test smoke after default-off trainer integration. It does
not authorize test, a full 30-epoch train, a nearby SIFER setting, or a current-
best command update.

Any A0 failure closes this exact SIFER mapping and nearby auxiliary position,
depth, width, interval, LR, optimizer, target-uniformity, fold, seed, budget,
and class-masked variants on the current keeper. Current-best command history
changes only after a later candidate wins the locked validation promotion gate.
