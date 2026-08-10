# TRKH 5-Class Hyperspherical Support A0 Protocol - 2026-07-21

## Status And Authorization

Prospectively locked before auditor implementation and before inspection of any
new candidate metric. This protocol authorizes one source-disjoint, train-only
information audit over an immutable frozen-keeper embedding cache. It does not
authorize validation/test access, image-model training, trainer/model edits, a
checkpoint, a threshold or hyperparameter sweep, a smoke/probe/full train, or
current-best command promotion.

The audit tests only whether class-conditional hyperspherical support
generation contains a precision-selective signal that is stronger than natural
sampling and exact-count duplicate oversampling. It is not a reproduction of
the complete FeatRecon training recipe.

## Primary Source And Reproducibility Boundary

- Accepted primary paper: Yi et al., "Geometry of Long-Tailed Representation
  Learning: Rebalancing Features for Skewed Distributions," ICLR 2025:
  `https://proceedings.iclr.cc/paper_files/paper/2025/hash/adb2075b6dd31cb18dfa727240d2887e-Abstract-Conference.html`.
- Official paper PDF:
  `https://proceedings.iclr.cc/paper_files/paper/2025/file/adb2075b6dd31cb18dfa727240d2887e-Paper-Conference.pdf`.
- OpenReview record: `https://openreview.net/forum?id=GySIAKEwtZ`; the official
  record identifies the accepted ICLR-2025 poster and licenses the paper under
  CC BY 4.0.
- Equations transferred independently: confidence-support estimation (Eq. 7),
  separable cap bound (Eq. 10), and truncated-angle/tangent generation
  (Eq. 14-16).
- The paper reports `alpha=0.99`, 128/1024-dimensional projection heads,
  supervised contrastive plus logit-compensated CE, and 90-400 epoch recipes.
  Those heads, losses, durations, benchmark results, and pretrained-backbone
  assumptions are not transferred to this <=30-epoch scratch TRKH project.
- Searches of the paper, authors' pages, OpenReview, GitHub, and the curated
  Awesome-LongTailed-Learning index found no official FeatRecon code release.
  No third-party implementation is copied. The paper does not report concrete
  values for `q`, `gamma`, or `m`; this A0 therefore fixes `gamma=1` implicitly
  by omitting head-statistic regularization and tests only the fully specified
  empirical-cap generation mechanism. It must never be described as complete
  FeatRecon.

## Screening And No-Repeat Boundary

- PLTR-SD is rejected before code: its balanced supervised contrastive and
  smooth-max multi-loss mechanism overlaps failed teacher-guided SupCon,
  boundary/multi-granularity contrastive, and multi-objective routes; its
  pathological imbalance setting is not comparable to TRKH's ratio `4.66`.
- GKP-GSA is rejected before code: grouped preservation plus sharpness-aware
  optimization overlaps failed SAM, A-GEM/GEM, V-REx, CAGrad, and GroupDRO
  families, while the current public source has no usable license lock and a
  200-epoch recipe.
- SEL, CurrMix, and IBC are rejected before code because their tail-region
  expansion, MixUp, MoCo/prototype, or multi-expert mechanisms overlap closed
  MixUp/CutMix, foreground counterexample, SSL, prototype, and ensemble routes.
- Natural/deferred reweighting, strict global class-1 oversampling, prior/logit
  correction, balanced classifiers, prototype/kNN, feature synthesis, capsule
  support constraints, supervised contrastive variants, and post-hoc routers
  are already closed. This A0 may not tune or combine those routes.
- The equation-distinct question is narrow: does synthetic angular diversity
  inside a fitted class cap outperform exact duplicate support at the same class
  counts without expanding class-1 false positives?
- Raw images, labels, YAML files, split declarations, and retained evidence are
  immutable. No generated value may be written below either dataset tree.

## Locked Inputs

- Pre-protocol HEAD/upstream:
  `48cfc1d29f67a0a99f8007cf01249ab0d081547c`.
- Frozen keeper train embedding cache:
  `runs/diagnostic_reslt_embedding_cache_keeper_yolof_20260712/train_embeddings.npz`,
  SHA-256
  `157805b449549c9ad87f56f11ece2ed756669bcf11332c860c99feca92314464`.
- Embedding-cache manifest SHA-256:
  `24c617e96f6863b8434edf03f8d611a4ea2da27cb33c30df9550ac2603a46dd2`.
- Fixed CIDT train-only source-fold CSV and summary SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`
  and
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.
- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Raw `yolo_f/data.yaml` SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Current-best command/history SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`
  and
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

The auditor must verify all locks before opening the cache. The cache path and
every embedded path must be train-only and must reject the markers `val`,
`valid`, `validation`, and `test`. Validation/test NPZs, CSVs, labels,
predictions, pixels, and metrics are forbidden.

## Locked Cohort And Folds

- All `9,215` ordered `yolo_f/train` object rows and 256-dimensional frozen
  keeper embeddings.
- Exact class counts: `[1941, 541, 1920, 2520, 2293]`.
- Exact sample indices: ordered integers `0..9214` with no duplicate.
- Reuse only the `condition=clean` CIDT fold assignment from the locked CSV.
  It must contain one row per sample, preserve labels and source stems, and
  have fold counts `[1843, 1830, 1828, 1851, 1863]`.
- Every fit/holdout pair must contain all five classes and have zero source-stem
  overlap. Embeddings are from a keeper trained on the complete train split;
  therefore results are source-held readout evidence, not fully OOF encoder
  evidence and not a generalization estimate.

## Fixed Geometry

For each held fold, L2-normalize the real fit and holdout embeddings. No PCA,
scaler, feature selection, projection, covariance model, temperature, or
keeper-probability concatenation is allowed.

For class `k` in fit rows, let:

```text
mu_k       = normalize(mean_i z_i)
theta_i    = acos(clip(z_i dot mu_k, -1, 1))
theta99_k  = quantile(theta_i, 0.99, method="linear")
simplex_r  = 0.5 * acos(-1 / (K - 1))
cap_r_k    = min(theta99_k, simplex_r), K=5
```

Fit one normal angle distribution with the population mean and standard
deviation of `theta_i`. Draw angles exactly from its inverse-CDF truncated to
`[0, cap_r_k]`; use float64 SciPy `ndtr/ndtri`, clipped uniform quantiles
`[1e-12, 1-1e-12]`, and NumPy PCG64. If the standard deviation is below
`1e-12`, use the clipped mean exactly.

For every angle, independently draw `s ~ N(0, I_256)`, project it into the
tangent plane, normalize it, and generate:

```text
nu = normalize(s - (s dot mu_k) * mu_k)
x  = normalize(cos(theta) * mu_k + sin(theta) * nu)
```

Generate `n_max - n_k` vectors for every class so each fit class contains
exactly `n_max` real-plus-synthetic rows. Seed is
`20260721 + 1009*fold + 131*class`. A seed-repeat role uses offset `+7919`.
Global Python/NumPy RNG state may not be consumed.

All generated norms must differ from one by at most `1e-10`; generated angles
must lie in `[0, cap_r_k]` within `1e-10`. Persist per-fold centers, empirical
angle statistics, radii, generated counts, pairwise center angles, cap overlap
margins, array hashes, and seed records.

## Matched Readout Roles

All roles fit multinomial scikit-learn logistic regression directly on the
normalized embeddings with `solver=lbfgs`, `C=0.3`, `max_iter=2000`,
`tol=1e-9`, no class weights, and seed `20260721`. No retry, coefficient
selection, class threshold, calibration, probability blend, or fallback is
allowed.

1. `natural_control`: fit only the natural-frequency real rows.
2. `duplicate_control`: balance every class to `n_max` by a deterministic
   seeded permutation of its real rows followed by cyclic repetition. This has
   exactly the candidate's class counts and no synthetic diversity.
3. `cap_candidate`: fit real rows plus the primary hyperspherical samples.
4. `cap_seed_repeat`: use the independently generated repeat samples.
5. `nearest_center_deranged`: generate the same counts/angles but around each
   class's most cosine-similar rival fit center while retaining the original
   labels. This is an information placebo and cannot become a model candidate.

The immutable keeper probabilities are reported only as an in-sample reference
and never enter a readout or a gate comparison that could promote training.

## Required Evidence

Persist one prediction row per sample with sample/source/fold/target, every
role's five probabilities and prediction, class-1 probability deltas, and all
candidate-versus-control transition flags. Persist fold protocols and
synthetic-geometry summaries separately.

Report for every role:

- accuracy, macro F1, per-class precision/recall/F1/support/predicted support,
  confusion matrix, NLL, Brier score, and 15-bin ECE;
- class-1 TP/FN and restricted `{0,2,4} -> 1` FP totals;
- candidate corrections/harms, class-1 FN rescue/TP break, and restricted-FP
  removal/creation versus natural and duplicate controls;
- the same metrics and transitions in every held source fold;
- primary/repeat prediction agreement and maximum probability difference;
- all geometry, fold, coefficient, input, artifact, runtime, and peak-RSS
  hashes/records.

The CSV must be independently replayed into a canonical analysis. Every
non-manifest payload must be hash-locked, and a second process must reproduce
the summary with maximum numeric difference at most `1e-12`.

## Conjunctive A0 Gates

Structural gates:

- every locked hash, row count/order, class count, fold count, source, class
  order, finite-probability, and train-only path check passes;
- source overlap is zero in all five folds and every role covers every row once;
- every logistic readout converges without retry;
- generated count balance, norm, angle, tangent, seed-isolation, and array-hash
  checks pass in all folds;
- compile, pyflakes, focused tests, PowerShell parse, exact replay,
  `git diff --check`, and artifact-manifest verification pass;
- validation/test/model forward/trainer/checkpoint/current-best files remain
  unopened and byte-identical.

Mechanism gates, all required:

- cap-candidate macro F1 is at least `duplicate + 0.002` and no more than
  `0.002` below natural control;
- class-1 F1 is at least `0.70` and at least `duplicate + 0.010`;
- class-1 precision is at least `duplicate + 0.015` and no more than `0.005`
  below natural control;
- class-1 recall is at least `0.75` and no more than `0.015` below duplicate;
- versus duplicate, restricted-FP net removal is at least `10`, class-1 TP net
  loss is at most `2`, corrections are not fewer than harms, and TP breaks do
  not exceed FN rescues;
- no class in `{0,2,3,4}` loses more than `0.010` F1 versus duplicate;
- at least four folds have non-worse class-1 precision and at least four have
  positive restricted-FP net removal versus duplicate; no fold loses more than
  one class-1 TP net;
- candidate class-1 F1 exceeds nearest-center derangement by at least `0.020`;
- primary/repeat prediction agreement is at least `0.99`, macro/class-1 F1
  differences are each at most `0.005`, and both seeds independently satisfy
  all aggregate precision/recall/FP safety thresholds.

No aggregate score can compensate for a failed gate. Precision obtained by
broad class-1 contraction is a failure. Cap-overlap margins are mandatory
diagnostics but are not a hard gate because a later train-time representation
objective is specifically intended to move class centers.

## Escalation And Stop Rules

If any A0 gate fails, close empirical-cap feature generation, nearby angle
distribution/radius/seed/readout sweeps, and a FeatRecon-like trainer fork on
the current keeper. Do not open validation/test, generate image XAI, edit the
model/trainer, train a smoke/probe/full run, or update current-best commands.

Only a complete A0 pass authorizes a separately locked, default-off
projection-head implementation that uses synthetic support only inside a
low-weight train-time representation objective. It must preserve default
bit-equivalence, strict checkpoint loading, export behavior, architecture
trace, class-specific XAI, and resource telemetry. It then receives one
matched short smoke and full metrics/robustness/XAI audit. A passing smoke may
authorize one short probe. Only a prospectively passing probe may authorize an
autonomous full train under 30 epochs, patience 3, and a fresh workers `2/4`
throughput/GPU-duty benchmark.

Current-best checkpoint, full-pipeline commands, and command-update history
remain unchanged unless that complete validation promotion sequence wins.
