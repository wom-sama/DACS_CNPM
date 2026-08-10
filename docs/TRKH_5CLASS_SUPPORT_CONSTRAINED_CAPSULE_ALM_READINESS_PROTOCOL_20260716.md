# TRKH 5-Class Support-Constrained Capsule ALM Readiness Protocol

Date locked: 2026-07-16

Status: immutable train-only Stage A protocol. Validation and test are forbidden
until every Stage A gate passes. This protocol defines one A0 only; it does not
authorize a hyperparameter, seed, fold, threshold, or architecture sweep.

## Question

Can routing-by-agreement over the keeper's final patch tokens create a genuinely
different part-to-class representation that reduces restricted class-1 false
positives without repeating the broad class-support suppression seen in the
natural-CE and Mutual-Channel adapters?

The candidate must start as the exact keeper, retain a natural multiclass CE
path, and apply explicit train-only class-1 support constraints under clean,
dim, bright, and low-contrast conditions. A precision gain obtained by breaking
true class-1 predictions is a rejection.

## Primary Sources

### Dynamic routing

- Sabour, Frosst, and Hinton, *Dynamic Routing Between Capsules*, NeurIPS 2017.
- Paper:
  `D:\DataAI\external_sources\papers\sabour2017_dynamic_routing_capsules.pdf`
- Paper SHA-256:
  `e87bc5186f7767d49d0ea6535183b82b84e3d330d147d41abf27fd98a423069d`
- Author-maintained TensorFlow source:
  `D:\DataAI\external_sources\official\dynamic_routing_capsules`
- Commit: `984fbc754943c849c55a57923f4223099a1ff88c`
- `research/capsules/models/layers/layers.py` SHA-256:
  `b6d94ee33c4a0bb4d29c8520fdd15937ad2a8c59995e79b63bcf95ee898a8b89`
- `research/capsules/models/capsule_model.py` SHA-256:
  `e1a115dbda2855241a9acc450e3f4bf6361721a99fb5007bd8120d4149de7862`
- `research/capsules/README.md` SHA-256:
  `d535b01f266ee3dba661cac5ad677c487b52448ac009900d9c39ce010ecaed07`
- Apache-2.0 `LICENSE` SHA-256:
  `b42e0eeff6d2d55ef63bc57c328b5a219a9fa20ddaecf7b9d48a6786c9e17e6b`

The A0 transfers the paper's 8D lower capsules, 16D class capsules, zero
routing logits, non-leaky softmax coupling, squash equation, dot-product
agreement, and exactly three routing iterations. It does not transfer the
MNIST reconstruction decoder or margin-loss constants.

### Critical-class augmented Lagrangian

- Sangalli et al., *Constrained Optimization to Train Neural Networks on
  Critical and Under-Represented Classes*, NeurIPS 2021.
- Main paper SHA-256:
  `908915f5d5f09ae218ab8c387e6f0b33a9b64a5e707418bb4ffc98fb0a05052c`
- Supplement SHA-256:
  `e78dc837a82e4c57c900068e4ecad7d2ab5deada8ca8eb632dbf26cc8122b6c3`
- Author repository:
  `D:\DataAI\external_sources\official\alm-dnn-github`
- Commit: `ab3da74d0d80dcffd3666d5ee4371f57908279e2`
- `utils/custom_losses.py` SHA-256:
  `d9cb0f1db8c8c1c7e369cc44d9a8c1b5a51ad1a4f84d43878d1d161c45bd3cf8`
- `utils/train_test_CIFAR.py` SHA-256:
  `5a513187e9510cf00e7b461838a207144cd78eae135085a4f65cf3f7f9323f93`
- `README.md` SHA-256:
  `1bde578e23dfaebae21c406a668f1063ee56c5c17a621e863557cb1fdc185652`

The repository does not provide a license file, so no repository code is
copied. The implementation follows paper Equations 5 and 6 and is covered by
an independent flat-tensor equation replay. The current GitHub helper returns
only the last positive term in its loop; this protocol instead accumulates all
positive constraints as required by Equation 5. This discrepancy must be
reported, not silently reproduced.

The paper sums each positive's hinge over negatives. A0 divides that sum by the
number of negatives. This leaves the feasible set `q_j = 0` unchanged and
removes batch-size dependence from the dual scale. This normalization is an
explicit TRKH adaptation, not a claim about the authors' implementation.

## Locked Inputs

- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Keeper launcher-args SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`
- Data YAML: `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`
- Data YAML SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`
- CIDT summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`
- CIDT prediction SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`
- MCL summary SHA-256:
  `cf4a0143afde61ce2a737fca4a859397d19d058ad3ca01c85d624a34e7183b0f`
- MCL prediction SHA-256:
  `1feff96cedf182b003163f2c22b14b075481de892ca3acdab878e728c925a1d5`
- Current-command SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`
- Command-history SHA-256:
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`

No path containing a validation or test split may be opened during Stage A.
The raw datasets are read-only.

## Locked Cohorts

- Source-grouped CIDT fold: `0`.
- Natural multiclass fit: `7372` rows and `6452` source groups.
- Holdout: `1843` rows and `1612` source groups.
- Source overlap: exactly zero.
- Fit-index SHA-256:
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d`
- Holdout-index SHA-256:
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`
- Class-1 references: all `432` class-1 rows in fit folds 1-4.
- Reference-index SHA-256:
  `1474f2d8e8fe83767e3c250fd89e827f1544fb058c762bc9c114b150a18d2574`
- Restricted hard negatives: all `186` fit rows with target in `0,2,4` and
  clean keeper prediction `1`; class counts are `135/42/9`.
- Hard-index SHA-256:
  `56dfdbc6180d99b80605799bcd8e2ef412f0abd5c0191054b4fe7d6554d7f020`

Natural CE uses every fit row at its natural frequency. Boundary constraints
use only the fixed class-1 and restricted-hard cohorts. There is no balancing,
relabeling, threshold fitting, test tuning, or raw-data modification.

## Locked Conditions

Condition order is fixed:

1. `clean`
2. `lighting_dim`: brightness `0.70`, contrast `0.90`
3. `lighting_bright`: brightness `1.25`, contrast `1.10`
4. `low_contrast`: brightness `1.00`, contrast `0.65`

Natural CE is computed on clean fit rows. Both boundary constraints are
computed separately on every condition, always pairing positives and negatives
from the same condition.

## Locked Capsule Residual

- Input: the keeper's final `167 x 256` retained patch tokens and original
  `16 x 16` patch indices.
- Shared primary projection: `256 -> 8`, followed by paper squash.
- Position-specific vote weights: `[256, 8, 5, 16]`.
- Class-capsule bias: `[5,16]`.
- Routing: three iterations, initial logits zero, non-leaky softmax over the
  five parent classes, dot-product agreement update after each iteration.
- Class evidence: Euclidean length of each 16D class capsule.
- Residual head: zero-initialized `Linear(5,5)`.
- Deployed logits: `raw_logits + 0.05 * residual_logits`.
- Exact adapter parameter count: `166006`.
- Seed: `42`, with constructor RNG isolation.

The initial candidate and control logits must be bit-exact to the raw keeper.
The keeper remains frozen. A same-weight one-iteration/uniform-coupling
ablation is diagnostic only and cannot be selected as a candidate.

## Locked Objective

For deployed logits `z`, define `p1(z) = softmax(z)[1]` and the deployed
class-1 decision margin

`m1(z) = z[1] - max(z[0], z[2], z[3], z[4])`.

For each class-1 row `j` and its 16 same-condition hard negatives `k`:

`q_rank_j = mean_k relu(-(p1(z_pos_j) - p1(z_neg_k)) + 0.05)`.

For the same positive row:

`q_support_j = relu(m1(raw_pos_j) - m1(z_pos_j))`.

The candidate loss is

`CE_natural + 0.5*mean(q_rank^2) + mean(lambda_rank*q_rank)
            + 0.5*mean(q_support^2) + mean(lambda_support*q_support)`.

Both penalty coefficients are fixed at `mu=1.0`. After each boundary step,
the detached pre-step constraints update nonnegative per-row/per-condition
multipliers:

`lambda <- lambda + q`.

The CE control has the identical capsule graph, initialization, natural order,
boundary schedule, optimizer, and diagnostics, but both constraint terms are
detached and multiplied by exact zero. Its parameter gradients must be
bit-exact to a pure-CE replay.

## Locked Training Schedule

- Epochs: exactly `30`; this is the local convergence ceiling, not permission
  for a longer run.
- Natural batch size: `32`.
- Boundary batch: `16` class-1 rows and `16` hard negatives.
- Optimizer: SGD, learning rate `0.1`, momentum `0.9`, weight decay `2e-4`.
- No scheduler, warmup, AMP training, early stopping, validation callback, or
  checkpoint selection.
- Natural occurrences: `221160`.
- Natural-order SHA-256:
  `9fefbbac66a1ce22190744092edbfa3a9b44c0c1b7d4accf257b09e2b01ffb5b`
- Boundary events: `3240` (`30 x 4 x 27`).
- Every positive appears exactly once per condition per epoch.
- Negative slots are filled by deterministic repeated permutations without
  label-based selection beyond the locked hard cohort.
- Boundary-schedule SHA-256:
  `54d9a40365e250d10e582258194064e936428b08541d132b6a299afafa7b0788`
- The 108 boundary events in an epoch are evenly interleaved across the 231
  natural batches; no optimizer-only boundary step is added.

Candidate and control start from independent deep copies of one deterministic
prototype. The formal run is not allowed to alter these constants after seeing
any behavior metric.

## Structural Gates

All are mandatory:

- Every locked hash, commit, source cleanliness check, cohort count, source
  disjointness check, order hash, and schedule hash is exact.
- No validation/test path was opened and no output directory is created by
  `--preflight-only`.
- Keeper parameter count/state hash is unchanged before and after both fits.
- Adapter parameter schema/count is exact; control/candidate initialization is
  bit-identical and raw-logit equivalence is exact.
- PyTorch squash, vote, coupling, agreement, and three-iteration routing match
  an independent flat reference within `1e-6`.
- Couplings sum to one over parent classes within `1e-6`.
- Candidate and control gradients are finite; every trainable parameter family
  receives a nonzero gradient during the formal fit.
- Detached-control gradients equal pure CE bit-exactly.
- The candidate rank gradient is nonzero and differs from CE.
- Initial support violation is exactly zero.
- All dual values remain finite and nonnegative; at least one rank multiplier
  becomes positive.
- The 30-epoch curve is finite and both variants consume exactly the locked
  natural and boundary schedules.

Any structural failure rejects Stage A independently of behavior.

## Train-Holdout Behavior Gates

Metrics use all `1843` holdout rows, never a selected subset. Restricted class-1
false positives mean targets `0,2,4` predicted as `1`.

### Clean candidate versus raw keeper

- Macro F1 delta `>= -0.002`.
- Class-1 F1 delta `>= +0.005`.
- Class-1 precision delta `>= +0.010`.
- Class-1 recall delta `>= -0.005`.
- Class-1 TP breaks `<= 1` and FN rescues `>=` TP breaks.
- Restricted class-1 FP reduction `>= 3`.
- Corrections strictly exceed harms.
- Maximum F1 drop among classes `0,2,3,4` is `<= 0.010`.

### Clean candidate versus matched CE control

- Macro F1 delta is nonnegative.
- Class-1 F1 delta `>= +0.003`.
- Class-1 precision delta `>= +0.003`.
- Class-1 recall delta `>= -0.005`.
- Restricted class-1 FP reduction `>= 1`.
- Corrections strictly exceed harms.

### Illumination safety versus raw

For each of dim, bright, and low contrast:

- Class-1 F1 and precision deltas are nonnegative.
- Class-1 recall delta `>= -0.005`.
- Class-1 TP breaks `<= 1` and FN rescues `>=` TP breaks.
- Restricted class-1 FP count does not increase.
- Maximum non-focus class F1 drop is `<= 0.010`.

Across the three conditions, mean class-1 F1 gain must be `>= +0.003`, mean
precision gain `>= +0.005`, and at least two conditions must remove a
restricted false positive.

### Retained comparators

- Candidate class-1 F1, precision, and recall must each exceed the locked MCL
  candidate under every condition.
- Raw and MCL probabilities loaded from prior evidence must replay their locked
  metrics and sample ordering exactly.

## Mechanism Gates

- Candidate mean rank violation is `<= 0.90x` matched control on clean holdout
  and on every shifted condition.
- Candidate support-violation mean and p95 are no greater than control under
  every condition.
- Candidate class-1 capsule residual-margin AUROC versus targets `0,2,4` is
  `>= 0.70` and no lower than control.
- Three-iteration routing differs from the same-weight one-iteration/uniform
  ablation on every audited parameter path and on at least one holdout logit.
- Normalized class-1 routing entropy is in `[0.20,0.995]`; routing may be
  selective but may not collapse to one patch or remain uniform.
- Effective class-1 routed-patch count is in `[4,160]`.
- Candidate dual rank multipliers have nonzero spread and the largest source
  share is `<= 0.02`.
- Constraint reduction cannot be claimed if the clean behavior gates fail.

## XAI Gates

The required cohort is the union of all clean candidate-vs-raw and
candidate-vs-control class-1 TP breaks/rescues, restricted FP removals/creations,
corrections, and harms, with at least 12 deterministic rows when available.

For every row retain:

- RGB input and target/raw/control/candidate probabilities.
- Candidate and control 16x16 class-1 coupling maps with pruned positions zero.
- Candidate-minus-control coupling map.
- Dynamic-versus-uniform routing map.
- Deterministic 4x4 zero-occlusion saliency for deployed class-1 margin.
- Bbox/valid-mask overlays and foreground/border/background mass.

All defined maps must be finite and nonzero, pruned/invalid mass must be zero,
selection replay must be exact, and contact sheets must be inspected before a
decision. Cleaner maps alone never pass a behavior gate.

## Deployment Gates

- Static-batch-1 adapter ONNX and full keeper-plus-adapter ONNX export succeed.
- PyTorch/ONNX maximum logit error `<= 1e-5` and argmax mismatch count is zero.
- Median batch-32 runtime ratio versus raw keeper `<= 1.15`.
- Peak full-model VRAM `<= 3.25 GiB` and adapter-only incremental peak
  `<= 0.75 GiB`.
- Exported routing is exactly three iterations and has no Python-only fallback.

## Stage Decision

`stage_b_authorized=true` only when every structural, behavior, illumination,
mechanism, XAI, replay, resource, and export gate passes. Passing Stage A
authorizes one full `yolo_f/val=2606` adapter evaluation with no test access.
It does not authorize shared-trainer integration, a probe, a full train, test,
or current-best command promotion.

If any gate fails, close exact CapsALM without changing capsule dimensions,
routing count/leak, residual scale, margin, dual normalization, support floor,
optimizer, LR, weight decay, epoch count, batch schedule, fold, seed, cohort,
condition, or decision threshold. A future revisit requires a new primary
source or representation and a new precommitted protocol.

## Required Artifacts

- `summary.json`, `report.md`, and `training_curve.csv`.
- `predictions_all_conditions.csv` for all `4 x 1843` holdout rows.
- Constraint/routing mechanism JSON and CSV.
- Independent replay JSON.
- XAI tensor archive, manifest, and contact sheets.
- Adapter and full static ONNX evidence only; no `.pt`, `.pth`, `.ckpt`, or
  optimizer-state artifact.
- Complete SHA-256 artifact manifest and read-only retention audit.

The current-best command packet and its three-revision history remain unchanged
unless a later, separately locked validation promotion gate is won.
