# TRKH 5-Class Supervised-Minority Condition-View Readiness Protocol

Date locked: 2026-07-16

Status: immutable train-only Stage A protocol. Validation and test are forbidden
until every Stage A gate passes. This document authorizes exactly one A0. It
does not authorize a loss, temperature, architecture, schedule, seed, fold,
threshold, view, or residual-scale sweep.

## Question

Can the CVPR 2025 Supervised Minority objective reorganize the keeper's pooled
representation so that class 1 remains coherent while non-class-1 samples stay
instance-discriminative, thereby removing restricted class-1 false positives
without breaking keeper class-1 true positives?

The A0 uses deterministic clean/illumination condition pairs as its two views.
It is a paper-grounded TRKH adaptation, not a reproduction of the paper's
100-epoch ResNet-50 experiment. A precision gain obtained by suppressing true
class-1 support is a rejection.

## Primary Source

- Mildenberger et al., *A Tale of Two Classes: Adapting Supervised Contrastive
  Learning to Binary Imbalanced Datasets*, CVPR 2025.
- CVF paper:
  `D:\DataAI\external_sources\papers\mildenberger2025_tale_two_classes.pdf`
- Paper SHA-256:
  `9919077444fcd45005f1114128524bc593fb1852823e7e849f31b696584c95d8`
- Official MIT-licensed repository:
  `D:\DataAI\external_sources\official\ttc-cvpr2025`
- Commit: `0b1e6974254993b074ad27a226c7ce864da7f95c`
- Tree: `707c9b34ece30ac4a23f1de28be9d0ee803710b7`
- `loss.py` SHA-256:
  `200e763967904f1c498d2e028485b8e7026f4c469c480897ffcf6a97fb934c71`
- `models/sup_cont.py` SHA-256:
  `9a73df402cd160cf2b4188efcd19739157543a88c71a6fba54af5fcfd1d491f1`
- `README.md` SHA-256:
  `b48559693415782936966c88a4dd3a067e53abedad770729f6d53443cf7e2491`
- MIT `LICENSE` SHA-256:
  `2c25d4b5ace923d0e3d2f2105f1639e2604335de4cb0a3d4b2bd16e5cb74087f`

For two normalized views per sample, the paper's Supervised Minority loss uses
all same-class positive pairs for the minority class and only the paired second
view of the same sample for the majority class. In the official implementation
this is `ratio_supervised_majority=0`. The standard SupCon comparator is the
same implementation with all same-class majority positives enabled.

The official recipe uses scratch ResNet-50, batch 256, temperature `0.07`, SGD
with learning rate `0.0625`, momentum `0.9`, weight decay `1e-4`, ten warm-up
epochs, and 100 total epochs. A0 keeps the equation and optimizer constants but
scales the warm-up fraction to `2/20` representation epochs because TRKH is
limited to 30 total epochs. The source paper then trains a linear classifier;
A0 instead trains a five-class keeper-relative residual probe for ten epochs.
Both departures must remain disclosed.

## No-Repeat Boundary

This is not the previously rejected teacher-guided SupCon route. That route
formed supervised positives for several semantic classes, optionally filtered
or weighted them with teacher confidence, and optimized them jointly with the
existing image trainer. A0 instead:

- defines a binary class-1 versus non-class-1 representation problem;
- keeps all non-class-1 samples instance-discriminative;
- uses two frozen-keeper condition views of the same sample;
- trains an isolated zero-residual representation adapter before a separate
  natural-frequency multiclass probe;
- compares against both standard SupCon and an identity-adapter CE-only probe.

Do not reinterpret this protocol as permission to revisit teacher confidence,
memory queues, SNSCL stochastic weights, sub-centers, prototypes, MCL channel
groups, capsules, post-hoc thresholds, or validation-tuned routers.

Fresh source screening rejects the following nearby routes before code:

- Feature Magnitude Regularization reports minimal benefit from scratch
  initialization and is intended to remove pretrained feature bias.
- LoDisc is a global/local SSL pretraining program without an official source
  release found in the screen and does not fit the fixed 30-epoch evidence
  contract without a much larger unverified adaptation.
- Supervised Prototypes is excluded because fixed and multi-prototype geometry
  is already closed on the keeper representation.
- Global CYFLOD, standard SupCon, teacher-guided SupCon, co-teaching, and
  self-paced loss filtering are already closed.

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
- CIDT predictions SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`
- Current-command SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`
- Command-history SHA-256:
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`

The raw dataset is read-only. No path containing a validation or test split may
be opened during Stage A.

## Locked Cohorts

- Source-grouped CIDT fold: `0`.
- Natural multiclass fit: `7372` rows and `6452` source groups.
- Holdout: `1843` rows and `1612` source groups.
- Source overlap: exactly zero.
- Fit class counts: `[1561,432,1527,2017,1835]`.
- Binary fit counts: class 1 `432`, non-class 1 `6940`.
- Fit-index SHA-256:
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d`
- Holdout-index SHA-256:
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`
- Clean holdout class-1 support: `109`.
- Clean keeper class-1 TP/FN: `107/2`.
- Clean restricted hard negatives: `36` holdout rows whose target is in
  `{0,2,4}` and whose keeper prediction is class 1.

Contrastive training uses every fit row at its natural frequency. Class 1 is
not oversampled. Class 3 belongs to the non-class-1 majority for the paper
objective but remains excluded from the restricted agricultural FP count.

## Locked Conditions And Feature Path

Condition order is fixed:

1. `clean`
2. `lighting_dim`: brightness `0.70`, contrast `0.90`
3. `lighting_bright`: brightness `1.25`, contrast `1.10`
4. `low_contrast`: brightness `1.00`, contrast `0.65`

The frozen keeper is run with its exact classification metadata, object crop,
image-valid mask, bbox prior, checkpoint normalization, and FP16 autocast path.
Each cache row stores FP32 copies of:

- the exact `features["pooled"]` tensor with shape `[256]`;
- raw five-class logits;
- target and sample index.

All four conditions cover the same ordered `9215` train objects. Clean cache
probabilities and argmax must replay the locked CIDT clean rows within `2e-3`
maximum probability error and with zero decision mismatch. Cache extraction
must be deterministic across an immediate 32-row replay within `1e-6`.

## Locked Pre-Training Selectivity Gate

The gate uses normalized pooled features, fit rows as the reference bank, and
all holdout rows as queries. It is computed before any adapter update.

- Neighborhood size is the paper's `5%` of the fit bank: `369` neighbors.
- Class Alignment Consistency (CAC) uses binary class-1 labels.
- Sample Alignment Accuracy (SAA) asks whether the same sample's secondary
  condition view is closer than every other holdout secondary view.
- The class-1 score is the fraction of class-1 rows among the 369 fit
  neighbors.
- TP-versus-hard-FP AUROC and mean separation use the fixed clean keeper TP and
  restricted-hard cohorts under each of the four feature conditions.

Training is forbidden unless every condition satisfies:

- all values are finite;
- TP-versus-hard-FP AUROC `>= 0.60`;
- mean class-1 neighbor score on TP exceeds hard FP by `>= 0.02`;
- class-1 CAC is strictly between `0.10` and `0.95`;
- each non-clean clean-to-condition SAA is `>= 0.85`.

The pre-protocol clean diagnostic on an older signed-embedding cache produced
binary CAC `0.956563`, class-1 CAC `0.551801`, neighborhood TP/hard AUROC
`0.718977`, and centroid-direction TP/hard AUROC `0.683541`. These values
motivated A0 but are not formal gate results because that cache is not the
locked pooled feature path.

## Locked Adapter And Controls

The representation adapter consumes a 256D pooled keeper feature:

- `LayerNorm(256)`;
- `Linear(256,64)`, GELU, `Linear(64,256)`;
- the up-projection weight and bias start at exact zero;
- adapted feature: `raw_feature + 0.10 * adapter_residual`;
- official-form projection head: `Linear(256,256)`, ReLU,
  `Linear(256,128)`, followed by L2 normalization;
- five-class residual probe: zero-initialized `Linear(256,5)`;
- deployed logits: `raw_logits + 0.10 * residual_probe(adapted_feature)`.

Exact parameter counts are:

- representation adapter plus projection head: `132288`;
- residual probe: `1285`;
- total per variant: `133573`.

Seed `42` constructs one prototype under RNG isolation. Three deep copies start
bit-exact:

1. `ce_identity`: representation stage skipped; only the residual probe trains.
2. `standard_supcon`: all same-binary-class pairs are supervised positives.
3. `supervised_minority`: all class-1 pairs are positive; each non-class-1
   sample has only its own second view as a positive.

The keeper is frozen and its state hash must remain exact. No variant may use
raw labels to select a threshold, route an inference decision, or alter the
dataset.

## Locked Representation Schedule

- Epochs: exactly `20`.
- Batch size: `256`; `29` batches per epoch including the final partial batch.
- Primary view: always `clean`.
- Secondary view for sample `i` in zero-based epoch `e`:
  `[dim, bright, low_contrast][(e + i) mod 3]`.
- Each epoch order is
  `default_rng(42 + e).permutation(fit_indices)`.
- Total occurrences per trained representation variant: `147440`.
- Secondary counts: dim `49145`, bright `49126`, low contrast `49169`.
- Canonical schedule SHA-256:
  `d73ba25938afb8ba69061ea7c1ae79ac58b254d8a7c9dbb952566b130da1b52a`.
- Temperature/base temperature: `0.07/0.07`.
- SGD: learning rate `0.0625`, momentum `0.9`, weight decay `1e-4`.
- Linear warm-up: two epochs from `0.00625` to `0.0625`.
- Cosine minimum learning rate: `0.0000625` at epoch 20.
- No AMP training, gradient clipping, early stopping, checkpoint selection, or
  validation callback.

`ce_identity` consumes the same schedule only as a detached audit replay. Its
representation/projector state must remain bit-exact to initialization.

## Locked Residual-Probe Schedule

- Epochs: exactly `10`, making `20+10=30` total epochs.
- Batch size: `1024`; eight batches per epoch including the final partial batch.
- Clean fit features only, natural five-class frequency.
- Epoch order is
  `default_rng(1042 + e).permutation(fit_indices)`.
- Total occurrences per variant: `73720`.
- Canonical schedule SHA-256:
  `bf5568009a9e0480aec7f6f8d7f3cce7f08409ac74325930747f5d022a4f10f0`.
- Adapter and projector are frozen; only the zero-init residual probe trains.
- Objective: ordinary five-class cross entropy on deployed logits.
- Adam: learning rate `3e-4`, weight decay `1e-4`, no scheduler or warm-up.

All three probes start bit-exact and consume identical rows in identical order.

## Structural And Equation Gates

All are mandatory:

- Every locked file hash, source commit/tree, license, cohort count, order,
  schedule, condition, and parameter count is exact.
- Official TTC worktree and tracked TRKH worktree are clean. The three protected
  user-owned untracked paths do not invalidate tracked cleanliness.
- `--preflight-only` creates no output directory and opens no image data.
- The local SupMin and standard-SupCon masks/losses match an independent flat
  implementation and the official source within `1e-6`.
- At ratio zero, every non-class-1 anchor has exactly one positive (its paired
  view), and every class-1 anchor has all other class-1 views as positives.
- Candidate/control initialization, probe initialization, and optimizer states
  are bit-exact where the schedules require equality.
- Initial deployed logits are bit-exact to raw for all three variants.
- Every trained parameter family receives finite nonzero gradients; the keeper
  receives no gradient and its state hash never changes.
- `ce_identity` adapter/projector remains bit-exact to initialization.
- Every curve value, cached feature, logit, probability, and metric is finite.
- Each trained representation and each probe consumes exactly its locked
  schedule with no skipped or duplicated row.

Any structural failure rejects Stage A independently of behavior.

## Representation Mechanism Gates

On every condition, `supervised_minority` must satisfy all of the following:

- class-1 CAC delta from raw `>= +0.02`;
- TP-versus-hard-FP AUROC delta from raw `>= +0.02`;
- TP-versus-hard-FP AUROC is at least the standard-SupCon comparator;
- non-class-1 CAC does not fall by more than `0.02` from raw;
- clean-to-condition SAA does not fall by more than `0.02` from raw;
- class-1 adapted-feature mean pairwise cosine increases, while the mean
  non-class-1 off-instance cosine does not increase by more than `0.02`;
- final SupMin loss is at most `90%` of its first-epoch mean;
- adapter residual RMS is nonzero and the projection head does not collapse:
  every output dimension has finite variance and effective rank is `>= 16`.

Mechanism success alone never authorizes validation.

## Train-Holdout Behavior Gates

Metrics use all `1843` holdout rows. Restricted class-1 FP means target in
`{0,2,4}` and prediction class 1.

### Clean supervised-minority versus raw keeper

- Macro F1 delta `>= -0.002`.
- Class-1 F1 delta `>= +0.005`.
- Class-1 precision delta `>= +0.010`.
- Class-1 recall delta `>= -0.005`.
- Class-1 TP breaks `<= 1`; FN rescues `>=` TP breaks.
- Restricted class-1 FP reduction `>= 3`.
- Corrections strictly exceed harms.
- Maximum F1 drop among classes `0,2,3,4` is `<= 0.010`.

### Clean supervised-minority versus both trained controls

For each of `ce_identity` and `standard_supcon`:

- macro F1 delta is nonnegative;
- class-1 F1 delta `>= +0.003`;
- class-1 precision delta `>= +0.003`;
- class-1 recall delta `>= -0.002`;
- restricted class-1 FP count does not increase;
- corrections are at least harms.

### Illumination safety versus raw keeper

For each of dim, bright, and low contrast:

- class-1 F1 delta `>= -0.005`;
- class-1 precision delta is nonnegative;
- class-1 recall delta `>= -0.010`;
- restricted class-1 FP count does not increase;
- class-1 TP breaks `<= 2` and FN rescues `>=` TP breaks;
- corrections are at least harms.

At least two shifted conditions must improve class-1 F1 by `>= +0.003`.

## XAI Gates

XAI is mandatory after the probe even when behavior rejects A0. The required
cohort is every clean holdout row for which candidate versus raw changes a
class-1-related decision, plus every candidate class-1 TP break, FN rescue,
restricted-FP removal, and restricted-FP creation under any condition.

For every required row, retain:

- exact image path, sample index, condition, target, raw/control/candidate
  prediction and probabilities;
- native keeper attention provenance;
- raw and candidate gradient-times-activation patch maps;
- bbox foreground, boundary, and outer-context mass;
- object-desaturation, background-blur, and background-gray probability drops;
- a contact-sheet cell with the unmodified crop and aligned overlays.

No fallback map may be labeled native attention. Tensor rows, page rows, and
case hashes must align exactly; required coverage must be `100%`. Candidate
class-1 evidence must remain foreground-majority on average, and candidate
outer-context or boundary mass may not increase by more than `0.05` versus raw.

## Deployment And Resource Gates

- Isolated adapter/probe ONNX opset 17 matches PyTorch with maximum logit error
  `<= 1e-5` and exact argmax.
- Full keeper-plus-adapter ONNX matches packaged PyTorch with maximum
  probability error `<= 5e-4` and exact argmax on the locked deployment batch.
- Batch-32 PyTorch runtime ratio candidate/raw `<= 1.10` after warm-up.
- Incremental peak CUDA allocation `<= 0.25 GiB`; full peak allocation
  `<= 4.0 GiB`.
- Parameter count and exported graph contain no training-only projector path.
- No TensorRT claim is made at Stage A.

## Evidence And Stop Rules

- Output root:
  `runs/audit_supervised_minority_condition_view_readiness_20260716`.
- The formal root must be absent or empty before execution.
- Permitted binary artifact: one isolated ONNX and one full ONNX for deployment
  replay. No PyTorch checkpoint, optimizer state, validation prediction, or
  test payload may be written.
- Preserve summary, all-condition predictions, curves, representation metrics,
  XAI tensors/pages/manifest, ONNX diagnostics, independent replay, report, and
  a complete SHA-256 artifact manifest.
- Any failed pre-training selectivity gate stops before representation training.
- Any later failed gate denies validation, test, shared-trainer integration,
  full train, keeper replacement, and command-file revision.
- On rejection, do not sweep ratio, temperature, projection size, adapter size,
  residual scale, optimizer, LR, warm-up, epochs, batch, condition pairing,
  fold, seed, sampling, threshold, or router on this keeper.
- Promotion requires every gate plus independent replay. Only then may a
  separately precommitted Stage B evaluate validation without test.

