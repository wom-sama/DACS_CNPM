# TRKH 5-Class Restricted-Negative Learnable ISDA A0 Protocol

## Status And Authorization

This is a prospective, train-only information and mechanism gate for a
restricted-negative, sample-wise implicit semantic augmentation readout. The
formal protocol and its machine-readable lock must be committed and pushed
before candidate code is committed, candidate metrics are produced, or a
candidate artifact is opened.

The A0 may use only the frozen 9,215-row `yolo_f/train` embedding cache and the
already locked source-disjoint CIDT folds. The frozen encoder saw the complete
training split, so this is not encoder-OOF evidence and cannot establish
validation, test, or end-to-end model improvement.

A pass authorizes only:

- default-off implementation in the production trainer;
- a separately locked, matched CE-versus-candidate validation smoke;
- production XAI and robustness audits after that smoke.

A pass does not authorize test access, a probe, a full train, checkpoint
promotion, or a current-best command/history update. A failure closes the
locked neighborhood before any production integration.

## Research Question

The non-gating engineering screen exposes a precise trade-off on the frozen
features:

| Readout | Macro F1 | Class-1 precision | Class-1 recall | Class-1 F1 |
|---|---:|---:|---:|---:|
| Natural CE | 0.939956 | 0.816029 | 0.828096 | 0.822018 |
| Restricted class-wise ISDA | 0.937421 | 0.844311 | 0.781885 | 0.811900 |

The class-wise robust loss raises class-1 precision by `0.028282` but loses
`0.046211` recall and `0.010118` F1. These values were observed in a
control-only, train-cache engineering screen before candidate code or
candidate metrics. They are rationale, not formal evidence.

The A0 asks whether a sample-wise covariance predictor, optimized by
source-disjoint and class-balanced inner meta exchange, can retain the useful
false-positive pressure of restricted class-wise ISDA while recovering true
class-1 recall.

## Primary Sources And Adaptation Boundary

The method is derived independently from equations in three accepted primary
sources:

- ISDA, NeurIPS 2019:
  <https://papers.nips.cc/paper_files/paper/2019/hash/15f99f2165aa8c86c9dface16fefd281-Abstract.html>.
  Local paper SHA-256:
  `6df013289f38f2709a84fd41c5e9c63e5eeb91060a68fae4959d72858de6936b`.
- MetaSAug, CVPR 2021:
  <https://openaccess.thecvf.com/content/CVPR2021/html/Li_MetaSAug_Meta_Semantic_Augmentation_for_Long-Tailed_Visual_Recognition_CVPR_2021_paper.html>.
  Local paper SHA-256:
  `d13adcb04e3144a07d5ee3d11787dfa575e8df8711f0de59f57c1cae0bd9fc60`.
- Fine-Grained Recognition With Learnable Semantic Data Augmentation,
  IEEE TIP 2024, DOI `10.1109/TIP.2024.3364500`:
  <https://arxiv.org/abs/2309.00399>.
  Local paper SHA-256:
  `32ec2779e3e7e15f9542cd5a70556fd6de631f54d49fa4008bd7f445e67233c2`.

The official LearnableISDA reference is pinned only for provenance at commit
`9473a481d9639efd64ebec2452ce2ec4f5dc2223`, tree
`1374f08c765e6e049f5936a0d1ca5add10712f6f`. No root license is visible.
No source code may be copied, imported, vendored, installed, or adapted
line-by-line. The TRKH implementation must be an independent expression of the
paper equations and must pass a separate NumPy/FP64 oracle.

The adaptation is not an exact reproduction:

1. LearnableISDA predicts a positive sample-wise diagonal covariance. TRKH
   predicts a bounded scale over partition-local empirical class variance.
2. TRKH applies the robust term only to true classes `{0,2,4}` and only to
   rival logit `1`; true classes `1` and `3` receive exact clean CE.
3. TRKH uses the paper's P2P combination settings that fit the current hybrid:
   `lambda_0=5`, one hidden layer of half the 256-dimensional feature width,
   and a meta update every ten paired iterations.
4. Following the balanced meta-objective principle in MetaSAug, each CovNet
   update is evaluated on exactly `30` rows from the opposite source-disjoint
   partition, `6` per class. Unlike MetaSAug, this A0 does not reserve or open a
   validation split: it cycles deterministically through train-only rows, and
   the real head batches retain their natural distribution.
5. The A0 trains only linear heads and covariance predictors over frozen
   train embeddings. It does not claim an end-to-end LearnableISDA result.

## Locked Restricted-Negative Equation

Let `x_i in R^256` be a frozen feature, `y_i` its true class, and
`W in R^(5x256), b in R^5` the clean linear head:

```text
z_i = W x_i + b
```

For partition `p` and class `c`, compute the population diagonal variance
from only the real rows in that partition:

```text
v[p,c,d] = max(Var({x[k,d] : k in p and y[k]=c}, ddof=0), 1e-8)
```

The covariance predictor is a one-hidden-layer MLP:

```text
h_i = ReLU(A x_i + a)
s_i = 2 * sigmoid(B h_i + q)
sigma_i^2 = v[p,y_i] * s_i
```

The output layer `B,q` is initialized to exact zero. Therefore
`s_i = 1` and `sigma_i^2 = v[p,y_i]` before the first meta update. The hidden
layer uses the standard PyTorch linear initialization. The predictor input is
detached from the classifier feature.

For an eligible negative class `y_i in {0,2,4}`, only rival logit `j=1`
receives the implicit robust increment:

```text
Delta[i,1] =
  lambda(epoch) / 2
  * sum_d ((W[1,d] - W[y_i,d])^2 * sigma_i^2[d])
```

Every other increment is exactly zero:

```text
Delta[i,j] = 0 for j != 1
Delta[i,:] = 0 for y_i in {1,3}
L_RN-LISDA = CE(z_i + Delta_i, y_i)
```

The real classifier update uses `sigma_i^2.detach()`. CovNet receives gradients
only through the meta objective, except in the explicit joint-no-meta collapse
control.

Feature augmentation is ephemeral. Every implicit child inherits the exact
`sample_index`, `source_stem`, outer fold, and inner partition of its real
parent. It may be consumed only by the update for that partition and is
discarded immediately; no synthetic vector may move to another partition or
be written into the raw dataset.

The explicit Gaussian vectors used by the post-training semantic audit are
scientific evidence, not training rows. They are generated only after the
outer-holdout clean probabilities have been frozen, inherit the anchor's
identity and outer fold, and may be persisted only under the formal audit
directory. They cannot enter a gradient, nearest-neighbor search pool,
selector, threshold, or later training corpus.

Any future pixel-space augmentation or synthetic-image route requires its own
prospective lock. It must fit generators and all generation/filtering
parameters on fit-train parents only; every descendant inherits all parent
source IDs and fold assignments. After the corpus is frozen, validation/test
may be read only by a separate duplicate audit that can invalidate the entire
corpus or method. It may not remove individual rows, replenish them, choose a
generator, tune a threshold, or otherwise feed information back into training.

## Immutable Inputs

The final machine lock records exact path, byte count, and SHA-256 for:

- `runs/diagnostic_reslt_embedding_cache_keeper_yolof_20260712/train_embeddings.npz`;
- `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`;
- the keeper checkpoint;
- keeper `resolved_config.json`;
- `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`;
- current-best commands and current-best history;
- this protocol; each run records the machine-readable JSON lock SHA-256;
- the three local papers and the two pinned official repository states.

Expected cache declarations:

- rows: `9,215`;
- embedding shape: `[9215,256]`;
- class counts: `[1941,541,1920,2520,2293]`;
- outer-fold counts: `[1843,1830,1828,1851,1863]`;
- cache paths: `yolo_f/images/train` only;
- outer source overlap: zero;
- ordered sample indices: exactly `0..9214`.

The runtime lock is Python `3.9.11`, NumPy `1.26.4`, PyTorch
`2.6.0+cu124` with CUDA `12.4`, scikit-learn `1.6.1`, Pillow `11.3.0`,
psutil `7.2.2`, and `NVIDIA GeForce RTX 4060 Laptop GPU`. Preflight and
replay fail before candidate scoring if any value differs.

The aggregate embedding-cache manifest and CIDT summary are deliberately not
formal inputs because they are unnecessary and can describe non-train
payloads. The executable may hash declarations listed above but semantically
loads only the train cache and clean train CIDT rows.

Before the first candidate input is opened, the executable installs a
process-wide Python audit hook for file-open events and requires a custom
installation-probe event to be observed before continuing. The hook records every
open under `D:\DataAI\AIEx\newdataset` or `D:\DataAI\AIEx\TRKH\runs`,
including access from library code and worker threads. Any observed path with
a complete component equal to `val`, `valid`, `validation`, or `test` raises
before the open completes. The ordered normalized-path/mode ledger, per-path
counts, blocked-attempt count, and SHA-256 digest are written to
`data_access_ledger.json`; replay must reproduce the zero-blocked,
train-only contract. Dataset access is additionally rooted at the parent of
the hash-locked `yolo_f/data.yaml`; `class_f/train` is not an RN-LISDA A0
input and is blocked despite being a valid view for other TRKH experiments.
Repository source, protocol, lock, output artifacts,
Python packages, fonts, and operating-system files are outside this
data-domain ledger and remain covered by their separate immutable-input,
repository, manifest, and process gates.
This hook is evidentiary defense-in-depth for hash-locked, non-adversarial
candidate code, not a security sandbox. CPython 3.9 documents that audit hooks
can observe events and abort operations by raising, while warning that
Python-level hooks are bypassable by malicious code:
<https://docs.python.org/3.9/library/sys.html#sys.addaudithook>. The locked
runtime's `open` event contract is `(path, mode, flags)`.

During protocol engineering, the aggregate embedding-cache manifest was
accidentally inspected once. It exposed historical split counts and hashes
already present in project records, but no RN-LISDA candidate output, metric,
state, threshold, role comparison, or visual evidence. The manifest was then
removed from the immutable-input set and cannot be opened by formal execution.
This disclosure prevents a global validation-blind claim; it does not authorize
using the disclosed counts or hashes to choose any method detail.

Historical keeper validation metrics and split counts already documented in
the project were known before this protocol; therefore this A0 is not claimed
to be globally validation-blind. They are not executable inputs and cannot
choose this method, equation, role, seed, gate, or visual row. During A0,
validation/test pixels, labels, paths, predictions, embeddings, checkpoints,
statistics, thresholds, and metrics are forbidden. Raw `class_f` and `yolo_f`
files are read-only.

## Outer And Inner Source Isolation

The five fixed CIDT source folds are outer holdouts. For each outer fold `f`,
the other four folds are split once into inner partitions `A_f` and `B_f`
using:

```text
StratifiedGroupKFold(
    n_splits=2,
    shuffle=True,
    random_state=20260724 + f,
)
groups = source_stem
```

The complete index/source hashes and class counts are stored in the JSON lock.
Every inner source overlap must be zero. No candidate metric, state, or visual
may influence either split.

Every numeric-array hash is SHA-256 over ASCII dtype, an `int64` shape vector,
and contiguous array bytes. Every string-sequence hash includes an `int64`
sequence count and an `int64` UTF-8 byte length before each string. A complete
partition assignment is an `int8[9215]` array with held `-1`, `A=0`, and
`B=1`. The auditor must reproduce all 65 partition/index/source/derangement/
held-donor/balanced-meta-schedule hashes in the JSON lock before training.

During training, `A_f` and `B_f` exchange roles:

1. pseudo train on a natural `A_f` batch, clean meta objective on a balanced
   `B_f` batch, then real update on the natural `A_f` batch;
2. pseudo train on a natural `B_f` batch, clean meta objective on a balanced
   `A_f` batch, then real update on the natural `B_f` batch.

The balanced meta sampler is deterministic, source-aware, and least-used
first. For each slot it rotates which class receives first source choice, then
ranks rows within every class first by prior meta-use count and then by a
locked NumPy `default_rng` permutation seeded from
fold/partition/slot/class. Every meta batch has `6` rows from each class, `30`
distinct rows, and `30` distinct source stems. Reuse across later slots is
allowed only in the meta objective and is spread across rows by the least-used
rule; within each partition/class, the largest and smallest row-use counts may
differ by at most one. The sampler never duplicates, reweights, drops, or
reorders rows in the real head-training stream. Because `A_f` and `B_f` are
source-disjoint, every pseudo-train/meta pair also has zero source overlap.

The outer holdout is scored clean exactly once, after epoch 30. It cannot
select an epoch, head, threshold, covariance state, visual row, or role.

## Locked Outer-Holdout Mechanism View

Mechanism and visual evidence is computed only after the clean outer-holdout
probabilities are frozen. For fold `f`, `v_fit[f,c]` is the population
diagonal variance of the real `A_f union B_f` rows of class `c`, clamped at
`1e-8`. It is never used by a training update.

For every outer-holdout row, the same-label swap donor is fixed without model
output:

1. rank outer-fit rows of the same label by SHA-256 of
   `rn_lisda_holdout_donor_rank|20260724|fold|class|index|source_stem`;
2. choose rank
   `stable_seed(rn_lisda_holdout_swap,20260724,fold,held_index) mod N`;
3. assert donor label equality, donor membership in outer fit, and source
   disjointness.

The lock records the held-index and donor-index array hashes for every fold.
Candidate and joint-no-meta held covariance is
`v_fit[f,y] * CovNet(input)`. Effective rank is the participation ratio
`sum(sigma2)^2 / sum(sigma2^2)`. Sample-wise variation is the population
standard deviation, over eligible held rows, of each row's mean scale.
Same-label input-swap change is the mean row-wise
`L1(sigma2(x)-sigma2(donor)) / max(L1(sigma2(x)),1e-12)`. Candidate versus
joint covariance compares the global mean over all eligible held rows and
dimensions. None of these quantities may affect training or row selection.

## Fixed Visual Anchors

Twenty outer-holdout anchors are selected before candidate training. Each fold
contains:

- the correctly predicted class-1 row with the smallest positive keeper
  class-1 margin;
- the strongest keeper class-1 false positive for each true class `0,2,4`;
- when a fold has no such false positive, the nearest class-1 boundary row for
  that true class is used without replacement.

Locked sample indices by fold:

| Fold | Class-1 TP | True 0 | True 2 | True 4 |
|---:|---:|---:|---:|---:|
| 0 | 4002 | 3195 | 3284 | 4613 |
| 1 | 3217 | 2127 | 2707 | 5213 |
| 2 | 4616 | 4770 | 3955 | 4012 |
| 3 | 4624 | 3248 | 2718 | 5732 |
| 4 | 5327 | 6113 | 4916 | 3624 |

Fold 1 has no true-class-4 keeper class-1 false positive; index `5213` is the
prospectively fixed boundary fallback. The immutable CIDT CSV SHA-256 plus the
ordered anchor-index hash fixes every corresponding label, source, path,
keeper probability, and prediction; the JSON lock also fixes every held
same-label swap donor.

## Matched Roles

Exactly eight roles are trained:

1. `ce_control`: clean CE for all 30 epochs.
2. `restricted_classwise_isda`: ten CE epochs, then the restricted equation
   with fixed partition-local class variance and no CovNet.
3. `rn_lisda_candidate`: the primary restricted-negative meta candidate.
4. `rn_lisda_covnet_seed_repeat`: the same method, folds, head initialization,
   and batch orders with only an independent CovNet hidden-layer seed namespace.
5. `restricted_all_rival_meta`: sample-wise covariance for true classes
   `{0,2,4}`, but robust increments are applied to every non-target rival.
6. `full_all_class_meta`: sample-wise covariance for all true classes and all
   non-target rivals.
7. `rn_lisda_deranged_input`: the candidate equation, but each CovNet input is
   replaced by a fixed same-label, same-partition, different-source
   derangement; labels, heads, batch order, and base variance stay unchanged.
8. `rn_lisda_joint_no_meta`: the candidate equation, but CovNet is optimized
   directly to minimize the augmented training loss and receives no meta
   gradient.

All roles share byte-identical head initialization and batch orders within a
fold. All CovNet-bearing roles except `rn_lisda_covnet_seed_repeat` share the
exact initial CovNet state. The paired repeat applies the locked `+100000`
offset only to the CovNet initialization; its head, batch orders, data
partitions, optimizer, schedule, equations, gates, and scoring remain
identical. This common-random-number design isolates sensitivity to the learned
covariance mechanism instead of confounding it with a different classifier or
sample order. No role receives class weights, global class-1 oversampling,
duplicate rows in the real head update, threshold calibration, or early
stopping. Balanced repetition is confined to the CovNet meta objective and is
identical across all meta roles.

## Locked Optimization

- device: CUDA FP32;
- deterministic algorithms: enabled;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- TF32: disabled for CUDA matmul and cuDNN;
- epochs: exactly `30`;
- clean warm-up phase: epochs `0..9`;
- semantic phase: epochs `10..29`;
- total paired batch size: `64`, as `32` from `A_f` and `32` from `B_f`;
- paired steps: `min(floor(|A_f|/32), floor(|B_f|/32))`;
- drop-last: true within each partition;
- head optimizer: independently implemented SGD, LR `0.03`, momentum `0.9`,
  dampening `0`, weight decay `0`, no Nesterov;
- CovNet optimizer: independently implemented SGD, LR `0.001`, momentum `0`,
  weight decay `0`;
- LR schedule for both: paper-compatible five-epoch linear warm-up followed
  by cosine decay over the 30-epoch horizon;
- both exchange directions use the same LR for a paired iteration;
- augmentation strength:
  `lambda(epoch) = 5 * (epoch - 9) / 20` for epochs `10..29`;
- meta schedule: at semantic paired-iteration indices `0,10,20,...`, counting
  continuously across epochs; both exchange directions perform a meta update;
- each meta objective uses a deterministic `30`-row opposite-partition batch,
  exactly `6` rows per class and `30` unique source stems, selected by the
  rotating-priority least-used rule; this schedule is hashed prospectively for
  every fold and partition;
- natural real-head batches remain unchanged and receive no class balancing;
- no outer-holdout evaluation before the final clean score.

The differentiable pseudo-SGD step must include the current real head momentum
buffer. A focused test compares the functional update with the real update for
both the first-buffer and existing-buffer cases.

## Pre-Candidate Synthetic Resource Benchmark

Before any RN-LISDA candidate metric or TRKH candidate input was opened, the
locked implementation shape was exercised on `9,215` independent synthetic
normal vectors with the locked class counts and representative partition
sizes. The benchmark reused the formal peak-resource monitor, but it opened no
TRKH cache, dataset, checkpoint, prediction, validation/test artifact, or
metric. Its immutable evidence is
`docs/TRKH_5CLASS_RESTRICTED_NEGATIVE_LEARNABLE_ISDA_A0_SYNTHETIC_BENCHMARK_20260724.json`.

The superseding v3 run projected `23.663158` formal minutes after a `25%`
margin, measured `1.142296 GiB` peak process-tree RSS, and measured
`0.084456 GiB` peak CUDA allocation. These are below the prospectively chosen
`45 minute`, `4 GiB`, and `2 GiB` gates, respectively, so no resource threshold
was relaxed. An initial harness attempt stopped before role metrics because the
deterministic cuBLAS workspace setting was not installed; v2 then completed
without RSS sampling, and v3 superseded it with the exact formal monitor. No
attempt produced a candidate performance metric. This benchmark establishes
runtime feasibility only and is not scientific performance evidence.

## Required Metrics And Transitions

For every role, aggregate and per-fold evidence must include:

- confusion matrix;
- per-class precision, recall, F1, support, and predicted count;
- macro F1 and accuracy;
- NLL, multiclass Brier score, and 15-bin equal-width ECE;
- class-1 TP/FN and restricted `{0,2,4}->1` FP;
- clean prediction and probability arrays;
- training loss, LR, lambda, update counts, and gradient norms by epoch.

Every candidate-like role is compared with `ce_control`. Required transition
counts are:

- corrections: CE wrong, role correct;
- harms: CE correct, role wrong;
- class-1 FN rescues and TP breaks;
- restricted-FP removals and newly created restricted FP;
- net class-1 TP and net restricted-FP removal.

The primary candidate is also compared directly with every matched role.

## Feature-Semantic Audit

The formal run creates one fixed, clearly labeled contact sheet over all 20
anchors. Its image tiles are real train images used as nearest-neighbor
proxies; they are not generated pixels.

For each eligible anchor in true class `{0,2,4}`:

1. show the exact model object crop;
2. show the nearest same-label, different-source real row from the outer-fit
   set;
3. draw one explicit Gaussian feature from `lambda_0 * v_fit[f,y]` and show
   its nearest different-source real outer-fit proxy;
4. draw two explicit Gaussian features from the candidate covariance using
   `lambda_0 * v_fit[f,y] * CovNet(x)`, fixed sample-index seeds, and show
   their nearest different-source real outer-fit proxies;
5. show the corresponding deranged-input candidate proxy.

Every explicit draw is FP32
`x + sqrt(5 * sigma2) * epsilon`, where `epsilon` comes from NumPy
`default_rng(stable_seed("rn_lisda_visual",20260724,fold,sample_index,role,draw))`.
Nearest-proxy search uses cosine distance between L2-normalized frozen
embeddings, searches only real outer-fit rows with a different source, and
breaks exact ties by ascending sample index. The deranged-input cell uses the
locked held-row donor through the final `rn_lisda_deranged_input` control
CovNet before applying the exact candidate draw-0 epsilon. The separate
same-label input-swap mechanism metric instead passes that donor through the
primary candidate CovNet; the two quantities must not be conflated.

Nearest-proxy search is unrestricted by label after sampling; the displayed
label and distance expose semantic drift. No anchor or proxy may come from
validation/test or the anchor's source. For class-1 anchors, all augmentation
cells must state `disabled by restricted-negative design`; no Gaussian sample
is drawn.

Metadata records covariance mean, standard deviation, effective rank, scale
range, proxy labels/distances, role probabilities, and prediction
transitions. Manual review covers every row at original raster detail and must
reject systematic background, basket, hand, padding, silhouette, source-style,
or class-semantic shortcuts. It must also reject a candidate visually
indistinguishable from the fixed class-wise or deranged controls.

## Conjunctive Structural Gates

All must pass:

1. every immutable path/hash/repository revision matches;
2. repository tracked state is clean, HEAD equals upstream, and the protected
   untracked set is unchanged;
3. cache keys/shapes/order/class counts and the process-wide dynamic
   data-access ledger match; every observed data-domain open is train-only
   and no validation/test open is attempted or completed;
4. all 65 partition/provenance/balanced-meta hashes match and all outer and
   inner source overlaps are zero;
5. all folds/partitions contain every class;
6. every update consumes only its declared inner partition;
7. outer holdout is scored only after epoch 30 and never enters a gradient,
   scheduler, selector, threshold, state, or visual-row choice;
8. every implicit or explicit feature child has complete parent
   `sample_index/source_stem/fold/partition` provenance and cannot cross a
   partition boundary;
9. held mechanism/visual evidence is created only after probabilities freeze
   and cannot feed back into training or selection;
10. validation/test access counters are zero;
11. raw dataset changes are zero;
12. exact role count, initial states, paired batch-order hashes, balanced
    meta-batch hashes/class counts, partition consumption hashes, head updates,
    meta updates, and scored rows match;
13. no NaN/Inf occurs;
14. deterministic CUDA/TF32 settings match;
15. start available physical RAM is at least `4 GiB`, peak process RSS is
    below `4 GiB`, peak CUDA allocated memory is below `2 GiB`, and formal
    elapsed time is below `45 minutes`;
16. no unrelated Python/TensorRT process exists during formal execution;
17. current-best commands/history and keeper checkpoint remain byte-identical;
18. artifact manifest verifies every recursive payload.

## Conjunctive Mechanism Gates

All must pass:

1. restricted equation matches the FP64 oracle within `1e-6`;
2. target classes `1` and `3` have exactly zero robust increments;
3. eligible rows have exactly one potentially nonzero rival column, class 1;
4. pre-meta zero output makes candidate covariance equal fixed class-wise
   covariance within `1e-7`;
5. the first scheduled meta update has finite nonzero CovNet output-layer
   gradient and hidden-layer gradient at most `1e-12`;
6. by the second scheduled meta update, both output and hidden gradients are
   finite and strictly nonzero;
7. every later scheduled meta update has finite, strictly nonzero output and
   hidden gradient;
8. each primary candidate fold changes CovNet state from initialization;
9. candidate covariance is positive, finite, and within `[0,2]` times the
   partition-local base variance;
10. median candidate covariance effective rank over eligible held rows is at
    least `8`;
11. eligible held-row sample-wise scale standard deviation is at least `0.01`;
12. the mean relative covariance change under the fixed same-label
    different-source input swap is at least `0.05`;
13. candidate mean covariance is at least `1.10` times joint-no-meta mean
    covariance;
14. every scheduled meta batch has exactly `[6,6,6,6,6]` class counts,
    `30` unique rows, `30` unique source stems, and zero source overlap with
    its pseudo-train partition; per-class row-use spread is at most one;
15. fixed contact metadata is complete for `20/20` anchors and all proxy
    source/fold constraints pass.

Gradient-order checks 5-7 apply independently to the primary
`rn_lisda_candidate` trace in every fold. Other meta roles must still have
finite states, losses, gradients, and nonzero scheduled updates, but they
cannot substitute for a failed primary mechanism trace.

## Conjunctive Performance Gates

### Control compatibility

1. CE macro F1 is at least `0.935`;
2. CE class-1 F1 is at least `0.80`;
3. class-wise ISDA class-1 precision is at least CE `+0.015`;
4. class-wise ISDA removes at least `10` net restricted FP versus CE.

### Primary candidate versus CE

1. macro F1 gain is at least `+0.001`;
2. class-1 precision gain is at least `+0.015`;
3. class-1 F1 gain is at least `+0.005`;
4. class-1 recall is at least `0.80` and no more than `0.010` below CE;
5. net restricted-FP removal is at least `10`;
6. net class-1 TP is at least `-3`;
7. corrections are at least harms;
8. maximum F1 loss over non-focus classes is at most `0.005`;
9. NLL and ECE are each no more than `0.010` above CE.

### Fold stability

1. class-1 precision is non-worse than CE in at least `4/5` folds;
2. class-1 F1 improves in at least `3/5` folds;
3. no fold loses more than `0.030` class-1 recall;
4. no fold loses more than two class-1 TP.

### Candidate versus fixed class-wise ISDA

1. class-1 recall recovers at least `+0.025`;
2. class-1 F1 improves at least `+0.010`;
3. candidate precision preserves at least half the class-wise precision gain
   over CE;
4. candidate net restricted-FP removal preserves at least half the class-wise
   net removal.

### Causal controls

1. candidate class-1 F1 exceeds restricted-all-rival meta by at least `0.002`,
   while macro F1 is no more than `0.001` lower;
2. candidate class-1 F1 exceeds full-all-class meta by at least `0.005`, while
   macro F1 is no more than `0.001` lower;
3. candidate exceeds deranged-input class-1 precision and F1 by at least
   `0.005` each;
4. candidate class-1 F1 exceeds joint-no-meta by at least `0.005`.

### Paired CovNet-seed repeat

1. the repeat passes all primary-versus-CE, class-wise recovery, and
   fold-safety gates under the paired head/order controls;
2. primary/repeat prediction agreement is at least `0.97`;
3. absolute macro-F1 and class-1-F1 differences are at most `0.01`;
4. absolute class-1 precision and recall differences are at most `0.02`.

No metric gate compensates for another failed gate.

## Replay, Manifest, And Manual Finalization

Formal execution writes:

- fixed fold/partition protocol;
- epoch/update trace;
- OOF probabilities and row-level CSV;
- role states and array hashes;
- mechanism telemetry;
- dynamic data-access ledger;
- fixed visual arrays, metadata, and contact sheet;
- immutable summary and a recursive pre-replay artifact manifest.

A fresh Python process must:

1. verify the pre-replay formal manifest;
2. re-verify all immutable inputs and clean pushed repository state;
3. reconstruct every inner split, derangement, initialization, natural batch
   order, balanced meta schedule, update, role state, OOF probability, metric,
   transition, covariance telemetry, dynamic data-access contract, and fixed
   visual array;
4. compare probabilities and states within `1e-7`, integer actions exactly,
   nested metrics within `1e-10`, and fixed visual arrays exactly;
5. write `replay.json` without changing a scientific artifact.

The final manual pass/reject file is written only after the fresh replay and
complete visual inspection. It does not mutate the formal summary or replay.
A separate final manifest covers the pre-replay manifest, replay, manual
review, and final decision. Authorization is the conjunction of structural,
mechanism, performance, replay, and manual visual gates.

## Escalation And Stop Rules

If any gate fails:

- reject A0;
- do not integrate with the production model or trainer;
- do not open validation/test, smoke, probe, full train, or current commands;
- do not sweep nearby `lambda_0`, covariance floor/scale, hidden width/depth,
  meta interval/LR/balanced-batch size, warm-up, eligible classes, rival set,
  batch size, seed, fold, epoch, or gate threshold;
- preserve the compact formal/replay/visual evidence and close this exact
  restricted-negative LearnableISDA neighborhood.

If all gates pass, the next stage is a separately committed default-off
trainer implementation and one matched CE-versus-RN-LISDA validation smoke.
That smoke must preserve the 30-epoch ceiling, test isolation, class-1
precision priority, explicit recall budget, complete XAI/robustness audit, and
independent reload. Current-best commands may change only after a real locked
validation winner.
