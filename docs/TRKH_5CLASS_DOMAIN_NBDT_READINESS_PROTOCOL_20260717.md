# TRKH 5-Class Domain NBDT A0 Readiness Protocol (2026-07-17)

Status: prospectively locked before implementation, preflight, or any data
epoch. Official validation and test are forbidden during A0 method selection.

## Decision And Scope

This protocol authorizes one bounded test of Neural-Backed Decision Tree
(NBDT) supervision and soft-path inference on the scratch TRKH classifier. It
does not authorize a new backbone, pretrained weights, a raw-data change, or a
full 30-epoch run.

The agricultural hypothesis is that the five labels already define a useful
sequence of decisions:

1. classes `0,1,2,3` versus class `4`: non-rotten versus rotten/inedible;
2. classes `0,1` versus classes `2,3`: unripe versus ripe;
3. class `0` versus class `1`: undamaged versus lightly damaged/risky unripe;
4. class `2` versus class `3`: naturally ripe/transportable versus the final
   non-transportable ripe state.

This tree directly separates class 1 from class 4 at the root and from classes
2/3 at the maturity node, while retaining the hard class-0/class-1 leaf
decision. The intended benefit is higher class-1 precision without sacrificing
its true positives.

The topology is fixed as `((0,1),(2,3))|4`. It may not be reordered after any
metric is observed. This is a domain adaptation, not a claim that the stock
NBDT induced hierarchy was reproduced.

## Primary Sources And Provenance

Authority is the accepted primary paper and author repository, not a secondary
report:

- Wan et al., "NBDT: Neural-Backed Decision Tree," ICLR 2021 Poster:
  https://openreview.net/forum?id=mCLVeEpplNE
- Accepted paper PDF:
  https://openreview.net/pdf?id=mCLVeEpplNE
- Official repository:
  https://github.com/alvinwan/neural-backed-decision-trees

Observed on 2026-07-17, the official repository exposed 626 stars, 126 forks,
830 commits, and an MIT license. Popularity is supporting provenance, not
evidence that the method will solve TRKH.

Locked local source:

- root: `D:/DataAI/external_sources/official/nbdt-iclr2021`;
- commit: `a7a2ee6f735bbc1b3d8c7c4f9ecdd02c6a75fc1e`;
- tree: `681fd2ca86d6ad238f2586fd4d4d6357d809b8df`;
- `LICENSE` SHA-256:
  `bc292c305fa922306b16498fc543439d1f2f4f908b2b86558b8c601b03a1abd3`;
- `nbdt/loss.py` SHA-256:
  `c9638f8f364d5ab7c01e047900544c476ad7b078289f6ef5d4982eb1209c16ed`;
- `nbdt/model.py` SHA-256:
  `c7a89f5e7a39c8fee0ff15dc27bd1e8415d63424badb06d60e32f7c984bdec7f`;
- `nbdt/tree.py` SHA-256:
  `c36955bc156030b90db41f9f3edd74e024088e4bfc5cbd38b9db31a0c7d20c06`;
- `nbdt/graph.py` SHA-256:
  `d1824ea3152cde71696e7472b85b58711f1a7ec77df091c8740258616a38f5cd`;
- accepted-paper local copy:
  `D:/DataAI/external_sources/official/nbdt-iclr2021-paper.pdf`, SHA-256
  `b6e2c4a0864453460ee87c64829fccf59031c38dfbf9b88200264a1b653ea1e0`.

The repository's latest commit is from 2021 and its complete package import
requires old optional dependencies. TRKH must not install or depend on that
package. It may implement only the small MIT-licensed equations needed here,
with attribution, and must compare them against the locked source plus an
independent direct equation oracle.

## Source Claims And Adaptation Limits

The paper and code establish the following mechanism:

- leaf vectors are the original class vectors;
- each inner-node vector is the mean of descendant leaf vectors;
- child probabilities are a softmax over child node scores;
- each leaf probability is the product of conditional probabilities on its
  root-to-leaf path;
- training adds cross entropy over the soft-path class distribution;
- a linearly increasing tree-loss weight works better than a constant weight;
- soft-path inference can recover from an uncertain early decision;
- tree supervision can also improve the original flat classifier.

The paper reports stronger results from a hierarchy induced by normalized,
fully trained class weights than from a conceptual hierarchy. It also reports
that mid-training hierarchies underperform fully trained induced hierarchies.
TRKH cannot derive an induced hierarchy from the locked A0 holdout or official
validation without leakage, and a random-initialized hierarchy has no visual
meaning. Therefore this A0 uses one fixed domain tree and treats the hierarchy
choice itself as a risk. A result may be described as "domain NBDT" only.

Stock NBDT results use substantially longer schedules and often pretrained or
reproducible baseline models. They do not support a prior claim that a scratch
five-epoch TRKH pair will improve.

## No-Repeat Screen

This route is close to, but not identical to, the rejected semantic-attribute
hierarchical loss from 2026-07-02:

- the old loss independently grouped normalized class probabilities into three
  overlapping tasks (`maturity`, `transport`, `quality`);
- it used constant weight `0.02` during a 24-batch keeper-resume smoke;
- it reached validation macro/class-1 F1 `0.8796/0.6685`, class-1 precision
  `0.5735`, and broadened class-1 false positives;
- it did not form one pure-leaf tree, multiply path probabilities, use NBDT
  soft inference, or use the paper's increasing schedule.

NBDT A0 is allowed only because those causal differences are material and are
locked prospectively. If A0 fails, do not return to nearby semantic groups,
tree topologies, hierarchy weights, hierarchical softmax, ordinal losses, or
post-hoc path thresholds on the current representation.

The following already-closed families are also forbidden as additions to this
pair: semantic-attribute loss, ordinal/EMD losses, pairwise binary heads,
precision routers, teacher distillation, background/bbox forcing, token
selection/recovery, DAT/BRA/FAA, Cropr, EViT, part heads, and pretrained
ensembles. A0 changes only the loss and optional deterministic inference map.

## Locked Inputs

- Repository pre-protocol HEAD:
  `6c8c0442f7a39116eb91755a59824a352833f5b1`.
- Repository pre-protocol tree:
  `bd21056038193dac205876dc07557ea05b8118a9`.
- Raw data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Declaration CSV SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Declaration summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`.
- Fold data/summary SHA-256:
  `4195ba89995d4eee483cae31626717381df910f56657086e7116322ca971388e` /
  `2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763`.
- Reference resolved config SHA-256:
  `c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97`.
- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Current-best command/history SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

Reuse only `runs/yolof_cidt_fold0_trainonly_20260716`:

- fit: 7,372 rows, 6,452 source stems, class counts
  `[1561,432,1527,2017,1835]`;
- holdout: 1,843 rows, 1,612 source stems, class counts
  `[380,109,393,503,458]`;
- source overlap: zero;
- fit/holdout sample-index SHAs:
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d` /
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`.

The pair launcher must use a method-specific fit-only YAML whose train, val,
and test loader paths all point to the 7,372-row fit partition. It must pass
`SkipFinalTest`, use `last.pt`, and disable early stopping as a selection
mechanism. Only the post-pair auditor may construct the 1,843-row holdout
loader. Official `yolo_f/val` and `yolo_f/test` must never be constructed.

The class order and names are immutable:

0. `Xoai_Song_Chua_KhoDap`
1. `Xoai_Song_ChuaNhe_CoNguyCo`
2. `Xoai_Chin_NgotThanh_DeDap`
3. `Xoai_ChinGia_NgotGat_KhongVanChuyen`
4. `Xoai_Hu_KhongAnDuoc`

Raw images, labels, split declarations, and hardlinks are read-only.

## Exact Domain NBDT Equations

Let flat TRKH logits be `z` with shape `[B,5]`. Child logits are means of all
descendant leaf logits, matching the official code's
`EmbeddedDecisionRules.get_node_logits`:

```text
root_logits     = [mean(z0,z1,z2,z3), z4]
maturity_logits = [mean(z0,z1), mean(z2,z3)]
unripe_logits   = [z0, z1]
ripe_logits     = [z2, z3]
```

Apply a two-way softmax independently at every node, giving `r`, `m`, `u`,
and `v`. The five soft-path probabilities are:

```text
p0 = r0 * m0 * u0
p1 = r0 * m0 * u1
p2 = r0 * m1 * v0
p3 = r0 * m1 * v1
p4 = r1
```

They must be finite, non-negative, preserve input dtype where safe, and sum to
one per row within `1e-6` FP32 and `2e-3` CUDA BF16.

Source-faithful tree supervision uses the official implementation behavior:

```text
L_tree = torch.nn.functional.cross_entropy(path_probabilities.float(), target)
```

This intentionally passes the path-probability vector to PyTorch cross entropy,
as `SoftTreeSupLoss` passes `SoftEmbeddedDecisionRules(outputs)` to its wrapped
criterion. It must not be silently replaced by `NLLLoss(log(p))`, hierarchical
softmax, summed node CE, class weighting, label smoothing, focal loss, or a
temperature. Proper path NLL may be reported as an audit-only diagnostic but
may not affect training or promotion.

The candidate objective is:

```text
L_candidate = L_control + w(epoch) * L_tree
```

For the locked five epochs, use inclusive paper endpoints:

```text
w = [0.000, 0.125, 0.250, 0.375, 0.500]
```

This is `0.5 * (epoch-1)/(epochs-1)`, reaching the paper's stated `0 -> 0.5`
small-dataset schedule. The official interpolation equation must replay
exactly; the inclusive endpoint is a declared adaptation to the five-epoch,
one-based TRKH loop. The control weight is always zero.

The function adds no parameter, buffer, stochastic operation, optimizer group,
or checkpoint key. Default-off training and inference must remain bit-exact.

## Four Locked Evaluation Modes

Every checkpoint is evaluated with deterministic normal deployment logits and
with the same pure soft-path map:

1. `control_flat`: control checkpoint, flat logits;
2. `control_soft`: control checkpoint, domain soft-path probabilities;
3. `candidate_flat`: tree-supervised checkpoint, flat logits;
4. `candidate_soft`: tree-supervised checkpoint, domain soft-path probabilities.

The primary method comparison is `candidate_soft` versus `control_flat`.
`candidate_flat` versus `control_flat` isolates representation/loss benefit;
`candidate_soft` versus `candidate_flat` isolates inference benefit; and
`control_soft` is an inference-only compatibility diagnostic. No mode may use a
threshold, calibration fit, router, ensemble, changed class prior, or logits
from a trace-only forward.

## Engineering Preflight

Before any data epoch, a committed and pushed implementation must pass every
material check:

1. official commit/tree/license/paper/source hashes, repository/fold/input/
   protected hashes, class order, and prospective protocol hash are exact;
2. an AST-extracted replay of the locked official `get_node_logits`, softmax,
   and path multiplication matches TRKH node logits/probabilities within
   `1e-7` FP32 on fixed synthetic logits;
3. a fully independent direct equation oracle matches every node logit,
   conditional probability, leaf probability, prediction, official CE value,
   input gradient, and Hessian-vector product within `1e-6`;
4. the fixed schedule is exactly `[0,0.125,0.25,0.375,0.5]`, candidate total
   loss replays independently, and epoch 1 is exactly control-equivalent;
5. probabilities are finite, non-negative, normalized, and map all five target
   paths correctly on analytical high-margin examples;
6. a finite-difference check agrees with autograd for every input-logit family,
   and all five targets produce finite nonzero gradients to relevant leaves;
7. enabling training or inference adds zero parameters/buffers/state keys and
   consumes no constructor or forward RNG; independently constructed matched
   models have bit-identical states and RNG checkpoints;
8. feature disabled/default construction, train loss, inference logits,
   predictions, trace, public tensor shapes, and checkpoint strict load are
   bit-exact to current TRKH;
9. the same standard deployment forward supplies logits for flat and path
   modes; standard/trace parity is exact in FP32 and CUDA BF16 for a fixed full
   batch, with no manual `forward_features` prediction path;
10. FP32/CUDA-BF16 forward and backward are finite; BF16 versus FP32 path
    probabilities differ by at most `0.01` and predictions agree on at least
    `99%` of non-tie synthetic rows;
11. a static batch-1 wrapper exporting image to soft-path probabilities matches
    PyTorch within `1e-4`, preserves argmax, and exposes only standard model
    operators plus mean/softmax/multiply path operations;
12. candidate median inference and training runtime are at most `1.02x/1.05x`
    control, peak allocated VRAM is at most `1.02x` control, and parameter
    counts are equal;
13. the exact `7372/1843` fold, source disjointness, fit-only YAML, no-official-
    holdout construction, and no raw-data mutation checks pass;
14. a keeper-based fit-partition compatibility audit, explicitly non-
    generalization evidence, shows finite/noncollapsed node probabilities and
    no more than `0.03` absolute class-1 precision loss or `10%` relative
    restricted-FP increase from `flat` to `soft`;
15. the wrapper refuses overwrite, dirty tracked worktree, unpushed HEAD,
    mismatched hashes, official validation/test paths, pretrained/resume input,
    any epoch count other than five, and any pair before `formal_pair_permission`
    is true.

A material failure closes A0 before training. A correction after complete
artifacts requires preserving and hashing the failed output, using a new output
directory, and byte-identically replaying unaffected artifacts.

## Sole Five-Epoch Train-Only Pair

Only a complete preflight may authorize one pair:

- random initialization, seed 42, no pretrained or resume checkpoint;
- exact 7,372-row fit partition, no holdout loader during training;
- exactly 5 epochs, 120 train batches per epoch, scheduler horizon 10;
- identical architecture, transforms, sampler order, optimizer, scheduler,
  AMP, EMA, losses, batch size, workers, and checkpoint policy;
- control first, candidate second;
- sample-occurrence hashes must match for every epoch;
- candidate causal difference is only domain NBDT tree-loss weight;
- both are evaluated from `last.pt`, never a fit-metric-selected `best.pt`;
- final test is skipped and official validation is untouched.

The matched recipe is inherited from the current locked scratch TRKH config.
The launcher may only override values required to enforce the fit-only fold,
five epochs, 120-batch cap, deterministic pair, and NBDT candidate flag.

## Post-Pair Quantitative Gates

The 1,843-row source-disjoint train-only holdout is opened once, after both
checkpoints and occurrence hashes exist. Promotion requires all of the
following for `candidate_soft` versus `control_flat` on clean images:

- macro F1 delta at least `+0.003`;
- class-1 F1 delta at least `+0.015`;
- class-1 precision delta at least `+0.025`;
- class-1 recall delta at least `-0.015` and no more than two net class-1 true
  positives lost;
- restricted class-1 false positives from true classes `0,2,4` decrease by at
  least five and do not increase for any individual source class;
- corrections exceed harms, and class-1 TP rescues are not fewer than class-1
  TP breaks;
- no individual non-class-1 F1 drops by more than `0.010`;
- clean path probabilities remain finite and normalized for every row.

The same four modes are evaluated without fitting on fixed `dim`, `bright`,
and `low_contrast` conditions. Candidate class-1 precision and F1 may not fall
below control in more than one condition, and neither may fall by more than
`0.015` in any condition. Restricted false positives may not increase in any
condition. Predictions must come from normal full-batch deployment inference.

Mechanism checks must also show:

- candidate improves or preserves root, maturity, class-0/1 leaf, and class-2/3
  leaf conditional accuracy and log loss on their applicable rows;
- candidate class-1 false positives are not merely moved from one path node to
  another;
- `candidate_flat` is reported even if only `candidate_soft` wins;
- soft inference does not account for a precision gain by an unacceptable
  class-1 recall collapse;
- path entropy, minimum path probability, and first-wrong-node statistics are
  complete for every sample and condition.

No holdout result authorizes topology, weight, schedule, probability transform,
threshold, or epoch changes. A failed gate closes the route.

## Required XAI And Causal Audit

Quantitative success is not sufficient. The auditor must generate and review:

- confusion matrices and per-class precision/recall/F1 for all four modes and
  all four conditions;
- every changed clean prediction, partitioned into correction, harm, neutral,
  class-1 rescue, class-1 TP break, and restricted-FP change;
- node-level conditional probabilities, entropy, target child, predicted child,
  first wrong node, and root-to-leaf product for every row;
- fixed full-batch standard-forward parity hashes for all XAI-selected rows;
- stem Grad-CAM, final attention/rollout, and path-node Grad-CAM for the root,
  maturity node, and relevant leaf decision;
- background blur/gray, object desaturation, dim, bright, and low-contrast
  perturbation effects on flat logits and each path node;
- bbox foreground mass, border mass, center distance, and object/background
  probability-drop summaries for every XAI method;
- complete visual pages for at least all corrections/harms plus a deterministic
  balanced cohort of true class 1 and restricted class-1 false positives.

Path-node saliency must differentiate the log conditional probability for the
actual branch, while the model logits are obtained from the standard deployment
forward. A trace/manual forward that changes logits or predictions invalidates
the visual audit. Every generated page must be opened and reviewed; missing,
blank, clipped, mismatched, or batch-composition-dependent pages fail the gate.

Desired causal evidence is that the root and maturity decisions reject classes
4 and 2/3 before the class-0/1 leaf while the final leaf remains focused on fruit
surface evidence. Cleaner maps without precision, TP, and perturbation gains do
not count as improvement.

## Promotion And Closure

Passing A0 permits a separately prospectively locked official-validation probe;
it does not directly permit full training, final test, or current-best command
updates. Current-best commands remain at three revisions and two metric-bearing
updates until a candidate wins the independent-reload official-validation gate.

If A0 fails, preserve compact summaries, predictions, path traces, visual pages,
source/protocol hashes, and a cleanup manifest; remove only reproducible
checkpoints/ONNX artifacts after retention passes. Record the result in the
research journal, TODO, and TRKH skill, then close all nearby domain-NBDT
topology/weight/schedule/inference variants on the current representation.

