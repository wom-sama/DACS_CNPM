# TRKH 5-Class Sparse Over-Parameterization A0 Readiness Protocol - 2026-07-17

## Status

Precommitted before implementation, data output, or behavior measurement. This
document first authorizes one isolated fit-only compatibility gate for Sparse
Over-Parameterization (SOP). Only a complete compatibility pass may authorize
a default-off trainer integration, an engineering preflight, and one exact
ten-epoch scratch control/candidate pair on the locked train-only fold.

It does not authorize official validation or test access, raw-data changes,
hyperparameter sweeps, a full train, or a current-best command update.

## Research Question

Can per-example sparse label-noise variables prevent the current scratch
CNN/Transformer hybrid from memorizing ambiguous labels without weakening the
gradient that must remove class-1 false positives or breaking true class-1
support?

SOP is relevant because it does not edit a label file or add an inference
module. During training it decomposes the observed target into a network
prediction plus a sparse per-example noise term. At deployment the noise
variables are absent, so any gain must be carried by the unchanged hybrid.

The mechanism also has a project-specific risk. For a correctly labelled
class-0/2/4 sample that the model predicts as class 1, SOP may treat the
network prediction as the latent clean signal and absorb the disagreement in
its per-example variable. That can suppress the very class-1 corrective
gradient required by the agricultural precision priority. A direct gradient
compatibility gate therefore precedes trainer work.

## Primary-Source Lock

User-supplied and secondary reports are hypothesis sources only. The accepted
paper and the authors' licensed repository define A0.

- Accepted paper: Liu, Zhu, Qu, and You, "Robust Training under Label Noise by
  Over-parameterization," ICML 2022, PMLR 162:14153-14172:
  `https://proceedings.mlr.press/v162/liu22w.html`.
- Accepted PDF:
  `https://proceedings.mlr.press/v162/liu22w/liu22w.pdf`.
- Local PDF:
  `D:/DataAI/external_sources/papers/liu2022_sop_icml.pdf`.
- Paper SHA-256:
  `087e1c05425bf9682351a27097ce25cb2ecc94a912b715bbebc97589506bac12`.
- Official repository: `https://github.com/shengliu66/SOP` (66 stars and
  seven forks observed on 2026-07-17; acceptance and source provenance are the
  authority, while popularity is supporting context only).
- Locked commit/tree:
  `4d991cedf1fafec98f858a213ccc31e52318a77f` /
  `dadbf1b288599c67d7fcced4db2cbdc3db10c6cd`.
- Official MIT license SHA-256:
  `05d6838d881952883e6da340405ba12c3c94372f7f30298beec0f1d902cb8886`.
- Reviewed official loss `model/loss.py`, SHA-256:
  `66b252e62b1e0e1d8ed50a896f1eaaed9506a35a68a96f1b0cf0966ba744af48`.
- Reviewed official loop `trainer/trainer.py`, SHA-256:
  `8c53028d7e4b220b0ddaac7010a1eedf34a41a35427cdb7120e3b8ee6392a8db`.

For one-hot target `y`, network logits `z`, one positive scalar `u_i`, and
one five-vector `v_i`, the locked official implementation computes

```text
q       = softmax(z)
U       = clamp(u_i^2 * y, 0, 1)
V       = clamp(v_i^2 * (1 - y), 0, 1)
p_raw   = clamp(q + U - stopgrad(V), min=1e-4)
p       = clamp(p_raw / ||p_raw||_1, 1e-4, 1)
y_hat   = one_hot(argmax(stopgrad(z)))
L_CE    = mean(-sum(y * log(p)))
L_MSE   = sum((y_hat + U - V - y)^2) / batch_size
L_SOP   = L_CE + L_MSE
```

The paper explains why `v` needs the MSE term: under the normalized CE path,
its classwise gradient cannot identify the network-label discrepancy. The
official source detaches `V` from `L_CE`, uses SGD without momentum or weight
decay for `u/v`, initializes both from `N(0, 1e-8)`, and uses one scalar `u`
plus one class-vector `v` per training example.

The source evidence is not a guarantee for TRKH. CIFAR schedules are 120-300
epochs. The only ten-epoch result uses one million Clothing1M examples and a
pretrained ResNet-50. A0 borrows the ten-epoch real-noise schedule as the
largest source-grounded duration comfortably inside the project limit; it
does not claim equivalent convergence from scratch.

## No-Repeat And Candidate Screen

SOP is distinct from the closed robust-loss and sample-selection routes:

- GCE, SCE, ELR, Tversky, AUC/pAUC, CYFLOD, class-balanced self-paced loss,
  label-transition correction, and static sample weights alter a shared loss
  or target but do not jointly learn sparse per-example additive noise.
- Co-teaching and Cleanlab choose, downweight, or relabel examples using peer
  or out-of-fold predictions. A0 neither edits raw labels nor removes rows.
- Structural Labels, Jo-SNC/SNSCL, prototypes, and neighborhood policies use
  feature-neighbor agreement; A0 has no neighbor graph.
- NBDT and semantic/ordinal hierarchies change class structure or inference;
  A0 leaves the five-class deployment graph unchanged.

Two nearby primary routes were screened but not selected. ICLR-2024 SGN is a
newer input-conditioned Gaussian regression model, but its official repository
does not publish a reusable license. ICLR-2024 Label Wave changes checkpoint
selection rather than the learned class-boundary mechanism and therefore does
not answer the current precision question. Neither may be silently substituted
after seeing A0 results.

## Locked Inputs And Provenance

- Repository pre-protocol HEAD/upstream:
  `38dce3b6af5eda64437e538576ef7a6a85c200c1`.
- Data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Declaration CSV SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Current-best command/history SHAs:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
- Domain NBDT A0 closure SHA-256:
  `d1747782630eaa57f38239025e4925f1b1d393562bc65028e7b5dc3231a9dd3b`.

Reuse only `runs/yolof_cidt_fold0_trainonly_20260716`:

- fit: 7,372 rows, 6,452 source stems, class counts
  `[1561,432,1527,2017,1835]`;
- holdout: 1,843 rows, 1,612 source stems, class counts
  `[380,109,393,503,458]`;
- source overlap: zero;
- fit/holdout sample-index SHAs:
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d` /
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`;
- fold data/summary SHAs:
  `4195ba89995d4eee483cae31626717381df910f56657086e7116322ca971388e` /
  `2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763`.

Raw images and labels are read-only. The generated `test` path mirrors holdout
only for loader compatibility. Every training launcher must use a fit-only
YAML whose `train/val/test` entries all reference fit, and must pass
`SkipFinalTest`. Pair decisions use `last.pt`; fit-mirror metrics may not
select an epoch.

## Phase A: Full-Fit Gradient Compatibility Gate

Phase A is the only action initially authorized. It creates no checkpoint and
never constructs the 1,843-row holdout, official validation, or test loader.

1. Load the hash-locked keeper and run the normal metadata-aware deployment
   forward with the deterministic evaluation transform over all 7,372 fit
   rows. `return_attention` and trace logits are forbidden for behavior.
2. Require exact sample-index coverage/order hashes and replay the known flat
   fit result within `1e-6`: macro/class-1 F1 `0.937980/0.809204`, class-1
   precision/recall `0.690671/0.976852`, class-1 TP/FP `422/189`, and 185
   restricted `{0,2,4}->1` false positives.
3. Freeze every keeper tensor. Initialize `u:[7372,1]` and `v:[7372,5]` from
   `N(0,1e-8)` with seed 42. Replay ten natural-frequency passes, batch 128,
   with deterministic epoch shuffles, official `L_SOP`, independent SGD
   groups `lr_u=0.1`, `lr_v=1.0`, zero momentum, and zero weight decay. These
   are the paper's real-noise Clothing1M values. No consistency or balance
   regularizer is allowed.
4. Preserve standard CE gradients with `u=v=0` as the comparator. After the
   tenth pass, independently recompute every row's SOP probability, CE/MSE,
   noise energy, and gradient with respect to the original network logits.

Before the fit scan, synthetic and random-tensor checks must match the locked
official AST and an independently written equation for outputs, losses,
`u/v/z` gradients, one optimizer step, and finite differences within `1e-6`.
FP32 is authoritative; BF16 must be finite with maximum probability error
`<=2e-3`. The auditor must refuse source/hash mismatch, overwrite, any path
outside fit, and any dirty tracked code implementation.

Phase A passes only if every condition below passes:

- all 7,372 rows replay exactly and every tensor/gradient is finite;
- at least `1%` of rows have nonzero effective `V` above `1e-6`, proving the
  mechanism did not remain at initialization, while no `u/v` entry leaves the
  paper's `[-1,1]` domain;
- for all 185 restricted class-1 false positives, `dL/dz_1` retains the
  positive corrective sign, its median SOP/CE magnitude ratio is at least
  `0.95`, and at most `5%` of rows lose more than half their corrective
  magnitude;
- for all 422 class-1 true positives, `dL/dz_1` retains the negative
  support sign, its median SOP/CE magnitude ratio is at least `0.95`, and at
  most `5%` of rows lose more than half their support magnitude;
- no class-1 false-positive row is clamped at `p_1=1e-4` in a way that makes
  its corrective logit gradient exactly zero;
- restricted-FP mean effective noise energy is no greater than the mean over
  all other raw errors. Otherwise SOP is preferentially explaining away the
  project's precision failures rather than making the hybrid learn them;
- the final corrected distributions are finite, sum to one within `1e-6`,
  and do not collapse to one class.

These gates operate on class decisions and logit gradients, so background
localization cannot satisfy them. The keeper has previously seen these fit
rows; consequently a Phase-A pass is only a mechanism-compatibility result,
never a generalization or promotion result. Any material failure closes A0
before trainer integration.

## Conditional Default-Off Integration

Only a complete Phase-A pass authorizes this section.

Add `classification_loss=sparse_overparameterization` as a default-off
training-only criterion. It must:

- require stable contiguous `sample_index` values and an exact dataset-index
  hash before epoch 1;
- keep `u/v` in FP32 outside the deployment model state dictionary;
- use the exact official equation above, independent SGD parameter groups
  `0.1/1.0`, no momentum/weight decay, and no SOP+/UDA/balance term;
- support AMP by unscaling and stepping both the model and noise optimizers
  exactly once per accumulated optimizer step;
- preserve `u/v`, their optimizer, occurrence hash, and criterion metadata in
  `last.pt` so resume is exact;
- exclude `u/v`, their optimizer, and every SOP tensor from evaluation, ONNX,
  TensorRT, video, and XAI forward graphs;
- report CE, MSE, active-noise fraction, classwise noise energy, saturation,
  class-1 TP/restricted-FP gradient retention, and sample-occurrence hashes.

Default-off model construction, logits, traces, parameter names, checkpoint
loading, export, and current-best wrappers must remain bit-exact.

## Conditional Engineering Preflight

Before an image epoch, a committed and pushed integration must pass:

1. all primary-source, protocol, fold, and protected hashes;
2. official-source and independent equation/gradient/finite-difference replay;
3. exact accumulation semantics and one `u/v` update per model optimizer step;
4. exact sample-index mapping under shuffle, worker loading, and resume;
5. deterministic ten-step synthetic asymmetric-noise recovery with no clean
   class-1 TP loss and fewer `{0,2,4}->1` errors than matched CE;
6. common model state/RNG and identical control/candidate sample order;
7. FP32/BF16 finiteness, stable loss scale, and no `u/v` saturation;
8. default-off and deployment parity, plus ONNX graph absence of SOP tensors;
9. candidate training runtime `<=1.20x` control and peak VRAM `<=1.10x`
   control and `<=7.5 GiB`;
10. fail-closed refusal of holdout/validation/test leakage, unpushed HEAD,
    dirty tracked scope, overwrite, or pair execution without permission.

Any material failure closes A0. A correction after complete artifacts requires
a preserved failure manifest, a new directory, and byte-identical replay of
unaffected outputs.

## Conditional Sole Ten-Epoch Train-Only Pair

Only complete Phase-A and engineering passes authorize one random-initialized
seed-42 pair on the exact `7372/1843` source-disjoint fold. Neither role may
resume or use pretrained weights. Both use the same current scratch hybrid,
ordered samples, transforms, optimizer/scheduler steps, auxiliary losses, EMA,
and natural-frequency occurrence stream.

Shared recipe: image 256, current conv stem, `256/8/8` embedding/depth/heads,
four registers, color/edge branch tokens, detail enhancement, bbox/CNN/fine-
grained fusion, pruning at blocks `2,5` with keep rates `0.85,0.65`, batch 32,
accumulation 2, AdamW `2.5e-4`, weight decay `0.05`, warmup 1, cosine horizon
10, minimum LR `1e-6`, clip `0.7`, EMA `0.995`, and ten complete epochs.

The control uses the current `ldam_focal` main classification loss. The
candidate replaces only that main term with locked `L_SOP`; metric-learning,
pairwise-margin, transforms, and every other enabled term remain identical.
Attention-view, teacher caches, distillation, sample/target manifests, paired
views, MixUp/CutMix/copy-paste, thresholds, TTA, and final test are disabled.

Evaluate both hash-locked `last.pt` checkpoints through the standard BF16
deployment forward on all 1,843 holdout rows under clean, dim
(`0.70/0.90` brightness/contrast), bright (`1.25/1.10`), and low contrast
(`1.00/0.65`). Holdout may first be loaded only after both last checkpoints
and occurrence hashes are fixed.

The candidate passes only if every gate below passes:

- clean macro F1 delta `>=0.000`;
- clean class-1 F1 delta `>=+0.015` and precision delta `>=+0.025`;
- class-1 TP is no more than two below control and recall delta is no worse
  than `-0.010`;
- restricted `{0,2,4}->1` false positives fall by at least four and `10%`;
- corrections exceed harms and no other class loses more than `0.015` F1;
- mean class-1 precision across all four conditions improves by `>=0.015`,
  with no condition losing more than `0.010` class-1 F1 or `0.015` macro F1;
- candidate fit noise remains sparse and nonsaturated, and its restricted-FP
  gradient-retention median remains `>=0.95` throughout training;
- object suppression changes class-1 margin more than matched far-background
  suppression, without increasing restricted FP under either perturbation;
- standard-forward logits and predictions are reproduced exactly by the XAI
  batch path, and hash-locked stem/block-2/block-5 Grad-CAM, rollout,
  perturbation, transition, tiny/edge, and class-1 TP/FP visual review passes.

A pair failure closes SOP loss, buffer-learning-rate, initialization,
regularizer, epoch, sampler, loss-combination, class-specific, and seed/fold
variants on the current representation. A pass authorizes a separately
precommitted official-validation comparison; it does not update current-best
files by itself.

## Stop Rules

- No raw-data modification, official validation/test access, pretrained
  teacher, ensemble, calibration, threshold, or class-1 oversampling.
- No trainer implementation or epoch unless the preceding gate is complete.
- No `u/v` shape, learning-rate, initialization, clamp/projection, loss,
  consistency, balance, warmup, duration, fold, or seed sweep.
- Do not rescue a failed SOP result with SGN, Label Wave, ELR, Cleanlab,
  co-teaching, or another closed noisy-label method in the same loop.
- Preserve summaries, manifests, hashes, gradients, and XAI for rejection.
  Remove only reproducible bulky payloads through a verified cleanup manifest.
- Current-best command/history files remain at three revisions/two updates
  unless a later locked official-validation comparison replaces the keeper.
