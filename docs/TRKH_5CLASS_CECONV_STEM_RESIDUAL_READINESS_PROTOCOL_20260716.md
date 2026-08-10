# TRKH 5-Class CEConv Stem Residual Readiness Protocol - 2026-07-16

## Status

Precommitted before implementation. This document locks one exact train-only
Stage-A experiment. It authorizes neither validation/test access nor a shared
trainer, probe, full train, or parameter sweep.

## Research Question

Can an early color-equivariant residual path remove restricted class-1 false
positives caused by illumination/hue variation while preserving every current
class-1 true positive and the native color-specific TRKH path?

Current XAI repeatedly shows that far background is weakly causal, while fruit
surface color, glare, pale-green/yellow boundaries, and illumination are much
more influential. A useful mechanism must therefore add selective color
robustness, not globally suppress class 1 or discard color, because color is
also genuine maturity evidence.

## Primary-Source Lock

- Paper: Lengyel et al., "Color Equivariant Convolutional Networks," NeurIPS
  2023.
- Proceedings PDF:
  `https://proceedings.neurips.cc/paper_files/paper/2023/file/5f173562e7662b14fb5c5695f225ea46-Paper-Conference.pdf`
- Official repository: `https://github.com/Attila94/CEConv`
- Locked local paper:
  `D:/DataAI/external_sources/papers/ceconv_neurips2023.pdf`
- Paper SHA-256:
  `1c41da4c99ffc46ef0c318e7145ab749efb14b6ef63927bba9a8071937ad8f83`
- Locked official repository commit:
  `8f46c78c3a7cf91ad905d0255b756d13cd1e0c94`
- Official `ceconv/ceconv2d.py` SHA-256:
  `c9a730691e3186179b64b40f966dbd60d21add3040b0911786408cefc75e6469`
- Official `ceconv/pooling.py` SHA-256:
  `882425da67ed9700f8bd2c12d2e708e8d8ae857a816f39cebae389d68ab9351c`
- Official MIT license SHA-256:
  `4d846d51cf0de71d223d0ba9ef4a2ddf41088e0b0eb92af4a6d17ca98cc3b6df`

The official input CEConv rotates learned RGB filters around the RGB diagonal,
constructs `N` group responses, uses `BatchNorm3d` while the group dimension is
present, and removes that dimension with group-coset max pooling. The paper
reports that one or two early equivariant stages can preserve later
color-specific capacity. Its experiments use ResNets and 200 epochs on an A40;
they do not establish that this bounded scratch-TRKH adapter will work.

## Local No-Repeat Boundary

This route is not a reopening of fixed color transforms, color-statistic
fusion, GrayEdge/CIConv normalization, MixStyle, input desaturation, frequency
experts, texture descriptors, frozen readouts, post-hoc thresholds, or
gradient-combination families. It adds a trainable equation-traceable early
representation path and keeps the complete native stem path.

This is not claimed to reproduce the paper's full CE-ResNet. Failure closes
only this exact zero-init first-block residual adapter. It does not establish
that every full CEConv architecture fails.

## Exact TRKH Adaptation

Method identifier: `ceconv_stem_residual_a0`.

Wrap only `stem.blocks.0` of the locked keeper. The original block remains
unchanged and frozen. For input `x` and original block `B`, return

`B(x) + tanh(g) * P(A(T(W), x))`,

where:

1. `W` is a trainable clone of the keeper's first `3x3`, `3 -> 32` convolution.
2. Candidate `T` is the official input-filter equation with exactly `N=3`
   fixed powers of the `120 degree` RGB-diagonal rotation matrix.
3. Matched control has the same schema and computation but repeats the identity
   matrix three times. No candidate/control parameter differs at construction.
4. The transformed convolution gives `[B,32,3,H,W]`; apply
   `BatchNorm3d(32, eps=1e-5, momentum=0.1)`, GELU, the original block's
   spatial max-pool independently per group, and official group-coset max.
5. `g` is one trainable scalar per output channel, initialized exactly to zero.
   Therefore candidate and control logits are bit-exact to the unmodified
   keeper before adaptation. The native path remains available after training.
6. Add no classifier, auxiliary head, threshold, attention module, dropout,
   pretrained weight, augmentation, external data, or raw-data write.
7. Freeze the entire keeper, including the native first block and classifier.
   Train only branch `W`, branch BN affine values, and `g`; BN running statistics
   may update. Candidate and control must have identical trainable counts.

The expected trainable count is `32*3*3*3 + 32 + 32 + 32 = 960`: branch filter,
BN weight/bias, and channel gate. BN running statistics are buffers.

## Locked Inputs

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
- Locked raw/margin comparator predictions:
  `runs/audit_class1_boundary_cagrad_readiness_20260715/predictions_all_conditions.csv`
- Comparator SHA-256:
  `92dc296345e34d3f0ac026db2dcb2f982e40af79e1179136dfbb17f1cc2b9d9f`
- Comparator summary/manifest SHA-256:
  `dab24c6dd7dcf873e523915aadbdd1f8ef5337801278d9ffc55ce8c867a37750` /
  `7f1f2a18846cd54fb7a7eafe2964a17681b99d009a70aadfaa09be54d43d1ced`.

The declaration contains exactly `9215` ordered `yolo_f/train` object rows.
Fold 0 is the source-disjoint `1843`-row holdout; the other `7372` rows are fit
only. The locked fit cohort is all `432` class-1 rows plus exactly `186` raw
restricted `0/2/4 -> 1` false positives. Validation and test paths are forbidden.

## Stage A Execution

- CUDA deterministic algorithms, TF32 off, seed/fold `42/0`.
- Batch size `32`; every batch has `16` class-1 and `16` restricted hard-negative
  rows. Cycle independent seeded permutations within each side as needed.
- Budget: exactly `60` optimizer steps / `1920` rows. This is a step budget,
  not an epoch claim and must not be extended after observing results.
- Deterministic keeper evaluation transform; no stochastic augmentation.
- AdamW, learning rate `5e-4`, weight decay `0.01`.
- Standard five-class hard-label cross entropy only.
- Candidate and control use the exact same row order, initial parameter values,
  optimizer settings, and RNG checkpoints.
- Selection is the single final step-60 state. No checkpoint selection, LR,
  step, rotation, pooling, gate, loss, seed, fold, or cohort sweep.

## Structural and Equation Gates

All must pass:

1. Verify every local/official SHA, official commit/license, and a clean official
   worktree; tracked TRKH files must be clean before the formal run.
2. Preflight constructs no output directory and accesses no validation/test
   path, prediction, dataset, or loader.
3. The official rotation matrix is orthogonal, preserves the RGB diagonal,
   has determinant `+1`, and its third power is identity within `1e-6`.
4. Candidate transformed filters match a direct official-equation replay within
   `1e-7`; group zero equals `W` exactly. Candidate and control group stacks
   must differ for a non-degenerate RGB filter.
5. Zero gates give exact native stem output and exact keeper logits in FP32.
6. Candidate/control initial model state, trainable tensors, and parameter count
   match; only fixed rotation buffers differ. The frozen keeper subset remains
   bit-exact after training.
7. Every branch filter/BN affine/gate gradient is finite and seen nonzero during
   adaptation. Every trainable tensor moves; BN running statistics update.
8. Final gates are finite, bounded by `abs(tanh(g)) <= 0.25`, nonzero in at least
   24/32 channels, and candidate/control outputs are noncollapsed.
9. Hue-cycle equivariance is checked directly before coset pooling; cyclically
   rotating input RGB by one group step must cyclically shift candidate group
   responses within FP32 tolerance. The identity control must fail this same
   distinction check.

## Decision Gates

Evaluate raw keeper, matched identity control, and CEConv candidate on the exact
ordered 1843-row train holdout. Candidate must satisfy all:

- versus raw: macro F1 `>= +0.001`, class-1 F1 `>= +0.003`, and class-1
  precision `>= +0.005`;
- versus matched control: macro F1 `>= 0`, class-1 F1 `>= +0.002`, and
  class-1 precision `>= +0.003`;
- class-1 recall delta versus raw and control each `>= -0.005`;
- zero raw class-1 TP breaks after accounting for FN rescues;
- remove at least two restricted `0/2/4 -> 1` false positives versus raw and at
  least one versus control;
- corrections strictly exceed harms versus both raw and control;
- maximum nonfocus-class F1 drop versus raw `<=0.010`;
- no more than two new `3 -> 2` harms.

The raw comparator is mandatory. A candidate-placebo win cannot authorize code
if both trained variants damage the keeper.

## Illumination, XAI, Export, and Resource Gates

Evaluate the same ordered holdout under dim `(0.70,0.90)`, bright
`(1.25,1.10)`, and low contrast `(1.00,0.65)` brightness/contrast settings.

- Candidate-minus-control class-1 precision and macro F1 must be nonnegative in
  at least `2/3` conditions.
- Worst class-1 recall delta versus raw is `>=-0.020`; no condition has net
  class-1 TP loss greater than two.
- Aggregate restricted-FP removals exceed creations versus both raw and control.

The auditor must export changed-case contact sheets with source image, native
Grad-CAM, candidate/control Grad-CAM, three pre-coset group-energy maps, winner
map, gate values, and clean/dim/bright/low-contrast predictions. At least all
clean corrections, harms, and raw class-1 TP breaks must be represented. XAI is
a required diagnostic, not a substitute for decision gates.

Export an isolated adapter and one full metadata-aware candidate at static batch
1, ONNX opset 17. ONNX Runtime CPU maximum absolute error must be `<=1e-5` with
argmax match. Across three matched BF16 forward/backward repeats at batch 32,
candidate runtime must be `<=1.35x` the unwrapped keeper and peak allocated VRAM
must be `<=3.25 GiB`.

Any failed gate rejects Stage B. Still complete equation replay, decision CSVs,
XAI, export/resource diagnostics where technically possible, independent replay,
artifact manifest, retention, tests, journal, TODO, and skill closure.

## Stage B Boundary

Only a complete Stage-A pass permits implementation as a default-off shared
model option and one separately precommitted keeper-initialized validation
smoke. A Stage-A failure forbids validation, test, a nearby adapter variant,
probe/full train, and current-best command revision.

The current command file must remain at SHA-256
`36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`
unless a later independently reloaded candidate wins the locked validation gate.
