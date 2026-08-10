# TRKH 5-Class Domain NBDT A0 Closure (2026-07-17)

Status: closed before training. No source-disjoint holdout, official
validation, test, full train, or current-best command was opened or changed.

## Decision

The fixed domain NBDT route `((0,1),(2,3))|4` failed its full 7,372-row
fit-partition compatibility gate. The deterministic soft-path map predicted
class 4 for every row. The prospectively locked five-epoch control/candidate
pair is therefore forbidden and was not run.

The development-only runtime integration was removed rather than leaving a
rejected training option in the shared trainer. The prospective protocol is
retained as the exact record of the hypothesis and stop rule.

## Primary Sources

Authority was the accepted paper and author repository, not the supplied
secondary research reports:

- Wan et al., "NBDT: Neural-Backed Decision Tree," ICLR 2021 Poster:
  https://openreview.net/forum?id=mCLVeEpplNE
- Accepted paper PDF: https://openreview.net/pdf?id=mCLVeEpplNE
- Official MIT repository:
  https://github.com/alvinwan/neural-backed-decision-trees
- Locked official commit/tree:
  `a7a2ee6f735bbc1b3d8c7c4f9ecdd02c6a75fc1e` /
  `681fd2ca86d6ad238f2586fd4d4d6357d809b8df`

The local AST replay matched the official descendant-logit means, node
softmaxes, path multiplication, probability-vector cross entropy, gradients,
and linear weight interpolation. Independent equations, gradients,
Hessian-vector products, finite differences, FP32/BF16 probability replay,
trace-only deployment parity, ONNX Runtime, and one-batch resource checks also
passed. The rejection is behavioral, not an equation-copy failure.

## Locked Data Scope

- Fit partition only: 7,372 object rows from 6,452 source stems.
- Fit class counts: `[1561,432,1527,2017,1835]`.
- Fit sample-index SHA-256:
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d`.
- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Evaluation transform and normal metadata-aware deployment forward were used.
- The source-disjoint 1,843-row holdout was not constructed.
- Official `yolo_f/val`, `yolo_f/test`, and all final-test data were not used.
- The scan used `num_workers=0` because it was invoked from a Windows stdin
  development process; this changes throughput, not row order or predictions.

## Full-Fit Result

Flat keeper:

- accuracy / macro F1: `0.960662 / 0.937980`;
- class-1 precision / recall / F1: `0.690671 / 0.976852 / 0.809204`;
- class-1 TP / FP: `422 / 189`;
- restricted class-1 FP from true classes `0,2,4`: `185`.

Domain soft-path inference:

- accuracy / macro F1: `0.248915 / 0.079722`;
- class-1 precision / recall / F1: `0 / 0 / 0`;
- class-1 TP / FP: `0 / 0`;
- prediction histogram: `[0,0,0,0,7372]`;
- class-1 precision delta: `-0.690671`;
- class-1 TP delta: `-422`.

Flat confusion matrix, rows=true and columns=predicted:

```text
[[1416, 135,    5,    0,    5],
 [   9, 422,    0,    0,    1],
 [   4,  41, 1455,   22,    5],
 [   0,   4,   33, 1978,    2],
 [  13,   9,    2,    0, 1811]]
```

Soft-path confusion matrix:

```text
[[0, 0, 0, 0, 1561],
 [0, 0, 0, 0,  432],
 [0, 0, 0, 0, 1527],
 [0, 0, 0, 0, 2017],
 [0, 0, 0, 0, 1835]]
```

Node probabilities were finite and nonconstant, so a NaN or dead-node bug is
not the explanation. Mean per-child variances were:

```text
root      0.0066008433
maturity  0.0054664630
unripe    0.0080558788
ripe      0.0111279823
```

## Mechanism Diagnosis

The fixed tree gives class 4 a one-edge path but classes 0-3 three-edge paths:

```text
p0 = r0*m0*u0   p1 = r0*m0*u1
p2 = r0*m1*v0   p3 = r0*m1*v1
p4 = r1
```

At equal leaf logits, every binary node is `[0.5,0.5]`. Therefore
`p0..p3=0.125` while `p4=0.5`. The official multiplication is a normalized
tree distribution, but the current flat logits were not trained to calibrate
the singleton class-4 root branch against deeper singleton leaves. The unequal
depth makes the incompatibility decisive before tree supervision can be
considered safe for the protocol's primary `candidate_soft` deployment mode.

This does not invalidate NBDT generally. It rejects this prospectively fixed
domain tree and soft-inference adaptation for the current TRKH representation.
A balanced/dummy topology, induced hierarchy, path-length correction, proper
path NLL, summed node CE, flat-only inference, different schedule, or different
weight would be a new method selected after observing this failure. Those
nearby variants are prohibited by the A0 no-repeat rule.

## Closure Rule

- Do not run the five-epoch A0 pair.
- Do not sweep NBDT topology, path depth, loss, weight, schedule, inference
  transform, threshold, seed, fold, or combinations on this representation.
- Do not revive the previously rejected semantic-attribute/hierarchical loss
  under another name.
- Preserve the protocol and this closure document; remove the uncommitted
  runtime integration and temporary ONNX/dev artifacts.
- Keep the current-best command/history hashes and selected checkpoint exact.
- The next route must come from a distinct accepted primary source with
  official licensed code and a precision-first pre-training gate.
