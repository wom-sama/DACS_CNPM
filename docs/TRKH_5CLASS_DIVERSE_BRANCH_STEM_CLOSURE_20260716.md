# TRKH 5-Class Diverse-Branch Stem A0 Closure - 2026-07-16

## Decision

Reject the exact three-block Diverse Branch Block (DBB) stem before training.
The committed formal preflight passed 53 of 58 checks but did not grant
`formal_pair_permission`. Therefore the five-epoch control/candidate pair,
behavior audit, XAI, Stage B, official validation, test, full training, and
current-best command promotion are forbidden.

This is an engineering and resource rejection of the locked dense four-path
DBB implementation. It is not a behavior result and it does not establish
class-1 quality in either direction.

## Frozen Provenance

- Prospective protocol SHA-256:
  `578b208f2f50da69b3b3ac2113705750302ffc5b6f205fc20bd60133045d22ff`.
- Implementation commit, pushed before execution:
  `ed45a46ef95f66bbc1a2677fa1afc28d767e5333`.
- Accepted primary paper: Ding et al., CVPR 2021, "Diverse Branch Block:
  Building a Convolution as an Inception-Like Unit."
- Official Apache-2.0 commit/tree:
  `8d2b16b6aee45a33236b2d11685be6857f9ba929` /
  `b038d8e014664312f8979d0b7917466e70877b30`.
- Official implementation/transformation SHAs:
  `5f67f20081ffef0a9f7ad6f9f5cc1a7bb4e59902dc31c86a8e59c37fd477ccb0` /
  `1424673513c4883ffb2e88b24f81a41cb69569f2621aa18f76a8fe341c92b11f`.
- Exact source-disjoint fold: 7,372 fit rows and 1,843 holdout rows, with
  zero source overlap. Validation and test paths were not loaded.

All locked source, data, fold, checkpoint, command, and Git checks passed.
The user-supplied research reports were not used as equation or gate authority.

## Passed Engineering Evidence

- Independent BN fusion, branch addition, 1x1 padding, sequential fusion,
  average-pool conversion, and BN-aware border-padding replay had maximum
  error `7.152557e-7`.
- Random train/eval and real-fit eval replay against official code passed.
  All remapped gradients were finite and closely matched.
- Control/candidate construction preserved every non-stem tensor and the three
  origin conv/BN pairs bit-exactly. CPU/CUDA RNG states matched.
- All 12 paths were finite, nonzero, mutually distinct within each block, and
  received finite nonzero FP32 and BF16 gradients.
- True-FP32 conversion used CPU kernel composition and disabled TF32 during
  comparison. Candidate stem/logit errors were `6.675720e-6` and
  `1.564622e-7`, with zero decision mismatch.
- The deployed candidate and folded control each contained exactly three
  bias-enabled 3x3 stem convolutions, no DBB-only training operator, and the
  same 167,104 stem parameters.
- Static batch-1 ONNX export passed with maximum logit error
  `1.899898e-7` and matching argmax.
- Candidate training and inference remained finite and used 3.944 GiB peak
  allocated VRAM, below the absolute 7.5 GiB limit.

## Failed Immutable Gates

The following five checks failed; one failure was sufficient to deny training.

1. `official_random_real_train_eval_replay`: the real-fit train output error
   was `1.907349e-6`, above `1e-6`. Its input-gradient, parameter-gradient,
   equivalent-kernel, and bias errors were respectively `1.091394e-11`,
   `3.576279e-7`, `0`, and `1.043081e-7`.
2. `deployment_parity_structure`: under real CUDA BF16, branch-structure versus
   deployed stem maximum error was `0.09375`, above `0.002`. Logit error was
   `0.001953125` with zero decision mismatch, but the locked stem-output gate
   still failed. This is expected from branchwise BF16 rounding versus one
   fused convolution and cannot be hidden by changing the metric after seeing
   the result.
3. `train_runtime_ratio`: median candidate/control train-step time was
   `0.296953/0.161301` seconds, or `1.840986x`, above `1.50x`.
4. `candidate_vram_ratio`: candidate/control peak allocated training VRAM was
   `3.943974/2.533129` GiB, or `1.556958x`, above `1.25x`.
5. `inference_runtime_ratio`: converted candidate/folded-control median
   inference time was `0.071763/0.066250` seconds, or `1.083225x`, above
   `1.05x`.

The candidate added 211,951 train-time parameters (`7,457,541` versus
`7,245,590`). The resource failures make the rejection independent of the two
strict numerical tolerances.

## Evidence And Cleanup

- Exact retained preflight summary:
  `runs/evidence_dbb_stem_preflight_rejected_20260716/preflight/summary.json`,
  SHA-256
  `a60983ffb1118816b186067ca4d7631d64bb573aea77c798d335875501b55c8e`.
- Exact retained artifact manifest SHA-256:
  `cd1e0fa6499bdd63f54f05deca6a82d4625913321ec4382b6c4ae99963416c37`.
- Compacted evidence manifest SHA-256:
  `16b266742cfff190a0c709cd3945526c81f80cb57ab952258a5201b45b5ff8d7`.
- Cleanup manifest SHA-256:
  `3372453024f08b3736aa8ed2a0a052eb6a0227c462dc31a42a3ec3f259bde722`.
- Compaction excluded the reproducible 30,419,262-byte ONNX and verified the
  original preflight directory was removed. No checkpoint existed.
- Full pytest before formal execution passed `1261/1261`. PowerShell parsing,
  Python compilation, focused tests, and locked hashes also passed.
- Read-only retention passed over 721 directories with `blockers=[]`; all 204
  compacted originals were absent as declared. Retention summary SHA-256 is
  `77cc4282bc2798ac10f070f728b6b6d932b04022769aeebd3c4adecef9ec381d`,
  with 71.100 GiB free.

## No-Repeat Boundary

Close this exact dense DBB replacement and every nearby sweep of branch subset,
internal width, BN gamma, block count, insertion position, activation, pooling,
optimizer, LR, weight decay, epoch count, fold, seed, precision threshold, or
combination with RepVGG, ACB, OREPA, FENet, or RepOptimizer. Do not weaken the
five failed gates or rerun the preflight on the current keeper.

The default-off implementation may remain as reproducible negative evidence and
as tested deployment infrastructure. A future structural-reparameterization
study requires a genuinely different primary-source mechanism and a new
prospective protocol.

Current-best command/history files remain unchanged at three revisions, two
updates, and zero keeper replacements.
