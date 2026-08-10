# TRKH 5-Class Quaternion Color-Rotation A0 Closure - 2026-07-21

## Decision

Reject the prospectively locked central-bbox `64x64`, three-layer gray-axis
quaternion color branch on the current keeper. It does not expose selective
class-1 precision information beyond keeper log probabilities, an ordinary
real CNN, red-axis rotation, or channel-dephasing placebos. Do not integrate
it into TRKH and do not run an image-model smoke, probe, validation, test, or
full train from this result.

## Locked Scope

- Protocol:
  `TRKH_5CLASS_QUATERNION_COLOR_ROTATION_A0_PROTOCOL_20260721.md`, SHA-256
  `87d59f49ec9f06ac6ac549eb932626e189c19f58bfb29b4fd5112e8df2db7fa8`.
- Infrastructure commit: `0cb10ed28bf7619fb07a4a684a014fa106fdf3b9`.
- Visual-status reconciliation commit:
  `6e55dc6`; it only mirrors the already locked top-level manual `fail` into
  `xai.manual_review` and records the previous summary hash. It changes no
  score, threshold, action, gate, model state, or scientific conclusion.
- Cohort: exactly 750 train-only CIDT clean rows, comprising 528 true class-1
  keeper predictions and 222 restricted `0/2/4 -> 1` false positives across
  five source-disjoint folds.
- No validation or test path, metric, prediction, label, or pixel was opened.
  No raw dataset file was modified.

## Engineering And Execution

- The independent NumPy/FP64 Rodrigues oracle agrees with the explicit
  gray-axis matrix within `6.66e-16`; FP32/BF16 errors are
  `1.13e-7/0.007873`.
- Gray-axis projection, grayscale invariance, cyclic RGB symmetry,
  multi-group accumulation, singleton batches, invalid-input rejection,
  finite-difference gradients, and exact static materialization all pass.
- The real control has 3,898 parameters versus 4,006 for the quaternion
  candidate, a `2.696%` difference within the locked 5% bound.
- Every declared parameter receives a finite nonzero first-step gradient and
  changes. Matched quaternion trunks begin bit-exact, and all roles consume
  the same fold occurrence order.
- The formal loader uses requested/effective workers `4/4`, pin memory, and
  persistent workers. Peak CUDA allocation is `0.18775 GiB`.
- Static candidate versus real-CNN runtime is `0.63449/0.35279 ms` per
  batch-64 iteration, ratio `1.79850`, below the `2.25x` gate.
- Materialized logits are exact; ONNX error is `2.38419e-7`. The deployment
  artifact contains ordinary static `Conv2d` operators only.
- Focused tests pass `18/18`; full pytest passes `1721/1721` after the
  reconciliation guard was added.

## Clean OOF Results

| Role | AUROC | TP retention | FP rejection | Corrections | Harms |
|---|---:|---:|---:|---:|---:|
| keeper log-probability | 0.808635 | 0.965909 | 0.229730 | 51 | 18 |
| RGB grid covariance + base | 0.752483 | 0.948864 | 0.144144 | 32 | 27 |
| matched real CNN + base | 0.789269 | 0.971591 | 0.211712 | 47 | 15 |
| red-axis quaternion + base | 0.790549 | 0.964015 | 0.225225 | 50 | 19 |
| trained channel-dephased quaternion + base | 0.791573 | 0.973485 | 0.202703 | 45 | 14 |
| **gray-axis quaternion + base** | **0.794414** | **0.971591** | **0.207207** | **46** | **15** |
| gray-axis quaternion only | 0.559378 | 0.962121 | 0.063063 | 14 | 20 |
| gray-axis seed repeat | 0.810913 | 0.967803 | 0.229730 | 51 | 17 |
| same-weight channel dephasing | 0.792878 | 0.973485 | 0.202703 | 45 | 14 |
| same-weight theta zero | 0.789073 | 0.975379 | 0.229730 | 51 | 13 |

The candidate fails 14 clean checks. Its AUROC is `0.014222` below base and
only `0.005144/0.003865/0.002841/0.001536/0.005341` above the real CNN,
red-axis, trained dephasing, same-weight dephasing, and theta-zero roles. These
deltas are far below the prospective margins.

The action rule also fails: candidate FP rejection is `20.72%`, below the
required 25%, and it rejects five fewer FP than the base role. It rejects only
one more FP than trained dephasing while breaking one more TP. It beats the
real CNN in only two of five folds, although it beats covariance in five,
red-axis in four, and trained dephasing in five.

## Mechanism Diagnosis

- The quaternion-only branch has AUROC `0.559378`, so the image branch does
  not independently carry useful class-1 precision information. Most usable
  discrimination remains in the five keeper log probabilities.
- Candidate effective rank is only `5.2692` across 48 pooled channels and at
  least one channel has exactly zero variance. Learned scale and angle tensors
  themselves remain finite and variable, so collapse occurs in activated
  pooled features rather than in parameter initialization.
- The seed repeat reaches AUROC `0.810913`, `0.016499` above the candidate but
  still near the base and below every required candidate margin. This is
  seed-sensitive weak fitting, not evidence for choosing the favorable seed.
- Same-weight dephasing changes AUROC by only `-0.001536`; theta-zero changes
  it by only `-0.005341`. Therefore aligned chromatic rotation and learned
  angles are not causal selective signals on this cohort.
- Robustness is correctly skipped because the conjunctive clean gate fails.

## XAI Review

The automatic XAI checks pass numerically and candidate mean border mass is
`0.306891`, below 0.45. Manual review fails. Candidate maps form dense
high-frequency grids across the full fruit surface and broad silhouette.
Aligned, same-weight dephased, and theta-zero maps remain visually similar;
none consistently isolates lesions, ripeness boundaries, or peel regions that
separate true class 1 from restricted false positives. Automatic spatial
safety therefore cannot establish the intended mechanism.

## Replay And Artifacts

- Final summary SHA-256:
  `23554b3d9f6a84343b9dbc0b33b6f9ed7d02c800196e75db44037fd143d4d6bb`.
- Final manifest SHA-256:
  `ff19434c085708ea775368518448fbc6aeb9c4c6218b379bc0354bc6cc22344a`.
- OOF predictions SHA-256:
  `d9b29a5b8284ba1734612f3b0bc0017ead3227d41bc5d70a73a9b381f0e3e2c2`.
- XAI sheet SHA-256:
  `f85ff880e7de08e4dd5fcfddc22c87d96b5215794c10e9167687ddb2b86e57bb`.
- Static ONNX SHA-256:
  `c1c42f819595d41296e0c765d02d1ec63a025042db51a57061e868cf7334465f`.
- Independent replay preserves row order and exact actions; maximum score
  difference is `8.11855e-8`, below the locked `1e-7` tolerance.
- The temporary 8.79 MiB RGB cache is deleted after geometry, states, XAI,
  and hashes are persisted. The complete retained formal is about 4.0 MiB.
- Final read-only retention passes over 810 run directories and 51 valid
  object-schema compaction manifests. All 233 expected-absent originals remain
  absent, no file is deleted, `blockers=[]`, free space is `100.822 GiB`, and
  retention summary SHA-256 is
  `1e04726b7b751711da78b859eac9baa4c88325b880ff842f50b9f886dfcc00b5`.

## Closed Boundary

Close this exact central-80%-bbox `64x64` crop, three-layer widths/kernels,
gray-axis quaternion equation, red-axis/dephasing/theta-zero controls,
20-epoch AdamW schedule, seed/fold, and 97%-TP threshold family on the current
keeper. Do not sweep crop inset/size, quaternion width, axis, angle range,
pooling, LR, weight decay, epochs, seed, or threshold. Do not fuse quaternion
features with closed covariance, color-equivariant, Gabor, topology, or
attention families.

Current-best full-train commands and history remain unchanged. The A0 failure
does not authorize production integration, smoke, probe, validation/test, or
full training.
