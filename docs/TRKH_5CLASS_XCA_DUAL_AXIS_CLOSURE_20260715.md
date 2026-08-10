# TRKH 5-Class Shared-Projection XCA Dual-Axis Closure - 2026-07-15

## Decision

**Rejected at train-only Stage A.** The locked Stage-B keeper-initialized
validation smoke is not authorized. No validation or test loader was
constructed, no full train was run, and the current-best command packet remains
unchanged.

The XCA branch is structurally correct and measurably active, but it adds no
differential class-1 decision signal on the precommitted source-disjoint
train-only holdout. Increasing its scale or sweeping layers, learning rate,
seed, or budget after seeing this result would violate the protocol and would
not be supported by the observed behavior.

## Locked Experiment

- Protocol:
  `docs/TRKH_5CLASS_XCA_DUAL_AXIS_READINESS_PROTOCOL_20260715.md`
- Protocol SHA-256:
  `5a7a46e28772002089b1daadbfa8e2e468ca16c53559fa669382b401c6c5590f`
- Official XCiT commit:
  `82f5291f412604970c39a912586e008ec009cdca`
- Reviewed `xcit.py` SHA-256:
  `3e2d4be847dc8c88d98fe0f965bca3119aeeb588975eff0bc62fdfb20e8b3e9e`
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Dataset scope: `yolo_f/train=9215` only. Fold 0 contains `7372` fit rows
  and `1843` holdout rows from `6452/1612` source groups with zero source
  overlap. Both adaptations consumed the same first `20 x 32 = 640` fit rows.
- Train-order SHA-256:
  `38a893d6eb2a5f4e3fcb1ba2b9115b0a064ce24561ae7284386d82eaf9247f65`
- Locked candidate: patch-only shared-qkv/shared-projection XCA in Transformer
  layers `2,5`, separate LayerNorm and per-head temperature, fixed residual
  scale `0.10`, and no LPI or XCA dropout.

## Structural and Mechanism Evidence

- Candidate adds exactly `1,040` parameters. Loading the keeper produces the
  exact six expected XCA missing keys, no unexpected keys, and every existing
  tensor remains bit-identical.
- Constructor and train-forward RNG states match the control. All FP32 and
  BF16 XCA/shared-projection gradient families and input gradients are finite
  and nonzero. All six new parameter tensors move during adaptation.
- Spatial MHSA remains active in all eight blocks, spatial attention remains
  available for pruning/XAI, and pruning remains after layers `2,5`.
- Layer-2/layer-5 channel maps are `[1,8,32,32]`, row-normalized, and finite.
  Their normalized entropies are `0.967196/0.971133`, diagonal masses are
  `0.027247/0.028291`, and q/k token-norm errors are below `1.2e-7`.
- Removing layer 2 or 5 changes logits by `0.00188267/0.000900626`, proving
  both branches are connected. Their residual-norm ratios are only
  `0.002686/0.002146`; combined ratio `0.00239789` is below the precommitted
  `0.005` floor.
- Runtime ratio is `0.969343x` control and peak VRAM is `2.602996 GiB`, so
  resources are not the rejection reason.

These checks establish a live, inexpensive channel mixer. They do not establish
class-selective usefulness.

## Train-Only Decision Evidence

Before adaptation, control and candidate make exactly the same `1843` holdout
decisions. Both have macro/class-1 F1 `0.948369/0.845850` and class-1
precision/recall `0.743056/0.981651`; the candidate changes probabilities
slightly but changes zero hard decisions.

After the matched 20-batch adaptation:

| Metric | Control | Candidate | Delta |
|---|---:|---:|---:|
| Macro F1 | `0.827659` | `0.827641` | `-0.00001795` |
| Class-1 F1 | `0.452055` | `0.452055` | `0.000000` |
| Class-1 precision | `0.891892` | `0.891892` | `0.000000` |
| Class-1 recall | `0.302752` | `0.302752` | `0.000000` |

The candidate changes only two decisions: one correction and one harm. It
rescues/breaks zero class-1 FN/TP, removes zero restricted `0/2/4 -> 1` false
positives, creates zero class-1 precision gain, and has maximum nonfocus F1 drop
`0.000914`. The large class-1 recall loss after adaptation is shared by both
models; the causal candidate-minus-control result remains exactly zero for all
class-1 metrics.

## Failed Gates and Export Caveat

The following six locked checks failed:

1. `combined_residual_ratio_in_range`
2. `onnx_error_lte_1e5`
3. `class1_f1_delta_gte_0p002`
4. `class1_precision_delta_gte_0p005`
5. `restricted_focus_fp_reduction_gte_2`
6. `corrections_gt_harms`

The static-batch-2 full-graph candidate ONNX export completed, but ONNX Runtime
maximum logit error was `0.496950`. The earlier dynamic-batch attempt exposed an
existing keeper reshape limitation, and this Stage A did not run a matched
control ONNX export. Therefore the export mismatch is recorded as an unresolved
deployment blocker, not attributed solely to XCA. The independent behavioral
gate failures already reject the method without relying on this attribution.

## Retained Evidence

Evidence root: `runs/audit_xca_dual_axis_readiness_20260715_stage_a`

- `summary.json`: SHA-256
  `5429a64f254dd045b0670bccbeb807c72da94b26853633fa01beaf150a49f7b0`
- `holdout_predictions.csv`: SHA-256
  `7f2a5a7111b7ce7c713e5e77324ff038e930c07effa34c27fd433b56e4553e71`
- `artifact_manifest.json`: SHA-256
  `0a9f6b0f876e9bfb5aef72c6ee3c9a41e0c8ac4e5fd59de9994e3e029060d63c`
- `report.md`: SHA-256
  `6445256c95f802259350c7f6aa07d00bdf3ef2c38793217d6697ce0bb04a5b33`
- `xca_candidate.onnx`: SHA-256
  `582681c1e4bb49145206d12232d1947fceb05ccfc6a236212879a1d963c59186`

The 31 MB ONNX file remains with its manifest because it is the exact failed
export evidence and disk pressure is not material. No rejected training
checkpoint was created by Stage A.

## Closure Rule

Do not run Stage B, validation, test, five epochs, a probe, or a full train for
this route. Do not sweep XCA layer placement, residual scale, temperature,
head count, LPI, dropout, learning rate, seed, fold, or adaptation budget on the
current keeper. A future channel-interaction route must be mechanistically
distinct and must prove class-1 false-positive reduction plus true-positive
retention on train-only evidence before validation access.

Current-best command SHA-256 remains
`36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`.
