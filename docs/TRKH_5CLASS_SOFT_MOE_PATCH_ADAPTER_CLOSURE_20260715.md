# TRKH 5-Class Soft-MoE Patch Adapter Closure - 2026-07-15

## Decision

Close the exact `soft_moe_patch_adapter` route at train-only Stage A.
`stage_b_smoke_authorized=false`. Do not run validation, test, a longer
adaptation, a probe, a full train, or a nearby Soft-MoE sweep.

The adapter was live, noncollapsed, exportable, and inexpensive enough, but it
did not change one clean holdout decision. Learned routing signatures differed
between class-1 true positives and restricted false positives without turning
that distinction into a safer class boundary.

## Locked Scope

- Protocol:
  `docs/TRKH_5CLASS_SOFT_MOE_PATCH_ADAPTER_READINESS_PROTOCOL_20260715.md`
- Protocol SHA-256:
  `d46319582c8fba1c3f061e987b3f9bbdc4442d188c5f39bd20e0dfe1ca57110b`
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Dataset: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`
- Raw dataset modified: `false`
- Validation used: `false`
- Test used: `false`

Stage A constructed only `yolo_f/train=9215`. Source-disjoint fold 0 supplied
`1843` ordered holdout rows, folds 1-4 supplied `7372` fit rows, and the fixed-
router control and learned-router candidate consumed the same deterministic
`60 x 32 = 1920` fit rows.

## Implementation Result

The default-off adapter was inserted after the native attention/local residual
and before the dense FFN in Transformer layers `2,5`. Each layer used four
one-slot experts, a normalized linear router, `256->64->256` expert MLPs,
fixed residual scale `0.10`, and zero output projections for exact identity.
Prefix tokens bypassed the adapter and patch indices stayed traceable through
token pruning.

All exact checkpoint and mechanism checks passed after audit correction:

- exactly `266,754` added parameters and 36 expected state tensors;
- exact constructor/forward CPU and CUDA RNG parity;
- exact keeper state and initial logits;
- all eight native spatial-MHSA maps and pruning after layers `2,5` preserved;
- every required FP32/BF16 gradient finite and nonzero after warm adaptation;
- control changed only output projections; candidate moved every router,
  input-projection, and output-projection group;
- both layer and every expert ablation changed logits by at least `0.003906`.

## Audit Infrastructure Repairs

The first completed adaptation exposed a post-audit metadata bug: the FP32
gradient cohort sliced images from 32 to 5 but retained 32-row bbox/mask
metadata. The auditor now slices and validates every tensor metadata field;
the regression test and the formal retry pass.

The formal summary then reported a false equation failure because a BF16
`einsum` slot was compared with a separately recomposed FP32 slot. The
reference is now recomposed under the same autocast contract. Direct GPU
diagnostics report zero slot/residual/update error in FP32 and BF16, while
dispatch/combine normalization errors remain at about `1e-7`.

The immutable formal summary SHA is
`1a2fe8909e9e73539779fdbfee88f161ca9847bf5ee82a4a1629e92d78bd64ec`.
The hash-locked correction artifact SHA is
`a478b96cfb663fc2cef687ffe5aae87e5597e93be038a137e77288dac85855aa`.
It changes only the equation structural gate and preserves eight behavioral
failures, so the experiment outcome does not change.

## Train-Only Decision Failure

| Metric | Raw keeper | Fixed-router control | Soft-MoE candidate |
|---|---:|---:|---:|
| Macro F1 | `0.948369` | `0.948369` | `0.948369` |
| Class-1 F1 | `0.845850` | `0.845850` | `0.845850` |
| Class-1 precision | `0.743056` | `0.743056` | `0.743056` |
| Class-1 recall | `0.981651` | `0.981651` | `0.981651` |

Candidate-minus-raw and candidate-minus-control deltas were exactly zero for
all four metrics. The candidate changed zero clean argmax decisions, corrected
zero cases, harmed zero cases, removed zero of 37 restricted
`0/2/4 -> 1` false positives, rescued zero class-1 false negatives, and broke
zero class-1 true positives.

The branch was not numerically identical. Candidate-versus-raw probability L1
was nonzero on `1501/1843` rows, with mean `0.0005081` and maximum `0.0188007`;
candidate-versus-control mean/max L1 was `0.0004638/0.0186579`. The learned
effect was real but too small and not selectively aligned with clean errors.

## Routing XAI And Illumination

Routing did not collapse. Candidate expert masses were `0.210-0.316` in layer
2 and `0.227-0.276` in layer 5; normalized combine entropy was
`0.9153/0.9290`. Class-1 TP versus restricted-FP routing-signature L1 was
`0.02585/0.01025`, above the locked separation floor. Candidate-minus-control
residual bbox-mass deltas were only `-3.59e-5/+4.59e-6`; the learned router did
not create a materially different object/context allocation.

- Dim: one restricted FP correction, class-1 F1/precision
  `+0.002462/+0.005463` versus raw.
- Bright: one restricted FP correction, class-1 F1/precision
  `+0.002532/+0.003873` versus raw.
- Low contrast: one new nonfocus `3->2` harm, macro F1 `-0.000441`, and no
  class-1 change.

These isolated corruption changes do not satisfy the clean decision gate and
do not justify using illumination as an oracle or tuning the adapter scale.

## Export And Resources

- Static-batch-1 full ONNX passed with max logit error `2.31e-7` and matching
  argmax.
- Candidate/keeper median inference ratio was `1.122715x`.
- Peak candidate training allocation was `0.755037 GiB`.
- Control/candidate adaptation time was `194.29/194.49 s` with identical
  train order and forward RNG checkpoints.

Deployment and resource checks were not blockers. Clean class-selective
behavior closes the route.

## No-Repeat Rule

Do not sweep expert count, slot count, layer placement, hidden width, router
temperature/scale, residual scale, initialization, optimizer, LR, weight
decay, fold, seed, batch budget, loss, augmentation, or run length on this
keeper. Do not convert routing signatures or corruption-only changes into a
post-hoc class-1 filter.

Future conditional-computation work would need a genuinely different
supervision signal that directly ties expert specialization to clean
foreground surface/boundary errors while preserving class-1 TP. Merely making
this residual larger is not new evidence.

## Evidence And Cleanup

Original Stage-A payload hashes before compaction:

- summary: `1a2fe890...bd64ec`;
- correction: `a478b96c...855aa`;
- holdout predictions: `f2d34634...ec2d`;
- illumination predictions: `32c97f82...6b0a`;
- report: `903080f4...fd7`;
- artifact manifest: `35c00426...484a`;
- adapted extension: `145910f5...ea5a`;
- ONNX: `3cb03f2f...a7432f`.

Compaction retained eight verified nonbinary payloads under
`runs/evidence_soft_moe_patch_adapter_stage_a_rejected_20260715`; payload
manifest SHA is `75de0592...b0cd`. The reproducible extension and ONNX binaries
totaling `33,805,837` bytes were excluded. Source deletion was verified and
observed free space increased by `36,806,656` bytes.

Retention passed over `676` run directories with `blockers=[]`; summary SHA is
`7efd920f...f609`. Keeper, scratch complement, current command, and command-
history hashes remained unchanged. No full-train command revision is
justified.

## Engineering Verification

- Package and tests compileall passed.
- Focused Soft-MoE/integration tests passed `33/33`.
- Full pytest passed `1103/1103` in `76.14 s`.
- Soft-MoE readiness, V8, current-best pipeline, TensorRT export, and video
  PowerShell launchers parsed with zero errors.
- All five operational preflight checks passed and created no run/deploy
  directory. Readiness remained train-only and V8 exposed the Soft-MoE config.
- `BaoCao/` and both user-owned deep-research reports remained untouched and
  unstaged.
