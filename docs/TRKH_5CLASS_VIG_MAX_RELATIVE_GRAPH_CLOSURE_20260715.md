# TRKH 5-Class ViG Max-Relative Graph Closure - 2026-07-15

## Decision

Close the exact `vig_max_relative_graph` route at train-only Stage A.
`stage_b_smoke_authorized=false`. Do not run validation, test, a five-epoch
continuation, a probe, a full train, or a nearby graph sweep.

The graph mechanism was active and raised clean class-1 precision slightly,
but it removed far more true class-1 evidence than false positives. This is
not an acceptable precision trade for the agricultural classifier.

## Locked Scope

- Protocol:
  `docs/TRKH_5CLASS_VIG_MAX_RELATIVE_GRAPH_READINESS_PROTOCOL_20260715.md`
- Protocol SHA-256:
  `9a0ad9f928e1915417c198b12dfeaaf07327734eb10d532566aa34f76382dd76`
- Official ViG commit:
  `f90e129b645c3b1684fe07cd361cd557d0ad71f7`
- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Stage-A run:
  `runs/audit_vig_graph_readiness_20260715_stage_a_retry1`
- Validation used: `false`
- Test used: `false`
- Raw dataset modified: `false`

Stage A used only `yolo_f/train=9215`. Source-disjoint fold 0 supplied
`1843` ordered holdout rows, folds 1-4 supplied `7372` fit rows, and the two
variants consumed the same deterministic `30 x 32 = 960` fit rows.

## Implementation Result

The default-off candidate added ViG-style patch graphs after spatial MHSA in
Transformer layers `2,5`:

- deterministic `256 -> 64` projection;
- L2-normalized dense kNN with `k=9`, including self;
- official max-relative message `max_j(x_j-x_i)`;
- `128 -> 256` GELU residual with zero-initialized output projection;
- no relative position, graph dilation, stochastic edge selection, graph loss,
  or change to the CNN stem, spatial MHSA, pruning, classifier, or XAI source.

Checkpoint extension and deterministic-construction gates passed:

- exactly ten new state tensors and no unexpected keys;
- exactly `99,840` added trainable parameters;
- every existing keeper tensor remained bit-identical;
- constructor/forward CPU and CUDA RNG states matched control;
- initial logits and direct graph residual were exactly equal to control;
- all eight spatial-MHSA layers and pruning after layers `2,5` remained intact.

## Mechanism Evidence

This route did not fail because the branch was dead:

- all FP32 and BF16 graph/head/input gradient families were finite and nonzero;
- all ten graph tensors moved while frozen existing state remained bit-exact;
- residual-to-patch norm ratios were `0.096046/0.095173`;
- mean non-self neighbors were `8.070/8.048`;
- non-local neighbor fractions were `0.681464/0.669135`;
- normalized neighbor-selection entropy was `0.968467/0.972371`;
- layer-2/5 ablation changed logits by `0.261719/0.148438`;
- peak VRAM passed at `0.786735 GiB`.

The graph therefore created a diverse, strongly non-local and materially used
signal. Its failure is behavioral: feature-neighbor aggregation suppressed
class-1 modes instead of selectively separating their confusing boundaries.

## Clean Decision Failure

After matched head-only versus head-plus-graph adaptation on the train-only
holdout:

| Metric | Control | Graph candidate | Delta |
|---|---:|---:|---:|
| Macro F1 | `0.940928` | `0.906509` | `-0.034419` |
| Class-1 F1 | `0.829268` | `0.681818` | `-0.147450` |
| Class-1 precision | `0.885417` | `0.895522` | `+0.010106` |
| Class-1 recall | `0.779817` | `0.550459` | `-0.229358` |

The candidate changed `38` decisions with only `8` corrections and `29`
harms. It removed four restricted `0/2/4 -> 1` false positives, but rescued
zero class-1 false negatives and broke `25` true class-1 predictions. Maximum
nonfocus F1 drop was `0.020964`, also outside the locked safety gate.

The positive precision delta cannot promote the route. Precision rose because
the model predicted class 1 much less often, not because it learned a safer
boundary while preserving disease/maturity evidence.

## Illumination Failure

All three candidate-minus-control robustness comparisons reduced macro and
class-1 F1:

| Condition | Macro F1 delta | Class-1 F1 delta | Precision delta | Recall delta | Rescue / TP break |
|---|---:|---:|---:|---:|---:|
| Dim | `-0.024830` | `-0.109557` | `-0.067775` | `-0.082569` | `0 / 9` |
| Bright | `-0.008640` | `-0.048884` | `+0.064815` | `-0.045872` | `1 / 6` |
| Low contrast | `-0.027990` | `-0.120833` | `-0.047059` | `-0.110092` | `0 / 12` |

The graph removed more focus false positives than it created across these
conditions, but again did so by breaking true class-1 predictions. This is the
same unsafe precision mechanism seen on clean input.

## Export And Resource Result

- Full static-batch-1 three-input ONNX export passed with maximum logit error
  `3.58e-7` and matching argmax.
- The isolated graph residual exported and ran, but its maximum element error
  was `3.492e-4`, above the locked `1e-5` gate. Near-tied kNN/TopK neighbor
  ordering is the likely cause; the result is retained as an unresolved
  isolated-equation parity issue, not repaired after behavioral rejection.
- Candidate/control median forward-backward time was
  `0.090292/0.059616 s`, ratio `1.514575x`, above the locked `1.30x` ceiling.
- Candidate peak VRAM was only `0.786735 GiB`, so memory was not the blocker.

Behavioral failures independently close the method even if isolated ONNX tie
handling or runtime were optimized.

## Attention And XAI Boundary

The precommitted Stage A required graph telemetry, hard-decision transitions,
three illumination conditions, gradients, ablations, export, runtime, and
VRAM. All were completed. Image-space native-attention/rollout/Grad-CAM and
perturbation XAI belonged to Stage B and was authorized only after every Stage
A gate passed. Because Stage A failed and no candidate checkpoint was promoted,
no validation/XAI Stage B was run. Existing spatial MHSA remained untouched and
was never replaced by a graph-neighbor map.

## No-Repeat Rule

Do not sweep graph `k`, layers, bottleneck width, projection initialization,
graph type, self-neighbor policy, residual scale/form, LR, weight decay,
budget, fold, seed, loss, augmentation, or run length on this keeper. Do not
convert the result into a graph confidence router or post-hoc class-1 filter.

The mechanism teaches a useful negative lesson: strongly non-local patch
similarity is not class-1 reliability. Future representation work must prove
direct TP protection and class-specific boundary encoding before validation.

## Evidence Hashes

- Stage-A summary:
  `8bcacda2dbf511d86a02aad97cfac208d0994d02b277a144cbd5dabca62bde77`
- Report:
  `b653aa2e85040151dffa9cf7faa3df29916358e1b2c3aed34e527a25cbaea7f7`
- Artifact manifest:
  `eef913da874a73159b4dedb04070254796d78d66d8142fbf9d8e24a8c12f0158`
- Holdout predictions:
  `fb282e28533843ecf435a78622df51df448684d86c56a3506a7460288d8fbec1`
- Illumination predictions:
  `8fea08ab61c4110976cd36b950d29caf85008174d1c141fb5f6a7e439550f7e9`
- Isolated/full ONNX:
  `b51c31227d53bc3eca4fb4c128d1f0bc590e70b59428da1d8c665b3eb761ddd1` /
  `575310b6bf7a9239c2b2e3ec4f9689741b5b238bd0f226ecde5ecbe45b83ef10`

After recording these hashes, rejected-run compaction retained all five
nonbinary source files plus its verification metadata as seven verified
payloads under `runs/evidence_vig_graph_stage_a_rejected_20260715`. Payload
manifest SHA is
`7452c05a9aa18a9a6788050b8a8c3be10aacfc249aca858b67343d16d36e7671`.
The two reproducible ONNX binaries totaling `31,168,862` bytes were excluded;
source deletion was verified and observed free space increased by `33,783,808`
bytes. Cleanup manifest SHA is
`996841bfd3a18fa2c736e9f1a197d288e2b4fa795bb8aeb8d7844e10a607cbc8`.

Read-only retention then passed over `672` run directories, found zero
remaining compacted originals and `blockers=[]`. Retention summary SHA is
`4c68424454adf7b9e57cea2fa11aa9c8f24ecc92767771d751dcb3d8e27f69d9`.

## Engineering Verification

- Package compileall passed.
- Focused ViG/XCA/SRM regression tests passed `37/37` before Stage A.
- Full pytest passed `1078/1078` in `90.60 s` after closure.
- V8, ViG readiness, current-best full pipeline, TensorRT export, and video
  PowerShell launchers parsed with zero errors.
- The current-best full-pipeline preflight still resolves the protected keeper,
  explicit `yolo_f`, `skip_final_test=true`, and the independent raw-validation
  promotion gate.
- TensorRT export preflight passed with TensorRT `10.7.0` and CUDA; PyTorch
  video preflight passed and retained classification-only support.
- `BaoCao/` and both user-owned deep-research reports remained untouched and
  unstaged.

Keeper, scratch complement, and current-command hashes remain
`1f49d577...482677`, `f8bd6309...1a549`, and `36b9aa1a...40faf`.
No current-best command revision is justified.
