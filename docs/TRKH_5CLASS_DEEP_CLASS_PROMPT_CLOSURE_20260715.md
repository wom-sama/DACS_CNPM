# TRKH 5-Class Deep Class-Prompt Closure - 2026-07-15

## Decision

Close the exact `deep_class_prompt` route at train-only Stage A.
`stage_b_smoke_authorized=false`. Do not run validation, test, a five-epoch
continuation, a probe, a full train, or any nearby prompt sweep.

Deep prompts increased class-1 precision by predicting class 1 much less often.
They did not learn a safe class-specific foreground boundary: class-1 recall
fell sharply, no false negative was rescued, 25 true positives were broken,
and the learned class-1 map moved more strongly toward the confuser region.

## Locked Scope

- Protocol:
  `docs/TRKH_5CLASS_DEEP_CLASS_PROMPT_READINESS_PROTOCOL_20260715.md`
- Protocol SHA-256:
  `15703848b396beb2adbcc88f0f07f58e56c69ec449276d7fb76e1b833a865f9d`
- Implementation commit: `8eb0bc9`
- Prompt-CAM official commit:
  `4d35f3fb2eb99a63859465fcc1c7c3f4879f3ac5`
- MCTformer official commit:
  `0acc27ada87a5582053efb14648442d8644168aa`
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Dataset: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`
- Raw dataset modified: `false`
- Validation used: `false`
- Test used: `false`

Stage A used only `yolo_f/train=9215`. Source-disjoint fold 0 supplied `1843`
ordered holdout rows, folds 1-4 supplied `7372` fit rows, and control/candidate
consumed the same deterministic `60 x 32 = 1920` fit rows.

## Implementation Result

The default-off module inserted one prompt per class at each of the eight
existing Transformer blocks, removed prompts before the next base-token step,
and exposed reconstructed class-to-patch maps separately from legacy native
attention. It used one shared LayerNorm and scalar head and fused logits as:

`base_logits + 0.10 * class_prompt_logits`.

The candidate added exactly `11,009` parameters and the five expected state
tensors. Existing keeper state remained bit-exact, pruning stayed after layers
`2,5`, and all eight spatial-MHSA/native-attention paths retained their legacy
schema. Candidate prompt tensors moved, every required FP32/BF16 gradient was
finite and nonzero, and frozen keeper tensors remained exact.

The formal run exposed one infrastructure defect: `fork_rng(devices=[])`
restored CPU RNG but not CUDA RNG after `torch.manual_seed()`. The constructor
now forks every visible CUDA device. Focused regression and a keeper-shape
post-fix check pass CPU/CUDA constructor RNG equality, exact keeper state, and
the `11,009`-parameter contract. The immutable formal summary retains its
original failed `constructor_rng_equal` field. This repair does not reopen the
experiment because nine independent behavior/XAI gates also failed.

## Train-Only Decision Failure

| Metric | Control | Prompt candidate | Delta |
|---|---:|---:|---:|
| Macro F1 | `0.928091` | `0.898436` | `-0.029655` |
| Class-1 F1 | `0.826087` | `0.744681` | `-0.081406` |
| Class-1 precision | `0.785124` | `0.886076` | `+0.100952` |
| Class-1 recall | `0.871560` | `0.642202` | `-0.229358` |

The candidate changed 90 decisions with only `27` corrections and `62` harms.
It correctly removed `17` restricted `0/2/4 -> 1` false positives, but rescued
zero class-1 false negatives and broke `25` class-1 true positives. Maximum
nonfocus F1 drop was `0.028366`, well outside the locked `0.005` ceiling.

The class-1 predicted support fell from `121` to `79`. The precision gain is
therefore class suppression, not a deployable high-precision improvement.

## Illumination Failure

| Condition | Macro F1 delta | Class-1 F1 delta | Precision delta | Recall delta |
|---|---:|---:|---:|---:|
| Dim | `-0.009524` | `-0.115617` | `+0.149024` | `-0.155963` |
| Bright | `-0.041948` | `-0.155844` | `+0.137037` | `-0.275229` |
| Low contrast | `-0.032274` | `-0.112972` | `+0.078755` | `-0.155963` |

Every condition repeated the unsafe suppression pattern. Positive precision
deltas cannot compensate for the large F1/recall losses.

## Prompt XAI

The mechanism was active rather than collapsed:

- mean off-diagonal prompt-map cosine improved `0.891699 -> 0.672559`;
- effective rank improved `2.811462 -> 3.531364`;
- minimum class-logit variance improved `4.746270 -> 9.701262`.

However, spatial evidence moved in the wrong direction:

- target bbox mass delta: `+0.008471`;
- strongest-confuser bbox mass delta: `+0.042853`;
- target-versus-confuser separation delta: `-0.034383`;
- target core/ring/context deltas: `-0.011508/+0.019979/-0.008471`.

Prompt diversity is not sufficient. The learned maps became more distinct but
more confuser-aligned, matching the class-1 TP collapse.

## Export And Resource Result

- Static-batch-1 full ONNX passed with max logit error `2.98e-7` and matching
  argmax.
- Candidate/keeper median inference time was `0.058213/0.055868 s`, ratio
  `1.041970x`.
- Peak training allocation was `1.081265 GiB`.
- Control/candidate adaptation time was `183.80/172.14 s` with identical train
  order.

Deployment and resource checks were not blockers. Behavior closes the route.

## No-Repeat Rule

Do not sweep prompt count, insertion layers, initialization, logit fusion,
prompt-only classifier, dense MCTformer head, optimizer, LR, weight decay,
fold, seed, budget, loss, augmentation, or run length on this keeper. Do not
convert prompt logits or maps into a post-hoc class-1 filter.

Future work must protect class-1 TP directly and distinguish foreground
surface/boundary evidence without treating reduced class-1 support as a
precision win.

## Evidence And Cleanup

Original Stage-A hashes before compaction:

- summary: `c4e4584126ab1aba0457cd796d29065303175efb99b0335453d15d77e4bc0d4c`;
- report: `86e0084b41c7de198642ef142f4d939485dfbb7a5120e9ff6d6ddb0c3ffcaa08`;
- artifact manifest:
  `a68c933e5362050042aea5050a4535a049853cd8342f6cd933675850c5e69faf`;
- holdout predictions:
  `cd284a382a7bb27aca7270439decd57d9b0baa9cb0e00daba61e71b540a0bdb5`;
- illumination predictions:
  `ecb93e5cc24f81791a53d49a282ca3801c88b4ecee274eb3b47bd05de89f2fad`;
- adapted prompt state:
  `3758e29a480818f2b5aa5b6c812bc6dcb2521270a9df0b23779391d347dfdddb`;
- ONNX:
  `390217902c9a93b8345388ecd40d577a764a3b500f15a91348e1586061c7b82f`.

Compaction retained all five nonbinary source files plus verification metadata
as seven verified payloads under
`runs/evidence_deep_class_prompt_stage_a_rejected_20260715`. Payload manifest
SHA is `d7b6b64b...e137d2`. The reproducible prompt-state and ONNX binaries
totaling `30,699,520` bytes were excluded. Source deletion was verified and
observed free space increased by `33,320,960` bytes. Cleanup manifest SHA is
`5c85fdc4...32a8c0`.

Retention passed over `674` run directories with `blockers=[]`; summary SHA is
`6bfd164a...57b71`. Keeper, scratch complement, current command, and command
history hashes remain unchanged. No full-train command revision is justified.

## Engineering Verification

- Package and tests compileall passed.
- Focused deep-prompt/auditor tests passed `10/10`.
- Full pytest passed `1088/1088` in `43.29 s`.
- Deep-prompt readiness, V8, current-best pipeline, TensorRT export, and video
  PowerShell launchers parsed with zero errors.
- Deep-prompt preflight preserved `validation_used=false` and
  `test_used=false`.
- Current-best full-pipeline preflight resolved the protected keeper, explicit
  `yolo_f`, `skip_final_test=true`, and the independent raw-validation gate.
- TensorRT `10.7.0`/CUDA export and PyTorch classification-only video
  preflights passed.
- `BaoCao/` and both user-owned deep-research reports remained untouched and
  unstaged.
