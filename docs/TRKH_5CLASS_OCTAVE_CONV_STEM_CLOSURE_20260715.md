# TRKH 5-Class Octave Convolution Stem Closure - 2026-07-15

## Decision

The exact parameter-matched Octave Convolution stem is rejected and closed on
the current scratch TRKH recipe. It must not receive an alpha, path, width,
depth, learning-rate, seed, loss, augmentation, or run-length sweep. No
five-epoch continuation, probe, full train, or test evaluation is permitted.

The current-best full-train, final-audit, TensorRT export, and video commands
remain unchanged. The candidate did not pass the locked independent validation
gate and is not a precision improvement for the agricultural use case.

## Locked Experiment

- Protocol:
  `docs/TRKH_5CLASS_OCTAVE_CONV_STEM_READINESS_PROTOCOL_20260715.md`
- Primary method: Chen et al., ICCV 2019, Octave Convolution.
- Official source commit:
  `87c44f79162f3a2ef316bf924ad3697b8957a463`
- Reviewed source SHA-256:
  `06e60037b8fd5d4cd9e063ab1d439b25ddc98c4a109f770cbc0adcb9d7295675`
- Fixed low-frequency channel ratio: `alpha=0.125`.
- Change: replace only the three vanilla stem convolutions with
  first/persistent/final OctConv paths while retaining the stride-8 output,
  patch projection, `16x16` token grid, eight spatial-MHSA blocks, pruning at
  layers `2,5`, loss, augmentation, optimizer, scheduler, and seed.
- Smoke: one matched scratch `120 batches x 2 epochs`, full `yolo_f/val=2606`,
  with no test split.

## Stage A

Train-only readiness passed every locked check at
`runs/audit_octave_conv_stem_stage_a_20260715`.

- Summary SHA-256:
  `b27335e71b8e228c065aa26cbc0015e89521f2bd03c039f39dc8d35be04e159b`
- Total parameters were exactly matched at `7,245,590`; stem kernel parameters
  were exactly matched at `166,752`.
- Downstream non-stem initialization and post-constructor RNG state were paired;
  reconstructed virtual vanilla kernels were bit-identical.
- All high/high, high/low, low/low, low/high, patch-projection, FP32, and BF16
  gradient checks passed. Path ablations were material and low-frequency
  activation was smoother without collapse.
- ONNX maximum error was `8.94e-7`.
- Candidate peak VRAM was `2.8077 GiB`; runtime ratio was `1.1917x`, within the
  `3.25 GiB` and `1.50x` limits.
- Stage A constructed train data only (`9215` object rows); validation and test
  loaders were not constructed.

## Matched Validation Result

The comparison summary is
`runs/audit_octave_conv_matched_smoke_pair_20260715/comparison/summary.json`,
SHA-256
`db74f347b2654fedfd6dbb8a16e3b808605cf0252cd751ab9524461d18a6347b`.

| Metric | Control | OctConv | Delta |
|---|---:|---:|---:|
| Macro F1 | 0.784059 | 0.775768 | -0.008290 |
| Class-1 precision | 0.395652 | 0.389381 | -0.006272 |
| Class-1 recall | 0.602649 | 0.582781 | -0.019868 |
| Class-1 F1 | 0.477690 | 0.466844 | -0.010847 |
| Class-1 TP | 91 | 88 | -3 |
| Class-1 FP | 139 | 138 | -1 |
| Class-1 FN | 60 | 63 | +3 |

The candidate changed `116` decisions: `40` corrections versus `59` harms,
two class-1 FN rescues versus five TP breaks, two net focus-FP removals, and
nine new `3->2` harms. Runtime was `652.395 s` versus `607.142 s` control
(`1.07453x`). Eight material continuation checks failed.

Control checkpoint SHA-256 is
`3f4cad1c3115e585b1e22b93b17df54f05ad92711ba8dbf1f4027efc333d9525`;
candidate checkpoint SHA-256 is
`f674dde7bbbb0bcaf85f128f7412da4b4e6baa3bfafdf38e1fb2cf24886fa254`.

## Robustness And XAI

The complete post-smoke summary is
`runs/audit_octave_conv_matched_smoke_pair_20260715/postsmoke_audit/summary.json`,
SHA-256
`4eff90635650ba7c257d37178d3104569daa7d0f1b495046e5e7a6eb66f1897a`.

- OctConv won macro F1 in only `2/5` conditions: bright `+0.021073` and low
  contrast `+0.001398`. It lost clean `-0.007486`, center occlusion
  `-0.003818`, and dim `-0.015049`.
- Worst class-1 recall delta was `-0.013245`; the robustness gate requires at
  least three macro wins and therefore failed.
- The paired 16-case XAI summary SHA-256 is
  `11270d54f834bffb45b1330eadb2720405f31fd4a5252aeb24961f0d59336c04`.
  All cases used native block-7 MHSA and eight gradient-bearing rollout layers;
  attention and grad-rollout fallback counts were zero.
- Candidate attention foreground mass rose `0.934277 -> 0.942699`; stem
  Grad-CAM foreground rose `0.930686 -> 0.945499`, and Grad-CAM border mass
  fell `0.157529 -> 0.128220`.
- Despite cleaner localization, candidate accuracy on the changed-case XAI
  cohort fell `0.500 -> 0.375`. On TP breaks, attention border mass rose by
  about `0.0819`; heatmap cleanliness is therefore not a decision-quality gate.
- Candidate object-desaturation causal drop was `0.058269`, while background
  gray/blur drops were only `0.001826/0.001949`. The unresolved signal remains
  foreground surface/color discrimination, not far-background rejection.

## Operational Boundary

- `test_used=false` is present in Stage A, pair comparison, trace, cohort,
  paired XAI, and post-smoke closure artifacts.
- Raw data was not modified.
- Keeper checkpoint SHA-256 remains
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Random-init complement SHA-256 remains
  `f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549`.
- Current VS Code command packet SHA-256 remains
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`.

## Evidence Compaction And Retention

- `runs/evidence_octave_conv_stem_rejected_20260715` preserves `322` copied
  files and `324` verified payloads. File-manifest SHA-256 is
  `d2772bfff85ab78af5c81c7334648de0ee563ccb1ac2ee27878070e14df87abd`.
- Four checkpoint binaries totaling `348,783,986` bytes were intentionally
  excluded. Cleanup-manifest SHA-256 is
  `0c6a38a7f3f6765e7852d1117b63c3fc55985d336869348e3e73d2a7240929ab`.
- The two rejected smoke roots were removed only after inventory and copied
  payload verification. The Stage-A root and complete matched-pair audit remain
  live.
- Retention audit
  `runs/artifact_retention_audit_20260715_octave_conv_closure` passed over
  `665` run directories with `blockers=[]`; summary SHA-256 is
  `e66ab1d2bc83d92888ed250f9db314a4413b19f6f155c28be6df7f95f14dfcea`.
- Free space on `D:` was `75.759 GiB` after cleanup. Raw dataset content was
  outside the operation.

## Engineering Verification

- Package compileall passed.
- Focused OctConv/model/gate/trace/compaction/retention tests passed `24/24`.
- Full pytest passed `1040/1040`; only existing dependency deprecation warnings
  were emitted.
- Nine affected/current PowerShell launchers parsed with zero errors.
- Six operational preflights passed without training or opening test data:
  current-best full pipeline with an optional final-test request, frozen
  precision package, direct-keeper TensorRT export, direct-keeper PyTorch
  video, OctConv Stage A, and the locked OctConv matched smoke.
- The post-smoke launcher was parser-checked only after intentional checkpoint
  compaction. Its completed status and hash-verified pair/post-smoke artifacts
  are the replay evidence; rejected checkpoints were not recreated.

## Next Distinct Direction

The next research protocol may evaluate a limited Cross-Covariance Attention
(XCA) residual derived from official XCiT, while preserving the existing CNN
stem, spatial MHSA, pruning, and XAI path. This is a new channel-interaction
hypothesis, not an OctConv continuation. It requires a primary-source lock,
train-only decision-level readiness, deterministic pairing, direct class-1 TP
protection, and focus-FP reduction before any validation smoke.
