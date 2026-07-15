# TRKH 5-Class Patch-Style SRM FFN Closure - 2026-07-15

## Decision

Close the exact Patch-Style SRM FFN route at Stage A. The implementation is
mechanically valid and deployable, but the locked train-only comparison shows
no class-1 precision benefit and a small loss of class-1 F1/recall. Stage B,
validation, test, probe, full train, and nearby parameter sweeps are not
authorized.

Raw data was not modified. Validation and test datasets were not constructed.
The current keeper, scratch complement, and VS Code command packet remain
unchanged.

## Locked Method

- Protocol:
  `docs/TRKH_5CLASS_PATCH_STYLE_SRM_FFN_READINESS_PROTOCOL_20260715.md`
- Protocol SHA-256:
  `e71918dc236b28681f6db8d69e6ff9ccb3c15af0e0bc7805d0a2306932486779`
- Primary source: Lee, Kim, and Nam, ICCV 2019 SRM.
- Official source commit:
  `f6221c77e797c4530dddba03616153708d636bd1`
- Official source file SHA-256:
  `5232416ff7eabecd43a3d0e9e118c64f2dc2e97ded55d5141863ec8fe0e32d13`
- Exact adaptation: hidden mean/std SRM after GELU in FFN layers `2,5`, patch
  tokens only, zero-initialized CFC, BatchNorm, sigmoid gate, `8,192` added
  trainable parameters.

## Stage-A Provenance

- Completed run:
  `runs/audit_patch_style_srm_readiness_20260715_stage_a_retry1`
- Summary SHA-256:
  `80af1fd4bc39a55213c85e708a1c622cd6840395df64a53c3ed90fad1108b196`
- Report SHA-256:
  `2c9e5d52b4efd3a0a05c270bf2511d8ca13ff28c17232517c397ba26a48796e2`
- Artifact manifest SHA-256:
  `12635a20098f413106f74e5887392639f25cd9eac93eda42a85c0c74f86b4a5c`
- Dataset scope: ordered `yolo_f/train=9215` declaration only.
- Source-disjoint fold 0: `7372` fit rows and `1843` holdout rows, zero
  source overlap.
- Matched adaptation: `30 x 32 = 960` rows, identical order SHA-256
  `e94fe9bdd44df1b5ffdd6d3fde9b53be79ba44225234c52fc80c8a8f10b7a4a8`.
- Control trained only the 1,285-parameter classifier. Candidate trained the
  same classifier plus 8,192 SRM parameters.

## Mechanism Checks

The route passed every locked mechanism check except runtime:

- existing checkpoint tensors remained bit-identical; the exact 12 SRM state
  additions and `8,192` trainable parameters were present;
- constructor and forward RNG checkpoints matched;
- FP32/BF16 equations, patch-only prefix preservation, all classifier/CFC/BN/
  FFN/input-gradient families, frozen-state equality, and state movement passed;
- adapted gates were finite and noncollapsed. Layer 2/5 maximum deviations
  from `0.5` were `0.010157/0.007955`;
- bypassing layers 2/5 changed logits by `0.059570/0.022461`;
- all three illumination shifts materially changed both SRM gates;
- isolated/full static-batch-1 ONNX Runtime errors were
  `5.96e-8/2.68e-7`, with matching argmax and the full
  `images,bbox,image_mask` contract;
- peak allocated VRAM passed at `0.764486 GiB`;
- median matched runtime was `0.058903 s` control versus `0.095214 s`
  candidate, ratio `1.616455x`, above the locked `1.15x` ceiling.

The runtime failure is retained as a resource blocker, but it is not needed to
reject the route because the decision-level failures are independent.

## Train-Only Decision Result

Before adaptation, adding the initial `0.5` SRM gate already moved four hard
decisions and slightly worsened the holdout:

- macro/class-1 F1 delta: `-0.001434/-0.003330`;
- class-1 precision/recall delta: `-0.005125/0.000000`;
- corrections/harms: `1/3`;
- restricted focus-FP reduction: `-1`.

After the matched adaptation:

- control macro/class-1 F1: `0.940928/0.829268`;
- candidate macro/class-1 F1: `0.940045/0.823529`;
- macro/class-1 F1 delta: `-0.000883/-0.005739`;
- class-1 precision delta: `-0.001206`;
- class-1 recall delta: `-0.009174`;
- changed/corrections/harms: `6/3/3`;
- restricted focus FP control/candidate: `11/11`, reduction `0`;
- class-1 FN rescues/TP breaks: `1/2`;
- new `3->2` harms: `0`; maximum nonfocus F1 drop: `0`.

The candidate therefore failed the precommitted class-1 F1, precision,
focus-FP reduction, TP-protection, and correction-balance gates.

## Illumination Result

- Dim: macro delta `-0.000766`; class-1 F1/P/R deltas all `0`; corrections/
  harms `4/5`; no focus-FP removal or creation.
- Bright: macro delta `-0.001290`; class-1 F1/P/R deltas all `0`; corrections/
  harms `1/4`; one FN rescue and one TP break.
- Low contrast: macro/class-1 F1 delta `-0.002973/-0.014984`; class-1
  precision/recall delta `-0.031674/-0.009174`; corrections/harms `3/4`;
  zero focus-FP removals, one creation, and one TP break.

No condition improved macro F1. Low contrast violated the precision floor and
the aggregate focus-FP direction was nonpositive. SRM responds to lighting,
but the learned response is not class-selective false-positive suppression.

## Interpretation And No-Repeat Boundary

The evidence separates mechanism liveness from model utility. Hidden-channel
style gates learned, varied by sample/channel/illumination, affected both
selected layers, and exported accurately. They nevertheless reduced true
class-1 decisions without removing false positives. This dataset's remaining
surface/color boundary problem is not solved by per-channel first/second-moment
reweighting at these FFNs.

Do not sweep SRM layers, CFC/BN design, gate scaling, LR, weight decay, budget,
fold, seed, trainable parameter set, loss, augmentation, or run length on this
keeper. Future work must introduce a genuinely different representation signal
with train-only hard-decision FP suppression and class-1 TP protection gates.

## Retention And Commands

- All five manifest-listed final artifacts rehashed exactly. Holdout and
  illumination prediction SHA-256 values are `aba1a021...f05d54c` and
  `d9d93fbd...3134e7`; isolated/full ONNX SHA-256 values are
  `e8ba3a41...2097de` and `e5b1ce3e...9a4ba`.
- Two superseded infrastructure roots were compacted only after recording each
  binary hash. Evidence manifest SHA-256 is
  `c55a99679d2c3a0bc0c648674a32ea0aee39c2ffd27d5ceaddf83c7ecacee64d`;
  cleanup manifest SHA-256 is
  `f57be6ae28b9b16c0eea499d54df08ee1b38c44dba8b2e2c4f15ee4f875fdac8`.
  Three superseded ONNX files totaling `30,614,685` bytes were excluded, and
  observed free-space gain was `30,621,696` bytes.
- Retention audit
  `runs/artifact_retention_audit_20260715_patch_style_srm_closure` passed over
  `670` directories with `blockers=[]`; summary SHA-256 is
  `e91bff5dfac9048e68e148d2aa2997f8aa0e52be90d45ae4b9845aff024339f5`.
- Keeper/scratch/command SHA-256 values remain
  `1f49d577...482677`, `f8bd6309...1a549`, and `36b9aa1a...40faf`.
- No best-command update is recorded because no validation gate was opened and
  the train-only candidate did not win.

## Engineering Verification

- Package compileall passed.
- Focused SRM model/auditor/export tests passed `14/14`.
- Full pytest passed `1065/1065` in `76.93 s`.
- SRM Stage-A, V8 training, and current-best full-pipeline PowerShell launchers
  parsed with zero errors.
- The SRM wrapper preflight resolved direct native Python invocation and
  reported `validation_used=false`, `test_used=false`.
- The current-best full-pipeline preflight passed with an optional final-test
  request while retaining the independent raw-keeper promotion gate and the
  direct `$LASTEXITCODE` native-stderr policy.
