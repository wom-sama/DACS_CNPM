# TRKH 5-Class MS-Lacunarity Stem Closure - 2026-07-20

## Decision

Reject this exact multiscale-lacunarity stem route before condition replay,
XAI, an image-model smoke, validation/test access, or full training. The
candidate exposes additional class-1 recall but does not distinguish true
class-1 samples from restricted class `0/2/4` false positives. The current-best
checkpoint and full-train commands remain unchanged.

## Locked Basis

- Prospective protocol:
  `docs/TRKH_5CLASS_MS_LACUNARITY_STEM_READINESS_PROTOCOL_20260720.md`, SHA-256
  `a6151debe79add4148ea31e5d67cba5e8ebff08cc3abfdea01580320b137cb9e`.
- Primary paper: Mohan and Peeples, CVPRW Vision4Ag 2024, "Lacunarity Pooling
  Layers for Plant Image Classification using Texture Analysis."
- Official MIT repository commit/tree:
  `6e464b4c326513c206eb9377c7ff9139ddee65f7` /
  `9f1a7c618db236cd8673e2cb797308fb210e39fe`.
- Protocol and implementation were committed and pushed before the formal at
  `c99b265` and `a709506`.
- Frozen keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Source-disjoint generated train-only fold: `7,372/1,843` fit/holdout object
  rows, `6,452/1,612` source stems, and zero source overlap. Test was not read.

## Implemented Audit

The isolated auditor captures the frozen keeper's `[B,256,32,32]` final CNN
stem activation, applies the official `X=(tanh(F)+1)/2` scaling, and computes

`L(X) = E[X^2] / (E[X]^2 + epsilon) - 1`.

It compares a 512-dimensional native-mean control with a 1,536-dimensional
candidate containing `mean_native * lacunarity_level` over full/object regions
and native/one fixed Gaussian-pyrdown scale. Natural-frequency multinomial
logistic readouts use one locked `C=0.3`, five source-group folds, and one
fit-all/holdout evaluation. No image-model parameter is trained.

Engineering checks pass: dimensions and direct-moment equations are exact,
all features are finite, every one of 12 readouts converges, candidate-only
effective rank is `42.753535`, peak CUDA allocation is `995.75 MiB`, and all
fit/holdout plus fold source intersections are empty. Formal runtime is
`162.061 s`; train/holdout extraction takes `72.735/29.845 s`.

The prefix run exposed two harness defects before formal: NumPy arrays do not
provide the expected `.square()` method, and a fold missing a class returns a
reduced `predict_proba` matrix. The implementation now uses explicit squaring
and maps `model.classes_` into a fixed five-class probability matrix. A
regression test covers the latter. Both non-decisional prefix directories were
removed after formal: one empty directory and one directory containing eight
files (`345,978` bytes).

## Formal Result

| Metric | Control | Candidate | Delta |
|---|---:|---:|---:|
| OOF macro F1 | 0.850891 | 0.860888 | +0.009997 |
| OOF class-1 F1 | 0.501370 | 0.543943 | +0.042573 |
| Holdout macro F1 | 0.863452 | 0.868954 | +0.005502 |
| Holdout class-1 precision | 0.653846 | 0.588235 | -0.065611 |
| Holdout class-1 recall | 0.467890 | 0.550459 | +0.082569 |
| Holdout class-1 F1 | 0.545455 | 0.568720 | +0.023266 |

Macro F1 improves in every OOF fold and class-1 F1 improves in four of five,
but class-1 precision decreases in four of five folds. On holdout, the
candidate makes `48` corrections and `48` harms, removes/creates `10/25`
class-1 false positives, rescues/breaks `18/9` class-1 examples, and worsens
restricted `0/2/4 -> 1` false positives from `27` to `42`. Candidate-minus-
control direction AUROC is `0.653780` OOF but only `0.583014` holdout.

Three prospective gates fail:

- holdout class-1 precision gain `>= +0.025`;
- at least four net restricted-FP removals;
- holdout class-1 direction AUROC `>= 0.60`.

The most frequent changed cohort is target class 0 moving from control class 0
to candidate class 1 (`18` rows), while only `13` target-class1 rows are
rescued from class 0. This confirms broad class-1 support expansion instead of
the requested precision-selective surface cue.

## Evidence And Retention

- Formal directory:
  `runs/audit_ms_lacunarity_stem_full_trainonly_20260720`.
- Summary SHA-256:
  `23f9149c1438dfe329a2af354b24dcd16e6606a78707657f9db00e03c06db585`.
- Artifact manifest SHA-256:
  `d7ff114b58175fb493ec8e982c0bfb969c62a9bb306b66c55b503f97df38a7fe`.
- Seven payloads total `4,106,353` bytes; aggregate payload-manifest SHA-256 is
  `fa47e6d5071e1352e1909a4e556a7c74a6f2d71e9c98d4b74535a4cf08f55c4c`.
  The manifest contains no checkpoint, model binary, or test payload.
- Both prediction CSVs were independently replayed. Metrics, transitions,
  direction AUROCs, row counts, classes, groups, folds, and payload hashes
  reproduce exactly.
- Read-only retention scans `770` run directories and all `49` current object
  compaction manifests. All `216` listed compacted originals remain absent,
  `blockers=[]`, free space is `102.6 GiB`, and retention-summary SHA-256 is
  `6f2a64596e3b8aa92467a0ed42af0efb5082d780fd1c50e4fed2514d452da458`.

The prospective clean gate is conjunctive, so condition replay and XAI were
correctly not generated. Their absence is a protocol result, not missing audit
work. Do not sweep scales, Gaussian kernels, scaling, regions, erosion,
interactions, readout regularization, folds, class weights, thresholds, or
combine this route with closed texture branches.

## Verification

- Focused lacunarity tests: `10/10`.
- Full repository pytest before formal: `1500/1500`.
- Compile and `git diff --check`: pass.
- Raw dataset modified: no.
- Validation/test used for selection: no.
- Production model/trainer/config changed: no.
- Current-best command revision: unchanged.
