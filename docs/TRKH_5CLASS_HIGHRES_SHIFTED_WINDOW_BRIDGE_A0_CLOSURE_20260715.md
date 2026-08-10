# TRKH High-Resolution Shifted-Window Bridge A0 Closure - 2026-07-15

## Decision

Reject the exact stride-8 Haar-supported shifted-window bridge before model
implementation. A0 did not authorize architecture code, Stage-A adaptation,
validation, test, probe, or full training. The current-best command and its
three-revision history remain unchanged.

## Scope And Provenance

- Frozen keeper SHA:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- `yolo_f/data.yaml` SHA:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Reused CIDT fold CSV SHA:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Locked protocol SHA:
  `526d209a132a6ce46982f5c82cf8925f1ac0d0b2953ab0eddd1c8f102e2dfd87`.
- Formal summary SHA:
  `44de7fab9a1fb2c24215cddc70f1dd572ec8f8c011df68c9b458d76e9136436b`.
- Artifact-manifest SHA:
  `34725588f76c5155a8a701668170fdd8fc50df5ab3f992974844059c05c79f13`.

The run used only all `9215` ordered `yolo_f/train` object rows, `8064`
source groups, and the fixed source-disjoint folds
`1843/1830/1828/1851/1863`. Dataset index, label, path, source stem, and keeper
argmax aligned exactly. Recomputed FP32 keeper probabilities differed by at
most `0.009427`, without any argmax mismatch. Validation and test were never
constructed.

## Result

The aligned detail descriptor was active and noncollapsed: shape
`9215x288`, effective rank `72.269949`, and all three Haar bands had nonzero
variance. Every placebo permutation stayed within its fit or holdout partition
and had zero same-source fixed points.

| Variant | Macro F1 | Class-1 F1 | Class-1 P | Class-1 R | Restricted FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen keeper | 0.939876 | 0.815444 | 0.700265 | 0.975970 | 222 |
| Matched control | 0.854975 | 0.527930 | 0.647849 | 0.445471 | 130 |
| Aligned Haar candidate | 0.822973 | 0.466151 | 0.488844 | 0.445471 | 249 |
| Permuted-detail placebo | 0.807471 | 0.438545 | 0.468487 | 0.412200 | 250 |

Candidate minus control was macro/class-1 F1
`-0.032002/-0.061779`, class-1 precision `-0.159006`, recall `0.000000`, and
restricted FP `+119`. It changed `752` decisions with `204/464`
corrections/harms and `71/71` class-1 FN rescues/TP breaks. All five folds had
negative macro, class-1 F1, and class-1 precision deltas. Candidate did beat
the placebo, proving some sample-aligned information, but it did not beat the
lower-capacity control and was behaviorally unsafe.

The control also failed its keeper-preservation guards by macro/class-1 F1
`-0.084901/-0.287514`. Thus no positive conclusion may be drawn from a weak
comparator. All 15 LBFGS fits reached the exact 200-iteration cap with large
residual norms. Post-run self-review corrected the future gate from
`iterations <= 200` to strict convergence before the cap. The immutable formal
summary therefore overstates one structural optimizer pass; this is an
additional rejection reason and cannot change the outcome.

## Audit And Retention

Independent recomputation from `predictions_oof.csv` exactly reproduced all
four metric rows and candidate-control transitions. The evidence directory
contains six hash-verified nonbinary payloads totaling `4,815,340` bytes, with
no checkpoint, model binary, ONNX, validation payload, or test payload. No
compaction is needed. Read-only retention passed across `680` run directories
with `blockers=[]`; retention summary SHA is
`fc565106377fa6c2989a2ac67c43379cfb9afae39661c03836ea2e2d327a5b7b`.

Do not sweep Haar ranks, erosion, pooling, residual scale, regularization,
optimizer iterations, window size, shifted-window depth, heads, LR, seed, or
run length. Do not reinterpret the placebo win as architecture permission.
Any future high-resolution route needs a new source-group-held-out objective
that preserves the frozen baseline and demonstrates precision gain without
true-class suppression before architecture code.

Closure verification passed targeted compileall, focused pytest `7/7`, full
pytest `1122/1122` in `77.69 s`, PowerShell parse, preflight with no output
directory, artifact hash recomputation, retention, `git diff --check`, and
protected-artifact hash checks.
