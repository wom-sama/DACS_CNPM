# TRKH 5-Class GSFL Shared-Feature Closure - 2026-07-16

## Scope and provenance

This gate asked whether GSFL-style shared/discriminative decomposition adds useful
class-1 boundary information to the frozen no-pretrain TRKH keeper embedding. It is
not an exact reproduction of Li and Monga's BMVC 2019 system: the official recipe
uses pretrained VGG, cross-validated class groups, 150+200 epochs, and test-selected
models. The official repository is unlicensed, and its shared-center loop applies
`labels[0]` to later batch rows. No official source was copied into TRKH.

The fixed local adaptation replaced the incompatible sigmoid decoder with a linear
decoder because keeper embeddings are signed. It retained the paper's decomposition,
reconstruction, class-center, and shared-center mechanisms under a prospective
train-only protocol. Paper SHA-256 is `f4cf6a6c...a42506`; official repository commit
is `8527039...65fe5`; protocol SHA-256 is `1d5dba1...eaafb`.

## Locked formal run

- Data: exactly `yolo_f/train`, source-disjoint fit/holdout `7372/1843` rows and
  `6452/1612` source groups, overlap zero.
- Input: L2-normalized frozen 256-dimensional keeper embeddings.
- Candidate: paired 128-dimensional shared/discriminative encoders, linear
  reconstruction decoder, and five-class discriminative classifier.
- Loss: CE + `0.01` reconstruction + `0.1` class center + `0.1` shared center.
- Training: Adam, LR `0.001`, weight decay `1e-5`, batch `42`, seed `20260715`,
  exactly 30 epochs, final holdout evaluation once, no sweep or early selection.
- Control: CE adapter with exactly matched discriminative/classifier initialization
  and paired dropout masks.
- Validation/test/model binaries/raw-data writes: none.
- Precommit/fix commits: `99b3f58` and `ca6fdb8`.

All structural gates passed: exact cohort/provenance, paired initialization/dropout,
finite nonzero gradients, decreasing objectives, reconstruction reduction, updated
class/shared centers, normalized probabilities, and a candidate distinct from control.

## Results

| Variant | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | Class-1 TP | Restricted FP |
|---|---:|---:|---:|---:|---:|---:|
| Raw keeper | 0.949323 | 0.748252 | 0.981651 | 0.849206 | 107 | 36 |
| Aggregate-margin A-GEM | 0.951161 | 0.769784 | 0.981651 | 0.862903 | 107 | 32 |
| Matched CE adapter | 0.941276 | 0.861386 | 0.798165 | 0.828571 | 87 | 14 |
| GSFL adapter | 0.945650 | 0.853211 | 0.853211 | 0.853211 | 93 | 16 |

Versus raw, GSFL raised class-1 precision by `+0.104959` and removed 20 restricted
false positives, but broke 14 class-1 true positives. Recall fell by `-0.128440`, macro
F1 fell by `-0.003673`, and class-1 F1 gained only `+0.004005`, below the locked
`+0.005` gate. Of 50 changed decisions, only 20 were corrections and 27 were harms.
The direction AUROC was `0.263889`, so the score moved more strongly on false positives
than on the vulnerable true positives it needed to preserve.

The visual audit inspected close-crop and wide-context examples on both sides. Removed
false positives and broken true positives both contain narrow and broad backgrounds;
the split is therefore not a stable context nuisance. The adapter is suppressing a
subtle surface/ripeness boundary rather than learning a safe foreground/background rule.

## Decision

- Shared-trainer integration is denied. The precision gain is broad class-1
  suppression and is unsafe for the agricultural requirement.
- Close nearby GSFL width, class-group, decoder, center-weight, reconstruction-weight,
  optimizer, epoch, seed, and sampling sweeps on the current keeper representation.
- Do not revive the official pretrained/test-selected recipe under the no-pretrain,
  train-only, and `<=30`-epoch TRKH contract.
- Current-best commands remain unchanged at three revisions and two updates after the
  initial revision. No validation winner or checkpoint promotion occurred.

## Evidence and verification

Evidence root: `runs/audit_gsfl_shared_feature_readiness_20260715`.

- Summary SHA-256: `e1396669...d801d`.
- Holdout predictions SHA-256: `65e28ddb...a3668`.
- Report SHA-256: `c385d0a7...ac41`.
- Training curve SHA-256: `45f365b1...8f24`.
- Artifact manifest SHA-256: `7c5d68a9...099f`.

Independent replay matched all 1,843 rows, probability normalization, four confusion
matrices, metrics, transitions, restricted-FP counts, direction AUROC, 30 training
epochs, and every payload hash. The five evidence files contain no checkpoint, ONNX
graph, engine, validation payload, or test payload.

Compilation, focused tests `5/5`, full pytest `1170/1170`, PowerShell parsing and actual
preflight all passed. Read-only retention passed over 697 run directories with
`blockers=[]`, no deletion, raw data untouched, and `72.265 GiB` free; retention summary
SHA-256 is `0c545fe6...0756`. Keeper, scratch complement, current command, and command
history hashes remained exact.

## Next boundary

The next method must create class-1-positive local surface evidence while preserving
the 107 raw true positives. Another frozen global embedding head, prototype/density
classifier, sample filter, gradient combiner, uncertainty head, context suppressor, or
GSFL decomposition is not sufficiently distinct. A primary-source/no-repeat screen is
required before any new code or training run.
