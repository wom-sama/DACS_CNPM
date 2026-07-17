# TRKH 5-Class DART Recurrent Aggregation A0 Closure - 2026-07-17

## Decision

Reject DART recurrent aggregation on the current keeper before XAI, validation,
test, trainer integration, smoke, probe, or full training. The train-only A0
shows that branch averaging improves recall relative to one mixed trajectory by
expanding the class-1 decision region, but it materially reduces precision and
creates restricted false positives. Recurrent averaging also produces exactly
the same decisions as the matched final-only average, so the locked evidence
does not support a benefit from the `Repeat` step.

No checkpoint is retained and current-best commands remain unchanged.

## Locked authority and execution

- Accepted source: Jain et al., *DART: Diversify-Aggregate-Repeat Training
  Improves Generalization of Neural Networks*, CVPR 2023.
- Official MIT repository commit/tree:
  `62274e83d2f08eb416db61d0957476c53fde9361` /
  `83d56bedfc1159eb16ecbf8b4363e014d15c6f75`.
- Corrected protocol SHA-256:
  `4887d71f8be119569ba963581d3e87eb1b62d303872f412ada07b33738c47fa8`.
- Formal implementation/upstream commit:
  `76252ef2e983b74662ec9128ff56a51c27346cb4`.
- Formal evidence:
  `runs/audit_dart_recurrent_aggregation_a0_20260717`.
- Formal summary SHA-256:
  `418f82d3bdbddd760798c1f3596616262651a0773f3044bd5f6cd3c20b3fe8e4`.
- Artifact manifest SHA-256:
  `20ef9c4135fe17abc6089c53a292eabe9dd981951941538103eb912adefab925`.

The first preflight stopped before CUDA because the initial protocol copied an
incorrect CIDT SHA. The first formal launcher invocation then stopped before
output/model loading because the in-process check misidentified the Windows
venv launcher parent. Both defects were documented and corrected before any
formal measurement. The sole completed formal run began after the wrapper saw
no existing Python/TensorRT process and an isolated GPU at `7%`, `1629/8188`
MiB, and `55 C`.

## Cohort and optimization integrity

- Only `yolo_f/train=9215` was constructed. Fold 0 supplied the complete
  source-disjoint holdout (`1843`); folds 1-4 supplied four 256-row update
  branches and a separate 256-row fit probe.
- Update/probe ordered-index SHAs are `2279a58a...72bb7` and
  `6490523a...c5bd`. Update/probe/holdout source overlaps are all zero.
- All three variants replayed the same 32 logical batches and RNG seeds
  exactly. Phase-1 final-only and recurrent branch/optimizer hashes are exact.
- Every parameter group received finite nonzero gradients and moved. AdamW
  steps are exactly `32` for mixed and `8` per branch; branch optimizer states
  remain distinct and unchanged across broadcasts.
- FP32 averaging preserved all nonfloating buffers bit-for-bit and every
  recurrent broadcast hash is exact. Sequential training peak is
  `3.914190 GiB <= 4.5 GiB`.
- Final-only/cycle-1/cycle-2 fit-probe barriers are
  `-0.000572160/-0.000147499/-0.000556454`; all are low and finite. The final
  recurrent barrier is nevertheless strictly above the final-only barrier by
  `1.57e-5`, failing the prospectively locked recurrence comparison.
- Pairwise branch-delta cosines are noncollapsed: final-only
  `0.7779..0.8465`, cycle 1 `0.5977..0.7279`, and cycle 2 `0.8610..0.8838`.

## Train-holdout behavior

On all 1,843 clean fold-0 rows:

| Variant | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | Restricted FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| mixed control | 0.939627 | 0.878788 | 0.798165 | 0.836538 | 12 |
| final-only average | 0.946848 | 0.774436 | 0.944954 | 0.851240 | 30 |
| recurrent DART | 0.946848 | 0.774436 | 0.944954 | 0.851240 | 30 |

Relative to mixed training, recurrent DART changes 48 decisions, rescues 16
class-1 false negatives, breaks no class-1 true positive, and raises macro/
class-1 F1 by `+0.007222/+0.014701`. It simultaneously lowers class-1
precision by `-0.104352`, creates 17 focus false positives, removes none, and
worsens restricted `{0,2,4}->1` FP by `12 -> 30`. This is recall expansion,
not precision-safe complement internalization.

Final-only and recurrent DART change zero decisions relative to each other on
clean, dim, bright, and low-contrast cohorts. Independent replay finds their
maximum probability difference is only `0.001123` on clean and at most
`0.004117` across lighting conditions. Recurrent class-1 F1 delta versus the
matched final-only control is exactly zero, failing the required `+0.002`.

Lighting confirms the same failure:

- dim: class-1 precision delta `-0.087121`, restricted FP `18 -> 37`;
- bright: class-1 precision delta `-0.168595`, restricted FP `13 -> 48`;
- low contrast: class-1 precision delta `-0.134773`, restricted FP `17 -> 43`.

The independent CSV replay covers `4 x 1843` rows and reproduces every
confusion, transition, and gate value exactly (`maximum error=0`).

## Deployment and artifacts

- Standard ONNX export passes at opset 17 with maximum error `3.66e-7` and
  matching argmax; the 30,535,555-byte ONNX was deleted after verification.
- Native/candidate FP32 batch-32 inference medians are
  `86.685/84.203 ms`, ratio `0.971363`; peak inference memory ratio is `1.0`.
- No checkpoint, ONNX, engine, visual, validation, or test artifact is retained.
- The compact evidence payload retains predictions, batch/RNG manifest,
  training/aggregation history, summary, and artifact manifest. Key SHAs are:
  predictions `974cae46...a1819`, history `a2295749...26899`, and batch
  manifest `dc37afee...53432`.
- XAI did not run because nonvisual gates failed; visual evidence cannot rescue
  a measured precision/FP and recurrence failure.

## No-repeat rule

Do not sweep DART branch count, role/fold mapping, augmentation strengths,
aggregation interval, cycle count, LR, optimizer, loss, seed, budget, average
weights, or a final-only/DART interpolation on this keeper. Do not combine it
with DropKey, SWAD, EMA, SAM, pretrained weights, distillation, output routing,
feature stitching, or a closed architecture route. Do not promote the mixed
control from this train-holdout diagnostic; it has no separately locked
validation evidence.

No validation, test, XAI, longer A0, smoke, probe, full train, raw-data edit, or
current-best command update is authorized from this route.

## Closure verification

- Compile, pyflakes, PowerShell AST, focused `12/12`, and full pytest
  `1395/1395` pass on the final code.
- Read-only retention passes over 749 run directories and all 48 compaction
  manifests: all 210 compacted originals remain absent, `blockers=[]`, and
  free space is `103.05 GiB`. Retention summary SHA-256 is
  `8751127615a420b4798a38dd23818b5ee6fed125d4c01843e851dd98abe3442e`.
- Current-best command/history SHAs remain
  `36b9aa1a...40faf` / `39bd2879...98f53`, still three revisions and two
  actual updates.
