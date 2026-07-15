# TRKH 5-Class Class-1-Protected RSC Closure - 2026-07-15

## Decision

Close the exact class-1-protected pooled-channel RSC route at train-only
Stage A. `stage_b_smoke_authorized=false`. Do not run validation, test, a
longer adaptation, a probe, a full train, or a nearby RSC sweep.

The intervention was equation-correct, deterministic, class-1-safe at the
mask level, exportable, and moderately inexpensive. It improved matched
class-1 F1 and recall slightly, but the precision gain was too small, it did
not reduce restricted class-1 false positives, bright lighting regressed, and
peak training allocation exceeded the locked gate.

## Locked Scope

- Protocol:
  `docs/TRKH_5CLASS_CLASS1_PROTECTED_RSC_READINESS_PROTOCOL_20260715.md`
- Protocol SHA-256:
  `9227f5e5122184b89b6b2ee4239f62fc31671343f346e5a9e0b4efc8b9959366`
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Dataset: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`
- Raw dataset modified: `false`
- Validation used: `false`
- Test used: `false`

Stage A constructed only `yolo_f/train=9215`. Source-disjoint fold 0 supplied
`1843` holdout rows, folds 1-4 supplied `7372` fit rows, and control/candidate
consumed the same deterministic `60 x 32 = 1920` fit rows.

## Mechanism Result

The candidate masked the 86 largest signed true-logit-gradient channels of
the 256-dimensional pooled representation. It selected at most the top third
of positive-drop true classes `0,2,3,4`; true class 1 was protected exactly.

- selected/eligible/positive-drop rows: `622/1804/1786`;
- class-1 fit rows and masked class-1 channels: `116/0`;
- exact train-order, initial-state, parameter-count, and forward-RNG parity;
- every required gradient group was finite, nonzero, and moved;
- all source, split, CIDT, equation, and mask invariants passed.

Two pre-training infrastructure stops occurred before the formal run. The
first added the deterministic cuBLAS workspace setting. The second found that
native-grid position embeddings still invoked nondeterministic bicubic CUDA
backward. The model now bypasses interpolation only when the requested grid
equals the base grid; the native output is bit-exact to the former forward
path. Neither stopped attempt reached an optimizer step or left evidence used
for selection.

## Clean Train-Only Decision

| Metric | Raw keeper | CE control | Protected-RSC candidate | Candidate - control |
|---|---:|---:|---:|---:|
| Macro F1 | `0.949323` | `0.910629` | `0.912176` | `+0.001548` |
| Class-1 F1 | `0.849206` | `0.728205` | `0.734694` | `+0.006489` |
| Class-1 precision | `0.748252` | `0.825581` | `0.827586` | `+0.002005` |
| Class-1 recall | `0.981651` | `0.651376` | `0.660550` | `+0.009174` |
| Restricted `0/2/4 -> 1` FP | `36` | `15` | `15` | `0` |
| Predicted class-1 support | `143` | `86` | `87` | `+1` |

Candidate versus control changed five decisions: three corrections, two
harms, one class-1 FN rescue, and zero class-1 TP breaks. It removed one
restricted false positive but created another, so net reduction was zero.
Class-1 F1 passed its minimum delta, but precision missed `+0.005` and the
required net FP reduction missed `2`.

The raw row is not the matched promotion comparator, but it exposes an
important training risk: both short full-model adaptations sharply suppressed
class-1 recall relative to the keeper. RSC recovered only one class-1 TP over
the control and did not solve that adaptation drift.

## Illumination Safety

- Dim: macro/class-1 F1/precision/recall deltas
  `+0.002805/+0.010063/+0.007451/+0.009174`; restricted FP `19 -> 19`.
- Bright: deltas `-0.006010/-0.012579/-0.020000/-0.009174`; restricted FP
  `12 -> 13`, with `2/11` corrections/harms.
- Low contrast: deltas `+0.004826/+0.012121/+0.017857/+0.009174`; restricted
  FP `19 -> 18`.

Bright lighting failed the class-1 F1 floor and created a net restricted
false positive. Aggregate FN rescues still exceeded TP breaks (`4/3`), but
every illumination gate was required, so corruption gains cannot promote the
method.

## Export And Resources

- Structural/clean/illumination checks passed: `28/29`, `7/9`, `3/5`.
- Candidate/control median step ratio: `1.063282x`, within `1.35x`.
- Candidate peak allocation: `3.907542 GiB`, above the locked `3.25 GiB`.
- Static-batch-1 ONNX max error: `5.96e-7`, finite with matching argmax.
- The verified ONNX was deleted automatically; no candidate checkpoint was
  written because Stage B was denied.

The five failed gates were:
`peak_vram_lte_3p25_gib`, `class1_precision_delta_gte_0p005`,
`restricted_focus_fp_reduction_gte_2`,
`all_class1_f1_deltas_gte_minus_0p010`, and
`no_condition_increases_restricted_focus_fp`.

## Evidence

Retained nonbinary Stage-A payloads:

- summary: `049dfad64e6f80ed90e616f42c19612a9e25a2c184d309dfa41cc55f08dce3dc`;
- clean predictions: `93f212e69a0ff077283b23b434684f2e14a84ab929acfa33dafb9deefa42dde0`;
- illumination predictions: `9575ba9a865be15431c3272bd81a11112c949802b57bd5e076ddc5baa9d0970d`;
- training history: `5af554488df5885880fd5c4a1c1485682b4f04179916338da4fe337754c33d7b`.

An independent CSV recomputation matched both clean confusion matrices and
all decision deltas exactly. The completed run contains no checkpoint or ONNX
binary, so no additional compaction is needed.

Read-only artifact retention passed over `678` run directories with
`blockers=[]`; retention summary SHA-256 is
`b22337fa17e51cfae4ee6d929065eea66180803415b2434a6dcb3346a973f6f4`.
Keeper, scratch complement, current-command, and command-history hashes remain
`1f49d577...482677`, `f8bd6309...1a549`, `36b9aa1a...40faf`, and
`39bd2879...98f53`.

## Engineering Verification

- Targeted compileall passed.
- Focused RSC/native-grid tests passed `12/12`.
- Full pytest passed `1115/1115` in `55.06 s`.
- RSC, current-best, TensorRT-export, video, and V8 PowerShell launchers parsed
  with zero errors.
- RSC and current-best pipeline preflights passed and created no run directory.
- TensorRT export and PyTorch video preflights passed and created no deploy or
  output directory.
- `BaoCao/` and both user-owned deep-research reports remained untouched and
  unstaged.

## No-Repeat Rule

Do not sweep RSC drop fraction, batch fraction, minimum drop, protected or
eligible classes, representation location, LR, weight decay, optimizer, fold,
seed, budget, loss, augmentation, or run length on this keeper. Do not turn
the challenged confidence drop into a post-hoc class-1 filter.

Future feature-challenging work needs a genuinely different objective that
directly rewards restricted-FP removal while constraining class-1 TP and
bright-light behavior. A larger or longer version of this exact RSC policy is
not new evidence.

The keeper, scratch complement, current-best command, and three-revision
command history remain unchanged. This train-only failure does not justify a
best-command revision.
