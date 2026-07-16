# TRKH 5-Class Supervised-Minority Condition-View A1 Numeric Protocol

Date locked: 2026-07-16

Status: immutable train-only Stage A1 infrastructure correction. Validation
and test are forbidden until every inherited Stage A gate passes. This
document authorizes exactly one A1 and no sweep.

## Authority And Inheritance

A1 incorporates every scientific and behavioral clause of
`TRKH_5CLASS_SUPERVISED_MINORITY_CONDITION_VIEW_READINESS_PROTOCOL_20260716.md`
at SHA-256
`d4fb528cc45dbcfbbdfd0218971cde05e7ba203b3ce0ad29f6080c3f561b4ce9`,
except for the two numeric-cache clauses explicitly replaced below.

The method authority is the accepted CVPR 2025 paper served by the Computer
Vision Foundation and its official MIT-licensed implementation:

- https://openaccess.thecvf.com/content/CVPR2025/html/Mildenberger_A_Tale_of_Two_Classes_Adapting_Supervised_Contrastive_Learning_to_CVPR_2025_paper.html
- https://github.com/aiforvision/TTC
- Official local commit:
  `0b1e6974254993b074ad27a226c7ce864da7f95c`.
- Official local tree:
  `707c9b34ece30ac4a23f1de28be9d0ee803710b7`.

No user-supplied or secondary research report is an authority for this A1.

## A0 Failure Being Corrected

The immutable A0 completed deterministic clean FP16 extraction but failed its
CIDT gate with maximum probability error `0.014133721590042114` and one
argmax mismatch. Train-only isolation proved:

- both model-construction paths and all 185 state tensors are bit-exact;
- FP32 matches CIDT within `5.960464477539063e-08` over the 64 smallest-margin
  clean rows with zero decision mismatch;
- FP16 changes sample `2589` because its class-0/class-1 probabilities round
  to an exact tie;
- locked CIDT keeper probabilities were produced in FP32.

The A0 closure document is
`TRKH_5CLASS_SUPERVISED_MINORITY_CONDITION_VIEW_A0_NUMERIC_CLOSURE_20260716.md`.
A0 did not reach selectivity or training, so A1 is an infrastructure replay
correction rather than a method or hyperparameter retry.

## Replacement Numeric-Cache Clauses

These clauses replace only the A0 sentences that require FP16 cache extraction
and a 32-row immediate replay:

- Frozen-keeper clean/dim/bright/low-contrast cache extraction runs in FP32
  with autocast disabled, matching the locked CIDT forward path.
- Extraction batch size is exactly `64`, matching the CIDT runtime batch.
- Each cache still stores FP32 copies of exact `features["pooled"]`, raw logits,
  target, and sample index for all ordered 9,215 train objects.
- Clean probabilities must replay all 9,215 locked CIDT clean rows with maximum
  error `<=1e-6` and zero argmax mismatch.
- An immediate first-64-row FP32 replay must match pooled features and logits
  within `1e-6`, with exact sample indices and targets.
- Cache diagnostics must state `fp32_autocast_disabled`; an FP16 cache may not
  satisfy this contract.

The FP16 runtime benchmark, XAI forward path, and deployment replay remain the
inherited A0 resource/deployment checks. This correction changes neither the
deployed architecture nor its runtime claim.

## Unchanged Scientific Contract

The following remain byte-for-byte or semantically identical to A0:

- keeper/data/CIDT/source/current-command hashes;
- source-disjoint fold-0 `7372/1843` fit/holdout cohorts;
- clean/dim/bright/low-contrast condition definitions;
- pre-training k-NN CAC/SAA/selectivity thresholds;
- 256D pooled feature, `256->64->256` zero-up adapter, 128D projector, and
  zero-initialized five-class residual probe;
- `ce_identity`, `standard_supcon`, and `supervised_minority` variants;
- exact Supervised Minority and standard SupCon equations;
- 20-epoch representation schedule and 10-epoch probe schedule, including all
  optimizer, learning-rate, temperature, order, and occurrence hashes;
- clean/shifted precision, F1, recall, true-positive, restricted-false-positive,
  correction/harm, mechanism, XAI, ONNX, runtime, and VRAM gates;
- no raw-data modification, validation/test access, thresholding, routing,
  trainer integration, checkpoint export, or current-best command revision.

## Execution And Stop Rules

- Contract identifier: `a1_fp32_cidt`.
- Output root:
  `runs/audit_supervised_minority_condition_view_a1_fp32_20260716`.
- The output root must be absent or empty before execution.
- The A1 launcher must explicitly select this contract, this document, batch
  64, seed 42, fold 0, four requested workers, and XAI batch 4.
- Formal execution is forbidden until implementation, focused tests, full
  tests, PowerShell parse, preflight, protected hashes, and tracked-worktree
  cleanliness pass from a committed and pushed state.
- Any failed cache/selectivity/mechanism/behavior/XAI/deployment gate stops A1
  and denies validation, test, shared-trainer integration, full train, keeper
  replacement, and current-best command revision.
- A1 may not alter or sweep precision, dtype, batch, loss, ratio, temperature,
  projection/adapter/probe dimensions, residual scale, optimizer, LR, warm-up,
  epoch count, conditions, fold, seed, sampling, threshold, or router.
