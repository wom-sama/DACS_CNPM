# TRKH 5-Class Supervised-Minority A0 Numeric Closure

Date: 2026-07-16

Status: closed before method evaluation. This is an infrastructure-contract
failure, not a behavioral rejection of Supervised Minority. Validation, test,
shared-trainer integration, full train, keeper replacement, and current-best
command revision remain unauthorized.

## Formal A0 Outcome

- Protocol SHA-256:
  `d4fb528cc45dbcfbbdfd0218971cde05e7ba203b3ce0ad29f6080c3f561b4ce9`.
- Precommit implementation commit:
  `35f767b1317bc69eab1a8fc13b500c9be2f82927`.
- The formal run completed the ordered 9,215-row clean FP16 cache and an
  immediate 32-row replay. Pooled features, logits, sample indices, and targets
  replayed exactly.
- The next mandatory CIDT provenance gate failed: maximum clean probability
  error was `0.014133721590042114` with one argmax mismatch, versus locked
  limits `0.002` and zero mismatches.
- The run stopped before shifted-condition extraction, pre-training
  selectivity, representation/probe training, XAI, or deployment export. The
  formal output root was never created.

## Root-Cause Isolation

The diagnostic used train data only and did not create an artifact directory.

- `create_model` plus strict state load and the normal inference
  `build_model_from_checkpoint` path both constructed
  `VisionTransformerWithRegisters` with 185 state tensors. Every state tensor
  was bit-exact and both paths produced identical FP32 and FP16 logits.
- On the 64 CIDT clean rows with the smallest top-two margins, both FP32 model
  paths matched locked CIDT probabilities within `5.960464477539063e-08` and
  had zero argmax mismatches.
- Both FP16 paths had one mismatch in that cohort. Sample index `2589` has CIDT
  probabilities `p0=0.24788621068000793` and
  `p1=0.24808543920516968`, so CIDT predicts class 1 with margin
  `0.00019922852516174316`. FP32 reproduces that decision within
  `1.4901161193847656e-08`; FP16 rounds both values to
  `0.24801132082939148`, and `argmax` resolves the tie to class 0.
- CIDT generated its locked keeper probabilities in FP32 without autocast.
  A0 instead required an FP16 cache to reproduce those FP32 decisions. The
  all-row zero-mismatch requirement was therefore internally infeasible for
  this near-tie checkpoint.

## Decision

A0 remains immutable and closed. Its gate is not loosened and it is not rerun.
Because no method update occurred, this result does not activate the A0
no-repeat boundary against Supervised Minority itself. Exactly one corrected
A1 may proceed under a separately hashed protocol that matches the CIDT FP32
numeric path. A1 may not change the paper objective, adapter, schedules,
cohorts, conditions, behavioral gates, or validation/test prohibition.

Current-best command tracking remains three revisions, two updates after the
initial revision, and zero keeper replacements.
