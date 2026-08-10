# TRKH Validity Partial-Conv Stage-B Audit Correction

Date: 2026-07-20

Status: locked after the first complete matched training/evaluation pair and
before a corrected audit is produced. No train, validation prediction, metric
threshold, placebo rule, source fold, checkpoint, or raw-data artifact may be
changed by this correction.

## Incident Boundary

- The pair launched from clean synchronized commit `e259974` and completed both
  two-epoch/120-batch runs plus both full 2,606-row validation evaluations.
- Control/candidate ordered-occurrence files are byte-identical at SHA-256
  `da5417d9930120c3349e94336219d903f84b76053ede0db548ff2c448a1492be`.
- Control/candidate checkpoint SHAs are
  `b37e3fc3802cc54a46640a8097ce9ceb7c4ee25d478a88a464ec9f58b6acbd96`
  and
  `4c50473579cc629f607d8883913262c56d715c79e966f5f838ebea52a836a462`.
- Control/candidate detailed-validation CSV SHAs are
  `00e241e5f15ab2cfb01b9ff230dab18144983c7fb9c23e1547187b798d18078b`
  and
  `8c9b7e466b7dde9db7c913cced1aff67c506b66b4db157ebcc8deef359336e35`.
- The first audit wrote summary/gate-input SHAs
  `a632bbcfcfea819203744cd543dd06a46ce8a78c4fef98959543158017d417b3`
  and
  `f4c57ff82f635c52a095c696fdfb961a56bb6847253a96eb6d43c48c51a80d71`,
  then stopped before its manifest because strict replay compared the ordered
  `failed_checks` list.

## Root Causes And Fixed Scope

1. `gate_inputs.json` is emitted with sorted mapping keys. Replaying the same
   false checks therefore produced the same set of failures in another list
   order. Sort `failed_checks` in the pure assessor so semantic and serialized
   replay are identical.
2. The resolved config mirrors the two role-specific cartography paths under
   both `train_config` and `data.data_cartography`. Add the two nested output
   paths to the already intended artifact-path allowlist. No model, data,
   optimizer, augmentation, loss, or scheduler field is allowed.
3. Persist independent argmax equality separately from the prospectively locked
   probability threshold. The observed BF16 replay error remains subject to
   `5e-4`; this correction does not relax it.

## Corrected Audit Rule

- Reuse the exact checkpoints, validation CSVs, run summaries, histories,
  occurrence hashes, traces, Stage-A evidence, and locked protocol above.
- Write corrected output to `comparison_corrected`; preserve the first failed
  audit directory read-only as incident evidence.
- Run the same independent 2,606-row control/candidate/placebo inference with
  batch size `32`, workers `2`, device `cuda`, and no test access.
- Require internal and fresh external-process gate replay to be exact. A failed
  scientific or provenance check still closes the route and never authorizes a
  probe, test, full train, or current-best command update.
