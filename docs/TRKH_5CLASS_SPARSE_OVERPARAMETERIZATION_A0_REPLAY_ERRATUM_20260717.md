# TRKH 5-Class SOP A0 Replay-Contract Erratum - 2026-07-17

## Status

Precommitted after the keeper replay stopped and before any SOP noise-buffer
optimization, SOP row gradient, image epoch, trainer integration, holdout,
validation, or test access. The original readiness protocol remains immutable
at SHA-256
`d2b94048da9d2a3ed261fda4f97c333c51bada3b008bd2303629aefd96b687c8`.

This erratum corrects only the flat-keeper replay constants used to authorize
the SOP computation. It does not change the SOP equation, initialization,
optimizer, number of passes, cohort definitions, gradient thresholds, stop
rules, or any downstream authorization rule.

## Preserved Failure

At pushed repository commit
`5a8b03f1e1b713960ce683a386e9d7559f697819`, the formal launcher completed the
standard FP32 metadata-aware keeper forward over all 7,372 fit rows. Row
identity and targets matched, and all 7,372 predictions matched the hash-locked
CIDT declaration. The auditor then stopped before `u/v` initialization because
the resulting confusion matrix did not match constants copied from the earlier
NBDT development scan.

No original output directory was created because the fail-closed replay check
preceded output creation. A compact reconstructed failure record is preserved
under
`runs/evidence_sop_a0_replay_contract_mismatch_20260717`:

- `summary.json` SHA-256:
  `1dde47f0f222f35272837b721932872e64bb25c279266025a30d543ced971638`;
- `artifact_manifest.json` SHA-256:
  `d9570f2d1d13a767837ca8d4817eb389055ab32a371e8e2665cc0f8a93a9dea1`.

The completed fit inference took `303.7133035` seconds. A separate initial
launcher call was killed by a one-second command timeout before an output
directory or surviving process existed; it produced no measurement and is
also declared in the evidence record.

## Diagnosis

The original constants were:

```text
accuracy / macro F1                     0.960662 / 0.937980
class-1 precision / recall / F1         0.690671 / 0.976852 / 0.809204
class-1 TP / FP / restricted FP         422 / 189 / 185
```

They came from the prior NBDT development scan, which used a different batch
execution context. Three near-boundary class decisions differ from the formal
CIDT/deployment batch-64 path. The NBDT collapse conclusion is unaffected
because its soft tree predicted class 4 for every row, but its flat metrics
must not be reused as the canonical batch-64 SOP baseline.

The authoritative replay evidence is the conjunction of:

1. keeper checkpoint SHA-256
   `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`;
2. declaration SHA-256
   `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`;
3. ordered fit-index SHA-256
   `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d`;
4. exact `7,372/7,372` declaration-prediction agreement from the completed
   formal FP32 batch-64 deployment replay; and
5. an independent metric recomputation from the locked declaration.

## Corrected Replay Contract

The corrected confusion matrix, rows=true and columns=predicted, is:

```text
[[1416, 135,    5,    0,    5],
 [  10, 421,    0,    0,    1],
 [   4,  42, 1454,   22,    5],
 [   0,   4,   33, 1978,    2],
 [  12,   9,    2,    0, 1812]]
```

Corrected exact metrics are:

```text
accuracy                               0.9605263157894737
macro F1                               0.9375834511834891
class-1 precision                      0.6890343698854338
class-1 recall                         0.9745370370370371
class-1 F1                             0.8072866730584852
class-1 TP / FP                        421 / 190
restricted {0,2,4}->1 FP              186
```

The corrected formal replay is fixed to FP32 inference batch size `64`, seed
`42`, and the same standard transform/metadata forward. Worker count may only
affect throughput, but the launcher remains at `4`. It must require exact
target order, exact confusion matrix, and exact agreement with every locked
declaration prediction before initializing `u/v`.

## Corrected Run Rule

After the erratum and auditor correction are committed and pushed, exactly one
replacement formal run is authorized at a new path:

`runs/audit_sparse_overparameterization_a0_corrected_replay_20260717`

The old absent output path must not be reused. The replacement must reproduce
the corrected replay contract above. If it does not, the auditor must preserve
a failure artifact and stop before SOP.

If replay passes, the original Phase-A mechanism gate proceeds unchanged:

- ten deterministic natural-frequency passes, batch `128`;
- seed `42`, `u/v ~ N(0,1e-8)`;
- SGD `lr_u=0.1`, `lr_v=1.0`, no momentum or weight decay;
- at least `1%` effective-V active rows;
- all `186` restricted false positives retain a positive class-1 corrective
  gradient, median magnitude ratio at least `0.95`, and at most `5%` lose more
  than half magnitude;
- all `421` class-1 true positives retain a negative support gradient under
  the same magnitude thresholds;
- no class-1 clamp-to-zero gradient, preferential restricted-FP noise
  absorption, invalid probability, parameter-domain violation, or collapse.

Any mechanism failure still closes SOP A0 without a learning-rate, buffer,
loss, pass, epoch, seed, fold, SGN, or Label-Wave rescue. Passing still does not
authorize validation, test, full train, or a current-best command update.
