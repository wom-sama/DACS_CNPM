# SIFER Feature-Sieve A0 Replay Erratum - 2026-07-20

## Scope

The sole formal SIFER A0 run completed from pushed commit `b732d72` and wrote
the immutable pre-review summary with SHA-256
`883a87d57eecbfa730cce4183e738e5b7ffa7c52a824890feb95463e958ce65d`.
The launcher then stopped in independent replay before visual finalization.
No formal rerun, validation/test access, trainer integration, checkpoint, or
current-best command update is authorized by this erratum.

## Numerical mismatch

The formal audit computes the stem mean absolute difference by accumulating
FP32 tensor sums per batch and dividing by the exact element count. The replay
computes the mathematically equivalent mean of serialized per-row FP32 means.
All rows have the same stem shape. The two values are:

- formal: `0.004924195830225621`;
- replay: `0.004924195810138767`;
- absolute difference: `2.008685405507915e-11`;
- FP32 ULP at this magnitude: `4.656612873077393e-10`.

Both values map to the same FP32 number. A diagnostic in-memory replay using
`1e-10` reproduced all 1,843 clean rows, 5,529 illumination rows, metrics,
gate decisions, forget sequence, contact-sheet hashes, and summary artifact
bindings. It wrote no change to the immutable summary or source CSV/history.

## Correction

Only `mechanism.stem_mean_absolute_difference` receives an absolute replay
tolerance of `5e-10`, approximately one FP32 ULP. Every other numeric replay
comparison remains at `1e-12`; paths, hashes, row counts, decisions, booleans,
and sequences remain exact. Regression coverage requires the observed
same-FP32 pair to pass, a `6e-10` stem discrepancy to fail, and a `2e-12`
non-stem discrepancy to fail.

The corrected replay must run against the original summary SHA above and pass
before any visual decision. Because the automatic gate already contains nine
failed checks, this correction cannot convert SIFER A0 into an automatic pass.
