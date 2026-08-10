# TRKH 5-Class NSA Replay Geometry-Order Diagnosis - 2026-07-24

## Scope

This diagnosis explains the failed `all_geometry_hashes_exact` check in:

`runs/audit_nsa_class_pair_poisson_a0_formal_v2_20260724/replay.json`

It does not change the formal summary, replay, prospective thresholds, manual
decision, or authorization. The NSA A0 remains rejected.

## Observation

The fresh replay reproduced:

- all readout actions exactly;
- every readout score within `2.220446049250313e-16`;
- metrics, query diagnostics, aggregate diagnostics, and fold mismatch maps
  exactly;
- all 266 fixed visual arrays exactly;
- every saved adapter state and the keeper state exactly.

Two replay checks failed:

1. peak virtual-memory fraction was `0.899`, above the locked `<0.82`;
2. all five ordered geometry-record hashes differed.

## Root Cause

The geometry records themselves are deterministic. Their container order is
different:

- formal training appends each batch as
  `[all same][all cross0][all cross1]`;
- `replay_training_geometry` appends each sample as
  `[same,cross0,cross1]`.

For every batch, the diagnosis transformed only the already saved formal
record order:

```text
formal: same[0:b] + cross0[0:b] + cross1[0:b]
replay: for i in 0:b -> same[i], cross0[i], cross1[i]
```

No image, geometry, model, dataset, adapter, or metric was recomputed.

| Fold | Records | Reordered formal SHA-256 | Replay SHA-256 | Match |
|---:|---:|---|---|:---:|
| 0 | 12,960 | `3debf8b774c7b316db493f8cbcc7fa071b006ad4535175657834c7e8df9faddd` | `3debf8b774c7b316db493f8cbcc7fa071b006ad4535175657834c7e8df9faddd` | yes |
| 1 | 12,780 | `e3d324af3bc00f83d9c54171df55474404160961cc83ac1559a11f8735c5efe5` | `e3d324af3bc00f83d9c54171df55474404160961cc83ac1559a11f8735c5efe5` | yes |
| 2 | 13,110 | `e165a57866c35792888d28426b83039b8b79c19a0a3a2303b4cb0f37084f25a0` | `e165a57866c35792888d28426b83039b8b79c19a0a3a2303b4cb0f37084f25a0` | yes |
| 3 | 13,140 | `006c1f02b6ef27750fee4ee267a8ecd15ac151e0f3d56af90bc6bf61990b1922` | `006c1f02b6ef27750fee4ee267a8ecd15ac151e0f3d56af90bc6bf61990b1922` | yes |
| 4 | 12,930 | `33cd9d47154cfc9fec68d404d59da1ae0e11b78a4d6fca27cb39e061307ac0e9` | `33cd9d47154cfc9fec68d404d59da1ae0e11b78a4d6fca27cb39e061307ac0e9` | yes |

All `5/5` reordered hashes equal the hashes independently reported by replay.
The failed ordered-hash check is therefore a checker-order contract defect,
not evidence of geometry nondeterminism.

## Decision Boundary

The replay file stays failed because its prospective ordered-hash and resource
checks failed. No rerun or post-result relaxation is authorized. This diagnosis
only prevents the failure from being misreported as changed synthetic pixels.
The candidate independently fails AUROC, localization, class-1 TP protection,
causal-control, and correction-to-harm gates, so the final scientific decision
is unchanged.

Evidence SHAs:

- formal summary:
  `576c061fd78d14f1cf848a32d40437a510a3c0222a853aa0aefac9a9cac37e13`;
- replay:
  `ea6ed9630c1bb84a3fc7fa4c68210942a85f620d17194d2d26c5b8aa6a10fdf4`;
- final visual review:
  `93a4da577a7c5e0cd34fd9028398f875d96d60a899926e65a050e0b222e46f89`;
- final artifact manifest:
  `9b978d3abc0ea186a7ccdb1e9881dc7108717709b046c451670a41eb999aabc6`.
