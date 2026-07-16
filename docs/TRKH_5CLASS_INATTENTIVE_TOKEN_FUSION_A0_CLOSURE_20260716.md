# TRKH 5-Class Inattentive-Token Fusion A0 Closure - 2026-07-16

## Decision

Close the exact EViT-style inattentive-token fusion A0 after its sole locked
five-epoch train-only pair. The candidate is an active, numerically correct,
parameter-free fusion mechanism, but it improves class-1 recall by widening the
class-1 decision region. It fails the precision, restricted-false-positive,
illumination, tiny/edge, selectivity, and visual gates.

Official validation, test, full training, nearby sweeps, and current-best
command promotion remain forbidden.

## Primary-Source Provenance

- Accepted paper: Liang et al., "Not All Patches are What You Need:
  Expediting Vision Transformers via Token Reorganizations," ICLR 2022
  Spotlight, arXiv 2202.07800.
- Official Apache-2.0 repository: `https://github.com/youweiliang/evit`.
- Locked commit/tree: `97e58f610c51d4b74a070341739e41647dced32c` /
  `7d907e81f4a764972baa96a639a16164af6f08a6`.
- Official `evit.py`, `helpers.py`, license, and paper SHAs are recorded in the
  readiness protocol. The final prospective protocol SHA is
  `d3f8de6fbef7ad5d88ca33e5db71f9c5ca587f32c1ad5ee4797fd400aaf3ee4b`.
- The user-provided research reports were treated only as hypothesis sources;
  they did not define the mechanism, implementation, or gates.

The adaptation deliberately preserves TRKH's selector, prune layers `2,5`,
keep rates `0.85,0.65`, and spatial patch identities. Dropped tokens contribute
one non-spatial, unnormalized class-attention-weighted context token. This is a
TRKH adaptation at post-block prune points, not a claim of exact stock-EViT
architecture equivalence.

## Preflight And Pair Provenance

The formal preflight passed after a hash-locked correction replay proved that
the initial engineering-pruning failure came only from nine BatchNorm buffers
mutated by the backward/resource audit. Pristine replay gave exact first-prune
identity and mean second-prune Jaccard `0.9996279478` with one changed row.

- final preflight summary SHA:
  `acb9cf60c84ed06988a098705c4ea8ebf6aef16a42fa61e286019d42095e5440`;
- visual-review SHA:
  `e02eafc9400724cf6eafbc35050ad81df744fb1b1a948a748cf90766e5049f50`;
- preflight artifact-manifest SHA:
  `e2bcd90d1f4204c27028426ac8a1bdff66aae9d51b5f4fcf355c65e27f081d0`.

The sole pair ran from pushed commit `ed9bb1248fee1f6ad98794f6a431603edba42b40`
on the frozen source-disjoint `yolo_f/train` fold: 7,372 fit rows and 1,843
holdout rows, zero source overlap, no official validation/test. The pair
manifest SHA is
`4b9bce29f4828a91cedf5a3892dae03670c16034c612832c8cf46731f2f9604f`.

Both roles completed exactly five epochs, used identical occurrence records,
contained 7,245,590 parameters, and differed only by fusion false/true.
Architecture traces covered all five classes. Both pruned `256 -> 218 -> 167`
patches. The candidate carried exactly one additional internal context token
after each prune, while the public final token shape remained `[1,174,256]`.

Original checkpoint hashes before compaction:

| Role | best.pt SHA-256 | last.pt SHA-256 |
| --- | --- | --- |
| Control | `9f5f8b33af569b1a20b4abe53e798fbd954077c279e2f5187dbf4802e90281cf` | `c4c70c5ebd8a808048d1fe84b2eee9fded30530e9cec539343d0857022363393` |
| Fusion | `1ace19113cf1a7210771a5af624549f6d6972d5674beab4d1137f205c57e438b` | `1ce3d551247042ad350d0224087d12e29accaf75f9043faada815c0f749987d6` |

## Independent Train-Only Audit

Standard BF16 deployment inference, never trace logits, scored all 1,843 rows
under four conditions:

| Condition | Macro F1 delta | Class-1 F1 delta | Precision delta | Recall delta | Restricted FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | +0.004534 | +0.017105 | -0.027903 | +0.018349 | 13 -> 16 |
| Dim | +0.004258 | +0.000000 | +0.063280 | -0.018349 | 36 -> 26 |
| Bright | -0.004792 | -0.041418 | -0.081720 | -0.027523 | 14 -> 16 |
| Low contrast | +0.005010 | +0.031983 | -0.034839 | +0.027523 | 8 -> 11 |

Clean macro/class-1 F1 improved, but clean class-1 precision fell from
`0.617647` to `0.589744`; restricted false positives increased by three.
Across conditions the mean precision gate failed, and bright illumination
degraded both precision and recall.

The frozen 1,123-row tiny/edge union also regressed:

- macro F1: `0.790212 -> 0.787656` (`-0.002556`);
- class-1 F1: `0.318841 -> 0.305556` (`-0.013285`);
- class-1 precision: `0.578947 -> 0.500000` (`-0.078947`);
- restricted false positives: `8 -> 11`.

Class-1 keeper-TP versus restricted-hard-negative AUROC was only
`0.594496` clean, `0.576064` dim, `0.541667` bright, and `0.609294` low
contrast. The mechanism therefore did not create a reliable conservative
class-1 margin.

## Context And XAI Result

Fusion mechanics were healthy: every row was finite, trace and standard logits
matched exactly, mass replay error stayed below `2.46e-8`, and shifted context
cosine means were `0.957-0.976`. The failure is semantic rather than numeric.

On clean data, weighted dropped-object mass rose from `0.067836` raw to
`0.093073`, but mean effective context outside the bbox was `0.906927`.
Across conditions the outside fraction stayed `0.883-0.917`. Object
perturbation remained more causal than far background, yet the fused context
itself was overwhelmingly outside-object.

All 54 class-1 decision events, eight representative categories, 60 unique XAI
rows, and all 15 hash-locked contact sheets were reviewed. Border/background
lineage dominated most pages, frequently with `out=1.000`. Candidate stem
Grad-CAM foreground mass fell `0.319174 -> 0.285430` (`-0.033744`) and was
often weak or zero on hard cases.

The Grad-CAM path matched standard predictions on only `58/60` rows for each
role. This is an auditor limitation and a failed gate; future auditors must hook
the standard deployment forward directly. It is not needed to reject A0,
because nine independent behavioral/selectivity/tiny-edge gates already fail.

## Auditor Rendering Correction

The first post-pair audit reached rendering after writing complete predictions,
context rows, events, and the frozen cohort, then failed because the reused
preflight overlay expected a preflight-only `second_prune_jaccard` field.
The failed attempt was retained at manifest SHA
`0bb4d7c50880592400dfa61aa1c0d586f1bf5cfa7bc50e75150d278907d5bdce`.

Commit `78d41e15a02d4441daa75793cc4984cd164df96a` marks that field not applicable
for post-train visualization and adds a fail-closed correction replay. The
replay reproduced the four prior quantitative artifacts byte-for-byte before
rendering. Final audit summary, predictions, visual review, and artifact
manifest SHAs are:

- `d7b3d1fed759d5ab29e538b89556f8cecd6a0e82fc0587a1e0d438d2bfc76348`;
- `82b417971925f3b01eeb2cf296505d451435ea7e8f7a479323588bf0fafa1ff4`;
- `500c53746a58b8bbb5b306839099b1540c98adcfc10ed48bb8b0984fc67251c4`;
- `ad9416849425127f037c8a12b67084bc050cb37abf47b170bdeface7d4438b3f`.

## Stop Boundary

- Do not sweep fusion weights, normalization, context-token count, keep rates,
  prune layers, warmup, epochs, optimizer, loss, augmentation, fold, seed,
  threshold, router, pretrained variant, or combination with another closed
  attention family on the current keeper.
- Do not reinterpret the positive clean class-1 F1 delta as progress. It is a
  recall-driven widening that violates the user's agricultural precision
  priority and worsens tiny/edge precision.
- Any future dropped-token route needs a distinct accepted primary source and
  must prospectively prove object-conditioned outside-context rejection before
  training. Simply retaining or fusing discarded context is now closed.
- Official validation/test and current-best commands remain untouched. Tracking
  stays at three revisions, two updates, and zero keeper replacements.

## Evidence Retention

Before binary compaction, both run summaries, histories, launchers, resolved
configs, occurrence hashes, architecture traces, and all four checkpoint hashes
above were recorded. Fail-closed compaction then copied and verified 332
non-binary files (`46,238,611` bytes), excluded exactly the four recorded
checkpoint files (`348,760,480` bytes), deleted only the two rejected pair run
directories, and reclaimed `332.312 MiB` of measured free space.

- compact evidence:
  `runs/evidence_inattentive_fusion_a0_pair_rejected_20260716`;
- compact payload-manifest SHA:
  `7eed039e9cdf0f29cc18aed24a2e96a6c2f63bb438389fcc858bee46a34d0ec6`;
- compact summary SHA:
  `67b7e13cf83c8983f887ed4b41ad34f9bea7ba2f3d84722e3f6a9649d23dfa56`;
- cleanup-manifest SHA:
  `bb03882b13a754afe88cf6c39aea22e31f2dfdc8eacfaf1a72876bc516134626`.

The final audit directory and all 15 XAI pages remain intact. Read-only
retention passed over 728 run directories and 46 object-compaction manifests
with `blockers=[]`, 70.803 GiB free, and summary SHA
`260a34b235ccd30ea6fa637ad6b34b4e64f1a02c220df9e140d0af8378afaf24`.
Keeper/current-best checkpoint, command, history, and final-audit hashes all
remain exact.
