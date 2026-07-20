# TRKH Validity Partial-Conv Stem Closure

Date: 2026-07-20

Decision: reject the exact three-block validity-partial stem after the sole
matched keeper adaptation smoke. Do not run post-smoke robustness, additional
XAI, a five-epoch probe, test, full train, or current-best command promotion.

## Locked Pair

- Stage A passed all 32 readiness checks and authorized only the precommitted
  pair. Pair infrastructure was committed/pushed at `e259974`; the audit replay
  correction was committed/pushed at `59c82ce`.
- Both roles resumed the same keeper, reset epoch/optimizer/scheduler/scaler,
  used seed `42`, `120` train batches for each of two epochs, scheduler horizon
  `10`, batch/accumulation `32/2`, LR `8e-5`, and workers `4/2`.
- The sole model difference was `standard` versus `validity_partial`. The two
  occurrence-hash files are byte-identical at SHA
  `da5417d9...92be`; both contain `3,840` balanced occurrences per epoch with
  exact epoch hashes.
- Full validation contains 2,606 rows with support `[549,151,544,712,650]`.
  Test was never opened and raw-data identity remained
  `a7caeccb...5fb6` before/after.

## Validation Result

| Class | Control P/R/F1 | Candidate P/R/F1 | F1 delta |
|---|---:|---:|---:|
| 0 | 0.9454 / 0.8834 / 0.9134 | 0.9235 / 0.8579 / 0.8895 | -0.0239 |
| 1 | 0.6091 / 0.7947 / 0.6897 | 0.5204 / 0.7616 / 0.6183 | -0.0714 |
| 2 | 0.8803 / 0.9467 / 0.9123 | 0.8289 / 0.9081 / 0.8667 | -0.0456 |
| 3 | 0.9864 / 0.9157 / 0.9497 | 0.9338 / 0.8919 / 0.9124 | -0.0374 |
| 4 | 0.9615 / 0.9615 / 0.9615 | 0.9416 / 0.8677 / 0.9031 | -0.0584 |

- Accuracy falls `0.919800 -> 0.874520`; macro F1 falls
  `0.885324 -> 0.837989` (`-0.047336`). Candidate epoch 2 is also below
  control (`0.857715` macro and `0.623656` class-1 F1), so the selected
  epoch-1 result is not hiding a late recovery.
- Candidate changes 206 decisions: `37` corrections versus `155` harms. It
  removes/creates `20/46` restricted class-1 false positives (net `-26`),
  rescues/breaks `5/10` class-1 FN/TP, and lowers class-1 precision in all five
  adequate source folds by `0.057-0.123`.
- Candidate beats the source/mask-deranged placebo by `+0.057868` class-1 F1,
  `+0.062379` class-1 precision, and 25 net restricted-FP removals. Therefore
  aligned validity geometry carries signal relative to a wrong mask, but the
  exact adaptation remains substantially worse than the standard control.
- Candidate Brier/NLL worsen from `0.595517/1.174075` to
  `0.630086/1.240634`. Lower ECE is accompanied by lower mean confidence and
  does not offset the accuracy/F1 regression.

## Provenance And Audit Correction

- The first audit completed every inference and wrote summary SHA
  `a632bbcf...17b3`, then stopped before its manifest because JSON key sorting
  reordered the same `failed_checks` set. The correction sorts that list,
  recognizes the two nested role-specific cartography output paths, and records
  probability and argmax replay separately. No threshold or scientific input
  changed; correction protocol SHA is `5fee986e...33a`.
- Corrected internal and external gate replay are exact. Control/candidate
  independent argmax are exact, but BF16 probability errors
  `0.034772/0.030549` exceed the prospectively locked `5e-4`, so both replay
  probability checks correctly remain failed.
- Eleven conjunctive checks fail: independent probability replay, macro/class-1
  F1, class-1 precision/recall, TP safety, corrections-versus-harms,
  non-focus preservation, restricted FP reduction, and source-fold stability.
  Runtime/VRAM pass at `1.020118x` and `6.071 GiB` candidate peak.
- Corrected summary/manifest/internal-replay/external-replay/gate-input SHAs are
  `7505bac2...4b7`/`56eaa9f8...bd5`/`d24db3b6...276`/
  `3b827a78...8ac`/`5631f7e9...9fa`. Corrected status/top-manifest SHAs are
  `1326d0d8...f39`/`5629bcf8...7dc`.

## Trace Interpretation

- All five input/preprocessing image pairs are pixel exact. All five stem,
  patch-embedding, attention-score, and Transformer token maps change; mean
  rendered MAE is `2.2735` at the stem, `4.9083` at patch embedding, and
  `5.10-7.51` across blocks 1-8.
- Attention-score MAE is only `0.7246`; pruning images are exact for two of five
  classes and change only `0.625%/0.469%` of pixels on average at the two prune
  stages. The mechanism therefore amplifies global token-state drift without a
  corresponding selective foreground/pruning improvement.
- The fixed class-1 trace increases its first pairwise route weight
  `0.2128 -> 0.3282`, consistent with the observed expansion of class-1 support
  into restricted negatives. This is single-sample explanatory evidence, not a
  causal aggregate estimate.

## Retention And Boundary

- Preserve both replayable `best.pt` checkpoints at SHAs
  `b37e3fc3...bd96`/`4c504735...a462`, both validation CSVs, all 240 trace
  PNGs, both occurrence hashes, the incident, and corrected audit.
- Delete only the two rejected optimizer-bearing `last.pt` files after hash
  verification, freeing `232,337,412` bytes (`221.57 MiB`). Cleanup manifest
  SHA is `b97b5160...c27`.
- Read-only retention passes over 790 run directories and all 50 compaction
  manifests; all 219 compacted originals remain absent, `deleted_anything=false`,
  and `blockers=[]` at summary SHA `cd49f50b...035`.
- Pycompile, pyflakes, PowerShell parse, focused tests `26/26`, and full pytest
  `1602/1602` pass after the replay correction.
- Close epsilon, mask-update, pooling, layer-subset, fill, normalization, LR,
  epoch, seed, threshold, and partial-conv-plus-attention variants on this
  keeper. Current-best checkpoint/command/history hashes remain
  `1f49d577...2677`/`36b9aa1a...0faf`/`39bd2879...8f53`.
