# Hamburger-NMF Surface A0 Closure (2026-07-20)

## Decision

Reject the exact frozen-keeper Hamburger/NMF surface descriptor. It removes
more restricted class-1 false positives than the controls only by breaking too
many true class-1 predictions, is unstable across the locked NMF seed, and is
indistinguishable from source-deranged surface information. Do not run Stage B,
integrate this branch, open validation/test, launch a smoke/probe/full train, or
update current-best commands from this result.

## Locked Scope

- The accepted source is Geng et al., ICLR 2021. Paper SHA is
  `4eed8898...ae9f1`; the authors' GPL-3.0 repository is pinned read-only at
  commit/tree `d9b51f6...57c08`/`a399506...02c2e`. No GPL source was copied.
- TRKH independently implements the paper equation and checks it with a
  separately written NumPy oracle and BSD scikit-learn `1.6.1` reference.
- The final resource-amended protocol SHA is `0bc33ac9...d7996`.
  Infrastructure and the prospective RAM/worker erratum were pushed at
  `e6c0dc6` and `bee1503` before candidate metrics.
- The fixed cohort is 750 train-only rows: 528 keeper true positives and 222
  restricted target-`0/2/4` false positives over five source-disjoint folds.
  Validation and test were unused.

## OOF Result

| Role | AUROC | TP retention | FP rejection | FP rejects | Corrections | Harms |
|---|---:|---:|---:|---:|---:|---:|
| Base only | 0.831900 | 0.965909 | 0.229730 | 51 | 46 | 18 |
| Raw surface | 0.793927 | 0.937500 | 0.279279 | 62 | 56 | 33 |
| Rank-8 SVD | 0.743158 | 0.888258 | 0.297297 | 66 | 58 | 59 |
| Rank-8 NMF candidate | 0.769127 | 0.876894 | 0.373874 | 83 | 77 | 65 |
| NMF seed repeat | 0.744514 | 0.850379 | 0.351351 | 78 | 72 | 79 |
| Source deranged | 0.769554 | 0.876894 | 0.355856 | 79 | 72 | 65 |

Candidate AUROC deltas versus base/raw/SVD/source-deranged are
`-0.062773/-0.024800/+0.025969/-0.000427`. It wins raw and SVD together in
only `1/5` folds. It rejects `56/24/3` target-`0/2/4` false positives from
supports `158/54/10`, but breaks 65 of 528 true class-1 rows. Aggregate and
minimum-fold TP retention are only `0.876894/0.831683`, below the locked
`0.95/0.90` floors.

Ten of 21 mechanism gates fail: absolute AUROC, gains over base/raw/deranged,
TP retention, correction-to-harm ratio, fold consistency, minimum-fold TP
retention, seed action agreement, and seed AUROC stability. The extra 17 FP
rejects over the strongest control cannot compensate for the recall loss.

## Mechanism Diagnosis

- Candidate effective rank is healthy at `34.454303`; all 284 dimensions are
  active. This is not descriptor collapse.
- Median two-seed descriptor cosine is `0.972960`, but the minimum is
  `0.811616`. Seed-repeat AUROC differs by `0.024613` and decisions agree only
  `0.809333`, so the low-rank factorization is not decision-stable.
- Candidate and source-deranged AUROCs differ by only `-0.000427`, with the
  same aggregate TP retention and harms. The learned readout is exploiting
  broad surface statistics without a source-specific precision signal.
- NMF beats the rank-8 SVD control but loses both raw tokens and keeper logits.
  Nonnegativity therefore changes the representation without adding the
  required class-conditional evidence.
- Close rank, iteration, initialization, projection, component ordering,
  descriptor subset, logistic `C`, threshold, seed voting, and nearby
  post-hoc NMF/Hamburger variants on this keeper. A future method must learn a
  stable supervised precision representation rather than factorize frozen
  final tokens with an unsupervised per-image objective.

## Replay And XAI

- Production/NumPy decomposition error is at most `9.09e-13`; analysis and
  applied actions match exactly. The prelocked `1e-10` score/state tolerance
  still fails at `2.23e-9` score and `4.92e-10` readout-state error. Preserve
  this structural miss; do not relax the tolerance after metric access.
- CIDT replay retains exact argmax with maximum probability error
  `2.31564e-5`, below the locked `3e-5` bound. Model state is bit-identical.
- The 12-row contact sheet is manually failed. Rows `58` and `4012` render as
  extreme thin strips, and several supports include hand/background regions.
  The finite/no-padding tensor checks pass, but the visual object-alignment
  requirement does not.
- Final summary/manifest/sheet SHAs are
  `0a0cf04d...e110a`/`79e37691...83280`/`1c61c4eb...618a6`.

## Resources And Retention

- Requested/effective workers are `2/2`, CUDA batch is `64`, clean extraction
  takes `26.124 s` at `28.709` source images/s, and peak allocation is
  `1,202,673,664` bytes. Focused/full tests pass `15/15` and `1659/1659`.
- Preserve all 11 formal files (`18.57 MiB`) in
  `runs/audit_hamburger_nmf_surface_a0_20260720`; there is no checkpoint,
  optimizer, smoke, or probe artifact to remove.
- Read-only retention covers 798 run directories and all 50 valid compaction
  manifests. All 219 compacted originals remain absent,
  `deleted_anything=false`, and `blockers=[]` at summary SHA
  `81e40d26...f5af`.
- Keeper/current-best command/history hashes remain
  `1f49d577...2677`/`36b9aa1a...0faf`/`39bd2879...8f53`. Stage B, trainer,
  smoke, probe, validation/test, full train, and command promotion remain
  unauthorized.
