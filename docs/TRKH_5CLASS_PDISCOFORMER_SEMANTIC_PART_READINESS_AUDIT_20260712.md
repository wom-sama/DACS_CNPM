# TRKH 5-Class PDiscoFormer-Style Semantic-Part Readiness Audit (2026-07-12)

## Decision

Reject before image-model smoke. The official-form semantic regularizers make
the frozen keeper part maps smoother, sharper, and more rotation-equivariant,
but they collapse foreground utilization and materially reduce direct class-1
recall. No test split, raw-data edit, integrated trainer smoke, checkpoint, or
current-best command update was permitted.

## Primary-source review and scope

- PDiscoFormer paper: <https://arxiv.org/abs/2407.04538>.
- Official MIT repository: <https://github.com/ananthu-aniraj/pdiscoformer>.
- Inspected local official commit:
  `1a872e2bed2ea38c4b078fd69294126e9f6b2f33`.
- Inspected official model, trainer, and loss implementations before writing
  TRKH code. The paper recipe assigns every patch to `K` foreground parts plus
  background using negative squared prototype distance, then combines part
  dropout, shared per-part classification, presence, equivariance,
  orthogonality, total variation, background-presence, and pixel-entropy loss.
- The reported recipe depends on a frozen pretrained DINOv2 representation.
  The current TRKH keeper is trained from scratch. This diagnostic therefore
  tests readiness of the keeper representation; it is not a reproduction or a
  novelty claim.
- This is distinct from the rejected `AdaptivePartTokenLearner`: the old head
  used four foreground/bbox-biased attention pools and pairwise residuals but
  no competing background slot or combined semantic-part objective.

The locked protocol was written before implementation in
`docs/TRKH_5CLASS_PDISCOFORMER_SEMANTIC_PART_READINESS_PROTOCOL_20260712.md`.

## Fixed diagnostic

- Data: only `yolo_f/train` and `yolo_f/val`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper forward: FP32, exact final post-pruning tokens, grid `16x16`, retained
  token shape `167x256`; token caches were FP16 and shared by both arms.
- Views: unchanged tensor and exact clockwise 90-degree tensor rotation. Bbox,
  valid mask, and patch indices were transformed back to the original frame;
  grid-valid mismatch was `0` for all rows.
- Head: two foreground parts plus one background part, temperature `1.0`, soft
  Gumbel assignment in training, deterministic softmax in evaluation, part
  dropout `0.30`, shared bias-free classifier.
- Matched control: identical parameters, initialization, data order,
  stochastic original-view seeds, CE, Adam, StepLR, batch 64, and 12 epochs;
  semantic losses were the only difference.
- Candidate weights followed the official form: presence/equivariance/
  orthogonality/TV/entropy `1`, edge-background `2`, CE `1`.
- Five `StratifiedGroupKFold` folds grouped by source stem, then one final pair
  trained on all train rows. No early stopping, validation-selected epoch, or
  hyperparameter sweep was used.
- The keeper backbone was trained on all train rows. Head OOF is therefore
  source-grouped head-OOF over an in-sample keeper representation, not a true
  OOF keeper backbone.

## Engineering review

Two preflight issues were caught before the full run:

1. The keeper class does not expose `forward_heads`; the tool now uses the
   repository-standard `classification_logits_from_features` fallback.
2. `Subset` metadata carried empty path strings. Without the established
   `dataset.sample_paths()` fallback, all 128 preflight rows collapsed into one
   source group. The invalid 29-file/`44,136,781`-byte cache was inventoried,
   hashed, and deleted under manifest SHA `6e26844e...4ed6`; a regression now
   preserves subset path order and the extractor fails closed on empty paths.

The successful 128/128 contract preflight had `127/127` source groups, zero
cross-split/fold overlap, exact `16x16` rotation alignment, finite gradients,
matched initial-state hashes, and 33 independently replayed payload hashes.
Capped metrics were not used to approve or reject the method.

## Full support and direct metrics

- Rows: `9215 train / 2606 validation`.
- Source groups: `8064 / 2577`; train/validation and all fold overlaps are `0`.
- Class counts: train `1941,541,1920,2520,2293`; validation
  `549,151,544,712,650`.
- Direct FP32 keeper validation was reproduced exactly at macro/class-1
  `0.8829248142 / 0.6782608696`, class-1 P/R
  `0.6030928 / 0.7748344`.

| Split/arm | Macro F1 | Class-1 F1 | Class-1 P | Class-1 R |
|---|---:|---:|---:|---:|
| Train head-OOF control | 0.882467 | 0.619691 | 0.648485 | 0.593346 |
| Train head-OOF candidate | 0.865934 | 0.574402 | 0.657143 | 0.510166 |
| Validation keeper | 0.882925 | 0.678261 | 0.603093 | 0.774834 |
| Validation control | 0.846990 | 0.560606 | 0.654867 | 0.490066 |
| Validation candidate | 0.812924 | 0.477733 | 0.614583 | 0.390728 |

Candidate class-1 F1 exceeded control only in fold 1 (`1/5`). Fold candidate
class-1 F1 was `0.64681, 0.59119, 0.50617, 0.59898, 0.50962`; matched control
was `0.65812, 0.57609, 0.52121, 0.63256, 0.67227`.

## Direction and transition audit

- Candidate versus control OOF: `408` changes,
  corrections/harms/neutral `134/223/51`, class-1 FN rescue/TP break `30/75`,
  FP remove/create `73/43`.
- Candidate versus control validation: `164` changes,
  `41/103/20`, FN rescue/TP break `7/22`, FP remove/create `14/12`.
- Candidate versus keeper validation: `265` changes,
  `66/172/27`, FN rescue/TP break `2/60`, FP remove/create `52/12`.
- Candidate-minus-control p1 direction AUROC transferred above chance at
  `0.621852 -> 0.571047`, but both candidate and control directly lose too much
  recall. This AUROC does not authorize a router, residual, threshold, sample
  weight, or margin policy.

## Semantic-part and visual audit

The regularizers did optimize their structural targets:

| Validation statistic | Control | Candidate |
|---|---:|---:|
| Foreground part masses | 0.38005 / 0.20030 | 0.05334 / 0.12963 |
| Normalized part-use entropy | 0.92965 | 0.87066 |
| Background inside bbox | 0.38622 | 0.79041 |
| Background outside bbox | 0.52603 | 0.90175 |
| Foreground equivariance cosine | 0.71559 | 0.83517 |
| Total variation | 1.02609 | 0.33383 |
| Pixel entropy | 0.04341 | 0.01001 |

The apparent background-gap pass is misleading: candidate background rises
both outside and inside the fruit. Total candidate foreground mass is only
`0.18297`; part 0 fails the locked minimum at `0.05334`.

The 14-row full-resolution review covers every class, keeper class-1 TP/FN/FP,
candidate corrections, and candidate harms. Part 0 repeatedly contracts to the
lower silhouette, hands, crop edges, or sparse residual patches. Part 1 often
covers a broad color region or most of the fruit. Corrections and harms share
the same broad-color/silhouette assignments; no repeatable class-1-positive
surface part appears. Manual verdict is `fail`, recorded under contact-sheet
SHA `918057cd...b3423`.

## Gate and no-repeat decision

Ten locked checks fail: candidate OOF and validation macro/class-1 versus
control, class-1 fold wins, candidate macro/class-1 versus keeper,
corrections-versus-harms, FN-rescue-versus-TP-break, and foreground part mass.

Do not sweep `K`, loss weights, transform, temperature, selected feature layer,
token-pruning mode, optimizer, LR, epochs, part dropout, Gumbel hard/soft mode,
background prior, residual, or router on this keeper. Reopen semantic part
discovery only after a new representation independently supplies stable
foreground class-1-positive support; do not use prettier maps as evidence.

## Integrity, cleanup, and closure

- Independent reconstruction from train/validation CSVs exactly reproduced
  all direct metrics, five folds, transitions, and AUROC without importing the
  diagnostic metric helpers.
- All 26 cache payload records (`2,041,168,926` bytes) were replayed before
  deletion. Keeper and current-command hashes remained
  `1f49d577...482677` and `3c718130...57dd`.
- Full compact evidence has 13 top-level payloads/`7,284,956` bytes, artifact
  manifest SHA `4bf77a95...97ed5`, and cache-cleanup manifest SHA
  `2be94ebd...630d`. The cleanup removed 29 cache files/`2,041,187,420` bytes
  with `2,041,257,984` observed free-byte gain.
- Successful preflight compact evidence has 11 payloads/`2,305,506` bytes,
  artifact manifest SHA `9c9e5d5c...46840`; its 29 cache files reclaimed
  `44,277,760` observed bytes.
- Final manifest replay found no artifact/retained-summary mismatch; all 26
  NPY records per run match their cleanup manifests and both cache directories
  are absent.
- Retention audit passes over 610 run directories with `blockers=[]`, summary
  SHA `95ade55b...a614`.
- Compileall passed; focused semantic-part plus old part-token tests passed
  `16/16`; full pytest passed `834/834`; all four PowerShell wrappers parse
  with zero errors.
- Current-best `-PreflightOnly` still resolves the exact keeper, 30 epochs,
  effective batch 64, empty class-conditional augmentation policy, and
  train-side test skipped. It created no run directory.
- Test was not opened, raw datasets and protected user paths were untouched,
  and the current-best full-train/export/video commands remain unchanged.
