# TRKH 5-Class Spectral Decoupling A0 Closure - 2026-07-20

## Scope and provenance

This closes the single prospectively locked Spectral Decoupling A0. The
accepted NeurIPS-2021 equation was taken from the official paper and MIT source
at commit/tree `bc5c29631fc3abe72e7d05c7ab3424c342e7fd49`/
`2fbce6eaeceb46ecadcd2c2ede600c2faa1478a7`. The immutable TRKH
protocol is `TRKH_5CLASS_SPECTRAL_DECOUPLING_A0_PROTOCOL_20260720.md`, SHA-256
`fe7c14ee923f2b60ac5744831a4524ab721c61ba5f19f7745c5daa13830ac576`.

- Production implementation and matched launcher were committed and pushed as
  `4de1f7b` before the formal run.
- The train-only objective was exactly per-sample ordinary CE plus
  `lambda / 2 * mean_class(logit^2)`, with `lambda=0.01`.
- Both roles used the same random initialization protocol, seed 42, 120 train
  batches per epoch, five epochs, scheduler horizon 30, batch 32 with gradient
  accumulation 2, no pretraining, no resume, weight decay 0, label smoothing
  0, full 2,606-row validation, and no final test.
- Stage-A v2 passed all 16 equation/config/provenance checks at summary SHA
  `c2daaa7d4133b64fd7c6bbf3ba6083805feca5679de35ab94c7c56238e91d348`.
  The first Stage-A artifact is retained as an infrastructure trace: its sole
  false failure compared the intentional FP32 loss path with an FP64 oracle.
- A same-session Windows loader benchmark selected train/eval workers `4/2`.
  Workers 4 reached `196.346 images/s` versus `107.006 images/s` for workers 2,
  with data-wait fractions `10.85%` versus `52.48%`; benchmark SHA is
  `b55552f02ed51ffab0925f3e618fc9ad142e6dbde1de3917e4e38321402cc90a`.

## Locked full-validation result

The independently reloaded best checkpoints were control epoch 5 and candidate
epoch 4. Metrics are from all 2,606 `yolo_f` validation objects.

| Variant | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | Restricted class-1 FP |
|---|---:|---:|---:|---:|---:|
| CE control | 0.792608 | 0.382353 | 0.602649 | 0.467866 | 145 |
| SD lambda 0.01 | 0.785405 | 0.387097 | 0.556291 | 0.456522 | 131 |
| Candidate minus control | -0.007203 | +0.004744 | -0.046358 | -0.011345 | -14 |

Spectral Decoupling changed 106 decisions. It made 44 corrections and 58
harms, removed 30 restricted class-1 FP, created 16, rescued five class-1 FN,
and broke 12 class-1 TP. The main transition pattern explains why the raw FP
count is insufficient evidence:

- 24 class-0 `1 -> 0` corrections;
- 13 class-2 `2 -> 1` harms;
- 11 class-1 `1 -> 0` harms;
- 18 class-3 `3 -> 2` harms.

Thus only the restricted-FP gate passed. Precision missed its prospective
`+0.005` threshold, while class-1 recall/F1, macro F1, and corrections-versus-
harms all failed. Pair summary SHA is
`3e623fa4d29e6537c2b7019b6678296d66c0a10e14470819d7fcc6335e542b0c`.

## Calibration and architecture audit

The intended confidence contraction was active but not selective. Mean
confidence fell `0.828498 -> 0.808307`, while NLL worsened
`0.417319 -> 0.444295`, Brier worsened `0.221819 -> 0.235614`, and ECE-10
worsened `0.028414 -> 0.040267`. On changed cases, mean class-1 probability
fell `0.123943` for FP removals and `0.124061` for broken TP; the almost equal
magnitudes show broad class-1 contraction rather than a TP-safe precision cue.

Both architecture traces completed on one sample per class with identical
input/stem/token/grid geometry and exact retained-token counts. The selected
token identities changed as expected from separately trained weights; layer-2
and layer-5 Jaccard overlap ranged `0.8712-0.9818` and `0.8352-0.9881`.
Pairwise routing was zero on all five trace samples in both roles, while bbox
spatial logits changed by mean/max absolute `0.0570/0.1311`. Control/candidate
trace SHAs are
`8fe8d4e7ebc55e02796bfe0d6c03158049f3092031ecb83e5e23870f9c900ec5`
and `b0a55cf4177833456b5993d912cd54d5efeb371992b78e36b56f70d17825d9d2`.
Parameter counts and peak reserved VRAM were identical (`7,245,590`,
`7,110 MiB`); runtime ratio was `0.968812`, treated only as timing noise.

The clean metric gate failed before robustness. Per the locked stop rule, no
shifted-condition evaluation, changed-cohort XAI, test, longer probe, or full
train was run. This is an evidence-based stop, not a missing audit: Stage A,
training histories, full confusion transitions, source-group forensics,
calibration, and both architecture traces were inspected before closure.

## Evidence retention and decision

The six Stage-A/benchmark/control/candidate/pair directories were compacted
after hash verification. Rejected checkpoints and redundant generated PNGs
were omitted; metrics, predictions, configs, histories, source-group
forensics, and architecture trace JSON/JPEG were retained.

- Compact evidence: `runs/evidence_spectral_decoupling_rejected_20260720`,
  54 verified payloads, `5.647 MiB`, payload-manifest SHA
  `52d09f11099e10ff461fec13d9f92031114e01dee69c26f773a0a616d9723d20`.
- Cleanup manifest SHA:
  `6999cb2ccc9a3563d1c6dcf9c589e39f650a3e6960c1d1404ce7eff2c656aa18`;
  observed recovery was `379.719 MiB`.
- The first relative-output invocation placed the verified evidence under a
  redundant `runs/runs` prefix. It was moved to the canonical path, absolute
  metadata was corrected, and all payload hashes were regenerated and
  reverified. This did not alter any experiment artifact.
- Retention passed over 766 run directories and 50 object manifests, with all
  230 compacted originals absent and `blockers=[]`; summary SHA is
  `2b3f0a40af6b97f379f873dab9b2eda2389145a449c1f4e2499e43016f94c718`.

Reject this exact Spectral Decoupling A0. Do not sweep lambda, class-specific
penalties, weight decay, label smoothing, seed, short budget, checkpoint
selection, or combine it with LogitNorm/confidence penalties on the current
keeper. It suppresses useful and harmful class-1 evidence together and also
damages the class-3/class-2 boundary.

The current-best command/history/keeper SHAs remain respectively
`36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`,
`39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`, and
`1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
The command packet remains three revisions/two actual updates; no promotion
occurred.
