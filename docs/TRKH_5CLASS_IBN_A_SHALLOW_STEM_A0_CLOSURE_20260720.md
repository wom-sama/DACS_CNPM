# Shallow IBN-a Stem A0 Closure (2026-07-20)

## Decision

Close `stem_normalization=ibn_a_first` on the current no-pretrain keeper. The
formal matched smoke fails six conjunctive gates and creates a broad class-1
false-positive expansion. It does not authorize post-smoke robustness, a
five-epoch probe, test access, or full training.

## Locked Evidence

- Accepted source: ECCV-2018 IBN-Net paper and authors' MIT repository at
  commit/tree `d1673389...da915`/`e113673c...ab4b`.
- Original protocol SHA: `7e329a7c...e576`.
- Prospective missing-resume erratum SHA: `2ba11cfc...f91e`.
- Implementation commit: `5d08496`; matched-smoke infrastructure commit:
  `508582e`.
- Stage-A summary/manifest/launcher SHAs:
  `14b484ff...b422`/`8595174c...dff6`/`b38c9067...e3f3`.
- Formal pair comparison/top-level-manifest SHAs:
  `b0e9c527...30da`/`9b033a33...8f42`.
- Dataset remained unchanged. Stage A was train-only. The smoke used only
  `yolo_f/train` and full `yolo_f/val=2606`; no test output exists.

The historical v4 resume checkpoint named in the keeper launcher had been
deleted by the documented legacy cleanup. Both roles therefore used the same
current keeper SHA `1f49d577...2677`, reset epoch/optimizer/scheduler/scaler,
and retained all other source arguments. The only control-versus-candidate
model/config difference is `batch -> ibn_a_first` plus run identity.

## Stage A

All engineering gates pass. The 50/50 first-block IN/BN equation has maximum
oracle/direct error `9.54e-7/0.0`, preserves the BN half bit-exactly, strict-
loads the same keeper state, and keeps `7,245,590` parameters. FP32/BF16
gradients are finite and nonzero. Candidate BF16 runtime is `1.083251x`, peak
allocation is `4.58305 GiB`, and standard-op ONNX parity error is
`3.0518e-5` under the locked `5e-5` engineering tolerance.

## Matched Smoke

Both roles use seed 42, two epochs, 120 train batches per epoch, full
validation, batch size 32, accumulation 2, workers `4/2`, BF16, architecture
trace, and no final test. Independent checkpoint reload reproduces the selected
metrics exactly.

| Metric | Control BN | IBN-a first | Delta |
|---|---:|---:|---:|
| Macro F1 | 0.882914 | 0.794264 | -0.088650 |
| Class-1 precision | 0.597990 | 0.381988 | -0.216002 |
| Class-1 recall | 0.788079 | 0.814570 | +0.026490 |
| Class-1 F1 | 0.680000 | 0.520085 | -0.159915 |
| Runtime | 300.183 s | 297.607 s | 0.991418x |

Non-focus F1 drops are `0.067652/0.106154/0.089445/0.020082` for classes
`0/2/3/4`. Candidate-versus-control decisions change on 304 rows, with only 33
corrections and 254 harms. It removes eight restricted `{0,2,4}->1` false
positives but creates 108, a net worsening of 100. Class-1 FN rescues/TP breaks
are `15/11`; the small recall gain is overwhelmed by false-positive growth.

The most frequent prediction changes are `3->2=93`, `0->1=67`, `2->1=42`,
`3->1=19`, and `4->1=15`. Mean candidate-minus-control class-1 probability is
`+0.02537` for true class 0 and `+0.01148` for true class 2; it rises on 80.7%
of class-0 rows. NLL/Brier worsen from `1.17503/0.59601` to
`1.22315/0.62084`. Lower ECE is not a benefit here because accuracy collapses
from `0.91865` to `0.83384` while mean confidence also falls.

## Trace Review

The mandatory five-class architecture traces are complete for both roles.
Quantitative rendered-map comparison gives control/candidate stem correlations
`0.981/0.944/0.998/0.953/0.996` and attention correlations
`0.947/0.979/0.997/0.996/0.998` for classes `0..4`. Manual review agrees: the
candidate retains almost the same broad fruit/edge focus and does not expose a
new class-conditional lesion cue. The large decision shift with nearly
unchanged attention is consistent with early feature-statistic/logit drift,
not improved localization.

The protocol makes changed-case XAI and dim/bright/low-contrast replay
conditional on a clean smoke pass. Running them after six clean-gate failures
would violate the prospective stop rule and could only invite tuning a closed
family. Existing trace maps are retained as the required visual closure
evidence.

## Stop Rule

Do not sweep IBN split ratio, channel order, insertion depth, normalization
family, seed, learning rate, adaptation budget, augmentation, or loss. This
experiment does not claim that every from-scratch IBN network is ineffective;
it rejects the exact shallow IBN-a candidate under the locked current-keeper
research protocol. A future route must add an equation-distinct
class-conditional surface/boundary signal that removes restricted class-1 false
positives without broad support contraction or expansion.

## Evidence Retention

The rejected pair was compacted only after source paths, keeper protection,
and payload hashes were verified. The retained evidence directory is
`runs/evidence_ibn_a_shallow_stem_rejected_20260720`: it preserves 350
verified payloads, including predictions, metrics, configs, histories, and all
five-class trace PNGs. Exactly four rejected `.pt` files (348,762,272 bytes)
were excluded, then only the three rejected source directories were deleted.
The cleanup freed 395,141,120 bytes.

- Evidence payload manifest SHA256: `b1b68c1ca1b742433857041a2bee8f1d85ba41093db48916873cfec754b277fe`
- Evidence summary SHA256: `459883455363576406ebbc0a0408d960ec5c1feb54eee91b8beee159c81206fa`
- Source inventory SHA256: `9c858c90f4488422e0c826be8fa088be7c351a6a356cc3f587d651d2c9f6012d`
- Cleanup manifest SHA256: `1ef2912dd7cd850085799aa42482eb3e74fe25eea19f5cc65507669be8e9d7f3`

The read-only post-cleanup audit covers 777 run directories and all 50 valid
compaction manifests, with `retention_audit_passed=true`, `blockers=[]`, no
deletion, no raw-data change, and no test use. Its summary SHA256 is
`117acf93aec59dc32fba57658bd52b1fe127b95a5b622c388db4faf4451da3f7`.
Current-best checkpoint and command/history hashes remain unchanged.
