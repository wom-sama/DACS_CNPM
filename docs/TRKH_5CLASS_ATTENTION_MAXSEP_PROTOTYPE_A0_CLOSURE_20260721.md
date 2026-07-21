# TRKH 5-Class Attention-MaxSep Prototype A0 Closure - 2026-07-21

## Decision

Reject and close the prospectively locked attention-aware maximal-separation
prototype adaptation on the current frozen keeper. The candidate preserves
most keeper class-1 true positives, but it is substantially less selective
than keeper confidence and the cluster-only control, rejects too few restricted
false positives, and collapses to a low-rank, partially unused prototype bank.
Do not integrate this branch, open robustness/validation/test, run an image-
model smoke/probe/full train, or update the current-best commands.

This closure applies to the exact independent TRKH adaptation below. It does
not claim that the ACCV-2020 method or ProtoPNet is generally invalid.

## Locked Scope

- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Protocol:
  `TRKH_5CLASS_ATTENTION_MAXSEP_PROTOTYPE_A0_PROTOCOL_20260721.md`,
  SHA-256
  `43a965074e1a9dc2f9074598fdedd0810ed3977a3b60df19bc4a88a0f3b5907b`.
- The accepted ACCV-2020 paper/supplement SHAs are
  `68a9c61c...b4ff` / `4e360302...8f75`. The authors' unlicensed source is
  provenance only at commit/tree `2093938...fe431` / `5d26bc4...e949b` and
  was neither imported nor copied.
- ProtoPNet is the licensed architecture reference. Its MIT authors' source is
  pinned at commit/tree `81bf2b7...59321` / `b6275c5...ae01` and was not
  imported by the auditor.
- Use all 9,215 natural `yolo_f/train` rows in five source-disjoint folds with
  class counts `[1941,541,1920,2520,2293]`. Decision gates use exactly 763
  train-only CIDT rows: 528 keeper class-1 TP, 13 keeper class-1 FN, and 222
  restricted `0/2/4 -> 1` FP. Cohort-order SHA-256 is
  `59d2146617f2642f99c081f49a9aa5812eae4206b27887a1fa5476f6f61c85b2`.
- Capture frozen transformer block-2 patch tokens at `[256,16,16]`; exclude
  padding but retain the full object crop and its context. Four byte-identical
  roles isolate cross-sample MaxSep, self-attention MaxSep, cluster-only, and
  channel-dephased learning. Same-weight attention-roll and class-cycle plus
  keeper, bbox, GAP, and no-keeper controls are fixed before metrics.
- Train each role for exactly 20 epochs in every fold, batch 64, with 19,526
  trainable head parameters. Validation/test are never opened, the production
  model/trainer is untouched, and raw `yolo_f` remains byte-stat unchanged.
- Infrastructure is committed and pushed at `53c6786`; extraction telemetry is
  corrected without changing scientific settings at `6412e84`.

## Engineering Result

Seventeen of 19 structural gates pass. The equations, training path, numeric
precision, export, resource envelope, state isolation, and temporary-cache
cleanup are valid. The strict replay and automatic XAI reconstruction gates
remain failed and cannot be relaxed after metric access.

- The independent NumPy/FP64 oracle agrees with Torch to at most
  `2.84217e-14` across the locked equations. Prototype analytic versus
  finite-difference gradient error is `3.19744e-14`.
- All trainable parameter families receive finite nonzero gradients and change.
  Matched initial states and sample occurrence order are exact across roles.
- Requested/effective Windows workers are `4/4`, with pin memory, persistent
  workers, and prefetch factor 2. Full extraction takes `107.028` s at
  `86.0991` images/s and peaks at `1,208,097,280` CUDA bytes. Head training
  takes `464.807` s.
- Batch-32 head latency is `1.32895 ms`, extra peak CUDA is `4,768,768` bytes,
  and all resource gates pass.
- FP32/FP64 and BF16/FP32 score errors are `2.19622e-7` and `0.00277349`.
  NumPy/Torch descriptor error is zero. Static ONNX Runtime output error is
  `2.98023e-7`, non-near-tie actions are exact, and no custom domain exists.
- Compile, pyflakes, PowerShell parsing, focused `13/13`, and full pytest
  `1764/1764` pass. The formal launcher completes in about 693.6 seconds.
- The temporary full 9,215-row feature cache is deleted in `finally`; only the
  763-row cohort feature bank is retained.

## Clean OOF Result

Only 4 of 25 conjunctive mechanism gates pass.

| Role | AUROC | AUPRC | TP retention | FN support | FP rejection | Precision | TP broken |
|---|---:|---:|---:|---:|---:|---:|---:|
| `keeper_logprob` | 0.796540 | 0.910144 | 0.984848 | 2/13 | 0.121622 | 0.728033 | 8 |
| `bbox_geometry_plus_keeper` | 0.788163 | 0.906167 | 0.982955 | 3/13 | 0.103604 | 0.723994 | 9 |
| `block2_gap_plus_keeper` | 0.730504 | 0.879346 | 0.910985 | 3/13 | 0.252252 | 0.744615 | 47 |
| `cross_sample_cluster_only` | 0.792043 | 0.906768 | 0.984848 | 3/13 | 0.112613 | 0.726389 | 8 |
| **`cross_sample_maxsep`** | **0.671488** | **0.773264** | **0.988636** | **5/13** | **0.076577** | **0.719945** | **6** |
| `self_attention_maxsep` | 0.647200 | 0.759191 | 0.979167 | 3/13 | 0.112613 | 0.725244 | 11 |
| `cross_sample_channel_dephased` | 0.710013 | 0.871424 | 0.854167 | 1/13 | 0.310811 | 0.747107 | 77 |
| `same_weight_attention_rolled` | 0.668432 | 0.771946 | 0.992424 | 6/13 | 0.063063 | 0.718157 | 4 |
| `same_weight_class_cycled` | 0.674685 | 0.773989 | 0.994318 | 5/13 | 0.067568 | 0.719132 | 3 |
| `candidate_without_keeper` | 0.497802 | 0.704264 | 0.992424 | 13/13 | 0.000000 | 0.707510 | 4 |

The candidate rejects only 17 restricted FP and supports five keeper FN while
breaking six TP. It misses AUROC/AUPRC `0.85/0.90`, precision `0.75`, FP
rejection `0.25`, eight-FN support, four-of-five fold wins, all required
control margins, and the standalone-information gate. Every fold fails both
the control-margin and FP-rejection checks. High TP retention therefore comes
from weak intervention rather than useful precision selectivity.

## Collapse And XAI Diagnosis

- Projected-feature effective rank is only `5.3739/32`. Only 12 of 25
  prototypes are used, with used counts by class `[3,4,3,0,2]`; the class-3
  prototype group is completely unused.
- The class-1 own-versus-rival prototype margin has AUROC `0.488793`. Mean
  margins are negative for both true class 1 (`-0.089028`) and restricted FP
  (`-0.104270`), so the learned prototype relation is not a reliable class-1
  separator.
- Nearest held-patch class purity is only `0.228411`; mean nearest cross-class
  prototype distance is `0.986061`. Distinct vectors alone do not establish
  semantic or local-morphology prototypes.
- `294/763` attention maps are all-zero/low-peak (`38.53%`). Padding leakage is
  exactly zero, but nondegenerate attention and anti-collapse gates fail.
- Fixed 15-row review fails. Candidate/self/cluster attention maps are often
  blank; own/rival prototype maps broadly cover fruit color and silhouette;
  the dephased control remains responsive; own-minus-rival and gradient maps
  do not consistently distinguish TP, FN, and restricted FP. Surface/edge
  spots in some gradients do not establish candidate-specific causal evidence.

## Replay, XAI Reconstruction, And Retention

- In-process and independent second-process replay preserve exact states,
  actions, and analysis values, but both fail the prospectively locked numeric
  tolerance: output/descriptor error is `8.58307e-6`, score error is
  `1.67261e-7`, and threshold error is `4.26299e-8`.
- This is consistent with the already measured GPU/BF16 batch-shape path:
  formal features were captured in batch 64 and reconstructed after state
  reload. The historical CIDT probability comparison is telemetry, not a gate.
  Do not repair or relax replay after observing the candidate metrics.
- Automatic XAI is finite, deterministic to `1.16415e-10`, padding-clean, and
  uses the exact fixed 15 rows. Its score reconstruction error is
  `0.00258765` because selected samples are rerun at batch 1 rather than the
  formal batch-64 path, so the locked `1e-5` gate remains failed. Manual review
  independently fails the visual evidence.
- Final status is `rejected_automated_gate_visual_review_recorded`. Summary/
  manifest SHA-256 are
  `ef5a6d254b453ba523368fb67650acb89d84c452439d9afdb2bba7298aed3888` /
  `1acbc5689edaea0dfb57f2ff2e419762965530060bd23bc9abd7c19f2c9f5e66`.
  XAI sheet SHA-256 is
  `56d9f829710dffe05b47dc2c0765f8fdfb80442efb7cddabea12f92f44f1cf81`.
- All 12 manifest files verify byte-exact and total `107.824 MiB`. The
  1,207,828,608-byte temporary train cache is absent; no final artifact needs
  compaction.
- The first post-route retention command accidentally omitted the ignored
  nested historical manifest and is superseded. The corrected all-51 read-only
  audit scans 817 run directories, verifies all 233 expected-absent originals,
  deletes nothing, and has `blockers=[]`. Its summary SHA-256 is
  `9884a8c288cfdce91c7b67525609a77ece12073aaa971c0aa95c408b24fbb1b0`.

## Closed Boundary

Close prototype count, projector width, MaxSep/cluster/separation weights,
learning rate, warm/joint schedule, epoch count, seed, fold, readout C,
threshold, captured layer, attention normalization, attention roll/class
cycle, and nearby post-metric fusion or voting on this keeper. Do not integrate
the exact MaxSep loss into the image trainer or add a post-hoc prototype head.

The next route must provide a stable supervised class-conditional signal that
beats keeper confidence and a matched simple control without prototype or
attention-map collapse. It must preserve precision-oriented class-1 TP/FP
gates prospectively. Keeper, current-best command file, and command-update
history remain byte-unchanged.
