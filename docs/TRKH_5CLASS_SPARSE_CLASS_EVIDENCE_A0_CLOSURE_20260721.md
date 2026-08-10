# TRKH 5-Class Sparse Class-Evidence A0 Closure - 2026-07-21

## Decision

Reject and close the prospectively locked copied-third-block sparse class-
evidence branch on the current frozen keeper. The implementation, replay,
static export, resource checks, and fixed XAI artifacts are valid, but the
map-L1 candidate does not provide selective class-1 evidence. Do not integrate
the branch, run shifted-condition robustness, open validation/test, launch an
image-model smoke/probe/full train, or update the current-best commands from
this result.

## Locked Scope

- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Protocol:
  `TRKH_5CLASS_SPARSE_CLASS_EVIDENCE_A0_PROTOCOL_20260721.md`, SHA-256
  `c046891266bd07b59723b8f4d1a8f5db52e070404f0f3c8d71db1c415f286a7b`.
- Prospective protocol and implementation were pushed before formal metrics at
  commits `5f24b1a` and `4ba52cf`; the clean formal ran from pushed HEAD
  `7d667c5`.
- Cohort: exactly 763 train-only CIDT rows, comprising 528 keeper class-1 TP,
  13 keeper class-1 FN, and 222 restricted `0/2/4 -> 1` FP over five immutable
  source-disjoint folds. Ordered-index SHA-256 is
  `59d2146617f2642f99c081f49a9aa5812eae4206b27887a1fa5476f6f61c85b2`.
- Frozen input: keeper `stem.blocks[1]` output `[64,64,64]`, complete valid-
  rectangle crop `[64,48,48]`, copied third stem block `64 -> 256`, and a
  `1x1` class-evidence projection producing two `[24,24]` maps.
- Learned roles: matched dense, map-L1 sparse, global-logit-L1, and separately
  trained channel-dephased heads. The candidate is also evaluated with the
  same learned weights after deterministic marginal-preserving channel
  dephasing.
- Training: five folds, batch 64, AdamW `lr=1e-3`, weight decay `1e-4`, exactly
  20 epochs, and map/logit regularization calibrated to 5% of first-batch CE.
  Each threshold is fit only on four folds to retain at least 97% fit class 1.
- Validation and test were not opened, and no raw dataset file was changed.

## Engineering Result

All 18 structural gates pass.

- Independent NumPy/FP64 Eq. (1)-(2) reconstruction agrees with Torch FP64 to
  `2.22e-16`; the regularizer covers every valid map element in every batch
  item. Common spatial-permutation symmetry, deterministic dephasing,
  marginal preservation, and RNG isolation pass.
- Hooked versus ordinary keeper probability error is exactly `0.0`, and keeper
  state is bit-exact before/after extraction. Historical CIDT-cache drift is
  retained only as BF16/batch-shape telemetry and is not substituted for the
  locked ordinary-forward equivalence gate.
- Every one of the five trainable tensors in every matched role receives a
  finite nonzero gradient and changes. Parameters and optimizer states remain
  FP32 and finite; occurrence order and matched initial states are attested.
- Formal extraction uses requested/effective workers `4/4`, pin memory, and
  persistent workers at `27.3945` images/s. Peak extraction CUDA allocation is
  `1,346,222,592` bytes. No unknown process is terminated.
- Candidate/dense runtime is `5.54483/5.14515 ms` per measured batch, ratio
  `1.07768`; candidate extra peak activation is zero.
- BF16-versus-FP32 and FP32-versus-FP64 probability errors are
  `3.19824e-4` and `2.74460e-8`. Static ONNX error is `1.19209e-7`, actions are
  exact, and no custom operator domain is present.
- Focused/full tests pass `13/13` and `1734/1734`. The temporary FP16 feature
  cache `[763,64,48,48]` is hash-attested and deleted.

## Clean OOF Result

| Role | AUROC | AUPRC | Class-1 retention | TP retention | FN support | FP rejection | TP broken | FP rejected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense_aligned` | 0.623187 | 0.811417 | 0.948244 | 0.952652 | 10/13 | 0.076577 | 25 | 17 |
| `logit_l1_aligned` | 0.634103 | 0.815743 | 0.940850 | 0.945076 | 10/13 | 0.103604 | 29 | 23 |
| **`sparse_aligned`** | **0.626959** | **0.811392** | **0.940850** | **0.945076** | **10/13** | **0.058559** | **29** | **13** |
| `sparse_channel_dephased` | 0.612013 | 0.797357 | 0.863216 | 0.867424 | 9/13 | 0.243243 | 70 | 54 |
| `sparse_same_weight_dephased` | 0.613812 | 0.800090 | 0.902033 | 0.907197 | 9/13 | 0.148649 | 49 | 33 |

Only 8 of 24 conjunctive mechanism gates pass. The candidate gains only
`+0.003772` AUROC over dense while losing `0.000025` AUPRC and `0.018018` FP-
rejection rate. Global-logit-L1 is better by `0.007144` AUROC, `0.004350`
AUPRC, and `0.045045` FP-rejection rate.

The action rule is harmful: the candidate supports 10 FN and rejects 13 FP,
but breaks 29 keeper TP, so corrections plus supports (`23`) do not cover TP
harms. It misses the locked `0.95` TP-retention, `0.25` FP-rejection, `0.85`
AUROC, and `0.90` AUPRC floors.

## Mechanism And XAI Diagnosis

- Candidate Hoyer sparsity is `0.239164`, only `+0.024617` over dense instead
  of the required `+0.10`. Top-10% mass share is `0.282555`, only `+0.016993`
  over dense instead of `+0.10`.
- Mean foreground mass is adequate at `0.740494`, but mean border mass worsens
  from dense `0.364951` to candidate `0.390825` and fails the `<=0.30` gate.
  Padding leakage, zero maps, constant maps, and logit reconstruction are all
  clean, so this is not an artifact-corruption failure.
- Same-weight dephasing lowers AUROC by only `0.013147`, below the required
  causal margin `0.02`; the candidate beats the separately trained dephased
  role in only folds 1 and 3. Spatial alignment is therefore not a sufficiently
  causal source of selectivity.
- Manual review of the fixed 15-row direct and feature-gradient sheets fails.
  Evidence maps are broad color fields and fruit contours. Gradients emphasize
  fruit-background boundaries, stems, hands, shadows, and context transitions;
  lesion or peel texture is sparse and inconsistent. The visual result agrees
  with the quantitative border/dephasing failures and cannot rescue the clean
  rejection.

## Replay And Retention

- Independent second-process replay covers all 763 rows with exact thresholds,
  scores, actions, and analysis (`0.0` maximum differences).
- Final summary SHA-256:
  `52d623f6c9b7e7854864db0bca5a989c76800c99b33c176ac531bce454de58ec`.
- Final artifact-manifest SHA-256:
  `06a6addcde70e6d77b6cb054f162af08696df3e7fe96c5bb92cf88f2af6e97c4`.
- OOF predictions SHA-256:
  `235a5455dfac4b93889083991012368cdd4e74a3f5a2f27595aadf80d5b413a6`.
- ONNX SHA-256:
  `d79732b7148f646371d61cdbc7f945dec3193c814aa7b0981e6dd74885a39a51`.
- Direct/gradient XAI sheet SHA-256:
  `328e1c7cb03f18f113a56a0301d7a15f8fdf2ac6cd954f6415d5811309548f8d` /
  `5dc23db7fc9fc8d453df10067ba4a345b60d239e65b255043a2e2d2952b0bb48`.
- Branch-state/fit-score SHA-256:
  `f6e1f86523d0836fad78dcc556f96accb5358415dcc60aab6648e538f8494689` /
  `6c1c81b7bff66ccc0161061369597f397eaed29d6286550871a8309af66069e`.
- The complete 16-file formal is 26.980 MiB and remains intact; no cleanup is
  useful. Read-only retention passes over 813 run directories and 51 valid
  object-schema compaction manifests, with all 233 expected-absent originals
  still absent, `deleted_anything=false`, `blockers=[]`, and `100.552 GiB`
  free. Retention-summary SHA-256 is
  `aebdc14c2ff8023f0cb9a4cbdb6cef2093cafd4fea01555a5d99d6e9b202cba2`.

## Closed Boundary

Close the copied-third-block `64 -> 256` branch, block-2 `48x48` valid crop,
two-channel `24x24` class-evidence map, map-L1 calibrated at 5% CE, matched
dense/global-logit-L1 controls, 20-epoch head schedule, fold/seed/threshold,
and trained/same-weight channel-dephasing family on this keeper. Do not sweep
nearby map-L1 weights, widths, map grids, stem layers, optimizers, epochs,
folds, seeds, thresholds, or fuse this evidence map with already closed color,
covariance, frequency, part-selection, or attention routes.

The next route must improve class-conditional surface evidence before spatial
averaging, demonstrate a causal lesion/texture signal against matched
non-spatial and boundary/context placebos, and protect class-1 TP. Current-best
checkpoint, full-pipeline command file, and command-update history remain
unchanged.
