# TRKH 5-Class BoxInst Foreground-Mask A0 Closure - 2026-07-21

## Decision

Reject and close the prospectively locked class-agnostic BoxInst mask-head
adaptation on the current frozen keeper. The formal audit is complete and
replayable, but the aligned projection-plus-LAB-affinity head converges to an
almost all-valid mask instead of an object-tight mask. It provides no useful
class-1 precision signal beyond simpler controls. Do not integrate this branch,
run shifted-condition robustness, open validation/test, launch an image-model
smoke/probe/full train, or update the current-best commands from this result.

This decision applies to the exact independent TRKH adaptation below. It does
not claim that the original instance-segmentation system or paper is invalid.

## Locked Scope

- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Protocol:
  `TRKH_5CLASS_BOXINST_FOREGROUND_MASK_A0_PROTOCOL_20260721.md`, SHA-256
  `cb6335e787df64eb624db4abe1c677e16eaf248d8ab74d3b28a5db6ac3643425`.
- The accepted CVPR-2021 paper is pinned at SHA-256
  `95b01b2bfaa56f522aa84e7817cbf39ccaa72f01f6572723655a7989552f66e0`.
  The authors' AdelaiDet source is retained only as provenance at commit/tree
  `5e19cb172b8363820b409ed1a2754fb19ad3acb8` /
  `bd7918078c2b7145ca06809b9e5f976f1286a702`; no official source was imported.
- Formal mask training uses all 9,215 clean `yolo_f/train` object rows with
  immutable source-disjoint folds and class counts
  `[1941,541,1920,2520,2293]`. Classification gates use exactly 763 train-only
  CIDT rows: 528 keeper class-1 TP, 13 keeper class-1 FN, and 222 restricted
  `0/2/4 -> 1` FP. Cohort index SHA-256 is
  `59d2146617f2642f99c081f49a9aa5812eae4206b27887a1fa5476f6f61c85b2`.
- Every learned role receives the same frozen keeper block-2 feature tensor
  `[64,64,64]`. The candidate combines bbox projection Dice with positive LAB
  pairwise affinity. Controls are projection-only, separately trained
  affinity-dephased, valid-uniform, bbox rectangle, same-weight mask roll,
  keeper log-probability, and candidate-without-keeper.
- Training is five source-held folds, batch 64, AdamW `lr=1e-3`, weight decay
  `1e-4`, exactly 20 head epochs. Readout thresholds are fit only on four folds
  and retain at least 97% fit class 1.
- The formal ran from clean pushed implementation HEAD `1b72f8b`. Validation
  and test were never opened; the raw dataset and keeper were not modified.

## Engineering Result

Thirteen of 17 structural gates pass. The independent equations, formal
training, artifact replay, and data isolation are valid, but four deployment
or explanation gates fail and remain failures.

- No-pixel preflight reproduces all train/holdout/cohort counts and both locked
  index hashes. Torch and independent NumPy FP64 projection/pairwise errors are
  at most `2.22e-16`; LAB conversion differs from scikit-image by at most
  `5.260085e-5`.
- A real two-row no-output forward preserves hooked versus ordinary keeper
  probabilities exactly, keeps every bbox inside valid letterbox support,
  gives every mask-head parameter a finite nonzero gradient, and deletes both
  temporary caches. Historical CIDT probability drift is `7.364154e-5` with
  exact argmax and remains telemetry only.
- Requested/effective formal workers are `4/4`, with pin memory and persistent
  workers. Extraction throughput is `71.842895` images/s, extraction time is
  `128.266` s, training time is `1026.392` s, and peak extraction CUDA
  allocation is `1,317,182,976` bytes.
- The candidate branch costs `5.99285 ms` versus `3.12535 ms` for bbox
  descriptor construction, ratio `1.917497`, failing the locked `<=1.15`
  resource gate. Extra peak CUDA is only `12,593,152` bytes and total peak is
  `236,653,568` bytes.
- FP32/FP64 mask and descriptor errors pass at `7.98933e-8` and `2.62641e-8`,
  but score error is `2.82996e-5`, above `1e-5`. BF16/FP32 mask and descriptor
  errors pass at `0.00192368` and `0.000220057`; score error is `0.347777` and
  non-near-tie actions are not exact. The nearly singular readout amplifies
  otherwise small descriptor differences.
- Static ONNX Runtime mask/descriptor errors pass at `1.78814e-7` and
  `2.62260e-6`, actions are exact, and no custom domain exists. Score error is
  `0.00486332`, so static export remains failed rather than being relaxed
  after observation.
- Direct-mask automatic XAI is finite with zero padding leakage. Feature-
  gradient XAI reconstructs scores to `3.75567e-7` with zero padding leakage,
  but the fixed row-3 same-weight rolled control is all-NaN because its mask is
  degenerate; the automatic gradient-XAI gate therefore fails.
- Compile, pyflakes, PowerShell parsing, focused tests, and the full suite pass.
  The formal implementation passed `16/16` and `1750/1750`; replay hardening
  raises the focused/full counts to `17/17` and `1751/1751`.

## Clean OOF Result

Only 6 of 18 conjunctive mechanism gates pass.

| Role | AUROC | AUPRC | Class-1 retention | TP retention | FN support | FP rejection | Precision | TP broken |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `keeper_logprob` | 0.796540 | 0.910144 | 0.964880 | 0.984848 | 2/13 | 0.121622 | 0.728033 | 8 |
| `valid_uniform` | 0.789829 | 0.909535 | 0.963031 | 0.979167 | 4/13 | 0.139640 | 0.731742 | 11 |
| `bbox_rectangle` | 0.786573 | 0.907389 | 0.961183 | 0.979167 | 3/13 | 0.144144 | 0.732394 | 11 |
| `projection_aligned` | 0.776790 | 0.903369 | 0.946396 | 0.965909 | 2/13 | 0.135135 | 0.727273 | 18 |
| **`boxinst_aligned`** | **0.793068** | **0.910206** | **0.959335** | **0.975379** | **4/13** | **0.144144** | **0.732017** | **13** |
| `boxinst_affinity_dephased` | 0.787730 | 0.908181 | 0.963031 | 0.979167 | 4/13 | 0.139640 | 0.731742 | 11 |
| `boxinst_same_weight_mask_rolled` | 0.789354 | 0.909194 | 0.950092 | 0.965909 | 4/13 | 0.157658 | 0.733238 | 18 |
| `boxinst_aligned_without_keeper` | 0.622554 | 0.812638 | 0.940850 | 0.945076 | 10/13 | 0.076577 | 0.712885 | 29 |

The candidate rejects 32 restricted FP and supports 4 FN while breaking 13
keeper TP. Its AUROC is `0.003472` below keeper log-probability, and its FP
rejection equals the bbox control while trailing the same-weight rolled
control. It misses the locked `0.85` AUROC, `0.25` FP-rejection, `0.75`
precision, eight-FN-support, and causal-margin requirements. High AUPRC and TP
retention alone therefore do not establish a useful new mechanism.

## Collapse And XAI Diagnosis

- The aligned candidate has mean valid-area mass `0.99999936`, mean bbox-area
  ratio `1.19221830`, outside-bbox mass `0.16039238`, and saturation
  `0.99999863`. All 9,215 rows exceed 99.5% saturation and 5,091 are constant
  all-one rows.
- The separately trained affinity-dephased role collapses similarly: valid
  mass `0.99999966`, bbox-area ratio `1.19221865`, outside mass `0.16039239`,
  all 9,215 rows saturated, and 7,485 constant/all-one rows.
- The positive-only same-label affinity term reaches nearly zero loss by
  assigning the same foreground label throughout valid support. Its mean
  aligned pairwise loss is only `3.57849e-7`, while projection loss remains
  `0.0875963`; the projection term is too weak to prevent this all-valid
  attractor in the locked adaptation.
- Projection-only masks are geometrically credible: bbox-area mean/std are
  `0.727165/0.095143`, outside-bbox mass is `0.015898`, projection loss is
  `0.021700`, and there are no degenerate rows. Their lower AUROC and weak FN/
  FP action balance show that object-tight shape alone is not the missing
  class-conditional disease/ripeness signal.
- Manual review of the fixed 15-row direct sheet fails. Projection-only maps
  generally follow the mango, aligned/dephased maps fill the entire valid
  image, and rolled controls form shifted rectangular bands.
- Manual review of the feature-gradient sheet also fails. Roles are visually
  similar and emphasize fruit silhouette, object-background transitions,
  stems, hands, broad peel, and occasional lesion dots without stable
  candidate-specific class evidence. The blank row-3 rolled gradient agrees
  with the recorded non-finite automatic gate.

## Replay And Retention

- The first external replay exposed an engineering mismatch: the standalone
  process had not restored deterministic CUDA and TF32 settings used by the
  formal process. It reported mask/descriptor/score errors
  `0.00341797/0.00022910/8.6194e-5`. Commit `2258da0` sets
  `CUBLAS_WORKSPACE_CONFIG`, deterministic seeding, and TF32-off before replay;
  it changes no scientific metric or persisted model state.
- The corrected independent second-process replay covers all 763 rows with
  exact masks, descriptors, actions, and analysis. Score/threshold errors are
  only `4.44089e-16` and `5.55112e-17`.
- Final status is `rejected_automated_gate_visual_review_recorded`.
- Final summary/manifest SHA-256:
  `80a80391f3811dd91ce2de7af48929398cd609433f5b2e8b6606bbe1e6b8bd09` /
  `98368287becc6ed4195ea418d2b58698a2e8965dc2a2b149f196200bfb5c2102`.
- OOF predictions and ONNX SHA-256:
  `59d1d4e2c5fed2a5bd562c677c5ee05cbabeba2f8c55f9fdde334b33e962878c` /
  `6b8edc213ca469dd3c2ae0e4ae92f57791a6da92e37be1b977bdf70404d3f374`.
- Direct/gradient XAI sheet SHA-256:
  `11520c96c41b18bf05c5169d1737f43aba4ffc8f558befc4054690526ae43cea` /
  `b3f539dec4db849d044a315d8af85c0ccca3569d1fab2157943d52e05db0d24d`.
- Mask-head/readout-state SHA-256:
  `5fdc93a733aeb6f8bc2ed4cc0e5ded96530e0d3d2aa2c46a30398ce1a9f05c57` /
  `d7fc51ee8bfa6d5ce085cb6899fb3809fbd082dfe133b7df4c0591cea6fbbe8c`.
- The complete 20-file formal is `457.257 MiB`. Both multi-gigabyte temporary
  extraction caches are absent; no partial artifact deletion is useful.
- Read-only retention passes over 814 run directories and all 51 valid object-
  schema compaction manifests. All 233 expected-absent originals remain
  absent, `deleted_anything=false`, `blockers=[]`, and `100.099 GiB` is free.
  Retention-summary SHA-256 is
  `1f857adae5dbe76a9855096d48add575351152f68640644949c8eac6bdff5e7a`.

## Closed Boundary

Close the exact class-agnostic `64x64` mask head, positive-only LAB pairwise
affinity, projection Dice, widths, neighborhood/dilation/color threshold,
loss balance, optimizer, 20-epoch schedule, fold/seed/readout/TP threshold,
dephasing, and nearby post-metric sweep family on this keeper. Do not repair
the collapse by tuning a negative-pair term, mask-area prior, thresholds, or
loss weights after seeing these metrics under the same A0 claim.

The next route must learn class-conditional lesion or ripeness evidence rather
than another class-agnostic foreground mask. It should include a prospectively
locked anti-collapse or causal control, demonstrate selectivity beyond keeper
confidence and bbox/shape/context, and protect class-1 TP before any image-
model integration. Current-best checkpoint, full-pipeline command file, and
command-update history remain byte-unchanged.
