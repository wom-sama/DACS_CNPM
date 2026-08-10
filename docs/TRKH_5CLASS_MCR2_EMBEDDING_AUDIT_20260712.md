# TRKH MCR2 Frozen-Embedding Readiness Audit

Date: 2026-07-12

## Question

Can maximal coding rate reduction (MCR2) reorganize the current keeper's
frozen representation into compact within-class and separated between-class
subspaces, especially for class 1, without changing the raw dataset or using
test data?

Primary sources:

- NeurIPS 2020 paper:
  <https://proceedings.neurips.cc/paper_files/paper/2020/hash/6ad4174eba19ecb5fed17411a34ff5e6-Abstract.html>
- Official implementation: <https://github.com/ryanchankh/mcr2>
- Official loss implementation:
  <https://github.com/ryanchankh/mcr2/blob/master/loss.py>
- Official nearest-subspace evaluation:
  <https://github.com/ryanchankh/mcr2/blob/master/evaluate.py>
- Efficient variational MCR2, CVPR 2022:
  <https://openaccess.thecvf.com/content/CVPR2022/html/Baek_Efficient_Maximal_Coding_Rate_Reduction_by_Variational_Forms_CVPR_2022_paper.html>

The official supervised reproduction command uses LR `0.01`, 500 epochs,
batch 1000, 128D output, epsilon `0.5`, and gamma `1/1`. The checked
`train_sup.py` parser currently defaults to LR `0.001` and 800 epochs. This
audit locks the reproduction LR but caps training at the project's required 30
epochs; the official scheduler milestones begin after epoch 200 and therefore
would not act inside this budget.

## Locked Protocol

- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Reusable keeper cache:
  `runs\diagnostic_reslt_embedding_cache_keeper_yolof_20260712`.
- Immutable `yolo_f`: train `9215`, validation `2606`; test unopened.
- Train/validation source groups: `8064/2577`; overlap `0`.
- Input: frozen 256D keeper embeddings. The keeper was trained on the full
  train split, so its train representation/probabilities are explicitly
  in-sample and are never described as OOF evidence.
- Projection: official-form `Linear(256,256,bias=False) -> BatchNorm -> ReLU
  -> Linear(256,128) -> L2 normalization`.
- Objective: exact empirical MCR2 global and class-conditional log-determinant
  terms with epsilon `0.5` and gamma `1/1`.
- Training: SGD, LR `0.01`, momentum `0.9`, weight decay `5e-4`, natural
  frequency, deterministic shuffle, batch `1000`, `drop_last=true`, 30 epochs.
- The official supervised loader does not request shuffle. This audit uses one
  fixed-seed shuffle so cache row order cannot define batches; every-class
  batch support is measured and fail-closed. This adaptation is explicit.
- Evaluation: official centered nearest-subspace PCA residual, 30 components
  per class. Residual probabilities use only a fit-side median temperature.
- Five `StratifiedGroupKFold` source folds. The matched control uses L2-
  normalized raw keeper embeddings and the same folds/subspace evaluator.
- The six projection instances are RAM-only and discarded. No adapter,
  checkpoint, model, trainable manifest, raw-data write, test input, class
  weighting, oversampling, parameter sweep, or validation tuning is allowed.

Implementation and retained evidence:

- `trkh/tools/audit_mcr2_embedding_readiness.py`
- `tests/test_audit_mcr2_embedding_readiness.py`
- `runs/diagnostic_mcr2_keeper_embedding_full_20260712`

The exact audit-tool SHA-256 that generated the retained payload is
`beb496de98728e123a0f147090a9c649139807a96902b247f94bcac4e5958fce`.

## Protocol Checks

All `9215/2606` rows are unique, finite, and have normalized probabilities.
Every fit/hold fold and train/validation source comparison has zero overlap.
Every batch contains every class; the minimum class support is `32`. MCR2 loss
decreases in all five folds and the full fit. Rate reduction rises from roughly
`10.8-11.7` at epoch 1 to `27.3-28.0` at epoch 30.

The candidate improves class-1 F1 over the matched raw-subspace control in all
five train folds. This proves that the implementation learns the intended
subspace geometry; it does not prove that this geometry beats the deployed
keeper classifier.

Independent CSV recomputation reproduced all reported macro/class-1 F1 values,
the validation confusion matrix, and transition counts. Candidate probability
sums differ from one by at most `2.98e-8`; no validation path references test.

## Results

| Split/method | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 |
| --- | ---: | ---: | ---: | ---: |
| train OOF raw-subspace control | 0.891109 | 0.653061 | 0.650647 | 0.651852 |
| train OOF MCR2 | 0.939210 | 0.746951 | 0.905730 | 0.818713 |
| validation raw-subspace control | 0.852374 | 0.585714 | 0.543046 | 0.563574 |
| validation MCR2 | 0.875310 | 0.632258 | 0.649007 | 0.640523 |
| validation direct keeper | 0.884073 | 0.608247 | 0.781457 | 0.684058 |

MCR2 gives large matched-control gains: OOF `+0.048101/+0.166862` and
validation `+0.022936/+0.076949` macro/class-1 F1. However, it still loses to
the direct keeper by `-0.008764/-0.043535`, and class-1 recall falls by
`0.132450`. The stronger geometric readout is therefore not a keeper-level
class-1-positive representation.

| Comparison | Changed | Corrections/harms | FP removed/created | FN rescued/TP broken |
| --- | ---: | ---: | ---: | ---: |
| OOF MCR2 vs control | 441 | 329/96 | 90/69 | 146/8 |
| val MCR2 vs control | 140 | 78/46 | 18/17 | 25/9 |
| val MCR2 vs keeper | 91 | 42/44 | 24/5 | 2/22 |

Against its matched control, MCR2 looks recall positive. Against the keeper it
is mainly a conservative class-1 suppressor: it removes 24 false positives but
breaks 22 true positives and rescues only two false negatives. The candidate-
minus-control FN-versus-FP AUROC also reverses from `0.717030` on OOF train to
`0.385807` on validation. This reversal prohibits a residual, router,
threshold, or sample-weight interpretation.

## Decision

Reject MCR2 on the current frozen keeper representation before image-model
smoke. Seven locked checks fail, including keeper macro/class-1/recall
preservation, class-1 `0.70`, direct correction safety, FN rescue, and
validation direction. `image_smoke_permission=false`.

Do not sweep MCR2 LR, epsilon, gamma, projection width, PCA rank, batch size,
optimizer, epochs, folds, or subspace temperature. Do not add this adapter as a
post-hoc head or train an image branch from the same frozen signal. Reopen MCR2
only after a genuinely new image representation changes direct keeper FN/FP
support; then use a matched CE classifier and keeper-preserving initialization.

The final artifact contains eight payloads, `4863531` bytes, aggregate SHA-256
`f5a326772966adc999303ceea992992f625cc28c74dae9daf52f683629f6ae49`.
It contains no checkpoint, model binary, or test payload. There were no MCR2
preflight roots or duplicate logs to delete. Retention
`runs\artifact_retention_audit_after_mcr2_reject_20260712` passed over 588 run
directories with `blockers=[]` and performed no deletion.

Keeper and current-best command remain unchanged. The command SHA-256 is
`3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.
Closure passed py-compile, compileall, focused tests `7/7`, full pytest
`754/754`, independent CSV metric reconstruction, payload/retention checks,
and explicit protected-path review.
