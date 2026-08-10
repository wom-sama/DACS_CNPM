# TRKH 5-Class DeepBDC Stem Joint-Dependence A0 Closure - 2026-07-21

## Decision

Reject and close the exact DeepBDC stem-final joint-dependence family on the
current frozen keeper. Do not integrate a BDC side head, run a matched image-
model smoke, open validation/test, launch a probe/full train, or update the
current-best commands from this evidence.

The audit is structurally valid and independently replayed. It fails because
aligned Brownian channel dependence is not selective enough for true class-1
predictions, loses to covariance, and loses to the same trained head after its
spatial channel alignment is destroyed.

## Locked Scope

- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Final pre-pixel protocol SHA-256:
  `4148b61ecf40168a82796030dcb9fd4648e5312fe60e6ea21c3e27c47837f93d`.
- Infrastructure commit:
  `9e31058b3adc4995ff1761ddf9d3649a864fcc0f`, clean and pushed before
  formal execution.
- Dataset access: `yolo_f/train` only. Validation and test were not opened.
- Cohort: 750 keeper-predicted class-1 rows, containing 528 true class-1
  positives and 222 restricted class `0/2/4` false positives in five immutable
  source-disjoint CIDT folds.
- Frozen input: final `HybridConvStem` map `[256,32,32]`; crop only the complete
  valid rectangle from `image_valid_mask`, then resize to `[256,16,16]`.
- Learned roles: keeper log probability, projected mean, covariance, trained
  dephased BDC, aligned BDC with base, and aligned-only BDC. A seventh causal
  control evaluates the trained aligned head with the same weights after
  deterministic per-channel spatial dephasing.
- Training: natural frequency, five source-held folds, batch 64, AdamW
  `lr=1e-3`, weight decay `1e-4`, exactly 20 epochs, no AMP/scheduler/early
  stop/weighting/resampling. Thresholds are fit on four folds to retain at
  least 97% fit TP and applied once to the held fold.

## Engineering And Structural Result

All 11 structural gates pass.

- Independent explicit-pairwise NumPy/FP64 and Torch FP64 BDC agree within
  `1.67e-16`; FP32/BF16 descriptor errors are `6.24e-8/0.004516`.
- Symmetry, translation, orthonormal-transform, normalized-scale,
  singleton-batch, invalid-input, analytic-gradient, and finite-difference
  checks pass.
- Ordinary versus hooked keeper probability error is exactly `0.0`; keeper
  state is bit-exact before/after extraction. The formal CIDT cache delta is
  `2.315640e-5`, below its telemetry bound, with exact argmax.
- Every trainable projection, BN affine, BDC temperature, and classifier
  parameter receives a finite nonzero gradient and changes. Optimizer state,
  initial/final parameter state, occurrence order, and all fold heads are hash
  attested.
- Formal loader requested/effectively used `4/4` Windows workers with pin and
  persistent workers. No unknown process was terminated.
- Candidate extraction/crop/head runtime ratio versus ordinary keeper forward
  is `0.926464`, below `1.15`. Peak CUDA allocation is `1,274,713,600` bytes,
  below 3.5 GiB.
- The temporary FP16 feature cache has shape `[750,256,16,16]`, SHA-256
  `afcac089...a1ce5f`, was never persisted, and was discarded after artifact
  finalization.

## Clean OOF Result

| Role | AUROC | TP retention | FP rejection | Corrections | Harms |
|---|---:|---:|---:|---:|---:|
| `base_logprob` | 0.738423 | 0.973485 | 0.148649 | 33 | 14 |
| `mean32_base` | 0.764708 | 0.958333 | 0.153153 | 34 | 22 |
| `cov32_base` | 0.789141 | 0.960227 | 0.148649 | 33 | 21 |
| `bdc32_dephased_base` | 0.770432 | 0.954545 | 0.184685 | 41 | 24 |
| `bdc32_aligned_base` | 0.779603 | 0.964015 | 0.144144 | 32 | 19 |
| `bdc32_aligned_only` | 0.621579 | 0.960227 | 0.054054 | 12 | 21 |
| `bdc32_same_weight_dephased` | 0.790123 | 0.962121 | 0.225225 | 50 | 20 |

Only 7 of 17 conjunctive mechanism gates pass. Material failures are:

- candidate AUROC is `0.779603`, below required `0.85`;
- aligned-only AUROC is `0.621579`, below required `0.65`;
- aligned candidate minus base/mean/covariance/trained-dephased/same-weight-
  dephased AUROC is respectively
  `+0.041180/+0.014896/-0.009538/+0.009171/-0.010519`;
- candidate FP rejection is `0.144144`, below `0.25`, and rejects only 32 FP,
  fewer than every useful control rather than at least ten more;
- candidate breaks 19 TP, five more than the base role, violating the maximum
  additional-break allowance;
- fold AUROC wins versus base/mean/covariance/trained-dephased/same-weight-
  dephased are `3/4/2/4/2`, not four of five against every control.

Candidate fold AUROC is `0.723261/0.833333/0.753750/0.839490/0.751807`.
Every fold retains at least 95% TP, so the failure is not one bad fold or a
threshold collapse. The descriptor effective rank is healthy at `24.6980`.

## Mechanism Interpretation

The decisive causal result is that the same trained aligned head becomes
better after independent channel-wise spatial permutation: AUROC rises
`0.779603 -> 0.790123`, FP rejection rises `0.144144 -> 0.225225`, and FP
corrections rise `32 -> 50` with only one additional TP harm. This destroys the
aligned joint spatial dependence that DeepBDC is intended to use while
preserving each projected channel marginal.

Covariance also beats aligned BDC by `0.009538` AUROC. Therefore this audit
does not support the claim that nonlinear aligned channel dependence adds a
new class-1 precision signal beyond the already closed second-order family.
The healthy rank rules out trivial descriptor collapse; the missing property
is selective information.

## XAI Review

The fixed 16-row sheet was rendered after the automatic decision and manually
recorded as fail. Aligned BDC and covariance gradients mainly trace fruit
silhouette, boundary transitions, padding/context edges, and isolated hot
pixels. They do not consistently isolate class-specific lesion/surface regions.
The same-weight dephased maps are diffuse but remain at least as predictive,
matching the quantitative causal failure. Visual review cannot rescue the
automatic rejection.

## Replay And Retention

- Independent second-process replay reconstructs scores from persisted OOF
  features plus serialized classifier state. Maximum score difference is
  `2.22e-16`, analysis difference is `1.07e-14`, and all actions/sample/fold/
  label axes are exact.
- Final summary SHA-256:
  `1d0503b0c0f919bc85313e434f24cd4c73ebbdb2be73e4fa25e4b82444fa91f7`.
- Final artifact-manifest SHA-256:
  `3a1e2d0da38242a0a44d1de6af2ba5ecdbd3c9296254196b6d730e9ead5eca6b`.
- Prediction CSV SHA-256:
  `6621a86ba34ae172a96eaa82810323104ece351033b227e931358b9359777a72`.
- Contact-sheet SHA-256:
  `c4c078b9b82b0510d98e79c28045a6f2b5be22dca2d5fbc021b58bb17261a2dc`.
- Head-state/replay-feature SHAs:
  `9987fee7...4f1906` / `04cbb7c3...f46b5`.
- The complete formal is 12.741 MiB and remains intact; no cleanup is useful.
- Read-only retention passes over 807 run directories and all 50 valid object-
  schema compaction manifests. All 219 compacted originals remain absent,
  `deleted_anything=false`, `blockers=[]`, and retention summary SHA-256 is
  `84b1f209913d8c226e117ecff213767a01dc4fa3f4cea4a86c789044d7d75609`.

## No-Repeat Boundary

Do not sweep or combine nearby values of BDC dimension, stem layer, valid
crop/resize, temperature, epsilon, centering, normalization, upper-triangle
readout, projection block, optimizer, epoch count, LR, weight decay, fold,
seed, TP-retention threshold, or trained/same-weight dephasing on this keeper.
Do not convert these audit actions into a post-hoc router.

A future route must provide supervised class-conditional surface evidence that
beats covariance and a marginal-preserving spatial placebo while retaining
class-1 TP. A generic higher-order pooling head is not enough.

Current-best checkpoint, full-pipeline commands, and command update history
remain byte-exact. No command revision is warranted.
