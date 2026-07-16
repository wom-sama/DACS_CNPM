# TRKH 5-Class Support-Constrained Capsule ALM Closure - 2026-07-16

## Scope and provenance

This closes the single prospectively locked train-only CapsALM A0. The route
combined Sabour et al. dynamic routing with Sangalli et al. augmented
Lagrangian critical-class constraints, adapted to the frozen 167-patch TRKH
keeper. The immutable protocol is
`docs/TRKH_5CLASS_SUPPORT_CONSTRAINED_CAPSULE_ALM_READINESS_PROTOCOL_20260716.md`
with SHA-256
`4344e6b16957b4e75ce432c9d04e70b511198d983fa380060fd190d5e7cd9667`.

- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Source-disjoint `yolo_f/train` fit/holdout: `7372/1843` rows and
  `6452/1612` source groups, with zero overlap.
- Boundary cohort: all 432 fit-fold class-1 rows and 186 restricted hard
  negatives. Natural and boundary schedule SHAs remained exact.
- Candidate/control each trained exactly 30 epochs, consumed 221,160 natural
  occurrences and 3,240 boundary events, and started bit-identically.
- Validation and test were not accessed. No checkpoint or other trainable
  model binary was written, and the raw datasets were not changed.

Official source commits remained clean: Google dynamic-routing capsules
`984fbc7...1ff88c` with Apache-2.0, and the disclosed unlicensed ALM repository
`ab3da74...279e2`.

## Formal behavior result

All metrics below use the complete 1,843-row source-disjoint train holdout.
Restricted false positives are targets `0,2,4` predicted as class 1.

| Condition | Variant | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | Restricted FP |
|---|---|---:|---:|---:|---:|---:|
| Clean | Raw | 0.948369 | 0.743056 | 0.981651 | 0.845850 | 37 |
| Clean | CE control | 0.858059 | 0.947368 | 0.330275 | 0.489796 | 2 |
| Clean | CapsALM | 0.935708 | 0.736111 | 0.972477 | 0.837945 | 38 |
| Dim | Raw | 0.825963 | 0.535354 | 0.486239 | 0.509615 | 46 |
| Dim | CapsALM | 0.827447 | 0.518519 | 0.513761 | 0.516129 | 51 |
| Bright | Raw | 0.872482 | 0.561644 | 0.752294 | 0.643137 | 63 |
| Bright | CapsALM | 0.850429 | 0.525974 | 0.743119 | 0.615970 | 72 |
| Low contrast | Raw | 0.854165 | 0.547826 | 0.577982 | 0.562500 | 51 |
| Low contrast | CapsALM | 0.823898 | 0.480620 | 0.568807 | 0.521008 | 56 |

Clean CapsALM versus raw changed 33 decisions: four corrections and 29 harms.
It broke one class-1 TP, rescued no class-1 FN, created one net restricted FP,
and reduced macro/class-1 F1 by `0.012661/0.007905`. Class-1 precision and
recall fell `0.006944/0.009174`. The candidate also caused `36/40/57` net
harms over corrections under dim/bright/low contrast and increased restricted
FP in every shifted condition.

The CE control obtained high precision by collapsing class-1 recall to
`0.330275`; it is not a usable baseline result. CapsALM protected class-1
support from this collapse, but did not improve raw behavior. Stage B,
validation, test, shared-trainer integration, and command promotion are denied.

## Mechanism audit

The ALM support constraint was active rather than silently detached:

- Final clean support violation mean was `0.020531` for CapsALM versus
  `0.649686` for CE control. Candidate support dual mean/max reached
  `0.433360/14.954435` and remained finite and nonnegative.
- Clean rank violation changed only `0.028232 -> 0.027954`, far short of the
  required 10% reduction. Dim, bright, and low-contrast rank behavior also
  failed the locked gate.
- Candidate class-1 residual-margin AUROC was only `0.652595` on clean and
  `0.645541/0.673347/0.649964` under the three shifts.
- Normalized routing entropy was about `0.9997`, and effective routed-patch
  count was about `163/167` in every condition. Three routing iterations did
  differ numerically from the one-iteration ablation, but the learned routing
  remained practically uniform.

The mechanism therefore demonstrates support preservation against an unsafe
natural-CE adapter, not selective positive class-1 evidence or improved
precision against the frozen keeper.

## XAI, export, and resource audit

- XAI covered all 107 required clean transitions over 36 pages. Pages 0, 17,
  21, and 35 were visually inspected. Candidate/control class-1 coupling maps
  were nearly identical and spread over most retained fruit patches; the
  dynamic-minus-uniform signal was weak. Close crops and wide-context samples
  appeared on both correction and harm sides.
- All maps were finite and invalid/pruned coupling mass was exactly zero. One
  bbox overlay was correctly all-zero: sample 3576 is a tiny class-0 object at
  the lower image edge whose retained tokens have no positive bbox prior. It
  became a class-1 false positive and is retained as a real context/pruning
  failure, not hidden as an audit exception.
- Full keeper-plus-adapter ONNX passed with maximum error `3.5763e-7` and exact
  argmax. Formal isolated adapter export failed because cached tensors created
  under `torch.inference_mode()` were passed to ONNX tracing. Commit `2d4a40f`
  clones them into normal tensors; a fresh isolated synthetic export then
  passed with zero error and exact argmax. The full formal behavior was not
  rerun merely to change this rejected deployment gate.
- Batch-32 runtime ratio was `0.958760`; full peak VRAM was `0.404137 GiB` and
  adapter incremental peak was `0.000620 GiB`.

The formal audit also reported one raw-versus-CIDT clean argmax difference at
sample 7240, a near tie. Current raw probabilities replay the locked MCL raw
probabilities bit-exactly across all 7,372 condition rows; CIDT had
`p0=0.244257, p1=0.243177`, while the current/MCL path had
`p0=0.242732, p1=0.244636`. CIDT supplies immutable folds/cohorts, but exact
CIDT argmax replay was not a protocol gate. Commit `2d4a40f` retains the
diagnostic and removes it from structural authorization.

## Replay and artifact closure

- Independent CSV recomputation verified all `4 x 1843 = 7372` rows, unique
  sample indices, probability argmax values, confusion matrices, macro/class-1
  metrics, restricted-FP counts, and correction/harm counts.
- All 47 manifest payloads matched their declared lengths and SHA-256 values.
  The evidence directory contains 66,105,330 manifest-declared bytes and no
  forbidden model binary.
- Evidence root:
  `runs/audit_support_constrained_capsule_alm_readiness_20260716`.
- Summary/prediction/artifact-manifest/independent-replay SHAs:
  `17bad124...7d7a66`, `3801ffd0...92330`, `804e8b31...29f39`, and
  `2e6ade36...11603`.
- XAI manifest/tensor/full-ONNX SHAs:
  `58f35124...b473`, `b6d986e5...67cbc`, and `b02574f0...b02a`.
- Compileall, focused tests `7/7`, full pytest `1203/1203`, launcher parse,
  clean-worktree preflight, protected hashes, and post-fix isolated ONNX replay
  passed.
- Read-only retention
  `runs/artifact_retention_audit_support_constrained_capsule_alm_closure_20260716`
  passed over 707 directories with `blockers=[]`, deleted nothing, and reported
  71.278 GB free. Retention summary SHA is
  `1157f7f58373e13d4603b7806693eb3ff525f06a72cddf4611617ff3eed1b72b`.

Keeper, random-init complement, current-best command packet, and command
history hashes remain exact. Current-best tracking stays at three revisions,
two updates after the initial revision, zero keeper replacements, and one
optional precision package.

## Decision and no-repeat boundary

Reject the exact support-constrained final-patch capsule ALM A0. Do not sweep
capsule dimensions, routing iterations, vote initialization, residual scale,
rank/support margins, ALM penalty or dual updates, optimizer, learning rate,
weight decay, epochs, cohort, condition, fold, seed, or a post-hoc threshold or
router on this keeper.

Dynamic routing remained almost uniform and did not create the missing
surface-boundary evidence. A future route must use a different representation
mechanism that can reject a candidate before training when its train-only
spatial/class signal is not selective. It must retain the raw, natural-CE,
MCL, and CapsALM results as comparators and preserve class-1 TP and precision
under all four lighting conditions.
