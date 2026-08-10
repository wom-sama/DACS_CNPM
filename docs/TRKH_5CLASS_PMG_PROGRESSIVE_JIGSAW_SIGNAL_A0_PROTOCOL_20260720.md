# TRKH 5-Class PMG Progressive-Jigsaw Signal A0 Protocol (2026-07-20)

## Decision being tested

This is a train-only information gate for one equation-distinct fine-grained
classification hypothesis. It does not authorize a model change by itself.

The hypothesis is that stage-aligned local-to-global jigsaw views expose a
class-1-positive surface representation that is absent from both the clean
keeper representation and non-progressive multi-view controls. Only a complete
pass may authorize one matched short training pair that implements sequential
PMG updates in TRKH.

The route is selected from Du et al., *Fine-Grained Visual Classification via
Progressive Multi-Granularity Training of Jigsaw Patches*, ECCV 2020. The
accepted paper defines, for the last `S=3` stages, the sequence

`n_l = 2^(L-l+1) = {8,4,2}`

and performs one optimizer update for each stage-specific cross-entropy loss,
followed by a clean fused-output update. This sequential optimizer ordering is
the material distinction from the already rejected TRKH multi-granularity CE,
contrastive, HERBS-style refinement, and DCL-lite routes.

## Primary-source lock

Only the accepted paper and authors' MIT-licensed repository are normative.
No implementation is copied into TRKH.

- Accepted paper:
  `D:\DataAI\external_sources\papers\PMG_ECCV_2020_accepted.pdf`
- Paper SHA-256:
  `ae70129965ea54459d6d59f1a6257bac7361d4972c63234740225e7be0f2da85`
- Official repository:
  `D:\DataAI\external_sources\repositories\PMG-Progressive-Multi-Granularity-Training`
- Commit:
  `db7a7d7ab5fd91c2e322e1fcf200cbfbe192b51a`
- Tree:
  `85949974008e279c885510bf564396c5f9dbe99e`
- License SHA-256:
  `d50df6b22917cf3f93a9d9987339e88b972342edd83aedb49897c91b034f5270`
- `train.py` SHA-256:
  `8331cb58e5e37c77e7e8755b619c69e8b9bfb11dfe0ca532be64a019ecc9fed0`
- `model.py` SHA-256:
  `fc0ad1b6718e38d84699d7e19f096c5347fbb3f949c021bcc47edbdad4d421c9`
- `utils.py` SHA-256:
  `44706f38646ac8c2bb2346ee89d6b4d0d9836e028f5cceadaa3f9f8c9fc06d28`

The paper and source impose adverse evidence that must remain visible in every
decision. Their reported recipe uses pretrained convolutional layers, input
`448`, batch `16`, up to `200` epochs, and four forward/backward/optimizer
updates per logical batch. The repository README says "train from scratch",
but its released `train.py` calls `load_model(..., pretrain=True)`. Therefore
the reported benchmark is not evidence that a scratch TRKH model will converge
within 30 epochs or fit this laptop. A0 exists to prevent that unsupported leap.

## Immutable project inputs

- Repository branch: `classification-only-research`.
- Current keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Dataset declaration:
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`
- Dataset declaration SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`
- CIDT prediction cache:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`
- CIDT prediction SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`
- CIDT summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`
- Current-best command SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`
- Current-best history SHA-256:
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`

The formal must use all `9,215` `yolo_f/train` object rows in their immutable
dataset order. Validation and test pixels, labels, predictions, thresholds, or
metrics are forbidden. Reading only validation/test filenames to verify zero
source overlap is allowed and must be declared.

## Fixed folds and cohorts

- Five deterministic source-disjoint folds must exactly replay the existing
  CIDT fold column and use seed `20260714`.
- No source stem may cross folds or train/validation/test source boundaries.
- Focus class is class `1`.
- Restricted negative classes are `0`, `2`, and `4`.
- Keeper cohorts are fixed before candidate inference from the CIDT clean
  prediction: class-1 true positives, class-1 false negatives, restricted
  false positives into class 1, and all remaining rows.

## Fixed jigsaw transform

For normalized image tensor `x` and grid `n in {8,4,2}`, split `x` into
non-overlapping `n x n` patches. Apply a permutation without replacement and
reassemble the patches without resize, interpolation, crop, color change, or
pixel replacement. Apply the identical permutation to `image_mask`.

The permutation is deterministic per `(seed, sample_index, n)` and is generated
from one SHA-256-derived 64-bit seed. It is not selected from labels, keeper
outputs, fold, or any metric. Image size `256` is divisible by every locked
grid. Each output patch must have exactly one input origin and pixel/mask
multisets must be preserved bit-exact.

The clean view retains original metadata. Jigsaw views retain immutable sample
identity and class target. Bbox coordinates are not used by the frozen stage
descriptor and may not be converted into a post-hoc selector.

## Frozen stage descriptor

The keeper remains in `eval` mode and FP32. Its parameters and buffers must be
hashed before and after extraction and remain bit-identical. No auxiliary head,
adapter, threshold, or model parameter is trained.

Capture the post-block, pre-current-pruning token tensor at Transformer blocks
`2`, `5`, and `8`. The keeper has no deep prompts and a fixed prefix count.
For stage `l` and view `v`, remove prefix tokens, apply the keeper's final
LayerNorm independently to every patch token, then compute

`z_l(v) = L2Norm(0.5 * mean_patch(LN(T_l(v))) + 0.5 * max_patch(LN(T_l(v))))`.

The fixed global component `g(clean)` is the keeper's clean final pooled
feature after its normal forward path. Every role has exactly `4 * 256 = 1024`
features:

1. `clean_control = [z_2(clean), z_5(clean), z_8(clean), g(clean)]`.
2. `pmg_aligned = [z_2(P_8), z_5(P_4), z_8(P_2), g(clean)]`.
3. `reverse_placebo = [z_2(P_2), z_5(P_4), z_8(P_8), g(clean)]`.
4. `deepest_placebo = [z_8(P_8), z_8(P_4), z_8(P_2), g(clean)]`.

All four roles are assembled from the same extracted feature bank and have the
same dimension, row order, fold assignment, clean global component, and readout
budget. The reverse role tests whether the paper's stage/granularity ordering
matters. The deepest role tests whether multiple destructive views alone are
sufficient without progressive stage specialization.

## Fixed OOF readout

For each role and each holdout fold:

1. fit `StandardScaler` on the other four folds only;
2. fit one natural-prior multinomial `LogisticRegression` with `C=0.05`, L2
   penalty, `lbfgs`, no class weights, `tol=1e-6`, and `max_iter=1000`;
3. emit probabilities only for the untouched holdout fold.

There is no `C`, threshold, class-weight, feature, grid, seed, solver, or fold
sweep. Every fit must converge before `max_iter`. Persist all OOF probabilities
and enough model telemetry to independently reconstruct metrics and gates.

## Required reports

The formal output must contain:

- locked file, source commit/tree, repository state, and dataset identity;
- exact CIDT row/target/fold/path/source and clean-keeper probability replay;
- transformation permutation checks and a fixed five-class contact sheet with
  columns `clean`, `P_8`, `P_4`, and `P_2`;
- extraction runtime, source-image/view throughput, worker resolution, peak
  CUDA allocation, and model state before/after hashes;
- OOF macro F1, five-class precision/recall/F1, confusion matrix, ECE, and
  convergence telemetry for every role;
- candidate-minus-control transitions, corrections, harms, class-1 FN rescue,
  class-1 TP break, restricted-FP remove/create/net counts, and all five fold
  deltas;
- class-1-TP-versus-restricted-FP directional AUROC for candidate, reverse,
  and deepest `delta_p1` relative to the clean control;
- replay from persisted probabilities, payload hashes, and a complete artifact
  manifest.

## Structural gates

Every structural gate is mandatory:

- all locked hashes and official commit/tree/license match;
- tracked worktree is clean, HEAD equals upstream, and protected untracked
  user paths are neither staged nor modified;
- only all `9,215` train rows are opened and dataset identity matches;
- row order, targets, paths, source stems, and folds exactly match CIDT;
- clean keeper argmax is exact and probability maximum error is at most `2e-5`;
- all four descriptor roles have shape `[9215,1024]`, finite values, identical
  clean-global blocks, and nonzero effective rank;
- every jigsaw patch mapping is bijective and the image/mask patch multisets are
  preserved exactly on the locked engineering checks;
- all 20 OOF fits converge, probabilities are finite and normalized, and every
  row is predicted exactly once by a model that did not fit its fold;
- model parameter/buffer state is bit-identical before and after extraction;
- requested loader workers are `4`, the safe loader resolves without fallback,
  and peak CUDA allocation is at most `7.5 GiB`;
- independent replay differs by at most `1e-12` and manifest verification passes.

## Mechanism gates

All mechanism gates are conjunctive. Compare `pmg_aligned` against
`clean_control` unless another comparator is named.

- macro F1 delta is at least `+0.003`;
- class-1 F1 delta is at least `+0.010`;
- class-1 precision delta is at least `+0.010`;
- class-1 recall delta is at least `-0.005`;
- corrections exceed harms;
- restricted-FP net removal is at least `10`;
- class-1 TP breaks do not exceed FN rescues by more than one;
- at least four folds have non-worse class-1 precision;
- at least three folds have positive class-1 F1 delta;
- no fold loses more than `0.03` class-1 recall;
- candidate class-1 F1 exceeds both reverse and deepest placebos by at least
  `0.005`;
- candidate directional AUROC is at least `0.60` and exceeds both placebo
  directional AUROCs by at least `0.03`;
- candidate effective rank is at least `16` and no descriptor block is constant.

No aggregate score can compensate for one failed gate. Manual review must also
confirm that the contact sheet preserves local mango pixels/masks and does not
introduce resize, interpolation, fill, or label-dependent geometry.

## Stop and promotion rules

If any structural, mechanism, replay, or visual-geometry gate fails, close exact
PMG stage-aligned jigsaw on the current keeper before model/trainer integration.
Do not sweep neighboring grids, permutation seeds, readout `C`, fusion weights,
stage choices, jigsaw probability, or a post-hoc router.

If every gate passes, the only authorized next step is one prospectively locked
matched short pair. The candidate must implement the official semantic order:
three separate jigsaw stage losses with three sequential optimizer steps, then
one clean fused loss and optimizer step. Its control must use the same number of
forward/backward/optimizer operations without stage/granularity alignment. The
pair must still pass full validation, robustness, runtime, strict replay, all
audits, and XAI before any probe or full train.

A full train remains forbidden until that matched pair passes. If later
authorized, it is capped at 30 epochs, uses early stopping and workers selected
for the RTX 4060 laptop, and may update current-best command files only after a
locked single-model validation winner. Test remains sealed until final locked
evaluation.
