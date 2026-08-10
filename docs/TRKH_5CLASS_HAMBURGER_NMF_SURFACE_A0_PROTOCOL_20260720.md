# TRKH 5-Class Hamburger-NMF Surface Decomposition A0 Protocol - 2026-07-20

## Status And Scope

Prospectively locked before auditor implementation and before any new candidate
metric. This protocol authorizes one source-disjoint, train-only information
audit of nonnegative low-rank decomposition over frozen current-keeper object
tokens. It does not authorize validation/test access, model or trainer edits,
a checkpoint, a smoke/probe/full train, or current-best command promotion.

A complete A0 pass may authorize only one later default-off implementation and
a separately locked matched short smoke. It can never directly authorize a
full train. A negative result closes the exact projection, rank, iteration,
descriptor, and readout route on this keeper without claiming that every NMF
architecture is impossible.

## Prospective Provenance Erratum - 2026-07-20

The first no-output preflight stopped before model loading, dataset pixels,
descriptor construction, or candidate metrics because the official
`seg/HamNet/hamburger/ham.py` SHA-256 had one transcribed character out of
place. Direct `Get-FileHash`, the pinned clean repository, and a second hash
check agree on
`c6a261aa8fd7f932246b6e57addde4c3e971dbaa1f6a42e9a70bcbebeecc2829`.
This erratum corrects only that provenance string. It does not change the
source file, projection, cohort, token selection, NMF equation, descriptor,
readout, seed, gate, resource limit, or stop rule.

## Prospective Token-Support Erratum - 2026-07-20

The first two-row engineering forward stopped before projection,
decomposition, labels/readouts, or candidate metrics because the second row
had no final token with the ordinary keeper `patch_bbox_prior >= 0.5`. A
geometry-only diagnostic established that this keeper receives source-image
`bbox` coordinates for its ordinary forward, while `crop_bbox` contains the
same object rectangle after object-crop resize and square padding. For the two
fixed rows, the ordinary keeper prior selected `104/0` valid tokens; an
independently reconstructed transformed-crop prior selected `160/32`.

Keep the ordinary keeper forward exactly unchanged with source `bbox`, so its
logits, pruning, and final patch tokens remain faithful to the checkpoint.
After that forward, independently build a second audit-only overlap prior from
`crop_bbox` on the original `16x16` grid using the keeper's locked bbox margin
ratio `0.04`, per-row min-max normalization with denominator clamp `1e-6`, and
gather by the final `patch_indices`. Use this transformed-crop audit prior,
intersected with the unchanged key-padding validity mask, for the locked
`>=0.5` support rule. Persist both the ordinary keeper prior and the audit-only
prior. The at-least-16 requirement and no-fallback rule remain unchanged.

This correction aligns support geometry without changing any model input or
output. It adds no label, segmentation, top-k selection, adaptive dilation, or
metric-dependent choice. All projection, NMF, descriptor, readout, seed, gate,
resource, and stop settings remain unchanged.

## Prospective Host-Resource Erratum - 2026-07-20

The first formal launcher attempt stopped after focused and full tests passed,
but before the engineering forward, output-directory creation, dataset pixels,
or candidate metrics, because only `4.83 GiB` physical RAM was available versus
the locked `5 GiB` floor. A post-stop measurement showed `4.70 GiB` available
on this `15.64 GiB` host, no Python/TensorRT/FFmpeg process, and an idle RTX
4060. User-facing applications are left untouched.

Keep the CUDA batch at `64` and every scientific setting unchanged. Reduce only
the Windows DataLoader worker count from `4` to `2` and the fail-closed physical
RAM floor from `5 GiB` to `4.25 GiB`. Two workers bound duplicated loader and
pinned-prefetch state while batch `64` preserves the GPU work unit. The final
formal run must record requested/effective workers `2/2`; the two-row
engineering forward must be repeated under this resource lock. This is a
prospective operational correction and cannot be changed again after any
candidate metric is emitted.

## Research Question

Does a compact nonnegative low-rank reconstruction of the frozen keeper's
object-surface patch tokens distinguish true class-1 predictions from
restricted `0/2/4 -> 1` false positives better than:

- the keeper's clean probabilities alone;
- an equal-dimensional identity/raw-surface descriptor;
- an equal-dimensional optimal unconstrained rank-8 SVD descriptor; and
- a source-deranged NMF descriptor?

The question is deliberately narrower than whether Hamburger improves a full
classifier. Existing TRKH evidence shows foreground localization is generally
clean, distant-background perturbations are nearly inert, and the remaining
class-1 ambiguity lies mainly on fruit surface/boundary appearance. NMF is
therefore tested as an optimization-defined surface representation, not as
another attention selector, post-hoc probability sweep, or background mask.

## Primary Source And Licensed References

- Accepted primary paper: Zhengyang Geng, Meng-Hao Guo, Hongxu Chen, Xia Li,
  Ke Wei, and Zhouchen Lin, *Is Attention Better Than Matrix Decomposition?*,
  ICLR 2021, OpenReview `1FvkSpWosOl`.
- Local paper:
  `D:/DataAI/external_sources/papers/hamburger_iclr2021.pdf`, SHA-256
  `4eed8898973d9aa9a1101e32d436c95e57f0f61c5ef459b46f673e33666ae9f1`.
- Authors' official repository:
  `https://github.com/Gsunshine/Enjoy-Hamburger`, pinned locally at
  `D:/DataAI/external_sources/official/hamburger-iclr2021`:
  - commit `d9b51f6f197486df68c6e059e396520680157c08`;
  - tree `a399506b5556b8d243a80b2cfeee47ba64b02c2e`;
  - GPL-3.0 `LICENSE` SHA-256
    `230184f60bae2feaf244f10a8bac053c8ff33a183bcc365b4d8b876d2b7f4809`;
  - reference `seg/HamNet/hamburger/ham.py` SHA-256
    `c6a261aa8fd7f932246b6e57addde4c3e971dbaa1f6a42e9a70bcbebeecc2829`.
- Independent licensed numerical reference: installed scikit-learn `1.6.1`:
  - BSD-3-Clause `COPYING` SHA-256
    `1b74e02d0cb8e6502091124787fc91695b9dd92c9cb146d9d34830f9a300ab3c`;
  - `sklearn/decomposition/_nmf.py` SHA-256
    `6f0ed846e1531f773eb82de20dfa4f07e57f53fa472f3f8c576e8ee3d7007369`.

The paper defines Hamburger as a low-rank reconstruction block whose matrix
decomposition optimizer supplies the global-context mechanism. The official
source provides the authors' NMF defaults and multiplicative-update reference,
but is GPL-3.0. No official source code is copied into TRKH. The auditor is an
independent implementation of the published equations, with a separately
written NumPy oracle and a scikit-learn BSD reference check on the NMF
objective. Source hashes prove what was reviewed, not code provenance.

The original reported tasks are semantic segmentation and image generation,
not five-class mango classification. Its large segmentation backbones and
160k-iteration recipes provide no evidence that this scratch TRKH model will
converge within 30 dataset epochs. A0 transfers only the nonnegative low-rank
factorization mechanism.

## Literature Screen And No-Repeat Boundary

- DeepTEN residual encoding, learned/fixed histograms, Gabor/LHO/FCM texture,
  bilinear/second-order pooling, nearest prototypes, part discovery, MIL/
  WILDCAT, attention voting/selection, and frozen perturbation responses have
  already been tested and rejected on this keeper.
- Hamburger-NMF differs by solving a nonnegative low-rank reconstruction per
  object-token matrix. It does not assign residuals to learned codewords,
  compute covariance, select top-attention tokens, paste pixels, or use a
  pretrained teacher.
- The SVD role is a mandatory low-rank placebo. NMF must beat the optimal
  unconstrained rank-8 reconstruction, not merely outperform probabilities.
- The raw role is a mandatory capacity placebo. Every non-base role has the
  same number and ordering of descriptor dimensions.
- Raw images, labels, YAML, split declarations, and retained evidence are
  immutable. No generated value is written into either dataset tree.
- No rank, projection, ReLU, normalization, token threshold, iteration,
  epsilon, seed, descriptor, logistic `C`, fold, threshold, or gate sweep is
  allowed after candidate metrics.

## Locked Inputs

- Pre-protocol HEAD/upstream:
  `b05e226edd3cf6e1bcd3aaa45713e6be9ee55cc9`.
- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper `launcher_args.json` and `resolved_config.json`, SHA-256
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`
  and
  `e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674`.
- Dataset declaration: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- CIDT train-only source-fold summary and prediction CSV:
  `runs/audit_cidt_readiness_full_train_20260714/summary.json` and
  `predictions_all_conditions.csv`, SHA-256
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`
  and
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Current-best command/history SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`
  and
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

The formal auditor must verify every lock, the official repository commit/tree
and clean state, installed scikit-learn version, and all seven protected
untracked payload hashes before creating its output directory. The tracked
TRKH worktree must be clean and HEAD must equal upstream.

## Locked Cohort And Folds

Reuse only CIDT `clean` train rows whose keeper prediction is class 1 and whose
target is in `{0,1,2,4}`. The cohort is exactly 750 objects:

- 528 true class-1 positives;
- 222 restricted false positives;
- fold TP/FP counts:
  `107/36`, `112/45`, `100/48`, `101/52`, `108/41`;
- restricted-FP target totals by fold:
  `0:[23,36,37,36,26]`, `2:[12,9,8,13,12]`,
  `4:[1,0,3,3,3]`;
- ordered sample-index SHA-256, serialized as comma-joined ASCII integers with
  no trailing delimiter,
  `55913ec45265b156a611dc96c779e46b08f28582737af8269105afe6f65694d7`.

Use immutable CIDT source folds `0..4`. Every path must resolve below
`yolo_f/images/train`; any validation/test path, prediction, label, pixel, or
metric access is fatal. Fit and held source stems must have zero overlap.

## Frozen Token Extraction

Run the ordinary metadata-aware keeper path in evaluation mode and float32,
with unchanged `image_mask` and source object `bbox`. Persist five logits,
probabilities, final post-pruning `patches`, aligned `patch_indices`, ordinary
keeper `patch_bbox_prior`, transformed-crop audit prior, and
`memory_key_padding_mask`. Model state before and after each condition must be
bit-identical. The prospective token-support erratum supersedes the ambiguous
"aligned normalized object-bbox prior" wording below: selection uses only the
independently reconstructed transformed-crop audit prior.

For each row:

1. a token is valid only if its patch index is in `[0,255]` and its key-padding
   mask is false;
2. select valid tokens whose aligned normalized object-bbox prior is at least
   `0.5`;
3. require at least 16 selected tokens; no top-k or fallback is allowed;
4. sort selected tokens by ascending original patch index;
5. multiply the float32 `N x 256` token matrix in float64 by the locked fixed
   `256 x 32` projection, apply ReLU, then divide each projected channel by its
   per-row RMS plus `1e-6`; cast the resulting `N x 32` matrix to float32 and
   use that persisted value as the sole decomposition input.

The projection is generated by NumPy `default_rng(20260720).standard_normal`
in float64, reduced QR, and a sign correction that makes each diagonal entry
of `R` nonnegative, then cast to float32. Its dtype/shape/byte-array SHA-256 is
`b93eafaee2cd389e5483c42a9db048f7734905a4df94f35353fc859ebb11a42a`;
maximum float32 orthogonality error must be at most `1e-6`.

Labels may not enter token selection, projection, normalization,
factorization, descriptor construction, seed generation, or source
derangement. They are exposed only after all descriptor banks and the
derangement mapping have been persisted.

## Exact Decomposition Roles

Let `X in R_+^(N x 32)` be one projected object-token matrix, `D=32`, rank
`R=8`, update count `T=6`, and denominator epsilon `1e-6`.

### Raw Surface Control

Use reconstruction `X_hat = X` and residual `E = X - X_hat = 0`. This is an
equal-dimensional identity/capacity control, not a candidate.

### SVD Rank-8 Control

Compute the float64 reduced SVD of `X` and use its unconstrained optimal
rank-8 Frobenius reconstruction. Do not clamp negative reconstructed values.
The reconstruction itself is invariant to singular-vector signs.

### Hamburger-NMF Candidate

Work with the transposed nonnegative matrix `V = X^T in R_+^(32 x N)`.
Initialize positive bases `B in R_+^(32 x 8)` from a local NumPy PCG64 uniform
generator, normalize each basis column to unit L2 norm, then initialize
coefficients `C in R_+^(N x 8)` by row-wise softmax of `V^T B`.

For exactly six alternating steps, independently implement:

```text
C <- C * (V^T B) / (C (B^T B) + 1e-6)
B <- B * (V C)   / (B (C^T C) + 1e-6)
```

Then perform one final coefficient update with the final `B` and reconstruct
`X_hat = (B C^T)^T`. Initial and every full-step Frobenius objectives must be
finite and non-increasing within `1e-10` relative tolerance. All factor values
must stay nonnegative within `1e-12`.

The candidate local seed is
`20260720 + 1000003*sample_index + 17`. A mandatory seed-repeat role uses the
identical equation and descriptor with seed offset `+7919`. Global Python,
NumPy, and Torch RNG state may not be consumed.

The production Torch-float64 equation must match a separately written
NumPy-float64 oracle on fixed synthetic matrices with maximum reconstruction,
basis, coefficient, and objective error `<=1e-10`. A scikit-learn BSD
reference run is used only to verify nonnegativity and that the independently
implemented result is a valid objective-reducing NMF solution; sklearn output
is not used as a feature or target.

## Equal-Dimensional Surface Descriptor

For raw, SVD, candidate NMF, and seed-repeat NMF, compute the same 284 values
from `X`, reconstruction `X_hat`, and absolute residual `A=abs(X-X_hat)`.
All NumPy quantiles use `method="linear"`.

1. For each of 32 reconstruction channels, concatenate moment-major
   `mean`, population `std`, `max`, and `q75`: 128 values.
2. For each of 32 absolute-residual channels, concatenate the same four
   statistics: 128 values.
3. For each token, calculate reconstruction L2 norm, residual L2 norm, and
   cosine similarity between `X` and `X_hat` with epsilon `1e-12`. Summarize
   each vector by `mean`, population `std`, `min`, `max`, `q25`, `q50`, `q75`,
   and `q90`: 24 values.
4. Append four method-neutral scalars: relative Frobenius residual,
   explained energy, effective rank of `X_hat` divided by `min(N,32)`, and
   reconstruction-to-input L1 ratio.

Every non-base readout appends these 284 values to the same six-value base
vector: five clipped (`1e-8`) log probabilities plus the class-1 logit margin
against maximum class `0/2/4`. Locked roles and dimensions are:

- `base_only`: 6;
- `raw_surface_control`: 290;
- `svd_rank8_control`: 290;
- `nmf_candidate`: 290;
- `nmf_seed_repeat`: 290;
- `source_deranged`: 290.

For `source_deranged`, cyclically reassign the clean candidate surface
descriptor inside each held fold and clean base-margin quartile. Every source
must come from a different source stem. Construct and persist this mapping
before labels are exposed, then reuse exactly the same mapping under lighting
conditions.

## Locked Readout And Actions

For each role and each held CIDT fold, fit on the other four folds only:

- `StandardScaler` fit on training folds;
- L2 logistic regression, `C=0.1`, `class_weight="balanced"`, solver `lbfgs`,
  `max_iter=4000`, tolerance `1e-9`, seed `20260720`;
- binary target 1 for true class 1 and 0 for restricted false positive;
- fit-fold threshold chosen to retain at least 97% positive rows, allowing
  exactly `floor(0.03*n_positive)` lowest positive scores below threshold;
- apply scaler, model, and threshold once to the held fold.

If accepted, the hypothetical action remains class 1. If rejected, it becomes
the highest-probability restricted rival among classes `0/2/4`. This action is
an information-gate diagnostic only; it is not added to inference code.

## Clean Automated Gate

All checks are conjunctive. The candidate must satisfy:

- AUROC `>=0.85`;
- AUROC gain `>=0.02` over base-only;
- AUROC gain `>=0.01` over raw surface;
- AUROC gain `>=0.01` over rank-8 SVD;
- AUROC gain `>=0.02` over source derangement;
- TP retention `>=0.95` and restricted-FP rejection `>=0.25`;
- at least ten more FP rejections than the strongest base/raw/SVD control;
- corrections at least twice harms;
- higher AUROC than both raw and SVD in at least four of five held folds;
- minimum held-fold TP retention `>=0.90`;
- target-0 and target-2 FP rejection each `>=0.15`, and at least one target-4
  FP rejection;
- candidate descriptor effective rank `>=12` across rows;
- median row-wise cosine similarity between candidate and seed-repeat surface
  descriptors `>=0.95`;
- absolute candidate/seed-repeat AUROC difference `<=0.01` and action agreement
  `>=0.95`;
- every NMF objective/nonnegativity, source-isolation, dimensionality,
  provenance, model-state, resource, replay, and action-semantics check passes.

The clean gate fails if any control is identical to the candidate, any readout
fails to converge, any selected row has fewer than 16 object tokens, or any
candidate metric is accessed before the prospective protocol and committed
auditor locks agree.

## Conditional Lighting Stage B

Run Stage B only if every clean automated gate passes. Reuse the same 750 train
rows under the already established deterministic conditions:

- `lighting_dim`: brightness `0.70`, contrast `0.90`;
- `lighting_bright`: brightness `1.25`, contrast `1.10`;
- `low_contrast`: brightness `1.00`, contrast `0.65`.

Do not refit any scaler, coefficient, threshold, projection, factorization
setting, or source mapping. Apply the clean-fitted fold states to each held-fold
condition descriptor. Every condition must satisfy:

- candidate AUROC `>=0.78`;
- candidate AUROC gain `>=0.005` over the strongest base/raw/SVD role;
- TP retention `>=0.92` and FP rejection `>=0.15`;
- corrections not below harms;
- target-0 and target-2 FP rejection each `>=0.08`;
- candidate/seed-repeat action agreement `>=0.90`;
- all extraction, state, source, finite-value, and replay checks pass.

Any clean failure skips all three conditions. Any condition failure closes A0
before model integration.

## Replay, XAI, Resource, And Stop Rules

- Persist padded projected token matrices, counts, patch indices, bbox priors,
  logits/probabilities, all descriptor banks, derangement mapping, fold states,
  scores, actions, objectives, and artifact hashes.
- A second process must rebuild raw/SVD/NMF/seed-repeat descriptors from the
  persisted projected matrices with the independent NumPy oracle, reconstruct
  every scaler/logistic score/threshold/action, and match metrics/gates and
  hashes. Maximum score and threshold error is `1e-10`; actions and gates must
  match exactly.
- Render a fixed 12-row sheet using the three smallest sample indices from each
  target category `1/0/2/4`. Columns are input crop, selected object-token
  support, dominant NMF component, NMF reconstruction strength, and residual
  heat. The sheet must show finite, object-aligned support and no invalid/padded
  token use. Manual review cannot rescue an automated failure.
- Use CUDA batch size `64`, requested loader workers `2`, seed `20260720`, and
  record requested/effective workers. Peak allocated CUDA memory must be
  `<=3.5 GiB`; available system RAM preflight must be `>=4.25 GiB`.
- Compare clean probabilities with the locked CIDT cache at maximum absolute
  tolerance `3e-5` and require exact argmax. This bound is prospectively set
  above the already documented same-keeper BF16 replay difference
  `2.261996e-5`; it is not derived from NMF metrics.
- Formal output is create-once. A partial or failed output may be removed only
  after its hashes/reason are recorded and its absolute path is verified below
  the intended `runs` directory.
- No validation/test access, model/trainer edit, smoke/probe/checkpoint/full
  train, export, video test, shutdown, or current-best command/history update
  is authorized unless the entire automated clean/lighting gate, independent
  replay, and fixed visual review pass.
