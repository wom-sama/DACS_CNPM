# TRKH 5-Class PDiscoFormer-Style Semantic-Part Readiness Protocol (2026-07-12)

## Decision to test

Test whether the frozen no-pretrain TRKH keeper patch representation can support
two stable foreground parts plus one background assignment whose direct
classification decisions improve class 1. This is a bounded readiness diagnostic,
not a PDiscoFormer reproduction and not permission to start a full image-model run.

The current `AdaptivePartTokenLearner` is not this mechanism. It learns four
foreground/bbox-biased attention pools and a residual/pairwise classifier, but it
does not use a competing background slot, spatial total variation, foreground
presence, pixel assignment entropy, part-feature orthogonality, transform
equivariance, or part dropout as a combined semantic-part objective. Its exact
four-part pairwise route is already rejected and must not be repeated.

## Primary-source lock

- PDiscoFormer paper: <https://arxiv.org/abs/2407.04538>
- Official MIT repository: <https://github.com/ananthu-aniraj/pdiscoformer>
- Locally inspected official commit:
  `1a872e2bed2ea38c4b078fd69294126e9f6b2f33`
- Official reference files inspected:
  `models/individual_landmark_vit.py`,
  `engine/distributed_trainer_pdisco.py`, and `engine/losses/*`.

The official recipe assigns every spatial feature to `K` foreground parts plus a
background channel using negative squared prototype distance and soft/Gumbel
softmax. It pools one feature per part, applies part dropout, and averages shared
per-part class logits. Its paper recipe uses DINOv2 pretrained features and freezes
the backbone. That pretrained representation is a material dependency, so the
scratch TRKH keeper must pass this diagnostic before any trainer integration.

## Immutable inputs

- Repository branch: `classification-only-research`.
- Dataset: `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`.
- Allowed splits: `train` and `val` only. `test` is forbidden.
- Keeper checkpoint:
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Locked independent FP32 keeper validation macro/class-1 F1:
  `0.8829248142 / 0.6782608696` over all `2606` rows.
- Class order comes only from the data specification/checkpoint and must contain
  exactly five classes with focus class index `1`.
- No raw image, label, YAML, split, or object row may be edited.

## Fixed feature evidence

- Rebuild the keeper from the checkpoint and run it in evaluation mode with all
  model parameters frozen.
- Use independent FP32 keeper forwards. Save direct keeper probabilities before
  token-cache quantization and require the locked validation metrics above.
- Use the keeper's final post-pruning patch tokens with original patch indices.
  Expected geometry is a `16 x 16` source grid and a fixed retained-token count.
- Store token caches as FP16 only for bounded disk/RAM use; candidate and matched
  control read the exact same cache. Record shapes, dtypes, paths, labels,
  `sample_index`, source stem, valid mask, crop bbox, checkpoint hash, and every
  cache-file hash.
- Extract exactly two views:
  1. the unchanged evaluation tensor;
  2. an exact clockwise 90-degree tensor rotation.
- Rotate `image_mask` and transform normalized `crop_bbox` as
  `(cx, cy, w, h) -> (1-cy, cx, h, w)`. Map rotated patch indices back to the
  original coordinate frame before equivariance comparison.
- Missing pruned or padded grid positions are explicit background positions. They
  must not be silently removed from spatial losses.

## Locked head

- Foreground parts `K=2`, plus one background channel. This single value is fixed
  because mango maturity is a whole-surface problem without reliable anatomical
  subparts; it also avoids repeating the rejected four-part TRKH head. Do not
  sweep `K`.
- Learn one prototype per channel. Assignment logits are the negative squared
  Euclidean distance between LayerNorm patch features and prototypes.
- Training assignment: soft Gumbel softmax, temperature `1.0`, non-hard.
- Evaluation assignment: deterministic softmax, temperature `1.0`.
- Pool foreground part features over the full grid, apply one shared per-part
  LayerNorm, part dropout `0.30`, one shared bias-free linear classifier, and mean
  the two foreground-part logits.
- Candidate and control must have identical modules, parameter counts,
  initialization, batches, classification loss, optimizer, scheduler, and epochs.
  Their only difference is that the candidate enables the semantic-part losses.

## Locked optimization

- Five `StratifiedGroupKFold` train folds, grouped by case-folded source stem.
- Seed `20260712`; one seed only.
- Adam, weight decay `0`, learning rate `0.01414`. This is the effective official
  scratch-layer rate from `1.414e-6 * 1e4`; it is not a searched TRKH value.
- StepLR, step size `4`, gamma `0.5`, matching the official schedule family.
- Batch size `64`, exactly `12` fixed epochs, no early stopping and no
  validation-selected checkpoint.
- Standard cross-entropy on the unchanged view is weight `1.0` for both arms.
- Candidate-only official-form weights:
  - foreground presence `1.0`;
  - transform equivariance `1.0`;
  - part-feature orthogonality `1.0`;
  - total variation `1.0`;
  - enforced edge-background presence `2.0`;
  - pixel-wise assignment entropy `1.0`.
- The rotated view supplies only the part-equivariance term, as in the official
  training structure. It does not receive an extra classification label.
- Train one final candidate/control pair on all train rows for the same 12 epochs,
  then evaluate once on the untouched full validation split.
- The keeper backbone was trained on all train rows. Source-grouped head OOF is
  therefore head-OOF evidence over an in-sample keeper representation, not a true
  OOF keeper backbone. This limitation must be stated in every result.

## Preflight and full-run sequence

1. Compile and run focused unit tests for rotation, assignment reconstruction,
   losses, source grouping, matched initialization, and fail-closed gates.
2. Run a capped `128 train / 128 val` extraction and two-epoch execution only to
   validate shapes, hashes, determinism, finite gradients, and output contracts.
   Capped metrics cannot approve or reject the method.
3. Delete the capped token payload after its contract evidence is recorded.
4. Run one full `9215 train / 2606 val` cache extraction and the single locked
   five-fold/final-head experiment.
5. Independently reconstruct metrics, transitions, gate booleans, and payload
   hashes before interpreting the result.
6. Build one balanced visual review from validation: each class plus class-1
   keeper TP, FN, FP, candidate corrections, and candidate harms. Show source
   image, crop bbox, both discovered parts, and background assignment.

## Promotion gates

All support/provenance gates must pass:

- train rows `9215`, validation rows `2606`;
- train/validation source overlap `0`;
- five disjoint source-grouped folds with complete one-time OOF assignment;
- keeper hash and direct FP32 metrics match the lock;
- finite losses/gradients and candidate/control parameter/init equality;
- no test access and no raw-data write.

All classification gates must pass:

- candidate OOF macro and class-1 F1 are not below matched control;
- candidate validation macro and class-1 F1 are not below matched control;
- candidate wins class-1 F1 in at least `3/5` train folds;
- candidate validation macro F1 is at least `0.8829248142`;
- candidate validation class-1 F1 is at least `0.6782608696`;
- versus keeper on validation: corrections are at least harms, class-1 FN rescues
  are at least TP breaks, and class-1 FP removals are at least FP creations;
- candidate-minus-control class-1 delta direction AUROC is at least `0.55` in
  both head-OOF train and validation, with absolute transfer gap at most `0.10`.

All semantic-part gates must pass on validation:

- each foreground part receives dataset-mean assignment mass in `[0.10, 0.75]`;
- normalized entropy of aggregate foreground-part use is at least `0.75`;
- candidate background assignment outside the crop bbox exceeds its inside-bbox
  assignment by at least `0.10`;
- candidate foreground-map equivariance cosine is at least `0.75` and exceeds
  matched control by at least `0.02`;
- candidate total variation and pixel-wise entropy are each no greater than the
  matched control values;
- no visual-review cohort is dominated by padding, hands, image edges, or broad
  color while being described as a semantic fruit part.

## Stop and follow-up rules

- Failure of any classification/direction gate closes this exact frozen
  PDiscoFormer-style route before an image-model smoke. Do not sweep part count,
  temperature, loss weights, transform, optimizer, LR, epochs, dropout, map layer,
  token-pruning mode, or add a router/residual from the same evidence.
- Passing every gate permits a default-off integrated TRKH semantic-part module
  and one matched image-level smoke. It does not promote the command file by
  itself.
- `docs\TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt` and deployment
  wrappers remain unchanged unless a later integrated candidate independently
  beats the locked validation gate and passes the full audit suite.
- After closure, retain compact metrics, predictions, visual review, provenance,
  and manifests; hash and delete reproducible multi-GB token caches and rejected
  checkpoints using the guarded cleanup workflow.
