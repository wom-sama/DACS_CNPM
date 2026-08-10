# TRKH 5-Class Deep Class-Prompt Readiness Protocol

Date locked: 2026-07-15

This protocol is immutable after the first formal Stage-A run. Any later
change to the architecture, optimizer, fold, budget, gate, or source hashes is
a new experiment, not a retry of this one.

## 1. Research question

Can one prompt per mango class, inserted at every existing TRKH Transformer
block, learn class-specific foreground evidence that increases class-1
precision without discarding true class-1 positives?

The experiment keeps the current no-external-pretrain keeper backbone and raw
dataset unchanged. It is not another late class query, output router,
prototype, detector curriculum, local stem, or background suppressor.

## 2. Primary-source lock

### Prompt-CAM

- Paper: *Prompt-CAM: Making Vision Transformers Interpretable for
  Fine-Grained Analysis*, CVPR 2025.
- Official repository: `https://github.com/Imageomics/Prompt_CAM`.
- Reviewed commit: `4d35f3fb2eb99a63859465fcc1c7c3f4879f3ac5`.
- `model/vision_transformer.py` SHA-256:
  `f409d3e101e01dffd05342b1cff798fb739046ced730d7451ddf5e0f49aa647d`.
- `experiment/build_model.py` SHA-256:
  `7b1e59dd88dfd269cf9c494426f9f2d546cc5903023f56f2aeff1cddcbfa3a62`.
- Source behavior used here: deep prompts are inserted at each Transformer
  block, removed before the next base-token step, normalized at the output,
  and scored by one shared scalar head. The official implementation trains
  prompts and the head while freezing the pretrained vision tower.

### MCTformer+

- Paper: *Multi-Class Token Transformer for Weakly Supervised Semantic
  Segmentation*, CVPR 2022, plus the MCTformer+ extension.
- Official repository: `https://github.com/xulianuwa/MCTformer`.
- Reviewed commit: `0acc27ada87a5582053efb14648442d8644168aa`.
- `models.py` SHA-256:
  `39fd884077bfb56d5bb21635ab0fc7d80148813d920a7c0e9329051fc6a5e00e`.
- Source behavior used here: one class token per class produces explicit
  class-to-patch attention, and class-token and dense-patch evidence are
  combined for prediction.

The TRKH module is an equation- and source-traceable adaptation, not an exact
reproduction. Prompt-CAM uses externally pretrained ViTs and long schedules;
those results do not establish success for this no-external-pretrain keeper.

## 3. Local no-repeat boundary

The following routes are already closed and are not reopened:

- frozen INTR-style class-query readout: validation class-1 F1
  `0.6841 -> 0.4426`;
- CaiT-style late class-attention pool and pool-plus-head: class-1 F1
  `0.6765/0.6784`, both below the keeper;
- bbox-interior token-label supervision: class-1 F1 `0.6648` and wider
  false-positive behavior;
- whole-patch bilinear, interior covariance, part tokens, objectness,
  detector/PCGrad, Finer-CAM routing, and border suppression;
- CAL at official commit
  `0ba9d5084f2532eeb21c9ef051c23f8b339595ff`, whose
  `fgvc/models/cal.py` SHA is
  `ecc9c453a527c765daf2437ed479d64a1a3f76a380444265a38607e6213f4749`.
  Its bilinear attention pooling plus attention crop/drop overlaps closed
  local families.

This experiment is distinct because every class has a prompt at every block,
the model exports class-to-patch maps directly, and the matched comparison
isolates learning the prompts from merely fitting their shared scalar head.

## 4. Locked TRKH adaptation

- Feature: `deep_class_prompt`, default off.
- Classes/prompts per block: exactly `5`, in canonical dataset order.
- Blocks: all existing `8` Transformer blocks.
- Prompt dimension: keeper embedding dimension `256`.
- Prompt insertion: after the existing CLS/register/branch prefix and before
  patch tokens.
- Prompt removal: immediately after each block and after any pruning action.
  The base sequence entering the next block therefore has its legacy shape.
- Prompt initialization: isolated deterministic truncated normal, standard
  deviation `0.02`, without advancing global constructor RNG.
- Prompt output: final-block class prompts pass through one shared LayerNorm
  and one shared `Linear(256, 1)` head, matching Prompt-CAM's class-slot/shared
  scalar formulation.
- Logit fusion: `base_logits + 0.10 * class_prompt_logits`.
- No prompt dropout, class-specific head, prototype distance, routing,
  threshold, bbox bias, extra loss, or new data transform.
- Existing CLS attention remains the pruning signal. Main native attention
  exports must remove prompt rows/columns and preserve their legacy schema;
  class-prompt-to-patch maps are exported separately with original patch-index
  reconstruction.

Expected added trainable parameters for the locked keeper are:

- prompts: `8 * 5 * 256 = 10,240`;
- shared LayerNorm: `512`;
- shared scalar head: `257`;
- total: exactly `11,009`.

## 5. Locked data and provenance

- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Dataset: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`.
- Locked data-spec SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Source-fold authority: retained CIDT train-only rows, with the exact locked
  CIDT summary and prediction hashes already used by XCA/SRM/ViG readiness.
- Stage A may construct only the `train` split. Validation and test loaders,
  prediction files, thresholds, and metrics are forbidden.
- Fold: source-disjoint fold `0`; expected fit/holdout rows `7372/1843` and
  zero normalized source overlap.

## 6. Matched Stage-A adaptation

Both variants use the prompt-enabled architecture and the same initialized
extension tensors, keeper state, batch order, transform, optimizer state,
forward RNG, and holdout order.

- Control trainable tensors: shared prompt LayerNorm and scalar head only.
- Candidate trainable tensors: the same tensors plus all deep prompt
  embeddings.
- Every keeper tensor is frozen and must remain bit-exact.
- Optimizer: SGD, learning rate `0.005`, momentum `0.9`, weight decay `0.001`.
- Batch size: `32`.
- Batches: exactly `60`, hence `1920` deterministic shuffled fit rows.
- Seed: `42`.
- Input: keeper evaluation geometry and normalization, with no stochastic
  augmentation.
- Precision: BF16 autocast for adaptation and holdout inference; FP32 is used
  for equation, attribution, and export checks.

The fixed-prompt control separates useful class-specific prompt learning from
capacity gained by fitting a scalar readout over arbitrary prompt slots.

## 7. Mandatory functional gates

All must pass:

1. Exactly `11,009` added parameters and the exact expected state additions.
2. Constructor RNG equality with the legacy keeper construction.
3. Every keeper tensor bit-exact after loading and after Stage-A adaptation.
4. Prompts inserted and removed at all eight blocks; base token and original
   patch-index contracts remain valid through both pruning layers.
5. Legacy native MHSA maps remain valid for all eight blocks, and eight
   separate class-prompt-to-patch maps are available without fallback.
6. FP32 and BF16 gradients are finite and nonzero for prompt embeddings,
   shared normalization, and shared scalar head.
7. Candidate prompt embeddings move; control prompt embeddings remain exact.
8. Final candidate prompt maps are not collapsed: mean off-diagonal cosine
   below `0.995`, effective rank above `1.5`, and every class logit has
   nonzero sample variance on a balanced real batch.
9. On true class-1 audit rows, candidate class-1 maps improve target-versus-
   confuser spatial separation over control by at least `0.005`, without
   increasing outside-bbox context mass by more than `0.02`.
10. Static-batch-1 three-input ONNX Runtime max logit error is at most `1e-5`
    with identical argmax.
11. Prompt-enabled inference runtime is at most `1.25x` the keeper and peak
    training allocation is at most `3.25 GiB` for the locked audit batch.

## 8. Mandatory train-only decision gates

Candidate minus matched control on all `1843` fold-0 holdout rows must satisfy:

- macro F1 delta at least `0.000`;
- class-1 F1 delta at least `+0.005`;
- class-1 precision delta at least `+0.005`;
- class-1 recall delta at least `-0.005`;
- at least two correct removals of restricted `0/2/4 -> 1` false positives;
- class-1 FN rescues at least class-1 TP breaks;
- corrections strictly greater than harms;
- maximum F1 drop over classes `0/2/3/4` at most `0.005`.

The same candidate/control pair is evaluated on fixed dim, bright, and
low-contrast holdout views. All three class-1 F1 deltas must be at least
`-0.005`, worst class-1 recall delta at least `-0.010`, and at least two of
three class-1 precision deltas must be nonnegative.

Raising precision by suppressing class-1 predictions while breaking more true
positives is an automatic failure regardless of aggregate foreground mass.

## 9. Stage-B permission

Only a complete Stage-A pass authorizes one matched keeper-initialized
`120-batch x 2-epoch`, full-validation, no-test smoke. Stage B must run
independent reload, transition accounting, clean/dim/bright/low-contrast and
occlusion robustness, architecture trace, native attention, class-prompt maps,
Grad-CAM, rollout, perturbation audit, and manual contact-sheet review before
any continuation.

Failure closes this route without prompt-count/layer/init/fusion/LR/optimizer/
seed/fold/budget/loss/augmentation/run-length sweeps. It does not permit a
Prompt-CAM-only classifier, MCTformer dense head, post-hoc prompt router, or
test opening.

The current keeper, scratch complement, precision package, and VS Code
full-train command remain unchanged unless a single checkpoint later passes
the existing promotion gates.
