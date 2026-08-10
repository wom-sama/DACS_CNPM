# TRKH 5-Class AugSelf Color-Adapter Readiness Protocol

Date: 2026-07-16
Status: immutable before implementation or metric access

## Decision question

Does augmentation-aware color-difference prediction give the scratch TRKH
keeper a precision-safe class-1 representation update, or does it merely make
the already color-sensitive decision boundary broader?

This is a train-only A0 information gate. It cannot authorize validation, test,
shared-trainer integration, a probe/full train, or a current-best command
revision unless every gate below passes.

## Primary-source basis and limits

- Lee et al., *Improving Transferability of Representations via
  Augmentation-Aware Self-Supervision*, NeurIPS 2021:
  `https://proceedings.neurips.cc/paper/2021/hash/94130ea17023c4837f0dcdda95034b65-Abstract.html`.
- Paper SHA-256:
  `1289ef0c57bd9fb1bf29db51540a944ba1435f080483de5b59263cc1f44eb3a1`.
- Supplemental SHA-256:
  `c0355e117d20975e5da896acf601c0162796f27b932c7d3210447f4f09e58c03`.
- Official repository:
  `https://github.com/hankook/AugSelf`, commit
  `c131db66b5ade96af86774bc43a2cb797390bba5`.
- Official `README.md`, `models.py`, `trainers.py`, and `transforms.py` SHAs are
  respectively
  `0d86607921f9d506c316dfb48a993a65147dd811b99436ecca97a1aa0f5d7088`,
  `860b86ea0a8c03f0b6d3c2d4b53e149408f22f40c981f4dc75f2b59e7e3ee014`,
  `bed42e208164afdef32a0cb4f063ee2e1867b9bd6eade70107e6e7e3e65b9e96`,
  and
  `eb394ae72fa0276f854c0f51127b3a61da812144545f9023bc4c3a4f02dbb62d`.
- The official repository has no license file. Local code must therefore be a
  clean-room implementation of the equations and architecture stated in the
  paper; official source is used only to audit semantics and provenance, not
  copied.
- The paper predicts differences between augmentation parameters with a
  three-layer MLP. Color uses a four-dimensional brightness/contrast/
  saturation/hue target, `tanh` output, and squared error. It reports strong
  color-sensitive transfer gains on Flowers and few-shot Plant Disease.
- The original evidence is not directly comparable to TRKH: ResNet-18/50,
  `100-500` pretraining epochs, batch `128/256`, transfer/few-shot accuracy,
  and no class-wise precision/F1 or source-group mango split. The A0 below is
  intentionally much narrower and cannot be reported as a reproduction.

## Why this is not a repeated route

- Internal VICReg, DINO, paired-token consistency, illumination consistency,
  and CEConv attempted to make representations invariant/equivariant across
  views. They did not create conservative class-1 false-positive control.
- Light RandAugment and class-dependent augmentation changed the images but
  provided no target that preserved the discarded color information.
- AugSelf has the opposite representation pressure: it predicts the difference
  between view parameters, so the shared embedding must retain photometric
  information while classification still enforces label stability.
- This A0 uses only the paper's strongest relevant color task. Crop prediction
  is excluded because changing crop geometry would alter the keeper's bbox/
  context semantics; flip and blur are excluded because the paper's standalone
  ablation found them weak and local evidence already rejects nearby routes.

## Immutable data contract

- Dataset: `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, train paths only.
- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Reuse the exact source-disjoint fold-0 split: `7372` fit rows and `1843`
  holdout rows, source overlap zero. Fit/holdout index SHAs are
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d`
  and
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`.
- Fit cohort: all `432` class-1 rows plus the exact `186` restricted
  `0/2/4 -> 1` raw-keeper hard negatives. Their index SHAs are
  `1474f2d8e8fe83767e3c250fd89e827f1544fb058c762bc9c114b150a18d2574`
  and
  `56dfdbc6180d99b80605799bcd8e2ef412f0abd5c0191054b4fe7d6554d7f020`.
- Training order is exactly 60 batches of 32, each containing 16 class-1 rows
  and 16 hard negatives. The 1920-row order SHA is
  `8dd3b2b74b5975b2ce430f85b833cd75e98e3f96ae518850737aef80df31fe5a`.
- AugSelf representation evaluation uses only the holdout's 109 class-1 rows
  and 36 restricted raw false positives. Clean/lighting behavior uses all
  1843 holdout rows.
- No validation/test path or prediction may be constructed or read. No raw
  image, label, YAML, or split may be modified.

## Deterministic color views

- Begin from the keeper's clean evaluation tensor and checkpoint input
  normalization. Denormalize to RGB `[0,1]`, apply color operations, clamp,
  then apply the same normalization again. Bbox and image-mask metadata remain
  unchanged.
- For every occurrence and each of two views, derive a private CPU RNG seed
  from SHA-256 of
  `augself-color|phase|seed|step|sample_index|view`. The global seed is `42`.
- Apply color jitter with probability `0.8`. When active, sample brightness,
  contrast, and saturation factors uniformly from `[0.6,1.4]` and hue from
  `[-0.1,0.1]`; apply the four operations in a uniformly sampled order.
- Record official centered parameters
  `w=(brightness-1)/0.8, (contrast-1)/0.8,
  (saturation-1)/0.8, hue/0.2`. Inactive jitter has `w=0`. Every component is
  in `[-0.5,0.5]`.
- The target for ordered views `(1,2)` is `d12=w1-w2`; the reverse target is
  `d21=-d12`. Both lie in `[-1,1]`. Independent replay must reproduce every
  parameter, operation order, view tensor hash, and reversal identity exactly.
- No crop, affine, flip, grayscale, blur, solarization, obstacle, MixUp,
  CutMix, mosaic, copy-paste, or other augmentation is allowed in this A0.

## Candidate and matched control

- Freeze the complete 7,245,590-parameter keeper in evaluation mode. Its final
  classification head input is 256-dimensional.
- Add a deployment adapter
  `h' = h + W_up GELU(W_down LayerNorm(h))`, with parameter-free LayerNorm,
  `W_down: 256 -> 64`, `W_up: 64 -> 256`, no biases, and exactly zero
  `W_up` initialization. The adapter has exactly 32,768 parameters and initial
  deployed logits must be bit-exact to the raw keeper.
- Add the paper's non-deployed color predictor: concatenate ordered pairs of
  256D adapted embeddings, then use
  `Linear(512,512,bias=False)-BN-ReLU-
  Linear(512,512,bias=False)-BN-ReLU-Linear(512,4,bias=True)`.
  It has exactly 528,388 parameters. Prediction is `tanh(output)`.
- Candidate and control have bit-exact adapter/predictor initialization,
  optimizer state, batches, views, targets, and random states.
- Candidate color loss receives gradients through adapted embeddings. In the
  control, only the predictor sees color loss: both adapted embeddings are
  detached before concatenation. Classification CE still updates both
  adapters. An isolated initial gradient audit must prove candidate color loss
  gives a finite nonzero adapter gradient, control gives exactly zero adapter
  gradient, and predictor gradients match.

## Locked optimization

- Seed `42`, deterministic algorithms, batch `32`, 60 steps, no epoch/schedule
  interpretation, and no retry with changed settings.
- For each batch, forward both color views. Classification loss is the equal
  mean of five-class CE on view 1 and view 2.
- Symmetric color loss is the equal mean of MSE for
  `(concat(h1,h2), d12)` and `(concat(h2,h1), -d12)`, after `tanh`.
- Total loss is `classification_loss + 1.0 * color_loss`. The paper's small-
  dataset value `lambda=1.0` is transferred once; no lambda sweep is allowed.
- Train only the 32,768 adapter parameters and 528,388 predictor parameters
  with AdamW, learning rate `5e-4`, weight decay `0.01`. The keeper and original
  classification/fusion heads remain bit-exact frozen.

## Required audits

1. Provenance, source hashes, clean official worktree, absent official license,
   exact cohort/order, train-only paths, RNG isolation, equation, parameter
   count, zero-init, initial logits, gradient isolation, and frozen-state audit.
2. Source-disjoint focused-holdout color prediction: candidate/control MSE,
   per-channel MSE, zero-predictor MSE, explained variance, symmetry, and two-
   view classification metrics.
3. Clean and deterministic dim/bright/low-contrast predictions for raw,
   control, and candidate on the exact 1843-row holdout. Independently replay
   every CSV-derived metric, transition, and confusion matrix.
4. Deployment-only adapter runtime/VRAM benchmark against raw. The auxiliary
   predictor is forbidden at inference. Export isolated adapter and full
   image+bbox+mask candidate to static-batch-one ONNX opset 17 and compare with
   ONNX Runtime.
5. XAI selection must include every clean class-1 TP break/rescue and every
   restricted FP creation/removal, then deterministic corrections/harms until
   at least 12 rows when available. Render original image, raw/control/
   candidate class-1 stem Grad-CAM, the two locked color views, view absolute
   difference, and candidate color-loss input saliency. All maps and selection
   categories must be independently manifest-replayable.

## All-or-nothing Stage-A gates

### Structural and representation gates

- Every provenance/data/equation/RNG/gradient/frozen-state/replay check passes.
- Candidate and control each execute exactly 60 batches and 1920 ordered rows;
  every trainable parameter sees a finite nonzero gradient and changes.
- Focused-holdout candidate color MSE is at most `0.95x` zero-predictor MSE and
  at most `0.995x` control MSE. Candidate explained variance is positive.
- Candidate two-view classification accuracy and class-1 F1 are each no lower
  than control. These checks prevent a lower auxiliary loss from hiding worse
  augmented classification.

### Clean decision gates

Versus raw keeper, all must pass:

- macro F1 gain at least `+0.001`;
- class-1 F1 gain at least `+0.003`;
- class-1 precision gain at least `+0.005`;
- class-1 recall delta at least `-0.005` and zero net class-1 TP break;
- at least two fewer restricted false positives;
- corrections strictly exceed harms;
- maximum non-class-1 per-class F1 drop at most `0.010`;
- no more than two new `3 -> 2` harms.

Versus matched detached-gradient control, all must pass:

- macro F1 delta nonnegative;
- class-1 F1 gain at least `+0.002`;
- class-1 precision gain at least `+0.003`;
- class-1 recall delta at least `-0.005`;
- at least one fewer restricted false positive;
- corrections strictly exceed harms.

### Illumination gates

- Candidate macro F1 and class-1 precision are nonnegative versus raw in at
  least two of dim/bright/low contrast, and separately nonnegative versus
  control in at least two of three.
- Worst class-1 recall delta is at least `-0.020` versus both raw and control.
- Net class-1 TP loss is at most two in every condition versus either
  comparator.
- Aggregate restricted-FP removals strictly exceed creations versus raw and
  separately versus control across all three conditions.

### Deployment and XAI gates

- Deployment adapter runtime ratio is at most `1.10x` raw and candidate peak
  VRAM is at most `0.75 GiB` under the matched benchmark.
- Both ONNX exports are finite, argmax-exact, and have maximum absolute error
  at most `1e-5`.
- Required XAI coverage is exact; all maps are finite; candidate color-loss
  saliency is not identically zero.

## Decision rule and no-repeat boundary

- If any gate fails: reject this exact final-head AugSelf color-adapter route.
  Do not sweep lambda, jitter strength/probability/order, adapter width/depth,
  predictor width/depth, detach location, optimizer, LR, weight decay, steps,
  batch balance, fold, seed, or add crop/flip/blur/solarization afterward.
- A failure does not reject the AugSelf paper universally. Reopening requires a
  genuinely different prospectively sourced insertion point or supervision
  target, not a neighboring hyperparameter change.
- Only if every gate passes may Stage B implement the same default-off adapter
  in shared code and run one short source-disjoint validation probe. A later
  current-best revision still requires winning the existing locked validation
  promotion gate; test remains final-only.

## Artifact contract

- Write summary JSON, all-condition prediction CSV, training curve, focused
  color-pair replay, XAI manifest/contact sheets, ONNX exports, report, and a
  SHA-256 artifact manifest.
- Do not write a `.pt`, `.pth`, `.ckpt`, TensorRT engine, trainable manifest,
  validation/test prediction, or current-best command revision.
- On closure, rerun compile/focused/full tests, PowerShell parse/preflight,
  protected hashes, artifact replay, retention, journal/TODO, and skill. Remove
  only superseded failed technical roots after exact inspection; never delete
  the formal evidence or user-owned untracked paths.
