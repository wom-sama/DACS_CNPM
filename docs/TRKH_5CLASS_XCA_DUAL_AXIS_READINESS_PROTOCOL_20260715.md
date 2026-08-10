# TRKH 5-Class Shared-Projection XCA Dual-Axis Readiness Protocol - 2026-07-15

## Status

Precommitted before implementation. This document locks one exact train-only
readiness experiment and, only if it passes, one deterministic keeper-initialized
validation smoke. It does not authorize test access, a full train, or a parameter
sweep.

## Research Question

Can a limited channel-axis attention residual improve class-1 surface/color
decision precision while preserving the current TRKH spatial attention,
wide-context CNN stem, token pruning, class-1 true positives, and XAI path?

The hypothesis is deliberately narrower than replacing TRKH with XCiT. Existing
audits show that the model already localizes fruit and rejects far background,
but its foreground surface representation does not reliably separate class 1
from classes 0, 2, and 4. The new mechanism must therefore change channel
interactions without discarding learned spatial interactions.

## Primary-Source Lock

- Paper: El-Nouby et al., "XCiT: Cross-Covariance Image Transformers",
  NeurIPS 2021.
- Proceedings:
  `https://papers.neurips.cc/paper/2021/hash/a655fbe4b8d7439994aa37ddad80de56-Abstract.html`
- Paper PDF:
  `https://papers.neurips.cc/paper_files/paper/2021/file/a655fbe4b8d7439994aa37ddad80de56-Paper.pdf`
- Official repository: `https://github.com/facebookresearch/xcit`
- Locked official commit:
  `82f5291f412604970c39a912586e008ec009cdca`
- Reviewed official file: `xcit.py`
- Reviewed file SHA-256:
  `3e2d4be847dc8c88d98fe0f965bca3119aeeb588975eff0bc62fdfb20e8b3e9e`

The equation and official code both transpose Q/K so each head forms a
channel-by-channel covariance attention matrix. Q and K are L2-normalized over
the token axis, a learnable per-head temperature scales the covariance, Softmax
is applied over the last channel axis, and the matrix mixes V. The paper
describes this as a data-dependent dynamic `1x1` convolution. Heads make the
covariance block diagonal and easier to optimize.

## Local No-Repeat Boundary

The following nearby routes are already closed and must not be silently
reintroduced:

- VCA/full-grid attention replacement;
- gated relative position attention;
- local patch mixer, LeFF, concurrent local-global coupling, and late class
  attention;
- PWCA token interaction, MogaNet, StarNet, InceptionNeXt, OctConv, Gabor,
  high-frequency experts, and fixed second-order texture statistics;
- output calibration, verifier/router, loss-only, and class-1 recall-pressure
  variants.

ODConv is not selected because its four kernel-attention dimensions overlap the
failed multiplicative/dynamic-convolution stem family and its official recipes
use long training schedules. Stock XCiT is not selected because replacing
spatial MHSA would remove the current pruning and XAI evidence path.

## Exact TRKH Adaptation

The method identifier is `shared_projection_xca_dual_axis`.

1. Keep the current `conv_pool` CNN stem, patch projection, branch/register
   tokens, all eight standard spatial-MHSA blocks, MLPs, class head, pruning at
   layers `2,5`, foreground/bbox priors, losses, and augmentations.
2. Enable XCA only in one-based Transformer layers `2,5`.
3. Apply XCA only to patch tokens. CLS, register, and branch-prefix tokens get
   no direct XCA residual. Subsequent spatial MHSA still aggregates the updated
   patch representation into prefix tokens.
4. In each selected block, run spatial MHSA first. Then normalize current patch
   tokens with a separate `LayerNorm(eps=1e-6)`, run XCA, add the fixed XCA
   residual, and continue to the existing MLP.
5. Reuse that block's existing learned spatial-MHSA `qkv` and `proj` linear
   layers. Do not register a second random projection. The only new learnable
   values are the deterministic LayerNorm scale/bias and one temperature per
   head initialized to `1.0`.
6. Use the official per-head dimensions: TRKH `dim=256`, `heads=8`, and
   `head_dim=32`. Each XCA map must be `[B,8,32,32]` regardless of patch count.
7. Use a fixed residual scale `0.10`. It is an explicit TRKH adaptation, chosen
   from the existing bounded residual convention. It is not a claim from the
   XCiT paper and is not learnable or sweepable.
8. XCA attention/projection dropout is exactly zero. This is a legal official
   setting and prevents the candidate-only branch from consuming random draws;
   existing TRKH dropout remains unchanged.
9. Do not add XCiT Local Patch Interaction, class-attention blocks, Fourier
   positional encoding, a new classifier, a gate, or a post-hoc threshold.
10. Spatial MHSA remains the attention returned for pruning and image-space XAI.
    XCA must expose separate channel-attention temperature, normalized entropy,
    diagonal mass, and residual-norm telemetry; a channel covariance matrix must
    never be presented as a spatial heatmap.

Default-off configuration surface:

- `cross_covariance_attention=false`
- `cross_covariance_attention_layers="2,5"`
- `cross_covariance_attention_residual_scale=0.10`

## Locked Inputs

- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Keeper launcher-args SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`
- Data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`
- Data YAML SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`
- Reusable train-only source-group/fold declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`
- Fold CSV SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`
- Companion summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`

The train declaration contains exactly `9215` clean rows and source-disjoint
fold counts `1843/1830/1828/1851/1863`. Stage A uses fold 0 as holdout. This is
a differential readiness audit over a keeper representation that historically
saw train; it is not true OOF backbone evidence and cannot support a paper
metric or model promotion.

## Stage A - Train-Only Readiness

### Locked execution

- device `cuda`, seed `42`;
- deterministic algorithms on, cuDNN benchmark off, TF32 off;
- source-group fold `0` holdout, exactly `1843` ordered rows;
- fit budget: exactly `20` batches x `32` rows from folds `1-4`, selected by one
  deterministic shuffled order;
- clean keeper evaluation transform only; no stochastic image augmentation;
- two separately constructed variants: unchanged control and XCA candidate;
- both load the exact keeper model state, reset optimizer state, and use the
  same input order and dropout RNG sequence;
- optimize all trainable model parameters with AdamW, LR `8e-5`, weight decay
  `0.05`, no scheduler, no gradient accumulation, BF16 AMP when available;
- evaluate both unadapted and adapted variants on the same ordered fold-0
  holdout;
- no validation or test loader may be constructed.

### Structural and numerical gates

All must pass:

1. official commit/file SHA and all locked input SHA values match;
2. `9215` train rows, exact class/fold counts, and zero fit/holdout source
   overlap;
3. default-off control remains checkpoint strict-load compatible and bit exact;
4. candidate missing checkpoint keys are exactly the two selected XCA
   LayerNorm/temperature groups; every existing model tensor is bit identical;
5. control/candidate constructor and post-forward CPU/CUDA RNG states match;
6. only layers `2,5` contain XCA, standard MHSA remains in all eight blocks,
   and pruning still occurs after layers `2,5`;
7. XCA map shape is `[B,8,32,32]`, rows sum to one, Q/K token-axis norms are
   one within `1e-5`, and all values are finite in FP32 and BF16;
8. prefix tokens receive an exactly zero direct XCA delta;
9. every XCA LayerNorm/temperature and shared qkv/proj family receives finite,
   nonzero gradients; candidate input gradients are finite;
10. disabling either XCA layer changes logits by at least `1e-6`, and the
    combined residual-to-patch norm ratio is in `[0.005,0.20]`;
11. normalized channel-attention entropy is in `[0.20,0.9995]` and diagonal
    mass is below `0.95`; uniform/identity collapse is forbidden;
12. candidate ONNX maximum absolute logit error is `<=1e-5`;
13. matched batch-32 candidate/control median runtime ratio is `<=1.30`;
14. candidate peak VRAM is `<=3.25 GiB` in the readiness benchmark;
15. train order, batch count, optimizer settings, forward RNG checkpoints, and
    ordered holdout rows match exactly between control and candidate.

### Hard-decision readiness gates

All adapted candidate-versus-adapted-control gates must pass on the `1843`
train-only holdout:

- at least two changed decisions;
- macro F1 delta `>=-0.002`;
- class-1 F1 delta `>=+0.002`;
- class-1 precision delta `>=+0.005`;
- class-1 recall delta `>=-0.010`;
- restricted focus FP (`0/2/4->1`) reduction `>=2`;
- class-1 TP breaks `<=` class-1 FN rescues;
- corrections `>` harms;
- new `3->2` harms `<=2`;
- maximum non-focus per-class F1 drop `<=0.015`.

The unadapted active candidate must also avoid catastrophic intervention:
macro F1 delta `>=-0.010`, class-1 recall delta `>=-0.020`, and no more than
three class-1 TP breaks above rescues.

Passing Stage A authorizes one Stage B smoke only. Failing any material gate
closes the exact XCA route without validation, test, layer, residual-scale,
head, dropout, optimizer, LR, seed, or budget sweeps.

## Stage B - Sole Deterministic Validation Smoke

Only after Stage A passes:

- control and candidate both resume the locked keeper;
- reset epoch, optimizer, scheduler, and scaler;
- `seed=42`, deterministic execution, batch `32`, grad accumulation `2`;
- exactly `120` train batches x `2` epochs;
- LR `8e-5`, minimum LR `1e-6`, warmup `1`, scheduler horizon `15`, weight
  decay `0.05`, patience `3`;
- full `yolo_f/val=2606`, `max_val_batches=0`;
- `skip_final_test=true` and no test loader;
- candidate-only change is the three locked XCA configuration fields.

Continuation requires every condition below:

- candidate absolute macro F1 `>=0.885925`;
- candidate absolute class-1 F1 `>=0.683261`;
- candidate absolute class-1 precision `>=0.613093`;
- candidate absolute class-1 recall `>=0.764834` and class-1 TP `>=117`;
- candidate-minus-control macro F1 `>=+0.002`;
- candidate-minus-control class-1 F1 `>=+0.004`;
- candidate-minus-control class-1 precision `>=+0.008`;
- candidate-minus-control class-1 recall `>=-0.006`;
- restricted focus-FP reduction `>=4`;
- TP breaks `<=` FN rescues, corrections `>` harms, and new `3->2` harms
  `<=3`;
- maximum non-focus per-class F1 drop `<=0.015`;
- runtime ratio `<=1.30` and all provenance/resource checks pass.

Regardless of clean metrics, complete full confusion/forensics, source-group
review, five-condition robustness, five-class architecture trace, XCA channel
telemetry, and exact changed-case paired native-attention/rollout/Grad-CAM plus
background/object perturbation audit. Require at least `3/5` robustness macro
wins, zero attribution fallback, and no hidden replacement of spatial attention
with channel covariance.

## Promotion And Stop Rules

- A Stage-B pass authorizes one short probe, not a full train or test.
- Full train remains gated by an independently reloaded single checkpoint and
  the existing class-1 milestone policy.
- Test remains final-only and cannot select XCA settings.
- The current-best VS Code command file changes only after an independent
  single-checkpoint validation win. Smoke-only or train-holdout evidence cannot
  change it.
- A failed Stage A or Stage B closes this exact route. Do not sweep layers,
  scale, heads, temperature initialization, projection sharing, LPI, dropout,
  optimizer, LR, seed, loss, augmentation, or run length.
- Raw datasets are read-only throughout.
