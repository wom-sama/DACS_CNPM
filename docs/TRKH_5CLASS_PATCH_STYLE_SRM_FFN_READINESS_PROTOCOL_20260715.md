# TRKH 5-Class Patch-Style SRM FFN Readiness Protocol - 2026-07-15

## Status

Precommitted before implementation. This document locks one exact train-only
readiness experiment and, only if every Stage-A gate passes, one
keeper-initialized validation smoke. It does not authorize test access, a full
train, or a parameter sweep.

## Research Question

Can a lightweight style-statistic recalibration inside selected Transformer
feed-forward branches suppress illumination/color styles that create class-1
false positives while preserving true class-1 surface evidence, the current
wide-context CNN path, spatial MHSA, token pruning, and XAI provenance?

The current scratch complement improves class-1 recall but creates 33 extra
validation false positives. Repeated XAI shows that far background is already
weakly causal, while glare, pale-green/yellow maturity, crop-edge effects, and
surface/color overlap remain important. A useful intervention must therefore
learn which hidden-channel styles are relevant instead of applying more class-1
recall pressure or another foreground-localization mechanism.

## Primary-Source Lock

- Paper: Lee, Kim, and Nam, "SRM: A Style-Based Recalibration Module for
  Convolutional Neural Networks," ICCV 2019.
- Proceedings:
  `https://openaccess.thecvf.com/content_ICCV_2019/html/Lee_SRM_A_Style-Based_Recalibration_Module_for_Convolutional_Neural_Networks_ICCV_2019_paper.html`
- Paper PDF:
  `https://openaccess.thecvf.com/content_ICCV_2019/papers/Lee_SRM_A_Style-Based_Recalibration_Module_for_Convolutional_Neural_Networks_ICCV_2019_paper.pdf`
- Author repository:
  `https://github.com/hyunjaelee410/style-based-recalibration-module`
- Locked repository commit:
  `f6221c77e797c4530dddba03616153708d636bd1`
- Reviewed official file: `models/recalibration_modules.py`
- Reviewed file SHA-256:
  `5232416ff7eabecd43a3d0e9e118c64f2dc2e97ded55d5141863ec8fe0e32d13`

For each channel, SRM pools spatial mean and standard deviation, combines the
two values with a channel-independent two-input linear weight, normalizes the
encoded scalar with batch normalization, applies a sigmoid, and multiplies the
feature channel by the resulting sample-specific gate. The paper reports that
SRM is lightweight and outperforms SE/GE under its supervised ImageNet and
CIFAR experiments. Those results motivate this test; they are not a claim that
the TRKH adaptation will converge or improve within 30 epochs.

## Local No-Repeat Boundary

This route is distinct from, and must not silently reopen, the following closed
families:

- MixStyle and input/feature statistic randomization. MixStyle exchanges style
  statistics between samples; SRM learns an end-to-end per-channel gate from
  each sample's own hidden mean/std.
- GrayEdge/CIConv/Lab/color-statistic and input-normalization routes. The new
  module does not alter pixels or append a fixed color descriptor.
- XCA/PWCA/second-order covariance routes. SRM uses independent first/second
  moments per hidden channel; it does not construct a channel covariance map.
- Gabor, wavelet, LBP, high-frequency, OctConv, and other texture experts.
- VCA, relative-position, local-patch-mixer, LeFF, concurrent local-global,
  token-selector, and background-localization routes.
- Residual/deep classifier heads, calibration, verifier, router, ensemble,
  loss-only, sample-weight, or class-1 recall-pressure changes.

ConvNeXt-V2 GRN is not selected. Its primary paper says the supervised-only
gain is relatively small and newly adding GRN only at fine-tuning performs
poorly; its strongest evidence is the FCMAE co-design already outside this
closed no-pretrain route. SRM has direct supervised classification and
multi-style evidence and better matches the current illumination/style error.

## Exact TRKH Adaptation

Method identifier: `patch_style_srm_ffn`.

1. Keep the current `conv_pool` CNN stem, patch embedding, positional/prefix
   tokens, all eight spatial-MHSA blocks, token pruning after layers `2,5`,
   bbox/foreground priors, classifier, losses, and transforms unchanged.
2. Enable patch-style recalibration only in one-based Transformer layers
   `2,5` and only in the standard `FeedForward` path.
3. Preserve the existing FFN order except for one insertion:
   `Linear(256,1024) -> GELU -> SRM -> existing Dropout -> Linear(1024,256)
   -> existing Dropout`.
4. Compute style statistics only from patch hidden activations after GELU.
   For hidden channel `c`, use population statistics over the current patch
   tokens: `mu_c = mean_n(x_nc)` and
   `sigma_c = sqrt(mean_n((x_nc-mu_c)^2) + 1e-5)`. Population variance follows
   the paper's `1/HW` equation rather than modern PyTorch's default sample
   correction.
5. Use one independent two-value CFC weight per hidden channel,
   `W in R^(1024 x 2)`, initialized to zero exactly as the official code.
6. Use `BatchNorm1d(1024, eps=1e-5, momentum=0.1, affine=true,
   track_running_stats=true)` on the `[B,1024]` encoded style values. This is
   mathematically the official `BatchNorm2d` on `[B,1024,1,1]` without dummy
   spatial axes.
7. Use the official sigmoid gate `g=sigmoid(BN(W*[mu,sigma]))` and output
   `x_patch*g`. Zero CFC plus default BN initializes every patch gate to
   exactly `0.5` in evaluation. Do not multiply it by two or add a separate
   gate/residual scale.
8. Do not directly gate CLS, register, or branch hidden activations. They pass
   through the original FFN unchanged. The parent Transformer FFN residual
   remains the only residual connection; subsequent spatial MHSA can aggregate
   recalibrated patch evidence into prefix tokens.
9. Add no dropout, spatial attention, auxiliary head, loss, augmentation,
   threshold, pretrained weight, or external data.
10. Spatial MHSA remains the pruning and image-space XAI source. SRM must expose
    style/gate telemetry separately and must never be visualized as a spatial
    attention map.

Default-off configuration surface:

- `patch_style_recalibration=false`
- `patch_style_recalibration_layers="2,5"`

The exact candidate adds `8,192` trainable parameters: per selected layer,
`2,048` CFC values plus `1,024/1,024` BN affine values. BN running mean,
running variance, and batch counter are buffers, not trainable parameters.

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
- Train-only source-group/fold declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`
- Fold CSV SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`
- Companion summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`

The declaration contains exactly `9215` ordered clean train rows. Fold 0 is a
source-disjoint `1843`-row holdout; the remaining `7372` rows are fit-only.
Validation and test paths are forbidden in Stage A.

## Stage A: Train-Only Readiness

Locked execution:

- device: CUDA, deterministic algorithms, TF32 off;
- seed/fold: `42/0`;
- batch size: `32`;
- deterministic evaluation transform for both fit and holdout; no stochastic
  augmentation;
- train order: one NumPy permutation of fold-not-0 indices with seed `42`;
- budget: first `30` batches, exactly `960` rows;
- optimizer: AdamW, learning rate `3e-4`, weight decay `0.01`;
- hard-label cross entropy only;
- control trainable parameters: `head.weight`, `head.bias`;
- candidate trainable parameters: the same head plus all CFC/BN affine
  parameters in layers `2,5`;
- freeze every other parameter. Keep the parent model and all existing dropout/
  BN modules in evaluation mode; put only candidate SRM BN modules in training
  mode so their running statistics are learned.

This corrects a weakness exposed by the XCA audit: all-parameter short
adaptation collapsed class-1 recall for both variants and obscured the small
candidate signal. The new comparison isolates whether SRM adds useful evidence
beyond the same head adaptation without changing the frozen representation.

### A1. Source, schema, and deterministic construction

All must pass:

1. Verify every locked source SHA and official source commit/SHA.
2. Construct no validation/test dataset or loader.
3. Control strict-loads the keeper. Candidate reports exactly these six new
   state entries per selected layer and no others:
   `cfc`, `bn.weight`, `bn.bias`, `bn.running_mean`, `bn.running_var`, and
   `bn.num_batches_tracked`.
4. Every pre-existing tensor is bit-identical after candidate load.
5. Added trainable parameters equal `8,192`; selected layers are exactly
   `2,5`; all eight spatial-MHSA blocks and pruning positions remain unchanged.
6. Complete CPU/CUDA RNG states after construction and forward match the
   control. The new module must not consume random draws.
7. Config, checkpoint, and V8 PowerShell CLI round-trip the feature while the
   default-off model remains schema/behavior exact.

### A2. Equation, patch-only, gradient, and mechanism checks

All must pass in FP32 and BF16 where applicable:

1. Module output and gate match an explicit implementation of paper equations
   1-9 within numerical tolerance.
2. Evaluation gate is exactly `0.5` at initialization.
3. Direct SRM application changes no prefix hidden value; patch hidden values
   equal input times the reported gate.
4. CFC, BN affine, shared FFN `fc1/fc2`, classifier, and input gradients are
   finite and nonzero under a five-class loss.
5. After adaptation, all CFC and BN affine tensors move, running statistics
   update, and each layer's maximum gate deviation from `0.5` is at least
   `1e-4`.
6. Adapted gates are finite, strictly within `(0,1)`, have channel standard
   deviation at least `1e-4`, and sample standard deviation at least `1e-5`.
7. Bypassing either selected layer changes logits by at least `1e-4`.
8. Gate response changes by at least `1e-4` under at least one locked lighting
   shift relative to clean input. Activity alone is not a promotion signal;
   decision gates below remain mandatory.

### A3. Export and resource gates

All must pass:

1. Export an isolated SRM-FFN equation wrapper and a full metadata-aware
   candidate at static batch `1`, opset 17. ONNX Runtime CPU maximum logit
   error must be `<=1e-5` and argmax must match.
2. Full export must accept `images`, `bbox`, and `image_mask`; it must not claim
   dynamic-batch support.
3. Candidate median BF16 forward/backward runtime at batch 32 is no more than
   `1.15x` control across three matched repeats.
4. Candidate peak allocated VRAM is no more than `3.25 GiB`.

### A4. Train-only decision gates

Evaluate control and candidate on the exact ordered 1843-row holdout before
and after adaptation. For the unadapted candidate require:

- macro-F1 delta `>= -0.010`;
- class-1 recall delta `>= -0.020`;
- net class-1 TP breaks minus FN rescues `<=3`.

For the adapted candidate minus adapted control, all must pass:

- changed decisions `>=2`;
- macro-F1 delta `>=-0.002`;
- class-1 F1 delta `>=+0.002`;
- class-1 precision delta `>=+0.005`;
- class-1 recall delta `>=-0.010`;
- at least two correct `0/2/4 -> 1` false-positive removals;
- class-1 TP breaks no greater than FN rescues;
- corrections strictly greater than harms;
- new `3 -> 2` harms `<=2`;
- maximum nonfocus-class F1 drop `<=0.015`.

### A5. Train-only illumination gates

Evaluate the adapted pair on the same holdout under the existing fixed
conditions:

- dim: brightness `0.70`, contrast `0.90`;
- bright: brightness `1.25`, contrast `1.10`;
- low contrast: brightness `1.00`, contrast `0.65`.

Require candidate-minus-control macro F1 to be nonnegative in at least `2/3`
conditions, class-1 precision delta to be nonnegative in at least `2/3`, worst
class-1 precision delta `>=-0.010`, worst class-1 recall delta `>=-0.030`, and
aggregate restricted focus-FP removals to exceed creations. No condition may
have more than three net class-1 TP breaks.

Any Stage-A failure closes this exact route. Do not change layers, CFC/BN,
gate equation, LR, weight decay, budget, fold, seed, trainable prefixes, or
thresholds after observing the result.

## Stage B: Sole Keeper-Initialized Validation Smoke

Stage B is authorized only if every Stage-A check passes.

- Run exactly one `120 train batches x 2 epochs` smoke on full `yolo_f/val=2606`.
- Resume the locked keeper.
- Train only `head` and the two SRM modules with AdamW `LR=3e-4`, weight decay
  `0.01`; keep the backbone frozen and existing modules in eval mode.
- Use the keeper's existing train transforms/losses except no new auxiliary
  loss or augmentation. Set scheduler horizon to `10`, patience `1`, and
  `SkipFinalTest=true`.
- Require an independently reloaded raw checkpoint. EMA/trainer-only metrics
  cannot promote the candidate.

Minimum validation continuation gates relative to the independent keeper:

- macro F1 `>=0.884925`;
- class-1 F1 `>=0.683261`;
- class-1 precision `>=0.613093`;
- class-1 recall `>=0.768212` and at least `116/151` class-1 TP;
- at most `73` class-1 FP;
- corrections greater than harms and at least four net restricted focus-FP
  removals;
- maximum nonfocus-class F1 drop `<=0.010`;
- no new `3->2` harm expansion above two rows;
- all three illumination gates remain satisfied;
- full architecture trace, generic forensics/confusions, SRM telemetry,
  native-attention/rollout/Grad-CAM XAI, and perturbation audit complete with
  zero unlabeled attribution fallback.

Failure closes the route without a five-epoch continuation, probe, full train,
test access, or nearby sweep.

## Promotion and Command Boundary

Passing Stage A only permits Stage B. Passing the Stage-B continuation gates
only permits a separately precommitted validation confirmation and deployment
equivalence review; it does not automatically authorize test access.

Update `scripts/run_trkh_current_best_full_pipeline.ps1` and
`docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt` only after a candidate
is independently reloaded, materially beats the keeper validation gate, passes
all XAI/resource/export checks, and has a frozen final-test protocol. Otherwise
the current command SHA-256 must remain
`36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`.
