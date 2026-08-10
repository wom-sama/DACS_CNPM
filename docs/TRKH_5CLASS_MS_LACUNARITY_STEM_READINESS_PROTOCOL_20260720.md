# TRKH 5-Class MS-Lacunarity Stem Readiness Protocol - 2026-07-20

## Status

Prospectively locked before implementation. This protocol authorizes one
frozen-keeper, train-only readiness audit. It does not authorize an image-model
epoch, official validation or test access, a hyperparameter sweep, a full
train, or a current-best command update.

## Research Question

Does multiscale lacunarity of the current TRKH CNN stem expose spatial
heterogeneity that distinguishes true class 1 mango surfaces from visually
similar classes while preserving class-1 recall? The candidate must add useful
information beyond first-order pooling on the exact same object and wide-crop
regions. Precision is the primary focus metric.

## Primary-Source Lock

- Accepted workshop paper: Mohan and Peeples, "Lacunarity Pooling Layers for
  Plant Image Classification using Texture Analysis," CVPRW Vision4Ag 2024:
  `https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/html/Mohan_Lacunarity_Pooling_Layers_for_Plant_Image_Classification_using_Texture_Analysis_CVPRW_2024_paper.html`.
- Local paper: `D:/DataAI/external_sources/papers/lacunarity_pooling_cvprw2024.pdf`.
- Paper SHA-256: `a110acaf0c4b43910b0e861f657f96039dc5ed484a422d1f2adb4c7097655ce2`.
- Official MIT repository:
  `https://github.com/Advanced-Vision-and-Learning-Lab/2024_V4A_Lacunarity_Pooling_Layer`.
- Locked commit/tree:
  `6e464b4c326513c206eb9377c7ff9139ddee65f7` /
  `9f1a7c618db236cd8673e2cb797308fb210e39fe`.
- Reviewed source SHAs for base, multiscale, and fusion implementations:
  `1485e75d...f08d`, `b0ae4c0a...222f`, and `0ea99209...107e`.
- MIT license SHA-256: `845f803c30acc03f14c3445c44a5ce51a42b7de5a20ea944d726a41b33c2ca73`.

The paper defines lacunarity as the normalized second moment

`L(X) = E[X^2] / (E[X]^2 + epsilon) - 1`

after `X = (tanh(F) + 1) / 2`, evaluates it on a Gaussian pyramid, learns a
channelwise combination across two scales, and multiplies the result by GAP.
The official experiments freeze ImageNet-pretrained ConvNeXt, ResNet18, or
DenseNet161 backbones. Their absolute accuracies therefore do not transfer to
scratch TRKH. Only the equation and the reported agriculture/texture
motivation are imported.

## Local No-Repeat Boundary

- Fixed RGB LoG surface-blob morphology is closed. This audit operates on the
  keeper's learned 256-channel stem map and measures low-pass heterogeneity,
  not handcrafted peak counts or lesion rules.
- Deep-TEN, bilinear/covariance pooling, wavelet, Gabor, LBP/HOG, spectral
  experts, DDHTS, and color statistics are closed. Base lacunarity alone is a
  coefficient-of-variation statistic and would overlap them. The only tested
  addition is the two-scale change in latent spatial heterogeneity after a
  fixed Gaussian pyrdown.
- FENet/CLASSNet are not imported: FENet's displayed official repository lacks
  a usable license and both routes rely on pretrained backbones and materially
  larger fractal encoders. CAPTN D2P plus degree-2 Chebyshev encoding is
  rejected because channel permutation can be absorbed by an unconstrained
  `1x1` convolution and the polynomial reduces to elementwise linear/quadratic
  terms already covered by failed second-order routes.
- The official DBC implementation is excluded. It uses `ceil`, a global sum
  over the whole batch, and did not consistently beat GAP in the paper.
- No new raw data, label edit, split edit, class weighting, threshold, router,
  ensemble, teacher, distillation, augmentation, optimizer, or model branch is
  allowed in this readiness stage.

## Locked Inputs

- Repository pre-protocol HEAD/upstream:
  `87dabbc594d1e29f6a1e6153bf85938fdbb7e832`.
- Keeper checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Raw `yolo_f/data.yaml` SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Frozen fold declaration SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Generated fold YAML/summary SHAs:
  `4195ba89995d4eee483cae31626717381df910f56657086e7116322ca971388e` /
  `2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763`.
- Current-best command/history SHAs remain
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

Use only `runs/yolof_cidt_fold0_trainonly_20260716/data.yaml`:

- fit: 7,372 object rows, 6,452 source stems, class counts
  `[1561,432,1527,2017,1835]`;
- holdout: 1,843 object rows, 1,612 source stems, class counts
  `[380,109,393,503,458]`;
- source overlap: zero;
- fit/holdout index SHAs:
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d` /
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`.

The generated `test` path mirrors holdout for loader compatibility and must
never be opened.

## Locked Descriptor

Capture the frozen keeper's final `HybridConvStem` activation before patch
projection. Its expected shape at 256-pixel input is `[B,256,32,32]`.

1. Convert each activation to `X=(tanh(F)+1)/2` in FP32.
2. Build exactly two levels: the native map and one fixed separable
   `[1,4,6,4,1]/16` Gaussian blur followed by stride-2 sampling. No learnable
   filter or scale sweep is permitted.
3. At each level compute weighted means and lacunarity independently over:
   the full valid crop and the 12%-eroded object bbox interior. Downsample masks
   by area interpolation, retain fractional occupancy, and clamp support at
   one effective cell.
4. Control descriptor: native-scale means for both regions, 512 dimensions.
5. Candidate-only block: `mean_native * lacunarity_level` for both regions and
   both levels, 1,024 dimensions. Candidate descriptor is the 1,536-dimensional
   concatenation of control and candidate-only blocks.

The candidate contains the official GAP-times-lacunarity interaction while
keeping GAP as a zero-init-compatible residual path. Separate scale features
represent the paper's trainable depthwise scale mixing without introducing an
image-model parameter during readiness.

## Locked Readout And Audit

- Standardize each feature using fit rows only.
- Fit natural-frequency multinomial logistic regression with no class weights,
  intercept enabled, `C=0.3`, `lbfgs`, and at most 2,000 iterations.
- Use five `StratifiedGroupKFold` folds by normalized source stem for fit OOF
  evidence. Control and candidate share exact folds, ordering, solver, and
  stopping settings.
- Refit each role once on all 7,372 fit rows and evaluate once on all 1,843
  holdout rows. No holdout-driven setting change is allowed.
- The frozen keeper predictions are descriptive only because the checkpoint
  has seen the original train rows; they are not a gate baseline and are not
  used as readout inputs.
- Persist protocol, metrics, folds, all OOF/holdout predictions, descriptor
  rank/correlation telemetry, resource telemetry, and an exact artifact
  manifest. Write no model, checkpoint, engine, raw-data payload, or test result.

One prefix run may use at most 512 fit and 256 holdout rows to catch engineering
faults. It is non-decisional and must be removed after the full result is
recorded.

## Engineering Gates

Before the full audit:

1. source, data, fold, keeper, and command hashes match;
2. FP32 output reproduces an independent direct-moment equation within `1e-6`;
3. uniform maps produce zero lacunarity within `1e-6`, clustered maps produce
   larger positive lacunarity, and Gaussian pyrdown is finite/deterministic;
4. control/candidate dimensions are exactly `512/1536`, all values are finite,
   and candidate-only effective rank is at least 8;
5. fit/holdout and every OOF fit/hold source intersection are empty;
6. only `train` and `val` keys of the generated train-only YAML are opened;
7. extraction peak CUDA allocation is at most 7.5 GiB and no batch is dropped;
8. focused tests, full pytest, compile checks, and `git diff --check` pass on a
   committed and pushed implementation.

## Precision-First Promotion Gates

Every gate must pass on both OOF and frozen holdout unless explicitly scoped:

- candidate-minus-control macro F1 is at least `+0.003` on holdout and
  nonnegative OOF;
- class-1 F1 gain is at least `+0.015` on holdout and positive in at least
  three of five OOF folds;
- holdout class-1 precision gain is at least `+0.025`;
- holdout class-1 recall delta is at least `-0.010`;
- restricted `{0,2,4}->1` false positives decrease by at least four;
- total corrections are at least harms, class-1 FN rescues are at least TP
  breaks, and maximum non-focus class F1 loss is at most `0.010`;
- candidate-minus-control class-1 direction AUROC is at least `0.60` on both
  OOF and holdout;
- all solvers converge, descriptors remain finite, and candidate-only
  effective rank remains at least 8.

Passing the clean gate authorizes a separate dim/bright/low-contrast replay and
changed-case XAI review. Only passing those audits may authorize one paired
image-model smoke. It still does not authorize full training, validation/test,
or command promotion.

## Stop Rules

Any material failure closes this exact route. Do not sweep scales, Gaussian
kernel, tanh scaling, region definitions, erosion, feature interactions,
readout `C`, solver, folds, class weights, thresholds, or combine lacunarity
with FENet, CLASSNet, CAPTN, Deep-TEN, wavelets, morphology, or other closed
texture branches. Keep the current-best command unchanged unless a later
prospective image-model candidate beats all validation, robustness, XAI,
deployment, and final-test gates.
