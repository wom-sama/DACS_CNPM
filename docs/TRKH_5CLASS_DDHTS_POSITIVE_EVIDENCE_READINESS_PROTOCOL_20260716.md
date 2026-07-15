# TRKH 5-Class DDHTS Positive-Evidence Readiness Protocol - 2026-07-16

## Decision question

Can the cross-layer and cross-frequency sign histograms from DDHTS-Net provide
independent positive surface evidence for class 1, strong enough to remove
restricted `0/2/4 -> 1` false positives without breaking any raw keeper class-1
true positive?

This is a single train-only information gate. It does not modify raw data, train
the shared TRKH model, read validation/test, or create a deployable checkpoint.

## Primary source and transfer limits

- Source: Li et al., *DDHTS-Net: Dual-Domain Hierarchical Texture Supervision
  Network for Plant Texture Analysis*, Frontiers in Plant Science, 2026,
  `https://doi.org/10.3389/fpls.2026.1872418`.
- Locked local PDF:
  `D:\DataAI\external_sources\papers\ddhts_net_frontiers_2026.pdf`.
- PDF SHA-256:
  `d3fa72ef3345f4ef23403eab904855df15ebe772146b6323fe502f2d9dff9a7d`.
- The paper uses AlexNet/VGG feature extractors, an SVM, random stratified
  `80/20` image splits repeated 50 times, and tunes the retained-channel count
  to `C=250`. It does not report source-group separation, class F1, SVM kernel
  settings, backbone weight provenance, code, or an end-to-end training loss.
  Its reported accuracy is therefore not directly comparable to TRKH.
- This gate transfers only Equations 2-18: isotropic undecimated wavelet
  contrast views, adjacent-layer sign coding, cross-frequency sign coding, and
  histogram concatenation. It uses the already scratch-trained frozen keeper
  instead of importing pretrained AlexNet/VGG weights.

## Immutable data and model contract

- Data spec: `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, which references
  `canbang.yaml` for balance statistics. Use split `train` only, all `9215`
  object rows in canonical order and `8064` source groups.
- Data-spec SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Keeper:
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Reuse the prospectively locked CAGrad source-disjoint split: `7372` fit rows
  and `1843` holdout rows, with zero source overlap.
- Fit-index SHA-256:
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d`.
- Holdout-index SHA-256:
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`.
- Reuse the locked boundary cohorts: all `432` fit class-1 references and the
  exact `186` restricted hard negatives. No row selection or class sampling is
  allowed after descriptor extraction.
- Validation and test paths are forbidden. The output directory must contain
  no binary model, checkpoint, dataset copy, or modified image.

## Immutable DDHTS adaptation

1. Denormalize the keeper input with ImageNet mean/std and clamp RGB to
   `[0,1]`; compute luminance as `0.2989 R + 0.5870 G + 0.1140 B`.
2. Apply three IUWT scales using the paper's separable low-pass kernel
   `[1,2,1] / 4` at dilations `1,2,4`. At each scale compute
   `H_k=L_(k-1)-L_k`, `d_k=H_k/(L_k+1e-6)`, then
   `D_k=sigmoid((d_k-mean(d_k))/(std(d_k)+1e-6))`.
3. Replicate each `D_k` to RGB, normalize it with the keeper input mean/std,
   and run the frozen keeper CNN stem. Use the original view plus the three
   contrast views.
4. Capture the outputs of the keeper's three `ConvStemBlock` layers. This fixes
   `N=3`; the source equations permit a variable number of selected layers.
5. Standardize each map to the deepest `32x32` grid with adaptive average
   pooling. Preserve the common minimum `C=32` channels; for wider layers,
   contiguous channel groups are averaged to 32. `C` is dictated by the
   scratch keeper and is not tuned.
6. IntraHSU: for each of the four views, encode signs of adjacent-layer
   differences into a two-bit pattern and produce one L1-normalized four-bin
   histogram. Dimension: `4 views x 4 bins = 16`.
7. InterGSU: at each of the three layers, encode signs of `D1-I`, `D2-I`, and
   `D3-I` into a three-bit pattern and produce one L1-normalized eight-bin
   histogram. Dimension: `3 layers x 8 bins = 24`.
8. The sole candidate descriptor is `[InterGSU, IntraHSU]`, exactly 40 finite
   values per row. Intra-only and inter-only metrics may be reported as fixed
   mechanism diagnostics, but cannot select the candidate.

## Immutable positive-evidence readout

- Fit one `StandardScaler` and one binary linear SVM with `C=1.0`, balanced
  class weights, and seed `20260716` on the exact `432 + 186` clean fit cohort.
  Positive means true class 1; negative means restricted hard negative.
- The veto threshold is the minimum clean-fit SVM score among raw keeper
  class-1 true positives minus `1e-6`. This construction must preserve every
  clean-fit raw class-1 true positive.
- On holdout, only rows whose raw keeper prediction is class 1 are eligible.
  If their score is below the locked threshold, replace class 1 with the raw
  runner-up class. All other predictions remain bit-exact.
- Fit the SVM and threshold once on clean fit. Apply them unchanged to clean,
  dim `(0.70,0.90)`, bright `(1.25,1.10)`, and low-contrast `(1.00,0.65)`
  holdout views. No condition-specific fitting is permitted.
- Raw keeper and aggregate-margin A-GEM remain comparators. The DDHTS route is
  not allowed to borrow their labels, scores, gradients, or parameters.

## Promotion gates

All structural checks must pass, including exact hashes/order/cohorts, zero
fit/holdout source overlap, descriptor dimension 40, finite and normalized
histograms, complete condition rows, deterministic SVM replay, no validation or
test access, and no binary artifact.

Clean holdout must satisfy every condition below:

- break `0` raw class-1 true positives and preserve raw class-1 recall exactly;
- remove at least `4` restricted class-1 false positives and create `0`;
- improve class-1 precision by at least `0.02`;
- improve class-1 F1 by at least `0.005`;
- do not reduce macro F1;
- corrections must be at least harms;
- class1-TP versus restricted-FP direction AUROC must be at least `0.65`.

Robustness must also preserve every raw class-1 TP in each shifted condition,
must not reduce class-1 F1 in any shifted condition, and must remove at least
one restricted false positive in at least two of the three shifted conditions.

## Stop rule

- If any gate fails, deny shared-trainer integration, validation, test, probe,
  full train, command promotion, and sweeps over wavelet scales, channels,
  layers, histogram variants, SVM kernel/C/weights, thresholds, seeds, or
  condition settings. This closes exact DDHTS-style fixed binary histograms on
  the current keeper, alongside the prior LBP/wavelet/Deep-TEN evidence.
- If every gate passes, authorize only one separately precommitted Stage B that
  makes the descriptor deployable and measures complete XAI, runtime, VRAM,
  ONNX, precision, and TP-retention behavior before any validation access.
