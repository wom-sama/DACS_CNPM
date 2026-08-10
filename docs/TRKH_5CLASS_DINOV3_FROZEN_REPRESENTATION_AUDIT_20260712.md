# TRKH 5-Class DINOv3 Frozen Representation Audit 2026-07-12

## Decision

Reject frozen DINOv3-S representation transfer before an image-model smoke.
The fixed candidate loses to a freshly re-extracted FP32 DINOv2-S control on
train OOF, improves validation class-1 F1 by only `0.002484`, wins class 1 in
only `2/5` source folds, and remains below the direct TRKH keeper by `0.040459`
class-1 F1. Ten predeclared gates fail. Test remains closed and current-best
commands remain unchanged.

## Research Basis

- The [DINOv3 paper](https://arxiv.org/abs/2508.10104) introduces Gram
  anchoring to preserve dense feature quality during long self-supervised
  training. The [official Meta repository](https://github.com/facebookresearch/dinov3)
  reports strong frozen dense features, making a dense surface descriptor a
  materially new representation check after DINOv2.
- The released candidate is
  [`vit_small_patch16_dinov3.lvd1689m`](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m):
  21,586,944 parameters, 384 dimensions, one CLS plus four register tokens,
  and a native 256-pixel recipe. The exact safetensors SHA-256 is
  `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`.
- The control is DINOv2-S/14 from the
  [DINOv2 paper](https://arxiv.org/abs/2304.07193): 21,628,800 parameters,
  384 dimensions, one CLS token, and the already audited 224-pixel recipe. Its
  exact safetensors SHA-256 is
  `04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081`.
- This is a released model-plus-preprocessing comparison, not a pure
  architectural ablation. Both recipes produce a 16x16 patch grid and use the
  same fixed descriptor and readout dimensions.

## Locked Protocol

- Input views: immutable `class_f` object crops, strictly remapped to
  `yolo_f` object-level `sample_index` order.
- Coverage: train/validation `9215/2606`; source groups `8064/2577`; no
  missing keys, duplicate keys, label mismatch, or train/validation source
  overlap.
- Test, raw-data writes, trainable manifests, checkpoints, and descriptor
  caches: disabled.
- Features per model:
  - normalized CLS token, 384D;
  - fixed `CLS + patch mean + patch std + 2x2 spatial means`, 2688D.
- Readouts: StandardScaler, fixed PCA-128, fixed linear logistic and RBF-SVC;
  RBF-SVC is primary.
- Split: five deterministic `StratifiedGroupKFold` folds using source stems and
  seed `20260712`.
- Feature selection: maximize train-OOF `macro F1 + class-1 F1` over the two
  fixed features. Validation is not used for feature selection.
- No model/readout/descriptor/size/fold/threshold sweep was allowed.

## Numerical Precision Gate

The retained DINOv2 audit used AMP. A 16-image full-batch-versus-8+8 probe
showed that AMP descriptors depend materially on batch geometry:

| Model | Feature | AMP geometry mean abs | AMP geometry max abs | FP32 geometry max abs |
| --- | --- | ---: | ---: | ---: |
| DINOv2-S | CLS | `0.022799` | `0.129207` | `0.0` |
| DINOv2-S | CLS+dense | `0.011536` | `0.129207` | `0.0` |
| DINOv3-S | CLS | `0.003479` | `0.040240` | `0.0` |
| DINOv3-S | CLS+dense | `0.001504` | `0.040240` | `0.0` |

Both models were therefore re-extracted in FP32 under the same runner. FP32
changed only `10` OOF and `8` validation DINOv2 decisions versus the retained
AMP audit, but improved DINOv2 validation dense macro/class-1 F1 from
`0.882850/0.636364` to `0.884337/0.641115`. The matched result below supersedes
AMP for this comparison without rewriting the historical audit.

## Results

OOF selected `CLS+dense` for both models.

| Model | Train OOF macro | Train OOF class 1 | Validation macro | Validation class 1 | Val class-1 P/R |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINOv2-S FP32 | `0.880034` | `0.602432` | `0.884337` | `0.641115` | `0.676471/0.609272` |
| DINOv3-S FP32 | `0.878180` | `0.598714` | `0.883516` | `0.643599` | `0.673913/0.615894` |
| TRKH keeper | n/a | n/a | `0.884073` | `0.684058` | `0.608247/0.781457` |

DINOv3 minus DINOv2 is `-0.001854/-0.003718` OOF and
`-0.000821/+0.002484` validation macro/class-1 F1. DINOv3 minus keeper is
`-0.000557/-0.040459`; class-1 recall drops by `0.165563`.

### Source-Fold Stability

| Fold | DINOv2 macro/class 1 | DINOv3 macro/class 1 | Gain macro/class 1 |
| --- | ---: | ---: | ---: |
| 1 | `0.891541/0.660633` | `0.875471/0.602740` | `-0.016070/-0.057894` |
| 2 | `0.882705/0.622010` | `0.880864/0.607843` | `-0.001842/-0.014166` |
| 3 | `0.870451/0.552764` | `0.869548/0.546341` | `-0.000903/-0.006422` |
| 4 | `0.874706/0.571429` | `0.885089/0.626728` | `+0.010383/+0.055300` |
| 5 | `0.879686/0.599156` | `0.879272/0.606557` | `-0.000414/+0.007401` |

The gain is concentrated in folds 4-5 and reverses strongly in fold 1. This is
not a stable representation improvement.

## Transition and Direction Audit

- DINOv2 to DINOv3 OOF: `242` changes, `103` corrections, `120` harms;
  class-1 FN rescue/TP break `35/31`, FP remove/create `33/49`.
- DINOv2 to DINOv3 validation: `88` changes, `39` corrections, `43` harms;
  FN rescue/TP break `13/12`, FP remove/create `14/15`.
- Keeper to DINOv3 validation: `185` changes, `99` corrections, `74` harms,
  `12` neutral; FN rescue/TP break `9/34`, FP remove/create `48/17`.
- DINOv3 is complementary but acts mainly as a class-1 suppressor. Its
  keeper+DINOv3 binary class-1 oracle is `0.830065`, but direct recall safety
  fails badly.
- DINOv2-to-DINOv3 FN-vs-FP direction AUROC is stable at
  `0.633019/0.699923` OOF/validation. That establishes complementarity; it does
  not supply fold-safe keeper OOF targets or authorize a validation-trained
  router.

## XAI and Visual Review

- The full preview and a balanced 32-case review use patch-token deviation
  energy, not causal attention. Review selection is fixed to the eight largest
  absolute class-1 probability changes in each FN-rescue, TP-break, FP-remove,
  and FP-create group.
- DINOv3 energy maps are smoother and less interior-detailed than DINOv2.
  High energy often follows crop/object boundaries or nearby background while
  broad fruit interiors remain low-energy.
- Rescued false negatives and broken true positives share pale-green,
  yellow-green, mottled, spotted, and shadowed fruit. FP-created pale-green
  class-0 crops also overlap the rescued class-1 group.
- FP removals include visibly yellow, damaged, class-3/class-4 fruit and some
  class-0 fruit, consistent with conservative class-1 suppression. The same
  suppression breaks 34 keeper true positives while rescuing only 9 false
  negatives.
- No visual action rule separates corrections from harms well enough for a
  router, KD target, sample weight, or training smoke.

## Gate and No-Repeat Rule

Ten checks fail: OOF macro/class-1 gain, validation macro/class-1 gain,
class-1 fold consistency, validation class-1 `0.70`, keeper macro/class-1 gain,
keeper recall preservation, and keeper FN-rescue/TP-break safety.

Do not sweep DINOv3 input size, descriptor layout, PCA, readout, threshold,
folds, model variants, or ensemble/router weights. Do not use these outputs for
KD, sample weighting, or command promotion. Reopen DINOv3 only if a new
train-OOF keeper signal makes the `9/34` rescue/break direction learnable
without validation fitting.

## Evidence and Verification

- Full audit:
  `runs/diagnostic_dinov3_dense_patch_matched_readiness_20260712`, 11 payloads,
  `8,168,884` bytes, manifest SHA-256
  `2ef946a91889e6a3e129245720e1ba0d939f92c9b01fd91b9bc1c4ee4fa7fe2b`.
- Changed-case review:
  `runs/diagnostic_dinov3_changed_case_review_20260712`, 5 payloads,
  `471,146` bytes, manifest SHA-256
  `a3630a1e9790bd2a6972d5c15a41c5e9ebab6b7e3457c01279b72e5b79536a57`.
- Both manifests report no checkpoint, model binary, or test payload.
- Independent CSV reconstruction verified sample order, finite normalized
  probabilities, prediction columns, per-class confusion matrices, macro F1,
  and class-1 precision/recall/F1 exactly for all eight variants on
  `9215/2606` rows.
- Focused compile/tests passed before and after the full run. Full-suite and
  retention closure are recorded in the research journal after completion.
