# TRKH 5-Class Class-1-Protected RSC Readiness Protocol

Date locked: 2026-07-15  
Status: locked before implementation or adaptation  
Scope: `yolo_f/train` only; validation and test are forbidden in Stage A

## 1. Question

Can a class-1-protected adaptation of Representation Self-Challenging (RSC)
force the existing CNN-ViT to learn alternate foreground maturity cues for
non-class-1 samples, reduce `0/2/4 -> 1` false positives, and preserve every
available class-1 cue?

This is a training intervention. It adds no inference parameter, head, router,
or post-hoc decision rule.

## 2. Primary-source lock

- Paper: Huang, Wang, Xing, and Huang, *Self-Challenging Improves
  Cross-Domain Generalization*, ECCV 2020.
- Paper URL:
  `https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470120.pdf`
- Paper SHA-256:
  `fa76aec1fe8aaf430548d938f008adb3936d7ace53061e63412ceac1da0ea48e`
- Official source: `https://github.com/DeLightCMU/RSC`
- Official commit:
  `bf6d280c5d74910f009ea8963c59167252659666`
- Domain-generalization `models/resnet.py` SHA-256:
  `c2837b190a61be481a7ff7aae31e9459647ec22edc94db88f9d2a0004abf9bf8`
- ImageNet `resnet.py` SHA-256:
  `bc898ca8abe7edb3319f8697fabb6dcc552f1c2f6a1badaee006062db5875bb7`
- License: BSD-2-Clause.

The source computes a ground-truth-logit gradient, masks the most predictive
representation elements, ranks batch rows by the true-class confidence drop,
and updates the model from the challenged representation. The paper reports
that the best drop ratio is data-specific. Its main domain-generalization code
uses ImageNet-pretrained ResNets and environment-sensitive settings. Those are
material mismatches with no-pretrain TRKH and justify a fail-closed train-only
screen rather than direct use in a full train.

## 3. No-repeat boundary

This hypothesis is distinct from closed input attention-view drop, obstacle,
CutMix/SnapMix, FriendlyAdv, MixStyle, token pruning, and post-hoc class-1
filters:

- selection is deterministic and uses the true-label logit gradient;
- masking acts on a learned feature vector, not pixels or raw attention;
- the challenged CE updates the original CNN, Transformer, pooling, and head;
- class 1 is protected during training rather than suppressed at inference.

Do not reinterpret this as another input crop/drop, random dropout, confidence
router, or class-1 threshold.

## 4. Exact TRKH adaptation

Let `z` be the 256-dimensional `features["pooled"]` vector after final token
normalization and `cls_branch_register_mean`, but before fine-grained patch
pooling. Let `h(z)` be the unchanged fine-grained pooling, classifier, and
enabled additive CNN path. For target `y`:

1. Compute `g = d h(z)[y] / d z` from the clean true-class logit.
2. Set exactly the largest `ceil(256 / 3) = 86` entries of `g` to zero in a
   binary channel mask. Ranking uses signed gradients, not absolute values.
3. True class-1 rows receive an all-one mask and are never challenged.
4. Eligible rows are true classes `0,2,3,4`.
5. With top modules temporarily in eval mode, compute
   `drop = p_y(clean) - p_y(masked) - 1e-4`.
6. Among eligible rows with positive `drop`, challenge at most
   `ceil(eligible_count / 3)` rows with the largest drop. No random selection
   or tie-breaking noise is allowed.
7. Restore train mode and compute ordinary cross-entropy from the mixed batch:
   selected rows use `z * mask`; every other row uses clean `z`.
8. The RSC CE replaces, rather than supplements or blends with, the ordinary
   CE for the candidate. No new loss weight exists.

The locate pass must consume no random number. Candidate and control each
consume one stochastic backbone pass and one stochastic final-head pass per
batch. Class-1 protection is a hard invariant, not a tunable option.

## 5. Locked A0 evidence

A no-training FP32 screen used only source-disjoint fold-0 train rows:

- fold rows: `1843`;
- restricted `0/2/4 -> 1` false positives: `36`;
- true class-1 rows: `109`;
- matched correct non-class-1 rows: `36`;
- exact mask width on eligible rows: `86/256`;
- class-1 mask width and prediction changes: `0`, `0/109`;
- restricted-FP changes/corrections: `1/1`;
- restricted-FP mean class-1 probability delta: `-0.0096854`;
- matched-correct mean class-1 probability delta: `+0.0258051`;
- CIDT prediction mismatches and non-finite values: `0`, `0`.

This proves that the operation is active, class-1 protection is exact, and the
mask exposes class-1 pressure on clean negatives. It does not authorize
validation or establish a metric improvement.

## 6. Stage-A data and optimization lock

- Dataset: only `D:\DataAI\AIEx\newdataset\yolo_f\train`.
- Dataset YAML SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- CIDT clean train source-fold rows are the immutable split source.
- Fold: source-disjoint fold 0; fit/holdout rows `7372/1843`.
- Train order: NumPy PCG64 seed `42`, first `60 x 32 = 1920` fit rows.
- Transform: deterministic keeper evaluation transform; no augmentation.
- Variants: bit-identical keeper-initialized control and candidate.
- Trainable parameters: every existing model parameter for both variants.
- Control objective: ordinary cross-entropy.
- Candidate objective: the exact protected RSC objective in Section 4.
- Optimizer: AdamW, LR `1e-5`, weight decay `0.05`, betas `(0.9,0.999)`.
- Precision: FP32 adaptation and gate evaluation; TF32 disabled.
- Seed: `42`; deterministic algorithms enabled.
- No scheduler, EMA, sample weights, teacher, auxiliary loss, or augmentation.

No LR, drop ratio, batch ratio, protected class, target classes, fold, seed,
budget, optimizer, representation location, or loss sweep is allowed.

## 7. Required structural and mechanism checks

All checks below must pass:

- keeper/data/CIDT/paper/source/protocol hashes are exact;
- no validation or test path is constructed or read;
- control and candidate start with identical states and parameter schemas;
- inference parameter count is unchanged;
- train orders and initial forward RNG states are identical;
- class-1 rows always have an all-one mask and are never selected;
- selected rows are positive-drop true classes `0,2,3,4` only;
- every selected row masks exactly 86 signed-top-gradient channels;
- locate equations match a direct FP32 reference;
- clean and challenged logits, gradients, and loss are finite;
- all major parameter families receive finite nonzero gradients and move;
- candidate peak VRAM is at most `3.25 GiB`;
- candidate training-step runtime is at most `1.35x` control;
- unchanged inference graph exports with static batch-1 ONNX error `<=1e-5`.

## 8. Clean train-only behavior gates

On all 1843 fold-0 holdout rows, candidate versus matched control must satisfy
every gate:

- macro F1 delta `>= 0.000`;
- class-1 F1 delta `>= +0.005`;
- class-1 precision delta `>= +0.005`;
- class-1 recall delta `>= -0.005`;
- restricted `0/2/4 -> 1` false-positive reduction `>= 2`;
- class-1 FN rescues are at least class-1 TP breaks;
- total corrections are greater than harms;
- maximum non-class-1 per-class F1 drop `<= 0.010`;
- predicted class-1 support is at least `95%` of control support.

Precision gained by collapsing class-1 support or recall is an automatic
failure, even if false positives fall.

## 9. Illumination safety gates

Evaluate the same holdout under locked dim, bright, and low-contrast transforms.
For candidate versus control:

- every condition has class-1 F1 delta `>= -0.010`;
- every condition has class-1 recall delta `>= -0.015`;
- at least two of three conditions have nonnegative class-1 precision delta;
- aggregate class-1 TP breaks do not exceed FN rescues;
- no condition creates more restricted class-1 FP than it removes.

## 10. Decision

Stage B is authorized only if every structural, mechanism, clean, illumination,
runtime, memory, and export gate passes. Stage B permits exactly one
`120 batches x 2 epochs` full-validation, no-test smoke after the method is
wired default-off into the main trainer and V8 launcher.

That smoke must reach at least:

- validation macro F1 `0.882925`;
- class-1 F1 `0.683261`;
- class-1 precision `0.613093`;
- class-1 recall `0.774834` and at least `117/151` TP;
- at least four locked focus-FP removals with no net class-1 TP loss.

Any Stage-A failure closes this exact route before validation. Do not sweep a
nearby setting, run test, launch a full train, or update the current-best
command/history. A command-history revision is allowed only after a later
candidate wins the locked validation promotion protocol.
