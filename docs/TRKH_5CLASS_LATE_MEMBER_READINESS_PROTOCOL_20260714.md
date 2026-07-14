# TRKH Late-Member Readiness Protocol - 2026-07-14

## Question

Can the illumination-recall complement in the 2026-07-14 scratch member be
retained by a TRKH member that shares the keeper's early representation and
branches only after token pruning, without importing the scratch member's
class-1 false-positive behavior?

This is a no-test architecture-readiness audit. It is not a checkpoint soup,
not a fork-depth sweep, and not permission to promote a two-checkpoint model.

## Fixed Evidence

- Dataset: `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`.
- Audit split: complete validation only, exactly `2606` object rows.
- Keeper checkpoint:
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Scratch complement checkpoint:
  `runs\full_v8_yolof_randominit_30e_20260714_105524\checkpoints\best.pt`.
- Scratch SHA-256:
  `f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549`.
- Frozen comparison rule: keeper weight `0.60`, complement weight `0.40`,
  class-1 probability offset `0.034`. The rule is reused without search.
- Frozen rule source summary SHA-256:
  `6c6397d3561ec917d9716ecfd2c6b65a616433e3a07c45031df898578d07a8c8`.
- Test is closed. No test prediction, metric, threshold, weight, margin, or
  model-selection decision may be produced by this stage.

## Research Basis

- Hydra uses a shared body with separate heads to preserve member behavior
  instead of distilling only the ensemble mean:
  https://openreview.net/forum?id=ByeaXeBFvH
- MIMO demonstrates that independently trained subnetworks can retain useful
  predictive diversity inside one model:
  https://arxiv.org/abs/2010.06610
- MixMo shows that member-specific paths and outputs matter; simple feature
  summation can erase diversity:
  https://openaccess.thecvf.com/content/ICCV2021/html/Rame_MixMo_Mixing_Multiple_Inputs_for_Multiple_Outputs_via_Deep_Subnetworks_ICCV_2021_paper.html

These papers motivate the shared-body/separate-member form, but they do not
justify copying arbitrary trained weights across incompatible feature spaces.
That compatibility is the first local question tested here.

## Locked Diagnostic Splice

The architecture has eight transformer blocks and prunes after blocks 2 and 5.
The one declared fork is after block 6, so both paths receive the same final
post-pruning token set and only blocks 7-8 plus the readout are member-specific.

Use the keeper checkpoint as the complete base. Copy only these prefixes from
the scratch checkpoint:

- `blocks.6.` and `blocks.7.`
- `norm.` and `head.`
- `fine_grained_pool.`
- `pairwise_margin_norm.` and `pairwise_margin_head.`
- `cnn_fusion_norm.` and `cnn_fusion_head.`
- `bbox_spatial_fusion_head.`

Everything before the fork, including stem, patch embedding, positional and
register tokens, branch-token fusion, detail enhancer, and blocks 1-6, remains
bit-identical to the keeper. State schemas and tensor shapes must match before
writing. The generated checkpoint must record source hashes, exact copied
keys, exact retained keys, fork depth, and output hash.

The splice is a compatibility control only. A failed splice rejects direct
weight transplant, not the trainable late-member architecture. A successful
splice permits implementation, but does not bypass smoke/probe gates.

## Fixed Evaluation

1. Independently evaluate the generated splice in FP32 on all `2606`
   validation rows and export detailed probabilities.
2. Verify the original keeper probability CSV hash and row alignment.
3. Apply the already frozen `0.60/0.40 + 0.034` rule exactly once to keeper
   plus splice predictions. Do not run a weight, offset, temperature, or
   threshold search.
4. Reconstruct confusion matrices and class-1 TP/FP/FN from CSVs.
5. If the fixed ensemble reaches the gates below, run matched native-attention
   and Grad-CAM review on class-1 rescue/harm/FP transitions. Otherwise, record
   the failure and proceed only with keeper-initialized trainable branching.

## Readiness Gates

- Validation support is exactly `2606`; labels, sample indices, paths, class
  names, and source groups align with the keeper CSV.
- No test path or test row appears in any input or output.
- All untouched tensors remain bit-identical to the keeper.
- Splice class-1 recall is at least `0.80` and class-1 false positives are at
  most the scratch member's `110`.
- Fixed keeper-plus-splice class-1 precision is at least `0.679245`.
- Fixed keeper-plus-splice class-1 recall is at least `0.715232`.
- Fixed keeper-plus-splice class-1 F1 is at least `0.696774`.
- Fixed keeper-plus-splice macro F1 is at least `0.890298`.
- Class-1 false positives decrease versus keeper and class-1 TP breaks do not
  exceed class-1 FN rescues.
- Corrections are not fewer than harms.

All thresholds are the existing frozen dual-checkpoint result, not newly
selected targets. Failure means the scratch late weights are not compatible
with the keeper shared representation. It must not trigger a fork-depth or
prefix sweep.

## Next Architecture If Direct Splice Fails

Create a default-off TRKH late member with shared stem/blocks 1-6 and separate
blocks 7-8 plus readout. Initialize both paths from the keeper so the primary
path reproduces the keeper exactly. Freeze the primary path during the first
probe. Train only the complement path on train rows, with explicit class-1 TP
protection and conservative `0/2/4 -> 1` control derived from train-only
teacher disagreement. Full 30-epoch training remains closed until a smoke and
short probe pass the next validation milestone.

## Direct-Splice Result

The direct splice failed and is closed. Its standalone validation macro/class-1
F1 was `0.142509/0.134868`. The frozen keeper/splice rule reached
`0.877641/0.666667`; it made `35` corrections and `40` harms, rescued no
keeper class-1 false negatives, and broke `19` keeper class-1 true positives.
The scratch blocks are therefore incompatible with the keeper feature basis.
No fork depth, copied-prefix, ensemble weight, or offset sweep was run.

## Keeper-Initialized Control

The default-off trainable late member was initialized from the keeper and the
primary path was frozen and held in evaluation mode. Before training, full-val
macro/class-1 F1 was `0.880391/0.662338`, with class-1 precision/recall
`0.649682/0.675497`. A matched one-epoch, 40-batch fused-CE control reached
`0.880918/0.670588`, with class-1 precision/recall `0.603175/0.754967`.
Relative to the raw keeper it made `8` corrections and `10` harms, rescued no
class-1 false negatives, broke `3` class-1 true positives, removed `4` class-1
false positives, and created `2` new ones. All `185` shared tensors remained
bit-identical while all `56` late-member tensors changed.

Native late-member XAI used candidate block 7 for every reviewed case. Mean
foreground mass was `0.9203` for native attention, `0.9468` for grad-rollout,
`0.8918` for Grad-CAM, and `0.9043` for rollout. Object desaturation changed
class-1 probability by `0.1589` on average, versus `0.0121/0.0109` for
background blur/gray. This confirms surface/color ambiguity rather than a new
wide-background failure. The control is rejected because ordinary fused CE
raises candidate class-1 probability broadly and loses precision.

## Locked Directional Smoke

The next smoke starts from the same keeper-initialized checkpoint, uses the
same seed, 40 train batches, batch size, accumulation, learning rate, complete
validation, and no test. The scratch complement is an online frozen teacher;
global KD and candidate auxiliary CE remain disabled. Only teacher/keeper
disagreements that agree with train labels are transferred:

- true class 1 where teacher predicts class 1 and teacher `p1` exceeds keeper
  `p1` by at least `0.04`;
- true class `0/2/4` where teacher predicts non-class-1 and teacher `p1` is at
  least `0.04` below keeper `p1`;
- teacher binary targets use hard-target blend `0.25`;
- directional loss weight is `0.20`;
- keeper-correct class-1 and `0/2/4` probabilities are protected with slack
  `0.01` and preservation weight `0.20`.

The smoke may advance to a 120-batch, at-most-two-epoch probe only if full-val
class-1 precision is at least `0.64`, class-1 F1 is at least `0.68`, macro F1
is no more than `0.002` below the raw keeper, false positives do not exceed
`70`, class-1 TP breaks do not exceed FN rescues, and corrections are not fewer
than harms. A full train remains closed until the probe reaches class-1 F1
`0.70` under the repository milestone rule.

### Invalid First Invocation

Run `smoke_v10_yolof_latemember_directional020_preserve020_fork6_precision034_40b_1e_20260714`
is excluded from method selection. Passing `--late-member-branch` while loading
checkpoint configuration activated the model-extension path and replaced the
frozen class-1 offset `0.034` with the CLI default `0.0`. The resolved config
proves the mismatch. Its macro/class-1 F1 `0.878760/0.664740` and class-1
precision/recall `0.589744/0.761589` are retained only as invalid-configuration
evidence. It receives no transition or XAI audit and cannot satisfy or fail the
locked directional gate. The corrected invocation omits model extension and
inherits all late-member architecture fields from the initializer checkpoint.

## Directional-Smoke Result

The corrected directional run
`smoke_v10_yolof_latemember_directional020_preserve020_fixedoffset034_40b_1e_20260714`
is valid but rejected before a 120-batch probe. Independent FP32 full validation
reached macro/class-1 F1 `0.880357/0.668622` and class-1 precision/recall
`0.600000/0.754967`, below the raw keeper `0.882925/0.678261` and
`0.603093/0.774834`. Relative to the keeper it changed `24` rows, made `10`
corrections and `13` harms, rescued/broke class-1 FN/TP `1/4`, and
removed/created class-1 FP `4/3`.

The train-only directional masks selected only about `14` positive and `7`
negative rows across all `40` batches (`1.09%/0.55%` of samples). This is too
sparse to counter the dense fused cross-entropy objective. It also confirms
that a scratch member evaluated in-sample is not a useful hard-disagreement
selector even though its soft probabilities remain complementary on held-out
validation.

The locked 12-case XAI cohort contains the one FN rescue, all four TP breaks,
all three created FP, and four removed FP/corrections. Every case used native
late-member block-7 attention. Mean foreground mass was `0.905834` attention,
`0.944412` grad-rollout, `0.841837` Grad-CAM, and `0.909739` rollout. Object
desaturation changed class-1 probability by `0.065915`, versus
`-0.008455/-0.005140` for background blur/gray. All 12 cases were near-ties;
the harms are surface/color/boundary ambiguities, not evidence that wider
background context should be emphasized. No weight, gap, blend, or run-length
sweep is permitted for this rejected sparse objective.

## Locked Dense Precision-Rule Distillation

The next method changes the supervision formulation rather than tuning the
rejected sparse masks. It follows two primary references:

- Hydra preserves ensemble-member behavior with a shared body and separate
  heads: https://openreview.net/forum?id=ByeaXeBFvH
- Positive-Congruent Training defines negative flips and applies focal
  distillation with extra weight on rows the reference model predicts
  correctly: https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html

For every train image, compute the already frozen validation-selected target
in probability space:

1. `q = 0.60 * softmax(keeper) + 0.40 * softmax(scratch)`;
2. subtract `0.034` from `q[class1]`, clamp it to `1e-8`, and renormalize;
3. minimize `KL(q || softmax(late-member fused logits))` on every row;
4. use focal-congruence weight `1 + 5 * I(keeper prediction == label)` and
   normalize by the sum of weights.

The model's own candidate weight, focus class, and focus offset are the single
source of truth; duplicate CLI values are forbidden. The primary path and
teacher remain frozen. Standard fused CE stays enabled as in Equation 4 of
Positive-Congruent Training; global KD, candidate auxiliary CE, sparse
directional loss, and probability-hinge preservation are disabled. This is a
single predeclared setting: precision-rule loss weight `1.0`, focal correct
boost `5.0`, start epoch `1`.

The matched smoke starts from
`runs\late_member_keeper_init_fork6_20260714\checkpoints\init.pt`, uses seed
`42`, batch `32`, accumulation `2`, LR `5e-5`, scheduler horizon `10`, one
epoch/`40` train batches, all `2606` validation rows, and no test. It may
advance to a 120-batch, at-most-two-epoch probe only if independent FP32
validation reaches class-1 precision at least `0.64`, class-1 F1 at least
`0.68`, macro F1 no more than `0.002` below the raw keeper, class-1 FP at most
`70`, TP breaks no greater than FN rescues, and corrections no fewer than
harms. Architecture trace, strict transition audit, and a changed-case native
attention/grad-rollout/Grad-CAM/perturbation cohort are mandatory before the
gate decision. Test and current-best commands remain closed.

## Dense KL Result

The dense precision-rule KL smoke is valid but rejected before a 120-batch
probe. Its checkpoint SHA-256 is
`1b37d480f123168c5a893070dff6914da61ef171b62ccef44f11c70ab22d2ceb`.
Independent FP32 full validation reached macro/class-1 F1
`0.879181/0.664723`, class-1 precision/recall `0.593750/0.754967`, and exact
support `2606`. It is below both the raw keeper and every locked smoke gate.

Relative to the keeper, it changed `24` predictions: `9` corrections, `14`
harms, and one wrong-to-wrong transition. Class-1 TP/FP/FN changed from
`117/77/34` to `114/78/37`; FN rescue/TP break was `1/4`, and FP
remove/create was `3/4`. No test row or test metric was used.

The target was dense on every train row and the primary path was correct on
`96.33%` of sampled rows, but the frozen rule disagreed with the primary
argmax on only `1.09%`. More importantly, mean KL was only `0.001541` versus
total train loss `0.837789`; target/student mean absolute probability error
was `0.008258`. At temperature 1 this objective is too weak to counter the
dense fused CE and does not compress the member complement.

The mandatory 12-case XAI cohort covered every class-1 FN rescue, TP break,
FP create, and FP remove. All cases used native
`forward_features.return_attention.late_member_blocks[7]` and all were
near-ties. Attention/grad-rollout/Grad-CAM/rollout foreground mass was
`0.9066/0.9445/0.8657/0.9108`; object-desaturation prediction drop was
`0.0731`, while background blur/gray was `-0.0059/-0.0038`. Visual review
again found pale-green surface, shadow, damage, and fruit-boundary ambiguity;
Grad-CAM sometimes reached hands/image edges, but background perturbation was
not causal. Do not increase context/background supervision or sweep KL weight,
focal boost, seed, LR, or run length.

## Locked Focal Logit Matching

Positive-Congruent Training reports that KL reduces negative flips only at a
very high temperature, where it approaches logit matching, and that direct
logit matching is marginally better. The next and final nearby distance
variant therefore uses its declared FD-LM form rather than tuning the failed
temperature-1 KL:

- construct the same frozen probability target `q` with model-owned
  `0.60/0.40` member weight and class-1 offset `0.034`;
- use target logits `log(clamp(q, 1e-8))` and the model's normalized fused
  logits;
- minimize per-row `0.5 * ||student_logits - target_logits||^2`;
- retain focal-congruence weight `1 + 5 * I(keeper correct)`, standard fused
  CE, frozen primary/teacher, loss weight `1.0`, and all matched runtime
  settings;
- keep global KD, candidate CE, sparse directional loss, and hinge
  preservation disabled.

A one-batch runtime preflight may verify finiteness, exact offset/config, and
gradient routing but may not select a weight. The 40-batch smoke uses full
validation and no test. Because the requested deployment mode explicitly
prioritizes agricultural precision, its advancement gate is class-1 precision
`>=0.65`, recall `>=0.71`, F1 `>=0.69`, macro F1 `>=0.887`, FP `<=60`, TP
breaks `<=10`, and corrections not fewer than harms. These thresholds bracket
the already frozen dual rule (`0.679245/0.715232/0.696774`, macro
`0.890298`) without fitting a new inference margin. Full train remains closed
until a subsequent 120-batch, at-most-two-epoch probe reaches the repository
class-1 F1 milestone `0.70` with the same precision/recall safeguards.

## Focal Logit-Matching Result

The one-batch preflight was finite and inherited the exact model-owned
candidate weight `0.40`, class-1 offset `0.034`, focal correct boost `5`, and
`logit_l2` distance. Its precision-rule loss was `0.005402` with logit RMSE
`0.04643`, full dense-row coverage, and gradients confined to the late-member
path. The matched 40-batch checkpoint SHA-256 is
`8ffdc348018551767d720562dbe54a7d25eac8c2542de33da5ca3efa86eabed5`.

Independent FP32 validation over all `2606` rows reached macro/class-1 F1
`0.879775/0.666667` and class-1 precision/recall `0.596859/0.754967`. Class-1
TP/FP/FN was `114/77/37`, versus keeper `117/77/34`. Strict alignment found
only `23` changed decisions: `9` corrections, `13` harms, and one
wrong-to-wrong transition. It rescued/broke class-1 FN/TP `1/4` and
removed/created class-1 FP `3/3`. It therefore fails every advancement gate
except recall and receives no 120-batch probe or test evaluation.

The train objective was no longer numerically negligible: mean FD-LM was
`0.008318`, logit RMSE `0.057633`, and total train loss `0.843914`. Dense-row
coverage was `1.0`, primary correctness `0.9625`, and rule/primary argmax
disagreement only `0.011719`; target/student mean class-1 probability was
`0.171352/0.176558` with probability MAE `0.008184`. Direct logit matching
therefore strengthened the distance but could not create a stable signal where
the frozen rule itself rarely disagreed with the keeper on train rows.

Mandatory XAI covered the one FN rescue, four TP breaks, three removed FP, and
three created FP. Every one of the 11 cases used native
`forward_features.return_attention.late_member_blocks[7]`; no fallback was
accepted, and all 11 were near-ties. Attention/grad-rollout/Grad-CAM/rollout
foreground mass was `0.9009/0.9426/0.8672/0.9082`. Object desaturation changed
prediction probability by `0.0669`, while background blur/gray changed it by
`-0.0073/-0.0052`. Visual inspection found genuine fruit-surface and damage
focus in several cases, but also edge, surrounding-object, and background
peaks in the precision harms. This is unstable boundary evidence, not support
for a wider-context objective.

FD-LM and direct late-member rule compression are closed. Do not sweep the
distance, loss weight, focal boost, temperature, seed, learning rate, fusion
weight, margin, fork, or run length. The current-best command remains on the
single-checkpoint keeper. A future late-member attempt must first demonstrate
train-held-out feature-basis compatibility through direct representation
matching; output-only teacher matching is no longer an admissible nearby
variant.
