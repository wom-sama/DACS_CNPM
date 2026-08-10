# TRKH Friendly Foreground Adversarial Protocol - 2026-07-12

## Decision question

Can a small, fruit-interior, class-1-directed adversarial neighborhood supply
new supervision that protects true class-1 predictions and suppresses
`0/2/4 -> 1` false positives without moving the clean keeper gradient in an
unsafe direction?

This is not another logit margin, current-embedding router, background filter,
or architecture substitution. The candidate creates new input-space examples
from the existing train images. Raw files and labels remain unchanged.

## Primary references

- Friendly Adversarial Training, ICML 2020:
  <https://proceedings.mlr.press/v119/zhang20z.html>
- AdvProp, CVPR 2020:
  <https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Adversarial_Examples_Improve_Image_Recognition_CVPR_2020_paper.html>
- Learnable Boundary Guided Adversarial Training, ICCV 2021:
  <https://openaccess.thecvf.com/content/ICCV2021/html/Cui_Learnable_Boundary_Guided_Adversarial_Training_ICCV_2021_paper.html>

The protocol adopts FAT's earliest-boundary-crossing principle and AdvProp's
treatment of adversarial images as auxiliary examples. It does not claim
adversarial robustness and does not copy auxiliary-BN machinery: the keeper is
evaluated with frozen normalization state and the candidate is gated only on
clean five-class validation behavior.

## Locked keeper and data

- Checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`
- Checkpoint SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Data: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`
- Splits allowed: full train `9215` and full validation `2606` only.
- Test is forbidden for readiness and smoke selection.
- Expected independent FP32 validation keeper macro/class-1 F1:
  `0.882925/0.678261`.

## Locked readiness audit

The readiness tool may write metrics, CSV, plots, and previews. It must not
write a checkpoint, fit a classifier/router, modify raw data, or select a
threshold from validation.

1. Reproduce full train and validation predictions in FP32 with the keeper's
   exact eval transform, bbox metadata, image-valid mask, and bbox-token prior.
2. Define the positive-protection pool as clean-correct `y=1,pred=1` rows.
3. Define the negative-suppression pool as clean-correct rows with
   `y in {0,2,4}`, class 1 in top-2, and `pred=y`.
4. Define correction targets independently as class-1 false negatives and
   `0/2/4 -> 1` false positives.
5. Build five deterministic source-grouped train folds. The keeper train
   representation is in-sample; fold results describe update-direction
   transfer only and must not be called keeper OOF performance.
6. Generate deterministic FP32 early-stopped PGD examples only from the two
   clean-correct pools:
   - RGB `L_inf` radius `2/255`;
   - step size `1/255`, at most two steps, no random start;
   - perturb only the valid-mask intersection with the crop bbox eroded by
     `10%` on every side;
   - positive protection maximizes `max(logit_0,logit_2,logit_4)-logit_1`;
   - negative suppression maximizes `logit_1-logit_y`;
   - retain the first step that crosses the declared boundary; otherwise keep
     the final bounded step.
7. Detach generated pixels before the defense loss. Compare parameter-gradient
   directions on all keeper parameters whose checkpoint-loaded
   `requires_grad` flag is true. Record the exact name/shape list and its hash;
   do not select layers after observing results. Each objective uses at most 32
   rows, ordered by smallest clean class-1 boundary margin and then sample
   index, with fixed mini-batches of eight:
   - positive defense CE versus held-fold class-1-FN correction CE;
   - negative defense CE versus held-fold `0/2/4 -> 1` correction CE;
   - combined adversarial defense CE versus clean boundary CE.
8. Repeat the fixed direction comparison against validation correction rows as
   transfer-only evidence. No validation-driven epsilon, step, mask, sample
   count, parameter subset, or loss-weight search is allowed.

## Readiness gates

Every check is fail-closed. Smoke permission requires all checks:

- checkpoint hash and class order match;
- support is exactly `9215/2606`, train/validation source overlap is zero, and
  test access is false;
- FP32 validation control matches macro/class-1 `0.882925/0.678261` within
  `1e-6`;
- perturbation outside the declared mask is exactly zero and maximum RGB delta
  is at most `2/255 + 1e-7`;
- both attack directions cross the boundary on at least `10%` of their eligible
  rows in train and validation;
- every train fold contains both correction-target directions;
- each train direction has median fold gradient cosine at least `0.05` and at
  least four of five fold cosines are positive;
- validation positive-protection and negative-suppression correction cosines
  are each at least `0.02`;
- combined adversarial-defense versus clean-boundary gradient cosine is at
  least `0.10` on train and non-negative on validation.

Failure closes this exact `epsilon=2/255`, two-step, eroded-bbox friendly
adversarial formulation before trainer integration. No nearby epsilon, step,
erosion, attack-objective, cohort, or parameter-subset sweep is permitted from
the failed evidence.

## Predeclared smoke if readiness passes

- Resume the locked keeper; preserve the current clean recipe.
- One epoch, `40` train batches, full validation, final test skipped.
- Auxiliary adversarial CE weight `0.05`.
- Use the exact readiness attack and frozen normalization statistics.
- Per batch, cap at eight positive-protection and eight
  negative-suppression rows; keep the two directions equally weighted when both
  exist.
- Run independent FP32 checkpoint reload, boundary audit, architecture trace,
  and balanced XAI after the smoke.

Permission for a longer probe requires all of:

- validation macro F1 at least `0.882925`;
- class-1 F1 at least `0.700000`;
- class-1 recall at least `0.761589` (at least 115/151);
- no more than `70` total `0/2/3/4 -> 1` false positives;
- background perturbations remain materially below object-desaturation effect;
- no test access and no current-best command update.

Only an independently reloaded candidate that later beats the locked keeper
and promotion gate may change the full-train command file.
