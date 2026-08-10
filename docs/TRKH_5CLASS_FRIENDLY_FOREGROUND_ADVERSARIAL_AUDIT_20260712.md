# TRKH 5-Class Friendly Foreground Adversarial Audit - 2026-07-12

## Decision

Reject the locked friendly foreground adversarial candidate after its first
40-batch smoke. Do not open a longer probe, test split, or current-best command
promotion. Keep the implementation default-off for reproducibility, but do not
sweep its epsilon, step count, erosion, loss weight, cohort, or attack objective
on the current keeper.

The locked protocol is
`docs/TRKH_5CLASS_FRIENDLY_FOREGROUND_ADVERSARIAL_PROTOCOL_20260712.md`.
Full no-test readiness evidence is retained at
`runs/diagnostic_friendly_foreground_adversarial_full_20260712`; compact smoke
evidence is at
`runs/evidence_friendly_foreground_adversarial_w005_smoke_reject_20260712`.

## Research Basis

The method was fixed before implementation from three primary sources:

- [Friendly Adversarial Training, ICML 2020](https://proceedings.mlr.press/v119/zhang20z.html)
- [AdvProp, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Adversarial_Examples_Improve_Image_Recognition_CVPR_2020_paper.html)
- [Learnable Boundary Guided Adversarial Training, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Cui_Learnable_Boundary_Guided_Adversarial_Training_ICCV_2021_paper.html)

The fixed candidate perturbed only the valid-mask intersection with the crop
bbox eroded by 10% per side. It used RGB `L_inf=2/255`, step `1/255`, at most
two deterministic steps, no random start, and stopped at the first declared
class-1 boundary crossing. Clean-correct class 1 rows were attacked away from
class 1; clean-correct classes `0/2/4` with class 1 in top-2 were attacked
toward class 1. Detached adversarial images then received ordinary true-label
CE at weight `0.05`.

## Readiness Result

The FP32 audit covered all `9215/2606` train/validation rows, `8064/2577`
source groups, zero source overlap, and no test access. It reproduced the
keeper validation macro/class-1 F1 exactly as
`0.882924855/0.678260863`.

| Check | Train | Validation | Gate |
| --- | ---: | ---: | ---: |
| Protect crossing rate | 0.662879 | 0.735043 | >=0.10 |
| Suppress crossing rate | 0.786755 | 0.822727 | >=0.10 |
| Max RGB delta | 0.007843167 | 0.007843167 | <=2/255 |
| Outside-mask delta | 0 | 0 | exactly 0 |
| Empty attack masks | 0 | 0 | 0 |
| Combined clean cosine | 0.972652 | 0.915201 | >=0.10 / >=0 |

All five source folds contained both correction directions. Median fold
cosines were `0.895480` protect and `0.828468` suppress, with `5/5` positive
folds. Validation correction cosines were `0.601035/0.741000`. All 21 locked
checks passed, so the predeclared smoke was permitted.

This result is a safety/direction check, not evidence of useful
generalization. The adversarial CE and correction CE share true labels, so a
high all-parameter cosine can be dominated by ordinary class supervision. The
smoke result below is the deciding gate.

## Trainer Integration

- Added a shared deterministic attack primitive and default-off `TrainConfig`,
  CLI, v8 launcher, primary-step, and SAM-replay plumbing.
- Candidate selection is clean-correct and nearest-boundary, capped at eight
  protect plus eight suppress rows per batch.
- Attack forwards use eval mode; every module's prior train/eval flag is
  restored before adversarial CE and again in `finally`.
- The primitive now rejects `step_size > epsilon`, and start epoch is enforced
  as 1-based.
- A one-batch runtime preflight activated both directions without OOM. The
  40-batch smoke averaged protect/suppress counts `5.8/4.475`, crossing rates
  `0.801310/0.588839`, adversarial CE `1.482813`, and max RGB delta
  `0.007843167`.

## Independent Smoke Gate

The smoke resumed the locked keeper for one epoch and 40 train batches with
full validation, batch 32, accumulation 2, LR `8e-5`, scheduler horizon 10,
and no test. The independent FP32 batch-32 reload covered all 2606 rows.

| Metric | Keeper FP32 | FriendlyAdv smoke | Required | Result |
| --- | ---: | ---: | ---: | --- |
| Macro F1 | 0.882925 | 0.882354 | >=0.882925 | fail |
| Class-1 F1 | 0.678261 | 0.676471 | >=0.700000 | fail |
| Class-1 precision | 0.603093 | 0.608466 | diagnostic | +0.005373 |
| Class-1 recall | 0.774834 | 0.761589 | >=0.761589 | pass at floor |
| Class-1 TP | 117 | 115 | >=115 | pass at floor |
| Class-1 FP | 77 | 74 | <=70 | fail |

Independent confusion was:

```text
[[486, 49,  1,  1, 12],
 [ 19,115, 12,  0,  5],
 [  5,  9,513, 10,  7],
 [  0,  4, 50,656,  2],
 [  8, 12,  4,  1,625]]
```

Only 12 decisions changed versus the exact keeper reload: six corrections and
six harms. The candidate removed three class-1 false positives and created
none, but rescued zero class-1 false negatives and broke two keeper class-1
true positives. The intended conservative FP effect was too small and was
recall-unsafe.

## Boundary And XAI Audit

The boundary manifest selected 142 validation rows: 36 class-1 FN, 26
class-1 FP, 40 low-margin errors, and 40 low-margin correct boundary rows. A
fixed 12-case XAI set used the two highest-confidence errors for each of
`1->0`, `1->2`, `1->4`, `0->1`, `2->1`, and `4->1`.

- Attention foreground/background/border mass: `0.87564/0.12436/0.16313`.
- Grad-rollout foreground/background/border: `0.92765/0.07235/0.18201`.
- Grad-CAM foreground/background/border: `0.82971/0.17029/0.24671`.
- Background blur/gray predicted-probability drops: `0.00641/0.00602`.
- Center occlusion drop: `0.01698`; object desaturation drop: `0.13788`.
- Flags: object-color sensitive `6/12`, Grad-CAM background `4/12`, Grad-CAM
  border `6/12`, rollout background/border `8/12` and `6/12`.

The robustness context gate passed because object desaturation was about 22x
larger than either background intervention. Visual review still showed broad
fruit color, silhouette, endpoint/stem, lesions, crop edge, and occasional
background hotspots in both FN and FP directions. The attack did not supply a
stable class-1-positive interior cue.

## Cleanup And Verification

- Readiness manifest: six payloads, `2,910,654` bytes, SHA-256
  `9e473c2c...adf65`, correct friendly-adversarial manifest mode, no model,
  checkpoint, or test payload.
- Compact rejected-smoke evidence: 346 verified payloads, `46,895,735` copied
  source bytes, SHA-256 `c7c7418e...56a6c`.
- Four checkpoint files totaling `348,748,896` bytes were excluded but retain
  source hashes. Smoke `best.pt` SHA-256 is `b8dca372...ce332`.
- Cleanup deleted six verified source roots and observed `396,460,032` bytes
  of free-space gain. Raw data, test, keeper, command packet, and protected
  user paths were untouched.
- Compileall, PowerShell parse, focused tests `15/15`, and full pytest
  `814/814` passed. Retention passed over 603 run directories with
  `blockers=[]`.
- The current-best one-command wrapper passed a direct Windows PowerShell
  `-PreflightOnly` run with epochs 30, effective batch 64, train-side final
  test skipped, the locked keeper resolved, and FriendlyAdv default-off at
  weight `0`. It created no run directory.
- Keeper SHA remains `1f49d577...482677`; current-best command SHA remains
  `3c718130...57dd`.

## Closed Variants

Do not sweep this method's epsilon, PGD step/count/random start, bbox erosion,
focus/negative cohorts, sample caps, loss weight, start epoch, LR, or nearby
attack margin. Do not infer that auxiliary BN, a different adversarial paper,
or a longer run will fix the observed zero-FN-rescue/two-TP-break direction
without a new independent keeper-relative readiness target. The current-best
full-train, engine-export, and video-test commands remain unchanged.
