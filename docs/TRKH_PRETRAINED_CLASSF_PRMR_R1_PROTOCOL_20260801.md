# PRMR-R1 matched probe protocol — 2026-08-01

Protocol IDs:

- `TRKH_PRETRAINED_CLASSF_PRMR_R1_CONTROL_20260801`
- `TRKH_PRETRAINED_CLASSF_PRMR_R1_20260801`

Status: locked paired development probe. Full training is not authorized by
this protocol. Canonical test inference remains forbidden.

## Evidence and hypothesis

On canonical `class_f` validation (2,479 images), the current DINOv3 B9
reference has macro-F1 `0.876324` and class-1 F1 `0.701149`. Under the fixed
bright condition, class-1 F1 falls to `0.532438` and restricted
`0/2/4 -> 1` false positives rise from `68` to `170`. B0 shows the same
failure direction. A validation-only class-1 bias sweep improves B9 only to
approximately `0.7032`, so threshold tuning is not an adequate remedy.

PRMR-R1 tests whether already-correct class-1 decision margins can be retained
under controlled luminance changes without forcing the full output
distribution to match.

For focus class `c=1`, rivals `R={0,2,4}`, clean logits `z`, relit logits `z'`,
and retention `rho=0.8`:

```text
y=c:   a=stopgrad(z_c-z_r), b=z'_c-z'_r, r in R
y in R:a=stopgrad(z_y-z_c), b=z'_y-z'_c
d=max(0, rho*a-b), only where a>0
```

Rival terms are averaged per class-1 sample. The protect and suppress
directions are then averaged with equal direction weight when both exist.
Soft/mixed targets fail closed. PRMR adds no parameter or inference operation;
it adds one selected-subset forward only during training.

The paired view changes luminance while preserving the RGB chroma residual:

```text
Y  = 0.299R + 0.587G + 0.114B
Y' = gain * (mean(Y) + contrast * (Y-mean(Y)))
RGB' = clamp(RGB + (Y'-Y), 0, 1)
```

Each selected subset is exactly balanced between dim and bright polarities
when its size is even. For an odd subset the difference is one and the extra
polarity is randomized, avoiding a systematic bright/dim bias. Probe settings
are brightness delta `0.25`, contrast delta `0.10`, probability `0.50`, loss
weight `0.15`, and start epoch `3` after the new classifier head has warmed up.
Violation fractions in history are computed from epoch-wide violation and
eligible-pair counts, not by averaging per-batch fractions.

## Matched arms

Both arms use fresh locked DINOv3-S/16 initialization plus a new five-class
head, seed 42, the same canonical train/validation split, tempered sampler
`q_c proportional to n_c**0.5`, LDAM-Focal recipe, EMA, BF16, batch
`24 x accumulation 2`, five epochs, at most 120 train batches per epoch, and
full validation every epoch.

- `prmr-r1-control`: PRMR default-off.
- `prmr-r1`: only the locked PRMR flags above are added.

No sweep, rescue run, checkpoint cherry-pick, threshold tuning, test read, or
change of selector is permitted after observing the pair.

## Decision gate

Use independently reloaded FP32 predictions from the predeclared selected
checkpoint. PRMR-R1 is favorable only if all conditions hold versus its paired
control:

- clean class-1 F1 improves by at least `0.010`, or reaches `0.72`;
- clean macro-F1 is no worse than control minus `0.0015`;
- class-1 recall delta is at least `-0.020` and TP retention is at least `0.97`;
- restricted `0/2/4 -> 1` and `2 -> 1` errors each fall by at least 20%;
- bright class-1 F1 improves by at least `0.030`, and dim class-1 F1 by at
  least `0.010`.

A favorable probe only authorizes writing a separate confirmation/full
protocol. A failed gate closes R1 without tuning its weight, probability,
retention, start epoch, or relighting strength on this validation result.

## Scientific claim boundary

The mechanism is new within the TRKH lineage. Broad literature novelty is not
claimed. Paired-view consistency, stop-gradient, margin objectives, and
photometric robustness all have prior art; the defensible distinction is this
specific error-driven, bidirectional, positive-clean-margin retention rule.
Do not call it adversarial robustness or illumination invariance. Report it as
synthetic luminance-shift robustness. Strong publication claims require the
locked ablations and multiple seeds after R1 passes.

Nearest audited primary references include
[MaCS](https://arxiv.org/abs/2603.05812),
[CR-Aug](https://arxiv.org/abs/2205.12461),
[AugMix](https://arxiv.org/abs/1912.02781), and
[TRADES](https://proceedings.mlr.press/v97/zhang19p.html). No exact formula
match was found in this bounded review; this is not a claim of exhaustive
literature novelty.

## Authorized commands

Run control first, then candidate, from the same clean commit:

```powershell
$env:PYTHONPATH = 'D:\DataAI\AIEx\TRKH_pretrained'
$Python = 'D:\DataAI\.venv\Scripts\python.exe'
$Recipe = 'trkh.recipes.pretrained_classf_b0'
$Data = 'D:\DataAI\AIEx\newdataset\class_f\data.yaml'
$Dev = 'D:\DataAI\AIEx\TRKH_pretrained\configs\class_f_5class_dev.yaml'
$Attest = 'D:\DataAI\AIEx\TRKH_pretrained\runs\canonical_classf_contract_20260731.json'
$Dino = 'C:\Users\ADMIN\.cache\huggingface\hub\models--timm--vit_small_patch16_dinov3.lvd1689m\snapshots\3bf4720a82ec2066db88137180ff1f83a675cef0\model.safetensors'

& $Python -m $Recipe --mode probe --experiment prmr-r1-control `
  --python $Python --data $Data --canonical-attestation $Attest `
  --train-data $Dev --dino-checkpoint $Dino --output-dir runs `
  --run-tag 20260801_prmr_r1_control_r1 --batch-size 24 `
  --num-workers 0 --eval-num-workers 0 --amp-dtype bf16

& $Python -m $Recipe --mode probe --experiment prmr-r1 `
  --python $Python --data $Data --canonical-attestation $Attest `
  --train-data $Dev --dino-checkpoint $Dino --output-dir runs `
  --run-tag 20260801_prmr_r1_candidate_r1 --batch-size 24 `
  --num-workers 0 --eval-num-workers 0 --amp-dtype bf16
```
