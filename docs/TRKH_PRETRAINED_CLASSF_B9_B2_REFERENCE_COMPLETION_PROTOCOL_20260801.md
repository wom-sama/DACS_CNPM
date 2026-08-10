# B9 B2-shaped canonical reference completion — 2026-08-01

Protocol: `TRKH_PRETRAINED_CLASSF_B9_B2_REFERENCE_COMPLETION_20260801`

Role: `exploratory_canonical_reference_completion`

Promotion eligible: `false`

## Why this run exists

B2 is the best observed pretrained canonical `class_f` probe, but it did not
pass its locked promotion gate: class-1 F1 was `0.653409 < 0.660000` (gap
`-0.006591`). Its other probe values were accuracy `0.889875`, macro-F1
`0.851462`, class-1 precision/recall `0.592784/0.727848`, and `2 -> 1 = 43`.
Consequently, the original B2 protocol does not authorize a B2 full train.

B9 asks only whether the already-observed B2-shaped recipe is useful after a
complete, fixed 30-epoch-cap schedule. It creates a canonical teacher/reference
for presentation, matched B0 comparison, later distillation, and long-horizon
diagnosis. It does not make the failed B2 probe pass retroactively and is not a
confirmatory novelty result. B8 is rejected and contributes no component to
B9.

## Locked training contract

B9 is training-semantics-equivalent to B2. The recipe test removes only run
name, protocol ID, and derived contract hash and then requires the full trainer
argument lists to be byte-identical.

- Fresh DINOv3-S/16 initialization, revision
  `3bf4720a82ec2066db88137180ff1f83a675cef0`, SHA-256
  `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`,
  plus a new five-class head; no B2 checkpoint resume.
- Canonical `class_f` hard labels; 8,278 train and 2,479 validation crops;
  ordered five-class schema; test absent from the development YAML.
- Seed 42; `256x256` pad/bicubic/ImageNet preprocessing; the existing fixed
  light augmentation contract.
- Train sampler `q_c proportional to n_c**0.5`; no class weights, teacher,
  KD, cache, router, B8 descriptor, or post-hoc class correction.
- LDAM-Focal: margin `0.3`, scale `18`, gamma `1`, focal mix `0.1`, label
  smoothing `0.02`.
- AdamW: LR `1.5e-4`, backbone scale `0.1`, minimum LR `1e-6`, weight decay
  `0.05`, gradient clip `0.7`; EMA `0.995`.
- At most 30 epochs; scheduler 30; warmup 2; patience 6; full validation every
  epoch; batch `24 x accumulation 2 = 48`; BF16; local workers `0/0`.
- The existing `fair_macro_f1` checkpoint selector is fixed. No epoch, power,
  threshold, seed, loss, AMP, batch, or selector sweep is allowed after seeing
  validation.
- `--skip-final-test` remains active. Test inference is forbidden in B9.

Early stopping is a valid completion. A run is invalid if any canonical hash,
class order, source commit/tree, parameter count (`21,588,869`), focused test,
finite-training, output-collision, or test-lock check fails. Any failure is
retained. Resume is allowed only for the same run tag, commit, source-tree
digest, protocol/config lineage, BF16 request, and validated `last.pt`; it may
not change the recipe to rescue the run.

The recipe must fail closed for full mode on the old B1/B2 experiment keys.
Only B0 matched-control and B9 reference-completion keys are currently
full-train-authorized.

## Reporting and interpretation

Always retain `best.pt`, `last.pt`, history, summary, resolved config,
preflight/completion manifests, console logs, hashes, full-validation
predictions, and confusion evidence. The selected checkpoint remains the
predeclared fair-selector checkpoint even if another epoch has a more attractive
single metric.

B9 remains exploratory regardless of its score. After a matched B0 full run,
it may be described as a **favorable exploratory reference** only if all hold:

- macro-F1 delta versus B0 `>= +0.005`;
- class-1 F1 delta `>= +0.010`;
- class-1 recall delta `>= -0.020` and TP retention `>= 0.97`;
- combined class `0/2/4 -> 1` false positives fall by at least 20%;
- class `2 -> 1` errors fall by at least 20%.

Otherwise it is a neutral/negative reference. Absolute class-1 F1 `>= 0.72`
is reported as a practical milestone, not a promotion rule. A future current-
best or scientific superiority claim requires a separately locked confirmation
protocol and independent data/evidence.

## Required post-training audits before any test decision

On validation only: independent checkpoint reload and exact confusion/per-class
metrics; class-1 TP/FP/FN transitions; paired bootstrap and McNemar comparison
against B0; calibration; fixed brightness/contrast/occlusion robustness; XAI
attention/rollout plus perturbation checks; parameter/FLOP/VRAM/latency profile;
ONNX opset-17 parity and deployment feasibility. DINOv3-S/16 is a
teacher/reference, not the final mobile model; mobile deployment requires a
separate distilled student protocol.

## Authorized command

```powershell
$env:PYTHONPATH = 'D:\DataAI\AIEx\TRKH_pretrained'
$Python = 'D:\DataAI\.venv\Scripts\python.exe'
$Data = 'D:\DataAI\AIEx\newdataset\class_f\data.yaml'
$DevData = 'D:\DataAI\AIEx\TRKH_pretrained\configs\class_f_5class_dev.yaml'
$Dino = 'C:\Users\ADMIN\.cache\huggingface\hub\models--timm--vit_small_patch16_dinov3.lvd1689m\snapshots\3bf4720a82ec2066db88137180ff1f83a675cef0\model.safetensors'

& $Python -m trkh.recipes.pretrained_classf_b0 `
  --experiment b9-b2-reference-completion --mode full `
  --data $Data --train-data $DevData --dino-checkpoint $Dino `
  --output-dir D:\DataAI\AIEx\TRKH_pretrained\runs `
  --run-tag 20260801_b9_b2_ref_full_r1 `
  --batch-size 24 --num-workers 0 --eval-num-workers 0 `
  --amp-dtype bf16 --confirm-full
```
