# B5: partial-order semantic factorization on Q/V-LoRA

## Locked hypothesis and no-repeat boundary

B5 starts from the exact B2 selected EMA checkpoint and repeats the complete
B4 Q/V-LoRA recipe. It does **not** resume a trained B4 checkpoint. This keeps
B4 and B5 matched at initialization and makes the train-only loss the sole B5
delta.

The earlier rejected semantic-attribute experiment trained a scratch model and
readout with weight `0.02` and generic `maturity+transport+quality` groups. B5
is materially different but remains a one-shot exploratory probe:

- the DINOv3 classifier, normalization, and base backbone stay frozen;
- only the same `24,576` Q/V-LoRA parameters as B4 are optimized;
- the parameter-free loss must therefore move the representation through the
  fixed classifier rather than learn a replacement readout;
- the three tasks directly bracket class 1:
  `lower:0|1,2,3;upper:0,1|2,3;rotten:0,1,2,3|4`;
- `lower`, `upper`, and `rotten` respectively target `0->1`, `2->1`, and
  `4->1` while protecting the reverse class-1 transitions;
- no cumulative head, residual logit adjustment, ordinal/EMD loss, pAUC,
  margin, router, or post-hoc threshold is enabled.

The POSF weight is locked to `0.006`. No weight, grouping, seed, sampler, loss,
or decision-threshold sweep is permitted after validation is observed.

## Bounded contract

- Modes: engineering preflight, 4-batch smoke, or 5-epoch/120-batch probe only.
- Probe seed `42`, batch `24`, accumulation `2`, full validation, no test.
- Resume only B2 `best.pt`, SHA-256
  `4d3205e7029fa25c313ac622f3ea1abbe0bbf4103d804ee9ac87f527759352a6`.
- Dataset image-tree SHA-256
  `70a1b7d2b4c6f80e28fe3f0f714f1ba3e8ab654a90ce50b1e9cae8e4dac4a503`.
- DINOv3/B2 provenance, tempered `p=0.5` sampler, LDAM-Focal, augmentation,
  EMA, optimizer, scheduler, LoRA rank/layers/alpha/dropout, trainable prefixes,
  checkpoint selection, and deployment contract are inherited unchanged from
  B4.
- `train_semantic_attribute_tasks` must equal `3`; its loss must be finite and
  positive; total trainable parameters must remain exactly `24,576`.
- POSF adds zero model parameters and no inference graph operation. B4 merge,
  materialization, ONNX parity, latency, and package-size gates remain binding.

## Predeclared promotion gate

B5 must pass every B4 absolute gate: accuracy `>=0.8879`, macro-F1 `>=0.8495`,
class-1 P/R/F1 `>=0.610/0.708/0.665`, TP `>=112`, restricted
`0/2/4->1 <=69`, and `2->1 <=39`.

Against the matched B4 selected checkpoint it must also satisfy all:

- accuracy delta `>=-0.0020`, macro-F1 delta `>=-0.0015`;
- class-1 precision/F1 deltas `>=+0.010/+0.005`;
- class-1 recall delta `>=-0.012` and TP delta `>=-2`;
- at least `5` fewer restricted false positives and `3` fewer `2->1` errors;
- neither `0->1` nor `4->1` may increase;
- full validation support `2479`, zero non-finite steps, and no test inference.

Failure closes this exact B5 recipe and does not authorize a nearby POSF or
hierarchy sweep. A pass licenses the existing audits/full-train decision; it is
not itself confirmatory scientific evidence.

## Commands

```powershell
$env:PYTHONPATH = 'D:\DataAI\AIEx\TRKH_pretrained'
$Python = 'D:\DataAI\.venv\Scripts\python.exe'
$Data = 'D:\DataAI\AIEx\newdataset\class_f\data.yaml'
$DevData = 'D:\DataAI\AIEx\TRKH_pretrained\configs\class_f_5class_dev.yaml'
$Dino = 'C:\Users\ADMIN\.cache\huggingface\hub\models--timm--vit_small_patch16_dinov3.lvd1689m\snapshots\3bf4720a82ec2066db88137180ff1f83a675cef0\model.safetensors'
$B2 = 'D:\DataAI\AIEx\TRKH_pretrained\runs\pretrained_dinov3_classf_tempered_p05_probe_20260731_local_b2_tempered_p05\checkpoints\best.pt'
$Attestation = 'D:\DataAI\AIEx\TRKH_pretrained\runs\canonical_classf_contract_20260731.json'

& $Python -m trkh.recipes.pretrained_classf_b5 --mode preflight `
  --data $Data --train-data $DevData --dino-checkpoint $Dino `
  --b2-checkpoint $B2 --canonical-attestation $Attestation `
  --output-dir runs --run-tag 20260731_preflight

& $Python -m trkh.recipes.pretrained_classf_b5 --mode smoke `
  --data $Data --train-data $DevData --dino-checkpoint $Dino `
  --b2-checkpoint $B2 --canonical-attestation $Attestation `
  --output-dir runs --run-tag 20260731_smoke

& $Python -m trkh.recipes.pretrained_classf_b5 --mode probe `
  --data $Data --train-data $DevData --dino-checkpoint $Dino `
  --b2-checkpoint $B2 --canonical-attestation $Attestation `
  --output-dir runs --run-tag 20260731_local_b5_posf
```

No B5 smoke or probe had been launched when this protocol was created.
