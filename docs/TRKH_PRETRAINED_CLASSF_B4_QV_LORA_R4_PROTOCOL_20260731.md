# B4: DINOv3 Q/V-LoRA rank 4 stability probe

## Hypothesis and single model delta

B2 tempered sampling improved the current canonical `class_f` result to
accuracy/macro-F1 `0.889875/0.851462` and class-1 P/R/F1
`0.592784/0.727848/0.653409`, but it still has `43` class-2-to-class-1 errors.
Train-only B3b then showed that raw patch moments improve pairwise OOF AUROC in
`13/15` folds, increase aggregate class-1 TP `1365 -> 1376`, and reduce rival
FP `262 -> 245`, although that static readout correctly failed its magnitude
gate.

B4 keeps the B2 best EMA checkpoint, tempered sampling, LDAM-Focal objective,
augmentation, seed, batch contract, and validation lock. Its sole model delta
is a parameter-efficient update of the Q and V projections in DINOv3 blocks
`8,9,10,11`:

- LoRA rank `4`, alpha `8`, scale `2`, dropout `0.05`;
- base DINO weights frozen;
- trainable: only the eight Q/V low-rank branches;
- exact LoRA parameters: `24,576`;
- exact total trainable parameters: `24,576`;
- zero-up initialization, so the resumed B2 logits are bit-exact before the
  first optimizer step;
- Q/V deltas are merged and the wrappers materialized as plain fused QKV
  linear layers before deployment, adding zero deployed parameters/FLOPs and
  no LoRA tensors to the exported state.

This route is distinct from the rejected static B3/B3b heads: it changes how
the pretrained representation aggregates local evidence rather than adding a
post-hoc boundary rule.

## Locked probe

- Resume source: B2 `best.pt`, selected/EMA weights, epoch 4, SHA-256
  `4d3205e7029fa25c313ac622f3ea1abbe0bbf4103d804ee9ac87f527759352a6`.
- Canonical dataset image-tree SHA-256:
  `70a1b7d2b4c6f80e28fe3f0f714f1ba3e8ab654a90ce50b1e9cae8e4dac4a503`.
- B2 provenance: source commit `f1d79def090e347c0586efae4003b2579d368a28`,
  source-tree SHA-256
  `61b860a5549ac172d06c64ee1c5f9fe035cc9853074312f2fe5a2de80390e747`,
  recipe SHA-256
  `2f53fd5e3f4fc154a86a72b1985328c59fffd9a5d376b14e89f9c70ad8311aa6`.
- Five epochs, 120 train batches per epoch, full validation, seed 42.
- Batch 24, accumulation 2, effective batch 48.
- LR `2e-4`, backbone LR scale `1`, cosine horizon 5, one warmup epoch
  starting at `0.1` of the target LR, minimum LR `1e-6`.
- AdamW weight decay `0.05`, EMA `0.995`.
- No rank/alpha/layer/LR/sampler/loss/threshold sweep.
- No test inference.
- Checkpoint selection remains the inherited
  `fair_macro_f1_min_class_gap_penalty` rule; gates are applied once to that
  selected epoch, never used to choose a different epoch.

The probe is eligible for full training only when every check passes:

- no non-finite optimizer step;
- accuracy `>= 0.8879`;
- macro-F1 `>= 0.8495`;
- class-1 precision `>= 0.61`;
- class-1 recall `>= 0.708` and TP `>= 112`;
- class-1 F1 `>= 0.665`;
- restricted `0/2/4 -> 1 <= 69`;
- class `2 -> 1 <= 39`;
- initial B2/LoRA maximum logit error is zero;
- trained unmerged/materialized FP32 maximum logit error `<= 1e-5` with
  validation argmax parity `2479/2479`;
- materialized ONNX has no LoRA branch/initializer, latency `<= 1.01x` B2,
  and package bytes `<= 1.001x` B2.
- FP32 ONNX Runtime is finite and matches materialized PyTorch on all 2,479
  validation samples with maximum logit error `<= 1e-5`, exact argmax and
  exact confusion/metric parity.

B2 and B4 deployment latency/bytes must be produced by the same exporter,
opset, provider, resolution, batch size, warm-up, timed iterations and power
mode. The B2 artifact is generated first because no matched B2 deployment
baseline existed when this protocol was locked.

The class-1 F1 `>= 0.665` gate licenses a full run; the research teacher
milestone remains class-1 F1 `>= 0.72`.

A failing probe closes this exact Q/V-LoRA recipe. It does not authorize a
rank, layer, LR, sampling, or validation-threshold sweep.

Because B2, B3/B3b and B4 all reuse the development validation split, B4 is
an exploratory research candidate, not confirmatory evidence of scientific
superiority. Confirmation is deferred to the still-sealed test set or a new
prospective holdout.

B4 is an accuracy-oriented research teacher. Matching B2 after LoRA
materialization proves zero *incremental* adapter overhead; it does not make
the DINOv3 teacher itself a mobile model. The final mobile deliverable still
requires a distilled/quantized student and absolute target-device p50/p95
latency, peak-RAM, package-size and thermal/energy gates.

## Trainable prefixes

```text
blocks.8.attn.qkv.q_lora,blocks.8.attn.qkv.v_lora,
blocks.9.attn.qkv.q_lora,blocks.9.attn.qkv.v_lora,
blocks.10.attn.qkv.q_lora,blocks.10.attn.qkv.v_lora,
blocks.11.attn.qkv.q_lora,blocks.11.attn.qkv.v_lora
```

## Reproducible launcher

```powershell
$env:PYTHONPATH = 'D:\DataAI\AIEx\TRKH_pretrained'
$Python = 'D:\DataAI\.venv\Scripts\python.exe'
$Data = 'D:\DataAI\AIEx\newdataset\class_f\data.yaml'
$DevData = 'D:\DataAI\AIEx\TRKH_pretrained\configs\class_f_5class_dev.yaml'
$Dino = 'C:\Users\ADMIN\.cache\huggingface\hub\models--timm--vit_small_patch16_dinov3.lvd1689m\snapshots\3bf4720a82ec2066db88137180ff1f83a675cef0\model.safetensors'
$B2 = 'D:\DataAI\AIEx\TRKH_pretrained\runs\pretrained_dinov3_classf_tempered_p05_probe_20260731_local_b2_tempered_p05\checkpoints\best.pt'
$Attestation = 'D:\DataAI\AIEx\TRKH_pretrained\runs\canonical_classf_contract_20260731.json'
```

Preflight only:

```powershell
& $Python -m trkh.recipes.pretrained_classf_b4 --mode preflight `
  --data $Data --train-data $DevData --dino-checkpoint $Dino `
  --b2-checkpoint $B2 --canonical-attestation $Attestation `
  --output-dir runs --run-tag 20260731_preflight
```

Bounded integration smoke:

```powershell
& $Python -m trkh.recipes.pretrained_classf_b4 --mode smoke `
  --data $Data --train-data $DevData --dino-checkpoint $Dino `
  --b2-checkpoint $B2 --canonical-attestation $Attestation `
  --output-dir runs --run-tag 20260731_smoke `
  --batch-size 24 --num-workers 4 --eval-num-workers 2
```

Locked exploratory probe:

```powershell
& $Python -m trkh.recipes.pretrained_classf_b4 --mode probe `
  --data $Data --train-data $DevData --dino-checkpoint $Dino `
  --b2-checkpoint $B2 --canonical-attestation $Attestation `
  --output-dir runs --run-tag 20260731_local_b4_qv_lora_r4 `
  --batch-size 24 --num-workers 4 --eval-num-workers 2
```
