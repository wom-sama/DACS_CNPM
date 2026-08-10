# TRKH 5-Class Visual-Contrast Attention Protocol (2026-07-14)

## Research question

Can NeurIPS 2025 Visual-Contrast Attention (VCA) preserve the complete mango
patch grid while learning positive-versus-distracting regional evidence that
improves class-1 precision without breaking true class-1 recall?

Primary sources:

- *Linear Differential Vision Transformer: Learning Visual Contrasts via
  Pairwise Differentials*, NeurIPS 2025:
  https://proceedings.neurips.cc/paper_files/paper/2025/file/5820ad65b1c27411417ae8b59433e580-Paper-Conference.pdf
- Official implementation at commit
  `3fb5ee100cfdd5c6e482cf1bade58a61dc24e3d1`:
  https://github.com/LeapLabTHU/LinearDiff
- Reviewed classification implementation SHA-256:
  `df6b87825cc8628ac3fc28a06a9fe17f83f9ecae0b83908cbb7b159da4f6c8ab`.

VCA is distinct from locally closed routes. DCAL/PWCA used a second natural
image as a train-only distractor and removed the branch at inference. VCA uses
two learned positive/negative streams derived from pooled queries within the
same image and is the inference attention itself. It is also not gated relative
position attention, local/window attention, a class-query head, a part-map
regularizer, or a post-hoc output rule.

## Locked evidence and constraints

- Dataset: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Current scratch reference:
  `runs/full_v8_yolof_randominit_30e_20260714_105524`.
- Scratch resolved-config SHA-256:
  `c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97`.
- Scratch history SHA-256:
  `a6eeddd7b1ef29222e78764bad854c9d670b0bc98de38362c01e2373359404e2`.
- Scratch best checkpoint SHA-256:
  `f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549`.
- Scratch launcher-argument SHA-256:
  `a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6`.
- Image size is `256`, patch grid is `16x16`, embedding is `256`, depth is
  `8`, heads are `8`, and the locked run has seven prefix tokens: one class,
  four register, one color-branch, and one edge-branch token.
- Raw data is immutable. Validation is allowed only for the matched smoke gate;
  test remains closed. No pretrained weights or external images are used.

## Locked architecture

- Replace MHSA in all eight transformer blocks with VCA. Keep the CNN stem,
  positional/register tokens, FFN, residual paths, heads, and training losses
  unchanged between matched variants.
- Use `64` visual-contrast tokens (`8x8`), the same one-quarter spatial ratio as
  the official `49` tokens on a `14x14` DeiT grid. No token-count sweep.
- Generate pooled tokens only from the `16x16` patch grid. Prefix tokens remain
  queries/keys/values but never enter spatial pooling or depthwise convolution.
- Use the paper's two stages: pooled positive/negative tokens attend to all
  keys/values; every original query then attends differentially to the compact
  contrast representation. Use
  `lambda_init(d)=0.8-0.6*exp(-0.3*d)`, official `N(0,0.1)` lambda vectors,
  per-head RMSNorm, and the patch-value depthwise `3x3` residual.
- Preserve TRKH `qkv` and `proj` parameter names. This makes state provenance
  explicit without claiming output equivalence to MHSA.
- Disable token pruning in both MHSA control and VCA candidate. The paper
  assumes a dense spatial query grid, and the hypothesis specifically tests
  whether retaining all fine-grained surface patches helps. A configuration
  that enables VCA and pruning together must fail closed.
- Return a full-grid XAI map only on explicit attention requests. Label it
  `vca_effective_positive`: multiply the signed Stage-II and Stage-I affinities,
  keep positive support, and normalize over keys. It is a derived VCA map, not
  standard softmax MHSA and not a pruned-QKV fallback.
- No VCA-specific class loss, distillation, margin, router, threshold, paired
  view, or augmentation change is allowed in the first matched run.

## Stage A: functional and resource preflight

Require all checks before image training:

1. Exact output shape for arbitrary prefix count plus dense patch grid.
2. Finite FP32 and CUDA AMP outputs and gradients for QKV, positive/negative
   embeddings, both lambda stages, depthwise convolution, RMSNorm, and output
   projection.
3. Full-grid derived attention has shape `[B,H,N,N]`, is finite,
   nonnegative, and sums to one over keys.
4. Sparse/mismatched patch grids, non-square contrast-token counts, invalid
   head dimensions, and VCA-plus-token-pruning configurations fail closed.
5. Deterministic same-seed initialization and checkpoint round trip.
6. All eight expected VCA blocks are present; added parameters are at most
   `350,000`; a batch-32 forward/backward stays below `7.75 GiB` peak VRAM.
7. One real `yolo_f/train` batch produces finite CE, nonzero gradients in every
   VCA component family, and no validation/test access.

## Stage B: matched short smoke

Run exactly two scratch variants with seed `42`:

- control: dense no-prune MHSA;
- candidate: dense no-prune VCA with the locked settings above.

Both use batch `32`, gradient accumulation `2`, `120` train batches per epoch,
`2` epochs, full `2606`-row validation, scheduler horizon `15`, warmup `1`,
the existing AdamW recipe with LR `5e-4`, minimum LR `1e-6`, weight decay
`0.05`, patience `3`, and the same balanced sampler/LDAM-focal/TRKH auxiliary
recipe. Attention dropout is fixed to zero in both variants because the
released VCA classifier does not apply its declared attention-dropout module.
Final test is skipped. Attention-view supervision is disabled for both variants
so differing attention semantics do not become a confound.

The candidate advances only if every gate passes:

- exact full validation support and finite probabilities;
- macro F1 no worse than control by more than `0.003`;
- class-1 F1 gain at least `0.010`;
- class-1 precision gain at least `0.020`;
- class-1 recall at least `control - 0.015`;
- total `0/2/4->1` false positives decrease by at least `5`;
- class-1 TP breaks are no more than FN rescues plus `3`;
- total corrections exceed harms versus control;
- newly created `3->2` harms are at most `5`;
- no nonfocus class F1 falls by more than `0.020`;
- candidate runtime is at most `1.35x` control and peak VRAM stays below the
  Stage-A ceiling.

After both smokes, independently reload both selected checkpoints and run
confusion/transition, calibration, robustness, architecture trace, native
attention, rollout, Grad-CAM, and changed-case XAI. A metric pass with invalid
or background-collapsed VCA maps does not advance.

## Advancement and promotion boundary

A complete Stage-B pass authorizes one matched continuation to five epochs,
not a full 30-epoch run. The five-epoch gate must retain the same directional
precision/recall criteria and show at least four favorable source-group
bootstrap folds before a full run is considered.

No VCA smoke or five-epoch probe can change the current-best command file.
Promotion still requires a single-checkpoint candidate to beat the existing
independent full-validation keeper gate, complete every audit, and follow the
unchanged one-time final-test policy. Failure forbids layer/token/LR/seed/
lambda/embedding-initialization sweeps under this formulation.

## Stage-A result (completed 2026-07-14)

- Evidence: `runs/audit_vca_stage_a_train_only_20260714/summary.json`, SHA-256
  `c904e7659ab5c584eb15b290e80cb99c9c66b0d01c021b52d80f8cf92ec30b5e`.
- All checks passed and `smoke_permission=true`; validation/test were not
  loaded. The audit used all `9215` train objects and one real batch of `32`.
- VCA added `285184` parameters (`7530774` versus `7245590`), instantiated all
  eight blocks, and passed exact same-seed and strict state round trips.
- FP32 and AMP losses/logits were finite. QKV, positive/negative embeddings,
  both lambda stages, both RMSNorms, depthwise value convolution, and projection
  all received finite nonzero gradients.
- AMP batch-32 forward/backward took `1.462 s` and peaked at `2.979 GiB`, below
  the locked `7.75 GiB` ceiling.
- All eight derived maps were `1x8x263x263`, finite and nonnegative, with
  maximum row-sum error `2.384186e-7` and provenance
  `vca_effective_positive`.

## Stage-B result and closure (completed 2026-07-15)

- Ran the one authorized matched `120b x 2e`, full-`2606` validation, no-test
  pair. Control is
  `runs/smoke_vca_dense_mhsa_control_120b_2e_20260714`; candidate is
  `runs/smoke_vca_dense_candidate_120b_2e_20260714`. The locked pair summary is
  `runs/audit_vca_matched_smoke_pair_20260714/comparison/summary.json`, SHA-256
  `bdd9b73c7da8d4efd974201b47d243d874ab586b1109126b952d59be591e7cbf`.
- Independent reload gave control/candidate macro F1
  `0.770209/0.760897`. Class-1 precision/recall/F1 changed from
  `0.383621/0.589404/0.464752` to
  `0.333333/0.529801/0.409207`. The macro, class-1 precision, recall, and F1
  deltas were `-0.009312/-0.050287/-0.059603/-0.055545`.
- The candidate changed `163` decisions: `64` corrections and `72` harms. It
  rescued `5` class-1 false negatives but broke `14` class-1 true positives;
  focus false positives increased `141 -> 158`, and new `3->2` harms were
  `10`. Eight locked checks failed. Five-epoch and full-train permission are
  both false.
- Full-validation forensics confirms the precision failure: class-1
  `TP/FP/FN` changed `89/143/62 -> 80/160/71`; `4->1` increased by `14` and
  `2->1` by `5`. Pair F1 deltas were negative for `0-1`, `1-2`, `2-3`, and
  `4-rest`.
- Robustness macro F1 deltas for clean, center occlusion, dim, bright, and low
  contrast were `-0.007369/-0.014145/-0.002757/+0.002388/-0.002085`. VCA won
  only one of five conditions and therefore provides no stable illumination or
  occlusion benefit.
- The trace is structurally healthy: all eight layers are present and finite,
  both contrast stages are nonzero, and lambda values are active. Failure is
  behavioral rather than an implementation collapse.
- Provenance-correct paired XAI used a deterministic 16-case validation cohort,
  SHA-256
  `38bbf41d01d5d8731f177454ea02eca2da27eac8bd06b47b7f54fb4a476463e5`.
  Candidate native attention moved foreground/background/border mass by
  `-0.067451/+0.067451/+0.023086`; ordinary rollout foreground fell
  `0.043898`. TP-break and harm Grad-CAM cases shifted toward hands, leaves,
  bright background, and borders. Some newly added false positives remained
  fruit-focused, so the failure also includes class selectivity, not only
  localization.
- Candidate `vca_effective_positive` is derived after the logit path and has no
  usable gradient path. All `16` candidate grad-rollout cases therefore used
  ordinary-rollout fallback (`8/8` layers per case), while control had `8/8`
  valid gradient layers. These `grad_rollout_*` deltas are invalid for
  attribution and are excluded from the hardened closure summary at
  `runs/audit_vca_matched_smoke_pair_20260714/postsmoke_audit_hardened/summary.json`,
  SHA-256
  `d398c76713cdc924d156eaa76dc57493ccc548686ace4de918992d79525aa5eb`.
- VCA is closed without layer/token/lambda/LR/seed/epoch/loss/initialization
  sweeps. Test was not used, raw data was not modified, and current-best
  full-train/export/video commands remain unchanged.

## Evidence retention

- Rejected smoke roots were compacted to
  `runs/evidence_vca_matched_smoke_rejected_20260715`. A CLI bug initially let
  four checkpoints through because custom image globs replaced mandatory
  binary exclusions. The repair verified every copied checkpoint against the
  source inventory SHA before removal, excluded `280` files / `393620720`
  bytes, retained `26` source payloads plus metadata, and produced payload
  manifest SHA-256
  `22e572a2146b0ccee56f836038a83d9a848c4ec0381e920e54aa6f63d3d7f998`.
- Superseded pre-provenance XAI roots were removed only after exact cohort and
  replacement checks. Cleanup manifest
  `runs/cleanup_manifest_20260715_vca_superseded_xai.json` records
  `32480814` logical bytes removed. Retention audit
  `runs/artifact_retention_audit_20260715_vca_closure` passed over `643` run
  directories with no blockers. Closure also passed compileall, full pytest
  `956/956`, five PowerShell parser checks, and current-best/VCA preflights.
