# TRKH 5-Class AugSelf Color-Adapter Closure

Date: 2026-07-16
Decision: reject the locked final-head AugSelf color adapter; Stage B denied

## Scope and provenance

- The immutable protocol is
  `docs/TRKH_5CLASS_AUGSELF_COLOR_ADAPTER_READINESS_PROTOCOL_20260716.md`,
  SHA-256
  `248f1ade029bf641cbaef0be3125c7390d4670ce0b78100d2d1867839f3203ae`.
- Primary evidence is Lee et al., *Improving Transferability of
  Representations via Augmentation-Aware Self-Supervision*, NeurIPS 2021,
  plus official repository commit
  `c131db66b5ade96af86774bc43a2cb797390bba5`. The paper, supplement, and
  four official source-file SHAs were precommitted. The official repository
  has no license, so the local implementation was written from the paper
  equations without copying source code.
- Only source-disjoint `yolo_f/train` fit/holdout rows `7372/1843` were used.
  The fit cohort contained all `432` class-1 rows plus the exact `186`
  restricted `0/2/4 -> 1` hard negatives. The 60 batches each contained
  `16/16` rows and the 1,920-row order SHA was
  `8dd3b2b74b5975b2ce430f85b833cd75e98e3f96ae518850737aef80df31fe5a`.
- Validation and test paths or predictions were never constructed. No raw
  dataset file, shared model/trainer code, or current-best command was changed.

## Structural and training audit

- The frozen keeper had exactly `7,245,590` parameters. The deployment
  adapter had exactly `32,768` parameters and used
  `h + W_up GELU(W_down LayerNorm(h))`, `256 -> 64 -> 256`, without bias or
  LayerNorm affine parameters. Zero-initialized `W_up` made initial raw,
  control, and candidate logits bit-exact.
- The non-deployed three-layer color predictor had exactly `528,388`
  parameters. Candidate and control model/predictor states were bit-exact at
  initialization.
- Private SHA-256 color RNG reproduced every active flag, factor, operation
  order, centered parameter, tensor hash, and reverse target. Candidate and
  control used bit-exact view records. Independent replay matched all `1,920`
  train pairs and all `145` focused holdout pairs.
- The isolated auxiliary-only gradient audit gave a finite nonzero candidate
  adapter gradient, exact-zero detached-control adapter gradient, and
  bit-exact predictor gradients with maximum error `0`.
- Both variants executed exactly 60 steps. Every trainable parameter saw a
  finite nonzero gradient and changed; every frozen keeper tensor remained
  bit-exact. Control CE/color loss changed
  `1.540536/0.369551 -> 1.046499/0.039928`; candidate changed
  `1.540536/0.369551 -> 1.052421/0.039424`.

## Focused representation result

- The fixed focused holdout contained `109` class-1 rows and `36` restricted
  raw false positives. Zero-predictor color MSE was `0.137060`.
- Detached control color MSE was `0.047476`, explained variance `0.653614`.
  Candidate color MSE was `0.050474`, explained variance `0.631740`.
  Candidate was therefore `1.06315x` the control MSE and failed the required
  `<=0.995x` candidate/control gate despite beating the zero predictor.
- Two-view candidate accuracy was `0.637931` versus control `0.641379`.
  Candidate class-1 F1 was `0.771930` versus control `0.773626`; candidate
  made five different decisions with `2` corrections and `3` harms.
- The auxiliary representation gradient did not improve held-out color
  prediction or augmented classification beyond training the predictor and CE
  adapter alone.

## Locked train-holdout behavior

| Condition | Variant | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | Restricted FP |
|---|---|---:|---:|---:|---:|---:|
| clean | raw keeper | 0.949323 | 0.748252 | 0.981651 | 0.849206 | 36 |
| clean | detached control | 0.327441 | 0.082017 | 1.000000 | 0.151599 | 718 |
| clean | AugSelf candidate | 0.308590 | 0.080681 | 1.000000 | 0.149315 | 740 |
| dim | raw keeper | 0.825093 | 0.535354 | 0.486239 | 0.509615 | 46 |
| dim | detached control | 0.254802 | 0.064698 | 0.844037 | 0.120183 | 831 |
| dim | AugSelf candidate | 0.247675 | 0.065217 | 0.853211 | 0.121173 | 834 |
| bright | raw keeper | 0.872779 | 0.565517 | 0.752294 | 0.645669 | 62 |
| bright | detached control | 0.267317 | 0.072085 | 0.935780 | 0.133858 | 817 |
| bright | AugSelf candidate | 0.252577 | 0.071429 | 0.935780 | 0.132726 | 828 |
| low contrast | raw keeper | 0.852525 | 0.538462 | 0.577982 | 0.557522 | 53 |
| low contrast | detached control | 0.258106 | 0.059841 | 0.761468 | 0.110963 | 806 |
| low contrast | AugSelf candidate | 0.247284 | 0.060043 | 0.770642 | 0.111406 | 817 |

- On clean holdout, candidate changed macro/class-1 F1 by
  `-0.640733/-0.699891`, class-1 precision by `-0.667571`, and restricted FP
  `36 -> 740`. Its `1,217` decision events contained only `2` corrections and
  `1,215` harms. Recall reached `1.0` by predicting class 1 for broad regions
  of classes 0, 2, and 4, not by learning conservative class-1 evidence.
- Detached control already collapsed to restricted FP `718`, proving that CE
  adaptation on the narrow balanced boundary cohort was unsafe. Candidate
  was still worse than control: macro/class-1 F1
  `-0.018852/-0.002284`, 31 harms with zero corrections, and 22 additional
  restricted FP.
- Dim/bright/low-contrast candidate restricted FP rose to `834/828/817`.
  Candidate macro F1 was negative versus both raw and control in all three
  conditions. The apparent recall gains were support expansion and fail the
  agricultural precision requirement.

## XAI and deployment audit

- Required XAI coverage selected all `708` clean TP-rescue/break and
  restricted-FP create/remove rows, producing 59 contact-sheet pages. The
  categories contained `704` restricted-FP creations versus raw, 24 versus
  control, only two removals versus control, and no clean TP break.
- Visual review of pages `1`, `15`, `30`, `45`, and `59` covered yellow/green
  healthy fruit, damaged/dark fruit, partial fruit, obstacles, and wide
  context. Raw, control, and candidate Grad-CAM remained fruit-centered; the
  collapse was not caused by a new background-localization failure.
- Quantitative replay over all 708 maps gave candidate/control Grad-CAM mean
  correlation `0.998049` and mean absolute difference `0.005852`. Candidate/
  raw correlation was `0.753266`. Color-loss occlusion saliency was finite and
  nonzero but broad, with mean `99.55%` of pixels above `1e-6` after map
  normalization. The auxiliary branch did not create a selective spatial
  mechanism beyond the CE control.
- Deployment-only candidate runtime was `0.064525 s` versus raw `0.063174 s`,
  ratio `1.02139x`; peak VRAM was `0.396273 GiB`. Both resource gates passed.
- Isolated/full static-batch-one ONNX exports passed finite and argmax checks
  with maximum errors `5.960e-8` and `3.576e-7`. The auxiliary predictor was
  not exported. Deployability cannot override the representation and behavior
  rejection.

## Replay, artifacts, and engineering closure

- Independent CSV replay reproduced all `4 x 1843 = 7372` rows, comparisons,
  confusion-derived metrics, and transitions exactly. Training/focused color
  view replay and XAI selection replay were also exact.
- Evidence is under
  `runs/audit_augself_color_adapter_readiness_20260716`. Its 69 manifest
  payloads total `759,894,298` bytes. Summary/prediction/artifact-manifest/XAI-
  manifest SHAs are respectively
  `b928b2bf417df00483ada9af90af586f054d5b28931c32e73fd2a2613cd4740a`,
  `dc8da86ccc29723204ab87e747ff7e29b9b8906a374d005f6ca61a9e37deb1a3`,
  `a63a333df379efe0a9d501a28b81c95411bbd85ac3be9069f49feb2ac7e45f0e`,
  and
  `99f8d53876c2ff4bf7b34123e9f3ef563992ac046528a8dbba0cee4101a99541`.
- No checkpoint, TensorRT engine, trainable manifest, validation/test output,
  or raw-data write exists. Current-best command tracking remains three
  revisions and two updates after the initial revision.
- Compileall passed; focused tests passed `6/6`; full pytest passed
  `1188/1188` with 257 warnings in `75.87 s`. PowerShell parse and real
  preflight passed.
- Read-only retention audit
  `runs/artifact_retention_audit_augself_color_adapter_closure_20260716`
  passed over `703` run directories with `blockers=[]`, deleted nothing, and
  reported `71.473 GiB` free. Summary SHA-256 is
  `ff2ee94192e579a5c84e9350d7a66404fdc83a61ac7d165c2519887a9aff64d9`.
- Keeper, scratch complement, command packet, and command-history hashes remain
  exact.

## Decision and no-repeat boundary

Stage B, validation, test, probe, full train, shared-trainer integration, and
current-best command promotion are denied. Do not sweep auxiliary weight,
jitter probability/range/order, adapter width/depth, predictor width/depth,
detach location, optimizer, learning rate, weight decay, steps, balanced
cohort, fold, or seed. Do not add crop, blur, flip, or solarization to this
final-head adapter.

This is a rejection of the exact final-head AugSelf color-adapter transfer and
nearby parameter changes, not a universal rejection of augmentation-aware
self-supervision. Reopening AugSelf requires a genuinely different sourced
insertion point or supervision target and a new prospectively locked protocol.
The next route must protect the natural multiclass geometry before any
boundary-focused update and must not obtain recall by broadening class-1
support.
