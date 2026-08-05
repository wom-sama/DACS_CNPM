# TRKH pretrained class_f: refocus decision (updated 2026-08-05)

## Decision

- Keep `class_f` immutable, accept its labels as the user-authoritative real-world
  ground truth, and keep the test split sealed. Similar-looking cross-class
  samples are a representation challenge, not a relabelling task.
- Close PRMR R1 as **diagnostic-positive but not promotable**.
- Do not reuse the two-backbone A0 hybrid.
- Close class-conditioned group exposure and PR-SPR-V3 after their matched failures.
- Preserve B9 pure DINO as the current presentation/teacher reference. XCNorm A0/A1 and B11-CGAER are closed by matched TRAIN-only screens. Do not launch another late pooled-feature adapter, uncertainty router, or width/loss/LR sweep on this representation. Mobile promotion still requires a later locked distillation and device audit.
- Close the single B14 iFormer-S pooled-320 screen after a valid TRAIN-only failure. Its mobile qualification passed, but its released representation did not retain the DINO class-1 boundary/AUROC signal; no iFormer variant, alternate stage/readout/input, fine-tuning or backup architecture is authorized by this result.
- Close B15 at its synthetic/mobile preflight. The exact parameter-free aligned prefix-to-patch pool exceeded the locked p95 overhead, so no TRAIN descriptor or metric was opened. This closes the operator under the mobile budget, not its unmeasured representation quality.
- Close exact B16 SwiftSurface-XS at its label-free deployment preflight. The candidate's relative overhead passed, but the locked absolute host-latency gate failed for the stock backbone itself and QDQ INT8 was slower than FP32. No TRAIN/data/metric evidence was opened, so this is an engineering-protocol closure rather than evidence that the spatial hypothesis is weak.
- Close exact B18 after its single TRAIN authorization on commit `9bb1232`. Its preflight passed, but fold 0 stopped after all eight epochs because the LR evidence gate compared a one-ULP warmup endpoint with `==`; no fold state, OOF metric, validation or test result was produced. A B19 protocol-only successor may change only bit-exact LR endpoints and real-fold scheduler tests.
- Close exact B19 after its valid single TRAIN-only OOF screen on commit `b39f4d0`. The scheduler correction, all 15 states, metric barrier and integrity contracts passed, but the active spatial factor did not beat its matched mean-surface control and widened `2 -> 1`; validation and test remain sealed.
- Keep B20 `NOT_OPENED`. Haar plus log/opponent chromaticity is not a new
  observable: it is a prohibited kernel/magnitude neighbour of two already
  failed families. This closure no longer depends on a label review.
- Close only the exact five-epoch joint-finetune recipe
  `B21-DINO-ConvPass-A0`. Its spatial path improves accuracy and macro-F1 but
  reduces class-1 recall/F1 and increases `2 -> 1`; validation and test remain
  sealed. Because the run ended near peak LR on a 30-epoch horizon, it is not
  evidence that ConvPass or pretrained hybrids reached their limit.
- Close exact `B22`: a scalar class-1 bias almost restores the direct B21
  boundary, while logits/features over-expand class 1. This rejects another
  post-pooling repair, not a pre-pooling specialist initialized from B9.
- Close exact B23 after a valid inherited-and-frozen B9 screen: despite a full
  six-epoch cosine schedule, uniform multi-depth ConvPass reduces macro/C1 F1
  and increases `2 -> 1`. This closes uniform all-patch local mixing, not the
  hybrid ceiling.
- Close B24 as signal-positive but not promotable: DINO+ConvNeXt improves the
  point estimates and false-positive boundary, but its 1,152-dimensional
  readout hits the fixed solver limit in three folds and fails the unchanged
  bootstrap harm guard.
- Close B25 after the solver-only resolution: all folds converge, no argmax
  changes, and the unchanged bootstrap guard still fails. ConvNeXt remains
  useful mechanism evidence but is not authorized as a teacher.
- Close B26 after its one locked run. It preserved B9 numerically and trained
  only the intended query decoder, but its attention stayed nearly uniform and
  moved only three validation decisions. The exact soft-query decoder therefore
  repeats global token mixing rather than selecting new regional evidence.
- Close B27/B28: decision-margin degradation fixes the selector diagnostics but
  the raw-DINO subpatch descriptor still fails its information gates.
- Close the exact conventional-KL B29 pilot as mechanism-positive but not
  promotable. On the fixed TRAIN design fold, the canonical four-stage
  EfficientViM residual transfers enough ConvNeXt-teacher signal to improve
  accuracy/macro/C1 F1 over its identical control, but trades two C1 true
  positives for a large false-positive reduction. This supports the hybrid
  family and isolates calibration/knowledge transfer as the next problem; it
  does not authorize validation, test, full training, or a post-hoc bias claim.
- Lock one B30 fold-1 loss ablation. It keeps the B29 teacher, model,
  initialization, sampler and schedule; only coupled KL versus trust-margin
  residual distillation changes. No architecture/backbone sweep is open.
- Close B30 TMORD after its valid fold-1 failure. Its coupled-KL control is new
  positive evidence: it improves every primary metric, retains more C1 TP and
  removes FP even though the fusion teacher's own C1 metric is worse. Complete
  only folds 2--4 of the coupled-KL OOF; do not tune either loss.

### Objection and decision state

Update rows in place; do not append chronology. States move `OPEN -> LOCKED -> CLOSED`; a closed row reopens only on its stated new evidence. `GUARD` is permanent.

| ID | State | Objection / exact scope | Decision and evidence anchor | Exit or reopen condition |
|---|---|---|---|---|
| `DATA-01` | `CLOSED_BY_SCOPE` | Similar-looking cross-class samples imply that canonical labels should be reviewed or changed. | Rejected by project scope. The `class_f` labels are authoritative and no loader/class-map bug was found. pHash/component evidence is retained only to build source-safe folds and describe hard visual boundaries; it is not evidence against the labels. | Permanent for this project unless the user explicitly supplies a replacement canonical dataset. No human-review, relabel or automatic-edit gate. |
| `HYB-CAUSE-01` | `CLOSED` | V2/V3 lose only because class_f is noisy. | Rejected for V2/V3: matched arms share the same data, while V2 actively widens `2->1` and V3 loses recall across stable, relabelled, mixed and near cohorts. This does not claim every hybrid must fail. | Reopen only for a materially different matched hybrid whose branch adds measured pre-pooling information. |
| `V2-01` | `CLOSED_FAIL` | More CNN depth or tuning can rescue V2. | Branch is active, not collapsed; duplicated absolute colour plus a free `64->384` projection overrides DINO semantics (`C1 F1 0.633609`, `2->1=52` versus B2 `38`). | New constrained evidence source, capacity-matched control and pre-pooling integration; no same-recipe tuning. |
| `V3-01` | `CLOSED_FAIL` | The classifier direction was wrong. | Direction finite differences pass; the zero-init 160-parameter branch is starved by a moving 21.6M backbone (`p95=7.37e-5`, zero argmax correction). | A successor must inherit/freeze a verified teacher and prove non-collapse before metrics. |
| `B12-01` | `CLOSED_FAIL` | A larger SSM/attention block over DINO depth tokens will recover missing signal. | Candidate gain versus latest is tiny/inconclusive and it is worse than the native B9 margin with CI below zero. | Independently measured missing information, not more capacity on the same tokens. |
| `B13-01` | `CLOSED_FAIL` | EfficientViM-M1 final feature is already a superior mobile transfer representation. | Rejected TRAIN-only: versus raw DINO, M1 final-320 loses mean pair AUROC `-0.014154`, macro-F1 `-0.032695` and C1 F1 `-0.042563`; every bootstrap interval is below zero. Mobile eligibility passes, but it cannot authorize fine-tuning. | No same-evidence reopen: the locked B13 stop rule forbids M2/M3/M4, four-stage fusion and readout replacement after seeing the result. Reopen only with a newly acquired, independently sealed dataset/holdout fixed before model choice, not another interface on current `class_f`. |
| `B14-01` | `CLOSED_FAIL` | A compact local/global backbone can be selected by trying several pretrained families until one wins. | Rejected as a process and then tested once as preregistered. Official iFormer-S v0.9 pooled-320 passed source, strict-load, ONNX and mobile gates (`0.196x` median, `0.211x` p95 DINO latency), but lost macro-F1 `-0.012474`, C1 F1 `-0.024587` and mean pair AUROC `-0.005353`; pair-AUROC and C1 non-inferiority failed. Summary SHA256 `9ad53ca882ceec174cbe19aca9dfd8e268c1c4380ae8172ee3c745d0f166ba90`. | No same-evidence reopen: no iFormer variants, stage/readout/input/fine-tune sweeps or backup architecture. Reopen this exact route only on a newly acquired independently sealed dataset/holdout fixed before model/interface choice. |
| `B15-01` | `CLOSED_FAIL` | Prefix tokens can select more useful final DINO patches at negligible mobile cost. | The committed synthetic preflight on `53274c1` preserved zero trainable parameters, ONNX parity and standard domains, but aligned-versus-uniform ORT CPU overhead was `+4.579%` median and `+11.591%` p95. The p95 gate was at most `+5%`; preflight SHA256 `4bd2f3f99c6b9b35c2f9d2e25e47c6224fdacfd5fa5fbe186631d7e2977f4a1c`. No TRAIN/validation/test descriptor or metric was read. | Do not retry, relax latency, or inspect B15 representation metrics. A successor must absorb selection into a separately preregistered mobile student/backbone rather than append this exact final-token operator. |
| `B16-01` | `CLOSED_FAIL` | A hybrid should be rescued by another unconstrained branch/backbone sweep, or raw pretraining alone proves the design. | Rejected as a sweep, then tested once as locked B16. Commit `f203a3c` passed strict weights, synthetic mechanism, CUDA fit, ONNX, size and full QDQ coverage, but failed the absolute 12 ms host p95 gate: FP32 stock/control/candidate `12.928/13.387/13.327` ms and INT8 QDQ `31.450/31.045/31.553` ms. Candidate p95 ratios `1.031x/1.003x` did pass the `1.10x` relative gate. No dataset, TRAIN, validation, test or architecture metric was opened. Diagnostic SHA256 `09942f6944ca96a0434a521e3d95da3109b212d2116c3f8ead1dba22d91a2956`. | Exact B16 remains closed; do not relax 12 ms after seeing it or infer representation quality. A successor requires a prospectively locked backend/device contract that first qualifies its stock baseline, while retaining the same causal controls if the spatial hypothesis is tested. |
| `B17-01` | `CLOSED_FAIL` | A lightweight hybrid should add measured local-direction information where its auxiliary objective can directly train the new mechanism, rather than append another free late branch. | Architecture/code review passed, but the first formal run on `954655c` failed at `strict_load`: CUDA had already been initialized by the device contract, while `torch.manual_seed(20260805)` ran inside `fork_rng(devices=[])`, which restored CPU RNG but not CUDA RNG. A direct probe reproduced `CPU preserved=true, CUDA preserved=false`; `fork_rng(devices=[0])` preserved both. Failure artifact SHA256 is `ad1ded2a42808cf0fd39843b07efe0f197471b8e7ff90fe449a486ac2e672848`. No control/candidate, deployment metric, dataset, TRAIN, validation or test was opened. | Exact B17 never reopens. A separately committed B18 may change only full-device RNG isolation and target-matched host packaging/measurement; architecture, causal controls, data protocol and statistical gates remain fixed. |
| `B18-01` | `CLOSED_FAIL` | Passing synthetic scheduler tests proves the exact real-fold LR evidence gate. | Rejected. B18 preflight passed with SHA256 `b32697dd4911f38ae2fb3ab42bc7aeb635ec92e9ad21a288aa02c763ad73dfda`, then the one-use TRAIN run stopped after fold-0 training because `3e-5 * 201 / 201` produced `2.9999999999999997e-5`; exact dictionary equality rejected the one-ULP difference (`3.388e-21`). Failure SHA256 is `75370288a392497b045a977924b0fdf8c3dc168a3849863ca69024c75afbc8ed`; validation/test are false and no fold state/OOF metric exists. | Exact B18 never reruns. B19 may only return literal `peak_lr/MIN_LR` at the two endpoints and test all locked update counts `202/206/212/207/210`; no architecture, data, loss, optimizer, schedule shape or gate change. |
| `B19-01` | `CLOSED_FAIL` | An active, directly supervised, foldable spatial factor is sufficient to improve the class-1 boundary. | Rejected by a valid TRAIN-only screen. Candidate C1 F1 was `0.632939` versus control `0.630553`, delta `+0.002386 [-0.009669,+0.014988]`; pair-2 AUROC delta was `-0.000533`, `2->1` rose `80->84`, and restricted FP rose `131->135`. Active-minus-delta-off C1 F1 was real (`+0.043440 [+0.019285,+0.069964]`), so this is target/quality failure, not branch collapse. | Exact B19 never reruns or retunes. A successor must change the target and placement equation, use a matched primary backbone/control, and pass a bounded TRAIN-only screen. |
| `OBS-01` | `CLOSED_FAIL` | Combining Haar with log/opponent chromaticity is a materially new interaction and may be probed immediately. | Rejected. Aligned Haar lost C1 F1 `-0.061779` to its control and added `119` restricted FP; CCR reached AUROC `0.535520`, changed C1 F1 `-0.026845`, and beat both controls in only `2/5` folds. For fixed linear operators, `H(A log RGB) = A(H log RGB)`, so Haar-on-log-opponent is not an interaction. The CCR stop rule also explicitly forbids post-failure signed/magnitude and derivative-kernel sweeps. | Closed independently of label status. Reopen only for a mathematically different observable supported by a new measured precondition. |
| `B21-01` | `CLOSED_FAIL` | A hybrid cannot beat B9 because the previous local branches already exhausted spatial evidence. | The premise is too broad, but exact B21 fails its class-1 gate. On the same fold/seed/state, spatial versus direct changes accuracy `0.867800 -> 0.880417` and macro-F1 `0.839621 -> 0.846775`, while C1 F1 falls `0.691892 -> 0.674033`, C1 TP `64 -> 61`, and `2 -> 1` rises `15 -> 17`. Adapter p95 residual is `0.032020`; gradients, updates and strict reload are valid. B21 also updated the full 21.6M primary model and stopped after five epochs with LR still `1.458e-4` on a 30-epoch horizon. | Do not rerun the exact short joint-finetune recipe or open its validation/test. A successor may strict-load a verified supervised reference, freeze/preserve its primary path, and use a screen-aligned schedule; that is a different initialization and optimization claim. |
| `B22-01` | `CLOSED_FAIL` | B21's useful spatial representation may be hidden by a single long-tail classifier, but another signed residual could again buy precision by losing C1 recall. | Frozen spatial B21 has macro/C1 F1 `0.846775/0.674033`. Non-negative bias-only reaches `0.850331/0.691489`; logits-only reaches `0.797839/0.541935`; the 390-parameter feature/logit expert reaches `0.816621/0.584980`, with C1 FP `83` and `2->1=41`. The feature arm fits its binary objective (`0.465207 -> 0.226587`) but expands rather than separates the C1 region. | Exact B22 never retunes. Another post-pooling head on the same final representation is closed. Reopen only with new pre-pooling/local evidence and a matched ordinary-backbone control. |
| `B23-01` | `CLOSED_FAIL` | B21 may have failed because it restarted raw DINO, moved the full primary model at the same time as a zero-init adapter, and stopped before meaningful cosine decay. | B23 removed all three confounds: it strict-loaded B9, froze all B9/head parameters, trained only `170,880` adapters and completed the matched six-epoch cosine. B9 macro/C1 F1 `0.876324/0.701149` fell to `0.870704/0.687679`; C1 TP `122 -> 120`, restricted FP `68 -> 71`, and `2 -> 1` `41 -> 45`. Only 18 validation argmaxes changed and 13 were correct-to-wrong. | Do not tune B23 LR/width/loss/epochs or repeat a uniform all-patch ConvPass. A successor needs sample-specific regional evidence or a materially different local teacher, with its signal measured before training. |
| `B24-01` | `CLOSED_FAIL` | A locally biased DINOv3 teacher may contain complementary maturity cues that uniform adapters cannot create, but trying multiple backbones/fusions after seeing results would be a sweep. | ConvNeXt-only is worse than raw DINO, while standardized fusion raises macro/C1 F1 `0.802076/0.517520 -> 0.818554/0.542021`, retains C1 TP `288 -> 287`, and reduces restricted FP `317 -> 268`. Macro delta CI is positive, but pair-AUROC lower CI is `-0.005649` versus the `-0.004` guard; fusion folds 1/3/4 hit `2000` iterations. | No new backbone, feature interface or gate change. B25 may change only the solver ceiling to establish the same objective's converged result; B24 itself never promotes. |
| `B25-01` | `CLOSED_FAIL` | The B24 point gain may be a partially optimized readout artifact rather than stable complementary representation evidence. | Solver-only B25 converges in `1735/2110/1971/2053/2044` iterations. Scores move by at most `0.001093` but zero argmax changes; metrics and bootstrap intervals remain identical, so the pair-AUROC lower CI still fails. | Exact ConvNeXt-T teacher route is closed. Its descriptive FP reduction may inform the problem statement, but cannot supervise or select a successor. |
| `B26-01` | `CLOSED_FAIL` | Uniform local mixing fails, final-token selection is deployment-closed, and old no-pretrain deep prompts suppressed C1 recall; a pretrained successor must add regional evidence without perturbing B9. | The strict-reloaded query residual is active (`p95=0.02463`) but its normalized attention entropy is `0.995904`, above the nonuniformity limit. Versus B9, macro/C1 F1 changes `0.876324/0.701149 -> 0.875392/0.697143`; C1 TP stays `122`, while restricted FP and `2->1` both rise by two. Only three argmaxes change: two correct class-2 rows become class 1 and one class-3 row is repaired. | Do not tune B26 query width, loss, LR or duration and do not run its mean-token control. A successor must route a sparse, input-dependent set of regions and add sub-patch/pixel information absent from the fixed 16x16 DINO tokens; test stays sealed. |
| `B27-01` | `CLOSED_FAIL` | B26 failed because it softly pooled existing tokens; a sparse high-resolution route may expose information lost by patch-16 compression. | Max-over-head attention has mean entropy `0.889948`, but `49.85%` of its selected patches fall in the outer two grid cells and the most frequent top-1 sites lie on row 0/column 0. Candidate-minus-base macro/C1 F1/pair-AUROC is `-0.003302/-0.007406/-0.001741`; restricted FP rises `317 -> 337`. The spatially shifted control is also better than the candidate by `0.004326` C1 F1. | Close the max-head route and do not sweep K, subdivision or PCA. This does not close subpatch information because the matched route test failed. One successor may replace only head aggregation with prospectively defined decision-margin degradation. |
| `B28-01` | `CLOSED_FAIL` | Feature-distance head selection from SubViT may still favor DINO outliers irrelevant to the five-class boundary; the router teacher must target the actual predicted-class margin without using the row label. | The full run confirms a real selector change: positive margin degradation `91.00%`, border selection `52.01% -> 39.05%`, route equality with max-head only `13.77%`. Nevertheless candidate-minus-base macro/C1 F1/pair-AUROC is `-0.001367/-0.005477/-0.000570`, restricted FP rises `317 -> 326`, and only 1/5 C1 folds improves. | Close raw-DINO final-attention plus patch-projection/Haar subpatchs on current evidence. Reopen only with a different measured local representation, not another head/K/PCA selector. |
| `B29-01` | `CLOSED_FAIL` | B24's useful ConvNeXt complementarity should not require a 27.8M second inference backbone; a fast state-space student may learn its correction. | The fixed-fold pilot is strongly mechanism-positive: candidate-minus-control accuracy/macro/C1 F1 is `+0.020296/+0.020326/+0.028594`, restricted FP falls `69 -> 52`, and strict reload is exact. It fails only locked TP retention (`50 -> 48`). A 5,000-draw whole-component bootstrap keeps accuracy and macro lower bounds positive, while C1 F1 remains uncertain. | Do not rerun ordinary coupled KL or promote a bias tuned on fold 0. A successor may keep the exact backbone/teacher and change only the transfer/calibration equation, then confirm on TRAIN folds not used to choose that equation. Validation/test/full train remain sealed. |
| `B30-01` | `CLOSED_FAIL` | B29's fusion signal is useful, but coupled KL can transmit harmful teacher margins and its unweighted retention can sacrifice minority TP. | TMORD beats raw DINO C1 F1 by `+0.056022` but loses to coupled KL by C1/macro/accuracy `-0.032447/-0.015821/-0.013577`, loses one TP and adds seven restricted FP. Positive-gain routing covers `78.75%` of pairs with mean `1.446` under cap `2`, so it is insufficiently selective and removes useful non-target structure. | Do not tune cap, C1 weight or gate. Preserve the coupled-KL control as evidence and complete that unchanged equation on TRAIN folds 2--4. |
| `B31-01` | `CLOSED_PASS` | Fold-0 TP loss may be fold variance rather than a coupled-KL defect; fold 1 independently reverses it and produces a large all-metric gain. | Confirmation folds 2--4 improve accuracy/macro/C1 F1 by `+0.027935/+0.031605/+0.058224`, add two TP and remove 61 restricted FP; C1 improves in 3/3 folds. Whole-component C1-gain CI is `[+0.03018,+0.08997]`. Full five-fold C1 F1 is `0.579051` versus `0.517520`. | Authorizes one train-all/matched exploratory validation of this exact raw-DINO+M1 equation. It does not authorize a B9 graft, test, train+val finalfit or presentation claim. |
| `B32-01` | `CLOSED_FAIL` | OOF success may disappear on validation or remain below supervised pure B9; attaching ConvNeXt knowledge directly to B9 is not justified. | B32 validates the mechanism versus raw DINO (`+0.019766/+0.021777/+0.031702` accuracy/macro/C1 F1; `+3` TP), but reaches only `0.887455/0.837360/0.583815` versus B9 `0.912061/0.876324/0.701149`, with 18 more restricted FP. | Do not tune raw-primary B32 on validation or run its audits. Its image-only M1 residual may be screened once as a fixed feature beside B9 under source-fold TRAIN readout. |
| `B33-01` | `CLOSED_FAIL` | B32's M1 residual may contain useful knowledge even though its raw-DINO operating point is weaker than supervised B9. | The apparent TRAIN gain did not generalize. On design-exposed validation, exact B9 / B9-only readout / B9+M1 gives C1 F1 `0.701149 / 0.705202 / 0.656716`; the candidate loses 12 C1 TP while reducing restricted FP by only one versus B9. | The source-fold readout was not end-to-end OOF because both upstream generators had seen all TRAIN rows. Do not tune B33. Any successor must cross-fit every upstream state and evaluate a fold unseen by the primary, auxiliary branch and fusion readout. |
| `B34-01` | `LOCKED` | B33 may fail because its upstream states saw all TRAIN, or because the raw-DINO-aligned M1 residual is intrinsically misaligned with a supervised DINO primary. | Reuse the preserved B21 spatial EMA and B29 candidate M1, both trained without component fold 0. Fit a balanced 5-D primary-only readout and a matched 10-D primary+M1 readout on folds 1--4; score fold 0 once. | Require C1 F1 gains `>=0.005` over raw primary and `>=0.010` over readout control, accuracy/macro no worse than `-0.002`, no C1 TP loss versus primary, and no restricted-FP or `2->1` increase versus control. Pass permits one B9-aligned cross-fitted residual fold; never validation/test. |
| `KD-01` | `GUARD` | B19 proves that raw-DINO relation distillation itself improves SwiftFormer. | False. Stock, control and candidate all receive the same cosine-neighbour relation loss; B19 isolates spatial atoms under that objective, not relation KD versus CE-only. The normalized relation also does not preserve photometric magnitude by construction. | Any causal KD claim requires a separately preregistered matched experiment on prospectively sealed evidence; B19 cannot be reinterpreted as that ablation. |
| `TELEM-01` | `LOCKED` | Final factor norms and scalar loss curves are enough to diagnose optimization if a successor fails. | Rejected. They prove activation and fit behaviour but cannot distinguish backbone absorption from branch starvation. | B21 records per-role gradient and update/parameter norms plus adapter residual ratios by epoch. Gradient-conflict telemetry is required only when an auxiliary objective exists; B21 deliberately has none. |
| `LR-01` | `CLOSED` | Exact float equality is safe when a mathematically identical endpoint is produced through multiplication and division. | Rejected by the B18 real-fold counterexample. Endpoint values must be returned explicitly, while interior points retain the original formula; tests must exercise every locked fold horizon. | Permanent scheduler implementation rule. |
| `RNG-01` | `CLOSED` | CPU-only RNG tests are sufficient for a CUDA training protocol. | Rejected. `torch.manual_seed` seeds CUDA as well as CPU; a CPU-only test left CUDA uninitialized and therefore missed the leak caused by `fork_rng(devices=[])`. | Every successor must initialize the exact GPU first, snapshot every CUDA generator, exercise construction/head reset, and prove bit-exact restoration. |
| `DEPLOY-01` | `LOCKED` | Host ORT QDQ latency is a valid proxy for mobile efficiency and quantization must be faster. | Not established. B16 remains closed under its original 12 ms rule. For a successor whose factors fold into an exactly stock-topology graph, one prospectively locked TRAIN-only causal screen may follow stock-only source/ONNX/ORT-format/mobile-checker/host qualification; it must remain `DEVICE_PENDING`. This exception evaluates representation without pretending the Windows host is an Android proxy. | Before validation, promotion or any mobile-ready claim, freeze a physical Android ARM64/SoC/API/ORT-AAR/power/thermal contract, qualify stock first on CPU and XNNPACK, lock one backend and its application-derived absolute budget, then require candidate p95 at most `1.03x` stock. No backend/threshold choice may use candidate results. |
| `CLEAN-01` | `CLOSED` | Old pretrained runs can be deleted by name or age. | Rejected. Tiers A--D used exact reviewed manifests. Tier D preserved 153 compact evidence copies, verified 550 non-last files including 16 `best.pt` plus 38 anchors, and deleted only 16 obsolete `last.pt` files (`5,084,665,336` bytes). Result SHA256 `978433fdbb1b95882c4eb78c16520436cdb5a507ccc7ffa07ad7326d51e33285`. | Any later cleanup needs a new exact manifest and independent verification; completed tiers never authorize name/age/wildcard deletion. |
| `SEAL-01` | `GUARD` | Freezing a protocol permits reuse of the historical test. | False. B13 is TRAIN-only; current validation is exploratory/design-exposed and historical test cannot select or confirm a new model. | A new claim requires a prospectively sealed source/time/site holdout. |

## Evidence, not assumptions

- Canonical contract: 12,019 images; train/val/test `8278/2479/1262`; no declared `source_image` or `leakage_group` crosses a split.
- Train has `8278` rows but only `5361` leakage groups. Validation has `2479/2471`; test has `1262/1262`. Repeated views are therefore concentrated in train.
- The current tempered sampler operates on rows. Under its quota, one class-2 group can be replayed about 54 times per epoch and one class-3 group about 41 times.
- Class 1 is `738/12019 = 6.14%`; `116/497` class-1 train rows belong to components containing another label. This quantifies a difficult visual boundary and source dependence; the canonical labels remain correct by project definition.
- The best exact B9 full-val export is accuracy `0.912061`, macro-F1 `0.876324`, class-1 P/R/F1 `0.642105/0.772152/0.701149`. Its class-1 false positives are `2->1: 41`, `0->1: 21`, `4->1: 6`.
- The valid B14 TRAIN-only result is iFormer-S accuracy/macro-F1/C1-F1/pair-AUROC `0.847065/0.789602/0.492933/0.940750` versus locked DINO `0.858178/0.802076/0.517520/0.946103`. The paired 95% intervals are macro `[-0.030516,+0.007787]`, C1 `[-0.067381,+0.017714]` and AUROC `[-0.014060,+0.003169]`; the exact pooled-320 route is closed. Evidence: `runs/pretrained_iformer_s_classf_b14_frozen_transfer_c797a6f_r1/summary.json`.
- B15 stopped before TRAIN because the aligned DINO ONNX graph measured `75.324 ms` p95 versus `67.500 ms` for uniform pooling (`1.115911x`); its graph added only `8,987` bytes, so the rejection is an observed tail-latency cost rather than a parameter-count failure. No claim about class-1 representation quality is permitted from this preflight.
- Class 1 is not one uniform visual cohort. B9 recall is `0.852` on samples whose historical records changed and only `0.688` on the stable-history cohort. Frozen-DINO embeddings place the latter near class-0 false positives and the former near class-2 false positives. This motivates better representation; it does not authorize relabelling.
- The two false-positive streams mirror those cohorts photometrically: B9 `0 -> 1` errors have red-minus-green mean `-0.0005` versus stable class 1 `+0.0017`; `2 -> 1` errors are `+0.0248` versus relabelled class 1 `+0.0150`. Brightening raises `0 -> 1` from `21` to `128`, while dimming raises `2 -> 1` from `41` to `72`. This supports illumination-normalized local fruit-surface evidence, not a provenance flag at inference.
- Exact-file integrity is good, but the declared grouping is not session/crop safe. It reports no cross-split group or SHA-exact overlap, yet adjacent IDs from the same source form 1,115 same-label train/val pairs; 193 have crop pHash distance `<=8` versus zero in a random control, and B9 is correct on all 196 val rows with such a same-label train neighbour. Conversely, train contains 105 declared groups mixing class 1 with another label, affecting `116/497` class-1 rows; `88/497` have a cross-label crop within pHash `<=8`. `_source_family` currently treats consecutive frames such as `Image_10383` and `Image_10384` as different families. Current validation is therefore suitable for matched exploratory ablation, not a clean generalization claim. The older `docs/dataset_leak_audit_5class` describes 13,088 rows and is stale for the current 12,019-image corpus; do not cite it.
- The stricter TRAIN-only mechanism audit found no class-map/loader failure and no exact-SHA cross-label duplicate. It did find `120` luminance-pHash-radius-3 pairs crossing classes, including `55` involving class 1; this is retained as evidence that texture/chroma and multi-depth context must separate visually near samples. Historical label-change fields and numeric adjacency are not training targets or permission to question canonical ground truth.
- The historical A0 hybrid has no canonical-class_f comparison. Its only yolo_f smoke trained `691,841/29,524,375` parameters, collapsed DINO patch geometry into global mean/std, and ended with an effective gate near `2.4e-5`.
- The valid B19 TRAIN-only candidate reached accuracy/macro-F1/C1-F1 `0.903721/0.856104/0.632939`. Its spatial gain over the matched mean control was inconclusive, while active-versus-delta-off added `47` C1 true positives but also added `45` and removed only one non-C1 false positive into C1. The mechanism broadens the class-1 region rather than separating its boundary.
- B19 exposes a representation ceiling for its own target: candidate C1 recall is `0.719697` on single-class components but `0.446352` on cross-class components; aggregate accuracy is `0.949206` versus `0.714464`. These facts do not excuse the matched failure and make a stronger same-DINO integration path worth testing.

## Closed experiment: B19 bit-exact SurfaceFold-XS

The one-use B19 run completed all five component folds, 15 final states and 5,000 locked whole-component bootstrap draws on commit `b39f4d0`. All endpoint LR bytes, source/data hashes, strict reloads, folded parity and resource contracts passed. It opened every TRAIN row exactly once from its held fold and opened neither validation nor test.

| TRAIN-only OOF arm | Accuracy | Macro-F1 | C1 P/R/F1 | C1 TP/FP/FN | `2 -> 1` | Restricted FP |
|---|---:|---:|---:|---:|---:|---:|
| Stock relation | `0.903842` | `0.855189` | `0.678404/0.581489/0.626219` | `289/137/208` | `83` | `134` |
| Mean-surface control | `0.904325` | `0.856356` | `0.683099/0.585513/0.630553` | `291/135/206` | `80` | `131` |
| Spatial candidate | `0.903721` | `0.856104` | `0.680556/0.591549/0.632939` | `294/138/203` | `84` | `135` |
| Same candidate, delta off | `0.902150` | `0.846630` | `0.724340/0.496982/0.589499` | `247/94/250` | `70` | `92` |

Candidate beat control on C1 F1 in four folds, but the aggregate delta was only `+0.002386` with interval `[-0.009669,+0.014988]`; macro-F1 delta was `-0.000253 [-0.003689,+0.003212]`, mean pair-AUROC delta `-0.000045 [-0.000762,+0.000690]`, and pair-2 AUROC delta `-0.000533`. Versus its own delta-off checkpoint, candidate C1 F1 increased `+0.043440 [+0.019285,+0.069964]`, but this came from `+47` TP together with a net `+44` FP into C1. The factor is active and causally used; its learned direction repeats the historical recall-for-precision failure.

The post-hoc locked-cohort diagnosis is descriptive, never a selection gate. Relative to control, candidate gains five C1 TP and loses none inside cross-class components, but loses two net TP on single-class components. Near-cross-label endpoints remain near chance (`0.505155` accuracy, C1 recall `0.229167`). Therefore B19 contains a technical target/placement mismatch: its normalized 84-edge teacher target discards magnitude, chroma and class direction. No B19 width/rank/loss/LR/epoch retry is allowed.

Authoritative artifacts:

- Summary SHA256 `6731b018deeea1a42b50809cabb3ca016b06cbf3780002ff5fec4b2b9e984fe9`: `runs/pretrained_surfacefold_xs_b19_train_oof_b39f4d0_r1/summary.json`
- Gate SHA256 `1e15775ae56fc7a14e7ad099221b8eaf3d7a0f8e7214889beadbbeb16b44de97`; metrics SHA256 `fa0f5f06adaa15f9d09c99242aa5f698f950db5a9847aa98a5845042d82d316e`
- Bootstrap SHA256 `437c42fa4bb1ea748416184900058d5848c4082eb90ea898ef52f007b404f38b`; OOF SHA256 `5ea1c6a26005dd3bc28b3d026f2b7b10765527f2a3f57555271d07cc61bac0c7`
- Metric-barrier SHA256 `17a32f4d7c960580234fdebb96edc6025d23c06edbf4e9dee52453f2e83e2f2b`; closure tag `trkh-pretrained-b19-closed-b39f4d0`

## Closed screen: B21 DINOv3 multi-depth spatial bypass

B21 keeps the exact DINOv3-S/16 primary architecture, verified pretrained
asset, 256-pixel preprocessing, B9 sampler/loss and discriminative LR split.
It adds a ConvPass-inspired bottleneck parallel to both MHSA and MLP in all 12
EVA blocks:

`u = x + DP1(gamma1 Attn(LN1(x))) + A_attn(LN1(x))`

`y = u + DP2(gamma2 MLP(LN2(u))) + A_mlp(LN2(u))`

Each `A` is `384 -> 8 -> QuickGELU -> dense 3x3 -> QuickGELU -> 384`.
The dense convolution starts as a center identity and the up projection starts
at exact zero. Scale is locked once at `1.0`, the official implementation
default and the median of its released task configs. Adapter dropout and an
extra DropPath are deliberately removed: B21 is a deterministic TRKH variant,
not a claim of exact ConvPass reproduction, and direct/spatial arms must consume
the same native EVA RNG sequence. The design follows the official evidence that
parallel full MHSA+MLP placement is stronger than sequential or attention-only
placement ([ConvPass paper](https://arxiv.org/abs/2207.07039),
[official code](https://github.com/JieShibo/PETL-ViT/blob/main/convpass/vtab/convpass.py)).

The 24 adapters add exactly `170,880` parameters (`0.7915%` of the 21,588,869
parameter five-class DINO-S), not the paper's ViT-B percentage. Five DINO prefix
tokens pass independently through the same 3x3 center response; the 256 patch
tokens use the true `16x16` grid. A confirmatory `dephased` arm retains identical
weights/operators but applies a fixed patch permutation before each 3x3 and its
inverse afterwards, destroying only neighbourhood topology.

Static and real-backbone preflight passed: construction preserves CPU and
initialized-CUDA RNG, direct and spatial logits are bit-exact native DINO at
step zero (`max_abs=0`), the first backward reaches every up projection while
leaving Conv/down at zero as expected, and the second update reaches Conv/down.
The accepted preflight on commit `9bbd899` also proves a bit-exact first batch
with two spawned Windows workers. Observed CUDA peak for the one-image two-step
probe is `487.5 MiB`; total model parameters are `21,759,749`.

The only authorized Phase-B screen uses TRAIN component fold 0, direct then
spatial, seed `20260805`, five fixed epochs, batch `16`, accumulation `3`, BF16,
AdamW, task/adapter LR `1.5e-4`, backbone LR `1.5e-5`, weight decay `0.05`, clip
`0.7`, two-epoch warmup on the fixed 30-epoch cosine horizon, EMA `0.995`,
two deterministically seeded loader workers, tempered class power `0.5`, and
B9 LDAM-Focal/augmentation. It constructs no
validation or test dataset and selects no epoch. Spatial advances only if its
final EMA has C1-F1 delta at least `+0.005` versus direct **or** reduces restricted
`0/2/4 -> 1` FP by at least `5%` while retaining at least `98%` of direct C1 TP;
macro-F1 delta must be at least `-0.002`, `2 -> 1` cannot increase, adapter
residual p95 must lie in `[1e-4,0.1)`, and all updates/telemetry must be finite.
A pass would have authorized fixed five-fold TRAIN OOF with the dephased
control, not validation or test. B21 did not pass.

The completed paired screen is numerically and operationally valid:

| TRAIN fold-0 EMA arm | Accuracy | Macro-F1 | C1 P/R/F1 | C1 TP/FP/FN | `2 -> 1` | Restricted FP |
|---|---:|---:|---:|---:|---:|---:|
| Direct DINO | `0.867800` | `0.839621` | `0.719101/0.666667/0.691892` | `64/25/32` | `15` | `25` |
| Spatial B21 | `0.880417` | `0.846775` | `0.717647/0.635417/0.674033` | `61/24/35` | `17` | `24` |

Spatial improves accuracy `+0.012617` and macro-F1 `+0.007154`, but C1 F1
changes `-0.017859`, TP retention is only `0.953125`, restricted FP reduction
is only `4%`, and `2 -> 1` increases by two. The final adapter gradient norm is
`4.975993`, update/parameter norm is `0.022106`, and residual p95 is `0.032020`;
the branch is active and bounded. Strict safetensors reload reproduces BF16
logits exactly (`max_abs=0`). Exact B21-A0 is therefore closed without
validation/test access.

The error sets are nevertheless complementary. Direct-only and spatial-only
C1 true positives are `6` and `3`; a descriptive C1-union on top of the spatial
predictions gives accuracy/macro/C1-F1 `0.880965/0.851623/0.697917` with
`67` TP and `29` FP. This is a post-result upper-bound diagnosis, not a model
or promotion result. It motivates B22's single-pass recall-monotonic expert;
it does not authorize a two-model ensemble or threshold sweep.

Authoritative artifacts:

- Preflight: `runs/preflight_b21_convpass_9bbd899_r1/preflight.json`
- Paired result: `runs/b21_dinov3_convpass_fold0_9bbd899_r1/summary.json`
- Spatial EMA SHA256: `69b5902735be2b916ff127eed653bc128713398ae6f47db13aa99be4a491857c`

The five-epoch outcome is a valid rejection of its locked Phase-B recipe, but
not a convergence result. The run used a two-epoch warmup on a 30-epoch cosine
horizon and ended at LR `1.458193e-4`, still `97.2%` of the `1.5e-4` peak.
Further, it restarted from the raw DINO asset with a reset head and jointly
updated all `21,759,749` parameters. The official ConvPass training recipe
freezes the primary backbone and optimizes adapters plus the task head. A
successor must therefore fix initialization/optimization causally instead of
merely adding epochs to this exact run.

## Closed diagnostic: B22 recall-monotonic post-pooling expert

B22 froze the exact B21 spatial EMA and added only a non-negative bounded
correction to class-1 logits. That construction guarantees that a row already
predicted as class 1 cannot leave class 1. Three fixed arms used only TRAIN
fold 0: one scalar bias, six logits-only parameters, and a 390-parameter
standardized pooled-feature/logit expert.

| Frozen TRAIN-fold arm | Accuracy | Macro-F1 | C1 P/R/F1 | C1 TP/FP | `2 -> 1` |
|---|---:|---:|---:|---:|---:|
| Spatial B21 base | `0.880417` | `0.846775` | `0.717647/0.635417/0.674033` | `61/24` | `17` |
| Bias only | `0.880965` | `0.850331` | `0.706522/0.677083/0.691489` | `65/27` | `18` |
| Logits only | `0.838727` | `0.797839` | `0.381818/0.875000/0.541935` | `84/130` | `70` |
| Feature + logits | `0.856829` | `0.816621` | `0.471338/0.770833/0.584980` | `74/83` | `41` |

The bias-only arm nearly recovers direct B21 C1 F1 `0.691892`, but does not
beat it. The feature arm lowers its fit objective from `0.465207` to `0.226587`
while widening the C1 region, so the missing information is not recoverable by
another flexible head over the already pooled representation. No validation or
test data was constructed.

The first replay correctly failed before training because eval batch `64`
changed BF16 near-boundary predictions relative to B21's batch `32`, despite
identical state and preprocessing. Binding B22 to batch `32` restored bit-exact
replay. Weight hashes alone are therefore insufficient for numerical evidence.

- Result: `runs/b22_recall_monotonic_expert_diag_695b65d_r1/summary.json`
- Summary SHA256: `e11d0254dcb96e22635ee93b8170ff4fcc43268b6a6e4897e8cb1872399f8c72`

## Closed screen: B23 supervised-B9 guarded ConvPass

B23 corrects two confounds rather than inventing another unrelated backbone.
The official ConvPass recipe freezes the pretrained primary network and trains
adapters plus the task head; B21 instead jointly moved the full primary model
([official training code](https://github.com/JieShibo/PETL-ViT/blob/main/convpass/vtab/train.py)).
Fine-grained ViT work also supports retaining local/middle-depth patch evidence
rather than relying only on a final global vector
([FFVT](https://arxiv.org/abs/2107.02341)). B23 therefore strict-loads the
selected B9 EMA, freezes both its ViT and already-supervised head, and learns
only the existing 24 spatial adapters before pooling. Its class-1 retention
hinge permits stronger class-1 evidence and suppression of hard negatives, but
penalizes loss of labelled class-1 margin relative to B9. The weak KL term is a
trust region, not output-only distillation presented as a new representation.

The fixed exploratory screen completed six full TRAIN epochs, BF16
training/FP32 evaluation, batch `16`, accumulation `3`, adapter LR `1.5e-4`,
one-epoch warmup and six-epoch cosine decay. It replayed B9's exact FP32
validation confusion before training and never opened test.

| FP32 design-exposed validation | Accuracy | Macro-F1 | C1 P/R/F1 | C1 TP/FP/FN | `2 -> 1` |
|---|---:|---:|---:|---:|---:|
| Inherited B9 | `0.912061` | `0.876324` | `0.642105/0.772152/0.701149` | `122/68/36` | `41` |
| B23 spatial EMA | `0.908431` | `0.870704` | `0.628272/0.759494/0.687679` | `120/71/38` | `45` |

The adapter is active (residual p95 `0.004726`) and strict reload is exact, but
all class-1 advancement gates fail. Only 18 argmaxes change: 13 move from
correct to wrong and four from wrong to correct; the dominant harms are four
new true-class-2 predictions into class 1 and four true-class-3 predictions
into class 4. Thus the corrected optimization does not rescue uniform local
mixing. A descriptive scalar class-1 bias over B9 peaks at only `0.703170` C1
F1, so calibration alone cannot reach the target either.

- Result: `runs/b23_b9_guarded_convpass_5a1c310_r1/summary.json`
- Summary SHA256: `da68b1ee75bd4dd1db674ce459bc0dd6f681d92917d6ec99740a98479c5fbcfb`

## Closed screen: B24 DINOv3-ConvNeXt local-teacher signal

B24 asks one bounded question before another hybrid is designed: does an
official locally biased DINOv3 ConvNeXt-T representation add complementary
TRAIN-only signal to the retained raw DINO-S representation? It strict-loads
the official `27,820,128`-parameter ConvNeXt-T asset, extracts only its final
768-dimensional descriptor using the released 224-pixel preprocessing, and
uses the existing source-component folds. The retained DINO descriptor and OOF
scores must replay byte-exactly. ConvNeXt-only is reported as a mechanism
control; only the predeclared standardized 384+768 fusion is gated. This is a
teacher oracle screen, not a mobile architecture or permission to deploy two
backbones. A pass permits designing a single-backbone distillation/region
student; failure closes this exact teacher route without trying S/B/L variants.

The formal screen is integrity-complete and used no validation/test data.
ConvNeXt-T alone does not beat raw DINO; the useful result is complementarity:

| TRAIN-only OOF arm | Accuracy | Macro-F1 | C1 P/R/F1 | C1 TP/FP | Restricted FP |
|---|---:|---:|---:|---:|---:|
| Raw DINO | `0.858178` | `0.802076` | `0.467532/0.579477/0.517520` | `288/328` | `317` |
| ConvNeXt-T | `0.856608` | `0.797819` | `0.454839/0.567404/0.504924` | `282/338` | `333` |
| Standardized fusion | `0.874003` | `0.818554` | `0.510676/0.577465/0.542021` | `287/275` | `268` |

Fusion reduces `0 -> 1` `159 -> 131` and `2 -> 1` `118 -> 107` while losing
one C1 TP. Macro-F1 delta is `+0.016478 [0.005152,0.028059]`; C1-F1 delta
is `+0.024501 [-0.012338,0.062710]`; pair-AUROC delta is `+0.000734
[-0.005649,0.007292]`. The last lower bound fails the locked `-0.004` harm
guard. More importantly, fusion folds 1, 3 and 4 hit the `2000`-iteration
ceiling, so the exact screen cannot promote despite its useful point evidence.

- Result: `runs/b24_dinov3_convnext_train_oof_9433474_r1/summary.json`
- Summary SHA256: `9de8cd82ccd0a8fb7be7a3e53bc6533c1f0707efe9866bff35db33666d6c5c7e`

## Closed numerical resolution: B25

B25 changes no representation, sample, fold, preprocessing, objective,
regularization or gate. It reuses the exact retained B13 DINO and B24 ConvNeXt
descriptors and raises only LBFGS `max_iter` from `2000` to `10000`. This is the
sole authorized numerical resolution of B24. It does not read any image split;
validation/test/full training remain forbidden.

B25 converged all fusion folds in `1735/2110/1971/2053/2044` iterations.
Relative to B24, decision scores move by at most `0.001093` and no argmax
changes. Therefore every point metric and bootstrap interval is unchanged;
the sole remaining failure is still the predeclared pair-AUROC harm guard.

- Result: `runs/b25_b24_fusion_readout_df613bc_r1/summary.json`
- Summary SHA256: `96b7f929188c69c4f25177ba7becfd4269af86a231e3f02d5f9cbe33d551d95f`

## Closed screen: B26 one-way multi-depth evidence queries

B26 uses the class-query idea from Prompt-CAM while correcting the historical
TRKH failure mode: query tokens can read patch evidence, but the frozen B9
tokens never attend to queries and therefore cannot drift. FFVT motivates
multi-depth patch evidence; the recent SubViT result further supports allocating
extra capacity only to sample-specific discriminative regions rather than
uniformly refining all tokens
([Prompt-CAM](https://openaccess.thecvf.com/content/CVPR2025/papers/Chowdhury_Prompt-CAM_Making_Vision_Transformers_Interpretable_for_Fine-Grained_Analysis_CVPR_2025_paper.pdf),
[FFVT](https://arxiv.org/abs/2107.02341),
[SubViT](https://arxiv.org/abs/2607.09086)).

The fixed decoder carried five persistent 64-D class queries through one
shared four-head cross-attention and FFN, reading the frozen patch streams after
blocks `3`, `7` and `11`. A zero-initialized shared scalar head makes B26 exactly
B9 at step zero; `0.5*tanh(score)` bounds every class-logit residual. The decoder
adds `58,753` parameters (`0.272%` of B9) and exposes class-to-patch maps for XAI.
Training is twelve full TRAIN epochs with SGD `0.005`, momentum `0.9`, weight
decay `0.001`, two-epoch warmup and the same twelve-epoch cosine horizon. B9 is
always frozen/eval. LDAM-Focal is augmented by a weight-`1.0` labelled-C1
margin-retention hinge, weight-`0.10` rival hard-negative hinge and weight-`0.10`
KL trust region. The design-exposed validation is opened once after exact B9
replay; test remained sealed. The run finished all `3108` optimizer updates and
strict reload was exact (`max_abs=0`). The residual was active
(`p50/p95/max=0.01203/0.02463/0.03041`), but attention collapsed toward uniform
over patches (normalized entropy `0.995904`, std `0.000847`). B26 kept all `122`
class-1 true positives but added two class-2 false positives into class 1;
accuracy/macro/C1 F1 became `0.911658/0.875392/0.697143` versus B9
`0.912061/0.876324/0.701149`. It changed only three decisions, so the exact
decoder is closed as a weak global-boundary residual, not as a limit on sparse
regional refinement.

- Result: `runs/b26_evidence_query_b1616ea_r1/summary.json`
- Summary SHA256: `af425cf1aab177c17008d02eacdd650eb0ce2c8489f23ee174370c0c715b3d9e`
- Query-state SHA256: `5b7d6cbb8675879bc06950aaa06631f832e88f273aedfa026ff18e6d2547cfe4`

## Closed screen: B27 sparse subpatch information

B27 tests the missing-information premise before another hybrid is trained.
It follows SubViT's central observation that patch-16 tokenization can discard
fine detail and that random extra tokens are not sufficient, but uses a fixed
`K=3` (`1.17%`) and an explicit same-capacity spatially dephased control rather
than a dataset-specific sweep. The candidate reads new 8x8 pixel regions using
the frozen DINO patch projection; it does not reweight B9 logits or reuse B26's
soft attention decoder. Only canonical TRAIN images and the existing five
source-component folds may be constructed. A pass is evidence to build one
lightweight deterministic router and bounded refiner; it is not permission to
open validation or test.

The locked run failed. Candidate accuracy/macro/C1 F1 was
`0.855158/0.798774/0.510114` versus raw DINO
`0.858178/0.802076/0.517520`; restricted FP increased `317 -> 337` and all
information gates failed. Only one of five folds improved C1 F1 materially.
The dephased route reached C1 F1 `0.514440`, so extra readout capacity cannot
explain the failure. The route changed `319` base decisions with `160`
correct-to-wrong and `135` wrong-to-correct.

- Result: `runs/b27_sparse_subpatch_train_oof_bfa1aee_r1/summary.json`
- Summary SHA256: `22af39f72fe9e51b6a3cfd83be785131b26e0cd5afd471949cc0acdd3f3f6f99`

## Closed screen: B28 margin-degradation subpatch teacher

B28 keeps B27's raw DINO, `K=3`, 2x2 subdivision, pretrained patch projection,
three embedding-space contrasts, PCA/readout, source folds, shifted control and
promotion gate. It changes only the selector. Each held row uses its already
retained fold-specific readout to define the predicted top-1/runner-up margin;
labels are not inputs to head selection. The head whose token removal maximally
reduces that margin supplies the route. This is a boundary-targeted adaptation
of SubViT's degradation teacher, not a head/K sweep.

The formal selector behaved as designed: mean selected-map entropy was
`0.835655`, positive decision-margin degradation covered `91.00%` of rows,
and only `13.77%` of routes matched max-head. This did not create new useful
classification information. Candidate accuracy/macro/C1 F1 was
`0.857091/0.800709/0.512043` versus raw DINO
`0.858178/0.802076/0.517520`; the shifted route reached C1 F1 `0.520147`.
The exact raw-DINO attention/subpatch family is therefore closed.

- Result: `runs/b28_margin_subpatch_train_oof_4510d43_r1/summary.json`
- Summary SHA256: `c8cc82bff8f8c5dfeabe4f48d3e55e53b5b3dcb04738e28256270c7057c3df15`

## Closed design screen: B29 EfficientViM local-teacher residual

B29 uses ConvNeXt only to construct a fold-fit TRAIN teacher. At inference the
candidate consists of the unchanged DINO primary plus official pretrained
EfficientViM-M1, whose canonical four stage heads produce a bounded score
residual. The matched control has the same M1 initialization, trainable
parameters, sampler, task loss, DINO trust target and schedule; candidate changes
only the KD target to the DINO+ConvNeXt fusion. This tests transfer of measured
local complementarity across architectures rather than frozen M1 quality or a
two-large-backbone ensemble. Fold-0 success permits five-fold OOF, not validation.

The one fixed TRAIN-fold run completed `606` updates per arm in `449.98 s`,
used `1.225 GB` peak CUDA allocation, strict-reloaded both student states with
maximum error zero, and constructed neither validation nor test. Results are:

| Fold-0 held TRAIN arm | Accuracy | Macro-F1 | C1 P/R/F1 | C1 TP | Restricted FP |
|---|---:|---:|---:|---:|---:|
| Raw DINO readout | `0.801975` | `0.746171` | `0.390625/0.520833/0.446429` | `50` | `75` |
| Same-M1 DINO control | `0.806912` | `0.751753` | `0.409836/0.520833/0.458716` | `50` | `69` |
| M1 fusion-teacher candidate | `0.827208` | `0.772079` | `0.475248/0.500000/0.487310` | `48` | `52` |

Candidate-minus-control accuracy/macro/C1 F1 is
`+0.020296/+0.020326/+0.028594`; the candidate also removes 17 restricted FP.
A 5,000-draw whole-component bootstrap gives accuracy
`+0.02025 [0.01100,0.02921]`, macro `+0.02030 [0.00837,0.03183]`, and C1 F1
`+0.02883 [-0.00737,0.06736]`. The exact gate fails only because C1 TP changes
`50 -> 48`. This is a calibration/transfer failure, not a collapsed hybrid:
candidate residual absolute p95 is `2.62194` under the `6.0` bound.

A diagnostic-only class-1 logit shift shows that `+0.16157` would recover the
two TP while keeping candidate C1 F1/macro/accuracy at
`0.502513/0.775483/0.828305`. Fold 0 has already been exposed, so this value
must not be promoted or scored there again. The next equation must be fixed
before using other TRAIN folds. It should replace coupled KL with trustworthy,
class-aware transfer: gate teacher corrections by true-margin improvement,
separate target/non-target knowledge, and learn any calibration only from the
fit side. This follows the mechanisms in
[DKD](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html),
[logit adjustment](https://openreview.net/pdf?id=37nvvqkCo5), and
[trust-calibrated collaborative learning](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_Trust-calibrated_Collaborative_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2026_paper.html)
without changing `class_f` or adding another backbone.

- Accepted preflight SHA256 `c618c226569562930bc2ebe61f2cdceb5c5afe26069dee675b851a294a9ecd2b`.
- Result: `runs/b29_m1_teacher_residual_c1f67cb_r1/summary.json`
- Summary SHA256: `b33519099671eb434c82a43edeca34fd8e4c90b04f65a6c7eb04f3915a4244f9`

## Locked confirmation screen: B30 trust-margin residual distillation

B30 uses TRAIN fold 1, which was fixed before its teacher or student metric was
read. Both arms add the same official pretrained EfficientViM-M1 residual to a
fold-fit DINO primary. The control repeats B29's coupled KL toward the same
DINO+ConvNeXt teacher. For candidate true label `y` and rival `j`, TMORD defines
`m_p=p_y-p_j`, `m_t=t_y-t_j`, and
`m*=m_p+clip(m_t-m_p,0,2)`. A weighted Huber loss moves the student margin to
`m*`; teacher corrections that reduce the true margin are therefore never
targets. Class-1 margin pairs and primary-correct retention receive weight `4`,
while task CE and the optimizer/sampler/six-epoch cosine remain unchanged.

This is one loss-equation ablation, not another pretrained model search. It
runs a focused unit preflight inside the single command and records source/Git,
strict reload, score and state hashes. Its gate is the `B30-01` row above.
Regardless of outcome, validation, test and full training remain closed; a pass
permits only one additional unseen TRAIN fold.

B30 completed all `618` updates per arm and strict reload was exact. On held
TRAIN fold 1, raw DINO / coupled KL / TMORD reached C1 F1
`0.571429/0.659898/0.627451`, macro-F1
`0.830518/0.865475/0.849654`, and accuracy
`0.876033/0.902597/0.889020`. Coupled KL raises C1 TP `60 -> 65` and cuts
restricted FP `47 -> 30`; TMORD reaches `64/37`. The TMORD residual is active
but smaller (p95 `1.47873` versus KL `2.82174`), and its final task loss is
higher (`0.04150` versus `0.02915`). Its routing is nearly global: `78.75%` of
true-vs-rival pairs receive positive teacher gain and mean gain is `1.446` under
the cap `2.0`.

This rejects the exact positive-margin filter. It discarded useful non-target
relations even though those relations let task CE produce a student better than
the teacher's own C1 score. That mechanism is consistent with DKD's finding
that non-target-class knowledge is a principal source of logit-distillation
benefit. Do not tune TMORD. Its matched KL arm is retained as prospective fold-1
evidence for the unchanged B29 equation.

- Result: `runs/b30_tmord_fold1_7c77f68_r1/summary.json`
- Summary SHA256: `0b788bc94cf5c4bab90c1e7e6b6794df52a95145d519cd49d6f3f798d3857fb2`

## Locked OOF completion: B31 unchanged coupled KL

B31 trains only folds 2--4 with the exact B29 candidate/B30 control equation:
official M1 initialization, DINO primary, fold-fit DINO+ConvNeXt teacher,
coupled KL at temperature `4`, the same task/retention weights, tempered sampler
and complete six-epoch schedule. Folds 0--1 are loaded from their immutable
score artifacts, not rerun. The primary promotion statistic is the aggregate of
the previously unseen folds 2--4 under the `B31-01` gate; the five-fold OOF is
reported secondarily. No validation/test dataset is constructed by B31.

B31 completed `1,902` new updates, strict-reloaded all three states with zero
error, and passed every gate. On confirmation folds 2--4, raw base versus
coupled KL is accuracy `0.873346 -> 0.901281`, macro-F1
`0.816688 -> 0.848294`, C1 P/R/F1
`0.468421/0.595318/0.524300 -> 0.564263/0.602007/0.582524`, TP
`178 -> 180`, and restricted FP `195 -> 134`. C1 improves in all three folds.
A 5,000-draw union-component bootstrap gives accuracy gain
`+0.02805 [0.02148,0.03479]`, macro `+0.03185 [0.02323,0.04073]`, and C1 F1
`+0.05848 [0.03018,0.08997]`; the FP change is
`-61.15 [-85,-39]`.

Across all five OOF folds, accuracy/macro/C1 F1 changes
`0.858178/0.802076/0.517520 -> 0.885238/0.833168/0.579051`, C1 TP
`288 -> 293`, restricted FP `317 -> 216`, and mean pair AUROC
`0.946103 -> 0.956651`. Full-OOF whole-component C1-gain CI is
`[+0.03940,+0.08450]`. This promotes the exact coupled-KL hybrid mechanism to
one matched exploratory validation; it is not yet evidence that it beats B9.

- Result: `runs/b31_coupled_kd_oof_d5e3816_r1/summary.json`
- Summary SHA256: `b117dcd1256bd85b53c330d782626d67dfbca84af19faefc4769c3e243737210`
- OOF score SHA256: `d1e7e9538d3587b4b2ff8beac533d7415871e9f4989a9dedbea4f30a58124ae0`

## Locked matched validation: B32 full-TRAIN raw-DINO + M1

B32 fits the same balanced raw-DINO and DINO+ConvNeXt readouts on all 8,278
TRAIN descriptors, trains one official M1 residual with the unchanged B31
sampler/loss/six-epoch schedule, and extracts raw-DINO validation descriptors
with the exact B13 preprocessing. Validation inference contains raw DINO, its
saved linear primary readout and M1; ConvNeXt is TRAIN-teacher-only. The matched
raw-DINO primary and retained exact B9 predictions are evaluated on the same
ordered 2,479 rows under the `B32-01` gate. Current validation is design-exposed,
so even a pass remains exploratory and cannot open test or finalfit by itself.

B32 completed `780` updates, extracted all 2,479 DINO validation descriptors,
and strict-reloaded with zero error. Raw primary to M1 hybrid changes validation
accuracy/macro/C1 F1 `0.867689/0.815583/0.552113 ->
0.887455/0.837360/0.583815`, TP `98 -> 101`, and restricted FP `98 -> 86`.
The branch is active (residual p95 `3.28688`) and improves every matched gate.
It nevertheless misses B9 by accuracy/macro/C1 F1
`-0.024607/-0.038964/-0.117334`, has 21 fewer C1 TP and 18 more restricted FP.
Thus raw-primary B32 is a validated mechanism but not the target model.

- Result: `runs/b32_full_train_validation_b835798_r1/summary.json`
- Summary SHA256: `6e689d825c187453892487cf97ffe78ddc36937a0d6c85ab6ef384a3e77a2f84`
- Validation score SHA256: `958d9d6e7471d58f0bc2ede99166c45da12836659e0d30476f0f0c1d2b8965d5`

## Locked B9 fusion: B33 distilled M1 residual readout

The B32 M1 output is image-only, so raw DINO is not required after training.
Its extracted TRAIN residual (SHA256
`490f07f210870312c7137ba9a883f23de150fcb33a9765f10cf04b81f3482062`)
is concatenated with exact B9 logits. A balanced standardized 10-D logistic
readout is the candidate; the same 5-D readout on B9 logits is the calibration
control. Both use the immutable source folds, and the candidate must first
reproduce the TRAIN result in `B33-01` before one full fit. Validation inference
then consists only of the existing supervised B9 backbone, the distilled M1 and
the saved 10-D readout. No raw DINO or ConvNeXt model is present. The retained
B9 checkpoint is evaluated directly to obtain ordered validation logits and
must reproduce its historical predictions exactly.

B33 completed on commit `624b543` and failed every quality gate except the
restricted-FP nonincrease check. Exact B9, the B9-logit-only calibration
control, and B9+M1 respectively reach validation accuracy
`0.912061 / 0.912465 / 0.902783`, macro-F1
`0.876324 / 0.877177 / 0.861061`, and C1 P/R/F1
`0.642105/0.772152/0.701149`, `0.648936/0.772152/0.705202`, and
`0.621469/0.696203/0.656716`. The candidate changes C1 TP `122 -> 110` and
restricted FP `68 -> 67`; the single-FP reduction is therefore obtained by
sacrificing recall, not by learning the intended boundary correction.

The TRAIN source-fold score (`0.949318` C1 F1) was optimistic because the
readout split alone did not cross-fit its inputs: both B9 and the B32 M1 state
had already trained on all TRAIN rows. This closes exact B33 and invalidates
that screening shortcut for future fusion promotion. A valid successor needs a
primary and auxiliary state trained without the held component, followed by a
fusion rule fitted only on their fit-side outputs. Validation remains
design-exposed and test remains sealed.

- Result: `runs/b33_b9_m1_fusion_624b543_r1/summary.json`
- Summary SHA256: `01d35527392338ed1eaf58967f6b3eb1d6c76b75e9e3535e7aa12f0c1500201f`
- Validation score SHA256: `53549cfd3ce95c17e9f4abee6500294443d0f8cc50ebe17da6ebe0e64cab0d36`

## Locked cross-fitted diagnostic: B34 supervised primary + M1

B34 is the smallest check that removes B33's upstream TRAIN exposure without
training another large model. The preserved B21 spatial EMA and B29 candidate
M1 were both fitted on component folds 1--4. B34 extracts their logits/residual
on the same fit rows and held fold 0, fits matched balanced logistic readouts
only on folds 1--4, and scores fold 0 once. The 5-D primary-only readout is the
calibration control for the 10-D primary+M1 candidate. Validation and test are
never constructed.

This remains a diagnostic, not a confirmatory OOF claim: fold 0 was unseen by
the upstream optimizers and readout but its separate B21/B29 component results
were already observed. A pass only justifies the expensive next step--one
fresh fold-specific B9-like primary and a residual target defined relative to
that primary. A failure closes reuse of the current raw-DINO-aligned M1 for B9
fusion; it does not define the ceiling of a newly aligned hybrid.

## PRMR R1 closure

Matched five-epoch probe, seed 42, effective batch 48, test closed:

| Full-val FP32 condition | Control macro/C1 F1 | PRMR macro/C1 F1 | Delta macro/C1 |
|---|---:|---:|---:|
| clean | `0.847922 / 0.644068` | `0.850915 / 0.661017` | `+0.002993 / +0.016949` |
| dim | `0.805755 / 0.528302` | `0.816444 / 0.556355` | `+0.010689 / +0.028053` |
| bright | `0.819916 / 0.557789` | `0.823922 / 0.580153` | `+0.004007 / +0.022364` |
| low contrast | `0.842498 / 0.624339` | `0.846192 / 0.634146` | `+0.003694 / +0.009808` |
| center occlusion | `0.845053 / 0.658462` | `0.850068 / 0.674772` | `+0.005014 / +0.016310` |

Clean class-1 TP changes `114 -> 117`, but restricted `0/2/4 -> 1` FP only changes `80 -> 76` (`-5%`) and `2 -> 1` only `46 -> 43` (`-6.5%`). The locked gate required `-20%`; bright C1 improvement also misses the required `+0.030`. Do not tune PRMR weight again on this validation split.

Artifacts:

- Control: `runs/pretrained_dinov3_classf_prmr_r1_control_probe_20260801_prmr_r1_control_r1/postaudit/val_only/robustness_val_full_r2/robustness_summary.json`
- Candidate: `runs/pretrained_dinov3_classf_prmr_r1_probe_20260802_prmr_r1_candidate_r2/postaudit/val_only/robustness_val_full_r2/robustness_summary.json`

## Closed experiment: B10 group-tempered exposure

Keep the B2/B9 tempered class quota unchanged. Within each class, sample leakage groups uniformly, then rows uniformly within the selected `(class, group)` bucket. This is class-conditioned group balancing, not global group balancing. It must be default-off, manifest-driven, fail-closed, and must not alter raw data, splits, augmentation, loss, or model. Its matched probe control is a fresh same-commit/same-dtype B2 probe; B9 remains the 30-epoch deployment reference, not the sampler-effect comparator.

The matched deterministic BF16 probe rejected B10. Relative to B2, macro-F1 changed `0.851013 -> 0.844122`, class-1 F1 `0.657061 -> 0.626087`, and `2 -> 1` false positives `40 -> 46`; all three locked gates failed. B10 therefore is not the Hybrid foundation. Because the current manifest misses consecutive-session/crop relations, this result rejects only that manifest's group-uniform sampler; it does not reject a future source-session-safe split or weighting scheme. The hashed decision artifact is `runs/pretrained_dinov3_classf_b10_group_tempered_p05_probe_matched_bd4ae00_seed42_bf16_r1/postaudit/val_only/b10_vs_b2_metric_gate.json`.

## Closed experiment: DINOv3 spatial surface hybrid V2

Use DINOv3-S as the primary path. Four non-overlapping `2x2/2` mobile stages create exact `16x16` patch bins from RGB plus an exact-bin mean. Per-cell channel LayerNorm avoids BatchNorm leaking statistics across patches or samples. Fuse a per-patch residual **before DINO block 11 (zero-based)**, then reuse that pretrained attention/MLP block to integrate local evidence. A smooth `r/sqrt(1+||r||²)` bound scales the residual by the detached primary-patch norm, so its relative norm stays below the learned gate (`0.05` initial, `0.25` maximum). Zero-initialized branch output projections make the configured Hybrid bit-exact to native DINO at step 0 without zeroing the gate: the projection learns on step 1 and the upstream specialist/gate open after that update. Do not add the old keeper, global descriptor collapse, handcrafted texture bank, or a zero scalar gate.

Matched arms:

1. pure DINO;
2. capacity-matched generic DINO-token adapter;
3. DINO plus local RGB surface adapter;
4. the same V2 architecture with random DINO initialization.

Arms 1--3 now inherit B2's `n_c**0.5` class quota with image-uniform sampling inside each class. They do not inherit rejected B10 group-uniform sampling. The protocol IDs and run prefixes include the B2 lineage so earlier B10-based smoke artifacts cannot be mistaken for matched evidence.

The local adapter adds `29,825` parameters (`+0.138%`); its generic control adds `29,990` (difference `165`, or `0.55%` of adapter capacity). Construction is isolated with `torch.random.fork_rng`, so adding either branch cannot shift later augmentation/dropout RNG. On 20 development images, configured step-0 logits match native DINO exactly; actual-DINO backward checks show the output projection learning on step 1 and the specialist/gate learning from step 2, including with gradient checkpointing. All three current Hybrid preflights passed after this identity-preserving change; the local preflight also passed all 122 focused tests. No test inference was performed. PRMR remains off in all arms.

The random-initialized arm is only a matched initialization ablation under the fixed fine-tuning optimizer/schedule. A low score cannot establish the general value of pretraining because a fair from-scratch model would require its own train/val-tuned schedule. Run this arm only after the pretrained local adapter passes its pure/generic controls.

Current-graph engineering mobile proxy (Windows ONNX Runtime CPU, 4 threads, batch 1, 15 warmups then five alternating 100-run trials): direct DINO median/p95 `46.493/51.645 ms`, local Hybrid `49.964/55.317 ms`, median/p95 ratios `1.0747x/1.0711x`; FP32 ONNX size ratio `1.00164x`; PyTorch-to-ONNX maximum absolute error `1.79e-7` with identical argmax for both graphs. The zero-initialized projection was filled with fixed nonzero weights only to prevent constant-folding the untrained branch. This passes the `<=1.10x` architecture proxy, but a trained checkpoint still needs physical-device INT8 accuracy and latency audits before any mobile-ready claim.

Architecture tracing explicitly exports the surface descriptor/map, pre-fusion patches, gate, gated residual and post-fusion patches. For XAI, run both `--feature-source auto` (DINO patch path) and `--feature-source last_conv` (local surface branch); either view alone is incomplete.

Scientific metric gate: class-1 F1 `>=0.72`, at least `+0.010` over both pure and capacity controls, non-negative macro delta, restricted `0/2/4 -> 1` FP reduction `>=20%`, neither the `0 -> 1` nor `2 -> 1` stream may increase, class-1 TP retention `>=97%`, and no hidden test access. The executable comparator never authorizes full training by itself. A fixed post-audit must additionally show stable recall non-decreasing, relabelled recall loss no worse than `0.02`, and gate-off class-1 F1 loss at least `0.005`; otherwise the local branch has not demonstrated useful causal contribution.

The matched result rejects the local V2 branch. Exact FP32 validation gives:

| Arm | Macro-F1 | Class-1 F1 | `0 -> 1` | `2 -> 1` | `4 -> 1` |
|---|---:|---:|---:|---:|---:|
| B2 pure DINO | `0.851080` | `0.658892` | `16` | `38` | `16` |
| Generic token adapter | `0.856577` | `0.670391` | `19` | `43` | `16` |
| Local RGB adapter | `0.846433` | `0.633609` | `14` | `52` | `22` |

This V2 B2 row is one internally consistent exact export, not the later V3 control: checkpoint SHA `6ee4a39a...27411`, selected-state SHA `645da0e8...e9b47`, prediction CSV SHA `9450ff78...33aa4`. Its CSV gives `2 -> 1 = 38`; the value `40` in older gate summaries came from in-training metrics. The V3 table below deliberately uses its own separately hashed B2 control and must not be spliced into this row.

The local branch is active, not collapsed: removing it after co-adaptation lowers class-1 F1 from `0.633609` to `0.539171`. Its mean residual/token ratio is about `0.069`, versus `0.038` for the generic control, while its gate is nearly always on. On true class 2, the local branch preferentially raises class-1 probability for low-saturation, low-`R-G` images. The old six-channel input `[RGB, patch_mean_RGB]` therefore duplicates absolute colour while the free `64 -> 384` projection can override DINO semantics in an unconstrained direction. This is an architecture failure, not evidence that more CNN depth is needed.

## Locked next experiment: PR-SPR-V3

The single preregistered hypothesis is **Patch-Relative Semantic Pair Residual**. It changes both candidate and control to the same small pair-residual mechanism; the only trained-arm difference is the source of seven evidence features.

- Candidate evidence per exact `16x16` patch: RMS patch-relative RGB (3), mean chromaticity (3), and illumination-normalized luma standard deviation (1).
- Generic control evidence: a fixed seven-component summary of detached DINO patch tokens; it receives no additional RGB evidence.
- Both unit-normalize each seven-vector and use the identical bias-free `7 -> 16 -> 3` head. The zero-initialized output produces signed evidence for `1 <-> {0,2,4}`, adds only `160` trainable parameters, and keeps a zero/black padding patch at zero evidence after training.
- A residual may only follow detached classifier directions `normalize(W_1 - W_j)`. Competitor weights come from detached pre-final preview logits. There is no free `64 -> 384` projection, learned scalar gate, auxiliary loss, saturation threshold, or architecture sweep.
- Injection remains before DINO block 11. The smooth relative residual cap is fixed at `0.04`; class-3 probability supplies only a fixed safety attenuation. Step zero must be bit-exact native DINO.

The initial untrained-graph preflight passed exact native-DINO logits at step zero, real-DINO BF16 gradients opening the pair output on step 1 and its hidden layer on step 2, and strict checkpoint reconstruction after a CUDA smoke. The final parameter, ONNX and latency contracts must be regenerated from the clean committed graph before probe; earlier optimization measurements are diagnostic only.

Run exactly one BF16 seed-42 probe for each new arm under the B2 sampler, effective batch 48 and sealed test. Do not tune width, cap, descriptor or loss after reading validation. The primary endpoint is class-1 F1; macro-F1, TP retention and restricted false positives are joint safety gates.

Promotion requires all of: absolute class-1 F1 at least `0.72`; class-1 F1 at least `+0.010` over both B2 and the V3 generic control; macro-F1 delta at least `-0.002`; `2 -> 1` at least `20%` below the generic control; total `0/2/4 -> 1` at least `10%` lower; class-1 TP at least `98%` of control and no more than two lost TP; active-minus-off class-1 F1 at least `0.005`; residual/token p95 at most `0.04`; no skipped/non-finite update; added parameters below `5,000`; and mobile p95/size no more than `1.05x`/`+0.1 MB` versus B2. Failure closes V3 without same-validation retuning. Passing only authorizes a locked multi-seed replication, not test access or a scientific claim.

The executable V3 verdict accepts only full canonical validation exports with class order/support `[558,158,380,494,889]`, an identical ordered sample-identity hash, and CSV/metrics consistency. The branch-off artifact must reuse the candidate checkpoint hash and evaluate its co-adapted `model.backbone` path. Active exports record the actual residual/token distribution and separate signed `1 <-> 0/2/4` maps; the coefficient-L1 routing proxy is never reported as a learned gate.

Because near-duplicate sequences inflate the aggregate score, the post-probe mechanism audit must also report stable-label, relabelled, mixed-group, near-train-neighbour and far/boundary cohorts. Improvement confined to the near-neighbour cohort is a failure even if aggregate validation passes.

Because validation informed this design, V3 remains exploratory. A final confirmatory claim needs a new prospectively sealed source/time/site holdout. Freezing a protocol does not reopen the historical project test, which remains forbidden for selection and confirmation.

## Closed experiment: PR-SPR-V3

The locked seed-42 BF16 probe completed without non-finite or skipped updates, but the hardened full-validation FP32 verdict rejected V3:

| Arm | Accuracy | Macro-F1 | Class-1 F1 | Restricted `0/2/4 -> 1` FP |
|---|---:|---:|---:|---:|
| B2 pure DINO | `0.890278` | `0.852012` | `0.655172` | `74` |
| Pair-generic control | `0.875353` | `0.830924` | `0.597101` | `82` |
| Relative-surface candidate | `0.883017` | `0.843025` | `0.639296` | `71` |
| Candidate checkpoint, branch off | `0.883017` | `0.842870` | `0.637427` | `72` |

The descriptor is useful relative to the same-capacity generic arm (`+0.01210` macro-F1, `+0.04219` class-1 F1, restricted FP `82 -> 71`), but it does not recover B2. More importantly, the active branch changes no argmax correction and only removes one restricted FP relative to its own branch-off path. Its residual/token p95 is only `7.37e-5`, far below the allowed `0.04` cap: the adapter is starved and most of the observed difference was absorbed into the co-adapted backbone.

The mechanism audit rejects the earlier “wrong direction through block 11” hypothesis. On 25 deterministic train images, finite differences along all three `W_1-W_{0,2,4}` directions increased their intended margin in `25/25` cases; candidate directions also retain cosine `0.987--0.990` with B2. The actual design errors are that V3 restarted a full 21.6M-parameter fine-tune from the generic DINO source instead of inheriting B9, while a 160-parameter zero-initialized branch competed with that moving backbone and remained nearly closed.

Dataset cohorts do not rescue V3. Relative to B2, candidate class-1 recall is lower on stable labels (`0.5584` versus `0.5844`), relabelled labels (`0.8148` versus `0.8519`), mixed-source sessions (`0.6308` versus `0.6923`) and source-near rows (`0.6628` versus `0.7209`). It improves accuracy/macro-F1 on source-far rows (`0.8845/0.8378` versus `0.8810/0.8311`) but retains the same class-1 recall; therefore the failure is neither confined to near duplicates nor explainable by data alone.

Authoritative artifacts:

- Locked verdict: `runs/evidence_v3_exact_36eca38/locked_verdict.json`
- Exact predictions and routing: `runs/evidence_v3_exact_36eca38/{b2,generic,candidate,branch_off}`
- Train/validation integrity review (pHash radius 3): `runs/classf_integrity_review_trainval_p3_20260802`
- Final mobile proxy: `runs/engineering_v3_mobile_clean_36eca38/summary.json`

## Locked next screen: frozen-B9 Post-Norm XCNorm A0

Do not tune V3 on validation. The next single hypothesis first strict-loads the selected EMA state from B9 `best.pt`, freezes the complete DINO backbone/norm/classifier, and trains only a `3,672`-parameter local adapter. Candidate and control share `384 -> 8`, an `8 x 8 x 3 x 3` local operator, and `8 -> 3`; they differ only in normalized cross-correlation versus ordinary convolution. The pair residual is injected after the final DINO norm and before average pooling, where `W_1-W_j` has exact classifier geometry. Step-zero and branch-off logits must be bit-exact B9.

The first screen uses five immutable component-group folds derived only from train: same crop source, declared train group, adjacent source IDs and pHash distance at most 3 remain in one component. B9 has already seen all train rows, so this is a conditional adapter comparison, not an independent generalization estimate. No validation/test dataset is constructed. XCNorm must beat its equal-parameter convolution control by at least `0.020` mean pair AUROC, win at least four of five folds, retain at least `98%` class-1 TP, reduce restricted FP by at least `10%`, keep macro-F1 within `0.002`, and demonstrate a non-collapsed residual p95 in `[1e-3, 0.04)`. Failure closes A0; passing authorizes one frozen protocol evaluation on validation, not test access.

## Closed experiment: frozen-B9 Post-Norm XCNorm A0

The locked train-only run completed from clean commit `1dcc566`; it constructed neither validation nor test. B9 remains a frozen in-sample substrate here, so the absolute rows below are not end-to-end OOF generalization estimates. The scientific comparison is the paired XCNorm-versus-Conv delta under identical component folds, initialization, capacity, batches and update count.

| Train component-fold arm | Macro-F1 | Class-1 F1 | Class-1 TP | Restricted `0/2/4 -> 1` FP |
|---|---:|---:|---:|---:|
| Frozen B9, branch off | `0.955234` | `0.902486` | `472` | `74` |
| Capacity-matched Conv | `0.952637` | `0.894325` | `457` | `65` |
| XCNorm A0 | `0.953084` | `0.896750` | `469` | `77` |

XCNorm's mean pair-AUROC delta versus Conv is `-0.0000645`, with gains `-0.0000344/-0.0001548/-0.0000042` for `1` versus `0/2/4`, and only two of five fold wins. The original `+0.020` AUROC gate was incorrectly specified: Conv AUROCs are already `0.996387/0.996680/0.999938`, so the maximum possible mean gain is only `0.002335`. That impossible gate is withdrawn and is not evidence against A0. The paired AUROC interval still gives no superiority (`[-0.000224,+0.0000635]`, bootstrap probability of a positive delta `0.178`). All `980/980` paired updates completed, with zero skipped/non-finite update; branch-off error is exactly zero and residual/token p95 is active at `0.02164`.

XCNorm recovers 12 class-1 true positives relative to Conv, but creates 12 new false positives into class 1 and removes none. A 1,000-draw component-cluster bootstrap puts XCNorm-minus-Conv class-1 F1 at `[-0.00435, +0.00957]`, while its restricted-FP increase is `[+6, +19]`. Against frozen B9, the class-1 F1 delta interval is entirely negative, `[-0.01130, -0.00033]`, and macro-F1 is also negative `[-0.00406,-0.00035]`. XCNorm's pairwise training loss exceeds Conv in all `25/25` fold-epochs (one-sided sign probability `2^-25`), with median gap `+0.00798`. These feasible endpoints close A0 despite the invalid original AUROC threshold. Do not retune A0 or run its photometric/validation audits.

The paired failure isolates a technical cause that data quality alone cannot explain: both arms saw the same rows and schedule, yet Conv achieved a lower pairwise training loss in every final fold/epoch and XCNorm supplied no AUROC gain. Final post-norm DINO tokens have already been globally contextualized and normalized; a local operator applied there cannot recover missing pixel evidence, and immediate average pooling reduces it to a small decision-boundary shift. This also explains why the light pure ViT can outperform the Hybrid despite having no explicit CNN branch.

Boundary structure remains a separate modeling constraint. Train support is `[1987,497,1326,2080,2388]`; class 1 is the minority and `233/497` class-1 rows lie in a cross-class source component. Across train, `8278` rows collapse to `3117` conservative source/pHash components, of which `145` contain multiple classes and `1604` samples. Historical review/change fields describe dataset construction only. These facts require source-safe evaluation and stronger discrimination, not another label audit, and they do not excuse A0's matched loss.

The independent TRAIN-only cohort audit found no class-order or loader/manifest label-map bug. Instead, `23/25` frozen-B9 class-1 false negatives and `66/77` false positives into class 1 occur inside cross-class components; class-1 F1 is `0.975791` on single-class components, `0.825147` on cross-class components and `0.596154` on the near-crop cohort. Eighteen class-1 rows are both near-crop and source-adjacent to another class, and B9, Conv A0 and XCNorm A0 all classify only `7/18` correctly. This is a hard-representation cohort for reporting, not a review queue. Source `0` supplies `93.01%` of train and `96.18%` of class 1, which is insufficient evidence for a source router or a domain-robustness claim.

The only admissible A1 changes placement: feed the same capacity-matched Conv/XCNorm bottleneck from the normalized input of the final attention block, add its bounded patch residual in parallel with final MHSA, and let the frozen final MLP plus final norm integrate it. The backbone/head remain strict-loaded B9 and frozen. A1 uses 5,000 fixed-seed bootstrap draws that resample whole union groups within folds. XCNorm must have mean pair-AUROC delta versus Conv at least `+0.0002`, lower bootstrap bound above zero, at least four of five fold wins and no pair below `-0.0001`; class-1 F1 versus B9 must improve at least `+0.005` with lower bound above zero; macro-F1 non-inferiority lower bound must be at least `-0.002`; class-1 TP retention must be at least `98%`; every `0/2/3/4 -> 1` count must be no higher than both B9 and Conv; total non-class-1 FP into class 1 must fall at least `10%` versus B9 and its rate delta versus Conv must have bootstrap upper bound at most zero. Structural gates also lock exact branch-off, `3,672` parameters, matched initialization/response scale, residual caps and finite complete updates. Failure closes XCNorm rather than starting another width/cap/learning-rate/placement sweep.

The observed backbone is timm `Eva`, not a generic ViT abstraction. Its final block uses learned `gamma_1/gamma_2`, so the exact A1 path is `z=norm1(x)`, `u=x+drop_path1(gamma_1*MHSA(z,RoPE))`, `v=u+r(z)`, `y=v+drop_path2(gamma_2*MLP(norm2(v)))`, then final norm, average of the 256 patch tokens and the frozen head. Residual norm is capped at `4%` of `u`, the tensor actually perturbed; p95 downstream final-token perturbation must remain at most `8%`. Preflight must execute this graph on fixed train rows, hash the full selected B9 state and final-block LayerScale tensors, reproduce branch-off logits exactly and prove that gradients reach only the adapter. A post-hoc Response-Norm variant is deliberately not allowed: it would keep the already rejected late placement while changing the operator after seeing A0.

## Closed experiment: parallel-final-MHSA XCNorm A1

The sole locked A1 run completed on commit `74f046d` without constructing validation or test. All `980/980` paired updates were finite, branch-off error was zero, frozen-B9 hashes remained exact and both arms had `3,672` parameters. B9/Conv/XCNorm macro-F1 was `0.955234/0.953454/0.955799`; class-1 F1 was `0.902486/0.894789/0.905950`; false positives into class 1 were `77/65/73`. XCNorm recovered 17 class-1 true positives relative to Conv but removed no Conv FP and created eight new FP (`0:+4, 2:+3, 3:+1`). Its FP-rate delta versus Conv was strictly positive under the 5,000-draw component bootstrap, `+0.001028 [ +0.000384,+0.001810 ]`.

XCNorm's mean pair-AUROC gain over Conv was only `+0.0000282`, interval `[-0.0000518,+0.0001024]`, with three of five fold wins. Its class-1 F1 gain over B9 was `+0.003464`, interval `[-0.000607,+0.008058]`, below the locked effect size and not distinguishable from zero. XCNorm pair-BCE exceeded Conv in all `25/25` fold-epochs (median gap `+0.004179`, sign probability `2^-25`). Pure-component class-1 F1 was unchanged from B9 at `0.975791`; all eight new FP versus Conv and 16 of 17 recovered TP were in mixed-label components. XCNorm's injection p95 was only `0.01718` versus Conv `0.03096`. These results close the exact A1 and the local XCNorm family: normalization discarded useful magnitude/contrast, and the branch mostly returned to B9's operating point rather than creating new discrimination. Do not sweep width, cap, learning rate or another late placement.

The next family must not repeat the rejected simple ordinal/cumulative heads already recorded in `TODO_TRKH_5CLASS.md`. The admissible hypothesis is a lightweight ambiguity-aware bridge on the preserved B9 teacher: keep class `3/4` categorical, model the ordered `0/1/2` boundaries, learn a sample-visible uncertainty gate from component-conflict supervision, and compare against an equal-parameter categorical control under the same source-union OOF/bootstrap. Group distributions are soft supervision only, never automatic relabelling or inference metadata. This direction is supported by work on adjacent-category sequence decisions ([Ord2Seq, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Ord2Seq_Regarding_Ordinal_Regression_as_Label_Sequence_Prediction_ICCV_2023_paper.html)), ordinal learning under class-conditional label noise ([PMLR 2020](https://proceedings.mlr.press/v129/garg20a.html)), and ordinal label-distribution learning ([ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wen_Ordinal_Label_Distribution_Learning_ICCV_2023_paper.html)). It remains train-only until a separately preregistered gate passes.

## Locked next screen: B11 conflict-gated adjacent-energy residual

`B11-CGAER-A0` is a representation--decision Hybrid rather than another CNN graft. It freezes B9 and consumes only its mean post-final-norm patch feature `z in R^384` and logits `b in R^5`. Both arms use fixed non-affine LayerNorm followed by `384 -> 8 -> 5`, SiLU and a zero-initialized output layer: exactly `3,125` trainable parameters and `3,112` dense MACs per sample. A disabled branch returns `b` directly. The candidate interprets its outputs as ordered location, spread, two categorical contrasts and conflict logit `(m~,s~,c1~,c2~,u)`:

`m=2*tanh(m~)`, `alpha=1+0.5*tanh(s~)`, `g=sigmoid(u)`, `T=1+g`, and for centers `mu=(-1,0,1)`, `phi_k=-alpha*(m-mu_k)^2/(2*T^2)`. Center `phi-phi_at_(m=0,alpha=1,g=0.5)` over classes `0/1/2`, then subtract `(g-0.5)` times the centered B9 `0/1/2` logits. Two fixed orthonormal categorical contrasts retain unrestricted cluster-versus-`3/4` and `3`-versus-`4` corrections. A smooth L2 trust region `r/sqrt(1+(||r||_2/0.5)^2)` caps every five-logit residual below `0.5` without changing its direction. At zero output this graph is B9-exact.

The matched control has the identical trunk, gate, categorical contrasts, trust region, initialization, batches and objective, but replaces ordered energy with two free zero-sum categorical contrasts over `0/1/2`. Their fixed basis gains equal the two candidate zero-state Jacobian singular values; the control coordinates themselves remain unbounded before the common trust region. Preflight must numerically prove this Jacobian match. Consequently a candidate win cannot be attributed to parameter count, initial response scale or a weaker control.

The union-component graph and five folds remain the immutable A0/A1 label-independent construction; assignment CSV SHA256 is `afde4fec6b344b867d31e0e01b89f16c5f4be7fd551fdc1fdfd0ea34219d725f`. For component `C`, let `p_C` be its empirical label histogram and let `q_C=(1-sum_k p_Ck^2)/(1-1/r_C)`, where `r_C` is the number of represented labels. Pure non-singletons receive conflict target zero; singletons are masked. Both arms optimize hard one-hot five-way CE plus `0.10 * BCEWithLogits(u,q_C)`; the ambiguity term weights each eligible row by inverse component size before normalization so a long source/pHash chain cannot dominate it. The histogram never becomes a class target or model input, so this is neither CORN/cumulative decoding nor another soft-label experiment. At inference only image-derived `z` and `b` exist.

The single screen uses FP32, seed `20260731`, five fixed epochs, AdamW `lr=1.5e-4`, weight decay zero, batch `128`, tempered class power `0.5`, identical sampled indices and final-epoch OOF only. It reuses the exact lossless A0 token/logit/label cache (`tokens SHA256 7d720ef0f6897d5339ac060ac0cc70d8e718621aaee00e473108c91fe8304dbd`) and constructs no dataset object. Statistical intervals use 5,000 fixed-seed paired bootstrap draws of whole union components within folds.

Promotion requires every gate: candidate class-1 F1 improves by at least `+0.005` versus both B9 and control with both bootstrap lower bounds above zero; at least four of five fold wins versus control; class-1 TP is at least `98%` of B9 and no lower than control; total non-class-1 FP into class 1 is at most `90%` of B9 and no higher than control, with its rate-delta upper bounds below zero versus B9 and at most zero versus control; each `0/2/3/4 -> 1` count is no higher than both; macro-F1 lower bounds are at least `-0.002` versus both; mixed-versus-pure gate AUROC is at least `0.75` with lower bound at least `0.70`; the gate ranks the binary event “this row is a class-1 FP or FN” against all train rows with AUROC at least `0.75`; fixing `g=0.5` loses at least `0.002` class-1 F1 without reducing FP; residual p95 is in `[0.01,0.5)`; and all paired updates, hashes, branch-off and finite-value checks pass. The error-triage AUROC is not a claim that uncertainty alone knows the FP-versus-FN correction direction; the ordered-energy coordinates must prove that through the matched F1/TP/FP gates. This population clarification was committed after the first formal preflight created no artifact and before any candidate OOF update; it preserves the originally measured train-only diagnostic rather than silently replacing it with the narrower conditional-boundary AUROC. Failure closes exact B11 without width/LR/loss/cap/gate sweeps. Passing authorizes a separately frozen validation protocol, not test access or a generalization claim. ONNX, CPU/mobile and robustness audits occur only after this train-fold gate passes.

## Closed experiment: B11 conflict-gated adjacent-energy residual

The one locked FP32 TRAIN-only run on commit `6a008d7` completed all `1305/1305` paired updates with zero skipped/non-finite update, exact branch-off and verified cache/fold/source hashes. It constructed neither validation nor test. The accepted preflight already showed that a fixed linear score in frozen B9 space can rank mixed versus pure components (`AUROC 0.825639`) and the event “row is a class-1 FP or FN” versus all train rows (`0.846424`); the narrower FP-versus-FN boundary population was only `0.577538` and was never promoted as a directional claim.

| TRAIN-only OOF arm | Accuracy | Macro-F1 | C1 P/R/F1 | C1 TP/FN | FP into C1 (`0/2/3/4`) |
|---|---:|---:|---:|---:|---:|
| Frozen B9 | `0.966900` | `0.955234` | `0.859745/0.949698/0.902486` | `472/25` | `77` (`36/34/3/4`) |
| Equal-parameter categorical control | `0.963397` | `0.950075` | `0.889113/0.887324/0.888218` | `441/56` | `55` (`23/26/1/5`) |
| B11 candidate | `0.963155` | `0.950157` | `0.870906/0.909457/0.889764` | `452/45` | `67` (`27/31/2/7`) |
| Candidate with gate forced to `0.5` | — | `0.951093` | `—/—/0.888889` | `444/53` | `58` |

B11 loses class-1 F1 versus B9 by `-0.012722`, with a 5,000-draw component-bootstrap interval entirely below zero `[-0.024204,-0.002457]`; macro-F1 also falls by `-0.005077 [-0.008498,-0.001932]`. Its tiny class-1 F1 delta over the control is inconclusive, `+0.001546 [-0.006033,+0.009470]`, with only two of five fold wins. Although it removes 13 B9 false positives, it loses 20 B9 true positives. Against the control it recovers 11 true positives but creates 12 false positives and removes none; its FP-rate interval is strictly worse, `+0.001542 [+0.000665,+0.002597]`.

The mechanism failed, not merely the promotion threshold. Candidate residual p95 reaches `0.468004` under a `0.5` cap, yet its learned gate inverts the preregistered signal: mixed-versus-pure AUROC is `0.480222 [0.399984,0.553564]` and class-1 error-event AUROC is `0.378410 [0.324786,0.437553]`. Forcing the gate to `0.5` changes class-1 F1 by only `+0.000875 [-0.005746,+0.007584]`. Pair AUROCs remain effectively unchanged around `0.996--1.000`, so the pooled post-norm B9 feature contains an uncertainty ranking but supplies no learned correction direction. End-to-end CE repurposed the shared gate as a class-routing coordinate while its auxiliary BCE memorized the training folds. A separately frozen/two-stage gate could recover triage AUROC, but cannot solve this missing-direction result and is therefore not an admissible B12.

This closes the exact B11 family and the broader strategy of correcting B9 with a tiny head after final pooling. No validation, test, robustness, ONNX, or mobile audit is authorized from B11. The next experiment must either expose genuinely new pre-pooling/local evidence with a measured precondition, or add independently defensible supervision; it must not retune B11's width, cap, loss weight, learning rate, gate, or ordinal coordinates.

Authoritative artifacts:

- Accepted B11 preflight SHA256 `04ba96201dea663ab1e47f3dc51a8197746fc66bab1c869545aaf3f5b39c3745`: `runs/preflight_cgaer_b11_6a008d7_r1/preflight.json`
- Closed B11 summary SHA256 `b63b0100a060e4272d5465deaa204dbe7277a6d6f140451d09fadd61d2aca5f5`: `runs/pretrained_dinov3_classf_b11_cgaer_sourcefold_6a008d7_r1/summary.json`
- Closed B11 OOF SHA256 `95ee6a1c990cd48f55530ee9a2355dfb0f7c53a5a6d2b1ed6897af581ebdb5c8`: `runs/pretrained_dinov3_classf_b11_cgaer_sourcefold_6a008d7_r1/train_oof_logits.npz`
- Accepted end-to-end A1 integration preflight SHA256 `2039c1165a0d6d44fe460fb54ac1aa1397942b1920d042f97dc2cb4b7d7fa0d6`: `runs/preflight_xcnorm_a1_actual_b9_287ab18_r1/preflight.json`
- Closed A1 summary SHA256 `1d07c0afdf46d57e80b6b516f1c0004101a39c0b8bdbc0dbd2bb7a906861a10`: `runs/pretrained_dinov3_classf_xcnorm_a1_sourcefold_74f046d_r1/summary.json`
- Closed A1 OOF SHA256 `c49bcdad986805c25ad7f13830dd678291cf34bcdf305ce477f9c3eb10702e6b`: `runs/pretrained_dinov3_classf_xcnorm_a1_sourcefold_74f046d_r1/train_oof_logits.npz`
- Summary SHA256 `628712ae01f2d7edff02447cf1c42a7992114c0de90c00fe5adf697b75fca109`: `runs/pretrained_dinov3_classf_xcnorm_a0_trainfold_1dcc566_r1/summary.json`
- OOF logits SHA256 `5f982bb8119bbbba0367288c4a5d6f9ac3a7f1c2e2b47ad51a8d49e02075cf49`: `runs/pretrained_dinov3_classf_xcnorm_a0_trainfold_1dcc566_r1/train_oof_logits.npz`
- Fold assignment SHA256 `afde4fec6b344b867d31e0e01b89f16c5f4be7fd551fdc1fdfd0ea34219d725f`: `runs/pretrained_dinov3_classf_xcnorm_a0_trainfold_1dcc566_r1/train_fold_assignments.csv`

## Closed experiment: B12 same-patch depth trajectory

B12 tested one fixed hypothesis: intermediate DINOv3 block trajectories from the same patch might add class-1 boundary information that is absent from the latest block. The corrected R2 run reproduced the immutable B9 cache bit-exactly (`max_abs=0`) and used only TRAIN descriptors, the fixed five source/union-component folds, a preregistered linear readout and whole-component bootstrap. It constructed neither validation nor test.

| TRAIN-only pair screen | Mean C1-vs-rival AUROC | C1 TP | FP into C1 |
|---|---:|---:|---:|
| Raw B9 margin | `0.997776` | `1466` | `77` |
| Latest-block control | `0.997424` | `1457` | `76` |
| Deranged-depth control | `0.997511` | `1455` | `70` |
| B12 trajectory candidate | `0.997589` | `1458` | `71` |

The candidate's gain over the latest-block control is only `+0.000165 [-0.000023,+0.000380]`, and over the deranged control only `+0.000077 [-0.000101,+0.000265]`. It is significantly worse than the native B9 margin by `-0.000188 [-0.000355,-0.000021]`. Four of five fold means improve over latest, but only eight of fifteen rival-fold comparisons are positive. Recall/FP safety passes; every information-effect gate fails. Therefore the exact B12 trajectory family is closed: a larger SSM or attention module over these depth tokens would add capacity to relearn a signal already better represented by B9, not repair a demonstrated missing feature.

- Accepted R2 preflight SHA256 `954b9db6daf3965bc3a1adfd6ae08fcabc36021d9bb0f6782b56fbd97c3c33ea`: `runs/preflight_b12_depth_trajectory_ae6b70a_r2/preflight.json`
- Closed B12 summary SHA256 `083eca35eb2b6b78299c9d6c3130b2c6959f5de3533e7e71d25c759904e3775d`: `runs/pretrained_dinov3_classf_b12_depth_trajectory_signal_ae6b70a_r2/summary.json`
- B12 feature SHA256 `fb7afeac35ad76e356ef7ced6691d20c58a28adf759e7bbee125121e132a06ec`; OOF SHA256 `d910851e2b793f38bf131a6aa8be693e61b368ea60a019640a447cea84ff44dc`.

## Closed experiment: B13 EfficientViM-M1 final feature

B13 compared raw pretrained representations on the same `8278` TRAIN rows, immutable five component folds, fold-local balanced linear readout and 5,000 paired whole-component bootstrap draws. It constructed neither validation nor test.

| TRAIN-only OOF arm | Accuracy | Macro-F1 | C1 P/R/F1 | Mean C1-vs-rival AUROC | Restricted FP rate |
|---|---:|---:|---:|---:|---:|
| DINOv3-S final patch mean, 384-D | `0.858178` | `0.802076` | `0.467532/0.579477/0.517520` | `0.946103` | `0.055604` |
| EfficientViM-M1 final spatial feature, 320-D | `0.826407` | `0.769381` | `0.416036/0.553320/0.474957` | `0.931949` | `0.065076` |

Candidate-minus-DINO intervals are macro-F1 `-0.032695 [-0.050192,-0.014300]`, C1 F1 `-0.042563 [-0.085055,-0.000096]`, and mean pair AUROC `-0.014154 [-0.022956,-0.005367]`; all three locked information gates fail. Recall retention (`0.9549x`) and restricted-FP control (`1.1703x`) pass, so this is not a C1-collapse result. Loss is broader: `2->1` rises `118->140`, `4->1` `40->54`, while correct class-2/class-4 predictions fall `1065->975` and `2186->2086`.

The mobile precondition passes: the 5-class M1 graph has `0.264964x` DINO parameters, standard-domain ONNX parity, and Windows ORT CPU median/p95 latency ratios `0.074249/0.076133`. These are eligibility proxies, not a physical-phone claim. Post-hoc error overlap is descriptive only: prediction agreement is `81.05%`; M1 alone fixes `581` DINO errors while breaking `844` DINO-correct rows, including `79/92` respectively for true C1. This complementarity does not authorize an ensemble or a score sweep.

The exact scientific conclusion is narrow: a final-stage-only M1 transfer route loses fine-grained information; it does not prove that EfficientViM or SSM architectures are intrinsically inferior. Official M1 assigns only `0.50765` fusion mass to its final feature and `0.49235` to the other three stages, but that fact and the four-stage interface were already known and explicitly excluded before formal B13. They cannot be used post-hoc to reopen this `class_f` evidence line. The process error was selecting a non-canonical final-only interface at B13 design time; a future architecture screen must preregister its canonical transfer interface and conditional sequence before any metric is observed, on fresh prospectively locked evidence.

- Accepted preflight SHA256 `51178282afa4a0c2283ea3f2b774beb23ab75082dc7e919b99fa20faf9870b96`: `runs/preflight_b13_efficientvim_frozen_transfer_f229649_r1/preflight.json`
- Closed summary SHA256 `b017d34955f1c9126a611cf064cb919b6696ed9a19bf0f2b676838ad97955a75`: `runs/pretrained_efficientvim_classf_b13_frozen_transfer_f229649_r1/summary.json`
- OOF SHA256 `82bcef6d293223b819f559122bffee822b7f2a2c80a200113dd6c0b7843def39`; final-feature SHA256 `b7da2700eb317c97adba6106b3fd5a7d38be290d4d483c94f10e4387ecb59014`.

## Completed cleanup: pretrained Tier A

The reviewed cleanup copied and rehashed `21` small evidence artifacts (`3,197,478` bytes), then rehashed and deleted ten exact superseded/incomplete roots totaling `9,073,944,908` bytes. All 14 protected roots and 125 anchor hashes survived; remaining free space was `37.055 GiB`. Tier B-D remain untouched.

- Inventory SHA256 `94323fb180e5f05cdb2345e72adcaaf09e71607855042e8a0c850deefd704c93`
- Preserved-evidence manifest SHA256 `9b67cb7d6133fd2c73feb2308d38f7dfff012af413b4dfeed466295b153a7417`
- Execution result SHA256 `4808dd861d2d5cf9bfd6f96d65b064765126a2ad9c6266d3797fece3392c4d48`: `runs/cleanup_pretrained_tier_a_20260804`

## Completed cleanup: pretrained Tier B

The reviewed Tier-B manifest copied and rehashed `36` configuration/metric/history/stage files (`730,695` bytes), then deleted exactly six obsolete one-epoch smoke roots totaling `3,184,449,161` bytes. The measured free-space increase was `3,183,550,464` bytes; all 18 B9/B13/B14/assets/data/presentation anchors passed before and after deletion.

- Inventory SHA256 `043dfba698d9092842c5191a2bc368acb8b697c522a554c8f3f593966d70a40f`
- Preserved-evidence manifest SHA256 `09e04951db9c5f7740c6ad558d52cf911f1750859629826f9b0f212f6d8c593a`
- Execution result SHA256 `1c0f77c8e4112bd1bc6a0134fd38c90eccb336e8733b61aeedb27fc0644a8fbe`: `runs/cleanup_pretrained_tier_b_20260804`

## Completed cleanup: pretrained Tier C

The twice-audited exact manifest preserved and rehashed 73 evidence copies, then deleted 23 superseded descriptor/checkpoint files totaling `15,584,026,418` bytes. All candidates are absent; all 29 protected anchors and all evidence source/copy pairs passed post-execution verification. Measured free-space gain was `15,584,075,776` bytes.

- Plan SHA256 `009b87450bb1e425f64dc00308fe7fed2857be201c6cff379b0c9f3e1789f8b4`; evidence manifest SHA256 `18367d350bdbbe7af2d668d64536f3f98fedbb08c56e1b38423fcce969dbd687`
- Execution result SHA256 `28f8c7a115ace12bc0812e5eb75dae8d4324204be90c6bf3923c6b3c892b7274`: `runs/cleanup_pretrained_tier_c_20260805/cleanup_result.json`

## Completed cleanup: pretrained Tier D

An independently reviewed literal allowlist deleted exactly 16 obsolete
optimizer/resume `last.pt` checkpoints totaling `5,084,665,336` bytes. All
16 `best.pt` files and the other 534 affected-run artifacts remained in place;
153 compact evidence copies, 38 project anchors and all six B16
failure/diagnostic files passed post-verification. Measured free-space gain was
`5,084,692,480` bytes. No directory, wildcard, age rule or recursive deletion
was used.

- Plan SHA256 `bdf80119bea015bfe222f2664c810374dfa77e31fe8cedd5c77e2cf4abaa0bda`
- Execution result SHA256 `978433fdbb1b95882c4eb78c16520436cdb5a507ccc7ffa07ad7326d51e33285`: `runs/cleanup_pretrained_tier_d_20260805/cleanup_result.json`

## Retired non-gating boundary-review packet

The previously prepared blinded packet is retained as historical diagnostic
evidence only. Do not send it to reviewers, do not use it to challenge or edit
`class_f`, and do not gate architecture work on it. Its pHash/component records
remain useful solely for source-safe TRAIN folds and cohort reporting. Canonical
labels are user-authoritative real-world ground truth.

## Process corrections

- Never compare scores across yolo_f and canonical class_f as if they were matched.
- Resolve checkpoint preprocessing in independent exporters; an ImageFolder alphabetical transform/order is not a valid substitute for the trainer's data contract.
- Record total/trainable/frozen parameters separately; trainable-only count is not deployment size.
- Add new architecture code in a separate module. Do not expand the existing `model.py`/`train.py` monolith with another large implementation.
- Reject any promotable hybrid that injects after the last transformer block and immediately average-pools: A0 used this location only as an explicitly conditional operator screen and confirmed why it must not be a final architecture.
- Matched arms must preserve post-construction RNG state; a shared numeric seed is insufficient if architectures consume different initialization RNG.
- Probe/full recipes pin `--deterministic` and require an explicit AMP dtype; `auto` is rejected so BF16/FP16 cannot silently change across GPUs under one recipe hash.
- A numerical replay must also pin evaluation batch size, backend flags and the
  exact forward/pooling path. B22 showed that BF16 batch `64` versus `32` can
  flip near-boundary argmax values with the same checkpoint.
- If the claim is improvement of B9, strict-load its selected supervised EMA or
  first produce a matched fold-specific pure checkpoint. Raw DINO plus a reset
  head is not B9 inheritance.
- A short screen whose last LR remains near its peak closes only that short
  optimization recipe. It cannot support an asymptotic architecture-ceiling
  claim; any successor must change the initialization/optimization hypothesis
  prospectively rather than silently extending the failed run.
- Archive rejected one-shot tools only through a hashed manifest; do not mass-delete research evidence.
- `build_classification_grouped_split.py` now groups adjacent `Image_N` frames label-independently (`9bb1232`) and has a cross-class transition regression test. There is still no evidence that this tool generated canonical `class_f`; the fix is a forward provenance guard, not a retrospective corruption claim.
- Use adjacency/pHash/union graphs only for fold exclusion and component bootstrap. Mixed-component status is not an ambiguity label: only `519/1604` mixed rows are direct endpoints of a cross-label relation, while `1085/1604` inherit the status transitively; prevalence is also class-confounded (`46.88%` for class 1 versus `7.08%` for class 4). This explains why B11's fixed component-Gini diagnostic can look strong while its learned OOF gate reverses.
- A committed protocol stop rule overrides a later refocus-table interpretation. If a canonical multi-stage/readout alternative is already known, preregister its conditional sequence before the first formal result; do not reopen it because the excluded arm becomes attractive after failure.
