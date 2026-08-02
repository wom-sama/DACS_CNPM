# TRKH pretrained class_f: refocus decision (2026-08-02)

## Decision

- Keep `class_f` immutable and keep the test split sealed.
- Close PRMR R1 as **diagnostic-positive but not promotable**.
- Do not reuse the two-backbone A0 hybrid.
- Close class-conditioned group exposure and PR-SPR-V3 after their matched failures.
- Preserve B9 pure DINO as the current presentation/teacher reference. XCNorm A0/A1 and B11-CGAER are closed by matched TRAIN-only screens. Do not launch another late pooled-feature adapter, uncertainty router, or width/loss/LR sweep on this representation. Mobile promotion still requires a later locked distillation and device audit.

## Evidence, not assumptions

- Canonical contract: 12,019 images; train/val/test `8278/2479/1262`; no declared `source_image` or `leakage_group` crosses a split.
- Train has `8278` rows but only `5361` leakage groups. Validation has `2479/2471`; test has `1262/1262`. Repeated views are therefore concentrated in train.
- The current tempered sampler operates on rows. Under its quota, one class-2 group can be replayed about 54 times per epoch and one class-3 group about 41 times.
- Class 1 is `738/12019 = 6.14%`; `116/497` class-1 train rows belong to groups containing another label. This proves an ambiguity/provenance risk, not that those labels are wrong.
- The best exact B9 full-val export is accuracy `0.912061`, macro-F1 `0.876324`, class-1 P/R/F1 `0.642105/0.772152/0.701149`. Its class-1 false positives are `2->1: 41`, `0->1: 21`, `4->1: 6`.
- Class 1 is not one uniform failure cohort. B9 recall is `0.852` on recently relabeled class-1 samples but only `0.688` on the stable cohort. Frozen-DINO embeddings also place stable class 1 near class-0 false positives and relabeled class 1 near class-2 false positives. Do not globally oversample class 1 or auto-relabel it.
- The two false-positive streams mirror those cohorts photometrically: B9 `0 -> 1` errors have red-minus-green mean `-0.0005` versus stable class 1 `+0.0017`; `2 -> 1` errors are `+0.0248` versus relabelled class 1 `+0.0150`. Brightening raises `0 -> 1` from `21` to `128`, while dimming raises `2 -> 1` from `41` to `72`. This supports illumination-normalized local fruit-surface evidence, not a provenance flag at inference.
- Exact-file integrity is good, but the declared grouping is not session/crop safe. It reports no cross-split group or SHA-exact overlap, yet adjacent IDs from the same source form 1,114 same-label train/val pairs; 193 have crop pHash distance `<=8` versus zero in a random control, and B9 is correct on all 196 val rows with such a same-label train neighbour. Conversely, train contains 105 declared groups mixing class 1 with another label, affecting `116/497` class-1 rows; `88/497` have a cross-label crop within pHash `<=8`. `_source_family` currently treats consecutive frames such as `Image_10383` and `Image_10384` as different families. Current validation is therefore suitable for matched exploratory ablation, not a clean generalization claim. The older `docs/dataset_leak_audit_5class` describes 13,088 rows and is stale for the current 12,019-image corpus; do not cite it.
- The stricter TRAIN-only mechanism audit separates real supervision conflict from provenance suspicion. There is no class-map/loader failure and no exact-SHA cross-label duplicate, but `120` luminance-pHash-radius-3 pairs cross labels, including `55` involving class 1. Of `30` pHash-identical cross-label pairs, six are also near-identical in RGB (`SSIM >= 0.95`, `MAE <= 0.01`); B9 gives the same prediction to `27/30`, and every pair has at least one error. Class 1 also has strong policy churn: `264/497` final rows were relabelled and `247/497` previously carried class 2. These are human-review evidence, never automatic relabel evidence. Numeric adjacency is not a temporal target: among `159` cross-label adjacent-ID pairs, label direction is balanced (`81` increasing, `78` decreasing), and manifest `source` equals the crop box index for all `8278` train rows rather than a fruit/track ID. Sequence/ordinal label propagation is therefore forbidden until immutable fruit-track/session/time metadata and double-blind consensus exist.
- The historical A0 hybrid has no canonical-class_f comparison. Its only yolo_f smoke trained `691,841/29,524,375` parameters, collapsed DINO patch geometry into global mean/std, and ended with an effective gate near `2.4e-5`.

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

Because validation informed this design, V3 remains exploratory. A final confirmatory claim needs a new source/time/site holdout; the current test split stays sealed until the complete protocol is frozen.

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

Dataset quality remains a separate ceiling. Train support is `[1987,497,1326,2080,2388]`; class 1 is the minority, `330/497` class-1 rows carry a train-review flag, `264/497` were relabelled, `247/497` originated from class 2, and `233/497` lie in a mixed-label source component. Across train, `8278` rows collapse to only `3117` conservative source/pHash components, of which `145` mix labels and contain `1604` samples. These facts require source-safe evaluation and human review; they do not justify changing labels automatically or blaming A0's matched loss on the dataset.

The independent TRAIN-only cohort audit found no class-order or loader/manifest label-map bug. Instead, `23/25` frozen-B9 class-1 false negatives and `66/77` false positives into class 1 occur inside mixed-label components; class-1 F1 is `0.975791` on pure components, `0.825147` on mixed components and `0.596154` on the cross-label near-crop cohort. Eighteen class-1 rows are both near-crop and source-adjacent to another label, and B9, Conv A0 and XCNorm A0 all classify only `7/18` correctly. This is the smallest high-value double-blind review cohort; pHash was computed from luminance only, so it is a review priority rather than evidence for automatic relabelling. Source `0` supplies `93.01%` of train and `96.18%` of class 1, which is insufficient evidence for a source router or a domain-robustness claim.

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

## TRAIN-only blinded boundary review queue

The remaining data question is now a bounded human-review task, not an excuse for architecture sweeps. A sealed queue contains the 18 class-1 rows that are simultaneously endpoints of a cross-label `pHash<=3` relation and an adjacent-ID relation, plus six RGB-near-identical cross-label pairs (`pHash=0`, global SSIM at least `0.95`, RGB MAE at most `0.01`). B9 is correct on only `7/18` priority class-1 rows, but its predictions are hidden from the contact sheet to prevent reviewer anchoring. Two independent blinded reviewers and adjudication are required; no automatic relabeling is permitted.

- Review directory: `runs/classf_train_boundary_review_priority_20260802`
- Priority-P1 CSV SHA256 `6103ee946fb48147499fb167e23a8407614c04cf85d49b57c4e1d9e3cad20b46`
- Priority-P2 CSV SHA256 `7987a7c639df84e494c4603eca10712c1d510775c88ee8078aff3fba91aec56d`
- Contact-sheet SHA256 `c8bd6a05d984bb0f078185b48c5d16b02e5e369ff1a10e12946d75595bc324d7`; review-summary SHA256 `7cb6efe68c2677bcd3a37e69937d9d27331eec2ce56051b1162e58c61fe88609`.

## Process corrections

- Never compare scores across yolo_f and canonical class_f as if they were matched.
- Resolve checkpoint preprocessing in independent exporters; an ImageFolder alphabetical transform/order is not a valid substitute for the trainer's data contract.
- Record total/trainable/frozen parameters separately; trainable-only count is not deployment size.
- Add new architecture code in a separate module. Do not expand the existing `model.py`/`train.py` monolith with another large implementation.
- Reject any promotable hybrid that injects after the last transformer block and immediately average-pools: A0 used this location only as an explicitly conditional operator screen and confirmed why it must not be a final architecture.
- Matched arms must preserve post-construction RNG state; a shared numeric seed is insufficient if architectures consume different initialization RNG.
- Probe/full recipes pin `--deterministic` and require an explicit AMP dtype; `auto` is rejected so BF16/FP16 cannot silently change across GPUs under one recipe hash.
- Archive rejected one-shot tools only through a hashed manifest; do not mass-delete research evidence.
- Do not reuse `build_classification_grouped_split.py` for a prospective split until its adjacent-frame grouping is made label-independent: its current class-conditioned grouping can separate a physical `0/1/2` transition sequence. There is no evidence that this tool generated canonical `class_f`; this is a forward provenance guard, not a retrospective corruption claim.
- Use adjacency/pHash/union graphs only for fold exclusion and component bootstrap. Mixed-component status is not an ambiguity label: only `519/1604` mixed rows are direct endpoints of a cross-label relation, while `1085/1604` inherit the status transitively; prevalence is also class-confounded (`46.88%` for class 1 versus `7.08%` for class 4). This explains why B11's fixed component-Gini diagnostic can look strong while its learned OOF gate reverses.
