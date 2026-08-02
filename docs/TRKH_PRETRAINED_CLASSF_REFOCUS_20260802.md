# TRKH pretrained class_f: refocus decision (2026-08-02)

## Decision

- Keep `class_f` immutable and keep the test split sealed.
- Close PRMR R1 as **diagnostic-positive but not promotable**.
- Do not reuse the two-backbone A0 hybrid.
- Close class-conditioned group exposure after its matched failure; isolate a spatial surface adapter on top of the winning B2 pure-DINO path.

## Evidence, not assumptions

- Canonical contract: 12,019 images; train/val/test `8278/2479/1262`; no declared `source_image` or `leakage_group` crosses a split.
- Train has `8278` rows but only `5361` leakage groups. Validation has `2479/2471`; test has `1262/1262`. Repeated views are therefore concentrated in train.
- The current tempered sampler operates on rows. Under its quota, one class-2 group can be replayed about 54 times per epoch and one class-3 group about 41 times.
- Class 1 is `738/12019 = 6.14%`; `116/497` class-1 train rows belong to groups containing another label. This proves an ambiguity/provenance risk, not that those labels are wrong.
- The best exact B9 full-val export is accuracy `0.912061`, macro-F1 `0.876324`, class-1 P/R/F1 `0.642105/0.772152/0.701149`. Its class-1 false positives are `2->1: 41`, `0->1: 21`, `4->1: 6`.
- Class 1 is not one uniform failure cohort. B9 recall is `0.852` on recently relabeled class-1 samples but only `0.688` on the stable cohort. Frozen-DINO embeddings also place stable class 1 near class-0 false positives and relabeled class 1 near class-2 false positives. Do not globally oversample class 1 or auto-relabel it.
- The two false-positive streams mirror those cohorts photometrically: B9 `0 -> 1` errors have red-minus-green mean `-0.0005` versus stable class 1 `+0.0017`; `2 -> 1` errors are `+0.0248` versus relabelled class 1 `+0.0150`. Brightening raises `0 -> 1` from `21` to `128`, while dimming raises `2 -> 1` from `41` to `72`. This supports illumination-normalized local fruit-surface evidence, not a provenance flag at inference.
- Exact-file integrity is good, but the declared grouping is not session/crop safe. It reports no cross-split group or SHA-exact overlap, yet adjacent IDs from the same source form 1,114 same-label train/val pairs; 193 have crop pHash distance `<=8` versus zero in a random control, and B9 is correct on all 196 val rows with such a same-label train neighbour. Conversely, train contains 105 declared groups mixing class 1 with another label, affecting `116/497` class-1 rows; `88/497` have a cross-label crop within pHash `<=8`. `_source_family` currently treats consecutive frames such as `Image_10383` and `Image_10384` as different families. Current validation is therefore suitable for matched exploratory ablation, not a clean generalization claim. The older `docs/dataset_leak_audit_5class` describes 13,088 rows and is stale for the current 12,019-image corpus; do not cite it.
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
| B2 pure DINO | `0.851080` | `0.658892` | `16` | `40` | `16` |
| Generic token adapter | `0.856577` | `0.670391` | `19` | `43` | `16` |
| Local RGB adapter | `0.846433` | `0.633609` | `14` | `52` | `22` |

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

## Process corrections

- Never compare scores across yolo_f and canonical class_f as if they were matched.
- Resolve checkpoint preprocessing in independent exporters; an ImageFolder alphabetical transform/order is not a valid substitute for the trainer's data contract.
- Record total/trainable/frozen parameters separately; trainable-only count is not deployment size.
- Add new architecture code in a separate module. Do not expand the existing `model.py`/`train.py` monolith with another large implementation.
- Reject any hybrid that injects after the last transformer block and immediately average-pools: spatial tensor shape alone does not imply spatial interaction.
- Matched arms must preserve post-construction RNG state; a shared numeric seed is insufficient if architectures consume different initialization RNG.
- Probe/full recipes pin `--deterministic` and require an explicit AMP dtype; `auto` is rejected so BF16/FP16 cannot silently change across GPUs under one recipe hash.
- Archive rejected one-shot tools only through a hashed manifest; do not mass-delete research evidence.
