# B15 aligned prefix-to-patch pooling protocol — 2026-08-04

Protocol ID: `TRKH_PRETRAINED_CLASSF_B15_ALIGNED_PREFIX_PATCH_POOL_20260804`

## Question and claim boundary

B15 asks one prospective TRAIN-only mechanism question: do the five final DINOv3-S prefix tokens contain image-matched query information that can recover class-1 boundary evidence discarded by uniform averaging of the 256 final patch tokens? The single candidate is a frozen, parameter-free prefix-to-patch pooling rule. Its primary comparator is the exact raw-DINO uniform-patch-mean arm from B13; prefix-only pooling and one fixed union-group-disjoint derangement are falsification controls.

The candidate is not a pretrained-model sweep and it does not add a CNN, SSM, colour branch, classifier feedback or learned attention head. A pass establishes only that this fixed pooling transformation exposes aligned prefix-conditioned information to the locked linear readout on the existing TRAIN folds. It does not establish causal prefix-token semantics, independent-sample generalization, superiority of ViT over a hybrid, or deployment readiness. Raw DINOv3-S is not the mobile final model; a pass permits only a separately preregistered distillation or adapter protocol for an operator-friendly student.

B13 and B14 remain closed for their exact tested routes. B15 does not reopen EfficientViM-M1 final-320, iFormer-S pooled-320, another backbone, another DINO layer, or any validation/test evidence.

## Locked DINO source, weight and historical evidence

- Model ID: timm/Hugging Face `vit_small_patch16_dinov3.lvd1689m`, snapshot revision `3bf4720a82ec2066db88137180ff1f83a675cef0`.
- Cached `model.safetensors`: exactly `86,362,376` bytes, SHA256 `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`. Load with `safetensors`, never pickle, into `timm.create_model(..., pretrained=false, num_classes=0, img_size=256)` and require a strict state load.
- Runtime source is timm `1.0.27`; this model resolves specifically to module/class `timm.models.eva.Eva`, not `VisionTransformer`. The locally verified `D:\DataAI\.venv\Lib\site-packages\timm\models\eva.py` has SHA256 `23314ef536d7ce9cc3737e54f424841b46f426e93af651c357b4a75a8b00a12d`. The relevant `timm\models\_factory.py` dispatcher has SHA256 `30a6eecdaba750af470cfae3196186fd647052c72338a21c928914aac06163e4`. Preflight must assert the resolved type exactly; any source, model geometry or preprocessing drift invalidates it.
- Require final `forward_features()` geometry `[B,261,384]`, exactly five prefix tokens in order `[CLS, register_0, register_1, register_2, register_3]`, followed by the `16x16=256` patch grid. These are already the final post-normalization tokens: do not apply the model's final norm a second time. No intermediate block is accessible to B15.
- Locked B13 root: `runs/pretrained_efficientvim_classf_b13_frozen_transfer_f229649_r1`, protocol ID `TRKH_PRETRAINED_CLASSF_B13_EFFICIENTVIM_M1_FROZEN_TRANSFER_20260803`, Git HEAD `f229649fed92b56d3ec43f13bd1c23919d956abf`. Require `summary.json` SHA256 `b017d34955f1c9126a611cf064cb919b6696ed9a19bf0f2b676838ad97955a75`, `train_oof_readouts.npz` SHA256 `82bcef6d293223b819f559122bffee822b7f2a2c80a200113dd6c0b7843def39`, `train_paths.json` SHA256 `a472608c3d0b1ce1b941a1fa4bfa72b9f3032140d7f4c2792cb9bdd357b1c0fb`, and `train_dino_final_f32.npy` SHA256 `0429260ab2f619cf8312848e08af1a13f4c81caa95a93daebda6e6e22240dc38`. Bind B13 runner SHA256 `641791100aa626379637a2d939f7c36c7971779ab4b723ce40898d2a457fa7cb` for readout, metric and bootstrap semantics.
- Locked B14 evidence root: `runs/pretrained_iformer_s_classf_b14_frozen_transfer_c797a6f_r1`, protocol ID `TRKH_PRETRAINED_CLASSF_B14_IFORMER_S_FROZEN_TRANSFER_20260804`, Git HEAD `c797a6f69db2ea2f4cc120c5abbc2ab117d6a353`. Require `summary.json` SHA256 `9ad53ca882ceec174cbe19aca9dfd8e268c1c4380ae8172ee3c745d0f166ba90`, `train_oof_readouts.npz` SHA256 `ae3ad2420f070d79300701e4c222e8465afa45079a5c81a7ac0f5e30e329ffde`, `train_paths.json` SHA256 `a472608c3d0b1ce1b941a1fa4bfa72b9f3032140d7f4c2792cb9bdd357b1c0fb`, `train_iformer_s_pooled320_f32.npy` SHA256 `255141e85097a1f671ab53e06b5d19b1890f1d96bcd597137e877deae6ce3217`, accepted preflight SHA256 `aa59fef3d61be981c6540569eb35d1c7d885de5cf73675c28230af18aaf77f0a`, B14 protocol SHA256 `74302e9ded89099866aed060203973f297e71fbed8779fb2893e509a52d5a8c5`, and B14 runner SHA256 `c9eafebeec471db175e397119ecb648d591b98593d2cdc2e35a602c5a0c080fa`.
- B14 artifacts are provenance/closure anchors only. No iFormer or EfficientViM descriptor, score, metric or checkpoint may enter a B15 descriptor or readout. A focused negative test must fail if B15 requests `efficientvim_scores`, iFormer scores, B9 logits or any non-DINO feature.

## Locked formal environment and determinism

- Python `3.9.11`; NumPy `1.26.4`; PyTorch `2.6.0+cu124`; torchvision `0.21.0+cu124`; timm `1.0.27`; scikit-learn `1.6.1`; ONNX `1.19.1`; and distribution `onnxruntime-gpu==1.19.2`, with `CPUExecutionProvider` used for every ONNX parity/latency measurement. Version or provider drift is invalid infrastructure, not a scientific result.
- Formal descriptor extraction must resolve `cuda:0` on `NVIDIA GeForce RTX 4060 Laptop GPU`, CUDA capability `(8,9)`, total memory `8,585,216,000` bytes. CPU fallback, another CUDA device or multi-GPU execution is forbidden.
- Global mechanism/extraction seed is `20260804`: set Python `random`, NumPy, `torch.manual_seed` and `torch.cuda.manual_seed_all`. Donor seeds remain exactly `20260804 + fold_id`; B13 readout/bootstrap seed remains exactly `20260803`. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization, `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`, and `torch.set_num_threads(4)`. Record every setting and fail closed if deterministic execution is unavailable.
- Formal extraction remains FP32, batch `32`, workers `0`, canonical row order, one CUDA process and no autocast/TF32. Set `torch.backends.cuda.matmul.allow_tf32=False` and `torch.backends.cudnn.allow_tf32=False`. ONNX benchmarking remains batch one, four intra-op threads and one inter-op thread.

## Immutable TRAIN boundary

- Worktree/branch: `D:\DataAI\AIEx\TRKH_pretrained`, `research/pretrained-classf-b1`; formal execution requires one clean committed HEAD.
- Canonical YAML bytes: `D:\DataAI\AIEx\newdataset\class_f\data.yaml`, SHA256 `312eb376e89023f42e9df24fea8723b64a6b885c15c24f85d8e319d1ff4f1fa8`.
- TRAIN only: exactly `8278` ordered rows, class counts `[1987,497,1326,2080,2388]`, aggregate content SHA256 `e9319e6fbddd03382050fadc42bcf156195c590926700b3a7287adc38d4373c8` under `ordered relative_path + byte_length + file_SHA256`. Recompute before and after token extraction.
- Reuse the exact B13 ordered paths, labels, fold assignments, opaque union-group IDs and raw-DINO OOF scores. Fold-assignment CSV SHA256 is `afde4fec6b344b867d31e0e01b89f16c5f4be7fd551fdc1fdfd0ea34219d725f`; assignment-vector SHA256 is `fca50be670fac7da20dd23cb382f9096c3c071d1a1e0dbaa1fd123e65c8aeb4b`; group-vector SHA256 is `1243b61c91497459fa12036d1b9f63770ba4f1f11bb132ca17fb3b6013f1ade2`; path/fold SHA256 is `0f3856bf6020ba31d8564a48394e2a6fba7ad6f415e279e0063c5baadff20cb7`.
- Resolve only those locked paths directly beneath the canonical root and hash YAML bytes without invoking a generic data loader. Before descriptor extraction, path safety may check only normalized `train/` prefix, resolved containment beneath the fixed TRAIN root, and existence as a regular file; the class-directory segment is opaque and must not be parsed, mapped or compared with labels. Only after every descriptor is frozen and its start hash recorded may the runner load labels/class order and require exact class-directory-name to numeric-label agreement for all 8278 rows. Validation and test datasets, paths, files, predictions and metrics must not be constructed or opened. Persist `train=true`, `validation=false`, `test=false`, `validation_not_constructed=true`, and `test_not_constructed=true`.

DATA-01 remains unresolved: class-1 boundary supervision and fruit/session provenance still require two blinded reviewers plus adjudication or new immutable provenance. B15 cannot relabel data, resolve the review queue, rehabilitate the exposed validation split, or support a generalization claim. A confirmatory claim still requires a prospectively sealed source/time/site holdout.

## Exact preprocessing and four frozen descriptors

Apply PIL EXIF transpose, convert to RGB, resize directly to `256x256` with torchvision bicubic interpolation and `antialias=true`, scale to FP32 `[0,1]`, then normalize by mean `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`. Use `eval()`, FP32 inference, batch `32`, workers `0`, fixed canonical row order. Let the final token tensor for one row be

\[
T=[G;P],\quad G\in\mathbb{R}^{5\times384},\quad P\in\mathbb{R}^{256\times384},\quad
m=\frac{1}{256}\sum_{i=1}^{256}P_i.
\]

The four and only four `384-D` descriptor arms are:

1. **uniform**: `z_uniform = m`. This must reproduce the locked B13 raw-DINO descriptor and uses the locked B13 raw-DINO OOF scores as the definitive comparator.
2. **prefix_only**: `z_prefix = (1/5) * sum_j G_j`, the raw mean of all five final prefix tokens. There is no subtraction, concatenation or projection.
3. **aligned**: the single B15 candidate. Define non-affine LayerNorm over the 384 feature coordinates as `LN(x)=(x-mean(x))/sqrt(mean((x-mean(x))^2)+1e-6)`. Then

   \[
   q_j=\operatorname{LN}(G_j-m),\qquad k_i=\operatorname{LN}(P_i-m),
   \]

   \[
   s_i=\frac{\sum_{j=1}^{5}q_j^\top k_i}{5\sqrt{384}},\qquad
   a_i=\frac{e^{s_i}}{\sum_{r=1}^{256}e^{s_r}},\qquad
   z_{aligned}=\sum_{i=1}^{256}a_iP_i.
   \]

4. **deranged**: preserve one donor row's complete prefix-to-own-patch-mean residual block while keeping the current row's patches and values. For donor `d` and current row `r`, use `q_j=LN(G^(d)_j-m_d)`, `k_i=LN(P^(r)_i-m_r)`, and current `P^(r)` in the weighted sum. Equivalently reconstruct `G'_j=m_r+(G^(d)_j-m_d)` at the current center and apply `LN(G'_j-m_r)`. Never use the old out-of-distribution miscentering `LN(G^(d)_j-m_r)` and never mix prefix positions across donors.

Every operation above is FP32. The LayerNorm has `elementwise_affine=false` and `eps=1e-6`; it is the only additional relation normalization and never repeats the DINO final norm. The denominator is exactly `5*sqrt(384)`; softmax is only across 256 patch positions. All four arms have equal width `384`. There is no candidate residual addition of `m`, blend coefficient, temperature, top-k, threshold, dropout, learned scalar, learned parameter, classifier direction, label, logit, pHash distance, colour statistic, bbox, source ID, review flag or augmentation in any descriptor. B7/B8 machinery is explicitly excluded: no B2/global logits, action mask, Gaussian threshold, learned pair head or learned prefix-residual branch; the fixed donor-centering operation above is only a diagnostic control, not that learned B7 residual branch. Labels enter only after all four descriptor tensors have been frozen; opaque fold/group IDs enter only the negative-control donor map.

## Fixed union-group-disjoint deranged-prefix control

Construct one donor map before reading labels or scores. For each locked assignment fold `f in {0,...,4}`, take its canonical row indices and exact B13 union-group IDs, then run the following NumPy `1.26.4` algorithm with `default_rng(20260804 + f)`:

1. collect rows by union-group ID in canonical insertion order;
2. shuffle the group-key list, shuffle members within every group, and concatenate the resulting contiguous group blocks;
3. let `L` be the largest group size in that fold and circularly roll the concatenated row vector by `-L` to obtain donors;
4. map the rolled donor vector back to the canonical recipient order.

Formal execution must fail unless every fold satisfies `2*L <= fold_rows` and the five maps jointly form five bijections with zero fixed points, zero fold-assignment crossings and zero recipient/donor union-group matches. Every recipient uses one whole five-token donor block and that donor's own uniform patch mean `m_d`; current patches/values never cross rows. Labels, predictions, class counts, metrics, raw pHash values and image content are forbidden inputs to map construction. Persist and hash the donor vector before formal labels/scores are requested.

The deranged arm is deliberately a transductive, label-blind deterministic falsification diagnostic: a held row can borrow another held row's centered prefix residual block, joining otherwise distinct B13 union components. It is not an inference algorithm and cannot be deployed. No bootstrap interval or inferential gate may compare against it. Because derangement also changes donor visual/class content, a positive point comparison supports only aligned prefix-conditioned information under this fixed diagnostic, not a causal semantic interpretation of prefix tokens or a generalization claim.

## Matched readout, endpoints and bootstrap

For `prefix_only`, `aligned` and the fold-fixed `deranged` arm, use the unchanged B13 readout independently in every immutable fold: fit `StandardScaler` on fit rows only, then sklearn multinomial `LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", tol=1e-8, max_iter=2000, random_state=20260803)`, and predict the held rows. Sklearn consumes float64 after fold-local scaling. Thus every `384-D` arm uses the same scaler policy, `C`, class weights, solver and OOF folds. Uniform is the exact scientific baseline, not a capacity-matched learned control, because aligned adds zero capacity. No hyperparameter selection, calibration, thresholding or early fold inspection is allowed. The locked B13 `dino_scores` are the uniform arm; a freshly extracted uniform descriptor must have SHA256 `0429260ab2f619cf8312848e08af1a13f4c81caa95a93daebda6e6e22240dc38` before those scores may be reused.

Report OOF accuracy, macro-F1, per-class precision/recall/F1 with `zero_division=0`, class-1 precision/recall/F1, full confusion matrix, the exact `1 -> 2` and `2 -> 1` counts, and AUROC for each `1-vs-{0,2,4}` pair using only rows from that pair and decision margin `score_1-score_rival`. Mean class-1-pair AUROC is the unweighted arithmetic mean of the three pair AUROCs. Report all metrics by arm, by held fold and, for AUROC, all 15 pair-fold cells. Gates consume unrounded float64 values.

Use exactly the B13 `5000` paired, fold-stratified whole-union-component bootstrap draws with seed `20260803`; never resample individual rows. The draw vector must reproduce SHA256 `d8967cbc13b17b78dd225841bd82a1a3074a7f57c3728fc6dd600aba646ec469`. Compute aligned-minus-uniform and aligned-minus-prefix_only deltas on each common draw and percentile intervals with NumPy linear quantiles `0.025/0.975`. Every stored per-replicate metric array and every paired-delta array for uniform, prefix_only and aligned must have exactly 5000 entries and be entirely finite before any quantile is computed; filtering, dropping or imputing a replicate is forbidden. These are conditional TRAIN component-stability intervals after adaptive reuse across B3-B15, not confirmatory nominal-confidence intervals, coverage guarantees or independent generalization inference. Deranged joins components through its donor edges, so it is excluded from bootstrap entirely and receives only the preregistered point diagnostic below.

## All-or-nothing B15 gate

The aligned arm passes only if every condition holds:

1. versus uniform, class-1-vs-2 AUROC point delta is at least `+0.010` and the lower endpoint of its conditional 95% TRAIN component-stability interval is strictly greater than `0`;
2. versus uniform, mean class-1-pair AUROC point delta is at least `+0.003` and its conditional component-stability lower endpoint is strictly greater than `0`;
3. versus uniform, class-1 F1 point delta is at least `+0.010` and its conditional component-stability lower endpoint is strictly greater than `0`;
4. versus uniform, macro-F1 delta has conditional component-stability lower endpoint at least `-0.002`;
5. versus uniform, class-1 recall point delta is at least `-0.005`;
6. aligned `1 -> 2` count is no greater than uniform's count;
7. aligned reduces the uniform `2 -> 1` count by at least `10%`, computed as `(uniform_count-aligned_count)/uniform_count >= 0.10`; a zero uniform denominator invalidates the gate rather than passing it;
8. class-1-vs-2 AUROC aligned-minus-prefix_only has conditional component-stability lower endpoint strictly greater than `0`;
9. as a non-inferential diagnostic only, aligned class-1-vs-2 OOF AUROC is strictly greater than deranged class-1-vs-2 OOF AUROC under the one frozen donor map;
10. aligned class-1-vs-2 AUROC is strictly greater than uniform in at least `4/5` held folds, and aligned AUROC is strictly greater than uniform in at least `10/15` class-1 pair/fold cells;
11. all source, weight, artifact, TRAIN integrity, token geometry, donor-map, descriptor isolation, finite-value, convergence, OOF-completeness and bootstrap-draw checks pass.

No gate may be rounded, averaged across seeds, rescued by another arm or replaced after any metric is observed.

## Label-free synthetic preflight

Before any TRAIN image, dataset object, B13 array, label, score or metric is read, a clean-commit preflight must be absolutely synthetic-only and use synthetic FP32 tokens/images for numerical checks. It must not open the canonical dataset or any B13/B14 run array. The real donor map is constructed and hashed only at formal start: after exact artifact byte hashes are validated, load only the locked fold/group vectors, build and validate the map, persist its hash/report, and only then request labels or scores. Preflight must establish:

- the standalone aligned pooling module has exactly zero parameters and zero mutable/learned buffers;
- direct formula parity, correct `[B,384]` output, finite values, softmax weights summing to one, invariance to permutation of the five prefix-token order, and invariance to a common permutation of patch tokens and their weights;
- uniform-query parity: with every `G_j=m`, centered non-affine LayerNorm produces zero queries, all 256 attention weights are exactly `1/256`, and aligned output versus uniform mean pooling has maximum absolute error at most `1e-6`;
- donor-centering parity: for synthetic donor/current rows, `LN(G_d-m_d)` and `LN((m_current+(G_d-m_d))-m_current)` agree within `1e-6`; adding a common offset to donor `G_d` and `m_d` changes neither query nor output within `1e-6`. A focused negative test must demonstrate that the forbidden `LN(G_d-m_current)` is not silently substituted;
- attention entropy `H=-sum_i(a_i*log(a_i))` and effective patch count `exp(H)` are reported for synthetic cases, finite and bounded by `[0,log(256)]` and `[1,256]` respectively; these diagnostics have no selectable threshold beyond numerical validity;
- opset-17 ONNX exports of the full locked DINO image-to-uniform-descriptor and image-to-aligned-descriptor graphs use only standard `ai.onnx`/empty operator domains;
- PyTorch/ONNX Runtime CPU maximum absolute descriptor error is at most `1e-5` for both graphs;
- on the same machine and ONNX Runtime CPU provider, four threads, batch one, 15 warmups and five alternating 100-run trials, aligned p95 latency overhead over the uniform locked-DINO graph is at most `5%`, i.e. `p95_aligned/p95_uniform <= 1.05`; report medians, p95 values, trial means, graph bytes and operators;
- DINO weights/source/geometry, protocol/runner/tests and dependency versions are exact; synthetic miniature donor-map tests establish the algorithmic zero-same-union-group, zero-fixed-point and zero-fold-crossing contracts without reading formal arrays. B13/B14 artifact anchors and the real donor vector are verified only at formal start.

The two ONNX graphs are synthetic preflight temporaries, not TRAIN evidence. After successful parity/benchmark inspection, they may be removed through exact paths only; `preflight.json` must retain each graph's byte count, SHA256, operator domains/list, parity and latency telemetry. No formal/TRAIN artifact may be deleted under this exception.

The preflight root prefix is `runs/preflight_b15_aligned_prefix_patch_pool_`. Formal acceptance requires the exact non-empty nested synthetic-check and mobile-operator-check key sets, literal Boolean `true` values, and exact equality between their merged map and the top-level operator-check map; an edited-and-rehashed artifact with missing, empty, extra or divergent nested evidence is invalid. A synthetic scientific-gate failure closes B15 before descriptor access. Preflight success is an engineering prerequisite, not TRAIN evidence.

## Stop rule, artifacts and permissions

Commit and hash this protocol, runner and focused tests before preflight. Formal execution requires the exact accepted preflight artifact/SHA, identical clean HEAD and dependencies, an unused direct child of `runs` prefixed `pretrained_dinov3_classf_b15_aligned_prefix_patch_pool_`, and unchanged start/end TRAIN hashes. Start the monotonic wall timer and reset CUDA peak-memory telemetry before formal source/artifact validation. Write atomically through an owned sibling `.partial` directory and quarantine, hash and retain any failed partial output.

The first extraction pass keeps C-contiguous FP32 prefix blocks `G` and uniform means `m` in bounded RAM. Uniform `m` must remain in RAM through final integrity verification: serialize the exact `[8278,384]` FP32 array with `np.save(io.BytesIO(), m, allow_pickle=False)`, hash those in-memory bytes, and require SHA256 `0429260ab2f619cf8312848e08af1a13f4c81caa95a93daebda6e6e22240dc38`. Repeat the same in-memory serialization/hash at formal end. Uniform must never be materialized as a B15 file and then deleted. The second pass recomputes current `P/m`, obtains each donor query only from cached `G_d-m_d`, and writes deranged output; `G`, `m` and `[8278,261,384]` tokens are never retained on disk.

Retain exactly three new descriptor files: `train_prefix_only_f32.npy`, `train_aligned_prefix_patch_pool_f32.npy`, and `train_deranged_prefix_patch_pool_f32.npy`, each C-contiguous FP32 shape `[8278,384]`. Also retain the fixed donor artifact `donor_indices_i64.npy`, the exact B13-byte-identical `train_paths.json`, source/weight/evidence hashes, permissions, preprocessing, donor report, complete scalers/coefficients/intercepts, OOF scores, fold/pair metrics, confusion matrices, bootstrap draw hash/arrays/intervals, latency/parity evidence, wall time, peak allocated VRAM and byte accounting.

Immediately after each retained descriptor, donor vector and `train_paths.json` is atomically finalized, record its start SHA256. Before any promoted summary, recompute every file SHA256 and require exact start/end equality. The uniform in-memory NPY-serialization start/end SHA must also match each other and the locked expected SHA. Require `train_paths.json` start/end SHA to equal `a472608c3d0b1ce1b941a1fa4bfa72b9f3032140d7f4c2792cb9bdd357b1c0fb`. Any mismatch is integrity-invalid and cannot enter metric construction or a valid summary.

The formal limits are `900` monotonic seconds, `6*1024^3` peak CUDA allocated bytes and `100*1024^2` retained artifact bytes. After descriptors, deferred class-name/label validation and start hashes are complete—but before creating `stage_metric_construction_started.json`—apply a pre-metric resource gate: elapsed time must be at most `900`, peak VRAM at most `6 GiB`, and the exact recursive current-file byte sum plus a locked `16*1024^2` (`16 MiB`) finalization reserve must be at most `100 MiB`. A failure here is pre-metric infrastructure-invalid and is eligible for exactly one identical retry; it is not evidence against the representation. Create the metric sentinel only after this gate passes.

After the metric sentinel exists, any elapsed-time, peak-VRAM or final-byte breach is a terminal registered operational failure: set every representation/distillation/validation/test/full-train permission false, close the exact B15 route for operational infeasibility, state explicitly that this is not evidence against representation quality, and forbid retry. Before writing/promoting `summary.json`, construct its deterministic JSON bytes in memory, include those exact bytes with every retained file in `final_total_bytes`, update/reserialize until the encoded `final_total_bytes` value and byte length are stable, and require the exact total at most `100 MiB`, elapsed at most `900` seconds and peak VRAM at most `6 GiB`. The final wall value is the monotonic post-serialization elapsed time conservatively rounded upward to the next whole second; if serialization crosses the recorded bound, repeat the in-memory projection before deciding validity. A summary must not contain `integrity_complete=true`, `execution_status=valid`, `signal_gate_passed=true` or any positive permission before this final resource/integrity check passes. On failure, write only a terminal operational-failure manifest; never promote a valid-looking summary.

One infrastructure retry is allowed only before metric construction, at the identical commit, protocol, preflight, environment and arguments, without inspecting partial descriptors or scores. Once any arm's OOF metric construction starts, the registered scientific result or registered operational failure is final. Any scientific gate failure closes exactly final-postnorm `[5+256]` DINOv3-S / non-affine-LN-eps-`1e-6` / parameter-free aligned pooling / `256x256` / B13 balanced-linear-readout route; a post-sentinel resource failure closes that same route only for operational infeasibility under the locked budget.

After failure, do not try temperature, residual/blend, layer choice, attention head, prefix/register subset, top-k, alternate normalization, learned projection, readout C, loss, input size, threshold, classifier feedback or another pooling variant on the same evidence. After a pass, do not open validation/test or call raw DINO the mobile final. The only permission is to write one separate, prospectively frozen distillation/adapter protocol with matched controls and explicit mobile student gates. DATA-01 and the absence of a new sealed holdout continue to block any generalization or confirmatory scientific claim.
