# TRKH 5-Class MORE Readiness Protocol (2026-07-14)

## Research question

Can NeurIPS 2025 MOdel REbalancing (MORE) reserve mergeable low-rank
convolutional capacity for rare class 1 while reducing its false positives and
preserving its true positives? This is a parameter-space representation test,
not another classifier head, class weight, post-hoc threshold, or LoRA-only
adapter.

Primary source:

- MORE, *Long-tailed Recognition with Model Rebalancing*, NeurIPS 2025:
  https://proceedings.neurips.cc/paper_files/paper/2025/file/c4889bd7f7ce643003746526da2c2fc4-Paper-Conference.pdf

The locked implementation follows equations 1-6: each convolution is
`W = W_g + B_t A_t`; the tail contribution is measured by squared L2 logit
discrepancy between full and general-only paths; the discrepancy is weighted by
the natural class prior; and its coefficient is sinusoidal over training. The
paper reports that decomposition without `L_MORE` is comparable to baseline,
so a matched low-rank control is mandatory.

## Locked evidence

- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Data YAML:
  `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`
- Data YAML SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`
- CIDT train-only fold/probability source:
  `runs/audit_cidt_readiness_full_train_20260714`
- CIDT summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`
- CIDT ordered prediction CSV SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`
- Rows/classes/source groups: `9215`, `[1941,541,1920,2520,2293]`,
  `8064`; focus class is `1`.
- Existing clean-train keeper class-1 P/R/F1 is
  `0.700265/0.975970/0.815444`. There are `213` class-1 false positives and
  `13` false negatives, so precision is the dominant train-only target.

Validation and test pixels, labels, predictions, and metrics are forbidden in
readiness. The CIDT CSV supplies only fixed train sample order, targets,
source-group folds, and keeper reference probabilities.

## Locked parameterization

- Wrap all five keeper `Conv2d` modules; no layer selection:
  `stem.blocks.0.block.conv`, `stem.blocks.1.block.conv`,
  `stem.blocks.2.block.conv`, `patch_embed.proj`, `detail_enhancer.proj`.
- Flatten each convolution as `[out, in*k_h*k_w]` and set
  `rank=max(1, round(0.1*min(out,in*k_h*k_w)))`.
- Keep `W_g` bit-identical to the keeper. Initialize `A_t` with fixed Kaiming
  seed `42` and `B_t=0`, making the initial full path keeper-identical.
- Base parameters are frozen in readiness; only tail parameters train. This is
  a conservative capacity probe, not a claim of reproducing full end-to-end
  paper training.
- Matched control: identical tail modules, initialization, source folds,
  minibatch order, optimizer, LR, and budget with sinusoidal amplitude `A=0`.
- MORE candidate: paper normalized amplitude `A'=A/C=2.0`; with `C=5`, use
  fixed peak `A=10.0` and `alpha(t)=10*sin(pi*t/T)`.
- Optimizer: SGD, LR `3e-4`, momentum `0.9`, weight decay `0.0002`; no
  scheduler beyond the paper sinusoidal MORE coefficient. The LR is the
  paper's reported low-rate setting and a 64-row train-only numerical check
  produced mean/max logit deltas `0.00110/0.00386` after two steps. Accuracy,
  F1, validation, and test were not used to choose it. No rank/LR/layer/
  amplitude/seed/budget sweep is allowed.
- Deterministic keeper eval transform and model eval mode are used so dropout
  cannot contaminate the full-versus-general discrepancy.
- Disable CUDA matmul TF32 and cuDNN TF32 for zero-tail/merge gates; enable
  deterministic cuDNN and disable benchmark selection. A prior apparent
  `4.5e-5` merge error came from TF32 convolution choice; exact deterministic
  FP32 replay reduced the keeper error to approximately `1e-7` without
  changing the `1e-5` gate.
- Deployment equivalence requires algebraically merging `B_t A_t` into `W_g`;
  no tail branch may remain at inference.

## Stage A: functional preflight

Before any full train-only OOF audit, require all of the following:

1. All five expected convolutions are wrapped and no grouped-convolution
   approximation is used.
2. Zero-tail maximum logit error versus the unwrapped keeper is `<=1e-6`.
3. Merge maximum logit error versus the unmerged full path is `<=1e-5`.
4. Base parameters stay frozen; both low-rank factors receive finite gradients
   after the zero-initialized factor has been updated.
5. General-only logits remain keeper-identical while full logits become
   non-identical after optimizer steps.
6. Control and MORE use identical ordered sample indices and tail
   initialization hashes.
7. Run exactly `20` ordered natural-frequency batches from the fit side of
   CIDT fold 0 for each matched variant, then all `1843` fold-0 holdout rows.
   MORE must differ from control on at least two hard decisions, produce a
   nonzero discrepancy after zero initialization, and avoid more than five
   excess harms over corrections. This is an early material-effect gate, not a
   reportable model metric.

Failure closes the method without an image smoke.

## Stage B: full train-only OOF gate

Use the exact five CIDT source folds. For each fold, train matched control and
MORE tail adapters only on the other four folds, then predict every held-out
row exactly once. The output must contain all `9215` unique sample indices,
zero fit/holdout source overlap, and no validation/test access.

All gates are predeclared:

- finite normalized probabilities and complete ordered replay;
- candidate changes at least `47/9215` hard decisions versus matched control;
- candidate macro F1 is no worse than control by more than `0.001`;
- class-1 precision gain versus control is at least `0.015`;
- class-1 F1 gain versus control is at least `0.005`;
- class-1 recall is at least `control - 0.015`;
- class-1 FP removals exceed FP creations by at least `10`;
- class-1 TP breaks are no more than FN rescues plus `5`;
- total corrections exceed total harms;
- at least four of five folds have nonnegative class-1 precision gain, and at
  least three have positive gain;
- mean full-versus-general logit L2 discrepancy on true class 1 is at least
  `1.25x` the mean on non-class-1 rows, matching the paper's tail-capacity
  ordering;
- newly created non-focus `3->2` harms are at most `5`.

A failed gate forbids trainer integration, validation, probe, test, parameter
sweeps, and current-best command changes. A full pass authorizes exactly one
matched 40-batch/full-validation/no-test control/MORE smoke from the keeper,
followed by transition, confusion, calibration, architecture, native-attention,
all-method XAI, and robustness audits before any 120-batch probe.

## Promotion boundary

No result from train OOF or a 40-batch smoke can replace the keeper. Promotion
still requires independent full-validation reload, the existing absolute
macro/class-1/precision/recall gates, a successful 120-batch probe, complete
audits, and an unchanged final-test policy. The VS Code full-train command file
is updated only after that sequence is passed.
