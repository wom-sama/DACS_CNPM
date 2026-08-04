# B13 EfficientViM-M1 frozen-transfer protocol — 2026-08-03

Protocol ID: `TRKH_PRETRAINED_CLASSF_B13_EFFICIENTVIM_M1_FROZEN_TRANSFER_20260803`

## Question and claim boundary

B13 asks one question before any image fine-tuning: does the final frozen representation of the official ImageNet-pretrained EfficientViM-M1 retain enough class-1 boundary signal to justify it as a mobile student/backbone candidate? It is compared with raw pretrained DINOv3-S/16 under the same TRAIN rows, fixed folds and linear readout.

This screen is not a claim that SSM is superior to ViT: the two sources used different pretraining objectives and corpora. It also does not test another DINO hybrid. Its only possible promotion is permission for one separately frozen EfficientViM fine-tuning/distillation protocol. Failure closes this exact M1 final-feature route without trying M2/M3/M4, distillation weights, multi-stage fusion, input-size, readout-C or loss sweeps.

Primary sources are the [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_EfficientViM_Efficient_Vision_Mamba_with_Hidden_State_Mixer_based_State_CVPR_2025_paper.pdf) and [official MIT repository](https://github.com/mlvlab/EfficientViM), pinned at commit `304340cb9c339b61669250d058525c9cdadd5e93`.

## Immutable inputs

- Worktree/branch: `D:\DataAI\AIEx\TRKH_pretrained`, `research/pretrained-classf-b1`.
- Canonical data: `D:\DataAI\AIEx\newdataset\class_f\data.yaml`; TRAIN only, exactly `8278` rows with class counts `[1987,497,1326,2080,2388]` and fixed class order from the data YAML.
- A0 ordered TRAIN cache supplies only the immutable path/label ledger; its image tensors, B9 tokens and B9 logits are not candidate inputs.
- The accepted preflight hashes every one of the `8278` ordered TRAIN image files as `relative_path + byte_length + file_SHA256`; formal execution recomputes this aggregate before and after descriptor extraction. Validation/test files are never opened. Any TRAIN byte drift aborts the run.
- Fold assignment CSV SHA256: `afde4fec6b344b867d31e0e01b89f16c5f4be7fd551fdc1fdfd0ea34219d725f`. Its five label-independent source/adjacency/pHash union components never cross fit/held folds.
- Raw DINO source: timm `vit_small_patch16_dinov3.lvd1689m`, cached safetensors SHA256 `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`.
- EfficientViM source checkpoint: official non-distilled M1 e450, SHA256 `c04c83b982a9a136cec8dca98c5397540bb7d8b18acaa22e559edca042c937a6`. B13 explicitly selects `model_ema`: the checkpoint records `73.4800001` for EMA versus `73.4560000` for the ordinary model. This is a preregistered policy choice, not a claim that the official loader always prefers EMA.
- The EfficientViM checkpoint must be read with `torch.load(..., weights_only=True)` and a temporary inert allowlist for its serialized `yacs.config.CfgNode`; unsafe pickle loading is forbidden. All 307 selected tensors must strict-load the 1000-class M1 before feature extraction.

Validation and test datasets, paths, images, predictions and metrics are forbidden. B9 validation numbers are historical context only and do not enter a gate.

## Frozen descriptors and matched readout

- DINO: model-specific 256-pixel bicubic resize, its published mean/std, FP32 inference, final post-norm patch tokens only, remove five prefix tokens and average the `16x16` patch grid to one `384-D` vector.
- EfficientViM: official evaluation transform (bicubic resize to 256, center crop 224, ImageNet mean/std), FP32 inference, final stage output after `norm[3]` and global average pool to one `320-D` vector. The pretrained four-head fusion and ImageNet logits are not readout features.
- Both arms use their final representation only. No intermediate stage, colour statistic, bbox, source ID, review flag, component label, B9 logit or augmentation is supplied.
- For each immutable fold, fit `StandardScaler` on fit rows, then one sklearn multinomial `LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", tol=1e-8, max_iter=2000, random_state=20260803)` and predict the held rows. No hyperparameter selection or early fold inspection is allowed.
- FP32 descriptor extraction is batched at `32`, workers `0`; sklearn consumes float64 after fold-local scaling. Every row appears in exactly one OOF holdout.

## Endpoints and promotion gate

Primary endpoints are OOF macro-F1, class-1 precision/recall/F1, restricted `0/2/4 -> 1` false positives, and mean AUROC over `1-vs-{0,2,4}` using paired decision margins. Report every fold and confusion matrix.

Use `5000` fixed-seed paired bootstrap replicates. Resample whole union components within each fold, never individual rows. Candidate-minus-DINO intervals and fold deltas are frozen before extraction.

EfficientViM passes the representation gate only if every condition holds:

1. mean pair-AUROC delta is at least `-0.005`, with 95% lower bound at least `-0.010`;
2. macro-F1 delta is at least `-0.025`, with lower bound at least `-0.040`;
3. class-1 F1 delta is at least `-0.035`, with lower bound at least `-0.060`;
4. class-1 recall is at least `90%` of DINO recall;
5. restricted-FP rate is no more than `1.25x` DINO's rate;
6. at least three of five folds have class-1 F1 delta no worse than `-0.050`;
7. all hashes, strict loads, finite values, path/label order, component isolation, convergence and OOF-completeness checks pass.

The margins deliberately trade a bounded frozen-representation loss for a materially smaller deployment graph; passing does not mean accuracy parity and does not authorize test access.

## Static mobile precondition

Before the formal TRAIN run, a clean-commit preflight must use synthetic inputs only and show:

- the 5-class M1 skeleton has exactly `5,720,278` parameters and no more than `30%` of the 5-class DINO graph;
- fixed `224x224` PyTorch forward is finite and the vendored graph matches the official graph after strict EMA load;
- opset-17 ONNX export uses no custom selective-scan/Triton operator, ONNX Runtime argmax matches PyTorch and maximum absolute error is at most `1e-5`;
- on this machine with ONNX Runtime CPU, four threads, batch one, 15 warmups and five alternating 100-run trials, M1 median latency is at most `0.25x` DINO and p95 is at most `0.30x` DINO.

These are an engineering proxy, not a physical-phone result. A passed fine-tuned model would still require locked INT8/FP16 accuracy, thermal latency and memory audits on target devices.

## Stop rule and artifacts

Preflight is committed and hashed before formal descriptors are computed. The formal runner must require that exact preflight SHA, a clean tracked worktree and an unused output directory. Persist descriptor hashes, ordered paths/labels/folds/groups, OOF scores, fold scalers/readout coefficients, bootstrap draw hash, metrics, confusion matrices, dependencies, source hashes, wall time and explicit `train=true, validation=false, test=false` flags.

If any promotion gate fails, write the closure result and stop. Do not fine-tune M1, open validation/test, add DINO features, enable four-stage feature fusion or replace the readout after seeing B13.
