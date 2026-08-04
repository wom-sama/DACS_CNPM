# B14 iFormer-S frozen-transfer protocol — 2026-08-04

Protocol ID: `TRKH_PRETRAINED_CLASSF_B14_IFORMER_S_FROZEN_TRANSFER_20260804`

## Question and claim boundary

B14 asks one prospective TRAIN-only question: does the canonical pooled representation of the official supervised-ImageNet iFormer-S checkpoint preserve enough five-class and class-1 boundary information to justify this 6.24M-parameter graph as a mobile fine-tuning candidate? The only comparator is the locked raw-DINO arm from B13 on the same ordered TRAIN rows, folds, groups and readout.

The architectural hypothesis is fixed before descriptor extraction. iFormer-S combines an overlapping 5x5 FusedIB stem and early high-resolution local convolution with late single-head modulation attention. This may retain local mango-surface detail while adding global shape and illumination context, without attaching an unconstrained colour branch to DINO. B14 tests the released representation, not the mechanism in isolation.

This is not a causal claim that a convolution-attention hybrid is better or worse than ViT. iFormer-S uses supervised ImageNet-1K pretraining, 224-pixel input and a pooled 320-D feature; DINOv3-S/16 uses self-supervised LVD-1689M pretraining, 256-pixel input and averaged 384-D post-norm patch tokens. Objective, corpus, resolution, pooling and feature width are all confounded. A causal architecture claim requires matched pretraining.

Primary sources are the [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/396a3fc4560b1fe85574ebebe2b2a739-Paper-Conference.pdf), [official repository](https://github.com/ChuanyangZheng/iFormer) and [official v0.9 release](https://github.com/ChuanyangZheng/iFormer/releases/tag/v0.9). PoolFormer, SHViT and other backup architectures are outside this protocol.

## Locked source and checkpoint

- Official checkout: `D:\DataAI\external_sources\official\iformer_iclr2025`, tag `v0.9`, commit `063d2afd851f77c84be0f15c67c9bb5d828c5f2b`, with a clean tracked and untracked worktree.
- `models/iformer.py` SHA256 `a83a7903a9b9e68c072f219974f3a99f5bd80d24965a3437179e8c9ea029a652`.
- `configs/iFormer_s.yaml` SHA256 `754ea8f96b0d480c7ef435cb1ba056b927f1850fa63c975f7792a1db581b37d7`.
- `datasets.py` SHA256 `a046faac72944294051fe6fd202e02939b25b67cb86ee5751f7ce9a70524abc4`.
- `LICENSE` SHA256 `63e8210e6bf3e8c032dc0c69b1d1d2e3ab72c14b02cabcc0dada2618bb188b97`; code is MIT. The release gives no separate weight-license statement, so checkpoint redistribution remains unauthorized unless clarified.
- Official release asset `iFormer_s.pth`, asset ID `210177750`, exactly `79,363,642` bytes, SHA256 `dba81d99d9b6491b18fccd022f795d2b43e31bcb75f7aeea59be37b00bc7d7a1`. The GitHub API URL, size and a fresh in-memory download must agree before formal use.
- Load only `checkpoint["model"]`. There is no EMA arm. `torch.load(..., weights_only=False)` is forbidden; use `weights_only=True` with a scoped allowlist containing only `argparse.Namespace`, `numpy.core.multiarray.scalar`, `numpy.dtype` and the concrete float64 dtype class.

The serialized `args` metadata is known stale: it says `model=convnext_t1_v35`, an alias absent from v0.9, and `layer_scale_init_value=1e-6`, while the state contains no gamma tensors. It must never construct the model. The locked constructor is official `iFormer_s(num_classes=1000, drop_path_rate=0, layer_scale_init_value=0, head_init_scale=1, distillation=False)`: depths `[2,2,19,6]`, widths `[32,64,176,320]`, FusedIB stem, kernels `[5,3,3,3]`. All 505 state entries must strict-load; the 1000-class graph has exactly `6,563,368` parameters. Constructing with `1e-6` yields 24 missing gamma keys and must fail.

The current timm `1.0.27` factory injects an incompatible `pretrained_cfg` argument into this v0.9 source. The runner must use the pinned direct constructor through a small compatibility adapter and must not alter the official model file.

## Immutable TRAIN inputs and locked comparator

- Worktree/branch: `D:\DataAI\AIEx\TRKH_pretrained`, `research/pretrained-classf-b1`.
- Canonical YAML: `D:\DataAI\AIEx\newdataset\class_f\data.yaml`, SHA256 `312eb376e89023f42e9df24fea8723b64a6b885c15c24f85d8e319d1ff4f1fa8`.
- TRAIN only: exactly `8278` ordered rows, class counts `[1987,497,1326,2080,2388]`, aggregate content SHA256 `e9319e6fbddd03382050fadc42bcf156195c590926700b3a7287adc38d4373c8`. Recompute the ordered `relative_path + byte_length + file_SHA256` aggregate before and after extraction.
- Fold assignment CSV SHA256 `afde4fec6b344b867d31e0e01b89f16c5f4be7fd551fdc1fdfd0ea34219d725f`; the five label-independent source/adjacency/pHash union components never cross fit and held folds.
- Locked B13 comparator root: `runs/pretrained_efficientvim_classf_b13_frozen_transfer_f229649_r1`. Require `summary.json` SHA256 `b017d34955f1c9126a611cf064cb919b6696ed9a19bf0f2b676838ad97955a75`, `train_oof_readouts.npz` SHA256 `82bcef6d293223b819f559122bffee822b7f2a2c80a200113dd6c0b7843def39`, and `train_paths.json` SHA256 `a472608c3d0b1ce1b941a1fa4bfa72b9f3032140d7f4c2792cb9bdd357b1c0fb`.
- Reuse only B13 labels, folds, groups, ordered paths and raw-DINO OOF scores. EfficientViM descriptors, scores and metrics are not B14 inputs. Require exact vector hashes, row order, DINO metrics and OOF completeness from the locked summary.

Validation and test datasets, paths, images, predictions and metrics are forbidden. The runner must positively record `validation_not_constructed=true` and `test_not_constructed=true`. Historical validation numbers are context only and cannot select B14.

## Canonical iFormer descriptor and readout

- Preprocess each TRAIN image with resize-short-side `256` using bicubic interpolation, center crop `224`, tensor conversion, ImageNet mean `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`.
- Use FP32 evaluation, batch `32`, workers `0`. Run the exact four official downsample/stage pairs, then official adaptive global average pooling before `model.classifier`, yielding one `320-D` vector. The misleading official `forward_features()` returns logits and is forbidden as a descriptor API.
- A compatibility wrapper must prove on synthetic input that passing its pooled vector to the untouched classifier reconstructs the official forward logits exactly. No classifier logits, BN-transformed feature, intermediate stage, stage concatenation, alternative pooling, colour statistic, bbox, source ID, review flag, group label, B9 feature/logit or augmentation is supplied.
- For each fixed fold, fit `StandardScaler` on fit rows, then sklearn multinomial `LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", tol=1e-8, max_iter=2000, random_state=20260803)` and predict the held rows. No hyperparameter selection or fold inspection is allowed.
- Every row appears in one OOF holdout; all descriptors/scores must be finite and all five readouts must converge. Persist the candidate descriptors and complete scalers/coefficients/intercepts.

## Endpoints and representation gate

Report OOF accuracy, macro-F1, per-class precision/recall/F1, class-1 precision/recall/F1, restricted `0/2/4 -> 1` false positives, mean AUROC over `1-vs-{0,2,4}`, all five folds and confusion matrices.

Use exactly `5000` paired fold-stratified whole-union-component bootstrap replicates with seed `20260803`; never resample individual rows. The draw vector must reproduce B13 SHA256 `d8967cbc13b17b78dd225841bd82a1a3074a7f57c3728fc6dd600aba646ec469` before computing candidate-minus-DINO intervals.

iFormer-S passes only if every condition holds:

1. mean pair-AUROC delta is at least `-0.005`, with 95% lower bound at least `-0.010`;
2. macro-F1 delta is at least `-0.025`, with lower bound at least `-0.040`;
3. class-1 F1 delta is at least `-0.035`, with lower bound at least `-0.060`;
4. class-1 recall is at least `90%` of locked DINO recall;
5. restricted-FP rate is no more than `1.25x` locked DINO;
6. at least three of five folds have class-1 F1 delta no worse than `-0.050`;
7. all provenance, strict-load, integrity, convergence, component-isolation and OOF checks pass.

These are mobile-aware non-inferiority margins, not an accuracy-superiority claim.

## Synthetic mobile precondition

Before any TRAIN descriptor is read, a clean-commit preflight must reproduce with synthetic inputs only:

- exact 5-class graph size `6,243,973` parameters, no more than `30%` of locked DINO's `21,588,869`;
- opset-17 ONNX no larger than `50 MiB`, using only standard `ai.onnx`/empty operator domains;
- ONNX Runtime CPU argmax exactly equal to PyTorch and maximum absolute error at most `1e-5` for both graphs;
- four threads, batch one, 15 warmups and five alternating 100-run trials on this machine, with iFormer median latency no more than `0.35x` DINO and p95 no more than `0.40x` DINO;
- finite official forward, exact pooled-feature reconstruction and no source/checkpoint drift.

The initial engineering qualification observed a parameter ratio `0.28922`, standard-domain ONNX with `25,032,211` bytes, parity error `1.67e-6`, median ratio `0.17965` and p95 ratio `0.17089`. These observations do not satisfy the gate: the committed runner must independently reproduce and hash them. The official `0.85 ms` iPhone 13/CoreML result is external eligibility evidence, not local phone certification. Android, INT8/FP16 accuracy, thermal latency and memory remain unaudited.

## Stop rule and artifacts

Commit and hash the protocol and synthetic preflight before extracting descriptors. Formal execution requires that exact preflight SHA, a clean tracked worktree, a clean pinned source checkout and an unused output directory. Persist dependency versions, source/release/checkpoint hashes, permissions, descriptor/path/OOF hashes, readout state, bootstrap draw hash, metrics, gates and start/end TRAIN integrity.

If the synthetic gate fails, close before data access. If any representation gate fails, close the exact official iFormer-S-v0.9 / pooled-320 / 224-pixel / balanced-linear-readout route. Do not then try iFormer-T/M/L, the CoreML package, classifier logits, other stages, concatenation, alternate pooling, input size, readout C, loss, fine-tuning or distillation on this evidence. Do not substitute PoolFormer or SHViT inside B14.

Passing authorizes only one separately committed fine-tuning protocol. It does not authorize validation, test, full train, XAI claims, model comparison publication or deployment promotion. Human blinded review of the unresolved class-1 queue remains an independent requirement.
