# TRKH 5-Class Spatial Worst-Shift Robust Optimization A0 Protocol

Date: 2026-07-17

## Question

Can label-aware spatial robust optimization turn the keeper's measured
one-pixel instability into higher class-1 precision without sacrificing true
class-1 support, changing inference, touching raw data, or using validation or
test?

This is a train-only readiness gate. It is not a validation experiment, a
production integration, or authority to update the current-best commands.

## Primary authority and provenance

- Accepted paper: Engstrom et al., *Exploring the Landscape of Spatial
  Robustness*, ICML 2019, PMLR 97:1802-1811:
  <https://proceedings.mlr.press/v97/engstrom19a.html>.
- Local accepted paper:
  `D:\DataAI\external_sources\papers\Engstrom_Exploring_Spatial_Robustness_ICML2019.pdf`.
- Paper SHA-256:
  `220d573aef26395e8414c9c59d5f91bb97d4be8fc51ef02a031fbf8894f31019`.
- Paper-linked official repository:
  <https://github.com/MadryLab/adversarial_spatial>.
- Local official repository:
  `D:\DataAI\external_sources\official\adversarial_spatial`.
- Official commit/tree:
  `a1c9e364d4179d410209ba3a3b06fdcc73ad2ff9` /
  `60fe12b9c388b00d00068e07c2e93d134f126cf3`.
- Official source SHA-256 values:
  - `README.md`: `7d1098457088ab740f4f46c29e096bb84ca2df9fc5056ff0c7f53814a83ba256`;
  - `spatial_attack.py`: `f663f8628e50b2c89b68b90284559c46e830908adf12410dd4c34d0b831b7b6e`;
  - `train.py`: `9c27b7cfb9bbfb7472fb3b8d03e217a8d9d480c90c42f44bd3007cb38aa7aeac`;
  - `LICENSE`: `9787e942613613893ea455e11094c625e902a4a31dd549a408b57158684b0925`.
- The source is MIT licensed. GitHub reported 47 stars and 13 forks when
  checked on 2026-07-17. Popularity is disclosed but is not efficacy evidence.

The official implementation enumerates or samples spatial transformations in
evaluation mode, prioritizes a misclassifying transformation if one exists,
otherwise selects the largest cross-entropy, and performs the training update
on that selected transformation. A0 independently transcribes that selection
rule in current PyTorch/TRKH semantics; it does not copy TensorFlow model or
dataset code.

## No-repeat screen

This route is equation-distinct from the closed TRKH families.

- R-Drop, illumination/chroma/AugMix-JSD, surface, paired-view, and hflip
  consistency regularized two predictions or representations. This candidate
  instead minimizes supervised CE on one per-sample worst spatial transform.
- Hflip TTA and hflip-instability routing were recall-unsafe. A0 does not use
  hflip, TTA output aggregation, a router, a threshold, or a post-hoc filter.
- SoftPool, max-soft, SPT, BlurPool/APS-adjacent work, and NeurIPS-2022 LPD
  changed inference or downsampling. A0 leaves the model graph and inference
  path unchanged.
- WACV-2025 TIPS was screened out as an inference-time polyphase-pooling
  neighbor immediately after LPD failed runtime, memory, and backend parity.
- NeurIPS-2021 AugMax was screened out because its adversarial composition of
  AugMix-like corruption chains plus DuBIN overlaps failed photometric/style
  routes and does not isolate the observed one-pixel signal.
- TMLR DAIR was screened out because it is another consistency regularizer;
  its official repository also exposes no software license. No DAIR source is
  copied.
- CVPR-2021 MaxUp supports the broad worst-augmentation idea, but its paper did
  not provide an official licensed implementation. The selected ICML source is
  both directly about translations and backed by paper-linked MIT code.

## Locked inputs

- Repository: `D:\DataAI\AIEx\TRKH`.
- Branch: `classification-only-research`.
- Keeper checkpoint:
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher-args SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- Dataset YAML: `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`.
- Dataset YAML SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- CIDT train-only fold source:
  `runs\audit_cidt_readiness_full_train_20260714\summary.json` and
  `predictions_all_conditions.csv`.
- CIDT summary/prediction SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad` /
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Current command/history SHA-256 must remain
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

Only `yolo_f/train` may be decoded. Any validation or test path, row, metric,
cache, or checkpoint-selection input is a hard failure.

## Locked source-disjoint cohorts

Reuse the already hash-locked CIDT/DART whole-source partition without
resampling or reassigning a source.

- update: folds 1-4, 1,024 rows, 932 source groups, class counts
  `[217,60,212,280,255]`, ordered sample-index SHA-256
  `2279a58ace27fbb5840c31a037a3e9871d6173799279327c8729f9298b572bb7`;
- readiness probe: folds 1-4 after excluding update sources, 256 rows, class
  counts `[54,16,52,70,64]`, ordered sample-index SHA-256
  `6490523a2de91848dc06e3ace3f78be5cfb1cb1670a28e05468527eff050c5bd`;
- final train-only holdout: all fold-0 rows, 1,843 rows, ordered sample-index
  SHA-256
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`.

Update, probe, and holdout source overlap must be exactly zero. The keeper saw
the complete training set during its historical training, so fold 0 is an
adaptation holdout, not a claim of scratch OOF generalization. It is used only
for an equal-initialization differential gate; a later candidate still needs
the normal locked validation protocol.

## Locked transforms

Start from the deterministic keeper evaluation transform at 256 px, including
the existing illumination normalization. The exact transform set and order is

`T = [(0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]`.

For `(dx,dy)`:

1. translate the normalized image tensor by replicate-pad then crop;
2. translate `image_mask` by false-pad then crop;
3. add `dx/256` and `dy/256` to both `bbox` and `crop_bbox` centers and clamp
   centers to `[0,1]`;
4. preserve target and `sample_index` exactly.

The implementation must independently test direction, shape, identity,
metadata movement, target/index invariance, and exact replay. Raw dataset files
must remain byte-untouched.

## Official worst-transform rule

For sample `i`, target `y_i`, condition order `t in T`, prediction
`p_i(t)`, correctness `c_i(t)`, and CE `l_i(t)`:

1. if any `c_i(t)=false`, select the misclassifying condition with maximum
   `l_i(t)`;
2. otherwise select the condition with maximum `l_i(t)`;
3. exact ties keep the earlier condition in the locked order.

Selection runs with `model.eval()` and `torch.no_grad()`. The optimizer update
runs with `model.train()` on only the selected transformed row. An independent
oracle and finite synthetic cases must match selected indices and losses
exactly before real training.

## Matched A0 variants

Both variants start from byte-identical keeper state, use identical ordered
batches, perform 32 updates of batch 32, and execute all nine no-grad selection
forwards plus one gradient forward per update.

### Random-shift matched control

Select one condition per sample from `T` by a stable SHA-256 function of
`seed=42`, update step, and `sample_index`. The nine evaluation forwards still
run and are recorded so forward count and selection overhead match the
candidate. Selection cannot depend on logits, target, process RNG, or batch
order.

### Spatial worst-shift candidate

Select one condition per sample using the official rule above. No clean-loss
blend, KL/JSD term, class weight, class-1 oversampling, threshold, soft target,
teacher, bbox loss, or extra architecture is allowed.

Both use full-parameter AdamW, learning rate `1e-5`, betas `(0.9,0.999)`,
weight decay `0.05`, no scheduler, no gradient accumulation, and CE only.
Logical update RNG is reset identically before the gradient forward of both
variants. Every parameter group must receive finite gradients and move.

## Pre-training signal stop

Before optimizer work, run all nine conditions on the 256-row source-disjoint
probe. Continue only if:

- all rows and transforms replay exactly and all outputs are finite;
- at least six of nine conditions are selected by the official rule;
- at least 5% of rows select a non-clean transform;
- selection exposes at least one true-class1 clean-correct row and at least
  one restricted `{0,2,4}->1` clean false positive with target-loss span
  greater than `1e-6` across the nine conditions;
- the official rule and independent oracle select the same condition for every
  row.

This gate proves only that the training signal is active. It is not a metric
win.

## Evaluation protocol

Evaluate the immutable keeper, matched random-shift control, and spatial
worst-shift candidate on all 1,843 holdout rows. Use the standard deployment
forward and exact full batch composition. For each model evaluate all nine
conditions under:

- clean photometry;
- `lighting_dim`: brightness `0.70`, contrast `0.90`;
- `lighting_bright`: brightness `1.25`, contrast `1.10`;
- `low_contrast`: brightness `1.00`, contrast `0.65`.

For every photometric condition report:

- metrics on `(0,0)`;
- metrics from the per-target official worst transform;
- every individual shift metric;
- shift-consistency rate and clean-to-shift transition counts;
- macro F1, all per-class precision/recall/F1/support;
- class-1 TP, FP, FN, precision, recall, and F1;
- restricted `{0,2,4}->1` false positives;
- class-1 entries and exits relative to `(0,0)`.

Mean-probability shift aggregation may be reported only as a diagnostic. It
cannot select a model, threshold, or future command.

## Conjunctive advancement gates

All structural gates and all metric gates must pass.

### Structural and resource gates

- paper, source, license, repository, keeper, data, CIDT, protocol, command,
  cohort, and implementation hashes are exact;
- tracked worktree is clean and the implementation commit is pushed before
  formal execution;
- no unrelated `python.exe`, `pythonw.exe`, or `trtexec.exe` is active; no
  unknown process is terminated;
- initial model states, batch bytes, logical RNG, update counts, optimizer
  steps, and forward counts are exact;
- both variants perform exactly 32 updates and 320 model forwards over update
  batches: 288 no-grad selection forwards plus 32 gradient forwards;
- official/oracle selection parity is exact, selected rows are finite, random
  selection replay is exact, and candidate selection differs from random on at
  least 25% of update rows;
- all required parameter groups have finite nonzero gradients and move;
- model state schema and parameter count remain unchanged;
- candidate and control training peak allocation are each `<=4.5 GiB` and
  candidate/control wall-time ratio is `<=1.10`;
- standard inference runtime and memory ratios versus the keeper are each
  `<=1.02`; static batch-1 standard-domain ONNX export is finite, argmax exact,
  and has maximum absolute error `<=1e-5`;
- independent CSV replay differs from live summaries by at most `1e-12`.

### Clean holdout gates

Let `best_ref` mean the better of immutable keeper and matched random control
for the named metric.

- candidate macro F1 `>= best_ref + 0.001`;
- candidate class-1 F1 `>= best_ref + 0.005`;
- candidate class-1 precision `>= best_ref + 0.010`;
- candidate restricted false positives are at least two below both references;
- candidate class-1 TP is no more than one below the larger reference TP;
- candidate corrections versus keeper exceed harms, and corrections among
  restricted class-1 FP exceed broken keeper class-1 TP.

### Worst-shift holdout gates

- candidate macro F1 and class-1 F1 exceed both references, with class-1 F1
  delta at least `0.005`;
- candidate class-1 precision exceeds both references by at least `0.010`;
- candidate restricted false positives are at least three below both
  references;
- candidate class-1 TP is no more than two below the larger reference TP;
- candidate shift consistency exceeds both references without satisfying the
  gate through constant or majority-class collapse.

### Illumination safety gates

For each of dim, bright, and low-contrast, on both clean and official
worst-shift predictions:

- candidate class-1 F1 and precision may not trail either reference by more
  than `0.005`;
- candidate class-1 TP may not trail the larger reference by more than two;
- candidate restricted FP may not exceed either reference.

Additionally, candidate class-1 precision and F1 must exceed both references
under at least two of the three photometric conditions, and pooled restricted
FP across the three must be lower than both references.

## XAI and next-stage rule

If any nonvisual gate fails, status is `rejected`; do not retain a checkpoint,
run XAI, touch validation/test, integrate the trainer, or sweep shifts, update
count, LR, optimizer, padding, batch size, class weighting, or selection rule.

If all nonvisual gates pass, retain one hash-locked candidate checkpoint and
status `passed_nonvisual_xai_required`. Before any validation or production
integration, run a separately hash-locked deployment-forward XAI audit on 16
holdout rows covering restricted-FP corrections, class-1 TP safety, candidate
harms, and shift-sensitive unchanged errors. Require exact normal-forward
logit parity, native attention, grad-rollout, Grad-CAM, bbox/valid-mask overlays,
background blur/gray, object desaturation, all finite maps, and no systematic
foreground-to-border/background collapse.

Only a passed nonvisual gate plus passed XAI may authorize one separately
locked matched five-epoch scratch pair. Validation, test, a full train, engine
export, video test, and current-best command/history updates remain forbidden
at A0.

## Artifacts and retention

The sole formal run writes to
`runs\audit_spatial_worst_shift_robust_optimization_a0_20260717` and must not
overwrite an existing directory. Preserve:

- `summary.json` and `report.md`;
- exact cohort/batch/selection manifests;
- training history;
- all-condition prediction CSV and independent replay;
- artifact manifest and hashes;
- candidate checkpoint only after every nonvisual gate passes.

Temporary ONNX files are deleted after parity checks. A rejected run keeps
compact JSON/CSV/markdown evidence, records a closure in the research journal
and skill, passes retention, and leaves the current-best command count at three
revisions/two actual updates.
