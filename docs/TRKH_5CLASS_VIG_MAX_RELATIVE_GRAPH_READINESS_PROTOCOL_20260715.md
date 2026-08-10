# TRKH 5-Class ViG Max-Relative Graph Readiness Protocol - 2026-07-15

## Status

Precommitted before implementation. This document locks one exact train-only
readiness experiment and, only if every Stage-A gate passes, one
keeper-initialized validation smoke. It does not authorize test access, a full
train, or a parameter sweep.

## Research Question

Can a sparse, content-adaptive graph over patch tokens improve class-1
surface/boundary precision while preserving the keeper's wide-context CNN
stem, spatial MHSA, pruning, class-1 true positives, and image-space XAI path?

Current audits repeatedly show that the keeper already localizes fruit and
that far-background perturbations are weak. The unresolved errors are
foreground class boundaries, especially `0/2/4 -> 1` false positives. A useful
graph branch must therefore connect patches with similar learned appearance
even when they are spatially separated and must produce a class-selective
decision change, not merely a live residual.

## Primary-Source Lock

- Paper: Han et al., "Vision GNN: An Image is Worth Graph of Nodes," NeurIPS
  2022.
- Proceedings:
  `https://proceedings.neurips.cc/paper_files/paper/2022/hash/3743e69c8e47eb2e6d3afaea80e439fb-Abstract-Conference.html`
- Paper PDF:
  `https://proceedings.neurips.cc/paper_files/paper/2022/file/3743e69c8e47eb2e6d3afaea80e439fb-Paper-Conference.pdf`
- Official repository: `https://github.com/huawei-noah/Efficient-AI-Backbones`
- Locked repository subproject: `vig_pytorch`
- Locked repository commit:
  `f90e129b645c3b1684fe07cd361cd557d0ad71f7`
- Reviewed official files and SHA-256:
  - `vig_pytorch/gcn_lib/torch_vertex.py`:
    `c0b198b0f21fe20947de3317b3df500ffc76a224fe5fa46bc3c5585b5a465ef5`
  - `vig_pytorch/gcn_lib/torch_nn.py`:
    `28a9b32f630b0774823c5ff315bdccad4ffd7e1affdb1dadd068e8e39388cff9`
  - `vig_pytorch/gcn_lib/torch_edge.py`:
    `88bcd6b8bdeb628f2f9fb21e1dfe3affe31d833167402e9f12063544634cde53`
  - `vig_pytorch/pyramid_vig.py`:
    `ae596df75b019fad2e59080cf3a884276f65d695b3062dedad8cf3095fdd1089`

The official graph builder L2-normalizes node features, computes dense
pairwise distances without gradient, and selects nearest neighbors with
`topk(-distance)`. The official MRConv gathers center and neighbor features,
takes `max_j(x_j - x_i)`, concatenates it with `x_i`, and projects the result.
The official PVIG recipe uses `k=9`. These facts lock the graph equation below;
the small zero-init residual is an explicit TRKH adaptation.

## Local Architecture Screen And No-Repeat Boundary

### Re-Attention is not selected

DeepViT Re-Attention targets deep attention collapse, where maps become nearly
identical across layers. A read-only train-only diagnostic used 80 fold-0 rows,
exactly 16 per class, and all eight keeper attention layers. Adjacent full-map
cosines were `0.398-0.645`, adjacent same-head CLS-to-patch cosines were
`0.078-0.313`, late-layer inter-head full-map cosines were `0.113-0.185`, and
late-layer effective ranks were `7.52-7.74` out of eight. The keeper therefore
has evolving layers and diverse heads rather than the reported collapse
mechanism. Re-Attention would mix already-specialized heads and is closed by
this exploratory train-only mechanism screen.

### ODConv is not reopened

ODConv commit `0901cc1ccd75757b40420447db20603d30bd2f74` and official
`modules/odconv.py` SHA
`68198a42ffa4900e8321eeb0c51bd6fab5a58ebfe282cd3c8ca67d0bbea66537`
were reviewed. Its multiplicative kernel/channel/filter/spatial attention
overlaps the already failed dynamic/multiplicative convolutional stem family,
and official one-kernel recipes use long 100-150 epoch schedules. This was
already excluded in the XCA protocol and is not a new candidate.

### ViG boundary

This route must not silently become any previously closed family:

- no local depthwise convolution, LeFF, concurrent local-global path, LPI, or
  relative-position attention;
- no XCA/PWCA/channel covariance or attention-head mixing;
- no Gabor, frequency, OctConv, InceptionNeXt, StarNet, MogaNet, tokenizer, or
  stem replacement;
- no classifier/verifier/router, teacher, calibration, loss, augmentation,
  sample-weight, threshold, or class-1 recall-pressure change;
- no graph dilation, stochastic neighbor selection, relative position,
  multi-scale pooling, or graph-type sweep.

## Exact TRKH Adaptation

Method identifier: `vig_max_relative_graph`.

1. Preserve the keeper's `conv_pool` CNN stem, patch projection, prefix tokens,
   all eight spatial-MHSA blocks, FFNs, pruning after layers `2,5`, classifier,
   loss, transforms, bbox/foreground priors, and XAI source.
2. Add one patch-only dynamic graph residual after spatial MHSA and before the
   FFN in one-based Transformer layers `2,5`.
3. For current patch tokens `X in R^(B x N x 256)`, apply a separate
   `LayerNorm(256, eps=1e-6)` and a trainable deterministic projection
   `Linear(256,64,bias=false)`.
4. Initialize the projection without RNG: output row `r` averages input
   channels `4r..4r+3` with weight `0.5`. This preserves unit variance under
   independent standardized channels and remains trainable.
5. L2-normalize the 64-dimensional projected nodes. Under `no_grad`, compute
   squared pairwise Euclidean distance and select the nearest `k=9` nodes with
   `topk(-distance)`. Match the official implementation by allowing self as one
   of the nine neighbors.
6. Gather neighbors and compute the official max-relative message
   `m_i = max_j(x_j - x_i)` independently per channel. Concatenate
   `[x_i,m_i] in R^128`.
7. Apply `Linear(128,256,bias=true)` followed by GELU and add it to patch
   tokens. Initialize this output weight and bias to zero, so the candidate is
   exactly keeper-equivalent before adaptation and consumes no random draws.
8. Prefix tokens receive no direct graph residual. Graph construction uses
   only current surviving patch tokens, so it remains compatible with pruning.
9. Add no dropout, BatchNorm, learnable residual scale, auxiliary head, graph
   loss, external data, or pretrained weight.
10. Keep spatial MHSA as the returned attention for pruning and XAI. Expose
    graph neighbor and residual telemetry separately; neighbor matrices must
    never be presented as spatial attention.

Default-off configuration surface:

- `dynamic_graph_mixer=false`
- `dynamic_graph_mixer_layers="2,5"`
- `dynamic_graph_mixer_bottleneck_dim=64`
- `dynamic_graph_mixer_k=9`

Each selected layer adds `49,920` trainable parameters: LayerNorm `512`, input
projection `16,384`, and output projection `32,768+256`. The exact two-layer
candidate adds `99,840` trainable parameters and ten checkpoint entries.

## Locked Inputs

- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Keeper launcher-args SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`
- Data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`
- Data YAML SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`
- Train-only fold declaration:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`
- Fold CSV SHA-256:
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`
- Companion summary SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`

The declaration contains exactly `9215` ordered clean train rows. Fold 0 is a
source-disjoint `1843`-row holdout and folds 1-4 contain `7372` fit rows.
Validation and test paths are forbidden in Stage A. This is a differential
readiness audit over a representation that historically saw train, not a
paper/reportable OOF metric.

## Stage A: Train-Only Readiness

### Locked execution

- CUDA, deterministic algorithms, cuDNN benchmark and TF32 off;
- seed/fold `42/0`, batch size `32`;
- deterministic keeper evaluation transform only;
- one NumPy seed-42 permutation of fold-not-0 rows;
- exactly `30` fit batches (`960` rows);
- AdamW, LR `3e-4`, weight decay `0.01`, no scheduler;
- hard-label cross entropy only;
- control trainable parameters: `head.weight`, `head.bias`;
- candidate trainable parameters: the same head plus both graph modules;
- freeze every other tensor and keep all existing dropout/BN modules in eval;
- separately construct the matched pair and replay the same batch order and
  forward RNG states;
- evaluate unadapted/adapted variants on the same ordered fold-0 holdout;
- construct no validation or test dataset/loader.

### A1. Source, schema, and deterministic construction gates

All must pass:

1. Every locked local/official source commit and SHA matches.
2. Fit and holdout counts/classes are exact with zero source overlap.
3. Default-off control strict-loads the keeper and remains bit/behavior exact.
4. Candidate missing keys are exactly the five graph tensors in layers `2,5`;
   every pre-existing tensor is bit-identical after load.
5. Added trainable parameters equal `99,840`; only layers `2,5` contain graph
   modules; all eight spatial-MHSA blocks and pruning positions are unchanged.
6. Complete CPU/CUDA RNG states after construction and forward equal control.
7. Config, checkpoint resume extension, and V8 PowerShell CLI round-trip the
   feature while default-off schema remains exact.

### A2. Equation, patch-only, gradient, and graph mechanism gates

All must pass in FP32 and BF16 where applicable:

1. Pairwise distances, selected neighbors, max-relative message, and output
   match an explicit equation reference within numerical tolerance.
2. Candidate logits and every pre-existing activation are exact at zero-init.
3. Direct graph application changes no prefix token.
4. Input projection, output projection, graph LayerNorm, shared head, and input
   gradients are finite; every family receives a nonzero gradient across the
   locked adaptation and every graph tensor moves.
5. After adaptation, each layer's residual-to-patch norm ratio is at least
   `0.005` and no more than `0.20`.
6. Neighbor indices are finite/in-range, include exactly nine entries, and use
   at least four distinct non-self neighbors per sample on average.
7. Mean non-local neighbor fraction is at least `0.20`, where non-local means
   Chebyshev grid distance greater than one when original patch indices are
   available. This proves the route is not merely another local mixer.
8. Mean normalized neighbor-selection entropy across nodes is at least `0.20`;
   one repeated neighbor set for all nodes is forbidden.
9. Bypassing either graph layer changes logits by at least `1e-4`.

### A3. Export and resource gates

All must pass:

1. Export an isolated graph-equation wrapper and a full metadata-aware
   candidate at static batch `1`, opset 17. ONNX Runtime maximum logit error
   must be `<=1e-5` with matching argmax.
2. Full export accepts `images`, `bbox`, and `image_mask` and makes no dynamic
   batch claim.
3. Candidate median BF16 forward/backward runtime at batch 32 is no more than
   `1.30x` matched control across three repeats.
4. Candidate peak allocated VRAM is no more than `3.25 GiB`.

### A4. Train-only hard-decision gates

For the unadapted candidate require macro-F1 delta `>=-0.010`, class-1 recall
delta `>=-0.020`, and net class-1 TP breaks minus FN rescues `<=3`.

For adapted candidate minus adapted control, every gate must pass:

- changed decisions `>=2`;
- macro-F1 delta `>=-0.002`;
- class-1 F1 delta `>=+0.002`;
- class-1 precision delta `>=+0.005`;
- class-1 recall delta `>=-0.010`;
- correct restricted `0/2/4 -> 1` FP removals `>=2`;
- class-1 TP breaks no greater than FN rescues;
- corrections strictly greater than harms;
- new `3 -> 2` harms `<=2`;
- maximum nonfocus-class F1 drop `<=0.015`.

### A5. Train-only robustness gates

Evaluate the adapted pair on the same holdout under:

- dim: brightness `0.70`, contrast `0.90`;
- bright: brightness `1.25`, contrast `1.10`;
- low contrast: brightness `1.00`, contrast `0.65`.

Require candidate-minus-control macro F1 nonnegative in at least `2/3`,
class-1 precision delta nonnegative in at least `2/3`, worst class-1 precision
delta `>=-0.010`, worst class-1 recall delta `>=-0.030`, aggregate restricted
focus-FP removals greater than creations, and no condition with more than three
net class-1 TP breaks.

Any material Stage-A failure closes this exact route. Do not sweep k, layers,
bottleneck, graph type, initializer, residual form, LR, weight decay, budget,
fold, seed, trainable parameters, loss, augmentation, or run length.

## Stage B: Sole Keeper-Initialized Validation Smoke

Stage B is authorized only if every Stage-A gate passes.

- Run exactly one `120 train batches x 2 epochs` smoke on full
  `yolo_f/val=2606`.
- Resume the locked keeper and reset epoch/optimizer/scheduler/scaler.
- Train only the head and two graph modules with AdamW `LR=3e-4`, weight decay
  `0.01`; scheduler horizon `10`, patience `1`, and `skip_final_test=true`.
- Preserve keeper transforms/losses and add no auxiliary signal.
- Require independent raw-checkpoint reload; EMA/trainer-only metrics cannot
  promote the candidate.

Minimum absolute validation gates relative to the independent keeper:

- macro F1 `>=0.884925`;
- class-1 F1 `>=0.683261`;
- class-1 precision `>=0.613093`;
- class-1 recall `>=0.768212` and at least `116/151` class-1 TP;
- class-1 FP `<=73`;
- corrections greater than harms and at least four net restricted focus-FP
  removals;
- maximum nonfocus-class F1 drop `<=0.010`;
- new `3 -> 2` harms `<=2`;
- all robustness, graph mechanism, resource, export, and provenance checks pass.

Complete full confusion/forensics, source-group review, five-condition
robustness, five-class architecture trace, graph telemetry, and exact changed-
case paired native-attention/rollout/Grad-CAM plus background/object
perturbations. Require zero attribution fallback. Failure closes the route with
no five-epoch continuation, probe, full train, test access, or nearby sweep.

## Promotion And Command Boundary

Passing Stage A only permits Stage B. A Stage-B pass permits one separately
precommitted confirmation, not automatic full train or test access. Update the
current-best full-pipeline command and its history only after an independently
reloaded single checkpoint materially wins the locked validation gate and
passes all audit/export checks. Otherwise command SHA-256 remains
`36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`.
