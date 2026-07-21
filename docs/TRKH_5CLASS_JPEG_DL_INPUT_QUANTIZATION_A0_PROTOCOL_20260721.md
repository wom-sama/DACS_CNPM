# TRKH 5-Class JPEG-DL Input Quantization A0 Protocol

Status: prospectively locked before TRKH implementation, dataset pixel access,
candidate fitting, or candidate metric access.

## Scientific question

Can a trainable, class-agnostic 4:4:4 JPEG-DL input layer remove nuisance DCT
components that cause restricted class-1 false positives while preserving true
class-1 evidence and the keeper's wide-context behavior?

This is a train-only information gate around the frozen current keeper. It is
not a promoted image model and does not estimate final validation or test
performance. Production integration, smoke/probe training, full training, and
current-best command updates remain forbidden unless every hard A0 gate and the
manual visual gate pass.

## Primary-source and license lock

- Accepted paper: Ahmed H. Salamah, Kaixiang Zheng, Yiwen Liu, and En-Hui
  Yang, "JPEG Inspired Deep Learning," ICLR 2025.
- Proceedings PDF:
  `https://proceedings.iclr.cc/paper_files/paper/2025/file/2a50f08293e5f635655e8bec8f013d99-Paper-Conference.pdf`
- Local paper:
  `D:\DataAI\external_sources\papers\Salamah_JPEG_Inspired_DL_ICLR_2025.pdf`
- Paper SHA-256:
  `09ed1ad8c87f22f28ad2ef3efd259b4a867ebebdc6e93ea7ce0ea791ccc07b5c`
- Authors' repository:
  `https://github.com/AhmedHussKhalifa/JPEG-Inspired-DL.git`
- Local repository:
  `D:\DataAI\external_sources\official\JPEG-Inspired-DL`
- Commit/tree:
  `894fdb651326b7c68f13967886fea2ee04831b3a` /
  `c33f70fcb321ea9bc7431f4b6f9b998b5e107e84`
- License: MIT. License SHA-256:
  `7838cf441bbc8e87a1c903ad9ccc03476868334c306ee70a3f31f0b31fc3552c`
- Official source hashes:
  - `ImageNet/JPEG_layer.py`:
    `53c65c0fa52794c7e914f02cecca2674641f204a927522aceba3f86496fa954b`
  - `ImageNet/train.py`:
    `dcad3877e9c335b4f2a558d177d979dcc7a13af9cc91d7e0cd71c5001d98cb4a`
  - `TransformerBased/main_JPEG.py`:
    `60fe58066495010d3c3f6e0f60f05c5405ae7fb5ae4fe0955b3fb9b1d8b399c1`
  - `train_teacher_JPEG.py`:
    `3b9f8d6593a45aeba2d07eab4917f8225722cae1810c4d80463a6f183e498659`
  - `README.md`:
    `f24ab0a9245c07959e20bdf3c1a4788357ea475dfefbc95b770243151f1f1173`

The implementation in TRKH must be independent and small. No module from the
official repository may be imported at runtime. The official source is pinned
only for equation, implementation-semantics, and license provenance.

## Why this route is distinct

JPEG-DL is an input-space, supervised, blockwise nonlinear transform:

`RGB -> YCbCr -> 8x8 DCT -> learned soft quantization -> IDCT -> RGB`.

It is not another frequency pooling head, Fourier-amplitude augmentation,
fixed JPEG-quality augmentation, global texture statistic, prototype readout,
or post-hoc calibration method. The paper reports scratch fine-grained gains
for ResNet-18 and DenseNet-121 and smaller gains for EfficientFormer-L1. It
also reports that random JPEG-quality augmentation often hurts. Therefore this
A0 tests learned quantization, not random JPEG augmentation.

The paper's fine-grained runs use 200 CNN epochs and 600 EfficientFormer
epochs. TRKH is limited to at most 30 production epochs. This mismatch is
adverse prior evidence and is why a low-parameter frozen-keeper gate is
required before trainer integration.

## Immutable inputs and isolation

- Frozen keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Data declaration:
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`
- Data YAML SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`
- CIDT summary/predictions SHA-256:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad` /
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`
- Current-best command/history SHA-256:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`
- Population: only the immutable 9,215 `yolo_f/train` rows, class counts
  `[1941, 541, 1920, 2520, 2293]`.
- Source-disjoint holdout counts: `[1843, 1830, 1828, 1851, 1863]`.
- Locked ordered train-index SHA-256:
  `d41d14b04dfb70b60b5058f30d1c4cfe0189aef3b0efc0a22741f5672c1473c8`.
- Decision cohort: the existing 763 train-only CIDT rows containing all 541
  true class-1 rows and 222 keeper restricted `0/2/4 -> 1` false positives.
- Locked cohort-index SHA-256:
  `59d2146617f2642f99c081f49a9aa5812eae4206b27887a1fa5476f6f61c85b2`.

Validation and test paths must never be opened. Raw images, labels, YAML,
keeper files, current-best commands, and protected user payloads must remain
byte-identical. Dataset directory stat snapshots are recorded before and after
formal execution.

## Locked JPEG-DL equation

For DCT coefficient `z`, positive quantization step `q`, fixed `alpha=5`, and
the five local reconstruction indices centered on uniform rounding:

```text
r = round(z / q)
A5 = clamp(r - 2, -127, 123) + [0, 1, 2, 3, 4]
c_i = q * A5_i
w_i = softmax_i(-alpha * (z - c_i)^2)
Qd(z; q, alpha) = sum_i w_i * c_i
```

The rounded/clamped center indices are detached. Gradients flow through the
five reconstruction levels and their soft assignments to `q`, matching the
authors' executable masked-CPMF path. The paper describes a larger conceptual
support, while its implementation and Appendix A.12 use the five closest
levels. This A0 locks the executable five-level semantics and records the
paper/code support-range discrepancy rather than silently changing it.

The JPEG layer uses:

- BT.601 RGB/YCbCr conversion;
- no chroma subsampling (4:4:4);
- non-overlapping orthonormal 8x8 DCT-II and matching IDCT;
- one learned `8x8` luminance table and one shared `8x8` chroma table;
- exactly 128 trainable parameters for a full table;
- `q` initialized to all ones and projected after every optimizer step to
  `[1e-5, 255]`;
- no learned `alpha`, bitrate objective, hard rounding, image-dependent table,
  or extra JPEG round;
- no output clamp before keeper normalization, matching the official layer.

Input handling is exact: undo keeper normalization to RGB `[0,1]`, multiply by
255, subtract 128, apply JPEG-DL, add 128, divide by 255, and reapply the
keeper's checkpoint mean/std. The 256x256 keeper input is divisible by eight,
so formal execution must use zero JPEG padding. Existing `image_mask`, bbox,
crop bbox, and object-context metadata are passed unchanged to the keeper.

## Locked roles

1. `keeper_raw`: ordinary frozen keeper, no JPEG layer.
2. `fixed_q1`: untrained full JPEG-DL tables fixed at one.
3. `paper_ce_full`: 128 learned q values, standard multiclass CE only.
4. `precision_scalar`: two learned q values, one Y and one shared CbCr scalar
   expanded over all 64 frequencies, using the precision objective below.
5. `precision_full`: 128 learned q values using the precision objective. This
   is the sole candidate.
6. `precision_full_frequency_permuted`: same candidate weights, but a fixed
   seed-42 permutation independently reassigns the 63 AC entries of Y and
   CbCr; DC stays fixed and each table's value multiset is exact.
7. `precision_full_phase_shifted`: same candidate weights, but the input is
   toroidally rolled by `(4,4)` before block partition and inversely rolled
   after reconstruction. Pixel coordinates and metadata are restored before
   the keeper.

Roles 3-5 start byte-identically at q=1. Roles 6-7 are same-weight causal
interventions and are never trained. No role, loss weight, permutation, phase,
threshold, fold, seed, or epoch sweep is allowed after metric access.

## Locked optimization

- Frozen keeper in deterministic eval mode; every keeper parameter and buffer
  must be bit-exact before and after fitting.
- Seed `42`; folds `(0,1,2,3,4)`; natural-frequency fit rows only.
- Deterministic keeper eval transform; no stochastic augmentation in A0.
- Base loader batch `32`; roles 3-5 are concatenated into one effective
  keeper batch of `96` where memory permits. If a deterministic engineering
  check proves this exceeds available VRAM, the launcher may microbatch roles
  without changing gradients, order, or locked batch normalization state.
- Requested/effective Windows workers `4/4`, pin memory and persistent workers
  enabled through the established TRKH safety environment.
- Exactly two fit epochs per executed fold; no early stopping or scheduler.
- Adam for q only: LR `0.005`, betas `(0.9,0.999)`, epsilon `1e-8`, zero weight
  decay. This is the paper's fine-grained JPEG learning rate.
- FP32 JPEG transform and q optimizer. Keeper may use CUDA BF16 autocast, but
  logits and all losses are accumulated in FP32.
- Persist and hash each epoch order and q state. Every fit source must be
  absent from its held fold.

### Pre-metric engineering disposition (2026-07-21)

The first real two-row engineering forward accessed only locked train pixels and
reported no candidate decision metric. It found live keeper/CIDT probability
drift `7.596612e-5` at batch two with exact argmax, and a concatenated
three-role backward peak of `8.798651 GiB`, above the RTX 4060 physical-memory
budget. This activates the predeclared microbatch allowance as follows:

- Training roles are executed sequentially (`role_microbatch=1`) for the same
  natural batch, followed by one optimizer step over the accumulated independent
  q gradients. Keeper eval mode and every sample order remain unchanged.
- On the two-row engineering cohort, sequential and concatenated q gradients
  must be finite/nonzero, have cosine similarity at least `0.9999`, and maximum
  absolute difference at most `0.002`. A batch-32 sequential forward/backward
  must peak below `90%` of physical CUDA memory.
- The batch-two CIDT comparison remains explicit telemetry with exact argmax and
  error at most `1e-4`; it is not substituted for the locked formal tolerance.
- Before fitting any q table, a separate ordinary-keeper pass over all 9,215
  train rows uses the previously established FP32 CIDT replay batch `64`. It
  must retain exact CIDT argmax for every row and maximum probability drift
  `<=1e-6`. Failure aborts before any candidate metric or q optimization.

No validation/test row, candidate classification metric, threshold, role,
loss weight, optimizer setting, fold, seed, epoch, or decision gate was accessed
or changed by this engineering disposition.

### Pre-metric CIDT batch-shape erratum (2026-07-21)

The first formal invocation stopped in the keeper-only precondition before
output creation, q optimization, or candidate metrics. FP32 batch `32` retained
all `9,215` CIDT argmax values but reached maximum probability drift
`0.009427071`. Existing repository evidence had already established this
batch-shape effect: Supervised-Minority A0 observed up to `0.01413372`, while
its corrected A1 FP32 batch-64 contract reproduced all-row CIDT probabilities
within `8.940697e-8` with zero argmax mismatch.

Therefore the independent pre-candidate compatibility gate is corrected to the
existing FP32 batch-64 `<=1e-6` contract. Formal JPEG-DL fitting/evaluation stays
at batch `32`; its raw-CIDT comparison requires exact argmax and retains a
prospectively fixed `<=0.015` probability-drift telemetry bound. Candidate/raw
metrics are always computed from live forwards under the same batch and
metadata. No candidate scientific gate or optimization setting changes.

For logits `s`, class-1 margin
`m1 = s_1 - max_{k != 1}(s_k)`, fit-only restricted keeper FP set `R`, and
fit-only keeper class-1 TP set `T`:

```text
L_fp = mean_{i in R} softplus(m1_i + 0.15)
L_tp = mean_{i in T} softplus(0.10 - m1_i)
L_precision = CE(s, y) + 0.35 * L_fp + 0.20 * L_tp
```

An absent subset in a natural batch contributes exact zero for that term.
`paper_ce_full` uses only CE. `precision_scalar` uses exactly
`L_precision`. Masks come only from each fold's fit partition and the locked
CIDT keeper predictions; held labels never affect fitting or threshold choice.

## Prospectively staged formal execution

Formal A0 is one invocation with one immutable output directory.

### Stage A: fold-0 screen

Fit roles 3-5 on folds 1-4 and evaluate fold 0. Continue to the remaining four
folds only if all engineering gates pass and `precision_full` satisfies every
fold-0 gate versus `keeper_raw`:

- class-1 `p1` AUROC improves by at least `0.005`;
- class-1 precision improves by at least `0.005`;
- class-1 recall decreases by at most `0.02`;
- at least 5 of 36 restricted FP are removed;
- at most 2 of 107 keeper class-1 TP are broken;
- all-class macro F1 decreases by at most `0.005`;
- AUROC exceeds both `paper_ce_full` and `precision_scalar` by at least
  `0.002`;
- q/reconstruction anti-collapse gates pass.

Failure stops before folds 1-4 and shifted conditions. It still requires exact
replay, fold-0 controls, fixed fold-0 XAI, manifest, closure, and retention.

### Stage B: complete source-disjoint OOF

If Stage A passes, fit folds 1-4 under the identical lock and assemble one OOF
prediction for every train row. No fold metric may tune any setting.

## Clean OOF decision gates

All gates are conjunctive. Relative metrics compare OOF candidate predictions
with the corresponding live ordinary keeper prediction under the same batch
and metadata.

1. Candidate class-1 precision improves by at least `0.010` and is at least
   `0.72`.
2. Candidate class-1 F1 improves by at least `0.005`; recall decreases by at
   most `0.010`.
3. Keeper class-1 TP retention is at least `0.98`.
4. Restricted-FP rejection is at least `0.15` (at least 34 of 222), with at
   least 20 more corrections than harms after including FN rescue, TP break,
   FP removal, and FP creation.
5. On all 541 true class-1 rows versus 222 restricted FP, candidate `p1`
   AUROC improves over keeper by at least `0.010`, and AUPRC does not decrease.
6. All-row macro F1 decreases by at most `0.002`; no non-focus per-class F1
   decreases by more than `0.010`.
7. At least four folds have non-decreasing class-1 precision, at least three
   have non-decreasing class-1 F1, every fold retains at least `0.95` of keeper
   class-1 TP, and no fold creates more restricted FP than it removes.
8. `precision_full` exceeds both `paper_ce_full` and `precision_scalar` by at
   least `0.005` AUROC and removes at least five more restricted FP without
   breaking more keeper TP.
9. `fixed_q1` must not account for more than half of the candidate's AUROC or
   precision gain over keeper.
10. Each same-weight causal intervention loses at least `0.005` AUROC or at
    least five net restricted-FP removals relative to the aligned candidate.

## Quantizer and reconstruction anti-collapse gates

- DCT/IDCT, color conversion, soft quantizer, and block assembly are finite.
- Every learned q is finite and inside `[1e-5,255]`; at most 5% of entries may
  lie within `1e-4` of either bound.
- At least 16 of 128 full-table entries change by at least `0.01` from q=1.
- Full-table AC standard deviation is at least `0.01` in both Y and CbCr.
- Across completed folds, median pairwise Spearman correlation of aligned
  `delta_q` tables is at least `0.40`; sign-cosine similarity is at least
  `0.50`.
- On all held images, reconstructed RGB has mean PSNR at least `28 dB`, mean
  absolute RGB change at most `0.05`, absolute mean-channel shift at most
  `0.02`, and out-of-range pixel fraction at most `0.005` before display
  clamping.
- Candidate output variance must be nonzero in every channel, and no image may
  become constant, nonfinite, or more than 99% clipped for visualization.
- Full-table gains must survive source and target stratification; a single
  camera/source group may not account for more than 35% of net corrections.

## Shifted-condition gates

Open these only after every clean OOF gate passes. Reuse the learned fold
states without refitting on the same 763-row cohort:

- `dim`: brightness `0.70`, contrast `0.90`;
- `bright`: brightness `1.25`, contrast `1.10`;
- `low_contrast`: brightness `1.00`, contrast `0.65`.

For every condition, candidate class-1 precision must not decrease from the
condition-specific keeper, TP retention must be at least `0.95`, restricted FP
must not increase, and AUROC may decrease by at most `0.01`. At least two of
three conditions must improve class-1 F1. All reconstruction anti-collapse
checks remain active.

## Numeric, gradient, export, and resource gates

Before formal candidate metrics:

- Torch FP64 DCT/IDCT must match an independent NumPy matrix oracle within
  `1e-10`, with round-trip error at most `1e-10`.
- Torch FP64 RGB/YCbCr round trip must be within `1e-10`.
- Five-point Qd must match an independent NumPy FP64 implementation within
  `1e-10`; at `q=1, alpha=5, z=0.5`, it must agree with the paper's full-space
  reference within `1e-6`.
- Autograd q gradients must be finite, nonzero, and match central finite
  differences within `5e-4` relative error away from rounding boundaries.
- No-quantization bypass must reconstruct RGB within `1e-6` in FP32.
- Ordinary and metadata-aware keeper forwards must be bit-exact before adding
  JPEG-DL. The FP32 batch-64 compatibility pass must match CIDT argmax for every
  row with probability drift `<=1e-6`; live batch-32 evaluation retains exact
  argmax and probability-drift telemetry `<=0.015`.
- Every trained role receives a finite, nonzero q gradient in a real two-row
  engineering forward. Frozen keeper state remains exact.
- FP32/FP64 candidate probability error is at most `1e-4`.
- FP32-JPEG plus BF16-keeper probability error versus all-FP32 is at most
  `5e-3`, with at least `99.5%` argmax agreement.
- Static JPEG-layer ONNX export passes ONNX checker and ONNX Runtime with max
  reconstructed-input error `<=1e-5`; no custom domain is permitted.
- Warm batch-32 candidate inference latency is at most `1.45x` ordinary keeper
  latency. Additional peak CUDA allocation is at most `2 GiB` and total peak
  allocation must fit the RTX 4060 8 GiB GPU.
- Requested/effective loader workers, GPU identity, CUDA versions, latency,
  throughput, peak allocation, and external GPU contention snapshot are
  recorded.

## Fixed XAI and visual gate

If only Stage A executes, render the lowest-index fold-0 keeper TP, keeper FN,
and restricted FP. If Stage B executes, render the same three categories for
every fold, for 15 fixed rows total.

Each contact-sheet row contains:

- original wide-context crop with valid support and bbox;
- `fixed_q1`, `paper_ce_full`, `precision_scalar`, and `precision_full`
  reconstructions;
- magnified signed RGB difference and absolute-difference maps;
- raw/candidate Grad-CAM and genuine transformer attention or explicitly
  labeled fallback maps;
- phase-shifted and frequency-permuted reconstructions/maps;
- target, fold, source, raw/candidate probabilities, transition, PSNR, and
  per-image color-shift telemetry.

Also render Y/CbCr initial/final/delta q heatmaps in natural 8x8 frequency
layout for every executed fold. Automatic checks require finite deterministic
maps, zero attribution in padding, exact row selection, unchanged keeper
state, exact argmax, and batch-1 versus formal batch-32 probability error at
most `3e-3`. Manual review passes only if aligned JPEG-DL consistently preserves
or sharpens fruit-surface/lesion/ripeness evidence while reducing distractor
texture; broad blur, silhouette-only response, background removal, global
color suppression, checkerboard artifacts, or indistinguishable placebos fail.

The summary remains `pending_manual_visual_review` until a second-process
hash-locked finalization records `pass` or `fail`. No scientific metric may be
edited during finalization.

## Replay, artifacts, and retention

Persist only:

- resolved command/config/provenance and source hashes;
- fold/epoch orders, loss telemetry, q states, optimizer states, and state
  hashes;
- OOF probabilities/transitions and condition metrics for stages actually
  opened;
- engineering, numeric, export, resource, reconstruction, and XAI summaries;
- fixed contact sheets and q heatmaps;
- ONNX graph, final summary, and manifest with every payload SHA-256.

Do not persist a full image tensor cache. Independent replay must reconstruct q
states, probabilities, metrics, controls, gates, and contact-sheet selections.
Any numeric mismatch remains a failure; gates are never loosened after metric
access. Read-only retention must verify all protected artifacts and every
historical cleanup manifest before closure.

## Promotion and stop rules

- Any Stage-A failure closes this exact frozen-keeper JPEG-DL objective,
  q-initialization, alpha, five-point support, fold, seed, epoch, LR, scalar,
  frequency-permutation, and block-phase neighborhood. Do not sweep them after
  observing metrics.
- A clean OOF, shifted-condition, engineering, causal-control, replay, and
  manual-XAI pass authorizes one production integration with q initialized to
  one and jointly trained from scratch. It does not itself authorize command
  promotion or a full train.
- Integration must first pass compile, focused/full tests, one smoke, complete
  validation, and all available XAI/audit surfaces. Only a genuine locked
  validation improvement may update the current-best full-pipeline command.
- A future full train may use at most 30 epochs with reasonable early stopping
  and a separately measured worker count. It is authorized only after the
  integrated smoke/probe beats the existing gate with higher class-1 precision
  and no material all-class regression.
