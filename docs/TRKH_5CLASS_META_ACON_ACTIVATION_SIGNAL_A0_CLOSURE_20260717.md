# TRKH 5-Class Meta-ACON Activation Signal A0 Closure - 2026-07-17

## Decision

Status: rejected before trainer/model integration and before any image epoch.

The prospectively locked CVPR-2021 Meta-ACON fit-only gate exposed a small
sample-conditioned activation signal, but it was too weak and unstable to
support the class-1 precision objective. Clean OOF AUROC was `0.628292`, only
`+0.014814` over identity and `+0.015031` over static ACON. Clean class-1 TP
retention was `0.883610` and restricted-FP rejection was `0.161290`, both below
their locked gates. Only two of four source folds improved over static ACON.

No Meta-ACON trainer option, five-epoch pair, checkpoint, validation, test,
full train, or current-best command update is authorized. Raw data, labels,
bboxes, splits, and the keeper remain unchanged.

## Primary Sources And Screen

Authority was the accepted paper and official licensed source:

- Ma et al., "Activate or Not: Learning Customized Activation," CVPR 2021:
  `https://openaccess.thecvf.com/content/CVPR2021/html/Ma_Activate_or_Not_Learning_Customized_Activation_CVPR_2021_paper.html`;
- accepted PDF SHA-256: `acae6d8f...2f9218`;
- official MIT repository: `https://github.com/nmaac/acon`, observed at about
  207 stars during the screen;
- locked official commit/tree:
  `99fd67928a6ffb0543b54614303caada96c756f5` /
  `f269fee6201f0945fc154134f310704b59175837`;
- official `acon.py`/license SHAs: `670d4d70...e4087` /
  `e09e657f...ece60`.

The paper initializes `beta=p1=1, p2=0`; the official file initializes `p1/p2`
randomly. A0 followed the paper initialization and replayed the official
equation with copied state. Protocol
`TRKH_5CLASS_META_ACON_ACTIVATION_SIGNAL_PROTOCOL_20260717.md` has corrected
SHA-256 `c1fb84b3...2e436`. Its pre-measurement erratum removed an incorrect
ReLU between beta projections after both paper and source review. No cohort,
optimization, threshold, gate, or formal measurement existed before that fix.

The source screen also rejected nearby options before implementation:

- ODConv and API-Net were already closed by local no-repeat evidence.
- Official CVPR-2021 Involution at the actual third-stem geometry cost
  `3.640968x/7.428328x/7.145325x` runtime and
  `7.436997x/8.082161x/8.201651x` peak memory at batches `1/4/8`; its official
  fast path also relies on CuPy/custom CUDA.
- Focal Modulation overlaps closed local/global and large-kernel modulation
  families.
- Dynamic ReLU had an accepted paper but no author-maintained licensed source
  suitable for the locked implementation screen.

## Locked Cohort And Replay

The sole formal run is
`runs/audit_meta_acon_activation_signal_a0_20260717`. It used only the 607
predeclared fit rows:

```text
class-1 true positives                 421
restricted {0,2,4}->1 false positives 186
fold 1 TP/FP                           112 / 45
fold 2 TP/FP                           100 / 48
fold 3 TP/FP                           101 / 52
fold 4 TP/FP                           108 / 41
ordered-index SHA                      a2689d1b...c738bd
```

The keeper deployment replay matched 606/607 declarations. Sample `3657`,
target class 2, changed from declaration class 1 to replay class 2. The locked
declaration had `p1=0.23123835` and `p2=0.23106094`, only `0.00017741` apart.
This remains a strict structural failure.

A read-only sensitivity replay excluded that one negative from the already
written CSV without refitting. Meta-ACON clean AUROC changed only from
`0.628292` to `0.628183`, TP retention stayed `0.883610`, and FP rejection rose
only from `0.161290` to `0.162162`. The mismatch cannot rescue any material
information failure.

Independent replay of all `2,428 = 607 x 4` CSV rows with `sklearn` reproduced
every role/condition metric and all four fold deltas exactly
(`max_abs_diff=0`). Every sample had all four conditions, metadata was stable,
scores were finite, thresholds were constant per role/fold, and each source
appeared in one fold only.

## Equation And Deployment Audit

The implementation was faithful, active, and deployable:

```text
official output max error                 4.44e-16
independent output/beta max error         0 / 0
maximum parameter-gradient error          3.66e-15
finite-difference error                   7.27e-11
BF16 versus FP32 max error                0.015097
between-sample beta RMS                   0.225375
activation-delta RMS mean                 0.266386
added parameters                          9,520
runtime / peak-memory ratio               1.082251 / 1.000037
ONNX Runtime max error                    4.76837e-7
TensorRT parse / serialized build         true / true
```

All output and gradient checks were finite and nonzero where required. ONNX
used only standard `Add/Conv/Mul/ReduceMean/Sigmoid/Sub` operators. The ONNX
and TensorRT engine were ephemeral; no checkpoint, engine, reusable feature
cache, or raw image copy was retained.

## Fit-Only Information Result

Four source-disjoint OOF binary readouts used identical head initialization,
clean-only fitting, shared permutations, and fit-only recall-constrained
thresholds.

| Condition | Identity AUROC | Static ACON AUROC | Meta-ACON AUROC | TP retention | FP rejection |
|---|---:|---:|---:|---:|---:|
| Clean | 0.613478 | 0.613261 | 0.628292 | 0.883610 | 0.161290 |
| Dim | 0.597694 | 0.594961 | 0.607476 | 0.755344 | 0.365591 |
| Bright | 0.607833 | 0.608663 | 0.621242 | 0.907363 | 0.118280 |
| Low contrast | 0.613082 | 0.610822 | 0.611422 | 0.864608 | 0.193548 |

Clean Meta-ACON precision/F1 were `0.704545/0.783983`, versus static ACON
`0.697543/0.776842`. This modest threshold result does not compensate for low
AUROC separation, insufficient TP protection, or fold instability.

| Fold | Meta minus static clean AUROC | Meta clean precision | Meta clean TP retention |
|---:|---:|---:|---:|
| 1 | -0.005357 | 0.722222 | 0.928571 |
| 2 | -0.020625 | 0.684615 | 0.890000 |
| 3 | +0.042079 | 0.682540 | 0.851485 |
| 4 | +0.026649 | 0.726562 | 0.861111 |

Object-only beta clean AUROC was `0.623720`, only `+0.010459` over static ACON,
and its TP retention fell to `0.743468`. Full-map beta is active but does not
encode a sufficiently reliable object-conditioned class-1 precision signal.
The information gate failed 9 of 16 checks, including clean AUROC, both clean
deltas, fold stability, clean/shift TP retention, clean FP rejection, and the
object-beta gain.

## Artifacts And Retention

Formal artifact-manifest SHA-256 is `683b5cbb...cc20`. It records four payload
files totaling `730,512` bytes:

- OOF CSV SHA `f656d47a...1a1b8`;
- fold diagnostics SHA `9c8f69bd...36aa3`;
- summary SHA `1ef2786d...4f72`;
- report SHA `045d55aa...e89`.

The compact formal directory is retained intact. Post-Meta-ACON retention
`runs/audit_trkh_artifact_retention_post_meta_acon_20260717` passed over 740
run directories with `blockers=[]`, all 34 compacted originals absent, no
deletion or raw-data touch, and `70.283 GB` free. Its summary SHA is
`8d48a3fc...2b4c4`.

Current-best command/history hashes remain `36b9aa1a...9faf` and
`39bd2879...8f53`; tracking remains three revisions and two promoted updates.

Closure verification passed Python compileall, `pyflakes`, six PowerShell
parses, focused Meta-ACON tests `8/8`, and full pytest `1,353/1,353`. The 273
warnings were existing dependency deprecations and tracing notices, not test
failures. The official ACON checkout remained clean with no `__pycache__`, and
the formal manifest replayed every payload size/hash exactly with no retained
checkpoint, ONNX, engine, or reusable cache.

## No-Repeat Rule

- Do not integrate or train this post-stem Meta-ACON route.
- Do not rescue it by sweeping initialization, beta generator/reduction,
  insertion layer/count, activation family, LR, optimizer, epochs, batch,
  weight decay, class weighting, folds, thresholds, bbox masks, conditions, or
  residual scales.
- Do not reinterpret beta variance, equation/export success, mean clean gain,
  or two positive folds as five-epoch permission.
- Do not reopen Involution at this geometry without a standard-op path that
  prospectively passes both runtime and memory limits.
- This closure rejects the exact train-only activation-readout signal and its
  nearby hyperparameter variants; it does not claim Meta-ACON is generally
  ineffective in other architectures or datasets.
- Select the next route from a distinct accepted primary source with licensed
  official code and a cheap gate that tests restricted-FP separation plus
  class-1 TP safety before image training.
