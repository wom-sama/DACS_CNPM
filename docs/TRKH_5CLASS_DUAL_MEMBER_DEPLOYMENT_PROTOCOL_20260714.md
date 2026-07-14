# TRKH 5-Class Dual-Member Deployment Protocol - 2026-07-14

## Scope

This protocol packages the already frozen precision ensemble as one
self-contained inference checkpoint. It does not train, tune, recalibrate, or
open the test split. The deployed model must preserve the locked validation
decision rule exactly before it can replace the single-checkpoint keeper in
export or video commands.

The two complete member networks remain independent inside the package. Prior
output-distillation and feature-stitching attempts did not preserve the same
behavior, so this stage must not compress, share, average, or remap weights.

## Locked provenance

- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Scratch complement checkpoint:
  `runs/full_v8_yolof_randominit_30e_20260714_105524/checkpoints/best.pt`
- Scratch complement SHA-256:
  `f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549`
- Frozen validation protocol:
  `runs/full_v8_yolof_randominit_30e_20260714_105524/final_audit_manual/precision_ensemble_readiness_val_20260714/summary.json`
- Frozen protocol SHA-256:
  `6c6397d3561ec917d9716ecfd2c6b65a616433e3a07c45031df898578d07a8c8`
- Frozen rule: keeper weight `0.60`, complement weight `0.40`, focus class
  `1`, focus decision margin `0.034`.
- Frozen validation target: macro/class-1 F1
  `0.8902975586541526/0.696774193548387`, class-1 precision/recall
  `0.6792452830188679/0.7152317880794702` over exactly `2606` rows.

The builder must verify these hashes, the class order, model input geometry,
input normalization, and every evaluation-affecting transform field. It must
refuse a nonempty output directory or any incompatible member pair.

## Exact runtime contract and rule

The frozen validation path supplies three tensors: normalized `images`, the
post-resize `image_valid_mask`, and the transformed object `bbox` in normalized
`xywh` coordinates. Both members use token pruning and a bbox-spatial head.
An image-only forward is therefore not equivalent to the audited classifier.
The packaged PyTorch model accepts optional metadata for diagnostic fallback,
but its certified ONNX/TensorRT interface has three inputs:

- `images`: `[1,3,256,256]`, FP32;
- `image_valid_mask`: `[1,256,256]`, FP32 values in `{0,1}`;
- `bbox`: `[1,4]`, FP32 normalized `xywh`.

The certified export is deliberately fixed to batch `1`. Legacy tracing in
the member routing heads materializes Python batch dimensions, and a preflight
graph that advertised a dynamic batch failed at runtime for batch `2`. Video
inference is batch `1`; no backend may advertise larger batches until a
separate row-level parity audit certifies them.

Object-crop evaluation and a detector/crop runtime can produce all three
inputs. A video frame without an ROI may use an all-valid mask and full-frame
box only as an explicitly unverified fallback; it cannot inherit the locked
validation metrics.

For member logits `z_k` and `z_c`:

1. `p_k = softmax(z_k)` and `p_c = softmax(z_c)` in FP32.
2. `p_blend = 0.60 * p_k + 0.40 * p_c`.
3. `s = p_blend - 0.034 * one_hot(class=1)`.
4. `q = clamp_min(s, 1e-8) / sum(clamp_min(s, 1e-8))`.
5. Return `log(q)` as ordinary classifier logits.

The formal readiness CSV stores `p_blend` and uses `s.argmax()` for its locked
decision. Therefore `softmax(log(q))` intentionally equals `q`, not the raw
blend, while `argmax(q)` must equal the frozen decision. The model must expose
member probabilities, raw blend, decision scores, normalized deployment
probabilities, and logits through an explicit audit method. Inference/export
continues to receive logits only.

The operator set is limited to softmax, multiply/add/subtract, clamp, reduce
sum, divide, and log. These are standard ONNX operations and are supported by
the installed TensorRT path, but actual export and row-level parity are still
mandatory; operator availability alone is not evidence of equivalence.

## Build and PyTorch gates

1. Build one checkpoint with two complete prefixed state dictionaries and
   immutable rule buffers. Do not depend on external member files at load time.
2. Re-read the saved checkpoint and strictly load every tensor. Record hashes
   for the package, both source checkpoints, the frozen protocol, configs, and
   prefixed member states.
3. On deterministic synthetic tensors, require member probability error and
   composite-logit arithmetic error `<= 1e-6`, with exact argmax agreement.
4. Evaluate `yolo_f/val` only, in FP32, over exactly `2606` rows. Require:
   - exact ordered sample/target alignment;
   - exact `2606/2606` frozen-decision agreement;
   - macro/per-class metrics equal to the frozen summary within `1e-9`;
   - independently loaded member versus packaged-member probability error
     `<= 1e-6` in the same batch geometry;
   - no test-derived path or row.
5. Verify that evaluator and metadata-aware inference use all three inputs,
   and that export still treats the package as classification-only.
   Separately verify that image-only video is labeled as an unverified
   full-frame fallback rather than silently claiming audited parity.
   TorchScript is not emitted for this package because the traced artifact did
   not load within the preflight timeout. The certified deployment paths are
   the packaged PyTorch checkpoint and ONNX Runtime CPU only.

## ONNX and TensorRT gates

1. Export one fixed-batch-1, three-input opset-17 ONNX graph from the packaged
   checkpoint. The graph must reject unsupported batch sizes rather than label
   a hard-coded trace as dynamic.
2. ONNX Runtime FP32 must match packaged PyTorch on full validation with
   maximum probability error `<= 1e-5` and exact `2606/2606` argmax agreement.
   Both sides must run at the certified batch geometry `1`; near-tie routing
   can legitimately select a different auxiliary path under another GPU batch
   kernel, so cross-batch probability comparison is not backend parity. The
   PyTorch probability reference uses deterministic cuDNN with benchmarking
   disabled. ONNX decisions are still compared directly with all frozen
   validation decisions, independently of this numerical-reference gate.
3. A TensorRT candidate must match packaged PyTorch on full validation with
   maximum probability error `<= 5e-3` and exact `2606/2606` argmax agreement.
   Any near-tie flip blocks certification rather than being silently exempted.
4. Persist runtime versions, graph/engine hashes, timings, confusion matrices,
   per-class metrics, and row-level mismatch artifacts. A failed backend may
   be debugged, but the frozen rule and model weights may not be changed.

## Validated backend matrix

The matrix below is frozen deployment evidence, not a request to relax a gate.

| Backend | Full validation result | Status |
| --- | --- | --- |
| Packaged PyTorch FP32 | `2606/2606` frozen decisions, exact packaged-member replay, all gates passed | Certified |
| ONNX Runtime CPU FP32 | probability max error `2.980232e-7`, `0/2606` PyTorch/frozen mismatches, all gates passed | Certified |
| ONNX Runtime CUDA | targeted 34-case max error `0.006675`, one decision mismatch; keeper route dominates the error | Rejected |
| TensorRT FP16 | full-validation max error `0.014805`, two decision mismatches | Rejected |
| TensorRT FP32, no TF32, builder O0 | max error `0.008846`, `0/2606` decision mismatches, probability gate failed | Rejected |

The last TensorRT result is not certified merely because its validation argmax
matches. Its probability error exceeds the declared tolerance and can move a
future near-boundary sample. TensorRT export and video use for this package are
therefore experimental until a new engine passes the unchanged full gate.

## XAI provenance

The package has no single native attention map. XAI commands must select the
`keeper` or `candidate` member explicitly and record that selection plus the
package/member hashes. Existing paired XAI cohorts remain the behavioral
comparison. An average, maximum, or concatenation of member attention maps
must not be labeled as ensemble attention.

At minimum, replay the locked precision-transition cohort for both members in
FP32 and verify native full-grid attention provenance, Grad-CAM, rollout,
background blur/gray, center occlusion, and object desaturation. No test case
may be added or selected during this deployment stage. Package-level member
selection has been verified for both members with in-memory-only autograd and
no optimizer step. The keeper smoke remains more border-heavy in Grad-CAM on
the checked transition, while the scratch member is a useful but recall-biased
complement; neither result authorizes aggregate "ensemble attention."

## Promotion boundary

The two-forward rule is promoted only as an optional high-precision deployment
mode on packaged PyTorch FP32 and ONNX Runtime CPU FP32. It does not replace the
current single-checkpoint full-train recipe, does not certify a TensorRT engine,
and does not establish the research target of per-class F1 above `0.98`.

For live PyTorch video without an ROI provider, the package can use all-valid
mask plus full-frame bbox fallback, but this path is explicitly unverified and
must not inherit object-crop validation/test metrics. The reproducible package,
full-validation parity, ONNX CPU audit, and member-specific XAI command is
`scripts/run_trkh_precision_ensemble_pipeline.ps1`.
