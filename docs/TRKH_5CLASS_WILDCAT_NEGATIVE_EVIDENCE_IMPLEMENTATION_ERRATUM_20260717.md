# TRKH 5-Class WILDCAT Negative-Evidence Implementation Erratum

Date: 2026-07-17
Status: immutable before the XAI correction, behavioral-result access, or corrected formal run

## Scope

This erratum does not change the method, dataset, source-disjoint split,
training order, head equations, initialization, optimizer, epoch budget,
holdout conditions, decision gates, deployment gates, or no-repeat boundary in
`TRKH_5CLASS_WILDCAT_NEGATIVE_EVIDENCE_READINESS_PROTOCOL_20260717.md`.
It corrects one XAI-only execution defect discovered by the first formal
attempt at implementation commit `32316ff1061a18faa8d204ff3cc655281e3d99b9`.

No summary was produced and no prediction, mechanism, or training metric was
read before this erratum was written. The failure is not a method gate result.

## Interrupted attempt

The first attempt completed the locked feature cache, all 20 head epochs, all
four train-holdout conditions, metric CSV serialization/replay, resource
benchmark, ONNX export, and TensorRT build. It then stopped before XAI output
or `summary.json` with:

`adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation`

Strict `torch.use_deterministic_algorithms(True)` correctly converted that
known CUDA-backward limitation into a `RuntimeError`. PyTorch documents that
`warn_only=True` retains deterministic selection where available but emits a
warning instead of throwing when an operation has no deterministic
implementation:
`https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html`.

The interrupted output is preserved at
`runs/audit_wildcat_negative_evidence_a0_interrupted_xai_determinism_20260717`.
Its manifest SHA-256 is
`4cf7cc1805bae15d6f2766d392752fbb7ca01685ad2a85404f82e01f4772f845`.
It contains no checkpoint, feature cache, XAI artifact, or summary. The five
pre-XAI references are:

- `training_curve.csv`:
  `ccbe8431f6f3ca378e25b141728d7b32fbe64f67481950e1f73355fd6632e977`;
- `predictions_all_conditions.csv`:
  `15daa1f74d6317af13684550b06b6a821868d62c9e7ddc34868145b6fe121e6c`;
- `mechanism_all_conditions.csv`:
  `998539c9a6d7ce204693b1619e2644e6931df32526b9a411e7d677f157ab05ad`;
- `wildcat_head.onnx`:
  `380c0f094916270f2af9446c6798e938479a026284f7dbffadd78fee4c19c3eb`;
- `wildcat_full_wrapper.onnx`:
  `73481247dd4428b8e5f4b229d6d7ed353b9d9ff9906e1ace6ac1ea34cec178fc`.

## Locked XAI correction

- Keep strict deterministic algorithms enabled for feature extraction,
  training, holdout evaluation, replay, resource measurement, and export.
- For only the input-saliency backward scope, switch to
  `torch.use_deterministic_algorithms(True, warn_only=True)` and restore the
  exact previous strict/warn state in `finally`.
- Execute two independent forward/backward passes for every selected XAI
  batch. Save their arithmetic-mean absolute input saliency.
- Require repeated class maps to be bit-exact, top and bottom indices to be
  exact, saliency maximum absolute error at most `1e-9`, normalized-saliency
  maximum absolute error at most `1e-6`, minimum cosine similarity at least
  `0.999999`, and saliency-argmax mismatches equal to zero.
- Record the two-pass contract, repeat diagnostics, warning-only scope, and
  strict-state restoration in the XAI manifest and all-or-nothing gate.
- A three-pass one-image implementation smoke before this lock had zero raw
  and normalized saliency error, cosine `1.0`, zero argmax mismatches, exact
  class maps/indices, and restored strict state. It accessed no label,
  prediction, or behavioral metric.

## Corrected-run authorization

Exactly one corrected formal run may use the original output path after the
XAI fix, focused tests, commit, push, and no-output preflight pass. Before any
XAI forward, that run must reproduce all five pre-XAI file SHAs above exactly.
Any mismatch aborts and rejects the recovery; it cannot be explained away as
timing noise or followed by metric inspection.

The interrupted directory remains immutable evidence. The corrected run may
not read validation/test data, modify raw data, write a checkpoint, update a
current-best command, or alter any original Stage-A threshold. Only a terminal
corrected summary followed by complete manual contact-sheet review can decide
the WILDCAT gate.
