# TRKH 5-Class Class-1 Boundary-Balanced CAGrad Closure

Date: 2026-07-16
Decision: reject the locked `c=0.4` CAGrad macrostep; Stage B denied

## Scope and provenance

- The immutable protocol is
  `docs/TRKH_5CLASS_CLASS1_BOUNDARY_CAGRAD_READINESS_PROTOCOL_20260715.md`,
  SHA-256
  `79047d2bd5bab18ee7b3da1c08d44b7e0224ef3677e6fc6512f6bb2ff48c9419`.
- Primary evidence was the NeurIPS-2021 CAGrad paper and official repository
  commit `dc3d48152b6196945cfd56144879b9d42353b095`. Preflight verified the
  paper, code, recipe, license, clean official worktree, keeper, launcher
  arguments, data YAML, CIDT cache, prior V-REx evidence, and protocol.
- Only source-disjoint `yolo_f/train` fit/holdout rows `7372/1843` were used.
  Validation and test were never constructed. The four environments reused
  identical sample indices and labels: clean, dim `(0.70, 0.90)`, bright
  `(1.25, 1.10)`, and low contrast `(1.00, 0.65)`.
- Each task gradient was the equal mean of CE over all `186` fit restricted
  `0/2/4 -> 1` false positives and all `432` fit class-1 rows. CAGrad used the
  sole precommitted paper transfer `c=0.4` and one all-parameter normalized
  step of ratio `1e-4`.

## Solver and gradient audit

- The deterministic nine-start SLSQP solve converged from every start. The
  uniform-start weights for clean/dim/bright/low contrast were
  `[0.45944099, 0, 0.54055901, 0]`; objective spread was `1.44e-13`.
- CAGrad had norm `7.529387` versus ERM norm `5.538492`, differed from ERM by
  relative L2 `0.400000`, and satisfied the exact radius ratio `0.400000`.
  Its minimum task dot improved over ERM by `+13.414594`; all four task dots
  were positive.
- Maximum normalized KKT residual was `1.054798e-7`, narrowly above the locked
  `1e-7` gate. The paper/code scale replay differed by `4.455624e-9` after
  FP32 tensor accumulation versus the locked `1e-10` gate, although an
  independent FP64 coefficient replay showed the direct paper/code weights
  are mathematically identical. Official-source direction replay passed at
  `5.88e-8`.
- The candidate/ERM actual update-norm mismatch was `2.19655e-7`, within the
  locked `5e-7` tolerance. These small numerical misses do not explain or
  reverse the independently reproduced behavioral failure.

## Clean fold-0 result

| Variant | Macro F1 | Class-1 F1 | Precision | Recall | Restricted FP |
|---|---:|---:|---:|---:|---:|
| raw keeper | 0.949323 | 0.849206 | 0.748252 | 0.981651 | 36 |
| aggregate-margin A-GEM | 0.951161 | 0.862903 | 0.769784 | 0.981651 | 32 |
| boundary ERM | 0.940933 | 0.823077 | 0.708609 | 0.981651 | 44 |
| boundary V-REx | 0.940933 | 0.823077 | 0.708609 | 0.981651 | 44 |
| boundary CAGrad | 0.940418 | 0.823077 | 0.708609 | 0.981651 | 44 |

- Versus raw, CAGrad changed macro/class-1 F1
  `-0.008905/-0.026129`, lowered precision `-0.039642`, and left recall
  unchanged. It changed 20 decisions with only `4/15` corrections/harms and
  created eight net restricted false positives (`36 -> 44`).
- It did not change class-1 decisions relative to ERM or V-REx and reduced
  macro F1 by another `0.000515`. It lost class-1 F1/precision
  `-0.039826/-0.061175` to aggregate-margin A-GEM and had twelve more
  restricted false positives.
- The optimized worst first-order task dot therefore did not translate into
  selective class-1 precision. It repeated the environment-mean tendency to
  widen class-1 support rather than remove incorrect class-1 predictions.

## Illumination and risk result

- Dim changed macro/class-1 F1 `-0.002376/-0.005550`, lowered class-1
  precision `-0.082799`, and created 29 net restricted false positives.
- Bright changed macro/class-1 F1 `-0.010877/-0.039171`, lowered precision
  `-0.065517`, and created 20 net restricted false positives.
- Low contrast changed macro/class-1 F1 `-0.011018/-0.017200`, lowered
  precision `-0.056447`, and created 17 net restricted false positives.
- CAGrad boundary-risk mean was `1.361059`, close to ERM `1.360547`, but its
  worst risk rose `1.410716 -> 1.414515`. Risk variance also rose
  `0.002743026 -> 0.002951173`. Thus even the local risk behavior did not
  dominate matched ERM.

## Independent verification and artifacts

- Independent CSV replay found exactly `1843` identically ordered rows in
  each condition (`7372` total), `36` immutable restricted hard rows, and
  `109` class-1 rows. It reproduced all metrics, transitions, and boundary
  risks.
- Every probability argmax matched and maximum probability-sum error was
  `1.27e-7`. Raw, aggregate-margin, ERM, and V-REx predictions/probabilities
  matched the prior V-REx CSV exactly on every row.
- Independent FP64 Gram replay reproduced the normalized KKT residual
  `1.054798e-7`, exact simplex sum, objective `38.5230831`, and zero direct
  paper/code coefficient discrepancy.
- Manifest/summary/prediction/gradient/solver SHAs are respectively
  `7f1f2a18846cd54fb7a7eafe2964a17681b99d009a70aadfaa09be54d43d1ced`,
  `dab24c6dd7dcf873e523915aadbdd1f8ef5337801278d9ffc55ce8c867a37750`,
  `92dc296345e34d3f0ac026db2dcb2f982e40af79e1179136dfbb17f1cc2b9d9f`,
  `bada3c59670d1c5346101aa37f113899fa41f6c553acebc5c51fafe74c08ed8f`,
  and
  `564a268640c4de1e4d86e655d787dc3a7eb0ae001330e8bf3e89481b8e491482`.
- Five manifest payloads total `4,482,405` bytes. No checkpoint, ONNX,
  TensorRT engine, validation/test prediction, or raw-data write was produced.

## Closure verification

- Python compilation and focused equation/gate tests passed `5/5`; full pytest
  passed `1161/1161` with `257` warnings in `50.39 s`.
- The PowerShell launcher parsed successfully. Post-run preflight reproduced
  every source hash, cohort, condition, and prior-prediction check without
  creating an output directory.
- Read-only retention audit
  `runs/artifact_retention_audit_class1_boundary_cagrad_closure_20260716`
  passed over `693` run directories with `blockers=[]`, deleted nothing, and
  reported `72.293 GB` free. Summary SHA-256 is
  `143e0c3cd20b03b29d99e84f742e677200068a6ba4c363e5d8da6cc36d17a45d`.
- Protected keeper, scratch complement, command packet, and command-history
  hashes remained exact. Command tracking remains three revisions and two
  updates after the initial revision; no CAGrad command revision occurred.

## Decision and no-repeat boundary

Stage B, validation, test, probe, full train, and current-best command
promotion are denied. Do not sweep `c`, task weights, environment definitions,
boundary weights, solver, normalized step, fold, seed, cohort, parameter set,
optimizer, or macrosteps. Do not convert the illumination recall expansion
into a post-hoc threshold or router.

This result closes condition-level CAGrad, MGDA, PCGrad, Agr-Sum/Norm,
GradNorm, and nearby gradient-combination variants on this keeper. The family
has now been tested through projection constraints, stratified memories,
variance equalization, and conflict-averse worst-direction geometry without a
precision-safe gain. Any continuation must use a genuinely different
representation or objective mechanism and preserve the raw keeper plus the
aggregate-margin comparator.
