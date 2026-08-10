# TRKH 5-Class Class-1 Boundary-Balanced V-REx Closure

Date: 2026-07-15
Decision: reject the locked beta-1 V-REx macrostep; Stage B denied

## Scope and provenance

- The immutable protocol is
  `docs/TRKH_5CLASS_CLASS1_BOUNDARY_VREX_READINESS_PROTOCOL_20260715.md`,
  SHA-256
  `4a798895a44e61b2c889b03edca7817cf5e223b6af55c4dd2a05c26ce0a48246`.
- Primary evidence was the ICML-2021 V-REx paper and DomainBed commit
  `b93c22a1cfc3b2428398272c1a116c8de1f4139e`. Preflight verified the paper,
  official code/license, keeper, launcher arguments, data YAML, CIDT cache,
  prior GEM artifacts, protocol, and clean DomainBed worktree.
- Only source-disjoint `yolo_f/train` fit/holdout rows `7372/1843` were used.
  Validation and test were never constructed. The four environments reused
  identical sample indices and labels: clean, dim `(0.70, 0.90)`, bright
  `(1.25, 1.10)`, and low contrast `(1.00, 0.65)`.
- Each environment risk was the equal mean of CE over all `186` fit restricted
  `0/2/4 -> 1` false positives and all `432` fit class-1 rows. V-REx used the
  locked population-risk variance coefficient `beta=1`; matched ERM used the
  same four risks and normalized `1e-4` parameter step.

## Gradient and risk audit

- Fit environment risks were `1.321332`, `1.448468`, `1.396166`, and
  `1.402412`; population variance was `0.002077080`.
- The ERM, variance, and V-REx gradient norms were
  `5.538492/0.164147/5.590776`. V-REx differed from ERM by only `0.029637`
  relative L2. Independent flattened equations matched within `8.67e-19`.
- The V-REx first-order objective, mean-risk, and variance changes were all
  negative, so the implementation was active and locally pointed in the
  intended analytical direction.
- Actual ERM/V-REx step ratios were
  `1.0000000914e-4/9.9999998464e-5`. Their relative update-norm mismatch was
  `1.06805e-7`, narrowly above the immutable `1e-7` gate. This numerical miss
  does not explain the much larger, independently replayed behavioral failure.
- On holdout, V-REx reduced risk variance versus ERM only
  `0.0027430256 -> 0.0027100944` (`0.987995x`, not the required `<=0.90x`).
  Mean risk rose `+0.000156`; worst risk fell only `-0.000702`.

## Clean fold-0 result

| Variant | Macro F1 | Class-1 F1 | Precision | Recall | Predicted class-1 |
|---|---:|---:|---:|---:|---:|
| raw keeper | 0.949323 | 0.849206 | 0.748252 | 0.981651 | 143 |
| aggregate-margin A-GEM | 0.951161 | 0.862903 | 0.769784 | 0.981651 | 139 |
| boundary ERM | 0.940933 | 0.823077 | 0.708609 | 0.981651 | 151 |
| boundary V-REx | 0.940933 | 0.823077 | 0.708609 | 0.981651 | 151 |

- V-REx and matched ERM changed no clean decisions relative to each other.
  Both were harmful versus raw: macro/class-1 F1 changed
  `-0.008390/-0.026129`, precision fell `-0.039642`, and recall stayed fixed.
- V-REx changed 19 raw decisions, with only `4/14` corrections/harms. It
  created eight net restricted false positives (`36 -> 44`) while rescuing
  and breaking zero class-1 cases.
- V-REx lost class-1 F1/precision `-0.039826/-0.061175` to the locked
  aggregate-margin comparator and had twelve more restricted false positives.
  Thus the variance term neither preserved the prior clean signal nor produced
  a distinct clean decision benefit over ordinary environment-mean ERM.

## Illumination result

- Under dim, V-REx raised class-1 recall `+0.119266` but lowered precision
  `-0.070565`, created 30 net restricted false positives, and was worse than
  matched ERM by class-1 F1 `-0.002104`.
- Under bright, V-REx changed macro/class-1 F1
  `-0.009941/-0.034760`, lowered precision `-0.059493`, and created 19 net
  restricted false positives. It was again worse than ERM.
- Under low contrast, V-REx changed macro/class-1 F1
  `-0.010328/-0.015012`, lowered precision `-0.052954`, and created 16 net
  restricted false positives.
- The method consistently widened class-1 support under shifted illumination.
  It did not provide the precision-selective worst-condition behavior required
  for agricultural deployment.

## Independent verification and artifacts

- Independent CSV replay found exactly `1843` unique, identically ordered
  sample indices in each condition (`7372` rows total), with exact fold-0
  provenance, `36` clean restricted hard rows, and `109` class-1 rows.
- Replay reproduced every clean/illumination comparison, transition, and all
  four variants' boundary-risk summaries with zero mismatch. All probability
  argmax values matched; maximum probability-sum error was `1.27e-7`.
- Prior raw/aggregate-margin argmax mismatch and raw-versus-CIDT mismatch were
  all zero. Every manifest size and SHA independently matched.
- Manifest/summary/prediction/gradient SHAs are respectively
  `1ed6bec4753f0f581f0bed0c558f4a65814ca6692524cdc56a935b8e8f1d3f1e`,
  `0b78da68af731456faaca8664a4b97dc111121a0a5c7630cfede1141e800babb`,
  `7ca25f7ffad843e7cc4da4cc70567c1372209f2b28e4710599536ea42edc8de9`,
  and
  `6e799fccc0a0bab8b42be4bb584883a0d36a7178b4640238892c82872be56f32`.
- Five verified nonbinary files total `3,719,745` bytes. No checkpoint, ONNX,
  TensorRT engine, validation/test prediction, or raw-data write was produced.

## Closure verification

- Python compilation and focused equation/gate tests passed `6/6`; full pytest
  passed `1156/1156` in `39.09 s`.
- The PowerShell launcher parsed successfully. Post-run preflight reproduced
  every source hash, cohort, condition, and prior-prediction check without
  creating an output directory.
- Read-only retention audit
  `runs/artifact_retention_audit_class1_boundary_vrex_closure_20260715`
  passed over `691` run directories with `blockers=[]`, deleted nothing, and
  reported `72.312 GB` free. Summary SHA-256 is
  `8ffca9fe29c8126d6c4139ddde78f1915071d7c08ec254af879795ee5fa829ac`.
- Protected keeper, scratch complement, command packet, and command-history
  hashes remained exact. Command tracking remains three revisions and two
  updates after the initial revision; no V-REx command revision occurred.

## Stop rule and next distinct screen

Do not sweep V-REx beta, environment definitions, boundary weights, normalized
step, fold, seed, cohort, parameter set, optimizer, batch size, or number of
macrosteps. Do not convert its recall gain into a post-hoc threshold/router.
Stage B, validation, test, probe, full train, and a current-best command
revision are denied.

The exact beta-1 term was only a three-percent direction perturbation and did
not change clean decisions beyond ERM. A continuation must be a newly sourced,
precommitted objective that directly handles conflicting condition risks while
preserving the average objective and class-1 TP/support. It must retain raw,
aggregate-margin, ERM, and V-REx evidence and cannot be a nearby V-REx weight
or environment sweep.
