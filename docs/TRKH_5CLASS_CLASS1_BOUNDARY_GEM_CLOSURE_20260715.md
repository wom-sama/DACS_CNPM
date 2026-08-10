# TRKH 5-Class Class-1 Boundary-Stratified GEM Closure

Date: 2026-07-15
Decision: reject the locked four-constraint GEM macrostep; Stage B denied

## Scope and provenance

- The immutable protocol is
  `docs/TRKH_5CLASS_CLASS1_BOUNDARY_GEM_READINESS_PROTOCOL_20260715.md`,
  SHA-256
  `ed44d3f397cb6e2127edf33d211c50b73f413e594a3ccce1c9c33ca8448962c9`.
- Primary sources were the NeurIPS-2017 GEM paper and official repository
  commit `34c6b8e9a0607db7567301c48b727430d20bee7e`.
- Preflight verified the official source/license, paper, keeper, launcher
  arguments, data YAML, CIDT cache, prior A-GEM artifacts, protocol, and clean
  official worktree.
- Only source-disjoint `yolo_f/train` fit/holdout rows `7372/1843` were used.
  Validation and test were never constructed.
- The current task contained the same `186` fit restricted `0/2/4 -> 1`
  false positives. All `432` fit class-1 rows were split by immutable keeper
  margin into four exact strata of `108` rows.

## Exact QP audit

- The hard-gradient dots against the four decision-margin memories were
  `[-114.865389, -81.369627, -47.552714, -23.426820]`; every constraint was
  violated before projection.
- The exact gamma-zero GEM solution selected active set `[0, 1]` with dual
  `[0.606059, 0.261924, 0, 0]`. Projected dots were approximately
  `[-8.84e-9, 4.18e-9, 3.390540, 5.347232]`.
- GEM retained `0.516920` of the hard-gradient norm. Its tensor/flat equation
  error was `2.54e-8`, and the independent SciPy direction relative error was
  `4.04e-15`.
- GEM differed from the aggregate-margin A-GEM direction by relative L2
  `0.389029`. Thus the multi-constraint QP was active and materially distinct.
- Aggregate-control/GEM parameter-step ratios were
  `1.0000001715e-4/1.0000001935e-4`; relative norm mismatch was `2.21e-8`.
  No optimizer, scheduler, augmentation, teacher, binary model, or checkpoint
  artifact was used.

## Clean fold-0 result

| Variant | Macro F1 | Class-1 F1 | Precision | Recall | Predicted class-1 |
|---|---:|---:|---:|---:|---:|
| raw keeper | 0.949323 | 0.849206 | 0.748252 | 0.981651 | 143 |
| prior CE A-GEM | 0.947871 | 0.848739 | 0.782946 | 0.926606 | 129 |
| aggregate-margin A-GEM | 0.951161 | 0.862903 | 0.769784 | 0.981651 | 139 |
| boundary GEM | 0.950749 | 0.852590 | 0.753521 | 0.981651 | 142 |

- GEM versus raw changed `14` decisions, made `8/6` corrections/harms,
  rescued/broke `1/1` class-1 cases, and removed only one restricted false
  positive. Macro/class-1 F1 and precision changed
  `+0.001426/+0.003383/+0.005269`; recall was unchanged.
- The locked clean gates required class-1 F1 `>= +0.005` and at least four
  restricted-FP removals. GEM failed both and was worse than the aggregate
  control by class-1 F1 `-0.010314` and precision `-0.016263`.
- The aggregate-margin control is a useful diagnostic signal: versus raw it
  raised class-1 F1/precision `+0.013697/+0.021532`, kept recall unchanged,
  and removed four restricted false positives. It cannot promote because it
  was a locked comparator and failed the robustness gates below.

## Illumination result

- GEM versus raw under dim changed macro/class-1 F1
  `+0.001191/-0.007413`, precision/recall `-0.052303/+0.036697`, and created
  `15` net restricted false positives.
- Under bright, GEM changed macro/class-1 F1 `+0.002382/-0.000508`, raised
  precision `+0.010022`, lowered recall `-0.018349`, and removed four
  restricted false positives.
- Under low contrast, GEM changed macro/class-1 F1
  `-0.006211/-0.011267`, precision/recall `-0.013038/-0.009174`, and created
  two net restricted false positives.
- Aggregate-margin A-GEM improved dim class-1 F1 `+0.023718` but created ten
  net restricted false positives; bright lost class-1 F1 `-0.006665` and six
  true positives; low contrast lost class-1 F1 `-0.027608` and created eight
  net restricted false positives. Its clean gain is therefore not robust.

## Independent verification and artifacts

- Independent CSV replay found exactly `1843` unique sample indices per
  condition, reproduced all seven comparison objects exactly, and found zero
  probability-argmax mismatches for raw, prior controls, margin A-GEM, or GEM.
- Raw clean predictions had zero mismatch against the immutable CIDT cache.
- Manifest/summary/prediction/gradient SHAs are respectively
  `960bbc3957f7a0eca4ee01379831f024923e5a72510c1dcc94513ea39e0f4183`,
  `1d2bdd20837ee1300698fd144601dffa7a2cd0a3b3a7df17c7f801cb994a3e0c`,
  `5e66813cb0e2cbe238d7eaccfe94202a3d8362550139a75236baffa02c67cfb3`,
  and
  `0e4776794c2be3b1598f14a0e8fd84a99e4423a7ba3d1c769a04c9003edb9278`.
- Five verified nonbinary files total `4,468,351` bytes. No checkpoint, ONNX,
  TensorRT engine, test prediction, or raw-data write exists.

## Closure verification

- Python compilation and focused tests passed `8/8`; full pytest passed
  `1150/1150` in `47.62 s`.
- The PowerShell launcher parsed successfully. Preflight reproduced every
  source hash, cohort, stratum, and prior-prediction check without creating an
  output directory.
- `git diff --check` passed.
- Read-only retention audit
  `runs/artifact_retention_audit_class1_boundary_gem_closure_20260715`
  passed over `689` run directories with `blockers=[]`; summary SHA-256 is
  `e9853a74b4a5f73a1065d3a072670fb242b97ea88bbd31ac895f3a3fba1f48ca`.
- Protected keeper, scratch complement, command packet, and command-history
  hashes remained exact. Command tracking remains three revisions and two
  updates after the initial revision.

## Stop rule and next distinct screen

Do not sweep the GEM strata, loss, gamma, ridge, QP solver, normalized step,
fold, seed, cohort, parameter set, batch size, conditions, or number of
macrosteps. Stage B, validation, test, probe, full train, and a current-best
command revision are denied.

The aggregate-margin control provides only a new hypothesis: a separately
sourced robust objective may need to preserve its clean precision gain while
optimizing the worst predefined illumination condition. Any continuation must
be a fresh, precommitted method with immutable clean/dim/bright/low-contrast
groups, raw/prior/GEM comparators, and class-1 TP/support gates. It must not be
a post-hoc GEM constraint or hyperparameter sweep.
