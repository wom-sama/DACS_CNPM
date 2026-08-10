# TRKH 5-Class Deferred Reweight A0 Result - 2026-07-20

## Decision

The locked no-training A0 passes all seven conjunctive gates and authorizes
exactly one five-epoch natural-only scratch smoke. It does not authorize
deferred-weight integration, a probe, test access, a full train, or current-best
command promotion.

## Evidence

- Strict batch-32 exposure is exactly `[1844, 1843, 1843, 1843, 1843]` over
  `9216` samples. Class 1 is exposed `3.406654x` relative to its 541 natural
  rows; no nonfocus class exceeds `0.959896x`.
- The scratch checkpoint predicts class 1 on `792` clean train rows for true
  support `541`: support ratio `1.463956`, P/R/F1
  `0.643939/0.942699/0.765191`, and `275` restricted `0/2/4 -> 1` FP.
- Scratch predicted-support ratios are
  `[1.33945, 1.33913, 1.54808, 1.57282, 1.53636]` over the five fixed source
  partitions. Recall-minus-precision gaps are
  `[0.23947, 0.22902, 0.33021, 0.35359, 0.33642]`; every partition passes.
- Fixed post-hoc prior controls are unsafe. The square-root correction lowers
  scratch class-1 recall to `0.092421`; the full correction removes every
  class-1 prediction. Neither is eligible as a calibrator or router.
- With natural sampling and effective-number `beta=0.999`, the prospective
  late class-1 weighted mass share is `0.116601`, between natural `0.058709`
  and strict `0.199978`. This is reweighting without class-1 oversampling.

## Integrity

- Summary/manifest/replay/fold/control SHAs are
  `cfa0d366...9176a5`/`53c8212a...d805c0`/
  `a00cf2f7...c0ce0`/`bf92c55b...63652a`/`aa81bf55...a197e`.
- Independent second-process replay is exact: persisted and recomputed
  canonical SHA are both `ab073792...25a51f`.
- Dataset pixels, checkpoints, model forwards, readout fitting, training,
  validation predictions, and test data were not used.
- Compile/pyflakes, PowerShell parse, focused `4/4`, and full pytest
  `1663/1663` pass.
- Read-only retention covers `800` run directories and all `50` valid
  object-schema compaction manifests. All compacted originals remain absent,
  `deleted_anything=false`, and `blockers=[]` at summary SHA
  `950282d0...e65b19`.

The first retention invocation included all 256 `cleanup_manifest_*.json`
files and correctly failed before output on an array-schema historical file.
The successful invocation filtered to the 50 object-schema files containing
`compacted_names`. Future retention reruns must retain that filter.

## Next Gate

Run the Stage-B natural-only scratch smoke exactly as locked: V8 random init,
seed 42, natural shuffled train sampling, class weights disabled, five epochs,
scheduler horizon 30, batch 32, train/eval workers 4/2, test skipped, full
validation, architecture trace, and complete robustness/XAI. Deferred weighting
may be implemented only if every Stage-B threshold passes.
