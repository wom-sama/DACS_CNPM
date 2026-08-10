# TRKH 5-Class Axial Color Topology A0 Closure - 2026-07-20

## Decision

Reject this exact low-frequency axial color-topology route before visual review,
image-model integration, validation/test access, or full training. The descriptor
acts as a broad class-1 suppressor: it removes many false positives and raises
precision, but suppresses substantially more true class-1 predictions and lowers
macro/class-1 F1 in every clean source fold. The current-best checkpoint and
full-train commands remain unchanged.

## Locked Basis

- Prospective protocol:
  `docs/TRKH_5CLASS_AXIAL_COLOR_TOPOLOGY_A0_PROTOCOL_20260720.md`, SHA-256
  `038faf959e2110a507b1aa4a41e21f1d6f70f9b02ffa769a84ecc1f4074758d1`.
- The protocol was committed and pushed before implementation at `34d43ff`;
  the isolated auditor, tests, and launcher were committed and pushed before
  formal measurement at `6f4e426`.
- The biological motivation is the spatial/temporal mango-ripening evidence in
  Nordey et al. (2014) and Ngamchuachit et al. (2016). It motivates a spatial
  color test but does not establish external-RGB label identifiability.
- Frozen keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Use only immutable `yolo_f/train=9215`, exact five source folds, and exact
  CIDT clean/dim/bright/low-contrast keeper probabilities. Validation and test
  were not constructed or read.

## Implemented Audit

Every transformed object crop is ROI-aligned to `128x128`, eroded by `0.05`,
restricted to a fixed radius-`0.92` ellipse, and filtered by a conservative
achromatic-highlight proxy. Five Lab/chromaticity/saturation maps are smoothed,
robust-standardized, and summarized over four unsigned axes and five
equal-support bins. Reflection-invariant profile values are sorted across axes,
yielding a `140D` axial descriptor.

The five-fold OOF comparison uses matched `165D` no-bias offset readouts:

- control: `25D` diffuse global color plus `140` zeros;
- candidate: the same global color plus axial topology;
- placebo: the same global color plus partition-local source-deranged topology.

All structural checks pass. All `15` readouts converge in `5-7` L-BFGS-B
iterations, descriptor values are finite, minimum bin support is `1308`, and
axial effective rank is `21.644701`. Focused descriptor tests verify exact
reflection and quarter-turn invariance, flat-color handling, and gradient
sensitivity.

## Formal Result

| Condition | Macro F1 delta | Class-1 P delta | Class-1 R delta | Class-1 F1 delta | Restricted FP reduction | Direction AUROC |
|---|---:|---:|---:|---:|---:|---:|
| clean | -0.022718 | +0.036436 | -0.109057 | -0.038105 | 38 | 0.354470 |
| dim | -0.006006 | +0.025621 | -0.086876 | -0.055139 | 56 | 0.543339 |
| bright | -0.029696 | +0.011765 | -0.060998 | -0.044545 | 21 | 0.499529 |
| low contrast | -0.020908 | +0.043114 | -0.059150 | -0.022810 | 54 | 0.546390 |

On clean OOF rows, control-to-candidate class-1 precision/recall/F1 moves
`0.788927/0.842884/0.815013 -> 0.825364/0.733826/0.776908`. Restricted
false positives improve `121 -> 83`, but the candidate makes only `96`
corrections versus `270` harms, rescues `2` class-1 false negatives, and breaks
`61` class-1 true positives. Candidate-minus-control class-1 direction AUROC is
`0.354470`, showing that the probability movement ranks the desired FN-versus-
FP direction backwards.

The candidate also loses to the source-deranged placebo by `-0.022696` macro F1
and `-0.034920` class-1 F1 on clean data. Every clean fold has negative macro
and class-1 F1 deltas, so this is not one bad split. Fourteen prospective
metric/robustness checks fail. Conditional contact sheets and XAI were therefore
correctly not generated; their absence is a protocol outcome, not missing work.

Do not sweep axes, channels, masks, sigma, bins, erosion, readout C, folds,
thresholds, class weights, or convert this descriptor into another post-hoc
class-1 suppressor on the current keeper. The new signal has precision pressure
but no TP selectivity, matching previously closed one-sided verifier failures.

## Runtime And Evidence

- Formal directory: `runs/audit_axial_color_topology_a0_20260720`.
- Summary SHA-256:
  `247070296b2e77f9dfe8fcfb2a0ef15db8abad747371ad7b75853eaf7d2cd224`.
- Artifact-manifest SHA-256:
  `4ad122d344f04aed2354582114c1ed8588906cc76d6058ab437acc87a2e811c4`.
- Six payloads total `40,576,913` bytes; aggregate payload-manifest SHA-256 is
  `314c95df766b775fd9659934aacb57364f87e46e6177606423d0ba745d69862f`.
  The manifest contains no checkpoint, model binary, or test payload.
- Independent replay reconstructs every metric and gate with maximum numeric
  difference `0.0`.
- The formal launcher elapsed `1379.7 s`. Each condition took `329-344 s`;
  peak CUDA allocation was `306.00 MiB`. The loader requested four workers but
  the shared Windows-safe helper correctly reported effective workers `0`;
  eight explicit descriptor threads performed the CPU-bound color analysis.

## Verification And Retention

- Focused axial tests: `10/10`; related integration tests: `48/48`.
- Full repository pytest: `1510/1510`.
- Compile, pyflakes, PowerShell AST, no-output preflight, provenance hashes,
  source isolation, and exact replay: pass.
- Read-only retention scans `772` run directories, leaves all `34` currently
  manifest-derived compacted originals absent, deletes nothing, reports
  `blockers=[]`, and leaves `102.553 GiB` free. Retention-summary SHA-256 is
  `41933ca4e0a7e2a26efe36b393c0424e0bcfc618c035e57ef04979aa808f8414`.
- Raw dataset modified: no. Validation/test used: no. Production model/trainer
  changed: no. Full train launched: no. Current-best command revision: unchanged.
