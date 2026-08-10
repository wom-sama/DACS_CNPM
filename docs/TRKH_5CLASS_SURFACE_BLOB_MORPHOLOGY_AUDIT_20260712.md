# TRKH Surface-Blob Morphology Readiness Audit

Date: 2026-07-12

## Question

Can fixed multiscale surface-blob morphology expose class-1-positive mango
surface evidence that the current CNN/Transformer keeper misses, while an
achromatic highlight mask prevents bright reflections from dominating the
descriptor?

Primary sources:

- Kim et al., CVPR 2013, single-image specular-reflection separation:
  <https://openaccess.thecvf.com/content_cvpr_2013/html/Kim_Specular_Reflection_Separation_2013_CVPR_paper.html>
- Liu et al., CVPR 2015, saturation-preserving specular-reflection separation:
  <https://openaccess.thecvf.com/content_cvpr_2015/html/Liu_Saturation-Preserving_Specular_Reflection_2015_CVPR_paper.html>
- Mango lenticel review: <https://doi.org/10.1016/j.scienta.2011.11.018>
- Mango reflectance/fluorescence maturity study:
  <https://doi.org/10.1016/j.jafr.2022.100477>

The reflection papers also make the main limitation explicit: physical
specular/diffuse separation from one unconstrained RGB image is ill-posed. The
audit therefore does not claim physical separation. Its top-quantile
`V * (1-S)` mask is only an achromatic-highlight exclusion proxy. The lenticel
review also does not justify assuming that simple spot count or density is a
monotonic ripeness label. This is a bounded readiness test, not a handcrafted
biological rule.

## Locked Protocol

- Checkpoint: current no-pretrain keeper, SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Data: immutable `yolo_f`; train `9215`, validation `2606`; test unopened.
- Source groups: train `8064`, validation `2577`; train/validation overlap `0`.
- ROI: exact runtime `crop_bbox`, eroded 15%, then CUDA `roi_align` to
  `128x128` with batch size 64.
- Highlight exclusion: top 5% of `HSV value * (1-saturation)`, used only to
  define the common valid mask for control and morphology and never supplied as
  a numeric predictive feature.
- Matched control: 25 fixed diffuse HSV/Lab/color-moment features on the same
  ROI and validity mask.
- Candidate: control plus 45 fixed local-max morphology features: dark and
  bright lightness blobs plus chroma blobs at LoG scales `1.2/2.4/4.8`,
  background sigma `6`, and a `2.5 MAD` peak threshold.
- Readout: natural-frequency mean CE, `C=0.3`, five source-grouped folds, and a
  zero-init-compatible no-bias offset on fixed keeper log probabilities.
- No feature/scale/threshold/C/fold sweep, class weighting, test access,
  raw-data write, model, checkpoint, or trainable manifest.

This differs from prior deterministic HSV/Lab statistics, LBP/HOG/wavelet,
Deep-TEN, high-frequency experts, and histogram heads: it tests scale-normalized
local extrema and their size/strength distribution after highlight exclusion.
It still had to beat the matched diffuse-color control before any image-model
smoke was allowed.

Implementation and retained evidence:

- `trkh/tools/audit_surface_blob_morphology_readiness.py`
- `trkh/tools/review_surface_blob_morphology_changed_cases.py`
- `tests/test_audit_surface_blob_morphology_readiness.py`
- `tests/test_review_surface_blob_morphology_changed_cases.py`
- `runs/diagnostic_surface_blob_morphology_full_20260712`
- `runs/review_surface_blob_morphology_changed24_20260712`

## Protocol Checks

All `9215/2606` descriptors are finite, all optimizers converge, every fold has
zero fit/hold source overlap, and train/validation source overlap is zero. Mean
highlight exclusion is `0.049855`, as expected from the locked 95th percentile,
and the 45D morphology block has effective rank `15.337912`.

The first prefix was class biased and was used only to catch implementation and
geometry faults. Process-pool extraction was bit-identical to threaded
extraction and materially faster after worker warm-up. The full decision uses
all rows. No prefix metric was promoted as evidence.

## Results

The morphology block contains generic class information, but no standalone
class-1 decision under natural-frequency risk:

| Split | Comparison | Macro-F1 gain | Class-1 F1 gain |
| --- | --- | ---: | ---: |
| train OOF | candidate vs diffuse control | +0.035733 | +0.000000 |
| validation | candidate vs diffuse control | +0.034452 | +0.000000 |

Both standalone branches predict zero validation class-1 rows. Adding the
morphology block to the keeper-relative residual produces only a tiny matched-
control improvement and remains far below the direct keeper:

| Validation method | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 |
| --- | ---: | ---: | ---: | ---: |
| direct keeper, same cache | 0.884073 | 0.608247 | 0.781457 | 0.684058 |
| keeper + diffuse control | 0.846421 | 0.626667 | 0.622517 | 0.624585 |
| keeper + morphology candidate | 0.848371 | 0.647887 | 0.609272 | 0.627986 |

Candidate-minus-control validation gains are only `+0.001950` macro and
`+0.003402` class-1 F1. OOF gains are negative: `-0.002120/-0.005881`.
Only two of five folds improve class-1 F1; fold gains are
`-0.01778/+0.01058/+0.00778/-0.00862/-0.01793`.

The required FN-versus-FP probability direction is inverted and stable in the
wrong direction: OOF/validation AUROC is `0.425281/0.430451`. This is not a
thresholding opportunity.

Transition accounting is equally decisive:

| Comparison | Changed | Corrections/harms | FP removed/created | FN rescued/TP broken |
| --- | ---: | ---: | ---: | ---: |
| OOF candidate vs control | 207 | 91/101 | 25/17 | 5/16 |
| val candidate vs control | 97 | 45/40 | 8/2 | 3/5 |
| val candidate vs keeper | 202 | 52/128 | 33/7 | 1/27 |

The candidate suppresses class 1 more often than it discovers missing positive
evidence. Against the keeper, macro/class-1 F1 falls by
`-0.035703/-0.056072` and class-1 recall falls by `0.172185`.

The direct keeper here is the fixed cached-probability path
`0.884073/0.684058`. It is neither the in-training AMP anchor
`0.884675/0.686047` nor the FP32 independent reload
`0.882925/0.678261`. Candidate and control comparisons use the same cache and
numeric path; the cache difference is disclosed rather than mixed into the
candidate effect.

## Exact Changed-Case Review

The prioritized review covers 24 rows across all transition kinds. It restores
the original CUDA batch-64 windows instead of recomputing selected images alone:
9 windows, 576 context rows, and maximum descriptor difference `0.0` at a
fail-closed tolerance of `1e-7`.

The contact sheet agrees with the metrics. Dark, bright, and chroma extrema are
dense across every class. They respond to lenticels, disease marks, bruises,
fruit edges, broad illumination changes, and context contamination. The
achromatic mask often removes glare but also selects pale fruit regions and
hands. The one rescued class-1 FN has no morphology pattern that separates it
from the four reviewed broken true positives or from false-positive removals.
The descriptor is surface sensitive but not class-1 positive.

## Decision

Reject this route before image-model smoke. Seventeen locked checks fail and
`image_smoke_permission=false`. Do not sweep LoG scales, peak threshold,
highlight quantile, ROI size/erosion, residual `C`, folds, class weights, or
build a morphology router/branch. Do not reinterpret the inverted direction as
a post-hoc class-1 suppressor; that would repeat the already rejected
validation-filter route and further damage recall.

The full artifact has 9 payloads, `9568281` bytes, aggregate SHA-256
`eaf1b5633330b2500c5f14cdc56fe66f7450a46284d136953b6c50c1e5f624de`.
The exact review has 5 payloads, `1851974` bytes, aggregate SHA-256
`9724432d917d82e646685c174180037882800bf0ff1f2d674c32c78e6b9a6cad`.
Neither contains a model, checkpoint, engine, or test payload.

Guarded cleanup removed four superseded preflights and three logs: 43 files,
`4242949` bytes, with observed free-space gain `4300800` bytes. Every deleted
file had a persisted SHA-256 and was reverified immediately before removal.
Retention then passed over 586 directories with `blockers=[]`. Keeper and
current-best command remain unchanged; command SHA-256 is
`3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.

Closure passed py-compile, compileall, focused tests `10/10`, full pytest
`747/747`, payload/cleanup/retention hash verification, `git diff --check`, and
explicit protected-path review. Only the seven dependency-coherent repo files
for this method are eligible for staging.
