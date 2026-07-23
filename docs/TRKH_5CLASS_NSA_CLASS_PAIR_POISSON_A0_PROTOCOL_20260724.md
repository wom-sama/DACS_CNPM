# TRKH 5-Class NSA-Inspired Class-Pair Poisson Mismatch A0 Protocol - 2026-07-24

## Status And Scope

Prospectively locked before generator implementation, image preview, adapter
training, or candidate metric access. This protocol authorizes one train-only,
source-held A0 and a preceding geometry-only preview. It does not authorize
validation/test access, a production model or trainer edit, a checkpoint
promotion, smoke/probe/full train, or current-best command/history changes.

Raw `class_f` and `yolo_f` data remain immutable. Synthetic images are
ephemeral train-derived tensors and auditable run artifacts only. A synthetic
image never receives a five-class classification label and is never written
back into either dataset.

## Prospective Geometry Erratum - 2026-07-24

The first 24-row geometry-only preview was executed after the original
protocol lock and before keeper loading, model forward, adapter construction,
or candidate metric access. All row, source, hash, RNG, train-only, and
changed-pixel checks passed, but the aggregate implementation flag was
incorrectly false because three desired negative states
(`validation_split_used=false`, `test_split_used=false`, and
`raw_dataset_modified=false`) were passed directly into `all()`. More
importantly, manual review rejected row 20 (`sample_index=3633`) because its
otherwise valid target rectangle crossed fingers occluding the mango.

Preserve the failed preview at summary/contact/row SHAs
`59a0faecf42e66f738e5897fa9038615e9295f9662e593974785c65271d1a426`,
`2e33244840da8bb629b855a6ad0cdc199cd4912573eda686b5b36a261daa21fa`,
and
`13240025e330f569ff2d05c8fd827e65f10726b08958659c0db17f7277b4e941`.
It generated no model score and authorizes no scientific conclusion.

Before a second geometry preview, apply exactly one additional deterministic
occluder exclusion after the original support intersection:

1. convert evaluation RGB uint8 to OpenCV HSV and YCrCb without changing RGB;
2. mark a pixel chromatically hand-like only when `Y>=70`,
   `138<=Cr<=175`, `105<=Cb<=135`, `S<=120`, and
   (`H<=18` or `H>=160`) on OpenCV's integer channel scales;
3. find 8-connected components in that mask;
4. select only components with at least 20 pixels inside original support,
   at least 50 pixels overall, and at least `0.35` of their pixels outside
   original support;
5. dilate selected components once with a `7x7` all-one kernel and subtract
   only their intersection with original support;
6. require at least 64 support pixels after exclusion; never fall back to the
   unfiltered support for an invalid target or donor.

The outside-support fraction prevents an isolated brown/lesion component
inside the mango from being removed, while the fixed dilation gives a
three-pixel placement margin around an entering occluder. Persist original,
selected-component, dilated-exclusion, and final-support hashes and areas for
every preview/formal row.

Correct the aggregate checks only by renaming the three negative states to
the positive predicates `validation_split_unused`,
`test_split_unused`, and `raw_dataset_unchanged`; their underlying observed
values remain false/false/false. No source plan, class pair, sampler
distribution, donor search, Poisson equation, adapter, optimizer, score,
threshold, or gate changes. A second fixed 24-row preview and manual review
must pass before the keeper may be loaded.

## Research Question

Can a small class-query-conditioned local head learn a surface compatibility
signal from source-disjoint Poisson composites that separates clean class-1
true positives from the keeper's restricted `0/2/4 -> 1` false positives,
beyond:

- keeper confidence plus bbox/shape/context metadata;
- the same head trained on clean class-query supervision only;
- a capacity-matched head without a class query; and
- a capacity-matched head whose query is independently permuted?

The causal hypothesis is not that a Poisson artefact is a mango class. A
same-class Poisson composite is explicitly a zero-mask hard negative, while a
cross-class composite has a local mismatch mask. The clean image is also
paired with its true query and all four false queries. A useful candidate must
therefore learn class-conditioned local compatibility and must gain over the
clean-only control; generic seam detection cannot pass.

## Primary Source, License, And Adaptation Boundary

- Accepted primary paper: Hannah M. Schlueter, Jeremy Tan, Benjamin Hou,
  and Bernhard Kainz, *Natural Synthetic Anomalies for Self-Supervised
  Anomaly Detection and Localization*, ECCV 2022.
- ECCV record:
  `https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7519_ECCV_2022_paper.php`;
  primary PDF:
  `https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910459.pdf`;
  supplement:
  `https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910459-supp.pdf`.
- Paper:
  `D:/DataAI/Tools/source_cache/papers/NSA_ECCV2022.pdf`, SHA-256
  `fdd14c4ee81f353d6f279f71b9516825fd84e4f603b050dc869db03dba474e44`.
- Supplement:
  `D:/DataAI/Tools/source_cache/papers/NSA_ECCV2022_supp.pdf`, SHA-256
  `5804ae0202b74149441baee1151d52da83fc1864a454e17fedcdfaaacf8d32da`.
- Authors' repository:
  `https://github.com/hmsch/natural-synthetic-anomalies`, locally pinned at
  commit `919591685307ce030fe27cb77687509dc277189c`, tree
  `6eebd6e9d7ce9ff77d7d0aa9a2640da7f304cd06`.
- MIT `LICENSE` SHA-256:
  `7f86283e43b5c69fe93ff2c73ddb2bd89f1dbb718995ece3a7b5fa41b2435aaa`.
- Official `self_sup_data/self_sup_tasks.py` SHA-256:
  `14f751362fa2f05935599e69b1334cf3568a2d01b2e8e7317eeb02185f3e4552`.

NSA contributes the different-image source patch, truncated-Gamma patch
scale, random shift/resize, source-gradient Poisson blend, and
intensity-derived localization target. This A0 is an independent,
TRKH-specific adaptation. It does not reproduce NSA's ResNet-18
encoder-decoder, normal-only anomaly assumptions, MVTec masks, 320/560-epoch
schedule, or deployment score. The original schedule is incompatible with the
TRKH ceiling; A0 is fixed to two adapter epochs per source fold.

DRAEM is rejected before code because its official recipe requires external
DTD textures, a roughly 97M-parameter reconstruction/segmentation stack, and
700 epochs. SimpleNet and recent diffusion anomaly methods are also rejected
because they require pretrained/external representations or overlap the
closed hyperspherical/diffusion routes.

## No-Repeat Boundary

Object-interior CutPaste response, Cutout, same/cross-class classification
paste, SnapMix, local zoom, RGB masked reconstruction, deterministic surface
statistics, LoG/Gabor/LBP/Deep-TEN, local MIL/WILDCAT, and frozen
perturbation-sensitivity readouts are closed. This route is distinct only
because it jointly requires:

1. a patch from a different train image;
2. source-gradient `cv2.NORMAL_CLONE`;
3. same-class Poisson hard negatives matched to cross-class geometry;
4. local intensity supervision for cross-class mismatch;
5. clean true/false class queries; and
6. a learned query-conditioned stem adapter that must beat clean-only and
   query-free controls.

Failure closes this exact combination and its nearby patch-size, blend,
adapter-width, top-k, loss-weight, learning-rate, and epoch variants. It does
not reopen any previously closed route.

## Locked Inputs

- Pre-protocol TRKH HEAD/upstream:
  `c4ff352627a461a8ea344cf47a3e76834f26b374`.
- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper `launcher_args.json` and `resolved_config.json`, SHA-256
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`
  and
  `e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674`.
- `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- CIDT train-only summary/predictions, SHA-256
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`
  and
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Current-best command/history SHA-256
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`
  and
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
- Runtime lock: OpenCV `4.13.0`, Torch `2.6.0+cu124`, torchvision
  `0.21.0+cu124`, scikit-learn `1.6.1`, NumPy `1.26.4`.

The auditor must verify these locks, the clean official NSA repository, all
seven protected untracked payload hashes, clean tracked TRKH state, and
HEAD/upstream equality before a formal metric run.

## Leakage Boundary And Source Folds

Only the 9,215 `yolo_f/images/train` object rows and immutable CIDT folds
`0..4` may be opened. For held fold `h`:

- adapter updates use rows whose fold is not `h`;
- every target, same-class donor, and cross-class donor is from those fit
  folds;
- donor and target `source_stem` must differ;
- threshold/readout fitting uses only fit-fold cohort rows;
- the held fold is used once for clean scoring and locked synthetic
  diagnostics;
- no validation/test path, pixel, label, prompt, embedding, statistic,
  checkpoint, or metric may be constructed.

Each fit panel is class-balanced by deterministic majority downsampling, not
minority repetition. Its per-class count equals the smallest available fit
class count, so every selected class-1 row appears once per adapter epoch and
no class-1 row is globally oversampled. Selection order is the SHA-256 rank of
`20260724|held_fold|sample_index|source_stem`.

All donor plans are built before image access from fit labels and source IDs.
For each target and epoch, select one same-class donor and two cross-class
donors. Across the two epochs, the four cross-class donor labels occur exactly
once in a deterministic cyclic order. A donor search may advance through at
most 64 deterministically ranked rows of the required class to find valid
geometry; exhaustion is fatal. Persist every target/donor ID, class, fold,
source stem, seed, and geometry hash.

## Locked Poisson Geometry

Start from the keeper's ordinary 256x256 evaluation RGB after undoing only
ImageNet normalization. Target and donor supports use the already reviewed
CutPaste support:

- deterministic surface-detail foreground mask, `margin=0.08`;
- intersection with `crop_bbox` and `image_mask`;
- centered bbox ellipse, radius factor `0.92`;
- two-pixel binary erosion after all intersections.

No background, hand, basket, padding, or invalid crop pixel may be a source or
destination. Source and destination rectangles must be fully contained in
their own support masks.

Use a local NumPy `PCG64` generator seeded from
`20260724 + 1000003*sample_index + 1009*epoch + 37*role_id`. Python, NumPy
global, and Torch global RNG state may not be consumed.

For each final half-height and half-width independently, sample the
NSA/Hazelnut form

`clip(0.03 + Gamma(shape=2, scale=0.05), 0.03, 0.35)`

relative to the target support bounding-box height/width. Full dimensions are
twice the rounded half-widths. Sample donor resize scale from
`clip(Normal(1, 0.5), 0.7, 1.3)`, crop the inverse-sized donor rectangle, and
resize bilinearly to the final dimensions. Resample geometry at most 32 times.

The same-class and first cross-class views share the final target rectangle
and final patch dimensions; donor source locations are independent. The second
cross-class view uses the next deterministic geometry draw. Use one patch per
view and OpenCV `NORMAL_CLONE` with a binary rectangle mask whose one-pixel
border is zero. RGB channel order is preserved end to end.

The cross-class target is the official Hazelnut logistic-intensity form:

- changed mask: mean absolute RGB uint8 delta `>1`, median-filtered with a
  `5x5` kernel;
- intensity: median-filtered mean absolute RGB delta;
- soft target inside the changed mask:
  `1 / (1 + exp(-(1/12)*(intensity-24)))`;
- zero outside the placed rectangle and target support.

The same-class target is identically zero despite receiving the matched
Poisson transform. Changed pixels outside the placed rectangle or target
support, fewer than 50 positive clone-mask pixels, an empty intensity target,
or any geometry failure is fatal.

## Geometry-Only Preview

Before loading the keeper, render exactly 24 fixed rows: four target classes
`0/1/2/4`, same-class plus all applicable class-1 boundary directions, with
clean target, donor, support masks, same-class composite, cross-class
composite, changed mask, and logistic target.

The preview may correct only an implementation-level geometry defect through
a prospective erratum committed before any keeper forward or adapter metric.
It may not change the scientific class pairs, distributions, epochs, controls,
score, or gates. Manual review must confirm visible mango-surface content,
plausible blending, no non-object support, and no systematic boundary halo.

## Frozen Stem And Adapter

The keeper is evaluation-only and bit-identical before/after A0. Run only its
existing CNN stem under `torch.no_grad`; the locked output is
`[B,256,32,32]`. Production transformer/head logits are not recomputed for
selection; immutable CIDT probabilities provide the keeper baseline.

The query-conditioned adapter is:

1. `GroupNorm(16,256)`;
2. `Conv2d(256,64,kernel_size=1)`, `GroupNorm(8,64)`, `GELU`;
3. a learned 64-D embedding for each of five class queries;
4. concatenate `f`, broadcast `q`, `f*q`, and `abs(f-q)`;
5. `Conv2d(256,64,kernel_size=3,padding=1)`, `GroupNorm(8,64)`, `GELU`;
6. `Conv2d(64,1,kernel_size=1)`.

There is no dropout, batch normalization, pretrained external feature,
transformer update, reconstruction decoder, or five-class loss. All adapter
parameters use the same seeded initialization.

Train four independent, capacity-matched roles:

- `poisson_query_candidate`: clean and Poisson supervision with correct query;
- `clean_query_control`: clean true/all-false queries only;
- `no_query_control`: same/cross Poisson supervision with one learned constant
  embedding;
- `permuted_query_control`: candidate inputs but query IDs independently
  deranged within each batch before labels are exposed.

For a clean image of class `y`, query `y` has a zero target and each query
`q != y` has the downsampled support target of one. Candidate Poisson targets
use query `y`; same-class is zero and both cross-class views use their
intensity masks. Pixel losses are means over support only:

`L_clean = 0.5*L_true + 0.5*mean(L_false_queries)`.

`L_candidate = mean(L_clean, L_same, mean(L_cross_views))`.

The clean-only and no-query controls use `L_clean` and
`mean(L_same, mean(L_cross_views))` respectively. The permuted control uses
the candidate equation. Use `BCEWithLogitsLoss`, AdamW
`lr=3e-4, weight_decay=1e-4`, batch size 16 target rows, requested workers 4,
two epochs, no scheduler, no gradient accumulation, gradient norm cap `5.0`,
and full-FP32 adapter updates. Each role receives exactly the same number of
optimizer steps. Non-finite loss/gradient is fatal.

## Held Scoring And Matched Readouts

For each adapter and clean image, query class 1 and compute mismatch
probabilities inside the downsampled support. Persist:

- support mean;
- top-quintile mean, using exactly `ceil(0.20*n_valid)` pixels;
- support 90th percentile with linear interpolation.

The fixed compatibility scalar is
`1 - 0.5*support_mean - 0.5*top_quintile_mean`; higher means more compatible
with class 1.

The locked 750-row CIDT cohort is unchanged: 528 clean class-1 true positives
and 222 restricted `0/2/4 -> 1` false positives, ordered SHA-256
`55913ec45265b156a611dc96c779e46b08f28582737af8269105afe6f65694d7`.

The matched base vector contains five clipped keeper log probabilities,
class-1 versus max-`0/2/4` logit margin, bbox `cx/cy/w/h`, log bbox area, log
bbox aspect ratio, valid-image fraction, and surface-support fraction. Each
role appends its three map summaries. `base_context` uses no map summaries.

For every held fold and role, fit only on the other four folds:

- `StandardScaler`;
- L2 `LogisticRegression`, `C=0.1`, `class_weight=balanced`,
  `solver=lbfgs`, `max_iter=4000`, `tol=1e-9`, seed `20260724`;
- positive label is a true class-1 keeper prediction;
- choose the lowest fit score retaining at least 98% fit true positives;
- apply once to the held fold.

Rejected rows switch only to the keeper's highest-probability non-class-1
class for diagnostic metrics. No threshold or adapter may be exported.

## Fixed Maps, Replay, And Resource Audit

Render a fixed contact sheet covering class-1 TP and target-`0/2/4` false
positives from every fold. Show clean RGB/support, same/cross composites and
targets, candidate/query controls, and candidate maps for all five queries.
Manual review cannot rescue an automatic failure.

Persist fit panels, donor plans, geometry, masks, losses, gradients, adapter
states, readout states, row scores/actions, fold/class summaries, timing,
peak RSS/VRAM, process snapshots, and all hashes. A fresh process must reload
saved adapter/readout states and reproduce geometry hashes, map summaries
within `1e-6`, exact decisions, metrics within `1e-10`, and the artifact
manifest. Retraining is not part of replay.

Requested/effective worker count, images/s, peak CUDA allocation, and
competing compute PIDs are mandatory. Peak CUDA allocation must be at most
`4.5 GiB`; host virtual-memory fraction must stay below `0.82`. The formal run
is invalid if an unknown Python/TensorRT compute process overlaps timing.

## Conjunctive Gates

Structural gates:

1. all locked hashes/revisions, protected payloads, train-only paths,
   clean worktrees, and HEAD/upstream equality pass;
2. exact 9,215-row declaration, five folds, 750-row cohort/order, zero
   fit/held source overlap, zero donor/target source equality, and zero
   validation/test construction;
3. every fit panel is majority-downsampled without duplicated sample index,
   every donor is fit-fold-only, and all four cross labels occur once per
   target over two epochs;
4. geometry, support, changed-mask, intensity-mask, RNG isolation, manual
   preview, and exact geometry replay pass;
5. keeper state is bit-identical, stem shape is exact, all adapter/readout
   values are finite, all readouts converge once, and replay tolerances pass;
6. requested/effective workers, process isolation, CUDA/RAM limits,
   compile, pyflakes, focused/full tests, launcher parse, and manifest pass.

Mechanism gates for `poisson_query_candidate`:

1. OOF TP-versus-restricted-FP AUROC is at least `0.855`;
2. AUROC gains are at least `+0.020` over `base_context`, `+0.010` over
   `clean_query_control`, and `+0.020` over both `no_query_control` and
   `permuted_query_control`;
3. aggregate action retains at least `0.98` of class-1 TP and rejects at least
   `0.12` of restricted FP;
4. it rejects at least five more FP than `clean_query_control` at the locked TP
   budget, with corrections at least three times TP harms;
5. it beats both `base_context` and `clean_query_control` AUROC in at least
   four of five folds, and no fold TP retention is below `0.95`;
6. target-0 and target-2 FP rejection are each at least `0.08`, and at least
   one of the ten target-4 FP is rejected;
7. true-query mismatch is lower than the mean false-query mismatch on at least
   `70%` of held clean rows, five-query compatibility accuracy is at least
   `0.45`, and every query embedding has nonzero gradient/update;
8. candidate maps have nonzero effective spatial rank, at least `75%` of
   cross-class held synthetic map mass lies inside the locked intensity mask,
   and at most `15%` of same-class support pixels have mismatch probability
   at least `0.5`;
9. fixed visual review finds no systematic seam-only, bbox-edge, padding,
   hand, basket, or background explanation for accepted candidate evidence.

No aggregate gain compensates for a failed gate. Precision obtained by broad
class-1 TP suppression is a failure.

## Escalation And Stop Rules

Any geometry-only failure stops before stem loading. Any structural or
mechanism failure rejects the exact A0 and forbids production integration,
validation/test access, smoke/probe/full train, and command promotion.

Only a complete A0 pass authorizes a default-off production auxiliary head
whose clean classification path is bit-identical at zero weight. The first
integration must use ephemeral fit-train composites only and run an identical
at-most `120 batches x 2 epochs` full-validation smoke against a clean-only
control. It must reach validation macro F1 `>=0.884675`, class-1 precision
`>=0.66`, recall `>=0.75`, F1 `>=0.70`, net restricted-FP removal, XAI,
robustness, replay, export, and performance gates before a probe.

Any later autonomous full train remains at most 30 epochs with measured
Windows-safe loader settings and early stopping. Current-best command/history
remain byte-identical until an independent-reload single-model validation
winner exists.
