# TRKH 5-Class Chromatic Cubical Persistence A0 Closure - 2026-07-20

## Decision

Reject this exact chromatic cubical-persistence route before visual review,
neural-surrogate integration, validation/test access, or image-model training.
The real topology descriptor is less predictive than the keeper-plus-global-
color control and the pixel-permutation placebo. It also loses TP safety under
every lighting shift. The current-best checkpoint and command files remain
unchanged.

## Locked Basis

- Prospective protocol:
  `docs/TRKH_5CLASS_CHROMATIC_CUBICAL_PERSISTENCE_A0_PROTOCOL_20260720.md`,
  SHA-256
  `6e91d163ef5a00dfefddfb9a271bfa07480b278266fbc0539b2f05dc3ce7892b`.
- The protocol was committed and pushed before implementation at `b021959`.
  Auditor/test/launcher commit `f136a5d` and GPU-idle hardening commit
  `5689fc0` were pushed before formal data access.
- Primary architecture evidence is Peng et al., *PHG-Net*, WACV 2024. The
  accepted CVF paper is pinned at SHA
  `41c862cb346b482f15f7a4a731909f41adcece65f974bb78096d23ac39af1b96`.
  The paper repository has no declared license, so no source was copied.
- Cubical persistence uses only the documented MIT-licensed GUDHI `3.11.0`
  wheel pinned at SHA
  `846ac5807d6a72ebc79a29b9405d12a68cfb8dad896ff5aa6a24fde343a8581f`.
- The frozen keeper, CIDT declarations, `yolo_f` YAML, current-best commands,
  and command-history hashes all matched before image access. Only immutable
  train folds `1..4` were used: `432` class-1 rows (`421` TP, `11` FN) plus
  `186` restricted `0/2/4 -> 1` FP. Validation and test were not opened.

## Implemented Audit

Each transformed bbox crop is ROI-aligned to `48x48`, eroded by `0.15`, and
restricted to a fixed ellipse after conservative highlight removal. Average-
rank Lab-a, Lab-b, and HSV-saturation maps feed sublevel and superlevel GUDHI
cubical complexes. Finite H0/H1 intervals above persistence `1/32` produce a
locked `252D` descriptor from Betti curves and interval summaries.

Four matched `282D` nested four-fold OOF roles compare keeper log-probabilities
plus global color with zero topology, real topology, within-row pixel-shuffled
topology, and partition-local source-deranged topology. Thresholds are selected
only from inner clean OOF scores at class-1 retention `>=0.98`; outer models and
thresholds are then applied unchanged to clean, dim, bright, and low-contrast
rows. All `64` readouts converge.

## Formal Result

| Condition | Control AUC | Candidate AUC | Pixel AUC | Source AUC | Candidate TP retention | FN accepted | FP removed | Score rho vs clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.793048 | 0.696709 | 0.712901 | 0.690076 | 0.983373 | 8/11 | 10/186 | 1.000000 |
| dim | 0.613575 | 0.584939 | 0.624701 | 0.554971 | 0.655582 | 3/11 | 83/186 | 0.536597 |
| bright | 0.628211 | 0.585300 | 0.624888 | 0.607614 | 0.923990 | 10/11 | 19/186 | 0.444778 |
| low contrast | 0.680668 | 0.635342 | 0.663742 | 0.639362 | 0.710214 | 1/11 | 92/186 | 0.540228 |

On clean rows, candidate-minus-control AUC is `-0.096339`; candidate-minus-
pixel-placebo is `-0.016192`, and candidate-minus-source-placebo is only
`+0.006633`, below the locked `+0.03` requirement. The candidate removes ten
restricted FP but breaks seven keeper TP and accepts only eight of eleven
keeper FN. Its action path makes nine corrections versus seven harms. Most
rejection is class 0 (`9/135`); only one class-2 and no class-4 FP are removed.

This is not one bad fold. Candidate clean AUC by held fold is
`0.694879/0.697716/0.722367/0.657650`, below control
`0.803285/0.795673/0.814787/0.763636` in every fold. Under dim and low contrast,
the candidate removes `83/92` FP but breaks `145/122` TP; under bright it
removes 19 FP and breaks 32 TP.

The clean topology matrix has effective rank only `3.025/252`. Outer-fold
topology-coefficient cosine is `0.558-0.639`, versus `0.890-0.926` for the base
columns, indicating an unstable low-dimensional fit. The raw descriptor is
still correlated across shifts, but its median relative L2 movement is
`0.199/0.446/0.291` for dim/bright/low contrast; candidate score Spearman then
falls to `0.537/0.445/0.540`. Real topology averages about `199` finite
intervals per clean row, while the histogram-matched pixel placebo averages
about `2722` and still predicts better. Spatial adjacency therefore does not
supply the required class-conditional identity signal.

All structural checks pass, but eight clean, sixteen shifted, and five
aggregate gates fail. Conditional contact sheets/XAI are correctly absent.
Do not sweep channels, polarity, H0/H1 selection, persistence floor, ranks,
resolution, ROI/mask/erosion/fill/sigma, Betti grid, readout C, threshold,
fold, seed, or convert this failed descriptor into a PHG-like neural surrogate
on the current keeper. A surrogate is not justified when the prospectively
required information signal loses to both the matched control and a destroyed-
adjacency placebo.

## Runtime And Evidence

- Formal directory:
  `runs/audit_chromatic_cubical_persistence_a0_20260720`.
- Summary SHA-256:
  `2dd9065c07f191c5110f31b32ff2cf22891efaef5c27598a7426929ceceaa711`.
- Artifact-manifest SHA-256:
  `accc41e57b9490684cca84b40bcda9ca844cac3353068690547173dbbeafedd4`.
- Five payloads total `4,314,680` bytes; aggregate payload-manifest SHA-256 is
  `59a5d90de61d87f476f22a7189cf342974e3bc00944759f2719b90fd30e21db0`.
  The manifest contains no checkpoint, model binary, contact sheet, or test
  payload.
- Independent replay reconstructs all inner thresholds and outer scores from
  saved scaler/coefficient records. Maximum score/threshold/inner-record
  differences are `2.22e-16/5.55e-17/5.55e-17`; metric and gate differences
  are exactly `0.0`.
- Four descriptor passes take `29.61/30.26/30.09/29.85 s`; the launcher takes
  about `148 s`. Peak CUDA allocation is `69 MiB`. Requested workers `4` are
  safely clamped to `0` on Windows; eight descriptor threads perform the
  CPU-bound persistence work.
- Two launcher attempts stopped before data/output because a brittle three-
  sample utilization gate observed one `21%` sample. Commit `5689fc0` adds a
  logged P8-and-`<=10 W` idle fallback without weakening compute-process
  isolation. The successful formal observed `20/19/20%`, P8, and
  `4.36-4.82 W`.

## Verification And Retention

- Focused tests: `13/13`; full repository pytest: `1523/1523`.
- Compile, pyflakes, PowerShell AST, no-output preflight, provenance hashes,
  source isolation, pixel-histogram preservation, synthetic H0/H1 oracle, raw-
  metadata guard, and exact independent replay: pass.
- Read-only retention scans `774` run directories and all `49` current object
  compaction manifests. All `216` listed originals remain absent, nothing is
  deleted, `blockers=[]`, and free space is `102.533 GiB`. Retention-summary
  SHA-256 is
  `6c456de96ae8de70b217bf5de8ab218b7d411d63928d38c9f4a20e9e9705a692`.
- Raw dataset modified: no. Validation/test used: no. Production model/trainer
  changed: no. Full train launched: no. Current-best command revision:
  unchanged.
