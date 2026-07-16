# TRKH 5-Class Cropr Token Selector A0 Closure - 2026-07-17

## Decision

Close the exact task-supervised Cropr token-selector A0 after its sole locked
five-epoch train-only pair. The learned selector is active and numerically
correct, and clean class-1 F1 rises, but the gain is recall-driven. Class-1
precision falls, restricted false positives increase, and candidate harms
outnumber corrections. The auxiliary selector is high-entropy and less aligned
with the object bbox than the native TRKH score, especially at layer 5.

Official validation, test, full training, selector sweeps, and current-best
command promotion remain forbidden.

## Primary-Source Provenance

- Accepted paper: Bergner, Lippert, and Mahendran, "Token Cropr: Faster ViTs
  for Quite a Few Tasks," CVPR 2025.
- Official MIT repository: `https://github.com/benbergner/cropr`.
- Locked commit/tree:
  `fa259e9030f5fddf4721ac75cdd18561524de6f9` /
  `4a83993890026c85cfd81b559b08c341fd86851d`.
- Paper SHA-256:
  `7525a21f05f1ddd982fea1d7d5777b1e87175aeb52d60729eedcb0e448f1a97d`.
- Final prospective protocol SHA-256:
  `42e912a6bdc320d98a9acbb79b6d48c1e372a5df861f279af1c6988b3ff6a847`.

The user-provided research reports were treated only as hypothesis sources.
The accepted paper and official source defined the scorer equation,
implementation, and locked gates.

The causal control and candidate both instantiate and train identical Cropr
queries, residual MLPs, and auxiliary classification heads after blocks 2 and
5. The control retains the native TRKH routing score; the candidate changes
only the routing source to the learned Cropr score. This isolates routing from
extra parameters and auxiliary supervision.

## Preflight And Pair Provenance

Formal preflight passed all `66/66` checks, including official-equation replay,
FP32/BF16 gradients, standard/trace parity, ONNX, runtime/VRAM, exact common
state/RNG, and train-only fold provenance.

- preflight summary SHA-256:
  `12902d6930ec89fd658c2b86e42a30f07ea44643a2fd67a8a1fddb5b7f1e873c`;
- preflight artifact-manifest SHA-256:
  `480f1e99b754918f50ca33b173e945a8fe55f7c1d6122fab2fea5205a5c90ee3`;
- pair manifest SHA-256:
  `09a89cbe4cf4f87e6ce2ac52dd6affd80b732357bc53c1bb75979f7a6180cc1e`.

The sole pair ran from pushed commit `479c9f1589919892cbf3eb1d89764bd67a6f838b`
on the frozen source-disjoint `yolo_f/train` fold: 7,372 fit rows, 1,843
holdout rows, zero source overlap, and exactly 7,372 natural-frequency
occurrences per role per epoch. Official validation and test were not loaded.

Original checkpoint hashes before compaction:

| Role | best.pt SHA-256 | last.pt SHA-256 |
| --- | --- | --- |
| Native control | `47a3c919dfef53cf14a81cf2959e177b0652f55f6895d8a2afc51ebf9f76a996` | `97c0ed6d922926c670c55799c972bb39d1c19e13f295623e44bac4cb66a7225d` |
| Learned candidate | `58736fdc4d54f32cad0f8ee17d5012d0252d4b3b1db8548fda3fe5c4f73f89db` | `6fa48de9b4d1a07ded560c9730e80987a535769c6b69d2d7af6940ca9fa7c98a` |

## Independent Train-Only Audit

Standard BF16 deployment inference, never trace logits, scored all 1,843
holdout rows under four frozen conditions:

| Condition | Macro F1 delta | Class-1 F1 delta | Precision delta | Recall delta | Restricted FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | +0.001804 | +0.021631 | -0.024828 | +0.018349 | 8 -> 10 |
| Dim | +0.004060 | +0.010549 | +0.010774 | +0.009174 | 32 -> 32 |
| Bright | +0.011586 | +0.040781 | +0.019608 | +0.027523 | 6 -> 7 |
| Low contrast | -0.000215 | -0.010023 | +0.024286 | -0.009174 | 13 -> 11 |

Clean macro/class-1 F1 changed from `0.782514/0.253731` to
`0.784319/0.275362`. Class-1 precision fell from `0.680000` to `0.655172`.
The candidate changed 25 decisions with only 8 corrections and 13 harms. It
rescued three class-1 false negatives but broke one class-1 true positive,
created one class-1 false positive, and removed none correctly. Low-contrast
class-1 F1 also crossed the locked `-0.010` failure boundary.

Fourteen automated checks failed, including precision, restricted-FP,
corrections-versus-harms, shifted-condition, selected-object-recall,
auxiliary-separability, and foreground-gain gates. The final pair decision is
`pair_pass=false`.

## Selector Mechanism Result

The clean candidate selector did not learn a useful sparse relevance ordering:

| Layer | Aux macro F1 | Aux class-1 F1 | TP-hard-negative AUROC | Entropy | High-entropy fraction | Native bbox mass | Cropr bbox mass | Selected-object recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.759627 | 0.230769 | 0.537462 | 0.992872 | 0.408573 | 0.434829 | 0.285063 | 0.925113 |
| 5 | 0.761569 | 0.223881 | 0.549185 | 0.996811 | 0.818231 | 0.442484 | 0.317162 | 0.830995 |

Foreground-mass gains relative to the native score were negative at both
layers (`-0.149766/-0.125322`). Layer-5 high-entropy fraction rose above 0.81,
and selected-object recall was lower than the control. The auxiliary head can
fit broad class evidence, but its query score is not a conservative object or
class-boundary selector.

Object perturbation remained genuinely more causal than far background:
candidate absolute class-1 margin change was `0.253820` for object removal and
`0.000000` for the frozen far-background perturbation. This confirms that the
base representation still uses fruit content; it does not rescue the learned
routing mechanism.

## XAI And Audit Corrections

The first complete audit used a trace forward for Grad-CAM and matched standard
predictions on only `57/57` control and `56/57` candidate requests. Its summary,
failure record, and artifact-manifest SHAs are:

- `6404590b0a25049d41238b019bba6da0ef84aea35184f68c588d922f4c04f9cb`;
- `5dc48089553d747656d6e014df71bcbc8ef0fd4c0adea53b5b61301f1a6c7311`;
- `af4ff08eebccd75d13ec5d7764fe4e0db0966ff44fffa5f0ae68b53f68fcbdc0`.

The first correction hooked the standard deployment forward, but evaluated
selected rows in batches of two. BF16 near ties still differed from the locked
batch-32 inference composition. That failed attempt was also preserved at
summary/failure/manifest SHAs:

- `b89bb1b5c28901aeec8f43dc54b7058b19c8d1ccc116cc5e1c15fc51d64f422f`;
- `5007bd44cb8caa6166d6112e9f6ef9344b6964a35537f0cd582e059f14444d41`;
- `9eb152ae4cc582fb5e208c5d72e7263f2bde50a481f065653784bf27d2ed5127`.

Commit `d082f918b4e41bfca153afe77eadcd6618b5e927` preserves the complete locked
batch-32 composition while collecting requested maps. The final audit has
exact standard and trace parity for all 57 requests per role, 50 event rows,
all required representatives, finite maps, and 29 hash-locked pages.

All four unaffected quantitative artifacts replayed byte-for-byte through
both corrections:

- predictions: `4ea78c5faf36c50e6c24833f9dd7ab8cd41dd00600e050f62988c51dc1e692e9`;
- selector rows: `828a463e82f077f48876323f3a1af618094ce3638304057fa8c95596ef977723`;
- perturbations: `7c97d60651870fee94f4df8793066d1dd331754ee95b39e52c47ef8130ae1d8c`;
- events: `3b049bd83ae6ddd3ef4182d7a4df1f0cca9d07b7dfb1c95c5677b3e2489c5aa3`.

All 29 pages were reviewed. Standard stem/block maps were often
foreground-focused, but candidate Cropr maps remained broad and high-entropy,
frequently covered padding/background on wide views, and usually lost layer-5
bbox mass. Mean candidate/control Cropr-L5 foreground mass was
`0.640683/0.719379`; candidate block-5 Grad-CAM also fell
`0.957391 -> 0.934013`. Visual review therefore failed.

Final audit summary, artifact-manifest, and visual-review SHAs are:

- `ee2d4456356256ee10aefb3004aee2e15d4e63823a42611eb272b85671238c5b`;
- `d6b94e14ca0552c6505397d637ea8f00188bb2550145adfdf5d33da9237b9b90`;
- `1f5c2e5749742253696dce17e10cad49b2b57d63d0388d6b7f715334cf969dc3`.

## Stop Boundary

- Do not sweep Cropr query/head count, residual MLP, auxiliary-loss weight,
  routing blend, score temperature, prune layers, keep rates, epochs,
  optimizer, augmentation, fold, seed, bbox loss, threshold, router, or a
  combination with another closed route on this keeper.
- Do not treat the positive clean class-1 F1 delta as progress. It widens the
  class-1 region while violating the agricultural precision priority.
- Any future learned selector needs a distinct accepted primary source and a
  prospective pre-training gate that proves low-entropy object/boundary
  selectivity plus class-1 TP and restricted-FP safety.
- Any BF16 XAI auditor must preserve both the normal deployment forward and the
  exact inference batch composition. Prediction parity alone is insufficient;
  logits must also match the locked inference artifact.
- Official validation/test and current-best commands remain untouched at three
  revisions, two updates, and zero keeper replacements.

## Evidence Retention

Fail-closed compaction copied and hash-verified 332 non-binary run files
(`46,130,253` bytes), excluded exactly the four recorded checkpoints
(`399,569,848` bytes), and deleted only the two rejected pair directories.
Measured free-space gain was `446,504,960` bytes.

- compact evidence:
  `runs/evidence_cropr_token_selector_a0_pair_rejected_20260717`;
- compact payload-manifest SHA-256:
  `75a54745b324f1931c2998c790edeeb1d7d49cebdb29eea89d4f818a4ea6df77`;
- compact summary SHA-256:
  `89307427396f1133b92c578d447984b7f12a40b2306694c1eb8a1e217eca5db6`;
- cleanup-manifest SHA-256:
  `1bcef94e3dbbf581a07d1457f302a9ddab52ac994a6fccbf7ccefad3d39b441a`.

The final audit, both failed correction attempts, all 29 final XAI pages,
preflight, pair manifest, compact evidence, and cleanup manifest remain intact.
Keeper/current-best checkpoint, command, and history hashes remain exactly
`1f49d577...482677`, `36b9aa1a...40faf`, and `39bd2879...98f53`.

Closure verification passed package compileall, PowerShell parse for both Cropr
wrappers, focused Cropr tests `32/32`, and full pytest `1332/1332`. Read-only
retention loaded 47 compaction manifests, covered 734 run directories, found
all 209 compacted originals absent, and passed with `blockers=[]` and
`70.438 GiB` free. Retention summary SHA-256 is
`95128604b2238d3bb26d41742be50f19903e77b520515ff8e107b6e4017382e9`.
