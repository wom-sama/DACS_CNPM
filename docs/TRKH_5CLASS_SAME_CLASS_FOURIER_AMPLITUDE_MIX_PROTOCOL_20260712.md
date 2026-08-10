# TRKH Same-Class Fourier Amplitude Mix Protocol - 2026-07-12

## Decision Status

- Stage: locked no-test readiness audit.
- Current-best checkpoint and full-train command: unchanged.
- Trainer integration, smoke training, probe training, and final test: forbidden until
  every readiness gate below passes.
- Raw `class_f` and `yolo_f` files remain read-only.

## Why This Is Not A Repeat

The repository already contains an FFT filter over the token embedding dimension and
has rejected high-frequency experts, texture descriptors, MixStyle, spatial mixing,
and several color/illumination normalizers. None of those interventions reconstructs
an RGB image from the target phase and a source-distinct peer amplitude spectrum.

The new hypothesis is narrower than generic Fourier augmentation:

1. The target image keeps its complete Fourier phase and target geometry.
2. The peer must have the same hard class and a different original YOLO source stem.
3. Validation targets can use peers from `yolo_f/train` only. No validation peer is
   used and no parameter is fit on validation.
4. A focus-boundary cross-class mix is emitted only as a destructive control. It can
   never become a training view.

This restriction matters because mango color and surface statistics are class cues,
not pure nuisance style. Same-class pairing is intended to vary illumination and
capture style without crossing the declared class-color distribution.

## Primary Sources And Locked Translation

- FACT, CVPR 2021: https://openaccess.thecvf.com/content/CVPR2021/html/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.html
- Official FACT implementation: https://github.com/MediaBrain-SJTU/FACT
- FDA, CVPR 2020: https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html
- Official FDA implementation: https://github.com/YanchaoYang/FDA
- Mango color/ripening review: https://www.mdpi.com/2076-3417/11/9/4249
- Mango maturity/color study: https://www.sciencedirect.com/science/article/abs/pii/S030442382200824X

FACT linearly interpolates two amplitude spectra while retaining each image phase.
Its released PACS recipe samples `lambda` from `[0, 1]` and mixes the full spectrum.
FDA instead replaces a centered low-frequency window and explicitly performs the
operation before normalization. The readiness audit uses the least ambiguous FACT
midpoint, not a sweep:

- RGB domain: exact deterministic keeper eval image, unnormalized to `[0, 1]`.
- Formula: `A_mix = 0.5 * A_target + 0.5 * A_peer`.
- Phase: `phase_target` only.
- Frequency area: full spectrum (`ratio=1.0`), matching official FACT default.
- Output: real inverse FFT, clipped to `[0, 1]`, then keeper normalization.
- Target metadata: original target `bbox`, transformed `crop_bbox`, and valid mask.
- Training randomness: not designed or enabled at this stage.

## Pairing Contract

The peer plan is deterministic from seed `20260712` and written to CSV.

- Train target -> train peer, same label, different `Path.stem.casefold()` source.
- Validation target -> train peer, same label, different source.
- Cross-class control mapping: `0->1`, `1->0`, `2->1`, `3->1`, `4->1`.
- Cross-class controls also require a different source.
- Duplicate sample indices, missing class support, wrong peer labels, same-source
  pairs, or incomplete path coverage fail closed.

## Full Readiness Evidence

The audit must cover all `9,215` train objects and all `2,606` validation objects
with the exact keeper SHA-256
`1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
It produces:

- exact clean FP32 keeper predictions;
- same-class Fourier-view predictions;
- focus-boundary cross-class control predictions;
- per-sample peer identity, probability deltas, transition outcome, RGB/object/
  background perturbation statistics;
- five source-grouped train fold summaries;
- validation summary using train peers only;
- identity reconstruction and imaginary-residual checks;
- a representative contact sheet;
- a hash manifest with no checkpoint, model binary, or test payload.

The direct class-1 cohorts are:

- clean class-1 false negatives: desired `delta p1 > 0` and rescue to class 1;
- clean class-1 true positives: must remain class 1;
- clean `0/2/4 -> 1` false positives: desired `delta p1 < 0` and removal;
- other clean `0/2/4` predictions: must not create a new class-1 false positive.

## Smoke Permission Gate

Every check is conjunctive. Capped preflight can verify implementation only and can
never grant smoke permission.

1. Full support, exact checkpoint hash, no train/validation source overlap, no test.
2. Same-class label and source-distinct pairing coverage is `100%`; validation peers
   are train-only; cross-control labels match the locked map.
3. Self-pair identity reconstruction max RGB error is at most `1e-5`; inverse FFT
   imaginary residual is at most `1e-5`; all outputs are finite.
4. Clean validation macro/class-1 F1 reproduces `0.882925/0.678261` within `0.001`.
5. Same-class RGB mean absolute perturbation is active (`>=1/255`) but bounded
   (`<=0.25`); clipping fraction is at most `0.05`.
6. Validation clean-correct retention and class-1 TP retention are both at least
   `0.95`, and no worse than the corresponding cross-class control retention.
7. On validation, class-1 FN mean `delta p1 > 0`, negative-FP mean `delta p1 < 0`,
   and their `delta p1` separation AUROC is at least `0.60`.
8. Validation FN rescues are at least TP breaks; FP removals are at least FP
   creations; transformed class-1 F1 does not fall below clean class-1 F1.
9. At least four of five source-grouped train folds reproduce the two signed
   probability directions with AUROC at least `0.55`.

Passing this gate permits one low-weight, default-off smoke implementation. It does
not permit a full train or current-best command update. A failed gate closes this
exact `same-class/full-spectrum/lambda=0.5` policy; nearby lambda or frequency-window
sweeps are not allowed without a new independent mechanism and protocol.
