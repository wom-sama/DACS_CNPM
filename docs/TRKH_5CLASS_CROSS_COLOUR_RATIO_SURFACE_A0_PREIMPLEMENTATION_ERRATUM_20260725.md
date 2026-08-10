# TRKH 5-Class CCR Surface A0 Pre-Implementation Erratum

Date: 2026-07-25

Erratum ID:
`trkh_cross_colour_ratio_surface_a0_preimplementation_erratum_r1_20260725`

State: prospective; no CCR candidate implementation, image pass, fit,
candidate metric, validation/test access, or downstream authorization exists

## Immutable Parent Boundary

This erratum follows the prospective boundary committed and pushed at
`2be317a20b95707b08072fba96ec6c3ee2a40384`.

- original protocol SHA-256:
  `1e408502c3fc20031d397d83bf24afb6c63aece539dda1bb8260160fc4eef347`;
- original machine-lock SHA-256:
  `6c2a604deb4f976a3556ffb0ad8ddb60c7f376e81e374ae16272c6e6d49fc835`;
- frozen keeper resolved-config SHA-256:
  `e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674`.

The original files are not rewritten. This document supersedes exactly one
ambiguous sentence in the frozen-input section and adds one implementation
ledger clarification. Cohort, folds, roles, equations, descriptor constants,
optimization, controls, gates, XAI rows, resource ceilings, and all forbidden
paths remain unchanged.

## Issue Found Before Candidate Code

The original protocol correctly requires inversion of the keeper's existing
model-normalized input tensor so a later CCR branch can consume the same
tensor. A later bullet said "no background suppression", which can be read as
constructing a different RGB tensor.

The frozen keeper config and ordinary evaluation path actually use:

- `image_size=256`;
- `crop_margin_ratio=0.05`;
- `resize_mode=pad`;
- `illumination_normalization=true`;
- `illumination_normalization_strength=0.35`;
- `background_suppression_mode=desaturate_blur`;
- `background_suppression_margin=0.08`;
- `background_suppression_blur_radius=7.0`;
- `foreground_crop_mode=none`;
- `surface_detail_amplification_mode=none`;
- `eval_surface_detail_amplification=false`;
- input mean `[0.485,0.456,0.406]`;
- input standard deviation `[0.229,0.224,0.225]`.

The relevant source SHAs are:

- `trkh/data/dataset.py`:
  `7ea29b5f3146995b63179600a5d31f338d154164221f1179b3352846ba4275d5`;
- `trkh/tools/build_precision_ensemble_checkpoint.py`:
  `481b7c1cdc0add86541e4be08175c3a42031d942cf2c86ed6d9e89320f43c7fe`;
- `trkh/evaluation/input_normalization.py`:
  `513f452063260c2f7724d40c621b4b34506e2463bfb78670f8cfb7d58c995e2f`.

## Corrected Same-Tensor Rule

The authoritative interpretation is:

> Apply no additional or different CCR-specific background suppression.
> Reproduce the frozen keeper evaluation tensor byte-for-byte, including its
> already-selected `desaturate_blur` preprocessing, then invert only the
> frozen mean/std normalization before sRGB-to-linear conversion.

The CCR branch may not rerun segmentation, alter the suppression mode,
strength, margin, blur, crop, padding, illumination normalization, or
mean/std. A future integrated branch receives the same tensor already supplied
to the RGB keeper.

## Locked Direct Loader Clarification

The general `MangoYOLOCropDataset` constructor scans every train label file to
build its full sample index. That broad read set is unnecessary and conflicts
with the CCR lock's exact 763-row scientific input declaration.

The independent CCR auditor must instead:

1. use the locked CIDT row for image path, target, source and sample identity;
2. use the locked geometry `model_boxes` as the original normalized object
   bbox and reproduce the exact margin-0.05 integer crop;
3. run the frozen keeper evaluation transform listed above;
4. prove the transformed bbox and downsampled valid mask agree with locked
   `crop_boxes` and `valid_masks` before descriptor extraction;
5. open only locked train image paths and no validation/test path;
6. keep the locked label files hash-protected but avoid opening the complete
   label directory or re-indexing unrelated train rows.

Any bbox, valid-mask, transform-source, config, path, target, or source
mismatch is fail-closed before a descriptor, model state, or candidate metric
is created.

## Authorization

This erratum authorizes only an independent equation engine and structural
preflight after the erratum is committed and pushed. It does not authorize an
image pass, fit, candidate metric, shifted condition, XAI, inference smoke,
validation/test access, production edit, full train, or current-best command
update.
