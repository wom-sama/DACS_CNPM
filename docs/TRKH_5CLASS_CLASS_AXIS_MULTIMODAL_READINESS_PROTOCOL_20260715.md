# TRKH 5-Class Class-Axis and Multimodal Readiness Protocol

Date: 2026-07-15

## Decision scope

This is a locked, train-only A0 information gate for two representation families:

1. class-conditional activation blocks inspired by CCAR; and
2. intra-class multimodal representations inspired by Goto et al., WACV 2024.

It is not a reproduction of either end-to-end method. It asks whether the frozen keeper
embedding already contains the class-axis or multimodal density signal needed to justify
shared-trainer work under the TRKH limit of at most 30 epochs.

## Inputs

- Keeper train embedding cache only: 9,215 `yolo_f/train` rows, 256 dimensions.
- Locked fold 0 from the existing source-grouped A-GEM/CAGrad audit:
  - fit: 7,372 rows;
  - holdout: 1,843 rows;
  - source overlap: zero.
- Raw keeper and aggregate-margin A-GEM predictions on the same holdout.
- Validation and test are forbidden.

The reusable embedding cache and the later CAGrad comparator replay use the same keeper
and image paths but differ slightly in extraction/evaluation plumbing. Their raw holdout
replays are therefore reconciled explicitly: exactly one of 1,843 argmax predictions may
differ and the maximum probability difference must be at most 0.04. Candidate gates use
the later locked CAGrad raw and aggregate-margin predictions as the canonical comparators.

Every input, this protocol, and both primary literature files are SHA-256 locked by the
auditor. The run may write JSON, CSV, Markdown, and a manifest only. It may not write a
checkpoint, adapter, model binary, or anything under the raw dataset tree.

## Fixed candidates

No candidate or hyperparameter sweep is allowed.

### Balanced class-axis diagnostic

1. Fit `StandardScaler` on fit rows only.
2. For class `c` and dimension `j`, compute:
   `s[c,j] = (mean(z_j^2 | y=c) - mean(z_j^2 | y!=c)) /
   (mean(z_j^2 | y!=c) + 1e-8)`.
3. Use one Hungarian assignment to allocate exactly 51 unique dimensions to each of the
   five classes. One of 256 dimensions remains unused. This deliberately gives the family
   a stronger, label-informed frozen-feature screen than CCAR's fixed contiguous blocks.
4. A class score is the mean squared standardized activation over its assigned dimensions,
   divided by that score's fit-only standard deviation. Apply softmax once.

### Multimodal density diagnostic

1. Reuse the fit-only standardized embeddings.
2. Fit one 10-component diagonal-covariance Gaussian mixture per class.
3. Lock `reg_covar=1e-4`, `max_iter=200`, `n_init=1`, and seed `20260715`.
4. Add the natural fit-class log prior to each class log likelihood and apply softmax once.

The mixture is a necessary multimodal-density proxy, not an implementation of the WACV
orthonormal-matrix optimizer.

## Authorization gate

Each candidate is compared independently with raw keeper and aggregate-margin A-GEM. A
candidate authorizes shared-trainer work only when all structural checks pass and it meets
all of these locked holdout checks:

- macro F1 is not below raw keeper;
- class-1 F1 gains at least 0.005 over raw keeper;
- class-1 precision gains at least 0.010 over raw keeper;
- class-1 recall loses no more than 0.005 versus raw keeper;
- class-1 F1 is within 0.002 of aggregate-margin A-GEM;
- no raw class-1 true positive is broken;
- at least four restricted class-1 false positives (classes 0, 2, and 4) are removed;
- corrections exceed harms; and
- the false-negative-positive direction AUROC is at least 0.60.

`shared_trainer_authorized` is true if at least one fixed candidate passes. Failure closes
only these head/regularizer families on the current keeper representation; it does not
claim that the original methods are universally ineffective.

## Primary references

- Goto et al., "Learning Intra-Class Multimodal Distributions With Orthonormal Matrices",
  WACV 2024: https://openaccess.thecvf.com/content/WACV2024/html/Goto_Learning_Intra-Class_Multimodal_Distributions_With_Orthonormal_Matrices_WACV_2024_paper.html
- Samanta et al., "CCAR: Intrinsic Robustness as an Emergent Geometric Property",
  arXiv:2604.16861 (unreviewed preprint): https://arxiv.org/abs/2604.16861
