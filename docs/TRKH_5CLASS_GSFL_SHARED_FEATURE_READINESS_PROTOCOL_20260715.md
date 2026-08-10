# TRKH 5-Class GSFL Shared-Feature Readiness Protocol

Date: 2026-07-15

## Question

Can a learned shared/discriminative decomposition remove class-common nuisance features
from the frozen keeper embedding while preserving every current class-1 true positive?

This is a train-only frozen-adapter A0. It cannot edit the dataset, train the shared TRKH
backbone, access validation/test, or write a model binary.

## Primary source and implementation boundary

- Li and Monga, "Group Based Deep Shared Feature Learning for Fine-grained Image
  Classification", BMVC 2019 / arXiv:2004.01817:
  https://arxiv.org/abs/2004.01817
- Author repository, commit `8527039b0eb6e0eb7470521844a91acfac865fe5`:
  https://github.com/xueluli/GSFL-Net

The repository has no license, so no source code is copied. The local adapter is a clean
paper-equation implementation. The following incompatibilities are resolved before code:

- The published experiments use pretrained VGG and 150-epoch head training followed by
  200-epoch fine-tuning. TRKH fixes one frozen adapter to 30 epochs.
- The paper chooses the number of class groups by cross-validation. TRKH fixes one group,
  because there are only five classes and the tested hypothesis is removal of a global
  shared/background component. No group-count or membership search is allowed.
- The official shared-center loop uses `labels[0]` for every row after the first. TRKH uses
  each row's own label as required by paper Equation 2.
- The official decoder ends in sigmoid although both its compact-bilinear input and the
  keeper embedding are signed. TRKH uses a linear final decoder layer so reconstruction
  can represent negative values. This is declared as an equation-compatible correction,
  not an exact source replay.
- The official loop evaluates a test loader every epoch. TRKH holds the source-disjoint
  train fold closed until epoch 30 and never constructs validation or test input.

## Locked data and comparators

- Frozen keeper train embedding cache: 9,215 `yolo_f/train` rows, 256 dimensions.
- Existing source-grouped fold 0:
  - fit: 7,372 rows and 6,452 source groups;
  - holdout: 1,843 rows and 1,612 source groups;
  - overlap: zero.
- Canonical raw keeper and aggregate-margin A-GEM probabilities from the CAGrad evidence.
- The one-argmax / maximum-probability-`0.0327522` cache replay reconciliation remains
  identical to the preceding class-axis protocol.

All inputs, this protocol, the primary PDF, official source files, and official commit are
locked by hash. Validation and test paths are forbidden.

## Fixed adapter and control

Input embeddings are L2-normalized per row, matching the paper's feature normalization.

The GSFL candidate has:

- discriminative encoder: `Dropout(0.5) -> Linear(256,128) -> Dropout(0.5) ->
  Linear(128,128) -> ReLU`;
- shared encoder: the same topology;
- decoder: `Linear(128,128) -> ReLU -> Linear(128,256)` applied to the sum of shared and
  discriminative components;
- classifier: `Linear(128,5)` on only the discriminative component.

The matched control contains the identical discriminative encoder and classifier, copied
bit-for-bit at initialization, and trains with cross entropy only. Candidate and control
receive the same natural-frequency batches and deterministic discriminative dropout masks.

The candidate loss is paper Equation 1 plus Equation 2:

`L = CE + 0.01 * reconstruction + 0.1 * class_center + 0.1 * shared_center`.

Reconstruction is element-mean MSE. Center terms sum per-row feature-mean MSE, matching
the official batch scaling. Class and total-class centers start at zero and use the
official `0.01 * sum(center-feature)/(1+count)` update after each optimizer step. With one
group, the shared center for every class is the mean of all five total-class centers.

The sole optimizer recipe is Adam, learning rate `0.001`, weight decay `1e-5`, batch 42,
natural shuffle, seed `20260715`, and exactly 30 epochs. The final epoch is evaluated once;
there is no early stopping, holdout checkpoint selection, or hyperparameter sweep.

## Structural gates

- Every locked hash, row, path, label, fold, and source group is exact.
- Candidate/control discriminative encoder and classifier are bit-identical initially.
- Candidate and control see identical batch indices and discriminative dropout masks.
- All losses, logits, gradients, probabilities, and centers are finite.
- Both train objectives decrease from epoch 1 to epoch 30.
- Candidate reconstruction loss decreases by at least 5%.
- Every class center is updated and shared/discriminative outputs are not identical.
- The output contains no checkpoint, adapter, ONNX, engine, validation, or test payload.

## Behavioral authorization gates

The GSFL candidate must pass all structural gates and all of these holdout checks:

- macro F1 is not below raw keeper or the matched CE adapter;
- class-1 F1 gains at least 0.005 over raw keeper and matched CE;
- class-1 precision gains at least 0.010 over raw keeper;
- class-1 recall loses no more than 0.005 versus raw keeper;
- class-1 F1 is within 0.002 of aggregate-margin A-GEM;
- zero raw class-1 true positives are broken;
- at least four restricted `0/2/4 -> 1` false positives are removed;
- corrections exceed harms; and
- false-negative-positive direction AUROC is at least 0.60.

Failure denies shared-trainer integration and all nearby width, group, decoder, center,
loss-weight, optimizer, epoch, seed, and sampling sweeps. It closes only this frozen GSFL
adaptation under the TRKH 30-epoch constraint, not the original method universally.
