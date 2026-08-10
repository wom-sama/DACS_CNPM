# TRKH 5-Class Sparse Over-Parameterization A0 Closure - 2026-07-17

## Decision

Status: closed before trainer integration and before any image epoch.

The exact ICML-2022 Sparse Over-Parameterization (SOP) A0 route failed its
prospectively locked full-fit mechanism gate. After ten deterministic passes,
none of the 7,372 rows had effective `V > 1e-6`; the learned label-noise term
remained numerically at initialization. The apparently perfect class-1
gradient retention is therefore an identity result, not evidence that an
active SOP mechanism safely protects agricultural precision.

The shared trainer/model, current-best commands, checkpoint, raw data,
source-disjoint holdout, official validation, and test were not changed or
opened. No SOP image epoch, trainer option, hyperparameter rescue, or full
train is authorized.

## Primary Sources

Authority was the accepted paper and authors' licensed source, not the user or
secondary research reports:

- Liu, Zhu, Qu, and You, "Robust Training under Label Noise by
  Over-parameterization," ICML 2022, PMLR 162:14153-14172:
  `https://proceedings.mlr.press/v162/liu22w.html`;
- accepted PDF:
  `https://proceedings.mlr.press/v162/liu22w/liu22w.pdf`;
- authors' MIT repository: `https://github.com/shengliu66/SOP`;
- locked official commit/tree:
  `4d991cedf1fafec98f858a213ccc31e52318a77f` /
  `dadbf1b288599c67d7fcced4db2cbdc3db10c6cd`.

The paper initializes `u/v` from zero-mean Gaussian noise with standard
deviation `1e-8`. Its Clothing-1M setting uses a pretrained ResNet-50, one
million training images, batch 64, ten epochs, and `lr_u/lr_v=0.1/1`. Those
values were used because they are the only source-grounded ten-epoch real-noise
schedule inside the project limit. They are not evidence that a 7,372-row
scratch TRKH problem has equivalent activation dynamics.

## Locked Protocol And Replay Correction

- Original prospective protocol SHA-256:
  `d2b94048da9d2a3ed261fda4f97c333c51bada3b008bd2303629aefd96b687c8`.
- Replay-only erratum SHA-256:
  `34b117ba422157c876c4f4cf650b846a961a2eead409f97cad08f2201591e8ec`.
- Auditor implementation commit: `5a8b03f1e1b713960ce683a386e9d7559f697819`.
- Replay correction commit: `e18d0634a191608d79c915483b4aaa399abdaaf8`.

The first complete fit forward stopped before SOP because baseline constants
copied from the NBDT development scan differed by three near-boundary
decisions from the canonical CIDT/deployment batch-64 path. All 7,372
predictions matched the locked declaration, so the original protocol was kept
immutable and a pre-SOP erratum corrected only the replay constants. Compact
failure evidence is preserved under
`runs/evidence_sop_a0_replay_contract_mismatch_20260717` at summary/manifest
SHAs `1dde47f0...71638` / `d9570f2d...9dea1`.

The corrected canonical flat keeper replay is:

```text
accuracy / macro F1                    0.9605263158 / 0.9375834512
class-1 precision / recall / F1        0.6890343699 / 0.9745370370 / 0.8072866731
class-1 TP / FP / restricted FP        421 / 190 / 186
```

Every corrected replay check passed, including exact target order, exact
confusion matrix, exact `7,372/7,372` declaration-prediction agreement, finite
logits, unchanged model-state hash, and the locked fit-index SHA
`22edca99...ce5d`.

## Equation And Source Replay

The isolated auditor directly loaded the official `overparametrization_loss`
class and compared it with a local exact expression and a separately written
equation. All checks passed:

- official/local loss and optimizer-step maximum error: `0`;
- official/local gradient maximum error: `0`;
- local/independent gradient maximum error: `2.9802322e-8`;
- local/independent probability maximum error: `5.9604645e-8`;
- finite-difference maximum error: `8.6513602e-11`;
- BF16/FP32 probability maximum error: `8.3780289e-4 < 2e-3`;
- every FP32/BF16 tensor and gradient was finite.

The `v` finite difference held the CE stop-gradient value fixed and varied the
MSE path, matching the source's deliberate `V.detach()` behavior.

## Formal Fit Result

Formal output:
`runs/audit_sparse_overparameterization_a0_corrected_replay_20260717`.

The fit forward used the normal metadata-aware FP32 deployment path and took
`237.5091` seconds. The launcher requested four workers; the Windows-safe
loader correctly clamped this to zero. No holdout loader was constructed.

Ten natural-frequency passes used seed 42, batch 128, SGD
`lr_u=0.1/lr_v=1`, zero momentum, and zero weight decay. Optimization took
`2.7498` seconds. Results were:

```text
effective-V active row fraction         0.0
final |u|max / |v|max                    3.9845e-8 / 4.2850e-8
effective V maximum                      1.8361e-15
effective U target maximum               1.5876e-15
maximum noise energy                     3.3766e-30
raw -> corrected changed decisions       0 / 7372
maximum probability change               5.9605e-8
```

Mean loss remained approximately `1.24328528`; CE remained `1.16433791` and
MSE remained `0.07894737`. The mechanism did not acquire a usable sparse-noise
signal.

All 186 restricted false positives retained the positive corrective class-1
gradient and all 421 class-1 true positives retained negative support. Both
cohorts had median SOP/CE magnitude ratio `1.0`, zero rows losing half their
gradient, and no class-1 clamp-to-zero event. This does not rescue A0: the same
CSV shows raw and SOP predictions are identical, probability changes are only
floating-point noise, and effective `V` is nine orders of magnitude below the
locked activation threshold.

## Mechanism Diagnosis

The official square parameterization gives `U=u^2` and `V=v^2`, so gradients
near zero are proportional to the tiny parameter itself. With one occurrence
per pass and initialization around `1e-8`, ten low-rate real-noise passes grew
the largest `u` only to `3.98e-8` and did not make any effective `V` measurable
at `1e-6`. The paper's successful ten-epoch Clothing-1M result also changes a
pretrained network jointly on a million-image stream; it is not a valid basis
for claiming rapid activation on this small scratch hybrid.

Changing initialization, `u/v` learning rates, update count, batch reduction,
projection, schedule, or jointly training the network after observing this
failure would be a post-result rescue. The protocol explicitly forbids those
nearby sweeps. This closure rejects exact A0 for TRKH; it does not claim that
SOP is generally invalid.

## Artifacts And Scope

Formal artifact SHAs:

- `gradient_rows.csv`: `8bd9960f...a75f2` (`2,858,865` bytes);
- `summary.json`: `1b88ca7b...e479f` (`15,922` bytes);
- `report.md`: `ea4be71f...d56b` (`547` bytes);
- `artifact_manifest.json`: `a1528f42...bd73`.

All three manifest payload hashes and byte counts independently replayed. The
formal directory is only `2,875,888` bytes and contains no checkpoint or
image, so it remains compact evidence and no cleanup is warranted. XAI was not
triggered because no model parameter, checkpoint, prediction, or image-forward
behavior changed; existing keeper XAI remains the applicable explanation.

The raw data/keeper/current-command/history SHAs remain respectively
`716e33df...884ef`, `1f49d577...82677`, `36b9aa1a...9faf`, and
`39bd2879...8f53`. Current-best tracking remains three revisions/two updates.

Closure verification passed Python compileall, PowerShell parsing for the SOP
wrapper plus all five current train/export/video wrappers (`6/6`), focused SOP
tests (`5/5`), and the full repository suite (`1,337/1,337`). The warnings were
existing dependency deprecations and tracing notices, not test failures.

## No-Repeat Rule

- Do not integrate this SOP A0 loss into the trainer.
- Do not run an SOP image epoch, holdout, validation, test, or full train.
- Do not rescue by changing initialization, `u/v` learning rates, passes,
  epochs, batch size/reduction, projection, network-loss coupling, SOP+,
  consistency, balance, SGN, Label Wave, seed, or fold.
- Do not treat identity gradient retention as mechanism safety.
- Preserve the protocol, replay erratum, closure, failure evidence, and formal
  gradient artifacts as the no-repeat record.
- Select the next route from a distinct accepted primary source with licensed
  official code and a pre-training gate tied directly to class-1 precision and
  true-positive preservation.
