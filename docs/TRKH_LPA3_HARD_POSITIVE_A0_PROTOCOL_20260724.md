# TRKH LP-A3-Inspired Hard-Positive A0 Protocol

Status: prospectively locked before implementation metrics.

## Question

Can a model-adaptive, label-preserving hard-positive view expose useful
class-1 surface/boundary variation without repeating the rejected
directional FriendlyAdv policy or sacrificing class-1 recall?

This is an A0 mechanism audit, not a training result. It may not access
validation or test, write a checkpoint, alter raw data, or update the
current-best command.

## Primary Sources

- Yang et al., *Adversarial Auto-Augment with Label Preservation: A
  Representation Learning Principle Guided Approach*, NeurIPS 2022:
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/8a1c4a54d73728d4d61701e320687c6d-Abstract-Conference.html
- Main-paper SHA256:
  `32e812a0f603792bb5dd68419349107078fb41319acd439077ab147a1627334e`
- Supplemental SHA256:
  `9e08ddb1b936e56af496fb2d12a69b9815ac2afbb6d37d49a9a35ec34efa1e0a`
- Official-code reference commit:
  `7059ca393b5cdfa04cf3304bb86c4c79712adfa3`
  (`https://github.com/kai-wen-yang/LPA3`). No repository license was
  visible when this protocol was locked. No source code will be copied,
  vendored, imported, or used as a dependency.

The paper defines a hard positive by maximizing intermediate-feature
distance while constraining the true-label log probability. Its supplemental
material uses five optimization steps and an exponentially increasing
Lagrange multiplier. The implementation below is an independently written,
bounded projected solver adapted to TRKH, not a claim of exact reproduction.

## Locked Inputs

- Keeper:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`
- Keeper SHA256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`
- Data spec: `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`
- Data-spec SHA256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`
- Split: `train` only. Validation and test dataset objects must not be built.
- Seed: `20260724`
- Five `source_stem`-grouped folds. A source may occur in exactly one fold.
- Focus class: `1`; restricted rivals: `0,2,4`.
- No global class-1 oversampling and no offline synthetic files.

## Locked Cohort

Run the keeper once over all `9215` train rows in FP32. Select:

- every restricted false positive: true class in `0,2,4`, prediction `1`;
- up to `256` clean-correct class-1 rows nearest the `1` versus `0/2/4`
  boundary;
- up to `256` clean-correct rival rows nearest class `1`.

Selection uses train labels and keeper outputs only. Ties are resolved by
sample index. The cohort is diagnostic and must not become a training sampler.

## Locked Transform

- Work in RGB `[0,1]`, then restore checkpoint normalization.
- Perturb only the valid bbox eroded by `0.10` on each side.
- L-infinity radius: `2/255`.
- Five projected steps.
- Initial noise: `0.01 * N(0,1)`, deterministically derived from seed and
  sample index, then masked and projected.
- Candidate feature: L2-normalized TRKH `features["pooled"]`.
- Candidate objective:
  `feature_distance - lambda * relu(clean_log_p_y - aug_log_p_y - 0.05)`.
- `lambda` schedule: `10 ** (step / 5)`.
- Step schedule: `(2/255) * 0.1 ** (step / 5)`.
- Keep the feasible iterate with greatest feature distance per row. An
  iterate is feasible only when `aug_log_p_y >= clean_log_p_y - 0.05`.
  Fall back to the clean row when no perturbed iterate is feasible.
- Clip pixels to `[0,1]`; outside-mask delta must be exactly zero.

## Locked Controls

1. `random_feasible`: five deterministic random masked candidates at matched
   radius; choose the feasible candidate with greatest feature distance.
2. `feature_only`: the same projected solver with the label-preservation
   penalty disabled. This tests whether the hinge, rather than feature
   distance alone, protects class identity.
3. Clean keeper predictions are the zero-perturbation reference. The rejected
   FriendlyAdv smoke remains a historical directional-adversarial reference;
   it is not rerun or retuned.

No bound, step, margin, layer, cohort, or gate sweep is permitted in A0.

## Required Artifacts

- `summary.json` and SHA256 artifact manifest.
- Full clean-train prediction CSV with source fold.
- Per-view row CSV with probabilities, transitions, feature distance,
  true-label log-prob drop, constraint status, mask fraction, and RGB deltas.
- Per-fold metric CSV.
- Deterministic replay on a fixed small subset.
- A fixed contact sheet containing clean/candidate/controls and amplified
  delta panels for the most consequential class-1 transitions.
- No raw-data write, generated training corpus, model fit, or checkpoint.

## Promotion Gates

All checks are required:

- exact keeper/data hashes, full train support `9215`, no validation/test;
- five complete source-disjoint folds and deterministic replay;
- no empty mask, RGB delta at most `2/255`, outside-mask delta exactly zero;
- candidate label-constraint violation rate `0`;
- class-1 TP retention at least `0.98`;
- clean-correct restricted-rival retention at least `0.98`;
- restricted class-1 FP rejection at least `0.05`;
- candidate FP rejection exceeds `random_feasible` by at least `0.02`;
- candidate TP retention exceeds `feature_only` by at least `0.02`;
- candidate median feature distance exceeds `random_feasible` by at least
  `25%`;
- at least four of five folds have candidate TP retention at least `0.97`
  and positive candidate-minus-random FP-rejection delta;
- visual panels remain faithful and do not expose padding/background-only
  shortcuts.

Passing A0 only authorizes default-off trainer integration and a prospective
smoke. Failure closes nearby radius/step/margin/lambda/layer variants on this
keeper unless a new causal signal is introduced.
