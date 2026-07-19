# TRKH 5-Class Spectral Decoupling A0 Protocol (2026-07-20)

## Status

- Prospective protocol: locked before candidate training or validation behavior.
- Scope: scratch/no-pretrain TRKH classification on `yolo_f`; raw data is read-only.
- Test split: forbidden for selection, tuning, or this A0 decision.
- Current-best train commands: unchanged unless all locked validation gates pass.

## Authority And Reproducibility

Primary paper:

- Pezeshki et al., *Gradient Starvation: A Learning Proclivity in Neural
  Networks*, NeurIPS 2021.
- Proceedings page:
  <https://proceedings.neurips.cc/paper/2021/hash/0987b8b338d6c90bbedd8631bc499221-Abstract.html>
- Paper PDF:
  <https://proceedings.neurips.cc/paper_files/paper/2021/file/0987b8b338d6c90bbedd8631bc499221-Paper.pdf>

Paper-linked implementation:

- Repository: <https://github.com/mpezeshki/Gradient_Starvation>
- License: MIT.
- Commit: `bc5c29631fc3abe72e7d05c7ab3424c342e7fd49`.
- Tree: `2fbce6eaeceb46ecadcd2c2ede600c2faa1478a7`.
- Relevant official blobs:
  - `Figure_4_and_table_1/cmnist.py`:
    `9dd67b3b1bc3b87af125dce53083fbb38973a342`.
  - `Table_2/loss.py`:
    `5fc71d46f7a3238d2864420ad9bd18048817dd9f`.
  - `Table_2/table_2.sh`:
    `f7e385e605ce1fedfc91bc19d64d44c1c6b20702`.

Locked local inputs:

- Source launcher arguments:
  `runs/full_v8_yolof_randominit_30e_20260714_105524/launcher_args.json`.
- Source launcher SHA-256:
  `a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6`.
- Data YAML: `D:/DataAI/AIEx/newdataset/yolo_f/data.yaml`.
- Data YAML SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Keeper validation predictions:
  `runs/eval_yolof_teacherfocusbinary015_best_val_20260702/predictions_detailed.csv`.
- Keeper prediction SHA-256:
  `9a3ee6b7ee1ab53ad8c140a864789d0dc29029ac17c78b55bca594c6af66a088`.

## Rationale And Exact Objective

The paper identifies cross-entropy gradient starvation: a strong feature can
reduce gradient support for other predictive features. Equation 17 replaces
parameter L2 weight decay with an L2 penalty on network outputs. This is a
plausible fit for TRKH because current XAI shows reliable fruit localization
but weak class-1 surface/boundary separation, while prior confidence-oriented
LogitNorm changed the loss geometry and failed the precision gate.

For TRKH's multiclass minibatch logits `z` with shape `[B, C]`, A0 uses the
class-count-normalized extension

```text
L_i = CE(z_i, y_i) + (lambda / 2) * mean_c(z_i,c^2)
L   = mean_i(L_i)
```

with `lambda = 0.01`. The mean over classes follows the official code's
`(logits ** 2).mean()` scaling while retaining the paper's explicit `lambda/2`
parameterization. The CE term supports existing hard/soft targets and class
multipliers. The output penalty is not class-weighted. Existing per-sample
weights, if present, apply to the complete per-sample objective; A0 uses no
sample-weight manifest.

Training uses Spectral Decoupling, but validation loss is ordinary CE with the
same label-smoothing setting. This prevents checkpoint reporting from grading
the candidate with its own train-only regularizer.

The Colored-MNIST delayed/scaled schedule and CelebA class-conditional shifted
penalties are out of scope. They solve different binary bias settings and are
not evidence for a TRKH schedule. No lambda sweep is allowed in A0.

## Stage A: Equation And Integration Preflight

Stage A may read code, the source argument file, and train metadata. It must not
load validation or test images or inspect candidate predictions.

Required checks:

- hard-target loss equals the locked equation;
- soft-target and class-multiplier behavior is finite and differentiable;
- `lambda=0` is exactly ordinary CE;
- negative lambda is rejected;
- train criterion is SD while eval criterion is ordinary CE;
- CLI and V8 launcher preserve the chosen lambda in resolved metadata;
- control and candidate arguments normalize to exact equality except
  `--classification-loss`;
- no pretrained weights, resume checkpoint, final test, or raw-data mutation;
- both roles use `weight_decay=0` because Eq. 17 replaces parameter L2 decay;
- both roles use `label_smoothing=0` to isolate the source objective.

Any failed check closes A0 before image training.

## Loader Benchmark

Before the formal pair, benchmark requested train workers `{2, 4}` with batch
size 32, image size 256, the same GPU, and Windows multiprocessing enabled.
Disable pin memory and persistent workers to match the known-stable V8 Windows
path. Select the higher-throughput setting only if startup and all measured
batches complete without worker errors. Both pair roles must use the same
chosen train worker count; evaluation uses two workers. Record requested and
effective values. GPU GUI load invalidates absolute timing but not a repeatable
same-session worker comparison; re-run if the result is within 3 percent.

## Stage B: Matched Scratch Pair

The source architecture and all non-loss training options come from the locked
full-run launcher arguments, with these prospectively fixed overrides:

| Setting | Control | Candidate |
| --- | --- | --- |
| initialization | scratch, seed 42 | scratch, seed 42 |
| deterministic | true | true |
| classification loss | ordinary CE | spectral decoupling |
| spectral lambda | ignored (`0.01` recorded) | `0.01` |
| label smoothing | `0` | `0` |
| weight decay | `0` | `0` |
| epochs | `5` | `5` |
| max train batches/epoch | `120` | `120` |
| max validation batches | `0` (full 2,606 rows) | same |
| batch / accumulation | `32 / 2` | same |
| LR / minimum LR | `2.5e-4 / 1e-6` | same |
| warmup / scheduler horizon | `1 / 30` epochs | same |
| patience | `3` | same |
| bbox token prior | `crop_bbox` | same |
| final test | skipped | skipped |

The formal order is control then candidate. A run directory may never be
overwritten. Architecture traces remain enabled. If either role crashes, the
pair is incomplete and no surviving metric may authorize a rerun with changed
settings.

## Locked Decision Gates

All gates compare independent full-validation reloads of candidate versus the
matched control. Class 1 is the precision-priority focus class. Restricted
class-1 false positives are targets in classes `0, 2, 4` predicted as class 1.

Required clean gates:

- class-1 precision delta `>= +0.005`;
- class-1 F1 delta `>= +0.005`;
- macro-F1 delta `>= -0.001`;
- class-1 recall delta `>= -0.010`;
- restricted class-1 FP reduction `>= 2`;
- candidate corrections `>=` harms versus control;
- full support is exactly `2,606` rows with identical order/targets.

Required robustness/XAI gates before any longer probe or full train:

- dim, bright, and low-contrast class-1 precision never decrease;
- each shifted class-1 F1 delta is `>= -0.010`;
- aggregate shifted restricted FP does not increase;
- paired XAI covers every selected changed case in FP32 with robustness probes;
- no new systematic background/border reliance;
- class-1 TP breaks do not exceed class-1 FN rescues on the XAI cohort;
- architecture trace contains one valid sample per class;
- test split remains untouched.

If the candidate misses any clean gate, it is rejected before expensive XAI.
If clean gates pass but robustness/XAI fails, it is rejected after those audits.
No nearby lambda, weight-decay, label-smoothing, epoch, seed, or threshold sweep
is permitted from the same validation result.

## Full-Train Permission

A longer scratch train is authorized only after every Stage A/B and
robustness/XAI gate passes. Its horizon must be selected from observed
convergence, remain at or below 30 epochs, use early-stopping patience 3, and
retain the measured worker configuration. It still cannot read test until the
existing locked validation-promotion pipeline independently accepts the raw
checkpoint. Passing A0 alone cannot update the current-best command files.
