# TRKH 5-Class Learnable Polyphase Downsampling A0 Closure - 2026-07-17

## Decision

Reject the locked three-stage Learnable Polyphase Downsampling (LPD) route
before production integration or an image epoch.

The current keeper has material one-pixel shift instability, so the motivating
signal is real. The exact source-grounded LPD adaptation is nevertheless too
slow, uses too much peak memory, and lacks locked PyTorch/ONNX phase parity.
No matched scratch pair, validation, test, full train, or current-best command
update is authorized.

## Locked identity

- Protocol:
  `docs/TRKH_5CLASS_LEARNABLE_POLYPHASE_DOWNSAMPLING_A0_PROTOCOL_20260717.md`.
- Protocol SHA-256:
  `6ba9f452dcf75b3c2815adf0055651769a5e362cec8d0c3b5e033e17bb3b8485`.
- Pushed implementation commit:
  `a28a7c7` on `classification-only-research`.
- Formal output:
  `runs/audit_learnable_polyphase_downsampling_a0_20260717`.
- Pre-visual summary SHA-256:
  `99defdf5360654ac2dd46d15c5978343184db04c4c6307d822b0ad244a3f93f8`.
- Final summary SHA-256:
  `a2c003d541c6b4a5836bdb7359ce2f60778f3d684f2222ffb5e8addebdeb6679`.
- Final artifact-manifest SHA-256:
  `a5cdc5df910060b07792d31e5a103c181778f8c8a88d489a6bdbb4ed84b27106`.
- Official LPS commit/tree:
  `ef28ff29aa058c5f3fdf05b9a096f196ee1d0f30` /
  `74cda577ec36cc9288a0f2cded289b9b3b3b2269`.
- Accepted-paper SHA-256:
  `cd13170aa45bc0b2f7afdd62e723bb2dc1674345012f26d45994137c4e3689dc`.

Preflight required exact source, paper, MIT license, dataset, keeper, CIDT,
current-command, protocol, repository, and upstream hashes. It passed without
creating an output directory. Formal started with no unrelated Python or
TensorRT process and GPU `11% / 1468 MiB`; no process was terminated.

## Equation evidence

All isolated equation gates pass:

- official and independent V2 phase-split errors: `0`;
- official selector-logit and hard-output errors: `0`;
- maximum official gradient error: about `4.16e-17`;
- central finite-difference error: about `3.38e-12`;
- circular phase-logit permutation error: about `6.94e-18`;
- circular hard-output global-mean error: about `5.55e-17`;
- fixed phase 0 is bit-identical to all three legacy max pools;
- strict BF16 output error: `0.007789`, with finite gradients and zero phase
  mismatch;
- both selector convolution kernels receive finite nonzero gradients and move
  under the locked Gumbel-Softmax step.

Real-image selector logits are finite and nonconstant in all three stages.
Random-initialization phase counts also use more than one phase at every stage.
The rejection is not caused by a fabricated or inactive implementation.

## Shift signal

The shift-signal gate passes on the exact train-only cohort:

- rows: `607`, with `421` class-1 TP and `186` restricted FP;
- decision-stable denominator after the sole locked sample-3657 near-tie:
  `606 = 421 TP + 185 restricted FP`;
- samples leaving class 1 under at least one one-pixel shift: `46`;
- TP exits: `11`;
- restricted-FP exits: `35`;
- fold exit counts: `14 / 15 / 9 / 8` for folds `1 / 2 / 3 / 4`;
- every cardinal and diagonal direction produces exits;
- median/P90 class-1 probability span: `0.016296 / 0.0309996`;
- independent replay is exact across `5,463` condition rows, with zero summary
  or sample-row difference.

This proves that one-pixel sampling/boundary sensitivity is material for the
keeper, especially among false class-1 decisions. It does not prove that LPD
can correct the direction of those changes or improve class-1 precision.

## Structural rejection

Four prospectively locked deployment checks fail:

| Check | Control | LPD | Result |
| --- | ---: | ---: | --- |
| Median batch-32 runtime | `74.19 ms` approximate ABBA median | `104.28 ms` approximate ABBA median | ratio `1.405570 > 1.15` |
| Peak allocated memory | `391,810,048` bytes | `1,031,835,648` bytes | ratio `2.633510 > 1.10` |
| ONNX feature max error | - | `0.530902` | `>1e-5` |
| ONNX phase mismatch by stage | - | `[0,3,22] / 32` | must be `[0,0,0]` |

The candidate adds only `97,456` parameters (`1.013450x` total), and standard
ONNX domains plus TensorRT parse/build pass. Those positives cannot override
runtime, memory, or backend-output failures. The untrained stage-2/3 selector
logits are close enough that backend arithmetic changes hard argmax phases;
this is a deployment risk, not permission to relax parity after observing it.

## Visual review

All four hash-locked contact sheets pass manual validity review:

- image translations move in the declared direction by one pixel;
- valid-mask boundaries and transformed bboxes remain aligned;
- no circular wrap, interpolation, unreadable panel, or metadata displacement
  artifact is visible;
- displayed probability/margin changes agree with the CSV transitions.

Visual validity cannot override the structural rejection. Final status is
`rejected` with matched-scratch and production-integration authorization both
false.

## Retention and cleanup

- Compact formal payload: `4,956,768` bytes across nine files.
- Shift prediction/sample CSV SHAs:
  `fa87636075bcf1bc25397c14f2ac58668625c1e663622468497b511d98e567d2` /
  `e128a25e30fdffaeb5eb1765a88e83f5b95600aed460b5e4c16d786b64f69903`.
- No checkpoint, ONNX graph, TensorRT engine, raw-data edit, or trainable
  manifest is retained.
- Keep the compact numeric and visual evidence as a no-repeat record; there is
  no useful LPD binary artifact to delete.
- Full retention audit uses all 48 object manifests, verifies all `210`
  compacted originals absent, returns `blockers=[]`, and reports `103.037 GiB`
  free at SHA
  `24945ef24a028524f44fedb90b0e5d9e93d1fe50aea7ced041ce6c231a96a871`.
- A weaker built-in-only retention output was verified, then removed after the
  all-manifest audit superseded it. No dataset or model evidence was deleted.

## No-repeat boundary

Do not rerun or sweep this LPD family on the current keeper through selector
width, selector architecture, phase order, layer subset, padding, temperature,
hard/soft mode, Gumbel schedule, antialias filter, APS/max-norm substitution,
shift size/fill, batch, seed, fold, runtime compiler, custom kernel, ONNX
tolerance, or relaxed resource gates. Do not copy BlurPool/APS code from the
inspected non-reusable licenses.

The keeper's shift signal may motivate a future equation-distinct and
deployment-light train-time method from an accepted paper and reusable
official source. Such a method must be screened against the existing closed
augmentation, consistency, SPT, pooling, frequency, and token-routing records
before a new protocol. It cannot be described as an LPD rescue.

Current-best commands remain at three revisions and two actual updates, with
lock-time command/history SHA-256 values
`36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf` /
`39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.
