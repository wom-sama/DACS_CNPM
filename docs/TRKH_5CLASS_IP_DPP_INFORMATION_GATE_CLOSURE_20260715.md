# TRKH 5-Class IP-DPP Information Gate Closure

Date closed: 2026-07-15

## Decision

Reject direct IP-DPP and the locked local `k=541` adaptation before trainer
integration. The official implementation selects a random initial subset; the
paper kernel cannot initialize through the pinned DPPy path at TRKH
cardinality. Do not tune k, seed, chain length, kernel scale, probability
source, resampling frequency, or continuation schedule around this result.

No validation or test split was constructed. No raw image, label, bbox, split,
keeper, scratch checkpoint, or current-best command file was modified.

## Locked Evidence

- Protocol SHA-256: `652c1fa4...9cd77f`.
- Train-only cache: four conditions x 9215 rows, 8064 source groups, exact
  class counts `[1941,541,1920,2520,2293]`, and five source-disjoint folds.
- Official repository commit: `20d69a6...65c41c0`.
- Official sampler SHA-256: `2754011d...3f57`.
- Runtime: `dppy==0.3.3`, `numpy==1.26.4`, `scipy==1.13.1`.
- Primary probabilities: scratch full-run `candidate_*`; independent
  consistency probabilities: keeper `keeper_*`.

The local diagnostic used exactly `k=N_min=541`, seed `42`, official reversed
class order `[4,3,2,1,0]`, and 256 precommitted random balanced controls. This
cardinality retains every class-1 object and introduces no repeats.

## Source-Level Failures

1. Paper default `k=10*N_min=5410` applies `min(k,N_c)` and therefore retains
   all 9215 TRKH rows. Direct IP-DPP is identity at the local `4.66x` maximum
   imbalance ratio.
2. Paper Eq. 16 uses off-diagonal scale `/N`; official code uses `/N^2`.
3. DPPy stores a default ten-state MCMC chain, but official code reads
   `list_of_samples[0][0]`: the uniform random initialization before any
   exchange.
4. Paper/kernel schedules also differ: the appendix describes resampling every
   10 epochs while official code defaults to 20. The full official recipe uses
   `1000+100+100` epochs and is outside the local `<=30` contract.

## Runtime Results

Both probability sources produced the same exact selections:

- official first/uniform/replay-first SHA-256:
  `3c55ff6f...718eb27`;
- returned final-state SHA-256: `941c9bd3...c30128`;
- first/final Jaccard: `0.986050` after only nine possible exchange steps;
- official first equals direct uniform initialization for every sampled class;
- selected rows: `2705`, all unique, exactly `541` per class, including all
  `541` class-1 rows.

The code-`N^2` determinant distribution is effectively uniform:

| Probability source | Code q95/q05 odds | Official restricted FP | Random FP q95 | Official self-info | Random self-info q95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scratch | 1.000610 | 59 | 88.00 | 1.160899 | 1.166647 |
| Keeper | 1.000721 | 56 | 73.25 | 1.154993 | 1.159714 |

Official restricted-FP and self-information values were below random q95 in
clean, dim, bright, and low-contrast conditions for both sources. No fold
exceeded random q95 for clean restricted-FP coverage. Source coverage and fold
composition passed, but they only prove a valid random balanced subset.

## Paper-Kernel Diagnostic

The paper-`N` kernel has nontrivial proposal-bank odds (`3.9238x` scratch,
`5.1570x` keeper), but every majority-class size-541 determinant upper bound
is below DPPy's `1e-9` initialization threshold. Bounds range from log-det
`-44.39` to `-52.49`, so the official DPPy path cannot initialize any such
subset.

The maximum-logdet entries among the same 256 random proposals exceeded their
own random q95, as expected after selecting a maximum, and paper/code chose the
same proposal for each probability source. This is a winner-selected random
diagnostic, not an IP-DPP sample, and cannot authorize training.

## Verification And Retention

- Six conjunctive checks failed: direct-default activity, non-random official
  extraction, paper/code scale agreement, paper-kernel DPPy feasibility,
  code-kernel determinant odds, and cross-condition hard-example enrichment.
- Independent replay reproduced exact `541 x 5` selections and confirmed every
  DPPy first state equals its cloned uniform initialization. Independent null
  q95 values matched the summary.
- Evidence root: `runs/audit_ip_dpp_information_gate_20260715`.
- Seven nonbinary payloads total `6,812,456` bytes; payload-manifest SHA-256 is
  `a77f9788...09608e`.
- Formal summary SHA-256 is `5c59baaa...77366b`; no checkpoint, ONNX, engine,
  model binary, validation payload, or test payload exists.
- Read-only retention passed over `684` run directories with `blockers=[]`,
  `72.34 GB` free, and all three compacted natural-prior originals absent.
  Retention summary SHA-256 is `95557ebf...96ef4`.

Trainer, validation smoke, test, probe, and full-train permissions are false.
Current-best full-train commands and their three-revision history remain
unchanged.
