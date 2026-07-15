# TRKH 5-Class IP-DPP Information Gate Protocol

Date locked: 2026-07-15

## Question

Can the official Information-Preservable Determinantal Point Process (IP-DPP)
select class-balanced, informative head-class examples that suppress class-1
false positives while retaining every true class-1 training object?

This is a no-training A0. It does not authorize a sampler, continuation,
validation smoke, test evaluation, or full train by itself.

## Primary Sources And Transfer Risk

- Paper: Lin and Yuan, *Long-Tailed Recognition via Information-Preservable
  Two-Stage Learning*, NeurIPS 2025:
  https://proceedings.neurips.cc/paper_files/paper/2025/file/431fe53889b90769f5b1678f9ce4cfe8-Paper-Conference.pdf
- Official repository: https://github.com/fudong03/BNS_IPDPP
- Locked official commit:
  `20d69a676215e854d34fbdefa4ba4b3e165c41c0`.
- Official sampler source:
  `sampling/ip_dpp_smapler.py`, SHA-256
  `2754011d4bf556fbd4a8ada40d0169b57c03a6e62cb18ad0c7292a14cb133f57`.
- Runtime dependency: `dppy==0.3.3`, `numpy==1.26.4`, and
  `scipy==1.13.1` in `D:/DataAI/.venv`.

The official recipe is adverse transfer evidence: BNS pretraining uses 1000
epochs, linear probing 100 epochs, and IP-DPP fine-tuning 100 epochs with five
warmup epochs. The official code resamples every 20 epochs; the paper appendix
describes every 10 epochs. Neither schedule fits the local no-pretrain,
`<=30`-epoch contract.

## Source Discrepancies To Audit

1. Paper Eq. 16 defines off-diagonal `S_ij = p_i*p_j/N`. Official code uses
   `p_i*p_j/N^2`.
2. DPPy `sample_mcmc_k_dpp` stores a 10-state exchange chain by default. The
   official source reads `list_of_samples[0][0]`, which is the uniformly
   initialized state before any exchange transition, instead of the returned
   final state.
3. DPPy initialization rejects a subset when its determinant is at most
   `1e-9`. The paper-`N` kernel must be checked analytically at TRKH scale
   before attempting expensive MCMC.
4. The paper recommends `k=10*N_min` and applies `min(k,N_c)`. TRKH has
   `N_min=541`, so official `k=5410` retains all 9215 rows and performs no
   selection at the observed imbalance ratio.

These are mechanism checks, not implementation details that may be silently
repaired after results are seen.

## Locked Inputs

- Train-only cache:
  `runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv`,
  SHA-256
  `2e0993752d58d99ea429bfeefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Required conditions: `clean`, `lighting_dim`, `lighting_bright`, and
  `low_contrast`, each with 9215 rows.
- Clean sample indices: exactly `0..9214`.
- Clean class counts: `[1941, 541, 1920, 2520, 2293]`.
- Source groups: `8064`; source-disjoint CIDT folds: five.
- Primary probability source: scratch full-run `candidate_*` columns.
- Independent consistency source: keeper `keeper_*` columns.
- Restricted class-1 false positives: targets `0,2,4` predicted as class 1 by
  the corresponding probability source.

All paths must resolve under `yolo_f/images/train`. Any validation/test path,
non-finite probability, probability normalization error, duplicate sample
index, source-fold overlap, or provenance hash mismatch aborts A0.

## Locked Cardinality And Randomness

The direct paper default is audited first and is expected to retain all rows.
For the explanatory local diagnostic only, use exactly `k=N_min=541` per
class. This is the unique exact-balanced cardinality that retains all 541
class-1 objects and introduces no repeated object.

- Official class traversal: reversed order `[4,3,2,1,0]`.
- Official DPP RNG: one shared `numpy.random.RandomState(42)` per probability
  source.
- Null proposal bank: exactly 256 balanced subsets. Replication `r` uses one
  shared `RandomState(20260715+r)` in the same reversed class order.
- No k, seed, proposal-count, chain-length, probability-source, class-set, or
  condition sweep is permitted.

## Locked Kernels And Diagnostics

For class probability vector `p`, class size `N`, and scale `alpha`, construct

`S = diag(1 - alpha*p*sum(p)) + alpha*p*p^T`.

- Paper kernel: `alpha=1/N`.
- Code kernel: `alpha=1/N^2`.

The exact subset log determinant is computed without a dense determinant:

`log det(S_A) = sum(log(d_i)) + log(1 + alpha*sum(p_i^2/d_i))`,

where `d_i=1-alpha*p_i*sum(p)` and `i` belongs to subset `A`. A dense small
fixture must match this rank-one identity.

For both scratch and keeper probabilities, A0 must report:

- symmetry, row-sum, diagonal, PSD/eigenvalue bounds, and rank-one/dense
  equation parity;
- paper/code log-determinant distribution over the same 256 random subsets;
- `q95-q05` determinant odds and correlation with mean self-information;
- official first-state selection, direct uniform initialization, and returned
  final 10-state selection;
- exact overlap/Jaccard, accepted exchanges, cardinality, duplicate samples,
  unique source coverage, and fold composition;
- selected target probability, self-information, wrong-class count,
  restricted class-1 FP count, and per-condition linked hardness;
- null `q05/q50/q95`, per-fold restricted-FP enrichment, and deterministic
  replay;
- a clearly labeled proposal-bank best-logdet diagnostic. It is not an
  IP-DPP sample and cannot authorize training.

For the paper kernel, compute a valid upper bound on every size-541 subset
determinant. If this upper bound is at most `1e-9`, record official DPPy
initialization as mathematically impossible and do not spend 100 failed
initialization trials.

## Conjunctive Gates

Trainer integration requires every check below:

1. Input/provenance integrity, train-only path scope, probability parity,
   source-fold disjointness, deterministic replay, and all spectral/equation
   checks pass.
2. The official `k=10*N_min` setting changes at least one majority-class
   selection. Identity selection fails.
3. The exact official first-state result differs from its direct uniform
   initialization and exceeds the random `q95` for both restricted-FP coverage
   and self-information. Exact equality is a source-implementation failure.
4. Paper Eq. 16 and official code use the same kernel scale. `/N` versus
   `/N^2` is a failure, even if one post-hoc variant looks useful.
5. The paper kernel can initialize through the pinned official DPPy path at
   `k=541`; its all-subset determinant upper bound must exceed `1e-9`.
6. Code-kernel `q95/q05` determinant odds are at least `1.10`, proving a
   materially non-uniform selection distribution.
7. The official selection retains all 541 class-1 rows, has no duplicate
   sample, preserves source coverage above random `q05`, and does not create a
   fold concentration outside the random `q05..q95` envelope.
8. Restricted-FP coverage and mean self-information exceed random `q95` for
   clean and at least three of four linked conditions, for both scratch and
   keeper probability sources.
9. Test use and raw-data modification are exactly false.

Any failure closes direct IP-DPP and the local `k=541` adaptation before
trainer integration. Do not repair source extraction, replace MCMC, tune k,
choose a lucky seed, or promote a proposal-bank maximum after reading A0.
