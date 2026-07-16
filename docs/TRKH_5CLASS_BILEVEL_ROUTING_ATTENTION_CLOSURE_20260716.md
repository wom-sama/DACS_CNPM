# TRKH 5-Class Bi-Level Routing Attention A1 Closure - 2026-07-16

## Decision

Reject the exact block-2 BiFormer Bi-Level Routing Attention (BRA) A1 before
training. The locked preflight did not grant `formal_pair_permission`, and the
hash-locked postflight replay confirms that independent runtime, route-coverage,
clean-consistency, dim-stability, and visual failures remain after correcting
three reporting defects. Therefore no five-epoch pair, official validation,
test, full train, or current-best command update is authorized.

This closes the exact `S=4`, topk-16/topk-4, depthwise-LCE-5 block-2 route. It
does not claim that stock BiFormer is ineffective; it shows that this bounded
scratch adaptation is not selective or stable enough for the current TRKH
data and pruning geometry.

## Frozen Provenance

- Prospective protocol SHA-256:
  `e7ce7cf3ba44361f29760ed2b1ed1f06538db50b44613ce03c1373a35cc3ede3`.
- Implementation commit pushed before formal execution:
  `063d1c9d0edc7f3efa55e4115cc77820b7067f73`.
- Postflight auditor-fix commit pushed before replay:
  `6eb272e`.
- Accepted primary source: Zhu et al., CVPR 2023, "BiFormer: Vision
  Transformer with Bi-Level Routing Attention."
- Official MIT repository commit/tree:
  `1697bbbeafb8680524898f1dcaac10defd0604be` /
  `313af0f24b31141cdde68fd775e75105278f8e52`.
- Exact source-disjoint fold: 7,372 fit rows and 1,843 holdout rows with zero
  source overlap. Official validation and test paths were not loaded.

The user-provided research reports were hypothesis sources only. The accepted
paper and official licensed repository defined the mechanism and replay.

## Engineering Evidence

- Official equation/output/attention replay errors were
  `2.38e-7/7.45e-8`; dense-MHSA maximum parameter-gradient error was
  `4.77e-7`.
- Common checkpoint state was bit-exact. The candidate added only 6,656
  parameters and all required FP32/BF16 gradients were finite and nonzero.
- BF16 route stability passed at mean Jaccard `0.984375` and exact fraction
  `0.9609375`.
- Static batch-1 ONNX export passed with maximum error
  `2.682209e-7`, and contained the required `TopK` and `Gather` operators.
- Candidate/control train-step runtime ratio was `0.910171`; peak train VRAM
  ratio was `1.000426`, with candidate peak allocation `2.617923 GiB`.
- Exact native standard/trace parity is zero in both FP32 and BF16 when the
  trace uses the deployment-compatible contract
  `return_trace=True, return_attention=False`.

## Material Gate Failures

The formal summary recorded 64 of 75 automated checks as passing. The locked
postflight replay reclassified six false negatives, but five automated failures
and the separate visual failure remain.

1. Inference runtime was `0.077585` versus `0.055795` seconds, or
   `1.390526x`, above the locked `1.35x` limit.
2. Samples `3536` and `5386` had transformed boxes with no 16x16 token center.
   Only `1841/1843` rows had a valid object-query region, so all-row geometry
   and coverage gates fail.
3. On the 1,841 valid clean rows, mean foreground gain was positive
   (`+0.047487`) and far-background reduction was positive (`+0.040557`), but
   only `0.486692` of rows had positive foreground gain versus the required
   `0.55`.
4. Dim-versus-clean route Jaccard was `0.616128`, below `0.65`. Bright and
   low-contrast values were `0.677098/0.657587`; the route was therefore not
   stable across every locked condition.
5. Visual review failed all three pages as a promotion gate. Routes repeatedly
   included far background, hands, borders, and illumination-dependent areas;
   tiny edge samples `3579` and `5386` showed severe object-route mismatch.

The positive aggregate foreground movement is not sufficient because it is
inconsistent across clean samples and fails exactly on difficult edge geometry.
No threshold can repair the missing token-center query without changing the
prospectively locked method.

## Postflight Reporting Corrections

The original formal artifact is retained unchanged at SHA-256
`e8f0c54000d2f4ce05199cc3fc7ee2f59f1643598c36c627174bcb2ac80f8e0b`.
Its row CSV is retained unchanged at SHA-256
`86c1086ed872c50cfc4253ec1b3c70588021efbbd2228eb1bdea9094bcbf728e`.

The read-only replay, SHA-256
`5e1e3b84d1adf1008baa06a48a5d7804dbd80f0959ed05f7d2e65e2cc44ccc29`,
corrects the following auditor classifications without granting any permission:

- constructor RNG parity compares the causal topk-16/topk-4 roles, which are
  equal; the unrelated baseline-MHSA constructor was not part of this contract;
- trace parity must preserve normal pruning behavior, and then block output and
  logits are exactly equal in FP32 and BF16;
- finite-only means report the 1,841 valid geometry rows while separately
  preserving both invalid rows as hard coverage failures. This restores the
  clean/shift aggregate foreground and far-background checks, but does not
  restore positive-gain fraction or condition stability.

The replay is locked to the original summary and CSV hashes and hard-codes
pair, validation, and test permission to false. It cannot overturn the formal
decision.

## Evidence, Cleanup, And Verification

- Compact evidence root:
  `runs/evidence_bra_preflight_rejected_20260716`.
- Compact payload-manifest SHA-256:
  `5e19d8332f505aa01b0f2757ecac7f9eead1064a6ca408754c749d1e87f38773`.
- Cleanup-manifest SHA-256:
  `cbbc2d8d1e27fe606a7636487909b6ba43b205d4ac14782696b2c4ad9a3bcb57`.
- Visual-review SHA-256:
  `642859d20b521048cbd91cbb27b33a827ece7b6981685c8a92053d78a9a92315`.
- The reproducible ONNX was recorded at 32,678,307 bytes and SHA-256
  `f12e1e51a24ea7754ede99fdc02b42fdea1954e2172b93df83b4ac109db95878`,
  then excluded. Summary, CSV, replay, review, overlays, and manifests remain.
- Focused tests passed `18/18`; full pytest passed `1279/1279`; `pyflakes` and
  Python compilation passed.
- Read-only retention passed over 723 directories, with all 205 compacted
  originals absent, all protected artifacts present, and `blockers=[]`.
  Retention summary SHA-256 is
  `015777877ef0b8501f4aa74da3ff1d6463925139d3a721b5602a78c1c779afef`.

Current-best checkpoint, command, and history hashes remain respectively
`1f49d577...482677`, `36b9aa1a...40faf`, and `39bd2879...98f53`.
Command tracking remains three revisions, two updates, and zero keeper
replacements.

## No-Repeat Boundary

Do not sweep BRA top-k, region count, LCE kernel, insertion block, scale,
routing detach policy, prefix policy, optimizer, schedule, epoch count, fold,
seed, runtime threshold, bbox handling, or combinations with DAT, FAA, ViG,
Soft-MoE, or token-pruning changes on this keeper. Do not rerun the formal
preflight after correcting its reporting layer.

The default-off implementation remains as reproducible negative evidence. The
next candidate must come from a distinct accepted primary source and official
licensed code, must address the observed tiny-edge/pruning and surface-boundary
failure rather than only average background mass, and must receive a new
prospective protocol before implementation or training.
