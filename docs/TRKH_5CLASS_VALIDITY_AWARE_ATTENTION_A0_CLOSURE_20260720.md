# Validity-Aware Attention A0 Closure (2026-07-20)

## Decision

Reject inference-time key masking of fully synthetic square-padding patches in
all eight Transformer blocks of the current no-pretrain keeper. The sole
train-only A0 passes all 18 structural, provenance, fidelity, isolation, and
replay gates, but fails all 24 clean-quality, alignment, and robustness gates.
It does not authorize model/trainer integration, validation or test access, a
smoke, a probe, full training, or current-best command promotion.

This closes the exact attention-only intervention and its nearby mask variants
listed in the prospective protocol. It does not close validity-aware CNN stem
processing or partial convolution, which changes the feature-extraction
equation before tokenization and requires a separate locked protocol.

## Locked Evidence

- Protocol SHA-256: `2b986af2...96df`; protocol/implementation/telemetry
  commits: `04f0101`/`9ba6e42`/`f769bb4`.
- Accepted NaViT paper SHA-256: `d4421cab...a396`.
- PyTorch 2.6 attention source/license SHA-256:
  `ed64d867...d67`/`47a26beb...4165`.
- Auditor/test SHA-256: `272c875f...632`/`7732c42c...fb92`.
- Keeper SHA-256: `1f49d577...2677`.
- Current-best command/history SHA-256:
  `36b9aa1a...0faf`/`39bd2879...8f53`.

The formal uses all 9,215 immutable `yolo_f/train` object rows and the five
fixed source-disjoint CIDT folds. Validation and test pixels, labels,
predictions, and metrics remain unopened. Raw-data and model-state hashes are
unchanged, the raw keeper replay differs from CIDT by at most
`4.470348e-08`, and the all-valid wrapper is exactly equivalent to the raw
keeper in logits and probabilities.

The dataset contains 9,204 padded rows and 11 square rows. There are 555,088
fully invalid 16x16-grid patches, averaging `60.237439/256` per row; no row is
fully masked. The intervention is therefore nontrivial and is not rejected for
lack of padding exposure.

## Clean Result

| Role | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 | Predicted class 1 |
|---|---:|---:|---:|---:|---:|
| Keeper control | 0.939876 | 0.700265 | 0.975970 | 0.815444 | 754 |
| Real validity mask | 0.933896 | 0.691057 | 0.942699 | 0.797498 | 738 |
| Same-class/fold source-placebo | 0.935056 | 0.691375 | 0.948244 | 0.799688 | 742 |

Candidate-minus-keeper macro/class-1 F1 is
`-0.005980/-0.017946`; class-1 precision/recall is
`-0.009208/-0.033272`. It makes 33 corrections and 69 harms, removes/creates
22/25 restricted `{0,2,4}->1` false positives, rescues one class-1 false
negative, and breaks 19 class-1 true positives. The class-1 direction AUROC is
`0.469018`, slightly below the source-placebo `0.469451`.

All five source folds lose macro and class-1 F1. Fold class-1-F1 deltas are
`-0.012553`, `-0.017483`, `-0.021997`, `-0.013104`, and `-0.024568`;
class-1 precision improves in only one fold and by only `+0.002549`. This is
not an aggregate result hiding a stable subgroup benefit.

## Robustness And XAI Diagnosis

Under dim light, the candidate gains `+0.008899/+0.014638` macro/class-1 F1,
but class-1 precision is still `-0.000692` and restricted FP worsen by 20 net.
Under bright light, precision improves `+0.008966` and 28 net restricted FP are
removed, but class-1 recall/F1 fall `-0.033272/-0.004765`. Under low contrast,
macro/class-1 F1 and precision fall
`-0.019633/-0.025627/-0.034697`, with 25 net new restricted FP and 238 harms.
The aggregate intervention is neither precision-selective nor robust.

Telemetry proves the implementation mechanism works. Mean clean class-query
attention assigned to invalid keys before masking is `0.061469` in block 1 and
`0.097678` in block 2; every block has maximum post-mask invalid attention
mass `0.0`. Token pruning reduces mean retained invalid keys from `60.237` in
blocks 1-2 to `28.452` in blocks 3-5 and `3.424` in blocks 6-8.

Manual review of `fixed_validity_attention_mechanism_sheet.png` confirms that
the red validity overlay aligns with synthetic side/top/bottom padding and the
absolute-change maps are concentrated along those invalid bands. The valid
attention pattern is then renormalized rather than becoming more selective for
class-1 lesion morphology. Some unpadded examples have no visible change, as
expected. The failure is therefore a frozen-representation/selectivity result,
not a mask-axis, QKV, pruning-index, or visualization defect.

Do not sweep mask thresholds, soft/query masks, layer subsets, padding fills,
class-specific masks, dilation, temperature, routing, blending, folds, seeds,
or training the same attention-only mask on this keeper. A new route must alter
pre-token CNN feature formation or supply independently sourced new evidence.

## Runtime, Interruption, And Retention

The successful formal takes `795.404 s` with requested/effective Windows
workers `4/4`, persistent workers, prefetch 2, and an RTX 4060 Laptop GPU. A
prior launch was manually stopped before any output, metric, or artifact write
because buffered progress obscured its actual runtime. The only subsequent
optimization computes the already-locked class-query telemetry row instead of
a redundant full second attention matrix; a focused exactness test proves the
same telemetry equation, and candidate logits are unchanged.

Independent replay reconstructs probabilities, analysis, telemetry, and all
42 gates with maximum difference `0.0`. Preserve the seven formal payloads,
totaling 13,364,010 bytes. Summary/manifest/probability/telemetry/prediction/
sheet/replay SHA-256 values are respectively
`1f28cccc...0a09`/`95e762d0...416c`/`7c1d44e3...afb8`/
`a967c0f7...778d`/`e27051c4...ab1`/`50468f92...e4f7`/
`78c280ca...6914`.

Pycompile and pyflakes pass, focused tests pass `7/7`, and the complete
repository suite passes `1578/1578` with 276 non-failing warnings. The
post-closure read-only retention audit covers 786 run directories and all 50
valid compaction manifests. All 219 manifest-derived originals remain absent,
`deleted_anything=false`, and `blockers=[]`; its summary SHA-256 is
`03f66d1e...e389b`. Current-best commands remain unchanged.
