# TRKH 5-Class Foveal Aggregated Attention A1 Closure - 2026-07-16

## Decision

Close the exact block-1 Foveal Aggregated Attention (FAA) route at its locked
pre-training mechanism gate. The five-epoch control/candidate pair, Stage B,
official validation, test, full training, and current-best command promotion
are not authorized.

This is a mechanism rejection, not an implementation failure. The
implementation reproduced the official equation, passed numerical, gradient,
deployment, and resource checks, but did not satisfy the prospectively locked
requirement that local and pooled evidence jointly compete for nearly every
patch query.

## Primary-Source Provenance

- Accepted paper: Shi et al., "TransNeXt: Robust Foveal Visual Perception for
  Vision Transformers," CVPR 2024.
- Paper SHA-256: `0d1a5d07b747eb7c02ed66a840f2f0242252697d883077325862509ad28e1c5e`.
- Official Apache-2.0 repository: `https://github.com/DaiShiResearch/TransNeXt`.
- Locked commit/tree: `c8a99743b60ac94ac8d2bf66ffe164a440dcfe21` /
  `d1ce1c6cf62f5d0ebe1290dfe64b213328d91d6a`.
- Reviewed native attention SHA-256:
  `f70bc20818c9b0456d6eaaf58669561f0ede8fa5aba3c3a557cb5094557d86fe`.
- Immutable protocol SHA-256:
  `1676ef25fde2d90ca0786730d0dd91ab9bba551976111b19912da452759bfec8`.
- Committed and pushed implementation: `6ce3114d31dca005d99a7fa2f7ba029fab08e20c`.

Secondary reports and user-supplied research notes did not define the method,
equations, or gates.

## Train-Only Fold

The generated hardlink view at `runs/yolof_cidt_fold0_trainonly_20260716`
contains only the frozen `yolo_f/train` source-disjoint fold:

- fit: 7,372 rows, 6,452 sources, class counts `[1561,432,1527,2017,1835]`;
- holdout: 1,843 rows, 1,612 sources, class counts `[380,109,393,503,458]`;
- source overlap: zero;
- fit/holdout index SHAs:
  `22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d` /
  `a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae`;
- fold summary SHA-256:
  `2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763`;
- 19,352 files were hardlinked and `raw_data_modified=false`.

No official validation or test path was loaded.

## Preflight Result

Forty-one of 42 locked checks passed. The sole failure was
`dual_route_fraction`:

- patch queries assigning at least 0.05 mass to both routes: `0.599609375`;
- locked minimum: `0.95`;
- mean local/pooled mass: `0.5148347616/0.4851652086`;
- dense attention shape: `[1,8,263,263]`;
- maximum row-sum error: `2.3841858e-7`;
- patch-to-prefix nonzero count: `0`.

The average local/pooled split looks balanced, but it hides substantial
head/query-level one-route dominance. The locked hypothesis required
near-universal joint competition, not merely balanced aggregate mass. Lowering
the threshold after observing `0.5996` would be post-hoc gate rewriting.

All engineering evidence was otherwise positive:

- official equation output and input-gradient maximum errors were exactly zero;
- common control/candidate state was bit-exact;
- every candidate FP32 and CUDA BF16 gradient family was finite and nonzero;
- parameters were `7,245,590 -> 7,320,246` (`+74,656`);
- median runtime ratio was `1.09972084`;
- peak VRAM was `2.63687 GiB`, ratio `1.00522744`;
- static batch-1 ONNX replay passed with maximum logit error `1.4901161e-7`
  and matching argmax.

Preflight summary and manifest SHAs are
`0d15bfa383663384b4f09e251eefab0f734702a732de1554a3b2316367941135` and
`140a1173757fc3ca37a3894b7108b6b785cbff0d2549b681111bccd1d8198245`.

## Stop Boundary

- Do not run the formal five-epoch pair or reinterpret the preflight as a
  behavioral result. There are no candidate metrics, XAI pages, or claims about
  class-1 precision/F1 because training was correctly denied.
- Do not sweep FAA window/pool size, CPB width, temperature, local tokens,
  prefix policy, insertion layer, layer count, loss, LR, augmentation, epoch,
  fold, seed, threshold, router, or pretrained variant on this keeper.
- The reusable train-only hardlink fold remains valid infrastructure. Compact
  only reproducible rejected-run payloads after preserving summary, manifest,
  closure, hashes, and a cleanup manifest.
- Current-best checkpoint and full-pipeline commands remain unchanged at three
  revisions, two updates, and zero keeper replacements.

The next route must be distinct from static local/global competition and must
be screened from an accepted primary paper plus an official maintained codebase
before a prospective protocol is written.

## Evidence Retention

The rejected preflight was compacted to
`runs/evidence_faa_preflight_rejected_20260716`. Exact `summary.json` and
`artifact_manifest.json` payloads are retained with inventory hashes; the
30,789,271-byte reproducible ONNX payload was excluded and the original
preflight directory was deleted only after source/hash verification.

- compacted payload manifest SHA-256:
  `f866d2b082c022d75e8d0ac838ad52e2840e426d15713afb539b81b6a9831439`;
- compacted summary SHA-256:
  `1e86a453c30ea321c11cadf925c58d7a8c163c4dd2bb6317a0826728e3460d7e`;
- cleanup manifest SHA-256:
  `e72781821cfa9f08e6bd25428cbf91150c4867e494dea6671182a6294a20ff9c`;
- read-only retention audit passed over 712 directories with `blockers=[]`,
  no raw-data access/change, no test use, and 71.226 GB free;
- retention summary SHA-256:
  `7bb8b6dbe54e3b9451d2400b41fa386d6b8d0b7145968606c4c4ae0eaf1a0b7e`.
