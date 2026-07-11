# TRKH 5-Class Deep-TEN Stem-Texture Audit

Date: 2026-07-12

Status: rejected before core-model image smoke. The current-best checkpoint and
full-train command remain unchanged.

## Question

Can a learnable residual texture dictionary over the keeper's CNN stem recover
interior mango-surface evidence that the existing CNN/Transformer classifier
misses, especially for class 1, without sacrificing direct keeper performance?

This is narrower than adding a generic texture expert. It tests the official
Deep-TEN encoding operation on a frozen, spatial stem representation before any
end-to-end architecture change.

## Primary references

- Deep TEN, CVPR 2017:
  <https://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Deep_TEN_Texture_CVPR_2017_paper.html>
- Deep Texture Encoding Network for material recognition, official code:
  <https://github.com/zhanghang1989/PyTorch-Encoding>, inspected at commit
  `ac748410dfc8d7d70a2ce7f5add08050af2fae20`.
- DEP, CVPR 2018:
  <https://openaccess.thecvf.com/content_cvpr_2018/html/Xue_Deep_Texture_Manifold_CVPR_2018_paper.html>

The implementation follows the official residual assignment form: learn K
codewords and per-codeword scales, soft-assign each local descriptor by scaled
squared distance, then aggregate assignment-weighted residuals.

## Locked protocol

- Frozen current keeper:
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt`.
- Keeper SHA-256:
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Data: `yolo_f`, all `9,215/2,606` train/validation object rows and
  `8,064/2,577` source groups; zero train/validation or fold fit/hold source
  overlap. Test is closed.
- Representation: final CNN stem activation, adaptive `12x12` grid,
  per-location LayerNorm, fixed seeded orthogonal projection `256 -> 32`.
- Spatial policy: transformed object bbox eroded by 12%, intersected with the
  valid-image mask, with a minimum nine-token nearest-valid fallback.
- Candidate: `K=8` masked Deep-TEN encoding, L2-normalized `8x32` vector,
  hidden width 64, zero-initialized no-bias keeper-log-probability residual.
- Matched control: masked mean plus standard deviation over the exact same
  projected stem grid, hidden width 64, and the same residual path.
- Training: five source-grouped folds, natural-frequency batches, AdamW
  `lr=1e-3`, `weight_decay=1e-4`, residual scale `0.12`, residual L2 `1e-3`,
  and 20 epochs. No class weights, test feedback, or validation sweep.
- The residual heads are OOF. The underlying keeper train predictions remain
  explicitly in-sample because the exact keeper has no fold checkpoints.

The first prefix preflight exposed an eight-token fallback. The fallback was
corrected and regression-tested before the full decision run. Both 4,096/1,024
prefix runs are class-order biased and were used only for protocol and runtime
checks; their compact history is retained inside the full artifact.

## Results

| Validation role | Macro F1 | Class-1 F1 | Class-1 precision | Class-1 recall |
| --- | ---: | ---: | ---: | ---: |
| Direct keeper | 0.884675 | 0.686047 | 0.611399 | 0.781457 |
| Mean/std control | 0.794480 | 0.363636 | 0.655172 | 0.251656 |
| Deep-TEN candidate | 0.779022 | 0.316327 | 0.688889 | 0.205298 |

Deep-TEN lost to the matched control in every source-grouped fold. Candidate
minus control macro gains were `-0.027159`, `-0.036628`, `-0.024318`,
`-0.013061`, and `-0.023050`; class-1 gains were all negative as well.

Against the direct keeper on validation, Deep-TEN changed 307 rows:

- 82 corrections, 199 harms, and 26 neutral changes;
- 62 class-1 false positives removed and one created;
- zero class-1 false negatives rescued and 87 true positives broken.

The representation did train rather than remaining collapsed. Codeword-use
entropy reached `0.896705`, assignment entropy `0.803272`, all scales stayed
nonpositive, and the minimum mask contained nine tokens. However, maximum
codeword cosine reached `0.995986`, showing near-collinear dictionary vectors,
and the final candidate residual absolute mean reached `8.154320`. The learned
action is therefore a strong class-1 suppressor, not a missing positive cue.

Candidate-minus-control class-1 direction AUROC was high and stable
(`0.848099/0.922566` OOF/validation), because the candidate suppresses false
positives more strongly than false negatives. It does not authorize a keeper
action: candidate-minus-keeper validation direction AUROC is only `0.446869`,
and true class-1 recall collapses.

## XAI review

The complete 12-case signed residual sheet was reviewed. Candidate positive
center mass fell from control `0.356843` to `0.278909`, while positive border
mass rose from `0.030891` to `0.040745`. Several keeper true-positive class-1
examples receive broad negative Deep-TEN evidence over the surface. The branch
also reacts to edges, stems, shadows, crop boundaries, and broad maturity color.

This agrees with the transition audit: global aggregated texture can remove
obvious class-1 false positives, but it cannot preserve subtle true class-1
surface evidence.

## Decision

Ten locked checks failed and `image_smoke_permission=false`. Do not implement a
core Deep-TEN branch or sweep codewords, grid size, projection rank, erosion,
hidden width, optimizer, residual scale, fold seed, or epochs. Do not build a
threshold/router/calibrator from the attractive control-relative AUROC; the
current keeper train probabilities are not true OOF and the direct keeper
direction is invalid.

Reopen learnable texture dictionaries only after a genuinely different local
representation or independent supervision produces a fixed source-safe action
that rescues keeper false negatives while preserving class-1 recall.

## Evidence and cleanup

- Full evidence:
  `runs\diagnostic_deepten_stem_texture_full_20260712`, 11 payloads,
  `5,671,831` bytes, payload manifest SHA-256
  `24ee35631bade5d70a1b9bca8714614d45eb81f1e2b983f6cf9ff4b852b46924`.
- It contains no model, checkpoint, feature cache, or test payload.
- Cleanup manifest:
  `runs\cleanup_manifest_20260712_deepten_preflights_superseded.json`.
  It removed exactly two preflight roots, 20 files, and `5,866,265` bytes;
  observed free-space gain was `5,902,336` bytes.
- Retention:
  `runs\artifact_retention_audit_after_deepten_stem_texture_cleanup_20260712`
  passed over 576 run directories with `blockers=[]`.
- Current-best command SHA-256 remains
  `3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.
