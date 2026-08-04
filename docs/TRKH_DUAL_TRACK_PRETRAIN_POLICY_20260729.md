# TRKH dual-track pretraining policy

Effective date: 2026-07-29
Decision owner: user
Clean base: `fdf9e40a3292813380e667c959e59bc72c55bf39`

## Decision

TRKH now has two independent scientific tracks:

| Track | Git location | External pretrained weights | Scientific identity |
|---|---|---:|---|
| No-pretrain | `classification-only-research` in `D:\DataAI\AIEx\TRKH` | Forbidden in the classifier, teacher, router and inference graph | Preserves the original scratch contribution and all existing claims |
| Pretrained | `research/pretrained-hybrid-v1` in `D:\DataAI\AIEx\TRKH_pretrained` | Allowed when declared and justified | Tests whether modern pretrained representations plus TRKH-specific reasoning improve the current class-1 boundary |

Active-branch amendment (2026-08-04): canonical `class_f` work continues on
`research/pretrained-classf-b1`, a direct descendant of
`research/pretrained-hybrid-v1` at commit `73c96f3d8f42e80c62ab2c4e3c0691ff81f45b77`.
`TRKH_pretrained` is a linked Git worktree whose common directory is
`D:\DataAI\AIEx\TRKH\.git`; it is not an independent copied project. The branch
rename narrows dataset/protocol scope and does not merge or overwrite the dirty
no-pretrain worktree.

The existing no-pretrain worktree, including its uncommitted files, must not be switched, reset, copied wholesale, or silently mixed into the pretrained track. Shared fixes may be ported only as reviewed, path-specific changes with their origin recorded.

As of 2026-07-31, `D:\DataAI\AIEx\newdataset\class_f\data.yaml` is the only
canonical dataset for new experiments in both tracks. Results trained on
`yolo_f` remain reproducible historical evidence, but their checkpoints,
teacher caches, thresholds and scores are incompatible with canonical model
selection because the target semantics changed materially.

## What is allowed on the pretrained track

- Load external weights into a ViT, CNN, feature encoder, teacher, fusion component, or another inference-time component when it addresses a measured evidence gap.
- Fine-tune, partially fine-tune, freeze, or use discriminative layer-wise learning rates according to a prospectively recorded experiment.
- Retain useful TRKH components such as bbox/valid-mask handling, local-detail evidence, pairwise class-1 boundaries, audit traces, XAI, and deployment tooling.
- Add a new branch only when a matched ablation can isolate its contribution and the result fits the RTX 4060 Laptop 8 GB deployment budget.

This permission is not an instruction to load every available model or to repeat a closed frozen-feature/readout experiment. Historical failures remain valid for the exact representation, frozen/short-probe recipe, split, and decision rule that was tested.

## Required provenance

Every pretrained run must record:

- track name, branch, worktree, Git commit, dirty status, and parent commit;
- provider, model ID, exact revision/commit, source URL, license, weight-file SHA-256, architecture and parameter count;
- input size, normalization, interpolation, crop policy, precision, batch geometry, and any register/prefix-token convention;
- which modules loaded external weights, which started randomly, which were frozen, and which were optimized;
- exact data YAML, source-group policy, seed, sampler exposure, optimizer/scheduler, checkpoint selector, and full command;
- validation, robustness, XAI, latency/p95, throughput, VRAM, ONNX parity, and TensorRT feasibility results.

If any required field is unknown, the run is exploratory and cannot become the reported best model.

## Isolation and promotion

- Use pretrained-specific run names, command files, evidence directories, and latest pointers. Never overwrite the scratch current-best command, checkpoint, or pointer.
- Tag every checkpoint and exported package with `research_track=pretrained` or `research_track=no_pretrain`; loading must fail closed on an incompatible declared track.
- Rebuild scratch, pretrained-backbone, MobileNetV3 and AIDT controls on the
  same full 2,479-sample canonical validation split. Old 2,606-row comparisons
  must be labelled historical and never mixed into the new ranking.
- A pretrained candidate must beat a matched pretrained-backbone control, not merely beat the scratch model. Report absolute metrics and deltas.
- Keep the historical project test retrospective and inaccessible during design, thresholding, model selection, and ablation. A new confirmatory claim still requires a prospectively sealed source/time/site holdout.
- Do not promote from a smoke, partial validation, oracle, ensemble weight sweep, or test-tuned result.

## Initial scientific objective

The immediate target is the unresolved class-1 boundary: preserve true class-1 recall while reducing `0/2/4 -> 1` false positives under illumination and surface variation. The first candidate family should use one strong modern pretrained representation as the primary path and add only a lightweight TRKH-specific residual/specialist whose value is measured against:

1. the same pretrained backbone alone;
2. the same backbone with a capacity-matched generic head;
3. the pretrained backbone plus the TRKH specialist;
4. the retained scratch and AIDT references.

The first bounded run remains smoke/probe only. Full train is permitted only after full-validation, class-1 TP/FP, robustness, XAI, resource, and export gates pass.

Mobile deployment is a separate Pareto objective: report full validation
quality together with parameters, package size, latency p50/p95, throughput,
peak memory and quantized parity. DINOv3 B0 is a teacher/control, not a mobile
claim. Promote a mobile student only after absolute quality and teacher-to-
student degradation gates pass on `class_f`.

## Scope of older rules

Any older statement saying pretrained models are “comparison-only”, “teacher-only”, or cannot enter the candidate is superseded for this pretrained branch as of 2026-07-29. It remains binding on the no-pretrain lineage. Exact no-repeat conclusions for already tested recipes remain binding until a materially different representation, optimization regime, or causal hypothesis is documented.
