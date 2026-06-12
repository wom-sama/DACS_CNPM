# TRKH Agent Rules - 2026-06-12

## Scope

Project: `D:\DataAI\AIEx\TRKH`

Dataset: `D:\DataAI\AIEx\newdataset\class_f`

Task: 5-class mango classification, no-pretrain TRKH first, with full auditability.

## Training Gate

No more full train just for tiny changes. A full 30-epoch run is allowed only
after:

- compile/import tests pass;
- focused unit tests pass;
- smoke run exits `0` and architecture trace completes;
- visual trace shows crop/drop/score on fruit surface, not padding/border;
- a probe or validation-only run reaches the next class-1 F1 milestone.

Milestones from v8 baseline:

- next: class-1 validation F1 `>= 0.70`;
- then: `0.75`, `0.80`, `0.85`, `0.90`.

If a change does not reach the next milestone, keep it as an ablation/audit and
do not spend time on full train unless the user explicitly overrides.

## Data Integrity

- Test split is audit only, never used for hard mining or model selection.
- Hard sample lists must be train-only.
- Do not globally oversample class 1. Use pairwise boundary handling and
  train-only hard negatives.
- If label ambiguity is suspected, create a review manifest and keep it separate
  from training until labels are corrected upstream.

## Current Evidence

- V8 is the best single TRKH checkpoint so far: test macro F1 `0.8869`, class 1
  F1 `0.6626`.
- Top-5 pretrained TTA is still stronger: macro F1 `0.9248`, class 1 F1
  `0.7786`.
- V9 `surface_detail` attention views passed smoke but did not pass the full
  train gate: best validation class 1 F1 `0.6783` `< 0.70`.
- XAI v8: background blur/gray has near-zero prediction impact; object
  desaturation has large impact. Model mostly depends on fruit color/texture,
  not background.
- Raw attention is noisy around border/background. Grad-rollout and stem Grad-CAM
  localize fruit surface better.

## Research Notes

- WS-DAN supports attention-guided crop/drop for fine-grained classification,
  but attention quality matters; noisy attention can crop/drop the wrong region.
- PMG supports multi-granularity learning for subtle local differences.
- ELR is relevant only after label audit or when noisy labels are clear; it
  should use warm-up because early class-1 predictions are biased toward
  neighboring classes.
- Background masking should not be intensified blindly because robustness probes
  show little background dependence.

## Tooling

- Repomix installed at `C:\Users\ADMIN\AppData\Roaming\npm\repomix.cmd`.
- 9Router and ConnectOnion were reviewed but not installed. They are useful for
  multi-provider routing/agent frameworks, not directly for the current local
  train/evaluate workflow.
- A personal Codex skill was installed at
  `C:\Users\ADMIN\.codex\skills\trkh-5class\SKILL.md` for future threads.
