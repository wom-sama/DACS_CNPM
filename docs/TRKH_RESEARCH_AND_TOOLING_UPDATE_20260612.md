# TRKH Research And Tooling Update - 2026-06-12

## Why V9 Is Surface-Detail First

V8 improved TRKH only slightly. XAI showed background blur/gray barely changes
predictions, while object desaturation changes them strongly. The next change
therefore should not be stronger global background suppression. It should make
the auxiliary crop/drop views focus on fruit surface texture/color details.

Relevant sources checked:

- WS-DAN: https://arxiv.org/abs/1901.09891
  - Uses attention-guided crop/drop for fine-grained classification.
  - Important warning for this project: the crop/drop signal is only useful if
    the attention map points to discriminative object regions.
- PMG: https://arxiv.org/abs/2003.03836
  - Fine-grained differences benefit from multi-granularity/local patch learning.
  - This supports adding surface-detail score and possibly later multi-scale
    patch branches, rather than only a global class head.
- ELR: https://proceedings.neurips.cc/paper_files/paper/2020/hash/ea89621bee7c88b2c5be6681c8ef4906-Abstract.html
  - Useful when label noise is confirmed.
  - Not enabled yet because class-1 early predictions are biased toward adjacent
    classes; use warm-up and label audit first.
- Background masking audit paper:
  https://openaccess.thecvf.com/content/ICCV2023W/OODCV/papers/Aniraj_Masking_Strategies_for_Background_Bias_Removal_in_Computer_Vision_Models_ICCVW_2023_paper.pdf
  - Background bias is real in FGVC, but our robustness probes show current TRKH
    errors are dominated by fruit color/texture and label boundaries.

## Token/Context Tooling

Installed:

- Repomix `1.14.1`
- Command: `C:\Users\ADMIN\AppData\Roaming\npm\repomix.cmd`
- Config: `repomix.config.json`

Usage for future context snapshots:

```powershell
C:\Users\ADMIN\AppData\Roaming\npm\repomix.cmd --config repomix.config.json --compress --token-count-tree 1000
```

Not installed:

- 9Router: useful for local provider routing, fallback and tool-output token
  saving, but it opens a dashboard/proxy and needs provider setup. It is not
  necessary for local TRKH train/evaluate.
- ConnectOnion: useful for building separate agents with context compression and
  subagents, but it does not plug directly into this Codex session or PyTorch
  pipeline.

References checked:

- 9Router: https://github.com/decolua/9router
- ConnectOnion: https://github.com/openonion/connectonion
- Repomix: https://github.com/yamadashy/repomix

## Operational Change

Full training is now gated by class-1 validation milestones. From v8, the next
full-train gate is class-1 validation F1 `>= 0.70` on a probe. If a change only
adds a tiny gain or only has smoke-level evidence, it is documented and not sent
to full train.

V9 result: `surface_detail` attention views passed smoke/trace, but the 80-batch
probe reached only validation class-1 F1 `0.6783`, below the `0.70` gate. It is
kept as an ablation and should not be sent to full 30-epoch training without a
stronger change.
