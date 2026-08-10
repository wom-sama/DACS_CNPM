# TRKH SaSPA F1 finalization lock

This closure combines the already fixed SaSPA generation, fidelity, blind,
duplicate, XAI, and visual-review evidence. It performs no model inference and
decodes no image.

F1 passes only when every required stage passes. XAI is diagnostic and cannot
repair a failed fidelity or blind-review gate. The locked evidence has already
failed both gates, so the expected outcome is `Rejected`.

The exact SaSPA configuration is closed. It must not be rescued by changing
prompts, seeds, diffusion steps/guidance, fidelity thresholds, blind-review
limits, or synthetic ratio after observing the outcome. A1, any training, a
full train, and current-best command promotion remain unauthorized.

The machine-readable authority is
`docs/TRKH_5CLASS_SASPA_F1_FINALIZATION_LOCK_20260724.json`.
