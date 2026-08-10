# TRKH SaSPA F1 XAI execution lock

This lock fixes the final diagnostic XAI step for the ten SaSPA F1 outputs.
Fidelity, blind-review, and duplicate decisions already exist and cannot be
changed by this audit.

The frozen TRKH keeper is run in FP32. For every output, the clean top-1 logit
is the objective for both stem Grad-CAM and the input-gradient map. The exact
normalized source bbox stored in generation provenance is passed as the model
bbox prior and defines the perturbation mask:

- background gray and Gaussian blur affect only pixels outside the bbox;
- object desaturation affects only pixels inside the bbox;
- all ten outputs are processed in lexicographic `output_id` order.

The audit may describe whether the keeper responds to fruit, surface,
boundaries, or synthetic artifacts. It cannot filter or regenerate outputs,
rank prompts or seeds, tune a threshold, authorize A1/training, or update the
current-best command. A visually clean heatmap is not evidence that the
synthetic method improves classification.

The machine-readable authority is
`docs/TRKH_5CLASS_SASPA_F1_XAI_EXECUTION_LOCK_20260724.json`.
