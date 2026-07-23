# TRKH SaSPA F1 Implementation Lock

Date: 2026-07-24

State: prospective, before any F1 synthetic output

## Scope

This lock makes the revision-2 SaSPA A0 protocol executable without changing
its sources, prompts, seeds, generator, inference settings, resource limits,
fidelity intent, duplicate thresholds, or downstream authorization.

F0 passed and replayed at summary/manifest SHAs
`cf9046c9...e257a` / `682f2c8b...aa712`. F1 remains limited to ten train-only
outputs, two per class. No validation/test metric, A1 pool, model training,
full train, or current-best command update is authorized.

## Generation

- Generate in ascending class/rank order in one isolated worker.
- Use the pinned BLIP-Diffusion ControlNet snapshot, FP16, model CPU offload,
  attention slicing, and VAE slicing. Sequential CPU offload is the only
  one-time fallback and is allowed only before the first synthetic file.
- Reproduce the official SaSPA aspect-preserving resize and Canny 120/200
  preprocessing, then let the pinned Diffusers processor resize the control to
  512x512.
- Use one fresh CPU `torch.Generator` per output with the locked seed.
- Save deterministic RGB 512x512 PNG files. Never regenerate after inspection.
- Preserve worker peak RAM/VRAM, latency, finiteness, process, and provenance
  telemetry for every output.

## Fidelity

Before generation, select 256 unique non-selected eligible train groups per
class by the locked SHA-256 ranking. Fit class-specific 95th-percentile
(`higher`) thresholds for:

1. LAB histogram square-root Jensen-Shannon distance.
2. Circular hue-histogram Wasserstein-1 distance.
3. Radius-1 8-neighbor LBP histogram chi-square distance.
4. Symmetric Canny edge-density ratio.

Every generated bbox crop must pass every threshold against its locked
source-disjoint subject crop. Thresholds are train-only and cannot be changed
after output.

## Blind And Leakage Audits

- Blind review uses deterministic A-J ordering. The reviewer sees only the
  generated image and class definitions; target/source/prompt/seed are hidden
  until all decisions are hash-locked.
- After all ten outputs are final, an isolated process scans every locked
  train/validation/test scope for decoded-RGB equality, 64-bit pHash distance,
  and pinned DINOv2-small cosine similarity.
- The process persists only nearest path/scope/distance/hash and pass/fail. It
  never persists reference embeddings, labels, images, or split aggregates.
- A duplicate rejection never triggers regeneration.

## XAI Boundary

Paired and blind contact sheets are required. Frozen-keeper Grad-CAM, input
gradient, background perturbation, and object-desaturation views are produced
for all ten outputs only after all F1 decisions are fixed. XAI is diagnostic:
it cannot filter outputs, tune generation, or authorize A1.

## Failure Policy

All gates are conjunctive. Preserve fail-closed artifacts on any process,
resource, finiteness, provenance, fidelity, review, diversity, leakage, or raw
mutation failure. Validation/test content is forbidden everywhere except the
isolated duplicate exclusion process.
