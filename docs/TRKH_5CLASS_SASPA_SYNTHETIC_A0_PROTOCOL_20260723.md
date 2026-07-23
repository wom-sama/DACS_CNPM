# TRKH SaSPA Dual-View Synthetic A0 Protocol

Date: 2026-07-23
State: prospective and pre-generation
Canonical specification:
`docs/TRKH_5CLASS_SASPA_SYNTHETIC_A0_PROTOCOL_20260723.json`

## Decision

Select one narrowly scoped SaSPA-inspired feasibility route. Use a full
`yolo_f/train` frame as the Canny edge/context condition and a same-class
`class_f/train` crop from a different `leakage_group` as the BLIP-Diffusion
subject reference. Generate only ten images, two per class, after a no-output
resource gate.

This A0 does not authorize trainer integration, validation/test metrics,
smoke/probe/full train, checkpoint promotion, or a current-best command update.
The raw datasets remain immutable.

## Why This Route

SaSPA directly targets fine-grained classification. Its official NeurIPS 2024
method combines:

1. a Canny edge map from one real image;
2. a subject reference from a different image of the same subclass;
3. a class-aware prompt; and
4. BLIP-Diffusion with ControlNet.

That structure matches the useful properties of the two accepted TRKH views:
`yolo_f` retains object position and broad context, while `class_f` provides a
clean subject crop. The reference and edge inputs must come from different
source groups so the candidate is not a disguised copy.

The SaSPA paper reports about 10 GB peak VRAM for BLIP-Diffusion ControlNet.
The local RTX 4060 has only 8,188 MiB. Therefore, the official repository's
direct `.to("cuda")` path is forbidden. The only prospective runtime ladder is:

1. `enable_model_cpu_offload`;
2. one restart with `enable_sequential_cpu_offload`, only if the first mode
   fails before any output is saved.

No third mode or post-output resource tuning is allowed.

## Prospective Runtime Correction

Revision 2 replaces only the pre-F0 runtime. A first isolated engineering
import showed that `diffusers==0.32.2` reaches a quantizer path referencing
`torch.float8_e4m3fn`, which does not exist in the initially recorded
`torch==2.0.1+cu118`. Inheriting global site packages also exposed unrelated
dependency conflicts.

The corrected runtime is a fully isolated Python 3.11.9 venv with its own
`torch==2.6.0+cu124` and `torchvision==0.21.0+cu124`. It passed `pip check`,
CUDA discovery, and import of `BlipDiffusionControlNetPipeline`. It does not
inherit from or modify `D:\DataAI\.venv`.

This correction occurred before model download, pipeline construction, source
selection, validation/test access, or synthetic-pixel generation. It does not
change the dataset hashes, generator/model revision, prompts, seeds, ten-output
cohort, resource/fidelity/leakage gates, or downstream policy.

## External-Evidence Screen

### SaSPA: selected

- Paper: NeurIPS 2024, official PDF SHA-256
  `4bf2b451fe720882bc3c0e7d7a80fa87e3fc53406832063cd08c3ed4ded3cfb7`.
- Repository: `EyalMichaeli/SaSPA-Aug`, commit
  `054230411863e8152ce475d48c1869e600a60f0c`, MIT license.
- Generator: `Salesforce/blipdiffusion-controlnet`, revision
  `e9e2aafc154c6a9d1593c8e4fb94c6a9a8a68593`, Apache-2.0,
  remote snapshot 7,688,436,558 bytes.
- The generator is a pretrained external prior and must be disclosed as such.

### DiffuseMix: not selected

The official CVPR 2024 code uses InstructPix2Pix and an external fractal-image
dataset. Its public repository at commit
`e58418d15d2ea5179dbedc2fcac80fe393dc6ec0` has no explicit license file.
Its autumn, sunset, rainbow, aurora, and artistic prompts can also change the
color signal that defines TRKH ripeness classes. Do not copy or deploy this
implementation in A0.

### BOB: deferred

BOB contributes the useful principle that pose/background attributes should be
class-agnostic to reduce class-context shortcuts. Its CVPR 2026 route fine-tunes
a text-to-image model for 400 epochs with batch size 80 and is aimed primarily
at low-shot generation. That training route is outside the current machine and
A0 budget.

## Immutable Data Boundary

The canonical inputs are:

- `yolo_f/data.yaml` SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`;
- `yolo_f/canbang.yaml` SHA-256
  `05808224579a5f73a31f8655056a9a4727f9185076019f7175242e374f0b5227`;
- `yolo_f/manifest.csv` SHA-256
  `eb16e09cd20fff8480b50aa79e7d5fec779e4614398911e2e257ba6a37642ff2`.

Only train rows may supply pixels, labels, prompts, source selection, reference
statistics, or filtering decisions. Validation/test labels, prompts,
embeddings, statistics, checkpoints, and metrics are prohibited.

Validation/test pixels may be opened once by an isolated final
near-duplicate audit. That process has fixed thresholds and may emit only
pass/fail, nearest path, distance, and hash. Its embeddings and images can
never flow back into generation or training. No rejected output is regenerated.

## Locked Tiny Cohort

Generate two outputs for each of the five classes, ten total.

An edge source is eligible only when:

- it is a `yolo_f/train` row;
- its source image contains exactly one object;
- normalized bbox area is within `[0.08, 0.70]`;
- normalized bbox border margin is at least `0.02`; and
- the matching `class_f/train/<class>/<stem>_box000.jpg` exists.

The edge source and subject reference must have the same class but different
image stems and different `leakage_group` values. No leakage group can be
reused anywhere in the ten-output cohort.

Selection is deterministic SHA-256 ranking defined in the canonical JSON.
Prompt assignment and per-output seeds are also derived from hashes. Manual
source cherry-picking, replacement, or regeneration is forbidden.

## Locked Generation

- Pipeline: `BlipDiffusionControlNetPipeline`.
- Resolution: `512 x 512`.
- Precision: FP16.
- Steps: `30`.
- Guidance scale: `7.5`.
- Canny thresholds: `120/200`.
- Source and target category: `mango`.
- Attention slicing: `auto`.
- VAE slicing: enabled.
- Negative prompt: the exact SaSPA quality prompt recorded in the JSON.

Two class-agnostic context templates are reused across every class. Only the
English subclass descriptor changes. This prevents context wording from
becoming a class label.

## F0: No-Output Gate

F0 may install the isolated environment, download the pinned model, build the
pipeline, install offload hooks, and select source rows. It must not call the
pipeline or produce synthetic pixels.

Minimum resources:

- 25 GiB free on `D:`;
- 4 GiB free physical RAM;
- 15 GiB free virtual memory;
- model snapshot no larger than 8 GiB;
- host virtual memory below 90%;
- pipeline load within 20 minutes; and
- no unknown Python/TensorRT GPU workflow.

Package versions, model files, source rows, and hashes must be written to the
run manifest before F1.

## F1: Tiny-Output Gate

The first locked output is the resource probe. Primary limits are:

- Torch peak allocated memory at most 6.5 GiB;
- Torch peak reserved memory at most 7.0 GiB;
- total NVML GPU memory at most 7.5 GiB;
- host virtual memory below 90%;
- first image within 180 seconds;
- ten-image estimate within 30 minutes; and
- no CUDA OOM or non-finite tensor.

If primary model offload fails before saving an image, F1 may restart once with
sequential CPU offload. A failure after an image is saved rejects A0.

## Fidelity And Leakage Gates

All gates are conjunctive.

Train-only reference distributions use 256 deterministic same-class pairs per
class and fix the 95th-percentile limits for:

- Lab color-histogram Jensen-Shannon distance;
- circular HSV hue-histogram distance;
- local-binary-pattern texture distance; and
- edge-density ratio.

Each generated object crop must stay within all class-specific train limits.

For blind review, output IDs are randomized and target labels are hidden. The
reviewer assigns one of five classes. At least 9/10 labels must be correct,
both class-1 outputs must be correct, no class-1 output may contain a severe
artifact, and at most one severe artifact is allowed overall.

The final immutable outputs are compared against all `class_f` and `yolo_f`
splits plus each other:

- any decoded RGB SHA-256 match fails;
- pHash Hamming distance `<= 4` fails;
- DINOv2-small cosine similarity `>= 0.995` fails.

The DINOv2 audit model is pinned to
`facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056`
and is used only for copy detection.

## If A0 Passes

The next step is not a full train. It is an A1 train-only synthetic pool:

- generate 5% of eligible rows independently within each class;
- keep the original class distribution and do not globally oversample class 1;
- use a fixed synthetic replacement probability of 0.10;
- compare against a real-only control with equal iterations; and
- lock generation, leakage, fidelity, and training recipes before opening
  validation.

Final test, full train, and current-best command promotion remain blocked until
later prospective gates explicitly authorize them.

## Failure Policy

Any mismatch, leak, raw-data mutation, resource failure, fidelity failure,
near-duplicate, missing provenance field, or manual-review failure rejects the
route. Do not sweep adjacent prompts, seeds, Canny thresholds, inference steps,
guidance, duplicate thresholds, or offload modes after observing outputs.
