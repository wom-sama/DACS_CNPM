# TRKH pretrained class_f B2 tempered-p05 protocol — 2026-07-31

## Hypothesis and single delta

B0 strict-balanced gives class 1 too much prior exposure: recall is high but
false positives, especially class `2 -> 1`, remain large. B1 natural sampling
reverses the error too strongly: precision rises while class-1 recall collapses.

B2 keeps the complete B0 DINOv3 model, initialization, LDAM-Focal loss
(`margin=0.3`), augmentation, optimizer, EMA, seed and evaluation contract.
Its only semantic delta is the train-only class prior

`q_c = n_c^0.5 / sum_j(n_j^0.5)`.

For canonical train counts `[1987, 497, 1326, 2080, 2388]`, batch size 24 and
the 120-batch probe, deterministic exposure is
`[649, 325, 530, 664, 712]`. Class 1 therefore receives 56.4% of B0 exposure
and 1.88 times its natural-prior exposure. Cumulative quota error must stay
strictly below one sample per class at every batch boundary.

The sampler is absent from inference, so it adds zero parameters, FLOPs,
latency and mobile memory.

## Frozen probe and promotion gate

- Canonical `class_f` fingerprint, ordered five-class schema and test lock must
  pass; no old-dataset checkpoint, teacher, cache or router is allowed.
- One seed only: `42`; five epochs; 120 train batches per epoch; full val.
- No sweep of the power value is permitted after observing validation.
- Probe must have no non-finite step and must satisfy all:
  - accuracy `>= 0.8760`;
  - macro-F1 `>= 0.8350`;
  - class-1 precision `>= 0.58`;
  - class-1 recall `>= 0.70`;
  - class-1 F1 `>= 0.66`;
  - confusion `class 2 -> class 1 <= 61` (at least 20% below B0's 76).

Only a passing probe is eligible for the matched 30-epoch full train and the
final XAI, robustness, calibration and mobile audits. A failure closes this
sampler hypothesis; it does not authorize another validation-guided power
sweep.
