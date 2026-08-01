# Pretrained metric comparability — 2026-08-01

## Current canonical `class_f` validation

All rows below use the current 2,479-image validation split and independent
FP32 reload. Test is locked.

| Model/run | Accuracy | Macro-F1 | Class-1 P/R/F1 | Status |
|---|---:|---:|---:|---|
| DINOv3 B0, strict-balanced | 0.906011 | 0.868593 | 0.610837 / 0.784810 / 0.686981 | matched control |
| DINOv3 B9, tempered p=0.5 | 0.912061 | 0.876324 | 0.642105 / 0.772152 / 0.701149 | current canonical reference |
| old AIDT checkpoint, direct transfer only | 0.819282 | 0.764197 | 0.437500 / 0.443038 / 0.440252 | diagnostic, not fair retrain |
| old MobileNetV3 checkpoint, direct transfer only | 0.811618 | 0.743564 | 0.389706 / 0.335443 / 0.360544 | diagnostic, not fair retrain |

B9 improves over B0, but misses the current class-1 milestones `0.72` and
`0.75`. A B0/B9 oracle reaches class-1 F1 `0.727273`; this only proves useful
complementary errors exist and is not a deployable score.

## Historical old `yolo_f` validation

These values used the old 2,606-row validation lineage. They are retained for
historical context and must not be subtracted directly from canonical scores.

| Historical model | Accuracy | Macro-F1 | Class-1 P/R/F1 |
|---|---:|---:|---:|
| AIDT ResNet50 + ViT-B/16 | 0.944743 | 0.910267 | 0.655738 / 0.794702 / 0.718563 |
| MobileNetV3-Large | 0.940522 | 0.904089 | 0.694268 / 0.721854 / 0.707792 |
| TRKH scratch keeper | 0.919033 | 0.882925 | 0.603093 / 0.774834 / 0.678261 |

## Why the pretrained number can look lower

- Canonical migration changed 1,924 of 12,019 targets; old and canonical
  class-1 membership agrees on only 299 of 738 affected candidates.
- Old AIDT/Mobile ImageFolder order was alphabetic. Research class 1 was old
  source index 3; canonical-to-old mapping is `[4,3,1,0,2]`.
- A pretrained backbone still receives a newly initialized five-class head.
  Low first-epoch class-1 F1 is head warmup, not evidence that DINO weights
  failed to load.
- B0/B9 preload, optimizer, scheduler, EMA and update telemetry are valid.
  The remaining failure is mainly class-1 boundary sensitivity to foreground
  luminance/color, especially bright `0 -> 1` false positives.

Therefore the fair next comparison is to retrain AIDT and MobileNetV3 on the
same canonical train/validation contract. Direct transfer and old `yolo_f`
scores are diagnostics only. In parallel, PRMR-R1 tests a train-only correction
for the measured luminance-sensitive class-1 boundary.
