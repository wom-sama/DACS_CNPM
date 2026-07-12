# TRKH 5-Class Research Journal

This journal records method attempts, audit conclusions, and cleanup decisions for the
`classification-only-research` branch. It is intentionally separate from the long audit
file so future work can quickly avoid repeating failed probes.

## Operating Rules

- Do not modify raw datasets under `D:\DataAI\AIEx\newdataset`.
- Test split is final audit only. Do not tune training, thresholds, routers, or
  ensembles on test.
- Every new method must have: code/test preflight, smoke, XAI/audit read, short probe
  only if smoke is sane, and a journal entry before moving on.
- Full 30-epoch train remains gated on validation class-1 F1 crossing the active
  milestone. Current milestone is `0.70`.
- Delete generated smoke/probe artifacts after their metrics and XAI conclusions are
  recorded, but preserve the best checkpoint, final test audits, AIDT comparison
  artifacts, and small manifests that explain cleanup.

## Current Keeper Artifacts

Date: 2026-07-02

- Best no-pretrain TRKH validation checkpoint:
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701`
  with validation macro/class-1 F1 `0.8847/0.6860`.
- Final no-pretrain TRKH test audit:
  `runs\eval_yolof_teacherfocusbinary015_best_test_final_20260702` with test
  macro/class-1 F1 `0.8865/0.6667`.
- AIDT pretrained comparison artifacts:
  `runs\aidt_classf_pretrained_val_20260702`,
  `runs\aidt_classf_pretrained_test_final_20260702`,
  `runs\aidt_yolof_mapped_test_metrics_20260702`, and
  `runs\aidt_classf_pretrained_train_20260702`.
- Frozen validation ensemble diagnostic:
  `runs\softensemble_trkh0775_aidt0225_val_20260702` reached validation
  macro/class-1 F1 `0.9139/0.7410`.
- Frozen final test ensemble audit:
  `runs\softensemble_trkh0775_aidt0225_test_final_20260702` reached test
  macro/class-1 F1 `0.8785/0.6494`, below the single TRKH test artifact, so it is
  not the deployable best.

## Cleanup Log

### 2026-07-02 obsolete smoke/probe cleanup

- D drive free space before cleanup was about `2.55 GB`.
- Created manifest:
  `runs\cleanup_manifest_20260702_obsolete_smoke_probe.json`.
- Deleted `175` generated obsolete run directories, totaling about `35.96 GB` by
  file-size estimate:
  - all `runs\smoke_*`;
  - all rejected `runs\probe_v8_yolof_*` except the best
    `probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701`;
  - `runs\yolof_oof_folds_train5_20260702`, because the OOF scratch schedule failed
    and the fold materialization can be regenerated from code if needed.
- D drive free space after cleanup was about `35.94 GB`.
- Keeper artifact existence was verified after deletion.

## Method Ledger

### Teacher-focus-binary route is still the best no-pretrain anchor

- Artifact:
  `probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701`.
- Result: validation macro/class-1 F1 `0.8847/0.6860`, class-1 P/R
  `0.6114/0.7815`.
- XAI: foreground focus is clean, background blur/gray perturbations are near zero,
  but border/near-tie/object-color flags remain.
- Decision: keep as current anchor; future methods should resume/compare against it.

### Wide-context and two-view data did not solve the class-1 boundary

- Tried direct `yolo_f` context, source-context feature fusion, source-context focus
  suppression, paired `class_f + yolo_f` sourceaux, static fusion, train-only router,
  and learned paired-view residual fusion.
- Representative paired fusion probe:
  `probe_v8_yolof_classf_pairfusion_headonly_bnfreeze_emafix_teacherfocusbinary015_120b_2e_20260702`
  reached validation macro/class-1 F1 `0.8837/0.6822`.
- Decision: do not repeat context-wide or paired-view probability/router variants with
  the same signal. If revisited, it needs a new train-safe signal such as real OOF
  base predictions or an uncertainty/cross-attention objective that changes
  representation, not just logits.

### Objectness, bbox, and foreground forcing are not the bottleneck by themselves

- Tried bbox prior, valid mask, boundary dropout, interior-boundary objectness, patch
  objectness, bbox-token context, object erasure, foreground-background mix, and
  surface/part consistency.
- XAI repeatedly showed foreground mass improves, but validation class-1 F1 does not
  cross `0.70`; several variants increase class-1 false positives.
- Decision: do not add another bbox/foreground regularizer unless it also introduces
  a new boundary/part representation signal and passes smoke XAI.

### Distillation and ensemble transfer are useful diagnostics but weak students

- Object-level weighted no-pretrain ensemble reached validation class-1 F1 around
  `0.6954`, but distilling it into the student lowered class-1 F1.
- Validation AIDT single is stronger (`0.9088/0.7169`), and AIDT+TRKH oracle/soft
  validation diagnostics show real complementarity.
- Global AIDT KD, AIDT focus-only, and AIDT rescue-margin probes did not transfer
  that advantage into the no-pretrain TRKH student.
- AIDT gated non-target/DKD probe
  `probe_v8_yolof_aidtnckd008_conf70_boundarydrop_bboxprior_120b_2e_20260702`
  also failed the gate: validation macro/class-1 F1 `0.8818/0.6763`, below the
  anchor `0.8847/0.6860`.
- Final test audit showed AIDT-mapped test `0.8575/0.5806` and frozen soft ensemble
  test `0.8785/0.6494`, both below single TRKH test `0.8865/0.6667`.
- Decision: AIDT explains why competitors can look stronger on validation
  (pretrained representation plus complementary errors), but it is not yet a safe
  deployable teacher. Do not repeat global/focus/rescue/DKD AIDT transfer with the
  current confidence and weight settings.

### Router, model soup, prototype, TTA, and OOF attempts are not current routes to gate

- Model soups and SWA did not improve class-1 F1 beyond the anchor.
- Frozen embedding centroid/kNN/logistic probes showed the current `yolo_f` head is
  better than simple prototype classifiers.
- Bbox-aware TTA reduced class-1 F1.
- Train-in-sample routers overfit badly; path-collapsed router evidence is invalid for
  multi-object `yolo_f`.
- OOF infrastructure is implemented, but scratch fold_00 2e/120b produced class-1 F1
  around `0.33`, so the current OOF schedule should not be expanded.
- Decision: router/stacking work needs real OOF predictions from a stronger legal
  fold training recipe, not the existing scratch schedule or train-in-sample outputs.

### SSL/MIM/reconstruction attempts did not create enough boundary signal

- Supervised MIM/RGB reconstruction, SSL-only VICReg warm-start, DINO short
  supervised-init, and surface-counterfactual consistency were tried.
- They generally kept foreground clean but failed validation class-1 gate and often
  damaged early class separation.
- Decision: do not repeat RGB-only reconstruction or SSL-only warm-start. If revisited,
  keep the supervised checkpoint as anchor and use a boundary/part/cross-view signal
  with smoke XAI before any probe.

### ELR on the current `yolo_f` anchor did not improve class 1

Date: 2026-07-02

- Rationale: older ELR V10 was run on an older V8/class_f route. Because current
  evidence points to ambiguous boundary labels, a very light ELR regularizer was
  tested on the current best `yolo_f` teacher-focus-binary anchor without modifying
  any raw data.
- Smoke:
  `runs\smoke_v8_yolof_elr003_teacherfocusbinary015_boundarydrop_bboxprior_20260702`.
  It resumed the current anchor, used ELR weight `0.03`, beta `0.70`, teacher
  focus-binary loss `0.015`, bbox boundary dropout, and pairwise routing. Compile,
  focused `IndexedSampleDataset` pytest, smoke train, and architecture trace passed.
  ELR state was `[9215, 5]` train-only and `train_elr_loss` was about `-0.2492`.
- Smoke XAI selected `12`: attention/Grad-CAM foreground mass `0.9760/0.9814`,
  background blur/gray prediction drop around `0`, object desaturate drop `0.0377`.
  Top confusions were still `3->2`, `0->1`, `1->0`, `0->4`, `4->1`.
- Probe:
  `runs\probe_v8_yolof_elr003_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702`.
  Full validation best epoch `2`: macro/class-1 F1 `0.8848/0.6856`, class-1 P/R
  `0.5990/0.8013`, confusion `0->1=53`, `1->0=14`, `1->2=11`, `2->1=12`,
  `4->1=12`.
- Probe XAI selected `20`: attention/grad-rollout/Grad-CAM foreground mass
  `0.9760/0.9741/0.9784`; background blur/gray prediction drop `0.0003/-0.0001`;
  object desaturate drop `0.0642`; near-tie `9/20`, Grad-CAM border `7/20`,
  rollout border `10/20`.
- Decision: reject full train. ELR preserved foreground focus but shifted the tradeoff
  toward more class-1 false positives. It did not beat the anchor class-1 F1
  `0.6860` and stayed below the `0.70` gate. Do not repeat ELR weight `0.03`
  on this anchor; ELR is not the missing boundary signal by itself.

### Teacher-guided supervised contrastive did not clear the boundary errors

Date: 2026-07-02

- Rationale: the existing supervised contrastive/boundary losses use hard labels.
  Because the train-only teacher cache contains soft probability structure, a
  filtered supervised contrastive loss was added to use only samples whose teacher
  top class agrees with the hard label and whose teacher confidence is above a
  low threshold. The teacher max-probability distribution is soft (`median=0.3128`,
  `max=0.4975`), so the trial used threshold `0.32` rather than the default `0.70`.
- Implementation: added `--teacher-guided-contrastive-*` CLI flags, config fields,
  source selection over `head,patch`, class whitelist `0,1,2,4`, teacher agreement
  filtering, class-balanced loss normalization, history columns, and unit tests
  in `tests\test_teacher_guided_contrastive.py`.
- Validation: `py_compile` passed for train/config/losses; focused pytest passed
  for `tests\test_teacher_guided_contrastive.py`,
  `tests\test_pretrained_distillation.py`, `tests\test_teacher_probability_dataset.py`,
  and `tests\test_trainable_module_prefixes.py`.
- Inactive smoke bug:
  `runs\smoke_v8_yolof_tgcontrast002_c032_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  had the dry-run config fields but the launcher did not yet pass the new flags
  into actual `TrainArgs`, so `resolved_config.json` and history showed
  `teacher_guided_contrastive_loss_weight=0.0`. The launcher was fixed before
  the active smoke; dry-run config alone is not sufficient evidence that a new
  loss is active.
- Active smoke:
  `runs\smoke_v8_yolof_tgcontrast002_c032_active_teacherfocusbinary015_boundarydrop_bboxprior_20260702`.
  The loss was active (`train_teacher_guided_contrastive_loss=1.4202`,
  fraction `0.3542`, count `11.33`). Smoke subset macro/class-1 F1 was
  `0.7455/0.7857`, but the subset missed class 4 and is not a gate metric.
  XAI selected `12`: top confusions still included `3->2`, `0->1`, `1->0`,
  `0->4`, `4->1`; attention/grad-rollout/Grad-CAM foreground mass was
  `0.9726/0.9715/0.9705`; object desaturate drop was `0.0319`.
- Probe:
  `runs\probe_v8_yolof_tgcontrast002_c032_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702`.
  Best validation epoch `2`: accuracy `0.9206`, macro/class-1 F1
  `0.8854/0.6857`, class-1 P/R `0.6030/0.7947`. Confusion:
  `0->1=52`, `1->0=14`, `1->2=12`, `2->1=11`, `4->1=12`.
  Compared with the anchor (`0.8847/0.6860`, class-1 P/R `0.6114/0.7815`),
  macro improved slightly but class-1 F1 and precision did not.
- Probe XAI selected `20`: top confusions `0->1=53`, `3->2=50`, `1->0=14`,
  `4->1=13`, `1->2=12`, `2->1=11`; attention/grad-rollout/Grad-CAM foreground
  mass `0.9724/0.9756/0.8796`; background blur/gray prediction drop
  `0.0006/-0.0002`; object desaturate drop `0.0734`; near-tie `7/20`,
  Grad-CAM border `7/20`, rollout border `11/20`.
- Decision: reject full train. Teacher-guided contrastive made the model more
  class-1-recall oriented but did not improve class-1 F1 and increased class-1
  false-positive pressure (`0->1` and `4->1`). Do not repeat the exact
  `weight=0.02`, threshold `0.32`, `head,patch`, classes `0,1,2,4`, agreement
  configuration on this anchor.

### Pairwise Confusion regularization worsened class-1 precision

Date: 2026-07-02

- Literature/source: Pairwise Confusion for FGVC reports a simple activation
  regularizer that reduces overfitting and improves localization without inference
  overhead ([arXiv:1705.08016](https://arxiv.org/abs/1705.08016),
  [CVF ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Abhimanyu_Dubey_Improving_Fine-Grained_Visual_ECCV_2018_paper.html)).
  This looked compatible with TRKH because the remaining errors are fine-grained
  inter-class boundary mistakes, not a background shortcut.
- Implementation check: `pairwise_confusion_loss` was already implemented and
  unit-tested, but the v8 launcher initially only exposed it in metadata. The
  actual `TrainArgs` were patched to pass `--pairwise-confusion-loss-weight`,
  `--pairwise-confusion-sources`, and `--pairwise-confusion-start-epoch`.
  `py_compile` passed and focused pytest
  `tests\test_detection_calibration.py -k pairwise_confusion` passed.
- Dry-run confirmed actual config:
  `pairwise_confusion_loss_weight=0.005`, sources `head,patch,registers,logits`,
  start epoch `1`, normalize enabled.
- Smoke:
  `runs\smoke_v8_yolof_pc005_all_teacherfocusbinary015_boundarydrop_bboxprior_20260702`.
  Loss was active (`train_pairwise_confusion_loss=1.0342`) with teacher
  focus-binary and bbox boundary-drop unchanged. Smoke subset macro F1 was
  `0.7458`; architecture trace completed.
- Smoke XAI selected `12`: top confusions already shifted against class 1
  (`0->1=57`, `4->1=16`, `2->1=13`, `1->0=14`), but foreground focus stayed
  clean (attention/grad-rollout/Grad-CAM foreground mass
  `0.9832/0.9768/0.9760`) and background blur/gray prediction drop was near `0`.
  This was enough for a short probe, but not a full train.
- Probe:
  `runs\probe_v8_yolof_pc005_all_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702`.
  Best validation epoch `2`: macro/class-1 F1 `0.8806/0.6685`, class-1 P/R
  `0.5769/0.7947`, confusion `0->1=57`, `1->0=14`, `1->2=11`, `2->1=12`,
  `4->1=15`. This is below the anchor macro/class-1 F1 `0.8847/0.6860`.
- Probe XAI selected `20`: top confusions `0->1=56`, `3->2=54`, `4->1=15`,
  `1->0=13`, `2->1=12`; attention/grad-rollout/Grad-CAM foreground mass
  `0.9838/0.9761/0.9833`; background blur/gray prediction drop
  `0.0006/-0.0007`; object desaturate drop `0.0745`; near-tie `8/20`,
  Grad-CAM border `8/20`, rollout border `11/20`.
- Decision: reject full train. PC regularized activations without creating the
  missing boundary signal, and it increased class-1 false positives more than it
  helped class-1 recall. Do not repeat `pairwise_confusion_loss_weight=0.005`
  with sources `head,patch,registers,logits` on this anchor.

### Mutual-Channel Loss did not improve the current anchor

Date: 2026-07-02

- Literature/source: Mutual-Channel Loss targets fine-grained recognition by
  making class-aligned feature channels discriminative and spatially diverse,
  without requiring part annotations or inference-time overhead
  ([arXiv:2002.04264](https://arxiv.org/abs/2002.04264),
  [IEEE TIP/code metadata](https://github.com/PRIS-CV/Mutual-Channel-Loss)).
  This was a plausible next test because TRKH's remaining bottleneck is subtle
  surface/boundary cue separation, not far-background reliance.
- Implementation check: the repo already had `mutual_channel_loss` on
  `stem_features`, CLI/config/launcher plumbing, and focused tests. `py_compile`
  passed for train/config/losses and
  `tests\test_detection_calibration.py -k mutual_channel` passed. The v8 launcher
  actual `TrainArgs` already passed `--mutual-channel-*`.
- Guardrail: dry-run initially showed the launcher's old default
  `HardSampleManifest`; the actual smoke/probe commands explicitly set
  `HardSampleManifest=''` and `HardSampleRepeatFactor=1` to avoid stale hard-repeat.
- Smoke:
  `runs\smoke_v8_yolof_mcl002_teacherfocusbinary015_boundarydrop_bboxprior_20260702`.
  It resumed the current best teacher-focus-binary anchor, kept bbox spatial fusion,
  boundary-band foreground dropout, pairwise-margin routing, and teacher-focus
  binary `0.015`, and added mutual-channel `weight=0.02`, `top_k=8`,
  diversity `0.20`, start epoch `1`. The loss was active
  (`train_mutual_channel_loss=1.7025`), architecture trace completed, and hard
  sample repeat was off. Smoke subset macro/class-1 F1 was `0.7607/0.8462`, but
  the subset missed class 4 and is not a gate metric.
- Smoke XAI selected `12`: top confusions were close to the anchor
  (`3->2=53`, `0->1=50`, `1->0=16`, `4->1=12`). Foreground mass for
  attention/grad-rollout/Grad-CAM was `0.9786/0.9771/0.9827`; background blur/gray
  prediction drops were `0.0002/-0.0003`; object desaturate drop was `0.0344`.
  This was sane enough for a short probe.
- Probe:
  `runs\probe_v8_yolof_mcl002_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702`.
  Best epoch `1`: validation macro/class-1 F1 `0.8816/0.6763`, class-1 P/R
  `0.6000/0.7748`, confusion `0->1=51`, `1->0=18`, `1->2=11`, `2->1=11`,
  `4->1=12`. This is below the anchor macro/class-1 F1 `0.8847/0.6860`.
  Epoch 2 did not recover (`val_macro_f1=0.8808`).
- Probe XAI selected `20`: top confusions `0->1=51`, `1->0=18`, `4->1=12`;
  attention/grad-rollout/Grad-CAM foreground mass `0.9675/0.9739/0.9828`;
  background blur/gray prediction drops `0.0007/-0.0002`; object desaturate drop
  `0.0518`; near-tie `8/20`, Grad-CAM border `8/20`, rollout border `11/20`.
- Decision: reject full train. Mutual-channel produced clean foreground focus but
  did not change the class-1 boundary in the right direction; it slightly worsened
  both `0->1` and `1->0` relative to the anchor. Do not repeat
  `mutual_channel_loss_weight=0.02`, `top_k=8`, diversity `0.20`, start epoch `1`
  on the current teacher-focus-binary anchor.

### Complement entropy did not control class-1 false positives

Date: 2026-07-02

- Literature/source: Complement Objective Training proposes optimizing the primary
  ground-truth objective while neutralizing probabilities assigned to complement
  classes ([arXiv:1903.01182](https://arxiv.org/abs/1903.01182),
  [OpenReview ICLR 2019](https://openreview.net/forum?id=HyM7AiA5YX)). This was
  a targeted test because the current TRKH anchor often lets class `1` dominate
  the non-target distribution for true class `0/2/4` images.
- Implementation: added optional `complement_entropy_loss_weight`,
  `complement_entropy_classes`, and `complement_entropy_start_epoch` to config,
  CLI, v8 launcher, train loop stats/history, and focused tests. The minimized
  term is `1 - H(non_target_probs) / log(C-1)`, so it is zero when complement
  probabilities are uniform. `py_compile` passed and focused pytest for
  complement entropy plus PC/MCL passed (`5 passed`).
- Smoke:
  `runs\smoke_v8_yolof_cot002_teacherfocusbinary015_boundarydrop_bboxprior_20260702`.
  It resumed the current anchor, kept bbox spatial fusion, boundary-band bbox
  dropout, pairwise-margin routing, and teacher-focus-binary `0.015`, disabled
  stale hard-repeat, and used complement entropy `weight=0.02`, classes
  `0,1,2,4`, start epoch `1`. The loss was active
  (`train_complement_entropy_loss=0.0063`) and architecture trace completed.
- Smoke XAI selected `12`: top confusions remained close to the anchor
  (`3->2=53`, `0->1=50`, `1->0=16`, `4->1=12`). Foreground mass for
  attention/grad-rollout/Grad-CAM/rollout was `0.9833/0.9733/0.9831/0.9518`;
  background blur/gray prediction drops were `0.0009/-0.0004`; object desaturate
  drop was `0.0675`. This was sane enough for a short probe.
- Probe:
  `runs\probe_v8_yolof_cot002_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702`.
  Best epoch `2`: validation macro/class-1 F1 `0.8821/0.6780`, class-1 P/R
  `0.5911/0.7947`, confusion `0->1=55`, `1->0=14`, `1->2=12`, `2->1=12`,
  `4->1=12`. This is below the anchor `0.8847/0.6860` and below the `0.70`
  full-train gate.
- Probe XAI selected `20`: foreground remained clean
  (`attention/grad-rollout/Grad-CAM/rollout` foreground mass
  `0.9882/0.9767/0.9829/0.9592`), background perturbation stayed near zero
  (`0.0002/-0.0007`), and object desaturate drop was `0.0704`. Review flags
  included `near_tie=6/20`, `object_color_sensitive=5/20`, `gradcam_border=6/20`,
  `rollout_border=10/20`.
- Decision: reject full train. Complement entropy regularized non-target
  distribution but did not fix the boundary; it increased class `0->1` false
  positives relative to the anchor. Do not repeat `weight=0.02`, classes
  `0,1,2,4`, start epoch `1` on this anchor unless paired with a mechanism that
  explicitly suppresses class-1 false positives.

### Ordinal-distribution / EMD smoke was rejected before probe

Date: 2026-07-02

- Literature/source: squared Earth Mover's Distance loss uses class-distance
  structure rather than treating all wrong classes equally
  ([arXiv:1611.05916](https://arxiv.org/abs/1611.05916)); Deep Label
  Distribution Learning uses soft label distributions for ambiguous labels
  ([arXiv:1611.01731](https://arxiv.org/abs/1611.01731)). This was a plausible
  check because the mango labels have ordered maturity-like neighbors and the
  remaining failures are boundary/ambiguity cases.
- Implementation check: the repo already had
  `_ordinal_distribution_loss_from_logits`, config/CLI/v8 launcher plumbing, and
  tests. `py_compile` passed for train/config/losses and
  `tests\test_cumulative_ordinal_head.py -k "ordinal_distribution or cumulative"`
  passed (`3 passed`). The dry-run initially caught wrong legacy launcher flag
  names; the accepted dry-run used `classification_folder_yolo_data=yolo_f`,
  `bbox_spatial_fusion=true`, `pairwise_margin_routing=true`, empty
  `hard_sample_manifest`, and ordinal distribution `weight=0.02`,
  classes `0,1,2,3,4`, target sigma `0.55`, start epoch `1`.
- Smoke:
  `runs\smoke_v8_yolof_orddist002s055_teacherfocusbinary015_boundarydrop_bboxprior_20260702`.
  It resumed the current teacher-focus-binary anchor and kept bbox spatial
  fusion, boundary-band bbox dropout, pairwise-margin routing, and
  teacher-focus-binary `0.015`. The ordinal loss was active
  (`train_ordinal_distribution_loss=0.1017`), architecture trace completed, and
  stale hard-repeat was disabled. The 6-batch validation subset had only class 0
  support, so its macro F1 was not used as a gate.
- Smoke XAI selected `12`: top full-val confusions were `3->2=54`,
  `0->1=50`, `1->0=19`, `2->1=13`, `1->2=12`, `4->1=10`. The class-1 false
  negatives worsened versus the anchor (`1->0` from `17` to `19`) and
  `0->1` did not improve materially. Heatmap focus also degraded:
  attention/grad-rollout/Grad-CAM/rollout foreground mass
  `0.8405/0.9267/0.7375/0.8597`, with Grad-CAM background mass `0.2625`,
  rollout background mass `0.1403`, and border flags on `6/12` Grad-CAM and
  `7/12` rollout cases. Background blur/gray prediction drops stayed near zero
  (`0.0019/0.0020`), while object desaturation drop rose to `0.1599`.
- Decision: reject before 2-epoch probe. The smoke failed the XAI sanity check:
  EMD/label-distribution smoothing made explanations more diffuse and did not
  reduce the class-1 boundary errors. Do not repeat ordinal-distribution
  `weight=0.02`, classes `0,1,2,3,4`, sigma `0.55`, start epoch `1` on this
  anchor.

### SAM rho 0.015 was rejected before probe

Date: 2026-07-02

- Literature/source: Sharpness-Aware Minimization optimizes for neighborhoods
  with uniformly low loss and reports improved generalization/noise robustness
  ([arXiv:2010.01412](https://arxiv.org/abs/2010.01412)); ASAM adapts the
  perturbation scale to reduce SAM's sensitivity to parameterization
  ([arXiv:2102.11600](https://arxiv.org/abs/2102.11600)). This was a plausible
  optimizer-level test because many loss/architecture add-ons had improved
  heatmap focus without improving the noisy class-1 boundary.
- Implementation check: the launcher dry-run used the corrected v8 arguments
  (`-DataYaml`, `-ClassificationFolderYoloData`, `-BBoxSpatialFusion`,
  `-PairwiseMarginRouting`) and confirmed `use_sam=true`, `sam_rho=0.015`,
  hard-repeat disabled, bbox spatial fusion enabled, and pairwise-margin routing
  enabled. A first smoke exposed a SAM replay bug where `_forward_train_loss`
  was sliced before unpacking; `trkh/training/train.py` now reads the returned
  tuple and takes index `0`. `py_compile` passed and focused tests for the
  touched training/model paths passed (`5 passed`).
- Smoke:
  `runs\smoke_v8_yolof_samrho015_teacherfocusbinary015_boundarydrop_bboxprior_20260702`.
  It resumed the current teacher-focus-binary anchor, disabled attention-view
  loss because SAM replay is incompatible with that auxiliary branch, kept bbox
  spatial fusion, boundary-band bbox dropout, pairwise-margin routing, and
  teacher-focus-binary `0.015`. SAM disabled AMP and made 24 batches slow
  (`train_seconds=52.61`, `gpu_memory_max_allocated_mb=6989.55`). The active
  losses were otherwise consistent with the anchor (`pairwise_margin_loss=0.5644`,
  `teacher_focus_binary_loss=0.5133`, bbox dropout active). The 6-batch
  validation subset had only class 0 support, so its macro F1 was not used as a
  gate.
- Smoke XAI selected `12`: top full-val confusions were `3->2=52`,
  `0->1=50`, `1->0=19`, `2->1=13`, `1->2=12`, `4->1=11`. The class-1 false
  negatives worsened versus the anchor (`1->0` from `17` to `19`), while
  `0->1` did not improve. Heatmap focus also failed the sanity check:
  attention/grad-rollout/Grad-CAM/rollout foreground mass
  `0.8397/0.9264/0.7459/0.8597`, with Grad-CAM background mass `0.2541`,
  rollout background mass `0.1403`, and border flags on `6/12` Grad-CAM and
  `7/12` rollout cases. Background blur/gray prediction drops stayed near zero
  (`0.0013/0.0003`), while object desaturation drop rose to `0.1645`.
- Decision: reject before 2-epoch probe. SAM rho `0.015` made the smoke slower,
  disabled AMP, did not reduce class-1 confusions, and produced diffuse
  explanations similar to the rejected ordinal-distribution smoke. Do not repeat
  SAM `rho=0.015`, non-adaptive, `AttentionViewLossWeight=0` on the current
  teacher-focus-binary anchor.

### SNSCL-lite weighted/stochastic teacher-guided SupCon was rejected before probe

Date: 2026-07-02

- Literature/source: LNL-FG/SNSCL targets fine-grained classification with noisy
  labels by using noise-tolerated supervised contrastive learning, a weight-aware
  mechanism, and stochastic feature embeddings
  ([arXiv:2303.02404](https://arxiv.org/abs/2303.02404)). Sel-CL also motivates
  selecting confident pairs for supervised contrastive learning under label noise
  ([arXiv:2203.04181](https://arxiv.org/abs/2203.04181)). This was a plausible
  next step because the remaining TRKH errors are fine-grained label-boundary
  cases and generic smoothing/logit losses had failed.
- Implementation: extended `SupervisedContrastiveLoss` with optional
  `sample_weights`, then extended teacher-guided contrastive with
  `weight_mode=confidence_margin`, stochastic feature embedding noise, minimum
  reliability, and confidence power. CLI/config/v8 launcher plumbing and focused
  tests were added. Defaults preserve the previous behavior (`filter`,
  stochastic `0`). `py_compile` passed and focused pytest for teacher-guided
  contrastive/train-one-epoch paths passed (`5 passed`).
- Smoke:
  `runs\smoke_v8_yolof_snscllite_tgcontrast012_cm_sstd004_teacherfocusbinary015_boundarydrop_bboxprior_20260702`.
  It resumed the current teacher-focus-binary anchor, kept bbox spatial fusion,
  boundary-band bbox dropout, pairwise-margin routing, attention-view loss, and
  teacher-focus-binary `0.015`, disabled hard-repeat, and used teacher-guided
  SupCon `weight=0.012`, temperature `0.18`, sources `head,patch`, classes
  `0,1,2,4`, `teacher_min_confidence=0.0`, `weight_mode=confidence_margin`,
  `stochastic_std=0.04`, confidence power `0.5`. The loss was active
  (`train_teacher_guided_contrastive_loss=0.8812`) but sparse
  (`fraction=0.1068`, average `3.42` selected samples/batch). The 6-batch
  validation subset had only class 0 support, so its macro F1 was not used as a
  gate.
- Smoke XAI selected `12`: top full-val confusions were `3->2=54`,
  `0->1=50`, `1->0=19`, `2->1=13`, `1->2=12`, `4->1=10`. It did not improve
  the class-1 boundary versus the anchor, and class-1 false negatives worsened
  (`1->0` from `17` to `19`). Heatmap focus failed the sanity check:
  attention/grad-rollout/Grad-CAM/rollout foreground mass
  `0.8401/0.9268/0.7411/0.8597`, with Grad-CAM background mass `0.2589`,
  rollout background mass `0.1403`, and border flags on `6/12` Grad-CAM and
  `7/12` rollout cases. Background blur/gray prediction drops were still near
  zero (`0.0019/0.0021`), while object desaturation drop was `0.1599`.
- Decision: reject before 2-epoch probe. The implementation is valid and generic,
  but this exact setting has too little selected signal and makes explanations
  diffuse without reducing class-1 confusions. Do not repeat teacher-guided
  stochastic weighted SupCon `weight=0.012`, temperature `0.18`,
  `confidence_margin`, stochastic std `0.04`, confidence power `0.5`,
  classes `0,1,2,4`, with teacher focus-binary `0.015` on this anchor.

## Do-Not-Repeat List

- Source-context/full-frame wide-context fusion or suppression with the current
  scale/gate settings.
- Static or train-in-sample `class_f + yolo_f` probability fusion/router.
- Bbox/objectness-only heads, patch objectness, bbox-token context, or stronger
  foreground-only attention forcing.
- Hard class-1 false-positive margin, hard-label blend, top-k residual, and
  focus-neighbor binary objectives with the current weights.
- Global AIDT KD, AIDT focus-only, and AIDT rescue-margin objectives with current
  confidence/margin settings.
- Model soup over the current checkpoint family, bbox-aware TTA, prototype/cosine
  head on the current embedding, and OOF scratch 2e/120b schedule.
- ELR weight `0.03` on the current `yolo_f` teacher-focus-binary anchor.
- Teacher-guided supervised contrastive with `weight=0.02`, confidence `0.32`,
  sources `head,patch`, classes `0,1,2,4`, and teacher agreement on the current
  `yolo_f` teacher-focus-binary anchor.
- Pairwise Confusion regularization with `weight=0.005`, sources
  `head,patch,registers,logits`, start epoch `1` on the current
  `yolo_f` teacher-focus-binary anchor.
- Mutual-Channel Loss with `weight=0.02`, `top_k=8`, diversity `0.20`, start
  epoch `1` on the current `yolo_f` teacher-focus-binary anchor.
- Complement entropy with `weight=0.02`, target classes `0,1,2,4`, start epoch
  `1` on the current `yolo_f` teacher-focus-binary anchor.
- Ordinal-distribution / EMD label smoothing with `weight=0.02`, classes
  `0,1,2,3,4`, target sigma `0.55`, start epoch `1` on the current
  `yolo_f` teacher-focus-binary anchor.
- SAM optimizer probe with `rho=0.015`, non-adaptive SAM, AMP disabled, and
  attention-view loss disabled on the current `yolo_f` teacher-focus-binary
  anchor.
- SNSCL-lite teacher-guided stochastic weighted SupCon with `weight=0.012`,
  temperature `0.18`, sources `head,patch`, classes `0,1,2,4`,
  `weight_mode=confidence_margin`, stochastic std `0.04`, confidence power
  `0.5`, and teacher focus-binary `0.015` on the current `yolo_f`
  teacher-focus-binary anchor.

## Cleanup Log 2026-07-02

- Deleted rejected ELR artifacts after preserving metrics and XAI conclusions:
  `smoke_v8_yolof_elr003_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  and
  `probe_v8_yolof_elr003_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702`.
  Manifest:
  `runs\cleanup_manifest_20260702_elr003_reject.json`; reclaimed about `577 MB`.
- Deleted rejected teacher-guided contrastive artifacts after preserving metrics and
  XAI conclusions:
  `smoke_v8_yolof_tgcontrast002_c032_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  (inactive launcher-flag bug),
  `smoke_v8_yolof_tgcontrast002_c032_active_teacherfocusbinary015_boundarydrop_bboxprior_20260702`,
  and
  `probe_v8_yolof_tgcontrast002_c032_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702`.
  Manifest:
  `runs\cleanup_manifest_20260702_tgcontrast_reject.json`; reclaimed about
  `763 MB`.
- Deleted rejected Pairwise Confusion artifacts after preserving metrics and XAI
  conclusions:
  `smoke_v8_yolof_pc005_all_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  and
  `probe_v8_yolof_pc005_all_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702`.
  Manifest:
  `runs\cleanup_manifest_20260702_pc005_reject.json`; reclaimed about `587 MB`.
- Deleted rejected Mutual-Channel Loss artifacts after preserving metrics and XAI
  conclusions:
  `smoke_v8_yolof_mcl002_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  and
  `probe_v8_yolof_mcl002_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702`.
  Manifest:
  `runs\cleanup_manifest_20260702_mcl002_reject.json`; reclaimed about `552 MB`.
- Deleted three older rejected non-keeper artifacts after confirming their outcomes
  were already documented: `ssl_dino_yolof_objectcrop_supervisedinit_60b1e_20260702`,
  `ssl_vicreg_yolof_objectcrop_120b2e_20260702`, and
  `probe_oof_yolof_v8_scratch_120b2e_20260702_fold_00`. Manifest:
  `runs\cleanup_manifest_20260702_ssl_oof_reject.json`; reclaimed about `625 MB`.
- Deleted rejected Complement Entropy artifacts after preserving metrics and XAI
  conclusions:
  `smoke_v8_yolof_cot002_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  and
  `probe_v8_yolof_cot002_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702`.
  Manifest:
  `runs\cleanup_manifest_20260702_cot002_reject.json`; reclaimed about `548 MB`.
- Deleted rejected ordinal-distribution smoke after preserving metrics and XAI
  conclusions:
  `smoke_v8_yolof_orddist002s055_teacherfocusbinary015_boundarydrop_bboxprior_20260702`.
  Manifest:
  `runs\cleanup_manifest_20260702_orddist002s055_reject.json`; reclaimed about
  `225 MB`.
- Deleted rejected SAM smoke after preserving metrics and XAI conclusions:
  `smoke_v8_yolof_samrho015_teacherfocusbinary015_boundarydrop_bboxprior_20260702`.
  Manifest:
  `runs\cleanup_manifest_20260702_samrho015_reject.json`; reclaimed about
  `226 MB`.
- Deleted rejected SNSCL-lite teacher-guided SupCon smoke after preserving
  metrics and XAI conclusions:
  `smoke_v8_yolof_snscllite_tgcontrast012_cm_sstd004_teacherfocusbinary015_boundarydrop_bboxprior_20260702`.
  Manifest:
  `runs\cleanup_manifest_20260702_snscllite_tgcontrast012_reject.json`;
  reclaimed about `225 MB`.
- Deleted invalid/rejected AIDT teacher-guided SupCon smokes after preserving
  conclusions:
  `smoke_v8_yolof_aidt_snscl_tgcontrast008_conf_sstd003_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  (wrong-view `yolo_f` sample-index CSV bound to `class_f`) and
  `smoke_v8_classf_aidt_snscl_tgcontrast008_conf_sstd003_teacherfocusbinary015_boundarydrop_bboxprior_20260702_fixmap`
  (corrected mapping but XAI/confusions did not improve). Manifest:
  `runs\cleanup_manifest_20260702_aidt_snscl_fixmap_reject.json`;
  reclaimed about `403 MB`.
- Deleted invalid/rejected angular-margin artifacts after preserving metrics and
  XAI conclusions:
  `smoke_v8_yolof_angmargin001_m006_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  (wrong primary view failure), the valid smoke
  `smoke_v8_yolof_angmargin001_m006_teacherfocusbinary015_boundarydrop_bboxprior_20260702_yolodata`,
  the auxiliary full-val export
  `eval_smoke_v8_yolof_angmargin001_m006_teacherfocusbinary015_val_20260702`,
  and the rejected probe
  `probe_v8_yolof_angmargin001_m006_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702`.
  Manifest:
  `runs\cleanup_manifest_20260702_angmargin001_m006_reject.json`; reclaimed about
  `502 MB`.

## Next Useful Checks

- If continuing router/ensemble research, first design a legal OOF or representation
  route that is stronger on fold_00 than the failed scratch 2e/120b recipe.
- If adding architecture, prefer a narrow boundary/part relation module that changes
  representation around class `0/1/2/4` ambiguity while preserving the current
  teacher-focus-binary anchor.

## Literature Check 2026-07-02

- [Decoupled Knowledge Distillation](https://arxiv.org/abs/2203.08679) separates
  target-class and non-target-class knowledge. This matches the current AIDT problem:
  AIDT has useful class-relation signal, but its target mass can create extra class-1
  false positives. TRKH already audited a gated non-target/DKD probe at confidence
  `0.70`; it was clean by XAI but did not improve class-1 F1, so do not repeat this
  exact setting.
- [Trusted Multi-View Classification](https://arxiv.org/abs/2102.02051) fuses
  multiple views at evidence/uncertainty level rather than simple probability
  averaging. This is relevant to future `class_f + yolo_f` work, but current paired
  logits/routers failed, so do not implement another multi-view module unless it has
  train-safe uncertainty supervision and a focused smoke metric.
- [Co-teaching](https://arxiv.org/abs/1804.06872) and
  [DivideMix](https://arxiv.org/abs/2002.07394) address noisy labels by dynamically
  selecting or relabeling likely-clean samples, usually with two networks and longer
  schedules. They are conceptually relevant to class-boundary noise, but expensive and
  risky for the current no-pretrain <=30 epoch gate. Treat them as a later diagnostic,
  not the next implementation target.
- [Pairwise Confusion for Fine-Grained Visual Classification](https://arxiv.org/abs/1705.08016)
  is relevant because it targets FGVC overfitting and inter-class similarity, but
  the current TRKH probe showed worse class-1 precision/F1. Do not repeat the
  audited `0.005` all-source PC setting on this anchor.
- [Mutual-Channel Loss](https://arxiv.org/abs/2002.04264) is relevant because it
  targets discriminative local regions through channel specialization, but the
  current `0.02/top_k=8` probe on the teacher-focus-binary anchor lowered
  class-1 F1 and increased boundary mistakes. Do not repeat this exact setting.
- [Squared EMD loss](https://arxiv.org/abs/1611.05916) and
  [Deep Label Distribution Learning](https://arxiv.org/abs/1611.01731) are
  relevant to ordered or ambiguous labels, but the audited ordinal-distribution
  smoke (`weight=0.02`, classes `0,1,2,3,4`, sigma `0.55`) made XAI focus more
  diffuse and did not improve class-1 confusions. Do not repeat this exact
  setting on the current anchor.
- [Fine-Grained Classification with Noisy Labels / SNSCL](https://arxiv.org/abs/2303.02404)
  and [Selective-Supervised Contrastive Learning](https://arxiv.org/abs/2203.04181)
  motivate weighted/confident contrastive pairs for noisy fine-grained labels, but
  the audited lightweight TRKH variant (`0.012`, confidence-margin weighting,
  stochastic feature std `0.04`) was too sparse and made XAI focus diffuse. Do
  not repeat this exact setting on the current anchor.
- [ArcFace](https://arxiv.org/abs/1801.07698) motivates angular-margin supervision
  when class separability is the bottleneck. [Sub-center ArcFace](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)
  is more relevant to this dataset than plain global contrastive pressure because
  it relaxes intra-class compactness under noisy/hard samples by allowing multiple
  centers per class. If trying margin-based representation next, prefer a narrow
  boundary-class or sub-center-style variant over another all-anchor SupCon/KD run.
- [TransFG](https://arxiv.org/abs/2103.07976),
  [Feature Fusion Vision Transformer / FFVT](https://arxiv.org/abs/2107.02341),
  and [TokenLearner](https://research.google/blog/improving-vision-transformer-efficiency-and-accuracy-by-learning-to-tokenize/)
  support the next FGVC direction: select or learn a small set of discriminative
  local tokens instead of relying only on the final CLS/global representation.
  This is relevant because current XAI repeatedly shows foreground focus is clean
  but class `0/1/2/4` boundary decisions remain ambiguous. However, TRKH already
  rejected part-token, local-zoom, high-frequency, and now micro-detail settings
  listed below; future token/part work must introduce a sharper training signal,
  not just another residual head over the same noisy boundary cases.

## Audit 2026-07-02 - AIDT Teacher CSV View Guard and Corrected SupCon Smoke

- Root cause found for the first AIDT teacher-guided contrastive smoke
  `smoke_v8_yolof_aidt_snscl_tgcontrast008_conf_sstd003_teacherfocusbinary015_boundarydrop_bboxprior_20260702`:
  the CSV was keyed by `yolo_f` sample_index while the launcher primary dataset
  was `class_f` with paired `yolo_f` bbox metadata. The CSV itself was correct
  (`argmax(prob_*)` matched `prediction_index`, train agreement `0.9861`), but
  binding it to `class_f` by sample_index produced loader agreement `0.1310` and
  sparse contrastive selection (`fraction=0.1016`). Treat this smoke as invalid,
  not as method evidence.
- Added `TeacherProbabilityDataset` path-overlap guard: sample-index teacher CSVs
  now report `sample_index_path_overlap_count/ratio` and raise when the CSV paths
  do not overlap the current primary dataset paths. Focused tests passed:
  `tests/test_teacher_probability_dataset.py`,
  `tests/test_teacher_guided_contrastive.py`, and
  `tests/test_remap_yolo_teacher_to_classification_folder.py` (`7 passed`).
- Generated a corrected class_f-order AIDT teacher artifact:
  `runs\aidt_classf_pretrained_train_20260702\teacher_probs_classf_train_sampleindex_aidtpretrained_ttaflip.csv`.
  Remap summary: `9215/9215` mapped, missing `0`, teacher-label agreement
  `0.9861`, teacher prediction counts `[1877, 627, 1903, 2517, 2291]`.
- Corrected smoke
  `smoke_v8_classf_aidt_snscl_tgcontrast008_conf_sstd003_teacherfocusbinary015_boundarydrop_bboxprior_20260702_fixmap`
  used the fixed CSV and confirmed loader agreement `0.9861` and
  `sample_index_path_overlap_ratio=1.0`. The contrastive signal was no longer
  sparse (`train_teacher_guided_contrastive_fraction=0.7734`, about `24.75`
  samples/batch), but the auxiliary loss was large (`3.1536`) and total train
  loss rose to `1.6605`.
- XAI selected12 for the corrected smoke did not improve the failure mode:
  top confusions remained `3->2=52`, `0->1=50`, `1->0=19`, `2->1=13`,
  `1->2=12`. Heatmap focus stayed diffuse for Grad-CAM
  (`attention/grad_rollout/gradcam/rollout foreground mass =
  0.8434/0.9267/0.7409/0.8599`; Grad-CAM background `0.2591`), while
  background perturbation remained negligible (`background_blur=0.0017`,
  `background_gray=0.0020`) and object desaturation was strong (`0.1609`).
  Reject before 2e probe; do not repeat AIDT all-anchor teacher-guided SupCon
  with `weight=0.008`, `temp=0.18`, `confidence`, `std=0.03`,
  `min_reliability=0.20` on this anchor.

## Audit 2026-07-02 - Narrow Angular Margin Smoke/Probe

- Tested a narrow ArcFace-inspired angular-margin auxiliary loss on the current
  `yolo_f` teacher-focus-binary anchor, using `weight=0.01`, margin `0.06`,
  scale `12.0`, start epoch `1`, and target classes `0,1,2,4`. The launcher used
  primary `yolo_f` data and the sample-index teacher CSV with
  `sample_index_path_overlap_ratio=1.0` and teacher-label agreement `0.9621`, so
  this result is a valid method result rather than a teacher-view mismatch.
- Smoke run
  `smoke_v8_yolof_angmargin001_m006_teacherfocusbinary015_boundarydrop_bboxprior_20260702_yolodata`
  had active angular-margin loss (`0.8566`). A full-val export gave
  macro/class-1 F1 `0.8846/0.6879`, class-1 P/R `0.6103/0.7881`, with confusion
  `0->1=50`, `1->0=16`, `1->2=11`, `4->1=12`. This was only a tiny class-1 gain
  over the anchor (`0.6860`) and still below the `0.70` gate, but it was sane
  enough to justify one short probe.
- Probe run
  `probe_v8_yolof_angmargin001_m006_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702`
  regressed to validation macro/class-1 F1 `0.8824/0.6761`, class-1 P/R
  `0.5882/0.7947`, and confusion
  `[[481,56,2,0,10],[15,120,11,0,5],[6,12,513,7,6],[0,4,50,655,3],[9,12,3,1,625]]`.
  The main failure is increased class-1 false positives, especially `0->1=56`.
- XAI selected12 for the probe stayed cleanly foreground-focused
  (`attention/grad_rollout/gradcam/rollout foreground mass =
  0.9822/0.9734/0.9856/0.9506`) and background perturbation remained negligible
  (`background_blur=0.0007`, `background_gray=-0.0002`), while object desaturation
  still changed predictions (`0.0645`). This confirms the loss is not solving the
  surface-boundary decision problem; it mainly shifts more samples into class 1.
  Reject before full train and do not repeat plain cosine/angular margin with
  `weight=0.01`, margin `0.06`, scale `12.0`, classes `0,1,2,4` on this anchor.

## Audit 2026-07-02 - Teacher-Gated Boundary Center Smoke/Probe

- Implemented a reliability-gated boundary-center auxiliary loss inspired by
  sub-center/noisy-label margin learning, but kept it narrow and train-only:
  centers/anchors are weighted from the existing `yolo_f` sample-index teacher
  CSV, with agreement required and teacher confidence used as the sample weight.
  Added config/launcher flags, focused tests, and a history audit for selected
  fraction/weight mean. Preflight passed with `8` focused tests.
- Smoke run
  `smoke_v8_yolof_tgbcenter010_conf030_agree_teacherfocusbinary015_boundarydrop_bboxprior_20260702_fixstats`
  confirmed the loss was active (`train_boundary_center_loss=0.2427`,
  selected fraction `0.7383`, weight mean `0.3291`). Full validation reached
  macro/class-1 F1 `0.8847/0.6879`, class-1 P/R `0.6103/0.7881`, but confusion
  stayed essentially at the anchor level: `0->1=50`, `1->0=16`, `1->2=11`,
  `4->1=12`, `3->2=53`. XAI stayed foreground-focused
  (`attention/grad_rollout/gradcam/rollout foreground mass =
  0.9789/0.9719/0.9831/0.9532`) and background perturbation stayed negligible.
- Probe run
  `probe_v8_yolof_tgbcenter010_conf030_agree_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702`
  regressed to validation macro/class-1 F1 `0.8817/0.6781`, class-1 P/R
  `0.5950/0.7881`, with confusion
  `[[482,53,2,0,12],[15,119,12,0,5],[6,11,513,7,7],[0,4,52,653,3],[9,13,3,1,624]]`.
  XAI remained foreground-clean (`attention/grad_rollout/gradcam/rollout
  foreground mass = 0.9701/0.9707/0.9810/0.9464`) and background blur/gray
  remained near zero, while object desaturation stayed meaningful (`0.0716`).
  Reject before full train: this center pressure still increases class-1 false
  positives and does not pass the `0.70` class-1 gate. Do not repeat
  `weight=0.01`, pairs `0-1,1-2,2-3,1-4`, margin `0.08`, teacher confidence
  gate `0.30`, agreement required, `confidence` weighting on this anchor.

## Audit 2026-07-02 - Micro-Detail Token Selection Smoke

- Retried the token/part-selection direction on the current stronger `yolo_f`
  teacher-focus-binary anchor using the existing `MicroDetailPatchExpert`, not the
  older pairroute-only anchor. Configuration: top-K `6`, hidden `128`, dropout
  `0.06`, temperature `0.10`, foreground power `1.5`, residual scale `0.12`,
  route pairs `0-1,1-2,2-3,1-4`, route margin `0.20`, auxiliary CE weight
  `0.012`, with hard-repeat disabled. Preflight passed (`9` tests), architecture
  trace completed, and the auxiliary loss was active (`train_micro_detail_aux_loss
  = 5.4505`, weighted about `0.0654`).
- Full validation for
  `smoke_v8_yolof_microdetail_k6_aux012_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  reached macro/class-1 F1 `0.8843/0.6879`, class-1 P/R `0.6103/0.7881`.
  This is below the anchor macro (`0.8847`) and still below the `0.70` class-1
  gate. Confusion remained effectively unchanged:
  `[[485,50,2,0,12],[16,119,11,0,5],[5,10,514,9,6],[0,4,52,653,3],[9,12,4,1,624]]`.
- XAI selected12 stayed foreground-clean
  (`attention/grad_rollout/gradcam/rollout foreground mass =
  0.9803/0.9726/0.9847/0.9534`), background blur/gray stayed negligible
  (`0.0005/-0.0003` pred drop), and object desaturation remained the main
  perturbation signal (`0.0628`). Trace confirmed the head selected exactly six
  patches, but route weights were sparse in class exemplars (`0` for classes
  `0/3/4`, about `0.25` for class `1`, `0.52` for class `2`). Reject before
  probe: the local-detail residual does not change the class-1 false-positive
  pattern. Do not repeat this exact micro-detail setting on the current anchor.

## Diagnostic 2026-07-02 - Class-1 Pairwise Calibration Guard

- Ran a validation-only post-hoc diagnostic on anchor probability CSVs to check
  whether the remaining class-1 error is mostly a calibration/routing issue. A
  global class-1 bias did not help: on
  `eval_yolof_teacherfocusbinary015_best_val_20260702`, the best class-1 F1 was
  only `0.6819`.
- A deterministic pairwise guard that only reassigns predicted class `1` to its
  strongest non-class-1 rival when the class-1 margin is small was much more
  informative. On the current-code eval CSV, the best rule reached macro/class-1
  F1 `0.8892/0.6993` with thresholds `1->0 <= 0.040`,
  `1->3 <= 0.076`, `1->4 <= 0.074`, and no `1->2` reassignment. This reduced
  false-positive `0->1` from `49` to `33`, but class-1 recall fell to `0.7086`
  and class-1 F1 still stayed just below the `0.70` gate.
- On the original run's `best_val_eval_predictions`, the same family reached
  macro/class-1 F1 `0.8895/0.7036` with thresholds `0.040/0.076/0.072`.
  Treat this as a diagnostic, not as a solved model: it is val-tuned postprocessing,
  is very close to the gate, and needs fold-safe or train-only calibration plus a
  final test audit before it can be reported. It does explain why many training
  losses fail: the model already contains enough probability signal to trade
  class-1 recall for precision, but the trade is brittle and pair-dependent.
- Fitting the same guard on available train predictions only did not generalize.
  The train-selected thresholds `1->0=0.042`, `1->3=0.050`, `1->4=0.048`
  achieved train macro/class-1 F1 `0.9454/0.8347`, but only
  `0.8861/0.6859` on current-code val and `0.8864/0.6901` on the original
  run-best val CSV. This rejects train-in-sample threshold fitting as a reporting
  route; use true OOF/fold-safe calibration only if this direction is revisited.
- Applied the fixed validation-derived guards once to the existing final-test
  prediction CSV in
  `runs\calibration_guard_audit_fixed_val_thresholds_20260702`. This was a
  final audit of the already chosen thresholds, not a test sweep. The result
  rejects the post-hoc guard: base final-test macro/class-1 F1 was
  `0.8865/0.6667`, while the val-best guard produced `0.8863/0.6622`
  (class-1 P/R `0.6533/0.6712`) and the train-fit guard produced
  `0.8829/0.6486`. Do not report or deploy the val-only class-1 calibration
  guard; its validation gain is not stable on test.

## Diagnostic 2026-07-02 - Why AIDT Looks Stronger

- Compared object-aligned validation predictions in
  `runs\aidt_classf_pretrained_val_20260702\aligned_trkh_yolof_aidt_val.csv`.
  TRKH current-code val is macro/class-1 F1 `0.8829/0.6783`, while AIDT is
  `0.9088/0.7169`. AIDT has better class-1 precision and recall
  (`0.6575/0.7881`) than TRKH (`0.6031/0.7748`), but it is not uniformly
  reliable for class 1.
- Correctness overlap: both correct `2347`, TRKH wrong/AIDT right `112`, TRKH
  right/AIDT wrong `48`, both wrong `99`. AIDT's useful corrections are mostly
  the exact hard regions: TRKH `3->2` fixed `28` times, `0->1` fixed `23`,
  `4->1` fixed `10`, `1->0` fixed `10`, and `2->1` fixed `6`.
- AIDT also injects class-1 risk. For TRKH false-positive class-1 cases
  (`77` samples), AIDT is correct on only `42`; it still predicts class `1` for
  many remaining errors (`0->1` `27`, `2->1` `4`, `4->1` `2`, `3->1` `1`).
  For TRKH class-1 false negatives (`34` samples), AIDT fixes `17`, but leaves
  `17` wrong. This explains why global AIDT KD, focus-only AIDT KD, DKD, rescue
  margin, and soft ensemble were brittle: AIDT carries stronger pretrained
  representation, but its target-class signal is not clean enough to transfer
  directly without a fold-safe/disagreement-aware gate.

## Diagnostic 2026-07-02 - AIDT Disagreement Gate

- Tested a validation-fitted, final-test-audited AIDT disagreement gate using
  only prediction CSVs, with no dataset edits and no test tuning. The gate keeps
  TRKH unless AIDT disagrees under a class-1-aware rule:
  `TRKH=1,AIDT!=1` can switch when AIDT confidence `>=0.65` and AIDT `p1<=0.35`;
  `TRKH!=1,AIDT=1` is allowed only at very high AIDT confidence `>=0.97` and
  low TRKH confidence `<=0.30`; non-class-1 disagreements switch when AIDT
  confidence `>=0.65` and AIDT confidence minus TRKH confidence `>= -0.05`.
  Audit artifacts: `runs\aidt_disagreement_gate_audit_20260702`.
- Validation improved strongly: TRKH base macro/class-1 F1 `0.8829/0.6783`,
  AIDT alone `0.9088/0.7169`, disagreement gate `0.9171/0.7541` with class-1
  P/R `0.7468/0.7616` and `120` switched samples. This confirms AIDT has useful
  complementary representation when selected by disagreement rather than blended
  globally.
- Final-test audit also improved over both base TRKH and AIDT alone, but not
  enough to pass the TRKH gate: base TRKH test `0.8865/0.6667`, AIDT alone
  `0.8575/0.5806`, disagreement gate `0.8888/0.6857`, class-1 P/R
  `0.7164/0.6575`, with `81` switched samples. Confusion became
  `[[251,11,0,0,1],[18,48,4,0,3],[0,7,226,8,16],[0,0,12,338,6],[4,1,0,0,313]]`.
  Treat this as a useful hybrid inference diagnostic, not a solved no-pretrain
  TRKH model: it depends on AIDT at inference and remains below class-1 `0.70`.
  Future AIDT transfer should try to compress this disagreement-aware selection
  into TRKH or build a fold-safe router, not repeat global soft KD.

## Cleanup 2026-07-02 - Documented Rejected Source-Aux Probes

- Deleted two documented rejected `class_f + yolo_f` source-aux probe directories:
  `probe_v8_classf_yolopair_sourceaux_noteacher_bboxprior_120b_2e_20260701` and
  `probe_v8_classf_yolopair_sourceaux_teacherfocus015_fixmap_bboxprior_120b_2e_20260701`.
  Both were already audited in TODO as below-anchor (`0.8824/0.6706` and
  `0.8818/0.6706`) and not useful as keepers. Manifest:
  `runs\cleanup_manifest_20260702_classf_yolopair_sourceaux_reject.json`;
  reclaimed `528.15 MB`.

## Cleanup 2026-07-02 - Historical Rejected Probe Artifacts

- Preserved the current no-pretrain keeper
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt`,
  final test audit, calibration audit, and AIDT disagreement audit, then deleted
  documented historical rejected probe/smoke/eval families (`probe_mango_cls_*`,
  `probe_groupclean_*`, old cartography/cleanlab/v8 policy probes, paired-fusion
  eval rejects, and older `class_f+yolo_f` probe variants).
- Cleanup manifest:
  `runs\cleanup_manifest_20260702_historical_reject_probes.json`; deleted `106`
  directories and reclaimed `16.233 GB`. D: free space after cleanup was about
  `53.28 GB`.

## Diagnostic 2026-07-02 - AIDT Disagreement Compression Check

- Applied the validation-selected AIDT disagreement gate rule to train-only TRKH
  and AIDT train predictions to see whether it can become a training signal.
  Artifact: `runs\aidt_disagreement_gate_train_analysis_20260702`.
  Train gate switched `230` samples and raised train macro/class-1 F1 from
  `0.9399/0.8154` to `0.9740/0.9097`, but the switched-flow was almost entirely
  train-label correction (`225` wrong->right, `4` right->wrong). It overlaps
  heavily with the previously failed AIDT rescue-margin manifest (`116` overlap,
  `5` rescue-only). This is a strong train-in-sample overfit warning.
- A TRKH-only calibration/gate diagnostic trained from train logits/probabilities
  could not reproduce AIDT gate gains on validation. Artifact:
  `runs\trkh_only_gate_calibrator_diagnostic_20260702`. Best class-1 validation
  F1 stayed at the base `0.6783` (or below), while the true AIDT disagreement gate
  validation diagnostic was `0.7541`. Conclusion: AIDT's useful correction signal
  is not recoverable from the current TRKH logits alone; do not build a train-logit
  router/calibrator from this signal.

## Audit 2026-07-02 - AIDT Reliability Sample Reweight Smoke

- Fixed sample-weight handling for `yolo_f`: `SampleWeightDataset` and
  `_load_sample_weight_manifest` now support `sample_index`, and rows with
  `sample_index` no longer create a path fallback. This prevents duplicate
  multi-object `image_path` collapse. Focused preflight passed:
  `6 passed` for sample-weight/sample-index, teacher-probability, and targeted
  margin tests.
- Built train-only AIDT reliability sample weights in
  `runs\aidt_reliability_sample_weights_20260702\sample_weights_train_only.csv`.
  The manifest has `245` object rows keyed by `sample_index`: `123` soft
  suppressions for TRKH false-positive class-1 cases confirmed by AIDT, `7`
  class-1 FN rescue cases, `67` non-focus TRKH errors confirmed by AIDT, and
  `39` downweight/noise-suspicion cases. It does not change raw data or labels.
- Smoke `smoke_v8_yolof_aidtreweight_soft_teacherfocusbinary015_boundarydrop_bboxprior_20260702_strictidx`
  correctly loaded `paths=0`, `sample_indices=245`, `matched_samples=245`, and
  `train_sample_weight_mean=1.0015`. Full validation eval
  `runs\eval_smoke_v8_yolof_aidtreweight_soft_strictidx_val_20260702` reached
  macro/class-1 F1 `0.8837/0.6841`, class-1 P/R `0.6082/0.7815`, below the
  anchor `0.8847/0.6860`. Confusion remained
  `[[486,50,2,0,11],[17,118,11,0,5],[5,10,515,8,6],[0,4,53,652,3],[9,12,4,1,624]]`;
  `0->1` stayed `50` and `4->1` stayed `12`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_aidtreweight_soft_strictidx_val_selected12_20260702`
  stayed foreground-clean
  (`attention/grad_rollout/gradcam/rollout foreground mass =
  0.9833/0.9732/0.9831/0.9518`), with background blur/gray near zero and object
  desaturation still the dominant perturbation (`0.0675`). Reject before probe:
  soft AIDT reliability sample reweighting does not change the class-1 boundary
  pattern. Do not repeat this exact manifest/weight policy on the current anchor.
- Deleted the failed config smoke, path-fallback invalid smoke, and strict-index
  rejected smoke checkpoints after metrics/XAI were recorded. Manifest:
  `runs\cleanup_manifest_20260702_aidtreweight_reject_smokes.json`; reclaimed
  `0.353 GB`.

## Audit 2026-07-02 - AIDT Feature Relation Distillation Smoke

- Researched relation/attention transfer directions as a representation-level
  alternative to unsafe AIDT logit KD: RKD
  (`https://arxiv.org/abs/1904.05068`), Attention Transfer
  (`https://arxiv.org/abs/1612.03928`), and fine-grained transformer token
  selection ideas from TransFG (`https://arxiv.org/abs/2103.07976`). Chose a
  dimension-agnostic distance RKD smoke from AIDT clean concatenated
  ResNet+ViT features into TRKH head embeddings, not target probabilities.
- Implemented `TeacherFeatureDataset`, strict `sample_index` feature loading for
  `yolo_f`, AIDT feature export, `class_f -> yolo_f` feature remap, launcher
  flags, RKD loss columns, and focused tests. Preflight passed:
  `py_compile` for touched modules and `9 passed` across teacher-feature,
  sample-index, teacher-probability, and remap tests.
- Exported train-only AIDT clean features from
  `D:\DataAI\AIDT\runs\resnet50_vit_b16_class_f_5class_pretrained\best.pt` to
  `runs\aidt_features_classf_train_20260702\features_classf_train_clean_aidtpretrained.npz`
  (`9215` rows, feature dim `2816`, export accuracy `0.9852`, macro F1
  `0.9722` on train). Remapped strictly to `yolo_f` object `sample_index` with
  full coverage `9215/9215` and no missing rows:
  `features_yolof_train_sampleindex_aidtpretrained_clean.npz`.
- Smoke
  `smoke_v8_yolof_aidtfeature_rkd020_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  loaded teacher features correctly (`sample_index_samples=9215`,
  `key_mode=sample_index`, no duplicate sample-index rows). RKD was active:
  `train_teacher_feature_rkd_loss=0.08646`, count `32`, distance-only,
  source `head`, loss weight `0.02`.
- Full validation eval
  `runs\eval_smoke_v8_yolof_aidtfeature_rkd020_val_20260702` reached only
  macro/class-1 F1 `0.8792/0.6707`, below the anchor `0.8847/0.6860`.
  Class-1 P/R was `0.6120/0.7417`; confusion was
  `[[484,48,1,2,14],[21,112,13,0,5],[6,10,513,9,6],[0,4,55,652,1],[8,9,4,1,628]]`.
  It slightly reduced `0->1` versus the anchor but worsened `1->0`, recall, and
  macro F1, so it fails the probe gate.
- XAI selected12
  `runs\xai_smoke_v8_yolof_aidtfeature_rkd020_val_selected12_20260702` stayed
  foreground-clean (`attention/grad_rollout/gradcam/rollout foreground mass =
  0.9784/0.9746/0.9794/0.9522`). Background blur/gray had near-zero effect
  (`0.00058/-0.00027` original-prediction drop), while object desaturation
  remained the dominant perturbation (`0.0682`). Reject before probe: global AIDT
  feature-distance RKD regularizes the embedding geometry but does not solve the
  surface/boundary class-1 decision and reduces class-1 recall. Do not repeat
  `teacher_feature_rkd_loss_weight=0.02`, distance-only, source `head`, AIDT
  clean concatenated features on the current anchor.

## Audit 2026-07-02 - Boundary-Gated AIDT Feature RKD Smoke

- Followed up the failed global feature RKD with a more local relation loss:
  added `teacher_feature_rkd_pair_mode` / `teacher_feature_rkd_pairs` so RKD can
  use only hard-confusion class pairs instead of the whole batch. The first smoke
  used `teacher_feature_rkd_loss_weight=0.01`, source `head`, distance-only,
  `pair_mode=boundary`, pairs `0-1,1-2,1-4,2-3`, plus the current
  teacher-focus-binary anchor recipe. Focused preflight passed:
  `py_compile` and `tests\test_teacher_feature_rkd.py` (`5 passed`), including a
  regression test proving boundary mode masks unlisted pairs.
- Smoke
  `smoke_v8_yolof_aidtfeature_boundaryrkd010_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  loaded the focus teacher with strict sample-index overlap `1.0` and loaded AIDT
  features with `9215/9215` coverage. Boundary RKD was active:
  `train_teacher_feature_rkd_loss=0.05643`, `train_teacher_feature_rkd_count=32`,
  and average `train_teacher_feature_rkd_pair_count=328.25`.
- Full validation eval
  `runs\eval_smoke_v8_yolof_aidtfeature_boundaryrkd010_val_20260702` reached
  macro/class-1 F1 `0.8801/0.6727`, class-1 P/R `0.6154/0.7417`, below the
  anchor `0.8847/0.6860`. Confusion was
  `[[485,47,1,2,14],[21,112,13,0,5],[5,10,514,9,6],[0,4,55,652,1],[8,9,4,1,628]]`.
  The pair gate reduced `0->1` to `47`, but class-1 recall stayed worse
  (`1->0=21`, `1->2=13`) and macro F1 remained below anchor.
- XAI selected12
  `runs\xai_smoke_v8_yolof_aidtfeature_boundaryrkd010_val_selected12_20260702`
  again stayed foreground-clean (`attention/grad_rollout/gradcam/rollout
  foreground mass = 0.9753/0.9714/0.9795/0.9506`). Background blur/gray remained
  near zero (`0.00059/-0.00007` original-prediction drop), while object
  desaturation was still dominant (`0.0704`). Reject before probe: boundary-gated
  AIDT feature RKD moves false-positive class-1 slightly, but it further weakens
  class-1 recall and does not create the needed surface/boundary separability.
  Do not repeat this exact `weight=0.01/source=head/distance-only/boundary
  pairs=0-1,1-2,1-4,2-3` setting on the current anchor.

## Audit 2026-07-02 - Patch-Source Boundary AIDT Feature RKD Smoke

- Tested the same boundary-gated AIDT feature RKD on the student `patch` source
  instead of the final head embedding, to target token-level surface cues while
  keeping the same strict train-only feature cache and focus-binary anchor
  recipe. Smoke
  `smoke_v8_yolof_aidtfeature_patchboundaryrkd010_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  loaded features with full `9215/9215` sample-index coverage and had active RKD:
  `train_teacher_feature_rkd_loss=0.06189`, `train_teacher_feature_rkd_count=32`,
  average `train_teacher_feature_rkd_pair_count=328.25`.
- Full validation eval
  `runs\eval_smoke_v8_yolof_aidtfeature_patchboundaryrkd010_val_20260702`
  reached macro/class-1 F1 `0.8808/0.6766`, class-1 P/R `0.6175/0.7483`, still
  below the anchor `0.8847/0.6860` and below the class-1 `0.70` probe gate.
  Confusion was
  `[[485,47,1,2,14],[20,113,12,0,6],[6,10,513,9,6],[0,4,55,652,1],[8,9,4,1,628]]`.
  It was slightly less harmful than head-boundary RKD but still did not solve
  class-1 recall/precision.
- XAI selected12
  `runs\xai_smoke_v8_yolof_aidtfeature_patchboundaryrkd010_val_selected12_20260702`
  stayed foreground-clean (`attention/grad_rollout/gradcam/rollout foreground
  mass = 0.9750/0.9712/0.9781/0.9507`). Background blur/gray was again near zero
  (`0.00058/-0.00003` original-prediction drop), and object desaturation remained
  dominant (`0.0699`). Reject before probe: simply moving AIDT relation RKD from
  head to patch does not add the missing class-1 surface/boundary signal. Do not
  repeat `weight=0.01/source=patch/distance-only/boundary pairs=0-1,1-2,1-4,2-3`
  on this anchor.

## Audit 2026-07-02 - Residual MLP Classification Head Smokes

- Researched whether a deeper decision head is a plausible architecture change
  before coding it. ViT's original paper notes that the classification head can be
  an MLP during pre-training (`https://arxiv.org/abs/2010.11929`), and MLP-Mixer
  shows MLP blocks are valid vision feature processors when sufficiently
  regularized (`https://arxiv.org/abs/2105.01601`). This made a small residual MLP
  head worth testing because AIDT's pretrained ResNet50+ViT-B/16 comparison uses a
  deeper `2816 -> 1024 -> 512 -> 5` classifier while TRKH's main head was a single
  `Linear(embed_dim, num_classes)`.
- Implemented `ResidualMLPClassificationHead` as a checkpoint-compatible head:
  existing `head.weight/head.bias` load directly from the old linear head, and
  only `head.residual_*` keys are new. The residual branch is zero-init, so
  predictions are unchanged at load time before training. Added CLI/config/launcher
  flags and a resume whitelist limited to `head.residual_`. Preflight passed:
  `py_compile` for `model.py/config.py/train.py` and focused pytest
  `tests/test_residual_mlp_classification_head.py tests/test_trainable_module_prefixes.py`
  (`7 passed`).
- Full-update smoke
  `smoke_v8_yolof_resmlphead512_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  loaded the old checkpoint with only `head.residual_*` missing and trained with
  the current teacher-focus-binary anchor recipe. Full validation eval
  `runs\eval_smoke_v8_yolof_resmlphead512_val_20260702` reached only
  macro/class-1 F1 `0.8732/0.6477`, class-1 P/R `0.5672/0.7550`, below the
  anchor `0.8847/0.6860`. Confusion was
  `[[477,55,2,1,14],[18,114,13,0,6],[3,12,516,7,6],[0,4,57,648,3],[6,16,4,1,623]]`.
  XAI selected12 stayed mostly foreground-focused
  (`attention/grad_rollout/gradcam/rollout foreground mass =
  0.9746/0.9710/0.9739/0.9505`) but kept high border flags
  (`Grad-CAM border 7/12`, rollout border 5/12) and background perturbations near
  zero (`background blur/gray 0.0006/-0.0000`). Object desaturation remained the
  dominant perturbation (`0.0782`). Reject: increasing head capacity while updating
  the backbone pushes extra false positives into class 1 (`0->1=55`, `4->1=16`).
- Residual-only head ablation
  `smoke_v8_yolof_resmlphead512_residualonly_lr5e4_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  froze everything except `head.residual_scale`, `head.residual_norm`, and
  `head.residual_mlp` (`397,318` trainable parameters). Full validation eval
  `runs\eval_smoke_v8_yolof_resmlphead512_residualonly_val_20260702` reached
  macro/class-1 F1 `0.8750/0.6551`, class-1 P/R `0.5825/0.7483`, still below
  the anchor and below the class-1 `0.70` gate. Confusion was
  `[[479,53,2,1,14],[18,113,14,0,6],[3,10,520,5,6],[0,4,60,645,3],[6,14,4,2,624]]`.
  XAI selected12 again showed foreground-clean but boundary-dominated behavior
  (`attention/grad_rollout/gradcam/rollout foreground mass =
  0.9630/0.9681/0.9638/0.9448`, `Grad-CAM border 7/12`, object desaturation
  `0.0736`, background blur/gray `0.0004/-0.0001`). Reject before probe:
  a deeper residual classifier on the current embeddings is not enough; the
  bottleneck is still the surface/boundary representation, not linear head
  capacity. Do not repeat residual MLP head full-update `hidden=512/dropout=0.08`
  or residual-only `lr=5e-4` on this anchor without a new representation signal.

## Cleanup 2026-07-02 - Legacy Detection/4-Class and Old Probe Artifacts

- Preserved the current no-pretrain keeper
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701`,
  the current-code validation/test audits, AIDT validation/test exports, soft
  ensemble audits, and the AIDT disagreement-gate audit before deleting anything.
- Deleted obsolete legacy detection/full-frame/4-class non-keepers recorded in
  `runs\cleanup_manifest_20260702_obsolete_legacy_detection_4class.json`: `16`
  directories, `4.496 GB` reclaimed. These included old `mango_hybrid_*`,
  `mango_detr_*`, and older object-crop/4-class artifacts that used old data,
  old class definitions, or obsolete detection framing. They should not be used as
  current `class_f`/`yolo_f` gates.
- Deleted old probe/benchmark/rejected-pretrain artifacts recorded in
  `runs\cleanup_manifest_20260702_legacy_probes_bench_pretrain_rejects.json`:
  `66` directories, `9.030 GB` reclaimed. This pass removed only
  `mango_cls_*probe*`, old `worker_bench_*`/`audit_perf_*`/`audit_smoke_train_*`,
  and rejected `pretrain_groupclean_*` directories. Current keeper/final/AIDT
  comparison directories were verified still present after deletion.
- D: free space after the second cleanup was about `70.26 GB`. Policy going
  forward: old legacy/detection/4-class metrics can remain as historical notes in
  docs, but they are not comparable baselines for the current
  classification-only `yolo_f`/`class_f` research loop. Keep checkpoints only when
  they are active keepers, reproducibility anchors, or final-audit comparisons.

## Audit 2026-07-02 - Self-Adaptive Target Loss Probe

- Researched noisy-label fine-grained learning before implementation. Self-adaptive
  training keeps per-sample target estimates as an EMA of model probabilities, so
  it was a distinct check from the earlier ELR run: SAT directly trains against the
  adaptive soft target instead of only regularizing it. Implemented train-only
  sample-index SAT state, CLI/config/launcher flags, history metrics, and focused
  tests. Preflight passed `py_compile` for `train.py/config.py` and focused pytest
  `tests/test_self_adaptive_target_loss.py
  tests/test_detection_calibration.py::DetectionCalibrationTests::test_early_learning_regularization_updates_target_state`
  (`4 passed`).
- Smoke
  `runs\smoke_v8_yolof_sat015_b90_hw10_teacherfocusbinary015_boundarydrop_bboxprior_rerun1_20260702`
  resumed the current keeper with `self_adaptive_target_loss_weight=0.015`,
  `beta=0.90`, `hard_weight=0.10`, and teacher-focus-binary `0.015`. Full-val
  macro/class-1 F1 reached `0.8847/0.6879`, class-1 P/R `0.6103/0.7881`. SAT was
  active (`loss=1.1878`, fraction `1.0`), but hard-agreement stayed `1.0`.
  XAI selected12 stayed foreground-clean
  (`attention/grad_rollout/gradcam/rollout foreground mass =
  0.9830/0.9691/0.9764/0.9458`), background blur/gray stayed near zero
  (`0.0001/-0.0004`), and object desaturation remained dominant (`0.0887`).
- Probe
  `runs\probe_v8_yolof_sat015_b90_hw10_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702`
  failed the gate. Epoch 1 reached val macro/class-1 F1 `0.8808/0.6743`, and the
  selected best checkpoint reached `0.8798/0.6760`, class-1 P/R
  `0.5845/0.8013`. Confusion was
  `[[477,58,2,0,12],[14,121,11,0,5],[6,13,511,6,8],[0,3,55,652,2],[9,12,3,1,625]]`,
  so SAT increased class-1 false positives (`0->1=58`, `4->1=12`) and reduced
  precision below the anchor.
- Probe XAI
  `runs\xai_probe_v8_yolof_sat015_b90_hw10_val_selected12_20260702` confirmed the
  same bottleneck: top confusions `0->1=56`, `3->2=55`, `2->1=13`, `4->1=12`;
  foreground mass stayed high (`attention/grad_rollout/gradcam/rollout =
  0.9876/0.9754/0.9850/0.9489`), background perturbations were negligible
  (`background_blur/background_gray pred-drop = 0.0007/0.0004`), and object
  desaturation was still the largest probe (`0.0914`). Reject before full train:
  this SAT setting mostly reinforces existing labels and does not add the needed
  surface/boundary signal. Do not repeat
  `weight=0.015/beta=0.90/hard_weight=0.10/start=2` on the current anchor.
- Cleanup after audit deleted the rejected SAT train directories and preserved
  both XAI directories plus current keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260702_sat015_reject_smoke_probe.json`, deleted `3`
  directories and reclaimed `364.30 MB`.

## Audit 2026-07-02 - Confusion-Pair Manifold Mixup Smoke

- Researched mixup-family methods before implementation. Classic mixup trains on
  convex combinations of samples and labels
  (`https://arxiv.org/abs/1710.09412`), Manifold Mixup moves this interpolation
  into hidden representations to smooth decision boundaries
  (`https://arxiv.org/abs/1806.05236`), and CP-Mix proposes selecting frequently
  confused class pairs for long-tailed recognition
  (`https://arxiv.org/abs/2411.07621`). For TRKH, image-space mixup is a bad fit
  because current classification-only crops still carry bbox/object metadata and
  earlier code explicitly disables mosaic/cutmix/mixup to preserve object
  semantics. I therefore implemented a head-input manifold variant over explicit
  confusion pairs only, without changing raw data or globally oversampling class 1.
- Implemented config/CLI/launcher support for `confusion_pair_mixup_loss_weight`,
  `alpha`, `pairs`, `max_pairs`, and `start_epoch`, plus history columns for loss,
  active fraction, pair count, and lambda mean. The loss samples within-batch
  pairs from `0-1,1-2,1-4,2-3`, mixes the extracted classification-head input,
  and applies soft-label CE through the existing head. Preflight passed
  `py_compile` for `trkh\training\train.py` and `trkh\core\config.py`, and
  focused pytest `tests/test_confusion_pair_mixup_loss.py
  tests/test_self_adaptive_target_loss.py` (`5 passed`). A launch bug was caught
  before useful training because validation referenced `args.num_classes`; it was
  fixed to use `expected_num_classes`. A second preflight run correctly rejected a
  teacher CSV whose sample-index paths belonged to `class_f` while the primary
  dataset was `yolo_f` (`sample_index_path_overlap_ratio=0`).
- Valid smoke
  `runs\smoke_v8_yolof_cpmix006_teacherfocusbinary015_boundarydrop_bboxprior_rerun2_20260702`
  resumed the current keeper with `confusion_pair_mixup_loss_weight=0.006`,
  `alpha=0.40`, `max_pairs=64`, pairs `0-1,1-2,1-4,2-3`, teacher-focus-binary
  `0.015`, and bbox spatial fusion. The auxiliary signal was active
  (`train_confusion_pair_mixup_loss=1.3871`, active fraction `0.7695`, average
  `24.625` mixed pairs/batch, lambda mean `0.8393`), but validation reached only
  macro/class-1 F1 `0.8805/0.6741`, class-1 P/R `0.5817/0.8013`, below the
  anchor `0.8847/0.6860`. Confusion was
  `[[479,56,3,0,11],[13,121,11,0,6],[3,12,516,6,7],[0,4,56,650,2],[9,15,3,1,622]]`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_cpmix006_val_selected12_20260702` confirmed the same
  failure mode: top confusions `0->1=57`, `3->2=56`, `4->1=15`, `1->0=14`,
  `2->1=12`; foreground mass stayed high
  (`attention/grad_rollout/gradcam/rollout = 0.9797/0.9663/0.9726/0.9315`),
  background perturbations stayed near zero (`background_blur/background_gray`
  original-prediction drop `0.0006/-0.0005`), and object desaturation remained the
  dominant perturbation (`0.1256`). Reject before probe: hidden-state mixing along
  noisy confusion pairs makes the class-1 decision region wider and increases
  false positives instead of adding a new surface/boundary cue. Do not repeat
  `confusion_pair_mixup_loss_weight=0.006/alpha=0.40/pairs=0-1,1-2,1-4,2-3` on
  the current anchor.
- Cleanup after audit deleted the two invalid CP-Mix preflight smoke directories
  and the valid rejected CP-Mix smoke train directory, while preserving the
  selected12 XAI audit and all keeper/final/AIDT comparison artifacts. Manifest:
  `runs\cleanup_manifest_20260702_cpmix006_reject_smokes.json`, deleted `3`
  directories and reclaimed `182.04 MB`; D: free space after cleanup was about
  `69.74 GB`.

## Audit 2026-07-02 - CNN Branch Token Smoke

- Researched lightweight CNN/ViT hybridization before implementation. ConViT
  argues that convolutional inductive bias can improve ViT sample efficiency when
  large external pretraining is not available (`https://arxiv.org/abs/2103.10697`),
  and CvT introduces convolutional token embedding/projection to combine CNN
  locality with transformer context (`https://arxiv.org/abs/2103.15808`). TRKH
  already has a CNN stem and a CNN fusion logit head, but the v8 launcher set
  `branch_cnn_tokens=0`, so the transformer prefix stream only received
  color/edge branch tokens. I tested a minimal checkpoint-compatible variant that
  adds one pooled CNN-stem branch token to the transformer sequence instead of
  changing the whole backbone.
- Implemented a narrow resume path for this architecture extension: when
  `branch_type_embed` grows from the old color/edge token count to include the new
  CNN token, old embeddings are copied into the matching prefix slots and only
  `branch_token_fusion.cnn_branch.*` keys are allowed to initialize from scratch.
  Added launcher parameter `BranchCnnTokens` and recorded it in preflight/resolved
  configs. Preflight passed with `branch_cnn_tokens=1`; `py_compile` passed for
  `model.py/train.py/config.py`; focused pytest
  `tests/test_branch_cnn_token_resume.py tests/test_confusion_pair_mixup_loss.py`
  passed (`3 passed`). The valid smoke confirmed partial load was restricted to
  `branch_token_fusion.cnn_branch.proj.*`.
- Smoke
  `runs\smoke_v8_yolof_branchcnn1_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  resumed the current teacher-focus-binary keeper with one CNN branch token,
  teacher-focus-binary `0.015`, bbox spatial fusion, 24 train batches, full
  validation, and final test skipped. It reached validation macro/class-1 F1
  `0.8841/0.6785`, class-1 P/R `0.6117/0.7616`, below the anchor
  `0.8847/0.6860`. Confusion was
  `[[490,47,1,1,10],[18,115,12,0,6],[3,11,518,5,7],[0,3,57,650,2],[9,12,2,1,626]]`.
  The new token slightly reduced `0->1` (`49 -> 47`) but increased class-1 false
  negatives (`1->0=18`, `1->2=12`), so precision stayed flat and recall dropped.
- XAI selected12
  `runs\xai_smoke_v8_yolof_branchcnn1_val_selected12_20260702` showed cleaner
  localization but no class-1 gain: foreground mass stayed high
  (`attention/grad_rollout/gradcam/rollout = 0.9834/0.9655/0.9821/0.9361`),
  border flags were lower than many prior smokes (`Grad-CAM border 2/12`, rollout
  border 3/12), background blur/gray were still negligible
  (`0.0008/0.0002` original-prediction drop), and object desaturation remained
  dominant (`0.1226`). Reject before probe: adding one CNN stem token improves
  the visual focus slightly but shifts class-1 errors toward false negatives and
  does not add the missing surface/boundary separability. Do not repeat
  `branch_cnn_tokens=1` on this anchor without a new training signal.
- Cleanup after audit deleted the rejected branch-CNN smoke train directory and
  preserved the selected12 XAI audit plus keeper/final/AIDT comparison artifacts.
  Manifest: `runs\cleanup_manifest_20260702_branchcnn1_reject_smoke.json`,
  deleted `1` directory and reclaimed `182.99 MB`; D: free space after cleanup
  was about `69.68 GB`.

## Diagnostic 2026-07-02 - Train-Holdout Class-1 Margin Guard

- Exported current anchor predictions on the train split only:
  `runs\eval_yolof_teacherfocusbinary015_best_train_20260702`. This produced
  train macro/class-1 F1 `0.9400/0.8164`, far above validation, so any threshold
  fit on these predictions is in-sample-biased and can only be used as a
  diagnostic, not as a gate.
- Re-tested the earlier class-1 margin-guard idea in a stricter train-only way:
  split train by `source_stem` into five deterministic holdout folds, fit a rule
  only on the remaining train folds, and apply the selected thresholds to val.
  The rule only acts when TRKH predicts class `1` and the top-2 rival is
  `0/3/4`, reassigning to that rival if the class-1 margin is small. Summary is
  saved at `runs\train_holdout_class1_guard_diagnostic_20260702\summary.json`.
- The best average train-holdout rule selected margin threshold `0.045` and
  `min_p1=0.0`. It improved in-sample train class-1 F1 from `0.8164` to
  `0.8333`, but validation reached only macro/class-1 F1 `0.8851/0.6817`,
  still below the no-pretrain anchor `0.8847/0.6860`. It made `34` val changes:
  `16` useful `0->1->0`, but also `9` harmful true class-1 `1->1->0` and `2`
  harmful `1->1->4`. Reject as a candidate for OOF expansion: even grouped
  train-holdout calibration cannot preserve class-1 recall on val.

## Audit 2026-07-02 - R-Drop Dropout Consistency Smoke

- Researched R-Drop as a lightweight regularizer for the observed train/validation
  gap. The original R-Drop paper proposes running each sample through two
  dropout-sampled submodels and minimizing bidirectional KL between the output
  distributions (`https://arxiv.org/abs/2106.14448`,
  `https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html`).
  This was a reasonable next test because the current anchor is very strong on
  train (`0.9400/0.8164` macro/class-1 F1) but much lower on val, while XAI keeps
  showing foreground-clean but surface/color-sensitive mistakes.
- Implemented optional `rdrop_loss_weight` and `rdrop_temperature` in
  `TrainConfig`, CLI, v8 launcher, train history, and focused tests. The loss is
  disabled by default and only runs in the classification path. It compares the
  final logits from the normal forward with a second train-mode forward on the
  same image/bbox metadata, using symmetric KL without detaching either branch.
  Preflight passed `py_compile` for `trkh\training\train.py` and
  `trkh\core\config.py`; focused pytest
  `tests/test_rdrop_loss.py tests/test_confusion_pair_mixup_loss.py
  tests/test_branch_cnn_token_resume.py` passed (`5 passed`).
- Smoke
  `runs\smoke_v8_yolof_rdrop010_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  resumed the current teacher-focus-binary keeper on `yolo_f` with
  `rdrop_loss_weight=0.10`, `rdrop_temperature=1.0`, teacher-focus-binary
  `0.015`, bbox spatial fusion, 24 train batches, full validation, and final test
  skipped. Resolved config confirmed the R-Drop flags and `history.csv` recorded
  active loss (`train_rdrop_loss=0.0020`). Validation regressed to macro/class-1
  F1 `0.8781/0.6648`, class-1 P/R `0.5681/0.8013`, below the anchor
  `0.8847/0.6860`. Confusion was
  `[[478,59,3,0,9],[13,121,11,0,6],[3,13,515,6,7],[0,4,57,649,2],[9,16,3,1,621]]`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_rdrop010_val_selected12_20260702` confirmed the same
  failure mode rather than a background issue: top confusions were `0->1=59`,
  `3->2=57`, `4->1=16`, `1->0=13`, `2->1=13`; foreground mass stayed high
  (`attention/grad_rollout/gradcam/rollout = 0.9796/0.9661/0.9727/0.9317`),
  background blur/gray remained negligible (`0.0006/-0.0004` original-prediction
  drop), and object desaturation was still dominant (`0.1262`). Reject before
  probe: R-Drop stabilizes dropout output but, on this anchor, mainly broadens the
  class-1 region and worsens false positives. Do not repeat
  `rdrop_loss_weight=0.10/temperature=1.0` on the current anchor.
- Cleanup after audit deleted the rejected R-Drop smoke train directory and
  preserved the selected12 XAI audit plus keeper/final/AIDT comparison artifacts.
  Manifest: `runs\cleanup_manifest_20260702_rdrop010_reject_smoke.json`, deleted
  `1` directory and reclaimed `180.05 MB`; D: free space after cleanup was about
  `69.61 GB`.

## Audit 2026-07-02 - Semantic Attribute Hierarchical Loss Smoke

- Researched auxiliary-task regularization for fine-grained classification before
  implementation. CO-TASK proposes creating auxiliary tasks from combinations of
  existing task labels without extra labeling and reports robustness to some label
  noise (`https://cdn.aaai.org/ojs/16949/16949-13-20443-1-2-20210518.pdf`).
  Fine-grained multi-task work also uses related auxiliary tasks to regularize
  subtle visual classification (`https://cvit.iiit.ac.in/images/ConferencePapers/2016/anoop_Fine-Grained.pdf`).
  For TRKH, I tested a checkpoint-safe logit-level hierarchy instead of adding a
  new head: group class probabilities with `logsumexp` and apply CE to semantic
  groups derived from the hard class label.
- Implemented optional `semantic_attribute_loss_weight` and
  `semantic_attribute_specs` in `TrainConfig`, CLI, v8 launcher, train history,
  and focused tests. Default specs were
  `maturity:0,1|2,3|4;transport:0,2|1|3,4;quality:0,1,2|3|4`, so no raw label or
  dataset artifact was changed. Preflight passed `py_compile` for
  `trkh\training\train.py` and `trkh\core\config.py`; focused pytest
  `tests/test_semantic_attribute_loss.py tests/test_rdrop_loss.py
  tests/test_confusion_pair_mixup_loss.py tests/test_branch_cnn_token_resume.py`
  passed (`8 passed`).
- Smoke
  `runs\smoke_v8_yolof_semattr020_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  resumed the current keeper with `semantic_attribute_loss_weight=0.02`, the
  default three attribute specs, teacher-focus-binary `0.015`, bbox spatial
  fusion, 24 train batches, full validation, and final test skipped. The auxiliary
  signal was active (`train_semantic_attribute_loss=0.7272`,
  `train_semantic_attribute_tasks=3.0`), but validation reached only
  macro/class-1 F1 `0.8796/0.6685`, class-1 P/R `0.5735/0.8013`, below the anchor
  `0.8847/0.6860`. Confusion was
  `[[480,57,3,0,9],[13,121,11,0,6],[3,13,515,6,7],[0,4,56,650,2],[9,16,3,1,621]]`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_semattr020_val_selected12_20260702` again showed no
  background bottleneck: top confusions `0->1=57`, `3->2=56`, `4->1=16`,
  `1->0=13`, `2->1=13`; foreground mass stayed high
  (`attention/grad_rollout/gradcam/rollout = 0.9795/0.9663/0.9724/0.9315`),
  background blur/gray remained negligible (`0.0006/-0.0004` original-prediction
  drop), and object desaturation dominated (`0.1249`). Reject before probe:
  semantic grouping regularizes coarse structure but still broadens/sustains the
  class-1 false-positive region. Do not repeat `semantic_attribute_loss_weight=0.02`
  with specs `maturity:0,1|2,3|4;transport:0,2|1|3,4;quality:0,1,2|3|4` on this
  anchor.
- Cleanup after audit deleted the rejected semantic-attribute smoke train
  directory and preserved the selected12 XAI audit plus keeper/final/AIDT
  comparison artifacts. Manifest:
  `runs\cleanup_manifest_20260702_semattr020_reject_smoke.json`, deleted `1`
  directory and reclaimed `180.09 MB`; D: free space after cleanup was about
  `69.55 GB`.

## Audit 2026-07-02 - GrabCut Foreground Crop Smoke

- Researched foreground segmentation/object-centric classification before this
  test. GrabCut uses iterative graph-cut segmentation from a loose foreground
  prior (`https://dl.acm.org/doi/10.1145/1015706.1015720`,
  `https://pub.ista.ac.at/~vnk/papers/grabcut_siggraph04.pdf`), and
  object-centric pooling/classification work argues that localizing object
  regions can help when scene context is distracting
  (`https://ai.stanford.edu/~olga/papers/eccv12-OCP.pdf`). For TRKH this was a
  cheap train-time transform because `yolo_f` already supplies object boxes, but
  it was intentionally kept as augmentation only and did not modify raw data.
- Smoke
  `runs\smoke_v8_yolof_grabcutcrop_m04_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  resumed the current teacher-focus-binary keeper with `ForegroundCropMode=grabcut`,
  probability `1.0`, margin `0.04`, min/max mask area `0.03/0.92`, max crop area
  `0.85`, teacher-focus-binary `0.015`, bbox spatial fusion, 24 train batches,
  full validation, architecture trace, and final test skipped. Validation reached
  only macro/class-1 F1 `0.8792/0.6667`, class-1 P/R `0.5742/0.7947`, below the
  anchor `0.8847/0.6860`. Confusion was
  `[[481,56,3,0,9],[14,120,11,0,6],[3,13,515,6,7],[0,4,56,650,2],[9,16,3,1,621]]`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_grabcutcrop_m04_val_selected12_20260702` confirmed the
  same bottleneck rather than a missing background filter: top confusions were
  `0->1=56`, `3->2=56`, `4->1=16`, `1->0=14`, and `2->1=13`; foreground mass
  stayed acceptable (`attention/grad_rollout/gradcam/rollout =
  0.9574/0.9643/0.9146/0.9246`), but border flags remained high
  (`Grad-CAM border 8/12`, rollout border 5/12). Background blur/gray still had
  negligible or negative original-prediction drop (`-0.0012/-0.0053`), while
  object desaturation dominated (`0.1219`). Reject before probe: forcing a tighter
  GrabCut crop increases class-1 false positives and does not add a useful
  surface/boundary cue. Do not repeat this exact foreground-crop setting on the
  current anchor.
- Cleanup after audit deleted the rejected GrabCut smoke train directory and
  preserved selected12 XAI plus keeper/final/AIDT artifacts. Manifest:
  `runs\cleanup_manifest_20260702_grabcutcrop_m04_reject_smoke.json`, deleted
  `1` directory and reclaimed `180.06 MB`; D: free space after cleanup was about
  `69.48 GB`.

## Audit 2026-07-02 - Illumination Consistency Smoke

- Researched illumination robustness before implementation. AugMix uses
  augmentation consistency/Jensen-Shannon style regularization to improve
  corruption robustness (`https://arxiv.org/abs/1912.02781`), and color-constancy
  work studies how object color perception can stay stable under changing
  illuminants (`https://pmc.ncbi.nlm.nih.gov/articles/PMC8976922/`). Fruit
  ripeness literature also treats illumination variation as a real source of
  brittleness for visual ripeness grading
  (`https://e-archivo.uc3m.es/entities/publication/4af224c6-635f-4725-b782-ba5d0a66e11a`).
  Because TRKH XAI repeatedly shows object color/surface sensitivity but not far
  background sensitivity, I implemented a conservative tensor-level
  brightness/contrast/gamma consistency loss without hue/saturation changes and
  without modifying raw data.
- Implemented optional `illumination_consistency_loss_weight`, probability,
  brightness, contrast, gamma, and temperature in `TrainConfig`, CLI, v8 launcher,
  train history, and focused tests. The helper denormalizes normalized RGB tensors,
  samples light brightness/contrast/gamma perturbations, normalizes back, and adds
  symmetric KL between original and illumination-perturbed logits for a sampled
  subset. Preflight passed `py_compile` for `trkh\training\train.py` and
  `trkh\core\config.py`; focused pytest
  `tests/test_illumination_consistency_loss.py tests/test_rdrop_loss.py
  tests/test_semantic_attribute_loss.py tests/test_confusion_pair_mixup_loss.py
  tests/test_branch_cnn_token_resume.py` passed (`10 passed`).
- Smoke
  `runs\smoke_v8_yolof_illumcons020_p50_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  resumed the current keeper with `illumination_consistency_loss_weight=0.02`,
  probability `0.50`, brightness/contrast/gamma `0.08/0.08/0.12`,
  teacher-focus-binary `0.015`, bbox spatial fusion, 24 train batches, full
  validation, architecture trace, and final test skipped. The auxiliary signal was
  active (`train_illumination_consistency_loss=0.0043`, fraction `0.5143`), but
  validation reached only macro/class-1 F1 `0.8796/0.6685`, class-1 P/R
  `0.5735/0.8013`, below the anchor `0.8847/0.6860`. Confusion was
  `[[480,57,3,0,9],[13,121,11,0,6],[3,13,515,6,7],[0,4,56,650,2],[9,16,3,1,621]]`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_illumcons020_p50_val_selected12_20260702` showed clean
  foreground focus but no class-1 fix: top confusions `0->1=57`, `3->2=56`,
  `4->1=16`, `1->0=13`, `2->1=13`; foreground mass stayed high
  (`attention/grad_rollout/gradcam/rollout = 0.9799/0.9667/0.9733/0.9317`),
  background blur/gray remained negligible (`0.0008/-0.0006` original-prediction
  drop), and object desaturation remained dominant (`0.1256`). Reject before
  probe: light illumination invariance does not address the class-1
  surface/boundary ambiguity and still broadens the class-1 false-positive region.
  Do not repeat `weight=0.02/probability=0.50/brightness=0.08/contrast=0.08/gamma=0.12`
  on the current anchor.
- Cleanup after audit deleted the rejected illumination-consistency smoke train
  directory and preserved selected12 XAI plus keeper/final/AIDT artifacts.
  Manifest: `runs\cleanup_manifest_20260702_illumcons020_p50_reject_smoke.json`,
  deleted `1` directory and reclaimed `180.19 MB`; D: free space after cleanup was
  about `69.39 GB`.

## Diagnostic 2026-07-02 - HSV/Lab/Texture Color Guard

- Ran a train/val diagnostic to test whether hand-crafted object-crop color and
  surface statistics contain a usable signal that TRKH is missing. Features were
  extracted from the same `yolo_f` object-crop logic with no raw data changes:
  HSV histograms/statistics, Lab histograms/statistics, Sobel magnitude/orientation
  histograms, and coarse 4x4 color-layout stats. Artifact:
  `runs\hsv_lab_texture_diagnostic_20260702`, size about `9.51 MB`.
- Standalone color/texture classifiers were far below the TRKH anchor. Best
  standalone model was ExtraTrees with validation macro/class-1 F1
  `0.8423/0.5498`; Logistic/RF variants stayed at class-1 F1 `0.439-0.529`.
  They can fix some TRKH class-1 false positives but create many harms and lose
  class-1 recall, so a direct branch/router from these features is not a strong
  candidate.
- A stricter 5-fold `StratifiedGroupKFold` train-only OOF guard was tested: train
  OOF color probabilities fit the threshold, then a final color model was applied
  to validation only when TRKH predicted class `1`. ExtraTrees selected threshold
  `0.05` and changed only `2` validation samples, reaching macro/class-1 F1
  `0.8841/0.6822`; RandomForest changed `3` validation samples and reached
  `0.8847/0.6842`. Both remain below the historical keeper class-1 F1 `0.6860`
  and below the next gate `0.70`.
- Conclusion: hand-crafted HSV/Lab/texture signal is complementary only at the
  margin and too weak to justify another model branch or post-hoc router. Do not
  expand this into a train-in-sample color router or a shallow color-stat branch
  unless a new feature set shows materially stronger OOF class-1 gains.

## Cleanup 2026-07-02 - Legacy Mango-Cls Large Runs

- Preserved the current no-pretrain keeper, final audits, AIDT comparison, soft
  ensemble comparison, and AIDT disagreement-gate audit before deleting anything.
  The remaining run directory above `200 MB` is the current keeper
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701`.
- Deleted `19` obsolete `mango_cls_*` legacy full/probe-era directories that are
  not valid gates for the current `class_f`/`yolo_f` research split. Their metrics
  are retained only as historical notes in this journal/skill; they should not be
  used as current evidence or resumed for new probes.
- Cleanup manifest:
  `runs\cleanup_manifest_20260702_legacy_mango_cls_large_runs.json`. Reclaimed
  `6721.93 MB`; D: free space after cleanup was about `75.97 GB`.

## Audit 2026-07-02 - BBox Object-Erasure Negative Smoke

- Researched object/patch masking before the run. Cutout masks input regions as a
  simple regularizer (`https://arxiv.org/abs/1708.04552`), Random Erasing uses
  randomly erased rectangles to improve occlusion robustness
  (`https://arxiv.org/abs/1708.04896`), Hide-and-Seek hides patches so models
  must use multiple object parts (`https://arxiv.org/abs/1811.02545`), and
  foreground/background masking has been used to reduce background bias
  (`https://openaccess.thecvf.com/content/ICCV2023W/OODCV/papers/Aniraj_Masking_Strategies_for_Background_Bias_Removal_in_Computer_Vision_Models_ICCVW_2023_paper.pdf`).
  For TRKH I used the existing train-only `yolo_f` bbox object-erasure negative
  loss instead of changing raw data: erase the crop bbox region and push the
  erased-view logits toward uniform so the model cannot stay confident from
  background/context alone.
- Preflight passed `py_compile` for `trkh\training\train.py`,
  `trkh\core\config.py`, and `trkh\models\model.py`; focused pytest
  `tests\test_yolo_bbox_spatial_fusion.py tests\test_focused_false_positive_margin.py`
  passed (`16 passed`).
- Smoke
  `runs\smoke_v8_yolof_objerase006_p50_blur_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  resumed the current keeper with `bbox_object_erasure_negative_loss_weight=0.006`,
  probability `0.50`, margin `0.04`, fill `blur`, teacher-focus-binary `0.015`,
  bbox spatial fusion, 24 train batches, full validation, architecture trace, and
  final test skipped. The auxiliary signal was active
  (`train_bbox_object_erasure_negative_loss=0.0472`, fraction `0.4909`,
  erased max probability `0.3172`), but validation reached only macro/class-1 F1
  `0.8783/0.6667`, class-1 P/R `0.5708/0.8013`, below the anchor
  `0.8847/0.6860`. Confusion was
  `[[477,58,3,0,11],[13,121,11,0,6],[3,13,515,6,7],[0,4,56,650,2],[9,16,3,1,621]]`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_objerase006_p50_blur_val_selected12_20260702` showed
  the same pattern as earlier background/context tests: top confusions were
  `0->1=58`, `3->2=56`, `4->1=16`, `1->0=13`, and `2->1=13`; foreground mass
  stayed acceptable (`attention/grad_rollout/gradcam/rollout =
  0.9223/0.9753/0.8863/0.9566`), but border flags remained high
  (`attention border 11/12`, rollout border 4/12). Background blur/gray had
  negligible original-prediction drop (`0.00075/0.00086`), while object
  desaturation was still much larger (`0.0709`). Reject before probe: erasing the
  object correctly makes erased views less confident, but it broadens the class-1
  false-positive region instead of adding a useful boundary/surface signal.
- Cleanup after audit deleted the rejected smoke train directory and preserved
  selected12 XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260702_objerase006_p50_blur_reject_smoke.json`,
  deleted `1` directory and reclaimed `180.16 MB`; D: free space after cleanup was
  about `75.90 GB`.

## Audit 2026-07-02 - Sample-Index Class-Cartography GroupDRO Smoke

- Researched GroupDRO and dataset cartography before this run. GroupDRO optimizes
  the worst predefined group loss and the original neural-network study stresses
  that regularization/early stopping matter for worst-group generalization
  (`https://arxiv.org/abs/1911.08731`). Dataset Cartography uses model training
  dynamics to separate easy, ambiguous, and hard examples
  (`https://arxiv.org/abs/2009.10795`). For TRKH I kept the signal train-only:
  a `yolo_f` object-level manifest grouped samples by true class plus current
  anchor train-pred cartography bucket, with quality columns recorded only for
  audit.
- Fixed quality-group plumbing before training: `_load_quality_group_manifest`
  and `QualityGroupDataset` now support `sample_index` keys, so multi-object
  `yolo_f` images do not collapse to one path-level group. Preflight passed
  `py_compile` for `trkh\data\dataset.py`, `trkh\training\train.py`, and
  `trkh\tools\build_quality_group_manifest.py`; focused pytest
  `tests\test_quality_group_manifest.py tests\test_data_centric_sample_weights.py
  tests\test_teacher_probability_dataset.py` passed (`7 passed`).
- Built train-only manifest
  `runs\quality_group_yolof_anchor_classcart_train_20260702\quality_groups_train_sampleindex.csv`
  from `9215/9215` `yolo_f` train object samples and the current anchor train
  predictions. Loader audit in smoke confirmed `sample_indices=9215`,
  `matched_samples=9215`, `paths=0`, `duplicate_sample_index_rows=0`, and
  `num_groups=23`. Group counts included `c1_medium=390`, `c1_ambiguous_boundary=90`,
  `c1_easy=60`, and `c1_hard_low_self=1`; this confirms class-1 hard groups are
  small and noisy, not a large clean reservoir.
- Smoke
  `runs\smoke_v8_yolof_groupdro015_classcart_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  resumed the current keeper with `GroupDroLossWeight=0.015`,
  `GroupDroTemperature=0.30`, `GroupDroMinSamples=1`, teacher-focus-binary
  `0.015`, bbox spatial fusion, 24 train batches, full validation, architecture
  trace, and final test skipped. GroupDRO was active
  (`train_group_dro_loss=5.5635`, observed groups/batch `8.125`, worst group loss
  `5.6165`), but validation reached only macro/class-1 F1 `0.8787/0.6648`,
  class-1 P/R `0.5714/0.7947`, below the anchor `0.8847/0.6860`. Confusion was
  `[[480,57,3,0,9],[14,120,11,0,6],[3,13,515,6,7],[0,4,56,650,2],[9,16,3,1,621]]`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_groupdro015_classcart_val_selected12_20260702` again
  showed that the failure is not background/context: top confusions were
  `0->1=57`, `3->2=56`, `4->1=16`, `1->0=14`, and `2->1=13`; foreground mass
  stayed acceptable (`attention/grad_rollout/gradcam/rollout =
  0.9222/0.9753/0.8836/0.9569`), background blur/gray original-prediction drop
  stayed near zero (`0.00079/0.00071`), and object desaturation was much larger
  (`0.0718`). Reject before probe: train-pred cartography GroupDRO amplifies the
  same noisy boundary groups and does not add a new local surface cue.
- Cleanup after audit deleted the rejected smoke train directory and preserved the
  train-only quality manifest, selected12 XAI, and keeper/final artifacts.
  Manifest:
  `runs\cleanup_manifest_20260702_groupdro015_classcart_reject_smoke.json`,
  deleted `1` directory and reclaimed `180.20 MB`; D: free space after cleanup was
  about `75.82 GB`.

## Audit 2026-07-02 - Foreground SnapMix Smoke

- Researched fine-grained image mixing before this run. SnapMix argues that
  pixel-area labels can be noisy for fine-grained recognition and instead uses
  semantic/CAM composition (`https://arxiv.org/abs/2012.04846`). CutMix motivates
  cut-paste region mixing with area-weighted labels
  (`https://arxiv.org/abs/1905.04899`), and SaliencyMix selects representative
  salient source patches (`https://arxiv.org/abs/2006.01791`). For TRKH I did not
  modify raw data: implemented a train-time auxiliary foreground SnapMix view that
  pastes bbox-constrained patches between in-batch confusion pairs and builds soft
  targets from source/target bbox semantic area, while the main teacher-focus
  forward still uses the original image.
- Implementation added optional config/launcher flags and tests:
  `foreground_snapmix_loss_weight`, probability, alpha, pair list, area range, and
  bbox margin. Preflight passed `py_compile` for `trkh\training\train.py` and
  `trkh\core\config.py`; focused pytest
  `tests\test_foreground_snapmix.py tests\test_confusion_pair_mixup_loss.py
  tests\test_yolo_bbox_spatial_fusion.py` passed (`12 passed`).
- Smoke
  `runs\smoke_v8_yolof_fgsnapmix015_p50_teacherfocusbinary015_boundarydrop_bboxprior_20260702`
  resumed the current keeper with `ForegroundSnapmixLossWeight=0.015`,
  probability `0.50`, pairs `0-1,1-2,1-4,2-3`, bbox area `0.06-0.18`,
  teacher-focus-binary `0.015`, bbox spatial fusion, full validation, architecture
  trace, and final test skipped. Because launcher epochs were not reduced, the
  smoke ran 15 short epochs with 24 train batches/epoch and early-stopped.
  Foreground SnapMix was active: near the best epoch it mixed about `4.3-5.4`
  samples/batch, source target weight about `0.116`, and target removed area about
  `0.116`.
- Best validation was still below the keeper and below gate: best macro/class-1 F1
  `0.8844/0.6838`, class-1 P/R `0.6000/0.7947`, accuracy `0.9194`. Confusion was
  `[[485,52,2,0,10],[14,120,11,0,6],[3,10,520,4,7],[0,3,56,649,4],[9,15,3,1,622]]`.
  It reduced neither the class-1 gate gap nor the persistent false positives
  enough: `0->1=52`, `4->1=15`, `1->0=14`, `1->2=11`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_fgsnapmix015_p50_val_selected12_20260702` confirmed the
  same bottleneck. Foreground mass stayed high
  (`attention/grad_rollout/gradcam/rollout = 0.9849/0.9703/0.9737/0.9380`), while
  background blur/gray remained negligible (`0.0005/-0.0001` original-prediction
  drop). Object desaturation was much larger (`0.1086`), and flags included
  `object_color_sensitive=6/12`, `gradcam_border_attention=6/12`, and
  `rollout_border_attention=5/12`. Reject before probe: bbox-constrained semantic
  patch mixing is active but behaves like another smoothing/augmentation signal
  over noisy boundaries rather than adding a reliable class-1 surface cue.
- Cleanup after audit deleted the rejected smoke train directory and preserved the
  selected12 XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260702_fgsnapmix015_p50_reject_smoke.json`, deleted `1`
  directory and reclaimed `182.15 MB`.

## Audit 2026-07-02 - Cross-Entropy Classification-Loss Ablation Smoke

- Rationale: after many boundary losses widened class-1 false positives, I tested
  whether removing the current non-CE classification shaping would tighten the
  class-1 decision region. This was a checkpoint-resume ablation only; raw data
  was not changed.
- Fixed the v8 launcher validation list so `ClassificationLoss` accepts
  `cross_entropy`/`ce`, matching the Python parser. Smoke
  `runs\smoke_v8_yolof_ce_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260702`
  resumed the current keeper with `ClassificationLoss=cross_entropy`,
  teacher-focus-binary `0.015`, bbox spatial fusion, 24 train batches, 1 epoch,
  full validation, architecture trace, and final test skipped.
- Validation was clearly worse than the keeper: macro/class-1 F1
  `0.8778/0.6611`, class-1 P/R `0.5694/0.7881`, accuracy `0.9152`.
  Confusion was
  `[[481,56,3,0,9],[14,119,12,0,6],[3,13,515,6,7],[0,4,56,650,2],[9,17,3,1,620]]`.
  CE widened class-1 false positives (`0->1=56`, `4->1=17`) and also kept
  `1->0=14`, `1->2=12`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_ce_teacherfocusbinary015_val_selected12_20260702`
  matched the previous failure pattern. Top confusions were `3->2=58`,
  `0->1=56`, `4->1=17`, `1->0=14`, `2->1=13`, and `1->2=12`. Foreground mass
  stayed high (`attention/grad_rollout/gradcam/rollout =
  0.9794/0.9657/0.9728/0.9320`), while background blur/gray original-prediction
  drop was negligible (`0.0009/-0.0004`). Object desaturation was much larger
  (`0.1285`), with `object_color_sensitive=7/12` and
  `gradcam_border_attention=6/12`. Reject before probe: plain CE removes useful
  boundary shaping and makes class 1 less precise; the bottleneck remains
  object-level surface/semantic ambiguity, not background.
- Cleanup after audit deleted the rejected smoke train directory and preserved the
  selected12 XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260702_ce_loss_reject_smoke.json`, deleted `1`
  directory and reclaimed `180.17 MB`; D: free space after cleanup was about
  `75.66 GB`.

## Audit 2026-07-02 - Quantized Label CPU/PU Ambiguous-Hard-Label Smoke

- Researched ambiguous hard labels and fine-grained noisy-label learning before
  this run. Quantized Label Learning frames a hard label as a quantized sample of
  an underlying soft class distribution and proposes class-wise positive-unlabeled
  risk for ambiguous labels (`https://arxiv.org/html/2501.01844v2`). The LNL-FG
  fine-grained noisy-label study also shows that generic label-noise methods can
  fail on fine-grained ambiguity
  (`https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wei_Fine-Grained_Classification_With_CVPR_2023_supplemental.pdf`).
  I therefore implemented a small train-time auxiliary CPU/PU-style loss without
  changing raw data: the target class is treated as positive and the other selected
  boundary classes are treated as unlabeled rather than strict negatives.
- Implementation added config, parser, launcher, metrics, and tests for
  `quantized_label_cpu_loss_weight`, selected classes, prior clamps, negative
  weight, non-negative risk floor, and start epoch. Preflight passed
  `py_compile` for `trkh\training\train.py` and `trkh\core\config.py`; focused
  pytest `tests\test_quantized_label_cpu_loss.py tests\test_foreground_snapmix.py
  tests\test_semantic_attribute_loss.py` passed (`8 passed`). The v8 launcher
  dry-run also resolved all new flags correctly.
- Smoke
  `runs\smoke_v8_yolof_qlcpu010_nw07_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260702`
  resumed the current keeper with `QuantizedLabelCpuLossWeight=0.01`, selected
  classes `0,1,2,4`, prior clamp `0.02-0.60`, negative weight `0.7`,
  teacher-focus-binary `0.015`, bbox spatial fusion, 24 train batches, 1 epoch,
  full validation, architecture trace, and final test skipped. The auxiliary loss
  was active (`train_quantized_label_cpu_loss=0.3140`, class count `4.0`,
  prior/positive fraction `0.1999`, negative risk `0.1903`), but validation was
  still below the keeper: macro/class-1 F1 `0.8818/0.6761`, class-1 P/R
  `0.5882/0.7947`, accuracy `0.9175`. Confusion was
  `[[485,52,3,0,9],[14,120,11,0,6],[3,12,516,6,7],[0,4,57,649,2],[9,16,3,1,621]]`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_qlcpu010_nw07_val_selected12_20260702` showed the
  same bottleneck as CE, SnapMix, and GroupDRO. Top confusions were `3->2=57`,
  `0->1=55`, `4->1=16`, `1->0=14`, `2->1=12`, and `1->2=11`. Foreground mass
  stayed high (`attention/grad_rollout/gradcam/rollout =
  0.9793/0.9657/0.9721/0.9321`), background blur/gray original-prediction drop
  stayed negligible (`0.0006/-0.0006`), and object desaturation remained dominant
  (`0.1249`). Flags were `object_color_sensitive=7/12`,
  `gradcam_border_attention=6/12`, and `rollout_border_attention=4/12`.
- Reject before probe: CPU/PU ambiguity risk is implemented and behaves as
  intended, but on this anchor it only softens hard-negative pressure around the
  same object-surface boundary. It does not add a new discriminative cue, does not
  improve class-1 precision/recall beyond the keeper, and repeats the foreground
  clean/background-insensitive XAI pattern.
- Cleanup after audit deleted the rejected smoke train directory and preserved the
  selected12 XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260702_qlcpu010_nw07_reject_smoke.json`, deleted `1`
  directory and reclaimed `180.29 MB`; D: free space after cleanup was about
  `75.60 GB`.

## Audit 2026-07-02 - FFVT-Lite Intermediate Layer-Token Fusion Smoke

- Researched fine-grained transformer token selection before this run. FFVT notes
  that deep CLS tokens can become too global and proposes selecting important
  tokens from multiple transformer layers to compensate local/low-level detail
  (`https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/0685.pdf`).
  TransFG likewise uses transformer attention links as indicators for
  discriminative regions in fine-grained recognition
  (`https://www.cs.jhu.edu/~alanlab/Pubs22/he22transfg.pdf`). For TRKH I kept the
  adaptation lightweight and checkpoint-safe: no raw data changes and no new
  trainable parameters; when enabled, `forward_features` collects top-k patch
  tokens from selected intermediate layers by CLS attention plus optional
  foreground/bbox priors, normalizes them, averages the selected layer features,
  and blends the result into the classification head input after fine-grained
  pooling.
- Implementation added `layer_token_fusion` model config/CLI/launcher flags:
  selected layers, top-k, blend, attention temperature, bbox weight, and
  foreground weight. Preflight passed `py_compile` for `trkh\models\model.py`,
  `trkh\training\train.py`, and `trkh\core\config.py`; focused pytest
  `tests\test_multi_granularity_aux_heads.py` passed (`5 passed`). Launcher
  dry-run confirmed `hard_sample_manifest=""`, `skip_final_test=true`, and
  `layer_token_fusion=true`.
- Smoke
  `runs\smoke_v8_yolof_layerfusion_ffvtlite_b12_top4_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260702`
  resumed the current keeper with layer fusion layers `2,4,6`, top-k `4`, blend
  `0.12`, bbox/foreground weights `0.20/0.10`, teacher-focus-binary `0.015`,
  bbox spatial fusion, 24 train batches, 1 epoch, full validation, architecture
  trace, and final test skipped. Architecture trace completed and confirmed the
  layer-token-fusion config.
- Validation was below the keeper and below the class-1 probe gate:
  macro/class-1 F1 `0.8798/0.6740`, class-1 P/R `0.5748/0.8146`, accuracy
  `0.9152`. Confusion was
  `[[476,60,2,1,10],[11,123,11,0,6],[3,10,518,6,7],[0,4,58,649,1],[9,17,4,1,619]]`.
  Layer fusion increased class-1 recall but widened false positives, especially
  `0->1=60` and `4->1=17`, worse than the keeper and worse than the current
  baseline failure pattern.
- XAI selected12
  `runs\xai_smoke_v8_yolof_layerfusion_ffvtlite_b12_top4_val_selected12_20260702`
  confirmed the regression. Top confusions were `0->1=60`, `3->2=58`,
  `4->1=17`, `1->0=11`, and `1->2=11`. Raw attention and Grad-CAM became more
  diffuse (`attention/gradcam foreground mass = 0.8785/0.8741`, background mass
  `0.1215/0.1259`, attention border `0.3065`), with flags
  `attention_border_attention=11/12`, `attention_background_attention=6/12`,
  and `gradcam_background_attention=3/12`. Background blur/gray still had near
  zero original-prediction drop (`0.0005/-0.0003`), while object desaturation
  remained much larger (`0.1125`).
- Reject before probe: FFVT-lite multi-layer token fusion is implemented and
  traceable, but this exact setting makes the model less localized and expands the
  class-1 decision region. Do not run a 2e probe or full train from this smoke.
- Cleanup after audit deleted the rejected smoke train directory and preserved the
  selected12 XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260702_layerfusion_ffvtlite_reject_smoke.json`,
  deleted `1` directory and reclaimed `180.19 MB`; D: free space after cleanup was
  about `75.53 GB`.

## Audit 2026-07-02 - Foreground Chroma Consistency Smoke

- Rationale from audit: repeated XAI runs show background blur/gray has near-zero
  effect while object desaturation remains a large perturbation. I researched
  color augmentation/color constancy before implementing this run: color jitter is
  commonly used to vary brightness/contrast/hue/saturation for robustness
  (`https://wiki.cloudfactory.com/docs/mp-wiki/augmentations/color-jitter`), but
  mango quality/ripeness studies also use color as a direct grading signal
  (`https://thesai.org/Downloads/Volume16No10/Paper_59-Quality_Classification_of_Harumanis_Mango_Based_on_External_Multi_Parameter.pdf`).
  Therefore the TRKH adaptation was deliberately weak and foreground-bounded:
  perturb chroma only inside the existing bbox, use a small KL consistency weight,
  and do not modify raw data.
- Implementation added `foreground_chroma_consistency_*` TrainConfig/CLI/v8
  launcher flags plus a bbox-masked tensor sampler. The perturbation uses a
  luminance/chroma decomposition, scales saturation lightly, rotates chroma by a
  small hue angle, blends only the bbox foreground back into the normalized image,
  and computes symmetric KL against the clean logits. Preflight passed
  `py_compile trkh\training\train.py trkh\core\config.py` and focused pytest
  `tests\test_illumination_consistency_loss.py` (`4 passed`). Launcher dry-run
  confirmed the new flags in `resolved_config` and `hard_sample_manifest=""`.
- Smoke
  `runs\smoke_v8_yolof_fgchroma012_p50_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260702`
  resumed the current keeper with `weight=0.012`, probability `0.50`,
  saturation delta `0.12`, hue delta `0.012`, bbox margin `0.02`,
  teacher-focus-binary `0.015`, bbox spatial fusion, 24 train batches, 1 epoch,
  full validation, architecture trace, and final test skipped. The auxiliary loss
  was active (`train_foreground_chroma_consistency_loss=0.00327`,
  fraction `0.4987`, mask fraction `0.6843`).
- Validation regressed below the keeper and below the probe gate:
  macro/class-1 F1 `0.8786/0.6648`, class-1 P/R `0.5749/0.7881`, accuracy
  `0.9156`. Confusion was
  `[[481,56,3,0,9],[14,119,12,0,6],[3,12,516,6,7],[0,4,57,649,2],[9,16,3,1,621]]`.
  Chroma consistency widened class-1 false positives (`0->1=56`, `4->1=16`)
  and did not improve recall enough to compensate.
- XAI selected12
  `runs\xai_smoke_v8_yolof_fgchroma012_p50_val_selected12_20260702` confirmed
  the same failure mode. Top confusions were `3->2=57`, `0->1=55`,
  `4->1=16`, `1->0=14`, and `1->2=12`. Heatmaps were not cleaner than the
  keeper (`attention/gradcam foreground mass = 0.8785/0.8751`, background mass
  `0.1215/0.1249`, attention border `0.3065`), flags remained high
  (`attention_border_attention=11/12`, `object_color_sensitive=7/12`), background
  blur/gray stayed negligible (`0.0004/-0.0006`), and object desaturation stayed
  dominant (`0.1250`).
- Reject before probe: a weak, bbox-bounded chroma consistency objective is
  implemented and active, but this exact setting reduces class-1 precision and
  repeats the color-sensitive boundary failure. Do not run a 2e probe or full
  train for this setting.
- Cleanup after audit deleted the rejected smoke train directory and preserved the
  selected12 XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260702_fgchroma012_p50_reject_smoke.json`, deleted
  `1` directory and reclaimed `180.36 MB`; D: free space after cleanup was about
  `75.46 GB`.

## Diagnostic 2026-07-02 - Probability Router And Teacher-Consensus Check

- Artifact:
  `runs\val_probability_router_diagnostic_20260702`. This is validation-only
  diagnostic evidence, not a reportable gate and not a final model selection.
- Baselines on validation from aligned sample-index CSVs:
  current TRKH base macro/class-1 F1 `0.8829/0.6783`; no-pretrain ensemble
  focus005 `0.8864/0.6939`; AIDT pretrained `0.9088/0.7169`.
- A grouped-CV router from base probabilities, no-pretrain ensemble probabilities,
  their deltas, top-2 margins, class-1 margins, and bbox size did not harvest the
  ensemble gain reliably. Best diagnostic router was logistic regression with
  grouped folds by `source_stem`: macro/class-1 F1 `0.8845/0.6880`, changing only
  `9` samples. Random forest reached `0.8841/0.6841`; HGB stayed at base
  `0.8829/0.6783`. This is below the no-pretrain ensemble itself and below the
  next class-1 gate.
- Switch summary also explains why another router is unlikely to be enough:
  no-pretrain ensemble fixes only a small net set over base, while AIDT has a much
  larger representation advantage but depends on pretrained inference. A simple
  probability router would be another val-tuned guard, not a new TRKH
  representation.
- I also checked a stricter train-side teacher-consensus manifest idea after the
  failed AIDT rescue-margin run. Requiring both AIDT and the no-pretrain ensemble
  to be correct on current train base mistakes within the class-1 rescue/suppress
  scope produced `0` rows under reasonable confidence/margin thresholds. There is
  therefore no hardsample/targeted-margin manifest that is both fold-safer and
  materially different from the rejected AIDT rescue manifest.
- Decision: do not implement another probability router, class-1 margin guard, or
  targeted-margin manifest from current base/ensemble/AIDT probabilities unless a
  new representation or true OOF signal changes the inputs.

## Audit 2026-07-02 - BBox-Weighted Train-Time Masked Reconstruction Smoke

- Rationale: after many logit/margin/consistency losses failed, I tried a
  representation-side signal that does not change raw data. MAE shows that
  masking image patches and reconstructing missing pixels can learn useful visual
  representations (`https://arxiv.org/abs/2111.06377`), while SimMIM reports that
  a simple raw-pixel reconstruction head and random masking can work without a
  complex decoder
  (`https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_SimMIM_A_Simple_Framework_for_Masked_Image_Modeling_CVPR_2022_paper.pdf`).
  Because short internal MAE pretraining had already failed on the older
  group-clean path, this run used only a light supervised auxiliary loss on the
  current `yolo_f` teacher-focus-binary anchor, with bbox/detail-biased patch
  selection and no raw dataset modification.
- Preflight passed `py_compile trkh\training\train.py trkh\models\model.py
  trkh\core\config.py` and focused pytest
  `tests\test_internal_ssl_mae.py tests\test_yolo_bbox_spatial_fusion.py`
  (`16 passed`). Launcher dry-run confirmed `hard_sample_manifest=""`,
  `skip_final_test=true`, the keeper resume checkpoint, and
  `masked_reconstruction_loss_weight=0.01`.
- Smoke
  `runs\smoke_v8_yolof_mim010_mask35_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260702`
  resumed the current keeper with teacher-focus-binary `0.015`, bbox spatial
  fusion, pairwise margin routing, boundary-band bbox dropout, 24 train batches,
  one epoch, full validation, architecture trace, and final test skipped.
  Masked reconstruction was active:
  `train_masked_reconstruction_loss=1.1148`, mask fraction `0.2461`, patch count
  `256`, bbox prior mean `0.6607`.
- Validation stayed below the keeper and below the probe gate:
  macro/class-1 F1 `0.8831/0.6784`, class-1 P/R `0.6073/0.7682`, accuracy
  `0.9194`. Confusion was
  `[[488,49,2,0,10],[19,116,11,0,5],[6,10,515,7,6],[0,4,54,652,2],[8,12,4,1,625]]`.
  Compared with the keeper (`0.8847/0.6860`), MIM did not reduce `0->1`, and it
  increased class-1 false negatives (`1->0=19`).
- XAI selected12
  `runs\xai_smoke_v8_yolof_mim010_mask35_val_selected12_20260702` showed the same
  failure family. Top confusions were `3->2=54`, `0->1=49`, `1->0=19`,
  `4->1=12`, and `1->2=11`. Heatmaps were not pathological
  (`attention/grad_rollout/gradcam/rollout foreground mass =
  0.9044/0.9722/0.9557/0.9499`), but border attention remained high
  (`attention_border_attention=11/12`, `gradcam_border_attention=7/12`).
  Background blur/gray original-prediction drops stayed negligible
  (`0.0005/-0.0008`), while object desaturation stayed larger (`0.0731`).
- Reject before probe: bbox/detail-weighted train-time MIM is implemented and
  active, but this exact light setting does not add a boundary cue strong enough
  to improve class 1. It mainly preserves foreground reconstruction and still
  leaves the same `0/1/2/4` surface-boundary errors.
- Cleanup after audit deleted the rejected smoke train directory and preserved
  the selected12 XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260702_mim010_mask35_reject_smoke.json`, deleted `1`
  directory and reclaimed about `185 MB`; D: free space after cleanup was about
  `75.38 GB`.

## Diagnostic 2026-07-02 - YOLO-F Foreground Surface Stats Object-Level Probe

- I inspected the boundary-error contact sheet and selected XAI crops before this
  diagnostic. The visible failures are mostly near-boundary object-surface cases:
  green `0/1/2` fruits with small maturity differences, `4->1` cases where
  dark/wrinkled defects are real but local, and `3->2` cases where large dark
  spots exist but Grad-CAM often stays on border/background or the clean yellow
  surface. This supports the prior conclusion that far background is not the
  main bottleneck, but it leaves open whether deterministic dark/brown/spot maps
  can be used as a secondary cue.
- `trkh.tools.probe_foreground_surface_stats` was updated to support
  object-level `yolo_f` via `MangoYOLOCropDataset`, preserving
  `sample_index`, `source_stem`, `object_index`, `label_path`, and bbox metadata
  in prediction CSVs. Regression test:
  `tests\test_probe_foreground_surface_stats.py` confirms a two-object YOLO
  image is not collapsed by path. Preflight passed `py_compile` for the tool and
  focused pytest (`1 passed`).
- Diagnostic artifact:
  `runs\yolof_foreground_surface_stats_diagnostic_20260702`. It used only train
  and validation splits, no test. Standalone foreground-surface stats remained
  weaker than TRKH: ExtraTrees validation macro/class-1 F1 `0.8384/0.5214`,
  class-1 P/R `0.7349/0.4040`; its best class-1 threshold diagnostic reached
  only class-1 F1 `0.6379`.
- A validation-only guard using surface stats to suppress low-surface-probability
  class-1 predictions was useful but not gate-worthy: best class-1 rule changed
  `16` validation samples, fixed `13` false positives into class 1, harmed `3`
  true class-1 samples, and reached macro/class-1 F1 `0.8874/0.6930`. This is
  below the next `0.70` class-1 gate and is val-threshold diagnostic only.
- A fold-safe version rejected the signal. Five-fold grouped OOF ExtraTrees
  surface predictions on train had class-1 F1 only `0.4367`; the threshold
  selected by train-OOF was `0.02`, changed `12` train samples but harmed `10`
  true class-1 cases. Applied to validation through a full-train surface
  classifier, it changed only `1` sample and reached macro/class-1 F1
  `0.8832/0.6802`, below the keeper `0.8847/0.6860`.
- Decision: keep the object-level diagnostic tool, but do not implement a
  surface-stat guard/head/router from this signal. It is another val-only
  suppressor that cannot survive train-OOF selection. Future defect/spot work
  needs a different supervision signal than deterministic HSV/Lab/spot summary
  statistics or validation-tuned post-processing.

## Smoke 2026-07-02 - Local Zoom Defect-Spot Score

- Research check before the smoke: TransFG uses transformer attention to select
  discriminative patches for fine-grained recognition
  (`https://arxiv.org/abs/2103.07976`), WS-DAN argues that attention-guided
  crop/drop can be more useful than random crops for FGVC
  (`https://ar5iv.labs.arxiv.org/html/1901.09891`), and NTS-Net localizes
  informative regions without part annotations
  (`https://arxiv.org/abs/1809.00287`). This supports testing a local-part cue,
  but only if the selected crop actually lands on surface defects instead of
  object borders/background.
- Code change: `LocalZoomImageExpert` now accepts
  `local_zoom_score_mode=defect_spot`, exposed through
  `--local-zoom-score-mode` and the v8 launcher. The score map adds dark/brown
  spot response and falls back to the older `foreground_detail` map when the
  defect response is weak. Focused test
  `tests\test_local_zoom_image_expert.py` checks a synthetic dark-brown region
  is selected. Preflight passed `py_compile` for `trkh\models\model.py`,
  `trkh\core\config.py`, `trkh\training\train.py`, and focused pytest
  (`tests\test_local_zoom_image_expert.py`,
  `tests\test_probe_foreground_surface_stats.py`; `5 passed`). Dry-run confirmed
  `hard_sample_manifest=""`, `skip_final_test=true`, keeper resume checkpoint,
  `local_zoom_score_mode=defect_spot`, `local_zoom_logit_scale=0.14`, and
  `local_zoom_aux_loss_weight=0.01`.
- Smoke
  `runs\smoke_v8_yolof_localzoom_defectspot_lzaux010_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260702`
  resumed the current keeper on `yolo_f`, with teacher-focus-binary `0.015`,
  bbox spatial fusion, pairwise margin routing, boundary-band bbox dropout,
  local zoom route pairs `0-1,1-2,2-3,1-4,4-rest`, 24 train batches, one epoch,
  full validation, architecture trace, and final test skipped. Local zoom aux
  loss was active (`train_local_zoom_aux_loss=5.3997`).
- Validation stayed below the keeper and below the probe gate:
  macro/class-1 F1 `0.8831/0.6784`, class-1 P/R `0.6073/0.7682`, accuracy
  `0.9194`. Confusion was
  `[[487,49,2,0,11],[19,116,11,0,5],[5,10,515,8,6],[0,4,52,653,3],[8,12,4,1,625]]`.
  Compared with the keeper (`0.8847/0.6860`), it did not reduce dominant
  `0->1`, kept `4->1=12`, and increased class-1 false negatives
  (`1->0=19`).
- Architecture trace found the failure mode directly: for the class-4 sample
  with visible dark defect spots, `09k_local_zoom_crop_box.png` selected the
  top-left clean/edge region and missed the defect cluster; `09j_local_zoom_score.png`
  was still dominated by object border/background edge energy. Class-0/1/3
  traces showed similar border/lower-edge attraction rather than stable small
  defect selection.
- XAI selected12
  `runs\xai_smoke_v8_yolof_localzoom_defectspot_val_selected12_20260702` confirmed
  this is not a background-context breakthrough. Top selected errors were
  `0->1`, `2->1`, `3->1`, `3->2`, and `4->1`. Heatmaps stayed mostly on fruit
  (`attention/grad_rollout/gradcam/rollout foreground mass =
  0.8982/0.9731/0.9461/0.9524`), but border flags remained high
  (`attention_border_attention=12/12`, `gradcam_border_attention=7/12`) and
  background blur/gray drops stayed negligible (`0.00008/-0.00042`). Object
  desaturation remained the dominant perturbation (`0.0882`), so the bottleneck
  is still color/surface-boundary separability rather than far background.
- Reject before probe: the code path is valid and may remain as an optional
  architecture hook, but this exact `defect_spot` scoring/weight setting should
  not be probed. It adds a crop expert initialized from scratch, yet the score
  is not defect-stable on real validation cases and worsens class-1 F1.
- Cleanup after audit deleted the rejected smoke train directory and preserved
  the selected12 XAI plus keeper/final/diagnostic artifacts. Manifest:
  `runs\cleanup_manifest_20260702_localzoom_defectspot_reject_smoke.json`,
  deleted `1` directory and reclaimed about `186.32 MB`; D: free space after
  cleanup was about `75.30 GB`.

## Smoke 2026-07-02 - Local Zoom Defect-Spot Interior Suppression

- Rationale: the previous `defect_spot` score selected border/background edge
  energy instead of true local surface defects. I added a stricter score mode,
  `local_zoom_score_mode=defect_spot_interior`, that soft-erodes foreground
  support, suppresses frame margins, and blends a center prior so local zoom
  should prefer interior brown/dark details over object edges. This keeps the
  change inside model/train-time logic and does not modify raw data.
- Code/preflight: `LocalZoomImageExpert` gained the interior/border-suppressed
  score path, exposed through `--local-zoom-score-mode` and the v8 launcher.
  Focused synthetic regression test
  `tests\test_local_zoom_image_expert.py::test_local_zoom_defect_spot_interior_suppresses_border_band`
  verifies an interior defect beats a dark border band. Preflight passed
  `py_compile` for `trkh\models\model.py`, `trkh\training\train.py`,
  `trkh\core\config.py`, and focused pytest (`5 passed`). Dry-run confirmed
  `hard_sample_manifest=""`, `distillation_weight=0`, `skip_final_test=true`,
  keeper resume, and `local_zoom_score_mode=defect_spot_interior`.
- Smoke
  `runs\smoke_v8_yolof_localzoom_defectspotinterior_lzaux010_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260702`
  used `yolo_f`, the current keeper checkpoint, teacher-focus-binary `0.015`,
  bbox spatial fusion, pairwise margin routing, boundary-band bbox dropout,
  route pairs `0-1,1-2,2-3,1-4,4-rest`, local zoom logit scale `0.14`, aux
  weight `0.01`, 24 train batches, one epoch, full validation, architecture
  trace, and final test skipped. The auxiliary was active
  (`train_local_zoom_aux_loss=5.4023`).
- Validation was worse than the keeper and below the probe gate:
  macro/class-1 F1 `0.8822/0.6764`, class-1 P/R `0.6042/0.7682`, accuracy
  `0.9186`. Confusion was
  `[[486,50,2,0,11],[19,116,11,0,5],[5,10,515,8,6],[0,4,52,653,3],[9,12,4,1,624]]`.
  It increased `0->1` (`50` vs keeper `49`), kept `4->1=12`, and left
  class-1 false negatives at `1->0=19`, so there is no short-probe case.
- Architecture trace still exposed the same failure mode. The synthetic test
  passed, but real samples did not: the class-4 trace selected a top-left
  crop touching `x=0` and focused leaves/background/object edge while ignoring
  a visible dark defect cluster on the fruit body. Class-0/1 traces still
  often selected lower/right edge regions. This means heuristic interior
  suppression is not reliable enough on real `yolo_f` crops.
- XAI selected12
  `runs\xai_smoke_v8_yolof_localzoom_defectspotinterior_val_selected12_20260702`
  confirmed no meaningful improvement. Top confusions were `3->2=51`,
  `0->1=50`, `1->0=19`, `4->1=12`, and `0->4=11`. Review flags:
  `attention_border_attention=12/12`, `attention_background_attention=6/12`,
  `gradcam_border_attention=7/12`, `gradcam_background_attention=1/12`, and
  `object_color_sensitive=5/12`. Heatmap foreground mass stayed high but
  border-heavy (`attention/grad_rollout/gradcam/rollout foreground =
  0.8836/0.9703/0.9463/0.9453`; border =
  `0.3158/0.1428/0.3180/0.2047`). Background blur/gray drops stayed negligible
  (`0.0005/-0.0007`), while object desaturation remained dominant (`0.1061`).
- Decision: reject before probe. `defect_spot_interior` fixes the toy border
  band case but not real validation failure cases, and it reduces class-1 F1.
  Do not repeat this exact local zoom setting. Any future local-part work must
  use a stronger reliability/supervision signal than hand-crafted dark/brown
  score maps, because deterministic spot maps keep confusing fruit edge, leaf
  texture, shadow, and true defects.
- Cleanup after audit deleted the rejected smoke train directory and preserved
  selected12 XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260702_localzoom_defectspotinterior_reject_smoke.json`,
  deleted `1` directory and reclaimed about `186.31 MB`; D: free space after
  cleanup was about `75.23 GB`.

## Smoke 2026-07-02 - Oracle-Teacher KD 0.03 From Train-Only Expert Oracle

- Rationale: use the train-only oracle teacher cache
  `runs\oracle_teacher_base_color_fgbg_train_20260701\teacher_probs_train.csv`
  as a very light diagnostic KD signal. The teacher selects a correct expert
  among base/color/foreground-background when available, otherwise falls back to
  averaging all experts. Its train metrics are high
  (macro/class-1 F1 `0.9699/0.9005`, class-1 recall `0.9871`), but it is
  explicitly train-label-informed, so this smoke used only `DistillationWeight=0.03`
  and was gated as a reject-if-not-immediate-improvement test.
- Preflight/dry-run: focused preflight had already passed for
  `trkh\training\train.py`, `trkh\models\model.py`,
  `trkh\tools\build_oracle_teacher_cache.py`, and tests
  `tests\test_build_oracle_teacher_cache.py`,
  `tests\test_teacher_probability_dataset.py` (`5 passed`). Dry-run confirmed
  `hard_sample_manifest=""`, oracle `distillation_teacher_csv`,
  `distillation_weight=0.03`, teacher-focus-binary `0.015`,
  bbox spatial fusion, pairwise margin routing, final test skipped, and
  architecture trace enabled.
- Smoke
  `runs\smoke_v8_yolof_oracleteacherkd003_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260702`
  used `yolo_f`, the current keeper checkpoint, 24 train batches, one epoch,
  full validation, teacher-focus-binary `0.015`, bbox spatial fusion,
  pairwise margin routing, boundary-band bbox dropout, and final test skipped.
  The oracle teacher matched by sample index with overlap ratio `1.0`, had
  train label agreement `0.9824`, and KD was active
  (`train_distillation_loss=0.1085` with weight `0.03`).
- Validation was below the keeper and below the probe gate:
  macro/class-1 F1 `0.8822/0.6764`, class-1 P/R `0.6042/0.7682`, accuracy
  `0.9186`. Confusion was
  `[[486,50,2,0,11],[19,116,11,0,5],[5,10,514,9,6],[0,4,52,653,3],[8,12,4,1,625]]`.
  Compared with the keeper, it worsened `0->1` (`50` vs `49`) and `1->0`
  (`19` vs `17`) while keeping `4->1=12`; there is no short-probe case.
- XAI selected12
  `runs\xai_smoke_v8_yolof_oracleteacherkd003_val_selected12_20260702`
  showed the same failure mode. Top confusions were `3->2=52`, `0->1=50`,
  `1->0=19`, `4->1=12`, and `0->4=11`. Review flags were
  `attention_border_attention=12/12`, `attention_background_attention=5/12`,
  `gradcam_border_attention=7/12`, `gradcam_background_attention=1/12`, and
  `object_color_sensitive=4/12`. Heatmaps remained mostly foreground but
  border-heavy (`attention/grad_rollout/gradcam/rollout foreground =
  0.8982/0.9732/0.9461/0.9524`; border =
  `0.3128/0.1528/0.3219/0.2142`). Background blur/gray drops stayed negligible
  (`0.0002/-0.0002`), while object desaturation remained much larger (`0.0881`).
  Manual review confirmed case `0->1` focuses on fruit surface blemish/color,
  while a `3->2` case still has edge/background heat but perturbation evidence
  does not support background as the main missing signal.
- Decision: reject before probe. The train-label-informed oracle teacher is
  not a useful transfer target for this anchor; even at low KD weight it widens
  the class-1 false-positive region and does not introduce a new surface/boundary
  cue. Do not repeat oracle KD from this train-in-sample teacher unless it is
  rebuilt as fold-safe/OOF or disagreement-confident and shows a clear smoke
  reduction in `0->1`/`4->1` without harming `1->0`.
- Cleanup after audit deleted the rejected smoke train directory and preserved
  selected12 XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260702_oracleteacherkd003_reject_smoke.json`,
  deleted `1` directory and reclaimed about `180.30 MB`; D: free space after
  cleanup was about `75.15 GB`.

## Smoke 2026-07-02 - LDAM-GCE Robust Loss

- Rationale: after the oracle-teacher KD failed, I checked a low-cost robust-loss
  direction before adding new architecture. Generalized Cross Entropy was chosen
  because the original NeurIPS/arXiv paper
  `https://arxiv.org/abs/1805.07836` frames it as an interpolation between
  cross entropy and MAE-like noise robustness, and the repo already had
  `gce`/`ldam_gce` support plus focused tests. This keeps the experiment inside
  train-time logic and does not modify raw data.
- Preflight passed `py_compile` for `trkh\training\train.py`,
  `trkh\training\losses.py`, `trkh\core\config.py`, and focused pytest
  `tests\test_generalized_cross_entropy_loss.py` (`3 passed`). Dry-run confirmed
  `classification_loss=ldam_gce`, `gce_q=0.7`, `hard_sample_manifest=""`,
  `distillation_weight=0`, teacher-focus-binary `0.015` with the correct
  focus005 teacher CSV, bbox spatial fusion, pairwise margin routing, final test
  skipped, and architecture trace enabled.
- Smoke
  `runs\smoke_v8_yolof_ldamgce_q070_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260702`
  used `yolo_f`, the current keeper checkpoint, 24 train batches, one epoch,
  full validation, teacher-focus-binary `0.015`, bbox spatial fusion,
  pairwise margin routing, boundary-band bbox dropout, and final test skipped.
  The classification loss changed from `ldam_focal` to `ldam_gce`; the loss was
  active (`train_cls_loss=0.1973`, `classification_loss=ldam_gce`, `gce_q=0.7`).
- Validation was below the keeper and below the probe gate:
  macro/class-1 F1 `0.8816/0.6744`, class-1 P/R `0.6010/0.7682`, accuracy
  `0.9183`. Confusion was
  `[[485,51,2,0,11],[19,116,11,0,5],[5,10,514,8,7],[0,4,52,653,3],[8,12,4,1,625]]`.
  Compared with the keeper, it worsened `0->1` (`51` vs `49`) and `1->0`
  (`19` vs `17`) while keeping `4->1=12`; there is no short-probe case.
- XAI selected12
  `runs\xai_smoke_v8_yolof_ldamgce_q070_val_selected12_20260702` confirmed no
  useful correction. Top confusions were `3->2=52`, `0->1=51`, `1->0=18`,
  `4->1=12`, and `0->4=11`. Review flags were
  `attention_border_attention=11/12`, `attention_background_attention=4/12`,
  `gradcam_border_attention=7/12`, `gradcam_background_attention=1/12`, and
  `object_color_sensitive=3/12`. Heatmaps stayed mostly foreground
  (`attention/grad_rollout/gradcam/rollout foreground =
  0.9044/0.9723/0.9559/0.9498`) but border mass remained high
  (`0.3041/0.1610/0.3021/0.2306`). Background blur/gray drops were negligible
  (`0.0005/-0.0005`), while object desaturation stayed larger (`0.0739`).
- Decision: reject before probe. `ldam_gce q=0.7` reduces the numeric CE scale
  but not the surface/boundary ambiguity; it broadens class-1 false positives
  and harms class-1 F1. Do not repeat this exact robust-loss setting on the
  current anchor. Any future robust-loss work needs a new sample-selection or
  fold-safe reliability signal, not just swapping the base criterion.
- Cleanup after audit deleted the rejected smoke train directory and preserved
  selected12 XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260702_ldamgce_q070_reject_smoke.json`, deleted `1`
  directory and reclaimed about `180.25 MB`; D: free space after cleanup was
  about `75.08 GB`.

## Smoke 2026-07-02 - Multi-Granularity Auxiliary CE Heads

- Rationale: test whether deep supervision on intermediate transformer layers
  can stabilize subtle boundary cues without adding inference-time blending.
  I checked the primary literature direction before running it: Deeply
  Supervised Nets (`https://arxiv.org/abs/1409.5185`) motivates companion
  objectives for intermediate layers, FFVT (`https://arxiv.org/abs/2107.02341`)
  motivates low/mid/high transformer features for fine-grained recognition, and
  Contrastive Deep Supervision (`https://arxiv.org/abs/2207.05306`) warns that
  direct supervised CE on shallow layers can conflict with low-level features.
  This smoke therefore acts as a cheap test of CE-style intermediate
  supervision before considering a contrastive variant.
- Preflight/dry-run: `py_compile` for `trkh\models\model.py`,
  `trkh\training\train.py`, `trkh\core\config.py` and focused pytest
  `tests\test_multi_granularity_aux_heads.py` passed (`5 passed`). The first
  launcher attempt used `class_f` as primary data and correctly failed the
  teacher CSV sample-index guard (`overlap_ratio=0.0000`); the partial run was
  deleted immediately because it produced no metrics. The corrected dry-run
  used `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml` as primary data and
  confirmed `hard_sample_manifest=""`, teacher-focus-binary `0.015`, bbox
  spatial fusion, pairwise margin routing, boundary-band bbox dropout, final
  test skipped, and `multi_granularity_aux_heads=true` with layers `2,5,8`.
- Smoke
  `runs\smoke_v8_yolof_multigranaux020_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260702`
  resumed the current keeper, trained one epoch over 24 batches, and completed
  architecture trace. The teacher CSV matched by sample index with
  `sample_index_path_overlap_ratio=1.0`. The new auxiliary heads loaded as a
  checkpoint-safe architecture extension and were active:
  `train_multi_granularity_aux_loss=8.8385`; trace confirmed logits from layers
  `2,5,8` with shape `[1,5]` for all five class samples.
- Validation was below the keeper and below the probe gate:
  macro/class-1 F1 `0.8830/0.6822`, class-1 P/R `0.6094/0.7748`, accuracy
  `0.9186`. Confusion was
  `[[487,49,2,0,11],[18,117,11,0,5],[5,10,513,10,6],[0,4,53,653,2],[9,12,4,1,624]]`.
  Compared with the keeper, it kept `0->1=49` and `4->1=12`, but increased
  class-1 false negatives (`1->0=18` vs keeper `17`) and reduced class-1
  recall (`0.7748` vs `0.7815`), so there is no short-probe case.
- XAI selected12
  `runs\xai_smoke_v8_yolof_multigranaux020_val_selected12_20260702` showed no
  new useful signal. Top confusions were `3->2=53`, `0->1=48`, `1->0=17`,
  `4->1=12`, and `0->4=11`. Review flags were
  `attention_border_attention=1/12`, `grad_rollout_border_attention=1/12`,
  `gradcam_background_attention=1/12`, `gradcam_border_attention=8/12`,
  `rollout_border_attention=5/12`, and `object_color_sensitive=4/12`. Heatmaps
  were mostly foreground (`attention/grad_rollout/gradcam/rollout foreground =
  0.9830/0.9686/0.9757/0.9461`) with Grad-CAM border mass still high
  (`0.2663`). Background blur/gray drops stayed negligible
  (`0.00013/-0.00040`), while object desaturation remained larger (`0.0883`).
  Manual spot check showed `0->1` still relying on highlight/border regions and
  a `4->1` case focusing on surface spots/wrinkles without separating class-4
  damage from class-1 boundary appearance.
- Decision: reject before probe. CE-style multi-granularity auxiliary heads add
  direct supervision but do not introduce a better surface/boundary cue on this
  anchor. Do not repeat this exact setting
  `multi_granularity_aux_heads=true/layers=2,5,8/loss_weight=0.02/dropout=0.08`.
  If deep supervision is revisited, prefer a genuinely different
  representation-level signal, such as contrastive deep supervision on
  intermediate pooled features with reliability/confusion-pair gating, rather
  than more CE heads on the same layers.
- Cleanup after audit deleted the rejected smoke train directory and preserved
  selected12 XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260702_multigranaux020_reject_smoke.json`, deleted
  `1` directory and reclaimed about `180.43 MB`; D: free space after cleanup was
  about `75.01 GB`.

## Smoke 2026-07-02 - Multi-Granularity Contrastive Deep Supervision

- Rationale: the CE auxiliary-head smoke confirmed that intermediate layers can
  be supervised, but direct CE did not create a better boundary/surface cue. I
  therefore implemented a representation-level variant aligned with the earlier
  literature check: DSN (`https://arxiv.org/abs/1409.5185`) for intermediate
  companion objectives, FFVT (`https://arxiv.org/abs/2107.02341`) for using
  low/mid/high transformer features in fine-grained recognition, and Contrastive
  Deep Supervision (`https://arxiv.org/abs/2207.05306`) for replacing conflicting
  shallow CE supervision with contrastive feature pressure.
- Implementation: `trkh\models\model.py` now exposes pooled intermediate
  `multi_granularity_features` whenever auxiliary heads are active, and
  `trkh\training\train.py` adds teacher-gated supervised contrastive loss over
  those layer features. The loss supports explicit confusion pairs
  `0-1,1-2,1-4,2-3`, teacher agreement filtering, and reliability weighting
  modes (`filter`, `confidence`, `confidence_margin`). The v8 launcher also
  normalizes optional path sentinels like `none` so `-HardSampleManifest none`
  correctly disables stale hard-repeat manifests.
- Preflight: `py_compile` for `trkh\models\model.py`, `trkh\training\train.py`,
  and `trkh\core\config.py` passed; focused pytest
  `tests\test_multi_granularity_aux_heads.py` passed (`7 passed`). Dry-run
  confirmed `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, keeper resume,
  `hard_sample_manifest=""`, teacher-focus-binary `0.015`, bbox spatial fusion,
  pairwise margin routing, boundary-band bbox dropout, final test skipped, and
  `multi_granularity_aux_heads=true/layers=2,5,8` with contrastive weight `0.006`.
- Smoke
  `runs\smoke_v8_yolof_multigrancontrast006_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260702`
  trained one epoch over 24 batches from the current keeper. Contrastive loss was
  active and dense: `train_multi_granularity_contrastive_loss=1.5623`,
  selected fraction `0.9740`, selected count `15.58`, positive pairs `34.17`,
  reliability mean `0.1963`; auxiliary CE loss was intentionally `0.0`.
- Validation tied the keeper on class 1 but did not beat it: macro/class-1 F1
  `0.8843/0.6860`, class-1 P/R `0.6114/0.7815`, accuracy `0.9194`.
  Confusion was
  `[[488,49,2,0,10],[17,118,11,0,5],[5,10,513,10,6],[0,4,53,653,2],[9,12,4,1,624]]`.
  Against the keeper it keeps class-1 F1/P/R unchanged, lowers macro slightly
  (`0.8843` vs `0.8847`), leaves `0->1=49` and `4->1=12`, and worsens `3->2`
  to `53`, so it is not a probe candidate.
- XAI selected12
  `runs\xai_smoke_v8_yolof_multigrancontrast006_val_selected12_20260702`
  also stayed effectively unchanged. Top confusions were `3->2=53`, `0->1=49`,
  `1->0=18`, `4->1=12`, and `1->2=11`. Review flags were
  `attention_border_attention=1/12`, `grad_rollout_border_attention=1/12`,
  `gradcam_background_attention=1/12`, `gradcam_border_attention=8/12`,
  `object_color_sensitive=4/12`, and `rollout_border_attention=5/12`. Heatmaps
  stayed mostly foreground (`attention/grad_rollout/gradcam/rollout foreground =
  0.9830/0.9686/0.9757/0.9461`), background blur/gray drops were near zero
  (`0.00025/-0.00047`), and object desaturation remained the stronger
  perturbation (`0.0883`). Manual checks of `0->1` and `4->1` mistakes still
  showed highlight/border and surface-spot ambiguity rather than a new separable
  cue.
- Decision: reject before probe. Multi-granularity contrastive deep supervision
  is implemented and works mechanically, but this setting only ties the current
  class-1 number and does not improve the real failure mode. Do not repeat
  `heads=true/layers=2,5,8/contrastive_weight=0.006/temp=0.18/pairs=0-1,1-2,1-4,2-3/teacher=focus005/weight_mode=confidence_margin/power=0.5`
  without a new target or representation signal.
- Cleanup after audit deleted the rejected smoke train directory and preserved
  selected12 XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260702_multigrancontrast006_reject_smoke.json`,
  deleted `1` directory and reclaimed about `180.69 MB`; D: free space after
  cleanup was about `74.94 GB`.

## Diagnostic 2026-07-02 - Current Keeper Color TTA

- Rationale: before adding another train-time loss, I checked whether the
  remaining class-1 boundary is partly an inference-time stability problem.
  Test-time augmentation is a standard way to aggregate predictions across
  transformed views, but the recent TTA aggregation literature warns that simple
  averaging can both correct and corrupt predictions. I checked
  `https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf`
  and `https://arxiv.org/html/2402.06892v1`; both support evaluating the
  augmentation set/aggregation empirically rather than assuming TTA helps.
- Eval
  `runs\eval_yolof_teacherfocusbinary015_best_val_tta_b08_20260702` used the
  current keeper, `yolo_f` validation, `--tta`, `--tta-brightness-delta 0.08`,
  and `--bbox-token-prior-source crop_bbox`. The current `evaluate.py` TTA path
  uses brightness/contrast/saturation views and averages logits; it does not add
  horizontal flip in this path.
- Result: reject. Macro F1 fell to `0.8763` and accuracy to `0.9144`, far below
  the keeper `0.8847/0.9198`. Class-1 F1 dropped to `0.6590`, with P/R
  `0.5808/0.7616`. Confusion became
  `[[484,52,2,0,11],[18,115,13,0,5],[5,13,512,7,7],[0,4,57,649,2],[9,14,3,1,623]]`,
  worsening `0->1`, `1->0`, `1->2`, `2->1`, `3->2`, and `4->1`.
- Per-sample comparison against the keeper validation CSV showed only `10`
  corrections but `23` harms over `37` changed predictions. Class-1-related harms
  included `0:0->1` (`4`), `4:4->1` (`3`), `2:2->1` (`3`), `1:1->0` (`3`), and
  `1:1->2` (`2`). This matches the previous XAI/robustness evidence: color and
  surface perturbations are label-relevant here, so color TTA is not a safe
  invariant.
- Decision: do not enable color TTA for the current keeper. If TTA is revisited,
  use a different augmentation family and an aggregation guard with train-only or
  fold-safe evidence; do not repeat simple color-logit averaging
  `tta_brightness_delta=0.08`.
- Cleanup deleted the rejected eval directory after recording metrics. Manifest:
  `runs\cleanup_manifest_20260702_color_tta_b08_reject_eval.json`, reclaimed
  about `2.85 MB`; D: free space after cleanup was about `74.94 GB`.

## Diagnostic 2026-07-02 - Reliable Entropy Test-Time Adaptation

- Rationale: since simple color TTA failed and many train-time losses only widen
  class-1 false positives, I tested a conservative label-free test-time
  adaptation diagnostic. TENT (`https://arxiv.org/abs/2006.10726`) adapts by
  minimizing prediction entropy during testing without extra labels. SAR/TTA
  stability work (`https://openreview.net/pdf?id=g2YraF75Tj` and
  `https://github.com/mr-eggplant/SAR`) warns that entropy minimization can
  collapse or be harmed by noisy samples, so the implementation uses reliable
  sample selection rather than updating on every validation image.
- Implementation: added `trkh.tools.evaluate_test_time_adaptation` as a separate
  diagnostic tool, not part of standard `evaluate.py`. It reuses the checkpoint
  loader/dataset/evaluator, freezes the model, updates only LayerNorm affine
  parameters plus `head.bias`, and selects unlabeled samples by confidence and
  low entropy. Regression tests in `tests\test_test_time_adaptation.py` passed
  after compile (`2 passed`).
- Smoke `runs\ttaadapt_smoke_lnheadbias_lr1e5_frac50_conf30_4b_20260702` ran
  four validation batches end-to-end, selecting `42/128` samples. It was only a
  tool smoke; the partial class distribution lacked class 4, so metrics were not
  used for model selection.
- Full validation
  `runs\ttaadapt_val_lnheadbias_lr1e5_frac50_conf30_20260702` used the current
  keeper, `selection_fraction=0.50`, `min_confidence=0.30`, `lr=1e-5`,
  one step per batch, and `bbox_token_prior_source=crop_bbox`. It selected
  `919/2606` samples (`0.3526`) and ran `41` optimizer steps with mean entropy
  loss `1.5428`.
- Result: reject. Full validation after adaptation reached macro/class-1 F1
  `0.8826/0.6764`, class-1 P/R `0.6042/0.7682`, and accuracy `0.9190`, below
  the keeper `0.8847/0.6860`. Confusion was
  `[[488,49,1,1,10],[19,116,11,0,5],[5,10,514,8,7],[0,4,52,653,3],[8,13,4,1,624]]`.
  It left `0->1=49`, worsened class-1 false negatives to `1->0=19`, and
  increased `4->1=13`.
- Per-sample comparison against the keeper changed only `7` predictions:
  `3` corrections, `4` harms, and no wrong-to-wrong moves. The class-1-related
  harms were `1:1->0` (`2`) and `4:4->1` (`1`), while the only class-1-related
  correction was `0:1->0` (`1`). This confirms that split-level entropy
  adaptation is too weak to create the missing class-1 surface/boundary signal
  and can reduce class-1 recall.
- Decision: keep the diagnostic tool because it is isolated and tested, but do
  not repeat exact reliable entropy TTA-adapt
  `layernorm_head_bias/lr=1e-5/selection_fraction=0.50/min_confidence=0.30/one_step`.
  Future test-time adaptation would need a different signal such as feature
  prototypes or explicit class-balance protection, not plain entropy pressure.
- Cleanup deleted the rejected smoke/full TTA-adapt diagnostic directories after
  recording metrics. Manifest:
  `runs\cleanup_manifest_20260702_ttaadapt_entropy_reject.json`, deleted `2`
  directories and reclaimed about `3.67 MB`; D: free space after cleanup was
  about `74.94 GB`.

## Cleanup 2026-07-02 - Legacy Mango Classification Leftovers

- After the TTA-adapt diagnostic, I rechecked the largest `runs` directories and
  found leftover legacy `mango_cls_*`, `bench_cls*`, and old group-clean XAI
  directories that were not current `class_f`/`yolo_f` gates. Current keeper,
  final evals, AIDT feature/disagreement artifacts, and current XAI audits were
  preserved.
- Cleanup manifest:
  `runs\cleanup_manifest_20260702_legacy_mango_cls_bench_leftovers.json`.
  Deleted `23` legacy/benchmark directories and reclaimed about `3441.33 MB`.
  D: free space after cleanup was about `78.30 GB`.

## Diagnostic 2026-07-03 - YOLO Source-Group Consistency Audit

- Rationale: the next tempting use of `yolo_f` wide context was a same-source or
  source-peer consistency loss, where objects from the same original label file
  supervise each other. Before adding another training loss, I audited whether
  this signal is actually broad and safe for class 1. This diagnostic reads YOLO
  labels only and does not modify raw images or labels.
- Implementation: added `trkh.tools.audit_yolo_source_groups` plus
  `tests\test_audit_yolo_source_groups.py`. The tool writes
  `source_group_summary.json`, `mixed_label_examples.csv`, and
  `multi_object_examples.csv` under a run directory. Preflight passed:
  `py_compile` for the new tool/test and focused pytest
  `tests\test_audit_yolo_source_groups.py` (`1 passed`).
- Artifact:
  `runs\yolof_source_group_consistency_audit_20260703`.
- Key counts on `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`:
  train has `8064` label files, `9215` object rows, and `691` multi-object
  source images. Of those, `547` are same-label multi-object images and `144`
  are mixed-label images. Class 1 appears in only `115` train multi-object
  source images; only `13` are same-label class-1 peer groups, while `102` are
  mixed-label source images. Validation has only `29` multi-object images, all
  same-label and only classes `0/3`; validation class 1 has no multi-object
  source-peer coverage. Test is entirely single-object (`1267/1267`).
- Decision: reject global same-image consistency. It is unsafe because train
  contains mixed-label source images (`0-1`, `1-4`, and other pairs), and it is
  not directly validated on class 1 because val/test lack class-1 multi-object
  peers. If this family is ever revisited, it must be same-label-only, very
  light, and treated as a minor auxiliary diagnostic that must show smoke-level
  improvement in class-1 F1 and reduced `0->1/4->1` without hurting `1->0`.
  It should not be the next main architecture/loss direction.
- Cleanup: no smoke/probe train run was created, and the diagnostic artifact is
  small, so nothing was deleted. Current keeper/final/AIDT artifacts remain
  preserved.

## Smoke 2026-07-03 - Low-LR Keeper Continuation

- Rationale: after rejecting another broad context/source signal, I checked a
  low-risk optimization hypothesis before adding more architecture: whether the
  current keeper can still improve with a small LR continuation. A direct replay
  of the original seed-variance probe was not possible because its historical
  resume source `runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\checkpoints\best.pt`
  had already been removed during obsolete-run cleanup; the preserved keeper
  itself was used instead.
- Implementation: exposed `-Seed` in
  `scripts\run_trkh_5class_attention_views_v8.ps1` while keeping default `42`.
  Dry-run passed for the continuation recipe:
  `yolo_f`, resume from current keeper
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt`,
  `Seed=7`, LR `1e-5`, `WarmupEpochs=0`, `MaxTrainBatches=24`, one epoch,
  hard-sample manifest disabled, bbox spatial fusion and pairwise-margin routing
  enabled, teacher-focus-binary `0.015`, final test skipped.
- Smoke
  `runs\smoke_v8_yolof_keeper_lowlr1e5_seed7_24b_1e_20260703` reached
  validation macro/class-1 F1 `0.8825/0.6764`, class-1 P/R
  `0.6042/0.7682`, below the keeper `0.8847/0.6860`. Confusion was
  `[[487,50,2,0,10],[19,116,11,0,5],[5,10,515,8,6],[0,4,53,652,3],[8,12,4,1,625]]`,
  so it worsened `0->1`, `1->0`, and `3->2` compared with the anchor.
- XAI selected12
  `runs\xai_smoke_v8_yolof_keeper_lowlr1e5_seed7_val_selected12_20260703`
  confirmed the same failure mode: top confusions were `3->2=53`, `0->1=49`,
  `1->0=19`, `4->1=12`, `1->2=11`; attention/Grad-CAM foreground mass stayed
  high (`0.9829/0.9851`), background blur/gray drops were near zero
  (`0.0006/-0.0000`), and object desaturation remained much larger (`0.0925`).
- Decision: reject before probe. Low-LR continuation from the keeper is not an
  improvement path; it mainly reduces class-1 recall and keeps the same
  surface/boundary ambiguity. Do not repeat `LR=1e-5`, `24b/1e`, seed `7`
  continuation from the current keeper unless a new signal is added.
- Cleanup deleted the rejected smoke train directory and preserved the XAI audit
  plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260703_keeper_lowlr1e5_reject_smoke.json`,
  reclaimed about `180.68 MB`.

## Diagnostic/Smoke 2026-07-03 - Spectral/FINE Embedding Ambiguity

- Rationale: after repeated losses/routers failed, I checked whether current
  TRKH embeddings themselves expose a reliable train-only ambiguity signal.
  The method was motivated by FINE-style noisy-label filtering
  (`https://arxiv.org/abs/2102.11628`), fine-grained noisy-label work
  (`https://arxiv.org/abs/2303.02404`), and PASS/peer-agreement caution that
  sample selection can confuse hard samples with label noise. This was treated
  as audit first, not as immediate data editing.
- Implementation: added `trkh.tools.probe_embedding_spectral_noise` plus
  `tests\test_probe_embedding_spectral_noise.py`. The tool extracts frozen
  checkpoint embeddings, keeps object-level `sample_index/source_stem/object_index`
  records, and scores class-local ambiguity from kNN label disagreement,
  centroid margin, and spectral/FINE alignment. A second utility
  `trkh.tools.build_spectral_margin_manifest` converts only high-ambiguity
  train rows into a `sample_index` targeted-margin candidate manifest.
  Preflight passed: `py_compile` for both tools/tests and focused pytest
  `tests\test_targeted_margin_sample_index.py`,
  `tests\test_build_spectral_margin_manifest.py`, and
  `tests\test_probe_embedding_spectral_noise.py` (`4 passed`).
- Full train-split audit artifact:
  `runs\spectral_noise_yolof_keeper_train_20260703`. It used current keeper
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt`
  on `yolo_f` train only (`9215` object rows, embedding dim `256`, no test read).
  Class 1 had only `0.7012` mean same-label neighbor fraction versus
  `0.9211/0.9409/0.9731/0.9774` for classes `0/2/3/4`, confirming that the
  current embedding manifold overlaps class 1 with classes 0 and 2. Top 10%
  high-risk class 1 rows had `0.2525` same-label neighbor fraction and `18.18%`
  base train error, with rival labels mostly `0` (`33`) and `2` (`19`).
  Top-risk class 0 had `74.36%` base train error and rival class 1 count `145`,
  so the signal is real but largely describes already-hard boundary rows.
- Candidate manifest artifact:
  `runs\spectral_margin_yolof_keeper_train_20260703\spectral_targeted_margin_train.csv`.
  It selected `285` train rows with strict `sample_index` keys:
  `233` spectral focus false-positive-risk rows and `52` false-negative-risk
  rows, distributed as `0->1=152`, `2->1=66`, `4->1=8`, `3->1=7`,
  `1->0=33`, `1->2=19`.
- Smoke
  `runs\smoke_v8_yolof_spectralmargin012_m08_teacherfocusbinary015_24b_1e_20260703`
  used this manifest with a light targeted-margin loss (`loss_weight=0.012`,
  margin `0.08`, max row weight `1.15`), teacher-focus-binary `0.015`,
  bbox spatial fusion, pairwise-margin routing, boundary-band bbox dropout, and
  hard-repeat/sample-weight/group-DRO disabled. The manifest matched
  `285/285` sample indices and the loss was active
  (`train_targeted_margin_loss=0.0989`, fraction `0.0417`).
- Result: reject before probe. Validation macro/class-1 F1 fell to
  `0.8822/0.6764`, class-1 P/R `0.6042/0.7682`, below keeper
  `0.8847/0.6860`. Confusion was
  `[[485,50,2,0,12],[19,116,11,0,5],[5,10,515,8,6],[0,4,52,653,3],[8,12,4,1,625]]`,
  so it increased `0->1` and `1->0` and did not reduce `4->1`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_spectralmargin012_m08_val_selected12_20260703`
  stayed in the same failure mode: top confusions `3->2=52`, `0->1=49`,
  `1->0=19`, `4->1=12`; foreground mass stayed high
  (`attention/gradcam=0.9828/0.9760`), background blur/gray drops were near zero
  (`0.0001/-0.0005`), object desaturation remained much larger (`0.0879`),
  and Grad-CAM border flags remained `7/12`.
- Decision: keep the spectral audit and manifest builder as diagnostics, but do
  not repeat representation-derived targeted margin
  `loss_weight=0.012/margin=0.08/max_weight=1.15/min_ambiguity_nonfocus=0.78/min_ambiguity_focus=0.74`
  on the current keeper. The signal identifies ambiguous/hard boundary rows, but
  pushing their current hard labels does not create a new surface/boundary cue and
  worsens class-1 recall. Future use of spectral embeddings should remain audit
  or be combined with a genuinely new representation/OOF signal, not another
  hard-label margin over the same anchor.
- Cleanup deleted the rejected spectral-margin smoke train directory and the
  temporary 128-sample spectral audit smoke after recording metrics/XAI. Manifest:
  `runs\cleanup_manifest_20260703_spectralmargin_reject_smoke.json`,
  reclaimed about `180.60 MB`; D: free space after cleanup was about `78.17 GB`.

## Smoke 2026-07-03 - Focus-Class Tversky/F1 Surrogate Loss

- Rationale: after spectral margin confirmed that hard-label pressure on the
  current embedding hurts class-1 recall, I tested a softer direct F-score
  objective for the problem class. The idea follows Tversky/Focal-Tversky style
  control of false positives and false negatives
  (`https://pmc.ncbi.nlm.nih.gov/articles/PMC8785124/`) and F-beta surrogate
  optimization (`https://arxiv.org/pdf/2104.01459`), with the practical goal of
  suppressing class-1 false positives (`0->1`, `4->1`) without editing raw data.
- Implementation: added optional focus-class soft Tversky loss to
  `trkh.training.train`, `TrainConfig`, and the V8 launcher. The loss uses the
  class-1 softmax probability against one-vs-rest hard targets and records
  train stats (`tp/fp/fn/index/mean_probability`) in `history.csv`. Preflight
  passed: `py_compile` for train/config/tests and focused pytest
  `tests\test_focus_tversky_loss.py tests\test_focus_class_head.py` (`4 passed`).
  Dry-run confirmed the resolved recipe on `yolo_f`, the current keeper resume,
  correct sample-index teacher CSV, bbox spatial fusion, pairwise routing, and
  no hard/sample/quality/targeted manifests.
- Smoke
  `runs\smoke_v8_yolof_focustversky010_a70b30_teacherfocusbinary015_24b_1e_20260703`
  used `focus_tversky_loss_weight=0.010`, class `1`, alpha/beta/gamma
  `0.70/0.30/1.0`, probability power `1.0`, start epoch `1`,
  teacher-focus-binary `0.015`, bbox spatial fusion, pairwise-margin routing,
  boundary-band bbox foreground dropout, and `24` train batches for one epoch.
  The loss was active (`train_focus_tversky_loss=0.6772`,
  `index=0.3226`, `tp=2.0553`, `fp=4.2786`, `fn=4.3600`).
- Result: reject before probe. Validation macro/class-1 F1 was
  `0.8831/0.6784`, class-1 P/R `0.6073/0.7682`, below the keeper
  `0.8847/0.6860`. Confusion was
  `[[486,49,2,0,12],[19,116,11,0,5],[5,10,516,7,6],[0,4,52,653,3],[8,12,4,1,625]]`.
  It did not reduce `0->1` (`49`) or `4->1` (`12`) and worsened class-1 false
  negatives to `1->0=19` versus the keeper.
- XAI selected12
  `runs\xai_smoke_v8_yolof_focustversky010_a70b30_val_selected12_20260703`
  stayed in the same failure mode: top confusions `3->2=52`, `0->1=49`,
  `1->0=19`, `4->1=12`; foreground mass stayed high
  (`attention/gradcam=0.9780/0.9793`), background blur/gray drops were near zero
  (`0.0002/-0.0006`), object desaturation was much larger (`0.0730`), and
  Grad-CAM border flags remained `7/12`.
- Decision: keep the implementation because it is isolated, tested, and may be
  useful only if a future representation changes the class-1 manifold. Do not
  repeat exact focus-class Tversky
  `weight=0.010/alpha=0.70/beta=0.30/gamma=1.0/prob_power=1.0/class=1/start=1`
  on the current keeper; it suppresses class 1 without creating the missing
  surface/boundary cue.
- Cleanup deleted the rejected Tversky smoke train directory and two stale
  background-launch log files after recording metrics/XAI. Manifest:
  `runs\cleanup_manifest_20260703_focustversky010_reject_smoke.json`, reclaimed
  about `180.58 MB`; D: free space after cleanup was about `78.10 GB`.

## Smoke 2026-07-03 - Class-Independent OVA Auxiliary Head

- Rationale: after direct class-1 F-score pressure failed, I tested whether
  decoupling class evidence into one-vs-all binary classifiers could reduce
  softmax class coupling under ambiguous/noisy fine-grained labels. This was
  motivated by Class-Independent Regularization for noisy labels
  (`https://ojs.aaai.org/index.php/AAAI/article/view/25434`) and the
  Fine-Grained Classification with Noisy Labels/SNSCL observation that
  inter-class ambiguity makes standard noisy-label methods weak on fine-grained
  classes (`https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_Fine-Grained_Classification_With_Noisy_Labels_CVPR_2023_paper.pdf`).
- Implementation: added a checkpoint-safe optional class-independent OVA head
  to `VisionTransformerWithRegisters`, `ModelConfig`, `TrainConfig`, train CLI,
  and the V8 launcher. The auxiliary head is zero-initialized and does not alter
  inference logits; it only adds BCE-with-logits one-vs-all training pressure.
  Preflight passed: `py_compile` for model/train/config/test files and focused
  pytest `tests\test_class_independent_head.py`,
  `tests\test_focus_class_head.py`, and `tests\test_focus_tversky_loss.py`
  (`6 passed`). Dry-run confirmed `yolo_f`, keeper resume, correct
  sample-index teacher CSV, `distillation_weight=0`, teacher-focus-binary
  `0.015`, bbox spatial fusion, pairwise routing, and empty hard/sample/
  quality/targeted manifests.
- Smoke
  `runs\smoke_v8_yolof_classindova020_pos4_teacherfocusbinary015_24b_1e_20260703`
  used `class_independent_head=true`, dropout `0.05`,
  `class_independent_loss_weight=0.020`, positive weight `4.0`,
  teacher-focus-binary `0.015`, bbox spatial fusion, pairwise-margin routing,
  and `24` train batches for one epoch. The loss was active
  (`train_class_independent_loss=1.1003`), but the auxiliary binary head was
  still nearly uninformative after the smoke window (`positive_probability=0.5042`,
  `negative_probability=0.4990`, margin `0.0054`).
- Result: reject before probe. Validation macro/class-1 F1 was
  `0.8825/0.6764`, class-1 P/R `0.6042/0.7682`, below the keeper
  `0.8847/0.6860`. Confusion was
  `[[486,50,2,0,11],[19,116,11,0,5],[5,10,515,8,6],[0,4,52,653,3],[8,12,4,1,625]]`,
  so it increased `0->1=50`, kept `4->1=12`, and worsened class-1 false
  negatives to `1->0=19`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_classindova020_pos4_val_selected12_20260703`
  stayed in the same failure mode: top confusions `3->2=52`, `0->1=50`,
  `1->0=19`, `4->1=12`; attention/Grad-CAM foreground mass stayed high
  (`0.9780/0.9793`), background blur/gray drops stayed near zero
  (`0.0002/-0.0005`), object desaturation remained much larger (`0.0732`),
  and Grad-CAM border flags remained `7/12`. Visual inspection showed the
  same surface/bright-region/border emphasis rather than a new separability cue.
- Decision: keep the implementation because it is isolated and tested, but do
  not repeat exact CIR-lite/OVA
  `class_independent_head=true/loss_weight=0.020/positive_weight=4.0/dropout=0.05`
  on the current keeper. One-vs-all decoupling did not create the missing
  surface/boundary signal and did not reduce class-1 false positives.
- Cleanup deleted the rejected OVA smoke train directory and preserved the XAI
  audit plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260703_classindova020_reject_smoke.json`, reclaimed
  about `180.81 MB`; D: free space after cleanup was about `78.03 GB`.

## Smoke 2026-07-03 - DCL-Lite BBox Region Shuffle

- Rationale: after OVA decoupling failed, I tested an image-view signal rather
  than another logit/embedding-only loss. The idea follows Destruction and
  Construction Learning for fine-grained recognition
  (`https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.html`):
  deliberately destruct the object region so the classifier must rely more on
  local fine-grained surface cues. I also checked part-centric fine-grained
  recognition evidence from PARTICLE
  (`https://openaccess.thecvf.com/content/ICCV2023W/VIPriors/papers/Saha_PARTICLE_Part_Discovery_and_Contrastive_Learning_for_Fine-Grained_Recognition_ICCVW_2023_paper.pdf`)
  and treated this as a lightweight smoke, not as a full architecture change.
- Implementation: added optional DCL-lite bbox region shuffle to `TrainConfig`,
  train CLI, and the V8 launcher. The helper selects samples by probability and
  class filter, shuffles a fixed grid inside `crop_bbox`/bbox, then applies a
  CE auxiliary loss on the destructed view with the original soft/hard target.
  It does not modify raw data and does not alter inference. Preflight passed:
  `py_compile` for train/config/test files and focused pytest
  `tests\test_dcl_region_shuffle.py`, `tests\test_foreground_snapmix.py`,
  `tests\test_illumination_consistency_loss.py`, and
  `tests\test_class_independent_head.py` (`10 passed`). Dry-run confirmed
  `yolo_f`, current keeper resume, correct sample-index teacher CSV,
  `distillation_weight=0`, teacher-focus-binary `0.015`, bbox spatial fusion,
  pairwise routing, and empty hard/sample/quality/targeted manifests.
- Smoke
  `runs\smoke_v8_yolof_dclshuffle010_p50_g4_teacherfocusbinary015_24b_1e_20260703`
  used `dcl_region_shuffle_loss_weight=0.010`, probability `0.50`, grid `4`,
  bbox margin `0.02`, classes `0,1,2,3,4`, teacher-focus-binary `0.015`, bbox
  spatial fusion, pairwise-margin routing, and `24` train batches for one epoch.
  The auxiliary loss was active (`train_dcl_region_shuffle_loss=1.6413`,
  fraction `0.4948`, about `15.83` samples/batch, bbox area fraction `0.6812`).
- Result: reject before probe. Validation macro/class-1 F1 was
  `0.8835/0.6784`, class-1 P/R `0.6073/0.7682`, below the keeper
  `0.8847/0.6860` and below the `0.70` class-1 smoke gate. Confusion was
  `[[488,49,2,0,10],[19,116,11,0,5],[5,10,516,7,6],[0,4,54,652,2],[8,12,4,1,625]]`,
  so `0->1=49` and `4->1=12` did not improve, `1->0=19` remained worse than
  the keeper, and `3->2` increased to `54`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_dclshuffle010_p50_g4_val_selected12_20260703`
  confirmed the same or slightly worse failure mode: top confusions `3->2=54`,
  `0->1=49`, `1->0=19`, `4->1=12`; attention/Grad-CAM foreground mass was
  `0.9828/0.9025`, with Grad-CAM background mass rising to `0.0975` and one
  `gradcam_background_attention` flag. Background blur/gray still barely
  mattered (`0.0004/-0.0003`), while object desaturation increased to `0.0931`
  and object-color sensitivity rose to `4/12`.
- Decision: keep the implementation because it is isolated and tested, but do
  not repeat exact DCL-lite
  `loss_weight=0.010/probability=0.50/grid=4/bbox_margin=0.02/classes=all` on
  the current keeper. Destructing the object bbox did not add a useful local
  surface cue; it made Grad-CAM more diffuse/background-sensitive and did not
  reduce class-1 false positives.
- Cleanup deleted the rejected DCL-lite smoke train directory and preserved the
  XAI audit plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260703_dclshuffle010_reject_smoke.json`, reclaimed
  about `180.75 MB`; D: free space after cleanup was about `77.97 GB`.

## Smoke 2026-07-03 - Boundary AIDT Feature CRD

- Rationale: after head/patch AIDT feature RKD failed, I tested a different
  feature-transfer shape inspired by contrastive representation distillation
  (`https://arxiv.org/abs/1910.10699`), relational KD
  (`https://arxiv.org/abs/1904.05068`), and feature distillation overhaul
  (`https://arxiv.org/abs/1904.01866`). Instead of matching pairwise distances,
  this smoke aligns each TRKH student feature with the same-object AIDT feature
  and uses in-batch boundary-pair negatives. It does not require AIDT at
  inference and does not modify raw data.
- Implementation: added optional teacher-feature contrastive loss to
  `TrainConfig`, `trkh.training.train`, the V8 launcher, and focused tests in
  `tests\test_teacher_feature_rkd.py`. The implementation uses a deterministic
  fixed projection for AIDT/TRKH feature dimensions, boundary pair filtering, and
  history columns for loss/count/negative_count/positive_similarity/
  negative_similarity/top1. Preflight passed: `py_compile` for train/config/test
  files and focused pytest `tests\test_teacher_feature_rkd.py` (`7 passed`).
  Dry-run confirmed `yolo_f`, current keeper resume, correct sample-index
  teacher CSV, correct clean AIDT feature NPZ, `distillation_weight=0`,
  teacher-focus-binary `0.015`, bbox spatial fusion, pairwise routing, and empty
  hard/sample/quality/targeted manifests.
- Smoke
  `runs\smoke_v8_yolof_aidtcrd006_boundary_teacherfocusbinary015_24b_1e_20260703`
  used `teacher_feature_contrastive_loss_weight=0.006`, temperature `0.20`,
  projection dim `128`, source `head`, pair mode `boundary`, pairs
  `0-1,1-2,1-4,2-3`, start epoch `1`, teacher-focus-binary `0.015`, bbox
  spatial fusion, pairwise-margin routing, and `24` train batches for one
  epoch. The loss was active but weakly aligned:
  `train_teacher_feature_contrastive_loss=2.3438`, count `32.0`,
  negative_count `10.2578`, positive similarity `-0.0205`, negative similarity
  `-0.0253`, and top1 `0.2279`.
- Result: reject before probe. Validation macro/class-1 F1 was
  `0.8828/0.6764`, class-1 P/R `0.6042/0.7682`, below the keeper
  `0.8847/0.6860` and below the `0.70` class-1 smoke gate. Confusion was
  `[[486,50,2,0,11],[19,116,11,0,5],[5,10,515,8,6],[0,4,51,654,3],[8,12,4,1,625]]`.
  It kept `0->1=50`, `4->1=12`, and worsened class-1 false negatives to
  `1->0=19`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_aidtcrd006_boundary_val_selected12_20260703`
  stayed in the established failure mode: top confusions `3->2=51`, `0->1=50`,
  `1->0=19`, `0->4=12`, and `4->1=12`; attention/Grad-CAM foreground mass was
  high (`0.9829/0.9761`), background blur/gray drops were near zero
  (`0.0005/-0.0002`), and object desaturation was much larger (`0.0881`).
  Grad-CAM still had border flags (`7/12`) and one background-attention flag;
  manual overlay checks showed occasional background hotspots but the decisive
  perturbation evidence remained object surface/color.
- Decision: keep the implementation because it is isolated and tested, but do
  not repeat fixed-projection boundary AIDT CRD
  `weight=0.006/temp=0.20/proj=128/source=head/pairs=0-1,1-2,1-4,2-3` on the
  current keeper. The fixed projection did not make same-object AIDT positives
  separable from boundary negatives; if feature distillation is revisited, it
  needs a learnable adapter or a stronger disagreement/OOF reliability signal,
  not another small weight on the same CRD shape.
- Cleanup deleted the rejected AIDT CRD smoke train directory and preserved the
  selected12 XAI audit plus keeper/final/AIDT artifacts. Manifest:
  `runs\cleanup_manifest_20260703_aidtcrd006_reject_smoke.json`, reclaimed about
  `180.92 MB`; D: free space after cleanup was about `77.90 GB`.

## Smoke 2026-07-03 - Self-Paced Small-Loss Reweighting

- Rationale: after AIDT feature CRD failed to add separability, I tested a
  small-loss/sample-selection loss wrapper for suspected noisy or ambiguous
  fine-grained labels. The idea follows the memorization/small-loss observation
  used by Co-teaching (`https://arxiv.org/abs/1804.06872`), JoCoR
  (`https://arxiv.org/abs/2003.02752`), and SuperLoss robust curriculum learning
  (`https://proceedings.neurips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf`).
  The fine-grained noisy-label warning remains important: LNL-FG work notes that
  standard noisy-label methods are weaker when inter-class ambiguity is high
  (`https://openaccess.thecvf.com/content/CVPR2023/html/Wei_Fine-Grained_Classification_With_Noisy_Labels_CVPR_2023_paper.html`).
  This smoke therefore used a conservative blend instead of hard dropping high
  loss samples, to avoid destroying rare class-1 boundary evidence.
- Implementation: added optional self-paced classification loss to
  `TrainConfig`, `trkh.training.train`, the V8 launcher, and
  `tests\test_self_paced_loss.py`. The wrapper computes per-sample
  classification loss, estimates a batch quantile threshold, downweights only
  higher-loss rows with detached weights, composes with existing sample weights,
  and records threshold/weight/selected-fraction telemetry in `history.csv`.
  Preflight passed: `py_compile` for train/config/tests, focused pytest
  `tests\test_self_paced_loss.py` (`2 passed`), combined pytest with teacher
  feature RKD (`9 passed`), and the sample-weight regression via `-k` (`1 passed,
  132 deselected`). Dry-run confirmed `yolo_f`, keeper resume, correct
  sample-index teacher CSV, `distillation_weight=0`, teacher-focus-binary
  `0.015`, bbox spatial fusion, pairwise routing, and empty hard/sample/
  quality/targeted manifests.
- Smoke
  `runs\smoke_v8_yolof_selfpaced025_p80_g075_min50_teacherfocusbinary015_24b_1e_20260703`
  used `self_paced_loss_weight=0.25`, percentile `0.80`, gamma `0.75`,
  min weight `0.50`, start epoch `1`, teacher-focus-binary `0.015`, bbox
  spatial fusion, pairwise-margin routing, and `24` train batches for one epoch.
  The loss was active: `train_self_paced_loss=0.3883`, raw loss `0.5919`,
  threshold `0.3029`, mean weight `0.9267`, selected fraction `0.7813`, and
  high-loss mean weight `0.6651`.
- Result: reject before probe. Validation macro/class-1 F1 was
  `0.8831/0.6784`, class-1 P/R `0.6073/0.7682`, below the keeper
  `0.8847/0.6860` and below the `0.70` class-1 smoke gate. Confusion was
  `[[487,49,2,0,11],[19,116,11,0,5],[5,10,515,8,6],[0,4,52,653,3],[8,12,4,1,625]]`.
  It did not reduce `4->1=12`, kept class-1 false negatives high
  (`1->0=19`, `1->2=11`), and only matched the usual `0->1=49` pattern.
- XAI selected12
  `runs\xai_smoke_v8_yolof_selfpaced025_p80_g075_min50_val_selected12_20260703`
  stayed in the same failure mode: top confusions `3->2=52`, `0->1=49`,
  `1->0=19`, `4->1=12`; attention/Grad-CAM foreground mass stayed high
  (`0.9828/0.9760`), background blur/gray drops were near zero
  (`0.0000/-0.0005` on predicted probability), and object desaturation was far
  larger (`0.0879`). Grad-CAM still had `7/12` border flags and one background
  flag; manual overlays showed object-surface emphasis with occasional external
  hotspot, but perturbation evidence still points to surface/boundary ambiguity,
  not usable background context.
- Decision: keep the implementation because it is isolated, tested, and may be
  useful with a stronger reliability/OOF signal, but do not repeat this exact
  self-paced setting on the current keeper. Batch-level small-loss reweighting
  alone behaves like another hard-label robust-loss wrapper and does not create
  the missing class-1 surface/boundary cue.
- Cleanup deleted the rejected self-paced smoke train directory and preserved
  the XAI audit plus keeper/final/AIDT artifacts. Manifest:
  `runs\cleanup_manifest_20260703_selfpaced025_reject_smoke.json`, reclaimed
  about `181.06 MB`; D: free space after cleanup was about `77.84 GB`.

## Smoke 2026-07-03 - Learnable Adapter AIDT Feature CRD

- Rationale: fixed-projection AIDT CRD failed because same-object positives and
  boundary negatives were barely separable. I tested a learnable projection
  adapter inspired by feature-hint distillation/FitNets
  (`https://arxiv.org/abs/1412.6550`), CRD
  (`https://arxiv.org/abs/1910.10699`), feature distillation overhaul
  (`https://arxiv.org/abs/1904.01866`), and projector-based KD work
  (`https://ojs.aaai.org/index.php/AAAI/article/view/28219/28433`). The adapter
  maps TRKH head features and AIDT 2816-dim features into a shared learned
  128-dim CRD space without using AIDT at inference and without modifying raw
  data.
- Implementation: added `TeacherFeatureProjectionAdapter` to
  `trkh.models.model`, config/CLI/launcher flags, checkpoint-compatible resume
  allowance for `teacher_feature_projection_adapter.*`, and a regression test in
  `tests\test_teacher_feature_rkd.py`. Also fixed the V8 launcher train call so
  Python stderr logging no longer aborts PowerShell under `$ErrorActionPreference
  = "Stop"`; launcher success/failure now follows the native Python exit code.
  Preflight passed: `py_compile` for model/train/config/tests, focused pytest
  `tests\test_teacher_feature_rkd.py` (`8 passed`), and resume-prefix tests
  (`6 passed`). Dry-run confirmed `yolo_f`, keeper resume, teacher-focus-binary
  `0.015`, bbox spatial fusion, pairwise routing, clean AIDT feature NPZ, and
  empty hard/sample/quality/targeted manifests.
- Smoke
  `runs\smoke_v8_yolof_aidtcrd_adapter004_boundary_teacherfocusbinary015_24b_1e_20260703`
  used `teacher_feature_contrastive_loss_weight=0.004`, temperature `0.20`,
  projection dim `128`, adapter dropout `0.02`, source `head`, pair mode
  `boundary`, pairs `0-1,1-2,1-4,2-3`, start epoch `1`, and `24` train batches
  for one epoch. The adapter was active in `resolved_config.json`
  (`student_dim=256`, `teacher_dim=2816`, `projection_dim=128`). CRD telemetry
  improved versus fixed projection but remained insufficient:
  loss `1.9771`, count `32.0`, negative_count `10.2578`, positive similarity
  `0.0739`, negative similarity `-0.0211`, and top1 `0.5169`.
- Result: reject before probe. Validation macro/class-1 F1 was
  `0.8825/0.6764`, class-1 P/R `0.6042/0.7682`, below the keeper
  `0.8847/0.6860` and below the `0.70` class-1 smoke gate. Confusion was
  `[[486,50,2,0,11],[19,116,11,0,5],[5,10,515,8,6],[0,4,52,653,3],[8,12,4,1,625]]`.
  It kept `0->1=50`, `4->1=12`, and class-1 false negatives
  (`1->0=19`, `1->2=11`).
- XAI selected12
  `runs\xai_smoke_v8_yolof_aidtcrd_adapter004_boundary_val_selected12_20260703`
  stayed in the same failure mode: top confusions `3->2=52`, `0->1=50`,
  `1->0=19`, `4->1=12`, `0->4=11`; attention/Grad-CAM foreground mass stayed
  high (`0.9827/0.9760`), background blur/gray drops were near zero
  (`0.0002/-0.0006`), and object desaturation was much larger (`0.0880`).
  Grad-CAM still had `7/12` border flags and one background-attention flag.
  Manual overlays showed case `0->1` still has external/border hotspots, while
  `4->1` focuses the wrinkled/damaged fruit surface but maps it to class 1.
- Decision: keep the adapter implementation because it is isolated,
  checkpoint-safe, and tested, but do not repeat this exact learnable-adapter
  AIDT CRD setting on the current keeper. A learned projector improves CRD
  alignment telemetry, but it still does not reduce class-1 false positives or
  add the missing surface/boundary cue. Future AIDT feature transfer needs a
  genuinely different signal, most likely fold-safe disagreement/reliability or
  case-level routing, not only another projector/weight/temp setting.
- Cleanup deleted the rejected adapter CRD smoke train directory and preserved
  the XAI audit plus keeper/final/AIDT artifacts. Manifest:
  `runs\cleanup_manifest_20260703_aidtcrd_adapter004_reject_smoke.json`,
  reclaimed about `190.26 MB`; D: free space after cleanup was about `77.77 GB`.

## Smoke/Probe 2026-07-03 - Balanced Softmax Tau 0.25

- Rationale: after repeated surface/boundary failures, I tested whether the
  current strict balanced sampler plus LDAM/focal shaping was over-expanding the
  rare class-1 decision region. This follows the class-prior correction ideas in
  Balanced Softmax (`https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html`)
  and logit adjustment (`https://arxiv.org/abs/2007.07314`). I also checked
  Seesaw Loss (`https://arxiv.org/abs/2008.10032`) as a related long-tail
  gradient-balancing idea, but started with the existing `balanced_softmax`
  implementation to avoid adding another custom loss before seeing signal.
- Preflight: `py_compile` passed for `trkh.training.train`,
  `trkh.training.losses`, and `trkh.core.config`; focused pytest for
  `BalancedSoftmaxFocalLoss` passed (`1 passed, 129 deselected`). Dry-run
  confirmed `ClassificationLoss=balanced_softmax`, `BalancedSoftmaxTau=0.25`,
  `yolo_f`, keeper resume, teacher-focus-binary `0.015`, bbox spatial fusion,
  pairwise routing, and empty hard/sample/quality/targeted manifests.
- Smoke
  `runs\smoke_v8_yolof_balsoftmax_tau025_teacherfocusbinary015_24b_1e_20260703`
  initially looked sane: validation macro/class-1 F1 `0.8850/0.6860`, class-1
  P/R `0.6114/0.7815`, confusion
  `[[487,49,2,0,11],[17,118,11,0,5],[5,10,516,7,6],[0,4,52,653,3],[9,12,4,1,624]]`.
  It tied the keeper class-1 F1 and slightly improved macro, mainly by reducing
  `1->0` to `17`.
- Smoke XAI
  `runs\xai_smoke_v8_yolof_balsoftmax_tau025_val_selected12_20260703` was not
  worse than the anchor: top confusions `3->2=51`, `0->1=49`, `1->0=17`,
  `4->1=12`; attention/Grad-CAM foreground mass `0.9781/0.9797`;
  background blur/gray near zero (`0.0005/-0.0004`); object desaturation
  `0.0755`. Because the smoke was at least keeper-like, I ran the gated short
  probe.
- Probe
  `runs\probe_v8_yolof_balsoftmax_tau025_teacherfocusbinary015_120b_2e_20260703`
  rejected the direction. Best epoch was epoch 1 with validation macro/class-1
  F1 `0.8747/0.6558`, class-1 P/R `0.5550/0.8013`. Epoch 2 dropped further to
  `0.8621/0.6272`. Best confusion was
  `[[476,58,3,0,12],[14,121,11,0,5],[3,16,510,7,8],[0,4,54,651,3],[8,19,3,1,619]]`.
  The loss increased class-1 recall but widened false positives (`0->1=58`,
  `2->1=16`, `4->1=19`), exactly the failure mode we need to suppress.
- Probe XAI
  `runs\xai_probe_v8_yolof_balsoftmax_tau025_val_selected12_20260703` confirmed
  the class-1 region widened rather than gained a new cue: top confusions
  `0->1=58`, `3->2=54`, `4->1=19`, `2->1=16`; attention/Grad-CAM foreground
  mass `0.9781/0.9681`; background perturbations stayed near zero; object
  desaturation jumped to `0.1391` and object-color-sensitive cases rose to
  `6/12`.
- Decision: reject. Balanced Softmax `tau=0.25` can make a one-epoch smoke look
  keeper-like, but under a slightly longer probe it expands class 1 and hurts
  precision. Do not repeat this exact setting or simply tune `tau` on the
  current keeper unless a new reliability/OOF signal constrains class-1 false
  positives.
- Cleanup deleted the rejected Balanced Softmax smoke and probe train
  directories while preserving both XAI audits and keeper/final artifacts.
  Manifest:
  `runs\cleanup_manifest_20260703_balsoftmax_tau025_reject_smoke_probe.json`,
  reclaimed about `363.26 MB`; D: free space after cleanup was about `77.64 GB`.

## Smoke 2026-07-03 - Seesaw Compensation-Only Loss

- Rationale: Balanced Softmax tau `0.25` widened class 1 during a longer probe.
  I therefore implemented Seesaw-style cross entropy from Seesaw Loss
  (`https://arxiv.org/abs/2008.10032`) but smoked a compensation-only setting:
  mitigation power `0.0`, compensation power `2.0`. This disables tail negative
  mitigation, which could worsen class-1 false positives, and keeps only the
  dynamic penalty for negative classes whose predicted probability exceeds the
  target-class probability.
- Implementation: added `SeesawCrossEntropyLoss` to `trkh.training.losses`,
  `classification_loss=seesaw` config/CLI/launcher plumbing, resolved-config
  logging for mitigation/compensation powers, and
  `tests\test_seesaw_loss.py`. Preflight passed: `py_compile` for
  `losses.py`, `train.py`, `config.py`, and test file; focused pytest passed
  (`4 passed`, including Seesaw and Balanced Softmax regression).
- Smoke
  `runs\smoke_v8_yolof_seesaw_p0_q2_teacherfocusbinary015_24b_1e_20260703`
  used `ClassificationLoss=seesaw`, `SeesawMitigationPower=0.0`,
  `SeesawCompensationPower=2.0`, teacher-focus-binary `0.015`, bbox spatial
  fusion, pairwise routing, and `24` train batches for one epoch. Validation
  macro/class-1 F1 was `0.8836/0.6842`, class-1 P/R `0.6126/0.7748`, below
  the keeper `0.8847/0.6860` and below the `0.70` gate. Confusion was
  `[[487,48,2,0,12],[18,117,11,0,5],[5,10,514,8,7],[0,4,52,653,3],[9,12,4,1,624]]`.
  It slightly reduced `0->1` (`48`) but increased class-1 false negatives
  (`1->0=18`) and did not reduce `4->1=12`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_seesaw_p0_q2_val_selected12_20260703` stayed in the
  same mode: top confusions `3->2=52`, `0->1=48`, `1->0=18`, `4->1=12`;
  attention/Grad-CAM foreground mass `0.9829/0.9764`; background blur/gray near
  zero (`0.0003/-0.0005`); object desaturation `0.0901`; Grad-CAM border
  `7/12` and one background flag.
- Decision: keep the implementation because it is isolated and tested, but do
  not repeat Seesaw compensation-only `p=0/q=2` on the current keeper. It only
  shifts the same precision/recall boundary and does not add a surface cue.
  Full Seesaw mitigation is not a good immediate next step because tail
  mitigation would likely further reduce negative pressure against class 1.
- Cleanup deleted the rejected Seesaw smoke train directory and preserved the
  selected12 XAI audit plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260703_seesaw_p0_q2_reject_smoke.json`, reclaimed
  about `180.97 MB`; D: free space after cleanup was about `77.57 GB`.

## Smoke 2026-07-03 - Disable Balanced Epoch Sampling

- Rationale: many recent losses widened class-1 false positives. I exposed the
  existing train.py `--disable-balanced-epoch-sampling` flag in the V8 launcher
  to test whether strict balanced class exposure itself was over-expanding the
  class-1 region. This does not modify raw data and, if anything, reduces
  class-1 exposure instead of oversampling it.
- Implementation: added launcher param `-DisableBalancedEpochSampling`, metadata
  logging, and conditional TrainArgs append. Dry-run confirmed
  `disable_balanced_epoch_sampling=true`, `yolo_f`, keeper resume,
  `ClassificationLoss=ldam_focal`, teacher-focus-binary `0.015`, bbox spatial
  fusion, pairwise routing, and empty hard/sample/quality/targeted manifests.
- Smoke
  `runs\smoke_v8_yolof_nobalanced_teacherfocusbinary015_24b_1e_20260703`
  used default LDAM/focal with balanced epoch sampling disabled for `24` train
  batches and one epoch. Validation macro/class-1 F1 was `0.8831/0.6804`,
  class-1 P/R `0.6105/0.7682`, below the keeper. Confusion was
  `[[488,48,1,1,11],[19,116,11,0,5],[5,10,513,9,7],[0,4,52,653,3],[8,12,4,1,625]]`.
  It reduced `0->1` to `48` but reduced class-1 recall and kept `4->1=12`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_nobalanced_val_selected12_20260703` stayed unchanged:
  top confusions `3->2=52`, `0->1=48`, `1->0=19`, `4->1=12`;
  attention/Grad-CAM foreground mass `0.9825/0.9760`; background blur/gray near
  zero (`0.0003/-0.0003`); object desaturation `0.0869`; Grad-CAM border `7/12`
  and one background flag.
- Decision: reject before probe. Balanced exposure is not the sole cause of the
  class-1 issue; disabling it trades a few fewer false positives for worse
  class-1 recall and does not change the XAI failure mode. Do not repeat this
  exact no-balanced-sampling continuation on the current keeper.
- Cleanup deleted the rejected no-balanced smoke train directory and preserved
  XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260703_nobalanced_reject_smoke.json`, reclaimed about
  `180.99 MB`; D: free space after cleanup was about `77.51 GB`.

## Smoke 2026-07-03 - Border Attention Suppression

- Rationale: repeated XAI showed high foreground mass but persistent Grad-CAM /
  rollout border flags and background-near-object hotspots on `0->1`, `4->1`,
  and `1->2` cases. I revisited attention-guided fine-grained ideas from
  WS-DAN (`https://arxiv.org/abs/1901.09891`) and attention dropout / region
  erasing from ADL (`https://arxiv.org/abs/1908.10028`,
  `https://openaccess.thecvf.com/content_CVPR_2019/papers/Choe_Attention-Based_Dropout_Layer_for_Weakly_Supervised_Object_Localization_CVPR_2019_paper.pdf`).
  Instead of adding another crop/drop view, I implemented a light train-time
  token-energy penalty on crop-frame patches plus bbox-edge-band patches. The
  intent was to keep `yolo_f` context available while discouraging the ViT from
  spending discriminative energy on frame/bbox edges.
- Implementation: added optional `border_attention_suppression_*` fields to
  `TrainConfig`, train CLI, `_forward_train_loss`, `train_one_epoch`, history
  telemetry, and V8 launcher metadata/TrainArgs. The helper uses patch-token
  squared energy as an attention proxy, supports token-pruned `patch_indices`,
  combines frame ring and bbox border-band masks, and filters classes
  `0,1,2,4`. Added `tests\test_border_attention_suppression.py`; preflight
  passed: `py_compile` for `train.py`/`config.py`, and pytest
  `tests\test_border_attention_suppression.py` (`3 passed`). Dry-run confirmed
  `yolo_f`, keeper resume, teacher CSV sample-index, bbox spatial fusion,
  pairwise routing, `hard/sample/quality/targeted` manifests empty, and real
  TrainArgs containing all `--border-attention-suppression-*` flags.
- Smoke
  `runs\smoke_v8_yolof_borderattn006_fw1_bb12_bw50_teacherfocusbinary015_24b_1e_20260703`
  used `loss_weight=0.006`, `frame_width=1`, `bbox_band=0.12`,
  `bbox_weight=0.50`, `temperature=0.20`, classes `0,1,2,4`,
  teacher-focus-binary `0.015`, bbox spatial fusion, pairwise routing, and
  `24` train batches for one epoch. The loss was active:
  `train_border_attention_suppression_loss=0.2678`, frame mass `0.1181`,
  bbox-band mass `0.3454`, valid fraction `0.7986`. Validation macro/class-1
  F1 was only `0.8828/0.6764`, class-1 P/R `0.6042/0.7682`, below the keeper
  `0.8847/0.6860` and far below the `0.70` probe gate. Confusion stayed
  `0->1=50`, `1->0=19`, `1->2=11`, `4->1=12`, `3->2=52`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_borderattn006_val_selected12_20260703` showed cleaner
  aggregate heatmaps but not a better decision boundary: attention/Grad-CAM
  foreground mass `0.9832/0.9836`, background mass `0.0168/0.0164`, Grad-CAM
  border flags `4/12`, rollout border `4/12`, object-color-sensitive `3/12`.
  Background blur/gray remained near zero (`0.0003/-0.0006`), while object
  desaturation was still the dominant perturbation (`0.0633`). Manual overlays
  for `0->1`, `4->1`, and `1->2` still highlighted bright surface regions,
  small defects, and near-object/border artifacts rather than adding a new
  robust class-1 surface cue.
- Decision: reject before probe. Border suppression made XAI look somewhat
  cleaner than several earlier rejects, but it reduced class-1 recall and did
  not reduce the decisive false-positive groups. Do not repeat this exact
  `weight=0.006/frame=1/bbox_band=0.12/bbox_weight=0.50/temp=0.20/classes=0,1,2,4`
  setting on the current keeper. If border attention is revisited, it needs a
  case-level reliability or OOF/disagreement signal, not another global penalty
  on the same patch-energy mask.
- Cleanup deleted the rejected border-attention smoke train directory plus
  background launch logs and preserved selected12 XAI plus keeper/final
  artifacts. Manifest:
  `runs\cleanup_manifest_20260703_borderattn006_reject_smoke.json`, reclaimed
  about `181.27 MB`; D: free space after cleanup was about `77.44 GB`.

## Smoke 2026-07-03 - Sub-Center Proxy Head

- Rationale: class 1 is visibly multi-modal and overlaps with classes `0/2/4`;
  the train-split spectral audit also showed class-1 same-label neighbor
  weakness. I checked sub-center ArcFace
  (`https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf`),
  ArcFace's note on K sub-centers for noisy samples
  (`https://arxiv.org/abs/1801.07698`), Proxy-NCA
  (`https://openaccess.thecvf.com/content_ICCV_2017/papers/Movshovitz-Attias_No_Fuss_Distance_ICCV_2017_paper.pdf`),
  and Proxy Anchor (`https://arxiv.org/abs/2003.13911`). The useful idea was
  not another one-center margin, but a small learnable multi-proxy head where a
  class sample can match its closest sub-center. This should tolerate hard/noisy
  modes better than the rejected plain angular-margin and boundary-center losses.
- Implementation: added checkpoint-safe `SubCenterProxyHead` with K normalized
  proxies per class, classifier-weight initialization plus tiny sub-center noise,
  train/config/CLI/V8 launcher flags, resume allow-list support, loss telemetry,
  and `tests\test_subcenter_proxy_loss.py`. The auxiliary CE subtracts an angular
  margin from the target class max-subcenter logit and applies only to configured
  classes. Preflight passed: `py_compile` for `trkh\models\model.py`,
  `trkh\training\train.py`, `trkh\core\config.py`, and focused pytest
  `tests\test_subcenter_proxy_loss.py` (`2 passed`). Dry-run confirmed `yolo_f`,
  keeper resume, teacher CSV sample-index overlap `1.0`, bbox spatial fusion,
  pairwise routing, empty hard/sample/quality/targeted manifests, and active
  `--subcenter-proxy-*` TrainArgs.
- Smoke
  `runs\smoke_v8_yolof_subcenterproxy006_k3_m08_s12_teacherfocusbinary015_24b_1e_20260703`
  used `loss_weight=0.006`, `subcenters=3`, margin `0.08`, scale `12.0`,
  classes `0,1,2,4`, start epoch `1`, teacher-focus-binary `0.015`, bbox
  spatial fusion, pairwise routing, and `24` train batches for one epoch. The
  loss was active (`train_subcenter_proxy_loss=0.9938`, valid fraction
  `0.7986`, positive-negative margin `0.1227`, selected subcenters `2.83`), but
  validation macro/class-1 F1 was only `0.8822/0.6764`, class-1 P/R
  `0.6042/0.7682`, below the keeper `0.8847/0.6860` and below the `0.70` probe
  gate. Confusion was
  `[[486,50,2,0,11],[19,116,11,0,5],[5,10,515,8,6],[0,4,53,652,3],[8,12,4,1,625]]`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_subcenterproxy006_val_selected12_20260703` showed no
  useful failure-mode change: top confusions `3->2=53`, `0->1=50`, `1->0=19`,
  `4->1=12`; attention/Grad-CAM foreground mass `0.9833/0.9835`; background
  blur/gray stayed near zero (`0.0003/-0.0007`); object desaturation remained
  larger (`0.0633`); Grad-CAM border flags were `4/12` and object-color-sensitive
  cases `3/12`.
- Decision: reject before probe. The sub-center proxies were active, but on the
  current frozen representation they did not split class-1 modes in a way that
  reduces `0->1/4->1` false positives or `1->0/1->2` misses. Keep the code path
  because it is isolated and tested, but do not repeat this exact
  `weight=0.006/K=3/margin=0.08/scale=12/classes=0,1,2,4` setting on the current
  keeper. A future proxy method would need a new representation or OOF/reliability
  target, not only different proxy count/weight around this recipe.
- Cleanup deleted the rejected sub-center smoke train directory plus background
  launch logs and preserved selected12 XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260703_subcenterproxy006_reject_smoke.json`, reclaimed
  about `181.76 MB`; D: free space after cleanup was about `77.37 GB`.

## Smoke 2026-07-03 - Compact Bilinear Patch Fusion

- Rationale: XAI keeps showing foreground/object-surface sensitivity while
  background perturbations stay negligible, so I tested a representation change
  rather than another margin/regularizer. Bilinear CNNs for fine-grained
  recognition (`https://arxiv.org/abs/1504.07889`) motivate pooled second-order
  local feature interactions for texture-like cues, and GSoP / covariance pooling
  (`https://arxiv.org/abs/1811.12006`) motivates global second-order statistics
  for stronger nonlinear representations. The repo already had a checkpoint-safe
  `CompactBilinearPatchFusion` residual head, so I used that instead of adding
  another duplicate module.
- Implementation/preflight: no new model code was needed. The existing head uses
  low-rank left/right projections over patch tokens, weighted bilinear pooling,
  signed square-root normalization, L2 normalization, and a zero-initialized
  residual classifier. Preflight passed: `py_compile` for `model.py`,
  `train.py`, `config.py`, and the two bilinear regression tests from
  `tests\test_detection_calibration.py` (`2 passed`).
- Dry-run confirmed `yolo_f`, keeper resume, teacher CSV sample-index path,
  teacher-focus-binary `0.015`, bbox spatial fusion, pairwise routing, empty
  hard/sample/quality/targeted manifests, and
  `bilinear_patch_fusion=true/rank=24/dropout=0.05`.
- Launcher trap: the first run used `-Smoke`, and the train smoke metric was only
  over `64` validation samples because V8 sets `MaxValBatches=2` when `-Smoke`
  is used. That metric was treated as a sanity check only. The checkpoint was
  then evaluated on the full validation split (`2606` samples) with
  `trkh.evaluation.evaluate`; future gated smoke metrics must either avoid
  `-Smoke` or explicitly override/check full-val support.
- Full validation on checkpoint
  `runs\smoke_v8_yolof_bilinear_rank24_drop005_teacherfocusbinary015_24b_1e_20260703\checkpoints\best.pt`
  produced macro/class-1 F1 `0.8824/0.6841`, class-1 P/R `0.6082/0.7815`,
  below the keeper `0.8847/0.6860`. Confusion was
  `[[484,50,1,1,13],[17,118,11,0,5],[5,10,513,9,7],[0,4,53,652,3],[9,12,4,1,624]]`.
  It kept class-1 recall but slightly reduced precision and did not reduce
  `0->1` or `4->1`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_bilinear_rank24_drop005_val_selected12_20260703`
  stayed in the same failure mode: top confusions `3->2=53`, `0->1=50`,
  `1->0=16`, `4->1=12`; attention/Grad-CAM foreground mass `0.9802/0.9846`;
  background blur/gray near zero (`0.0002/-0.0004`); object desaturation still
  dominant (`0.0628`); Grad-CAM border flags `4/12`, rollout border `5/12`, and
  object-color-sensitive `3/12`.
- Decision: reject before probe. Second-order patch texture residuals are a
  coherent FGVC idea, but this low-rank residual head did not create a new
  class-1 surface/boundary cue on the current keeper. Do not repeat
  `BilinearPatchFusion=true/rank=24/dropout=0.05` on the current anchor, and do
  not just tune nearby rank/dropout without a new reliability/OOF or case-level
  signal.
- Cleanup deleted the rejected bilinear smoke train directory, its background
  logs, and the full-val eval artifact after recording metrics; selected12 XAI
  plus keeper/final artifacts were preserved. Manifest:
  `runs\cleanup_manifest_20260703_bilinear_rank24_reject_smoke_eval.json`,
  reclaimed about `186.66 MB`; D: free space after cleanup was about `77.31 GB`.

## Smoke 2026-07-03 - Part-Token Pairwise Head

- Rationale: after whole-patch second-order pooling failed, I tested a more local
  part-token route. TransFG (`https://arxiv.org/abs/2103.07976`) motivates using
  transformer token importance to select discriminative patches for fine-grained
  recognition, and NTS-Net (`https://arxiv.org/abs/1809.00287`) supports the
  general idea of learning informative local regions without manual part labels.
  The repo already had a bbox/foreground-prior `AdaptivePartTokenLearner`, so I
  used only the pairwise head to keep the intervention narrow.
- Implementation/preflight: no new code was needed. The pairwise head learns
  part queries over patch tokens, biases attention by foreground/bbox prior,
  pools part descriptors, and routes pairwise residual logits only near configured
  class-pair boundaries. Focused test `tests\test_part_token_learner.py` passed
  (`7 passed`). Dry-run confirmed `yolo_f`, keeper resume, full-val
  `MaxValBatches=0`, teacher CSV sample-index path, empty stale manifests,
  `PartTokenPairwiseHead=true`, pairs `0-1,1-2,4-1,2-3`, logit scale `0.16`,
  dropout `0.05`, bbox weight `0.80`, foreground power `1.20`, and aux BCE
  weight `0.02`.
- Smoke
  `runs\smoke_v8_yolof_parttokenpair020_p4_teacherfocusbinary015_24b_1e_20260703`
  used `24` train batches for one epoch and evaluated the full validation split.
  The part-token pairwise loss was active (`train_part_token_pairwise_loss=0.6914`),
  but validation macro/class-1 F1 was only `0.8828/0.6784`, class-1 P/R
  `0.6073/0.7682`, below the keeper and below the `0.70` gate. Confusion was
  `[[486,49,2,0,12],[19,116,11,0,5],[5,10,515,8,6],[0,4,52,653,3],[8,12,4,1,625]]`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_parttokenpair020_p4_val_selected12_20260703` stayed
  unchanged: top confusions `3->2=52`, `0->1=49`, `1->0=19`, `4->1=12`;
  attention/Grad-CAM foreground mass `0.9796/0.9843`; background blur/gray near
  zero (`0.0003/-0.0003`); object desaturation still larger (`0.0627`);
  Grad-CAM border flags `4/12`, rollout border `5/12`, and object-color-sensitive
  `3/12`.
- Decision: reject before probe. The learned part-token route was active but did
  not find a new discriminative surface cue; it slightly reduced `0->1` vs keeper
  but lost class-1 recall and kept `4->1`. Do not repeat this exact
  `PartTokenPairwiseHead=true/weight=0.02/part_count=4/foreground_power=1.20/
  bbox_weight=0.80/pairs=0-1,1-2,4-1,2-3/logit_scale=0.16/dropout=0.05` setting
  on the current keeper.
- Cleanup deleted the rejected part-token smoke train directory plus background
  logs and preserved selected12 XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260703_parttokenpair020_reject_smoke.json`, reclaimed
  about `186.49 MB`; D: free space after cleanup was about `77.24 GB`.

## Smoke 2026-07-03 - Teacher Pairwise Margin Distillation

- Rationale: the AIDT/TRKH complementarity audit suggested that a teacher can
  identify some class-pair disagreements, but earlier global/focus KD variants
  transferred too broadly. I checked dark-knowledge KD
  (`https://arxiv.org/abs/1503.02531`), decoupled KD
  (`https://arxiv.org/abs/2203.08679`), relational KD
  (`https://arxiv.org/abs/1904.05068`), and noisy/fine-grained selective
  contrastive learning (`https://arxiv.org/abs/2203.04181`,
  `https://arxiv.org/abs/2303.02404`). The narrow test was to use teacher
  pairwise probabilities only as a margin signal on likely confusion pairs,
  while keeping inference as single no-pretrain TRKH.
- Implementation/preflight: no new code was needed; the existing
  teacher-pairwise margin path was used. `py_compile` passed for
  `trkh\models\model.py`, `trkh\training\train.py`, and `trkh\core\config.py`.
  Focused pytest passed for
  `tests\test_focused_false_positive_margin.py::test_teacher_pairwise_margin_uses_teacher_pair_probabilities_and_agreement`
  and then the full file (`8 passed`). Dry-run confirmed `yolo_f`, keeper
  resume, strict sample-index teacher CSV, empty stale manifests, full-val
  `MaxValBatches=0`, `teacher_pairwise_margin_loss_weight=0.006`,
  `mass_threshold=0.20`, `error_power=0.50`, hard target blend `0.05`, and
  agreement required.
- Smoke
  `runs\smoke_v8_yolof_teacherpair006_mass20_err05_teacherfocusbinary015_24b_1e_20260703`
  used `24` train batches for one epoch and evaluated the full validation split
  (`2606` samples). The loss was active
  (`train_teacher_pairwise_margin_loss=0.6517`, fraction `0.5358`, student
  pair probability `0.4613`, teacher pair probability `0.3669`), alongside
  teacher-focus-binary `0.015`. Validation macro/class-1 F1 was only
  `0.8828/0.6784`, class-1 P/R `0.6073/0.7682`, below the keeper
  `0.8847/0.6860` and below the `0.70` probe gate. Top confusions stayed
  `3->2=52`, `0->1=49`, `1->0=19`, `4->1=12`, `0->4=11`, `1->2=11`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_teacherpair006_mass20_err05_val_selected12_20260703`
  showed the same failure mode: attention/Grad-CAM foreground mass
  `0.9833/0.9834`, background mass `0.0167/0.0166`, Grad-CAM border flags
  `4/12`, rollout border `4/12`, and object-color-sensitive `3/12`.
  Background blur/gray stayed near zero (`0.0005/-0.0004` original-prediction
  drop), while object desaturation remained much larger (`0.0632`).
- Decision: reject before probe. Teacher pairwise margins were active, but they
  did not reduce the decisive `0->1/4->1` false positives or `1->0/1->2`
  misses. Do not repeat this exact
  `teacher_pairwise_margin_weight=0.006/mass_threshold=0.20/error_power=0.50/
  hard_target_blend=0.05/require_agreement=true` setting on the current keeper.
  Any future teacher-pairwise transfer needs a new fold-safe reliability,
  disagreement, or case-level routing signal, not another nearby weight/threshold
  tweak.
- Cleanup deleted the rejected teacher-pairwise smoke train directory plus
  background launch logs and preserved selected12 XAI plus keeper/final
  artifacts. Manifest:
  `runs\cleanup_manifest_20260703_teacherpair006_reject_smoke.json`, reclaimed
  about `181.31 MB`; D: free space after cleanup was about `77.17 GB`.

## Smoke 2026-07-03 - YOLO Primary With Class-F Auxiliary Training

- Rationale: the user asked to combine `yolo_f` and `class_f` without changing
  the raw datasets. I checked foreground/background recombination for ViT
  robustness (`https://arxiv.org/abs/2503.09399`), fine-grained domain
  generalization (`https://arxiv.org/abs/2406.09166`), and global/local context
  ViT design (`https://jankautz.com/publications/GCViT_ICML23.pdf`). The narrow
  test was to keep `yolo_f` as the validation/test/inference view, add `class_f`
  only as a train-time auxiliary view with low weight, and preserve the current
  bbox spatial fusion, boundary dropout, pairwise routing, and
  teacher-focus-binary anchor.
- Implementation/preflight: mixed training already existed, but offline teacher
  probabilities were sample-index strict and the original `yolo_f` teacher CSV
  covered only `9215` primary rows. I added
  `trkh.tools.build_mixed_teacher_probability_cache` plus
  `tests\test_build_mixed_teacher_probability_cache.py` so a mixed dataset can
  receive a strict teacher cache matching the primary+auxiliary sample-index
  layout. Preflight passed: `py_compile` for the new tool and touched train/data
  modules, plus focused pytest for the mixed-cache tool and mixed-dataset
  metadata test (`2 passed`).
- The generated cache
  `runs\mixed_teacher_cache_yolof_auxclassf025_focus005_20260703\teacher_probs_train_sampleindex_mixed_yolof_auxclassf025_focus005.csv`
  contains `11504` rows: `9215` from `yolo_f` and `2289` from `class_f`, with
  source names and effective weights preserved. A failed partial run with the
  old primary-only teacher cache was deleted and recorded in
  `runs\cleanup_manifest_20260703_auxclassf025_failed_teacher_cache_preflight.json`.
- Smoke
  `runs\smoke_v8_yolof_auxclassf025_teacherfocusbinary015_24b_1e_20260703`
  used `AuxiliaryTrainData=class_f`, `AuxiliaryTrainWeight=0.25`, paired YOLO
  metadata from `yolo_f`, the mixed teacher cache, `24` train batches for one
  epoch, and full validation support `2606`. The auxiliary mix and losses were
  active, but validation macro/class-1 F1 was only `0.8834/0.6804`, class-1
  P/R `0.6105/0.7682`, below the keeper `0.8847/0.6860` and below the `0.70`
  probe gate. Confusion was
  `[[489,48,2,0,10],[19,116,11,0,5],[5,10,514,8,7],[0,4,53,652,3],[8,12,4,1,625]]`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_auxclassf025_val_selected12_20260703` showed no new
  useful signal. Top confusions stayed `3->2=53`, `0->1=49`, `1->0=19`,
  `4->1=12`, and `1->2=11`. Attention/Grad-CAM foreground mass stayed high
  (`0.9779/0.9794`), background blur/gray original-prediction drops were near
  zero (`0.0004/-0.0002`), and object desaturation was much larger (`0.0731`).
  Grad-CAM border flags increased to `7/12` and rollout border flags to `6/12`.
  Manual overlay review of `0->1` and `4->1` cases showed the main heat still
  on fruit surface/edge with occasional bright-background hotspots, matching the
  prior surface/boundary bottleneck rather than proving that wider context helps.
- Decision: reject before probe. Plain auxiliary `class_f` exposure at weight
  `0.25` did not improve class 1 and slightly worsened border saliency. Do not
  repeat this exact `AuxiliaryTrainData=class_f/AuxiliaryTrainWeight=0.25` mixed
  recipe with the same teacher-focus-binary/bboxprior/boundarydrop anchor unless
  a new reliability, OOF, or foreground/background recombination signal changes
  the supervision. Cleanup deleted the rejected smoke train directory and
  preserved the selected12 XAI audit, mixed teacher cache, keeper, and final
  artifacts. Manifest:
  `runs\cleanup_manifest_20260703_auxclassf025_reject_smoke.json`, reclaimed
  about `181.16 MB`; D: free space after cleanup was about `77.11 GB`.

## Smoke/Probe 2026-07-03 - Frequency-Selective Pooling On Current Anchor

- Rationale: the previous XAI/audit pattern says far background is not the main
  bottleneck, but class-1 decisions remain sensitive to object surface/color and
  some border energy. I checked LaST-ViT / "Vision Transformers Need More Than
  Registers" (`https://arxiv.org/abs/2602.22394`) and its public implementation
  (`https://github.com/ChengShiest/LAST-ViT`) because frequency-selective pooling
  is a lightweight way to bias global tokens away from noisy high-frequency
  activation without adding a new classifier family. This had also shown a weak
  historical signal on an older V15 branch, so I retested it on the current
  yolo_f teacher-focus-binary keeper rather than relying on stale evidence.
- Implementation/preflight: no new code was needed; the existing
  `frequency_selective_pooling` path was used. Preflight passed:
  `py_compile` for `trkh\models\model.py`, `trkh\training\train.py`, and
  `trkh\core\config.py`, plus the focused regression tests from
  `tests\test_detection_calibration.py -k frequency_selective_pooling`
  (`3 passed`). Dry-run confirmed `yolo_f`, keeper resume, full validation
  (`MaxValBatches=0`), empty stale manifests, bbox spatial fusion, pairwise
  routing, boundary dropout, teacher-focus-binary `0.015`, and
  `FrequencySelectivePooling=true/top_k=1/blend=0.12/threshold=0.50`.
- Smoke
  `runs\smoke_v8_yolof_freqsel012_thr50_teacherfocusbinary015_24b_1e_20260703`
  used `24` train batches for one epoch and evaluated the full validation split
  (`2606` samples). Validation macro/class-1 F1 was `0.8856/0.6866`, class-1
  P/R `0.6250/0.7616`. This was a small macro/precision bump over the keeper,
  but still below the `0.70` class-1 probe milestone and with lower recall.
  Confusion was
  `[[488,45,1,2,13],[19,115,12,0,5],[5,10,512,11,6],[0,3,46,661,2],[9,11,4,1,625]]`.
- XAI smoke
  `runs\xai_smoke_v8_yolof_freqsel012_thr50_val_selected12_20260703` showed a
  cleaner but not decisive pattern: top confusions `0->1=45`, `3->2=45`,
  `1->0=19`, `0->4=13`, `1->2=12`; attention/Grad-CAM foreground mass
  `0.9828/0.9799`; background blur/gray drops near zero (`0.00027/-0.00032`);
  object desaturation `0.0810`; Grad-CAM border `7/12`, rollout border `5/12`.
  Manual overlay still showed the main class-1 false-positive heat on fruit top
  plus a bright-background hotspot. Because the smoke improved several
  confusions without a large XAI regression, it was allowed a short probe.
- Probe
  `runs\probe_v8_yolof_freqsel012_thr50_teacherfocusbinary015_120b_2e_20260703`
  did not hold the smoke gain. Epoch 1 reached macro/class-1 F1
  `0.8850/0.6784`, class-1 P/R `0.6073/0.7682`; the selected `best.pt` from
  epoch 2 had macro/class-1 F1 `0.8825/0.6800`, class-1 P/R
  `0.5980/0.7881`, with confusion
  `[[481,53,2,0,13],[15,119,12,0,5],[5,11,514,8,6],[0,4,51,654,3],[9,12,3,1,625]]`.
  Both epochs stayed below the keeper class-1 F1 and below the `0.70` gate.
- XAI probe
  `runs\xai_probe_v8_yolof_freqsel012_thr50_val_selected12_20260703` confirmed
  the regression: top confusions `0->1=53`, `3->2=51`, `1->0=17`,
  `0->4=13`, `1->2=12`, `4->1=12`; attention/Grad-CAM foreground mass
  `0.9881/0.9774`; background blur/gray still near zero (`0.00114/0.00018`);
  object desaturation increased to `0.11118`; Grad-CAM border `5/12` and
  object-color-sensitive cases `5/12`. The probe traded transient precision for
  higher object-color/surface sensitivity and did not stabilize the boundary.
- Decision: reject before full train. Do not repeat
  `FrequencySelectivePooling=true/top_k=1/blend=0.12/threshold=0.50` on the
  current keeper. If frequency pooling is revisited, it needs a stronger
  foreground reliability/OOF mask or case-level signal, not another small blend
  tweak. Cleanup deleted the rejected smoke and probe train directories while
  preserving smoke/probe XAI plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260703_freqsel012_thr50_reject_smoke_probe.json`,
  reclaimed about `363.69 MiB` (`381,357,605` bytes); D: free space after
  cleanup was about `76.98 GiB`.

## Smoke 2026-07-03 - Focus-Class Residual Head On Current Anchor

- Rationale: class 1 remains the limiting boundary and the user asked for hybrid
  logic that uses stronger classifier routing without changing the raw data. I
  checked Class-Independent Regularization
  (`https://dayan-guan.github.io/pub/CIR.pdf`) and binary class-decomposition
  ideas because reducing multi-class negative coupling can help under noisy
  fine-grained labels. The exact OVA/CIR-lite auxiliary head had already failed,
  so this smoke tested a narrower variant: a zero-init residual logit only for
  class 1, routed only when class 1 is plausible by top-2/margin logic, while
  preserving the current yolo_f keeper, bbox spatial fusion, boundary-band
  dropout, pairwise routing, and teacher-focus-binary anchor.
- Implementation/preflight: the focus-class residual head path was already
  implemented and checkpoint-safe. Preflight passed: `py_compile` for
  `trkh\models\model.py`, `trkh\training\train.py`, and `trkh\core\config.py`,
  plus `tests\test_focus_class_head.py` (`2 passed`). The first dry-run caught
  stale defaults that would have invalidated the comparison: `DistillationWeight`
  had defaulted to `0.1`, and bbox foreground dropout was not active. The final
  dry-run confirmed `DistillationWeight=0.0`, teacher-focus-binary `0.015`,
  bbox dropout loss/consistency/probability `0.08/0.03/0.45`, no stale hard
  sample manifest, `MaxValBatches=0`, yolo_f primary data, and focus-class
  routing active.
- Smoke
  `runs\smoke_v8_yolof_focushead012_aux010_pos125_teacherfocusbinary015_24b_1e_20260703`
  used `FocusClassHead=true`, logit scale `0.12`, dropout `0.05`, routing
  margin/probability `0.28/0.06`, aux loss `0.010`, positive weight `1.25`,
  `24` train batches for one epoch, and full validation support `2606`. The
  auxiliary signal was active (`train_focus_class_aux_loss=0.7275`) and
  distillation stayed disabled (`train_distillation_loss=0.0`), but validation
  macro/class-1 F1 was only `0.8834/0.6784`, class-1 P/R
  `0.6073/0.7682`, below the keeper `0.8847/0.6860` and below the `0.70`
  probe gate. Confusion was
  `[[487,49,2,0,11],[19,116,11,0,5],[5,10,515,8,6],[0,4,51,654,3],[8,12,4,1,625]]`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_focushead012_aux010_pos125_val_selected12_20260703`
  did not show a new useful signal. Top confusions stayed `3->2=51`,
  `0->1=49`, `1->0=19`, `4->1=12`, `0->4=11`, and `1->2=11`.
  Attention/Grad-CAM foreground mass stayed high (`0.9781/0.9794`), background
  blur/gray original-prediction drops were negligible (`0.00027/-0.00070`),
  and object desaturation was much larger (`0.0731`). Review flags were still
  border-heavy for Grad-CAM (`7/12`) and rollout (`6/12`). Manual overlays of
  `0->1` and `3->2` cases showed heat mostly on fruit surface/edge with some
  local bright/background artifacts, matching the established surface/boundary
  bottleneck rather than justifying a probe.
- Decision: reject before probe. A narrow class-1 residual head did not improve
  precision, recall, or XAI failure mode on the current embedding. Do not repeat
  `FocusClassHead=true/FocusClassLogitScale=0.12/FocusClassAuxLossWeight=0.010/
  FocusClassAuxPositiveWeight=1.25/FocusClassRouteMaxProbabilityMargin=0.28/
  FocusClassRouteMinProbability=0.06` on the current keeper. Cleanup deleted the
  rejected smoke train directory while preserving selected12 XAI, keeper, and
  final artifacts. Manifest:
  `runs\cleanup_manifest_20260703_focushead012_aux010_pos125_reject_smoke.json`,
  reclaimed about `181.34 MiB` (`190,147,312` bytes); D: free space after
  cleanup was about `76.91 GiB`.

## Diagnostic 2026-07-03 - Counterfactual Color Guard

- Rationale: XAI repeatedly shows object-color/surface sensitivity while
  background blur/gray has near-zero effect. I checked TTA/confidence-estimation
  work before implementation: Bahat and Shakhnarovich use test-time
  transformations for confidence estimation (`https://arxiv.org/abs/2006.16705`),
  Shanmugam et al. show naive TTA averaging can be suboptimal
  (`https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf`),
  and MEMO uses augmentation consistency for robustness
  (`https://proceedings.neurips.cc/paper_files/paper/2022/file/fc28053a08f59fccb48b11f2e31e81c7-Paper-Conference.pdf`).
  For this mango dataset, color is label-relevant, so the test was deliberately
  a narrow diagnostic/guard, not color-invariant averaging.
- Implementation/preflight: added
  `trkh.tools.evaluate_counterfactual_color_guard` and
  `tests\test_counterfactual_color_guard.py`. The tool runs the keeper on clean
  crops and on an object-desaturated counterfactual, can sweep a very narrow
  class-1 demotion guard, and writes CSV/summary artifacts without changing raw
  data or model weights. Preflight passed after a cleanup fix for dynamic
  `focus_class`: `py_compile` plus `tests\test_counterfactual_color_guard.py`
  (`3 passed`).
- Val smoke `max_samples=256`
  `runs\counterfactual_color_guard_keeper_val_smoke_20260703` showed the
  counterfactual itself is destructive: object desaturation changed
  `253/256` predictions, with `0` corrections and `247` harms. The desaturated
  predictions collapsed mostly to class 4, confirming color is a true semantic
  cue here and not a nuisance variable that can be averaged away.
- A full train split diagnostic was started but stopped because it made no
  artifact after several minutes and was CPU/I/O-bound; the smoke already showed
  the transform was too destructive. A faster train subset diagnostic
  `runs\counterfactual_color_guard_keeper_train1024_20260703` used `1024`
  train objects. Clean subset macro/class-1 F1 was `0.9503/0.8000`, but the
  desaturated view had macro F1 only `0.0054` and changed `1019/1024`
  predictions, with `0` corrections and `999` harms. The fold guard selected
  `drop_threshold=0.16`, `max_clean_margin=0.08`,
  `max_clean_confidence=0.34`, but the subset was class-distribution biased
  (`class4` support only `1`) and was used only to freeze a diagnostic rule.
- Applying that train-subset rule to full validation in
  `runs\counterfactual_color_guard_keeper_val_train1024fit_20260703` gave
  guarded macro/class-1 F1 `0.8840/0.6842`. It changed only one useful
  class-1 false positive (`0->1` from `49` to `48`) and moved one class-0 sample
  to class 4, leaving `1->0`, `1->2`, and `4->1` unchanged. This remains below
  the keeper class-1 F1 `0.6860` and far below the `0.70` probe gate.
- Decision: reject as a model/inference route. The diagnostic is useful evidence:
  object color is indispensable, and a narrow color-drop guard cannot reliably
  separate class 1 from the surrounding classes. Do not repeat
  object-desaturate counterfactual voting/guard with config
  `0.16/0.08/0.34` on the current keeper. If counterfactual color is revisited,
  it needs fold-safe reliability plus a less destructive, label-preserving
  perturbation; simple desaturation or color-logit averaging is ruled out.
- Cleanup deleted the rejected diagnostic artifacts after metrics were recorded:
  val smoke, aborted full-train diagnostic shell logs, train1024 subset, and
  full-val train1024-fit directories. Manifest:
  `runs\cleanup_manifest_20260703_counterfactual_color_guard_reject_diagnostic.json`,
  reclaimed about `1.72 MiB` (`1,806,565` bytes); D: free space after cleanup
  was about `76.91 GiB`.

## Diagnostic 2026-07-03 - Geometric HFlip TTA

- Rationale: after color counterfactual failed, I checked label-preserving
  geometric TTA instead of photometric/color TTA. The relevant research signal
  is mixed: classical TTA includes flips/crops, Shanmugam et al. show
  aggregation choice matters
  (`https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf`),
  Cyclic TTA with entropy weighting treats horizontal flip as a common but
  dataset-dependent transform
  (`https://proceedings.mlr.press/v180/chun22a/chun22a.pdf`), and 2026
  evidence warns that standard TTA can hurt classification under transform
  shift (`https://arxiv.org/abs/2604.09697`). Therefore this was run as a
  diagnostic only, not as a default inference change.
- Hygiene: the first full-val run accidentally used default
  `D:\DataAI\AIEx\dataset` instead of the valid
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`; it was rejected and deleted via
  `runs\cleanup_manifest_20260703_hflip_wrong_dataset_reject.json`. I also
  deleted obsolete `eval_smoke_*` artifacts and the empty failed hflip dir via
  `runs\cleanup_manifest_20260703_obsolete_eval_smoke_and_empty_hflip.json`
  (`7` directories, `17.04 MiB` reclaimed).
- Valid run: `runs\geometric_tta_hflip_keeper_yolof_val_full_noamp_20260703`
  used explicit `yolo_f` data, support `2606`, no AMP, and matched the existing
  evaluator baseline exactly: clean macro/class-1 F1 `0.8829/0.6783`, class-1
  P/R `0.6031/0.7748`, confusion
  `[[487,50,1,1,10],[18,117,11,0,5],[5,10,514,8,7],[0,4,52,653,3],[8,13,4,1,624]]`.
  The duplicate AMP run was removed via
  `runs\cleanup_manifest_20260703_hflip_amp_duplicate_reject.json`.
- Results: pure flip hurt (`0.8786/0.6553`). Average probability and
  entropy-weighted aggregation were effectively identical and only moved macro
  F1 to `0.8830` while lowering class-1 F1 to `0.6762`; class-1 P/R became
  `0.5960/0.7815`. Entropy-select gave class-1 recall `0.8013` but precision
  collapsed to `0.5874`, with class-1 F1 only `0.6779`.
- Audit of changed predictions for `avg_prob`: it changed `37` predictions,
  made `18` corrections and `17` harms. Useful class-1 FN repairs were
  `1->0->1` (`5`) and `1->2->1` (`1`), but this was offset by new class-1
  false positives: `0->0->1` (`6`), `4->4->1` (`2`), and `2->2->1` (`1`).
  Existing class-1 FP stayed mostly unchanged (`0->1`, `4->1`, `2->1`), so
  the transform improves recall by widening the class-1 region rather than
  learning a better boundary.
- Decision: reject as an inference route. Do not repeat hflip geometric TTA
  with simple probability/logit/entropy aggregation on the current keeper. If a
  geometric transform is revisited, it must be fold-safe and case-selective
  with a pre-declared reliability signal; simple hflip averaging is below both
  the keeper best (`0.8847/0.6860`) and the `0.70` class-1 gate.

## Diagnostic 2026-07-03 - Head Pooling Ablation On Current Keeper

- Rationale: before adding another local/global hybrid block, I checked whether
  the current branch/register aggregation itself was hiding a better decision
  boundary. This is a zero-training diagnostic on the same keeper checkpoint and
  explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, so it does not touch
  the raw dataset and does not tune the validation split.
- Valid run:
  `runs\head_pooling_keeper_yolof_val_diagnostic_20260703` evaluated full
  validation support `2606`, no AMP, and explicit `yolo_f`. The current
  `cls_branch_register_mean` path matched the evaluator baseline at
  macro/class-1 F1 `0.8829/0.6783`. Dropping the branch tokens to
  `cls_register_mean` fell to `0.8616/0.6310`, with `137` changed predictions,
  `38` corrections, and `84` harms. Pure `cls` fell further to
  `0.8589/0.6163`, with `133` changed predictions, `36` corrections, and
  `82` harms.
- Decision: keep the current branch/register pooling and do not train
  `cls_register_mean` or `cls` head-pooling variants on this anchor. The ablation
  confirms that the branch/register tokens are carrying useful complementary
  color/edge/local information; weakening the aggregation destroys class-1
  precision/recall rather than exposing a cleaner boundary. Future architecture
  work should add checkpoint-safe local evidence around the existing aggregation,
  not replace it.

## Smoke 2026-07-03 - Patch-Memory Local Adapter On Current Keeper

- Rationale: after head-pooling showed that current branch/register aggregation
  should be preserved, I checked lightweight local inductive bias work instead
  of another classifier-only head. LocalViT (`https://arxiv.org/abs/2104.05707`)
  and CeiT/LeFF
  (`https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_Incorporating_Convolution_Designs_Into_Visual_Transformers_ICCV_2021_paper.pdf`)
  both support injecting local convolutional evidence into ViT token processing.
  For this repo I implemented a much smaller checkpoint-safe post-transformer
  `PatchMemoryAdapter`: LayerNorm plus depthwise/pointwise convolution on patch
  tokens only, with zero-init residual output so old checkpoints load without
  changing inference before training.
- Implementation/preflight: added `patch_memory_adapter` and
  `patch_memory_adapter_dropout` to `ModelConfig`, `train.py`, the V8 launcher,
  and model construction. The first smoke attempt failed before training because
  resume partial-load allowlist did not yet include
  `patch_memory_adapter.*`; that failed artifact was deleted via
  `runs\cleanup_manifest_20260703_patchmem_failed_resume_allowlist.json`
  after adding the allowlist. Focused preflight then passed:
  `py_compile` for `model.py/train.py/config.py` and
  `tests\test_patch_memory_adapter.py` (`3 passed`), including a regression test
  that an old checkpoint state can load with adapter keys missing.
- Valid smoke:
  `runs\smoke_v8_yolof_patchmem000_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260703`
  used explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, resumed the
  current keeper checkpoint, kept bbox spatial fusion and boundary-band
  foreground dropout (`0.08/0.03/p=0.45/temp=1.2`), kept pairwise routing and
  teacher-focus-binary `0.015`, disabled KD (`DistillationWeight=0`), trained
  `24` batches for `1` epoch, and evaluated full validation support `2606`.
  Patch-memory added about `68.6k` parameters (`7,314,198` total vs keeper
  `7,245,590`) and was active, but full-val macro/class-1 F1 reached only
  `0.8822/0.6764`, class-1 P/R `0.6042/0.7682`, below both the keeper
  `0.8847/0.6860` and the `0.70` probe gate. Confusion was
  `[[486,50,2,0,11],[19,116,11,0,5],[5,10,514,8,7],[0,4,52,653,3],[8,12,4,1,625]]`.
  The active losses matched the intended recipe: metric learning `2.0659`,
  foreground consistency `0.0647`, bbox dropout `1.0485/0.0043`, pairwise margin
  `0.5649`, teacher-focus-binary `0.5317`, and KD `0.0`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_patchmem000_val_selected12_20260703` confirmed no new
  useful signal. Top confusions stayed `3->2=52`, `0->1=50`, `1->0=19`,
  `4->1=12`, and `0->4=11`. Attention/Grad-CAM foreground mass stayed high
  (`0.9780/0.9794`), background blur/gray original-prediction drops stayed near
  zero (`0.00036/-0.00052`), and object desaturation remained larger
  (`0.0730`). Review flags stayed border-heavy (`gradcam_border=7/12`,
  `rollout_border=6/12`). Manual overlay review of a `0->1` case showed Grad-CAM
  on fruit edge/dark adjacent artifacts and nearby background; a `3->2` case
  still focused on lower edge/shadow rather than a better surface cue.
- Decision: reject before probe. The zero-init local patch adapter is
  checkpoint-safe and technically clean, but the smoke did not improve the
  class-1 boundary or the XAI failure mode. Do not repeat
  `PatchMemoryAdapter=true/dropout=0.0` on the current keeper, and do not tune
  nearby dropout/weight around this adapter unless a new reliability, OOF, or
  case-level supervision signal changes the patch-token target. Cleanup deleted
  the rejected smoke train directory while preserving the selected12 XAI audit,
  keeper, and final artifacts. Manifest:
  `runs\cleanup_manifest_20260703_patchmem000_reject_smoke.json`, reclaimed
  about `182.42 MiB` (`191,276,944` bytes).

## Smoke 2026-07-03 - LogitNorm Loss On Current Keeper

- Rationale: class-1 still suffers from overconfident boundary mistakes, so I
  checked a loss-level calibration change before adding more architecture. The
  LogitNorm paper (`https://proceedings.mlr.press/v162/wei22d/wei22d.pdf`)
  normalizes logits and divides by a temperature to reduce confidence magnitude
  shortcuts. I also checked JoCoR
  (`https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.pdf`)
  and recent noisy/imbalanced label work, but chose LogitNorm first because it
  is a small single-loss swap and does not add a second network or another
  train-in-sample selection signal.
- Implementation/preflight: added optional `ClassificationLoss=logit_norm` with
  `LogitNormTemperature=0.04` to `trkh.training.losses`, `TrainConfig`,
  `train.py`, and the V8 launcher. The implementation supports hard labels,
  soft targets, class weights, and label smoothing. Focused preflight passed:
  `py_compile` for `losses.py/train.py/config.py` plus
  `tests\test_logit_norm_loss.py` (`2 passed`). Dry-run confirmed explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, keeper resume, full-val
  `MaxValBatches=0`, bbox spatial fusion, boundary-band dropout, pairwise
  routing, teacher-focus-binary `0.015`, and no stale hard-sample manifest.
- Valid smoke:
  `runs\smoke_v8_yolof_logitnorm_t004_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260703`
  used the same current keeper recipe as the patch-memory smoke, but replaced
  the main classification criterion with LogitNorm temperature `0.04`. It
  trained `24` batches for `1` epoch and evaluated full validation support
  `2606`. The result was validation macro/class-1 F1 `0.8818/0.6764`,
  class-1 P/R `0.6042/0.7682`, below the keeper `0.8847/0.6860` and below the
  `0.70` probe gate. Confusion was
  `[[486,50,2,0,11],[19,116,11,0,5],[6,10,514,8,6],[0,4,53,652,3],[8,12,4,1,625]]`.
  Active losses matched the intended recipe: LogitNorm CE `0.5329`, metric
  learning `2.0611`, foreground consistency `0.0672`, bbox dropout
  `0.7587/0.0042`, pairwise margin `0.5649`, teacher-focus-binary `0.5317`,
  and KD `0.0`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_logitnorm_t004_val_selected12_20260703` confirmed
  the same failure mode. Top confusions were `3->2=53`, `0->1=49`,
  `1->0=19`, `4->1=12`, and `0->4=11`. Attention/Grad-CAM foreground mass
  stayed high (`0.9780/0.9794`), background blur/gray drops stayed near zero
  (`0.00073/-0.00053`), and object desaturation stayed much larger
  (`0.0728`). Review flags stayed border-heavy (`gradcam_border=7/12`,
  `rollout_border=6/12`). Manual overlay review showed the same edge/shadow and
  local background-adjacent hotspots on `0->1`, `4->1`, and `3->2` mistakes.
- Decision: reject before probe. LogitNorm reduced neither class-1 false
  positives nor class-1 false negatives and did not change the XAI bottleneck.
  Do not repeat `ClassificationLoss=logit_norm/LogitNormTemperature=0.04` on
  the current keeper unless a new representation or OOF/reliability target
  changes the inputs. Cleanup deleted the rejected smoke train directory and
  launcher logs while preserving the selected12 XAI audit, keeper, and final
  artifacts. Manifest:
  `runs\cleanup_manifest_20260703_logitnorm_t004_reject_smoke.json`, reclaimed
  about `181.32 MiB` (`190,124,491` bytes).

## Smoke 2026-07-03 - Foreground/Background Recombination On Current Keeper

- Rationale: the user explicitly asked whether the wider `yolo_f` context can
  help ViT separate foreground from background. I checked foreground/background
  recombination work before running the repo's existing implementation. ForAug
  (`https://arxiv.org/html/2503.09399v4`) recombines foregrounds and backgrounds
  for image classification and reports reduced background/position/scale bias.
  The ImageNet-9/background-bias line
  (`https://openreview.net/forum?id=gl3D-xY7wLq`) also supports auditing whether
  background is a shortcut. This repo already had a train-time
  `_apply_foreground_background_mix_batch` path in the classification collator,
  so I used that instead of writing duplicate code.
- Implementation/preflight: no new code was needed. The existing collator keeps
  pseudo foreground from each normalized crop and replaces the remaining
  background with another batch image. Focused preflight passed:
  `py_compile trkh\data\dataset.py trkh\training\train.py trkh\core\config.py`
  and `tests\test_detection_calibration.py -k foreground_background_mix`
  (`2 passed`). Dry-run confirmed explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, keeper resume,
  `classification_loss=ldam_focal`, `hard_sample_manifest=""`, full-val
  `MaxValBatches=0`, bbox spatial fusion, boundary-band dropout, pairwise
  routing, teacher-focus-binary `0.015`, and
  `ForegroundBackgroundMixProbability=0.35`.
- Valid smoke:
  `runs\smoke_v8_yolof_fgbgmix035_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260703`
  trained `24` batches for `1` epoch and evaluated full validation support
  `2606`. Validation macro/class-1 F1 was `0.8834/0.6784`, class-1 P/R
  `0.6073/0.7682`, below the keeper `0.8847/0.6860` and below the `0.70`
  probe gate. Confusion was
  `[[488,49,2,0,10],[19,116,11,0,5],[6,10,514,8,6],[0,4,52,654,2],[8,12,4,1,625]]`.
  The augmentation made train loss noisier (`1.0543`) and active losses
  matched the intended recipe: CE `0.8450`, metric `2.1013`, foreground
  consistency `0.0574`, bbox dropout `1.1625/0.0045`, pairwise margin
  `0.5662`, teacher-focus-binary `0.5326`, and KD `0.0`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_fgbgmix035_val_selected12_20260703` showed no useful
  shift. Top confusions stayed `3->2=52`, `0->1=49`, `1->0=19`, `4->1=12`,
  and `1->2=11`. Attention/Grad-CAM foreground mass stayed high
  (`0.9781/0.9794`), background blur/gray drops stayed near zero
  (`0.00054/-0.00061`), and object desaturation stayed much larger (`0.0734`).
  Review flags were unchanged (`gradcam_border=7/12`, `rollout_border=6/12`).
  Manual overlays for `0->1` and `4->1` were visually almost identical to the
  previous smokes: edge/surface hotspots plus local background-adjacent artifacts.
- Decision: reject before probe. Foreground/background recombination is a valid
  idea for ViT background bias, but this dataset's current keeper is already
  mostly foreground-driven and the bottleneck remains fruit surface/boundary
  ambiguity. Do not repeat
  `ForegroundBackgroundMixProbability=0.35/margin=0.08/min=0.06/max=0.88/softness=5`
  on the current keeper. If background recombination is revisited, it needs a
  more reliable object mask or an OOF/case-level target, not another global
  probability tweak. Cleanup deleted the rejected smoke train directory and
  launcher logs while preserving the selected12 XAI audit, keeper, and final
  artifacts. Manifest:
  `runs\cleanup_manifest_20260703_fgbgmix035_reject_smoke.json`, reclaimed
  about `181.34 MiB` (`190,152,137` bytes).

## Smoke 2026-07-03 - Crop-BBox Foreground/Background Recombination On Current Keeper

- Rationale: the pseudo-mask foreground/background mix was a fair background-bias
  test but could still keep or remove the wrong pixels. I revisited the same
  ForAug/background-bias idea with the actual `yolo_f` object position by adding
  a checkpoint-neutral `foreground_background_mix_mask_source` option. The new
  `crop_bbox` mode keeps the transformed YOLO object box as foreground and
  swaps only the outside background, preserving raw data and validation/test
  splits.
- Implementation/preflight: added `foreground_background_mix_mask_source` to
  the collator, `AugmentationConfig`, `train.py`, and the V8 launcher. Defaults
  remain `pseudo`, and missing bbox metadata falls back to the old pseudo mask.
  Regression tests cover both direct bbox masks and collate metadata routing.
  Focused preflight passed: `py_compile` for `dataset.py/train.py/config.py`
  and `tests\test_detection_calibration.py -k foreground_background_mix`
  (`4 passed`). Dry-run confirmed explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, keeper resume, full-val
  `MaxValBatches=0`, no stale hard manifest, bbox spatial fusion, crop-bbox
  token prior, boundary-band dropout, pairwise routing, teacher-focus-binary
  `0.015`, and `ForegroundBackgroundMixMaskSource=crop_bbox`.
- Valid smoke:
  `runs\smoke_v8_yolof_fgbgmix_cropbbox035_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260703`
  trained `24` batches for `1` epoch and evaluated full validation support
  `2606`. Validation macro/class-1 F1 was only `0.8827/0.6784`, class-1 P/R
  `0.6073/0.7682`, below the keeper `0.8847/0.6860` and below the `0.70`
  probe gate. Confusion was
  `[[487,49,2,0,11],[19,116,11,0,5],[6,10,514,8,6],[0,4,52,653,3],[8,12,4,1,625]]`.
  Active losses matched the intended recipe: CE `0.9078`, metric learning
  `2.1127`, foreground consistency `0.0594`, bbox dropout `1.2352/0.0043`,
  pairwise margin `0.5671`, teacher-focus-binary `0.5331`, and KD `0.0`.
- XAI selected12
  `runs\xai_smoke_v8_yolof_fgbgmix_cropbbox035_val_selected12_20260703`
  showed the same bottleneck. Top confusions stayed `3->2=52`, `0->1=49`,
  `1->0=19`, `4->1=12`, and `1->2=11`. Attention/Grad-CAM foreground mass
  stayed high (`0.9828/0.9759`), background blur/gray drops stayed near zero
  (`0.00023/-0.00028`), and object desaturation stayed much larger (`0.0881`).
  Review flags remained border/background-adjacent (`gradcam_border=7/12`,
  `rollout_border=5/12`, `gradcam_background=1/12`). Manual overlays showed:
  a `0->1` case still activated the fruit top plus a strong nearby background
  object, a `4->1` case mixed fruit wrinkles with plate/background hotspots,
  and a `3->2` case focused on lower-edge shadow rather than a better surface
  cue.
- Decision: reject before probe. Using `yolo_f` crop boxes made the
  augmentation more geometrically faithful, but it still did not reduce the
  class-1 false-positive/false-negative boundary and did not change the XAI
  failure mode. Do not repeat
  `ForegroundBackgroundMixProbability=0.35/MaskSource=crop_bbox/Margin=0.0/Min=0.05/Max=0.95/Softness=5`
  on the current keeper. Cleanup deleted the rejected smoke train directory and
  launch logs while preserving the selected12 XAI audit, keeper, and final
  artifacts. Manifest:
  `runs\cleanup_manifest_20260703_fgbgmix_cropbbox035_reject_smoke.json`,
  reclaimed about `181.32 MiB` (`190,129,094` bytes).

## Smoke 2026-07-03 - Light Background-Focus Suppression Retest

- Rationale: before leaving the background/context branch, I checked the latest
  foreground/background robustness work again. ForAug v4
  (`https://arxiv.org/html/2503.09399v4`) and AutoBackSwap
  (`https://arxiv.org/html/2606.32018`) support controlled background swapping
  when background is the shortcut, while the CVPR 2022 RIVAL10 foreground/
  background sensitivity study
  (`https://openaccess.thecvf.com/content/CVPR2022/papers/Moayeri_A_Comprehensive_Study_of_Image_Classification_Model_Sensitivity_to_Foregrounds_CVPR_2022_paper.pdf`)
  argues for measuring foreground/background sensitivity instead of trusting
  saliency alone. Prior TRKH audits already rejected a stronger background-focus
  suppression probe, so this was a final light retest, not a new background
  sweep.
- Preflight/config: `py_compile` passed for `trkh\training\train.py` and
  `trkh\core\config.py`; PowerShell parse check for the V8 launcher passed; and
  `tests\test_focused_false_positive_margin.py -k background_focus_suppression`
  passed (`1 passed, 7 deselected`). Dry-run confirmed explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, keeper resume, full-val
  `MaxValBatches=0`, empty `HardSampleManifest`, strict sample-index teacher
  CSV, bbox spatial fusion, crop-bbox token prior, boundary-band dropout,
  pairwise routing, `TeacherFocusBinaryLossWeight=0.015`, and
  `BackgroundFocusSuppressionLossWeight=0.006/Probability=0.50/NegativeClasses=0,4`.
- Valid smoke:
  `runs\smoke_v8_yolof_bgfocus006_p50_neg04_teacherfocusbinary015_boundarydrop_bboxprior_24b_1e_20260703`
  trained `24` batches for `1` epoch and evaluated full validation support
  `2606`. Validation macro/class-1 F1 was only `0.8816/0.6725`, class-1 P/R
  `0.6021/0.7616`, below the keeper `0.8847/0.6860` and below the `0.70`
  probe gate. Confusion was
  `[[486,50,2,0,11],[19,115,12,0,5],[5,10,515,8,6],[0,4,53,653,2],[8,12,4,1,625]]`.
  The suppression loss was active but weak: loss `0.0108`, fraction `0.2101`,
  mean focus probability `0.2000`, and mean background-counterfactual delta
  `-0.0007`. That negative near-zero delta means neutralizing background did
  not consistently reduce the class-1 probability on the selected negatives.
- XAI selected12
  `runs\xai_smoke_v8_yolof_bgfocus006_p50_neg04_val_selected12_20260703`
  confirmed the same failure mode. Top confusions were `3->2=53`, `0->1=50`,
  `1->0=19`, `1->2=12`, and `4->1=12`. Attention/Grad-CAM foreground mass
  stayed high (`0.9779/0.9788`), background blur/gray drops stayed near zero
  (`0.00046/-0.00043`), and object desaturation was much larger (`0.0730`).
  Review flags remained border-heavy (`gradcam_border=9/12`,
  `rollout_border=6/12`, `object_color_sensitive=3/12`). Manual overlays
  showed the familiar pattern: `0->1` focused on the fruit top plus a local
  bright background artifact, `3->2` focused on lower-edge shadow/crop border,
  and `4->1` focused mostly on fruit wrinkles with some plate/object hotspots.
- Decision: reject before probe. The light setting avoids a very aggressive
  penalty but still hurts class 1 and does not create a stable background
  counterfactual signal. Do not repeat
  `BackgroundFocusSuppressionLossWeight=0.006/Probability=0.50/FocusClass=1/NegativeClasses=0,4/Margin=0.015/MinProbability=0.05/Power=1.5`
  on the current keeper, and do not continue background-focus/source-context
  suppression without a new fold-safe reliability or case-level signal. Cleanup
  deleted the rejected smoke train directory and launcher logs while preserving
  the selected12 XAI audit, keeper, and final artifacts. Manifest:
  `runs\cleanup_manifest_20260703_bgfocus006_p50_reject_smoke.json`,
  reclaimed about `181.42 MiB` (`190,231,958` bytes).

## Smoke 2026-07-03 - Peer Small-Loss Co-Teaching Diagnostic

- Rationale: after the background/context branch failed repeatedly, I revisited
  noisy-label sample selection with a true peer update rather than the earlier
  single-network self-paced small-loss wrapper. I checked Co-teaching
  (`https://arxiv.org/abs/1804.06872`), DivideMix
  (`https://arxiv.org/abs/2002.07394`), Generalized Cross Entropy
  (`https://arxiv.org/abs/1805.07836`), and PRODEN partial-label learning
  (`https://proceedings.mlr.press/v119/lv20a/lv20a.pdf`). The intended
  distinction was that two peers update on samples selected by the other model,
  so this is a valid sample-selection diagnostic only if the peers disagree
  enough to avoid reproducing the rejected self-paced loss-rank signal.
- Implementation/preflight: added `trkh.tools.coteach_finetune`, a standalone
  train-only finetune utility that loads two TRKH checkpoints, uses strict
  `yolo_f` bbox metadata with crop-bbox token priors, applies peer-selected
  per-sample CE losses, evaluates both peers on full validation each epoch, and
  saves the better peer checkpoint. Added focused tests in
  `tests\test_coteach_finetune.py` for low-loss peer selection and cross-peer
  loss routing. Preflight passed: `py_compile` for the tool and tests, plus
  `tests\test_coteach_finetune.py` (`2 passed`).
- Valid smoke:
  `runs\smoke_coteach_yolof_peer_anchor_soup_r80_lr1e5_24b_1e_20260703`
  used peer A as the current no-pretrain keeper
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt`
  and peer B as
  `runs\soup_v8_yolof_top2_teacherfocus_distill_20260701\checkpoints\best.pt`.
  It trained `24` batches for `1` epoch with LR `1e-5`, remember-rate `0.80`,
  batch size `24`, and full validation support `2606`. Best peer was peer A at
  validation macro/class-1 F1 `0.8697/0.6407`, class-1 P/R
  `0.5847/0.7086`, far below the keeper `0.8847/0.6860` and below the `0.70`
  probe gate. Peer B was slightly worse at `0.8691/0.6369`. Best-peer
  confusion was
  `[[494,45,2,1,7],[27,107,12,0,5],[4,13,497,16,14],[0,4,44,656,8],[11,14,4,2,619]]`.
  Telemetry showed why this did not create a new signal: selected sample overlap
  was `0.9625`, and both peers selected the same class-1 fraction `0.6870`, so
  the method behaved close to a harsher loss-rank continuation rather than a
  disagreement-aware co-teaching signal.
- XAI selected12
  `runs\xai_smoke_coteach_yolof_peer_anchor_soup_r80_lr1e5_val_selected12_20260703`
  confirmed the same bottleneck. Top confusions were `0->1=45`, `3->2=44`,
  `1->0=27`, `2->4=15`, `2->1=14`, and `4->1=14`. Attention/Grad-CAM
  foreground mass stayed high (`0.9628/0.9677`), background blur/gray drops
  stayed near zero (`-0.00004/0.00034` original-pred drop), and object
  desaturation stayed much larger (`0.0778`). Review flags remained
  border/object-color driven (`gradcam_border=7/12`, `rollout_border=4/12`,
  `object_color_sensitive=4/12`). Manual overlays for `0->1`, `1->0`, and
  `4->1` showed the heatmaps mostly on fruit surface/boundary regions with
  occasional local background hotspots, not a broad-context cue that would
  justify another yolo-context route.
- Decision: reject before probe/full train. The peer-small-loss setting reduced
  class-1 F1 dramatically and did not produce useful peer disagreement. Do not
  repeat `coteach_finetune anchor+soup/remember_rate=0.80/LR=1e-5/24b/1e` on
  the current keeper, and do not revisit small-loss sample selection with two
  near-identical checkpoints. If sample selection is revisited, it needs an
  OOF/fold-safe reliability signal or deliberately diverse peers with measured
  disagreement before smoke. Cleanup copied `summary.json`, `history.csv`, and
  both peer validation metrics into the XAI folder under `source_run_metrics`,
  then deleted the rejected smoke train directory and smoke logs while
  preserving the XAI audit, keeper, and final artifacts. Manifest:
  `runs\cleanup_manifest_20260703_coteach_peer_r80_reject_smoke.json`,
  reclaimed about `35.45 MiB` (`37,168,671` bytes).

## Diagnostic 2026-07-03 - Frozen Pairwise Feature Verifier

- Rationale: the rejected co-teaching run showed that loss-rank selection with
  near-identical peers is not useful, but the old val-only margin guard had
  nearly reached class-1 `0.70`. I checked whether the current frozen TRKH
  head embeddings contain a train-label-fittable pairwise boundary signal that
  simple logits and surface statistics missed. This was motivated by
  fine-grained pair/center literature such as "Exploration of Class Center for
  Fine-Grained Visual Classification" (`https://arxiv.org/pdf/2407.04243`),
  recent analysis that linear classifier heads can retain recoverable spatial
  class evidence (`https://arxiv.org/html/2606.14555v1`), and partial-label /
  instance-dependent ambiguity work (`https://arxiv.org/html/2603.04825v1`).
  The test is diagnostic only: it fits logistic one-vs-one verifiers on train
  split frozen features and applies fixed thresholds to validation; test is not
  read.
- Implementation/preflight: added `trkh.tools.probe_pairwise_feature_verifier`
  and `tests\test_pairwise_feature_verifier.py`. The tool reuses the existing
  `probe_embedding_prototypes` dataset/embedding extraction path, builds frozen
  features from head embeddings plus probabilities/log-probabilities/margin,
  fits balanced logistic pair verifiers for `0-1,1-2,1-4,2-3`, reports OOF
  local pair accuracy, and writes base/verified metrics plus prediction-change
  CSVs. Focused preflight passed: `py_compile` for the tool/test and
  `tests\test_pairwise_feature_verifier.py` (`4 passed`).
- Diagnostic run:
  `runs\pairwise_feature_verifier_yolof_keeper_t060_m040_20260703` used the
  current keeper checkpoint, explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`,
  train+val only, threshold `0.60`, max pair margin `0.40`, min pair probability
  `0.02`, and batch size `96`. It extracted full train `9215` and full val
  `2606`. Pair OOF local accuracies were high (`0-1=0.9222`, `1-2=0.9817`,
  `1-4=0.9954`, `2-3=0.9806`), showing the pair boundaries are learnable in
  train features, but validation transfer was modest. Validation base metrics
  from this extraction were macro/class-1 `0.8841/0.6841`; verified metrics were
  `0.8855/0.6901`, class-1 P/R `0.6667/0.7152`. The verifier changed `65` val
  rows with `32` corrections, `27` harms, and `6` neutral changes. Verified
  confusion was
  `[[495,40,2,0,12],[18,108,18,0,7],[6,5,521,6,6],[0,3,58,648,3],[10,6,4,1,629]]`.
  It improved class-1 precision but lost too much recall and stayed below the
  `0.70` probe gate.
- Conservative retest:
  `runs\pairwise_feature_verifier_yolof_keeper_t065_m030_20260703` used
  threshold `0.65`, max pair margin `0.30`, and batch size `192`. Batch size
  `192` used about `3.35 GB` VRAM but did not materially improve throughput on
  Windows because CPU transform/dataloader remained the bottleneck. The result
  was validation macro/class-1 `0.8859/0.6899`, class-1 P/R
  `0.6606/0.7219`, with `55` changed rows, `27` corrections, `22` harms, and
  `6` neutral changes. It slightly improved macro versus `t060_m040` but did
  not improve class 1, so I stopped instead of sweeping thresholds on val.
- XAI changed-case audit:
  `runs\xai_pairwise_feature_verifier_yolof_keeper_t065_m030_changed12_20260703`
  audited 12 explicit changed cases from the conservative run. Because this is
  the same keeper checkpoint, XAI reflects the base model's evidence before the
  verifier override. The changed cases were very low-confidence near-ties:
  `near_tie_top2=5/12`, attention/Grad-CAM foreground mass `0.9356/0.9429`,
  background blur/gray drops still near zero (`0.0005/-0.0025` original-pred
  drop), and object desaturation still larger (`0.0255` original-pred drop,
  `0.0328` target-prob drop). Rollout was border-heavy (`10/12`). Manual
  overlays showed both a harm `0->1` and a correction `1<-0` attending to
  visually similar fruit-surface/boundary regions; the verifier is shifting a
  fuzzy boundary, not uncovering a new cue.
- Decision: reject as an integration path before probe/full train. This is the
  best legal no-pretrain diagnostic in this thread since the keeper, but it
  still misses the class-1 `0.70` gate and relies on a post-hoc verifier rather
  than a trainable architecture improvement. Keep
  `runs\pairwise_feature_verifier_yolof_keeper_t060_m040_20260703` as a
  near-gate diagnostic and the changed-case XAI audit. Do not repeat frozen
  pairwise verifier thresholds around `t=0.60-0.65/margin=0.30-0.40` on the
  current keeper. If revisited, it needs a new representation signal, OOF
  verifier confidence, or an in-model pairwise feature target that improves
  recall without reintroducing `0->1/4->1` false positives. Cleanup copied the
  conservative run summary/metrics/case CSV into the XAI folder, deleted the
  inferior conservative run and both verifier progress logs, and preserved the
  best verifier diagnostic plus keeper/final artifacts. Manifest:
  `runs\cleanup_manifest_20260703_pairwise_feature_verifier_variant_reject.json`,
  reclaimed about `2.09 MiB` (`2,195,459` bytes).

## Diagnostic 2026-07-03 - Patch-Evidence MIL Dense Readout

- Rationale: the frozen feature verifier showed the current pooled head
  embeddings are close to the class-1 gate but not enough. I checked whether
  final patch tokens contain recoverable local class evidence that the
  CLS/register pooled prediction suppresses. This follows the dense-readout /
  MIL interpretation from "Rethinking Global Average Pooling: Your Classifier
  Is Secretly a Multi-Instance Learner"
  (`https://arxiv.org/html/2606.14555v1`), where a linear classifier can be
  applied pointwise to a spatial feature grid. I also used the FGVC token
  evidence rationale from TransFG
  (`https://cdn.aaai.org/ojs/19967/19967-13-23980-1-2-20220628.pdf`) and
  FFVT (`https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/0685.pdf`):
  subtle local patches can matter more than a single global token.
- Implementation/preflight: added `trkh.tools.probe_patch_evidence_mil` and
  `tests\test_patch_evidence_mil.py`. The tool applies the current
  classification head to final patch tokens, summarizes patch mean/max/top-k
  and bbox-weighted patch evidence, appends the current pooled embedding and
  probability context, then fits train-only balanced logistic pair verifiers.
  It writes full train/val metrics and changed-case CSVs. It uses explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, `crop_bbox` token prior, and
  does not read test. Preflight passed: `py_compile` for tool/test and
  `tests\test_patch_evidence_mil.py` (`3 passed`).
- Rejected all-pairs diagnostic:
  `runs\patch_evidence_mil_yolof_keeper_t060_m040_top4_cropbbox_20260703`
  fit pairs `0-1,1-2,1-4,2-3`. It extracted full train `9215` and full val
  `2606`. Validation base was macro/class-1 `0.8841/0.6841`, while
  patch-verified all-pairs fell to `0.8827/0.6792`. It changed `72` val rows
  with `33` corrections, `32` harms, and `7` neutral changes. Direct dense
  patch aggregation was not usable: best direct method had class-1 F1 only
  about `0.295`, confirming that naive max/top-k over patch logits is too
  noisy. Cleanup copied the all-pairs summary/val metrics into the 0-1 XAI
  source folder, deleted the rejected run, and reclaimed `2.38 MiB` via
  `runs\cleanup_manifest_20260703_patch_evidence_allpairs_reject.json`.
- Useful 0-1-only diagnostic:
  `runs\patch_evidence_mil_yolof_keeper_01only_t060_m040_top4_cropbbox_20260703`
  kept only pair `0-1` with the same threshold `0.60`, max pair margin `0.40`,
  min pair probability `0.02`, top-k `4`, and logistic `C=1.0`. Pair OOF local
  accuracy was `0.9243`. Train improved from macro/class-1
  `0.9399/0.8157` to `0.9478/0.8438` with `46` changes (`42` corrections,
  `1` harm, `3` neutral). Validation improved from `0.8841/0.6841` to
  `0.8892/0.7073`, class-1 P/R `0.6554/0.7682`, changing `31` rows with
  `16` corrections, `9` harms, and `6` neutral. Confusion became
  `[[495,41,1,1,11],[19,116,11,0,5],[6,9,514,9,6],[0,4,52,653,3],[13,7,4,1,625]]`.
  This is the first legal no-test diagnostic in this cycle to cross the
  class-1 `0.70` validation gate.
- Important diagnostic insight: target local evidence is usually present even
  when the image-level prediction is wrong. For validation, target class was
  recovered by at least one valid patch in `0.9873` of samples and by at least
  one bbox patch in `0.9866`; among `210` base-wrong val rows, `200` still had
  target evidence in both valid and bbox patch sets. Therefore the remaining
  problem is not simply missing visual cue; it is selecting/reweighting the
  right local evidence without letting noisy class-1-like patches widen
  `0->1` false positives.
- XAI changed-case audit:
  `runs\xai_patch_evidence_mil_yolof_keeper_01only_changed16_20260703` used a
  harm-first/correction CSV from the 0-1 diagnostic. It audited `16` changed
  cases with `method=all`, `stem_last`, robustness probes, explicit `yolo_f`,
  and `crop_bbox`. Foreground mass stayed high (`attention/grad-rollout/
  Grad-CAM/rollout` = `0.9510/0.9812/0.9934/0.9546`), while background
  blur/gray drops were still near zero (`0.0003/-0.0015`). Object desaturation
  remained more relevant (`0.0162` original-pred drop, `0.0380` target drop),
  and changed cases were mostly near-ties (`10/16`) with rollout border flags
  (`13/16`). Manual overlays showed that both corrections and harms attend to
  very similar fruit surface/boundary cues. One class-1 harm still had local
  leaf/branch/background hotspots, so a raw post-hoc verifier is not safe as a
  final model.
- Decision: keep the 0-1 patch-evidence MIL diagnostic and its XAI audit as a
  new useful signal, but do not treat the post-hoc verifier as a final model.
  Do not repeat all-pairs patch-evidence routing because `2-3` harms erase the
  class-1 gain. The next implementation should compress this into a trainable,
  checkpoint-safe in-model 0-1 patch-evidence/MIL auxiliary route with
  reliability gating, or use it as an OOF target; it must preserve class-1
  recall without reintroducing `0->1` harms.

## Smoke 2026-07-03 - Train-Time Patch-Evidence MIL Loss

- Rationale: the 0-1 patch-evidence diagnostic crossed the validation class-1
  gate as a post-hoc verifier, so I tried a checkpoint-safe train-time version
  that uses the same local evidence without adding a new inference dependency.
  The loss applies the existing classifier head pointwise to final patch tokens,
  takes top-k bbox-prior patch margins for pair `0-1`, and applies BCE only to
  true class `0/1` samples. This tests whether local target evidence can be
  absorbed into the model weights rather than kept as a post-hoc router.
- Implementation/preflight: added patch-evidence MIL loss fields in
  `trkh.core.config.TrainConfig`, CLI flags in `trkh.training.train`, the
  helper `_patch_evidence_mil_loss_from_features`, history/stat telemetry, and
  a resume override that allows short-run patch-evidence options to be supplied
  when resuming the keeper config. Added
  `tests\test_patch_evidence_mil_loss.py`. Focused preflight passed:
  `py_compile` for `trkh\core\config.py`, `trkh\training\train.py`,
  `tests\test_patch_evidence_mil_loss.py`; pytest for patch-evidence,
  verifier, and co-teaching tests passed (`11 passed`). I also fixed
  `trkh.core.utils.plot_all_training_metrics` to split very wide histories
  into `all_training_metrics_part*.png` chunks; the old single huge figure
  crashed after this smoke despite the train/val loop completing.
- Smoke run:
  `runs\smoke_patch_evidence_mil_yolof_keeper01_w004_top4_24b_1e_20260703`
  resumed the current keeper, used explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, full validation support
  `2606`, `batch_size=24`, `grad_accum=2`, `LR=1e-5`, `24` train batches,
  and `patch_evidence_mil_loss_weight=0.004`, pair `0-1`, top-k `4`,
  bbox threshold `0.05`, positive weight `1.0`, start epoch `1`.
  The loss was active in history: loss `0.6614`, selected fraction `0.3993`,
  positive score `0.5090`, negative score `0.2556`, margin `0.2534`, bbox
  fraction `0.4778`.
- Metrics: validation macro/class-1 F1 reached only `0.8835/0.6784`, below the
  keeper `0.8847/0.6860`. Class-1 P/R was `0.6073/0.7682`, weighted F1
  `0.9220`, accuracy `0.9198`. Confusion matrix:
  `[[488,49,2,0,10],[19,116,11,0,5],[5,10,515,8,6],[0,4,52,653,3],[8,12,4,1,625]]`.
  It did not preserve the diagnostic class-1 gain and still kept the dominant
  `0->1`, `1->0`, `1->2`, `4->1`, and `3->2` failure modes.
- XAI audit:
  `runs\xai_smoke_patch_evidence_mil_yolof_keeper01_w004_top4_selected12_20260703`
  audited 12 selected validation cases with `method=all`, robustness probes,
  explicit `yolo_f`, `stem_last`, and `crop_bbox` token prior. Top confusions
  were `3->2=52`, `0->1=50`, `1->0=19`, `4->1=12`, `1->2=11`.
  Review flags included attention background `3`, attention border `12`,
  Grad-CAM background `1`, Grad-CAM border `7`, near-tie top2 `1`,
  object-color-sensitive `3`, rollout border `4`. Foreground mass stayed high
  (`attention=0.9139`, `grad_rollout=0.9733`, `gradcam=0.9469`,
  `rollout=0.9529`), border mass remained material
  (`attention=0.3077`, `gradcam=0.2244`), background blur/gray drops were near
  zero, and object desaturation was still the larger perturbation signal.
  Manual overlays for representative `0->1`, `1->0`, and `4->1` cases showed
  fruit surface/boundary focus with occasional table/background-adjacent
  artifacts, not a clean new discriminative cue.
- Decision: reject this train-time MIL loss setting before probe/full train.
  The post-hoc 0-1 patch-evidence diagnostic remains useful, but plain top-k
  BCE did not compress it into the keeper. Do not repeat
  `patch_evidence_mil_loss_weight=0.004/pair=0-1/top_k=4/
  bbox_threshold=0.05/positive_weight=1.0/LR=1e-5/24b/1e` on the current
  keeper. If the patch signal is revisited, use a reliability-gated, OOF, or
  learned selector/router target that can decide when local class-1 evidence is
  trustworthy, not a raw top-k local-margin BCE. Cleanup copied smoke metrics
  into the XAI folder and deleted the rejected train directory; manifest
  `runs\cleanup_manifest_20260703_patch_evidence_mil_w004_reject_smoke.json`
  reclaimed `185,031,822` bytes.

## Smoke 2026-07-03 - Gated Patch-Evidence Router Head

- Rationale: after the post-hoc 0-1 patch-evidence verifier crossed the
  validation class-1 gate, I tested whether a checkpoint-safe learned selector
  could absorb that signal at inference without a separate verifier. I used
  primary MIL references for the design: gated attention MIL (Ilse et al.),
  CLAM-style attention and instance-level reliability, and TransMIL-style
  bag/patch aggregation. The implementation adds a zero-initialized
  `PatchEvidenceRouterHead` over patch tokens, context, pair patch margins, and
  crop-bbox prior, with conservative routing only on exact top-2 pair `0-1`.
- Implementation/preflight: added `PatchEvidenceRouterHead` in
  `trkh.models.model`, model/train config and CLI flags, resume-safe extension
  loading for `patch_evidence_router_head.*`, router-loss telemetry, and
  `tests\test_patch_evidence_router.py`. Focused preflight passed:
  `py_compile` for model/config/train/test and pytest for router,
  patch-evidence MIL loss, and part-token learner (`13 passed`).
- Smoke run:
  `runs\smoke_patch_router_yolof_keeper01_w020_top4_routeronly_24b_1e_20260703`
  resumed the current keeper, froze all non-router parameters, used explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, full validation support
  `2606`, `batch_size=24`, `grad_accum=2`, `24` train batches, LR `5e-4`,
  router pair `0-1`, hidden dim `128`, top-k `4`, bbox weight `0.75`, dropout
  `0.05`, logit scale `0.12`, route margin `0.25`, route min pair probability
  `0.02`, and router loss weight `0.020`.
- Metrics: validation macro/class-1 F1 was exactly the keeper level
  `0.8847/0.6860`, with class-1 P/R `0.6114/0.7815`, weighted F1 `0.9220`,
  accuracy `0.9198`, and confusion
  `[[487,49,1,1,11],[17,118,11,0,5],[5,10,514,9,6],[0,4,52,653,3],[8,12,4,1,625]]`.
  The router loss was active but weak: selected fraction `0.3993`, positive
  logit `-0.0270`, negative logit `-0.0328`, logit margin only `0.0057`, and
  probability about `0.4925`. The learned route did not produce a measurable
  decision change.
- XAI audit:
  `runs\xai_smoke_patch_router_yolof_keeper01_w020_top4_selected12_20260703`
  audited 12 selected validation cases with `method=all`, robustness probes,
  explicit `yolo_f`, `stem_last`, and `crop_bbox` prior. Top confusions stayed
  `3->2=52`, `0->1=50`, `1->0=18`, `4->1=13`, `1->2=11`. Foreground mass was
  still high (`attention/grad_rollout/gradcam/rollout` =
  `0.9140/0.9734/0.9480/0.9530`), background blur/gray drops stayed near zero
  (`0.0005/-0.0005` original-pred drop), and object desaturation remained the
  larger perturbation signal (`0.0665` original-pred drop). Review flags still
  included attention border `11/12`, Grad-CAM border `6/12`, rollout border
  `5/12`, and object-color-sensitive `3/12`.
- Manual XAI read: representative `0->1` errors still put Grad-CAM on the
  bright fruit edge/shadow band instead of the stable interior surface. A
  `1->0` case attended to interior peel but remained ambiguous in green/chua
  level. A `1->2` case was dominated by stem/table/edge lines and a partial
  surface region. This confirms the failure is not lack of wide background
  context; the model needs a more reliable interior-surface evidence signal and
  stronger control of boundary/lighting shortcuts.
- Decision: reject the router-only smoke before probe/full train. Do not repeat
  `PatchEvidenceRouterHead=true/pair=0-1/hidden=128/top_k=4/bbox_weight=0.75/
  logit_scale=0.12/loss_weight=0.020/router-only/LR=5e-4/24b/1e` on the
  current keeper. If revisited, the router must be trained with a stronger
  fold-safe reliability target or coupled representation update; a zero-init
  residual router alone does not move the boundary. Cleanup copied metrics into
  the XAI folder and deleted the rejected train directory; manifest
  `runs\cleanup_manifest_20260703_patch_router_w020_reject_smoke.json`
  reclaimed `131,759,604` bytes.

## Smoke 2026-07-03 - OOF Patch-Verifier Teacher for Pairwise Head

- Rationale: the only legal diagnostic to cross the class-1 validation gate was
  the 0-1 patch-evidence verifier, but the previous train-time top-k BCE and
  zero-init router did not compress it into the model. I therefore tried the
  leak-safe stacking direction: generate out-of-fold (OOF) train probabilities
  from the patch verifier, then distill those probabilities into the existing
  pairwise margin head. This follows the stacked generalization idea from
  Wolpert (`https://doi.org/10.1016/S0893-6080(05)80023-1`) and the confidence
  input finding from Ting & Witten's "Issues in Stacked Generalization"
  (`https://jair.org/index.php/jair/article/view/10228`), while keeping val/test
  out of the target generation.
- Implementation/preflight: extended `trkh.tools.probe_patch_evidence_mil` to
  write per-row verifier pair probabilities and
  `train\pair_verifier_teacher_oof.csv` with `sample_index`, `path`, and
  `prob_0..prob_4`. For pair `0-1`, train rows use 5-fold OOF probabilities;
  non-pair rows use hard-label teacher probabilities with tiny smoothing. Added
  a regression test for the writer. Also updated resume override in
  `trkh.training.train` so `--distillation-teacher-csv` and
  `--teacher-pairwise-margin-loss-weight` are not overwritten by the keeper
  checkpoint's old teacher config. Preflight passed:
  `py_compile` for probe/train/test and focused pytest (`14 passed`).
- OOF teacher artifact:
  `runs\patch_evidence_mil_yolof_keeper_01only_t060_m040_top4_cropbbox_oofteacher_20260703`
  reproduced the 0-1 verifier val result: base macro/class-1
  `0.8841/0.6841` -> patch-verified `0.8892/0.7073`. The generated teacher CSV
  covers all `9215` train samples, with `2482` pair rows, OOF local accuracy
  `0.9243`, teacher-label agreement `0.9796`, and mean confidence `0.9893`.
  Pair OOF confusion was `0->1=139`, `1->0=49`, so the target is mostly clean
  but close to hard labels.
- Smoke run:
  `runs\smoke_oof_patch_teacher_pairwise_yolof_keeper01_w020_pairhead_24b_1e_20260703`
  resumed the current keeper, used explicit `yolo_f`, full validation support
  `2606`, froze all parameters except `pairwise_margin_norm` and
  `pairwise_margin_head` (`1540` trainable params), disabled inherited
  teacher-focus binary loss, used the OOF teacher CSV, and trained `24` batches
  with LR `5e-4`, `teacher_pairwise_margin_loss_weight=0.020`,
  mass threshold `0.50`, error power `0.50`, and hard-target blend `0.0`.
- Metrics: validation macro/class-1 F1 fell to `0.8834/0.6822`, below the
  keeper `0.8847/0.6860`. Class-1 P/R was `0.6094/0.7748`; confusion matrix:
  `[[487,49,1,1,11],[18,117,11,0,5],[5,10,513,9,7],[0,4,52,653,3],[8,12,4,1,625]]`.
  The loss was active (`teacher_pairwise_margin_loss=0.5531`, valid fraction
  `0.5273`), but the student pairwise probability averaged `0.4740` against a
  teacher target average `0.3660`, and the small update reduced class-1 recall
  by one sample instead of improving the boundary.
- XAI audit:
  `runs\xai_smoke_oof_patch_teacher_pairwise_yolof_keeper01_w020_selected12_20260703`
  used full val, `method=all`, robustness probes, `stem_last`, explicit
  `yolo_f`, and `crop_bbox`. Top confusions stayed `3->2=52`, `0->1=49`,
  `1->0=19`, `4->1=13`, `1->2=11`. Foreground mass remained high
  (`attention/grad_rollout/gradcam/rollout` =
  `0.9143/0.9737/0.9477/0.9532`), background blur/gray drops were near zero
  (`0.0006/-0.0004` original-pred drop), object desaturation was larger
  (`0.0662` original-pred drop), and attention border was `12/12`. Manual
  overlay of the representative `0->1` case was still the same bright
  edge/shadow band.
- Decision: reject this OOF-teacher pairwise compression before probe/full
  train. Keep the OOF teacher artifact as a small reproducible diagnostic, but
  do not repeat `teacher_pairwise_margin_loss_weight=0.020/mass=0.50/
  error_power=0.50/hard_blend=0.0/trainable=pairwise_margin_norm+head/LR=5e-4/
  24b/1e` on the current keeper. The OOF verifier is useful for post-hoc
  routing, but its probabilities are too close to hard labels to create a new
  differentiable representation signal in the existing pairwise head. Cleanup
  copied metrics into the XAI folder and deleted the rejected train directory;
  manifest `runs\cleanup_manifest_20260703_oof_patch_teacher_pairwise_reject_smoke.json`
  reclaimed `127,712,535` bytes.

## Probe 2026-07-03 - Auxiliary class_f Patch-Evidence Verifier

- Rationale: plain mixed `yolo_f + class_f` training already failed, but the
  post-hoc 0-1 patch-evidence verifier remained the only legal class-1
  validation gate crossing. I tested a narrower data-combination route: keep
  validation on `yolo_f`, add `class_f/train` only to the train-side logistic
  patch verifier, down-weight auxiliary rows, and append a source-domain feature
  so crop-domain evidence cannot silently dominate the context-domain features.
  This follows the multi-view/local-token motivation from FFVT
  (`https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/0685.pdf`) and
  leak-safe stacking/OOS-probability practice from Wolpert/Ting-Witten
  (`https://doi.org/10.1016/S0893-6080(05)80023-1`,
  `https://jair.org/index.php/jair/article/view/10228`).
- Implementation/preflight: extended `trkh.tools.probe_patch_evidence_mil` with
  `--auxiliary-train-data`, `--auxiliary-train-weight`, and
  `--source-domain-feature`; extended `_fit_pair_models` to accept logistic
  sample weights. Existing no-aux behavior is unchanged. Focused preflight
  passed: `py_compile` plus
  `pytest tests\test_pairwise_feature_verifier.py tests\test_patch_evidence_mil.py -q`
  (`10 passed`).
- Probe:
  `runs\patch_evidence_mil_yolof_keeper_01only_auxclassf_w025_domain_t060_m040_top4_cropbbox_20260703`
  used primary `yolo_f` train/val, auxiliary `class_f/train`, pair `0-1`,
  `top_k=4`, `crop_bbox`, threshold/margin `0.60/0.40`, `C=1.0`, 5-fold OOF,
  auxiliary weight `0.25`, and source-domain feature enabled. The combined
  pair verifier saw `4964` train pair rows with class counts `3882/1082`,
  OOF local accuracy `0.9363`, and sample-weight mean `0.625`.
- Metrics: validation base macro/class-1 was `0.8841/0.6841`; auxiliary
  patch-verified reached `0.8883/0.7030`, class-1 P/R `0.6480/0.7682`, with
  `31` changes (`16` corrections, `10` harms, `5` neutral). This still crosses
  the `0.70` class-1 gate, but it is worse than the yolo-only verifier
  `0.8892/0.7073` and adds one more harm. Confusion after verification was
  `[[494,42,1,1,11],[19,116,11,0,5],[6,9,514,9,6],[0,4,52,653,3],[12,8,4,1,625]]`.
- XAI audit:
  `runs\xai_patch_evidence_mil_yolof_keeper_01only_auxclassf_w025_domain_changed16_20260703`
  audited 16 harm-first changed cases with `method=all`, robustness probes,
  `stem_last`, explicit `yolo_f`, and `crop_bbox`. Foreground mass was high
  (`attention/grad-rollout/Grad-CAM/rollout` =
  `0.9522/0.9802/0.9928/0.9542`), while background blur/gray drops stayed near
  zero (`0.0003/-0.0007`). Review flags still showed border-heavy routing
  (`attention_border=9/16`, `rollout_border=14/16`) and near ties (`9/16`).
  Manual overlays showed the same surface/edge and leaf/background-adjacent
  hotspots as the yolo-only verifier; a class-1 harm still highlighted leaf and
  background clutter.
- Decision: reject this auxiliary recipe before any in-model compression or
  test audit. `class_f` auxiliary patch evidence raised train OOF local accuracy
  but did not improve `yolo_f` validation beyond the yolo-only verifier, so it
  is likely domain mismatch rather than a new reliable cue. Do not repeat
  `auxiliary_train_data=class_f/weight=0.25/source_domain_feature=true/pair=0-1/
  t=0.60/margin=0.40/top_k=4/crop_bbox` on the current keeper. Cleanup copied
  source metrics into the XAI folder and deleted the weaker probe directory;
  manifest
  `runs\cleanup_manifest_20260703_auxclassf_patch_evidence_reject_probe.json`
  reclaimed `2,837,591` bytes.

## Probe 2026-07-03 - Spatial Interior/Border Patch-Evidence Features

- Rationale: the previous patch-verifier XAI repeatedly showed border/ring and
  leaf/background-adjacent harms. Rather than add more data or train a new
  model, I tested whether the verifier can learn reliability if its features
  explicitly separate object-interior patch evidence from border/ring evidence.
  The design keeps the default probe path unchanged and only activates the new
  feature block under `--spatial-evidence-features`.
- Implementation/preflight: added optional interior/border masks to
  `summarize_patch_evidence`, using square token grids, crop-bbox prior, and a
  one-token erosion (`--spatial-interior-erode 1`). The feature set appends
  interior and border mean/max/top-k/top1 and pair-margin summaries. Added a
  3x3 regression test proving that a border class cue and an interior class cue
  are split correctly. Preflight passed: `py_compile` plus focused pytest
  (`11 passed`).
- Probe:
  `runs\patch_evidence_mil_yolof_keeper_01only_spatial_t060_m040_top4_cropbbox_erode1_20260703`
  used primary `yolo_f` train/val only, pair `0-1`, `top_k=4`, `crop_bbox`,
  `spatial_evidence_features=true`, interior erode `1`, threshold/margin
  `0.60/0.40`, `C=1.0`, and 5-fold OOF. The verifier feature dimension grew
  from `318` to `388`, but 0-1 OOF local accuracy dropped slightly to `0.9226`
  from the yolo-only verifier's `0.9243`.
- Metrics: validation base macro/class-1 was `0.8841/0.6841`; spatial
  patch-verified reached `0.8883/0.7030`, class-1 P/R `0.6480/0.7682`, with
  `33` changes (`17` corrections, `11` harms, `5` neutral). This is still above
  the class-1 `0.70` diagnostic gate, but it is worse than the original
  yolo-only patch verifier `0.8892/0.7073` and increases harm count. Confusion
  after verification was
  `[[494,42,1,1,11],[19,116,11,0,5],[6,9,514,9,6],[0,4,52,653,3],[12,8,4,1,625]]`.
- XAI audit:
  `runs\xai_patch_evidence_mil_yolof_keeper_01only_spatial_erode1_changed16_20260703`
  audited 16 harm-first changed cases. Foreground mass stayed high
  (`attention/grad-rollout/Grad-CAM/rollout` =
  `0.9544/0.9811/0.9293/0.9557`), but attention and rollout remained
  border-heavy (`attention_border=11/16`, `rollout_border=14/16`), background
  perturbations stayed near zero, and object desaturation was still the larger
  perturbation. A new harm (`Image_6003`) showed attention dominated by
  leaf/background/edge structure while Grad-CAM was nearly blank over the object,
  confirming that the added spatial features did not suppress the shortcut.
- Decision: reject spatial interior/border features for this verifier before
  any in-model compression. Do not repeat
  `spatial_evidence_features=true/spatial_interior_erode=1/pair=0-1/t=0.60/
  margin=0.40/top_k=4/crop_bbox` on the current keeper. If spatial reliability
  is revisited, it needs a stronger supervised reliability target or a different
  representation, not more erode/ring features around the same frozen logits.
  Cleanup copied source metrics into the XAI folder and deleted the weaker
  probe directory; manifest
  `runs\cleanup_manifest_20260703_spatial_patch_evidence_reject_probe.json`
  reclaimed `4,717,951` bytes.

## Smoke 2026-07-03 - OOF Patch-Verifier Disagreement Sample Deweighting

- Rationale: the train-only OOF 0-1 patch verifier is still the strongest
  legal diagnostic signal, but direct distillation into the pairwise head failed.
  I tested a narrower route: use the OOF verifier only as a fold-safe sample
  reliability signal, deweighting train samples where the verifier confidently
  predicts the opposite 0/1 class. This does not change labels or raw data and
  keeps validation/test untouched.
- Implementation/preflight: added `trkh.tools.build_patch_oof_sample_weights`
  to build a `sample_index` keyed train-only manifest from
  `train\pair_verifier_teacher_oof.csv`, plus tests for pair filtering,
  disagreement thresholding, and output summaries. Patched resume overrides so
  `--sample-weight-manifest` is honored when resuming the keeper. Preflight
  passed: `py_compile` for the new tool and train path plus focused pytest
  (`2 passed`).
- Manifest:
  `runs\patch_oof_sample_weights_yolof_keeper01_t065_w035_20260703\patch_oof_sample_weights_train_only.csv`
  covered all `9215` train rows, selected `151` confident OOF disagreements
  from the 0-1 pair (`117` label `0` predicted `1`, `34` label `1` predicted
  `0`), and assigned selected rows weight `0.35`.
- Smoke:
  `runs\smoke_patchoof_sampleweight_yolof_keeper01_t065_w035_24b_1e_20260703`
  resumed the current keeper, used explicit `yolo_f`, full validation support
  `2606`, `24` train batches for `1` epoch, LR `1e-5`, no inherited
  teacher-focus binary loss, and the new sample-weight manifest. Loader audit
  matched all `151` sample-index rows; observed train weight mean was `0.98935`.
- Metrics: validation macro/class-1 F1 fell to `0.8823/0.6744`, below the
  no-pretrain keeper `0.8847/0.6860`. Class-1 P/R was `0.6010/0.7682`.
  Confusion matrix:
  `[[486,51,2,0,10],[19,116,11,0,5],[5,10,515,8,6],[0,4,52,653,3],[8,12,4,1,625]]`.
  Deweighting reduced no boundary error family; it kept `0->1=51`,
  `1->0=19`, `1->2=11`, `4->1=12`, and `3->2=52`.
- XAI audit:
  `runs\xai_smoke_patchoof_sampleweight_yolof_keeper01_t065_w035_selected12_20260703`
  used full val, `method=all`, robustness probes, `stem_last`, explicit
  `yolo_f`, and `crop_bbox`. Top confusions stayed `3->2=52`, `0->1=49`,
  `1->0=19`, `4->1=12`, `1->2=11`. Foreground mass stayed high
  (`attention/grad-rollout/Grad-CAM/rollout` =
  `0.9454/0.9781/0.9683/0.9636`), background blur/gray drops were near zero
  (`0.0001/-0.0008` original-pred drop), object desaturation was larger
  (`0.0345` original-pred drop), and review flags remained border-heavy
  (`attention_border=11/12`, `Grad-CAM_border=6/12`, `rollout_border=5/12`).
  Manual overlays again showed fruit surface/edge cues and local
  table/background hotspots, not a missing wide-context signal.
- Decision: reject this sample-deweighting policy before probe/full train. The
  OOF patch verifier remains useful as a post-hoc diagnostic, but its confident
  disagreement rows are not a safe train-loss reliability signal on the current
  representation; they remove support from ambiguous class-1 boundary examples
  without adding a new interior-surface cue. Do not repeat
  `build_patch_oof_sample_weights pair=0-1/threshold=0.65/weight=0.35` followed
  by `sample_weight_factor=1.0/sample_weight_max=1.0/LR=1e-5/24b/1e` on the
  current keeper. Cleanup copied metrics/config/history into the XAI folder and
  deleted the rejected train directory; manifest
  `runs\cleanup_manifest_20260703_patchoof_sampleweight_reject_smoke.json`
  reclaimed `185,597,270` bytes.

## Final-Test Diagnostic 2026-07-03 - 0-1 Patch-Evidence Verifier

- Rationale: the yolo-only 0-1 patch-evidence verifier was the first legal
  validation diagnostic to cross class-1 F1 `0.70` (`0.8892/0.7073`) and later
  in-model compression attempts failed. I ran one fixed-configuration test
  audit using the already selected validation setting (`threshold=0.60`,
  `margin=0.40`, `top_k=4`, `crop_bbox`, `C=1.0`) to measure generalization.
  No test threshold sweep was performed.
- Artifact:
  `runs\patch_evidence_mil_yolof_keeper_01only_t060_m040_top4_cropbbox_test_final_20260703`
  fit the verifier on train only (`9215` rows, 0-1 pair rows `2482`, OOF local
  accuracy `0.9243`) and evaluated test (`1267` rows). Metrics and changed-case
  predictions were copied into
  `runs\xai_patch_evidence_mil_yolof_keeper_01only_test_changed11_20260703\source_patch_evidence_test_metrics`.
- Test metrics: base TRKH was macro/class-1 F1 `0.8865/0.6667`; patch-verified
  reached `0.8905/0.6795`, with class-1 P/R changing from
  `0.5978/0.7534` to `0.6386/0.7260`. It changed `11` predictions
  (`7` corrections, `3` harms, `1` neutral). Main effect was reducing class-1
  false positives: `0->1` dropped `22 -> 16` and `4->1` dropped `3 -> 2`,
  but class-1 false negatives increased (`1->0` `7 -> 9`), so class-1 remains
  below the AIDT disagreement-gate test result (`0.6857`) and far below the
  target.
- XAI audit:
  `runs\xai_patch_evidence_mil_yolof_keeper_01only_test_changed11_20260703`
  audited all `11` changed test cases with `method=all`, robustness probes,
  `stem_last`, explicit `yolo_f`, and `crop_bbox`. Foreground mass stayed high
  (`attention/grad-rollout/Grad-CAM/rollout` =
  `0.9663/0.9818/0.9039/0.9738`), background blur/gray drops stayed near zero
  (`0.0001/0.0011` original-pred drop), and object desaturation affected the
  target probability more (`0.0489`). Review flags were still border/near-tie
  heavy (`attention_border=9/11`, `rollout_border=8/11`, `near_tie=5/11`).
  Manual overlays showed the same trade-off: corrections remove green/edge
  false positives, while harms are real class-1 surface-defect cases that the
  verifier routes back to class 0.
- Decision: keep this as a small post-hoc diagnostic artifact and current best
  no-pretrain post-hoc test macro result, but do not treat it as the final
  solution. It improves test macro and class-1 slightly, yet does not beat the
  AIDT disagreement gate on class-1 and demonstrates that the remaining problem
  is reliability of local surface/boundary evidence. Do not tune
  `threshold/margin/top_k` on test; any next patch-evidence work must add a
  train/validation-only reliability target that preserves class-1 recall while
  removing `0/4->1` false positives.

## Probe 2026-07-03 - Directional Selective Gate for Patch Verifier

- Rationale: the fixed patch verifier improves validation and test by reducing
  false positives into class 1, but it also harms true class-1 examples. I
  checked selective-classification/risk-control framing
  (`https://proceedings.neurips.cc/paper_files/paper/7073-selective-classification-for-deep-neural-networks.pdf`)
  and kept the stacking discipline of selecting thresholds from out-of-fold
  train predictions only. The goal was a direction-specific reliability gate:
  choose thresholds separately for `1->0` and `0->1`, then evaluate val without
  fitting anything on val/test.
- Implementation/preflight: added
  `trkh.tools.apply_patch_verifier_directional_gate`, which reads the existing
  patch-verifier `predictions.csv`, selects direction thresholds from train OOF
  by minimum correction precision and net corrections, writes gated metrics and
  changed cases, and records the leakage guard. Added
  `tests\test_patch_verifier_directional_gate.py`. Preflight passed:
  `py_compile` plus focused pytest (`1 passed`).
- Probe:
  `runs\patch_verifier_directional_gate_yolof_keeper01_oofrisk060_20260703`
  used train OOF predictions from
  `runs\patch_evidence_mil_yolof_keeper_01only_t060_m040_top4_cropbbox_fullprobs_20260703`,
  pair `0-1`, pair filters `min_pair_probability=0.02` and `max_pair_margin=0.40`,
  and risk target `min_correction_precision=0.60`. Train OOF showed `0->1`
  was unsafe (`18` candidates, `0` corrections, `17` harms), so that direction
  was disabled. The selected `1->0` threshold was `0.7273`, with train OOF
  `40` selected (`25` corrections, `13` harms, `2` neutral).
- Metrics: validation base was macro/class-1 `0.8841/0.6841`; directional
  gated reached `0.8883/0.7015`, class-1 P/R `0.6552/0.7550`, with `20`
  changes (`11` corrections, `4` harms, `5` neutral). Confusion became
  `[[497,39,1,1,11],[21,114,11,0,5],[6,9,514,9,6],[0,4,52,653,3],[12,8,4,1,625]]`.
  This is more leakage-disciplined than the default verifier and crosses the
  class-1 diagnostic gate, but it is still below the yolo-only fixed verifier
  `0.8892/0.7073`.
- XAI audit:
  `runs\xai_patch_verifier_directional_gate_yolof_keeper01_oofrisk060_changed20_20260703`
  audited all 20 changed val cases. Foreground mass stayed high
  (`attention/grad-rollout/Grad-CAM/rollout` =
  `0.9603/0.9778/0.9892/0.9621`), background blur/gray drops stayed near zero
  (`0.0007/-0.0000` original-pred drop), target probability was much more
  sensitive to object desaturation (`0.0460`), and review flags remained
  near-tie/border heavy (`near_tie=12/20`, `attention_border=12/20`,
  `rollout_border=16/20`). The gate is correctly suppressing the toxic
  `0->1` direction, but it still cannot distinguish class-1 true defects from
  class-0 green/edge artifacts reliably.
- Decision: keep this directional gate as a reproducible train-OOF reliability
  diagnostic and as evidence that `0->1` patch-verifier routing should be
  disabled unless a new target changes it. Do not run a test audit from this
  val result, because it does not beat the fixed verifier on validation. Future
  in-model work should encode the direction-specific finding: train only a
  conservative `1->0` false-positive suppressor, and add a separate mechanism
  to protect true class-1 recall before touching test again.

## Smoke 2026-07-03 - Patch-Directional Targeted Margin

- Rationale: after the directional gate showed `0->1` routing is toxic and
  `1->0` is only partly reliable, I tested whether that train-OOF signal can be
  compressed into the current TRKH checkpoint as a narrow hard-negative margin.
  This follows the selective train-OOF discipline from the previous gate and is
  related to hard-negative / negative-learning ideas
  (`https://openaccess.thecvf.com/content_ICCV_2019/papers/Kim_NLNL_Negative_Learning_for_Noisy_Labels_ICCV_2019_paper.pdf`)
  without modifying raw data or using val/test to select rows.
- Implementation/preflight: added
  `trkh.tools.build_patch_directional_margin_manifest` and
  `tests\test_build_patch_directional_margin_manifest.py`. The tool reads only
  train `changed_cases_for_xai.csv` rows from the directional gate, requires
  `split=train` and `\train\` paths, writes a `sample_index` keyed
  targeted-margin manifest, and records suppressor/protector counts. Preflight
  passed: `py_compile` plus focused pytest with the existing targeted-margin
  sample-index test (`2 passed`).
- Manifest:
  `runs\patch_directional_targeted_margin_yolof_keeper01_t0727_w15p08_20260703`
  emitted `40` train rows: `27` false-positive suppressors
  (`25` target `0 > 1`, `2` target `2 > 1`) with margin/weight `0.06/1.5`,
  and `13` class-1 recall protectors (`1 > 0`) with margin/weight `0.04/0.8`.
  Loader audit matched all `40/40` sample-index rows, with no path fallback.
- Smoke:
  `runs\smoke_patchdir_targetedmargin_yolof_keeper01_t0727_w010_120b_1e_20260703`
  resumed the no-pretrain keeper, used explicit `yolo_f`, full validation
  support `2606`, `120` train batches for `1` epoch, LR `1e-5`,
  teacher-focus-binary `0.015`, bbox spatial fusion, boundary-band bbox dropout,
  and targeted-margin loss weight `0.010`. The targeted loss was active but very
  sparse (`train_targeted_margin_fraction=0.0070`,
  `train_targeted_margin_loss=0.0089`).
- Metrics: validation macro/class-1 F1 fell to `0.8816/0.6763`, below the
  no-pretrain keeper `0.8847/0.6860`. Class-1 P/R was `0.6000/0.7748`.
  Confusion matrix:
  `[[486,51,2,0,10],[18,117,11,0,5],[6,11,512,8,7],[0,4,53,653,2],[9,12,4,1,624]]`.
  The policy did not reduce `0->1` or `4->1` and slightly hurt class 2/3
  boundary errors.
- XAI audit:
  `runs\xai_smoke_patchdir_targetedmargin_yolof_keeper01_t0727_w010_selected12_20260703`
  audited 12 high-confidence validation mistakes with `method=all`,
  robustness probes, `stem_last`, explicit `yolo_f`, and `crop_bbox`. Top
  confusions stayed `0->1=52`, `3->2=52`, `1->0=18`, `4->1=12`, `1->2=11`.
  Foreground mass was weaker than the keeper-like XAI
  (`attention/grad-rollout/Grad-CAM/rollout` =
  `0.8927/0.9650/0.9479/0.9392`), background blur/gray drops were still near
  zero (`0.0005/-0.0004` original-pred drop), and object desaturation dominated
  (`0.0886`). Review flags were border/background heavy
  (`attention_background=5/12`, `attention_border=11/12`,
  `Grad-CAM_border=6/12`). Manual overlays showed attention on background,
  stem/leaf, bright edges, and broad surface patches rather than a stable
  interior class-1 cue.
- Decision: reject before probe/full train. The OOF directional signal is useful
  for post-hoc reliability analysis, but compressing only `40` sparse rows into
  a targeted-margin continuation does not move the representation and can make
  attention more diffuse. Do not repeat
  `patch_directional_targeted_margin pair=0-1/direction=1->0/threshold=0.7273/
  suppressor=0.06x1.5/protector=0.04x0.8/loss_weight=0.010/LR=1e-5/120b/1e`
  on the current keeper. Cleanup copied metrics/config/history into the XAI
  folder and deleted the rejected train directory plus launch logs; manifest
  `runs\cleanup_manifest_20260703_patchdir_targetedmargin_reject_smoke.json`
  reclaimed `181.67 MB`.

## Cleanup 2026-07-03 - Obsolete Train-to-Val Reroute/Focus Probes

- Before continuing model work, I preserved the current no-pretrain keeper,
  final test audit, and fixed patch-evidence diagnostic, then cleaned a small
  set of obsolete train-to-val artifacts from the old group-clean line. These
  runs were already recorded as rejected in TODO/audit notes and are not valid
  gates for the current `class_f`/`yolo_f` workflow.
- Deleted artifacts included old `class1_reroute_probe_groupclean_v26*`,
  `trainfit_logit_bias_probe_groupclean_v26*`,
  `focus_class1_specialist_train_to_val_groupclean*`,
  `focus_specialist_train_to_val_groupclean*`, and the dry
  `foreground_surface_stats_probe_v46_dry_20260626` directory.
- Cleanup manifest:
  `runs\cleanup_manifest_20260703_obsolete_train_to_val_reroute_focus_probes.json`.
  Reclaimed `127.48 MB`. Preserved artifacts listed in the manifest include
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701`,
  `runs\eval_yolof_teacherfocusbinary015_best_test_final_20260702`, and
  `runs\patch_evidence_mil_yolof_keeper_01only_t060_m040_top4_cropbbox_test_final_20260703`.

## Diagnostic 2026-07-03 - Patch-Evidence Direction Filter Val Trap

- Artifact:
  `runs\patch_evidence_direction_filter_diagnostic_20260703`. This replays the
  fixed `0-1` patch-evidence verifier CSVs and compares the default verifier
  with a post-hoc filter that applies only default `1->0` changes, equivalent on
  these artifacts to removing `0->1` routing.
- Validation looked tempting: base `0.8841/0.6841`, fixed verifier
  `0.8892/0.7073`, and `1->0` only `0.8904/0.7103`, with `24` changes
  (`14` corrections, `4` harms, `6` neutral). This is exactly the kind of
  direction rule that can look better than the broader verifier on val.
- The already-frozen final test artifact rejects it: fixed verifier
  `0.8905/0.6795`, but `1->0` only falls to `0.8884/0.6710`, with fewer useful
  changes (`10` total, `6` corrections, `3` harms, `1` neutral). This test read
  is used only to reject a tempting post-hoc rule, not to tune a new threshold.
- Decision: do not implement or tune the `1->0` only patch-verifier direction
  filter. Keep the artifact as a negative control: direction-specific reliability
  is real, but post-hoc val filtering is unstable unless a new train-only/OOF
  target changes the representation.

## Diagnostic/Final Audit 2026-07-03 - Stacked Patch+Feature Selective Verifier

- Rationale/research: I checked stacked generalization and selective
  classification before changing code. Wolpert's stacked generalization uses
  out-of-fold base predictions for a second-level learner
  (`https://www.sciencedirect.com/science/article/pii/S0893608005800231`).
  Geifman & El-Yaniv frame selective classification as a risk/coverage gate for
  DNNs (`https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks`),
  and SelectiveNet makes the same reject-option idea explicit
  (`https://proceedings.mlr.press/v97/geifman19a.html`). The useful local
  signal so far is not another global threshold; it is a selective accept/reject
  gate over verifier proposals, trained only from OOF train evidence.
- Implementation/preflight: updated
  `trkh.tools.probe_pairwise_feature_verifier` so `predictions.csv` exports
  `verifier_0_1_prob_0/1`, replacing train pair rows with true OOF probabilities.
  Added `trkh.tools.stack_patch_feature_verifiers`, which merges patch-evidence
  and feature-verifier CSVs by `sample_index`, uses the patch artifact as the
  canonical base prediction, and fits a logistic accept/reject gate only on
  union verifier proposals. The chosen frozen recipe was
  `target_mode=class1_benefit`, threshold `0.5`, `C=1.0`, `5` OOF folds.
  Preflight passed: `py_compile` and focused pytest
  `tests\test_pairwise_feature_verifier.py tests\test_stack_patch_feature_verifiers.py`
  (`8 passed`).
- Development val artifact:
  `runs\stacked_patch_feature_verifier_yolof_keeper01_class1benefit_t050_20260703`.
  It used patch train OOF plus feature train OOF from
  `runs\pairwise_feature_verifier_yolof_keeper_01_oofprob_t060_m040_20260703`.
  The gate had `50` train candidates and `34` val candidates. Validation base
  was macro/class-1 `0.8841/0.6841`; union proposals reached `0.8895/0.7077`;
  stacked selective reached `0.8911/0.7125`, class-1 P/R `0.6746/0.7550`,
  with `25` changes (`15` corrections, `4` harms, `6` neutral). This crossed
  the class-1 `0.70` diagnostic gate and improved over the fixed patch verifier
  on validation.
- Val XAI:
  `runs\xai_stacked_patch_feature_verifier_yolof_keeper01_class1benefit_changed25_20260703`
  audited all accepted val changes. Foreground mass stayed high
  (`attention/grad-rollout/Grad-CAM/rollout` =
  `0.9489/0.9705/0.9894/0.9524`), background blur/gray drops stayed near zero
  (`0.0006/0.0004` original-pred drop), and object desaturation was still the
  strongest perturbation (`0.0378` target drop). Review flags remained
  near-tie/border heavy (`near_tie=12/25`, `attention_border=17/25`,
  `rollout_border=21/25`). The gate was suppressing class-1 false positives,
  not discovering a new background/context cue.
- Frozen final-test audit:
  `runs\stacked_patch_feature_verifier_yolof_keeper01_class1benefit_t050_test_final_20260703`.
  I froze the val-selected recipe and reran only the required feature-verifier
  test probabilities in
  `runs\pairwise_feature_verifier_yolof_keeper_01_oofprob_t060_m040_testfinal_20260703`.
  Test base was macro/class-1 `0.8865/0.6667`; fixed patch verifier was already
  known at `0.8905/0.6795`; stacked selective fell to `0.8842/0.6536`, with
  `12` changes (`6` corrections, `5` harms, `1` neutral). Feature-only test also
  reduced class-1 to `0.6581`. This rejects the method.
- Test XAI:
  `runs\xai_stacked_patch_feature_verifier_yolof_keeper01_class1benefit_test_changed12_20260703`
  confirmed the same failure mode. Foreground remained high
  (`attention/grad-rollout/Grad-CAM/rollout` =
  `0.9678/0.9898/0.9112/0.9777`), background blur/gray was negligible
  (`0.0002/-0.0006` original-pred drop), object desaturation dominated
  (`0.0535` target drop), and accepted cases included `5` true class-1 samples
  incorrectly suppressed to class `0`.
- Decision: reject the stacked post-hoc patch+feature verifier as a final route.
  It is a useful negative control showing that validation suppressor gates can
  look strong while over-suppressing true class 1 on test. Do not repeat
  `stack_patch_feature_verifiers target_mode=class1_benefit/threshold=0.5` or
  nearby post-hoc accept thresholds on the current keeper. Future work must add
  a recall-protection or representation signal for true class-1 surface defects,
  not just a stronger false-positive suppressor.

## Diagnostic 2026-07-03 - Top-5 Teacher Recall Protector Reject

- Rationale: after the stacked verifier failed final test by suppressing true
  class-1 samples to class 0, I checked whether the strongest pretrained top-5
  TTA teacher could act only as a recall protector for fixed patch-verifier
  `1 -> 0` suppressions. This keeps the teacher as a diagnostic signal, not a
  new final inference dependency, and does not use test for threshold selection.
- Artifact:
  `runs\top5_tta_teacher_yolof_sampleindex_20260703`. The existing top-5
  `class_f` teacher cache was strictly remapped to `yolo_f` sample-index order
  with `trkh.tools.remap_classification_teacher_to_yolo`. Coverage was full:
  validation `2606/2606`, test `1267/1267`, no missing keys, and class order was
  passed explicitly from `target_classes`.
- Validation diagnostic:
  `runs\top5_tta_teacher_yolof_sampleindex_20260703\recall_protector_val_summary.json`.
  The fixed patch verifier has validation macro/class-1 F1 `0.8892/0.7073`.
  There were only `24` fixed-patch suppressions from base class `1` to non-1,
  with true targets `{0:14, 1:4, 2:1, 4:5}`. Restoring class 1 by top-5
  teacher confidence/prediction did not help. The best metric was a no-op
  high threshold (`0` restored), while non-empty candidates all fell below the
  fixed verifier; for example `p1 >= 0.45` restored `5` rows but dropped to
  macro/class-1 `0.8880/0.7027`, and teacher-predicted-class-1 at `p1 >= 0.70`
  restored `1` row, a true class 0, dropping to `0.8886/0.7052`.
- Decision: reject simple top-5 teacher recall-protection for patch suppressions.
  The pretrained ensemble remains a strong upper-bound/representation diagnostic,
  but its class-1 confidence does not safely identify the true class-1 samples
  lost by the patch verifier. Do not freeze or test a top-5 restoration rule from
  this validation result. The next useful teacher work should compare top-5 vs
  TRKH error structure to find a trainable representation signal, not add a
  post-hoc restore threshold.

## Diagnostic 2026-07-03 - TRKH + Top-5 Soft Ensemble Upper Bound

- Rationale: since top-5 TTA is much stronger than the scratch TRKH anchor but
  not a safe recall-protector, I checked whether the two probability spaces are
  complementary. This is an upper-bound diagnostic with pretrained inference,
  not the final no-pretrain model. Weights were selected from validation only:
  one class-1 optimum and one macro optimum, then frozen for test.
- Validation complementarity: top-5 corrected `105` TRKH validation mistakes
  while breaking `34` TRKH-correct cases. By target class, top-5 fixed
  `31/12/22/23/17` TRKH mistakes for classes `0..4`, but broke `4/20/4/5/1`
  TRKH-correct cases respectively. The teacher is conservative for class 1:
  validation top-5 class-1 P/R/F1 was `0.7483/0.7285/0.7383`, while TRKH base
  was `0.6082/0.7815/0.6841`.
- Frozen class-1-opt ensemble:
  `runs\soft_ensemble_trkh075_top5tta025_yolof_val_20260703` selected
  `TRKH=0.75/top5=0.25` on validation, reaching macro/class-1
  `0.9166/0.7500`. The frozen final-test artifact
  `runs\soft_ensemble_trkh075_top5tta025_yolof_test_final_20260703` reached
  macro/class-1 `0.9196/0.7737`, class-1 P/R `0.8281/0.7260`.
- Frozen macro-opt ensemble:
  `runs\soft_ensemble_trkh048_top5tta052_yolof_val_20260703` selected
  `TRKH=0.48/top5=0.52` on validation, reaching macro/class-1
  `0.9166/0.7492`. The frozen final-test artifact
  `runs\soft_ensemble_trkh048_top5tta052_yolof_test_final_20260703` reached
  macro/class-1 `0.9242/0.7786`, class-1 P/R `0.8793/0.6986`.
- Decision: pretrained top-5 plus TRKH soft ensembling beats the scratch TRKH
  anchor and the AIDT disagreement-gate style baselines, but it does not beat
  top-5 TTA alone on frozen test (`0.9248/0.7786`). Keep
  `runs\top5_tta_teacher_yolof_sampleindex_20260703\trkh_top5_soft_ensemble_diagnostic_summary.json`
  as representation upper-bound evidence. Do not chase more soft-ensemble
  weights on test; future work should distill or reproduce the teacher's
  conservative false-positive control inside the no-pretrain model without
  losing class-1 recall.

## Diagnostic 2026-07-03 - Top-5 Train Teacher Cache Is In-Sample Hard-Label

- Rationale: to see whether the top-5 teacher could become a train-only
  distillation signal, I exported train predictions for the five pretrained
  TIMM experts using the same TTA recipe as val/test: horizontal flip plus
  brightness `-0.06,0.06` and contrast `0.92,1.08`. The checkpoints were the
  existing `image_baseline_experiments\outputs` models, and no raw data was
  changed.
- Artifacts:
  `runs\pretrained_expert_predictions_tta_5class_train_20260703`,
  `runs\ensemble_top5_tta_teacher_cache_5class_train_20260703`, and remapped
  `runs\top5_tta_teacher_yolof_sampleindex_20260703\teacher_probs_yolof_train_sampleindex_top5tta.csv`.
  Remap coverage was full (`9215/9215`) with no missing keys.
- Audit:
  `runs\top5_tta_teacher_yolof_sampleindex_20260703\train_teacher_reliability_audit.json`.
  The train ensemble has agreement `0.9998` with hard labels, only `2`
  disagreements out of `9215`, mean confidence `0.9919`, median confidence
  `0.99998`, and p05 confidence `0.9730`. The two disagreements are both
  predicted as class 1 (`target 2 -> 1` and `target 0 -> 1`).
- Decision: do not use the in-sample top-5 train cache as a direct KD,
  teacher-pairwise, or class-1 conservative false-positive target. It is almost
  identical to the hard labels and not fold-safe/OOF, so a smoke train would
  mostly repeat CE with a very confident teacher. A useful top-5 transfer would
  require OOF teacher predictions or a different teacher-derived representation
  signal, not this in-sample train cache alone.

## Diagnostic 2026-07-03 - Sample-Index Top-5/TRKH Router Reject

- Rationale: after soft ensembling failed to beat top-5 TTA, I tested whether a
  router can learn when to trust TRKH instead of top-5. To avoid the old
  path-collapse issue on multi-object `yolo_f` rows, I normalized both expert
  CSVs to numeric class names and `sample_index` keys in
  `runs\top5_trkh_router_inputs_20260703`, then used the existing
  `trkh.tools.train_trainval_expert_router`. Focused router tests passed
  (`tests\test_trainval_expert_router.py`, `3 passed`).
- Protocol: validation rows were used as the router train split with 5-fold OOF
  selection (`selection_source=train_oof`, focus class `1`, focus weight `0.25`);
  the test rows were evaluated only after candidate selection. This is still a
  pretrained inference diagnostic, not a no-pretrain solution.
- Artifact:
  `runs\top5_trkh_trainval_router_valfit_testfinal_20260703`. OOF selection
  chose `avg_temp_0p5` with OOF macro/class-1 `0.9166/0.7492`. Frozen final
  test for the selected candidate reached macro/class-1 `0.9226/0.7727`, class-1
  P/R `0.8644/0.6986`, below top-5 TTA alone (`0.9248/0.7786`). The tool also
  reports that `extra_trees_leaf10` would score test macro/class-1
  `0.9250/0.7704`, but it was not selected by OOF and must not be promoted from
  test hindsight.
- Decision: reject top-5/TRKH sample-index router as a final route. It confirms
  the remaining top-5/TRKH complementarity is small and unstable under
  fold-safe selection. Do not tune router candidates or selection weights after
  seeing test. The no-pretrain path still needs a new trainable representation
  cue rather than a post-hoc expert router.

## Smoke 2026-07-03 - AugMix/JSD-lite Consistency Reject

- Rationale: after background/context and patch-router attempts kept the same
  surface-boundary failure mode, I checked AugMix-style consistency as a
  no-pretrain robustness regularizer. The reference idea was AugMix
  (`https://arxiv.org/abs/1912.02781`): mix several simple stochastic
  augmentations and use a Jensen-Shannon consistency term across clean and
  augmented views. I also rechecked WS-DAN attention crop/drop
  (`https://arxiv.org/abs/1901.09891`), but did not choose that route because
  this branch has already failed many crop/drop/border/attention variants.
- Implementation: added configurable AugMix/JSD-lite train-time consistency to
  `trkh\training\train.py`, `TrainConfig`, and the v8 launcher. The operations
  are geometry-preserving photometric/detail perturbations
  (brightness/contrast/luma-saturation/gamma/soften-sharpen), run only for a
  sampled subset, and add JSD across clean plus two augmented forward passes.
  Focused preflight passed:
  `py_compile trkh\training\train.py trkh\core\config.py` and
  `pytest tests\test_illumination_consistency_loss.py -q` (`7 passed`). Dry-run
  confirmed explicit `yolo_f`, full validation, no stale hard-sample manifest,
  and active AugMix flags.
- Smoke:
  `smoke_v8_yolof_augmixjsd008_p50_s18_teacherfocusbinary015_24b_1e_20260703`
  resumed the current no-pretrain keeper with teacher-focus-binary, bbox spatial
  fusion, pairwise routing, boundary dropout, and
  `weight=0.008/prob=0.50/severity=0.18/width=2/depth=2/alpha=1.0/temp=1.0`.
  Full validation support was `2606`. The smoke reached macro/class-1 F1
  `0.8831/0.6784`, class-1 P/R `0.6073/0.7682`, below the keeper
  `0.8847/0.6860` and below the `0.70` class-1 gate. AugMix was active
  (`train_augmix_consistency_loss=0.00104`,
  `train_augmix_consistency_fraction=0.4974`) but did not move the main class-1
  tradeoff. Top confusions remained `3->2=53`, `0->1=49`, `1->0=19`,
  `4->1=12`, and `1->2=11`.
- XAI:
  `runs\xai_smoke_v8_yolof_augmixjsd008_p50_s18_val_selected12_20260703`.
  Foreground mass stayed very high
  (`attention/grad_rollout/gradcam/rollout fg = 0.9834/0.9733/0.9834/0.9529`).
  Background perturbation remained near zero
  (`background_blur` pred drop `0.0006`, `background_gray` `-0.0004`), while
  object desaturation was larger (`0.0633` pred drop). Review flags were still
  border/object-detail dominated: `gradcam_border_attention=4`,
  `rollout_border_attention=4`, `object_color_sensitive=3`.
- Cleanup and decision: reject this exact AugMix/JSD-lite setting before probe
  or full train. It is another regularizer that leaves the same
  surface/boundary ambiguity intact, and generic photometric invariance is risky
  because color/surface detail appears label-relevant. The smoke run was deleted
  after copying `history.csv`, `best_metrics.json`, `resolved_config.json`,
  `launcher_args.json`, and `color_audit.json` into the XAI artifact; manifest:
  `runs\cleanup_manifest_20260703_augmixjsd008_reject_smoke.json`. The launcher
  accidentally auto-created a `final_test` subdirectory during smoke; it was
  deleted with the smoke run and was not used for method selection. Future smoke
  commands should pass `-SkipFinalTest` or `--skip-final-test`.

## Smoke 2026-07-03 - High-Frequency Texture Expert Reject

- Rationale: recent FGVC work again points at subtle detail/frequency cues, for
  example SCOPE's spatial decomposition for subtle structural and texture cues
  (`https://arxiv.org/html/2508.06959v1`), FT-Former frequency-token extraction
  for preserving discriminative details (`https://www.espublisher.com/uploads/article_pdf/es2040.pdf`),
  and WST wavelet blocks for preserving low/high-frequency token information
  (`https://ojs.aaai.org/index.php/AAAI/article/download/34387/36542`). The repo
  already has `HighFrequencyTextureExpert`, so I did not write a duplicate
  texture branch. Instead I tested the existing checkpoint-safe branch on the
  current keeper.
- Preflight and dry-run: `py_compile trkh\models\model.py
  trkh\training\train.py trkh\core\config.py` passed, and
  `pytest tests\test_high_frequency_texture_expert.py -q` passed (`4 passed`).
  Dry-run confirmed explicit `yolo_f`, keeper resume, full validation, empty
  hard-sample manifest, `skip_final_test=true`, and active high-frequency
  settings.
- Smoke:
  `smoke_v8_yolof_hifreqtex_w010_pw010_teacherfocusbinary015_24b_1e_20260703`
  used the same keeper recipe plus
  `HighFrequencyTextureExpert=true/aux=0.010/pairwise=0.010/route_pairs=0-1,1-2,2-3,1-4/route_margin=0.25/logit_scale=0.16/analysis_size=96`.
  Full validation support was `2606`, with no final-test artifact created. The
  branch was active (`train_high_frequency_texture_aux_loss=5.4336`,
  `train_high_frequency_texture_pairwise_loss=0.6914`), but validation reached
  only macro/class-1 F1 `0.8828/0.6764`, class-1 P/R `0.6042/0.7682`, below the
  keeper and below gate. Confusions remained `3->2=52`, `0->1=50`, `1->0=19`,
  `4->1=12`, `1->2=11`.
- XAI:
  `runs\xai_smoke_v8_yolof_hifreqtex_w010_pw010_val_selected12_20260703`.
  Foreground mass stayed high
  (`attention/grad_rollout/gradcam/rollout fg = 0.9834/0.9733/0.9834/0.9529`),
  background blur/gray remained near zero (`0.0007/-0.0004` pred drop), and
  object desaturation stayed much larger (`0.0632`). Flags stayed border/detail
  oriented: `gradcam_border_attention=4`, `rollout_border_attention=4`,
  `object_color_sensitive=3`.
- Cleanup and decision: reject this exact high-frequency texture expert setting
  before probe. A raw high-pass/gradient/laplacian expert did not add a reliable
  class-1 cue and left the same surface/boundary ambiguity. The smoke run was
  deleted after copying metrics/config into the XAI artifact; manifest:
  `runs\cleanup_manifest_20260703_hifreqtex_w010_reject_smoke.json`, reclaimed
  `196.69 MB`.

## Smoke 2026-07-03 - OOF Patch-Verifier Soft Target Reject

- Rationale: the 0-1 patch-evidence verifier is the first train-only diagnostic
  that crossed the validation class-1 gate post-hoc, so I tested whether its
  train OOF probability can be used directly as soft classification targets.
  Before the smoke, I fixed a data-integrity issue: `yolo_f` has duplicate image
  paths for multi-object images, so ambiguous soft-target manifests must match by
  `sample_index`, not by path.
- Implementation: `_load_ambiguous_soft_target_manifest` now parses
  `sample_index`/`dataset_index` and returns a mapping with
  `by_sample_index`. `AmbiguousSoftTargetDataset` prefers the sample-index target
  before path fallback and reports sample-index coverage in
  `soft_target_summary()`. Focused preflight passed:
  `py_compile trkh\training\train.py trkh\data\dataset.py` and
  `pytest tests\test_detection_calibration.py -k "ambiguous_soft_target or soft_target_class_weights" -q`
  (`2 passed`).
- Smoke:
  `smoke_v8_yolof_patchoof_softtarget_teacherfocusbinary015_24b_1e_20260703`
  resumed the current no-pretrain keeper with teacher-focus-binary, bbox spatial
  fusion, pairwise routing, boundary dropout, and
  `AmbiguousSoftTargetManifest=runs\patch_evidence_mil_yolof_keeper_01only_t060_m040_top4_cropbbox_oofteacher_20260703\train\pair_verifier_teacher_oof.csv`
  with `AmbiguousSoftTargetAlpha=0.0`. The manifest audit showed `8064` path
  keys, `9215` sample-index keys, `1151` duplicate path rows,
  `0` duplicate sample-index rows, key mode `sample_index`, matched samples
  `9215/9215`, and no leakage. Full validation support was `2606`, and
  `SkipFinalTest` prevented any final-test artifact.
- Metrics: validation reached only macro/class-1 F1 `0.8828/0.6784`, class-1
  P/R `0.6073/0.7682`, below the keeper `0.8847/0.6860` and below the `0.70`
  class-1 gate. Confusions stayed `3->2=52`, `0->1=49`, `1->0=19`,
  `0->4=12`, `4->1=12`, and `1->2=11`. The normal keeper losses were still
  active (`teacher_focus_binary_loss=0.5288`, selected fraction `0.7760`), but
  the OOF soft labels did not shift the decision boundary in the useful
  direction.
- XAI:
  `runs\xai_smoke_v8_yolof_patchoof_softtarget_val_selected12_20260703`.
  Foreground mass stayed high
  (`attention/grad_rollout/gradcam/rollout fg = 0.9797/0.9723/0.9846/0.9536`).
  Background perturbation remained near zero
  (`background_blur` pred drop `0.0003`, `background_gray` `-0.0002`), while
  object desaturation was much larger (`0.0627` pred drop). Review flags stayed
  border/detail oriented: `grad_rollout_border_attention=2`,
  `gradcam_border_attention=4`, `rollout_border_attention=5`,
  `object_color_sensitive=3`, and `near_tie_top2=1`. Manual overlays confirmed
  the same pattern: `0->1` examples can still light up fruit-top plus bright
  adjacent background/object regions, while true class-1 `1->0` examples already
  cover the fruit surface but lack a reliable surface-boundary separator.
- Cleanup and decision: keep the sample-index soft-target loader fix, but reject
  this exact direct OOF patch-verifier soft-target recipe before probe. The OOF
  verifier remains useful as a post-hoc/reliability diagnostic, but its full-prob
  soft labels are too close to hard labels to create a new representation cue in
  the current head. The smoke run was deleted after copying metrics/config into
  the XAI artifact; manifest:
  `runs\cleanup_manifest_20260703_patchoof_softtarget_reject_smoke.json`,
  reclaimed `190.44 MB`.

## Diagnostic 2026-07-03 - Cleanlab on OOF Patch-Verifier Teacher

- Rationale: before trying any automatic label-policy or strict soft-target
  route, I audited the train-only OOF patch-verifier teacher with Cleanlab. This
  keeps the raw dataset unchanged and is diagnostic only; the goal was to see
  whether the OOF teacher identifies review candidates that are safer than the
  already rejected train-OOF directional patch gate.
- Implementation: `trkh.tools.audit_label_issues_cleanlab` now accepts
  `target_index`/`true_index` rows even when `true_name` is absent and preserves
  `sample_index`. `trkh.tools.build_oof_cleanlab_strict_soft_targets` also
  writes `sample_index` so any future train-time manifest can avoid duplicate
  multi-object path collapse. Focused preflight passed:
  `py_compile trkh\tools\audit_label_issues_cleanlab.py
  trkh\tools\build_oof_cleanlab_strict_soft_targets.py` and
  `pytest tests\test_cleanlab_label_issue_schema.py
  tests\test_build_oof_cleanlab_strict_soft_targets.py -q`.
- Audit:
  `runs\cleanlab_patchoof_teacher_yolof_train_20260703` used
  `runs\patch_evidence_mil_yolof_keeper_01only_t060_m040_top4_cropbbox_oofteacher_20260703\train\pair_verifier_teacher_oof.csv`
  over `yolo_f/train`. It read and used all `9215` train rows with no path guard
  mismatch. Cleanlab found `81` label-issue candidates, dominated by
  `0->1` (`70`) plus `1->0` (`11`). Low self-confidence below `0.45` was also
  concentrated in classes `0/1` (`134` and `44` rows).
- Decision: keep the schema/sample-index infrastructure, but do not auto-relabel
  or train strict soft targets from this artifact. The candidate distribution is
  mostly `0->1`, exactly the direction where the prior train-OOF directional
  gate showed `18` candidates with `0` corrections and `17` harms. Without
  manual labels, using these Cleanlab rows as automatic corrections would likely
  widen class-1 false positives instead of improving the class-1 boundary.

## Probe 2026-07-03 - ExtraTrees Patch-Evidence Verifier Reject

- Rationale: the logistic 0-1 patch-evidence verifier remains the strongest
  legal validation diagnostic, so I tested whether a nonlinear verifier can
  separate the same frozen patch/head evidence better without touching raw data
  or test. This is a train-label-only post-hoc probe.
- Implementation: `trkh.tools.probe_pairwise_feature_verifier` and
  `trkh.tools.probe_patch_evidence_mil` now expose
  `--verifier-model {logistic,extra_trees}` plus ExtraTrees tree/leaf/depth
  settings. Focused preflight passed:
  `py_compile trkh\tools\probe_pairwise_feature_verifier.py
  trkh\tools\probe_patch_evidence_mil.py
  trkh\tools\audit_label_issues_cleanlab.py
  trkh\tools\build_oof_cleanlab_strict_soft_targets.py`, and
  `pytest tests\test_pairwise_feature_verifier.py
  tests\test_patch_evidence_mil.py tests\test_cleanlab_label_issue_schema.py
  tests\test_build_oof_cleanlab_strict_soft_targets.py -q` (`18 passed`).
- Probe:
  `runs\patch_evidence_mil_yolof_keeper_01only_extratrees_leaf5_t060_m040_top4_cropbbox_20260703`
  used `pairs=0-1`, `top_k=4`, `crop_bbox`, threshold/margin `0.60/0.40`,
  `ExtraTreesClassifier(n_estimators=300,min_samples_leaf=5)`, and 5-fold train
  OOF. Train pair OOF local accuracy rose to `0.9331`, but validation regressed:
  base macro/class-1 F1 `0.8841/0.6841` became `0.8812/0.6744`, class-1 P/R
  `0.6010/0.7682`. It changed only `7` validation rows, with `1` correction,
  `5` harms, and `1` neutral; the harms included true class-1 cases pushed to
  class `0`.
- XAI/audit:
  `runs\xai_patch_evidence_extratrees_leaf5_reject_changed7_20260703` audited
  the exact 7 changed sample indices. All `7/7` were near top-2 ties and all had
  register/CLS attention divergence. Foreground mass was still high
  (`attention=0.9449`, `gradcam=0.9710`), background mass low
  (`0.0551/0.0290`), and Grad-CAM border mass high (`0.3507`). This confirms the
  nonlinear verifier is amplifying the same fragile 0-1 surface/border boundary,
  not adding a new reliable cue.
- Cleanup and decision: reject ExtraTrees patch-evidence verifier before any
  train integration or test use. The full probe directory was compacted into
  `runs\diagnostic_patch_evidence_extratrees_leaf5_reject_20260703`
  (`summary`, train/val metrics, changed cases) and deleted via
  `runs\cleanup_manifest_20260703_patch_evidence_extratrees_leaf5_reject_probe.json`,
  reclaiming `4.46 MB`.

## Smoke 2026-07-03 - CYFLOD SmoothStep Loss Damping Reject

- Rationale: I searched recent noisy-label fine-grained work before changing
  code. CYFLOD (CVPRW 2025) proposes cyclic filtering and SmoothStep loss
  damping for fine-grained label noise / inter-class overlap, where per-sample
  confidence `exp(-loss)` below a scheduled `delta` receives reduced gradient.
  This directly matches the class-1 boundary-noise hypothesis while keeping raw
  data unchanged. I implemented only the low-risk loss-damping piece first; the
  filtering stage is deferred unless damping gives a positive smoke signal.
- Implementation: `trkh.training.train` now has
  `_cyflod_damped_classification_loss_from_per_sample`, CLI/config fields, train
  history columns, and V8 launcher passthrough. The helper composes existing
  sample weights, schedules active `delta` over a short sine cycle, clamps
  damping to a minimum weight, and normalizes by the original active weight sum
  so high-loss rows are actually damped. Focused preflight passed:
  `py_compile trkh\training\train.py trkh\core\config.py` and
  `pytest tests\test_self_paced_loss.py -q` (`4 passed`).
- Smoke:
  `smoke_v8_yolof_cyflod_damp025_w035_teacherfocusbinary015_24b_1e_20260703`
  used explicit `yolo_f`, keeper resume, `BBoxSpatialFusion=true`,
  `crop_bbox`, teacher-focus-binary `0.015`, `SkipFinalTest`, full validation,
  and CYFLOD `weight=0.35/delta=0.25/cycle=2/min_weight=0.10/start=1`. The
  first launch was deleted because it omitted `BBoxSpatialFusion` and stopped
  before train with checkpoint module mismatch. The valid smoke activated
  damping (`active_delta=0.25`, damped fraction `0.0833`, damped weight mean
  `0.3129`) but regressed from keeper macro/class-1 F1 `0.8847/0.6860` to
  `0.8795/0.6685`. Class-1 recall rose to `0.7947`, but precision fell to
  `0.5769`, with top confusions `0->1=55`, `4->1=17`, `2->1=12`, `1->0=14`,
  and `1->2=11`.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_cyflod_damp025_w035_val_selected12_20260703` audited
  12 selected validation cases. Foreground mass stayed high
  (`attention=0.8964`, `grad_rollout=0.9555`, `gradcam=0.9487`,
  `rollout=0.9145`), background perturbation remained near zero
  (`background_blur=0.0004`, `background_gray=-0.0009`), and object
  desaturation was still the larger probe (`0.0191`). Review flags were mostly
  border/top-2 issues (`rollout_border=8/12`, `gradcam_border=6/12`,
  `near_tie=6/12`). This means the damping did not solve localization/context;
  it widened the class-1 decision region and added false positives.
- Cleanup and decision: reject this CYFLOD damping setting before longer probe or
  final test. Metrics/config/history were copied into the XAI artifact under
  `source_smoke_metrics`; the train run was deleted via
  `runs\cleanup_manifest_20260703_cyflod_damp025_w035_rejected_smoke.json`,
  reclaiming `185.99 MB`. Keep the implementation as an experimental knob, but
  do not repeat this exact setting on the current keeper. If the CYFLOD family is
  revisited, it must use a much more conservative class-pair/OOF policy with
  explicit false-positive protection, not global high-loss damping.

## Smoke 2026-07-03 - Register Attention Alignment Reject

- Rationale: after repeated XAI showed border-heavy rollout and some
  register/CLS divergence on changed cases, I implemented a lightweight
  last-block attention alignment loss. The loss aligns CLS-to-patch and
  register-to-patch distributions and optionally penalizes attention mass outside
  the crop bbox. This tests whether the register tokens need stronger
  foreground/object agreement without changing raw data.
- Implementation: `trkh.training.train` now supports
  `_register_attention_alignment_loss_from_features`, last-block attention
  capture on demand, CLI/config fields, V8 launcher passthrough, history columns,
  and focused unit coverage in `tests\test_border_attention_suppression.py`.
  Focused preflight passed:
  `py_compile trkh\training\train.py trkh\core\config.py` and
  `pytest tests\test_border_attention_suppression.py
  tests\test_bbox_token_prior_source.py -q` (`8 passed`), followed by a focused
  recheck after limiting attention capture to the final block.
- Smoke:
  `smoke_v8_yolof_regattn_align_w010_fg035_teacherfocusbinary015_24b_1e_20260703`
  used explicit `yolo_f`, keeper resume, `BBoxSpatialFusion=true`, `crop_bbox`,
  teacher-focus-binary `0.015`, `SkipFinalTest`, full validation, and register
  attention alignment
  `weight=0.010/classes=0,1,2,4/bbox_margin=0/agreement=1.0/foreground=0.35/start=1`.
  The loss was active (`0.1108`), but agreement was already tiny (`JS=0.0055`)
  and bbox foreground mass was only about `0.699`. Full validation regressed
  from keeper macro/class-1 F1 `0.8847/0.6860` to `0.8781/0.6648`. Class-1 P/R
  was `0.5714/0.7947`, with top confusions `0->1=55`, `4->1=17`,
  `2->1=13`, and `1->0=14`.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_regattn_align_w010_fg035_val_selected12_20260703`
  audited 12 selected validation cases. Foreground mass stayed high
  (`attention=0.9083`, `grad_rollout=0.9579`, `gradcam=0.9586`,
  `rollout=0.9229`), background blur/gray drops stayed near zero, and object
  desaturation remained the larger perturbation. Register diagnostics showed
  `cls_register_heatmap_similarity=0.9995`, so CLS/register disagreement is not
  a bottleneck on this anchor. Review flags were still mostly near-tie and
  border cues (`near_tie=6/12`, `rollout_border=8/12`, `gradcam_border=5/12`).
- Cleanup and decision: reject this exact register-attention alignment setting.
  The evidence and source metrics were kept under the XAI directory; the smoke
  run was deleted via
  `runs\cleanup_manifest_20260703_regattn_align_w010_rejected_smoke.json`,
  reclaiming `177.54 MB`. The implementation stays as an experimental knob, but
  the next valuable direction should not force more bbox/foreground attention on
  this keeper; the model already looks at the object, while the remaining error
  is a fragile surface/boundary decision.

## Smoke 2026-07-03 - YOLO Source-Group Sample Weights Reject

- Rationale: the train split has a source-layout mismatch that is visible without
  touching labels or images. `runs\yolof_source_group_consistency_audit_20260703`
  shows `yolo_f/train` has `691` multi-object images and `144` mixed-label
  images, while validation has only `29` same-label multi-object images and test
  is fully single-object. Class 1 appears in `102` mixed-label train images. I
  tested whether lightly downweighting these train-only source groups can reduce
  the class-1 false-positive region caused by adjacent/mixed objects.
- Implementation: added
  `trkh.tools.build_yolo_source_group_sample_weights`, which mirrors
  `MangoYOLOCropDataset` sample-index order by sorted label file and valid object
  row. It writes a train-only `sample_index` manifest and refuses non-train
  splits. Focused preflight passed:
  `py_compile trkh\tools\build_yolo_source_group_sample_weights.py` and
  `pytest tests\test_build_yolo_source_group_sample_weights.py
  tests\test_build_patch_oof_sample_weights.py -q` (`3 passed`).
- Manifest:
  `runs\source_group_sample_weights_yolof_train_mixed065_c1p085_same085_c4same075_20260703`
  covered `9215` train objects and wrote `1842` weighted rows. Policy:
  mixed-label non-class1 `0.65`, mixed-label class1 protected at `0.85`,
  same-label multi-object `0.85`, and same-label class4 `0.75`. Weighted rows
  by class were `0:234`, `1:139`, `2:271`, `3:37`, `4:1161`; mean train
  source-group weight was `0.9524`.
- Smoke:
  `smoke_v8_yolof_sourcegroup_w_mixed065_c1p085_same085_c4same075_60b_1e_20260703`
  used explicit `yolo_f`, keeper resume, `BBoxSpatialFusion=true`, `crop_bbox`,
  teacher-focus-binary `0.015`, existing sample-index teacher CSV, the
  source-group sample-weight manifest, `60` train batches, full validation, and
  `SkipFinalTest`. The sample-weight wrapper matched all `1842` sample indices
  and reported mean train weight `0.9524`, but full-val macro/class-1 F1 fell to
  `0.8779/0.6629`. Class-1 P/R was `0.5756/0.7815`, with confusion
  `0->1=54`, `4->1=16`, `2->1=13`, `1->0=15`, and `1->2=12`.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_sourcegroup_w_mixed065_c1p085_same085_c4same075_val_selected12_20260703`
  again showed high foreground mass (`attention=0.9156`, `gradcam=0.9538`),
  near-zero background blur/gray effect (`0.0005/0.0001`), and larger object
  desaturation (`0.0266`). Review flags remained border/near-tie dominated
  (`near_tie=6/12`, `rollout_border=7/12`, `gradcam_border=6/12`), and
  register/CLS heatmaps were already aligned (`0.9997`).
- Cleanup and decision: reject this exact source-group downweight policy before
  any longer probe. The tool and manifest are useful diagnostics, but simple
  downweighting of multi/mixed train source groups does not repair the boundary;
  it reduces useful class evidence and still widens class-1 false positives. The
  smoke run was deleted via
  `runs\cleanup_manifest_20260703_sourcegroup_sampleweight_rejected_smoke.json`,
  reclaiming `181.86 MB`. If source grouping is revisited, it needs a
  case-level reliability target or group-aware objective that protects true
  class-1 recall, not blanket downweighting.

## Smoke 2026-07-03 - Internal VICReg YOLOF Warmup Reject

- Rationale: after checking fine-grained noisy-label work and the MIL/TransFG
  family, I tested a different axis from the many rejected local loss/router
  tweaks: internal SSL representation warmup on train-only `yolo_f` objects.
  This is legal because it uses only the existing train split and does not alter
  raw images or labels. The hypothesis was that VICReg object-crop pretraining
  might give the scratch hybrid CNN/ViT a better surface representation before
  the class-1 boundary fine-tune.
- Preflight and SSL: `trkh.tools.pretrain_internal_barlow` was already present
  and focused tests passed for internal SSL (`tests\test_internal_ssl_vicreg.py`,
  `tests\test_internal_ssl_mae.py`, `tests\test_internal_ssl_dino.py`: `13
  passed`). The dry-run confirmed explicit `yolo_f`, train-only `9215` objects,
  balanced sampling, CUDA, and class counts `[1941,541,1920,2520,2293]`. The
  valid warmup
  `internal_vicreg_yolof_objectcrop_128b_1e_20260703` ran one epoch / 128 train
  batches with object crop, light color jitter, scale crop, and background
  suppression. It saved a checkpoint with best SSL loss `33.4361`.
- Fine-tune smoke:
  `smoke_v8_yolof_internalvicreg128b1e_ft120b_1e_20260703` resumed from the SSL
  checkpoint with explicit `yolo_f`, `BBoxSpatialFusion=true`, `crop_bbox`,
  teacher-focus-binary `0.015`, the existing train sample-index teacher CSV,
  no hard repeat, full validation support `2606`, and `SkipFinalTest`. Full
  validation regressed far below the no-pretrain keeper: macro/class-1 F1
  `0.7435/0.3801` versus keeper `0.8847/0.6860`. Class-1 P/R was only
  `0.3403/0.4305`, with confusion rows:
  `0:[466,60,2,2,19]`, `1:[53,65,22,0,11]`, `2:[5,21,455,48,15]`,
  `3:[1,1,109,584,17]`, `4:[37,44,29,10,530]`.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_internalvicreg128b1e_ft120b_val_selected12_20260703`
  audited 12 selected validation cases. The model still looked mostly at the
  fruit (`attention/grad_rollout/gradcam/rollout` foreground mass
  `0.9662/0.9445/0.9569/0.9340`), background blur/gray perturbations were near
  zero (`-0.0026/0.0005` original-prediction drop), and object desaturation was
  larger (`0.0592`). Review flags stayed in the same border/near-tie family:
  `register_attention_diffuse=12/12`, `rollout_border=10/12`,
  `attention_border=5/12`, `near_tie=2/12`, with almost identical CLS/register
  heatmaps (`0.99998` similarity). Manual review of a high-confidence
  `1->2` case again showed weak class-1 surface separability, not a context
  localization failure.
- Cleanup and decision: reject internal VICReg warmup before any longer probe.
  The SSL and fine-tune source metrics/config/history were copied into the XAI
  artifact under `source_smoke_metrics`, then the fine-tune run, SSL pretrain
  run, and dry-run directory were deleted via
  `runs\cleanup_manifest_20260703_internalvicreg_yolof_rejected_smoke.json`,
  reclaiming `365.84 MB`. Do not repeat the exact
  `internal_vicreg_yolof_objectcrop 128b/1e + V8 fine-tune 120b/1e` recipe on
  the current anchor. If SSL is revisited, it needs a substantially different
  schedule or target and a gate that can plausibly produce a class-1 surface
  signal before spending a full probe.

## Smoke 2026-07-03 - Focus Class-1 AUC-Rank Loss Reject

- Rationale: after the repeated post-hoc verifier and attention/context failures,
  I tested a direct train-time ranking objective for the actual class-1 boundary:
  encourage class-1 samples to score above likely false-positive rival classes
  `0,2,4`. This follows the same broad idea as class-specific ranking/AUC
  surrogates, but is implemented inside the existing V8 train loop and does not
  alter raw data or use the test split.
- Implementation and preflight: added optional `focus_auc_rank_*` config/CLI/V8
  launcher fields and history telemetry. The helper uses the binary score
  `logit[class1] - logsumexp(logits[0,2,4])`, selects the hardest pairwise
  positive-vs-negative comparisons by `hard_fraction`, and applies a softplus
  margin loss. Preflight passed:
  `py_compile trkh\training\train.py trkh\core\config.py` and
  `pytest tests\test_focus_auc_rank_loss.py tests\test_focus_tversky_loss.py
  tests\test_focused_false_positive_margin.py -q` (`12 passed`). During the
  first run I found a telemetry bug: `loss_details` contained
  `focus_auc_rank_*`, but `train_loss_components` did not accumulate those
  keys, so `history.csv` showed zeros. After fixing that, a 1-batch active-check
  confirmed the loss was active (`loss=0.00317`, `7` class-1 positives,
  `19` negatives, `133` pairs).
- Smoke:
  `smoke_v8_yolof_focusaucrank_w004_h50_teacherfocusbinary015_120b_1e_active_20260703`
  resumed from the current no-pretrain keeper with explicit `yolo_f`,
  `BBoxSpatialFusion=true`, `crop_bbox`, teacher-focus-binary `0.015`,
  `FocusAucRankLossWeight=0.004`, class `1`, negatives `0,2,4`, margin `0.04`,
  temperature `0.12`, hard fraction `0.50`, full validation support `2606`, and
  `SkipFinalTest`. The ranking loss was active across the smoke
  (`train_focus_auc_rank_loss=0.0107`, mean `6.4` positives, `19.2` negatives,
  `122.8` pairs per batch), but validation fell to macro/class-1 F1
  `0.8772/0.6611`, below the keeper `0.8847/0.6860`. Class-1 P/R was
  `0.5728/0.7815`, with confusion
  `0:[480,57,3,0,9]`, `1:[15,118,12,0,6]`,
  `2:[4,13,514,6,7]`, `3:[0,3,58,651,0]`,
  `4:[9,15,3,2,621]`.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_focusaucrank_w004_h50_teacherfocusbinary015_120b_1e_active_val_selected12_20260703`
  audited 12 selected validation cases. Foreground mass stayed high
  (`attention=0.9817`, `grad_rollout=0.9739`, `gradcam=0.9777`,
  `rollout=0.9539`), background blur/gray perturbation remained near zero
  (`0.0010/0.0006` original-prediction drop), and object desaturation stayed much
  larger (`0.0808`). Review flags were still border-dominated
  (`gradcam_border=6/12`, `rollout_border=5/12`), and
  `cls_register_heatmap_similarity=0.9997`. The failure is not missing
  background context; the ranking objective made the class-1 decision region too
  broad for hard `0/2/4` negatives.
- Cleanup and decision: reject this exact focus AUC-rank recipe before any probe.
  Metrics/config/history were copied into the XAI artifact under
  `source_smoke_metrics`, then obsolete active-check and smoke train directories
  were deleted via
  `runs\cleanup_manifest_20260703_focusaucrank_rejected_smokes.json`,
  reclaiming `727.80 MB`. Keep the implementation as a tested experimental knob,
  but do not repeat the plain one-vs-rest class-1 ranking setting on the current
  keeper. If ranking is revisited, it needs asymmetric false-positive protection
  or an OOF/case-level reliability target, not just a larger/smaller global
  class-1 pairwise pressure.

## Smoke 2026-07-03 - Surface-Detail Attention View Reject On Current Keeper

- Rationale: the old V9 surface-detail attention-view run was rejected on an
  earlier baseline, but the current keeper uses learned-attention crop/drop with
  teacher-focus-binary, bbox spatial fusion, and `crop_bbox` prior. I tested
  whether switching the attention-view score to deterministic surface-detail
  would make train-time crop/drop focus on fruit texture instead of the broad
  learned map, without changing raw data or touching test.
- Preflight:
  `scripts\run_trkh_5class_attention_views_v8.ps1 -PreflightOnly` confirmed
  explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, current keeper resume,
  `SkipFinalTest=true`, empty hard manifest, `AttentionViewScoreSource=surface_detail`,
  `AttentionViewLossWeight=0.30`, crop/drop probabilities `0.40/0.20`, and
  foreground weight `0.80`.
- Smoke:
  `smoke_v8_yolof_surfaceattention_score_currentkeeper_60b_1e_20260703` used
  60 train batches, full validation support `2606`, and no final test. The
  attention-view loss was active (`train_attention_view_loss=1.1368`,
  crop/drop fractions `0.3901/0.2057`), but validation reached only
  macro/class-1 F1 `0.8786/0.6667`, below the keeper `0.8847/0.6860`.
  Class-1 P/R was `0.5777/0.7881`, with confusion
  `0:[483,54,3,0,9]`, `1:[14,119,12,0,6]`,
  `2:[3,13,515,6,7]`, `3:[0,4,59,648,1]`,
  `4:[9,16,3,2,620]`.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_surfaceattention_score_currentkeeper_60b_val_selected12_20260703`
  audited 12 selected validation cases. Foreground mass remained high
  (`attention=0.9815`, `grad_rollout=0.9733`, `gradcam=0.9775`,
  `rollout=0.9534`), background blur/gray stayed near zero
  (`0.0004/0.0007` original-prediction drop), and object desaturation stayed
  much larger (`0.0789`). Review flags were still border/object-color dominated
  (`gradcam_border=6/12`, `rollout_border=5/12`,
  `object_color_sensitive=4/12`), with CLS/register heatmap similarity
  `0.9996`.
- Cleanup and decision: reject the current-keeper surface-detail attention-view
  setting before any 120-batch probe. Source metrics/config/history were copied
  into the XAI artifact under `source_smoke_metrics`, then the train run was
  deleted via
  `runs\cleanup_manifest_20260703_surfaceattention_currentkeeper_rejected_smoke.json`,
  reclaiming `182.21 MB`. Do not repeat
  `AttentionViewScoreSource=surface_detail/loss=0.30/crop=0.40/drop=0.20/fg=0.80`
  on the current keeper; deterministic surface-detail crop/drop changes where
  training perturbs the object, but it still does not produce a reliable
  class-1 surface/boundary discriminator.

## Smoke 2026-07-03 - Class-Balanced Self-Paced Loss Reject

- Rationale: I checked recent noisy-label/long-tail sample-selection work before
  changing the training loop. IJCAI 2024 "Learning from Long-Tailed Noisy Data
  with Sample Selection and Balanced Loss"
  (`https://www.ijcai.org/proceedings/2024/0605.pdf`) explicitly warns that
  losses from samples with different labels may be incomparable and motivates
  class-aware sample selection. I also checked CBS
  (`https://arxiv.org/html/2402.11242v1`) and SED
  (`https://nust-machine-intelligence-laboratory.github.io/project-SED/`) for
  class-balanced noisy-label selection/reweighting. The TRKH-specific hypothesis
  was that the earlier global self-paced loss was suppressing useful class-1
  hard/boundary samples, so per-class loss normalization might downweight noisy
  outliers without comparing class-1 losses directly against head-class losses.
- Implementation and preflight: added optional
  `self_paced_loss_class_balanced` to config, trainer CLI, V8 launcher, loss
  helper, telemetry, SAM replay path, and history columns. The default remains
  unchanged. The class-balanced path normalizes detached per-sample losses
  within each target class in the batch, applies the percentile threshold per
  class, and logs `train_self_paced_normalized_loss` plus
  `train_self_paced_class_count`. Preflight passed:
  `py_compile trkh\training\train.py trkh\core\config.py` and
  `pytest tests\test_self_paced_loss.py -q` (`5 passed`). V8 dry-run confirmed
  explicit `yolo_f`, keeper resume, empty hard manifest, `SkipFinalTest=true`,
  `BBoxSpatialFusion=true`, `crop_bbox`, teacher-focus-binary `0.015`, and
  `self_paced_loss_class_balanced=true`.
- Smoke:
  `smoke_v8_yolof_cbselfpaced_w018_p70_g075_min60_60b_1e_20260703` used
  `SelfPacedLossWeight=0.18`, percentile `0.70`, gamma `0.75`, min weight
  `0.60`, 60 train batches, full validation support `2606`, and no final test.
  The class-balanced self-paced path was active (`selected_fraction=0.6875`,
  `class_count=5.0`, normalized loss `0.3026`, high-loss weight mean
  `0.7229`), but full-val macro/class-1 F1 was only `0.8779/0.6629`, below the
  keeper `0.8847/0.6860`. Class-1 P/R was `0.5756/0.7815`, with confusion
  `0:[483,54,3,0,9]`, `1:[15,118,12,0,6]`,
  `2:[3,13,515,6,7]`, `3:[0,4,58,649,1]`,
  `4:[9,16,3,2,620]`.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_cbselfpaced_w018_p70_g075_min60_val_selected12_20260703`
  audited 12 selected validation cases. Foreground focus stayed high
  (`attention=0.9814`, `grad_rollout=0.9731`, `gradcam=0.9776`,
  `rollout=0.9534`), background blur/gray original-prediction drops stayed near
  zero (`0.0007/0.0007`), and object desaturation stayed much larger
  (`0.0783`). Review flags were still border/object-color dominated:
  `gradcam_border=7/12`, `rollout_border=5/12`, `object_color_sensitive=4/12`,
  `near_tie=2/12`. Manual overlay review of the `4->1` and `1->0` cases showed
  the same surface/edge/border ambiguity, not a new background-context failure.
- Cleanup and decision: reject this exact class-balanced self-paced recipe before
  any 120-batch probe. Source metrics/config/history/logs and trace summary were
  copied into the XAI artifact under `source_smoke_metrics`, then the smoke run
  and copied logs were deleted via
  `runs\cleanup_manifest_20260703_cbselfpaced_rejected_smoke.json`, reclaiming
  `182.30 MB`. Keep the implementation as a tested experimental knob, but do
  not repeat
  `self_paced_loss_weight=0.18/percentile=0.70/gamma=0.75/min_weight=0.60/class_balanced=true`
  on the current keeper; class-wise normalization still lowers class-1 precision
  and does not create the missing surface/boundary reliability signal.

## Smoke 2026-07-03 - Focus Partial-AUC FPR-Range Guard Reject

- Rationale: after the plain class-1 AUC-rank loss widened false positives, I
  tested a narrower pAUC/FPR-range variant based on the partial-AUC direction
  from NeurIPS 2022 and ICML 2022 pAUC/DRO work. The implementation is
  deliberately different from the rejected all-pair AUC-rank: it only penalizes
  the top-risk configured negatives `0/2/4 -> 1` by class-1 probability and
  separately protects low-margin true class-1 positives against rivals `0/2/4`.
  It does not alter raw data or touch the test split.
- Implementation and preflight: added optional `focus_partial_auc_*` config,
  trainer CLI, V8 launcher fields, resume override, SAM replay wiring, history
  telemetry, and `tests\test_focus_partial_auc_loss.py`. Preflight passed:
  `py_compile trkh\training\train.py trkh\core\config.py` and
  `pytest tests\test_focus_partial_auc_loss.py tests\test_focus_auc_rank_loss.py
  tests\test_focused_false_positive_margin.py -q` (`13 passed`). V8 preflight
  confirmed explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`,
  `BBoxSpatialFusion=true`, `crop_bbox`, teacher-focus-binary `0.015`, empty
  hard manifest, and `SkipFinalTest=true`. Note: `SkipFinalTest` is a switch in
  the V8 launcher, so pass `-SkipFinalTest` without `$true`; otherwise PowerShell
  can bind `$true` positionally into `ClassificationFolderYoloData`.
- Smoke:
  `smoke_v8_yolof_focuspartialauc_w004_nf25_pf30_pw045_min05_60b_1e_20260703`
  resumed from the current no-pretrain keeper for 60 train batches / 1 epoch
  with full validation support `2606` and no final test. The pAUC guard was
  active (`train_focus_partial_auc_loss=0.0957`,
  `negative_loss=0.0727`, `positive_loss=0.0510`, average `6.4` focus positives,
  `19.2` configured negatives, selected positive/negative counts `2.4/5.0` per
  batch), but full-val macro/class-1 F1 was only `0.8776/0.6629`, below the
  keeper `0.8847/0.6860`. Class-1 P/R was `0.5756/0.7815`, with confusion
  `0:[483,54,3,0,9]`, `1:[15,118,12,0,6]`,
  `2:[3,13,515,6,7]`, `3:[0,4,59,648,1]`,
  `4:[9,16,3,2,620]`.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_focuspartialauc_w004_nf25_pf30_pw045_min05_val_selected12_20260703`
  audited 12 selected validation cases. Foreground focus stayed high
  (`attention=0.9814`, `grad_rollout=0.9731`, `gradcam=0.9776`,
  `rollout=0.9534`), background blur/gray original-prediction drops stayed near
  zero (`0.0007/0.0007`), and object desaturation stayed much larger (`0.0783`).
  Review flags were again border/object-color dominated:
  `gradcam_border=7/12`, `rollout_border=5/12`, `object_color_sensitive=4/12`,
  `near_tie=2/12`. Manual review of `4->1` and `1->0` overlays showed the same
  fruit surface/edge/severity ambiguity, with only peripheral context artifacts;
  this is not a missing wide-background-context problem.
- Cleanup and decision: reject this exact focus partial-AUC recipe before any
  120-batch probe. Source metrics/config/history/logs and trace artifacts were
  copied into the XAI artifact under `source_smoke_metrics`, then the smoke train
  run was deleted via
  `runs\cleanup_manifest_20260703_focuspartialauc_rejected_smoke.json`,
  reclaiming `182.56 MB`. Keep the implementation as a tested knob, but do not
  repeat
  `focus_partial_auc_loss_weight=0.004/class=1/negative_classes=0,2,4/margin=0.04/temp=0.12/negative_fraction=0.25/positive_fraction=0.30/positive_weight=0.45/min_negative_probability=0.05`
  on the current keeper. This pAUC/FPR-range guard is narrower than plain
  AUC-rank, but it still lowers class-1 precision and does not add the missing
  surface/boundary reliability signal.

## Smoke 2026-07-03 - Coupled Patch-Evidence Router Reject

- Rationale: the fixed post-hoc `0-1` patch-evidence verifier remains the
  strongest near-gate diagnostic, so I checked MIL-style ideas before trying an
  in-model compression. DSMIL (`https://arxiv.org/abs/2006.05538`) motivates
  combining an instance stream with bag-level attention, CLAM
  (`https://arxiv.org/abs/2004.09666`) adds attention-based instance clustering
  for diagnostic subregions, and SMILE
  (`https://proceedings.mlr.press/v156/lu21a/lu21a.pdf`) uses sparse MIL
  attention plus contrastive patch representation. The TRKH-specific attempt was
  intentionally different from the rejected router-only smoke: expose the
  existing `PatchEvidenceRouterHead` through the V8 launcher and update the full
  model at low LR so patch routing can couple back into the representation,
  without changing raw data or touching test.
- Implementation and preflight: V8 launcher now exposes the existing
  `PatchEvidenceRouter*` flags, passes them into `TrainArgs`, records them in
  resolved config, and supports disabling router inference routing. Focused
  preflight passed:
  `py_compile trkh\training\train.py trkh\core\config.py` and
  `pytest tests\test_patch_evidence_router.py tests\test_focus_partial_auc_loss.py -q`
  (`7 passed`). Dry-run confirmed explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, current keeper resume,
  `BBoxSpatialFusion=true`, `crop_bbox`, teacher-focus-binary `0.015`, empty
  hard manifest, no trainable-prefix freeze, and `SkipFinalTest=true`.
- Smoke:
  `smoke_v8_yolof_patchrouter_coupled_w012_pos12_top4_lr1e5_60b_1e_20260703`
  resumed from the current no-pretrain keeper for 60 train batches / 1 epoch
  with full validation support `2606` and no final test. The router loss was
  active, but it did not separate positive and negative local evidence:
  `train_patch_evidence_router_loss=0.7625`, fraction `0.4000`,
  positive/negative logits `-0.000127/-0.000126`, logit margin about
  `-0.000001`, and router probability `0.49997`. Full-val macro/class-1 F1 was
  only `0.8788/0.6648`, below the keeper `0.8847/0.6860`. Class-1 P/R was
  `0.5784/0.7815`, with confusion `0:[484,53,3,0,9]`,
  `1:[15,118,12,0,6]`, `2:[3,13,515,6,7]`,
  `3:[0,4,57,650,1]`, `4:[9,16,3,2,620]`.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_patchrouter_coupled_w012_pos12_top4_lr1e5_val_selected12_20260703`
  audited 12 selected validation cases. Foreground focus stayed high
  (`attention=0.9815`, `grad_rollout=0.9732`, `gradcam=0.9775`,
  `rollout=0.9534`), background blur/gray original-prediction drops stayed near
  zero (`0.0005/0.0005`), and object desaturation stayed much larger
  (`0.0783`). Review flags stayed border/object-color dominated:
  `gradcam_border=7/12`, `rollout_border=5/12`,
  `object_color_sensitive=4/12`, `near_tie=2/12`. Manual overlays for the
  representative `4->1` and `1->0` cases showed heat concentrated on fruit
  surface, specular/edge regions, and severity cues; this is again not a
  missing wide-background-context issue.
- Cleanup and decision: reject this coupled patch-router recipe before any
  120-batch probe. Source metrics/config/history/logs and architecture trace
  were copied into the XAI artifact under `source_smoke_metrics`, then the
  smoke train run was deleted via
  `runs\cleanup_manifest_20260703_patchrouter_coupled_rejected_smoke.json`,
  reclaiming `186.31 MB`. Keep the V8 launcher exposure because it is valid
  infrastructure, but do not repeat
  `PatchEvidenceRouterHead=true/pair=0-1/hidden=128/top_k=4/bbox_weight=0.75/dropout=0.05/logit_scale=0.12/routing=true/route_margin=0.30/min_pair_prob=0.04/loss_weight=0.012/positive_weight=1.20/full-model/LR=1e-5/60b`
  on the current keeper. The router stayed effectively uninitialized around
  probability `0.5`; the next patch-evidence attempt needs a non-zero
  reliability/selector initialization or a stronger fold-safe instance target,
  not another zero-start BCE/router variant.

## Smoke 2026-07-03 - Patch-Evidence Router Max-Margin Prior Reject

- Rationale: the coupled router smoke showed the zero-start MLP did not move
  from probability `0.5`. I added a default-off DSMIL-style instance stream
  prior: `PatchEvidenceRouterHead` can now add a non-zero patch margin prior
  (`none|weighted|max|topk_mean|weighted_minus_mean`) to the router logit before
  the residual MLP learns. This keeps old checkpoints compatible and tests the
  hypothesis that the existing `0-1` patch evidence needs a warm-start rather
  than more BCE weight. Raw data and test were untouched.
- Implementation and preflight: added
  `patch_evidence_router_margin_prior_mode/scale` to model config, model
  constructor, trace, trainer CLI, V8 launcher preflight/resolved config, and
  router tests. Preflight passed:
  `py_compile trkh\models\model.py trkh\training\train.py trkh\core\config.py`
  and
  `pytest tests\test_patch_evidence_router.py tests\test_trainable_module_prefixes.py -q`
  (`10 passed`). V8 dry-run confirmed explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, current keeper resume,
  `SkipFinalTest=true`, empty hard manifest, `margin_prior_mode=max`,
  `margin_prior_scale=1.0`, route margin/min pair probability `0.30/0.04`,
  and `PatchEvidenceRouterLossWeight=0.006`.
- Smoke:
  `smoke_v8_yolof_patchrouter_maxprior_s100_w006_60b_1e_20260703` used
  60 train batches, full validation support `2606`, and no final test. The
  prior did make the router non-zero
  (`train_patch_evidence_router_loss=0.6804`,
  positive/negative logits `0.3524/0.1110`, margin `0.2414`,
  probability `0.5558`), but it widened class-1 false positives. Full-val
  macro/class-1 F1 was only `0.8794/0.6685`, below the keeper
  `0.8847/0.6860`. Class-1 P/R was `0.5701/0.8079`, with confusion
  `0:[479,58,3,0,9]`, `1:[11,122,12,0,6]`,
  `2:[2,14,515,6,7]`, `3:[0,4,57,650,1]`,
  `4:[9,16,3,2,620]`.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_patchrouter_maxprior_s100_w006_val_selected12_20260703`
  audited 12 selected validation cases. Top confusion shifted to `0->1=58`.
  Foreground focus stayed high (`attention=0.9789`, `grad_rollout=0.9703`,
  `gradcam=0.9757`, `rollout=0.9466`), background blur/gray drops stayed near
  zero (`0.0010/0.0009`), and object desaturation grew to `0.1016`. Review
  flags stayed border/object-color dominated (`gradcam_border=5/12`,
  `rollout_border=5/12`, `object_color_sensitive=5/12`). Manual overlay of a
  high-confidence `0->1` showed the model still reacting mostly to fruit
  surface/color cues; the extra background/object hotspots were not supported by
  the perturbation audit.
- Cleanup and decision: reject the max-margin prior recipe before any probe.
  Source metrics/config/history/logs and trace were copied into the XAI artifact
  under `source_smoke_metrics`, then the smoke train run was deleted via
  `runs\cleanup_manifest_20260703_patchrouter_maxprior_rejected_smoke.json`,
  reclaiming `186.34 MB`. Keep the default-off margin-prior infrastructure for
  diagnostics, but do not repeat
  `PatchEvidenceRouterMarginPriorMode=max/Scale=1.0/logit_scale=0.20/loss_weight=0.006/positive_weight=1.10/full-model/LR=1e-5/60b`
  on the current keeper. A raw max instance prior is not reliable enough; it
  increases class-1 recall by admitting more `0/2/4->1` false positives.

## Smoke 2026-07-03 - OOF-Reliability Patch-Router Teacher Reject

- Rationale: after the raw max-prior router increased class-1 false positives,
  I switched from patch-margin warm-starting to a fold-safe reliability target.
  The target uses the existing train-OOF `0-1` patch verifier CSV only when the
  teacher agrees with the hard side of pair `0-1` and has enough pair mass and
  confidence. This follows the same conservative idea as noisy-label/OOD
  reliability filtering and calibrated uncertainty distillation: use OOF
  predictions as a reliability signal, not as automatic relabeling. References
  reviewed for this step: Cleanlab cross-val predicted-probability guidance,
  ReCoV noisy-label detection, and calibrated uncertainty distillation. Raw
  datasets and test split were untouched.
- Implementation and preflight: added a default-off
  `PatchEvidenceRouterTeacherCsv` path plus teacher loss weight, confidence,
  pair-mass, and positive-weight gates to config, trainer CLI, V8 launcher,
  dataset metadata plumbing, history telemetry, and tests. The target is
  separate from global KD and from the already-rejected direct OOF soft-target
  CE path. Preflight passed:
  `py_compile trkh\data\dataset.py trkh\training\train.py trkh\core\config.py`,
  PowerShell parse of `scripts\run_trkh_5class_attention_views_v8.ps1`, and
  `pytest tests\test_patch_evidence_router.py tests\test_teacher_probability_dataset.py tests\test_trainable_module_prefixes.py -q`
  (`14 passed`). Gate diagnostic on
  `runs\patch_evidence_mil_yolof_keeper_01only_t060_m040_top4_cropbbox_oofteacher_20260703\train\pair_verifier_teacher_oof.csv`
  with `min_pair_mass=0.20/min_confidence=0.60` kept `2270/2482`
  pair rows, with positive/negative targets `477/1793`.
- Smoke:
  `smoke_v8_yolof_patchrouter_oofteacher_conf60_pw25_w010_60b_1e_20260703`
  used explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, current keeper
  resume, 60 train batches / 1 epoch, full validation support `2606`, empty
  hard manifest, and `SkipFinalTest=true`. Recipe:
  `PatchEvidenceRouterHead=true/pair=0-1/top_k=4/bbox_weight=0.75/dropout=0.05/logit_scale=0.12/routing=true/route_margin=0.30/min_pair_prob=0.04/hard_router_loss=0.0/PatchEvidenceRouterTeacherLossWeight=0.010/min_confidence=0.60/min_pair_mass=0.20/teacher_positive_weight=2.5/full-model/LR=1e-5`.
  The teacher loss was active (`0.6914`) with sample fraction `0.3656`,
  pair fraction `0.4000`, pair mass `0.9997`, confidence `0.9718`, target mean
  `0.4653`, and positive target fraction `0.4766`, but router probability
  stayed almost exactly `0.5` (`0.49996`). Full-val macro/class-1 F1 was only
  `0.8769/0.6592`, below the keeper `0.8847/0.6860`. Class-1 P/R was
  `0.5735/0.7748`, with confusion `0:[481,56,3,0,9]`,
  `1:[16,117,12,0,6]`, `2:[4,13,514,6,7]`,
  `3:[0,3,57,652,0]`, `4:[9,15,4,2,620]`.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_patchrouter_oofteacher_conf60_pw25_w010_val_selected12_20260703`
  audited 12 selected validation cases. Top confusion remained
  `3->2=57`, `0->1=56`, `1->0=16`, `4->1=15`, `2->1=13`.
  Foreground focus stayed high (`attention=0.9815`,
  `grad_rollout=0.9766`, `gradcam=0.9786`, `rollout=0.9549`).
  Background blur/gray original-prediction drops were near zero
  (`0.0001/-0.0004`), while object desaturation was much larger (`0.0752`).
  Review flags remained boundary/detail dominated: `gradcam_border=6/12`,
  `rollout_border=5/12`, `object_color_sensitive=4/12`, and `near_tie=2/12`.
  Manual overlays showed a `1->0` case with Grad-CAM almost entirely on fruit
  surface, a `0->1` case with fruit plus small background-adjacent hotspots, and
  a `4->1` case where surface damage/edge cues and table artifacts were mixed.
  This does not support another context/background forcing attempt.
- Cleanup and decision: reject this OOF-router teacher recipe before any probe.
  Source metrics/config/history/logs and trace were copied into the XAI artifact
  under `source_smoke_metrics`, then the smoke train run was deleted via
  `runs\cleanup_manifest_20260703_patchrouter_oofteacher_rejected_smoke.json`,
  reclaiming `186.43 MB`. Keep the default-off teacher-target infrastructure
  because it is a valid reliability primitive, but do not repeat the exact
  low-LR full-model teacher-router recipe above. The target was active, but the
  selector stayed at probability `0.5` and validation class-1 F1 fell; the next
  patch-evidence attempt must change optimization/parameterization or use a
  stronger case-level conservative signal, not only another OOF teacher BCE.

## Smoke 2026-07-03 - OOF-Router Teacher Head-Only High-LR Reject

- Rationale: the full-model OOF-router teacher loss was active but barely moved
  the selector. I tested whether the current frozen representation can compress
  that OOF reliability target if only `patch_evidence_router_head` is trainable
  with a higher LR, analogous to a head-only/linear-probe diagnostic. This is
  intentionally different from the old hard-BCE router-only smoke: it uses the
  filtered OOF teacher target and keeps the backbone/head logits fixed.
- Smoke:
  `smoke_v8_yolof_patchrouter_oofteacher_routeronly_lr5e4_w020_24b_20260703`
  used explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, current keeper
  resume, `TrainableModulePrefixes=patch_evidence_router_head`, 24 train batches
  / 1 epoch, full validation support `2606`, empty hard manifest, and
  `SkipFinalTest=true`. Trainable params were only `166,920` router-head
  parameters; `7,245,590` params were frozen. Recipe:
  `PatchEvidenceRouterTeacherLossWeight=0.020/min_confidence=0.60/min_pair_mass=0.20/teacher_positive_weight=2.5/hard_router_loss=0/router-only/LR=5e-4`.
  The teacher target became non-trivial (`teacher_loss=0.6999`,
  teacher fraction `0.3581`, pair fraction `0.3997`, pair mass `0.9997`,
  confidence `0.9754`, target mean `0.4546`, positive fraction `0.4682`) and
  router probability moved to `0.4847`, but the decision boundary still worsened.
  Full-val macro/class-1 F1 was only `0.8796/0.6685`, below the keeper
  `0.8847/0.6860`. Class-1 P/R was `0.5769/0.7947`, with confusion
  `0:[482,55,3,0,9]`, `1:[14,120,11,0,6]`,
  `2:[3,12,516,6,7]`, `3:[0,4,58,648,2]`,
  `4:[8,17,3,1,621]`.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_patchrouter_oofteacher_routeronly_lr5e4_w020_val_selected12_20260703`
  audited 12 selected validation cases. Top confusions worsened to
  `3->2=58`, `0->1=57`, and `4->1=17`. Foreground focus became more diffuse
  than the full-model OOF teacher smoke: `attention=0.9300`,
  `grad_rollout=0.9583`, `gradcam=0.9638`, `rollout=0.9223`, with background
  masses `0.0700/0.0417/0.0362/0.0777`. Background blur/gray perturbations
  still stayed near zero (`0.0004/-0.0004`), while object desaturation remained
  much larger (`0.0763`). Review flags now included background attention
  (`attention/grad_rollout/gradcam/rollout background=1/12`) plus
  `gradcam_border=6/12`, `rollout_border=6/12`, `object_color_sensitive=4/12`.
  Manual overlay of a near-tie low-confidence case showed heat spreading over
  background/table/phone-shaped regions without perturbation support; the
  `4->1` case remained surface/edge + table artifact driven.
- Cleanup and decision: reject before any probe. Source metrics/config/history,
  logs, and trace were copied into the XAI artifact under
  `source_smoke_metrics`, then the train directory was deleted via
  `runs\cleanup_manifest_20260703_patchrouter_oofteacher_routeronly_rejected_smoke.json`,
  reclaiming `131.12 MB`. Do not repeat
  `PatchEvidenceRouterTeacherCsv=pair_verifier_teacher_oof.csv/min_conf=0.60/min_pair_mass=0.20/teacher_loss_weight=0.020/teacher_positive_weight=2.5/hard_router_loss=0/router-only/LR=5e-4/24b-120b`.
  Head-only optimization can move the selector, but it mostly broadens class-1
  false positives and diffuses XAI, so patch-router work is no longer the best
  next direction unless a new signal beyond the current OOF verifier appears.

## Diagnostic 2026-07-03 - AIDT/Top-5/TRKH Consensus Is Useful But Not A No-Pretrain Solution

- Rationale: after patch-router OOF signals failed, I inspected whether the
  pretrained AIDT/top-5 models provide a conservative case-level signal that
  specifically suppresses `0/4->1` false positives without sacrificing true
  class-1 recall. This was run as a validation-only diagnostic; no threshold was
  tuned on test and no raw dataset was modified.
- Artifact:
  `runs\aidt_top5_consensus_val_diagnostic_20260703` combines current TRKH
  validation predictions, mapped AIDT validation teacher probabilities, and the
  top-5 pretrained TTA validation cache. Base TRKH validation macro/class-1 F1
  was `0.8829/0.6783`; AIDT alone was `0.9088/0.7169`; top-5 TTA alone was
  `0.9139/0.7383`.
- Best validation-only consensus rule:
  `combined_consensus/sup_conf=0.70/sup_p1=0.15/res_conf=0.80/trkh_max_conf=0.30`.
  It changed `38` validation rows (`32` corrections, `6` harms) and reached
  macro/class-1 F1 `0.8985/0.7319`, class-1 P/R `0.6988/0.7682`, with
  confusion `0:[500,37,1,1,10]`, `1:[19,116,11,0,5]`,
  `2:[5,5,519,8,7]`, `3:[0,2,52,655,3]`, `4:[8,6,4,1,631]`.
  Most changes were `both_non1_suppress` (`33`) plus a small number of
  `both_1_rescue` (`5`).
- Decision: useful diagnostic, not a deployable no-pretrain result. It remains
  below the top-5 pretrained TTA validation class-1 F1 and depends on pretrained
  inference. The top-5 train teacher cache is also not a valid direct KD target:
  its train agreement with hard labels is `0.9998` with only `2` disagreements,
  so it would mostly reinforce hard labels in-sample rather than provide
  fold-safe conservative control. Do not repeat validation/test threshold
  routing around AIDT/top-5/TRKH consensus. If this direction continues, the
  signal must become fold-safe and trainable, or be used only as a representation
  diagnostic.

## Smoke 2026-07-03 - Symmetric Cross Entropy Robust Loss Reject

- Rationale: after patch-router and loss-level class-1 guards repeatedly widened
  `0/2/4->1` false positives, I tested a different noisy-label robust-loss idea:
  Symmetric Cross Entropy (SCE), which combines CE with a bounded reverse-CE term.
  I reviewed SCE (`https://arxiv.org/abs/1908.06112`) plus adjacent robust
  uncertainty/loss families such as evidential deep learning
  (`https://papers.nips.cc/paper/7580-evidential-deep-learning-to-quantify-classification-uncertainty`)
  and bi-tempered logistic loss (`https://arxiv.org/abs/1906.03361`). Because
  GCE/logit-norm/global damping had already failed, the first SCE smoke used a
  very light reverse term instead of the stronger original SCE default.
- Implementation and preflight: added default-off `SymmetricCrossEntropyLoss`
  with hard/soft target support, label smoothing, class-weight multipliers, and
  `per_sample_loss()` for the existing sample-weight/self-paced paths. Exposed
  `symmetric_ce_alpha/beta/epsilon` through `TrainConfig`, trainer CLI, V8
  launcher validation/resolved config, and trainer criterion logging. Preflight
  passed:
  `py_compile trkh\training\losses.py trkh\training\train.py trkh\core\config.py`,
  PowerShell parse of `scripts\run_trkh_5class_attention_views_v8.ps1`, and
  `pytest tests\test_symmetric_cross_entropy_loss.py tests\test_generalized_cross_entropy_loss.py tests\test_seesaw_loss.py tests\test_trainable_module_prefixes.py -q`
  (`14 passed`).
- Smoke:
  `smoke_v8_yolof_sce_a010_b010_teacherfocusbinary015_60b_1e_r3_20260703`
  used explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, current keeper
  resume, 60 train batches / 1 epoch, full validation support `2606`, empty hard
  manifest, and `SkipFinalTest=true`. Recipe:
  `ClassificationLoss=symmetric_cross_entropy/SymmetricCeAlpha=0.10/SymmetricCeBeta=0.10/SymmetricCeEpsilon=1e-4/teacher_focus_binary=0.015/BBoxSpatialFusion=true/LR=1e-5`.
  Full-val macro/class-1 F1 was only `0.8785/0.6648`, far below the keeper
  `0.8847/0.6860`. Class-1 P/R was `0.5749/0.7881`, with confusion
  `0:[481,56,3,0,9]`, `1:[14,119,12,0,6]`,
  `2:[3,13,515,6,7]`, `3:[0,3,58,650,1]`,
  `4:[9,16,3,1,621]`. The pattern is the familiar class-1 false-positive
  widening (`0->1=56`, `4->1=16`, `2->1=13`) rather than conservative
  correction.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_sce_a010_b010_teacherfocusbinary015_val_selected12_20260703`
  audited 12 selected validation cases. Top confusions were `3->2=58`,
  `0->1=55`, `4->1=16`, `1->0=14`, `2->1=13`. Foreground focus stayed high:
  `attention=0.9815`, `grad_rollout=0.9731`, `gradcam=0.9778`,
  `rollout=0.9534`. Background blur/gray original-prediction drops stayed near
  zero (`0.0004/0.0006`), while object desaturation was much larger (`0.0813`).
  Review flags remained surface/border dominated: `gradcam_border=7/12`,
  `rollout_border=5/12`, `object_color_sensitive=4/12`, `near_tie=2/12`.
- Cleanup and decision: reject before probe. Source metrics/config/history/logs
  were copied into the XAI artifact under `source_smoke_metrics`, then the smoke
  train run and failed launch logs were deleted via
  `runs\cleanup_manifest_20260703_sce_a010_b010_rejected_smoke.json`,
  reclaiming `182.49 MB`. Keep SCE default-off infrastructure for completeness,
  but do not repeat
  `ClassificationLoss=symmetric_cross_entropy/alpha=0.10/beta=0.10/epsilon=1e-4`
  on the current keeper; it behaves like the other global robust/loss-damping
  recipes and broadens class-1 false positives without adding a new
  surface/boundary reliability cue.

## Diagnostic 2026-07-03 - High-Resolution / Multi-Scale Validation Reject

- Rationale: because the remaining class-1 failures are surface/boundary details,
  I checked whether input scale rather than another loss could recover useful
  signal. I reviewed FixRes / train-test resolution work
  (`https://arxiv.org/abs/1906.06423`, `https://arxiv.org/abs/2003.08237`),
  implementation trick audits (`https://arxiv.org/abs/1812.01187`), and
  multi-scale FGVC references such as MMAL-Net
  (`https://arxiv.org/abs/2003.09150`). Raw data and test split were untouched.
- Single-scale diagnostic:
  `runs\eval_yolof_teacherfocusbinary015_best_val_override320_20260703` evaluated
  the current keeper on `yolo_f/val` with `--override-image-size 320`,
  positional interpolation, crop-bbox token prior, batch `32`, and full support
  `2606`. It reduced `0->1` false positives from `50` to `45`, but class-1
  recall fell from `117/151` to `113/151`; macro/class-1 F1 became only
  `0.8805/0.6667`. Confusion was
  `[[491,45,3,0,10],[20,113,12,0,6],[4,10,518,7,5],[0,4,55,653,0],[8,16,7,1,618]]`.
- Multi-scale probability diagnostic:
  `runs\diagnostic_yolof_multiscale_256_320_val_20260703` blended the existing
  256 validation probabilities with the 320 probabilities. Best macro was
  `w320=0.70`, macro/class-1 F1 `0.8859/0.6806`; best class-1 was `w320=0.80`,
  macro/class-1 F1 `0.8858/0.6826`, P/R `0.6230/0.7550`, with
  `0->1=44`, `4->1=12`, `1->0=20`, `1->2=12`. This is a small macro bump over
  the same evaluator baseline but still below the current keeper gate and below
  the desired class-1 trajectory.
- Multi-scale plus the frozen train-only `0-1` patch-evidence verifier was also
  checked without new threshold tuning:
  `runs\diagnostic_yolof_multiscale320_patchverifier_val_20260703` used
  `w320=0.80`, pair `0-1`, min pair probability `0.02`, max margin `0.40`, and
  verifier confidence `0.60`. It reached validation macro/class-1 F1
  `0.8895/0.7015` with `27` changes (`13` corrections, `9` harms,
  `5` neutral), but that is still below the older yolo-only patch-verifier
  validation `0.8892/0.7073`. Because it underperforms an already frozen
  validation diagnostic, I did not spend test budget on this variant.
- Cleanup and decision: the raw override-320 eval directory was compacted into
  the multi-scale diagnostic artifact and deleted via
  `runs\cleanup_manifest_20260703_override320_rejected_eval.json`, reclaiming
  `2.99 MB`. Do not launch high-resolution train/probe or multi-scale+patch test
  from this evidence. Resolution changes can reduce some `0->1` errors, but they
  increase true class-1 misses and do not create the missing surface/boundary
  reliability signal.

## Smoke 2026-07-03 - FixRes-Style 320 Head/Position Adaptation Reject

- Rationale: after plain 320 inference and simple 256+320 probability blending
  failed, I tested the only resolution revisit still allowed by the current
  evidence: a FixRes-style short adaptation at image size `320`. To avoid
  destroying the current keeper, only `head,pos_embed,cls_token,register_tokens,
  bbox_spatial_fusion_head` were trainable (`107,822` params); the transformer
  backbone and CNN stem stayed frozen. This used explicit `yolo_f/data.yaml`,
  current no-pretrain keeper resume, `BBoxSpatialFusion=true`,
  `BboxTokenPriorSource=crop_bbox`, teacher-focus-binary `0.015`,
  boundary-band bbox dropout, `MaxTrainBatches=60`, full validation support
  `2606`, `MaxValBatches=0`, and `SkipFinalTest=true`. The hard-sample manifest
  was verified empty in `resolved_config.json`.
- Preflight and performance: `py_compile train.py/model.py/config.py`,
  `pytest tests\test_trainable_module_prefixes.py -q`, and PowerShell parse of
  `scripts\run_trkh_5class_attention_views_v8.ps1` passed. Training at `320`
  with batch `32` ran about `1.7-2.2` batches/s; the epoch took `174.4s`,
  with train `54.7s`, validation `118.0s`, and peak reserved GPU memory about
  `6.3GB`.
- Result:
  `smoke_v8_yolof_fixres320_headpos_bboxfusion_60b_1e_20260703` reached only
  validation macro/class-1 F1 `0.8780/0.6667`, below the keeper
  `0.8847/0.6860` and below the `0.70` class-1 probe gate. Class-1 P/R was
  `0.5850/0.7748`. Confusion was
  `[[488,49,3,0,9],[16,117,14,0,4],[3,9,522,5,5],
  [0,4,63,645,0],[8,21,8,2,611]]`. The adaptation did not protect recall
  failures (`1->2` rose to `14`) and worsened class-1 false positives from class
  `4` (`4->1=21`).
- XAI/audit:
  `runs\xai_smoke_v8_yolof_fixres320_headpos_bboxfusion_val_selected12_20260703`
  audited 12 selected validation cases. Top confusions were `3->2=63`,
  `0->1=49`, `4->1=21`, `1->0=16`, and `1->2=14`. Foreground focus stayed high
  (`attention/grad_rollout/gradcam/rollout` foreground
  `0.9173/0.9589/0.9663/0.9269`), background blur/gray drops stayed near zero
  (`0.0002/0.0003`), and object desaturation stayed much larger (`0.0937`).
  Review flags remained surface/border dominated: `gradcam_border=6/12`,
  `rollout_border=5/12`, `object_color_sensitive=4/12`, and `near_tie=2/12`.
- Cleanup and decision: reject before probe. Source metrics/config/history/logs
  and architecture trace were copied into the XAI artifact under
  `source_smoke_metrics`, then the smoke train directory was deleted via
  `runs\cleanup_manifest_20260703_fixres320_headpos_bboxfusion_rejected_smoke.json`,
  reclaiming `130.41 MB`. Do not repeat FixRes-style 320 head/pos/bbox-fusion
  adaptation on the current keeper; it worsens the same class-1 false-positive
  and recall boundary errors.

## Smoke 2026-07-03 - Natural-Prior cRT Head/Pair Adaptation Reject

- Rationale: I revisited classifier-only adaptation from a different angle after
  reading cRT / decoupled representation-classifier training
  (`https://arxiv.org/abs/1910.09217`) and recent logits-retargeting discussion
  (`https://arxiv.org/abs/2403.00250`). The intent was to keep the current
  representation fixed and retrain only the classifier/pair/bbox-fusion heads on
  the natural train prior, testing whether a calibrated head can reduce class-1
  false positives without another representation drift. Raw datasets and test
  split were untouched.
- Smoke:
  `smoke_v8_yolof_crt_natural_head_pair_bbox_lr5e4_60b_1e_20260703` resumed the
  current no-pretrain keeper, used explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, `ImageSize=256`,
  `MaxTrainBatches=60`, full validation support `2606`, `MaxValBatches=0`,
  `SkipFinalTest`, empty hard-sample manifest, and
  `DisableBalancedEpochSampling=true`. Only
  `head,pairwise_margin_norm,pairwise_margin_head,bbox_spatial_fusion_head,
  cnn_fusion_norm,cnn_fusion_head` were trainable (`6,199` params), with
  `LR=5e-4`, `MetricLearningLossWeight=0`, `BBoxSpatialFusion=true`,
  `BboxTokenPriorSource=crop_bbox`, and teacher-focus-binary `0.015`.
- Result: full-val macro/class-1 F1 was only `0.8815/0.6686`, below the keeper
  `0.8847/0.6860` and below the no-balanced full-model continuation reference.
  Class-1 P/R was `0.6000/0.7550`; confusion was
  `[[488,49,3,0,9],[19,114,12,0,6],[3,9,519,6,7],
  [0,4,55,651,2],[9,14,3,1,623]]`. The head became slightly more conservative
  than the FixRes 320 smoke on `4->1`, but it lost too many true class-1 cases
  (`1->0=19`, `1->2=12`) and still left `0->1=49`.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_crt_natural_head_pair_bbox_val_selected12_20260703`
  audited 12 selected validation cases. Top confusions were `3->2=55`,
  `0->1=48`, `1->0=18`, `4->1=14`, and `1->2=12`. Foreground focus stayed high
  (`attention/grad_rollout/gradcam/rollout` foreground
  `0.9774/0.9735/0.9776/0.9518`), background blur/gray drops stayed near zero
  (`0.0006/0.0006`), and object desaturation stayed much larger (`0.0777`).
  Review flags remained the usual surface/border failures:
  `gradcam_border=7/12`, `rollout_border=6/12`, `object_color_sensitive=4/12`,
  and `near_tie=1/12`.
- Cleanup and decision: reject before probe. Source metrics/config/history/logs
  and architecture trace were copied into the XAI artifact under
  `source_smoke_metrics`, then the smoke train directory was deleted via
  `runs\cleanup_manifest_20260703_crt_natural_head_pair_bbox_rejected_smoke.json`,
  reclaiming `127.26 MB`. Do not repeat this natural-prior cRT recipe on the
  current keeper; it does not add the missing surface/boundary reliability cue.
  If logits retargeting is explored, it must be a materially different
  classifier-training objective with explicit class-1 recall protection and a
  smoke gate, not another natural-prior head-only rerun.

## Smoke 2026-07-03 - LOS / Label-Over-Smooth cRT Reject

- Rationale: after the natural-prior cRT smoke failed, I checked the official
  LOS/LORT code path (`https://github.com/Thinklab-SJTU/LOS`) and the paper
  (`https://arxiv.org/abs/2403.00250`). Their stage-2 classifier retraining uses
  classifier-only cross entropy with very high label smoothing (`label_smooth
  0.98`), so I used the existing `ClassificationLoss=cross_entropy` and
  `LabelSmoothing=0.98` path instead of adding duplicate loss code. This was a
  targeted classifier-only smoke; raw datasets and test split were untouched.
- Smoke:
  `smoke_v8_yolof_los098_crt_head_pair_bbox_lr5e4_60b_1e_20260703` resumed the
  current no-pretrain keeper, used explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, full validation support `2606`,
  `MaxTrainBatches=60`, `MaxValBatches=0`, `SkipFinalTest`, empty hard-sample
  manifest, `DisableBalancedEpochSampling=true`, `MetricLearningLossWeight=0`,
  `TeacherFocusBinaryLossWeight=0`, `BBoxSpatialFusion=true`,
  `BboxTokenPriorSource=crop_bbox`, and only
  `head,pairwise_margin_norm,pairwise_margin_head,bbox_spatial_fusion_head,
  cnn_fusion_norm,cnn_fusion_head` trainable (`6,199` params). Preflight had
  passed before launch: `py_compile`, PowerShell launcher parse, and focused
  pytest (`11 passed`).
- Result: full-val macro/class-1 F1 was only `0.8800/0.6685`, below the keeper
  `0.8847/0.6860` and nearly identical to the failed cRT family. Class-1 P/R was
  `0.5805/0.7881`; confusion was
  `[[481,55,4,0,9],[15,119,11,0,6],[3,12,516,6,7],
  [0,4,56,650,2],[8,15,3,1,623]]`. The over-smooth head increased class-1
  recall relative to natural-prior cRT, but it did so by widening false positives
  (`0->1=55`, `2->1=12`, `4->1=15`) and did not approach the `0.70` class-1
  probe gate.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_los098_crt_head_pair_bbox_val_selected12_20260703`
  audited 12 selected validation cases. Top confusions were `3->2=56`,
  `0->1=54`, `1->0=15`, `4->1=15`, `2->1=12`, and `1->2=11`. Foreground focus
  stayed high (`attention/grad_rollout/gradcam/rollout` foreground
  `0.9806/0.9744/0.9774/0.9519`), background blur/gray drops stayed near zero
  (`0.0002/-0.0001`), and object desaturation was much larger (`0.0733`).
  Review flags remained surface/border dominated: `gradcam_border=7/12`,
  `rollout_border=4/12`, `object_color_sensitive=3/12`, and `near_tie=2/12`.
- Cleanup and decision: reject before probe. Source metrics/config/history/logs
  and architecture trace were copied into the XAI artifact under
  `source_smoke_metrics`, then the smoke train directory was deleted via
  `runs\cleanup_manifest_20260703_los098_crt_head_pair_bbox_rejected_smoke.json`,
  reclaiming `127.27 MB`. Do not repeat
  `ClassificationLoss=cross_entropy/LabelSmoothing=0.98` with this cRT head-only
  recipe on the current keeper; LOS-style over-smoothing behaves like the other
  global softening losses here and expands class-1 false positives rather than
  adding a new surface/boundary reliability cue.

## Diagnostic 2026-07-03 - Relative Mahalanobis / Class-1 Conformity Reject

- Rationale: after classifier-only cRT/LOS failed, I moved to a no-training
  diagnostic based on representation nonconformity. The primary references were
  DkNN / train-manifold nonconformity (`https://arxiv.org/abs/1803.04765`),
  selective classification / risk control (`https://arxiv.org/abs/1705.08500`),
  and Mahalanobis representation scoring (`https://arxiv.org/abs/2003.00402`).
  This did not edit raw data and did not read the test split.
- Implementation:
  added `trkh.tools.probe_relative_mahalanobis`, default-off. It extracts frozen
  train/val head embeddings from the current keeper, fits train-only class means,
  shared diagonal variance, class diagonal variance, and Ledoit-Wolf shared
  covariance on standardized and L2-standardized embeddings. The validation rule
  is threshold-free: keep the base prediction except when the base predicts
  class `1` and the Mahalanobis-nearest class is not `1`; variants either switch
  to any nearest rival or only switch `1 -> 0`. Preflight passed:
  `py_compile trkh\tools\probe_relative_mahalanobis.py trkh\tools\probe_embedding_prototypes.py`
  and `pytest tests\test_probe_relative_mahalanobis.py tests\test_probe_embedding_prototypes.py -q`
  (`7 passed`).
- Probe:
  `runs\relative_mahalanobis_yolof_keeper_valdiag_20260703` used explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, current no-pretrain keeper,
  train support `9215`, validation support `2606`, batch `128`, and no test.
  Train looked superficially better: best guard
  `guard1_any_ledoit_shared_l2std` improved train macro/class-1 from
  `0.9398/0.8157` to `0.9459/0.8354`, changing `88` train rows
  (`62` corrections, `24` harms, `2` neutral). This did not generalize.
- Validation result:
  best validation guard was only `guard1_to0_diag_shared_std`, which changed
  two rows and fell slightly below the keeper: base `0.8847/0.6860` became
  `0.8843/0.6842`, class-1 P/R `0.6126/0.7748`, with one correction
  (`0:1->0`, sample `1821`, `Image_6029`) and one harm (`1:1->0`, sample
  `1822`, `Image_6038`). More aggressive variants harmed recall harder:
  `guard1_any_ledoit_shared_l2std` changed `27` rows and fell to
  macro/class-1 `0.8818/0.6688`.
- XAI/audit:
  `runs\xai_relative_mahalanobis_yolof_keeper_guard1to0_diagshared_val_changed2_20260703`
  audited the two changed validation rows. Foreground focus was high
  (`attention/grad_rollout/gradcam/rollout` foreground
  `0.9855/0.9903/0.9770/0.9846`), background blur/gray drops stayed near zero
  (`0.0003/-0.0028` original-prediction drop), and object desaturation was larger
  (`0.0327` original-prediction drop, `0.0493` target drop). Both cases had
  Grad-CAM border flags, both had rollout border flags, and one was a near-tie.
  The harm sample is a true class-1 boundary case that the distance guard cannot
  distinguish from false-positive class-1 surface/border ambiguity.
- Cleanup and decision: reject as a current-keeper solution. The source summary,
  selected changed cases, and validation prediction CSV were copied into the XAI
  artifact under `source_probe_metrics`, then the probe directory was deleted via
  `runs\cleanup_manifest_20260703_relative_mahalanobis_rejected_probe.json`,
  reclaiming `2.87 MB`. Keep the diagnostic tool for future representation
  audits, but do not repeat the exact train-only relative-Mahalanobis class-1
  guard on this keeper; feature-distance uncertainty alone is not the missing
  recall-safe surface/boundary cue.

## Diagnostic 2026-07-04 - Fixed-ETF / Neural-Collapse Ridge Prototype Reject

- Rationale: after Mahalanobis nearest-manifold scoring failed, I checked a
  different representation geometry idea before changing the train loop. The
  relevant primary references were fixed ETF classifiers for neural collapse
  (`https://papers.neurips.cc/paper_files/paper/2022/file/f7f5f501282771c96bb3fedcc96bedfe-Paper-Conference.pdf`)
  and neural-collapse-inspired long-tailed learning
  (`https://arxiv.org/pdf/2203.09081`). The diagnostic asks whether the current
  frozen embedding can be mapped to a simplex ETF target with a train-only ridge
  projection, avoiding validation/test threshold tuning.
- Implementation:
  extended `trkh.tools.probe_embedding_prototypes` with `_simplex_etf_targets`
  and `_fit_predict_etf_ridge`, plus balanced sample-weight support. Preflight
  passed `py_compile trkh\tools\probe_embedding_prototypes.py` and
  `pytest tests\test_probe_embedding_prototypes.py -q` (`5 passed`).
- Probe:
  `runs\embedding_etf_probe_yolof_keeper_20260704` used explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, the current no-pretrain keeper,
  train support `9215`, validation support `2606`, no test, `--skip-logistic`,
  `knn-k=5`, and ETF ridge alphas `0.1/1/10/100`. The keeper remained best:
  base validation macro/class-1 was `0.8847/0.6860`. Best ETF macro was
  `etf_ridge_a100` at only `0.8749/0.6400`, class-1 P/R `0.7097/0.5828`.
  Best balanced ETF class-1 was `etf_ridge_balanced_a10` at `0.8707/0.6556`,
  class-1 P/R `0.5646/0.7815`. Both are below the existing frozen logistic and
  below the keeper.
- Change analysis:
  `etf_ridge_a100` changed `127` validation rows (`60` corrections, `61` harms,
  `6` neutral). It suppressed some false positives (`0:1->0=31`,
  `4:1->0=3`, `2:1->2=6`) but over-suppressed true class 1
  (`1:1->0=20`, `1:1->2=12`, `1:1->4=2`). This is the same precision/recall
  tradeoff seen in verifier and Mahalanobis suppressors.
- XAI/audit:
  `runs\xai_embedding_etf_ridge_a100_yolof_keeper_val_changed12_20260704`
  audited harm-first changed rows. Most selected rows were true class-1 cases
  that the base model got right but ETF would move away from class 1. Foreground
  mass remained high (`attention/grad_rollout/gradcam/rollout`
  `0.9651/0.9718/0.9499/0.9562`), background blur/gray drops stayed near zero
  (`0.0005/-0.0013`), and review flags were border-heavy (`gradcam_border=9/12`,
  `rollout_border=7/12`, `near_tie=3/12`). The fixed simplex projection is
  mostly changing already-fragile low-margin class-1 surface/border decisions.
- Cleanup and decision: reject before any train-loop implementation. Source
  summary, validation predictions, and changed-case CSV were copied under the XAI
  artifact's `source_probe_metrics`, then the probe directory was deleted via
  `runs\cleanup_manifest_20260704_embedding_etf_rejected_probe.json`,
  reclaiming `1.17 MB`. Do not implement fixed-ETF/cosine classifier training on
  this frozen representation unless a new recall-protecting representation
  signal appears first.

## Smoke 2026-07-04 - EfficientNetV2-S Feature CRD Adapter Reject

- Rationale: the post-hoc top-5/AIDT threshold and frozen-embedding geometry
  routes were exhausted, so I tested a representation-transfer route instead of
  another logit rule. The primary references were FitNets hint training
  (`https://arxiv.org/pdf/1412.6550`), relational knowledge distillation
  (`https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.pdf`),
  and contrastive representation distillation. This attempt used the stronger
  EfficientNetV2-S `class_f` retrieval feature cache, not the already-rejected
  AIDT feature cache and not the in-sample top-5 train logits.
- Feature remap:
  `trkh.tools.remap_classification_features_to_yolo` strictly remapped
  `runs\embedding_retrieval_efficientnetv2_s_ft_valonly_20260627` features into
  `yolo_f` sample-index order. Train and validation both had full coverage:
  `9215/9215` train and `2606/2606` val, `feature_dim=1280`, no missing rows,
  no duplicate sample-index rows. The remapped cache is kept at
  `runs\efficientnetv2_s_features_yolof_train_20260704`.
- Preflight:
  `py_compile trkh\training\train.py trkh\data\dataset.py
  trkh\models\model.py trkh\core\config.py
  trkh\tools\remap_classification_features_to_yolo.py` passed, and
  `pytest tests\test_teacher_feature_rkd.py
  tests\test_trainable_module_prefixes.py -q` passed (`13 passed`). The V8
  dry-run confirmed explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`,
  `MaxValBatches=0`, `SkipFinalTest=true`, empty hard manifest,
  `distillation_weight=0.0`, `BBoxSpatialFusion=true`, teacher-focus-binary
  `0.015`, and learnable feature projection adapter enabled.
- Smoke:
  `runs\smoke_v8_yolof_effv2feature_crd_adapter_w002_60b_1e_20260704`
  used the current no-pretrain keeper checkpoint, EfficientNetV2-S features,
  `teacher_feature_contrastive_loss_weight=0.002`, `temperature=0.16`,
  projection adapter `128` with dropout `0.02`, `source=head`,
  `pair_mode=boundary_or_intra`, pairs `0-1,1-2,1-4`, `60` train batches,
  one epoch, full validation support `2606`, and no final test. The contrastive
  telemetry was active but weak: loss `2.6476`, positive similarity `0.0258`,
  negative similarity `0.0413`, and top-1 `0.0844`, so the adapter did not learn
  useful teacher-student separation in the smoke window.
- Validation result:
  full-val macro/class-1 F1 was `0.8837/0.6802`, below the keeper
  `0.8847/0.6860` and below the `0.70` class-1 probe gate. Class-1 P/R was
  `0.6062/0.7748`. Top confusions remained `3->2=52`, `0->1=50`,
  `1->0=18`, `1->2=12`, and `4->1=12`; this is not a useful move relative to
  the anchor.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_effv2feature_crd_adapter_w002_val_selected12_20260704`
  audited 12 selected validation mistakes with `method=all` and robustness
  probes. Foreground focus stayed high (`attention/grad_rollout/gradcam/rollout`
  `0.9832/0.9685/0.9761/0.9464`), background blur/gray drops stayed near zero
  (`0.0005/-0.0002` original-prediction drop), and object desaturation stayed
  much larger (`0.0886`). Review flags were unchanged: `gradcam_border=7/12`,
  `rollout_border=5/12`, `object_color_sensitive=4/12`, plus one Grad-CAM
  background flag. The model is still solving or failing on fruit surface/border
  cues, not on missing wide background context.
- Cleanup and decision: reject before probe. Source metrics/config/history/logs
  and architecture trace were copied into the XAI artifact under
  `source_smoke_metrics`, then the smoke train directory and launch logs were
  deleted via
  `runs\cleanup_manifest_20260704_effv2feature_crd_adapter_rejected_smoke.json`,
  reclaiming `187.14 MB`; D: free space after cleanup was about `73.75 GB`.
  Do not repeat this exact EfficientNetV2-S feature CRD adapter recipe on the
  current keeper. A stronger pretrained retrieval feature does not transfer with
  a light boundary/intra CRD adapter here; future feature transfer needs a
  genuinely different fold-safe reliability, recall-protecting case signal, or
  batch/objective design, not another nearby adapter weight/temp tweak.

## Diagnostic 2026-07-04 - Entropy Test-Time Adaptation Reject

- Rationale: after train-time feature transfer failed, I checked whether a
  label-free inference-time adaptation could improve the current keeper without
  changing raw data or spending test budget. The relevant primary references
  were TENT entropy minimization (`https://arxiv.org/abs/2006.10726`), MEMO
  test-time adaptation with augmentation (`https://arxiv.org/abs/2110.09506`),
  and T3A classifier adjustment
  (`https://proceedings.neurips.cc/paper_files/paper/2021/file/1415fe9fea0fa1e45dddcff5682239a0-Paper.pdf`).
  This local diagnostic used validation only and did not read test.
- Implementation:
  `trkh.tools.evaluate_test_time_adaptation` already supported label-free
  entropy adaptation over LayerNorm/head parameters. I added
  `--save-adapted-checkpoint`, which writes `adapted_checkpoint.pt` for XAI after
  adaptation while dropping stale optimizer/scheduler/EMA states. Focused
  preflight passed: `py_compile trkh\tools\evaluate_test_time_adaptation.py` and
  `pytest tests\test_test_time_adaptation.py -q` (`3 passed`).
- Diagnostic:
  `runs\tta_entropy_yolof_keeper_layernorm_headbias_lr1e5_conf30_frac50_val_ckpt_20260704`
  used the current keeper, explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, split `val`, full support
  `2606`, batch `64`, `adapt_parameters=layernorm_head_bias`, LR `1e-5`,
  one step per batch, `selection_fraction=0.50`, `min_confidence=0.30`, and
  `bbox_token_prior_source=crop_bbox`. The adapter updated `13,317` parameters,
  selected `915/2606` validation samples (`0.3511`), and made `41` optimizer
  steps with mean entropy loss `1.5427`.
- Validation result:
  macro/class-1 F1 was `0.8841/0.6822`, below the keeper `0.8847/0.6860` and
  below the `0.70` gate. Class-1 P/R was `0.6094/0.7748`. Top confusions stayed
  effectively unchanged: `3->2=52`, `0->1=49`, `1->0=18`, `4->1=12`,
  `0->4=10/11` depending on ordering, and `1->2=11`.
- Change analysis:
  compared against the keeper's `best_val_eval_predictions\predictions_detailed.csv`,
  TTA changed only `3` validation rows: corrections `0:1->0` and `0:4->0`, plus
  one harm `1:1->0`. The net accuracy change is small, but the harm is exactly a
  true class-1 recall loss, so this does not solve the target class.
- XAI/audit:
  `runs\xai_tta_entropy_yolof_keeper_layernorm_headbias_lr1e5_conf30_frac50_val_selected12_20260704`
  audited the adapted checkpoint on 12 selected validation mistakes. Foreground
  focus stayed high (`attention/grad_rollout/gradcam/rollout`
  `0.9778/0.9697/0.9799/0.9440`), background blur/gray drops stayed near zero
  (`0.0006/-0.0006` original-prediction drop), and object desaturation remained
  much larger (`0.0726`). Review flags were still border/detail driven:
  `gradcam_border=6/12`, `rollout_border=6/12`,
  `object_color_sensitive=3/12`. Entropy adaptation slightly shifts margins but
  does not create a new surface/boundary reliability signal.
- Cleanup and decision: reject this TTA configuration as a current-keeper route.
  Metrics, setup, predictions, changed-row CSV, and the adapted checkpoint were
  copied into the XAI artifact under `source_tta_metrics`, then both TTA
  diagnostic run directories were deleted via
  `runs\cleanup_manifest_20260704_tta_entropy_rejected_diagnostic.json`,
  reclaiming `32.86 MB`. Keep the checkpoint-save tooling for future audits, but
  do not repeat `layernorm_head_bias/LR=1e-5/selection_fraction=0.50/
  min_confidence=0.30/one-step entropy` on the current keeper; it protects neither
  class-1 recall nor class-1 precision enough to justify more tuning.

## Diagnostic 2026-07-04 - T3A Target-Template Classifier Adjustment Reject

- Rationale: after plain entropy TTA failed, I tested the T3A idea from the
  same test-time adaptation literature, but as a validation-only diagnostic on
  frozen embeddings. This asks whether low-entropy target validation supports
  can adapt the classifier template without changing raw data, training on test,
  or spending a final-test budget.
- Implementation:
  `trkh.tools.probe_embedding_prototypes` now supports T3A-style source
  classifier templates from the checkpoint head plus normalized low-entropy
  target support means. I added `--t3a-support-per-class`,
  `--t3a-source-weight`, and `--t3a-min-confidence`, with tests for target
  support use and source-template fallback. Focused preflight passed:
  `py_compile trkh\tools\probe_embedding_prototypes.py` and
  `pytest tests\test_probe_embedding_prototypes.py -q` (`7 passed`).
- Diagnostic:
  `runs\embedding_t3a_probe_yolof_keeper_val_20260704` used explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, the current no-pretrain keeper,
  train support `9215`, validation support `2606`, no test, `knn-k=3`, no
  logistic/ETF fitting, support-per-class `16/32/64`, source weight `1/8/32`,
  and min confidence `0.0`. The base head stayed at validation macro/class-1
  F1 `0.8847/0.6860`. Every T3A variant was lower: the best T3A macro and
  class-1 variant was `t3a_k32_sw32_c0` at only `0.8649/0.6322`, class-1 P/R
  `0.5584/0.7285`.
- Change analysis:
  the best T3A variant changed `89` validation rows but had only `20`
  corrections against `56` harms and `13` neutral wrong-to-wrong changes. It
  harmed the exact target behavior: true class-1 rows that the keeper got right
  moved to `0/2/4` (`1:1->0=4`, `1:1->2=3`, `1:1->4=3`), while it also added
  non-class-1 false positives such as `2:2->1=9` and `0:0->1=8`.
- XAI/audit:
  `runs\xai_embedding_t3a_k32_sw32_yolof_keeper_val_changed12_20260704`
  audited the harm-first changed rows using the keeper checkpoint. Foreground
  focus remained high (`attention/grad_rollout/gradcam/rollout`
  `0.9823/0.9830/0.9821/0.9629`), background blur/gray drops stayed near zero
  (`0.0008/0.0021` original-prediction drop), and object desaturation was still
  larger (`0.0403`). Review flags showed the same fragile near-tie/border
  pattern: `near_tie=6/12`, `rollout_border=9/12`, `gradcam_border=4/12`.
  T3A is drifting templates on the current frozen embedding; it is not learning
  a better background/object separation or a new surface-boundary cue.
- Cleanup and decision: reject T3A for the current keeper. Source summary,
  validation predictions, and changed-case CSV were copied into the XAI artifact
  under `source_probe_metrics`, then the probe directory was deleted via
  `runs\cleanup_manifest_20260704_embedding_t3a_rejected_probe.json`,
  reclaiming `1.21 MB`; D: free space after cleanup was about `73.58 GB`.
  Keep the T3A tooling as a diagnostic, but do not repeat the
  `support_per_class=16/32/64`, `source_weight=1/8/32`, `min_confidence=0.0`
  sweep on this frozen representation. Future TTA/prototype work must add
  explicit class-1 recall protection or a new representation signal first.

## Smoke 2026-07-04 - HERBS-Style Multi-Granularity High-Temperature Refinement Reject

- Rationale: after TTA/prototype and feature-transfer routes failed, I checked a
  fine-grained multi-scale training idea rather than another post-hoc decision
  rule. The primary reference was HERBS, "Fine-grained Visual Classification
  with High-temperature Refinement and Background Suppression"
  (`https://arxiv.org/abs/2303.06442`). The repo already had
  multi-granularity auxiliary heads, but the prior CE and contrastive variants
  were rejected. This attempt added a lighter high-temperature KL refinement
  from the final head to intermediate heads, default-off, without changing raw
  data.
- Implementation:
  added `multi_granularity_refinement_loss_weight` and
  `multi_granularity_refinement_temperature` to config/CLI/V8 launcher, plus
  `_multi_granularity_refinement_loss_from_features`. The loss distills detached
  final logits into intermediate aux logits with high-temperature KL and records
  loss/count/teacher entropy in history. Focused preflight passed:
  `py_compile trkh\training\train.py trkh\core\config.py`, launcher dry-run with
  explicit `yolo_f`, and `pytest tests\test_multi_granularity_aux_heads.py -q`
  (`9 passed`).
- Smoke:
  `runs\smoke_v8_yolof_multigran_htrefine_w006_t32_60b_1e_20260704` used the
  current keeper checkpoint, explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`,
  empty hard manifest, no final test, full validation support `2606`, 60 train
  batches, and the keeper recipe (`BBoxSpatialFusion=true`, `crop_bbox`,
  teacher-focus-binary `0.015`, metric loss `0.04`, pairwise margin routing,
  bbox boundary-band dropout). The new refinement path was active:
  `train_multi_granularity_refinement_loss=0.0798`, `count=3`, teacher entropy
  `1.6094`.
- Validation result:
  macro/class-1 F1 was only `0.8837/0.6802`, below the keeper
  `0.8847/0.6860` and below the `0.70` class-1 gate. Class-1 P/R was
  `0.6062/0.7748`. The top confusions stayed the same: `3->2=52`, `0->1=50`,
  `1->0=18`, `4->1=12`, and `1->2=11`. This is not enough for a probe.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_multigran_htrefine_w006_t32_val_selected12_20260704`
  audited 12 selected validation mistakes. Foreground focus stayed high
  (`attention/grad_rollout/gradcam/rollout`
  `0.9820/0.9737/0.9865/0.9506`), background blur/gray drops stayed near zero
  (`0.0006/-0.0003` original-prediction drop), and object desaturation stayed
  larger (`0.0775`). Review flags were still border/detail driven:
  `gradcam_border=6/12`, `rollout_border=6/12`, `object_color_sensitive=3/12`.
  A visual check of the first `0->1` case showed heat on the fruit surface/border
  plus nearby background clutter, matching the old failure mode.
- Cleanup and decision: reject before probe. Metrics/config/history/logs and
  architecture trace were copied into the XAI artifact under
  `source_smoke_metrics`, then the smoke train directory was deleted via
  `runs\cleanup_manifest_20260704_multigran_htrefine_rejected_smoke.json`,
  reclaiming `182.74 MB`; D: free space after cleanup was about `73.51 GB`.
  Keep the default-off implementation and tests, but do not repeat
  `MultiGranularityAuxHeads=true` with `MultiGranularityRefinementLossWeight=0.006`
  and `Temperature=32` on the current keeper. It activates, but it just regularizes
  old intermediate features and does not add the missing class-1 surface/boundary
  reliability signal.

## Smoke 2026-07-04 - Self-Boosting CAM Attention Reject

- Rationale: after several attention/background routes failed, I tested a
  lighter self-boosting attention idea from the ECCV/arXiv self-boosting
  attention mechanism work (`https://arxiv.org/abs/2208.00617`,
  `https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850444.pdf`).
  The implementation uses the current classifier head to form detached
  hard-class token CAM pseudo-targets, then trains a class-agnostic token
  projection with KL. It does not change raw data and keeps inference behavior
  unchanged unless the optional head is enabled.
- Implementation:
  added default-off `self_boosting_attention_head`,
  `self_boosting_attention_loss_weight`, temperature/classes/start-epoch config
  and CLI/V8 launcher support. The loader can resume old checkpoints while
  allowing the new projection parameters. Focused preflight passed:
  `py_compile trkh\training\train.py trkh\models\model.py trkh\core\config.py`,
  PowerShell launcher parse/dry-run, and `pytest
  tests\test_self_boosting_attention_loss.py
  tests\test_multi_granularity_aux_heads.py -q` (`13 passed`).
- Smoke:
  `runs\smoke_v8_yolof_selfboostcam_w006_t040_60b_1e_20260704` used the
  current keeper checkpoint, explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`,
  empty hard manifest, no final test, full validation support `2606`, 60 train
  batches, and the keeper recipe (`BBoxSpatialFusion=true`, `crop_bbox`,
  teacher-focus-binary `0.015`, metric loss `0.04`, pairwise routing,
  boundary-band dropout, attention-view loss `0.35`). The self-boosting loss was
  active at `0.006`, temperature `0.40`, with `valid_fraction=1.0`,
  target entropy `4.8091`, target peak `0.0259`, projection peak `0.0239`, and
  map similarity `0.5602`. The high entropy means the pseudo-target was still
  broad over patches rather than a sharp new defect/surface signal.
- Validation result:
  macro/class-1 F1 was only `0.8834/0.6802`, below the keeper
  `0.8847/0.6860` and below the `0.70` class-1 gate. Class-1 P/R was
  `0.6062/0.7748`. Top confusions remained `3->2=52`, `0->1=50`,
  `1->0=18`, `4->1=12`, and `1->2=11`; this is not probe-worthy.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_selfboostcam_w006_t040_val_selected12_20260704`
  audited 12 selected validation mistakes with all methods and robustness probes.
  Foreground focus stayed high (`attention/grad_rollout/gradcam/rollout`
  `0.9832/0.9686/0.9762/0.9463`), background blur/gray drops stayed near zero
  (`0.0005/-0.0005` original-prediction drop), and object desaturation stayed
  much larger (`0.0887`). Review flags stayed border/detail driven:
  `gradcam_border=7/12`, `rollout_border=5/12`,
  `object_color_sensitive=4/12`, with one Grad-CAM background flag. A manual
  Grad-CAM overlay on `case_012_t3_p2` showed heat on bright surface spots,
  fruit border, and adjacent background, matching the old surface/border failure
  mode.
- Cleanup and decision: reject before probe. Metrics/config/history/logs and
  architecture trace were copied into the XAI artifact under
  `source_smoke_metrics`, then the smoke train directory and launch logs were
  deleted via
  `runs\cleanup_manifest_20260704_selfboostcam_rejected_smoke.json`,
  reclaiming `182.80 MB`; D: free space after cleanup was about `73.42 GB`.
  Keep the default-off implementation and tests, but do not repeat
  `SelfBoostingAttentionLossWeight=0.006`, `Temperature=0.40`, all classes on
  the current keeper, or nearby weight/temperature tweaks. The hard-label CAM
  target is too diffuse and reuses the current classifier's weak boundary
  evidence rather than adding class-1 reliability.

## Smoke 2026-07-04 - CMT/LeFF Block-Local Sparse Patch Mixer Reject

- Rationale: the next architecture route tested whether a small local
  convolutional mixer inside transformer blocks could add the local inductive
  bias described by CMT/CvT/CeiT-LeFF/Conformer-style hybrids
  (`https://arxiv.org/abs/2107.06263`, `https://arxiv.org/abs/2103.15808`,
  `https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_Incorporating_Convolution_Designs_Into_Visual_Transformers_ICCV_2021_paper.pdf`,
  `https://openreview.net/pdf?id=U1cPSPskTR`). This keeps the model no-pretrain
  and does not touch raw data.
- Implementation:
  added default-off `BlockLocalPatchMixer` and V8/config/CLI flags
  (`BlockLocalPatchMixer`, layers, dropout, residual scale), with old-checkpoint
  resume allowing the new `local_patch_mixer` parameters. The first adapter-only
  smoke exposed a real wiring bug: token pruning leaves only `167` patch tokens,
  so the initial mixer returned the input because it expected a full `16x16`
  grid, and the loss had no grad when only mixer prefixes were trainable. The
  mixer now accepts `patch_indices`, scatters sparse kept tokens back onto the
  full grid, runs depthwise/pointwise conv, then gathers the kept locations.
  A reproduction confirmed logits/loss regained gradients and `15/15`
  trainable tensors received gradients. Preflight passed
  `py_compile trkh\models\model.py trkh\training\train.py trkh\core\config.py`
  and `pytest tests\test_block_local_patch_mixer.py -q` (`3 passed`).
- Smoke:
  `runs\smoke_v8_yolof_blocklocal678_sparsefix_adapteronly_lr5e4_60b_1e_20260704`
  used the current keeper checkpoint, explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, empty hard manifest, no final
  test, full validation support `2606`, 60 train batches, adapter-only trainable
  prefixes `blocks.5/6/7.local_patch_mixer`, `LR=5e-4`, and the keeper recipe
  (`BBoxSpatialFusion=true`, `crop_bbox`, teacher-focus-binary `0.015`, metric
  loss `0.04`, pairwise routing, boundary-band dropout, attention-view loss
  `0.35`).
- Validation result:
  macro/class-1 F1 was `0.8837/0.6822`, below the keeper `0.8847/0.6860` and
  below the `0.70` class-1 gate. Class-1 P/R was `0.6094/0.7748`. Top confusions
  were `3->2=52`, `0->1=49`, `1->0=18`, `4->1=12`, `0->4=11`, and `1->2=11`;
  this is not probe-worthy.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_blocklocal678_sparsefix_val_selected12_20260704`
  audited 12 selected validation mistakes with all methods and robustness probes.
  Foreground focus stayed high but attention was border/background-heavy
  (`attention/grad_rollout/gradcam/rollout` foreground
  `0.9105/0.9769/0.9558/0.9564`). Background blur/gray drops stayed near zero
  (`0.0000/-0.0005` original-prediction drop), while object desaturation was
  much larger (`0.0919`). Review flags showed the same failure mode:
  `attention_border=12/12`, `attention_background=4/12`,
  `gradcam_border=7/12`, `rollout_border=3/12`,
  `object_color_sensitive=4/12`. Manual Grad-CAM overlays on `case_012_t0_p1`
  and `case_009_t3_p2` showed heat on fruit edge, adjacent background, and
  even off-object background rather than stable surface evidence.
- Cleanup and decision: reject before probe. Metrics/config/history/logs and the
  pre-fix crash evidence were copied into the XAI artifact under
  `source_smoke_metrics`; the sparsefix smoke train directory and the pre-fix
  crash run were deleted via
  `runs\cleanup_manifest_20260704_blocklocal_sparsefix_rejected_smoke.json`,
  reclaiming `134.32 MB`; D: free space after cleanup was about `73.35 GB`.
  Keep the sparse-token mixer implementation and tests because the wiring fix is
  valid, but do not repeat the exact adapter-only recipe
  `BlockLocalPatchMixer=true/layers=6,7,8/dropout=0.02/scale=0.10/
  TrainableModulePrefixes=blocks.5-7.local_patch_mixer/LR=5e-4/60b`. It repairs
  the graph but does not add a class-1 surface/boundary signal.

## Smoke 2026-07-04 - Block-Local + Part-Token Pairwise Reject

- Rationale: after the adapter-only block-local sparsefix failed, I tested the
  more connected version suggested by the previous audit: local convolutional
  token mixing should feed an explicit patch/part decision head instead of only
  acting as a residual adapter. This combines the CvT-style convolutional
  inductive bias (`https://arxiv.org/abs/2103.15808`) with adaptive token/part
  aggregation ideas from TokenLearner
  (`https://proceedings.neurips.cc/paper/2021/hash/6a30e32e56fce5cf381895dfe6ca7b6f-Abstract.html`),
  while keeping raw `yolo_f` data unchanged.
- Smoke:
  `runs\smoke_v8_yolof_blocklocal_partpair678_w020_lr5e4_60b_1e_20260704`
  resumed the current keeper checkpoint, used explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, empty hard manifest, no final
  test, full validation support `2606`, 60 train batches, and the keeper recipe
  (`BBoxSpatialFusion=true`, `crop_bbox`, teacher-focus-binary `0.015`, metric
  loss `0.04`, pairwise routing, boundary-band dropout, attention-view loss
  `0.35`). Trainable prefixes were
  `blocks.5.local_patch_mixer,blocks.6.local_patch_mixer,
  blocks.7.local_patch_mixer,part_token_pairwise_learner` (`409,888`
  parameters), with `BlockLocalPatchMixer` layers `6,7,8` and
  `PartTokenPairwiseHead=true`, pairs `0-1,1-2,4-1,2-3`, part count `4`,
  temperature `0.55`, foreground power `1.2`, bbox weight `0.85`, loss weight
  `0.02`, LR `5e-4`. Preflight passed
  `py_compile trkh\models\model.py trkh\training\train.py trkh\core\config.py`
  and focused pytest
  `tests\test_block_local_patch_mixer.py tests\test_part_token_learner.py
  tests\test_trainable_module_prefixes.py -q` (`15 passed`).
- Validation result:
  macro/class-1 F1 was `0.8837/0.6822`, still below the keeper
  `0.8847/0.6860` and below the `0.70` class-1 gate. Class-1 P/R was
  `0.6094/0.7748`. The confusion matrix stayed effectively unchanged:
  `3->2=52`, `0->1=49`, `1->0=18`, `4->1=12`, `0->4=11`, and `1->2=11`.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_blocklocal_partpair678_val_selected12_20260704`
  audited 12 selected validation mistakes with all methods and robustness probes.
  Foreground mass was high (`attention/grad_rollout/gradcam/rollout`
  `0.9044/0.9722/0.9557/0.9498`), but border mass remained high
  (`attention=0.3042`, `gradcam=0.3018`) and review flags still showed
  `attention_border=11/12`, `gradcam_border=7/12`,
  `attention_background=4/12`, `rollout_border=4/12`. Background blur/gray
  drops stayed near zero (`0.0003/-0.0005` original-prediction drop), while
  object desaturation was much larger (`0.0727`). Manual overlays confirmed the
  same failure pattern: a `0->1` case heated broad fruit surface and edge, while
  a `3->2` case put Grad-CAM partly off-object and attention on adjacent
  background/edge.
- Cleanup and decision: reject before probe. Metrics/config/history/logs,
  best-val metrics, and architecture trace were copied into the XAI artifact
  under `source_smoke_metrics`, then the smoke train directory was deleted via
  `runs\cleanup_manifest_20260704_blocklocal_partpair_rejected_smoke.json`,
  reclaiming `136.88 MB`; D: free space after cleanup was about `73.27 GB`.
  Do not repeat the combined recipe
  `BlockLocalPatchMixer=true/layers=6,7,8/dropout=0.02/scale=0.10 +
  PartTokenPairwiseHead=true/pairs=0-1,1-2,4-1,2-3/logit_scale=0.16/
  dropout=0.05/loss_weight=0.02/part_count=4/temp=0.55/fg_power=1.2/
  bbox_weight=0.85/TrainableModulePrefixes=blocks.5-7.local_patch_mixer,
  part_token_pairwise_learner/LR=5e-4/60b-120b`. Connecting local conv to the
  part head did not produce a new recall-safe class-1 surface/boundary signal.

## Smoke 2026-07-04 - Paired-View Local Patch-Token Alignment Reject

- Rationale: user specifically asked to revisit the wide-context `yolo_f` view
  together with the crop-style `class_f` view, but previous paired CE/KL/feature
  mean consistency and learned residual fusion had already failed. I therefore
  implemented a new default-off paired-view feature source, `patch_tokens`, based
  on local/dense correspondence ideas rather than another logit/fusion loss. The
  direction was grounded in CrossViT cross-attention between token streams
  (`https://arxiv.org/abs/2103.14899`), multi-view hybrid fusion/mutual
  distillation (`https://openaccess.thecvf.com/content/WACV2024/papers/Black_Multi-View_Classification_Using_Hybrid_Fusion_and_Mutual_Distillation_WACV_2024_paper.pdf`),
  and DenseCL-style local feature matching (`https://arxiv.org/abs/2011.09157`).
  Raw data was not modified.
- Implementation:
  `trkh.training.train` now accepts `--paired-view-feature-source patch_tokens`.
  Instead of averaging all patches like the old `patch` source, it selects the
  strongest local patch tokens from primary and paired features, applies a soft
  nearest-neighbor cosine alignment, respects `key_padding_mask`, and reuses
  existing `paired_view_feature_consistency_loss` telemetry. V8 launcher
  `PairedViewFeatureSource` now allows `patch_tokens`. Focused preflight passed
  `py_compile trkh\training\train.py trkh\core\config.py trkh\models\model.py`,
  PowerShell launcher parse, and
  `pytest tests\test_paired_view_patch_token_alignment.py
  tests\test_yolo_source_context_dataset.py tests\test_block_local_patch_mixer.py -q`
  (`11 passed`).
- Smoke:
  `runs\smoke_v8_yolof_pairtokenalign_classf_w004_t012_60b_1e_20260704`
  resumed the current keeper checkpoint, used explicit primary
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, paired auxiliary
  `D:\DataAI\AIEx\newdataset\class_f\data.yaml`, and
  `AuxiliaryTrainClassificationFolderYoloData=yolo_f` so mapping stayed
  object-level (`primary_base_samples=9215`, `auxiliary_base_samples=9215`,
  `effective_samples=9215`). It used empty stale manifests, no final test,
  full validation support `2606`, 60 train batches, and the keeper recipe
  (`BBoxSpatialFusion=true`, `crop_bbox`, teacher-focus-binary `0.015`, metric
  loss `0.04`, pairwise routing, boundary-band dropout, attention-view loss
  `0.35`). The only new objective was
  `PairedViewFeatureConsistencyWeight=0.004`,
  `PairedViewFeatureSource=patch_tokens`, `PairedViewTemperature=0.12`.
  Telemetry confirmed it was active
  (`train_paired_view_feature_consistency_loss=0.7152`,
  `train_paired_view_fraction=1.0`).
- Validation result:
  macro/class-1 F1 was `0.8850/0.6841`. Macro was slightly above the keeper
  `0.8847`, but class-1 F1 fell below the keeper `0.6860` and stayed far below
  the `0.70` gate. Class-1 P/R was `0.6082/0.7815`. Confusion matrix was
  `[[487,50,2,0,10],[17,118,11,0,5],[5,10,515,8,6],[0,4,52,654,2],[8,12,4,1,625]]`;
  the extra `0->1` false positive explains the class-1 regression.
- XAI/audit:
  `runs\xai_smoke_v8_yolof_pairtokenalign_classf_w004_t012_val_selected12_20260704`
  audited 12 selected validation mistakes with all methods and robustness
  probes. Foreground mass stayed high
  (`attention/grad_rollout/gradcam/rollout`
  `0.9831/0.9680/0.9757/0.9465`), but the same failure mode remained:
  `gradcam_border=9/12`, `rollout_border=5/12`, `object_color_sensitive=4/12`,
  and one Grad-CAM background flag. Background blur/gray drops were near zero
  (`0.0008/-0.0001` original-prediction drop), while object desaturation was
  much larger (`0.0895`). Manual overlays confirmed no new object/background
  separation: a `0->1` case heated both fruit surface and right-side background;
  a `3->2` case put Grad-CAM almost entirely off-object; another `3->2` case
  heated object border, background, and spots.
- Cleanup and decision: reject before probe. Metrics/config/history/logs,
  best-val artifacts, and architecture trace were copied into the XAI artifact
  under `source_smoke_metrics`, then the smoke train directory and launch logs
  were deleted via
  `runs\cleanup_manifest_20260704_pairtokenalign_rejected_smoke.json`,
  reclaiming `182.70 MB`; D: free space after cleanup was about `73.20 GB`.
  Keep the default-off `patch_tokens` source and tests because the code path is
  valid, but do not repeat the exact recipe
  `PairedViewTrain=true/yolo_f primary + class_f paired/
  PairedViewFeatureConsistencyWeight=0.004/FeatureSource=patch_tokens/
  Temperature=0.12/teacherfocusbinary015/BBoxSpatialFusion/crop_bbox/
  LR=8e-5/60b-120b`. Local token alignment transfers view invariance but still
  fails to add a recall-safe class-1 boundary signal.

## Diagnostic 2026-07-04 - YOLO Crop-Margin Sensitivity Reject

- Rationale: the user asked whether the wide-background `yolo_f` view might help
  ViT self-attention separate background from the object. Earlier train smokes
  with broad crop/context had failed, but we had not yet run a clean inference
  sweep around the current keeper's crop margin. The diagnostic was grounded in
  TransFG's discriminative patch selection (`https://arxiv.org/abs/2103.07976`)
  and HERBS' observation that FGVC background suppression only helps when it
  preserves discriminative object-scale features (`https://arxiv.org/abs/2303.06442`).
- Implementation:
  added `trkh.tools.probe_crop_margin_sensitivity`, a validation-only tool that
  loads a checkpoint, rebuilds `MangoYOLOCropDataset` with a `crop_margin_ratio`
  override, evaluates each margin on full support, and writes
  `crop_margin_sweep.csv`, per-margin metrics, baseline predictions, and
  changed-row CSVs versus baseline sample indices. It refuses `--split test` for
  method selection. Focused preflight passed
  `py_compile trkh\tools\probe_crop_margin_sensitivity.py` and
  `pytest tests\test_probe_crop_margin_sensitivity.py -q` (`3 passed`).
- Diagnostic:
  `runs\diagnostic_yolof_cropmargin_sensitivity_keeper_val_20260704` evaluated
  the current keeper checkpoint on explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, split `val`, full support
  `2606`, baseline margin `0.05`, margins
  `0,0.02,0.05,0.08,0.12,0.18,0.25`, and `crop_bbox` token prior. Best macro
  and best class-1 were both the baseline margin `0.05`: macro/class-1 F1
  `0.8841/0.6841`. Smaller margins increased recall but widened class-1 false
  positives (`m0` class-1 `0.6704`, `0->1=64`). Wider margins improved class-1
  precision but lost recall: `m0.12` reached class-1 P/R/F1
  `0.6201/0.7351/0.6727`, reducing `0->1` from `50` to `39` but increasing
  true class-1 misses (`1->0=22`, `1->4=9`). Large context was clearly harmful:
  `m0.18` fell to `0.8453/0.5875`, `m0.25` to `0.8015/0.5190`.
- Changed-case audit:
  `m0.12` changed `102` validation rows (`42` corrections, `49` harms,
  `11` neutral). It fixed some baseline class-1 false positives (`0:1->0=15`)
  and `3:2->3=11`, but harmed true class-1 (`1:1->0=8`) and introduced
  additional `2:2->1=7`/`2:2->4=5` errors. XAI on 12 representative changed
  rows in
  `runs\xai_cropmargin_m0p12_changed12_yolof_keeper_val_20260704` showed
  foreground mass `0.9192/0.9556/0.9180/0.9219`
  (attention/grad-rollout/Grad-CAM/rollout), but border/background flags stayed
  high (`gradcam_border=7/12`, `rollout_border=9/12`,
  `attention_background=3/12`, `gradcam_background=3/12`). Background blur/gray
  drops were near zero (`0.0000/-0.0014` original-prediction drop), while manual
  overlays showed heat on fruit edges, stem/hand/background, and strong
  off-object border bands. The margin-sensitive rows are still surface/boundary
  ambiguity, not a missing-context problem.
- Cleanup and decision: reject crop-margin/context-only direction before any
  training probe. Duplicate prediction exports were deleted via
  `runs\cleanup_manifest_20260704_cropmargin_duplicate_predictions.json`,
  reclaiming `2.93 MB`; the diagnostic summary, sweep, changed CSVs, baseline
  prediction CSV, and XAI audit were preserved. Do not repeat simple
  crop-margin sweeps or train/eval context widening on the current keeper. If
  context is revisited, it must be a reliability-gated selector with explicit
  class-1 recall protection, not a global margin/background-size change.

## Diagnostic 2026-07-04 - Train-Only Crop-Margin Router Reject

- Rationale: after the global crop-margin sweep rejected simple context
  widening, the only context direction still allowed by the audit was a
  reliability-gated selector: use train labels to learn when the `0.12` margin
  helps suppress false-positive class 1, while preserving true class-1 recall
  from the baseline `0.05` margin. This tested the user's hypothesis that
  `yolo_f` context may still help if applied conditionally, without editing raw
  data or touching test.
- Inputs:
  generated full train/validation prediction CSVs for the current keeper with
  `trkh.tools.probe_crop_margin_sensitivity`, explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, `crop_bbox` token prior,
  margins `0.05,0.12`, `--save-all-predictions`, and no test access. Train
  support was `9215`; validation support was `2606`. On train, `0.05` already
  beat `0.12` (`0.9401/0.8164` macro/class-1 F1 vs `0.9218/0.7667`), and on
  validation `0.05` also beat `0.12` (`0.8841/0.6841` vs `0.8792/0.6727`).
- Router:
  `runs\router_cropmargin_m005_m012_trainonly_noimg_yolof_keeper_20260704`
  used `trkh.tools.train_trainval_expert_router` with experts `m005,m012`,
  `--selection-source train_oof`, `--focus-class-weight 0.50`, 5-fold OOF, and
  `--disable-image-features` so the diagnostic stayed focused on the two margin
  probability views. Train OOF selected `logreg_c0p1` with OOF macro/focus F1
  `0.9464/0.8376`, but validation collapsed to `0.8758/0.6458`. The simple
  fixed average candidates were much more stable (`~0.8843/0.6845`) but still
  did not beat the keeper class-1 anchor `0.6860`.
- Changed-case audit:
  the selected train-only router changed `76` validation rows versus baseline
  margin `0.05`: `33` corrections, `34` harms, and `9` neutral. `58` changes
  were class-1 related. The main trade was unsafe: it fixed false-positive
  class-1 rows from class 0 (`0:1->0=20`) but also converted true class-1 rows
  that baseline got right into class 0 or 2 (`1:1->0=13`,
  `1:1->2=12`). Baseline class-1 F1 in router order was `0.6841`; router
  class-1 F1 was `0.6458`.
- XAI/audit:
  `runs\xai_router_cropmargin_m005_m012_trainonly_noimg_class1_harm12_20260704`
  audited 12 true-class-1 harm rows. The base keeper was correct on all 12 but
  low confidence (`0.226-0.271`) and near-tie-heavy (`near_tie_top2=7/12`).
  Heatmaps still focused mostly on the crop foreground
  (`attention/grad-rollout/Grad-CAM/rollout foreground`
  `0.9737/0.9795/0.9750/0.9589`), background perturbations were near zero
  (`background_blur=0.0004`, `background_gray=0.0020`), and border flags
  remained (`attention_border=3/12`, `gradcam_border=3/12`,
  `rollout_border=6/12`). Manual overlays showed the same failure mode: fruit
  surface/boundary ambiguity, occasional off-object heat, and no reliable
  background-context cue.
- Cleanup and decision: reject crop-margin reliability router on the current
  keeper. It overfits train OOF, is not recall-safe for class 1, and does not
  cross the `0.70` gate. Reproducible full prediction CSVs and the rejected
  router pickle were deleted after preserving sweeps, metrics, changed
  summaries, and XAI via
  `runs\cleanup_manifest_20260704_cropmargin_router_rejected_intermediates.json`,
  reclaiming `28.39 MB`. Do not repeat two-expert `m0.05/m0.12` train-only
  logit/probability routing, including nearby logistic/tree/router candidate
  sweeps, unless a new reliability signal explicitly protects true class-1
  recall.

## Diagnostic 2026-07-04 - BBox Token Region Separability Reject

- Rationale: before spending another smoke on background-uniform/adversarial
  region losses, I measured whether `yolo_f` bbox-derived foreground and
  background token descriptors contain a reliable class signal on the current
  keeper. This directly tests the user's hypothesis that wide context can teach
  ViT what is background versus object. I rechecked relevant literature:
  RIVAL10/CVPR 2022 (`https://arxiv.org/abs/2201.10766`) argues that
  foreground/background sensitivity must be measured quantitatively, ForAug
  (`https://arxiv.org/html/2503.09399v4`) and MaskTune
  (`https://proceedings.neurips.cc/paper_files/paper/2022/hash/93be245fce00a9bb2333c17ceae4b732-Abstract-Conference.html`)
  are useful when a background shortcut is actually active, and DANN/gradient
  reversal (`https://jmlr.org/papers/v17/15-239.html`) requires a nuisance
  signal worth removing. Prior TRKH audits already showed near-zero
  background blur/gray sensitivity, so this probe was a final quantitative
  check, not permission to add another background loss.
- Implementation:
  added `trkh.tools.probe_bbox_token_region_separability`, a train/val-only
  diagnostic that pools frozen patch tokens into foreground, background,
  foreground-minus-background, and concat descriptors using `patch_bbox_prior`
  from `crop_bbox`, fits only train-side decision layers, refuses test unless
  explicitly overridden, and writes descriptor metrics plus predictions.
  Focused preflight passed `py_compile` and
  `pytest tests\test_probe_bbox_token_region_separability.py -q` (`3 passed`).
- Diagnostic:
  `runs\diagnostic_yolof_bbox_region_separability_keeper_full_20260704` used
  the current keeper checkpoint, explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, splits `train,val`, full
  train/val support `9215/2606`, and `crop_bbox` token prior. The base head
  stayed the validation anchor (`0.8847/0.6860` macro/class-1 F1). Train-only
  decision layers on region descriptors did not match it: foreground-token
  logistic-balanced reached only `0.8658/0.6427`, background-token
  logistic-balanced only `0.8398/0.5833`, and foreground+background concat
  logistic-balanced only `0.8438/0.5836`. Background tokens had nonzero class
  signal, but it was unreliable and far below the keeper.
- Changed-case audit:
  comparing background-token logistic-balanced predictions with the base head
  changed `245` validation rows: `63` corrections, `157` harms, `25` neutral,
  with `136` class-1-related changes. It did fix some base false positives
  (`0:1->0=19`) but created many more class-1 errors, including true
  class-1 base-correct rows moved to class 0 or 2 (`1:1->0=13`,
  `1:1->2=8`) and new non-class1-to-class1 moves (`4:4->1=22`,
  `0:0->1=20`, `2:2->1=16`). Foreground-token logistic-balanced was also
  unsafe (`164` changes, `54` corrections, `95` harms).
- XAI/audit:
  `runs\xai_bbox_region_background_logregbalanced_harm12_yolof_keeper_20260704`
  audited 12 class-1-related harm rows where the base head was correct but the
  background-token classifier would change the decision. Foreground mass
  remained high (`attention/grad-rollout/Grad-CAM/rollout`
  `0.9598/0.9631/0.9446/0.9436`), while background blur/gray drops were near
  zero (`0.0005/-0.0001`) and object desaturation was larger (`0.0373`).
  Review flags were mostly border/local-background adjacent rather than broad
  background dependence (`rollout_border=9/12`, `gradcam_border=4/12`,
  `gradcam_background=3/12`, `attention_background=2/12`).
- Cleanup and decision:
  reject background-token routing and reject-before-smoke a new
  background-region uniform/adversarial loss on the current keeper. The
  diagnostic says background descriptors are not a recall-safe nuisance signal;
  making them uniform would likely repeat the already rejected
  background-focus/background-counterfactual family without adding a new
  surface-boundary cue. The smoke64 runtime check was deleted via
  `runs\cleanup_manifest_20260704_bbox_region_smoke64_deleted.json`,
  reclaiming `0.12 MB`; the full summary/metrics/predictions and XAI audit
  were preserved. Do not repeat background-only token classifiers, FG/BG
  concat routers, or background-region uniform/adversarial losses unless a new
  diagnostic first shows background perturbation has become causally important.

## Diagnostic 2026-07-04 - Token-Prune Keep-Rate Sensitivity Reject

- Rationale: after block-local, part-token, paired-token, and bbox-region paths
  failed, I tested whether the current checkpoint was losing subtle
  surface/boundary evidence because hard token pruning keeps only `167/256`
  patch tokens after layer 5. This is a cheap config-only diagnostic before any
  new training. The literature basis was TransFG's discriminative patch
  selection for fine-grained recognition (`https://arxiv.org/abs/2103.07976`),
  EViT's token reorganization tradeoff (`https://arxiv.org/abs/2202.07800`),
  and DynamicViT's progressive pruning motivation
  (`https://openreview.net/forum?id=kR95DuwwXHZ`).
- Implementation:
  added `trkh.tools.probe_token_prune_sensitivity`, which clones checkpoint
  `model_config`, overrides only `token_pruning/token_prune_layers/
  token_keep_rates`, refuses test unless explicitly allowed, records final
  kept-token counts from architecture trace, writes per-variant predictions,
  and exports changed-case CSVs against the checkpoint baseline. Preflight
  passed `py_compile` and
  `pytest tests\test_probe_token_prune_sensitivity.py -q` (`4 passed`).
- Diagnostic:
  `runs\diagnostic_yolof_tokenprune_sensitivity_keeper_full_20260704`
  evaluated the current keeper on explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, split `val`, full support
  `2606`, and `crop_bbox` token prior. Baseline `0.85,0.65` pruning kept
  `167` tokens and reproduced validation macro/class-1 F1 `0.8847/0.6860`.
  Keeping more tokens did not help: `0.95,0.85` kept `218` tokens and fell to
  `0.8778/0.6514`; `1.0,0.85` kept `218` tokens and reached `0.8788/0.6534`;
  disabling pruning kept all `256` tokens and reached only `0.8787/0.6554`.
- Changed-case audit:
  less pruning fixed some `3->2` rows but widened class-1 false positives and
  lost class-1 precision. `no_prune` changed `49` validation rows
  (`20` corrections, `25` harms, `4` neutral), with `24` class-1-related
  changes. Main harmful transitions were `2:2->1=6`, `0:0->1=4`,
  `4:4->1=5`, plus true class-1 recall loss `1:1->0=3`. `keep100_85` and
  `keep95_85` showed the same direction with fewer changes and still lower
  class-1 F1.
- XAI/audit:
  `runs\xai_tokenprune_no_prune_class1_harm12_yolof_keeper_20260704` used a
  temporary config-only `no_prune` checkpoint on 12 class-1-related harm rows.
  Foreground mass stayed high (`attention/grad_rollout/gradcam/rollout`
  `0.9601/0.9804/0.9383/0.9612`), but border flags remained
  (`attention_border=6/12`, `rollout_border=8/12`, `gradcam_border=4/12`) and
  near ties were common (`6/12`). Background blur/gray drops were near zero
  (`-0.0001/0.0041` original-prediction drop), while object desaturation was
  larger (`0.0214`). More tokens did not recover missing object detail; it
  amplified ambiguous surface/border evidence into class-1 false positives.
- Cleanup and decision:
  reject token-prune keep-rate loosening and no-prune evaluation on the current
  keeper. The smoke2 runtime check and temporary config-only checkpoint were
  deleted via
  `runs\cleanup_manifest_20260704_tokenprune_diagnostic_intermediates.json`,
  reclaiming `55.73 MB`; full diagnostic predictions/summaries and XAI were
  preserved. Do not run no-prune or higher keep-rate training/probes on this
  anchor unless a new recall-safe target first changes the surface/boundary
  representation; simply retaining more patch tokens increases `0/2/4->1`
  false positives and remains below the `0.70` class-1 validation gate.

## Diagnostic 2026-07-04 - Current Keeper Boundary Review and XAI Dataset-Crop Fix

- Rationale: after context/background, token-prune, part/local, color-stat, and
  robust-loss families failed, I rebuilt a current boundary atlas from the
  keeper's full `yolo_f/val` baseline predictions instead of launching another
  smoke without a new signal. This diagnostic uses validation for observation
  only; no threshold, weight, or training policy is selected from it.
- Boundary review artifact:
  `runs\boundary_review_yolof_keeper_val_current_20260704`. Input was
  `runs\diagnostic_yolof_tokenprune_sensitivity_keeper_full_20260704\predictions_baseline.csv`,
  full validation support `2606`, pairs `0-1,1-2,2-3,4-rest`, focus class `1`,
  and explicit split `val`. It selected `260` rows and rendered
  `boundary_review_report.html`; copied images were capped per reason. Reason
  counts were `focus_false_positive=75`, `focus_false_negative=33`,
  `high_confidence_error=8`, `low_margin_error=70`, and
  `low_margin_correct_boundary=74`. Pair counts were `0-1=110`, `1-2=56`,
  `2-3=63`, and `4-rest=31`.
- Manual visual check: top `1->0` and `1->2` rows are mostly green/yellow fruit
  sitting directly on the 0/1/2 maturity boundary; top `0->1` rows often have
  yellow patches, glare, or green-yellow surface tones that visually resemble
  class 1; `3->2` rows often include real dark spots but also clean yellow
  regions. This reinforces the earlier conclusion that the problem is mostly
  object surface/boundary semantics under lighting, not missing wide background.
- XAI tooling fix: while auditing the review cases, I found that
  `trkh.evaluation.xai_audit` collected predictions from the exact
  `MangoYOLOCropDataset` tensor but rendered/robustness-checked the case by
  re-reading the raw image path through `prepare_image_and_tensor`. For YOLO
  object crops this can mismatch the evaluated object-crop tensor. I changed
  XAI to prefer `dataset[sample_index][0]` for the audit tensor/crop image and
  to use that same tensor for `clean_recheck`. Preflight passed
  `py_compile trkh\evaluation\xai_audit.py` and
  `pytest tests\test_xai_audit_dataset_tensor.py -q` (`3 passed`). The
  pre-fix XAI artifact was deleted via
  `runs\cleanup_manifest_20260704_xai_boundary_review_mismatched_tensor_deleted.json`,
  reclaiming `86.34 MB`.
- Corrected XAI artifact:
  `runs\xai_boundary_review_yolof_keeper_val_current_cases16_datasetcrop_20260704`
  audited 16 representative review cases with all methods, robustness probes,
  explicit `yolo_f`, and `bbox-token-prior-source=crop_bbox`. Clean recheck now
  matches the selected prediction. Review flags were
  `rollout_background_attention=10/16`, `gradcam_border_attention=8/16`,
  `attention_background_attention=7/16`, `attention_border_attention=6/16`,
  `object_color_sensitive=7/16`, and `near_tie_top2=4/16`. Heatmap foreground
  mass remained meaningful but not perfectly clean:
  attention/grad-rollout/Grad-CAM/rollout foreground mass
  `0.8325/0.9443/0.9166/0.8850`; background mass
  `0.1675/0.0557/0.0834/0.1150`; Grad-CAM border mass `0.2588`.
- Robustness after the dataset-crop fix still does not support broad background
  as the causal bottleneck. Mean background blur/gray original-prediction drops
  were only `0.0077/0.0073`, while object desaturation drop was `0.1344` and
  center occlusion drop was `0.0295`. Thus the corrected audit strengthens,
  rather than weakens, the conclusion that surface/color evidence dominates.
- Basic prediction forensics artifact:
  `runs\forensics_yolof_keeper_val_current_basic_20260704`. Full-val metrics
  reproduce the keeper (`accuracy=0.9198`, class-1 P/R/F1
  `0.6114/0.7815/0.6860`). Calibration is very soft/poorly calibrated
  (`ECE=0.6052`), but prior logit/calibration routes have already failed to
  improve class 1. Error margins are much smaller than correct margins
  (`error p50=0.0569` vs correct p50 `0.1249`), and the `low_margin` bucket
  has `96/257` errors (`37.35%`). Lighting buckets are contributory but not
  decisive: `shadow_heavy` has `27/136` errors (`19.85%`) and
  `highlight_heavy` has `24/344` (`6.98%`). Pair metrics show the dominant
  bottleneck remains `0-1` (`49` `0->1`, `17` `1->0`) and `2-3` (`52`
  `3->2`).
- Decision: keep the current boundary review, corrected XAI audit, forensics,
  and XAI tooling fix. Do not launch another background/context/token-prune,
  color-stat guard, or hard-label margin smoke from this evidence. The next
  useful step must introduce a genuinely new train-only representation/target
  signal or a fold-safe pretrained-disagreement compression route; otherwise it
  will repeat low-margin routing around the same object-surface manifold.

## Smoke 2026-07-04 - Local-Neighbor CMSF Feature Cache Reject

- Rationale: after context/background/token-prune and frozen-embedding
  geometry routes failed, I tested a train-only representation target inspired
  by constrained mean-shift feature learning
  (`https://arxiv.org/abs/2112.04607`) and part-sensitive FGVC contrastive
  learning (`https://arxiv.org/abs/2309.13822`). The intended difference from
  old global KD/feature CRD and `focus_neighbor_binary` was to smooth each
  sample toward distant-but-same-class local neighbors in the current TRKH
  embedding, with stronger blending for true class 1 and mild blending for
  non-1 samples currently predicted as class 1. This does not edit raw data and
  refuses non-train splits.
- Implementation:
  added `trkh.tools.build_local_neighbor_feature_cache`, which extracts
  keeper train embeddings on explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, builds normalized
  `sample_index` teacher features, and reuses the existing teacher-feature CRD
  path. Preflight passed
  `py_compile trkh\tools\build_local_neighbor_feature_cache.py` and
  `pytest tests\test_build_local_neighbor_feature_cache.py
  tests\test_teacher_feature_rkd.py -q` (`12 passed`).
- Cache artifact:
  `runs\local_neighbor_feature_cache_yolof_keeper_cmsf_head_20260704`.
  It covers `9215/9215` train objects, `feature_dim=256`, no fallback rows,
  class counts `1941/541/1920/2520/2293`, and mean target-source cosine
  `0.99937` (`min=0.97047`). Reasons were `same_class_neighbor=8451`,
  `focus_same_class_neighbor=541`, and
  `boundary_predicted_focus_neighbor=223`. This confirms the target is gentle
  and train-only, but also suggests it may be too close to the current failed
  manifold.
- Smoke:
  `runs\smoke_v8_yolof_localneighbor_cmsfcrd_w002_head_val_20260704` used
  the current keeper checkpoint, explicit `yolo_f`, `BBoxSpatialFusion=true`,
  `BboxTokenPriorSource=crop_bbox`, teacher-focus-binary `0.015`, empty hard
  manifest, `SkipFinalTest`, `60` train batches, and full validation support
  `2606`. The CRD adapter was active
  (`weight=0.002`, `temp=0.18`, `proj=128`, `dropout=0.02`, source `head`,
  pair mode `boundary_or_intra`, pairs `0-1,1-2,1-4,2-3`), but validation fell
  to macro/class-1 F1 `0.8774/0.6592`, class-1 P/R
  `0.5700/0.7815`, below the keeper `0.8847/0.6860`. Main confusion worsened
  class-1 false positives: `0->1=57` vs keeper `49`, `4->1=16` vs `12`,
  with `1->0=15`, `1->2=12`, and `3->2=57`.
- Telemetry:
  teacher-feature contrastive loss was active but weakly separated
  (`loss=2.5369`, `count=32`, `negative_count=15.675`,
  positive similarity `0.0855`, negative similarity `0.0175`,
  top1 `0.1438`). This is better than completely inactive transfer but not
  enough to produce a recall-safe class-1 boundary.
- Audit:
  `runs\eval_smoke_v8_yolof_localneighbor_cmsfcrd_w002_head_val_20260704`
  reproduced the same full-val metrics and exported predictions.
  `runs\boundary_review_smoke_v8_yolof_localneighbor_cmsfcrd_w002_val_20260704`
  selected `180` rows: `focus_false_positive=45`,
  `focus_false_negative=33`, `high_confidence_error=12`,
  `low_margin_error=45`, and `low_margin_correct_boundary=45`.
  `runs\xai_smoke_v8_yolof_localneighbor_cmsfcrd_w002_val_selected12_20260704`
  audited 12 stratified rows using the dataset-crop-fixed path. XAI again
  shows object/color dominance over background: background blur/gray drops only
  `0.0106/0.0107`, while object desaturation drops `0.1887`;
  `object_color_sensitive=10/12`, `gradcam_border=8/12`,
  `rollout_background=5/12`, and `attention_background=4/12`. Visual overlays
  show class-1 false positives heating yellow/green surface patches and fruit
  edges, while true class-1 misses can heat adjacent leaves/background.
- Cleanup and decision:
  reject this exact local-neighbor CMSF CRD recipe before any probe. Metrics,
  config, history, launcher transcript, and architecture trace were copied to
  the XAI artifact under `source_smoke_metrics`; the smoke train directory was
  deleted via
  `runs\cleanup_manifest_20260704_localneighbor_cmsfcrd_rejected_smoke.json`,
  reclaiming `184.27 MB`. Do not repeat
  `build_local_neighbor_feature_cache min_rank=4/max_rank=24/neighbors=6 +
  TeacherFeatureContrastiveLossWeight=0.002/temp=0.18/proj=128/dropout=0.02/
  source=head/pair_mode=boundary_or_intra` on the current keeper. If neighbor
  evidence is revisited, it needs a materially different objective with
  explicit false-positive control and true-class-1 recall protection, not
  another gentle same-class feature smoothing target.

## Diagnostic 2026-07-04 - Train-Only Patch Part-Prototypes Reject

- Rationale: after patch routers, token-prune, part-token heads, and local
  neighbor feature smoothing failed, I tested whether a part-discovery readout
  over frozen patch tokens could expose a new surface-boundary cue before
  launching another train smoke. The primary-source basis was PARTICLE part
  discovery for fine-grained recognition
  (`https://openaccess.thecvf.com/content/ICCV2023W/VIPriors/papers/Saha_PARTICLE_Part_Discovery_and_Contrastive_Learning_for_Fine-Grained_Recognition_ICCVW_2023_paper.pdf`)
  and CCT-style noisy fine-grained co-training
  (`https://arxiv.org/abs/2303.02404`). This diagnostic does not edit raw data,
  clusters only train patch tokens, refuses test by default, and uses validation
  only for observation.
- Implementation:
  added `trkh.tools.probe_patch_part_prototypes`. It extracts frozen final patch
  tokens from the current keeper on explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, selects up to `14` bbox/crop-bbox
  foreground tokens per sample, fits `32` train-only MiniBatchKMeans part
  prototypes from `120000` train patch rows, then evaluates logistic readouts on
  `part_hist_only` and `head_prob_part_hist`. Preflight passed
  `py_compile trkh\tools\probe_patch_part_prototypes.py` and
  `pytest tests\test_probe_patch_part_prototypes.py -q` (`2 passed`).
- Diagnostic artifact:
  `runs\diagnostic_yolof_part_prototypes_keeper_full_20260704`. It covered full
  `yolo_f/train=9215` and `val=2606`, selected `129010` train patch rows and
  `36484` validation patch rows, and used `crop_bbox` as the token prior source.
  The base predictions reproduced the current-code keeper neighborhood at
  validation macro/class-1 F1 `0.8841/0.6841` with class-1 P/R
  `0.6082/0.7815`.
- Results:
  `part_hist_only` was not a useful classifier: train OOF macro/class-1 F1
  `0.7534/0.3562`, validation `0.7302/0.3727`, with `534` validation changes
  and `423` harms. The richer `head_prob_part_hist` readout looked strong
  in train OOF (`0.9390/0.8222`) but the train-OOF-selected `C=0.05` validation
  result fell to macro/class-1 F1 `0.8794/0.6667`, below the keeper, with
  `93` validation changes (`41` corrections, `43` harms, `9` neutral). It fixed
  some base class-1 false positives (`0:1->0=17`) but broke true class-1 rows
  (`1:1->0=11`, `1:1->2=8`) and added new class-1 false positives (`0:0->1=4`).
  The non-selected `C=0.3` candidate merely tied the old class-1 neighborhood
  and cannot be chosen by validation tuning.
- XAI/audit:
  `runs\xai_part_prototype_headprob_changed12_yolof_keeper_20260704` audited
  12 changed rows from the selected `head_prob_part_hist` diagnostic. It again
  showed object/surface dominance rather than a safe part cue:
  background blur/gray drops were negative or near zero (`-0.0049/-0.0090`),
  while object desaturation drop was `0.0978`. Flags remained near-tie and
  border-heavy (`near_tie=6/12`, `rollout_border=9/12`,
  `gradcam_border=4/12`, `gradcam_background=4/12`). Manual overlays showed
  true-class1 harms heating object edges/bottom surface/background-adjacent
  regions, and new class1 FP harms heating pale green/yellow fruit borders.
- Cleanup and decision:
  reject frozen patch part-prototype readouts on the current keeper and do not
  launch a train smoke from this signal. The runtime smoke
  `runs\diagnostic_yolof_part_prototypes_keeper_smoke_20260704` was deleted via
  `runs\cleanup_manifest_20260704_partprototype_smoke_deleted.json`; the full
  diagnostic and XAI artifacts were preserved. Do not repeat
  `num_prototypes=32/max_fit_patches=120000/max_patches_per_sample=14/
  bbox_threshold=0.05 + part_hist/head_prob_part_hist logistic readout` on this
  checkpoint. A future part route needs a genuinely new initialized selector or
  supervision target, not another frozen patch-token histogram over the same
  surface-boundary manifold.

## Smoke 2026-07-04 - Top-5 Expert Vote-Disagreement Sample Weights Reject

- Rationale: after frozen patch/readout and local-neighbor representation routes
  failed, I tested whether disagreement among the five pretrained TTA experts
  could provide a very narrow train-only uncertainty signal. This was inspired
  by ensemble uncertainty work
  (`https://arxiv.org/abs/1612.01474`) and noisy-label sample-selection ideas
  from co-teaching (`https://arxiv.org/abs/1804.06872`), but deliberately avoids
  direct top-5 average KD because the train top-5 average is almost hard-label
  reinforcement.
- Implementation:
  added `trkh.tools.build_top5_vote_disagreement_sample_weights`. It reads each
  individual expert prediction CSV, strictly remaps `class_f` crop keys to
  `yolo_f` sample-index order, writes sample-index weights only for train split,
  and refuses non-train manifests unless `--diagnostic-only` is used. The policy
  protects true class 1 by default and downweights only non-class1 rows with
  expert disagreement, especially if class-1 votes are involved. Preflight
  passed `py_compile trkh\tools\build_top5_vote_disagreement_sample_weights.py`
  and `pytest tests\test_build_top5_vote_disagreement_sample_weights.py -q`
  (`2 passed`).
- Train manifest artifact:
  `runs\top5_vote_disagreement_sample_weights_yolof_train_20260704`.
  It covered `9215/9215` train rows from 5 experts, found only `169`
  vote-disagreement rows, wrote `109` weighted non-class1 rows, and protected
  `60` true class-1 rows from downweighting. Weighted target counts were
  `0:53`, `2:47`, `3:8`, `4:1`, with mean manifest weight `0.8347`
  (`min=0.72`, `max=0.92`). Loader audit in smoke matched all `109` sample
  indices, mean effective train weight `0.9980`, no path fallback.
- Validation diagnostic artifact:
  `runs\top5_vote_disagreement_val_diagnostic_yolof_20260704` was
  diagnostic-only and produced no manifest. It found `219` val disagreement
  rows, of which `86` were keeper errors and `104` were class-1-related.
  The signal was real but mixed: possible class-1 FP suppressions existed
  (`14` cases where keeper predicted class 1 and expert majority did not), but
  there were also class-1 harm/new-FP risks (`12` and `9` cases respectively).
  No val threshold was selected from this diagnostic.
- Smoke:
  `runs\smoke_v8_yolof_top5votedisagree_sampleweight_w082_20260704` used the
  current keeper checkpoint, explicit `yolo_f`, `BBoxSpatialFusion=true`,
  `BboxTokenPriorSource=crop_bbox`, teacher-focus-binary `0.015`, empty hard
  manifest, `SkipFinalTest`, `60` train batches, full validation support
  `2606`, and the new sample-weight manifest. It failed clearly:
  validation macro/class-1 F1 fell to `0.8769/0.6574`, class-1 P/R
  `0.5673/0.7815`, below the keeper `0.8847/0.6860`. Confusion worsened the
  exact failure mode: `0->1=58` vs keeper `49`, `4->1=16` vs `12`,
  `2->1=12`, `1->0=15`, `1->2=12`, and `3->2=57`.
- Audit:
  `runs\eval_smoke_v8_yolof_top5votedisagree_sampleweight_w082_val_20260704`
  reproduced the full-val metrics and exported predictions. Boundary review
  `runs\boundary_review_smoke_v8_yolof_top5votedisagree_sampleweight_w082_val_20260704`
  selected `164` rows (`focus_false_positive=45`,
  `focus_false_negative=33`, `low_margin_error=41`,
  `low_margin_correct_boundary=45`), with buckets including `0->1=27`,
  `1->0=15`, `1->2=12`, `4->1=7`, and `3->2=27`.
- XAI:
  `runs\xai_smoke_v8_yolof_top5votedisagree_sampleweight_w082_val_selected12_20260704`
  audited 12 stratified rows using the dataset-crop-fixed path. XAI again
  showed the same non-background bottleneck: background blur/gray drops were
  only `0.0070/0.0045`, while object desaturation drop was `0.1471`.
  Review flags remained border/near-tie heavy (`attention_background=6/12`,
  `attention_border=7/12`, `gradcam_border=8/12`,
  `rollout_background=7/12`, `near_tie=6/12`, `object_color_sensitive=6/12`).
- Cleanup and decision:
  reject this top-5 vote-disagreement sample-weight route before any probe.
  Metrics, resolved config, history, launcher transcript, best-val metrics, and
  architecture trace were copied to the XAI artifact under
  `source_smoke_metrics`; the smoke train directory was deleted via
  `runs\cleanup_manifest_20260704_top5votedisagree_sampleweight_rejected_smoke.json`,
  reclaiming `191.53 MB`. Do not repeat
  `build_top5_vote_disagreement_sample_weights` with the policy
  `class1_vote_weight=0.82/class1_vote_strong=0.72/
  majority_conflict=0.80/other=0.92 + V8 sample-weight smoke` on the current
  keeper. Future pretrained-disagreement transfer needs a stronger
  fold-safe/OOF case signal or a representation target, not small downweighting
  of a sparse in-sample expert-disagreement set.

## Diagnostic 2026-07-04 - EfficientNetV2-S Feature OOF Readout Reject

- Rationale: the top-5 vote-disagreement smoke showed that sparse in-sample
  uncertainty weights are not enough. I next tested whether the existing
  strictly remapped EfficientNetV2-S pretrained feature cache contains a
  fold-safe class-1 boundary signal before attempting any TRKH distillation or
  training. This is diagnostic only and does not use test.
- Implementation:
  added `trkh.tools.probe_pretrained_feature_oof_readout`. It loads
  `features_train_yolof.npz` and `features_val_yolof.npz`, trains several
  sklearn readouts on train features, selects the candidate by train OOF
  composite score only, writes selected train-OOF and validation predictions,
  and refuses test feature caches. Preflight passed
  `py_compile trkh\tools\probe_pretrained_feature_oof_readout.py` and
  `pytest tests\test_probe_pretrained_feature_oof_readout.py -q` (`1 passed`).
- Diagnostic artifact:
  `runs\pretrained_feature_oof_readout_effv2s_yolof_20260704` used full
  `yolo_f/train=9215`, `val=2606`, feature dim `1280`, and 5-fold
  source-stem-aware OOF where possible. Candidate selection chose
  `logreg_balanced_c0p3` by train OOF. Train OOF looked unrealistically strong:
  macro/class-1 F1 `0.9960/0.9863`, class-1 P/R `0.9764/0.9963`.
- Validation result:
  the selected readout fell to validation macro/class-1 F1 `0.8876/0.6454`,
  class-1 P/R `0.6947/0.6026`, below the no-pretrain keeper class-1
  `0.6860`. The best validation class-1 among reported candidates was only
  `knn_cosine_k31` at `0.6691`, and the best validation macro was
  `extra_trees_leaf5` at `0.8917/0.6667`; both remain below the keeper class-1
  gate. The selected confusion was `0->1=31`, `1->0=20`, `1->2=35`,
  `3->2=41`, `4->1=5`.
- Decision:
  reject frozen EfficientNetV2-S feature OOF readouts as a compression target
  for the current keeper. The diagnostic explains part of the gap: pretrained
  features are extremely separable on train/OFF folds but do not carry the
  val/test surface-boundary decision reliably enough by themselves. Do not
  train another TRKH smoke from this readout or direct soft-target/KD from this
  selected OOF classifier. Future pretrained transfer needs either genuinely
  fold-held-out deep expert predictions or a representation objective that
  demonstrably improves validation class-1 before smoke.

## Smoke 2026-07-04 - Cumulative Ordinal Head Reject

- Rationale: after rejecting frozen readouts and sparse pretrained-disagreement
  sample weights, I tested a different loss family that is not the old
  ordinal-distribution/EMD smoothing recipe. The idea follows ordinal
  regression work such as CORAL (`https://arxiv.org/abs/1901.07884`), CORN
  (`https://arxiv.org/abs/2111.08851`), and cumulative-link ordinal
  classifiers (`https://arxiv.org/abs/1905.13392`): classes `0,1,2,3` have a
  partial maturity order, so a checkpoint-safe cumulative threshold head might
  sharpen the `0|1` and `1|2` boundaries without touching class `4`.
- Preflight:
  no code change was needed. Existing cumulative-ordinal infrastructure was
  checked with `py_compile trkh\models\model.py trkh\training\train.py
  trkh\core\config.py`, PowerShell V8 launcher parse, and
  `pytest tests\test_cumulative_ordinal_head.py -q` (`3 passed`).
- Smoke:
  `smoke_v8_yolof_cumulativeordinal_c0123_w006_s018_20260704` resumed the
  current keeper checkpoint on explicit `yolo_f`, with `BBoxSpatialFusion=true`,
  `BboxTokenPriorSource=crop_bbox`, teacher-focus-binary `0.015`, empty hard
  manifest, `SkipFinalTest`, full validation support `2606`, and
  `CumulativeOrdinalHead=true/classes=0,1,2,3/logit_scale=0.18/
  dropout=0.02/loss_weight=0.006/threshold_weights=1.7,1.7,0.6`.
  The new head loaded as an allowed missing architecture extension and the
  cumulative ordinal loss was active (`train_cumulative_ordinal_loss=0.6932`).
- Result:
  trainer best validation fell to macro/class-1 F1 `0.8762/0.6574`,
  class-1 P/R `0.5673/0.7815`, well below the keeper `0.8847/0.6860`.
  The standalone eval CSV used for boundary/XAI reproduced the same failure
  neighborhood at macro/class-1 F1 `0.8771/0.6592`, class-1 P/R
  `0.5700/0.7815`. Confusions widened the exact class-1 false-positive issue:
  `0->1` rose to about `56-57`, `4->1=16`, `2->1=13`, while true class-1
  misses stayed `1->0=15`, `1->2=12`, `1->4=6`.
- Audit:
  boundary review
  `runs\boundary_review_smoke_v8_yolof_cumulativeordinal_c0123_w006_s018_val_20260704`
  selected `165` rows with `focus_false_positive=45`,
  `focus_false_negative=33`, `low_margin_error=42`, and
  `low_margin_correct_boundary=45`. XAI
  `runs\xai_smoke_v8_yolof_cumulativeordinal_c0123_w006_s018_val_selected12_20260704`
  again showed the same non-background bottleneck: background blur/gray drops
  were only `0.0121/0.0102`, while object desaturation drop was `0.1357`.
  Review flags remained border/near-tie/object-color heavy
  (`gradcam_border=7/12`, `rollout_border=6/12`, `near_tie=6/12`,
  `object_color_sensitive=6/12`). Manual overlay checks showed false-positive
  class-1 heat on fruit edge/bottom-surface regions and false-negative class-1
  heat spilling to leaf/background adjacent to the fruit.
- Cleanup and decision:
  reject cumulative ordinal head/loss before probe. Metrics, resolved config,
  history, launcher transcript, best-val metrics, and architecture trace were
  copied under the XAI artifact's `source_smoke_metrics`; the smoke train
  directory was deleted via
  `runs\cleanup_manifest_20260704_cumulativeordinal_rejected_smoke.json`,
  reclaiming `182.70 MB`. Do not repeat this exact
  `CumulativeOrdinalHead=true/classes=0,1,2,3/logit_scale=0.18/
  loss_weight=0.006/threshold_weights=1.7,1.7,0.6/LR=1e-5/60b`
  route on the current keeper. More ordinal/logit-shaping variants are low
  priority unless paired with a new fold-safe reliability or representation
  signal, because this route behaved like earlier loss-level guards and widened
  class-1 false positives.

## Diagnostic 2026-07-04 - ConvNeXt Tiny Pretrained Head-Only Reject

- Rationale: after several no-pretrain loss/logit/readout routes failed, I used
  a pretrained ConvNeXt Tiny as a representation diagnostic only, not as a final
  TRKH replacement and not using test. ConvNeXt is relevant because the primary
  paper (`https://arxiv.org/abs/2201.03545`) modernizes CNN design toward
  Transformer-style training/architecture while retaining convolutional
  inductive bias, and the timm `convnext_tiny.fb_in22k_ft_in1k` weights are
  ImageNet-22k pretrained and ImageNet-1k fine-tuned.
- First smoke:
  `smoke_timm_convnexttiny_fb22k_yolof_ce_80b_1e_20260704` trained all
  parameters for `80` batches, but default warmup kept the effective LR too low
  for a one-epoch smoke. Validation macro/class-1 F1 was only
  `0.3927/0.1911`; treat this as an underfit configuration check, not an
  architecture conclusion.
- Head-only smoke:
  `smoke_timm_convnexttiny_fb22k_yolof_headonly_lr1e3_160b_20260704` froze the
  pretrained backbone and trained only `head` for `160` batches with
  `LR=1e-3`, no warmup, no class-aware crop margin, and full `yolo_f/val=2606`.
  It reached only validation macro/class-1 F1 `0.7243/0.3261`, far below the
  TRKH keeper `0.8847/0.6860`. Class-1 precision/recall were
  `0.2318/0.5497`, with severe false positives: `0->1=122`, `4->1=101`,
  `2->1=47`, plus true class-1 misses `1->0=44`, `1->2=23`.
- Audit and code fix:
  standalone eval
  `runs\eval_smoke_timm_convnexttiny_fb22k_yolof_headonly_val_20260704`
  reproduced macro F1 `0.7247`. Boundary review
  `runs\boundary_review_smoke_timm_convnexttiny_fb22k_yolof_headonly_val_20260704`
  selected `180` rows with `focus_false_negative=45`,
  `focus_false_positive=45`, `high_confidence_error=45`, and
  `low_margin_error=36`. While auditing, `xai_audit` and `attention_viz`
  were fixed to avoid calling TRKH metadata arguments on timm-style
  `forward_features`; preflight passed `py_compile` and
  `pytest tests\test_xai_audit_dataset_tensor.py -q` (`4 passed`).
- XAI:
  `runs\xai_smoke_timm_convnexttiny_fb22k_yolof_headonly_val_cases8_20260704`
  audited 8 stratified boundary rows. Because timm ConvNeXt has no TRKH
  attention rollout path, feature-map attention is reported as fallback, while
  Grad-CAM and robustness are usable. Grad-CAM foreground/background mass was
  only `0.6758/0.3242`, much less object-focused than the TRKH keeper audits.
  Object desaturation was highly causal (`pred prob drop=0.4156`), while
  background blur/gray were smaller (`0.0239/0.0077`). Review flags showed
  background/feature-map attention on all 8 cases, `gradcam_background=4/8`,
  `object_color_sensitive=7/8`, and manual overlays confirmed pad/background
  and edge/surface reliance on representative `0/1` and `4->1` errors.
- Cleanup and decision:
  reject ConvNeXt Tiny pretrained head-only as a candidate direction before any
  longer probe. Metrics/config/history from both ConvNeXt smoke directories
  were copied to the XAI artifact under `source_smoke_metrics`; the two smoke
  train directories were deleted via
  `runs\cleanup_manifest_20260704_convnexttiny_pretrained_rejected_smokes.json`,
  reclaiming `661.22 MB`. Do not repeat the exact
  `convnext_tiny.fb_in22k_ft_in1k/head-only/LR=1e-3/160b/no-warmup` or
  one-epoch warmup-heavy all-parameter ConvNeXt Tiny smoke on the current data.
  This diagnostic says a generic pretrained CNN head alone does not explain or
  solve the class-1 boundary; future pretrained work must use stronger
  fold-safe expert disagreement/representation transfer, or a carefully
  fine-tuned pretrained model with explicit class-1 false-positive control,
  before it can inform the no-pretrain TRKH model.

## Smoke 2026-07-04 - ConvNeXt Tiny Stage3+Head Partial-AUC Reject

- Rationale: the head-only ConvNeXt diagnostic was weak, but it did not test a
  real representation update. I ran one materially different pretrained smoke:
  unfreeze only `stages.3` plus `head`, add an explicit focus-class partial-AUC
  loss to suppress the high-risk negative tail `0/2/4 -> 1`, and protect
  bottom-scoring true class-1 positives. This is still diagnostic only and does
  not use test.
- Preflight:
  `py_compile trkh\training\train.py trkh\core\config.py trkh\models\model.py`
  passed, as did
  `pytest tests\test_focus_partial_auc_loss.py tests\test_timm_classifier_model.py
  tests\test_trainable_module_prefixes.py -q` (`10 passed`).
- Smoke:
  `smoke_timm_convnexttiny_fb22k_yolof_stage3head_pauc_w002_lr2e4_160b_20260704`
  used explicit `yolo_f`, `convnext_tiny.fb_in22k_ft_in1k`, trainable prefixes
  `stages.3,head` (`15.48M` trainable params), no warmup, `LR=2e-4`, `160`
  train batches, full validation `2606`, disabled class-aware crop margin and
  random erasing/affine, CE label smoothing `0.02`, and partial-AUC
  `weight=0.02/margin=0.05/temp=0.12/neg_fraction=0.35/
  pos_fraction=0.50/pos_weight=0.70/min_negative_p=0.02`.
  Partial-AUC telemetry was active (`train_focus_partial_auc_loss=0.8778`,
  positive/negative selected counts about `3.4/5.45`).
- Result:
  validation improved over head-only but remained far below the TRKH keeper:
  macro/class-1 F1 `0.8146/0.4595`, class-1 P/R `0.3793/0.5828` versus keeper
  `0.8847/0.6860`. Confusion still had many class-1 false positives and misses:
  `0->1=78`, `4->1=33`, `2->1=29`, `1->0=34`, `1->2=24`, `1->4=5`.
  Standalone eval reproduced macro `0.8142` and calibrated macro `0.8298`;
  calibration did not make it competitive.
- Audit:
  boundary review
  `runs\boundary_review_smoke_timm_convnexttiny_stage3head_pauc_val_20260704`
  selected `164` rows, including `focus_false_negative=45`,
  `focus_false_positive=45`, `high_confidence_error=45`, and
  `low_margin_error=13`. XAI
  `runs\xai_smoke_timm_convnexttiny_stage3head_pauc_val_cases8_20260704`
  showed Grad-CAM foreground/background `0.8254/0.1746`, attention fallback
  foreground/background `0.2500/0.7500`, background blur/gray drops
  `0.0169/0.0056`, and object desaturation drop `0.5266`. Flags included
  `attention_background=6/8`, `gradcam_background=2/8`, `gradcam_border=2/8`,
  and `object_color_sensitive=8/8`. Manual overlays showed strong heat on
  fruit edge/defect/color patches and adjacent/padded background, not a stable
  class-1 surface cue.
- Cleanup and decision:
  reject this stage3+head partial-AUC ConvNeXt Tiny route before probe. Metrics
  were copied to the XAI artifact under `source_smoke_metrics`; the smoke train
  directory was deleted via
  `runs\cleanup_manifest_20260704_convnexttiny_stage3head_pauc_rejected_smoke.json`,
  reclaiming `342.64 MB`. Do not repeat nearby
  `ConvNeXt Tiny stages.3+head + partial-AUC/FPR focus loss` smokes on the
  current data. The route proves that a light pretrained representation update
  can beat head-only but is still much worse than the no-pretrain TRKH keeper
  and remains edge/background/color sensitive.

## Diagnostic 2026-07-04 - DINOv2 Remapped Feature OOF Readout Reject

- Rationale:
  after the EfficientNetV2-S frozen readout failed to give a reliable class-1
  compression target, I tested whether DINOv2 self-supervised ViT features carry
  a different fold-safe object-surface signal. The source direction is relevant
  because DINO/DINOv2 learn visual features without labels
  (`https://arxiv.org/abs/2104.14294`, `https://arxiv.org/abs/2304.07193`),
  but this diagnostic still does not use test and is not a candidate final
  model.
- Tooling:
  `trkh.tools.evaluate_embedding_retrieval` now supports `yolo_f` object-crop
  extraction through a picklable classification-transform adapter and returns
  `sample_index/source_stem` metadata in feature caches. Preflight passed
  `py_compile trkh\tools\evaluate_embedding_retrieval.py
  trkh\tools\probe_pretrained_feature_oof_readout.py` and focused pytest
  `tests\test_evaluate_embedding_retrieval_yolo.py
  tests\test_probe_pretrained_feature_oof_readout.py` (`2 passed`), plus a
  follow-up adapter fix with `tests\test_evaluate_embedding_retrieval_yolo.py`
  (`1 passed`).
- Smoke:
  `runs\embedding_retrieval_dinov2_small_yolof_smoke_20260704` used direct
  `yolo_f` object crops with `vit_small_patch14_dinov2`, but only
  `15/15` train/val samples (`max_samples_per_class=3`) because DINOv2-small
  uses a `518x518` input and full direct extraction would be slow. The smoke
  selected `logreg c=0.1` at val macro/class-1 F1 `0.5603/0.5000`; this is a
  pipeline check only, not a gate metric.
- Remap:
  I reused the existing full `class_f` DINOv2-small feature cache and strictly
  remapped it to `yolo_f` sample-index order. Coverage was complete:
  train `9215/9215`, val `2606/2606`, feature dim `384`, missing rows `0`.
  The remapped features live in
  `runs\dinov2_small_features_yolof_remap_20260704`.
- OOF readout:
  `runs\pretrained_feature_oof_readout_dinov2small_yolof_20260704` selected
  `logreg_balanced_c0p3` by train OOF only. Train OOF reached
  macro/class-1 F1 `0.8470/0.5167`, class-1 P/R `0.4452/0.6155`.
  Validation reached only macro/class-1 F1 `0.8369/0.5112`, class-1 P/R
  `0.4439/0.6026`, far below the keeper `0.8847/0.6860`. Main validation
  class-1 confusions were `0->1=80`, `1->0=26`, `1->2=27`, `2->1=21`,
  `4->1=9`, and `3->2=48`. Other candidates were also worse
  (`logreg_c1` class-1 `0.5069`, `extra_trees_leaf5` `0.4286`,
  `knn_cosine_k31` `0.3487`).
- Complementarity audit:
  `runs\dinov2small_vs_keeper_complementarity_val_20260704` shows real but
  unsafe validation oracle complementarity. Against the current keeper eval
  (`0.8829/0.6783` on that CSV), DINOv2 rescues `13/34` keeper class-1 false
  negatives and correctly blocks `37/77` keeper class-1 false positives. But
  DINOv2 itself predicts class 1 for `205` validation rows with precision only
  `0.4439`, and the oracle macro/class-1 `0.9322/0.8100` uses validation
  labels, so it is not deployable.
- Decision:
  reject DINOv2-small remapped `class_f -> yolo_f` feature OOF readout as a
  current teacher/router target. Do not train direct soft targets, KD, sample
  weights, or a validation threshold router from this selected DINOv2 readout.
  If DINO is revisited, it needs genuine full `yolo_f` extraction or a
  materially different fold-safe signal with pre-evidence above the keeper, not
  the remapped frozen readout.
- Cleanup:
  the direct 15/15 `yolo_f` DINOv2 smoke metrics were copied to
  `runs\pretrained_feature_oof_readout_dinov2small_yolof_20260704\source_yolof_smoke`,
  then the obsolete smoke directory was deleted via
  `runs\cleanup_manifest_20260704_dinov2_yolof_smoke_deleted.json`
  (`0.07 MB` reclaimed). Keep the remap, OOF readout, and complementarity
  artifacts as the live evidence.

## Diagnostic 2026-07-04 - YOLO OOF Folds + MobileNetV3 Fold00 Reject

- Rationale:
  the pretrained/AIDT gap may require a fold-safe teacher rather than in-sample
  KD or validation-threshold routing. I built a current `yolo_f/train` 5-fold
  OOF dataset artifact, grouped by source image and materialized by hardlinks,
  then tested whether a lightweight pretrained MobileNetV3 fold expert was
  strong enough to justify full 5-fold OOF teacher generation. No raw dataset
  files were modified.
- Fold artifact:
  `runs\yolof_oof_folds_train5_20260704` was built from
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, split `train`, `folds=5`,
  `seed=42`, `link_mode=auto`. Source coverage was `8064` images and `9215`
  objects with class counts `[1941,541,1920,2520,2293]`; materialization was
  `hardlink=96768`, `copy=0`. Fold00 validation has `1858` objects, class
  counts `[390,108,387,503,470]`.
- Tooling/preflight:
  focused fold-tool tests passed (`tests\test_build_yolo_oof_folds.py` and
  `tests\test_stitch_yolo_oof_predictions.py`, `3 passed`). `attention_viz`
  now disables `inplace=True` activation modules before hooked Grad-CAM or
  grad-rollout, needed for timm MobileNetV3 Hardswish backward hooks; preflight
  passed `py_compile trkh\evaluation\attention_viz.py trkh\evaluation\xai_audit.py`
  and `tests\test_xai_audit_dataset_tensor.py` (`5 passed`).
- Probe:
  `probe_timm_mobilenetv3_yolof_oof_fold00_plain_3e_20260704` used
  `mobilenetv3_large_100.ra_in1k`, ImageNet pretrained, `image_size=224`,
  `epochs=3`, `batch_size=64`, CE label smoothing `0.02`, no balanced epoch
  sampling, no class-aware crop margin, no random erasing, and full fold00
  validation. It was run as diagnostic only with `SkipFinalTest`.
- Result:
  best epoch was `3`, validation macro/class-1 F1 `0.8648/0.5389`, class-1
  P/R `0.6118/0.4815`, far below the current no-pretrain keeper
  `0.8847/0.6860`. Epoch 2 had slightly higher class-1 F1 `0.5441`, but only
  by low precision (`0.4512`) and still far below the gate. Standalone eval
  reproduced macro F1 `0.8648`; confusion matrix rows were
  `[359,23,1,0,7]`, `[42,52,13,1,0]`, `[4,7,368,6,2]`,
  `[0,0,12,491,0]`, `[11,3,2,0,454]`.
- Audit:
  boundary review
  `runs\boundary_review_probe_timm_mobilenetv3_yolof_oof_fold00_plain_val_20260704`
  selected `86` rows: `focus_false_negative=30`, `focus_false_positive=26`,
  `high_confidence_error=23`, `low_margin_error=3`, and
  `low_margin_correct_boundary=4`. Top buckets included `1->0=23`,
  `1->2=6`, `0->1=16`, `2->1=7`, and `4->1=3`.
- XAI:
  `runs\xai_probe_timm_mobilenetv3_yolof_oof_fold00_plain_val_cases8_20260704`
  audited 8 explicit class-1 errors. All were high-confidence class-1 misses
  (`confidence=0.9589-0.9973`) predicted as class `0` or `2`. Because timm
  MobileNetV3 has no TRKH attention rollout path, attention is a feature-map
  fallback and was all-background by the pseudo-foreground metric. Grad-CAM was
  weaker than TRKH keeper audits: foreground/background/border mass
  `0.7556/0.2444/0.2878`, flags `attention_background=8/8`,
  `gradcam_background=7/8`, `gradcam_border=8/8`, and
  `high_confidence_misclassification=8/8`. Case metadata records disabled
  inplace modules such as `bn1.act` and `blocks.*.act`, confirming the timm
  Grad-CAM fix was active.
- Cleanup and decision:
  reject plain MobileNetV3-yolo OOF fold00 as a teacher route and do not expand
  it to 5 folds. Metrics/config/history/logs were preserved under
  `runs\xai_probe_timm_mobilenetv3_yolof_oof_fold00_plain_val_cases8_20260704\source_probe_metrics`;
  the command-check and probe train directories plus copied process logs were
  deleted via
  `runs\cleanup_manifest_20260704_mobilenetv3_oof_fold00_rejected_probe.json`,
  reclaiming `154.97 MB`. If pretrained OOF is revisited, it needs a materially
  stronger fold-safe expert or representation target before spending full
  5-fold training time.

## Smoke 2026-07-04 - EfficientNetV2-S Fold00 Reload/XAI Reject

- Rationale:
  after MobileNetV3 fold00 was too weak, I tested the stronger top-5-era
  `tf_efficientnetv2_s.in21k_ft_in1k` model as a fold-safe pretrained expert
  candidate. EfficientNetV2 is relevant as a stronger CNN family with fused
  MBConv-style blocks and progressive training ideas (`https://arxiv.org/abs/2104.00298`),
  but this run was still diagnostic only: fold00 validation only, no test.
- Command-check:
  `micro_timm_effv2s_yolof_fold00_cmdcheck_20260704` verified model load and
  command wiring. The first call downloaded the timm/HuggingFace weights; this
  run was later preserved and deleted after the real smoke.
- Smoke:
  `smoke_timm_effv2s_yolof_oof_fold00_plain_60b_2e_20260704` used explicit
  fold00 `yolo_f`, `image_size=224`, `batch_size=32`, `60` train batches per
  epoch, `2` epochs, `LR=5e-4`, no warmup, CE label smoothing `0.02`, focal/LDAM
  disabled, balanced epoch sampling disabled, rare-repeat disabled, and
  class-aware augmentation disabled. The trainer still applied the normal
  class-1 crop-margin policy, so this was not undercut by removing all class-1
  support.
- Trainer result:
  history improved from validation macro/class-1 `0.7696/0.3388` at epoch 1 to
  `0.8205/0.3804` at epoch 2, with class-1 P/R `0.4605/0.3241`. This is already
  far below MobileNetV3 fold00 (`0.8648/0.5389`) and the no-pretrain keeper
  (`0.8847/0.6860`).
- Reload eval:
  the deployable checkpoint artifact was worse when reloaded through the normal
  evaluation path. Standalone eval
  `runs\eval_smoke_timm_effv2s_yolof_oof_fold00_plain_val_20260704` reached only
  macro/class-1 F1 `0.6452/0.1120`, class-1 P/R `0.4118/0.0648`, accuracy
  `0.7670`. The confusion matrix rows were
  `[217,2,27,104,40]`, `[44,7,26,15,16]`, `[6,5,332,33,11]`,
  `[0,0,45,450,8]`, `[14,3,18,16,419]`. This reload mismatch makes the route
  unsafe as an OOF teacher even before considering its low trainer metric.
- Audit:
  boundary review
  `runs\boundary_review_smoke_timm_effv2s_yolof_oof_fold00_plain_val_20260704`
  selected `100` rows with `focus_false_negative=25`,
  `focus_false_positive=10`, `high_confidence_error=25`,
  `low_margin_error=25`, and `low_margin_correct_boundary=15`. Top reload
  confusions included `0->3=104`, `3->2=46`, `1->0=44`, `0->4=41`,
  `2->3=33`, `1->2=26`, and `1->4=16`.
- XAI:
  `runs\xai_smoke_timm_effv2s_yolof_oof_fold00_plain_val_cases8_20260704`
  audited 8 explicit class-1 false negatives. All were predicted as class
  `0/2/4` with high confidence (`0.8973-0.9653`). Attention was heavily
  border/background biased: attention foreground/background/border
  `0.5176/0.4824/0.6663`, Grad-CAM foreground/background/border
  `0.7072/0.2928/0.4177`, flags `attention_border=8/8`,
  `attention_background=7/8`, `gradcam_border=8/8`, and
  `gradcam_background=6/8`. Case metadata again confirms the timm inplace
  activation hook fix was active.
- Cleanup and decision:
  reject EfficientNetV2-S fold00 quick OOF expert route and do not expand it to
  5 folds. Metrics/config/history/eval/boundary/XAI artifacts were preserved
  under
  `runs\xai_smoke_timm_effv2s_yolof_oof_fold00_plain_val_cases8_20260704\source_smoke_metrics`;
  the micro and smoke train directories were deleted via
  `runs\cleanup_manifest_20260704_effv2s_oof_fold00_rejected_smoke.json`,
  reclaiming `644.44 MB`. If EfficientNetV2 is revisited, it needs either a
  full, reload-stable fine-tuning recipe with evidence near the keeper or a
  different fold-safe representation signal; do not spend 5-fold OOF time on
  this quick smoke family.

## Correction 2026-07-04 - EfficientNetV2-S Normfix Reload/XAI Reject

- Correction:
  the previous EfficientNetV2-S standalone eval/XAI artifact used the normal
  evaluator before it propagated checkpoint `model_config.input_mean/std` into
  `build_eval_transform`. `tf_efficientnetv2_s.in21k_ft_in1k` trains with timm
  normalization `(0.5,0.5,0.5)/(0.5,0.5,0.5)`, while the old reload path used
  ImageNet defaults. Therefore the old reload metric `0.6452/0.1120` and its
  XAI foreground/background masses are invalid evidence, although the trainer
  history was already far below the keeper.
- Fix:
  added `trkh.evaluation.input_normalization.checkpoint_input_normalization`
  and wired it through `trkh\evaluation\evaluate.py`,
  `trkh\evaluation\xai_audit.py`, and `trkh\evaluation\attention_viz.py`.
  XAI tensor-to-crop reconstruction now also accepts the same normalization.
  Preflight passed:
  `py_compile trkh\evaluation\input_normalization.py
  trkh\evaluation\evaluate.py trkh\evaluation\xai_audit.py
  trkh\evaluation\attention_viz.py` and
  `pytest tests\test_xai_audit_dataset_tensor.py -q` (`8 passed`).
- Normfix smoke:
  reran the same quick fold00 recipe as
  `smoke_timm_effv2s_yolof_oof_fold00_plain_normfix_60b_2e_20260704`.
  Trainer best epoch `2` reached only fold00 validation macro/class-1 F1
  `0.8125/0.3497`, class-1 P/R `0.4267/0.2963`. The independent reload eval
  `runs\eval_smoke_timm_effv2s_yolof_oof_fold00_plain_normfix_val_20260704`
  now matches trainer at macro F1 `0.8128`; the normalization mismatch is fixed.
  Confusion rows were `[361,21,1,0,7]`,
  `[52,32,22,1,1]`, `[3,8,370,4,2]`,
  `[0,0,27,475,1]`, `[13,14,4,1,438]`.
- Boundary/XAI:
  boundary review
  `runs\boundary_review_smoke_timm_effv2s_yolof_oof_fold00_plain_normfix_val_20260704`
  selected `108` rows: `focus_false_negative=30`,
  `focus_false_positive=30`, `high_confidence_error=30`,
  `low_margin_error=4`, `low_margin_correct_boundary=14`. Buckets included
  `1->0=23`, `0->1=18`, `3->2=18`, `1->2=7`, `2->1=7`.
  Robust XAI
  `runs\xai_smoke_timm_effv2s_yolof_oof_fold00_plain_normfix_val_cases12_robust_20260704`
  used explicit sample indices from the boundary review. It showed attention
  foreground/background/border `0.5433/0.4567/0.6339`, Grad-CAM
  `0.7112/0.2888/0.4234`, flags `attention_border=12/12`,
  `attention_background=9/12`, `gradcam_border=12/12`,
  `gradcam_background=9/12`, `high_confidence_misclassification=8/12`, and
  `object_color_sensitive=12/12`. Robustness drops were
  `background_blur=0.0323`, `background_gray=0.0007`,
  `center_occlusion=0.0050`, and `object_desaturate=0.5381`.
  Manual overlays show heat on pad/background/hand/fruit border rather than a
  stable class-1 surface cue.
- Cleanup and decision:
  reject EfficientNetV2-S fold00 quick OOF route after the normalization fix as
  well. The run is weaker than MobileNetV3 fold00 and much weaker than the
  no-pretrain TRKH keeper, so it must not be expanded to 5-fold OOF and must
  not be used as teacher evidence. Metrics were copied under the robust XAI
  artifact; the smoke train directory, duplicate non-robust XAI artifact, and
  process logs were deleted via
  `runs\cleanup_manifest_20260704_effv2s_normfix_rejected_smoke.json`,
  reclaiming `327.65 MB`.

## Smoke 2026-07-04 - CoAtNet-0 Fold00 Hybrid CNN-Attention Reject

- Rationale:
  the user asked to test stronger CNN/Transformer hybrid routes and the value of
  `yolo_f` context. CoAtNet is directly relevant because the paper proposes a
  hybrid family that combines convolutional inductive bias with attention
  capacity via depthwise-conv/relative-attention unification and staged
  conv-attention stacking (`https://arxiv.org/abs/2106.04803`). I used timm
  `coatnet_0_rw_224.sw_in1k` as a fold00 diagnostic only, not as a final model
  or test-tuned route.
- Command-check:
  `micro_timm_coatnet0_yolof_fold00_cmdcheck_20260704` verified that the timm
  model loads, downloads its weights, uses `(0.5,0.5,0.5)` normalization, and
  can run through the TRKH timm-classifier path. The micro directory was later
  deleted after metrics were preserved.
- Smoke:
  `smoke_timm_coatnet0_yolof_oof_fold00_plain_60b_2e_20260704` used fold00
  `yolo_f`, `image_size=224`, `batch_size=24`, `60` train batches per epoch,
  `2` epochs, `LR=3e-4`, no warmup, CE label smoothing `0.02`, focal/LDAM
  disabled, balanced epoch sampling disabled, rare-repeat disabled, and
  class-aware augmentation disabled. This kept the same wide-context object-crop
  path and checkpoint normalization fix used for the EfficientNetV2 normfix
  rerun.
- Metrics:
  trainer best epoch `2` reached fold00 validation macro/class-1 F1
  `0.7892/0.2550`, class-1 P/R `0.4634/0.1759`. Independent reload eval
  `runs\eval_smoke_timm_coatnet0_yolof_oof_fold00_plain_val_20260704`
  reproduced the result at macro/class-1 F1 `0.7916/0.2649`, class-1 P/R
  `0.4651/0.1852`, accuracy `0.8994`. Confusion rows were
  `[370,13,0,0,7]`, `[67,20,17,1,3]`, `[2,6,347,27,5]`,
  `[0,0,19,483,1]`, `[11,4,4,0,451]`. The main failure is severe class-1
  under-recall, especially `1->0` and `1->2`.
- Boundary/XAI:
  boundary review
  `runs\boundary_review_smoke_timm_coatnet0_yolof_oof_fold00_plain_val_20260704`
  selected `120` rows: `focus_false_negative=30`,
  `focus_false_positive=23`, `high_confidence_error=30`,
  `low_margin_error=9`, and `low_margin_correct_boundary=28`. Buckets included
  `1->0=14`, `1->2=13`, `1->4=2`, `1->3=1`, `0->1=13`, `2->1=6`, and
  `4->1=4`.
  Robust XAI
  `runs\xai_smoke_timm_coatnet0_yolof_oof_fold00_plain_val_cases12_robust_20260704`
  audited 12 explicit class-1 false negatives. It showed attention
  foreground/background/border `0.0833/0.9167/0.0000`, Grad-CAM
  `0.7480/0.2520/0.0129`, flags `attention_background=11/12`,
  `gradcam_background=3/12`, `object_color_sensitive=10/12`, and
  `register_cls_attention_divergent=12/12`. Robustness drops were
  `background_blur=-0.0003`, `background_gray=-0.0010`,
  `center_occlusion=0.0338`, and `object_desaturate=0.4745`. Manual overlays
  show `1->4` focusing on local dark defect spots and `1->2` cases with very
  weak heat, so this is not a useful context/background separation signal.
- Cleanup and decision:
  reject plain CoAtNet-0 fold00 OOF route and do not expand it to 5-fold. It is
  much weaker than MobileNetV3 fold00, EfficientNetV2 normfix, and the TRKH
  keeper, and its XAI is worse than the no-pretrain keeper for attention
  foreground. Metrics were copied into
  `runs\xai_smoke_timm_coatnet0_yolof_oof_fold00_plain_val_cases12_robust_20260704\source_smoke_metrics`;
  the micro command-check, smoke train directory, and process logs were deleted
  via `runs\cleanup_manifest_20260704_coatnet0_fold00_rejected_smoke.json`,
  reclaiming `839.10 MB`.

## Smoke 2026-07-04 - MaxViT-256 Fold00 Hybrid Local-Global Attention Reject

- Rationale:
  after CoAtNet-0 failed class 1, I tested a second hybrid pretrained family
  with a stronger explicit local-global design. MaxViT combines MBConv-style
  convolution with blocked local attention and dilated global grid attention
  (`https://arxiv.org/abs/2204.01697`). This was a fold00 diagnostic to answer
  whether a generic local-global pretrained hybrid could use `yolo_f` context
  better than TRKH; it was not a replacement final model.
- Command-check:
  `micro_timm_maxvit256_yolof_fold00_cmdcheck_20260704` verified timm
  `maxvit_rmlp_tiny_rw_256.sw_in1k`, `image_size=256`, checkpoint
  normalization `(0.5,0.5,0.5)`, and the trainer path. The first run also
  downloaded the weights.
- Smoke:
  `smoke_timm_maxvit256_yolof_oof_fold00_plain_60b_2e_20260704` used fold00
  `yolo_f`, `batch_size=16`, `60` train batches per epoch, `2` epochs,
  `LR=2e-4`, no warmup, CE label smoothing `0.02`, focal/LDAM disabled,
  balanced epoch sampling disabled, rare-repeat disabled, class-aware
  augmentation disabled, and no final test.
- Metrics:
  trainer best epoch `2` reached fold00 validation macro/class-1 F1
  `0.7447/0.0339`, class-1 P/R `0.2000/0.0185`. Independent reload eval
  `runs\eval_smoke_timm_maxvit256_yolof_oof_fold00_plain_val_20260704`
  matched at macro/class-1 F1 `0.7452/0.0339`, class-1 P/R
  `0.2000/0.0185`, accuracy `0.8977`. Confusion rows were
  `[378,3,2,0,7]`, `[76,2,25,1,4]`, `[3,2,344,34,4]`,
  `[0,0,8,493,2]`, `[15,3,1,0,451]`; only `2/108` class-1 samples were
  correct.
- Boundary/XAI:
  boundary review
  `runs\boundary_review_smoke_timm_maxvit256_yolof_oof_fold00_plain_val_20260704`
  selected `88` rows: `focus_false_negative=30`,
  `focus_false_positive=8`, `high_confidence_error=30`,
  `low_margin_error=9`, and `low_margin_correct_boundary=11`. Buckets included
  `1->0=21`, `1->2=9`, `2->3=22`, `4->0=6`, and only a small number of
  class-1 false positives.
  Robust XAI
  `runs\xai_smoke_timm_maxvit256_yolof_oof_fold00_plain_val_cases12_robust_20260704`
  audited 12 high-confidence class-1 false negatives. It showed attention
  foreground/background/border `0.5661/0.4339/0.5104`, Grad-CAM
  `0.9872/0.0128/0.0899`, flags `attention_background=12/12`,
  `attention_border=12/12`, `object_color_sensitive=12/12`, and
  `register_cls_attention_divergent=12/12`. Robustness drops were
  `background_blur=0.0021`, `background_gray=0.0003`,
  `center_occlusion=0.0052`, and `object_desaturate=0.6510`. Manual overlay
  confirms MaxViT often looks at the object, but the learned surface/color cue
  maps true class 1 to class `0/2` with high confidence.
- Cleanup and decision:
  reject MaxViT-256 fold00 and do not expand it to 5-fold OOF. This run is even
  weaker than CoAtNet-0 and gives no teacher signal for TRKH class-1 recall.
  Metrics were copied under
  `runs\xai_smoke_timm_maxvit256_yolof_oof_fold00_plain_val_cases12_robust_20260704\source_smoke_metrics`;
  the micro command-check, smoke train directory, and process logs were deleted
  via `runs\cleanup_manifest_20260704_maxvit256_fold00_rejected_smoke.json`,
  reclaiming `900.41 MB`.

## Smoke 2026-07-04 - Patch-Evidence Router Summary-Stats Reject

- Rationale:
  the offline `0-1` patch-evidence verifier in
  `runs\patch_evidence_mil_yolof_keeper_01only_t060_m040_top4_cropbbox_oofteacher_20260703`
  remains diagnostically useful because its validation post-hoc route reached
  macro/class-1 F1 `0.8892/0.7073`, but prior trainable router attempts did
  not start from the same feature space. I reviewed attention/MIL-style routes
  (attention MIL `https://arxiv.org/abs/1802.04712`, CLAM
  `https://arxiv.org/abs/2004.09666`, DSMIL
  `https://arxiv.org/abs/2011.08939`, TransMIL
  `https://arxiv.org/abs/2106.00908`) and chose a conservative TRKH-native
  variant: keep the existing `PatchEvidenceRouterHead`, but add default-off
  verifier-like summary statistics instead of a free attention/MIL branch.
- Implementation:
  added `patch_evidence_router_summary_stats` defaulting to false in
  `trkh\models\model.py`, `trkh\core\config.py`,
  `trkh\training\train.py`, and
  `scripts\run_trkh_5class_attention_views_v8.ps1`. When enabled, the router
  descriptor includes pair-margin min/top-k/bbox-weighted statistics, positive
  patch fractions, and base pair probability/top-confidence statistics. The
  default path remains checkpoint-safe; zero-init residual still leaves base
  logits unchanged. Preflight passed
  `py_compile trkh\models\model.py trkh\training\train.py trkh\core\config.py`,
  `pytest tests\test_patch_evidence_router.py -q` (`7 passed`), and the V8
  launcher parse check.
- Smoke:
  first launch failed before training because the wrong bbox flag omitted
  `bbox_spatial_fusion_head` at checkpoint load. The corrected rerun used the
  current keeper checkpoint, explicit `yolo_f`, full validation support
  `2606`, `60` train batches, `1` epoch, trainable router head only
  (`168610` params), `PatchEvidenceRouterSummaryStats=true`, pair `0-1`,
  `topK=4`, bbox source `crop_bbox`, router logit scale `0.12`, hard-router
  loss `0.004`, hard positive weight `1.8`, OOF patch teacher CSV from the
  diagnostic artifact above, teacher loss `0.020`, and `SkipFinalTest`.
- Metrics:
  smoke best validation macro/class-1 F1 was only `0.8777/0.6611`, class-1
  P/R `0.5728/0.7815`, below keeper `0.8847/0.6860` and below the `0.70`
  probe gate. Confusion widened class-1 false positives:
  `[482,55,3,0,9]`, `[16,118,11,0,6]`,
  `[3,12,516,6,7]`, `[0,4,58,648,2]`,
  `[8,17,3,1,621]`. Independent reload eval
  `runs\eval_smoke_v8_yolof_patchsummaryrouter_oofteacher_headonly_w020_pw18_val_20260704`
  matched the reject decision at macro F1 `0.8789`; top confusions included
  `0->1=53`, `4->1=17`, `1->0=16`, `2->1=12`, and `1->2=11`.
- Boundary/XAI:
  boundary review
  `runs\boundary_review_smoke_v8_yolof_patchsummaryrouter_summary_oofteacher_val_20260704`
  selected `153` rows: `focus_false_negative=33`,
  `focus_false_positive=40`, `low_margin_error=40`, and
  `low_margin_correct_boundary=40`; boundary pairs were `0-1=80`,
  `1-2=25`, `2-3=35`, `4-rest=13`. Robust XAI
  `runs\xai_smoke_v8_yolof_patchsummaryrouter_summary_oofteacher_val_cases12_robust_20260704`
  audited 12 explicit hard cases, balanced between class-1 false negatives and
  high-confidence non-class1 false positives into class 1. Heatmap means were
  attention foreground/background/border `0.9318/0.0682/0.1112`, rollout
  `0.9052/0.0948/0.2453`, Grad-CAM `0.7934/0.2066/0.3264`. Robustness drops
  were background blur/gray only `0.0084/0.0067`, center occlusion `0.0397`,
  and object desaturate `0.2082`; flags included `object_color_sensitive=11/12`
  and `gradcam_border=8/12`. Manual overlays confirmed hot regions on object
  surface/edge, with some class-1 false negatives showing adjacent leaf/edge
  activation, not a reliable broad-background cue.
- Cleanup and decision:
  reject this summary-stat router recipe and do not run a longer probe. It
  compresses the offline verifier poorly and increases class-1 false positives
  while preserving the same surface/color bottleneck. Metrics/config/history
  were copied into
  `runs\xai_smoke_v8_yolof_patchsummaryrouter_summary_oofteacher_val_cases12_robust_20260704\source_smoke_metrics`;
  the failed partial smoke and corrected trained smoke directories were deleted
  via
  `runs\cleanup_manifest_20260704_patchsummaryrouter_summary_rejected_smoke.json`,
  reclaiming `133.35 MB`. Retained artifacts are the reload eval, boundary
  review, XAI robust package, and cleanup manifest.

## Diagnostic 2026-07-04 - AIDT and Source-Group Status Check

- User-facing clarification:
  external/AIDT runs are diagnostics and baselines, not a replacement for TRKH.
  The active target remains the no-pretrain TRKH branch on the valid
  `class_f`/`yolo_f` datasets.
- AIDT artifact check:
  `D:\DataAI\AIDT\runs\resnet50_vit_b16_5class` is a pretrained
  `ResNet50 + ViT-B/16` feature-fusion model with `112721733` parameters,
  image size `224`, `30` epochs, pretrained `resnet50.a1_in1k` and
  `vit_base_patch16_224.augreg2_in21k_ft_in1k`, sqrt-inverse class weights, and
  class order different from TRKH. Its stored test metrics on `class_f` are
  macro F1 `0.8841`, accuracy `0.9329`, and class-1
  `Xoai_Song_ChuaNhe_CoNguyCo` F1 `0.6225` (P/R `0.6026/0.6438`, support
  `73`). This is below the current TRKH no-pretrain final single-model test
  anchor `runs\eval_yolof_teacherfocusbinary015_best_test_final_20260702`
  (`0.8865/0.6667`). AIDT is therefore not currently a superior class-1 target;
  it is useful mainly for architecture/pretraining contrast.
- Source-group audit refresh:
  `runs\yolo_source_group_audit_current_20260704` confirmed `yolo_f/train`
  has `8064` source images and `9215` objects, with `691` multi-object images
  and `144` mixed-label images. Class 1 appears in only `517` train images and
  `115` train multi-object images; `102` of those are mixed-label. Validation
  has only `29` multi-object images and all are same-label class `0/3`; test is
  fully single-object (`1267/1267`). This repeats the earlier warning: global
  same-source consistency is unsafe, and source-group tricks cannot be
  validated as a class-1 object-level cue on val/test.
- Decision:
  do not spend more time treating AIDT as a stronger target unless the
  comparison is remapped and evaluated on the same object-level split. Future
  TRKH work should use AIDT/top-5 only as diagnostic/pretrained upper-bound or
  fold-safe teacher evidence, not as proof that broad context or source-group
  layout solves class 1.

## Status Clarification 2026-07-04 - External Smoke Runs Remain Diagnostic

- User question:
  clarified why recent work tested AIDT/timm pretrained/hybrid backbones while
  the active objective is TRKH. The answer is that those runs are diagnostic
  baselines to isolate whether pretraining, broad context, or generic hybrid
  conv-attention capacity explains class-1 performance. They are not a pivot
  away from the no-pretrain TRKH branch.
- Current working rule:
  keep TRKH as the active model target. External models may be used only to
  measure complementarity, find fold-safe teacher evidence, or rule out weak
  architecture families. A new smoke should not be launched from an external
  model unless it provides validation evidence near or above the TRKH keeper or
  a train-only/fold-safe signal that can be compressed back into TRKH without
  touching raw data or the test split.

## Diagnostic 2026-07-04 - Direct `yolo_f` DINOv2 Feature Readout Reject

- Rationale:
  the earlier DINOv2-small result was a strict `class_f -> yolo_f` feature
  remap, so I closed the user's wide-context question by extracting DINOv2
  features directly from `yolo_f` object crops. This stayed validation-only for
  method selection and used no test split.
- Preflight:
  `py_compile trkh\tools\evaluate_embedding_retrieval.py` passed, and focused
  tests
  `tests\test_evaluate_embedding_retrieval_yolo.py
  tests\test_probe_pretrained_feature_oof_readout.py -q` passed (`2 passed`).
- Extraction/probe:
  cap diagnostics `max_samples_per_class=10/50` were used only to verify
  runtime. The full extraction
  `runs\embedding_retrieval_dinov2_small_yolof_direct_full_20260704` covered
  train/val `9215/2606` samples with `vit_small_patch14_dinov2`, feature dim
  `384`, input `518x518`, and `--skip-test`. The val-selected retrieval looked
  superficially decent on macro F1 but still failed class 1:
  macro/class-1 `0.8867/0.6414`.
- Fold-safe readout:
  `runs\pretrained_feature_oof_readout_dinov2small_yolof_direct_full_20260704`
  selected `logreg_balanced_c0p3` by train OOF, not validation. Train OOF was
  macro/class-1 `0.8521/0.5263`; validation was only `0.8449/0.5260`, with
  class-1 precision/recall `0.4486/0.6358` and predicted class-1 support
  `214`. Confusions were still dominated by unsafe class-1 expansion:
  `0->1=86`, `2->1=24`, `1->0=25`, and `1->2=24`.
- Complementarity audit:
  `runs\dinov2small_direct_vs_keeper_complementarity_val_20260704` showed real
  oracle complementarity, but not a deployable selector. DINO direct rescued
  `13/34` keeper class-1 false negatives and correctly blocked `43/77`
  keeper class-1 false positives; the oracle reached class-1 F1 `0.8254`.
  However DINO direct itself had class-1 F1 only `0.5260`. A simple diagnostic
  rule, "if TRKH predicts class 1 and DINO predicts non-1, switch to DINO",
  changed `84` rows, made `43` corrections, but caused `34` harms, all true
  class-1 rows; class-1 F1 fell from `0.6783` to `0.6360`.
- Decision:
  reject direct `yolo_f` DINOv2-small feature readout as a teacher, router,
  KD target, or sample-weight source for the current keeper. It confirms that
  wide context and generic self-supervised ViT features contain some
  complementary evidence, but the selector would need a new fold-safe
  recall-protecting signal; DINO's own class-1 precision is too low to guide
  TRKH directly. The obsolete cap10/cap50 diagnostics were deleted via
  `runs\cleanup_manifest_20260704_dinov2_direct_cap_diagnostics_deleted.json`,
  reclaiming `0.94 MB`; the full extraction, OOF readout, and complementarity
  audit are preserved.

## Diagnostic 2026-07-04 - MobileNetV3 Fold00 No-Warmup/Fair Probe Reject

- Rationale:
  I revisited the earlier MobileNetV3 fold00 result only to answer whether a
  materially corrected fold-safe pretrained expert could become an OOF teacher
  for TRKH. This was still validation-only on
  `runs\yolof_oof_folds_train5_20260704\fold_00\data.yaml`; the test split was
  not used. The new run differed from the rejected plain probe by removing LR
  warmup and selecting checkpoints with `fair_macro_f1` instead of plain macro
  F1.
- Probe:
  `probe_timm_mobilenetv3_yolof_oof_fold00_nowarmup_fair_6e_20260704` used
  `mobilenetv3_large_100.ra_in1k`, pretrained timm weights, image size `224`,
  CE-style classification, sqrt-inverse class weights, no class-aware
  augmentation, no rare-repeat, no balanced epoch sampling, full fold00 val
  support `1858`, and `--skip-final-test`. I stopped it after epoch 3 because
  the best class-1 validation F1 stayed far below gate:
  epoch1 `0.8267/0.4762`, epoch2 `0.8492/0.5158`, epoch3
  `0.8635/0.5361` macro/class-1. Reload eval of `best.pt` matched epoch 3:
  macro/class-1 `0.8635/0.5361`, class-1 P/R `0.6047/0.4815`, confusion row
  for true class 1 `[33,52,19,2,2]`.
- Boundary/XAI:
  `runs\boundary_review_probe_timm_mobilenetv3_yolof_oof_fold00_nowarmup_fair_val_20260704`
  selected `51` rows: `focus_false_negative=21`, `focus_false_positive=6`,
  `high_confidence_error=24`. Robust XAI
  `runs\xai_probe_timm_mobilenetv3_yolof_oof_fold00_nowarmup_fair_val_cases12_20260704`
  audited the first 12 high-confidence class-1 misses. Heatmap summary was
  attention foreground/background `0.0000/1.0000` due timm feature-map fallback,
  and Grad-CAM foreground/background/border `0.7117/0.2883/0.2877`. All 12
  cases were high-confidence misclassifications, object-color sensitive, and
  background/border flagged. Robustness drops were background blur/gray only
  `0.0005/0.0129`, while object desaturate dropped predicted probability by
  `0.6318`; visually, one selected case was a very thin crop dominated by pad
  and border, and typical `1->0` cases still relied on fruit surface/color.
- Decision:
  reject MobileNetV3 fold00 as a fold-safe teacher route even after no-warmup
  and fair checkpoint selection. It is weaker than the previous plain fold00
  probe and far below the TRKH keeper (`0.8847/0.6860` validation), so do not
  expand to 5-fold OOF or use it for KD/sample weights. Metrics/config/logs were
  copied into the XAI package under `source_probe_metrics\probe_train`, and the
  rejected checkpoint run was deleted via
  `runs\cleanup_manifest_20260704_rejected_mobilenetv3_fold00_nowarmup_probe.json`,
  reclaiming `71.46 MB`.

## Diagnostic 2026-07-04 - `class_f` MobileNetV3 Fold00 Probe Reject

- Rationale:
  I checked the user's hypothesis that the stronger pretrained MobileNetV3
  evidence might come specifically from `class_f`, not from `yolo_f`. This was
  validation-only on a train-only OOF artifact built from
  `D:\DataAI\AIEx\newdataset\class_f`; no raw dataset files and no test split
  were modified or used for selection.
- Infrastructure fixes:
  `build_classification_oof_folds` now writes absolute fold roots in generated
  `data.yaml` files, because relative fold paths were resolved under the YAML
  directory and produced zero-sample TRKH eval roots. The existing
  `runs\classf_oof_folds_train5_20260704\fold_*\data.yaml` files were updated
  as generated artifacts only. `import_timm_baseline_checkpoint` now records
  timm input mean/std and uses a new `resize_mode=stretch` path in
  `ClassificationTransform` so imported timm checkpoints can be audited through
  TRKH XAI with closer preprocessing. Preflight passed `py_compile` for the
  touched modules and focused fold tests
  `tests\test_build_classification_oof_folds.py
  tests\test_stitch_timm_oof_predictions.py -q` (`2 passed`).
- Probe:
  fold artifact `runs\classf_oof_folds_train5_20260704` contains `9215`
  train-source samples with hardlinks only (`55290`, copy `0`). The fold00
  MobileNetV3 probe used
  `D:\DataAI\AIEx\image_baseline_experiments\scripts\02_train_timm_classifier.py`,
  `mobilenetv3_large_100.ra_in1k`, pretrained weights, image size `224`,
  full fold00 val `1843`, batch `64`, LR `3e-4`, weight decay `0.05`, AMP,
  patience `4`, focus class `Xoai_Song_ChuaNhe_CoNguyCo`, and `--skip-test`.
  I stopped it after epoch 10 because the best focus-class fold-val F1 stayed
  below `0.60` and then regressed. Best epoch was 7 with macro/focus-class F1
  `0.8822/0.5933`, focus P/R `0.6139/0.5741`; direct export
  `runs\export_probe_timm_mobilenetv3_classf_oof_fold00_std_val_20260704`
  is the metric authority for this baseline.
- Boundary/XAI:
  boundary review
  `runs\boundary_review_probe_timm_mobilenetv3_classf_oof_fold00_std_val_20260704`
  selected `64` rows: `focus_false_negative=24`,
  `focus_false_positive=24`, and `high_confidence_error=16`. The selected
  focus false negatives were very confident, mostly class-1-to-raw-class
  `0/2/4` after class-order remap, with `20` over-bright/glare buckets.
  Robust XAI
  `runs\xai_probe_timm_mobilenetv3_classf_oof_fold00_std_val_cases12_20260704`
  audited 12 mapped sample-index cases. Top standard-order confusions were
  `0->1=32`, `1->0=27`, `1->2=16`, `2->3=12`, and `3->2=7`. Heatmap means
  were attention foreground/background `0.0000/1.0000` due timm fallback and
  Grad-CAM foreground/background/border `0.6601/0.3399/0.2395`. Flags included
  `high_confidence_misclassification=12`, `attention_background=12`,
  `gradcam_background=9`, `gradcam_border=10`, and
  `object_color_sensitive=9`. Robustness drops were background blur/gray only
  `0.0056/0.0031`, while object desaturate dropped predicted probability by
  `0.5853`. Manual overlay inspection confirmed focus on mango surface, color,
  and edge contrast rather than useful broad context.
- Cleanup and decision:
  reject the `class_f` MobileNetV3 fold-safe teacher route. It is below the
  no-pretrain TRKH keeper (`0.8847/0.6860`) and far below the `0.70` full-train
  gate, so do not expand it to 5-fold OOF and do not use it for KD, sample
  weights, or routers. Logs/import summary/export metrics were copied into the
  XAI package under `source_probe_metrics`; the raw and imported MobileNetV3
  checkpoint directories were deleted via
  `runs\cleanup_manifest_20260704_rejected_mobilenetv3_classf_fold00_probe.json`,
  reclaiming `32.54 MB`. The `class_f` OOF fold artifact is kept for now as a
  generated train-only utility, but this MobileNetV3 route is closed.

## Diagnostic 2026-07-04 - Late Class-Attention Pooling Reject

- Rationale:
  after checking CaiT-style class-attention and cross-attention ideas, I added a
  TRKH-native `LateClassAttentionPooling` module: the existing head feature
  queries patch tokens immediately before classification, with zero-init
  residual projection/MLP so old checkpoints load safely and the default path is
  unchanged. This is a no-pretrain TRKH change, not a pivot to external models.
- Implementation/preflight:
  `trkh\models\model.py`, `trkh\core\config.py`, `trkh\training\train.py`, and
  `scripts\run_trkh_5class_attention_views_v8.ps1` now expose default-off
  `late_class_attention_*` flags and resume extension support for
  `late_class_attention_pool.*`. Focused tests passed:
  `py_compile model.py config.py train.py`, PowerShell parser check for the V8
  launcher, and
  `pytest tests\test_late_class_attention_pooling.py tests\test_trainable_module_prefixes.py -q`
  (`8 passed`).
- Smoke:
  direct trainer smoke was used instead of V8 `-Smoke` because the launcher
  would force partial validation and `--resume-use-cli-config`. Recipe:
  resume from
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt`,
  explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, full validation
  support `2606`, `--skip-final-test`, train only
  `late_class_attention_pool`, LR `5e-4`, `60` train batches, `1` epoch,
  teacher-focus-binary `0.015`, and `crop_bbox` token prior.
  Validation reached only macro/class-1 F1 `0.8825/0.6765`, class-1 P/R
  `0.6085/0.7616`, below the keeper `0.8847/0.6860`. Confusion stayed
  dominated by `3->2=52`, `0->1=49`, `1->0=19`, `1->2=12`, `4->1=12`,
  and `2->1=9`.
- Boundary/XAI:
  reload eval preserved at
  `runs\eval_smoke_v8_yolof_lateclassattn_poolonly_resumeconf_val_20260704`
  matched the trainer metric. Boundary review
  `runs\boundary_review_smoke_v8_yolof_lateclassattn_poolonly_val_20260704`
  selected `79` rows (`focus_false_negative=24`,
  `focus_false_positive=19`, `low_margin_error=24`,
  `low_margin_correct_boundary=12`). Robust XAI
  `runs\xai_smoke_v8_yolof_lateclassattn_poolonly_val_cases16_20260704`
  on class-1 false negatives showed foreground mass still high
  (`attention/grad_rollout/gradcam/rollout` =
  `0.8999/0.9447/0.9478/0.8953`), background blur/gray drops near zero
  (`0.0010/0.0013`), and object-desaturate drop much larger (`0.0834`).
  Review flags still included rollout border/background (`8/16`, `9/16`) and
  Grad-CAM border (`5/16`). Manual overlays confirmed heat on fruit surface,
  edge, stem/hand/background-adjacent regions, not a new broad-context
  background-vs-object cue.
- Cleanup and decision:
  reject pool-only late class-attention on the current keeper and do not run a
  longer probe. It does not reach the `0.70` class-1 validation gate and repeats
  the same surface/boundary/color failure mode. Metrics/config/history/logs were
  copied into the XAI package under `source_smoke_metrics`; the rejected smoke
  train directory was deleted via
  `runs\cleanup_manifest_20260704_lateclassattn_poolonly_rejected_smoke.json`,
  reclaiming `135.08 MB`. Keep the implementation default-off as a possible
  building block only if coupled to a genuinely new recall-protecting target or
  trainable head/representation update; do not repeat the exact
  `TrainableModulePrefixes=late_class_attention_pool/LR=5e-4/60b/1e` recipe.

## Diagnostic 2026-07-04 - Late Class-Attention Plus Head Reject

- Rationale:
  because pool-only adaptation could have failed simply because the classifier
  head was frozen, I ran one non-identical smoke that also allowed the final
  `head` to adapt. This tested whether the new class-attention residual had a
  useful signal once the decision layer could use it, while still avoiding a
  broad full-model continuation.
- Smoke:
  direct trainer recipe was the same full-val/no-test protocol as pool-only,
  but `TrainableModulePrefixes=late_class_attention_pool,head`, LR `3e-4`,
  `60` train batches, `1` epoch, resume keeper, teacher-focus-binary `0.015`,
  and `crop_bbox` token prior. Trainable parameters were `528646`
  (`527361` late pool + `1285` head). Validation reached only macro/class-1
  F1 `0.8828/0.6784`, class-1 P/R `0.6073/0.7682`, still below the keeper
  `0.8847/0.6860` and below the `0.70` gate. Confusion stayed
  `3->2=51/52` range, `0->1=50`, `1->0=17-18`, `1->2=12`,
  `4->1=12`, and `2->1=9`.
- Boundary/XAI:
  reload eval
  `runs\eval_smoke_v8_yolof_lateclassattn_headpool_resumeconf_val_20260704`
  matched trainer metrics and exported predictions. Boundary review
  `runs\boundary_review_smoke_v8_yolof_lateclassattn_headpool_val_20260704`
  selected `78` rows (`focus_false_negative=24`,
  `focus_false_positive=19`, `low_margin_error=24`,
  `low_margin_correct_boundary=11`). Robust XAI
  `runs\xai_smoke_v8_yolof_lateclassattn_headpool_val_cases16_20260704`
  again showed background perturbation near zero
  (`background_blur=0.0008`, `background_gray=0.0008`) versus larger
  object-desaturate drop (`0.0861`). Heatmaps remained foreground/surface
  dominated but with border/background flags (`rollout_background=9/16`,
  `rollout_border=8/16`, `gradcam_border=6/16`). Manual overlays matched the
  same pattern: stem/edge/background-adjacent hotspots on `1->2`, bright
  surface heat on `1->0`, and edge/local-background heat on `1->4`.
- Cleanup and decision:
  reject the late class-attention head+pool adaptation. Letting the classifier
  adapt slightly improved class-1 F1 over pool-only (`0.6784` vs `0.6765`) but
  remained below the keeper and did not change the failure mechanism. Metrics,
  config, history, and logs were copied into the XAI package under
  `source_smoke_metrics`; the rejected smoke train directory was deleted via
  `runs\cleanup_manifest_20260704_lateclassattn_headpool_rejected_smoke.json`,
  reclaiming `135.08 MB`. Do not run nearby `late_class_attention_pool,head`
  60b-120b variants on the current keeper; any late class-attention revisit
  needs a new reliability/OOF/recall-protecting target, not just more head
  adaptation.

## Diagnostic 2026-07-04 - Current-Keeper Train Boundary Review Batch 01

- Rationale:
  after patch-router, pretrained fold probes, and late class-attention all
  repeated the same surface/boundary failure mode, I checked the remaining
  train-only label-policy route instead of launching another loss/router smoke.
  This uses generated artifacts only and does not edit raw images or labels.
- Artifact:
  `runs\manual_boundary_review_current_keeper_train_batch01_20260704` was built
  from
  `runs\boundary_review_teacherfocusbinary015_train_20260703\boundary_review_manifest.csv`.
  It selects `31` current-keeper train rows: `11` focus false negatives
  (`1->0`), `12` focus false positives (`0->1`), and `8` extra low-margin
  `0/1` boundary rows. The output contains `review_batch_unfilled.csv`,
  `contact_sheet_01.jpg`, `contact_sheet_02.jpg`, `summary.json`, and
  `visual_review_notes.md`.
- First-pass visual audit:
  the selected class-1 false negatives mostly look like subtle green/unripe
  mangoes where the risk cue is not visually obvious from the crop alone. The
  class-0 false positives often contain glare, shadows, small speckles,
  stem/skin marks, or crop-context artifacts. This again supports the current
  audit conclusion: the hard cases are fruit surface/label-boundary ambiguity,
  not a broad-background shortcut and not a missing generic hybrid backbone.
- Research check:
  recent noisy-label/fine-grained directions support explicit label-quality or
  VLM-assisted review signals, but the repo has already rejected the local
  SNSCL-style noise-tolerant SupCon variant on this keeper. A VLM/noisy-label
  detector could be useful only as a train-only review assistant; using it as an
  inference teacher or validation-threshold router would break the no-pretrain
  TRKH target. Primary references checked for this decision:
  SNSCL/fine-grained noisy labels (`https://arxiv.org/abs/2303.02404`),
  DeFT VLM noisy-label detection (`https://arxiv.org/abs/2409.19696`), and the
  noisy-label processing survey (`https://arxiv.org/abs/2404.04159`).
- Decision:
  do not train from this batch while `manual_label_status` is unfilled. Keep it
  as a review queue for a future conservative label policy. If reviewed labels
  are added, use train-only `correct`/`ambiguous` semantics with explicit
  class-1 recall protection before any smoke; do not auto-relabel or tune
  thresholds from validation/test.

## Diagnostic 2026-07-04 - YOLO BBox Geometry Selector Reject

- Rationale:
  I tested the user's wide-context/position hypothesis in the cheapest
  train-only way before adding another model block: fit simple selectors from
  `yolo_f` train metadata and evaluate on full validation. The diagnostic used
  bbox geometry, object index, source-object counts, mixed-label flags, and
  optionally the current keeper probabilities. Raw dataset files and the test
  split were untouched.
- Artifact:
  `runs\diagnostic_yolof_bbox_geometry_selector_keeper_runval_20260704` contains
  `summary.json` and `metrics_table.csv`. It fit on
  `runs\eval_yolof_teacherfocusbinary015_best_train_20260702\predictions_detailed.csv`
  and evaluated keeper-run validation probabilities from
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\best_val_eval_predictions\predictions_detailed.csv`,
  with bbox metadata merged by `sample_index` from the generated full-val
  detailed eval. Validation support was `2606`.
- Metrics:
  the base keeper-run validation metric in this artifact is macro/class-1 F1
  `0.8841/0.6841`, class-1 P/R `0.6082/0.7815`. Geometry-only logistic
  collapsed to `0.3619/0.1667` and changed `1539` rows with only `50`
  corrections versus `1388` harms. Geometry-only ExtraTrees collapsed to
  `0.3348/0.0252`. Base-probability logistic was below base at
  `0.8781/0.6563`; adding geometry stayed below base:
  logreg `0.8772/0.6627`, ExtraTrees `0.8773/0.6667`.
- Decision:
  reject bbox-position/source-count geometry as an inference router or training
  target on the current keeper. This directly argues against relying on wide
  YOLO context or object location as the missing class-1 signal. Future use of
  `yolo_f` should keep bbox metadata as a weak object prior only, not as a
  standalone context/background discriminator. A superseded earlier diagnostic
  directory was deleted after this result was documented; cleanup manifest:
  `runs\cleanup_manifest_20260704_bbox_geometry_superseded_deleted.json`.

## Diagnostic 2026-07-04 - AIDT/Top-5 Consensus Train-Signal Audit

- Rationale:
  because the user asked whether external-model checks meant the work had moved
  away from TRKH, I audited the strongest AIDT/top-5/TRKH consensus signal on
  train and validation. The goal was to decide whether it can become a
  fold-safe TRKH training target, not to replace TRKH with pretrained models.
  This was CSV analysis only: no train run, no raw dataset edits, and no test
  split read.
- Artifact:
  `runs\diagnostic_aidt_top5_consensus_train_signal_20260704` compares current
  TRKH train/val predictions with mapped AIDT and top-5 teacher probability
  caches. The previously documented validation rule was reported as
  `trkh_max_conf=0.30`, but the archived `38` changed / `32` correction /
  `6` harm validation result is reproduced from the current CSVs with effective
  `trkh_max_conf=0.35`. The artifact preserves both `0.30` and `0.35` rows so
  the mismatch is explicit rather than silently rewritten.
- Metrics:
  with reported `trkh_max_conf=0.30`, train changes were `93/9215`, all
  corrections, improving train macro/class-1 F1 from `0.9400/0.8164` to
  `0.9579/0.8812`; validation reached `0.8985/0.7329` with `33` changes
  (`29` corrections, `4` harms). With reproduced effective
  `trkh_max_conf=0.35`, train changes were `111/9215`, all corrections,
  improving train to `0.9616/0.8944`; validation matched the old consensus
  artifact at `0.8985/0.7319` with `38` changes (`32` corrections,
  `6` harms).
- Visual audit:
  I built a train-only manual review queue at
  `runs\manual_consensus_review_train_signal_20260704` from the `0.35` changed
  cases. It contains `111` rows, `102` `both_non1_suppress` cases and `9`
  `both_1_rescue` cases, with six contact sheets and an unfilled
  `review_batch_unfilled.csv`. Contact sheets show the same hard mode as the
  keeper audits: very green fruits, subtle surface/label-boundary differences,
  glare/shadow, dark spots, stems, and crop artifacts. This supports using the
  consensus only as a review/diagnostic cue, not as an automatic label or
  training signal.
- Decision:
  do not open a smoke/train from this consensus manifest while the review file
  has empty `manual_label_status`. The train-side consensus looks perfect
  because the teachers are in-sample and the top-5 train cache agrees with hard
  labels at `0.9998`; using it directly would repeat the failed AIDT/top-5
  sample-weight, KD, and router traps. Future use must be fold-safe or manual
  train-only review assisted, with explicit protection against true class-1
  recall loss. External models remain diagnostic/teacher/upper-bound tools; the
  active objective remains improving TRKH.

## Smoke 2026-07-04 - Manual Boundary Review Policy V1 Reject

- Safety/tooling fix:
  `build_boundary_review_manifest` now writes `sample_index` for new review
  rows, and `build_reviewed_boundary_training_manifests` preserves
  `sample_index/source_row_index` into sample-weight, targeted-margin,
  soft-target, and relabel candidate manifests. This fixes the multi-object
  `yolo_f` path-collapse risk for manually reviewed rows. Preflight passed
  `py_compile` for both tools and
  `pytest tests\test_boundary_review_manifest.py
  tests\test_reviewed_boundary_training_manifests.py -q` (`6 passed`).
- Train-only policy artifact:
  `runs\manual_boundary_review_current_keeper_train_batch01_policy_v1_20260704`
  was built from the existing train-only batch, with raw data untouched and no
  validation/test labels used. The policy marked `23` reviewed `0/1` boundary
  rows as `correct` and `8` low-margin clutter rows as `needs_crop`, producing
  `31` sample-weight rows and `23` targeted-margin rows keyed by `sample_index`.
  Loader guards matched all rows (`sample_weight=31`, `targeted_margin=23`).
- Smoke:
  `runs\smoke_v8_yolof_manualreview_policyv1_tm004_60b_1e_20260704` resumed the
  keeper, used full `yolo_f/val=2606`, `60` train batches, `1` epoch,
  teacher-focus-binary, crop-bbox prior, sample weights, and targeted-margin
  weight `0.004`. The trainer best validation was below keeper, and the
  independent reload eval preserved in the XAI package reported macro/class-1
  F1 `0.8797/0.6667`, class-1 P/R `0.5813/0.7815`, below keeper
  `0.8847/0.6860` and below the `0.70` gate. Confusion still widened class-1
  false positives: `0->1=52`, `4->1=16`, `2->1=13`, with true class-1 losses
  `1->0=15`, `1->2=12`, `1->4=6`.
- Boundary/XAI:
  boundary review
  `runs\boundary_review_smoke_v8_yolof_manualreview_policyv1_val_20260704`
  selected `153` rows: `focus_false_negative=33`,
  `focus_false_positive=40`, `low_margin_error=40`, and
  `low_margin_correct_boundary=40`; boundary pairs were `0-1=80`,
  `1-2=23`, `1-4=14`, `2-3=35`. Robust XAI
  `runs\xai_smoke_v8_yolof_manualreview_policyv1_val_cases12_robust_20260704`
  showed the same failure mode as earlier keeper audits: attention/Grad-CAM
  foreground mass remained high enough for object focus, but border/background
  flags persisted (`gradcam_border=8/12`) and the causal perturbation was object
  surface/color, not wide background. Background blur/gray drops were only
  `0.0083/0.0076`, while object desaturate drop was `0.1606`;
  `object_color_sensitive=8/12`. Manual overlays showed heat on fruit surface,
  bright/dark spots, and object edges rather than a usable broad-context cue.
- Cleanup and decision:
  reject manual-review policy v1 before any probe/full train. It is too sparse
  to create a new class-1 boundary signal (`train_targeted_margin_fraction`
  about `0.0036`) and it increases class-1 false positives instead of improving
  precision. Metrics/config/history/eval predictions/trace were copied into
  `runs\xai_smoke_v8_yolof_manualreview_policyv1_val_cases12_robust_20260704\source_smoke_metrics`;
  the rejected checkpoint run was deleted via
  `runs\cleanup_manifest_20260704_manualreview_policyv1_rejected_smoke.json`,
  reclaiming `185.53 MB`. Do not repeat this exact
  `correct + needs_crop_downweight + targeted_margin=0.004` policy on the
  current keeper; future manual-review use needs more coverage, an explicit
  recall-protecting policy, or a fold-safe reliability signal before smoke.

## Smoke 2026-07-04 - AIDT/Top-5 Consensus Hard-Case Policy V1 Reject

- Rationale:
  after rejecting the smaller manual boundary policy, I tested the larger
  train-only AIDT/top-5 consensus queue as a cautious hard-case target for TRKH,
  not as an external-model replacement. The source review queue
  `runs\manual_consensus_review_train_signal_20260704` contains `111`
  train-only rows where the current TRKH train prediction disagreed with the
  hard label and both AIDT/top-5 consensus supported the hard label. Contact
  sheets looked visually plausible enough for a no-relabel hard-margin policy:
  keep the official label as target, use TRKH's train prediction as the
  negative class, and leave raw data unchanged.
- Policy artifact:
  `runs\manual_consensus_review_train_signal_policy_v1_20260704` writes
  `review_batch_policy_v1.csv` with `manual_label_status=correct` for all
  `111` train rows and no relabel candidates. The generated manifests contain
  `111` sample-weight rows and `111` targeted-margin rows, all keyed by
  `sample_index`, with pairs `0->1=68`, `2->1=22`, `4->1=9`, `1->0=8`,
  `3->1=3`, and `1->4=1`.
- Smoke:
  `runs\smoke_v8_yolof_consensus_policyv1_tm0025_80b_1e_20260704` resumed the
  keeper, used full `yolo_f/val=2606`, `80` train batches, `1` epoch,
  teacher-focus-binary, crop-bbox prior, and targeted-margin loss weight
  `0.0025`. Loader guards matched all `111` sample-index rows. The targeted
  margin was active (`train_targeted_margin_fraction=0.0113`,
  `train_targeted_margin_loss=0.0431`), but validation worsened: trainer
  macro/class-1 F1 `0.8783/0.6611`, class-1 P/R `0.5728/0.7815`. Reload eval
  preserved in the XAI package reported macro F1 `0.8786`. Confusions widened
  class-1 false positives (`0->1=55`, `4->1=16`, `2->1=13`) while true
  class-1 misses stayed (`1->0=15`, `1->2=12`, `1->4=6`).
- Boundary/XAI:
  boundary review
  `runs\boundary_review_smoke_v8_yolof_consensus_policyv1_val_20260704`
  selected `153` rows, again `focus_false_negative=33`,
  `focus_false_positive=40`, and boundary pair `0-1=80`. Robust XAI
  `runs\xai_smoke_v8_yolof_consensus_policyv1_val_cases12_robust_20260704`
  matched the rejected policy v1 failure mode: attention/Grad-CAM stayed mostly
  object-focused but class-1 decisions were still surface/color/edge driven.
  Background blur/gray drops were only `0.0067/0.0073`, object desaturate drop
  was `0.1614`, and flags included `object_color_sensitive=8/12`,
  `gradcam_border=7/12`, `near_tie=4/12`.
- Cleanup and decision:
  reject this consensus hard-case policy before probe/full train. It is a
  stronger sparse target than the 31-row review policy, but it still pushes the
  current representation in the wrong direction: more class-1 false positives
  without new recall-protecting surface signal. Metrics/config/history/eval
  predictions/trace were copied into
  `runs\xai_smoke_v8_yolof_consensus_policyv1_val_cases12_robust_20260704\source_smoke_metrics`;
  the rejected checkpoint run was deleted via
  `runs\cleanup_manifest_20260704_consensus_policyv1_rejected_smoke.json`,
  reclaiming `185.53 MB`. Do not repeat in-sample AIDT/top-5 consensus as a
  direct targeted-margin/sample-weight/KD policy on this keeper; future external
  consensus work must be fold-safe or provide a representation-level signal with
  explicit class-1 recall protection.

## Smoke 2026-07-05 - Internal SSL-DINO Keeper-Init Surface-Detail Reject

- Rationale:
  this was still a TRKH improvement attempt, not a pivot to another model. The
  goal was to test whether a train-only self-supervised DINO phase on `yolo_f`
  could add a stronger representation signal before returning to the current
  V8 no-pretrain TRKH recipe. Raw data was untouched, validation/test were not
  used during SSL, and the smoke used `SkipFinalTest`.
- SSL phase:
  `runs\ssl_dino_yolof_keeperinit_surfdetail_64b_1e_20260705` initialized from
  the keeper checkpoint and ran `64` train batches for `1` epoch with
  foreground-luma surface-detail amplification. SSL loss was `2.8367` and DINO
  teacher entropy was `0.8788`. The SSL model did not include the
  `bbox_spatial_fusion_head`, so the later V8 smoke had to initialize that
  extension from scratch.
- Smoke:
  `runs\smoke_v8_yolof_ssl_dino_keeperinit_surfdetail64b_60b_1e_20260705`
  resumed the SSL checkpoint, used full `yolo_f/val=2606`, `60` train batches,
  `1` epoch, teacher-focus-binary `0.015`, crop-bbox prior, and bbox spatial
  fusion. Resolved config confirmed hard/sample-weight/targeted-margin manifests
  were disabled and `test_summary=null`. Validation fell to macro/class-1 F1
  `0.8730/0.6446`, class-1 P/R `0.5519/0.7748`, far below keeper
  `0.8847/0.6860`. Main confusions were `0->1=58`, `4->1=23`, `2->1=11`,
  plus true class-1 misses `1->0=17`, `1->2=14`, `1->4=3`.
- Boundary/XAI:
  reload eval `runs\eval_smoke_v8_yolof_ssl_dino_keeperinit_surfdetail64b_val_20260705`
  matched the launcher metrics. Boundary review
  `runs\boundary_review_smoke_v8_yolof_ssl_dino_keeperinit_surfdetail64b_val_20260705`
  selected `153` rows: `focus_false_negative=34`,
  `focus_false_positive=40`, `low_margin_error=40`, and
  `low_margin_correct_boundary=39`, with boundary pairs `0-1=80`,
  `1-2=19`, `1-4=19`, `2-3=35`. Robust XAI
  `runs\xai_smoke_v8_yolof_ssl_dino_keeperinit_surfdetail64b_val_cases12_robust_20260705`
  again showed surface/color dominance rather than useful wide-context
  reasoning: background blur/gray drops were only `0.0050/0.0043`, while
  object desaturate drop was `0.1457`; flags included
  `rollout_background=8/12`, `rollout_border=6/12`, and
  `object_color_sensitive=5/12`.
- Cleanup and decision:
  reject this SSL-DINO keeper-init surface-detail route before any probe/full
  train. It worsened class-1 precision and did not create a new background or
  boundary signal. Metrics/config/history/eval/boundary/XAI summaries were
  copied into
  `runs\evidence_ssl_dino_keeperinit_surfdetail64b_rejected_20260705`; the SSL
  checkpoint and smoke checkpoint directories were deleted via
  `runs\cleanup_manifest_20260705_ssl_dino_keeperinit_surfdetail64b_rejected.json`,
  reclaiming `357.66 MB`. Do not repeat
  `ssl-method=dino + keeper init + surface-detail 64b/1e + V8 60b/1e` on the
  current keeper. Future SSL/representation work needs a checkpoint-compatible
  path for bbox fusion or a genuinely different recall-protecting target before
  spending another smoke.

## Smoke 2026-07-05 - Light RandAugment On Current Keeper Reject

- Rationale:
  this remained a TRKH no-pretrain improvement attempt, not a switch to an
  external model. I checked a very light RandAugment-only regularization because
  the original RandAugment paper
  (`https://arxiv.org/abs/1909.13719`) removes the separate policy-search phase
  and exposes a small regularization search space. This smoke intentionally did
  not repeat the old heavy bundle (`MixStyle + foreground surface fusion +
  background-counterfactual consistency + RandAugment/local exposure/obstacle`)
  or AugMix/JSD consistency; it only set `RandAugmentNumOps=1` and
  `RandAugmentMagnitude=4` on the current V8 keeper recipe.
- Preflight:
  `py_compile` passed for `train.py`, `dataset.py`, `model.py`, `config.py`,
  `evaluate.py`, and `xai_audit.py`; focused pytest
  `tests\test_xai_audit_dataset_tensor.py tests\test_boundary_review_manifest.py`
  passed (`11 passed`). V8 dry-run confirmed explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, keeper resume checkpoint,
  full validation (`MaxValBatches=0`), `SkipFinalTest=true`, no stale hard,
  sample-weight, soft-target, or targeted-margin manifests, and active
  `randaugment_num_ops=1/randaugment_magnitude=4`.
- Smoke:
  `runs\smoke_v8_yolof_light_randaugment1m4_teacherfocusbinary015_60b_1e_20260705`
  resumed the keeper, used full `yolo_f/val=2606`, `60` train batches,
  `1` epoch, teacher-focus-binary `0.015`, crop-bbox prior, and bbox spatial
  fusion. Launcher validation fell to macro/class-1 F1 `0.8776/0.6629`,
  class-1 P/R `0.5756/0.7815`, below the keeper `0.8847/0.6860` and below the
  `0.70` probe gate. Reload eval
  `runs\eval_smoke_v8_yolof_light_randaugment1m4_val_20260705` confirmed full
  support `2606` and macro/class-1 F1 `0.8771/0.6629`. Main confusions worsened
  class-1 false positives: `0->1=53-54`, `4->1=16`, `2->1=13`, with true
  class-1 misses `1->0=15`, `1->2=12`, `1->4=6`.
- Boundary/XAI:
  boundary review
  `runs\boundary_review_smoke_v8_yolof_light_randaugment1m4_val_20260705`
  selected `153` rows: `focus_false_negative=33`,
  `focus_false_positive=40`, `low_margin_error=40`, and
  `low_margin_correct_boundary=40`; boundary pairs were `0-1=80`,
  `1-2=24`, `1-4=15`, `2-3=33`. Robust XAI
  `runs\xai_smoke_v8_yolof_light_randaugment1m4_val_cases12_robust_20260705`
  again showed surface/color dominance rather than useful context reasoning:
  background blur/gray drops were only `0.0010/-0.0001`, while object
  desaturate drop was `0.1423`. Flags included `rollout_background=7/12`,
  `rollout_border=6/12`, `attention_background=4/12`, and
  `object_color_sensitive=5/12`. Manual overlay inspection showed one
  `1->2` case hot near edge/background clutter and one `1->0` case hot over
  fruit surface but still not recall-safe.
- Cleanup and decision:
  reject this light RandAugment setting before probe/full train. The smoke
  lowered class-1 precision, widened class-1 false positives, and did not create
  a new background/context cue. Metrics/config/history/trace were copied into
  `runs\xai_smoke_v8_yolof_light_randaugment1m4_val_cases12_robust_20260705\source_smoke_metrics`;
  the rejected checkpoint run was deleted via
  `runs\cleanup_manifest_20260705_light_randaugment1m4_rejected_smoke.json`,
  reclaiming `182.69 MB`. Do not repeat nearby
  `RandAugmentNumOps=1/Magnitude=4` on the current keeper, and do not continue
  by simply increasing augmentation strength; color/surface signal is label
  relevant here, so generic stochastic augmentation is more likely to blur the
  boundary than solve class 1.

## Audit 2026-07-05 - Consensus Review Queue No-Train Decision

- Rationale:
  after the user asked whether the work was still focused on TRKH, I rechecked
  the train-only consensus review queue rather than treating AIDT/top-5 as a
  replacement model. The artifact
  `runs\manual_consensus_review_train_signal_20260704` contains `111`
  train-only rows: `102` `both_non1_suppress` rows and `9` `both_1_rescue`
  rows. Raw data, validation, and test were not modified or used for a new
  policy.
- Visual audit:
  all six contact sheets were reviewed. The suppressor rows are plausible
  hard cases, but they are dominated by green/yellow fruit surface, glare,
  shadow, dark speckles, stem scars, crop edges, and subtle surface aging.
  The rescue rows show the opposite risk: true class-1-looking samples can be
  pushed to class `0/4` if class-1 suppression is made too aggressive. This
  matches the prior XAI pattern: broad background is not the main causal cue;
  object surface/color/boundary is.
- Decision:
  do not build a policy v2 from this unfilled consensus queue. The previous
  direct consensus targeted-margin policy already failed (`0.8783/0.6611`),
  and the visual pass does not provide enough reliable manual labels to justify
  another sparse margin/sample-weight smoke. Keep the queue diagnostic-only
  unless a human-reviewed train-only policy or a true fold-safe/OOF teacher
  signal is added. The TRKH target remains unchanged: no test tuning, no raw
  dataset edits, and no 30e train unless smoke plus probe pass the class-1
  validation gate.

## Diagnostic 2026-07-05 - Patch-Evidence Verifier Full Export For TRKH Reuse

- Rationale:
  this stayed inside the TRKH improvement loop. External/AIDT/top-5 evidence
  was used only as diagnostic context; the concrete step here was to preserve
  the one current TRKH-native signal that still clears the next class-1
  validation milestone: the train-fit `0-1` patch-evidence verifier. I also
  rechecked relevant literature before choosing the next direction:
  SNSCL/noisy fine-grained classification
  (`https://openaccess.thecvf.com/content/CVPR2023/html/Wei_Fine-Grained_Classification_With_Noisy_Labels_CVPR_2023_paper.html`),
  FFVT multi-layer/local feature fusion (`https://arxiv.org/abs/2107.02341`),
  and TransFG part selection for fine-grained recognition
  (`https://arxiv.org/abs/2103.07976`). This supports local evidence and
  reliability targets, not another wide-background or generic augmentation
  sweep. Raw data and test split were untouched.
- Tooling:
  `trkh.tools.probe_pairwise_feature_verifier` now exports a fitted sklearn
  `StandardScaler + LogisticRegression` pipeline as raw-space coefficients
  (`raw_coef`, `raw_intercept`) plus standardizer parameters, and
  `trkh.tools.probe_patch_evidence_mil` writes the same export for
  patch-evidence verifiers. I added feature-schema helpers so future exports
  include exact feature names for the verifier vector layout:
  `[embedding, prob, log_prob, margin, confidence, patch-evidence stats]`.
  Preflight passed `py_compile` for both tools and focused pytest
  `tests\test_pairwise_verifier_model_export.py tests\test_patch_evidence_mil.py
  tests\test_pairwise_feature_verifier.py -q` (`15 passed`).
- Full no-test export:
  `runs\patch_evidence_mil_yolof_keeper_01only_exportparams_20260705`
  used the current keeper checkpoint, explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, pair `0-1`, `top_k=4`,
  `crop_bbox`, full train `9215`, and full validation `2606`. It fit on
  `2482` train samples from classes `0/1` (`1941/541`) with 5-fold OOF local
  accuracy `0.9243`. Exported verifier dimension is `318`, pair status is
  `exported`, and the raw coefficient count is `318`. This run started before
  the schema patch was loaded, so its JSON has `feature_names=0`; the follow-up
  schema-only smoke confirmed current code writes `feature_names=318`
  (`embedding_0` ... `pair_0_1_positive_fraction_bbox`).
- Metrics and changed-case audit:
  base full-val macro/class-1 F1 was `0.8841/0.6841`; patch-verified full-val
  reached `0.8892/0.7073`, class-1 P/R `0.6554/0.7682`, clearing the `0.70`
  diagnostic milestone. It changed `31` validation predictions: `16`
  corrections, `9` harms, and `6` neutral changes. Most changes were
  `1->0` suppressions (`24` cases) plus `0->1` rescues (`7` cases). The gains
  mainly come from reducing false class-1 positives, but the verifier still
  harms true class-1 recall in `4` cases and creates `5` new class-0-to-class-1
  harms. The changed-case CSV is preserved at
  `runs\patch_evidence_mil_yolof_keeper_01only_exportparams_20260705\val\changed_cases_for_xai.csv`.
- XAI/audit decision:
  I did not generate a duplicate XAI package for the export itself because it
  is not a trained model and reuses the already-audited patch-verifier behavior:
  previous fixed verifier and summary-router audits showed foreground-clean,
  object surface/color/boundary sensitivity, near-zero background perturbation,
  and recall risk on true class 1. The new value is the train-fit full-vector
  linear parameter export, not a new visual behavior claim.
- Cleanup and next step:
  obsolete command-check/schema smoke directories were deleted via
  `runs\cleanup_manifest_20260705_patch_evidence_exportparams_smokes_deleted.json`,
  reclaiming `0.394 MB`; the full export artifact is preserved. This is a
  keeper diagnostic and a valid next implementation target, not a final model:
  do not repeat the rejected summary-stat zero-init router. The next TRKH-native
  step should implement a checkpoint-safe in-model linear patch verifier using
  the full `318`-dim feature schema and explicit class-1 recall protection, then
  smoke full-val/no-test before considering any longer train.

## Implementation 2026-07-05 - In-Model Runtime Patch-Evidence Linear Verifier

- Scope:
  this directly continues TRKH, not an external-model switch. I converted the
  train-fit patch-evidence logistic verifier into a default-off runtime module
  inside `trkh.models.model`, and added `evaluate.py` CLI flags to load the
  exported JSON. Raw datasets and test split were untouched. The implementation
  is checkpoint-safe by default: old checkpoints have no verifier state unless
  the export is explicitly loaded at runtime.
- Code and preflight:
  `PatchEvidenceLinearVerifier` now reconstructs the full verifier vector
  `[head_input, prob, log_prob, top1-top2 margin, confidence, patch-evidence]`
  and applies the same pair route conditions as the post-hoc tool:
  current/top pair membership, minimum pair probability, maximum pair margin,
  verifier confidence, and candidate-change check. It supports optional
  `protect_right_min_probability`, but defaults it off for parity. Focused
  tests now cover feature-layout parity with
  `trkh.tools.probe_patch_evidence_mil.summarize_patch_evidence`, route/recall
  protection behavior, and default-off export loading. Preflight passed
  `py_compile trkh\models\model.py trkh\evaluation\evaluate.py` and focused
  pytest
  `tests\test_patch_evidence_router.py tests\test_pairwise_verifier_model_export.py
  tests\test_patch_evidence_mil.py tests\test_pairwise_feature_verifier.py -q`
  (`25 passed`).
- Full-val no-test evaluation:
  baseline on the same `evaluate.py` path is preserved at
  `runs\eval_yolof_keeper_full_val_baseline_20260705` with full
  `yolo_f/val=2606`: macro/class-1 F1 `0.8829/0.6783`, class-1 P/R
  `0.6031/0.7748`. Runtime verifier eval is preserved at
  `runs\eval_patch_linear_verifier_softboost001_full_val_20260705`, using
  `runs\patch_evidence_mil_yolof_keeper_01only_exportparams_20260705\pair_verifier_model_params.json`,
  pair `0-1`, feature_dim `318`, `top_k=4`, `min_pair_probability=0.02`,
  `max_pair_margin=0.40`, `confidence_threshold=0.60`, `logit_boost=0.01`,
  and `bbox-token-prior-source=crop_bbox`. The first hardboost eval proved the
  labels but inflated routed confidences and loss, so the default/runtime
  artifact now uses soft boost `logit_boost=0.01`. It reached full-val
  macro/class-1
  F1 `0.8887/0.7030`, class-1 P/R `0.6480/0.7682`, clearing the `0.70`
  validation milestone on the runtime path. Softboost preserved the same label
  metrics as hardboost but restored loss to `1.1858` instead of the hardboost
  `1.3285`, so calibration/confidence artifacts are less misleading. It is
  slightly below the post-hoc probe artifact (`0.8892/0.7073`) because the
  evaluator/probe data paths have tiny probability differences on near-tie
  cases; sample-index comparison found only `7` prediction differences against
  the post-hoc CSV.
- Changed-case audit:
  runtime-vs-baseline changed-case CSV is preserved at
  `runs\eval_patch_linear_verifier_softboost001_full_val_20260705\changed_cases_runtime_vs_baseline.csv`.
  The verifier changed `29` validation predictions: `16` corrections, `8`
  harms, and `5` neutral changes. Transitions were exactly `1->0` suppressions
  (`22`) plus `0->1` rescues (`7`). Class-1 gains come mostly from suppressing
  false class-1 positives (`14` class0 corrections), but it still harms three
  true class-1 rows by suppressing `1->0` and creates five class0 `0->1` harms.
- Recall-protection diagnostic:
  simple `protect_right_min_probability` is not useful on this evaluator. Among
  the `22` `1->0` suppressions, true-class1 harms have base p1 values
  `0.2369/0.2464/0.2859`, while correct class0 suppressions overlap broadly
  (`0.2451` to `0.3244`). Restoring suppressions by p1 threshold lowered the
  verifier result: thresholds `0.24`, `0.28`, `0.30`, and `0.32` gave class-1
  F1 `0.6782`, `0.6985`, `0.6988`, and `0.7009`, all below no-restore
  `0.7030`. Do not tune nearby one-dimensional p1 recall-protection thresholds
  on validation/test.
- XAI:
  selected runtime changed cases were first audited at
  `runs\xai_eval_patch_linear_verifier_full_val_changed12_20260705` using
  `8` harms plus `4` corrections from the changed-case CSV. That robust audit
  renders the underlying checkpoint feature signal and matches prior evidence:
  object desaturate drop `0.0870` is much larger than background blur/gray drops
  `0.0017/0.0013`, with `near_tie_top2=7/12`, rollout background `7/12`,
  rollout border `10/12`, and Grad-CAM border `6/12`. I then patched
  `xai_audit.py` to load the same verifier JSON as `evaluate.py`. Verifier-aware
  XAI smoke `runs\xai_eval_patch_linear_verifier_softboost001_changed4_verifieraware_20260705`
  confirmed routed predictions are rendered with realistic confidence after
  softboost (`0.272-0.284`, not hardboost `1.0`) and foreground-heavy heatmaps:
  attention foreground/background `0.9022/0.0978`, Grad-CAM
  `0.9610/0.0390`. This reinforces that the issue is foreground
  surface/boundary ambiguity, not a wide-background shortcut.
- Decision:
  keep the in-model linear verifier as the current best TRKH-native runtime
  diagnostic and deployable evaluation hook, but do not call the problem solved.
  It clears the `0.70` class-1 validation milestone, yet remains far from the
  final target and still has recall-risk harms. The next useful work should not
  be another threshold sweep; it should either add verifier-aware XAI/trace
  support and a fold-safe recall/FP reliability target, or move to a genuinely
  new representation signal that can push the next validation milestone
  (`0.75`) without losing true class-1 recall.
- Cleanup:
  superseded smoke eval `runs\eval_patch_linear_verifier_smoke_20260705` was
  deleted after the full-val artifact and docs were preserved. Manifest:
  `runs\cleanup_manifest_20260705_runtime_verifier_smoke_deleted.json`,
  reclaimed `1.128 MB`.
  Hardboost artifacts `runs\eval_patch_linear_verifier_full_val_20260705` and
  `runs\xai_eval_patch_linear_verifier_full_val_changed4_verifieraware_20260705`
  were deleted after softboost replaced them. Manifest:
  `runs\cleanup_manifest_20260705_runtime_verifier_hardboost_superseded_deleted.json`,
  reclaimed `4.609 MB`.

## Diagnostic 2026-07-05 - Train-OOF Patch Verifier Acceptance Meta-Gate Rejected

- Rationale:
  after the softboost runtime verifier cleared class-1 validation F1 `0.70`
  but still had recall-risk harms, I tested whether a train-only/OOF
  accept/reject gate could keep only safer verifier proposals. This was a
  diagnostic only: raw data and test were untouched, thresholds were selected
  from train OOF candidates, and validation was used only to evaluate the
  frozen train-selected rule.
- Artifact:
  `runs\patch_verifier_acceptance_meta_gate_yolof_keeper01_20260705` uses
  `runs\patch_evidence_mil_yolof_keeper_01only_exportparams_20260705`
  train/val prediction CSVs. Candidate proposals were reconstructed from
  verifier probabilities with pair `0-1`, `min_pair_probability=0.02`,
  `max_pair_margin=0.40`, and verifier confidence `0.60`. The meta features
  used base probabilities/log-probabilities, pair margin/confidence,
  verifier probabilities, candidate direction, and patch-method vote agreement.
  A `StandardScaler + LogisticRegression(C=0.5, class_weight=balanced)` gate
  was selected by train OOF for three target definitions:
  `exact_correct`, `not_harm`, and `class1_benefit`.
- Result:
  the diagnostic rejected this route. Train candidates were noisy:
  `81` OOF candidates (`1->0=69`, `0->1=12`), and accepting all proposals
  already reduced train class-1 F1 from `0.8157` to `0.8032`. Val all-proposal
  post-hoc metrics were still the known strong verifier reference
  `0.8892/0.7073`, but every train-selected meta-gate underperformed on val:
  `exact_correct` reached macro/class-1 `0.8834/0.6826`, `not_harm`
  `0.8831/0.6825`, and `class1_benefit` `0.8854/0.6930`. The best meta-gate
  accepted `16` val changes (`6` corrections, `4` harms, `6` neutral), all
  `1->0`, and stayed below both the softboost runtime verifier
  `0.8887/0.7030` and the post-hoc proposal reference.
- XAI/audit:
  `runs\xai_patch_verifier_acceptance_meta_gate_class1benefit_accepted8_20260705`
  audited `8` accepted cases from the best but rejected `class1_benefit` gate
  (`4` harms, `4` corrections). It again showed near-tie/border ambiguity:
  `near_tie_top2=7/8`, Grad-CAM border `4/8`, attention/Grad-CAM background
  `3/8`, attention foreground/background `0.9017/0.0983`, and Grad-CAM
  foreground/background `0.7597/0.2403`. There is no new reliability cue here;
  the gate mostly learns a brittle subset of the same foreground surface
  ambiguity.
- Decision:
  do not implement this acceptance meta-gate in TRKH and do not continue with
  nearby logistic/threshold variants on the current verifier CSVs. This closes
  the simple train-OOF accept/reject route for the current embeddings. The next
  useful path needs a genuinely new representation or fold-safe teacher signal
  that can move the next class-1 validation milestone `0.75`, not another
  selector over the same patch-verifier probabilities.

## Forensics 2026-07-05 - Softboost Verifier Remaining Error Atlas

- Rationale:
  after confirming that train-OOF accept/reject gates do not improve the
  runtime verifier, I froze the current TRKH-native softboost verifier and
  inspected the remaining validation errors before starting another method.
  This was a no-test, no-raw-data-change diagnostic over full `yolo_f/val`.
- Artifacts:
  `runs\forensics_softboost_verifier_remaining_errors_20260705` summarizes
  remaining errors from
  `runs\eval_patch_linear_verifier_softboost001_full_val_20260705` against the
  same baseline/evidence CSVs. Verifier-aware robust XAI for selected remaining
  errors is preserved at
  `runs\xai_softboost_verifier_remaining_errors16_20260705`.
- Error atlas:
  the softboost verifier leaves `203/2606` validation errors
  (accuracy `0.9221`). Dominant confusions are `3->2=52`, `0->1=41`,
  `1->0=19`, `4->0=12`, `1->2=11`, and `0->4=10`. Class-1 false positives
  remain numerous (`63` rows): target-in-top2 is high (`48/63=0.7619`) and
  patch evidence can often recover the official target
  (`patch_any_target=32/63`). Class-1 false negatives are fewer (`35` rows) but
  harder for the current embedding: target-in-top2 is only `16/35=0.4571`, and
  patch evidence recovers the target in only `13/35=0.3714`.
- XAI:
  the selected 16 remaining-error cases have high foreground focus but still
  show near-tie/border ambiguity. Review flags include `near_tie_top2=13/16`,
  rollout border `10/16`, Grad-CAM border `8/16`, rollout background `7/16`,
  and attention background `6/16`. Mean attention/rollout/Grad-CAM foreground
  masses are `0.8711/0.8946/0.8362`; background perturbation is again weak:
  background blur/gray drops are `0.0035/-0.0038`, while object desaturation
  drop is `0.0627`.
- Decision:
  the remaining class-1 FP bucket still contains some reroutable near-ties, but
  the class-1 FN bucket often lacks class 1 even in top-2 and is not solved by
  patch evidence. Do not spend another loop on nearby logit/verifier gates over
  the same current embedding. The next useful TRKH-native work should add a new
  representation signal for class-1 surface/boundary recall while keeping the
  verifier's conservative FP control as a diagnostic guard.

## Experiment 2026-07-05 - SSL-DINO BBox-Fusion-Compatible Retest

- Rationale:
  the previous internal SSL-DINO route was rejected, but it had one real
  implementation caveat: the SSL checkpoint omitted `bbox_spatial_fusion_head`,
  so the follow-up V8 smoke initialized that head from scratch. I patched
  `trkh.tools.pretrain_internal_barlow` so DINO/V8 SSL configs can preserve
  bbox spatial fusion, added focused test coverage, and reran a smaller
  train-only DINO pretrain before a full-val smoke. Raw data and test were
  untouched.
- Preflight:
  `py_compile` passed for `trkh\tools\pretrain_internal_barlow.py` and
  `tests\test_internal_ssl_dino.py`; focused SSL tests passed
  (`tests\test_internal_ssl_dino.py`, `tests\test_internal_ssl_vicreg.py`,
  `tests\test_internal_ssl_mae.py`: `14 passed`). The dry-run artifact
  confirmed `missing_key_count=0`, `unexpected_key_count=0`, and
  `dino_teacher_synced_after_init=true`.
- SSL phase:
  `runs\ssl_dino_yolof_keeperinit_bboxfusion_surfdetail32b_1e_20260705`
  used `yolo_f/train`, keeper init, `ssl_method=dino`, bbox spatial fusion
  enabled, foreground-luma surface-detail augmentation, `32` train batches,
  and no validation/test tuning. It completed in about `105s`, with
  `best_ssl_loss=4.4120` and teacher entropy `1.2361`. This checkpoint was
  only a representation retest, not a keeper.
- Supervised smoke:
  `runs\smoke_v8_yolof_ssl_dino_bboxfusion_surfdetail32b_48b_1e_20260705`
  resumed from the SSL checkpoint, used explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, `48` train batches, full
  `yolo_f/val=2606`, bbox spatial fusion with `crop_bbox`, teacher
  focus-binary `0.015`, no hard/sample-weight/targeted-margin manifests, and
  `SkipFinalTest`. Validation reached only macro/class-1 F1
  `0.8647/0.6213`, class-1 P/R `0.5278/0.7550`, far below the keeper
  `0.8847/0.6860` and far below the next gate. The main class-1 false
  positives widened sharply: `0->1=60`, `4->1=22`, `2->1=17`; class-1 misses
  remained `1->0=16`, `1->2=15`, `1->4=5`.
- Audit:
  an independent full-val evaluate
  `runs\eval_smoke_v8_yolof_ssl_dino_bboxfusion_surfdetail32b_48b_val_20260705`
  reproduced the result (`2606` samples, macro F1 `0.8656`). Boundary review
  selected `153` rows from `530` candidates, dominated by
  `focus_false_positive=102`, with buckets `0->1=60`, `4->1=22`, and
  `2->1=17`. Robust XAI
  `runs\xai_smoke_v8_yolof_ssl_dino_bboxfusion_surfdetail32b_48b_val_cases12_robust_20260705`
  selected `12` class-1 mistakes. Review flags were attention background
  `8/12`, rollout background `8/12`, rollout border `5/12`, and
  object-color sensitive `5/12`. Background perturbations stayed negligible:
  background blur/gray drops `0.0025/0.0017`, while object desaturation drop
  was `0.1392`. Visual inspection of case overlays confirmed heat stayed on
  fruit surface/color/border regions rather than a useful wide-background cue.
- Cleanup:
  compact evidence was preserved in
  `runs\evidence_ssl_dino_bboxfusion_surfdetail32b_rejected_20260705`, including
  SSL/smoke/eval metrics, boundary review, XAI summaries, prediction CSVs, and
  a few inspected overlays. The raw SSL/smoke/eval/boundary/XAI run directories
  were deleted via
  `runs\cleanup_manifest_20260705_ssl_dino_bboxfusion_surfdetail32b_rejected.json`,
  reclaiming `391.11 MB`.
- Decision:
  bbox-fusion-compatible SSL-DINO does not rescue the previous SSL route; it is
  worse than the keeper and worse than the already rejected non-compatible SSL
  smoke. Do not repeat nearby DINO keeper-init surface-detail schedules by only
  changing SSL batch count or preserving bbox fusion. Future representation work
  must introduce a materially different target or fold-safe recall-protecting
  signal, not another short DINO continuation over the current embedding.

## Experiment 2026-07-05 - Complementary Patch Suppression Head

- Rationale:
  after the softboost verifier forensics showed that another logit/verifier
  selector would not solve class-1 false negatives, I tested a small
  TRKH-native representation hook inspired by fine-grained feature
  boosting/suppression and transformer peak-suppression work
  (`https://arxiv.org/abs/2103.02782`,
  `https://arxiv.org/abs/2107.06538`). The implementation is a default-off,
  checkpoint-safe `ComplementaryPatchSuppressionHead`: it scores patch tokens
  against the head representation plus bbox/foreground priors, suppresses the
  top salient tokens, pools the remaining complementary evidence, and adds a
  zero-init residual logit head. Raw data and test were untouched.
- Preflight:
  `py_compile` passed for `trkh\models\model.py`,
  `trkh\core\config.py`, `trkh\training\train.py`, and
  `tests\test_complementary_patch_suppression_head.py`. Focused tests passed:
  `tests\test_complementary_patch_suppression_head.py`,
  `tests\test_trainable_module_prefixes.py`,
  `tests\test_late_class_attention_pooling.py` (`11 passed`). V8 launcher
  dry-run confirmed the new `complementary_patch_suppression_*` arguments,
  `TrainableModulePrefixes=complementary_patch_suppression_head`,
  `BBoxSpatialFusion=true`, and `bbox_token_prior_source=crop_bbox`.
- Smoke:
  `runs\smoke_v8_yolof_complementary_patch_suppression_60b_1e_20260705`
  resumed the current keeper, trained only
  `complementary_patch_suppression_head` (`168,847` trainable parameters),
  used explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, `60` train
  batches, full `yolo_f/val=2606`, bbox spatial fusion with `crop_bbox`,
  teacher focus-binary `0.015`, no hard/sample-weight/targeted-margin
  manifests, and `SkipFinalTest`. The smoke reached only validation
  macro/class-1 F1 `0.8777/0.6611`, class-1 P/R `0.5728/0.7815`, below the
  keeper `0.8847/0.6860` and below the next gate. Main confusion widened:
  `0->1=55`, `4->1=17`, `2->1=12`, while class-1 misses stayed
  `1->0=16`, `1->2=11`, `1->4=6`.
- Audit:
  independent full-val evaluate reproduced the reject
  (`runs\eval_smoke_v8_yolof_complementary_patch_suppression_60b_val_20260705`,
  `2606` samples, macro F1 `0.8771`, calibrated macro `0.8819`). Boundary
  review selected `153` rows from `357` candidates, with
  `focus_false_positive=89`, `focus_false_negative=33`, and buckets
  `0->1=56`, `4->1=17`, `2->1=12`, `1->0=16`, `1->2=11`.
  Robust XAI selected `12` class-1 errors and again showed foreground/border
  ambiguity rather than a new complementary cue: attention/grad-rollout/
  Grad-CAM/rollout foreground masses were `0.8732/0.9412/0.9006/0.8913`,
  rollout border/background flags were `6/12` and `7/12`, and background
  blur/gray drops were `0.0007/-0.0004` versus object desaturate drop
  `0.1402`. Visual inspection of Grad-CAM/attention overlays showed heat on
  stem, border, glare, and adjacent background/hand regions in several true
  class-1 errors, not a reliable supplemental surface signal.
- Cleanup:
  compact evidence was preserved in
  `runs\evidence_complementary_patch_suppression_rejected_20260705`, including
  source smoke metrics/config/launcher logs, independent eval metrics and
  predictions, boundary review, and the XAI case audit. Raw smoke/eval/
  boundary/XAI directories were deleted via
  `runs\cleanup_manifest_20260705_complementary_patch_suppression_rejected.json`,
  reclaiming `143.734 MB`.
- Decision:
  reject the complementary patch suppression head on the current keeper. Do
  not repeat nearby variants that only change top-k, suppression strength,
  bbox weight, logit scale, LR, or short batch count. The failure matches the
  earlier pattern: current embeddings still confuse class-1 surface/boundary
  evidence, and suppressing salient tokens widens class-1 false positives
  instead of improving recall support. Future representation work needs a
  stronger fold-safe recall signal or a materially different feature source,
  not another residual head over the same patch tokens.

## Experiment 2026-07-05 - Train-Time Soft Patch Linear Verifier

- Rationale:
  the runtime full-vector `PatchEvidenceLinearVerifier` is currently the best
  TRKH-native validation hook, improving full `yolo_f/val` from about
  `0.8829/0.6783` to `0.8887/0.7030` macro/class-1 F1 without touching test.
  I tested whether the same exported `318`-dim verifier could help the model
  learn a better class-0/class-1 boundary during training, instead of only
  acting as a runtime logit adjustment.
- Implementation:
  `PatchEvidenceLinearVerifier` now supports a default-off train-time soft
  adjustment. In training mode it computes a differentiable soft gate from
  pair probability mass, pair margin, and optional verifier confidence, then
  applies a small signed class-0/class-1 logit delta. The exported raw
  verifier coefficients are non-persistent buffers so checkpoints remain
  clean; evaluation mode keeps the existing hard runtime route. Config, train
  CLI, V8 launcher flags, resume-prefix handling, and focused tests were
  added.
- Preflight:
  `py_compile` passed for `trkh\models\model.py`,
  `trkh\core\config.py`, `trkh\training\train.py`, and
  `tests\test_patch_evidence_router.py`. Focused pytest passed
  `tests\test_patch_evidence_router.py` and
  `tests\test_trainable_module_prefixes.py` (`16 passed`). V8 launcher
  dry-run confirmed explicit `yolo_f`, `bbox_token_prior_source=crop_bbox`,
  verifier JSON loading, train-time soft adjustment flags, and `SkipFinalTest`.
- Smoke and eval:
  `runs\smoke_v8_yolof_patchlinear_softtrain_w005_60b_1e_20260705` resumed
  the current keeper, used `60` train batches, `1` epoch, full
  `yolo_f/val=2606`, teacher focus-binary `0.015`, runtime verifier
  `logit_boost=0.01`, and train-time soft scale `0.05`. Validation reached
  only macro/class-1 F1 `0.8808/0.6782`, class-1 P/R
  `0.5990/0.7815`, below the keeper `0.8847/0.6860` and below the runtime
  verifier `0.8887/0.7030`. Independent reload eval without verifier reached
  macro F1 `0.8785`; reload with verifier reproduced `0.8808`.
- Audit:
  boundary review selected `153` rows from `344` candidates, with
  `focus_false_positive=79`, `focus_false_negative=33`, and buckets
  `0->1=51`, `1->0=15`, `1->2=12`, `2->1=12`, `4->1=12`. Robust,
  verifier-aware XAI on `12` class-1 boundary mistakes showed the same failure
  mode: foreground masses stayed high, but attention/rollout still had
  background/border flags; background blur/gray prediction drops were only
  `0.0025/0.0012`, while object desaturation drop was `0.1496`. A visual
  Grad-CAM check showed heat on fruit border/stem/adjacent regions rather than
  a new stable class-1 surface cue.
- Cleanup:
  compact evidence was preserved in
  `runs\evidence_patchlinear_softtrain_rejected_20260705`, including smoke
  metrics/config/logs, eval metrics and predictions, boundary review, and XAI
  cases. Raw smoke/eval/boundary/XAI directories were deleted via
  `runs\cleanup_manifest_20260705_patchlinear_softtrain_rejected.json`,
  reclaiming `197.991 MB`.
- Decision:
  reject train-time soft patch-linear verifier over the current exported
  verifier and current embeddings. The runtime verifier remains useful as a
  conservative validation hook/diagnostic guard, but making it a soft training
  signal did not improve representation and slightly worsened the checkpoint.
  Do not repeat nearby scale/LR/batch variants over this same verifier unless
  a new fold-safe verifier, recall-protecting target, or materially different
  representation signal is introduced.

## Experiment 2026-07-05 - Bbox-Interior Token Label Auxiliary Loss

- Rationale:
  after the soft verifier forensics pointed to representation rather than
  another selector, I tested a TRKH-native token-level supervision route. The
  idea was inspired by token-label/dense-supervision work such as "All Tokens
  Matter: Token Labeling for Training Better Vision Transformers"
  (`https://arxiv.org/abs/2104.10858`) and fine-grained part/token selection
  work such as TransFG (`https://arxiv.org/abs/2103.07976`), but adapted
  without new data: use the existing `yolo_f` crop-bbox token prior to select
  interior patch tokens, reuse the current classifier head, and ask selected
  tokens to predict the image label. This does not modify raw data or inference.
- Implementation:
  added default-off `bbox_token_label_loss` in `trkh\training\train.py`,
  config fields in `trkh\core\config.py`, V8 launcher flags, and focused
  tests in `tests\test_bbox_token_label_loss.py`. The helper classifies
  `features["patches"]` using `model.head`; by default the classifier weights
  are detached so the gradient pressures patch representations rather than
  simply moving the head. It respects `patch_bbox_prior`,
  `memory_key_padding_mask`, class filters, focus-class weighting, and records
  token coverage/accuracy/margin telemetry.
- Preflight:
  `py_compile` passed for `trkh\training\train.py`,
  `trkh\core\config.py`, and `tests\test_bbox_token_label_loss.py`. Focused
  pytest passed `tests\test_bbox_token_label_loss.py` plus
  `tests\test_trainable_module_prefixes.py` (`8 passed`). V8 dry-run confirmed
  explicit `yolo_f`, `bbox_token_prior_source=crop_bbox`,
  `bbox_token_label_loss_weight=0.006`, focus weight `1.35`, teacher
  focus-binary `0.015`, no hard/sample-weight/targeted-margin manifests, and
  `SkipFinalTest`.
- Smoke and eval:
  `runs\smoke_v8_yolof_bboxtokenlabel006_p045_focus135_60b_1e_20260705`
  resumed the current keeper, used `60` train batches, `1` epoch, and full
  `yolo_f/val=2606`. Validation reached only macro/class-1 F1
  `0.8785/0.6648`, class-1 P/R `0.5784/0.7815`, below the keeper
  `0.8847/0.6860` and far below the runtime verifier `0.8887/0.7030`.
  Reload eval reproduced the result (`2606` samples, macro/class-1
  `0.8785/0.6648`). Confusion widened class-1 false positives:
  `0->1=53`, `4->1=16`, `2->1=13`, while class-1 misses remained
  `1->0=15`, `1->2=12`, `1->4=6`. The token loss was active but weakly
  aligned: selected-token fraction `0.7604`, sample fraction `0.7974`,
  token-label accuracy `0.3677`, and target margin `-0.0890`.
- Audit:
  boundary review
  `runs\boundary_review_smoke_v8_yolof_bboxtokenlabel006_val_20260705`
  selected `153` rows from `356` candidates, with `focus_false_positive=77`,
  `focus_false_negative=33`, and buckets `0->1=46`, `4->1=14`,
  `2->1=13`, `1->0=15`, `1->2=12`, `3->2=27`. Robust XAI
  `runs\xai_smoke_v8_yolof_bboxtokenlabel006_val_cases12_robust_20260705`
  selected `12` class-1 false negatives. Foreground mass stayed high
  (`attention/grad_rollout/gradcam/rollout = 0.8731/0.9405/0.8996/0.8922`),
  but rollout still had background/border flags (`7/12` and `6/12`).
  Background blur/gray prediction drops were negligible (`0.0013/0.0002`),
  while object desaturation was much larger (`0.1429`). Visual checks of
  Grad-CAM/rollout overlays showed heat on fruit edge, stem, glare, and
  adjacent non-fruit regions, not a reliable interior class-1 cue.
- Cleanup:
  compact evidence was preserved in
  `runs\evidence_bbox_token_label_rejected_20260705` (`20.945 MB`) with smoke
  metrics/history/config/logs, best-val plots, reload eval predictions,
  boundary review, XAI cases, and `evidence_summary.json`. Rejected
  checkpoints were intentionally not copied. Raw smoke/eval/boundary/XAI
  directories were deleted via
  `runs\cleanup_manifest_20260705_bbox_token_label_rejected.json`,
  reclaiming `197.212 MB`.
- Decision:
  reject bbox-interior token label supervision on the current keeper. The
  approach is a useful default-off diagnostic building block, but without
  reliable token-level labels it mostly forces every selected interior patch to
  inherit noisy image-level boundary labels and widens class-1 false positives.
  Do not repeat nearby weight/min-prior/focus-weight tweaks over the same
  current embeddings. A future token-level route needs a stronger fold-safe
  token teacher, manual train-only token/part policy, or recall-protecting
  reliability target before another smoke.

## Experiment 2026-07-05 - Color-Statistic Fusion Head-Only Adapter

- Rationale:
  this was still a TRKH-native/no-pretrain improvement attempt, not a pivot to
  an outside model. Since repeated XAI audits showed object color/surface
  sensitivity much larger than background sensitivity, I tested whether the
  existing default-off `ColorStatisticFusion` head could add a small
  checkpoint-safe residual from RGB/HSV/Lab summary statistics, center/border
  deltas, and histogram features while keeping the base TRKH keeper frozen.
- Implementation and preflight:
  no raw data was modified and test was not used. The smoke resumed
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701`,
  enabled `ColorStatFusion=true`, trained only `color_fusion_head`
  (`14,683` params) with LR `5e-4`, `60` train batches, full
  `yolo_f/val=2606`, crop-bbox token priors, teacher focus-binary `0.015`,
  empty hard/sample-weight/targeted-margin manifests, and `SkipFinalTest`.
  Preflight passed `py_compile` for `trkh\models\model.py`,
  `trkh\training\train.py`, and `trkh\core\config.py`; focused pytest
  passed `tests\test_detection_calibration.py::DetectionCalibrationTests::test_color_stat_fusion_can_extend_existing_vit_checkpoint`
  plus `tests\test_trainable_module_prefixes.py` (`6 passed`). The launcher
  dry-run confirmed explicit `yolo_f`, checkpoint-safe missing
  `color_fusion_head.*` extension keys, and `SkipFinalTest`.
- Smoke and reload eval:
  `runs\smoke_v8_yolof_colorstatfusion_headonly_lr5e4_60b_1e_20260705`
  reached only validation macro/class-1 F1 `0.8777/0.6611`, class-1 P/R
  `0.5728/0.7815`, below the keeper `0.8847/0.6860` and far below the
  runtime verifier `0.8887/0.7030`. Confusion widened class-1 false
  positives: `0->1=55`, `4->1=17`, `2->1=12`, with class-1 misses still
  `1->0=16`, `1->2=11`, `1->4=6`. Independent reload eval
  `runs\eval_smoke_v8_yolof_colorstatfusion_headonly_lr5e4_val_20260705`
  confirmed full validation support `2606` and macro/class-1
  `0.8771/0.6611`.
- Audit:
  boundary review selected `153` rows from `354` candidates, with
  `focus_false_positive=78`, `focus_false_negative=33`, and buckets
  `0->1=46`, `4->1=16`, `2->1=12`, `1->0=16`, `1->2=11`, `3->2=27`.
  Robust XAI on `12` class-1 false negatives showed no new reliable color-stat
  cue: attention/grad-rollout/Grad-CAM/rollout foreground masses were
  `0.8732/0.9412/0.9006/0.8913`, rollout background/border flags were
  `7/12` and `6/12`, background blur/gray prediction drops were
  `0.0007/-0.0004`, while object desaturation was much larger (`0.1402`).
  Manual overlay inspection still showed heat on fruit border, stem/glare,
  adjacent background/hand regions, or diffuse interior color, not a stable
  class-1 boundary signal.
- Cleanup:
  compact evidence was preserved in
  `runs\evidence_colorstatfusion_headonly_rejected_20260705` (`0.931 MB`),
  including smoke/eval metrics, config/history, boundary review, XAI summary,
  and three representative overlays. Rejected checkpoints were not copied.
  Raw smoke/eval/boundary/XAI directories were deleted via
  `runs\cleanup_manifest_20260705_colorstatfusion_headonly_rejected.json`,
  reclaiming `142.054 MB`.
- Decision:
  reject color-stat fusion head-only adaptation on the current keeper. This
  route confirms that shallow color/stat residuals over the existing embedding
  do not add the missing class-1 surface/boundary reliability and instead
  widen false-positive class 1. Do not repeat nearby LR/dropout/head-only
  variants or implement another shallow HSV/Lab/color-stat branch unless a
  new fold-safe feature set shows materially stronger class-1 OOF gains and
  explicit recall protection.

## Diagnostic 2026-07-05 - Current Patch-OOF Cleanlab + Feature-Neighbor Policy

- Rationale:
  after multiple smoke rejects showed that loss/head tweaks over the current
  embedding are not enough, I checked a stricter train-only reliability route
  before launching another model run. The method follows the same motivation
  as confident-learning label-error detection
  (`https://arxiv.org/abs/1911.00068`) plus feature-neighbor consensus, but is
  used here as a no-test diagnostic only: intersect the current patch-verifier
  OOF cleanlab manifest with kNN agreement from multiple train feature banks.
- Infrastructure fix:
  `trkh.tools.build_oof_neighbor_label_policy` now prefers `sample_index`
  matching when feature NPZ files provide it, and propagates `sample_index`
  into strict sample-weight/soft-target outputs. This matters for `yolo_f`
  because multiple object rows can share the same image path. Added a
  duplicate-path regression test and kept path fallback for older feature
  banks. Preflight passed `py_compile` plus
  `tests\test_oof_neighbor_label_policy.py` and
  `tests\test_data_centric_sample_weights.py` (`4 passed`).
- Diagnostic:
  output `runs\oof_neighbor_label_policy_patchoof_yolof_current_multibank_20260705`
  used `cleanlab_patchoof_teacher_yolof_train_20260703` and three train-only
  feature banks: current TRKH local-neighbor/head features, direct DINOv2-small
  `yolo_f/train` features, and remapped EfficientNetV2-S `yolo_f/train`
  features. Settings were `top_k=15`, min vote fraction `0.65`, issue rank
  `<=160`, OOF confidence `>=0.70`, self-confidence `<=0.20`, same-image
  neighbors excluded, and no validation/test input.
- Result:
  cleanlab read `500` review rows and produced `188` candidate rows; the true
  issue directions were only `0->1=70` and `1->0=11`. Strict all-bank
  consensus selected `0` rows, so the generated sample-weight and soft-target
  manifests are empty. A diagnostic relaxed count showed `81` clean
  OOF-confidence candidates, but only `14` had at least two feature banks
  agreeing, split `1->0=8` and `0->1=6`; their minimum suggested-vote
  fractions were often near zero, so they are not safe enough to train from.
- Decision:
  no smoke from this policy. Keep the sample-index-safe tool fix, but do not
  train direct relabel/sample-weight/soft-target policies from the current
  patch-OOF cleanlab + neighbor consensus. The evidence is too sparse and still
  lives on the same dangerous `0/1` boundary where prior train-OOF `0->1`
  routing was toxic. Revisit only with human-reviewed train-only labels or a
  stronger fold-safe teacher/reliability signal that explicitly protects true
  class-1 recall.

## Experiment 2026-07-05 - BlockLocalPatchMixer Layers 5-8 + Head Adapter

- Rationale:
  the earlier sparse `BlockLocalPatchMixer` adapter-only and local+part recipes
  were rejected, but the skill left one legal revisit: connect local
  convolutional mixing to the classification head instead of training only the
  in-block adapter. This smoke stayed TRKH-native/no-pretrain and tested
  whether CMT/LeFF-style local mixing in transformer blocks `5-8`, plus head
  adaptation, could improve surface/boundary representation without touching
  raw data or test.
- Preflight and smoke setup:
  preflight passed `py_compile` for `trkh\models\model.py`,
  `trkh\training\train.py`, `trkh\core\config.py`, and
  `tests\test_block_local_patch_mixer.py`; focused pytest passed
  `tests\test_block_local_patch_mixer.py` plus
  `tests\test_trainable_module_prefixes.py` (`8 passed`). The dry-run
  confirmed explicit `yolo_f`, crop-bbox priors, `SkipFinalTest`, and trainable
  prefixes `blocks.4-7.local_patch_mixer,head`. The smoke resumed the current
  keeper, used `BlockLocalPatchMixer=true`, layers `5,6,7,8`, dropout `0.02`,
  residual scale `0.10`, LR `3e-4`, `60` train batches, full `yolo_f/val=2606`,
  teacher focus-binary `0.015`, and empty hard/sample-weight/targeted-margin
  manifests.
- Smoke and reload eval:
  `runs\smoke_v8_yolof_blocklocal_head_layers5to8_lr3e4_60b_1e_20260705`
  trained `275,717` parameters and reached validation macro/class-1 F1
  `0.8797/0.6667`, class-1 P/R `0.5850/0.7748`, below the keeper
  `0.8847/0.6860` and below the runtime verifier `0.8887/0.7030`. Confusion
  stayed in the same failure mode: `3->2=58`, `0->1=51`, `1->0=16`,
  `4->1=16`, `1->2=12`, `2->1=12`. Independent reload eval
  `runs\eval_smoke_v8_yolof_blocklocal_head_layers5to8_val_20260705`
  reproduced macro F1 `0.8797` on full validation support `2606`.
- Audit:
  boundary review selected `153` rows from `354` candidates with
  `focus_false_positive=76`, `focus_false_negative=34`, and
  `low_margin_error=43`; top buckets were `0->1=45`, `1->0=16`,
  `1->2=12`, `4->1=15`, and `3->2=28`. Robust XAI on `12` class-1 errors
  showed attention/grad-rollout/Grad-CAM/rollout foreground masses
  `0.8832/0.9511/0.9720/0.9060`, but rollout background/border flags
  remained `6/12` and `5/12`, Grad-CAM border `4/12`, and
  `object_color_sensitive=5/12`. Background blur/gray prediction drops were
  only `0.0018/0.0007`, while object desaturation was much larger
  (`0.1430`). Manual overlays still heated object border, pad/background, stem,
  glare, or diffuse surface spots rather than a stable class-1 cue.
- Cleanup:
  compact evidence was preserved in
  `runs\evidence_blocklocal_head_layers5to8_rejected_20260705` (`6.292 MB`),
  including smoke/eval metrics, config/history/logs, boundary review, XAI
  summary, and three representative overlays. Rejected checkpoints were not
  copied. Raw smoke/eval/boundary/XAI directories and transient logs were
  deleted via
  `runs\cleanup_manifest_20260705_blocklocal_head_layers5to8_rejected.json`,
  reclaiming `148.079 MB`.
- Decision:
  reject `BlockLocalPatchMixer` layers `5-8` plus head adaptation on the
  current keeper and do not run a probe/full train. The sparse local convolution
  path is technically connected, but it still does not add the missing
  class-1 surface/boundary reliability. Do not repeat nearby layer/LR/scale/head
  variants over the same embeddings unless a new recall-protecting target or
  initialized local selector changes the signal first.

## Experiment 2026-07-05 - Teacher-Guided Contrastive Memory Queue

- Rationale:
  after rejecting repeated head/router/local-mixer tweaks over the current
  embedding, I tested a representation-level route inspired by supervised
  contrastive learning and noisy fine-grained classification. SupCon pulls
  same-class embeddings together and pushes different classes apart
  (`https://arxiv.org/abs/2004.11362`), while SNSCL for noisy fine-grained
  labels uses selective queue updates to avoid inserting noisy anchors
  (`https://arxiv.org/abs/2303.02404`). This implementation stayed
  TRKH-native/no-pretrain: a default-off FIFO memory queue was added to the
  existing teacher-guided contrastive loss, with detached cross-batch head
  embeddings and teacher-agreement filtering. Raw data and test were not used.
- Implementation and preflight:
  added `TeacherGuidedContrastiveMemoryQueue`, config/CLI/V8 flags
  `teacher_guided_contrastive_memory_queue_size` and
  `teacher_guided_contrastive_memory_min_count`, plus history telemetry for
  memory loss, memory anchor count, positive count, and queue size. Queue size
  defaults to `0`, preserving old batch-only behavior. Replay forward passes use
  the queue but do not enqueue again. Preflight passed `py_compile` for
  `trkh\training\train.py` and `trkh\core\config.py`; focused pytest passed
  `tests\test_teacher_guided_contrastive.py` (`6 passed`). Launcher dry-run
  confirmed explicit `yolo_f`, keeper checkpoint, `BboxSpatialFusion`, empty
  hard/sample/targeted manifests, `SkipFinalTest`, and queue params.
- Micro functional check:
  `runs\micro_v8_yolof_tgcmemory_head_q512_8b_1e_20260705` used only
  `8` train batches and `2` val batches as a runtime check, not a gate.
  Telemetry confirmed the queue was active: selected fraction `0.1445`,
  selected count `4.625`, memory loss `0.3087`, memory anchor count `3.0`,
  memory positive count `23.0`, and memory size `14.25`.
- Smoke and reload eval:
  `runs\smoke_v8_yolof_tgcmemory_head_q512_60b_1e_20260705` resumed the
  current keeper, trained full TRKH for `60b/1e`, full `yolo_f/val=2606`,
  source `head`, queue `512`, min-count `8`, loss weight `0.004`,
  teacher confidence `0.35`, weight mode `confidence_margin`, min reliability
  `0.02`, confidence power `0.5`, distillation teacher
  `weighted_ensemble_no_pretrain_6expert_val_gate_20260701`,
  distillation weight `0.1`, teacher-focus-binary `0.015`, and `SkipFinalTest`.
  Queue telemetry was active but sparse and high-variance:
  selected fraction `0.1422`, selected count `4.55`, memory loss `2.1984`,
  memory anchors `4.33`, memory positives `168.87`, average memory size
  `127.72`. Validation reached only macro/class-1 F1 `0.8777/0.6592`,
  class-1 P/R `0.5735/0.7748`, below the keeper `0.8847/0.6860` and far below
  runtime verifier `0.8887/0.7030`. Independent reload eval
  `runs\eval_smoke_v8_yolof_tgcmemory_head_q512_val_20260705` reproduced
  macro F1 `0.8777` on full validation support `2606`.
- Audit:
  boundary review
  `runs\boundary_review_smoke_v8_yolof_tgcmemory_head_q512_val_20260705`
  selected `153` rows from `452` candidates with
  `focus_false_positive=76`, `focus_false_negative=34`, and
  `low_margin_error=43`. Buckets stayed in the old failure mode:
  `0->1=45`, `1->0=16`, `1->2=12`, `4->1=15`, and `3->2=27`.
  Robust XAI
  `runs\xai_smoke_v8_yolof_tgcmemory_head_q512_val_cases12_robust_20260705`
  selected `12` cases and again showed background perturbation near zero:
  background blur/gray drops `0.0022/0.0010`, center occlusion `0.0217`,
  object desaturation `0.1448`. Foreground mass was high
  (`attention/grad-rollout/Grad-CAM/rollout`
  `0.8828/0.9510/0.9749/0.9075`), but rollout background/border flags were
  `6/12` and `5/12`, and Grad-CAM border `4/12`. Manual overlays showed
  foreground surface, edge/stem/glare, and adjacent pad/hand/background hot
  spots rather than a stable class-1 cue.
- Cleanup:
  compact evidence was preserved in
  `runs\evidence_teacher_guided_memory_contrastive_rejected_20260705`
  (`3.153 MB`), including smoke metrics/config/history, trace summary, reload
  eval metrics/predictions, boundary summary/manifest, XAI summary, and three
  representative overlays. Raw smoke/eval/boundary/XAI/micro/log dirs were
  deleted via
  `runs\cleanup_manifest_20260705_teacher_guided_memory_contrastive_rejected.json`,
  reclaiming `378.432 MB`.
- Decision:
  reject teacher-guided contrastive memory queue on the current keeper before
  probe/full train. The implementation is useful default-off infrastructure,
  but this recipe makes class-1 false positives wider and does not improve
  class-1 surface/boundary separability. Do not repeat nearby
  `source=head/q512/loss=0.004/confidence_margin/min_conf=0.35/min_rel=0.02`
  variants or simply change queue size/temperature/short batch count on the
  current embedding. A future contrastive revisit needs a stronger fold-safe
  clean-anchor policy, class-1 recall protection, or a different feature source
  whose train-only diagnostics show better class-1 positive support first.

## Experiment 2026-07-05 - Class-1 Intra-Class Foreground SnapMix

- Rationale:
  after rejecting cross-class foreground SnapMix, I tested a narrower
  same-class variant inspired by intra-class part swapping for fine-grained
  recognition (`https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Intra-Class_Part_Swapping_for_Fine-Grained_Image_Classification_WACV_2021_paper.pdf`).
  The goal was to increase class-1 surface/part diversity without creating a
  soft label across the dangerous `0/1/2/4` boundary. This used existing
  default-off TRKH training infrastructure only: `ForegroundSnapmixPairs=1-1`
  with `crop_bbox` metadata, no raw dataset edits, and no test use.
- Preflight and setup:
  preflight passed `py_compile trkh\training\train.py trkh\core\config.py`
  and `tests\test_foreground_snapmix.py` (`2 passed`). The dry-run confirmed
  explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, keeper resume,
  empty hard/sample/targeted manifests, `BboxSpatialFusion=true`,
  `BboxTokenPriorSource=crop_bbox`, `SkipFinalTest=true`, and
  `foreground_snapmix_pairs=1-1`.
- Smoke and reload eval:
  `runs\smoke_v8_yolof_intraclass1_snapmix_cropbbox_w006_60b_1e_20260705`
  resumed the current keeper and used `ForegroundSnapmixLossWeight=0.006`,
  probability `0.65`, area ratio `0.08-0.18`, bbox margin `0.01`,
  teacher-focus-binary `0.015`, distillation weight `0.1`, `60b/1e`, and
  full `yolo_f/val=2606`. The auxiliary loss was active
  (`train_foreground_snapmix_loss=2.6859`, fraction `0.0224`, count `0.7167`,
  source weight `0.0780`, target removed `0.0800`), but validation reached only
  macro/class-1 F1 `0.8802/0.6648`, class-1 P/R `0.5821/0.7748`, below the
  keeper `0.8847/0.6860` and below the runtime verifier `0.8887/0.7030`.
  Independent reload eval
  `runs\eval_smoke_v8_yolof_intraclass1_snapmix_cropbbox_w006_val_20260705`
  confirmed full support `2606`, macro F1 `0.8799`, and class-1 F1 `0.6648`.
- Audit:
  boundary review selected `153` rows from `344` candidates:
  `focus_false_positive=84`, `focus_false_negative=34`, and
  `low_margin_error=35`. Class-1 false positives widened versus the keeper and
  the recent rejected routes: buckets included `0->1=55`, `4->1=16`, and
  `2->1=9`, while true class-1 misses stayed `1->0=16`, `1->2=12`, and
  `1->4=6`. XAI on `12` class-1 false negatives again showed background
  perturbation near zero (`background_blur/gray=0.0019/0.0006`) versus large
  object-desaturate sensitivity (`0.1480`). A separate FP1 XAI pass on `8`
  `0/2/3/4->1` cases showed the harmful direction clearly:
  `object_color_sensitive=8/8`, Grad-CAM border `6/8`, Grad-CAM background
  `3/8`, background blur/gray drops only `0.0167/0.0186`, but
  object desaturate drop `0.2417`. Manual overlays showed heat on fruit
  bottom/border, pad/background bands, scars/glare, and broad green surface
  regions rather than a stable class-1 boundary cue.
- Cleanup:
  compact evidence is preserved in
  `runs\evidence_intraclass1_foreground_snapmix_rejected_20260705`, including
  smoke metrics/config/history, trace summary, reload eval metrics/predictions,
  boundary review, FN/FP1 XAI summaries, and four representative overlays.
  Raw smoke/eval/boundary/XAI directories were deleted via
  `runs\cleanup_manifest_20260705_intraclass1_foreground_snapmix_rejected.json`,
  reclaiming `201.591 MB`.
- Decision:
  reject class-1 intra-class foreground SnapMix before probe/full train. The
  same-class part swap is label-preserving, but on the current keeper it mainly
  expands the class-1 decision region and amplifies color/border/background-
  adjacent cues. Do not repeat nearby `ForegroundSnapmixPairs=1-1` settings by
  only changing weight, probability, area ratio, bbox margin, or short batch
  count. If foreground part swapping is revisited, it needs a conservative
  reliability target that explicitly suppresses `0/2/4->1` false positives
  while protecting true class-1 recall.

## Experiment 2026-07-05 - Class-1 Foreground Counterexample Mix

- Rationale:
  after the user asked whether the work was still TRKH-focused, I kept the
  scope explicitly TRKH-native. External/AIDT/top-5 models remain diagnostic or
  teacher/upper-bound evidence only. This experiment tested a harder
  false-positive-control augmentation: paste a small foreground patch from
  source class `1` into target classes `0,2,4`, but keep the target label
  unchanged. The intended signal was "a local class-1-looking surface/spot is
  not sufficient for class 1." This does not edit raw data and does not use the
  test split.
- Implementation and preflight:
  added default-off `foreground_counterexample_mix_*` config/CLI/V8 launcher
  flags, a replay-safe auxiliary loss hook, history telemetry, and unit tests
  in `tests\test_foreground_snapmix.py`. The batch helper excludes true class-1
  rows from the target side, draws source patches only from class `1`, and keeps
  original target probabilities unchanged. Preflight passed
  `py_compile trkh\training\train.py trkh\core\config.py` and focused pytest
  (`tests\test_foreground_snapmix.py -q`, `4 passed`). V8 dry-run confirmed
  explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, keeper resume,
  `BboxSpatialFusion=true`, `BboxTokenPriorSource=crop_bbox`, empty
  hard/sample/targeted manifests, and `SkipFinalTest=true`.
- Smoke and reload eval:
  `runs\smoke_v8_yolof_fgcountermix_src1_neg024_w006_60b_1e_20260705` resumed
  the current keeper and used loss `0.006`, probability `0.50`, source class
  `1`, target classes `0,2,4`, area ratio `0.04-0.10`, bbox margin `0.01`,
  teacher-focus-binary `0.015`, distillation weight `0.1`, `60b/1e`, and full
  `yolo_f/val=2606`. Telemetry confirmed the route was active:
  `train_foreground_counterexample_mix_loss=1.2382`, fraction `0.3026`, count
  `9.6833`, source area `0.0738`, target removed `0.0699`. Validation reached
  only macro/class-1 F1 `0.8773/0.6534`, class-1 P/R `0.5721/0.7616`, below
  the keeper `0.8847/0.6860` and below runtime verifier `0.8887/0.7030`.
  Independent reload eval
  `runs\eval_smoke_v8_yolof_fgcountermix_src1_neg024_w006_val_20260705`
  reproduced macro/class-1 `0.8773/0.6534` with full support `2606`; no test
  metric was used.
- Audit:
  boundary review selected `153` rows from `354` candidates:
  `focus_false_positive=86`, `focus_false_negative=36`, and low-margin error
  `31`. Key buckets were `0->1=55`, `4->1=16`, `2->1=11`, `1->0=18`, and
  `1->2=12`, so the intended FP-control did not happen and true class-1 recall
  was also weaker. XAI on `12` selected FN1/FP1 cases showed the same bottleneck:
  `object_color_sensitive=11/12`, Grad-CAM border `8/12`, rollout background
  `6/12`, background blur/gray drops only `0.0088/0.0073`, while object
  desaturate drop was `0.2098`. Visual overlays showed Grad-CAM pulled to
  object border, bright/stem-adjacent regions, and unstable surface spots, not a
  reliable class-1 interior cue.
- Cleanup:
  compact evidence is preserved in
  `runs\evidence_fgcountermix_src1_neg024_w006_rejected_20260705`, including
  smoke metrics/config/history, trace summary, reload eval metrics, boundary
  review, XAI metrics/report, case CSVs, and representative overlays. Raw
  smoke/eval/boundary/XAI/log artifacts were deleted via
  `runs\cleanup_manifest_20260705_fgcountermix_src1_neg024_rejected.json`,
  reclaiming `195.375 MB`.
- Decision:
  reject foreground counterexample mix before probe/full train. The idea is
  connected and label-safe, but on the current keeper it widens class-1 false
  positives and lowers class-1 F1. Do not repeat nearby variants by only
  changing source/target probability, loss weight, patch area, bbox margin, or
  short-batch count. Any revisit needs an explicit fold-safe reliability target
  that both suppresses `0/2/4->1` and protects true class-1 recall.

## Experiment 2026-07-05 - Register Diversity Loss Smoke

- Rationale:
  after checking that `register_attention_alignment` had already failed by
  forcing bbox/foreground agreement, I tested the lighter TRKH-native
  `register_diversity_loss` hook. The intent was to reduce collapsed register
  tokens and encourage different token roles without another selector/gate over
  the same logits. The V8 launcher previously hardcoded
  `--register-diversity-loss-weight 0.0`, so I exposed
  `RegisterDiversityLossWeight` in the launcher config/dry-run path and passed
  it through to the trainer. No raw data or test split was used.
- Preflight:
  `py_compile trkh\training\train.py trkh\models\model.py trkh\core\config.py`
  passed, focused pytest
  `tests\test_detection_calibration.py -k register_diversity -q` passed
  (`1 passed, 131 deselected`), and a V8 dry-run confirmed explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, keeper resume,
  `BboxSpatialFusion=true`, `BboxTokenPriorSource=crop_bbox`,
  `RegisterDiversityLossWeight=0.003`, and `SkipFinalTest=true`.
- Smoke and reload eval:
  `runs\smoke_v8_yolof_registerdiversity_w003_60b_1e_20260705` resumed the
  keeper, used teacher-focus-binary `0.015`, distillation weight `0.1`,
  `60b/1e`, full `yolo_f/val=2606`, and no final test. Validation reached only
  macro/class-1 F1 `0.8770/0.6554`, class-1 P/R `0.5714/0.7682`, below the
  keeper `0.8847/0.6860` and the runtime verifier `0.8887/0.7030`.
  Independent reload eval
  `runs\eval_smoke_v8_yolof_registerdiversity_w003_val_20260705` confirmed
  macro F1 `0.8764` on full support `2606`.
- Audit:
  boundary review selected `153` rows from `351` candidates:
  `focus_false_negative=35`, `focus_false_positive=88`, and low-margin error
  `30`. Key buckets worsened around the class-1 boundary:
  `0->1=57`, `4->1=16`, `2->1=11`, `1->0=17`, and `1->2=12`.
  Robust XAI on `6` FN1 plus `6` FP1 cases showed the same failure mode as the
  rejected foreground and head/loss variants: background blur/gray drops were
  small (`0.0092/0.0077`), object desaturate was large (`0.2105`),
  `object_color_sensitive=11/12`, and Grad-CAM border flags were `8/12`.
  Register attention remained nearly collapsed with
  `cls_register_heatmap_similarity=0.99945`, so the diversity loss at this
  short smoke weight did not create a usable class-1 surface/boundary signal.
- Cleanup:
  compact evidence is preserved in
  `runs\evidence_registerdiversity_w003_rejected_20260705`, including smoke
  metrics/config/history, architecture trace summary, reload eval metrics,
  boundary review, XAI summaries, case CSVs, logs, and a representative overlay.
  Raw smoke/eval/boundary/XAI/log dirs were deleted. The cleanup manifest
  `runs\cleanup_manifest_20260705_registerdiversity_w003_rejected.json` was
  written after deletion because the first manifest write failed while measuring
  PowerShell hashtable bytes; therefore reclaimed MB is intentionally recorded
  as `null` rather than reconstructed.
- Decision:
  reject `register_diversity_loss_weight=0.003` before probe/full train. Do not
  repeat nearby register-diversity-only variants on the current keeper by only
  changing the weight or short-batch count. If register tokens are revisited,
  they need a materially different target that improves true class-1 recall
  support while controlling `0/2/4->1` false positives.

## Experiment 2026-07-05 - Auxiliary Class-F Multi-Pair Patch Evidence Diagnostic

- Rationale:
  after the user asked whether using `yolo_f` wide context plus `class_f` crop
  evidence could help TRKH distinguish object from background, I ran a
  no-test diagnostic that fits patch-evidence verifiers on `yolo_f/train` plus
  auxiliary `class_f/train`. This was deliberately diagnostic-only because an
  earlier 0-1 auxiliary class-f verifier with source-domain features was already
  rejected; this variant removed the source-domain feature and expanded the
  pairs to `0-1,1-2,1-4,2-3` to test whether crop evidence helps the remaining
  class-1 and `3->2` boundaries. Raw datasets were not modified.
- Preflight and run:
  `py_compile trkh\tools\probe_patch_evidence_mil.py
  trkh\tools\probe_pairwise_feature_verifier.py trkh\models\model.py` passed,
  and focused pytest
  `tests\test_patch_evidence_mil.py tests\test_pairwise_verifier_model_export.py -q`
  passed (`8 passed`). The probe used the current keeper checkpoint, explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, auxiliary
  `D:\DataAI\AIEx\newdataset\class_f\data.yaml`, weight `0.25`,
  `top_k=4`, `crop_bbox`, logistic verifier, `oof_folds=5`, and full
  `yolo_f/val=2606`.
- Metrics:
  the combined train-side signal looked strong, which is the warning sign:
  train patch-verified macro/class-1 F1 rose from `0.9399/0.8157` to
  `0.9567/0.8723`, with `108` changes (`97` corrections, `3` harms,
  `8` neutral). Validation did not follow. Full-val base was
  macro/class-1 `0.8841/0.6841`; patch-verified dropped to
  `0.8836/0.6835`, below the yolo-only patch verifier and far below the
  runtime softboost verifier `0.8887/0.7030`. Class-1 precision increased
  from `0.6082` to `0.6545`, but recall fell from `0.7815` to `0.7152`.
  Validation changed `73` rows: `34` corrections, `32` harms, and `7`
  neutral. Main transitions were `1->0=22`, `3->2=17`, `1->2=12`,
  `2->3=11`, `0->1=8`, and `1->4=3`; pair changes were `0-1=30`,
  `2-3=28`, `1-2=12`, and `1-4=3`.
- Audit:
  robust XAI on `12` balanced changed cases is preserved at
  `runs\xai_patch_evidence_auxclassf_pairs_changed12_20260705`. The audit used
  the base keeper on the selected changed cases because the current runtime XAI
  verifier loader supports one pair, while this probe was offline multi-pair.
  XAI again showed object surface/border ambiguity rather than a wide-background
  solution: foreground mass was high (`attention/grad_rollout/gradcam/rollout`
  `0.8620/0.9502/0.9581/0.9020`), background blur/gray drops were only
  `0.0082/0.0071`, while object desaturation drop was `0.0748`.
  Review flags included `near_tie_top2=5/12`, rollout border `8/12`,
  rollout background `6/12`, attention background `5/12`, and Grad-CAM border
  `3/12`. Register heatmaps stayed collapsed
  (`cls_register_heatmap_similarity=0.999792`). Visual overlays for class-1
  harms, class-0 false-positive harms, and `2-3` harms all concentrated on
  fruit edge, glare, small spots, or crop-border-adjacent object surface.
- Cleanup:
  compact metrics, predictions, selected-case CSV, verifier params, and logs
  were copied into
  `runs\xai_patch_evidence_auxclassf_pairs_changed12_20260705\source_patch_evidence_metrics`
  and `source_logs`. The raw probe directory was deleted via
  `runs\cleanup_manifest_20260705_auxclassf_multipair_patch_evidence_rejected.json`,
  reclaiming `4.234 MB`. No test split was used.
- Decision:
  reject auxiliary `class_f/train` multi-pair patch-evidence fitting. This
  confirms that simply adding crop-domain rows to a current-embedding verifier
  overfits train/OOF and damages validation recall. Do not repeat by only
  changing auxiliary weight, source-domain flag, pair list, margin, threshold,
  or top-k on the current keeper. If the two dataset views are combined again,
  it needs a new cross-view representation objective or fold-safe reliability
  target with explicit true-class-1 recall protection, not another verifier
  selector over the same embeddings.

## Diagnostic 2026-07-05 - TRKH Scope vs External Upper Bounds

- Rationale:
  after the user asked whether the work was still focused on TRKH, I ran a
  no-train/no-test diagnostic over full `yolo_f/val=2606` to quantify why
  AIDT/top-5/timm-style models look stronger and whether their signal is safe to
  transfer. The base was the current TRKH-native runtime softboost verifier
  (`runs\eval_patch_linear_verifier_softboost001_full_val_20260705`), which
  has validation macro/class-1 F1 `0.8887/0.7030` and leaves `203` errors
  (`35` class-1 FN, `63` class-1 FP). Output is saved at
  `runs\diagnostic_external_scope_trkh_remaining_errors_20260705`.
- Findings:
  aligned AIDT validation predictions reached macro/class-1 `0.9088/0.7169`;
  they corrected `110/203` softboost errors, including `17` class-1 FN rescues
  and `37` class-1 FP blocks, but also harmed `54` softboost-correct rows,
  including `14` true-class-1 rows and `30` non-class1-to-class1 harms.
  Aligned top-5 TTA reached `0.9139/0.7383`; it corrected `97` errors and
  harmed `33` correct rows, with `10` class-1 FN rescues, `37` class-1 FP
  blocks, `16` true-class-1 harms, and `10` non-class1-to-class1 harms. The
  diagnostic TRKH+top5 val ensemble reached `0.9166/0.7500`, but it remains an
  external/pretrained validation diagnostic, not the no-pretrain TRKH target.
  Frozen DINOv2-small and EffV2 readouts were not safe teachers: DINOv2 reached
  only `0.8449/0.5260` and harmed `149` softboost-correct rows; EffV2 reached
  `0.8876/0.6454` and harmed `67`.
- Decision:
  TRKH is still the active implementation target. External/AIDT/top-5 models
  are useful because their pretrained representations expose an upper bound and
  complementarity, especially on `3->2`, `0->1`, and class-1 boundary cases,
  but direct KD, sample weighting, or validation-threshold routing from those
  predictions is not safe without a fold-safe/OOF reliability gate that protects
  true class-1 recall. Do not treat external validation superiority as a pivot
  away from TRKH; use it to design a TRKH-native representation or fold-safe
  reliability signal.

## Cleanup 2026-07-05 - Compact Obsolete Rejected XAI Case Images

- Rationale:
  after the current best TRKH-native softboost verifier and latest diagnostics
  were preserved, I compacted old rejected XAI smoke artifacts whose conclusions
  are already recorded in this journal/TODO/skill. These folders were dominated
  by per-case PNG overlays; the root `xai_audit_summary.json`,
  `xai_metrics.json`, CSV manifests, markdown audit note, and any
  `source_smoke_metrics` remain available for numeric audit and do-not-repeat
  tracking.
- Scope:
  compacted only `case_*` subdirectories inside `20` obsolete
  `runs\xai_smoke_v8_yolof_*` rejection folders from 2026-07-02 through
  2026-07-04. Protected artifacts included the keeper checkpoint
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701`,
  current runtime verifier eval
  `runs\eval_patch_linear_verifier_softboost001_full_val_20260705`,
  patch-verifier export, softboost remaining-error forensics/XAI, the external
  scope diagnostic, and the latest aux-classf changed-case XAI.
- Result:
  manifest `runs\cleanup_manifest_20260705_compact_obsolete_xai_case_images.json`
  reclaimed `1445.987 MB`; D: free space increased from about `72.39 GB` to
  `73.81 GB`. No raw dataset files, no current keeper/best artifacts, and no
  test metrics were touched.

## Diagnostic 2026-07-05 - Gradient Feature Readout / Token-Injection Precheck

- Rationale:
  before editing the TRKH model again, I tested whether explicit Sobel/gradient
  surface features could provide the missing class-1 boundary cue suggested by
  the remaining-error XAI. This was a no-test, no-data-edit precheck for a
  possible gradient-token or gradient-feature injection route. The diagnostic
  used train predictions from
  `runs\eval_yolof_teacherfocusbinary015_best_train_20260702` and full
  `yolo_f/val=2606` predictions from the current keeper baseline plus runtime
  softboost verifier.
- Method:
  output is saved at
  `runs\diagnostic_gradient_feature_readout_yolof_keeper_20260705`. For each
  object row, the image crop came from `image_path` and `bbox_0..3` with a
  small bbox margin, resized to `256`, then converted into 100 Sobel magnitude,
  orientation, center/border, quadrant, and patch-grid statistics. Train-only
  GroupKFold by `source_stem` fit logistic readouts; validation was evaluated
  once on the full split.
- Metrics:
  the current softboost baseline remained macro/class-1 F1
  `0.8887/0.7030`, class-1 P/R `0.6480/0.7682`. `gradient_only` was weak
  (`0.5952/0.1972` on val). `baseprob_gradient` reached only
  `0.8762/0.6540`. The best gradient-assisted candidate,
  `softprob_gradient`, reached macro/class-1 `0.8799/0.6710`, P/R
  `0.6541/0.6887`, with `69` changed rows: `25` corrections, `35` harms, and
  `9` wrong-to-wrong changes.
- Audit:
  changed-case transitions showed the feature hurts exactly the unstable
  boundaries: `3->2=19`, `1->2=12`, `1->0=8`, `2->3=7`, `0->2=5`, and
  `4->2=4`. Targeted errors included `t3:3->2=12`, `t1:1->2=8`, and
  `t1:1->0=4`, so gradient features lower true-class-1 recall and do not fix
  the `3/2` surface boundary.
- Decision:
  reject simple Sobel/gradient feature readouts and do not implement a
  gradient-token injection module on the current keeper. The result confirms
  that raw edge/texture summaries are not a safe complementary signal over the
  current softboost verifier. Any future gradient/edge route needs a materially
  different fold-safe target with explicit true-class-1 recall protection, not
  another shallow readout or token branch.

## Diagnostic 2026-07-05 - Border/Interior Counterfactual TTA Probe

- Rationale:
  the remaining-error XAI repeatedly shows object surface, border, stem/glare,
  and near-crop-edge ambiguity. Before adding another train-time module, I
  tested whether fixed inference-time border/interior views could stabilize the
  current TRKH-native runtime softboost verifier. This was a no-train,
  no-test, no-raw-data-edit diagnostic over full `yolo_f/val=2606`.
- Implementation and preflight:
  added `trkh.tools.probe_border_interior_tta`, which builds deterministic
  pseudo-foreground masks from the current object crop and evaluates fixed
  variants: `border_suppressed`, `core_only`, `interior_suppressed`,
  `crop_edge_suppressed`, plus fixed probability averages with the clean view.
  The run used the current keeper checkpoint, explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, `crop_bbox` token prior, and
  the runtime 0-1 patch-evidence linear verifier with `logit_boost=0.01`.
  `py_compile trkh\tools\probe_border_interior_tta.py` passed. A 64-sample
  smoke only validated plumbing; it was deleted after full-val and XAI were
  preserved via
  `runs\cleanup_manifest_20260705_border_interior_tta_smoke64.json`.
- Metrics:
  full output is saved at
  `runs\diagnostic_border_interior_tta_yolof_softboost_fullval_20260705`.
  Clean recheck was macro/class-1 F1 `0.8889/0.7052`, matching the current
  softboost verifier within eval-path rounding. Every fixed counterfactual
  route was worse: `border_suppressed` `0.8653/0.6319`, `core_only`
  `0.8652/0.6295`, `crop_edge_suppressed` `0.8625/0.6070`, and
  `interior_suppressed` `0.1489/0.0000`. Fixed averages also stayed below
  clean: `avg_clean_border_suppressed` `0.8785/0.6603`,
  `avg_clean_core_only` `0.8791/0.6624`,
  `avg_clean_crop_edge_suppressed` `0.8805/0.6625`, and
  `avg_clean_border_core` `0.8730/0.6452`.
- Audit:
  `border_suppressed` changed `131` rows with only `46` corrections versus
  `81` harms; it corrected `16` class-1 false positives but introduced `14`
  new class-1 false positives and harmed `20` true class-1 predictions.
  `core_only` was similar (`47` corrections, `81` harms, `21` true-class-1
  recall harms). The best-looking fixed average, `avg_clean_core_only`, still
  had only `30` corrections versus `38` harms and harmed `13` true class-1
  predictions.
- XAI:
  verifier-aware XAI on 12 balanced changed cases is preserved at
  `runs\xai_border_interior_tta_avgcore_changed12_20260705`. The selected
  cases were foreground-heavy rather than broad-background failures:
  attention/grad-rollout/Grad-CAM/rollout foreground mass was
  `0.824/0.930/0.870/0.864`; rollout border flags were `7/12`,
  attention-background flags `7/12`, and near-tie cases `5/12`. Robustness
  stayed object-surface dominated: background blur/gray original-pred drops
  were only `0.0228/0.0272`, while object desaturation dropped the original
  prediction by `0.1003` and target probability by `0.0702`. Visual inspection
  of overlays confirmed heat on fruit edge, stem/pad, crop border, glare, and
  surface spots.
- Decision:
  reject fixed border/interior counterfactual TTA and do not train from these
  masks on the current keeper. Suppressing the border or using only eroded
  interior removes real class-1 evidence faster than it removes false-positive
  evidence. Do not repeat nearby fixed border/core/crop-edge TTA, simple
  border-suppression augmentation, or train-time border erasure on this anchor
  unless a new fold-safe reliability target explicitly protects true class-1
  recall and controls `0/2/4->1` false positives.

## Smoke 2026-07-05 - Bbox-Compatible Surface-Amplified Auxiliary Retest

- Rationale:
  while reviewing old rejected surface-counterfactual/surface-amplified routes,
  I found a real implementation inconsistency: their auxiliary forward passes
  called `_forward_model_outputs(..., image_valid_mask=None)` and did not pass
  `bbox_metadata` or `bbox_token_prior`, even though the current keeper uses
  bbox spatial fusion with `BboxTokenPriorSource=crop_bbox`. This made the
  auxiliary views off-policy relative to the main keeper path. I patched the
  helpers to select and pass `image_valid_mask`, `bbox_metadata`, and
  `bbox_token_prior` for the selected rows. The code fix is retained; the
  train recipe below was smoke-gated separately.
- Implementation and preflight:
  added `_select_batch_tensor_metadata` in `trkh\training\train.py`, wired
  `_surface_counterfactual_consistency_loss` and
  `_surface_amplified_supervised_losses` to pass selected metadata into
  `_forward_model_outputs`, and added regression tests in
  `tests\test_surface_amplified_supervised_loss.py`. Preflight passed:
  `py_compile trkh\training\train.py tests\test_surface_amplified_supervised_loss.py`
  and `pytest tests\test_surface_amplified_supervised_loss.py -q` (`4 passed`).
- Smoke:
  run name
  `smoke_v8_yolof_surfaceamp_bboxcompat_sup004_bm003_p50_60b_1e_20260705`
  resumed the keeper checkpoint, used explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, `BboxSpatialFusion=true`,
  `BboxTokenPriorSource=crop_bbox`, teacher-focus-binary `0.015`,
  no hard/sample/targeted manifests, `60` train batches, full
  `yolo_f/val=2606`, and `SkipFinalTest`. Losses were
  `SurfaceAmplifiedSupervisedLossWeight=0.004`,
  `SurfaceAmplifiedBoundaryMarginLossWeight=0.003`,
  probability `0.50`, `foreground_luma`, strength `0.18`, boundary margin
  `0.08`.
- Metrics:
  validation reached only macro/class-1 F1 `0.8770/0.6572`, class-1 P/R
  `0.5743/0.7682`, below the keeper `0.8847/0.6860` and far below the
  runtime softboost verifier `0.8887/0.7030`. Confusion widened class-1 false
  positives: `0->1=53`, `4->1=16`, `2->1=13`; true class-1 misses stayed
  `1->0=17`, `1->2=12`, `1->4=6`. Surface-amplified telemetry was active
  (`train_surface_amplified_fraction=0.4896`, boundary terms about `15.67`),
  so this was not a dead loss.
- XAI:
  robust XAI is preserved at
  `runs\xai_smoke_v8_yolof_surfaceamp_bboxcompat_val_cases12_robust_20260705`
  and used all `2606` validation samples for case selection. It selected
  12 cases with top confusions `3->2=58`, `0->1=53`, `1->0=17`,
  `4->1=16`, and `2->1=13`. Review flags stayed surface/border driven:
  `object_color_sensitive=9/12`, `rollout_background=7/12`,
  `gradcam_background=6/12`, `gradcam_border=8/12`, and only
  `near_tie_top2=2/12`. Robustness again showed background was not the main
  lever: background blur/gray original-pred drops were only
  `0.0148/0.0163`, while object desaturation drop was `0.1883`. Overlay
  inspection showed heat on fruit surface, lower edge/stem/crop border, and
  occasional padded/background strips.
- Cleanup:
  compact evidence was copied into
  `runs\xai_smoke_v8_yolof_surfaceamp_bboxcompat_val_cases12_robust_20260705\source_smoke_metrics`.
  The rejected checkpoint smoke and launcher logs were deleted via
  `runs\cleanup_manifest_20260705_surfaceamp_bboxcompat_rejected_smoke.json`,
  reclaiming `183.154 MB`.
- Decision:
  keep the bbox-compatible auxiliary-forward bugfix and tests, but reject the
  surface-amplified supervised/boundary recipe on the current keeper. Do not
  repeat nearby surface-amplified or surface-counterfactual smokes by only
  changing loss weights, probability, strength, blur kernel, boundary margin,
  or short batch count. The corrected auxiliary path still removes/overweights
  true class-1 surface evidence faster than it adds a recall-safe
  representation signal.

## Cleanup 2026-07-05 - Compact More Obsolete V8 XAI Case Images

- Rationale:
  after the bbox-compatible surface-amplified route was documented and the
  current keeper/runtime-verifier artifacts were rechecked, I compacted another
  batch of old image-heavy XAI folders. These folders were all rejected
  `xai_smoke_v8_yolof_*` audits from 2026-07-02 through 2026-07-04 whose
  conclusions are already recorded in this journal/TODO/skill. The cleanup
  removed only per-case `case_*` image directories; root `xai_audit_summary.json`,
  `xai_metrics.json`, CSV/MD files, and any `source_smoke_metrics` remain.
- Scope:
  manifest
  `runs\cleanup_manifest_20260705_compact_more_obsolete_xai_case_images.json`
  selected 64 obsolete V8 XAI smoke folders and deleted 776 `case_*`
  subdirectories. Protected artifacts included the keeper checkpoint, the
  runtime softboost verifier eval, patch-verifier export, remaining-error
  forensics/XAI, and the current 2026-07-05 surfaceamp bbox-compatible XAI.
- Result:
  reclaimed `3786.209 MB`. A verification pass found `0` remaining `case_*`
  directories under the selected old `xai_smoke_v8_yolof_*` set, and all
  protected artifacts still existed. No raw dataset files and no test metrics
  were touched.

## Diagnostic 2026-07-05 - Remaining Error External Support Consensus

- Rationale:
  after the user's scope question, I made the external-model role explicit:
  AIDT/top-5/timm/DINO/EffV2 are diagnostic/support probes only, while the
  implementation target remains TRKH-native/no-pretrain. The current remaining
  error forensics showed `203` full-val errors after runtime softboost, but it
  did not yet separate "errors with broad external support" from "errors no
  available representation can rescue." This diagnostic is validation-only and
  does not select deployable thresholds, train from validation, touch test, or
  edit raw data.
- Implementation:
  added reproducible tool
  `trkh.tools.analyze_remaining_error_external_support`. It joins prediction
  CSVs strictly by `sample_index`, summarizes external correction/harm/support
  on remaining softboost errors, exports
  `remaining_error_external_support.csv`, and evaluates two diagnostic-only
  consensus routes. Regression coverage in
  `tests\test_analyze_remaining_error_external_support.py` verifies
  sample-index joins plus focus consensus correction/harm accounting. Preflight
  passed:
  `py_compile trkh\tools\analyze_remaining_error_external_support.py` and
  `pytest tests\test_analyze_remaining_error_external_support.py -q`
  (`1 passed`).
- Inputs:
  base was
  `runs\eval_patch_linear_verifier_softboost001_full_val_20260705\predictions_detailed.csv`;
  remaining errors were
  `runs\forensics_softboost_verifier_remaining_errors_20260705\remaining_errors.csv`.
  External support sources were pure sample-index aligned AIDT
  `runs\aidt_classf_pretrained_val_20260702\teacher_probs_yolof_val_sampleindex_aidtpretrained_ttaflip.csv`,
  pure top-5 TTA
  `runs\top5_tta_teacher_yolof_sampleindex_20260703\teacher_probs_yolof_val_sampleindex_top5tta.csv`,
  EfficientNetV2-S readout
  `runs\pretrained_feature_oof_readout_effv2s_yolof_20260704\val_predictions_selected.csv`,
  and direct DINOv2-small readout
  `runs\pretrained_feature_oof_readout_dinov2small_yolof_direct_full_20260704\val_predictions_selected.csv`.
  Output is preserved at
  `runs\diagnostic_remaining_error_external_support_20260705`.
- Results:
  TRKH runtime softboost remained full-val macro/class-1 `0.8887/0.7030`.
  AIDT was `0.9088/0.7169`, top-5 TTA `0.9139/0.7383`,
  EfficientNetV2-S readout `0.8876/0.6454`, and DINOv2-small readout
  `0.8449/0.5260`. Across the `203` remaining TRKH errors, external support
  count was: `56` with zero external correct, `39` with one, `26` with two,
  `21` with three, and `61` with all four correct.
- Class-1 breakdown:
  for the `35` class-1 false negatives, `13` had zero external correction
  support, only `9` had a correct two-vote-or-more external consensus, and
  `12` had at least two external focus-class votes. For the `63` class-1 false
  positives, `50` had at least one external predictor blocking class 1, `22`
  were blocked correctly by all four, and `32` had a correct two-vote-or-more
  non-class-1 consensus. This confirms the asymmetric bottleneck: suppressing
  false positives is more externally supported than rescuing true class 1.
- Diagnostic route check:
  replacing the base prediction whenever at least two external predictors
  agreed on a different class gave validation macro/class-1 `0.9131/0.7393`
  with `120` changes (`88` corrections, `27` harms, `5` wrong-to-wrong).
  This is useful upper-bound evidence but not a deployable no-pretrain TRKH
  route because it depends on external/pretrained predictors and validation
  case behavior. A focus-only consensus route was worse for the actual target:
  macro/class-1 `0.8901/0.6993`, below current softboost class-1 `0.7030`,
  with `98` changes (`51` corrections, `42` harms). Do not compress this
  focus-consensus rule into a router, KD target, sample-weight manifest, or
  margin policy.
- Visual audit:
  contact sheets
  `focus_fn_no_external_support_contact_sheet.jpg` and
  `focus_fp_all_external_correct_contact_sheet.jpg` were generated in the same
  diagnostic folder. The no-support class-1 false negatives include many green
  or yellow/defective-looking fruit that AIDT/top-5/EffV2/DINO also classify as
  `0/2/4`, so they are likely hard label-boundary/representation cases rather
  than wide-background failures. The all-external-correct class-1 false
  positives show TRKH being pulled by surface spots, yellow lighting, green
  texture, stem/edge, and crop-adjacent context; that matches prior XAI where
  object desaturation dominates background perturbations.
- Research cross-check:
  TransFG argues for attention-based discriminative part selection in
  fine-grained recognition (`https://arxiv.org/abs/2103.07976`); FFVT similarly
  aggregates important tokens across transformer layers
  (`https://arxiv.org/abs/2107.02341`). HERBS motivates multi-scale refinement
  plus background suppression (`https://arxiv.org/abs/2303.06442`). SNSCL
  specifically studies noisy-label fine-grained classification and uses
  noise-tolerated supervised contrastive learning
  (`https://arxiv.org/abs/2303.02404`). The local evidence matters more than
  the papers here: TRKH has already tried close analogues of part/token
  selection, high-temperature refinement, background suppression, and
  contrastive/noisy-label routes on this keeper, and they did not produce a
  recall-safe class-1 signal.
- Decision:
  keep this diagnostic as explanation for why external models look stronger
  and why a validation external router is not acceptable. Do not run a smoke
  from this validation consensus. The next useful TRKH-native attempt needs a
  fold-safe train-only reliability signal or genuinely new representation
  source that improves class-1 false-negative top2 support while preserving the
  softboost verifier's false-positive control.

## Diagnostic 2026-07-05 - Train OOF Support For Patch-Cleanlab Queue

- Rationale:
  the validation external-support diagnostic found a tempting upper bound, but
  validation support cannot become a training target. I therefore checked the
  closest existing train-only/fold-safe queue before launching any new smoke:
  patch-OOF cleanlab issue candidates intersected with EfficientNetV2-S and
  direct DINOv2-small train OOF readouts.
- Artifact:
  `runs\diagnostic_train_oof_support_cleanlab_patchqueue_20260705`. Inputs were
  `runs\cleanlab_patchoof_teacher_yolof_train_20260703\cleanlab_label_issue_manifest.csv`,
  `runs\pretrained_feature_oof_readout_effv2s_yolof_20260704\train_oof_predictions_selected.csv`,
  and
  `runs\pretrained_feature_oof_readout_dinov2small_yolof_direct_full_20260704\train_oof_predictions_selected.csv`.
  The check is train-only, no validation/test tuning, no raw-data edit, and not
  a manifest for training.
- Result:
  among `81` cleanlab-flagged patch-OOF issue rows (`70` `0->1`, `11`
  `1->0`), EffV2 OOF plus DINO OOF both supported the cleanlab suggested class
  only `4` times, both kept the hard label `19` times, and disagreed `58`
  times. For the dangerous `0->1` flagged rows specifically, both OOF sources
  supported class 1 only `3/70`; for `1->0`, only `1/11`. The larger
  review-only rows contain many easy same-label agreements, but they are not
  issue rows and do not solve the class-1 boundary.
- Decision:
  reject auto-policy creation from the current patch-cleanlab queue. This
  confirms the earlier multi-bank policy result (`strict_rows=0`) and means a
  new smoke from these train OOF signals would be low-value. Future fold-safe
  reliability work needs a stronger reviewed/manual signal or a different OOF
  source; do not simply relax thresholds around this queue.

## Smoke 2026-07-05 - Train-OOF External Reliability Margin Recall-Protect Reject

- Rationale:
  the user asked whether external model checks meant the work had moved away
  from TRKH. This smoke kept the final target TRKH-native/no-pretrain and used
  EfficientNetV2-S plus DINOv2-small only as train-OOF reliability signals. It
  intentionally did not repeat the rejected AIDT/top-5 in-sample consensus
  policy: the new manifest was train-only, keyed by `sample_index`, required
  EffV2 and DINO to agree with the hard label for false-positive suppressors,
  and added explicit class-1 true-positive recall protection.
- Manifest:
  `runs\oof_external_reliability_margin_policy_t07_20260705` generated
  `targeted_margin_oofext_t07_train_only.csv` from
  `runs\diagnostic_train_oof_external_reliability_full_20260705\train_class1_boundary_oof_support.csv`
  plus current train probabilities from
  `runs\train_predictions_teacherfocusbinary015_20260701\predictions_detailed.csv`.
  It used no validation/test rows and did not edit raw data. The manifest has
  `166` rows: `55` FP1 suppressors, `110` near-boundary TP1 recall-protect rows,
  and `1` FN1 rescue row. Target-negative pairs are `0->1=39`, `2->1=14`,
  `4->1=2`, `1->0=89`, `1->2=13`, `1->4=8`, and `1->3=1`.
- Preflight and smoke:
  `py_compile` passed for train/dataset/model/config/evaluate/xai, and focused
  pytest
  `tests\test_targeted_margin_sample_index.py tests\test_reviewed_boundary_training_manifests.py tests\test_boundary_review_manifest.py -q`
  passed (`7 passed`). V8 dry-run confirmed explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, keeper resume, `crop_bbox`,
  bbox fusion, empty hard/sample/soft-target manifests, matched
  sample-index targeted-margin rows, full validation, and `SkipFinalTest`.
  Smoke
  `runs\smoke_v8_yolof_oofextrel_t07_marginrecall_60b_1e_20260705`
  used full `yolo_f/val=2606`, `60` train batches, `1` epoch,
  teacher-focus-binary `0.015`, `TargetedMarginLossWeight=0.002`, and no test.
  Loader matched all `166` rows and the margin was active
  (`train_targeted_margin_fraction=0.0112`,
  `train_targeted_margin_loss=0.04375`), but validation fell to
  macro/class-1 `0.8779/0.6629`, class-1 P/R `0.5756/0.7815`, below the keeper
  `0.8847/0.6860` and runtime softboost `0.8887/0.7030`.
- Reload/boundary/XAI:
  reload eval
  `runs\eval_smoke_v8_yolof_oofextrel_t07_marginrecall_val_20260705`
  confirmed full-val macro/class-1 about `0.8771/0.6629`. Confusion stayed in
  the old failure mode: `3->2=58`, `0->1=53-54`, `4->1=16`, `2->1=13`,
  `1->0=15`, `1->2=12`, and `1->4=6`. Boundary review
  `runs\boundary_review_smoke_v8_yolof_oofextrel_t07_marginrecall_val_20260705`
  selected `153` rows with `focus_false_negative=33`,
  `focus_false_positive=40`, and boundary pair `0-1=80`. Robust XAI
  `runs\xai_smoke_v8_yolof_oofextrel_t07_marginrecall_val_cases12_robust_20260705`
  again showed foreground surface/border sensitivity rather than useful broad
  context reasoning: background blur/gray drops were only
  `0.0013/-0.0000`, while object desaturation drop was `0.1425`; flags included
  `object_color_sensitive=5/12`, rollout background `7/12`, rollout border
  `6/12`, and register heatmap similarity `0.99985`.
- Decision:
  reject this train-OOF external reliability targeted-margin policy before any
  probe/full train. Even with explicit TP1 recall protection, a sparse
  case-level margin over the current embeddings widens class-1 false positives
  and does not create a new recall-safe surface/boundary representation. Do not
  repeat nearby EffV2/DINO OOF targeted-margin variants by only changing
  threshold, FP/TP ratio, margin, loss weight, or short-batch count. Future
  external transfer needs a stronger fold-safe representation signal or manual
  train-only review, not another sparse current-embedding margin policy.
- Cleanup:
  compact evidence was preserved in
  `runs\evidence_oofextrel_t07_marginrecall_rejected_20260705`, including the
  policy CSV/summary, smoke config/history/metrics/trace summary, reload eval
  predictions, boundary manifest, and XAI summaries. Raw smoke/eval/boundary/XAI
  directories were deleted via
  `runs\cleanup_manifest_20260705_oofextrel_t07_marginrecall_rejected.json`,
  reclaiming about `193.542 MB`. The source diagnostic and compact manifest
  remain.

## Diagnostic 2026-07-05 - TRKH-Only Softboost + Internal Ensemble Blend

- Rationale:
  after rejecting the train-OOF external reliability margin, I checked whether
  the older no-pretrain six-expert TRKH ensemble still adds any useful
  validation signal beyond the current runtime patch softboost verifier. This
  stayed TRKH-native: inputs were the current softboost full-val predictions and
  `runs\weighted_ensemble_no_pretrain_6expert_val_gate_20260701\teacher_probs_val_sampleindex_focus005.csv`.
  It was validation-only, wrote no training manifest, touched no test split, and
  did not edit raw data.
- Artifact:
  `runs\diagnostic_trkh_internal_ensemble_softboost_blend_20260705` joins both
  prediction sets strictly by `sample_index` over the full `2606` `yolo_f/val`
  rows and sweeps probability and log-probability blends. Baselines were:
  current runtime softboost macro/class-1 `0.8887/0.7030`, and internal
  no-pretrain ensemble `0.8864/0.6939`.
- Result:
  the best validation blend was a log-probability average with `alpha=0.50`,
  reaching macro/class-1 `0.8903/0.7091`, class-1 P/R
  `0.6536/0.7748`. It changed only `6` rows versus softboost: `4`
  corrections and `2` harms. The changes were fragile near-ties rather than a
  broad new signal: examples include one `0:1->0` correction, one `1:0->1`
  correction, but also `2:2->3` and `3:3->2` harms.
- XAI:
  verifier-aware XAI on the 6 changed rows is preserved at
  `runs\xai_trkh_internal_ensemble_softboost_blend_changed6_20260705`. It
  selected exactly those rows and found `near_tie_top2=4/6`, Grad-CAM border
  `3/6`, rollout border `3/6`, and high foreground mass. Background blur/gray
  drops were small (`0.0061/-0.0098`) while object desaturation was larger
  (`0.0747`). Register heatmaps were still nearly collapsed
  (`cls_register_heatmap_similarity=0.9993`).
- Decision:
  keep this as a small TRKH-only upper-bound diagnostic, not as a new train or
  deployment route. The gain over softboost is only `+0.0061` class-1 F1 on
  validation, depends on a validation-selected blend, changes only six
  near-tie rows, and does not move toward the next milestone `0.75`. Do not
  sweep more internal-ensemble/softboost blend weights, train a router from
  these validation changes, or use this as a sample-weight/KD target. Future
  TRKH work still needs a genuinely new representation or fold-safe
  recall/false-positive reliability signal.

## Cleanup 2026-07-05 - Compact Obsolete Non-Smoke XAI Case Images

- Rationale:
  after confirming the active target remains TRKH-native and the latest route
  families are documented as rejected, I compacted image-heavy XAI case folders
  from older rejected/superseded diagnostics. This was disk hygiene only: no
  raw dataset files, model checkpoints, metrics, summaries, CSV/MD audit files,
  current keeper/softboost/remaining-error XAI, or final-test XAI were removed.
- Artifact:
  `runs\cleanup_manifest_20260705_compact_obsolete_non_smoke_xai_case_images.json`
  records `24` selected XAI roots and `311` deleted `case_*` subdirectories,
  reclaiming about `1801.301 MB`. Candidate roots were path-checked under
  `runs`; post-delete verification found `0` remaining selected case dirs, and
  protected XAI roots were still present.
- Protected:
  current/high-value XAI stayed intact, including
  `runs\xai_softboost_verifier_remaining_errors16_20260705`,
  `runs\xai_eval_patch_linear_verifier_softboost001_changed4_verifieraware_20260705`,
  `runs\xai_trkh_internal_ensemble_softboost_blend_changed6_20260705`,
  `runs\xai_boundary_review_yolof_keeper_val_current_cases16_datasetcrop_20260704`,
  and final-test XAI roots from patch/stacked verifier diagnostics.
- Decision:
  treat the compacted folders as preserved evidence without per-case image
  payloads. Do not regenerate those obsolete XAI cases unless a new route needs
  a direct before/after comparison; use the retained root summaries and source
  metrics for historical decisions.

## Diagnostic 2026-07-05 - Scope Clarification and Train-Only Softboost Review

- Scope:
  TRKH remains the active implementation target. AIDT/top-5/timm/pretrained
  experiments in this phase are diagnostic, teacher, or upper-bound evidence
  only; they are not a pivot away from the no-pretrain TRKH research target.
  External-model validation superiority should not be compressed into TRKH
  through validation-selected routers, KD, sample weights, or targeted-margin
  manifests without a new train-only/OOF reliability source that explicitly
  protects true class-1 recall and suppresses `0/2/4->1` false positives.
- Literature cross-check:
  recent noisy fine-grained work still points to the same families already
  tested locally: sample selection/co-teaching and contrastive queues
  (`https://arxiv.org/abs/2303.02404`), co-transformer/noisy-label selection
  (`https://arxiv.org/abs/2503.14193`), and VLM-assisted cleaning such as
  CLIPCleaner (`https://arxiv.org/html/2408.10012v2`). These are useful
  framing references, but the current TRKH evidence says not to rerun nearby
  loss/router/background variants over the same embedding. Any VLM-style route
  must be train-only/manual-review or OOF diagnostic first, because the mango
  class-1 boundary is visual and domain-specific, not a generic class-name
  prompt problem.
- Train eval:
  `runs\eval_patch_linear_verifier_softboost001_train_20260705` evaluated the
  current keeper checkpoint with the runtime full-vector `0-1` patch verifier
  softboost on explicit `yolo_f/train` (`9215` rows), no test and no raw-data
  edits. It reached train macro F1 `0.9478`; class-1 precision/recall/F1 were
  `0.7389/0.9834/0.8438`. This is intentionally treated as in-sample
  diagnostic evidence, not as a deployable score or automatic label policy.
- Boundary review:
  `runs\boundary_review_softboost_train_current_20260705` selected `260` train
  rows from `1171` candidates with the same thresholds used for the validation
  boundary review. The queue is dominated by class-1 false positives:
  `focus_false_positive=120`, `focus_false_negative=9`,
  `low_margin_error=94`, `low_margin_correct_boundary=34`; boundary pairs were
  `0-1=120`, `1-2=75`, `2-3=42`, `4-rest=23`. Buckets were led by
  `0->1=90`, `2->1=27`, `3->2=31`, and only `1->0=8`.
- XAI:
  verifier-aware robust XAI on 12 selected train FP/FN rows is preserved at
  `runs\xai_softboost_train_focus_fpfn12_20260705`. It found high foreground
  focus (`attention/gradcam foreground` `0.9055/0.9321`) but persistent
  border/surface ambiguity (`rollout_border_attention=9/12`,
  `object_color_sensitive=6/12`). Background perturbations were much weaker
  than object-surface perturbation: `background_blur/gray` drops
  `0.0341/0.0161` versus `object_desaturate=0.1479`. Register heatmaps remain
  nearly collapsed (`cls_register_heatmap_similarity=0.99986`).
- Decision:
  keep this train review as a manual/fold-safe reliability source candidate,
  not as a smoke trigger by itself. The evidence again rejects the idea that
  wider `yolo_f` background alone will let ViT self-attention solve class 1.
  The next meaningful route must add a new recall-protecting
  surface/boundary representation signal or a genuinely fold-safe/manual
  reliability target. Do not launch another smoke that only changes logit
  gates, crop context, background suppression, color/edge shallow stats,
  current-embedding contrastive queues, or patch-verifier thresholds.

## Review 2026-07-05 - Softboost Train Triage Batch 01

- Rationale:
  since the current evidence rejects another current-embedding smoke, I expanded
  the train-only review route instead. This is generated audit/review work only:
  no raw dataset edit, no validation/test tuning, and no training manifest was
  created.
- Artifact:
  `runs\manual_softboost_train_triage_batch01_20260705` selects `58` rows from
  `runs\boundary_review_softboost_train_current_20260705`: all `9` class-1
  false negatives, `30` high-confidence `0->1` false positives, `12`
  high-confidence `2->1` false positives, `3` other non-class1-to-class1 false
  positives, and `4` low-margin correct `0/1` anchors. It writes
  `triage_batch01.csv`, five contact sheets, `visual_triage_notes.md`,
  `triage_batch01_visual_suggestions.csv`, and summary JSON files.
- Visual audit:
  class-1 false negatives are pale green, low-margin, often glare/shadow or
  subtle-spot cases close to class `0`. The `0->1` false positives show
  class-1-like surface cues on the fruit itself: speckles, scars, green-yellow
  gradients, stems, mesh/leaf shadows, and bright sun patches. The `2->1`
  false positives are mostly yellow/changing-surface boundary cases. A few
  other-to-class1 samples are tiny-object, multi-fruit, or complex-context
  candidates better suited to `needs_crop`/quality review.
- Guardrail:
  `triage_batch01_visual_suggestions.csv` deliberately keeps
  `manual_label_status` empty; verification found `manual_label_status_filled=0`
  across all `58` rows. Suggestion columns are review notes only. Do not pass
  this artifact to `build_reviewed_boundary_training_manifests.py` as a policy,
  and do not smoke from it.
- Decision:
  keep the artifact as review-only evidence. It strengthens the conclusion that
  the remaining class-1 issue is a real visual surface/boundary ambiguity. A
  future manual policy needs explicit reviewed status fields, larger coverage
  across both true-class1 recall protectors and class1 false-positive
  suppressors, and a dry-run through the existing sample-index-safe manifest
  builder before any full-val smoke.

## Diagnostic 2026-07-05 - Selfreview Draft Dry-Run From Triage Batch 01

- Scope:
  in response to the concern that external models might be distracting from
  TRKH, I kept the next step TRKH-native and train-only: a generated
  self-review draft from the existing softboost triage contact sheets, with no
  raw dataset edit and no validation/test input.
- Artifact:
  `runs\manual_softboost_train_triage_batch01_selfreview_draft_20260705`
  contains `triage_batch01_selfreview_draft.csv`, `dryrun_summary.json`, and a
  README. The original triage artifact remains unchanged. The builder was run
  with `dry_run=True`, so no sample-weight, soft-target, targeted-margin, or
  relabel CSV outputs were written.
- Dry-run result:
  all `58` rows matched by `sample_index`; statuses were `54` `correct` and
  `4` `needs_crop`. The hypothetical builder policy would create `50`
  targeted-margin rows and no soft targets: `1->0=8` true-class1 recall
  protectors, `0->1=30`, and `2->1=12` class1 false-positive suppressors.
  Weight range would be `0.45-1.04`.
- Decision:
  do not smoke from this draft. It is useful because it proves the
  sample-index-safe manual-policy path works, but it is too small and
  asymmetrical: only `8` recall-protecting true-class1 cases versus `42`
  non-class1 suppressors. That repeats the risk seen in earlier targeted-margin
  and verifier policies, where suppressing `0/2/4->1` also damaged true class 1.
  Keep this as a documented dry-run and require a larger explicit review set or
  a stronger fold-safe reliability signal before generating non-dry-run
  training manifests.

## Diagnostic 2026-07-05 - Balanced Train Reliability Batch 02

- Rationale:
  batch 01 proved the manual-policy path but was too small and suppressor-heavy.
  I therefore generated a larger train-only review candidate set from the
  current softboost train predictions, explicitly adding true-class1 low-margin
  true positives as recall protectors. This remains a generated review artifact:
  no raw dataset edit, no validation/test input, and no train manifest.
- Artifact:
  `runs\manual_softboost_train_balanced_reliability_batch02_20260705` contains
  `165` rows and six contact sheets. Composition is `9` class1 FN, `60`
  low-margin true-class1 TP recall protectors, `36` `0->1` candidates, `24`
  `2->1` candidates, `12` complex `3/4->1` candidates, and `24` low-margin
  non-class1 anchors. `manual_label_status` is intentionally empty for all
  rows.
- Visual audit:
  the TP1 recall-protector sheet and the `0/2->1` suppressor sheets overlap
  heavily: green-yellow gradients, speckles, stem/edge cues, leaf/mesh shadows,
  and lighting changes appear in both true class 1 and non-class1 rows. Complex
  `3/4->1` rows include multi-fruit, tiny object, decay, or background clutter
  cases, so they are poor candidates for hard relabeling.
- Builder update:
  `trkh.tools.build_reviewed_boundary_training_manifests` now accepts an
  optional `manual_sample_weight` / `review_sample_weight` column. Existing
  manifests are unchanged; the override only affects sample-weight output and
  is still capped by quality flags. Focused preflight passed:
  `py_compile trkh\tools\build_reviewed_boundary_training_manifests.py
  tests\test_reviewed_boundary_training_manifests.py` and
  `pytest tests\test_reviewed_boundary_training_manifests.py -q` (`4 passed`).
- Dry-run:
  `runs\manual_softboost_train_balanced_reliability_batch02_selfreview_dryrun_20260705`
  writes only `batch02_selfreview_draft.csv`, `dryrun_summary.json`, and README.
  Builder `dry_run=True` produced no trainable CSV outputs. The hypothetical
  policy covers all `165` rows by `sample_index`, with `60` TP1 recall weights
  (`1.05`), `8` FN1 recall margins (`1->0`), `60` light FP suppressor margins
  (`0->1=36`, `2->1=24`, weight `0.97`, margin weight `0.0007`), `13`
  needs-crop rows, and `24` anchors.
- Similarity conflict diagnostic:
  `runs\manual_softboost_train_batch02_similarity_conflicts_20260705` audits the
  same `165` train-only rows with aHash/dHash plus average RGB. It found `133`
  clusters, `21` multi-item clusters, `10` mixed-target clusters, and `6` mixed
  recall/suppressor clusters covering `20` rows. The contact sheets show near
  duplicate or presentation-similar fruit/background cases where true class 1
  recall protectors sit beside `0/2/4->1` suppressors. This is direct evidence
  that auto sample weighting or targeted margins from generated review labels
  would encode label-boundary noise, not a reliable class-1 rule.
- Decision:
  do not launch smoke yet. Batch02 is a better manual/fold-safe reliability
  candidate than batch01 because recall protectors are now explicit, and the
  builder can represent them. But the visual overlap is still severe and the
  similarity-conflict audit confirms that the labels are generated self-review,
  not confirmed ground truth. A non-dry-run manifest should require explicit
  approval plus a stricter review pass that separates genuinely clean recall
  protectors from ambiguous class1-like negatives.

## Diagnostic 2026-07-05 - Batch02 Strict Review Filter

- Rationale:
  after the similarity-conflict audit, I made a stricter train-only review
  filter before considering any smoke. The goal was to remove mixed
  recall/suppressor clusters and complex rows, then check whether the remaining
  review pool is balanced enough to justify a future manual pass. No raw data,
  validation, test, or trainable manifest is touched.
- Artifact:
  `runs\manual_softboost_train_batch02_strict_review_filter_20260705` reads
  batch02 candidates plus the similarity diagnostic and writes
  `strict_review_candidates.csv`, `strict_review_rejects.csv`, `summary.json`,
  README, and three contact sheets. It keeps `125/165` rows and rejects `40`.
  Kept rows are `56` recall protectors, `50` false-positive suppressors, and
  `19` anchors. Rejections include `20` mixed recall/suppressor rows, `28`
  mixed-target rows, `12` complex `3/4->1` rows, and one ultra-low-margin class1
  FN uncertainty; rows can have more than one rejection reason.
- Visual audit:
  the strict contact sheets are more balanced than batch01 and no longer show
  the worst near-duplicate mixed clusters, but the remaining recall-protector
  and suppressor pools still share the same fruit-surface cues: green/yellow
  gradients, speckles, stem/edge marks, leaf shadows, and harsh lighting. This
  is still a human-review priority queue, not a reliable auto-label policy.
- Decision:
  do not smoke from the strict filter. It is useful because it narrows batch02
  into a balanced manual-review worklist, but the review fields are still empty
  and the visual boundary is still too ambiguous. A future non-dry-run manifest
  needs explicit reviewed `manual_label_status` values and another dry-run that
  remains recall-balanced after filtering.

## Diagnostic 2026-07-05 - Structural-Label Precheck on Batch02 Strict Rows

- Rationale:
  I checked whether a Structural Labels / reverse-kNN direction could provide a
  genuinely new train-only reliability signal before implementing another loss
  or smoke. This follows the idea of structural labels for noisy-label learning,
  but is only a diagnostic over current TRKH keeper train embeddings and the
  strict batch02 review rows.
- Artifact:
  `runs\diagnostic_structural_labels_batch02_strict_20260705` uses the existing
  train-only keeper embedding cache
  `runs\local_neighbor_feature_cache_yolof_keeper_cmsf_head_20260704\local_neighbor_teacher_features.npz`.
  It computes top-`30` forward kNN and reverse-kNN class fractions for all
  `9215` train objects, then joins `125` strict batch02 rows. It writes
  `strict_batch02_structural_rows.csv`, `summary.json`, README, and three
  contact sheets. No raw data, validation, test, or trainable manifest is used.
- Result:
  the structural class1 signal is inverted for this hard queue. Recall-protector
  rows have mean structural class1 blend `0.3488`, while false-positive
  suppressor rows have `0.6592`. AUC for recall-protector score greater than
  suppressor score is only `0.0996` for structural class1 blend, `0.1514` for
  forward class1 fraction, and `0.0727` for reverse class1 fraction. The only
  moderately separating signal is reverse same-label fraction (`AUC=0.6479`),
  but it still depends on the noisy hard label and does not produce a clean
  class1 recall-safe target.
- Visual audit:
  contact sheets show why the signal is inverted: many `0/2->1` suppressors are
  genuinely class1-like in current embedding space, while many true class1
  recall-protectors are pale/low-margin cases with weak class1 neighborhood
  support. A structural-label loss on this embedding would likely reinforce
  class1 false positives before rescuing true class1 recall.
- Decision:
  reject structural-label/reverse-kNN loss or manifest generation on the current
  keeper before smoke. This is useful negative evidence: structural-label ideas
  require a stronger feature source or confirmed manual labels here, not another
  current-embedding neighbor objective.

## Diagnostic 2026-07-05 - External-Feature Structural Support

- Rationale:
  since current TRKH embeddings invert the structural-label signal on batch02, I
  tested whether stronger existing train-only feature caches offer a different
  reliability surface. This stays diagnostic: EffV2-S/DINOv2 remain
  external/pretrained evidence, not the TRKH final target, and no trainable
  manifest is created.
- Train strict batch02 diagnostic:
  `runs\diagnostic_structural_labels_external_features_batch02_strict_20260705`
  computes same-source-excluded top-`30` structural support over strict batch02
  rows for three train feature banks. EffV2-S is the only source with strong
  separation: structural class1 blend AUC recall-protector > suppressor is
  `0.9957`, with recall mean `0.8812` and suppressor mean `0.1538`. Direct
  DINOv2-small remains weak/inverted (`AUC=0.2998`, recall mean `0.2087`,
  suppressor mean `0.3168`), as does the class_f-remapped DINOv2 cache
  (`AUC=0.2875`).
- Full-val softboost diagnostic:
  because the earlier EffV2-S OOF readout was already known to overfit train OOF
  and underperform on validation class1, I checked a fixed validation diagnostic
  before considering any route. Artifact
  `runs\diagnostic_effv2_structural_support_softboost_val_20260705` lets full
  `yolo_f/val=2606` query the EffV2-S train feature bank. Thresholds were fixed
  from the structural precheck (`low=0.35`, `high=0.55`) and not swept on val.
  Overall class1-vs-non1 structural support is real (`AUC=0.8976`, true1 mean
  `0.6596`, non1 mean `0.0213`), but it does not solve the remaining softboost
  boundary: class1 FP mean support is `0.3714`, class1 FN mean support is
  `0.2857`, and fixed guard changes `89` rows with `44` corrections, `41`
  harms, and `4` neutral.
- Gate result:
  fixed EffV2 structural guard lowers class1 F1 from current softboost
  `0.7030` to `0.6990` and leaves macro about flat/lower (`0.8887` to
  `0.8882`). Contact sheets show the same issue: EffV2 blocks many obvious
  class1 false positives and rescues some class1 false negatives, but the
  remaining high-p1 FP and low/medium-p1 FN are the same green/yellow
  surface-boundary ambiguity.
- Decision:
  keep EffV2-S structural support as a useful review-prioritization and
  representation diagnostic, but do not turn it into TRKH KD, router,
  sample-weight, targeted-margin, or structural-label loss on the current
  keeper. Any future use must be fold-safe/manual and must prove true-class1
  recall protection on a dry-run before smoke.

## Review 2026-07-05 - EffV2 Structural Priority Worklist

- Rationale:
  EffV2-S structural support is not safe enough as an automatic validation guard,
  but it can reduce manual-review effort by separating likely true-class1
  protectors from likely non-class1 suppressors inside the already strict
  batch02 queue. I therefore created a review worklist only, not a trainable
  artifact.
- Artifact:
  `runs\manual_softboost_train_batch02_effv2_structural_review_priority_20260705`
  reads the EffV2 strict-row diagnostic and writes
  `effv2_structural_priority_review_candidates.csv`, reject CSV, summary,
  README, three contact sheets, and
  `effv2_structural_priority_review_report.html`. It keeps `95/125` strict rows:
  `55` recall protectors with EffV2 structural p1 `>=0.55` and `40` FP
  suppressors with EffV2 structural p1 `<=0.35`. It rejects `30` rows:
  one recall protector below `0.55`, `10` suppressors above `0.35`, and all
  `19` anchors outside the mid-band.
- HTML report:
  `effv2_structural_priority_review_report.html` renders all `95` rows as local
  review cards with the source image, transition, softboost score, EffV2
  structural p1, review order, suggested status, and empty manual fields. The
  paired summary JSON verifies `manual_label_status_filled=0/95` and
  `missing_images_rendered=0`.
- Tooling:
  I added `trkh.tools.render_review_worklist_html` plus focused tests so future
  train-only review worklists can be rendered from arbitrary CSV schemas without
  ad hoc scripts or boundary-manifest assumptions. The tool only creates HTML
  and summary JSON; it does not create trainable manifests or edit raw data.
- Guardrail:
  manual fields remain empty, and no sample-weight, soft-target, relabel, or
  targeted-margin CSV is created. Contact sheets confirm the worklist is more
  reviewable than raw batch02, but it still contains visually hard green/yellow
  surface and lighting boundary cases.
- Decision:
  keep this as a manual review queue only. Do not feed it to
  `build_reviewed_boundary_training_manifests.py` until reviewed
  `manual_label_status` values are filled and a new dry-run proves the resulting
  policy remains recall-balanced.

## Literature Check 2026-07-05 - Review/Fold-Safe Reliability Before More Smokes

- Sources checked:
  imbalanced noisy-data sample selection
  <https://arxiv.org/html/2402.11242v1>, fine-grained noisy-label learning
  <https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wei_Fine-Grained_Classification_With_CVPR_2023_supplemental.pdf>,
  transformer-based FGVC survey/review
  <https://link.springer.com/article/10.1007/s00371-024-03545-6>, and VLM-based
  image-classification label renovation
  <https://arxiv.org/html/2505.16149v1>.
- Mapping to TRKH:
  class-balanced noisy sample selection and small-loss variants point toward
  per-class, confidence-aware review policies, but local self-paced,
  class-balanced self-paced, co-teaching, cleanlab/neighborhood, and OOF
  reliability attempts on the current keeper have already widened class-1 false
  positives or produced sparse/disagreeing signals. Fine-grained transformer and
  part-localization ideas overlap with already rejected local-token, part-token,
  late class-attention, block-local, patch-router, SnapMix, background/context,
  and high-frequency routes unless a new target is added.
- Decision:
  do not launch another generic sample-selection, part-attention, or VLM/teacher
  auto-cleaning smoke on the current embeddings. The literature supports using
  external/pretrained/VLM-style evidence as manual review prioritization or a
  fold-safe reliability source only after it explicitly protects true class-1
  recall and controls `0/2/4->1` false positives.

## Cleanup 2026-07-05 - Compact class_f OOF Fold Materialization

- Rationale:
  `runs\classf_oof_folds_train5_20260704` was a generated hardlink-only fold
  materialization used for the rejected class_f MobileNetV3 fold00 diagnostic.
  The useful evidence is the documented rejection plus `summary.json`; the fold
  trees can be rebuilt deterministically if a future fold-safe class_f route
  genuinely needs them.
- Action:
  deleted only `fold_00` through `fold_04` under
  `runs\classf_oof_folds_train5_20260704`, kept `summary.json`, and wrote
  `COMPACTED_README.md` plus
  `runs\cleanup_manifest_20260705_compact_classf_oof_fold_materialization.json`.
  All removed paths were verified under the target run directory before
  recursive deletion.
- Verification:
  raw `D:\DataAI\AIEx\newdataset\class_f` still exists; current
  `runs\yolof_oof_folds_train5_20260704\fold_00\data.yaml` still exists; the
  compacted class_f `fold_00` no longer exists. Measured deleted link targets
  total `4459.693 MB`, but actual free-space delta was only `11.336 MB`, which
  is expected for hardlinks.
- Preservation audit:
  `runs\artifact_inventory_current_trkh_20260705\summary.json` verifies the
  current keeper artifacts still exist after cleanup: best no-pretrain
  validation anchor, runtime softboost full-val hook, final no-pretrain test
  audit, patch-verifier final-test diagnostic, patch linear verifier export,
  current `yolo_f` OOF folds, EffV2 priority review worklist, journal, and TODO.
  It also records that `classf_oof_folds_train5_20260704\fold_00` is
  intentionally absent while `summary.json` remains.
- Guardrail:
  do not treat the class_f OOF materialized folders as live evidence. Regenerate
  them with `trkh.tools.build_classification_oof_folds` from
  `D:\DataAI\AIEx\newdataset\class_f\data.yaml`, train split, 5 folds, seed 42,
  if a future fold-safe class_f experiment needs folder materialization again.

## Guardrail 2026-07-05 - Review Worklist Readiness Audit

- Rationale:
  the only currently viable path is manual/fold-safe reliability, but earlier
  failed policies show that a small or imbalanced reviewed subset can suppress
  true class 1. I added a pre-builder audit so review worklists must prove
  manual fill, sample-index safety, train-only paths, recall-protector coverage,
  FP-suppressor coverage, and no mixed-side cluster conflicts before any
  trainable manifest is considered.
- Tool:
  `trkh.tools.audit_review_worklist_readiness` reads a review CSV and writes
  `readiness_summary.json`; it creates no sample-weight, soft-target,
  targeted-margin, relabel, or raw-data output. Focused tests cover empty manual
  fields, balanced actionable review, and mixed-side cluster blocking.
- EffV2 priority audit:
  `runs\manual_softboost_train_batch02_effv2_structural_review_priority_20260705\readiness_audit`
  checks the `95` priority rows. It confirms `missing_sample_index_rows=0`,
  `missing_image_path_rows=0`, `non_train_path_rows=0`, and no mixed-side
  clusters, but `manual_label_status_filled=0/95`, actionable recall `0<20`,
  actionable FP `0<20`, and therefore `ready_for_training_manifest=false`.
- Decision:
  keep EffV2 priority review-only. Do not pass it to
  `build_reviewed_boundary_training_manifests.py` or run a smoke until manual
  decisions are filled and the readiness audit passes with recall-balanced
  coverage.

## Guardrail 2026-07-05 - Review Queue Readiness Matrix

- Rationale:
  after adding the per-CSV readiness guard, I ran it across the active train-only
  review queues to avoid accidentally treating one unfilled queue as trainable.
- Artifact:
  `runs\review_worklist_readiness_matrix_20260705` audits
  `boundary_review_softboost_train_current_20260705`,
  `manual_softboost_train_triage_batch01_20260705`,
  `manual_softboost_train_balanced_reliability_batch02_20260705`,
  `manual_softboost_train_batch02_strict_review_filter_20260705`, and
  `manual_softboost_train_batch02_effv2_structural_review_priority_20260705`.
- Result:
  `ready_count=0/5`. Every queue has valid sample-index/train-path coverage, but
  all are blocked because `manual_label_status` is empty and actionable
  recall/FP counts are below the configured minimum. The strict batch02 queue
  has `2` mixed suppressor+anchor clusters, but the audit treats anchors as
  neutral warnings; blocking mixed actionable-side clusters are `0`.
- Decision:
  no reviewed-manifest build and no smoke from any current manual queue. The
  closest candidates remain strict batch02/EffV2 priority after real manual
  labels are filled and the readiness matrix passes.

## Guardrail 2026-07-05 - Reviewed-Manifest Builder CLI Readiness Gate

- Change:
  `trkh.tools.build_reviewed_boundary_training_manifests` now runs
  `trkh.tools.audit_review_worklist_readiness` from the CLI before non-dry-run
  manifest generation unless `--skip-readiness-check` is explicitly set. It
  infers the review side columns for strict review, triage-group, and boundary
  reason schemas. Non-dry-run builds are refused when readiness fails; dry-run
  still runs so policy/sample-index behavior can be inspected without writing
  trainable manifests.
- Verification:
  focused tests cover CLI refusal on unreviewed CSVs and dry-run warning
  behavior. The live EffV2 priority guard check at
  `runs\manual_softboost_train_batch02_effv2_structural_review_priority_20260705\builder_cli_guard_check`
  wrote a readiness summary, then produced no trainable CSVs because
  `manual_label_status_filled=0/95`.
- Decision:
  keep this guard in place. Any future non-dry-run reviewed-manifest build must
  either pass readiness or deliberately use the explicit skip flag with a
  documented reason.

## Cleanup 2026-07-05 - Compact Rejected Verifier XAI Case Images

- Rationale:
  two documented rejected verifier/XAI diagnostics still contained image-heavy
  `case_*` folders even though their decisions are already recorded. The root
  metrics, summaries, CSVs, and markdown audits are sufficient historical
  evidence; the per-case rendered images are no longer active work products.
- Action:
  deleted only `case_*` subdirectories under
  `runs\xai_stacked_patch_feature_verifier_yolof_keeper01_class1benefit_test_changed12_20260703`
  and
  `runs\xai_patch_evidence_extratrees_leaf5_reject_changed7_20260703`.
  The cleanup manifest is
  `runs\cleanup_manifest_20260705_compact_rejected_xai_test_extratrees_case_images.json`.
- Verification:
  both roots still exist, now with `0` `case_*` directories. Required root
  files `xai_audit_summary.json`, `xai_metrics.json`, and `xai_cases.csv` are
  still present. The manifest records `19` removed case directories,
  `97.376 MB` measured deleted case material, `97.824 MB` free-space delta,
  and `raw_dataset_touched=false`.
- Protected artifacts:
  verified current runtime softboost full-val hook
  `runs\eval_patch_linear_verifier_softboost001_full_val_20260705`,
  remaining-error XAI
  `runs\xai_softboost_verifier_remaining_errors16_20260705`, final-test
  patch-evidence XAI
  `runs\xai_patch_evidence_mil_yolof_keeper_01only_test_changed11_20260703`,
  final no-pretrain test audit
  `runs\eval_yolof_teacherfocusbinary015_best_test_final_20260702`, and raw
  `class_f` / `yolo_f` dataset roots.
- Guardrail:
  do not regenerate these compacted rejected case images unless a new method
  explicitly needs visual side-by-side evidence. Use the retained root metrics
  and CSVs for historical decisions.

## Diagnostic 2026-07-05 - Distribution-Aware Soft Centroid Prototype Precheck

- Rationale:
  DaSC proposes distribution-aware class centroid estimation by averaging all
  training features with weights from temperature-scaled model predictions
  (<https://arxiv.org/abs/2407.16802>). This is materially different from the
  previously rejected hard-label centroid, Mahalanobis, ETF, and kNN probes, so
  I added a no-test/no-manifest precheck before considering any prototype loss
  or centroid target.
- Implementation:
  `trkh.tools.probe_embedding_prototypes` now supports default-off
  `--soft-centroid-temperature` and `--soft-centroid-min-confidence`. It fits
  soft centroids only from `yolo_f/train` frozen embeddings and train model
  probabilities, then evaluates unchanged requested splits. Focused tests cover
  probability temperature scaling and probability-weighted centroid prediction.
- Artifact:
  `runs\diagnostic_soft_centroid_yolof_keeper_trainprob_20260705` uses the
  current keeper checkpoint, explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, full `train=9215` and
  `val=2606`, skips logistic/KNN/ETF, and writes `summary.json`,
  `val_prototype_predictions.csv`, and `soft_centroid_audit_summary.json`.
- Result:
  base head on this diagnostic path is macro/class-1 `0.8829/0.6783`.
  Hard centroid cosine is only `0.8417/0.5602`. Best soft centroid
  (`temperature=0.5`, all confidence settings identical because train max
  probability mean is only about `0.3168`) reaches only `0.8450/0.5455`,
  class-1 P/R `0.5548/0.5364`, and FP/FN `65/70`. It changes `171` validation
  rows versus base with `54` corrections, `103` harms, and `14` neutral:
  it fixes `0:1->0=29`, but harms true class 1 with `1:1->0=26` and
  `1:1->2=11`, and creates `4:4->1=16`.
- Decision:
  reject soft-centroid / distribution-aware prototype targets on the current
  keeper embedding. Do not convert this into a prototype loss, router, sample
  selection rule, or targeted manifest. It confirms that the current feature
  geometry still lacks recall-safe class-1 surface/boundary support; any future
  centroid/prototype route needs a new representation or a stronger fold-safe
  reliability signal first.

## Diagnostic 2026-07-05 - HFlip Decision-Instability Precheck

- Rationale:
  hflip TTA aggregation was already rejected on 2026-07-03, but the remaining
  errors are near-tie and border/surface sensitive, so I checked whether
  clean-vs-hflip instability itself could be a reliability signal before
  considering any equivariant consistency objective. This follows the same
  caution as MEMO-style augmentation consistency
  (<https://arxiv.org/abs/2110.09506>): label-preserving transforms can help
  only when the transform signal is reliable for the current error mode.
- Implementation:
  added `trkh.tools.audit_hflip_decision_instability` plus focused tests. The
  tool joins the previously validated full-val hflip artifact
  `runs\geometric_tta_hflip_keeper_yolof_val_full_noamp_20260703` with current
  runtime softboost full-val predictions
  `runs\eval_patch_linear_verifier_softboost001_full_val_20260705` by
  `sample_index`. It writes only audit CSV/JSON/README; no test, raw-data edit,
  threshold tuning, aggregation route, or train manifest.
- Artifact:
  `runs\diagnostic_hflip_softboost_instability_20260705`, full
  `yolo_f/val=2606`, target mismatches `0`. The tool now marks unsafe results
  as `reject_current_hflip_instability_as_training_signal`.
- Result:
  clean baseline from the hflip artifact is macro/class-1 `0.8829/0.6783`;
  pure hflip is worse at `0.8786/0.6553`; current softboost is
  `0.8887/0.7030`. Hflip changes only `59/2606` rows, with `26`
  clean-wrong to flip-right corrections versus `29` clean-right to flip-wrong
  harms. Against current softboost it would correct `28` errors but break `39`
  softboost-correct rows. For softboost class-1 FN it rescues only `7/35`,
  while true-class1 correct rows suffer `8` flip breaks. For softboost class-1
  FP, only `5` changed rows move clean class1 predictions away from class1, and
  non1-correct rows create `15` clean-non1 to flip-class1 FP risks.
- XAI follow-up:
  verifier-aware FN audit
  `runs\xai_hflip_softboost_instability_cases16_20260705` shows object
  desaturation drop `0.0764`, much larger than background blur/gray
  `0.0056/0.0035`, with near-tie `6/16` and rollout border `9/16`. FP audit
  `runs\xai_hflip_softboost_class1fp_cases16_20260705` shows object
  desaturation drop `0.1343` versus background blur/gray `0.0170/0.0169`,
  rollout border `11/16`, and Grad-CAM border `8/16`.
- Decision:
  reject hflip decision instability as a current training signal. Do not enable
  hflip TTA aggregation, hflip consistency loss, or a hflip-instability
  sample-weight/targeted-margin/router policy on the current keeper. It can
  recover a few true class1 FNs but is not recall-safe and can create
  `0/2/4->1` false positives.

## Cleanup 2026-07-05 - Compact Rejected HFlip XAI Case Images

- Rationale:
  after the hflip decision-instability route was rejected and its metrics/XAI
  conclusions were recorded in journal/TODO/skill, the per-case rendered XAI
  folders were no longer active work products. The root summaries, metrics,
  CSVs, and markdown audits are sufficient evidence for the decision.
- Action:
  deleted only `case_*` subdirectories under
  `runs\xai_hflip_softboost_instability_cases16_20260705` and
  `runs\xai_hflip_softboost_class1fp_cases16_20260705`. The cleanup manifest is
  `runs\cleanup_manifest_20260705_compact_rejected_hflip_xai_case_images.json`.
- Verification:
  removed `32` case directories, measured `25.367 MiB` of case material, and
  observed a free-space delta of `27.337 MB`. Both roots still contain
  `xai_audit_summary.json`, `xai_metrics.json`, `xai_cases.csv`, and
  `xai_audit.md`, with `case_dir_count_after=0`.
- Protected artifacts:
  verified current runtime softboost full-val metrics, remaining-error XAI,
  final no-pretrain test audit, and raw `class_f` / `yolo_f` data YAML files.
  `raw_dataset_touched=false`.

## Diagnostic 2026-07-05 - Class-Specific Patch-Query Readout Precheck

- Rationale:
  INTR-style class-specific queries (<https://arxiv.org/abs/2311.04157>) are a
  different representation surface from the rejected single-query late
  class-attention, part-token pairwise, and patch-verifier threshold routes. I
  added an offline no-test precheck before touching the main trainer: a tiny
  readout learns one query per class over selected frozen bbox/crop-bbox patch
  tokens, anchored by base log probabilities. MaskFeat
  (<https://arxiv.org/abs/2112.09133>) is kept only as related local-feature
  context; this route does not implement masked pretraining.
- Implementation:
  added `trkh.tools.probe_class_specific_patch_query` and
  `tests\test_probe_class_specific_patch_query.py`. The tool fits only the
  small readout on `yolo_f/train`, evaluates full `yolo_f/val`, refuses test by
  default, and writes no relabel/sample-weight/soft-target/targeted-margin
  manifest. Focused preflight passed `py_compile` and `pytest` (`2 passed`).
- Smoke checks:
  the `512/256` residual precheck with `residual_logit_scale=0.12` changed no
  predictions, so it only proved base logits dominated. The standalone no-base
  precheck was destructive (`61` changes, `57` harms, class-1 F1 `0` on the
  small first-256 val slice), so I ran one full diagnostic with base anchoring
  and a moderate residual instead of integrating the module into TRKH.
- Full artifact:
  `runs\diagnostic_classquery_patchreadout_full_base1_res05_20260705`, explicit
  `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, train `9215`, full val `2606`,
  max `24` selected patch tokens/sample, `query_epochs=8`,
  `base_logit_scale=1.0`, `residual_logit_scale=0.5`, no test.
- Result:
  base full-val macro/class-1 F1 on this diagnostic path is `0.8841/0.6841`;
  class-query readout falls to `0.7960/0.4426`. It changes `306` validation
  rows with only `74` corrections, `212` harms, and `20` neutral changes. The
  main harms are true class-1 recall losses (`1:1->0=44`, `1:1->2=22`) plus
  large class-4 and class-2 harms (`4:4->0=34`, `4:4->2=26`, `2:2->3=20`).
- XAI follow-up:
  `runs\xai_classquery_patchreadout_full_changed16_20260705` audited `12`
  true-class1 harms plus `4` corrections. Background perturbations remain weak
  (`background_blur/gray` original-probability drops `-0.0003/0.0034`), while
  object desaturation is much larger (`0.1336`). The changed cases are still
  foreground/surface/border driven: Grad-CAM border `9/16`, rollout border
  `11/16`, and register heatmaps remain collapsed
  (`cls_register_heatmap_similarity=0.99995`).
- Decision:
  reject offline class-specific patch-query readouts on the current keeper
  embedding. Do not integrate an INTR-style class-query head, class-query
  residual router, or nearby base/residual scale sweeps over the same frozen
  patch tokens. A future class-query revisit needs a new trainable
  representation target or verified fold-safe/manual reliability source that
  explicitly protects true class-1 recall before any smoke.

## Cleanup 2026-07-05 - Compact Rejected Class-Query Artifacts

- Action:
  deleted the two small class-query precheck directories and only the
  `case_*` subdirectories under
  `runs\xai_classquery_patchreadout_full_changed16_20260705`. The manifest is
  `runs\cleanup_manifest_20260705_classquery_patchreadout_rejected_aux.json`.
- Verification:
  removed `18` directories, measured `12.753 MiB`, observed `13.109 MB`
  free-space delta, and kept the full diagnostic `summary.json`,
  `predictions_class_query.csv`, `changed_class_query.csv`, plus root XAI
  `xai_metrics.json`, `xai_audit_summary.json`, `xai_cases.csv`, and
  `xai_audit.md`.
- Protected artifacts:
  verified current runtime softboost full-val metrics, remaining-error XAI,
  final no-pretrain test audit, and raw `class_f` / `yolo_f` data YAML files.
  `raw_dataset_touched=false`.

## Guardrail 2026-07-05 - EffV2 Priority Worklist Visual Pass

- Action:
  reviewed the three contact sheets in
  `runs\manual_softboost_train_batch02_effv2_structural_review_priority_20260705`
  and wrote `CODEX_VISUAL_PASS_20260705.md`.
- Result:
  the EffV2-priority split remains useful for ordering manual review, but it is
  not safe to auto-fill. Recall protectors and FP suppressors overlap heavily in
  green/yellow surface tone, glare, speckles, stem/edge marks, partial crops,
  and shadow. This matches prior evidence that sparse targeted-margin policies
  suppress some class-1 false positives while converting true class-1 rows to
  `0/2`.
- Decision:
  keep the EffV2 priority queue review-only. Do not build non-dry-run reviewed
  manifests until explicit review fields are filled and readiness passes with
  recall-balanced actionable rows.

## Cleanup 2026-07-06 - Compact Recent Rejected XAI Case Images

- Rationale:
  the recent rejected/superseded XAI roots already had their metric summaries,
  decisions, and source evidence recorded in TODO/skill/journal. The per-case
  rendered image folders were no longer needed for active comparison and were
  consuming disk.
- Action:
  compacted only `case_*` subdirectories from `16` documented rejected or
  superseded roots, including SSL-DINO surface-detail, light RandAugment,
  border/interior TTA, auxiliary `class_f` patch-evidence, surfaceamp bboxcompat,
  rejected timm fold00 XAI, internal-ensemble blend XAI, and meta-gate XAI. The
  manifest is
  `runs\cleanup_manifest_20260706_compact_rejected_recent_xai_case_images.json`.
- Verification:
  deleted `166` case directories, measured `89.454 MiB` of case material, and
  observed `92.605 MiB` free-space delta. Every compacted root still exists and
  keeps root `xai_metrics.json`, `xai_audit_summary.json`, `xai_cases.csv`,
  `xai_audit.md`, or source evidence where present. Protected artifacts all
  still exist: runtime softboost full-val, final no-pretrain test audit,
  softboost remaining-error XAI, softboost train-review XAI, verifier-aware
  runtime softboost XAI, final patch-evidence test XAI, and raw `class_f` /
  `yolo_f` data YAMLs. `raw_dataset_touched=false`.
- Intentionally retained case images:
  `xai_softboost_verifier_remaining_errors16_20260705`,
  `xai_boundary_review_yolof_keeper_val_current_cases16_datasetcrop_20260704`,
  `xai_softboost_train_focus_fpfn12_20260705`,
  `xai_eval_patch_linear_verifier_full_val_changed12_20260705`,
  `xai_patch_evidence_mil_yolof_keeper_01only_test_changed11_20260703`, and
  `xai_eval_patch_linear_verifier_softboost001_changed4_verifieraware_20260705`
  remain visual evidence for current/final decisions.

## Route Audit 2026-07-06 - VLM/Noisy-Label Literature Does Not Unlock Auto-Training

- Sources checked:
  DeFT (<https://arxiv.org/html/2409.19696v1>) learns positive/negative class
  prompts to detect noisy labels; CLIPCleaner
  (<https://arxiv.org/html/2408.10012v2>) uses zero-shot CLIP as an offline
  clean-sample selector; NLPrompt
  (<https://openaccess.thecvf.com/content/CVPR2025/papers/Pan_NLPrompt_Noise-Label_Prompt_Learning_for_Vision-Language_Models_CVPR_2025_paper.pdf>)
  purifies VLM prompt-learning data with robust losses and text prototypes;
  Jo-SNC (<https://arxiv.org/html/2601.12795v1>) combines self/neighbor
  consistency with per-class thresholds; CCT
  (<https://arxiv.org/html/2503.03042v1>) uses co-transformer contrastive
  learning for noisy labels; Early Cutting
  (<https://arxiv.org/html/2502.08227v3>) targets mislabeled easy examples.
- Local fit:
  these papers support the existing local conclusion: use external/VLM or
  neighbor evidence as review/fold-safe reliability signals, not as blind
  automatic relabeling or another loss/router over the current embedding.
  Current TRKH evidence is stricter than the generic assumptions: manual fields
  are empty, train OOF support is sparse/disagreement-heavy, focus-only external
  consensus hurts class 1, and EffV2 fixed validation guard falls below
  softboost despite high structural AUC on the review queue.
- Decision:
  do not launch a new automatic smoke from VLM/noisy-label literature alone.
  A valid next TRKH-native loop still needs a genuinely new fold-safe reliability
  source or representation target that explicitly protects true class-1 recall
  while controlling `0/2/4->1` false positives. Until then, keep the active
  path at runtime softboost diagnostics plus manual/fold-safe review work.

## Diagnostic 2026-07-06 - Softboost Class-1 Correction Budget

- Rationale:
  after the current automatic routes repeatedly traded class-1 recall against
  false positives, I quantified how large the remaining gap is before launching
  any new smoke. This is a validation-only sizing audit over current softboost
  predictions, not a threshold/router search.
- Artifact:
  `runs\diagnostic_softboost_class1_correction_budget_20260706`, based on
  `runs\eval_patch_linear_verifier_softboost001_full_val_20260705\predictions_detailed.csv`.
  It writes `summary.json`, `transition_correction_budget.csv`, and
  `class1_milestone_budget.csv`; no test split, raw-data edit, or trainable
  manifest is involved.
- Result:
  current softboost class-1 counts are `TP=116`, `FP=63`, `FN=35`, giving
  class-1 F1 `0.7030`. If a hypothetical route corrected only false negatives,
  the minimum arbitrary corrections are `13` for F1 `0.75` and `27` for
  F1 `0.80`. Higher targets require both sides: F1 `0.85` needs all `35`
  false negatives plus `10` false positives suppressed; F1 `0.90` needs all
  `35` false negatives plus `30` false positives; F1 `0.98` needs all `35`
  false negatives plus `57/63` false positives.
- Transition impact if fully solved:
  correcting all `0->1` rows would move class-1 F1 to about `0.8028`, all
  `1->0` rows to `0.7736`, all `1->2` rows to `0.7449`, and each of `2->1`
  or `4->1` to `0.7227`. These are oracle transition corrections and must not
  be converted into validation-tuned routing rules.
- Decision:
  this reinforces the gate for the next loop. A useful signal must correct many
  true class-1 false negatives or suppress many class-1 false positives without
  reciprocal harms. Small current-embedding selectors, local threshold tweaks,
  or sparse targeted-margin policies cannot plausibly close the gap.

## Diagnostic 2026-07-06 - External Support Scope Check for Current TRKH

- Rationale:
  the external/AIDT/top-5/EffV2/DINO checks are useful only if they explain or
  supply a fold-safe reliability signal for TRKH. I reran a narrow support audit
  on the current TRKH runtime softboost remaining validation errors to make that
  boundary explicit before launching any new smoke.
- Artifact:
  `runs\diagnostic_current_trkh_external_support_20260706`, using current
  softboost full-val predictions as base and the EffV2-S / DINOv2-small val
  readouts as support probes. It writes
  `softboost_remaining_errors_val.csv`, `remaining_error_external_support.csv`,
  `summary.json`, and `README.md`. This is validation-only diagnostic evidence;
  no test split, raw-data edit, or trainable manifest is involved.
- Result:
  current softboost still has `203/2606` validation errors. EffV2 alone changes
  many rows with `97` corrections and `67` harms; DINOv2 changes are noisier
  with `94` corrections and `149` harms. Two-model consensus is an upper-bound
  route only: it would reach validation macro/class-1 F1 `0.9034/0.7237` with
  `67` corrections, `28` harms, and `4` wrong-to-wrong changes. The focus-only
  consensus route also reaches class-1 `0.7237`, but it changes only `50` rows
  with `30` corrections and `18` harms.
- Class-1 transition support:
  external consensus mostly helps suppress class-1 false positives: `0->1`
  `9/41`, `2->1` `7/9`, `4->1` `5/9`, and `3->1` `3/4` consensus-correct.
  It is weak for true class-1 false negatives: `1->0` `5/19`, `1->2` `0/11`,
  and `1->4` `1/5`. This is below the correction budget needed to move class-1
  toward the next milestones and repeats the known recall-suppression risk of
  external guards.
- Decision:
  TRKH remains the active implementation target. External models explain the
  representation gap and can prioritize manual/fold-safe review, but the current
  EffV2/DINO consensus is not safe to compress into KD, router, sample weights,
  targeted margins, or another smoke. A valid next TRKH route still needs a new
  representation target or fold-safe/manual reliability source that covers
  `1->2` / `1->4` recall while controlling `0/2/4->1` false positives.

## Diagnostic 2026-07-06 - Signal-Gap Readiness Crosswalk

- Rationale:
  after the external-support scope check, I tied the remaining validation
  transition budget to every currently available legal support source:
  EffV2/DINO validation support, train-only EffV2/DINO OOF support, manual
  review queues, review-readiness audit, and patch-OOF cleanlab multibank
  policy. This answers whether any existing source justifies another automatic
  TRKH smoke without touching raw data or test.
- Artifact:
  `runs\diagnostic_trkh_signal_gap_readiness_20260706`. It writes
  `summary.json`, `transition_gap_crosswalk.csv`, and `README.md`. The artifact
  declares `raw_dataset_touched=false`, `test_split_used=false`, and
  `trainable_manifest_written=false`.
- Result:
  smoke gate is `false`. The current softboost class-1 F1 remains `0.7030`;
  the EffV2/DINO consensus upper bound is only `0.7237`, below the next
  `0.75` milestone. Current review readiness is `0/5`, all review queues have
  empty manual fields, and the patch-OOF cleanlab multibank policy still has
  `strict_rows=0`.
- Transition crosswalk:
  `0->1` is well represented but unsafe to automate: `41` validation errors,
  `9` val consensus-correct rows, `160` train-OOF rows, and `182` review rows,
  but `0` manual-filled rows. `2->1` similarly has `7/9` val consensus-correct
  rows and `84` review rows, but `0` manual-filled rows. The recall gaps are
  the blocker: `1->2` has `11` validation errors, `0` consensus-correct rows,
  `0` train-OOF rows, and `0` current review rows; `1->4` has `5` validation
  errors, only `1` consensus-correct row, `1` train-OOF row, and `2` review
  rows with `0` manual-filled rows.
- Decision:
  do not launch another automatic smoke from the existing external/review/OOF
  sources. They are useful for review priority and explanation, but not enough
  for KD/router/sample-weight/targeted-margin training. The next valid signal
  must be a genuinely new fold-safe/manual reliability source or representation
  target covering `1->2` / `1->4` recall while still suppressing `0/2/4->1`
  false positives.

## Route Audit 2026-07-06 - Literature After Signal-Gap Crosswalk

- Sources checked:
  SNSCL / Fine-Grained Classification With Noisy Labels
  (<https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_Fine-Grained_Classification_With_Noisy_Labels_CVPR_2023_paper.pdf>),
  DULL / Disentangling and Unlearning for Long-tailed Label-noise
  (<https://arxiv.org/html/2503.11414v1>), class-independent margin in dual
  representation space (<https://arxiv.org/html/2501.11053v1>), and Gradient
  Focal Transformer / GFT (<https://arxiv.org/pdf/2504.09852>).
- Local fit:
  SNSCL-style stochastic contrastive queues require reliable anchor/queue
  updates; the local memory-queue and SNSCL-lite smokes were sparse or harmful
  on current embeddings. DULL-style partial unlearning needs an identified
  wrong-feature region; local complementary patch suppression, bbox/background
  suppression, object-erasure, and partial-AUC losses widened class-1 false
  positives without rescuing `1->2` / `1->4`. Class-independent margin maps to
  the already-rejected OVA/CIR-lite and class-independent head family. GFT's
  gradient-attention/progressive patch-selection idea is only plausible with a
  trustworthy discriminative patch target; local gradient/stat, token-prune,
  part-token, local-mixer, and self-boosting CAM attempts show current patch
  selection still heats surface/border ambiguity and lacks recall-safe support.
- Decision:
  these papers reinforce the signal-gap result rather than opening an immediate
  smoke. Do not implement another contrastive queue, OVA/class-independent
  margin, partial-unlearning/suppression, or gradient/patch-selection variant on
  the current keeper unless a new fold-safe/manual reliability target first
  covers `1->2` / `1->4` true-class1 recall.

## Review Worklist 2026-07-06 - Class1 Rival2/4 Gap Coverage

- Rationale:
  the signal-gap crosswalk showed that `1->2` and `1->4` true-class1 recall
  gaps had almost no train-side review coverage. I built a train-only review
  worklist from the current softboost train predictions to cover low-margin
  class1-vs-rival2/rival4 cases plus matching `2/4->1` suppressors and anchors.
- Artifact:
  `runs\manual_softboost_train_class1_rival24_gap_review_20260706`. It writes
  `class1_rival24_gap_review_candidates.csv`,
  `class1_rival24_gap_review_report.html`, `summary.json`, and
  `readiness_audit\readiness_summary.json`. It declares
  `raw_dataset_touched=false`, `test_split_used=false`, and
  `trainable_manifest_written=false`.
- Coverage:
  selected `155` train rows: `65` recall protectors, `42` false-positive
  suppressors, and `48` anchors. Risk transitions are `1->4=33`, `1->2=32`,
  `2->1=32`, `4->1=10`, `2~1_anchor=24`, and `4~1_anchor=24`.
  The rendered HTML covers all `155` rows with `missing_images_rendered=0`.
- Readiness:
  training remains blocked. `manual_label_status_filled=0/155`, actionable
  recall and false-positive counts are both `0` because all manual fields are
  blank, and readiness found one blocking mixed-actionable cluster
  (`Image_4098`: one recall protector plus one FP suppressor).
- Decision:
  keep this artifact review-only. It closes the worklist coverage gap for
  `1->2` / `1->4`, but it must not be used for relabeling, sample weights,
  soft targets, targeted margins, or a smoke until explicit manual fields are
  filled, the mixed-side source cluster is resolved, and readiness passes.

## Diagnostic 2026-07-06 - Softboost Validation Rival2/4 Forensics + XAI

- Rationale:
  after adding train-only review coverage for the sparse `1->2` / `1->4`
  recall gap, I audited the current softboost validation behavior without
  training or touching test. The goal was to check whether `yolo_f` wide
  context/background still offers an actionable signal, or whether the remaining
  errors stay concentrated on fruit surface/boundary cues.
- Artifacts:
  `runs\diagnostic_softboost_val_rival24_forensics_20260706` and
  `runs\xai_softboost_val_rival24_cases16_20260706`. The forensic report uses
  full `yolo_f/val=2606` support with `image-stats-mode=none` after the
  foreground/basic image-stat passes were stopped for runtime. The XAI report
  uses only `16` validation case indices (`4` each for `1->2`, `1->4`,
  `2->1`, `4->1`) and the same patch-verifier softboost parameters as the
  current validation hook. Both artifacts are diagnostic-only:
  `raw_dataset_touched=false`, `test_split_used=false`, and no trainable
  manifest is written.
- Full-validation forensics:
  current softboost remains validation macro/class-1 `0.8887/0.7030`, class-1
  P/R `0.6480/0.7682`, with `TP=116`, `FP=63`, `FN=35`. Pair-subset
  performance is not the main weakness by itself: `1-2` pair subset is
  `0.9065` (`1->2=11`, `2->1=9`, `outside=45`) and `1-4` pair subset is
  `0.9238` (`1->4=5`, `4->1=9`, `outside=47`). The bottleneck remains
  class-1 one-vs-rest boundary control, not a single isolated pair classifier.
- XAI:
  the selected rival2/4 errors are mostly foreground/object driven. Mean
  heatmap foreground mass is high (`attention=0.9266`, `grad_rollout=0.9200`,
  `gradcam=0.8910`, `rollout=0.8896`), while background perturbation is tiny
  (`background_blur` drop `0.0009`, `background_gray` drop `0.0011`). Object
  desaturation is far stronger (`0.1043` original-prediction drop). Border
  sensitivity remains visible (`gradcam_border_mass=0.2847`, rollout border
  `0.2336`; `gradcam_border_attention=12/16`). By transition, `1->2` and
  `2->1` have very high Grad-CAM foreground mass but large border mass
  (`1->2` `0.9763/0.3337`, `2->1` `0.9327/0.4050`), while `1->4` has more
  background/rollout flags but background blur/gray still does not explain the
  decision (`-0.0060/-0.0035` mean drops).
- Visual pass:
  the contact sheet
  `runs\xai_softboost_val_rival24_cases16_20260706\rival24_xai_contact_sheet.png`
  shows pale/yellow/green fruit, edge glare, spots/decay, stem/crop borders, and
  surface marks overlapping across true class 1, class 2, and class 4. This
  supports the metric result: the hard errors are surface/border/label-boundary
  ambiguity, not a broad-background shortcut that a new context gate would
  likely solve.
- Decision:
  do not open a smoke from validation rival2/4 cases, background/context
  heuristics, or another patch-threshold variant. This diagnostic strengthens
  the current gate: a valid next TRKH-native method must add a new
  surface/boundary representation target or a reviewed/fold-safe reliability
  source that protects true class 1 while suppressing `0/2/4->1` false
  positives.

## Preservation 2026-07-06 - Current TRKH Artifact Inventory

- Artifact:
  `runs\artifact_inventory_current_trkh_20260706`.
- Result:
  verified `missing_count=0` for the current best no-pretrain checkpoint,
  current runtime softboost full-validation metrics/predictions, final
  no-pretrain test metrics, patch-verifier export parameters, both valid data
  YAMLs, the new rival2/4 review/forensics/XAI artifacts, the journal, TODO,
  and TRKH skill. `D:` free space is `79.33 GB`.
- Decision:
  no deletion was made in this step. The new rival2/4 forensic and XAI outputs
  are current evidence, not obsolete smoke/probe material. Continue cleanup only
  for documented rejected case directories after preserving summaries/metrics.

## Tooling 2026-07-06 - Reusable XAI Transition Summary

- Rationale:
  the rival2/4 XAI audit originally needed an inline script to group cases by
  `target->prediction` transition and build a contact sheet. I replaced that
  ad hoc step with a reusable diagnostic tool so future XAI reviews can be
  summarized consistently before any gate decision.
- Implementation:
  added `trkh.tools.summarize_xai_transitions` and
  `tests\test_summarize_xai_transitions.py`. The tool reads an existing
  `xai_audit_summary.json`, writes `transition_xai_summary.csv`,
  `transition_xai_summary.json`, `README.md`, and an optional contact sheet.
  It only reads existing XAI artifacts and declares
  `raw_dataset_touched=false`, `trainable_manifest_written=false`, plus a
  diagnostic-only guardrail.
- Verification:
  `D:\DataAI\.venv\Scripts\python.exe -m py_compile
  trkh\tools\summarize_xai_transitions.py` passed, and
  `D:\DataAI\.venv\Scripts\python.exe -m pytest
  tests\test_summarize_xai_transitions.py -q` passed (`1 passed`).
- Applied artifact:
  reran the tool on
  `runs\xai_softboost_val_rival24_cases16_20260706\xai_audit_summary.json`,
  writing
  `runs\xai_softboost_val_rival24_cases16_20260706\transition_summary_tool`.
  It summarized `16` validation cases across `4` transitions, rendered `48`
  crop/Grad-CAM/rollout images, and reported `missing_images=0`.
- Decision:
  use this tool for future XAI transition audits instead of inline scripts.
  Its outputs remain diagnostic-only and must not be used as a trainable
  manifest, threshold selector, router target, or validation-tuned policy.

## Tooling 2026-07-06 - Review Worklist Readiness Matrix

- Rationale:
  the previous readiness matrix covered five active manual review queues but
  did not include the new class1 rival2/4 gap worklist. I added a reusable
  matrix builder so future review queues can be audited from a JSON config
  without hard-coded shell scripts.
- Implementation:
  added `trkh.tools.build_review_worklist_readiness_matrix` and
  `tests\test_build_review_worklist_readiness_matrix.py`. The tool calls the
  existing `audit_review_worklist_readiness` for each configured queue, writes
  each per-queue readiness summary plus a matrix `summary.json`, and reads
  config JSON with `utf-8-sig` so PowerShell-generated configs work.
- Verification:
  `D:\DataAI\.venv\Scripts\python.exe -m py_compile
  trkh\tools\build_review_worklist_readiness_matrix.py
  trkh\tools\audit_review_worklist_readiness.py` passed. Focused pytest
  `D:\DataAI\.venv\Scripts\python.exe -m pytest
  tests\test_build_review_worklist_readiness_matrix.py
  tests\test_audit_review_worklist_readiness.py -q` passed (`6 passed`), and
  the single matrix test also passed after the BOM read fix.
- Applied artifact:
  `runs\review_worklist_readiness_matrix_20260706`. It audits six active
  train-only review queues: boundary softboost train, triage batch01, balanced
  batch02, strict batch02, EffV2 priority, and class1 rival2/4 gap review.
  Result is `ready_count=0/6`, `blocked_count=6`, with
  `raw_dataset_touched=false`, `test_split_used=false`, and
  `trainable_manifest_written=false`.
- Blocking evidence:
  every queue still has empty manual fields: boundary `0/260`, triage `0/58`,
  balanced batch02 `0/165`, strict batch02 `0/125`, EffV2 priority `0/95`,
  and rival2/4 `0/155`. The rival2/4 queue also has one blocking mixed
  actionable-side source cluster; strict batch02 has two mixed-side warning
  clusters but none blocking because anchors are neutral.
- Decision:
  no current review queue can be passed to non-dry-run reviewed-manifest
  tooling. This confirms there is still no permission to create relabels,
  sample weights, soft targets, targeted margins, or a smoke from review queues
  until explicit manual fields are filled and readiness passes.

## Tooling 2026-07-06 - TRKH Smoke Gate Audit

- Rationale:
  after the correction-budget, external-support, signal-gap, review-readiness,
  and XAI diagnostics all pointed to blocked automatic routes, I added a single
  gate auditor so future loops do not reinterpret diagnostic artifacts as train
  permission.
- Implementation:
  added `trkh.tools.audit_trkh_smoke_gate` and
  `tests\test_audit_trkh_smoke_gate.py`. The tool reads existing diagnostic
  summaries, computes a gate decision for the next class-1 milestone, and writes
  only `summary.json` plus `README.md`. It records
  `raw_dataset_touched=false`, `test_split_used=false`, and
  `trainable_manifest_written=false`.
- Verification:
  `D:\DataAI\.venv\Scripts\python.exe -m py_compile
  trkh\tools\audit_trkh_smoke_gate.py` passed, and focused pytest
  `D:\DataAI\.venv\Scripts\python.exe -m pytest
  tests\test_audit_trkh_smoke_gate.py -q` passed (`2 passed`).
- Applied artifact:
  `runs\audit_trkh_smoke_gate_current_20260706`. Current class-1 remains
  `TP=116`, `FP=63`, `FN=35`, F1 `0.7030`; the next milestone `0.75` needs
  at least `13` effective corrections. Review readiness is `0/6` with
  `0/858` manual fields filled, and the rival2/4 queue still has one blocking
  mixed actionable-side cluster. External consensus upper-bound class-1 is
  `0.7237`, below `0.75`, with insufficient true-class1 recall coverage for
  `1->2` (`0/11` consensus-correct) and `1->4` (`1/5`). XAI aggregation remains
  surface/boundary-dominant: background perturbation mean `0.0048` versus
  object-desaturation mean `0.1227`, foreground mass `0.9068`.
- Decision:
  `smoke_gate_ready=false`. Do not launch automatic TRKH smoke/full train from
  current external consensus, current review queues, patch-threshold variants,
  or validation XAI case indices. The next valid smoke still requires either a
  new surface/boundary representation target or a fold-safe/manual reliability
  source with true-class1 recall protection and conservative `0/2/4->1`
  false-positive control.

## Cleanup 2026-07-06 - Rejected Eval-Smoke Leftovers

- Rationale:
  after the smoke gate remained closed, I cleaned remaining rejected
  `eval_smoke_*` directories whose conclusions were already recorded in the
  journal and whose metrics were not current gate artifacts.
- Action:
  compacted `14` rejected/superseded `eval_smoke_*` directories into
  `runs\evidence_eval_smoke_rejected_leftovers_20260706`, copying only
  `metrics.json` and `metrics_detailed.json` for each run, then deleted the
  original eval directories. The cleanup manifest is
  `runs\cleanup_manifest_20260706_eval_smoke_rejected_leftovers.json`.
- Evidence:
  measured deleted payload was `39.238 MB` across `14` directories; compact
  evidence is `28` metric files. The deleted runs include rejected V8
  late-class-attention, SSL-DINO keeper-init, light RandAugment, top5
  vote-disagreement, cumulative ordinal, local-neighbor/patch-summary routes,
  and rejected timm fold00 eval-smokes.
- Verification:
  no `runs\eval_smoke_*` directories remain. Protected artifacts verified
  present after cleanup include the best no-pretrain checkpoint, current
  softboost full-val metrics, final no-pretrain test metrics, fixed
  patch-evidence final summary, rival2/4 review/forensics/XAI, current smoke
  gate audit, review readiness matrix, and `yolof_oof_folds_train5_20260704`.
  No raw dataset files, test tuning, or trainable manifest were touched.
- Decision:
  treat these eval-smoke runs as compacted historical evidence. Do not recreate
  them unless a future method explicitly needs their full prediction/plot
  payload; use the evidence root plus journal metrics for do-not-repeat context.

## Cleanup 2026-07-06 - Legacy MobileNetV3 class_f OOF Fold Train Dirs

- Rationale:
  the old `class_f` MobileNetV3 5-fold OOF artifact from 2026-06-27 is
  historical benchmark evidence only. It reached OOF train macro/class-1
  `0.8599/0.5403`, far below the current TRKH-native target, and the current
  fold-safe materialization to keep live is `runs\yolof_oof_folds_train5_20260704`.
- Action:
  compacted and deleted the five per-fold training/checkpoint directories
  `runs\oof_mobilenetv3_fold00_2e120b_20260627` through
  `runs\oof_mobilenetv3_fold_04_2e120b_20260627`. Metrics, history, and logs
  were copied to
  `runs\evidence_mobilenetv3_classf_oof_fold_train_dirs_20260706`; aggregate
  OOF predictions and cleanlab outputs were kept in place. Cleanup manifest:
  `runs\cleanup_manifest_20260706_mobilenetv3_classf_oof_fold_train_dirs.json`.
- Evidence:
  deleted payload measured `83.958 MB`, mostly checkpoint files
  (`16.238 MB` per fold checkpoint). The compact evidence root contains `16`
  small files across `5` fold evidence directories.
- Verification:
  aggregate `runs\oof_mobilenetv3_5fold_2e120b_20260627\summary.json` and
  `predictions_train_oof.csv` still exist, as does
  `runs\cleanlab_oof_mobilenetv3_5fold_2e120b_20260627\cleanlab_label_issue_manifest.csv`.
  Current protected artifacts also remain: `yolof_oof_folds_train5_20260704`,
  best no-pretrain checkpoint, current softboost full-val metrics, final
  no-pretrain test metrics, and the current smoke-gate audit. No raw dataset,
  test tuning, or trainable manifest was touched.
- Decision:
  do not look for the old MobileNetV3 per-fold checkpoint dirs as live evidence.
  Use the compact evidence root plus the aggregate OOF summary/CSV if historical
  comparison is needed.

## Audit 2026-07-06 - Current Artifact Retention Snapshot

- Rationale:
  after compacting rejected eval-smokes and old MobileNetV3 OOF fold train dirs,
  I created a retention audit so future cleanup passes do not re-scan the same
  artifact set or delete current gate evidence by mistake.
- Artifact:
  `runs\artifact_retention_audit_current_trkh_20260706`.
- Result:
  scanned `500` run directories and wrote an audit-only `summary.json` plus
  README. No files were deleted, no raw datasets were touched, no test tuning
  was done, and no trainable manifest was written. `D:` free space was
  `79.443 GB`.
- Key decisions:
  the largest live directory remains
  `runs\yolof_oof_folds_train5_20260704` at `2547.804 MB`, but it is retained
  because it is the current grouped `yolo_f` OOF materialization needed for
  fold-safe diagnostics. The best TRKH anchor
  `probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701`
  is also retained (`347.008 MB`) because it contains the current no-pretrain
  checkpoint and validation evidence.
- Verification:
  protected checks are all true for current `yolo_f` OOF summary, best
  no-pretrain checkpoint, current softboost full-val metrics, frozen final
  no-pretrain test metrics, smoke-gate audit, and review readiness matrix.
  The audit also confirms no `runs\eval_smoke_*` directories and no
  `runs\oof_mobilenetv3_fold*` per-fold checkpoint directories remain.
- Future cleanup queue:
  the audit flags `34` historical candidates (`484.105 MB`) as
  `FUTURE_CLEANUP_REVIEW`, including old embedding-retrieval, groupclean
  quality/boundary-review, selector, and posthoc sweep artifacts. These are not
  deletion permission; compact them only in a separate pass after preserving
  summaries and checking journal/TODO coverage.

## Tooling 2026-07-06 - Manual Review Minimum Fill Plan

- Rationale:
  the smoke gate is closed because current automatic signals do not cover the
  `1->2` and `1->4` true-class1 recall gaps, and all review queues still have
  empty manual fields. I added a diagnostic-only fill planner to turn that
  blocker into an explicit manual/fold-safe reliability work order without
  auto-labeling or generating trainable manifests.
- Implementation:
  added `trkh.tools.build_review_minimum_fill_plan` and
  `tests\test_build_review_minimum_fill_plan.py`. The tool reads the review
  readiness matrix plus the smoke-gate summary, prioritizes rows to fill from
  the existing train-only review queues, and writes only `summary.json`,
  `README.md`, and `priority_fill_plan.csv`.
- Verification:
  `D:\DataAI\.venv\Scripts\python.exe -m py_compile
  trkh\tools\build_review_minimum_fill_plan.py` passed. Focused pytest with
  readiness/gate tooling passed:
  `tests\test_build_review_minimum_fill_plan.py
  tests\test_build_review_worklist_readiness_matrix.py
  tests\test_audit_review_worklist_readiness.py
  tests\test_audit_trkh_smoke_gate.py -q` -> `9 passed`.
- Applied artifact:
  `runs\review_minimum_fill_plan_current_20260706`. It selected `360`
  priority fill-plan rows across the six active train-only queues, representing
  `184` unique sample indices. The artifact keeps
  `raw_dataset_touched=false`, `test_split_used=false`,
  `trainable_manifest_written=false`, `auto_labeling_performed=false`, and
  `smoke_permission=false`.
- Key finding:
  older broad queues have too little true-class1 recall-side coverage for the
  current gate (`boundary_softboost_train_current` and triage batch01 each have
  only `9` recall rows available versus a min of `20`). The rival2/4 worklist is
  the only current queue with direct blank coverage for the critical transitions
  (`1->2=32`, `1->4=33`), but it still has `manual_label_status=0/155` and one
  blocking mixed actionable cluster `Image_4098`.
- Decision:
  this fill plan does not open smoke permission. It defines the next manual or
  fold-safe reliability step: fill explicit review fields, resolve
  `Image_4098`, rerun the readiness matrix, then rerun the smoke gate before
  any non-dry-run manifest or train smoke.

## Review Packet 2026-07-06 - Current Manual Fill Plan HTML

- Rationale:
  the minimum fill plan CSV is useful for audit, but manual/fold-safe review
  needs an image-visible packet ordered by the same global priority. I rendered
  the plan into HTML without editing any source review fields.
- Artifact:
  `runs\review_minimum_fill_plan_current_20260706\priority_fill_plan_review.html`
  plus
  `runs\review_minimum_fill_plan_current_20260706\priority_fill_plan_review.summary.json`.
- Result:
  rendered `360/360` priority rows with `missing_images_rendered=0` and
  `manual_label_status_filled=0`. Group counts are `168` recall-required rows,
  `190` FP-required rows, and `2` mixed-cluster-resolution rows. The first rows
  are now the blocking `Image_4098` mixed actionable cluster followed by the
  critical `1->2`/`1->4` recall-gap rows.
- Verification:
  `D:\DataAI\.venv\Scripts\python.exe -m py_compile
  trkh\tools\render_review_worklist_html.py
  trkh\tools\build_review_minimum_fill_plan.py` passed. Focused pytest
  `tests\test_render_review_worklist_html.py
  tests\test_build_review_minimum_fill_plan.py -q` passed (`3 passed`).
- Decision:
  keep this as a review packet only. Do not write labels into this HTML, do not
  treat the fill-plan CSV as labels, and do not train from it. Any future
  manifest still requires explicit manual fields in the source review queues,
  a passing readiness matrix, and a rerun smoke gate.

## Cleanup 2026-07-06 - Legacy Embedding Retrieval Feature Caches

- Rationale:
  the retention audit flagged old `embedding_retrieval_*_20260627` directories
  as `FUTURE_CLEANUP_REVIEW`. These artifacts are historical pretrained feature
  diagnostics; their conclusions are already recorded in TODO/journal, while
  their `.npz` feature caches are not current TRKH gate evidence.
- Action:
  compacted `10` legacy embedding-retrieval directories into
  `runs\evidence_embedding_retrieval_legacy_20260706`, preserving all
  non-`.npz` summaries, metrics, predictions, and per-directory compact
  manifests, then deleted the original directories. Cleanup manifest:
  `runs\cleanup_manifest_20260706_legacy_embedding_retrieval_compacted.json`.
- Evidence:
  source payload was `223.476 MB`; preserved evidence is `4.486 MB` across
  `46` files; measured deleted feature-cache payload is `218.990 MB`. Rounded
  observed free-space delta was `0 MB`, likely below the filesystem/reporting
  granularity at this scale.
- Verification:
  all `10` original `embedding_retrieval_*_20260627` directories are absent and
  each evidence subdir has `summary.json`. Protected checks are true for the
  best no-pretrain checkpoint, current softboost full-val metrics, final
  no-pretrain test metrics, current smoke-gate audit, review readiness matrix,
  current fill-plan summary/HTML, `yolof_oof_folds_train5_20260704`, and both
  valid dataset YAMLs. No raw dataset, training launch, trainable manifest, or
  test tuning was touched.
- Decision:
  do not look for those old embedding-retrieval directories as live evidence.
  Use the compact evidence root plus the existing TODO/journal conclusions for
  historical comparison. Regenerate feature `.npz` caches only if a future,
  explicitly justified diagnostic needs them.

## Cleanup 2026-07-06 - Legacy Quality/Boundary Review Images

- Rationale:
  the retention audit also flagged old quality-group and boundary-review
  directories from 2026-06-25/26 as `FUTURE_CLEANUP_REVIEW`. They are historical
  review artifacts; the durable evidence is in CSV/JSON/HTML/README files, while
  `review_images` is image-heavy and no longer current gate evidence.
- Action:
  compacted `14` legacy quality/boundary review directories into
  `runs\evidence_legacy_quality_boundary_reviews_20260706`, preserving all
  CSV/JSON/HTML/README evidence plus a per-directory compact manifest, then
  deleted the original directories. Cleanup manifest:
  `runs\cleanup_manifest_20260706_legacy_quality_boundary_reviews_compacted.json`.
- Evidence:
  source payload was `170.612 MB`; preserved evidence is `20.320 MB` across
  `57` files; measured deleted rendered-image payload is `150.292 MB`. Rounded
  observed free-space delta was `0 MB`.
- Verification:
  all `14` original legacy quality/boundary review dirs are absent, and each
  evidence subdir has `summary.json`. Protected checks are true for the best
  no-pretrain checkpoint, current softboost full-val metrics, final no-pretrain
  test metrics, smoke-gate audit, review readiness matrix, fill-plan summary
  and HTML, compact embedding-retrieval evidence, `yolof_oof_folds_train5`,
  and both valid dataset YAMLs. No raw dataset, training launch, trainable
  manifest, or test tuning was touched.
- Decision:
  do not look for these old review-image directories as live evidence. Use the
  compact evidence root plus TODO/journal conclusions for historical comparison.
  Regenerate rendered review images only if a future, explicitly justified audit
  needs visual side-by-side evidence.

## Cleanup 2026-07-06 - Legacy Selector/Posthoc Sweep Artifacts

- Rationale:
  the retention audit also flagged old selector, class-1 specialist, and
  posthoc Q34 sweep artifacts. Their durable evidence is the CSV/JSON metrics,
  candidate tables, and predictions; the binary `selector.pkl`/`specialist.pkl`
  files and PR-curve plots are historical payloads and are not current TRKH
  gate evidence.
- Action:
  compacted `10` legacy selector/posthoc directories into
  `runs\evidence_legacy_selectors_posthoc_20260706`, preserving CSV/JSON/MD/TXT/HTML
  evidence plus per-directory compact manifests, then deleted the originals.
  Cleanup manifest:
  `runs\cleanup_manifest_20260706_legacy_selectors_posthoc_compacted.json`.
- Evidence:
  source payload was `90.017 MB`; preserved evidence is `34.802 MB` across
  `79` files; measured deleted binary/plot payload is `55.214 MB`. Original
  directories removed include old group-clean selector inputs, train-to-val
  selector evidence, top-5/TRKH selector artifacts, focus-class1 specialists,
  and Q34 max-det/score-NMS/best-recall posthoc sweeps.
- Verification:
  all `10` original selector/posthoc directories are absent, evidence file count
  is `79`, `remaining_original_dirs=[]`, and protected checks are true for the
  best no-pretrain checkpoint, current softboost full-val metrics, final
  no-pretrain test metrics, smoke-gate audit, review readiness matrix, fill-plan
  summary/HTML, compact embedding and quality-review evidence, current
  `yolof_oof_folds_train5`, and both valid dataset YAMLs. No raw dataset,
  training launch, trainable manifest, or test tuning was touched.
- Decision:
  do not look for these old selector/posthoc directories as live evidence. Use
  the compact evidence root plus TODO/journal conclusions for historical
  comparison. Historical posthoc test sweeps remain audit history only and must
  not be used as current gate evidence or threshold-tuning permission.

## Tooling 2026-07-06 - Current Signal-Gap Crosswalk Rerun

- Rationale:
  after adding the rival2/4 review worklist and fill-plan packet, the old
  `runs\diagnostic_trkh_signal_gap_readiness_20260706` still reflected an older
  review-source set for transition coverage. The smoke gate was still correctly
  closed, but its `1->2`/`1->4` review-row counts were stale, so future decisions
  could misread the actual blocker.
- Implementation:
  added `trkh.tools.build_trkh_signal_gap_readiness` plus
  `tests\test_build_trkh_signal_gap_readiness.py`. The tool reads existing
  softboost validation predictions, transition budget, external support,
  train-OOF support, review readiness, OOF-neighbor policy, and named review
  CSVs, then writes only `summary.json`, `transition_gap_rows.csv`, and
  `README.md`. It does not touch raw data, test split, labels, trainable
  manifests, sample weights, soft targets, targeted margins, or checkpoints.
- Verification:
  `D:\DataAI\.venv\Scripts\python.exe -m py_compile
  trkh\tools\build_trkh_signal_gap_readiness.py
  tests\test_build_trkh_signal_gap_readiness.py` passed, and focused pytest
  `tests\test_build_trkh_signal_gap_readiness.py -q` passed (`1 passed`).
- Applied artifact:
  `runs\diagnostic_trkh_signal_gap_readiness_current_20260706`. It now counts
  the rival2/4 worklist in transition coverage: `1->2` has `32` review rows and
  `1->4` has `36`, versus the old stale `0/2`. Both transitions still have
  `review_manual_filled_total=0`.
- Gate rerun:
  `runs\audit_trkh_smoke_gate_current_rerun_20260706` uses the updated
  signal-gap summary. It remains `smoke_gate_ready=false`: current class-1 F1 is
  `0.7030`, next milestone `0.75` still needs at least `13` effective
  corrections, review readiness is `0/6` with `0/858` manual fields filled,
  rival2/4 still has one blocking mixed actionable cluster, external consensus
  class-1 is only `0.7237`, and true-class1 recall coverage is still
  insufficient.
- Decision:
  the corrected crosswalk does not open training. It clarifies that the active
  rival2/4 worklist has enough blank review-row coverage to be the right manual
  target, but there is still no fold-safe/manual reliability signal until
  explicit fields are filled, `Image_4098` is resolved, readiness passes, and
  the smoke gate is rerun.

## Command Packet 2026-07-06 - Current Best Full Train/Test/XAI

- User-facing command file:
  `docs\TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt`.
- Contents:
  PowerShell commands for the current best no-pretrain TRKH-native full-train
  override on `yolo_f`, followed by raw final-test evaluation, verifier-aware
  patch-linear softboost final-test evaluation, verifier-aware XAI audit, and
  XAI transition summary export.
- Recipe:
  `teacher_focus_binary=0.015`, pairwise margin routing, bbox spatial fusion,
  boundary-band bbox foreground dropout, metric learning, no pretrained weights,
  explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, and `-SkipFinalTest`
  during training so test remains a separate explicit final-audit command.
- Decision:
  this file is a manual override packet for the VS Code terminal. It does not
  open the automatic smoke gate, and I did not run train/test/XAI from it in this
  loop.

## Tooling 2026-07-06 - Artifact Retention Audit Rerun

- Rationale:
  the first retention audit
  `runs\artifact_retention_audit_current_trkh_20260706` was created before the
  later embedding-retrieval, quality/boundary-review, and selector/posthoc
  compaction passes. It still pointed at the older smoke gate and listed several
  original directories that have since been compacted and deleted, so future
  cleanup decisions needed a fresh, reusable audit.
- Implementation:
  added `trkh.tools.audit_trkh_artifact_retention` plus
  `tests\test_audit_trkh_artifact_retention.py`. The tool scans immediate
  children of `runs`, computes size/file-count inventory, verifies protected
  artifacts, checks deleted-prefix absence, and verifies that names listed in
  compaction manifests are no longer present. It writes only `summary.json`,
  `largest_directories.csv`, `future_cleanup_review_candidates.csv`, and
  `README.md`; it deletes nothing and does not touch raw data, labels, test
  tuning, trainable manifests, or checkpoints.
- Verification:
  `D:\DataAI\.venv\Scripts\python.exe -m py_compile
  trkh\tools\audit_trkh_artifact_retention.py` passed, and focused pytest
  `tests\test_audit_trkh_artifact_retention.py -q` passed (`2 passed`).
- Applied artifact:
  `runs\artifact_retention_audit_current_trkh_rerun_20260706`. It scanned
  `472` run directories, `retention_audit_passed=true`, `blockers=[]`, and
  available free space was `79.861 GB`.
- Key checks:
  protected artifacts are all present, `eval_smoke_*` remaining count is `0`,
  `oof_mobilenetv3_fold*` remaining count is `0`, and all `34` compacted
  original directories from the 2026-07-06 compaction manifests are absent.
  The current largest live artifact remains
  `runs\yolof_oof_folds_train5_20260704` (`2547.804 MB`) and should be kept for
  fold-safe diagnostics.
- Cleanup queue:
  current future cleanup candidates are only `7` small historical items
  totaling `31.602 MB`, led by
  `embedding_retrieval_dinov2_small_yolof_direct_full_20260704` and old small
  selector/posthoc/quality artifacts. Treat them as review-only; compact or
  delete only in a separate pass after preserving summaries and checking
  journal/TODO coverage.
- Decision:
  use the rerun retention audit as the current inventory reference. The old
  `artifact_retention_audit_current_trkh_20260706` is historical/stale after
  subsequent cleanup.
- Later refresh:
  after the rejected crop-bbox token-prior probe cleanup, reran the inventory as
  `runs\artifact_retention_audit_after_cropbbox_cleanup_20260706`. It scanned
  `480` run directories, `retention_audit_passed=true`, `blockers=[]`,
  compacted originals remaining `0`, and free space was `79.838 GB`. This is
  the current retention reference after the cropbbox cleanup; the older
  `artifact_retention_audit_current_trkh_rerun_20260706` is now historical.

## Tooling 2026-07-06 - Patch Score Point-in-Box Audit

- Rationale:
  after reviewing recent token/patch-score literature, the next useful
  diagnostic was to measure whether the TRKH global/CLS-like query actually
  ranks object/bbox patches above background patches. This directly tests the
  user's question about whether wider `yolo_f` context plus ViT self-attention
  is helping the model separate object from background.
- Implementation:
  added `trkh.tools.audit_patch_score_pib` plus
  `tests\test_audit_patch_score_pib.py`. The tool runs a checkpoint on one
  split, computes cosine patch scores from `pooled`, `cls`, or
  `cls_register_mean` query features, measures top-k Point-in-Box and
  foreground/background score gaps from `patch_bbox_prior`, then writes
  `summary.json`, per-row CSV, group summary CSV, risk-transition CSV, and
  README. It refuses test unless `--allow-test` is explicit and writes no raw
  data, labels, trainable manifest, sample weights, soft targets, targeted
  margins, or checkpoints.
- Verification:
  `D:\DataAI\.venv\Scripts\python.exe -m py_compile
  trkh\tools\audit_patch_score_pib.py tests\test_audit_patch_score_pib.py`
  passed, and focused pytest `tests\test_audit_patch_score_pib.py -q` passed
  (`3 passed`).
- Applied artifacts:
  `runs\diagnostic_patch_score_pib_full_val_20260706` (`bbox`+`pooled`),
  `runs\diagnostic_patch_score_pib_clsreg_full_val_20260706`
  (`bbox`+`cls_register_mean`), and
  `runs\diagnostic_patch_score_pib_cropbbox_clsreg_full_val_20260706`
  (`crop_bbox`+`cls_register_mean`), all full `yolo_f/val=2606`, no test.
- Key result:
  using raw `bbox` as the prior makes PiB look artificially poor
  (`all_top1_pib` about `0.34-0.36`), but using transformed `crop_bbox` shows
  the query is already object-focused: all top-1 PiB `0.9279`, incorrect top-1
  PiB `0.9286`, target-class1 top-1 PiB `0.9338`, top-5 foreground fraction
  `0.9384`, and `background_shortcut_suspected=false`.
- Decision:
  do not treat the current failure as a broad background shortcut once bbox is
  measured in the transformed crop frame. The remaining bottleneck is still
  surface/boundary separability. The low raw-`bbox` PiB is useful only as a
  compatibility warning: be explicit about `bbox` versus `crop_bbox` in future
  audits and launch commands.

## Probe 2026-07-06 - Crop-BBox Token Prior Source Rejected

- Rationale:
  because the PiB audit showed transformed `crop_bbox` is the correct frame for
  token-level foreground measurement, I ran a short probe to test whether
  switching `-BboxTokenPriorSource crop_bbox` in the current v8 recipe improves
  the keeper.
- Run:
  `runs\probe_v8_yolof_cropbbox_tokenprior_120b_1e_20260706`, resumed from
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt`,
  1 epoch, `MaxTrainBatches=120`, full `yolo_f/val=2606`, `SkipFinalTest`,
  trace enabled, no raw-data edit.
- Metrics:
  validation macro/class1 F1 dropped to `0.8822/0.6763` with class1 P/R
  `0.6000/0.7748`, below the keeper `0.8847/0.6860` and below current
  softboost full-val `0.8887/0.7030`. Confusions worsened slightly versus
  keeper: `0->1` `49 -> 51`, `1->0` `17 -> 18`, `2->1` `10 -> 11`,
  `4->1` stayed `12`, and `1->2` stayed `11`.
- XAI:
  `runs\xai_probe_v8_yolof_cropbbox_tokenprior_val_cases12_20260706` plus
  transition summary
  `runs\xai_probe_v8_yolof_cropbbox_tokenprior_val_cases12_20260706\transition_summary`
  used validation only. Foreground mass stayed high
  (`attention=0.8766`, `grad_rollout=0.9372`, `rollout=0.8996`), background
  blur/gray drops were small (`0.0126/0.0118`), and object desaturation was much
  larger (`0.1698`). Flags remained surface/border-heavy
  (`object_color_sensitive=9/12`, Grad-CAM border `7/12`).
- Decision:
  reject simple `crop_bbox` token-prior switching on the current keeper. Do not
  repeat by only changing LR, epochs, batch count, or query token. Future work
  still needs a genuinely new surface/boundary representation target or a
  fold-safe/manual reliability signal with true-class1 recall protection.
- Cleanup:
  deleted the rejected probe checkpoint directory and the 12 rendered XAI
  `case_*` directories after preserving metrics/configs, XAI summaries, case
  CSV, audit markdown, and transition summary/contact sheet. Manifest:
  `runs\cleanup_manifest_20260706_cropbbox_tokenprior_rejected_aux.json`;
  measured payload removed `175.766 MB`. The probe root still retains
  `best_metrics.json`, `history.csv`, `launcher_args.json`, `resolved_config.json`,
  and `summary.json`; the XAI root still retains `xai_audit_summary.json`,
  `xai_metrics.json`, `xai_cases.csv`, `xai_audit.md`, and
  `transition_summary\transition_xai_summary.json`.

## Launcher 2026-07-10 - Full-Train Command Packet Revalidated

- Context:
  a manual full-train run named
  `runs\full_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_30e_20260706`
  was present from 2026-07-08, but it failed before training started. The
  transcript showed a PowerShell 5 `NativeCommandError` after the first Python
  INFO log because the launcher piped native stderr through `2>&1 |
  ForEach-Object`. No metrics or checkpoints were produced, so this is not a
  model failure.
- Fix:
  patched `scripts\run_trkh_5class_attention_views_v8.ps1` so it calls
  `& $Python -m trkh.training.train @TrainArgs` directly under a temporary
  `ErrorActionPreference="Continue"` guard. This preserves the native process
  exit code without letting benign Python stderr logs stop the PowerShell
  pipeline.
- Preflight:
  reran the full current-best command as
  `preflight_full_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_20260710`
  with the full teacher CSV argument. It passed with explicit `yolo_f`,
  `skip_final_test=true`, `bbox_token_prior_source=bbox`, teacher-focus-binary
  `0.015`, bbox spatial fusion, pairwise routing, metric learning, and
  boundary-band bbox foreground dropout resolved as intended.
- Launcher validation smoke:
  ran `smoke_launcher_stderrfix_v8_yolof_1b_20260710` with one train batch and
  one validation batch only. It exited `0`, used `SkipFinalTest`, and confirmed
  the launcher no longer dies on Python stderr logging. This smoke is not
  model-selection evidence because validation support was intentionally tiny.
- Cleanup:
  compacted the failed full-train run and the launcher-validation smoke into
  `runs\evidence_launcher_stderrfix_validation_20260710`, preserving
  launcher/status/config/history/metrics JSON/CSV/TXT evidence and deleting the
  original payload directories. Manifest:
  `runs\cleanup_manifest_20260710_launcher_stderrfix_validation.json`;
  observed free-space delta `178.082 MB`; raw dataset untouched.
- Retention:
  reran artifact retention as
  `runs\artifact_retention_audit_after_launcher_fix_20260710`. It scanned
  `482` run directories, `retention_audit_passed=true`, `blockers=[]`,
  protected artifacts all present, no `eval_smoke_*` or
  `oof_mobilenetv3_fold*` directories, compacted originals remaining `0`, and
  free space `79.992 GB`.
- Command packet:
  `docs\TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt` now uses run name
  `full_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_30e_20260710`.
  The command remains a manual override/final experiment packet, not automatic
  smoke permission; the current smoke gate remains closed until a genuinely new
  surface/boundary representation or fold-safe/manual reliability signal exists.

## Diagnostic 2026-07-10 - Self/Neighbor Consistency Rejected Before Smoke

- Literature hypothesis:
  Jo-SNC and SNSCL/LNL-FG suggest that self/neighbor consistency can identify
  reliable noisy-label anchors. This was worth checking because the rival2/4
  train queue explicitly covers the missing `1->2`, `1->4`, `2->1`, and
  `4->1` directions. The check remained diagnostic-only: no raw-data edit,
  test use, relabel, sample weight, soft target, targeted margin, or training.
- Tooling and preflight:
  added `trkh.tools.audit_self_neighbor_consistency`, explicit NPZ
  `--feature-key` support, and overlap-safe review deduplication by
  `sample_index`. A signal may pass an internal separation check, but the tool
  always returns `smoke_ready=false` and `training_permission=false`. Added
  reusable train-to-evaluation tool
  `trkh.tools.audit_neighbor_transfer_consistency`; the prototype probe can now
  save aligned embedding caches. Focused compile plus pytest passed
  (`19 passed` in the final combined preflight).
- Train-only result:
  `runs\diagnostic_self_neighbor_consistency_sourceemb_combined_20260710`
  used all `9215` keeper source embeddings, top-30 neighbors, same-source
  exclusion, and two review queues. It aligned `9215/9215` rows with zero label
  mismatch, deduplicated `280` queue rows to `248` unique samples, and found no
  global separation: best direct/clean signal AUC was `0.6317`; class1-neighbor
  label/probability AUC was only `0.4349/0.4091`.
- Queue instability:
  the rival2/4 queue alone looked moderately separable in-sample
  (`best AUC=0.7207`), while strict batch02 inverted (`best AUC=0.4511`). The
  earlier CMSF-smoothed `features` key made rival2/4 look stronger
  (`0.7714`) but is not the raw keeper representation; this discrepancy is why
  the explicit feature-key guard is now mandatory.
- Independent full-validation transfer:
  `runs\diagnostic_keeper_knn_transfer_k15_30_50_cache_20260710` performed one
  reusable train+val forward pass and saved aligned embedding caches. Base
  keeper validation was macro/class1 `0.8847/0.6860`; cosine kNN reached only
  `0.8663/0.6159` at k15, `0.8651/0.6154` at k30, and `0.8619/0.6063` at k50.
  The threshold-free transfer audit
  `runs\diagnostic_neighbor_transfer_consistency_keeper_val_20260710` was even
  more decisive: for validation `1->2/4` recall errors versus `2/4->1` false
  positives, stable class1-neighbor-label AUC was only `0.0582`; the remaining
  stable metrics were `0.0327-0.0426`. The train signal therefore reverses on
  independent validation rather than transferring.
- Decision:
  reject global Jo-SNC/SNSCL selection, current-embedding neighbor queues,
  neighbor sample weights, neighbor targeted margins, and the apparently
  promising rival2/4-only neighbor route. Do not smoke or tune k/thresholds on
  this representation. The next automatic route must change the underlying
  surface/boundary representation or provide genuinely fold-safe/manual
  reliability; current in-sample neighbor geometry cannot satisfy the gate.
- Cleanup:
  compacted nine superseded CMSF-smoothed, k-sensitivity, and queue-only audit
  directories into
  `runs\evidence_self_neighbor_consistency_sensitivity_20260710`, preserving
  each `summary.json` and `README.md`. Manifest
  `runs\cleanup_manifest_20260710_self_neighbor_sensitivity_compacted.json`
  records `27.930 MB` source, `0.079 MB` preserved, and `27.938 MB` observed
  free-space delta. The raw-source combined train audit, reusable train/val
  caches, and independent transfer audit remain live.

## Smoke 2026-07-11 - Detail-Preserving Stem Pooling Rejected

- Literature hypothesis:
  SoftPool (ICCV 2021, `https://arxiv.org/abs/2101.00440`) replaces hard max
  selection with exponentially weighted activation pooling; Detail-Preserving
  Pooling (CVPR 2018,
  `https://openaccess.thecvf.com/content_cvpr_2018/html/Saeedan_Detail-Preserving_Pooling_in_CVPR_2018_paper.html`)
  argues for adaptive downsampling that preserves spatial changes; BlurPool
  (ICML 2019, `https://proceedings.mlr.press/v97/zhang19a.html`) identifies
  aliasing in ordinary downsampling; and Early Convolutions Help Transformers
  See Better (NeurIPS 2021,
  `https://proceedings.neurips.cc/paper/2021/hash/ff1418e8cc993fe8abcfe3ce2003e5c5-Abstract.html`)
  supports a stronger convolutional inductive bias before transformer tokens.
  This was relevant because the current TRKH stem performs three hard
  `Conv3x3+BN+GELU+MaxPool2d` reductions before patch tokenization, which can
  discard subtle mango-surface evidence early.
- Implementation:
  added parameter-free `SoftPool2d` and conservative `MaxSoftPool2d`, with
  model/config/train/evaluate/V8 launcher options
  `stem_pooling_mode={max,soft,max_soft}` and `stem_softpool_blend`. The default
  remains legacy `max`; pooling modules add no state-dict keys, and checkpoint
  evaluation can explicitly override mode/blend. Added
  `tests\test_stem_pooling.py` for numerical range/shape, exact blend formula,
  state-dict schema identity, and strict checkpoint reload.
- Verification:
  compile passed and focused stem/resume/trainable-prefix pytest passed
  (`10 passed`). A broader local suite reported `137 passed, 4 failed`; the
  four failures were existing dirty-worktree contract mismatches (three old
  dataset tests expecting two-item tuples and one stale hybrid-config field),
  not failures in the pooling path.
- Direct override precheck:
  on the same first eight validation batches (`384` rows), legacy max reached
  macro/class1 `0.7607/0.8462`. Direct SoftPool fell to `0.7253/0.7826` and
  changed 21 predictions with only three corrections versus 18 harms.
  Conservative `max_soft` blend `0.15` reached `0.7492/0.8000`; its only two
  changes versus max were both harms (`3->2` and true `1->2`). This ruled out
  inference-only replacement but still left a short adaptation smoke as the
  valid test of the new representation.
- Gate and smoke:
  `runs\audit_trkh_smoke_gate_softpool_stem_candidate_20260710` opened only the
  short-smoke gate for the genuinely new stem representation. Run
  `runs\smoke_v8_yolof_maxsoft015_stemadapt_60b_1e_20260710` resumed the keeper,
  trained `60b/1e`, used full `yolo_f/val=2606`, retained the current bbox,
  teacher-focus-binary, pairwise-routing, metric, and boundary-drop recipe, and
  skipped test. Validation macro/class1 was only `0.8803/0.6627`, class1 P/R
  `0.5989/0.7417`, below keeper `0.8847/0.6860` and current runtime softboost
  `0.8887/0.7030`. True-class1 confusion was `[21,112,13,0,5]`, so the stem
  adaptation lost recall rather than adding the missing class1 cue.
- Independent verifier and boundary audit:
  full-val verifier-aware evaluation
  `runs\eval_maxsoft015_stemadapt_smoke_softboost001_full_val_20260710` fell
  further to macro/class1 `0.8714/0.6187`, class1 P/R `0.6772/0.5695`; class1
  `1->0` rose to `47`. The balanced diagnostic manifest
  `runs\boundary_review_maxsoft015_stemadapt_smoke_val_20260710` selected 48
  rows from 370 candidates: 12 each of class1 FN, class1 FP, low-margin error,
  and low-margin correct boundary. No test rows were used.
- XAI:
  `runs\xai_maxsoft015_stemadapt_smoke_val_cases12_20260710` audited 12 locked
  validation errors (four class1 FN, four class1 FP, four other boundary
  errors) with attention, rollout, gradient rollout, Grad-CAM, and robustness
  counterfactuals. Mean attention/grad-rollout foreground mass stayed high
  (`0.8709/0.8982`), but Grad-CAM background/border mass was
  `0.3544/0.3313`; flags counted Grad-CAM background `6/12`, Grad-CAM border
  `7/12`, and object-color sensitivity `8/12`. Mean original-prediction drop
  from object desaturation was `0.1371`, versus `0.0242/0.0254` for background
  blur/gray, and only one case received each global background-shortcut flag.
  The failure therefore remains local surface/border separability plus class1
  recall collapse, not a broad background shortcut fixed by stem pooling.
- Decision:
  reject direct SoftPool and `max_soft=0.15` keeper adaptation. Do not sweep
  nearby blend, LR, train batches, epochs, query tokens, or trainable prefixes
  on this checkpoint. Keep the default-off, checkpoint-compatible
  implementation for a future architecture ablation only if a materially new
  representation target justifies it. No probe or full train is permitted from
  this result.
- Cleanup:
  `runs\cleanup_manifest_20260711_softpool_stem_rejected.json` removed the two
  rejected checkpoints, 12 rendered XAI case directories, and 157 plot PNGs
  after preserving metrics/config/history/trace summary, full-val predictions,
  boundary manifest, and XAI JSON/CSV/markdown. Deleted payload was
  `145.954 MB` (`146.512 MB` observed free-space delta); raw data was untouched.
- Retention:
  `runs\artifact_retention_audit_after_softpool_reject_20260711` scanned 496
  run directories, passed with `blockers=[]`, protected artifacts present,
  compacted originals remaining `0`, and `79.960 GB` free. The seven small
  future cleanup candidates (`31.602 MB`) remain review-only.

## Smoke 2026-07-11 - Shifted Patch Token Residual Rejected

- Literature and route selection:
  Vision Transformer for Small-Size Datasets
  (`https://arxiv.org/abs/2112.13492`) proposes Shifted Patch Tokenization
  (SPT) and Locality Self-Attention to restore local inductive bias when ViTs
  train from scratch on small datasets. SPD-Conv
  (`https://arxiv.org/abs/2208.03641`) similarly argues that strided
  convolution/pooling loses fine detail, while LIFE
  (`https://arxiv.org/abs/2305.08551`) adds efficient local information to ViT
  attention. Local repo review found no prior SPT/LSA/space-to-depth route, but
  did find that broad context, paired views, local token heads, high-pass
  experts, and frequency pooling were already rejected. I selected SPT alone
  so the representation change could be attributed without combining LSA.
- Implementation:
  added default-off `ShiftedPatchTokenResidual`. It zero-pads and shifts the
  post-stem feature map in four diagonal directions, concatenates the views,
  projects them at the existing patch stride, and adds a fixed-scale residual
  to legacy patch tokens. Zero-initialization preserves old checkpoint logits
  exactly; the old state dict may miss only
  `shifted_patch_token_residual.proj.{weight,bias}`. Added ModelConfig/train CLI
  and V8 launcher fields, resume-extension allow-list support, trainable-prefix
  support, and trace residual norm/heatmaps. The tested branch adds `1,048,832`
  parameters; with the old classifier head, `1,050,117` parameters were
  trainable and `7,244,305` remained frozen.
- Preflight:
  tests cover non-wrapping zero-padded shifts, output shape, zero-init,
  gradients, exact base-logit equivalence, allowed checkpoint extension, V8
  scheduler forwarding, branch resume, and prefix freezing. Compile and focused
  suites passed (`13 passed`, then `10 passed` after the launcher fix), and the
  PowerShell parser passed.
- Scheduler audit and launcher fix:
  V8 previously hard-coded `--scheduler-total-epochs $Epochs`, so a one-epoch
  smoke decayed from base LR to min LR across its optimizer steps. The history
  LR is the end-of-epoch value, not proof that the whole epoch used min LR.
  Added independent `-SchedulerTotalEpochs` with validation, preflight/launcher
  recording, effective fallback to `Epochs`, and a regression test. Full-train
  behavior remains unchanged by default; short architecture smokes can now use
  a longer decay horizon explicitly.
- Smoke schedule sensitivity:
  all runs resumed the keeper, trained only the SPT residual plus legacy head,
  used shift `1`, scale `0.10`, `60` train batches, full `yolo_f/val=2606`, and
  no test. Base LR `2e-4` with one-epoch decay reached macro/class1
  `0.8840/0.6842`; base LR `1e-4` with one-epoch decay reached
  `0.8834/0.6822`; after the launcher fix, base LR `1e-4` with scheduler horizon
  `10` ended at LR `9.76e-5` and reached `0.8830/0.6822`, class1 P/R
  `0.6094/0.7748`. Every schedule remained below keeper `0.8847/0.6860` and
  far below current softboost class1 `0.7030`.
- Architecture trace:
  the effective smoke produced SPT residual norm mean `0.0694-0.0997` and max
  `0.1449-0.3051` across the five traced classes, so the branch was active.
  Residual maps mostly reproduced legacy patch-norm structure and emphasized
  object ends, bright spots, and crop/padding boundaries rather than a new
  interior surface cue.
- Independent verifier and boundary audit:
  the current runtime patch verifier reduced the effective SPT checkpoint to
  macro/class1 `0.8772/0.6502`, class1 P/R `0.6970/0.6093`, with `1->0=43`.
  The balanced validation review found 357 candidates and retained 48 rows,
  12 each for class1 FN, class1 FP, low-margin error, and low-margin correct
  boundary. No test rows were used.
- XAI:
  verifier-aware audit used 12 locked validation errors. Attention and
  grad-rollout foreground mass were `0.9462/0.8979`; Grad-CAM
  background/border mass was `0.2287/0.2218`. Flags were Grad-CAM background
  `3/12`, Grad-CAM border `3/12`, and object-color sensitive `8/12`.
  Object-desaturation prediction drop `0.1259` remained about ten times larger
  than background blur/gray `0.0115/0.0125`, while CLS/register heatmap
  similarity stayed `0.99979`. Visual FN/FP cases concentrated on surrounding
  branches, padding bands, and fruit endpoints instead of discriminative
  interior texture.
- Decision:
  reject the exact checkpoint-adapter route
  `shift=1/scale=0.10/trainable=shifted_patch_token_residual+head/60b`. Do not
  sweep nearby shifts, scale, LR, scheduler horizon, batch count, or head
  freezing on the current keeper. Keep SPT default-off as a reusable
  from-scratch architecture component; it did not create a transferable class1
  surface signal here, so no probe/full train is permitted.
- Cleanup and retention:
  compacted three smoke schedules plus verifier/boundary/XAI roots into
  `runs\evidence_shifted_patch_tokenization_rejected_20260711`, preserving 84
  files (`3.626 MB`) including metrics/config/history, prediction CSVs, five SPT
  heatmaps, and two visual XAI triplets. Manifest
  `runs\cleanup_manifest_20260711_shifted_patch_tokenization_rejected.json`
  records `467.494 MB` source, `463.868 MB` deleted payload, and `465.070 MB`
  observed free-space delta. Retention
  `runs\artifact_retention_audit_after_spt_reject_20260711` passed over 499 run
  directories with no blockers, all six compacted originals absent, protected
  artifacts present, and `79.955 GB` free.

## Smoke 2026-07-11 - Gated Relative-Position Attention Rejected

- Literature and design:
  ConViT/GPSA (`https://proceedings.mlr.press/v139/d-ascoli21a`) mixes
  content and fixed relative-position attention after softmax with a learned
  per-head gate, while SATA
  (`https://openaccess.thecvf.com/content/WACV2023/html/Chen_Accumulated_Trivial_Attention_Matters_in_Vision_Transformers_on_Small_Datasets_WACV_2023_paper.html`)
  motivates preserving useful local bias on small datasets. The implementation
  follows the GPSA principle only on patch-to-patch mass: prefix rows and
  patch-to-prefix links remain unchanged, and retained `patch_indices` preserve
  original coordinates after pruning. Eight fixed directional projections use
  `[dx, dy, distance^2]`; a per-head gate blends positional and content patch
  distributions while preserving each row's original patch mass.
- Implementation and preflight:
  added default-off `GatedPatchRelativePositionAttention` plus ModelConfig,
  train CLI, V8 launcher, resume-extension, trainable-prefix, and architecture
  trace wiring. Trace output records raw/effective gate, expected content,
  positional and mixed distance, local mass, and center positional maps. The
  keeper diagnostic had exact base-logit identity (`max_delta=0.0`), exactly
  eight allowed missing extension keys, `1,413` trainable parameters including
  the legacy head, and nonzero gate gradients in all four enabled blocks.
  Focused tests passed `12/12`; the broad relevant suite passed `138` tests
  with four pre-existing dirty-worktree failures in stale dataset/config tests.
- Infrastructure correction:
  the first smoke
  `smoke_v8_yolof_gpsa_relpos1234_max025_lr3e4_sched10_60b_1e_20260711`
  is invalid representation evidence. Partial checkpoint loading copied the
  keeper's mature EMA update counter while new attention keys started at zero,
  so the short-run EMA almost ignored the extension. Training now resets EMA
  updates for partial architecture extensions and reports
  `updates_source=reset_for_partial_architecture_extension`. Also, direct gate
  clamping could permanently kill a head after a negative update; training now
  uses a straight-through projected clamp while evaluation uses the true clamp.
  Regression tests cover both behaviors.
- Valid smoke:
  `smoke_v8_yolof_gpsa_relpos1234_emafixste_max025_lr3e4_sched10_60b_1e_20260711`
  resumed the keeper, enabled layers 1-4, used max mix `0.25`, LR `3e-4`,
  scheduler horizon `10`, `60` train batches, full `yolo_f/val=2606`, and no
  test. It reached macro/class1 F1 `0.87895/0.65455`, class1 P/R
  `0.60335/0.71523`, with `1->0=17` and `1->2=19`. EMA gate maxima were only
  `0.00136/0.00430/0.00306/0.00396` across layers 1-4; the optimizer left only
  `3/5/4/6` active heads. Positional expected distance was about `1.17-1.18`
  patches versus content `7.85-8.48`, but the tiny learned mix changed expected
  distance by only about `0.002-0.009` patches.
- Independent state and runtime checks:
  evaluating raw train weights gave macro/class1 `0.87503/0.65193`, class1 P/R
  `0.55924/0.78146`, so EMA was not hiding a useful solution. Applying the
  current patch-linear softboost verifier reduced the valid EMA checkpoint to
  `0.87113/0.61702`, class1 P/R `0.66412/0.57616`, with `1->0=38` and
  `1->2=19`. The balanced validation boundary review retained 48 rows, 12 each
  for class1 FN, class1 FP, low-margin error, and low-margin correct boundary.
- XAI:
  verifier-aware audit used 12 locked validation cases and no test. Attention,
  gradient-rollout, Grad-CAM and rollout foreground mass were
  `0.8965/0.9221/0.7008/0.8760`; Grad-CAM and rollout background/border flags
  were `7/9` and `7/7`. Object desaturation changed the original prediction by
  `0.14755` on average versus only `0.01544/0.01542` for background blur/gray;
  CLS/register heatmaps remained nearly identical (`0.9997805`). Visual class1
  FN and `0->1` FP cases still emphasized stems, crop borders, surrounding
  branches, endpoints, or broad color regions rather than a new discriminative
  interior texture cue.
- Decision:
  reject GPSA as a short checkpoint adapter. It is below the no-pretrain keeper
  (`0.8847/0.6860`) and far below current runtime softboost
  (`0.8887/0.7030`); raw weights and the verifier do not rescue it. Do not sweep
  nearby LR, max mix, enabled layers, batches, epochs, or head freezing on this
  checkpoint. Keep the rigorously tested implementation default-off for a
  future from-scratch architecture ablation. The EMA-reset rule applies to all
  future partial architecture extensions before their short-run metrics may be
  treated as valid evidence.
- Cleanup and retention:
  compacted both GPSA smokes plus raw-state, verifier, boundary, and XAI roots
  into `runs\evidence_gpsa_relative_position_rejected_20260711`. It preserves
  80 files (`7.240 MB`) including full-val detailed predictions, trace summaries
  and center maps, transition XAI, and three visual case groups. Manifest
  `runs\cleanup_manifest_20260711_gpsa_relative_position_rejected.json` records
  `328.130 MB` source and `329.340 MB` observed free-space delta. Retention
  `runs\artifact_retention_audit_after_gpsa_reject_20260711` passed over 502 run
  directories with `blockers=[]`, `14/14` protected checks present, all seven
  compacted originals absent, raw data untouched, and `79.947 GB` free.

## Diagnostic 2026-07-11 - Locality Self-Attention Precheck Rejected

- Literature and hypothesis:
  Vision Transformer for Small-Size Datasets
  (`https://arxiv.org/abs/2112.13492`) reports that LSA independently improved
  Tiny-ImageNet by `+3.60%`. Its two mechanisms are diagonal masking, which
  removes self-token relations, and a learned lower softmax temperature, which
  sharpens inter-token attention. Attention Temperature Matters in ViT-Based
  Cross-Domain Few-Shot Learning
  (`https://proceedings.neurips.cc/paper_files/paper/2024/hash/d2fe3a5711a6d488da9e9a78b84ee24c-Abstract-Conference.html`)
  provides a useful counterpoint: smoothing can improve transfer when learned
  query-key attention is wrong. Local no-repeat review confirmed LSA had only
  been cited during the SPT route and had never been implemented or measured.
- Diagnostic implementation:
  added validation-only `trkh.tools.probe_locality_self_attention`. Runtime
  hooks operate on the keeper's actual pruned-token path and measure patch mass,
  self mass, normalized entropy, top-1 mass, and self-top1 rate per layer. Fixed
  transforms preserve all prefix query rows, patch-to-prefix links, and each
  patch query's total patch mass. The matrix tests identity, smoothing `0.75`,
  sharpening `1.25`, diagonal suppression `0.50`, and sharpening with soft or
  hard diagonal suppression. It writes no model, trainable manifest, test
  result, or raw-data change. Compile plus focused GPSA/EMA/LSA tests passed
  `17/17`.
- Full-validation validity:
  `runs\diagnostic_lsa_patchonly_full_val_matrix_20260711` used all
  `yolo_f/val=2606` rows and reproduced the keeper exactly at macro/class1
  `0.884675/0.686047`, class1 P/R `0.611399/0.781457`. Baseline conditional
  self fractions on layers 1-4 were only
  `0.004446/0.002607/0.003150/0.004030`; self was top-1 only
  `0.005272/0.002299/0.003093/0.003903` of the time. The LSA diagonal therefore
  had almost no pathological self relation to remove. Normalized entropies were
  `0.941105/0.736427/0.775801/0.813968`.
- Fixed-transform results:
  smoothing reached only `0.882298/0.674419` with 5 corrections and 7 harms;
  sharpening reached `0.881644/0.676301` with 4 corrections and 9 harms.
  Diagonal `0.50` changed no prediction. Sharpening plus diagonal `0.50` reached
  `0.882578/0.680115` with 4 corrections and 8 harms; hard diagonal reached
  `0.883175/0.682081` with 4 corrections and 7 harms. The hard variant rescued
  one class1 FN but broke one class1 TP, removed two class1 FP, and created four.
  Sharpening's dominant new failure was `0:0->1`; no variant passed the fixed
  macro/class1, correction-harm, recall-protection, and FP-control gate.
- Decision:
  reject patch-only LSA and nearby attention-temperature tuning as a keeper
  adapter. Do not implement a trainable temperature/self-mask module or sweep
  layer subsets, suppression, or multipliers around these values. The low
  baseline self mass invalidates the diagonal-mask premise, while both
  smoothing and sharpening amplify more wrong decisions than they correct.
  No model weights changed, so a duplicate model XAI run was not warranted;
  row-level attention statistics and all changed transitions are the relevant
  audit for this inference-only gate.
- Cleanup and retention:
  losslessly gzipped full prediction and attention CSVs and removed the
  superseded eight-row dry-run. Manifest
  `runs\cleanup_manifest_20260711_lsa_precheck_compacted.json` reduced retained
  evidence from `18.874` to `6.154 MB` and observed `12.820 MB` freed. Retention
  `runs\artifact_retention_audit_after_lsa_precheck_20260711` passed over 504
  run directories with `blockers=[]`, `14/14` protected artifacts present, the
  dry-run absent, raw data untouched, and `79.940 GB` free.

## Full Train 2026-07-11 - Best-Recipe Continuation Rejected

- Purpose and validity:
  after fixing the PowerShell native-stderr launcher failure, completed the
  previously missing continuation of the current V8 `yolo_f` no-pretrain
  teacher-focus-binary recipe. Run
  `runs\full_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_30e_autorun_20260711`
  resumed the raw keeper, allowed at most 30 epochs with patience 3, wrote a
  five-class architecture trace, used full `yolo_f/val=2606`, and did not open
  test. It has `7,245,590` parameters and ran for `1078.843 s`.
- Optimization result:
  early stopping occurred after epoch 5, with best epoch 2. Raw best validation
  macro/class1 F1 was `0.886808/0.681690`, class1 P/R
  `0.593137/0.801324`, and fair selection metric `0.762266`; the keeper's
  selection metric is `0.762949` and its macro/class1 F1 is
  `0.884675/0.686047`. Epochs 1-5 class1 F1 were
  `0.674352/0.681690/0.672131/0.663014/0.668478`. Continuation therefore
  increased class1 recall while losing too much precision and never passed the
  class1 gate.
- Frozen runtime verifier:
  `runs\eval_full_v8_yolof_pairroute_tfb015_30eautorun_best_softboost001_full_val_20260711`
  reached only macro/class1 `0.877042/0.637288`, class1 P/R
  `0.652778/0.622517`. Its confusion matrix contains `1->0=41`, `1->2=10`,
  `0->1=26`, and `4->1=12`; the verifier over-suppresses the new checkpoint's
  true class1 predictions instead of recovering a useful precision-recall
  operating point.
- Boundary and XAI audit:
  the balanced review retained 48 validation rows, 12 each for class1 FN,
  class1 FP, low-margin error, and low-margin correct, then locked 12 cases for
  verifier-aware XAI. Attention/grad-rollout/Grad-CAM/rollout foreground mass
  was `0.92323/0.88368/0.72601/0.84174`; Grad-CAM background/border flags were
  `5/12` and `6/12`, rollout flags `9/12` and `7/12`, and object-color
  sensitivity `8/12`. Object-desaturation prediction drop `0.15491` remained
  much larger than background blur/gray `0.02342/0.02601`; CLS/register
  similarity was `0.999888`. Visual `1->0`, `3->1`, and `4->1` cases still
  emphasized stems, branches, endpoints, padding/borders, or broad color rather
  than a new interior-surface cue.
- Decision:
  reject this exact continuation as the new keeper. Preserve the original raw
  keeper and current softboost runtime reference; do not run test or extend the
  same checkpoint by changing only epoch count, patience, LR, or verifier
  strength. This result closes the infrastructure uncertainty but confirms that
  more optimization of the current representation does not solve class1.
- Cleanup and retention:
  compacted the full run, frozen-verifier evaluation, boundary review, and XAI
  roots into `runs\evidence_full_continuation_rejected_20260711`. It preserves
  58 files (`5.453 MB`) including config/history/metrics, full-validation row
  predictions, trace summaries, balanced review, transition XAI, and three
  visual groups; it contains no checkpoint or test result. The cleanup command
  completed deletion but its post-delete `DriveInfo.Refresh` measurement was
  unsupported, so unknown source-size/free-delta fields are explicitly `null`
  in `runs\cleanup_manifest_20260711_full_continuation_rejected.json` rather
  than estimated. Independent retention
  `runs\artifact_retention_audit_after_full_continuation_cleanup_20260711`
  passed over 506 run directories with `blockers=[]`, all protected artifacts
  present, and all four compacted originals absent.
- Research handoff pack:
  `docs\TRKH_DEEP_RESEARCH_REFERENCE_PACK_20260711.zip` contains 190 payload
  files plus an embedded manifest, `14.517 MB` uncompressed and `7.634 MB`
  compressed. It combines the current brief/journal/TODO, dataset metadata,
  model source snapshot, keeper/runtime/final-test summaries, negative-method
  evidence, XAI examples, and AIDT metadata. Every payload entry was read back
  from the ZIP and matched against SHA-256; checkpoints, raw dataset images,
  and full row-level dataset manifests are excluded.

## Diagnostic 2026-07-11 - BBox-Aligned Raw-Context/Crop Transfer Rejected

- Literature and no-repeat rationale:
  DropPos (`https://proceedings.neurips.cc/paper_files/paper/2023/hash/9098e2901b4eb54772f83535f89cb8ac-Abstract.html`)
  and LOCA
  (`https://openaccess.thecvf.com/content/WACV2024/papers/Caron_Location-Aware_Self-Supervised_Transformers_for_Semantic_Segmentation_WACV_2024_paper.pdf`)
  motivate explicit patch-location reasoning, while Siamese Image Modeling
  (`https://openaccess.thecvf.com/content/CVPR2023/papers/Tao_Siamese_Image_Modeling_for_Self-Supervised_Vision_Representation_Learning_CVPR_2023_paper.pdf`)
  uses crop-relative coordinates to remove ambiguity from dense cross-view
  prediction. Local review found that the earlier `patch_tokens` paired loss
  used energy top-k plus soft nearest-neighbor matching and never used the
  known bbox geometry. This precheck therefore measured exact object-relative
  correspondence before adding another trainable paired loss.
- Implementation and preflight:
  added diagnostic-only
  `trkh.tools.probe_paired_bbox_aligned_patch_correspondence`. It maps retained
  `patch_indices` through the transformed primary bbox into the paired crop,
  records nearest geometric matches after token pruning, and compares matched
  cosine/retrieval with a deterministic within-object control. It also applies
  a predeclared class1 gate covering FN correction, TP break, and FP
  create/suppress budgets. `raw_context` uses the full YOLO image with
  `classification_source_context` and `background_alpha=1.0`, so source pixels
  are unchanged; bbox metadata only identifies the object. Compile and focused
  paired/context tests passed `14/14`. A gate bug that treated exact numeric
  zero as a missing value was found by the 64-row dry run, fixed, and covered
  by regression test before full validation.
- Full-validation result:
  `runs\diagnostic_paired_bboxalign_rawcontext_full_val_20260711` used all
  `yolo_f/val=2606` object rows, mapped each to `class_f`, and used no test or
  trainable output. Geometry itself passed: match coverage `2605/2606`, mean
  pair count `43.29`, matched/control cosine `0.60074/0.29195`, margin
  `0.30879`, feature top1/random `0.07904/0.00599`, top1 lift `0.07306`, MRR
  `0.16106`, and mean geometric distance `0.03575`. The unmatched row was an
  extreme edge box only `0.0078` wide.
- Classification safety:
  the crop-trained keeper is not raw-context ready. Raw context reached only
  macro/class1 F1 `0.57793/0.19792`; paired class-f crop reached
  `0.88065/0.67055`. Crop predictions corrected `707` raw-context errors and
  harmed `53`; for class1 they rescued `97/132` FN and broke `1/19` TP. The
  decisive failure was false-positive control: paired crop created `67`
  class1 FP while suppressing only `12`. Geometric similarity was not an error
  detector either: correspondence margin was `0.3168` on primary errors versus
  `0.3048` on correct rows.
- Decision:
  reject direct bbox-aligned crop-to-context feature transfer before smoke.
  Geometry signal alone is real, but matching class-bearing crop features would
  transfer the same unsafe class1 bias and fails the predeclared FP budget. Do
  not implement or sweep this paired dense-alignment loss by changing weight,
  match radius, top-k, context alpha, or layer on the current keeper. A future
  location-aware route must predict geometry/position without using crop
  features as a class teacher, and must receive its own independent precheck.
- Cleanup and retention:
  preserved the 64-row object-crop control summary inside the full artifact and
  deleted both superseded dry-run roots. Manifest
  `runs\cleanup_manifest_20260711_paired_bboxalign_precheck_dryruns.json`
  records eight source files (`0.082 MB`) and `0.094 MB` observed free delta.
  Retention
  `runs\artifact_retention_audit_after_paired_bboxalign_precheck_20260711`
  passed over 508 run directories with `blockers=[]` and both originals absent.

## Diagnostic 2026-07-11 - BBox Position Reconstruction Target Rejected

- Hypothesis and implementation:
  after rejecting crop-to-context class-feature transfer, added diagnostic-only
  `trkh.tools.probe_bbox_position_reconstruction_readiness` to test a safer
  DropPos/LOCA-style target. The tool removes only patch positional embeddings
  while preserving prefix positions, samples retained tokens inside each
  transformed crop bbox, and fits an in-memory ridge decoder on train tokens to
  predict bbox-relative `(x,y)`. Full validation then measures both position
  decodability and the keeper classifier's response to the zero-position
  ablation. It writes no checkpoint, model, or trainable manifest and never
  opens test. Compile and the related focused suite passed `15/15` after adding
  logit/probability-delta telemetry.
- Position signal:
  `runs\diagnostic_bbox_position_readiness_poszero_train120_full_val_20260711`
  fit the frozen ridge decoder on `122868` tokens from `3840` train rows and
  audited all `2606` validation rows with `83390` tokens. All-class position
  mean R2/MAE was `0.215319/0.205873`; 4x4 joint accuracy was `0.154671`
  versus random `0.0625`. Class 1 was not the weak subset: its `4832` tokens
  reached mean R2/MAE `0.281719/0.197190` and 4x4 accuracy `0.164528`.
- Classification sensitivity:
  zeroing every patch positional embedding changed macro F1 only
  `0.884675 -> 0.884067` and class1 F1 `0.686046 -> 0.682216`. Only `8/2606`
  decisions changed, split evenly into `4` corrections and `4` harms. Mean/max
  absolute logit delta was `0.003426/0.154297`; mean/max probability delta was
  `0.000659/0.027983`. The predeclared support, R2, 4x4, MAE, class1-safety,
  and noncatastrophic checks all passed, but classification sensitivity failed
  because `8 < 25` predictions changed.
- Decision:
  reject DropPos or bbox-relative position reconstruction from this exact
  signal before smoke. Final tokens contain linearly decodable location, but
  the classification path is nearly invariant to the positional embeddings;
  adding an auxiliary merely because its target is learnable would optimize a
  largely orthogonal property. Do not sweep ablation scale, ridge strength,
  token cap, position-loss weight, or DropPos probability on this keeper. A
  future position route needs evidence that the proposed signal separates
  class1 recall errors from `0/2/4->1` false positives, not only coordinate
  recoverability. No model changed, so a duplicate image XAI run was not
  warranted; full row transitions and perturbation deltas are the relevant
  audit.
- Cleanup and retention:
  removed the superseded 4-train-batch/4-validation-batch dry run after the
  full artifact and expanded README were preserved. Manifest
  `runs\cleanup_manifest_20260711_bbox_position_readiness_dryrun.json`
  records three files (`0.010 MB`), `0.012 MB` observed free delta, no test or
  raw-data use, and zero remaining originals. Retention
  `runs\artifact_retention_audit_after_bbox_position_precheck_20260711`
  passed over 510 run directories with `blockers=[]`, all protected artifacts
  present, and `79.925 GB` free.

## Diagnostic 2026-07-11 - CIConv-W Shared-Trunk Complementarity Rejected

- Literature and no-repeat audit:
  CIConv (`https://openaccess.thecvf.com/content/ICCV2021/html/Lengyel_Zero-Shot_Day-Night_Domain_Adaptation_With_a_Physics_Prior_ICCV_2021_paper.html`)
  turns photometric invariants derived from a reflection model into a learnable
  convolutional prior; the authors' official MIT implementation
  (`https://github.com/Attila94/CIConv`) reports W as its strongest general
  invariant. Multiscale Retinex
  (`https://ntrs.nasa.gov/api/citations/19990005051/downloads/19990005051.pdf`)
  and intrinsic-image work
  (`https://www.cs.sfu.ca/~mark/ftp/Eccv04/`) independently motivate separating
  reflectance/structure from illumination. Local history confirmed this was not
  a repeat: TRKH had tried RGB brightness/gamma consistency, HSV/Lab summary
  statistics, and raw high-pass/gradient/Laplacian experts, but no physics-based
  color-invariant derivative view.
- Implementation and fixed protocol:
  added `trkh.models.photometric_invariant.ColorInvariantW`, following the
  paper's Gaussian color model and W equations, plus device-safe padding-mask
  halo suppression. Diagnostic-only
  `trkh.tools.probe_photometric_invariant_complementarity` runs the frozen
  keeper trunk on the fixed official-default `W/scale=0` view, clips only at
  `3` standardized units, preserves RGB as a separate branch, and concatenates
  L2-normalized RGB/W head embeddings. One predeclared balanced logistic
  readout (`C=0.3`) used five-fold `StratifiedGroupKFold` over source images;
  there was no scale/clip/readout sweep. The existing keeper RGB cache avoided
  a duplicate RGB pass. Compile and focused tests passed `6/6`, including
  intensity invariance, finite normalization, invalid-padding suppression, and
  the recall/FP gate.
- Full validity and visual audit:
  `runs\diagnostic_ciconv_w_complementarity_full_train_val_20260711` matched
  labels, paths, and order exactly for all `9215` train and `2606` validation
  object rows, with `8064` grouped train sources and no test. The preview shows
  genuine fruit spots, wrinkles, contour, and shadow structure; mean border
  mass was only `0.27434` and fewer than `0.001` values clipped. The frozen
  shared trunk did not collapse dimensionality: RGB/W effective ranks were
  `6.063/6.820` (W/RGB ratio `1.125`). The failure is lack of class-separating
  information, not an effective-rank collapse.
- Readout result:
  RGB-only grouped OOF macro/class1 was `0.93328/0.79934`, with validation
  `0.88106/0.66667`. W alone was weak at OOF `0.69225/0.29416` and validation
  `0.64965/0.21445`. Adding W made both controls worse: RGB+W OOF fell to
  `0.92939/0.78385`, and validation fell to `0.87421/0.63973`. This is below
  both the same RGB readout and direct keeper `0.88468/0.68605`; all OOF/val
  gain and class1 milestone checks failed.
- Error audit and decision:
  versus the direct keeper, RGB+W changed `117` validation decisions with `50`
  corrections, `55` harms, and `12` neutral transitions. It suppressed class1
  false positives (`32` removed versus `8` created), but did so by breaking
  `25` true class1 predictions while rescuing only `2/33` class1 false
  negatives. FN-versus-FP probability-delta AUC was only `0.55879`. Reject
  shared-trunk CIConv-W before smoke: the visible invariant edge map is mainly
  another unsafe class1 suppressor after this trunk, not a recall-safe surface
  representation. Do not sweep W scale, clipping, readout C, fusion weight, or
  a nearby zero-init shared-trunk adapter on this keeper. A future physics-view
  revisit would require an independently validated dedicated encoder or a
  different reflectance target that protects subtle true class1 color cues;
  this result gives no permission for one now.
- Cleanup and retention:
  deleted the superseded `1024/256` dry-run root after preserving the full
  caches, row audits, preview, summary, and expanded README. Manifest
  `runs\cleanup_manifest_20260711_ciconv_w_precheck_dryrun.json` records eight
  files (`2.034 MB`) and `2.051 MB` observed free delta. Retention
  `runs\artifact_retention_audit_after_ciconv_w_precheck_20260711` passed over
  512 run directories with `blockers=[]`, all protected artifacts present, and
  the dry-run absent.

## Research Synthesis 2026-07-11 - External Reports 14/15

- Preserved the two user-supplied UTF-8 reports as
  `docs\research_inputs\TRKH_DEEP_RESEARCH_REPORT_14_20260711.md` and
  `docs\research_inputs\TRKH_DEEP_RESEARCH_REPORT_15_20260711.md`. Their
  SHA-256 values are respectively
  `561BA5A677CCA7A2A52EC100EA1AAC91851F40760F746A4153238DFE94374D7F`
  and `7933984F468C90A394E5642088C56DA09518C8B88E66CFA5A64F38AA46A7D4B0`.
- The reports' strongest correction is about scope. Failed SPT/GPSA/LSA and
  the sparse CMT/LeFF adapter do not disprove a compact concurrent hybrid
  trained from initialization; failed kNN/prototype/sub-center routes disprove
  them only on the current embedding/queue policies; failed KD/router routes do
  not make external teachers useless as uncertainty critics. Future docs must
  retain those narrower conclusions.
- Local no-repeat comparison separated genuinely new work from renamed old
  work. Whole-patch compact bilinear pooling already failed twice, including
  `rank=24/dropout=0.05`; bbox foreground means, token-label supervision,
  part-token heads, sub-centers, neighbors, soft targets, ordinal heads, and
  generic color invariance also have negative evidence. The still-open
  second-order hypothesis is narrower: a fixed geometry `interior core`
  descriptor, with boundary/context kept as controls instead of mixed into the
  class descriptor. The still-open architecture hypothesis is a concurrent or
  stage-wise compact hybrid trained from epoch zero, not another zero-init
  adapter on the keeper.
- Primary-source checks support the mechanism but not guaranteed TRKH gains:
  B-CNN models local pairwise interactions for fine-grained recognition
  (`https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html`),
  iSQRT-COV reports fast covariance pooling and shorter convergence
  (`https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Towards_Faster_Training_CVPR_2018_paper.html`),
  CCT targets from-scratch small-data transformers
  (`https://arxiv.org/abs/2104.05704`), and CMT combines local convolution with
  long-range attention (`https://arxiv.org/abs/2107.06263`). These justify one
  focused diagnostic/implementation round, not a 30-epoch blind run.
- Adopted sequence: first run a train/grouped-OOF plus full-validation
  interior-core second-order readiness gate; only a passing result may become a
  trainable readout. A concurrent hybrid-from-scratch candidate remains open
  independently. Two-stage distributional/evidential supervision is phase 2,
  after a new backbone/readout demonstrates cleaner class-1 representation;
  sub-center/neighbor regularization remains phase 3.

## Diagnostic 2026-07-11 - Context Gray-Edge Color Constancy Rejected

- Motivation and fixed protocol: following Gray-Edge
  (`https://staff.science.uva.nl/th.gevers/pub/GeversTIP07.pdf`), Shades of Gray
  (`https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/12/1/art00008`),
  and the reports' structured illumination recommendation, the diagnostic used
  raw `yolo_f` full context only to estimate the nuisance illuminant. It excluded
  the target bbox plus margin `0.08`, used first-order Gray-Edge `sigma=2/p=6`,
  applied diagonal gains clipped to `0.5-2.0` only to the object crop, preserved
  RGB as a separate branch, and used one `C=0.3` five-fold source-grouped OOF
  readout. There was no candidate sweep, model training, test use, or raw-data
  edit. Compile plus focused dataset/photometric tests passed `15/15`.
- Full artifact
  `runs\diagnostic_context_grayedge_p6_s2_full_train_val_20260711` covered and
  aligned all `9215/2606` train/validation rows. Context support was healthy
  (train/val mean `0.4819/0.4561`, p01 `0.1299/0.1070`), gains never clipped,
  and corrected/RGB effective rank was preserved (`6.551/6.063`, ratio
  `1.080`). Mean context-to-neutral angle was only `3.64/3.43` degrees, while
  context-to-crop illuminant angle was `4.32/4.41` degrees. Visual inspection
  confirmed a conservative transform that retained spots and texture, although
  hard-shadow samples sometimes shifted toward cyan.
- Direct canonicalization failed decisively: keeper validation macro/class1
  `0.88468/0.68605` fell to `0.79411/0.45326`. It changed `309` decisions with
  `61` corrections and `232` harms, rescued/broke `10/48` true-class1 FN/TP,
  and removed/created `31/78` class1 false positives.
- Dual-view complementarity also failed. RGB grouped OOF macro/class1
  `0.93328/0.79934` fell to `0.93035/0.79149`; RGB validation readout
  `0.88106/0.66667` fell to `0.87857/0.66000`. Versus the direct keeper the
  fusion changed `120` rows (`54` corrections, `55` harms), rescued only one
  class1 FN while breaking `20` true class1 predictions, and its FN-vs-FP
  probability-delta AUC was `0.48040`.
- Decision: reject both direct Gray-Edge preprocessing and shared-trunk
  RGB+corrected fusion before smoke. Do not sweep p, sigma, exclusion margin,
  gain bounds, readout C, or fusion strength on this keeper. The transform is
  numerically valid but removes/rebalances class-bearing ripeness color in a
  direction the frozen representation cannot distinguish from illumination.
  A future color-constancy revisit requires a jointly trained interior-level
  consistency objective inside a genuinely new representation, with explicit
  class1 recall protection; this result gives no smoke permission now.
- Cleanup and retention: removed the superseded `1024/256` dry-run caches plus
  duplicate stdout/stderr logs after expanding the full artifact README.
  `runs\cleanup_manifest_20260711_context_grayedge_dryrun_logs.json` records
  `12` files, `2.771 MB` source, and `2.802 MB` observed free-space delta.
  `runs\artifact_retention_audit_after_context_grayedge_precheck_20260711`
  passed over `514` run directories with `blockers=[]`; the full artifact and
  all protected keepers remain present, while the dry root is absent.

## Diagnostic 2026-07-11 - Interior-Core Second-Order Readout Rejected

- Rationale and scope: external reports 14/15 correctly distinguish a
  geometry-isolated interior second-order readout from the already rejected
  whole-patch bilinear residual. Primary B-CNN, iSQRT-COV, and GSoP papers
  support local pairwise/covariance descriptors for fine-grained recognition,
  but do not establish that current TRKH patch tokens contain the needed cue.
  I therefore added diagnostic-only
  `trkh.tools.probe_interior_second_order_readiness` instead of another model
  head or smoke.
- Fixed protocol: transformed `crop_bbox` was eroded by `20%` on each side to
  define the core; the rest of the bbox was a boundary ring and outside-bbox
  tokens were context. Frozen 256-D patch tokens used one seed-`20260711`
  Gaussian-QR orthogonal projection to rank `24`, centered covariance, exact
  symmetric matrix square root, compact upper triangle, and normalized mean.
  Seven readouts were predeclared; only `head_core_second_order` was the
  candidate. All used one balanced `C=0.3` five-fold source-grouped OOF
  estimator. There was no rank/seed/erode/normalization/readout sweep, test,
  checkpoint, trainable manifest, or raw-data edit. `py_compile` and four
  focused geometry/covariance/gate tests passed.
- Dry-run validity: the `1024/256` run reproduced the same direct keeper slice
  as the Gray-Edge dry run and produced correct overlays, but one train class
  had only one row. Its apparent validation gain was explicitly ignored; it
  served only to validate geometry, descriptor finiteness, preview, and output
  plumbing.
- Full geometry and representation audit:
  `runs\diagnostic_interior_secondorder_rank24_erode020_full_train_val_20260711`
  covered `9215/2606` rows and `8064` train source groups. Train/validation
  mean core token counts were `48.924/49.384`, p01 `30/32`, with fallback only
  `0.000109/0.000384`. Visual overlays placed green core regions on central
  fruit surface, amber ring on fruit boundary/bbox corners, and blue context
  outside. Core covariance effective rank was `24.827` versus head `6.028`;
  this is not a mask-support or rank-collapse failure.
- Readout result: head OOF/validation macro-class1 was
  `0.93304/0.79901` and `0.88072/0.66667`. All-patch second-order reached only
  `0.87495/0.62241` and `0.83877/0.55621`; core second-order was worse at
  `0.84244/0.52125` and `0.82293/0.53261`. Boundary ring and context were also
  weak (`0.82280/0.52861` and `0.80364/0.50900` validation). Adding core
  second-order to the head reduced OOF to `0.91684/0.74561` and validation to
  `0.86914/0.63607`, below the direct keeper `0.88468/0.68605`.
- Error audit and decision: versus the direct keeper the candidate changed
  `127` rows (`47` corrections, `67` harms, `13` neutral), rescued only `3/33`
  class1 false negatives, broke `24` true class1 predictions, and
  removed/created `27/9` class1 false positives. Class1 precision/recall was
  `0.62987/0.64238`; FN-vs-FP delta AUC was `0.47717`. Reject before a trainable
  module or smoke. Do not sweep nearby rank, projection seed, erode ratio,
  covariance normalization, or readout C on the current frozen representation.
  This closes the report's interior second-order readout as a keeper adapter,
  not its separate proposal for a compact concurrent hybrid trained from
  initialization.
- Cleanup and retention: deleted the biased dry descriptor caches and four
  duplicate process logs after preserving the full `99.226 MB` evidence root
  and expanded README. Manifest
  `runs\cleanup_manifest_20260711_interior_secondorder_dryrun_logs.json`
  records `12` files, `11.762 MB` source, and `11.788 MB` observed free delta.
  `runs\artifact_retention_audit_after_interior_secondorder_precheck_20260711`
  passed over `516` run directories with `blockers=[]`; the dry root is absent
  and all protected keeper/runtime/final artifacts remain present.

## Probe 2026-07-11 - MobileViT-S Concurrent Compact Hybrid Rejected

- Literature and scope: MobileViT combines local convolutional representations
  with global transformer processing and describes transformers as
  convolutions (`https://arxiv.org/abs/2110.02178`; official Apple page
  `https://machinelearning.apple.com/research/vision-transformer`). This is a
  genuinely concurrent hybrid trained from initialization, so the old
  SPT/GPSA/LSA adapter failures did not pre-reject it. The experiment tests the
  exact MobileViT-S/TRKH recipe only, not every compact hybrid.
- Launcher and resource preflight: added
  `scripts\run_trkh_5class_compact_hybrid_scratch.ps1` with no pretrained
  weights, immutable `yolo_f`, image size 256, strict balanced sampling,
  LDAM-focal, EMA `0.995`, teacher-focus-binary `0.015`, keeper-style mild
  object/background augmentation, and mandatory `--skip-final-test`. The
  4,940,837-parameter model sustained about `195 img/s` synthetically at batch
  64 with about `6.43 GB` peak allocation; batch 80 thrashed. Worker-enabled
  full epochs used 120 train batches plus all 2,606 validation rows in roughly
  2.5 minutes.
- Epoch-5 gate: the initial five epochs improved monotonically from validation
  macro/class1 F1 `0.73510/0.38416` to `0.81299/0.50000`. Independent reload
  reproduced the endpoint exactly. Class1 precision/recall was
  `0.39689/0.67550`; confusion had 155 class1 false positives and 49 false
  negatives. This was improving but far below the keeper, so it received only
  a bounded continuation rather than a full run.
- Resume correctness: the launcher now accepts `-ResumeCheckpoint` and passes
  `--resume-use-cli-config` while preserving epoch, optimizer, scheduler,
  scaler, and EMA. Dry-run verified a total-epoch-10 target with the original
  cosine horizon 15. The actual continuation loaded epoch 5, optimizer and
  scheduler, retained EMA update 600, completed epochs 6-10, and never opened
  test.
- Best result and plateau: epoch 8 was best by the predeclared fair macro-F1
  selector. Independent full-validation reload reached macro/class1
  `0.82908/0.54595`, class1 precision/recall `0.46119/0.66887`, versus raw
  keeper `0.88468/0.68605`. Epochs 9/10 failed to improve the selector; epoch
  10 class1 F1 was `0.54408`. Best confusion was
  `[[484,60,2,0,3],[36,101,11,0,3],[3,30,481,16,14],`
  `[0,2,82,628,0],[19,26,10,4,591]]`.
- Full boundary/XAI audit: boundary candidates fell from 654 at epoch 5 to 575
  at epoch 8, but dominant errors remained `1->0`, `0/2/4->1`, and `3->2`.
  On the same 12 explicit class1-boundary cases, continuation corrected only
  one and left eleven wrong. Epoch-8 Grad-CAM foreground/background mass was
  `0.99610/0.00390`; background blur/gray prediction drops were only
  `0.00118/0.00063`, versus center occlusion `0.02536` and object desaturation
  `0.11190`. The model localized the fruit and ignored wide context, but its
  surface/color representation remained class-unsafe.
- Decision: reject this MobileViT-S loss/sampling/augmentation recipe at the
  total-epoch-10 gate. Do not extend it to 15/30 epochs, tune nearby LR/loss
  weights, or open test. EdgeNeXt remains distinct: its paper introduces split
  depth-wise transpose attention that mixes grouped depth-wise multi-scale
  convolution and channel attention (`https://arxiv.org/abs/2206.10589`;
  official repo `https://github.com/mmaaz60/EdgeNeXt`). One fixed X-small
  candidate is permitted; it is not evidence that it will pass.
- Cleanup and retention: compact evidence with config, history, independent
  predictions, boundary manifests, same-case XAI, overlays, summary, and
  SHA-256 manifest is retained at
  `runs\evidence_compacthybrid_mobilevit_s_yolof_scratch_10e_reject_20260711`.
  `runs\cleanup_manifest_20260711_mobilevit_compacthybrid_rejected.json`
  removed nine superseded roots (`268` files, `273.336 MB`) after verification;
  observed free-space gain was `273.953 MB`. No checkpoint, test output,
  dataset file, or protected keeper was touched.

## Probe 2026-07-11 - EdgeNeXt X-Small Rejected Before Continuation

- Rationale and fixed comparison: EdgeNeXt proposes split depth-wise transpose
  attention (SDTA), combining grouped depth-wise multi-scale convolution with
  channel attention (`https://arxiv.org/abs/2206.10589`, official code
  `https://github.com/mmaaz60/EdgeNeXt`). Local timm X-Small has 2,144,769
  parameters, native 256 input, and ImageNet mean/std. The probe changed only
  the MobileViT backbone: same immutable `yolo_f`, batch 64, 120 train batches,
  strict balanced sampler, LDAM-focal, EMA, teacher-focus-binary `0.015`, mild
  augmentation, scheduler horizon 15, full validation, and no test.
- Preflight and utilization: PowerShell parse/dry-run, `py_compile`, and timm
  model tests passed `2/2`. Synthetic training at batch 64 reached about
  `650 img/s` with `1.80 GB` peak allocation; batch 96 also fit, but batch 64
  was retained to preserve the one-variable comparison.
- Result: epochs 1-5 improved validation macro/class1 F1
  `0.66253/0.26250 -> 0.73157/0.38776`. Independent full-validation reload
  reproduced epoch 5 exactly. Class1 precision/recall was only
  `0.31535/0.50331`, producing 165 false positives and 75 false negatives.
  It was far below MobileViT epoch 5 `0.81299/0.50000` and the keeper
  `0.88468/0.68605`; all other class F1 values were also below MobileViT.
- Boundary/XAI: the scan found 1,292 error/boundary candidates, roughly twice
  MobileViT's 654 at the same epoch. Twelve balanced `1->0/2/4` and
  `0/2/4->1` cases showed attention foreground/background `0.99954/0.00046`
  and Grad-CAM `0.90663/0.09337`. Background blur/gray prediction drops were
  `0.00088/0.00406`, versus object desaturation `0.08546`. Visual overlays
  focused on fruit interior/defects. Wide context is again not the main
  failure; the X-Small representation is globally underfit and class-unsafe.
- Decision and scope: reject X-Small before continuation; do not tune its LR,
  loss, sampler, teacher, or epochs. This does not yet reject the
  parameter-matched 5.28M timm EdgeNeXt-Small corresponding to the paper's
  approximately 5.6M result. One unchanged-protocol Small probe is permitted
  to separate architecture from the obvious X-Small capacity deficit.
- Cleanup: evidence is retained at
  `runs\evidence_compacthybrid_edgenext_xsmall_yolof_scratch_5e_reject_20260711`
  with config, curves, full predictions, boundary manifest, 12-case XAI,
  overlays, summary, and SHA-256 manifest. Cleanup
  `runs\cleanup_manifest_20260711_edgenext_xsmall_compacthybrid_rejected.json`
  removed four source roots (`122` files, `71.752 MB`) and observed
  `72.039 MB` free-space gain. Test, datasets, and keepers were untouched.

## Probe 2026-07-11 - EdgeNeXt Small Parameter-Matched Check Rejected

- Purpose: X-Small was only 2.145M parameters, so one final stock EdgeNeXt
  check used timm `edgenext_small` at 5,283,357 parameters, matching
  MobileViT-S and the paper's approximately 5.6M variant. Every data, loss,
  sampler, teacher, augmentation, batch, scheduler, validation, and no-test
  setting was unchanged. Synthetic batch-64 training reached about `449 img/s`
  with `2.72 GB` peak allocation.
- Fast-convergence result: epochs 1-5 improved validation macro/class1
  `0.62795/0.24606 -> 0.72938/0.40302` in the trainer. Independent reload was
  `0.72898/0.40201`, class1 P/R `0.32389/0.52980`. This was no better than
  X-Small macro `0.73157` and remained far below MobileViT epoch-5
  `0.81299/0.50000` and keeper `0.88468/0.68605`. Larger capacity therefore
  did not repair the five-epoch optimization/class-structure deficit.
- Confusion and audit: independent confusion was
  `[[481,59,2,0,7],[58,80,10,0,3],[14,34,461,26,9],`
  `[1,1,142,560,8],[46,73,51,29,451]]`. The boundary pool grew to 1,341 rows,
  above X-Small's 1,292. Twelve balanced class1 errors had Grad-CAM
  foreground/background `0.99967/0.00033`; background blur/gray drops were
  `0.00117/0.00462`, center occlusion `0.00855`, and object desaturation
  `0.07966`. The generic timm attention fallback reported `0.41667` background
  mass, but it is not an EdgeNeXt SDTA export and conflicts with Grad-CAM plus
  perturbation, so it was not used as causal evidence.
- Decision: reject Small before continuation and close stock EdgeNeXt under
  the fast-convergence gate. Do not sweep EdgeNeXt sizes, LR, epochs, loss,
  sampler, teacher, or normalization. MobileViT/EdgeNeXt catalog-backbone
  substitution has now been tested with full audits and is not the next useful
  direction. New work must first demonstrate an uncertainty/surface signal on
  the current keeper or alter supervision/readout around measured confusions.
- Cleanup: compact evidence is retained at
  `runs\evidence_compacthybrid_edgenext_small_yolof_scratch_5e_reject_20260711`.
  Manifest `runs\cleanup_manifest_20260711_edgenext_small_compacthybrid_rejected.json`
  removed four source roots (`122` files, `143.372 MB`) and observed
  `143.656 MB` free-space gain; test, datasets, and keepers were untouched.

## Diagnostic 2026-07-11 - Teacher Uncertainty Critic Gate Rejected

- Motivation and distinction: reports 14/15 propose using external teachers as
  uncertainty critics with per-transition abstention rather than repeating
  global KD or a validation-fitted router. Existing audits measured consensus
  corrections but did not jointly measure continuous uncertainty, risk-coverage,
  train-OOF-to-validation direction transfer, and every critical class1
  transition under one fixed gate. I added diagnostic-only
  `trkh.tools.audit_teacher_uncertainty_critic_readiness` plus two focused tests;
  it requires strict `sample_index`, refuses locked-test rows, and writes no
  trainable manifest or raw-data change.
- Fold-safe protocol: current teacher-focus-binary train predictions define
  student error candidates only. Teacher reliability comes from the existing
  EffV2-S and DINOv2-S train OOF readouts; validation uses the corresponding
  two full-val readouts against the current softboost base. The audit covered
  all `9215/2606` rows. Fixed gates require AUROC `>=0.75` for both class1 FP
  and FN error detection, same-direction FP-vs-FN AUROC `>=0.60`, transition
  support for `0/2/4->1` and `1->0/2/4`, and class-safe action precision. No
  thresholds were swept and test remained closed.
- Uncertainty can rank errors: teacher-mean entropy reached overall-error AUROC
  `0.91833` train and `0.82851` validation. Teacher non-class1 probability
  detected base class1 false positives at AUROC `0.96365/0.79967` and false
  negatives at `0.78103/0.84507` for train/validation. Validation selective
  coverage is therefore meaningful for abstention: at `80.0%` coverage,
  teacher-entropy selection reduced accepted risk to `0.0350` and yielded
  macro/class1 F1 `0.9259/0.7521`. This is not a full-coverage closed-set gain.
- Correction direction does not transfer. The same non-class1 score separated
  FP from FN with AUROC `0.91763` on train OOF candidates but reversed to
  `0.43900` on validation. Train teacher-mean focus suppress/rescue actions
  looked precise (`0.9327/0.8667`), while validation collapsed to
  `0.5588/0.3077`. Direct teacher-mean routing reduced validation class1 F1
  from softboost `0.70303` to `0.66667`; across class1-boundary actions it made
  `46` corrections, `44` harms, and `8` wrong-to-wrong changes.
- Transition evidence explains the failure. Train OOF candidates supplied
  `139/45/7` corrections for `0/2/4->1` and `12/0/1` for `1->0/2/4`.
  Validation supplied `20/8/7` and only `6/1/1`, respectively. Thus the
  teacher remains useful for conservative FP review and selective abstention,
  but has no fold-safe coverage for `1->2`, almost none for `1->4`, and cannot
  decide whether a low class1 score means a false positive or a false negative
  consistently across splits.
- Decision: `smoke_gate_ready=false` in
  `runs\diagnostic_teacher_uncertainty_critic_readiness_20260711`. Do not train
  an uncertainty/evidential critic, per-transition KD head, soft target, router,
  or targeted-margin variant from these exact caches. This rejects the current
  teacher-supervised critic signal, not generic EDL/selective prediction; a
  future ambiguity head needs independently grounded ambiguity labels or a new
  representation whose train-OOF direction remains stable on full validation.

## Probe 2026-07-11 - CoAtNet-Nano Stage-Wise Hybrid Rejected

- No-repeat audit and rationale: CoAtNet vertically stacks convolutional
  MBConv stages before relative-attention stages and was proposed to improve
  generalization across data scales (`https://arxiv.org/abs/2106.04803`). The
  local implementation came from official timm
  (`https://github.com/huggingface/pytorch-image-models`). Existing CMT/LeFF
  sparse block-local adaptation and CCT/PARTICLE-style part-prototype evidence
  were already negative, and local timm exposed no stock CMT, Conformer, or
  CCT model. The historical pretrained CoAtNet-0 fold-00 run used plain CE and
  was not equivalent to the current scratch protocol, so one fixed
  `coatnet_nano_rw_224` candidate was allowed rather than treating that old run
  as a duplicate.
- Implementation and resource gate: the compact-hybrid launcher now validates
  an explicit image size and supports `coatnet_nano_rw_224`; a fixed-native-224
  forward test was added. The model has 14,630,809 parameters, with MBConv
  block counts `3/4` followed by 2D transformer block counts `6/3`. Its relative
  position table rejects image 256, so all runs used native 224. Synthetic
  batch-64 training measured about `274 img/s` and `4.96 GB` peak allocation.
  PowerShell parse/dry-run and the focused timm suite passed `3/3`.
- Controlled protocol: initialized from scratch on immutable `yolo_f`, image
  224, batch 64, 120 train batches per epoch, strict balanced sampling,
  LDAM-focal, EMA `0.995`, teacher-focus-binary `0.015`, the same mild
  augmentation, and cosine horizon 15. Resumes used `last.pt` and restored
  epoch, optimizer, scheduler, scaler, and EMA. Every gate used all 2,606
  validation rows and `--skip-final-test`.
- Learning trajectory: independent epoch-5 validation was macro/class1
  `0.83032/0.55340`, class1 P/R `0.43678/0.75497`. Epoch 10 reached
  `0.85530/0.61957`, P/R `0.52535/0.75497`. The final continuation selected
  epoch 13; epochs 14/15 did not improve the fair macro-F1 selector, so the
  recipe was stopped at its scheduler horizon rather than extended to 30.
- Final independent result: epoch 13 reached accuracy/macro/class1
  `0.90177/0.86161/0.62637`, class1 P/R `0.53521/0.75497`. Confusion was
  `[[480,55,1,1,12],[21,114,13,0,3],[1,22,502,11,8],`
  `[0,4,67,639,2],[9,18,5,3,615]]`. Class1 TP remained exactly `114` from
  epoch 5 through epoch 13; improvement came from reducing FP `147 -> 103 ->
  99`, while 37 FN remained. The result is below raw keeper
  `0.88468/0.68605` and runtime softboost keeper `0.88868/0.70303` by about
  `0.02707/0.07666` at the final comparison.
- Boundary and same-case XAI: candidate counts fell `584 -> 460 -> 422` over
  the three audited checkpoints. The same 12 explicit class1-boundary rows
  stayed `6/12` correct: epoch 10 exchanged one correction for one harm and
  epoch 13 made no net prediction gain. Final feature-map attention
  foreground/background/border mass was `0.87431/0.12569/0.28811`; Grad-CAM
  was `0.08312/0.91688/0.02377`. Background blur/gray prediction drops were
  `-0.00149/-0.00082`, center occlusion `0.00728`, and object desaturation
  `0.10312`. Causal evidence therefore remains object-color dominated; the
  increasingly diffuse CoAtNet Grad-CAM is not proof of a wide-background
  shortcut because perturbations contradict it.
- Decision and scope: reject this exact CoAtNet-Nano recipe as a keeper
  replacement, do not open test, do not extend to 30 epochs, and do not sweep
  nearby LR/loss/sampler/teacher/normalization settings. It is nevertheless
  the strongest stock scratch hybrid in the controlled MobileViT/EdgeNeXt/
  CoAtNet series, so stage-wise MBConv-to-attention is positive architecture
  evidence, not a selected model. Close further catalogue-backbone substitution;
  a next trainable route must be TRKH-native and must first prove a fold-safe
  class1 correction direction rather than repeat a confidence blend/router.
- Preservation and cleanup: verified compact evidence is retained at
  `runs\evidence_compacthybrid_coatnet_nano_yolof_scratch_15e_reject_20260711`
  with 320 files, configs/histories, three full-val prediction sets, three
  boundary manifests, three same-case XAI audits, logs, and a SHA-256 manifest;
  it contains no `.pt` or test output. Cleanup manifest
  `runs\cleanup_manifest_20260711_coatnet_nano_compacthybrid_rejected.json`
  removed 12 source roots plus 14 process logs (`1072.518 MB` source) and
  observed `1073.395 MB` free-space gain. Datasets and all protected keepers
  remained present.

## Diagnostic 2026-07-11 - CoAtNet/Softboost Complementarity Is Not Routable

- A strict `sample_index` validation-only comparison used retained CoAtNet
  epoch-13 predictions against current softboost over all 2,606 rows. This is
  label-using diagnosis only; no threshold, train manifest, or test output was
  created. The artifact is
  `runs\diagnostic_coatnet_nano_softboost_complementarity_val_20260711`.
- CoAtNet corrected 12 of 35 softboost class1 false negatives and correctly
  blocked 19 of 63 class1 false positives. However, it still predicted class1
  on 43 of those 63 false positives, so simple disagreement is mostly unsafe.
  The non-deployable focus-error oracle reached macro/class1
  `0.91221/0.79257`; the all-error oracle reached `0.92009/0.79257`. Even
  perfect validation-label routing between these two models remains far below
  the requested class-wise 0.98 target.
- Decision: do not build a CoAtNet/softboost blend, confidence gate, KD target,
  or router. The signal has complementarity but no fold-safe correction
  direction, agreeing with the earlier teacher-uncertainty critic rejection.

## Probe 2026-07-11 - TRKH Stage-Wise MBConv Stem Rejected

- Motivation and exact distinction: stock CoAtNet was the strongest scratch
  catalogue hybrid, so the only permitted TRKH-native architecture check took
  its verified inductive bias at the stage boundary rather than adding another
  head. New default-off `stem_architecture=coatnet_mbconv` uses the scratch timm
  CoAtNet-Nano stem and first two MBConv stages, preserves stride 8, projects
  to 256 channels, then leaves TRKH patch embedding, token pruning, branch
  tokens, bbox prior, pairwise head, and eight transformer blocks unchanged.
  Legacy `conv_pool` remains the default and retains its state-dict schema.
- Implementation verification: added ModelConfig/train/V8 wiring and fixed
  wrapper `scripts\run_trkh_5class_stagewise_mbconv_scratch.ps1`. New tests
  cover stride/shape, checkpoint rebuild, no-pretrain construction, default
  legacy schema, and invalid mode; compile, PowerShell parse/dry-run, focused
  stem/timm tests passed `11/11`. Dry-run confirmed immutable `yolo_f`, scratch,
  effective batch 64, scheduler horizon 15, and forced no-test behavior.
- Resource gate: the full current config increased from 7,245,590 to 8,242,518
  parameters. Synthetic BF16 batch-32 throughput fell `221.96 -> 140.27 img/s`
  and peak allocation rose `2.53 -> 4.00 GB`. A real 20-batch smoke completed
  full validation and a 132-file architecture trace; peak allocated/reserved
  was `6.88/7.07 GB`. Its early metric was intentionally not used for model
  selection. Smoke XAI had attention/Grad-CAM foreground `0.90951/0.97706`,
  background blur/gray drops `0.00623/0.00501`, and object-desaturation drop
  `0.03729`, so the pipeline and causal focus were sane.
- Performance interruption: the first fixed-recipe probe completed epoch 1 at
  macro/class1 `0.76663/0.36196`. When attention crop/drop activated at epoch
  2, batch time rose from about `0.5-0.8s` to `6-8s` and reserved VRAM reached
  `8.154 GB`. The entire process tree was stopped immediately, and
  `interruption_status.json` marks it diagnostic-only. It is not a failed
  metric result and was never used for selection.
- Efficient fixed gate: because attention-view was also unavailable to stock
  timm comparators, the architecture-only five-epoch gate disabled that loss
  while retaining LDAM-focal, balanced sampling, EMA, teacher-focus-binary,
  metric learning, bbox prior, boundary dropout, augmentation, 120 batches,
  full validation, and no test. Epoch macro/class1 progressed
  `0.76953/0.39024`, `0.79227/0.45953`, `0.79670/0.50246`,
  `0.81322/0.48675`, `0.81071/0.48227`. The fair selector chose epoch 3;
  later macro gains came with worse class1 precision.
- Independent result: reload reached accuracy/macro/class1
  `0.84421/0.79725/0.50495`, class1 P/R `0.40316/0.67550`, with confusion
  `[[460,73,2,0,14],[37,102,11,0,1],[1,32,466,40,5],`
  `[0,2,86,623,1],[33,44,15,9,549]]`. There were 151 class1 FP and 49 FN.
  This is below stock CoAtNet epoch 5 `0.83032/0.55340`, below its continuation,
  and far below the raw keeper `0.88468/0.68605`; 709 boundary candidates also
  exceed stock CoAtNet's 584.
- Final XAI and visual review: attention foreground/background/border was
  `0.75762/0.24238/0.29461`, while grad-rollout, Grad-CAM, and rollout
  foreground were `0.94547/0.99183/0.96235`. Background blur/gray drops were
  only `0.00124/0.00131`, versus center occlusion `0.01695` and object
  desaturation `0.14349`. Overlays showed correct focus on fruit color, spots,
  and damage but diffuse generic attention near padding/adjacent objects. The
  bottleneck is surface/maturity class structure, not wide context.
- Decision: reject at the five-epoch fast-convergence gate; do not extend to
  10/15/30 epochs or sweep batch, LR, loss, attention schedule, MBConv depth,
  width, or nearby stage subsets. Stock CoAtNet's advantage does not transfer
  with its early MBConv stem alone; its later relative-attention hierarchy is
  part of the representation. Further partial CoAtNet transplants would repeat
  this negative result rather than provide a new signal.
- Preservation and cleanup: compact evidence with smoke, interruption record,
  configs/history, architecture trace, independent predictions, boundary
  rows, two XAI audits, visuals, logs, benchmark, summary, and SHA-256 manifest
  is at `runs\evidence_stagewise_mbconv_stem_scratch_5e_reject_20260711` (437
  files, 32.888 MB, no checkpoint/test). Cleanup
  `runs\cleanup_manifest_20260711_stagewise_mbconv_stem_rejected.json` removed
  seven source roots and eight logs (`634.046 MB` source), observing
  `635.453 MB` free-space gain without touching datasets or keepers.
- Retention refresh
  `runs\artifact_retention_audit_after_stagewise_mbconv_cleanup_20260711`
  passed over 526 run directories with all 14 protected checks present,
  compacted originals absent, and `blockers=[]`.

## Diagnostic 2026-07-11 - Multi-Model Information Ceiling and Residual Neighbors

- Added strict validation-only tools
  `trkh.tools.audit_validation_information_ceiling` and
  `trkh.tools.audit_validation_residual_neighbors`, with five focused tests.
  They require aligned keys/targets/class order, reject test-derived inputs,
  bootstrap by source group, and compare exact-label and class1-vs-rest
  oracles across 17 models.
- Evidence root:
  `runs\diagnostic_validation_information_ceiling_multimodel_20260711`.
  It covers current TRKH raw/softboost/GPSA/SPT, scratch hybrid candidates,
  AIDT/top5/DINO/EfficientNetV2, and five individual pretrained experts over
  the same 2,606 validation objects.
- The softboost class1 state is TP/FP/FN `116/63/35`, F1 `0.70303`. Reaching
  F1 0.98 would require at least 92 of its 98 class1 binary errors to be
  corrected: all 35 FN plus 57 FP, or `93.88%` of the error budget.
- The class1 binary oracle is only `0.77273` for current TRKH members,
  `0.89968` for scratch hybrids, `0.94949` for pretrained members, and
  `0.97351` even when all 17 models may use validation labels. Four FN and four
  FP remain unanimous. The exact five-class all-model oracle reaches macro
  `0.98395` but class1 only `0.95765`; 29 rows are wrong for every model.
- The eight unanimous class1-binary residuals are `Image_3217`, `Image_546`,
  `Image_8344`, `Image_8346`, `Image_3818`, `Image_4298`, `Image_532`, and
  `Image_7579`. Direct image/sequence review found foreground-dominant fruit,
  adjacent frames with conflicting 0/1/2/3 labels, and no wide-background
  explanation. Three independent embedding spaces also gave weak target-label
  neighbor support; 52/70 nearby train/val sequence pairs were cross-label.
- Interpretation: `F1 > 0.98` for every class is not supported by the empirical
  information shared by current models. This is not a formal Bayes bound, but
  it is a stronger falsification than another single-model score. Future work
  must report ambiguity/selective risk honestly and cannot claim that more
  background filtering alone resolves these rows. Do not use `Image_N`
  sequence metadata as a predictive feature.

## Smoke 2026-07-11 - Deep Abstaining Classifier Rejected

- Literature gate used the original Deep Abstaining Classifier objective
  (`https://proceedings.mlr.press/v97/thulasidasan19a.html`) and the
  validation-only selective-risk framing from SelectiveNet
  (`https://proceedings.mlr.press/v97/geifman19a.html`). Penalty `1.30` was
  chosen between measured correct/wrong keeper CE means, not guessed from a
  sweep. The mature keeper supplies stage 1; DAC is enabled only for stage 2.
- Implemented a default-off scalar abstention head, stable published loss,
  exact logit-scale-independent q initialization, strict resume-extension
  support, V8/wrapper wiring, six history fields, evaluator CSV export, and
  `audit_deep_abstention_risk_coverage`. Focused tests passed `8/8`; preflight
  and dry-run used immutable `yolo_f`, full val, and `--skip-final-test`.
- Fixed smoke: `w=0.15`, `alpha=1.30`, q0 `0.01`, dropout `0.05`, 60 batches,
  one epoch, seed 42. The first launch stopped before training on the missing
  strict-extension whitelist; after a narrow whitelist fix, the corrected run
  completed in 98 seconds with no OOM/non-finite step.
- Independent full-val result was macro/class1 `0.87749/0.65487`, class1 P/R
  `0.59043/0.73510`, below raw keeper `0.88468/0.68605`. Dominant errors were
  `0->1=51`, `1->0=20`, `1->2=16`, `4->1=17`, and `3->2=45`.
- q was almost constant: mean/max `0.009133/0.010096`; correct/wrong means
  `0.009137/0.009083`. Error AUROC/AP was `0.42040/0.07065`; dropping the top
  10% q increased risk instead of reducing it. The eight unanimous residuals
  had mean q `0.009220`, not enough for useful selection.
- Full 24-case XAI found attention/grad-rollout/Grad-CAM/rollout foreground
  `0.84586/0.93574/0.79486/0.88934`, object-desaturation drop `0.13602`, and
  background blur/gray `0.00859/0.01015`. The loss did not add a boundary cue;
  it weakened closed-set discrimination and amplified color/border dependence.
- Decision: reject before a 120-batch probe. Do not sweep nearby DAC blend,
  penalty, q0, dropout, batch count, or threshold on this keeper. Compact
  evidence is retained at
  `runs\evidence_deep_abstention_w015_a130_reject_20260711`; no rejected
  checkpoint/test output is retained.

## Probe 2026-07-11 - Concurrent TRKH LeFF From Scratch Rejected

- Literature/no-repeat gate: LocalViT (`https://arxiv.org/abs/2104.05707`),
  CeiT/LeFF (`https://arxiv.org/abs/2103.11816`), and CvT
  (`https://arxiv.org/abs/2103.15808`) support injecting spatial locality into
  transformer feed-forward paths. This is distinct from the previously failed
  zero-initialized external block-local adapter: a true LeFF was placed inside
  layers 1-4 and trained concurrently from initialization.
- Implementation: default-off `LocallyEnhancedFeedForward` uses linear
  expansion, GELU/dropout, hidden-channel depthwise 3x3 convolution plus
  BatchNorm/GELU, and linear projection. Prefix tokens bypass the convolution;
  dense and pruned grids use deterministic scatter/convolve/gather with
  `patch_indices`. Config/train/V8/trace/wrapper wiring is complete in
  `scripts\run_trkh_5class_leff_scratch.ps1`; the disabled legacy state schema
  remains unchanged. Compile, parse, dry-run, gradient, pruning, selection,
  round-trip, and related regression tests passed `13/13`.
- Resource gate: parameters increased only `7,245,590 -> 7,290,646` (`0.62%`).
  Synthetic BF16 batch-32 throughput changed `222.09 -> 206.92 img/s`; peak
  allocation/reserve changed `2.533/3.264 -> 2.756/3.389 GB`. The real probe
  peaked at `4.734/5.594 GB`, completed in `654.5 s`, and retained a null test
  summary.
- Pipeline smoke: 20 batches/one epoch/full val/no test reached macro/class1
  `0.54277/0.26479`. It was not used for model selection. Twelve-case XAI had
  attention/grad-rollout/Grad-CAM/rollout foreground
  `0.9540/0.9447/0.9597/0.9487`; background blur/gray drops
  `-0.0031/0.0028` versus object desaturation `0.2176`, so the architecture and
  causal object focus earned the fixed five-epoch gate.
- Five-epoch trajectory (macro/class1) was `0.73415/0.35829`,
  `0.75446/0.40465`, `0.78447/0.46734`, `0.78484/0.46224`, and
  `0.79170/0.47465`. Epoch 5 was selected; independent reload exactly matched
  accuracy/macro/class1 `0.83960/0.79170/0.47465`, class1 P/R
  `0.36396/0.68212`. Confusion was
  `[[485,62,2,0,0],[35,103,13,0,0],[1,42,454,43,4],`
  `[0,4,84,622,2],[32,72,14,8,524]]`: 103 TP, 180 FP, and 48 FN for class1.
- Comparative gate: LeFF is below stage-wise MBConv macro/class1
  `0.79725/0.50495`, stock CoAtNet-Nano epoch 5 `0.83032/0.55340`, and the raw
  keeper `0.88468/0.68605`. It produced 784 boundary candidates, worse than
  stage-wise MBConv's 709 and CoAtNet's 584.
- Final XAI and visual review: attention/grad-rollout/Grad-CAM/rollout
  foreground was `0.9640/0.9379/0.8347/0.8658`; background blur/gray drops
  were `0.0060/0.0072`, center occlusion `0.0153`, and object desaturation
  `0.1262`. Direct `4->1`, `0->1`, `1->2`, and `1->0` overlays focus on fruit
  color, spots, wrinkles, contour, and stem/edge. Failure is class-unsafe
  maturity/surface discrimination, not wide-background dependence.
- Decision: reject the exact layers-1-4/kernel-3 LeFF route at the fixed
  fast-convergence gate. Do not extend to 10/15/30 epochs or sweep nearby
  layer subsets, kernels, width, LR, loss, sampler, teacher, or normalization.
  Keep the default-off implementation for ablation only; do not replace the
  keeper or open test.
- Preservation/cleanup: compact evidence is at
  `runs\evidence_trkh_leff1234_scratch_5e_reject_20260711` (591 files,
  60.839 MB, no checkpoint/test, manifest SHA-256
  `7ab8b7315eb7fa1946bfb7d5a01cbbe3717b4bc83dde49923ce4d35fd8c124aa`).
  `runs\cleanup_manifest_20260711_trkh_leff1234_scratch_rejected.json`
  removed six source roots/592 files (`395.740 MB` source), observing
  `396.520 MB` free-space gain. Retention
  `runs\artifact_retention_audit_after_trkh_leff_cleanup_20260711` passed over
  531 directories with all protected artifacts present and `blockers=[]`.

## Probe 2026-07-11 - Persistent Concurrent Local-Global Coupling Rejected

- Literature and no-repeat gate: Visual Conformer (arXiv `2105.03889`), CMT
  (arXiv `2107.06263`), and CoaT (ICCV 2021, arXiv `2104.06399`) motivate
  concurrent local and global states with explicit feature exchange. This is
  distinct from LeFF and the old single CNN branch token: TRKH retained a
  spatial 64-channel, 16x16 local map through layers 1-8 and coupled it
  bidirectionally with patch tokens after every selected transformer block.
- Implementation: new default-off `ConcurrentLocalInitializer` and
  `ConcurrentLocalGlobalCoupling` perform sparse-safe token-to-local scatter,
  depthwise bottleneck local updates, and local-to-token gather. CLS/register/
  branch prefix tokens are unchanged, layer-2/layer-5 pruning is supported,
  and the trace exports `03c_concurrent_local_activation.png` plus per-layer
  local/token norms and coupling scales. Config/train/V8/DETR/wrapper wiring is
  complete in `scripts\run_trkh_5class_concurrent_local_global_scratch.ps1`.
- Verification/resource gate: compile, PowerShell parse/dry-run/preflight,
  dense/pruned gradients, prefix preservation, default schema, checkpoint
  round-trip, trace, and related regressions passed `17/17`. Parameters rose
  `7,245,590 -> 7,670,190` (`+5.86%`); synthetic BF16 batch-32 throughput fell
  `221.13 -> 198.05 img/s`, while peak allocation/reserve rose
  `2.501/3.330 -> 2.770/3.480 GB`. The real probe peaked near
  `4.808/5.412 GB`, completed in `672.7 s`, and never evaluated test.
- Pipeline smoke: 20 batches/one epoch/full validation reached macro/class1
  `0.54090/0.22511`. XAI attention/grad-rollout/Grad-CAM/rollout foreground
  mass was `0.8577/0.9445/0.9738/0.9631`; background blur/gray drops were
  `0.0023/0.0033`, versus object desaturation `0.1815`. Finite telemetry and
  moving coupling scales established that the new branch was trainable.
- Five-epoch trajectory (macro/class1) was `0.74000/0.31939`,
  `0.75945/0.41686`, `0.75721/0.43320`, `0.78454/0.44898`, and
  `0.78784/0.45275`. The fair macro selector retained epoch 4; independent
  reload reproduced accuracy/macro/class1 `0.83423/0.78454/0.44898`, class1
  P/R `0.34138/0.65563`. Confusion was
  `[[466,76,2,0,5],[39,99,12,0,1],[1,42,474,23,4],`
  `[0,4,99,601,8],[24,69,18,5,534]]`: 99 TP, 191 FP, and 52 FN for class1.
- Comparative/XAI gate: 802 boundary candidates are worse than LeFF (784),
  stage-wise MBConv (709), and stock CoAtNet-Nano epoch 5 (584). Final
  attention/grad-rollout/Grad-CAM/rollout foreground was
  `0.9817/0.9401/0.7841/0.8356`; background blur/gray drops stayed only
  `0.0012/0.0009`, center occlusion was `0.0174`, and object desaturation
  `0.1539`. Local-map and overlay review confirmed real fruit-interior focus,
  but no class-safe color/spot/wrinkle/damage boundary.
- Decision: reject exact layers 1-8, dim 64, kernel 3 at the fixed scratch
  fast-convergence gate. Do not extend to 10/15/30 epochs or sweep nearby
  layer subsets, local width, kernels, scales, LR, loss, sampler, teacher, or
  normalization. Persistent CNN-token exchange is functioning but amplifies
  class1 false positives and does not solve the representation ambiguity.
- Preservation/cleanup: compact evidence is retained at
  `runs\evidence_trkh_concurrent_localglobal_scratch_5e_reject_20260711`
  (601 files, 58.949 MB, no checkpoint/test/raw data; manifest SHA-256
  `796c7403112e7cd6c88b46db46c35abe564d708f54129514aa0012bf8b1080de`).
  Cleanup manifest
  `runs\cleanup_manifest_20260711_trkh_concurrent_localglobal_rejected.json`
  removed six source roots/602 files (`411.945 MB` source) and observed
  `413.340 MB` free-space gain. Retention
  `runs\artifact_retention_audit_after_trkh_concurrent_localglobal_cleanup_20260711`
  passed over 533 run directories with `blockers=[]`.

## Smoke 2026-07-11 - Fixed LDR-KL Stage-2 Rejected

- Research triage used the two new deep-research reports plus primary sources.
  Generic EDL/TEDL mainly targets uncertainty quality and does not establish a
  closed-set class1 F1 mechanism; this overlaps the already failed DAC,
  ordinal, soft-target, and uncertainty-critic routes. CrossViT/Dual
  Cross-Attention was also gated by measured paired-view complementarity rather
  than implemented from a diagram.
- A reproducible full-validation paired-view audit compared the keeper on
  `yolo_f` and the corresponding `class_f` crops. The scores were
  `0.884073/0.684058` and `0.880428/0.668605` macro/class1. Only 28 predictions
  differed; the crop view rescued `0/33` context-view class1 FN and corrected
  only `2/76` class1 FP. Even the label-assisted exact oracle reached only
  `0.888270/0.688047`. Direct cross-view attention/fusion is therefore rejected
  before GPU training because the two views supply almost no complementary
  class1 decisions.
- Implemented the fixed LDR-KL objective from Zhu et al., ICML 2023
  (`https://proceedings.mlr.press/v202/zhu23o.html`) following the official
  Apache-2.0 code (`https://github.com/Optimization-AI/ICML2023_LDR`). The new
  default-off loss supports hard/soft labels, smoothing, class multipliers,
  per-sample output, config/CLI/V8 wiring, and the dedicated wrapper
  `scripts\run_trkh_5class_ldr_kl_stage2.ps1`. Compile, launcher parse,
  dry-run/preflight, and focused tests passed (`16/16`).
- Fixed stage-2 smoke used margin `2.0`, temperature `1.0`, keeper resume,
  LR `8e-5`, 60 batches, one epoch, scheduler horizon 15, full immutable
  `yolo_f/val=2606`, and no test. It completed in about 135 seconds with a
  finite active loss and reached accuracy/macro/class1
  `0.917114/0.880233/0.672515`, class1 P/R `0.602094/0.761589`. This is below
  raw keeper `0.884675/0.686047` and runtime keeper `0.888675/0.703030`.
- Confusion was
  `[[485,49,3,0,12],[18,115,13,0,5],[5,10,514,8,7],`
  `[0,3,53,653,3],[9,14,3,1,623]]`. Dominant `0->1=49`, `1->0=18`,
  `1->2=13`, and `4->1=14` show the same unsafe class1 tradeoff rather than a
  new separability signal.
- Twelve-case XAI gave attention/grad-rollout/Grad-CAM/rollout foreground mass
  `0.8804/0.9486/0.8438/0.9129` and border mass
  `0.1730/0.1595/0.2992/0.2460`. Background blur/gray drops were
  `0.0133/0.0129`, center occlusion `0.0499`, and object desaturation `0.2058`;
  9/12 cases were object-color sensitive. Visual review found scattered spot
  fixation on class1 FN and broad lesion/surface fixation on class1 FP, with
  some border/background activation. LDR-KL changed confidence/boundaries but
  did not create a class-safe maturity/surface representation.
- Decision: reject the exact fixed margin-2/temperature-1 LDR-KL stage-2 route
  before a longer probe. Do not sweep nearby margin, temperature, LR, sampler,
  teacher, batch count, or epochs. The mathematically distinct adaptive LDR
  variant may proceed only through a leakage-safe uncertainty-readiness audit.
- Preservation/cleanup: compact evidence is under
  `runs\evidence_ldrkl_m200_t100_stage2_smoke_reject_20260711` (38 payload
  files, about 2.03 MB, SHA-256
  `17b4ad7be22980264334280c0c788c5fe5aed115b6182c851fb7cc0db7778a6c`).
  `runs\cleanup_manifest_20260711_ldrkl_and_view_precheck_rejected.json`
  removed 293 source files and observed `193.090 MB` free-space gain. Retention
  `runs\artifact_retention_audit_after_ldrkl_cleanup_20260711` passed over 535
  directories with `blockers=[]`; no checkpoint, raw data, or test output was
  retained.

## Diagnostic 2026-07-11 - Adaptive LDR Readiness Rejected

- The ICML 2023 paper and official code were rechecked before implementation.
  In the official parameterization, the one-step update is
  `lambda_i/lambda_ref = 1 - KL(p_i || uniform)/(alpha*log(K))`; therefore its
  ordering is exactly a monotone transform of predictive entropy. Larger
  lambda is intended for uncertain/noisy rows, while smaller lambda gives
  confident/clean rows a stronger large-margin update.
- Added diagnostic-only `trkh.tools.audit_adaptive_ldr_readiness` and focused
  tests. It reconstructs logits from aligned probabilities up to the irrelevant
  additive constant, reproduces the official detached-lambda update, measures
  target-logit gradient strength, bootstraps AUC by source image, rejects test
  rows, and writes no trainable manifest or raw-data change. Compile plus the
  adaptive/existing uncertainty suite passed `4/4`.
- The fixed no-train gate used the current keeper's 9,215 in-sample train rows
  and all 2,606 validation rows, `lambda_ref=1`, safety-positive `alpha=2`,
  margin `1`, and 500 source-group bootstrap replicates. Train predictions are
  used only to inspect the proposed stage-2 update; validation is the primary
  generalization gate.
- Overall-error lambda AUROC was `0.76325` train and only `0.71463` validation
  (validation 95% group-bootstrap CI `0.6764-0.7584`). Class1-FP detection was
  `0.78912/0.62182`, while class1-FN detection was `0.92376/0.84691`. Thus the
  uncertainty direction is strongly asymmetric in the wrong way for the
  current problem.
- With positive class defined as FP, FP-versus-FN lambda AUROC was only
  `0.31748` train and `0.24637` validation. Mean validation lambda ratio was
  `0.98437` for class1 TP, `0.99376` for class1 FN, and `0.98836` for class1
  FP. ALDR would robustify/downweight the true-class1 FN correction path more
  than the FP path that needs suppression.
- Adaptivity is also numerically negligible because the keeper probabilities
  are high-entropy: validation lambda p90-p10 span was only `0.01619` against
  the predeclared `0.05` minimum, and the median absolute target-gradient
  change from fixed LDR was only `0.000715` against `0.01`. Train values were
  similarly `0.01442` and `0.000705`. This is effectively the already rejected
  fixed LDR objective, not a materially different stage-2 optimizer.
- Decision: `smoke_gate_ready=false` in
  `runs\diagnostic_adaptive_ldr_readiness_keeper_20260711`. Do not implement,
  smoke, or sweep adaptive LDR alpha/lambda/margin on the current keeper. The
  route fails both class1 FP safety and minimum-effect gates before GPU work;
  no duplicate image XAI is required because no model or prediction changed.

## Diagnostic 2026-07-11 - Class-Noise Transition Correction Rejected

- Primary-source gate: Patrini et al. CVPR 2017 forward correction multiplies
  latent clean predictions by a class-conditional corruption matrix, while the
  ICLR 2017 noise-adaptation layer learns the same latent-label communication
  channel. Both require the transition process to be identifiable; a model
  confusion matrix is not automatically a label-noise matrix.
- Added diagnostic-only `trkh.tools.audit_class_noise_transition_readiness`
  with two-expert train-OOF/validation alignment, locked-test refusal,
  row-stochastic proxy matrices `P(observed label | expert latent class)`,
  condition/support checks, and cross-expert/split stability gates. Focused
  transition plus uncertainty tests passed `6/6`; no model, manifest, or raw
  data changed.
- Inputs were the existing fold-safe EfficientNetV2-S and DINOv2-S train OOF
  readouts plus their aligned full 2,606-row validation readouts. Expert
  predictions are explicitly treated only as latent-clean proxies; the audit
  does not claim they reveal true clean labels.
- EfficientNetV2-S implied class1 diagonal `0.9764` on train OOF but only
  `0.6947` on validation; its train-validation class1 transition-row L1 shift
  was `0.5636`. DINOv2-S was more split-stable but implied class1 diagonal only
  `0.4628/0.4486`, so more than half of its latent-class1 rows map to observed
  non-class1 labels.
- The two experts disagreed on the class1 transition row by L1 `1.0272` on
  train and `0.5311` on validation, against the predeclared `0.15` maximum.
  All matrices were numerically invertible, but invertibility does not rescue
  the missing semantic identification.
- Decision: `smoke_gate_ready=false` in
  `runs\diagnostic_class_noise_transition_readiness_20260711`. Do not
  implement/sweep forward/backward correction, a learnable noise-adaptation
  layer, or a pseudo-clean transition matrix from these experts. It would turn
  expert-specific class1 mistakes into a presumed corruption process and can
  erase true class1 recall. Reopen only with independently validated clean
  anchors or stable fold-safe proxy matrices, neither of which exists here.

## Probe 2026-07-11 - EfficientFormerV2-S0 Scratch Hybrid Rejected

- Research gate: the primary EfficientFormerV2 ICCV 2023 paper
  (`https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Rethinking_Vision_Transformers_for_MobileNet_Size_and_Speed_ICCV_2023_paper.pdf`)
  and official Snap implementation
  (`https://github.com/snap-research/EfficientFormer`) were checked before the
  run. This candidate combines early local blocks with later multi-scale
  attention and is distinct from the rejected MobileViT, EdgeNeXt, CoAtNet,
  LeFF, and persistent TRKH coupling mechanisms. Local `timm 1.0.27` provides
  `efficientformerv2_s0`.
- Resource/pipeline gate: the model has 3,248,026 parameters. A synthetic BF16
  batch-64 benchmark at its native 224 resolution reached about 475.75 img/s
  with peak allocation/reserve `2.002/2.189 GB`. The fixed EfficientFormer
  attention grid rejects 256 input, so 224 was recorded as the only valid size
  rather than treated as a tunable sweep. A 20-batch one-epoch smoke completed
  without OOM/nonfinite values and mandatory XAI established a working
  pipeline.
- The predeclared five-epoch gate required both macro/class1 F1 to match
  CoAtNet-Nano epoch 5 (`0.830315/0.553398`). The 120-batch trajectory was
  `0.761111/0.453826`, `0.784701/0.488889`, `0.798224/0.510158`,
  `0.808905/0.513636`, and `0.807672/0.498896`. The fair selector retained
  epoch 3; epoch 4 had the highest raw macro but still failed both gates, and
  epoch 5 regressed class1.
- Independent full-validation reload of selected `best.pt` reproduced
  accuracy/macro/class1 `0.843438/0.799147/0.510158`, class1 P/R
  `0.386986/0.748344`. Confusion was
  `[[455,74,2,0,18],[27,113,9,0,2],[0,50,444,42,8],`
  `[0,3,81,627,1],[21,52,10,8,559]]`: 113 TP, 176 FP, and 38 FN for class1.
  `test_summary` remained null.
- Boundary/XAI audit: 853 validation boundary candidates are worse than
  CoAtNet epoch 5 at 584. On the same 12 locked cases the best checkpoint was
  correct on only 3. Attention/Grad-CAM foreground mass was
  `0.88101/0.88577`, background blur/gray prediction drops were
  `-0.00376/-0.00481`, center occlusion `0.02055`, and object desaturation
  `0.12554`. Direct visual review found fruit-color/lesion evidence mixed with
  stems, gloves, padding, and object boundaries. The failure is class-unsafe
  surface/boundary discrimination, not broad background dependence.
- Decision: reject this exact EfficientFormerV2-S0 scratch recipe. Do not
  continue to 10/15/30 epochs or sweep neighboring image size, LR, loss,
  sampler, teacher, normalization, or EfficientFormer variant. It is a valid
  compact negative architecture ablation but is below CoAtNet and both TRKH
  keeper references.
- Preservation/cleanup: compact evidence is at
  `runs\evidence_compacthybrid_efficientformerv2_s0_yolof_scratch_5e_reject_20260711`
  (52 hashed payload files, 4.657 MiB, no checkpoint/test/raw data; manifest
  SHA-256 `01fc0cfac225f93bcd9bb60f66df1b47131fd67dcccfe9cf5aa17b867a16c621`).
  Cleanup manifest
  `runs\cleanup_manifest_20260711_efficientformerv2_s0_compacthybrid_rejected.json`
  removed six source roots/238 files (`191.934 MiB` source) and observed
  `192.508 MiB` free-space gain. Retention
  `runs\artifact_retention_audit_after_efficientformerv2_s0_cleanup_20260711`
  passed over 539 directories with `blockers=[]` and all six originals absent.

## Infrastructure 2026-07-11 - VS Code Full Pipeline and Deployment Commands

- Failure analysis: the supplied VS Code screenshot reproduced the old
  PowerShell 5 failure mode where the launcher used
  `2>&1 | ForEach-Object`. Python logging legitimately writes INFO records to
  stderr; merging native stderr into a PowerShell object pipeline converted
  those records into terminating `NativeCommandError` values under the
  launcher's error policy. This was an orchestration failure, not a training
  exception.
- Added `scripts\run_trkh_current_best_full_pipeline.ps1`. It contains the
  selected keeper recipe and exposes one physical command for VS Code. Native
  processes are called directly under a temporary nonterminating native-log
  policy, every `$LASTEXITCODE` is checked, and no stderr object pipeline is
  used. Training always receives `SkipFinalTest`; explicit final test runs only
  after the train checkpoint, null trainer test summary, architecture trace,
  and independent validation evaluations pass.
- The post-train suite is candidate-specific and self-contained under
  `<run>\final_audit`: raw plus softboost001 validation evaluation, prediction
  forensics, confusion audits, boundary reviews, robustness, raw/verifier-aware
  XAI and transition summaries; then optional explicit raw/softboost final-test
  evaluation, test forensics/confusions/boundary/XAI, and artifact retention.
  A successful run writes `runs\latest_full_pipeline.json` for deployment
  wrappers. Test artifacts remain final-report evidence and must not feed back
  into model selection.
- Final-test promotion compares like with like: the independent raw evaluator
  must match both keeper reload macro/class1 `0.882925/0.678261`. The higher
  `0.884675/0.686047` pair is retained as the in-training anchor, not mixed into
  the reload gate. A failed gate still completes validation audits, writes
  `validation_gate_rejected`, skips test, and leaves the deploy pointer intact.
- Added `scripts\run_trkh_export_engine.ps1` for separate ONNX plus TensorRT
  FP16 export/benchmark and `scripts\run_trkh_test_video.ps1` for a separate
  video run. Both resolve the latest successful pipeline by default, accept an
  explicit checkpoint/run override, avoid stderr piping, and fail on nonzero
  exit codes. Export preflight found TensorRT `10.7.0`, CUDA available, and
  `C:\TensorRT\TensorRT-10.7.0.23\bin\trtexec.exe`.
- Fixed TensorRT video inference for classification-only TRKH engines. The old
  `stream_infer_trt.py` required bbox output and aborted on the current model.
  It now shares the PyTorch classification top-k/confidence result builder,
  temporal smoothing, drift input, and overlay path while preserving the
  existing detection-engine branch. Compile and focused regression tests pass
  (`6/6`).
- Replaced the fragile multiline command block in
  `docs\TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt` with absolute,
  one-line PowerShell commands for full pipeline, engine export, TensorRT
  video, and PyTorch video fallback. PowerShell AST parsing passed for all
  three wrappers. Full-pipeline preflight passed with the exact selected
  recipe, immutable `yolo_f`, current checkpoint/verifier, 30-epoch horizon,
  and final-test flag; no train or test was executed during this infrastructure
  verification.
- Update rule: a future version changes these commands only after it wins the
  locked validation gate and is actually selected. Update wrapper defaults,
  command-file metrics, journal/TODO/skill, then rerun PowerShell parse,
  full-pipeline preflight, engine preflight, and focused tests together. Never
  promote a smoke-only, oracle, or test-tuned gain.

### 2026-07-11 - Deep-research pack refresh and promotion invariant

- Refreshed `docs\TRKH_DEEP_RESEARCH_REFERENCE_PACK_20260711.zip` staging to
  `266` payloads (`~16.82 MiB` uncompressed before final documentation sync),
  including `63` selected follow-up files. The additions cover reports 14/15,
  information-ceiling diagnostics, and the closure evidence for CoAtNet-Nano,
  EfficientFormerV2-S0, LeFF, concurrent local-global coupling, LDR/ALDR, and
  class-noise transition readiness.
- The pack excludes checkpoints, raw datasets, ONNX files, and TensorRT engines.
  Every payload has a SHA-256 entry in `_MANIFEST.json`; a companion archive hash
  is regenerated only after ZIP stream verification succeeds.
- Command promotion is now an invariant, not a reminder: whenever a genuinely
  better checkpoint passes independent locked-validation selection, synchronize
  the full-pipeline wrapper, one-line VS Code command file, reported metrics,
  TODO, journal, and TRKH skill, then rerun parser/preflights/focused tests.
  Rejected smoke, oracle, and test-tuned candidates must leave the current-best
  commands and `runs\latest_full_pipeline.json` unchanged.
- Final archive verification passed `266/266` payload stream hashes, `267` ZIP
  entries including `_MANIFEST.json`, zero prohibited/extra entries, and exact
  payload bytes `17,636,816`. Compressed size is `8,951,829` bytes (`8.537 MiB`);
  archive SHA-256 is
  `82be9ffd37b4d6361e5cddf11313a5f065dc64d524228324ebe260c48cd49385`.
- Deleted the verified staging tree (`267` temporary files, `17,718,737` bytes)
  only after the companion archive hash matched. Final retention audit
  `runs\artifact_retention_audit_after_reference_pack_refresh_20260711` loaded
  all `22` cleanup manifests, checked `540` run directories and `67` expected
  absent source roots, found no missing protected artifact or remaining source,
  and passed with `blockers=[]`.

## Diagnostic 2026-07-11 - Generic Multi-Stage Teacher Features Rejected

- Research basis was checked before implementation: Knowledge Review
  (CVPR 2021), Feature Distillation Overhaul (ICCV 2019), Masked Generative
  Distillation (CVPR 2023), and fine-grained noisy-label learning (CVPR 2023).
  The diagnostic tests spatial/multi-stage representation transfer, not the
  pooled RKD/CRD variants already rejected on the keeper.
- Added `trkh.tools.audit_multistage_teacher_feature_readiness` and four focused
  tests. The protocol uses only the generic ImageNet-pretrained
  `tf_efficientnetv2_s.in21k_ft_in1k` teacher (20,177,488 parameters), never a
  TRKH-finetuned teacher checkpoint. It extracts fixed signatures from stages
  1/2/4/5 plus the head and fits source-grouped five-fold train-OOF linear
  readouts. Test, raw-data edits, trainable manifests, checkpoints, and model
  training are all disabled.
- Strict `class_f -> yolo_f sample_index` alignment passed for all train/val
  objects: `9215/2606` rows, `8064/2577` source groups, zero duplicate keys,
  missing rows, or label mismatches. The signature was non-collapsed
  (projected effective rank `37.10`), and stage-energy maps were neither
  collapsed nor border dominated. Visual review showed later stages focusing
  on fruit bodies, blemishes, and scratches; this confirms usable spatial
  localization but not class-safe maturity discrimination.
- The predeclared `head_plus_signature` readout reached train-OOF
  macro/class1 `0.85266/0.51155` and validation `0.85619/0.57658`. The direct
  keeper predictions used by the same audit were validation
  `0.88407/0.68406`, so the proposed signal lost `0.02789` macro-F1 and
  `0.10748` class1 F1. On validation it changed 254 rows: 102 corrections,
  135 harms, 17 neutral; class1 FN rescue/TP break was `11/33`, while class1
  FP removal/creation was `52/62`.
- Error direction was not transferable: train-OOF FN-versus-FP AUROC was
  `0.30735`, while validation was `0.68820`. The sign flip fails the required
  fold-safe action test even though generic teacher features improve their own
  weaker readout baselines.
- Decision: `smoke_permission=false` in
  `runs\diagnostic_multistage_generic_effv2s_readiness_20260711`. Eight gates
  failed, so no feature-distillation smoke/full train is permitted and no
  nearby stage/projection/readout/loss-weight sweep is justified. The compact
  5.05 MiB diagnostic, prediction audits, and stage-energy preview are retained
  as negative evidence. Current-best full-train/deployment commands remain
  unchanged because no checkpoint was created or promoted.

## Infrastructure 2026-07-11 - Worktree Hygiene Baseline

- Captured a hash inventory at
  `runs\worktree_hygiene_baseline_20260711` before staging, deleting, reverting,
  or committing anything. Branch and upstream were both
  `7020f0393a8d4260fc7c37372268ae9f27086224`; staged count was zero.
- Baseline scope is substantial: 33 tracked modifications
  (`+45,782/-6,020`) and 269 untracked files (`12.336 MiB`), dominated by
  147 tests and 85 `trkh` files. `git diff --check` reported zero whitespace
  findings; 33 stderr lines are `core.autocrlf=true` conversion warnings.
- The cleanup contract is recorded in
  `docs\TRKH_WORKTREE_HYGIENE_20260711.md`: explicit-path batches only,
  compile/tests plus cached-diff review before commit, no bulk line-ending
  rewrite, and no revert/delete of unclear user-owned changes. Generated run
  cleanup remains separate and still requires preservation manifests plus a
  retention audit.
- Full collection found `620` tests. The first complete run exposed four real
  accumulated contract regressions (`616` pass, `4` fail): three legacy tests
  still unpacked two-value classification samples or expected one-hot collate
  output after the dataset began returning pad `image_mask` metadata and
  preserving hard labels; the hybrid constructor also lagged `ModelConfig`.
- Updated the dataset tests to verify hard-label and batched `image_mask`
  contracts. Synchronized `DETRVisionTransformerWithRegisters` with all base
  model extension fields: four bbox-spatial-fusion fields and four teacher-
  feature-projection fields. An introspection gate now reported zero
  unconsumed fields across all 270 `ModelConfig` entries. Related tests passed
  `20/20`, the EOF hygiene test passed `1/1`, and the complete suite then
  passed `620/620` in 61.63 seconds.
- Batch 01 used an explicit 275-path manifest and excluded research docs,
  `BaoCao`, root reports, runs, datasets, and model artifacts. Compileall,
  33/33 PowerShell parses, cached diff check, prohibited-artifact scan, and
  secret-signature scan passed. The retained staged patch SHA-256 is
  `6100001308b16567b0ac745b24ab6beb9ec9314ee246dbf62830e42a64ae9a6c`.
- Created preservation commit
  `4c2c7cb824b48a76c9d796acaae944bf41369694` (`feat: checkpoint verified TRKH
  research runtime`): 275 files, `+105,611/-6,030`. Worktree status entries
  fell from 302 to 28 with an empty index. No behavior, dataset, checkpoint,
  selected recipe, or current-best command was changed by the commit itself.
- Batch 02 selected the remaining 21 TRKH documentation paths and continued to
  exclude the five `BaoCao/*` files plus root reports 9/10. UTF-8, conflict,
  relative-link, companion-hash, ZIP entry, and all 266 payload stream hashes
  passed. Reports 14/15 matched their archived copies byte-for-byte. The only
  raw `git diff --check` findings were 15 intentional Markdown hard breaks in
  the verbatim imported report 15; all non-source whitespace findings were
  fixed before commit.
- Created documentation commit
  `7548c1f72ad7837d0adbb10b99bc3be4f81cdef9` (`docs: preserve TRKH research
  evidence and hygiene record`): 21 files and 18,990 added lines, including the
  8.537 MiB verified reference pack. Worktree entries fell from 28 to the seven
  explicitly protected untracked files, with an empty index.
- Pushed both commits to `origin/classification-only-research` and independently
  resolved the remote ref. Local and remote heads matched at `7548c1f`, with
  ahead/behind `0/0`. GitHub therefore contains the verified runtime, VS Code
  commands, current research records, reports 14/15, and reference pack.

## Diagnostic 2026-07-11 - Fixed Wavelet Scattering Rejected

- Checked primary sources before implementation: invariant scattering
  (Bruna/Mallat), rotation/scale scattering (Sifre/Mallat), Kymatio, Wavelet
  Integrated CNNs, and Parametric Scattering Networks. This was a distinct
  surface/texture representation precheck, not a repeat of the rejected raw
  FFT/high-frequency expert.
- Added `trkh.tools.audit_wavelet_scattering_readiness`, pinned
  `kymatio==0.3.0`, and passed five focused tests. The fixed protocol uses
  `128x128` `class_f` crops, RGB plus luminance/red-green/blue-yellow channels,
  Kymatio `J=3/L=4/max_order=2`, mean/std/2x2 spatial descriptors (`2196`
  dimensions), train-only StandardScaler/PCA-128 whitening, and a predeclared
  RBF-SVC `C=3/gamma=scale`. A linear logistic readout is diagnostic only. No
  validation selection, raw-data change, test access, checkpoint, or trainable
  target manifest is permitted.
- Strict `class_f -> yolo_f sample_index` alignment covered all `9215/2606`
  train/validation objects with `8064/2577` source groups and zero missing rows,
  duplicate keys, or label mismatches. GPU scattering plus five grouped folds
  completed in about `135.1s` on the final run. No descriptor cache was retained.
- The raw and train-standardized projected effective ranks were only
  `4.7739/11.6378`; PCA-128 nevertheless explained `0.98808` of standardized
  variance, confirming a highly redundant representation rather than a broken
  extraction. Absolute scattering mass was stable train-to-validation: roughly
  `77.4%` zero-order, `14.5%` first-order, and `8.1%` second-order.
- Linear scattering reached OOF/validation macro-class1 F1
  `0.79693/0.42680` and `0.77613/0.45408`. The predeclared RBF readout improved
  to `0.84388/0.51383` OOF and `0.85632/0.63118` validation, but remained below
  direct keeper validation `0.88407/0.68406`. RBF validation class1 precision
  was `0.74107`, but recall was only `0.54967`.
- Both prediction audits were reviewed. RBF changed `248` validation rows:
  `88` corrections, `138` harms, and `22` neutral. It removed/created class1
  false positives `56/9`, but rescued only `9` class1 false negatives while
  breaking `44` true positives. Train OOF changed `741` rows with `113/616`
  corrections/harms and class1 FN-rescue/TP-break `2/270`. FN-versus-FP
  direction AUROC therefore shifted from `0.38087` train OOF to `0.64872`
  validation instead of transferring.
- The label-assisted validation binary oracle reached class1 F1 `0.85235`, but
  this only proves post-hoc complementarity. The fold-safe action needed to
  retain keeper true positives while applying scattering's conservative FP
  control is absent. Visual review of all five preview rows showed first- and
  second-order energy dominated by fruit boundaries, stems, scratches, and
  high-contrast background edges; it did not reveal a stable maturity-surface
  signal that resolves class 1.
- Decision: seven gates failed and `smoke_permission=false` in
  `runs\diagnostic_wavelet_scattering_rbf_c3_readiness_20260711`. Do not sweep
  J/L/order, opponent channels, PCA, SVM C/gamma/kernel, fusion thresholds, or
  a wavelet router on this keeper. No GPU train was launched and the current
  full-train/deployment command packet remains unchanged.
- Closure verification passed compileall and the complete suite (`625/625`).
  Retention
  `runs\artifact_retention_audit_after_wavelet_scattering_precheck_20260711`
  scanned `544` run directories and passed with `blockers=[]`. `pip check`
  continues to report pre-existing shared-venv conflicts between MambaVision's
  old exact pins and the repository runtime, plus OpenCV's NumPy requirement;
  Kymatio itself imports and runs correctly, so unrelated package versions were
  deliberately left unchanged.

## Probe 2026-07-11 - Full-Frame Detection Curriculum Rejected

- Checked NTS-Net, ELoPE, compact joint localization/classification, Cross-X,
  and InsLoc before implementation. The fixed hypothesis used complete
  `yolo_f` frames and supervised boxes to teach a DETR encoder where the fruit
  is, then transferred only an explicit encoder allowlist into the unchanged
  object-crop TRKH classifier. No raw dataset file was changed and the test
  split was never evaluated.
- Added a bidirectional curriculum-checkpoint builder with state-reset and
  allowlist reports. Classifier-to-detector initialization transferred
  `183/185` model keys (`7,245,048` elements) while excluding the classifier
  head; detector-to-classifier transfer copied `155` encoder keys
  (`6,909,381` elements) and preserved the original classifier head bit-exact.
  Generated detector checkpoints now declare `required_runtime`, and training
  fails closed unless both `--full-image-detection` and `--skip-final-test`
  are present. The trainer propagates this contract through subsequent
  best/last/interrupt checkpoints, while fresh curriculum payloads remove stale
  resume, calibration, stage, data-summary, and EMA-selection metadata.
- That guard was required: the first resource dry run omitted full-image mode,
  cropped around the primary object, and ignored `1,151` secondary training
  boxes. It is infrastructure-invalid. The corrected loader covered train
  `8,064` images / `9,215` objects and validation `2,577` images / `2,606`
  objects with `ignored_objects=0`. Batch 32 used about `2,656.7 MiB` allocated
  and `3,098 MiB` reserved, so it was selected without leaving GPU capacity
  idle.
- The fixed 20-query, decoder-depth-2, FFN-512 full-frame detector ran two
  120-batch epochs. Epoch 2 reached matched-query validation macro/class1 F1
  `0.928815/0.769760`, class1 precision/recall `0.800000/0.741722`, bbox IoU
  `0.483235`, and best detection F1@0.5 `0.414182`. The high classification
  score depends on ground-truth Hungarian matching and is therefore diagnostic,
  not a deployable classifier gate.
- After the detector encoder was transferred back under the original TRKH
  classifier head, independent validation before fine-tuning collapsed to
  macro/class1 `0.782285/0.522167`. The unchanged current V8 recipe recovered
  after two 120-batch epochs only to raw `0.874127/0.659574`, class1 P/R
  `0.551111/0.821192`. This failed the locked keeper minima
  `0.882925/0.678261` by `0.008798/0.018686`; keeper softboost also failed to
  transfer at `0.873939/0.650307` and reduced class1 recall to `0.701987`.
- Aligned full-validation comparison changed 81 predictions: 28 corrections,
  50 harms, and 3 wrong-to-wrong transitions. It rescued 8 class1 false
  negatives and broke 2 true positives, but created 33 new class1 false
  positives while removing only 8. The dominant regression was `0->1`.
- Reviewed raw and softboost prediction audits, 48-row boundary evidence,
  architecture traces, 24-case raw XAI, 24-case softboost XAI, and the full
  corruption audit. Raw `0->1` object-desaturation probability drop averaged
  `0.231264`, versus background blur/gray `0.007791/0.010052`. Center occlusion
  changed macro-F1 only `-0.008466`, while dim, bright, and low-contrast
  conditions changed it `-0.089595/-0.081702/-0.070126`. Attention usually
  covered the fruit, although selected Grad-CAM/rollout maps still emphasized
  boundaries and background. The unresolved bottleneck is photometric/surface
  class separability and class1 FP control, not basic localization.
- Corrected `robustness_eval` while performing this audit: `yolo_f` now loads
  classification object crops with original/transformed bbox metadata and pad
  masks, and model forwarding now supplies `bbox_token_prior`, `features["bbox"]`,
  and `image_valid_mask` exactly as the independent evaluator does. Focused
  regression tests cover the dataset/transform/forward contract.
- Decision: reject the exact direct DETR-20 curriculum, nearby detection-loss,
  LR/batch/epoch sweeps, and the same unrestricted encoder round-trip plus V8
  fine-tune. Do not run test, extend training, update the deploy pointer, or
  change `docs\TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt` from this
  result.
- Compact evidence is retained at
  `runs\evidence_detcls_fullframe_curriculum_rejected_20260711`; its 89 payloads
  contain no checkpoint/ONNX/engine and manifest SHA-256
  `32abb934337801851f9523e4d31c9dfa090efd0ce5b28ca146346010fb808f62`.
  Guarded cleanup deleted exactly seven superseded source roots and reclaimed
  an observed `1,458,147,328` bytes. The post-clean retention audit covered
  `547` run directories and passed with `blockers=[]`. Compileall,
  `git diff --check`, and the complete test suite passed (`637/637`).

## Diagnostic 2026-07-11 - Learned Context Illuminant Estimator Rejected

- A same-frame relative-maturity idea was screened first and closed without
  code or GPU work. `yolo_f/train` has only 306 cross-class object pairs in 144
  mixed-label frames; validation has 29 multi-object frames but zero mixed-label
  pair and zero class1 multi-object frame. A same-frame action therefore has no
  independent validation support and cannot establish fold-safe class1 FN/FP
  direction.
- Checked primary CLCC, Cheng et al. learning-based illuminant estimation,
  Convolutional Color Constancy, FC4 confidence-weighted pooling, and CNN color
  constancy before implementation. Their key distinction is that illuminant
  estimation needs an illuminant-dependent representation; forcing the class
  representation itself to ignore color repeats the already rejected generic
  consistency route. Because TRKH has sRGB rather than RAW images, this audit
  is explicitly a synthetic diagonal-photometric nuisance test, not a claim of
  ground-truth physical illuminant recovery.
- Added `trkh.tools.probe_context_illuminant_estimator_readiness` and four
  focused tests. The fixed protocol uses eight predeclared clean/dim/bright/
  RGB-cast/warm/cool gains, a 67D RGB/log-chroma/luminance/saturation descriptor
  from full-frame context outside the union of transformed object boxes, and
  `ExtraTreesRegressor(n=96,min_leaf=3,max_features=0.75)`. All casts from one
  source remain in one of five GroupKFold folds. The estimator is not saved;
  no checkpoint, trainable manifest, raw-data edit, or test access is allowed.
- Full support was exact: `8,064` train sources, `2,577` validation sources,
  `2,606` validation objects, and `64,512/20,616` train/validation descriptor
  rows. Gain recovery itself transferred strongly: minimum channel R2 was
  `0.98857` train OOF and `0.98282` validation; maximum log-gain MAE was
  `0.00409/0.00583`, with validation mean/p95 angular error only
  `0.16772/0.95596` degrees.
- Strong gain regression did not make correction class-safe. Clean keeper
  macro/class1 `0.884675/0.686047` fell after near-neutral estimated correction
  to `0.867572/0.638418`. Fifty-two predictions changed: only 8 corrections
  versus 40 harms, with class1 FN rescue/TP break `1/6` and FP remove/create
  `5/20`. A mean clean pixel change of only `0.00477` was already enough to
  erase subtle maturity color and expand `0/2/4->1` errors.
- Estimated correction recovered many severe synthetic failures but remained
  below the required clean envelope: dim raw/corrected macro-class1 was
  `0.722295/0.372222 -> 0.881802/0.676218`; red
  `0.484077/0.185022 -> 0.850795/0.595174`; green
  `0.134447/0.065445 -> 0.819418/0.500000`; blue
  `0.768656/0.448087 -> 0.882242/0.680115`; warm
  `0.731609/0.353659 -> 0.856235/0.608696`; and cool
  `0.514673/0.096089 -> 0.872015/0.651685`. Bright correction regressed
  `0.774372/0.455782 -> 0.758768/0.387234`, with 181 corrections versus 217
  harms and 178 newly created class1 false positives. Mean macro/class1 recovery
  was only `0.77547/0.69628`, below the locked `0.80/0.70` readiness thresholds.
- The full-validation known-gain oracle isolated the ceiling from estimator
  error. Exact dim inversion returned bit-near clean metrics and blue/cool were
  nearly recoverable, but bright remained `0.764950/0.399142`, red
  `0.852863/0.596774`, green `0.824396/0.500000`, and warm
  `0.863249/0.628099`. Clipping and sRGB nonlinear processing destroy class
  evidence that diagonal inversion cannot reconstruct even with the true gain.
  The five-class visual preview likewise shows correction cooling/greening the
  fruit and removing legitimate maturity warmth.
- Decision: eight gates failed and `smoke_permission=false`. Do not sweep tree
  family/count, descriptor statistics, synthetic gain strength, correction
  shrinkage/dead-zone, or implement an FC4/context-illuminant CNN on this sRGB
  route. The oracle ceiling proves those changes cannot meet the clean/class1
  gate. No GPU train, test evaluation, deploy update, or current-best command
  change is permitted.
- Full evidence is retained at
  `runs\diagnostic_context_illuminant_estimator_full_20260711`: six payloads,
  `6,140,666` bytes, no model/deploy binary, manifest SHA-256
  `010c610d8a7bda758e25ad9be74256f49090ad7231671027700c5e151bc58c76`.
  The superseded 100/50-source dry-run (`581,496` bytes) was removed under
  `runs\cleanup_manifest_20260711_context_illuminant_dryrun.json`; observed
  free-space delta was `593,920` bytes. Retention then covered `549` run
  directories and passed with `blockers=[]`. Py-compile, four focused tests,
  `git diff --check`, and the complete suite passed (`641/641`).

## Diagnostic 2026-07-11 - Pooled-Head API-Net Interaction Rejected

- Checked the primary API-Net AAAI/arXiv paper, DCAL CVPR 2022, and Pairwise
  Confusion ECCV 2018 before implementation. API-Net is distinct from the
  already rejected pairwise-confusion/margin losses: it learns a mutual vector
  from two images, generates image-specific channel gates, trains self/other
  residual features with cross entropy plus score ranking, and unloads the
  interaction module for single-image inference. DCAL independently supports
  training-only cross-image interaction, although its PWCA operates on tokens
  rather than the pooled feature used in this diagnostic.
- Added `trkh.tools.probe_api_pairwise_interaction_readiness` and three focused
  tests. The fixed no-sweep protocol uses the retained 256D keeper-head cache,
  mutual MLP `512->128->256`, `sigmoid(x_m * x_i)` gates, four self/other
  outputs, ranking weight `1.0`, margin `0.05`, AdamW `0.003`, and 20 epochs.
  A matched plain-linear control receives exactly the same initialization,
  natural-frequency batches, nearest intra/inter pairs, class weights,
  optimizer, and schedule. Each anchor occurs once per epoch; class 1 is not
  oversampled as an anchor. Every pair excludes the same `source_stem`.
- The audit covered all `9,215/2,606` train/validation rows, `8,064` train
  source groups, zero train-validation source overlap, and five
  StratifiedGroupKFold folds with zero fit-holdout source overlap. Every class
  had intra/inter pair support; the final plan contained `368,600` pairs,
  exactly 20 anchor visits per row, and zero same-source pair. The known
  limitation is explicit: keeper train embeddings are in-sample, so absolute
  OOF scores are optimistic and only API-versus-control direction transferring
  to untouched validation can open an image-model smoke.
- Direction failed in every fold. Matched-control OOF macro/class1 F1
  `0.911297/0.740431` fell with API to `0.907701/0.726531`; the five fold
  macro/class1 deltas were all negative. Validation control
  `0.848918/0.602041` fell to `0.846950/0.597590`. The direct frozen keeper
  remains `0.884675/0.686047`, so API single-image inference lost
  `0.037725` macro F1 versus the actual head.
- Full transition audits agree with the metrics. OOF control-to-API made 26
  corrections and 46 harms and removed/created class1 false positives
  `10/41`. Validation control-to-API made `15/22` corrections/harms, rescued
  six class1 false negatives without breaking a true positive, but
  removed/created class1 false positives only `2/19`. Versus the direct keeper,
  API changed 147 rows with `25` corrections, `111` harms, class1 FN
  rescue/TP break `9/3`, and FP remove/create `4/69`.
- Probability audit identifies the mechanism rather than an epoch shortfall.
  On validation, API lowered mean class1 probability for true class1 by
  `0.069506` while raising it for non-class1 rows by `0.042589`; it raised
  class1 probability on `2,303/2,455` negatives. The same direction appeared
  in OOF (`-0.099088` true class1, `+0.047158` non-class1). API pair-conditioned
  cross entropy did converge below control cross entropy, but the unloaded
  single-image classifier learned class-unsafe smoothing.
- Protocol self-review found that the first symmetric API run still reused
  nearest partners as label-bearing endpoints. Class1 was only `5.871%` of
  natural anchors but `14.535%` of selected partners (`53,576/368,600`), so
  symmetric endpoint loss amplified class1 despite the natural anchor sampler.
  This does not reverse the matched candidate-versus-control result, but it is
  a confound against the project's no-global-oversampling rule and cannot be
  the sole pooled-API closure evidence.
- The tool was therefore made explicitly reproducible with
  `--loss-scope symmetric|anchor_only`, partner endpoint/reuse telemetry, and a
  fourth test proving that anchor-only loss is invariant to partner labels.
  The one predeclared correction keeps nearest intra/inter images as mutual-gate
  context but computes CE/ranking only for each naturally sampled anchor; the
  partner is never a label target. All other folds, seeds, pairs, optimizer,
  20-epoch schedule, thresholds, and no-test guards stayed fixed.
- Anchor-only removed most of the symmetric control bias but still failed to
  transfer. Control/API OOF macro/class1 was
  `0.929138/0.798137 -> 0.928312/0.795699`, while validation was
  `0.862903/0.628743 -> 0.863613/0.631268`. Only two of five folds were
  positive; aggregate OOF gains were `-0.000827/-0.002438` and validation gains
  only `+0.000710/+0.002526`, below the fixed `0.002/0.015` gates and far below
  the direct keeper `0.884675/0.686047`.
- Anchor-only transition evidence is also class-unsafe. Versus its control it
  made `17/23` OOF corrections/harms and `10/10` on validation, with class1 FP
  remove/create `6/16` OOF and `0/3` validation. Versus the direct keeper it
  changed 96 rows with `24` corrections, `64` harms, class1 FN rescue/TP break
  `3/14`, and FP remove/create `10/16`. It continued to lower mean class1
  probability on true class1 by `0.082786` and raise it on non-class1 by
  `0.030825` (`2,335/2,455` negatives raised).
- No image-level XAI rerun is technically meaningful for this precheck: the
  frozen keeper image encoder, token path, attention, and checkpoint never
  changed, while the training-only API module was deliberately not saved.
  The complete applicable audit surface is retained instead: pair/source
  support, fold curves, per-class confusion, OOF/validation predictions,
  probability shifts, and all transition directions. Existing keeper XAI still
  describes the unchanged spatial representation.
- The first symmetric run failed 13 readiness checks; the corrected anchor-only
  run also failed 13 and retained `smoke_permission=false`. Do not integrate
  either pooled-head API module, run test, or sweep MLP width,
  optimizer/LR, class weights, batch size, ranking margin/weight, or epochs on
  the frozen keeper cache. A future token-level PWCA hypothesis is distinct,
  but it must first define a new fold-safe FN-versus-FP gate and cannot use this
  failed pooled interaction as smoke permission.
- Evidence is retained at
  `runs\diagnostic_api_pairwise_interaction_full_20260711`: seven payloads,
  `3,464,753` bytes, no checkpoint/model/test payload, manifest SHA-256
  `e4c847118ef3b82a700e6385e2dcd8a2afde8417e0daa7cc27bcf8f5f80770d0`.
  The full run took `29.445 s`; py-compile, three focused tests, manifest hash
  verification, compileall, complete pytest `644/644`, and `git diff --check`
  passed. Retention audit
  `runs\artifact_retention_audit_after_api_pairwise_precheck_20260711`
  covered 551 run directories with `blockers=[]`. Current-best commands and
  deploy pointers remain unchanged; the command TXT SHA-256 is still
  `3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.
- Corrected anchor-only evidence is retained separately at
  `runs\diagnostic_api_pairwise_anchoronly_full_20260711`: seven payloads,
  `3,485,228` bytes, no checkpoint/model/test payload, manifest SHA-256
  `d65f25b1c4d3117e4e72f1f12312b3e035b8de06691e824a18515c07c559bc83`.
  It completed in `31.154 s`; all payload hashes and `9,215/2,606` prediction
  rows were verified, stderr was empty, and no model-like file exists.
  Final compileall, four focused tests, complete pytest `645/645`, and
  `git diff --check` passed. Retention
  `runs\artifact_retention_audit_after_api_anchoronly_precheck_20260711`
  covered 553 run directories with `blockers=[]`; the current-best command hash
  remained unchanged.

## Diagnostic 2026-07-11 - Token-Level Natural-Distractor PWCA Below Gate

- Re-read the primary DCAL CVPR 2022 paper and supplement before coding. PWCA
  differs materially from pooled API: target queries attend jointly to target
  and distractor key/value tokens through weights shared with the ordinary
  self-attention branch, while PWCA is removed at inference. The paper's
  ablation found random same-dataset pairs from the natural distribution better
  than intra-only, inter-only, fixed 1:1, Gaussian, or external-COCO
  distractors. The fixed proxy therefore uses label-blind random partners and
  does not repeat nearest-pair API.
- Added `trkh.tools.extract_pwca_token_cache` and
  `trkh.tools.probe_pwca_token_interaction_readiness` with five focused tests.
  The extractor runs the immutable keeper under deterministic eval transforms,
  takes its final head token plus the 32 highest fine-grained-attention patch
  tokens, and applies one fixed QR-orthogonal `256->64` projection. It covers
  all `9,215/2,606` train/validation objects, `8,064/2,577` source groups, zero
  split overlap, and writes no test/model/trainable manifest.
- Cache alignment and representation checks passed. Direct validation from the
  extraction pass is exactly keeper macro/class1 `0.884675/0.686047`.
  Projected head effective rank is `5.951/6.274` train/validation and selected
  patch rank `14.407/14.076`. Top-32 tokens retain mean attention mass
  `0.886970/0.888785` (p05 `0.614710/0.624190`) and have mean bbox fraction
  `0.856538/0.872614`. Visual review of one train/validation row per class shows
  fruit-surface, spot, lesion, and decay focus, with occasional stem/background
  edge selection and low-tail bbox coverage; no top-k/rank was changed after
  viewing validation.
- The proxy is one shared pre-norm four-head SA/FFN/classifier over 33x64
  tokens. Its matched control trains self-attention CE only. The candidate adds
  equal-weight PWCA target CE, where target Q attends to concatenated target and
  random distractor K/V; validation uses only the shared self-attention branch.
  Both use identical initialization, natural-frequency anchor batches,
  inverse-frequency CE, AdamW `lr=0.002/wd=0.005`, cosine decay, dropout `0.05`,
  and 20 epochs. Distractors always come from a different source, never carry
  label loss, and are selected without labels.
- Full source-grouped five-fold support was exact. The final plan contains
  `184,300` pairs, every row is anchor and partner exactly 20 times, same-source
  count is zero, and natural inter-class fraction is `0.773880`. Fold-level
  macro gains were positive in four of five folds; class1 gains were positive
  in three. This is a real but sub-gate signal rather than the uniformly bad
  pooled API result.
- Matched-control OOF macro/class1 `0.933468/0.805324` became
  `0.934686/0.810855`, gains only `+0.001219/+0.005531` versus required
  `+0.002/+0.010`. Validation control `0.874168/0.636656` became
  `0.874874/0.647436`, gains `+0.000706/+0.010780` versus required
  `+0.002/+0.015`. The candidate remains below class1 `0.70` and loses
  `0.009801` macro F1 versus the direct keeper.
- Transition audits explain both value and risk. OOF PWCA made `48` corrections
  and `44` harms, rescued/broke class1 FN/TP `14/5`, and removed/created FP
  `17/22`. Validation versus matched control made `15/18` corrections/harms,
  rescued/broke `5/3`, and removed/created FP `4/3`. Versus the direct keeper,
  however, it made `38` corrections and `47` harms, removed/created class1 FP
  `22/7`, but rescued only two FN while breaking 19 true positives. It learns
  conservative class1 FP control but cannot preserve keeper recall.
- PWCA attention is active and target-dominant on average: distractor mass mean
  is `0.329867`, below the fixed `0.35` limit. Its p95 is `0.637411`, above the
  `0.55` tail gate; class1 FN/FP p95 values remain `0.604036/0.607195`. The
  explicit PWCA branch itself is weaker than self inference at validation
  macro/class1 `0.871155/0.636943`, consistent with a useful distractor only if
  the shared representation can absorb it safely.
- Decision: 11 readiness checks failed and `smoke_permission=false`. Do not
  launch an end-to-end PWCA smoke or sweep projection rank, top-k, heads,
  dropout, optimizer/LR, PWCA weight, pair distribution, or epochs from this
  proxy. Preserve the positive relative direction as research evidence. A
  future PWCA revisit must start from an exact keeper-preserving residual path
  and prove recall protection with a new predeclared gate; it cannot treat this
  below-threshold proxy as permission.
- Compact evidence remains at
  `runs\diagnostic_pwca_token_rank64_top32_full_20260711`: 14 payloads,
  `4,176,398` bytes, no model/test payload, manifest SHA-256
  `bbe9fea148340bcd3ce0646e348a60089d929a2e2e952771c8f35d50db42b0b6`.
  It includes full OOF/validation predictions and curves plus copied cache
  summary, original cache manifest, and five-class train/validation previews.
  Guarded cleanup
  `runs\cleanup_manifest_20260711_pwca_token_cache_rejected.json` deleted 11
  cache files (`53,500,161` bytes) after hash preservation and observed
  `53,526,528` free bytes reclaimed. Raw data, test, keeper, deploy pointers,
  and current-best commands were untouched. Py-compile, five focused tests,
  `git diff --check`, and retention
  `runs\artifact_retention_audit_after_pwca_token_precheck_20260711` passed;
  retention covered 555 run directories with `blockers=[]`, and the command TXT
  SHA-256 remained `3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.
  Compileall and the complete suite also passed (`650/650`).

## Diagnostic 2026-07-11 - Keeper-Relative PWCA Scalar Residual Rejected

- Before regenerating the deleted 53.5 MB token cache or opening a GPU smoke, I
  checked the narrowest keeper-preserving interpretation of the retained PWCA
  evidence. AdaptFormer motivates a parallel residual path that starts from the
  unchanged base function, while rate-constrained optimization motivates
  enforcing recall and false-positive requirements on train-only support. The
  fixed audit therefore applies
  `log(p_keeper) + alpha * (log(p_pwca) - log(p_control))`: `alpha=0` is exactly
  the keeper, and the matched PWCA-minus-control term isolates the prior
  training-only interaction effect.
- Added `trkh.tools.audit_pwca_keeper_residual_readiness` and three focused
  tests. It reads only the retained `9,215` train-OOF and `2,606` validation
  rows, rejects test paths, uses a fixed 201-point `[-1,1]` alpha grid, and
  selects alpha exclusively on train OOF. Eligibility requires macro F1,
  class1 F1, and class1 recall no lower than the keeper plus no increase in
  class1 false positives. Validation is transfer evaluation only.
- Exactly one coefficient was eligible: `alpha=0`. Every nonzero relative
  PWCA residual violated at least one train-OOF preservation constraint. The
  selected result consequently kept train OOF macro/class1
  `0.940013/0.816358` and validation `0.884675/0.686047` unchanged, with class1
  train OOF TP/FN/FP `529/12/226` and validation `118/33/75` unchanged.
- The label-assisted choose-keeper-or-PWCA upper bound is also limited:
  validation can fix only 38 keeper errors while 47 keeper-correct rows are
  harmable, and its class1 F1 is `0.733945`. This is not a selection rule and
  does not permit validation routing or test use.
- Decision: close scalar post-hoc PWCA/control residual interpolation and do
  not sweep coefficient ranges, steps, class-specific scales, thresholds, or
  routers. This result does not claim to reject a newly trained zero-initialized
  token adapter, but it removes the retained logits as standalone evidence for
  building one; a trainable revisit now needs a different train-only
  supervision signal and a predeclared nonzero recall-safe action before GPU.
- Evidence is retained at
  `runs\diagnostic_pwca_keeper_relative_residual_full_20260711`: five payloads,
  `2,525,541` bytes, manifest SHA-256
  `6996ed28dc16700eb24e7782a0f2c29739dabedb5028eb20def144aad3f4974f`,
  no model/checkpoint/test payload. The current-best command and deploy pointer
  remain unchanged.
- Closure passed compileall, focused API/residual tests `7/7`, complete pytest
  `653/653`, manifest hash verification, and `git diff --check`. Retention
  `runs\artifact_retention_audit_after_pwca_residual_precheck_20260711`
  covered 557 directories with `blockers=[]`; the current-best command SHA-256
  remains `3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.

## Diagnostic 2026-07-12 - MaskFeat RGB-HOG Target Rejected Before GPU

- Checked the primary MaskFeat CVPR 2022 paper before implementation. Unlike
  the already rejected RGB reconstruction and global VICReg/DINO routes,
  MaskFeat predicts a dense handcrafted feature only at masked regions. Its
  image ablation identifies separate RGB-channel HOG, nine unsigned orientation
  bins, `8x8` pixel cells, and local L2 contrast normalization as the effective
  default. The paper also uses hundreds of pretraining epochs, so under TRKH's
  `<=30`-epoch constraint this target had to show strong direct class1 and
  FN-versus-FP evidence before a trainable head could be justified.
- Added `trkh.tools.audit_maskfeat_hog_readiness` and three focused tests. The
  fixed audit computes dense RGB HOG with differentiable Torch operations at
  image size 128, preserves the full `16x16x3x9=6,912` cell descriptor, and
  uses the same fixed scaler/PCA-128 plus linear/RBF readouts under five
  StratifiedGroupKFold folds as the prior wavelet control. It aligns
  `class_f` crops to all `9,215/2,606` `yolo_f` object rows, has zero
  missing/duplicate/label mismatch, and reads no test split.
- A self-review caught that the first infrastructure-valid run had pooled the
  `16x16` HOG grid to `8x8`. Because MaskFeat predicts every cell, that proxy
  could discard the local detail being tested. I therefore reran one
  paper-faithful full-grid correction without looking for a favorable setting.
  Standardized projected effective rank rose from `29.7902` to `61.8220`, so
  the corrected descriptor is neither collapsed nor equivalent to the pooled
  version.
- Full-grid class signal still failed decisively. Primary RBF OOF macro/class1
  F1 was `0.733291/0.331096`; validation was `0.656872/0.262911`, class1
  precision/recall `0.451613/0.185430`, versus direct keeper validation
  `0.884073/0.684058`. The linear control was weaker at OOF
  `0.546913/0.198565` and validation `0.500041/0.168724`.
- Transition evidence shows the same unsafe suppressor mechanism as fixed
  gradients/wavelets, not a transferable MaskFeat target. Versus keeper on
  validation, HOG made `69` corrections and `539` harms, removed/created
  class1 FP `66/24`, but rescued only `3` class1 FN while breaking `93` true
  positives. The label-assisted binary oracle class1 F1 `0.858156` is large
  only because an unavailable oracle can choose when to apply suppression.
  FN-versus-FP action AUROC inverted from `0.309054` train OOF to `0.736045`
  validation, failing fold-safe direction and stability.
- Visual review of the five-class RGB/channel-energy contact sheet found HOG
  energy concentrated on fruit contours, stems, scratches, speckles, padding,
  and foreground-background edges. It exposes plausible shape/texture but no
  stable interior maturity cue that separates true class1 recall cases from
  `0/2/4->1` false positives. The preview is bit-identical across the pooled
  and full-grid runs because only the downstream readout grid changed.
- Decision: eight readiness checks failed and `smoke_permission=false`. Do not
  implement/sweep a MaskFeat-HOG auxiliary/pretraining head, mask ratio,
  HOG bins/cell size/channel space, decoder, target weight, schedule, or a HOG
  router under the current data and 30-epoch budget. This closes the distinct
  HOG target, not every future masked target; a revisit needs a new semantic
  surface target with train-OOF recall-safe direction rather than another
  handcrafted edge descriptor.
- Full evidence is retained at
  `runs\diagnostic_maskfeat_hog_rgb9c8_fullgrid16_readiness_20260712`: ten
  payloads, `3,906,228` bytes, manifest SHA-256
  `2782c40309425080c6281c97bf2aef29cb25bba43a7333afc555ec8f3d65886a`,
  no model/checkpoint/test payload. It includes the superseded pooled summary
  and manifest. Guarded cleanup
  `runs\cleanup_manifest_20260712_maskfeat_hog_pooled8_superseded.json`
  removed exactly nine pooled-run files (`3,882,869` bytes; observed free
  delta `3,903,488` bytes). Retention then covered 559 directories with
  `blockers=[]`; current-best commands remain unchanged.
- Closure passed py-compile, compileall, focused HOG/wavelet regressions `8/8`,
  complete pytest `656/656`, manifest hash verification, and
  `git diff --check`. No checkpoint, deploy pointer, test metric, or full-train
  command was created or changed.

## Diagnostic 2026-07-12 - DINOv2 Dense Patch Surface Target Rejected Before GPU

- Re-read the primary DINOv2 paper, Meta model card, and official
  `forward_features` implementation before coding. DINOv2-S/14 returns one
  normalized CLS token plus `16x16=256` normalized patch tokens for a 224-pixel
  input; its patch-level self-supervision and dense downstream use make those
  tokens a materially different semantic target from the already rejected
  RGB/HOG/wavelet descriptors and the prior global 384D DINOv2 cache.
- Added `trkh.tools.audit_dinov2_dense_patch_readiness` and four focused tests.
  The descriptor was locked before validation: a 384D CLS-only control versus
  `CLS + patch mean + patch std + 2x2 spatial patch means` (`2,688` dimensions).
  Both branches use the same scaler, fixed randomized PCA-128, linear and RBF
  readouts, and the same five source-grouped folds. There is no grid/readout
  sweep, descriptor cache by default, test access, image-model training,
  target manifest, or checkpoint write.
- Strict `class_f -> yolo_f` alignment covered all `9,215/2,606` object rows
  and `8,064/2,577` source groups with zero duplicate keys, missing rows, or
  label mismatch. Extraction used pretrained `vit_small_patch14_dinov2.lvd142m`
  at 224 pixels, resized the official positional embedding from `37x37` to
  `16x16`, took `72.9/21.3` seconds for train/validation, and peaked at about
  `644 MiB` allocated CUDA memory.
- Protocol self-review found that the first complete run had accidentally
  supplied different fold seeds to the control and candidate. Its negative
  result was not used. The tool now locks the model and one common seed and
  records `matched_source_folds=true`; the corrected fold telemetry is exactly
  identical (`7,377/1,838`, `7,363/1,852`, `7,384/1,831`, `7,387/1,828`,
  `7,349/1,866`) for both branches.
- Under matched folds, dense patches add real train-OOF signal but it does not
  transfer. Primary RBF CLS OOF macro/class1 F1
  `0.872683/0.581068` becomes `0.880335/0.603738`, while validation CLS
  `0.887625/0.664384` falls to `0.882850/0.636364`. Dense validation class1
  precision/recall is `0.674074/0.602649`, below direct keeper
  `0.884073/0.684058`. Linear readouts show the same weak regime: dense OOF
  `0.830536/0.472397` and validation `0.844524/0.543590`.
- Transition review explains why this is not a distillation permission. Versus
  the matched CLS control, dense patches make `122/78` OOF corrections/harms
  but only `40/40` on validation; validation class1 FN rescue/TP break is
  `10/16` and FP remove/create is `16/16`. Versus the keeper they make
  `98/71` corrections/harms and remove/create `49/17` class1 false positives,
  but break 31 true class1 predictions while rescuing only four. The
  label-assisted binary oracle class1 F1 `0.813333` is therefore an unavailable
  selector, not a deployable gain.
- FN-versus-FP direction also shifts across splits. Dense-minus-CLS direction
  AUROC is `0.592613` train OOF versus `0.771886` validation, exceeding the
  fixed stability gap; relative to keeper errors validation AUROC is only
  `0.598086`. On actual keeper FN rows, the dense-minus-CLS class1 probability
  delta is negative on both train/validation (`-0.035150/-0.088202`), so the
  signal cannot justify another keeper-relative scalar residual or router.
- The five-class patch-deviation contact sheet is non-collapsed but nearly
  uniform in spatial entropy (`0.996020`) with border mass `0.255176`. It
  responds to fruit texture, spots, lesions, lighting, edges, and remaining
  crop/background patches without isolating a stable recall-safe maturity
  region. Because the encoder is frozen and no image model was trained, a new
  TRKH Grad-CAM/rollout audit would only reproduce keeper XAI; the applicable
  audit surface is the patch map, full OOF/validation predictions, per-class
  metrics, transitions, and direction checks reviewed above.
- Decision: seven fixed readiness checks fail and `smoke_permission=false`.
  Do not sweep DINOv2 input size, model scale, spatial grid, patch moments,
  pooling, PCA/SVM, thresholds, residual coefficients, routers, adapters, or
  dense-feature distillation on this evidence. This closes the fixed DINOv2
  dense semantic-surface target on the current data; a future external target
  must prove stable keeper-FN rescue without sacrificing true class1 recall.
- Corrected evidence is retained at
  `runs\diagnostic_dinov2_dense_patch_semantic_surface_matchedfolds_readiness_20260712`:
  11 payloads, `3,866,659` bytes, manifest SHA-256
  `7a44e78857d8908eea651de6ea047b81009b995a3a7e2b86aae628c4bceb5f52`,
  no model/checkpoint/test payload. It contains the superseded summary and
  manifest. Guarded cleanup
  `runs\cleanup_manifest_20260712_dinov2_dense_unmatchedfolds_superseded.json`
  deleted exactly ten invalid-run files (`3,823,398` bytes; observed free
  delta `3,846,144` bytes). Retention then covered 561 run directories with
  `blockers=[]`; keeper, deploy pointers, and current-best command remain
  unchanged, with command SHA-256
  `3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.
- Closure passed py-compile, compileall, focused DINOv2 regressions `4/4`,
  complete pytest `660/660`, evidence-manifest hash verification, cleanup and
  retention assertions, and `git diff --check`. No full train, test evaluation,
  checkpoint, deploy artifact, or current-best command was produced.

## Diagnostic 2026-07-12 - Two-Stage Re-EDL Keeper Residual Rejected Before GPU

- Cross-checked the ambiguity-aware recommendations in deep-research reports
  14/15 against the no-repeat matrix and primary sources before implementation.
  TEDL (`https://arxiv.org/abs/2209.05522`) motivates a stable cross-entropy
  first stage followed by evidential training. Re-EDL
  (`https://arxiv.org/abs/2410.00393`) and its official source use
  `evidence=softplus(logits)`, `alpha=evidence+prior`, and direct optimization
  of the projected Dirichlet mean while removing the variance and KL terms.
  This audit deliberately composes TEDL's schedule with Re-EDL's documented
  objective; it is not presented as a reproduction of either paper.
- Added `trkh.tools.audit_two_stage_reedl_readiness` and five focused tests.
  The candidate is a zero-initialized linear residual over the exact cached
  keeper log probabilities, so identity is preserved before optimization. It
  uses natural-frequency batches, no class weights/oversampling, fold-fit-only
  standardization, AdamW `lr=0.003/wd=0.0001`, CE for 20 epochs, then compares
  CE versus Re-EDL for ten more epochs from the identical stage-1 state and
  batch order. Total branch budget is exactly 30 epochs.
- The locked CPU audit read all `9,215/2,606` train/validation rows and
  `8,064/2,577` source groups from the existing keeper cache. Five
  StratifiedGroupKFold folds had zero fit/hold source overlap; train and
  validation had zero source overlap. Test, raw data, checkpoint, trainable
  manifest, and validation hyperparameter selection were all absent. The
  grouped OOF limitation is explicit: the residual is held out, but the
  underlying keeper representation was originally trained on the full train
  split.
- Re-EDL adds only a small in-sample-head class1 change and does not transfer.
  OOF CE macro/class1 `0.939108/0.817204` became
  `0.938132/0.821662`; only two of five folds improved class1. Validation CE
  `0.877448/0.640288` fell to `0.858718/0.590406`, far below direct keeper
  `0.884675/0.686047`. Candidate validation class1 precision/recall was
  `0.666667/0.529801`; recall dropped `0.251656` from the keeper.
- Transition evidence rejects the apparent conservative class1 effect. Versus
  validation CE, Re-EDL made `18` corrections and `51` harms, removed/created
  class1 FP `5/7`, rescued four FN, and broke 13 true positives. Versus the
  keeper it made `45/71` corrections/harms, removed/created FP `38/3`, but
  rescued only one FN while breaking 39 true positives. It is another class1
  suppressor rather than a recall-safe ambiguity head.
- The loss was stable rather than divergent: full-train stage-2 Re-EDL loss
  decreased `0.4426 -> 0.2079`, with the same pattern in all five folds.
  Nonetheless its validation uncertainty error AUROC/AURC
  `0.838151/0.022503` was worse than CE entropy
  `0.885434/0.013678`; top-label ECE was also worse
  (`0.075548` versus `0.036798`). At 80% coverage it retained only `35/151`
  validation class1 samples and class1 F1 `0.311111`, so lower aggregate risk
  comes partly from abstaining on most of the difficult focus class.
- Candidate-minus-CE class1 probability has stable FN-versus-FP direction
  AUROC `0.954918/0.978778` on OOF/validation, but this is not a deployable
  action: it is defined relative to the already recall-damaging CE residual,
  while direct keeper transitions break 39 true positives for one FN rescue.
  The risk-coverage and uncertainty histograms confirm separation of some
  errors but consistently trail CE and do not restore closed-set decisions.
- Decision: 13 fixed checks failed and `smoke_permission=false`. Do not add an
  evidential head to the image model or sweep prior weight, CE/Re-EDL epoch
  split, LR, weight decay, residual width/depth, class weights, calibration,
  uncertainty thresholds, or post-hoc routing on this representation. Generic
  EDL/Re-EDL is closed for the current keeper cache; reopen only after a new
  representation independently changes class1 FN/FP support and then rerun a
  matched CE control.
- Evidence is retained at
  `runs\diagnostic_two_stage_reedl_keeper_residual_readiness_20260712`: ten
  payloads, `5,670,155` bytes, manifest SHA-256
  `24a71f2896d9a6aa0457a792322b6c330c30ca827f85a746ea929aa74177da1b`,
  no model/checkpoint/test payload. It contains complete OOF/validation
  probabilities, fold/training curves, calibration, risk-coverage, uncertainty
  plots, and the fixed gate. No full train, test evaluation, deploy pointer, or
  current-best command was produced or changed.
- Closure passed py-compile, compileall, focused Re-EDL tests `5/5`, complete
  pytest `665/665`, evidence-manifest hash verification, `git diff --check`,
  and retention
  `runs\artifact_retention_audit_after_two_stage_reedl_precheck_20260712` over
  563 run directories with `blockers=[]`. The keeper checkpoint remains
  present and current-best command SHA-256 remains
  `3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.

## Diagnostic 2026-07-12 - Joint Spatial-PCGrad Passed Gradient Readiness but Failed Image Smoke

- Revisited full-frame localization only as simultaneous multi-task supervision,
  not the already rejected sequential detector-to-classifier curriculum. The
  implementation follows the conflict-projection rule from PCGrad
  (`https://arxiv.org/abs/2001.06782`) and compares its component balancing to
  GradNorm's multi-task motivation (`https://proceedings.mlr.press/v80/chen18a.html`).
  The fixed candidate uses crop classification as the primary task and raw-context
  full-frame objectness/localization as auxiliary tasks. Detector classification
  gradients are excluded so class priors cannot be injected twice.
- Added `trkh.tools.audit_joint_cls_localization_gradient_readiness` and
  `trkh.tools.train_joint_spatial_pcgrad_smoke` with ten focused tests. The
  readiness audit used only `yolo_f/train`: 7,373 exact single-object paired
  sources, five held-out audit folds, 12 rows per class/fold, and a frozen keeper
  encoder. A DETR-20/depth-2/FFN-512 head was warmed for 120 batches of eight
  full frames while excluding every audit source; its loss ratio fell to
  `0.33354` and the encoder remained bit-exact.
- The predeclared component-normalized rule independently norm-matches
  objectness and localization to the natural-prior classification gradient,
  projects each conflicting component, then caps their combined contribution at
  `0.25`. It passed all readiness checks: raw total cosine median `0.173536`,
  localization retained norm `1.0`, class1 retained norm median `0.920122`,
  spatial-PCGrad CE delta `-0.000514` versus classification-only, and positive
  class1/true-margin deltas `+0.005872/+0.001055`. This established only local
  one-step geometry, not transferable validation improvement.
- A first runtime preflight exposed an important implementation mismatch. Tiny
  microbatch AdamW applies coordinate-wise moment normalization after gradient
  projection and collapsed validation to control/candidate macro-class1
  `0.877542/0.661017` and `0.860898/0.620112`. That path is retained only as an
  explicit rejected ablation; it must not be used for this hypothesis. The
  official smoke instead exactly reproduced the readiness macro-step: aggregate
  per-class gradients, weight by the natural single-object prior, and apply one
  encoder-only update with equal L2 parameter-step norm (`1e-4`) for control and
  candidate.
- The official 12-per-class smoke was numerically well controlled. Objectness
  raw cosine was `-0.289796` with projected norm retention `0.957088`;
  localization cosine/retention was `0.229522/1.0`. Requested step norm was
  `0.009468466`, actual control/candidate parameter ratios were
  `9.999998e-5/1.000000e-4`, and absolute step-norm difference was only
  `2.315e-9`.
- Full validation rejected the method. Keeper, matched classification control,
  and spatial candidate macro/class1 F1 were respectively
  `0.882925/0.678261`, `0.874714/0.654867`, and `0.873916/0.652941`.
  Candidate versus control changed 16 rows with 7 corrections, 8 harms, and 1
  neutral; class1 FP remove/create was `2/3`, while FN rescue/TP break was
  `1/1`. It regressed both `0-1` and `1-2` pair accuracy and failed six fixed
  promotion checks, so no probe or test was run.
- Forensics initially reported impossible near-random accuracy because the
  prediction exporter emitted `label/prediction` while the shared audit expects
  canonical `target_index/prediction_index` or `y_true/y_pred`. The exporter and
  tests now require both canonical forms; repaired files reproduce direct
  evaluator accuracy exactly. Treat cross-tool prediction schema as a tested
  contract, not an implicit convention.
- Complete changed-case XAI explains why locally safe gradients still fail.
  Candidate versus keeper Grad-CAM foreground mass fell
  `0.910216 -> 0.890438`, border mass rose `0.210001 -> 0.242096`, and near ties
  rose `11/16 -> 16/16`. Grad-rollout foreground also fell
  `0.937680 -> 0.932045`; object-desaturation prediction drop weakened
  `0.058020 -> 0.046398`. Contact-sheet review found occasional useful surface
  corrections, but class0/class1 harms shifted evidence toward stems, hands,
  borders, and context. Full-frame localization therefore did not add the
  missing class1 surface representation.
- Decision: reject before probe. Do not sweep macro-step size, auxiliary ratio,
  detector warm-up, seed, query count, nearby PCGrad/GradNorm weighting, raw
  total detection gradients, or the microbatch AdamW path; the CLI now requires
  a separate explicit rejected-ablation opt-in. The matched
  classification direction itself leaves the keeper optimum, so this route may
  reopen only after a new representation changes direct keeper FN/FP support and
  a matched control first preserves keeper performance.
- Evidence is retained at
  `runs\smoke_joint_spatial_pcgrad_normalized_macrostep_s12_20260712`: 492
  payloads, `40,643,232` bytes, manifest SHA-256
  `37d39be82c398b7bc3201891adf0198d0b143d0b5090cb9af76444ec588eedc6`,
  no checkpoint/model/test payload. Guarded cleanup preserved checkpoint hashes,
  deleted only two rejected smoke checkpoints and four exact `%TEMP%` preflights,
  and reclaimed `178,580,777` bytes. Focused tests pass `11/11`, compileall and
  complete pytest pass `676/676`; retention
  `runs\artifact_retention_audit_after_joint_spatial_pcgrad_cleanup_20260712`
  covers 566 directories with `blockers=[]`. Keeper and current-best command
  remain unchanged; command SHA-256 is
  `3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.

## Diagnostic 2026-07-12 - BBN Bilateral Classifier Space Rejected Before Image Smoke

- Re-read the primary long-tail multi-branch literature before implementation:
  [BBN (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhou_BBN_Bilateral-Branch_Network_With_Cumulative_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2020_paper.html),
  [RIDE (ICLR 2021)](https://iclr.cc/virtual/2021/poster/3018),
  [ResLT](https://arxiv.org/abs/2101.10633), and
  [SADE (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/dc6319dde4fb182b22fb902da9418566-Abstract-Conference.html).
  These methods motivate separate experts for natural, inverse, nested-tail, or
  test-agnostic class distributions. The fixed precheck isolates the cheaper BBN
  classifier-space claim before allocating another image-model branch; it is not
  presented as a reproduction of any full paper.
- Added `trkh.tools.audit_bbn_bilateral_classifier_readiness` and six focused
  tests. The conventional branch is a natural-frequency multinomial logistic
  readout. The reversed branch exactly realizes BBN class sampling proportional
  to `1/N_i` as natural-row loss weights proportional to `1/N_i^2`; raw logits
  are fused once at the paper's fixed inference `alpha=0.5`. There is no C,
  alpha, weighting, fold, or validation sweep.
- Reused a frozen keeper cache over all `9,215/2,606` train/validation objects,
  `8,064/2,577` source groups, and 256D embeddings. A source-guard self-review
  added a hard zero-overlap check between train and validation as well as every
  OOF fit/hold fold. The final run has both overlap counts at zero, reads no test,
  writes no model/checkpoint, and leaves raw data untouched. The reusable
  `12,496,767`-byte cache is retained with payload SHA-256
  `86df861525b2fddd1d11b853d213104e0e39e41d2d16144d0e89136299e090e6`.
- BBN fusion has a small same-readout OOF effect but does not transfer to the
  direct keeper boundary. Conventional versus bilateral OOF macro/class1 F1 is
  `0.934635/0.802208 -> 0.936151/0.809365`; the macro gain `+0.001516`
  misses its fixed `0.002` gate. On validation, conventional, reversed, and
  bilateral class1 F1 are `0.644928/0.650602/0.666667`, while the direct keeper
  remains `0.684058`; bilateral macro F1 `0.880665` is also below keeper
  `0.884073`.
- Transition and direction audits reject the apparent false-positive control.
  Versus keeper, bilateral removes/creates `32/7` class1 false positives but
  rescues only one false negative while breaking 18 true positives. Its class1
  precision/recall is `0.664474/0.668874` versus keeper
  `0.608247/0.781457`. Same-scale bilateral-minus-conventional FN-versus-FP
  direction AUROC inverts from `0.636277` OOF to `0.396057` validation; keeper
  probability deltas are deliberately excluded because the probability scales
  are incompatible.
- Decision: six fixed checks fail and `image_smoke_permission=false`. Do not
  implement or sweep BBN branch alpha, reversed weight, logistic C, folds, or a
  same-representation RIDE/ResLT/SADE expert family. This closes the current
  frozen-embedding class-prior axis, not every future multi-expert architecture;
  reopen only after a genuinely new image representation independently preserves
  keeper macro/class1/recall and changes direct keeper FN/FP support.
- Final evidence is retained at
  `runs\diagnostic_bbn_bilateral_classifier_readiness_sourceguarded_20260712`:
  eight payloads, `6,199,651` bytes, manifest SHA-256
  `c66558746eb9817d461938fcae051b99a945e45b34ebffa01a38f0fbf4aea45b`,
  no model/checkpoint/test payload. Two guarded manifests removed the original
  scale-invalid and intermediate no-source-guard outputs, exactly 18 files and
  `12,401,130` bytes; observed free-space gain was `12,443,648` bytes. Retention
  then covered 570 run directories with `blockers=[]`. Keeper and current-best
  command remain unchanged. Closure passed py-compile, compileall, focused BBN
  tests `6/6`, complete pytest `682/682`, evidence/cache/cleanup manifest hash
  verification, direct plot review, and `git diff --check`; current-best command
  SHA-256 remains
  `3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.

## Diagnostic 2026-07-12 - Fixed Interior LBP Surface Texture Rejected Before Image Smoke

- Re-read the primary multiresolution uniform-LBP work
  (`https://doi.org/10.1109/TPAMI.2002.1017623`), its implementation details,
  and color-constant LBP before selecting a fixed diagnostic. This stage uses
  grayscale order patterns only to test whether local surface microtexture adds
  a stable signal beyond the keeper. It is intentionally distinct from the
  already rejected HOG, wavelet-scattering, high-frequency, and generic
  pretrained-feature routes.
- Added `trkh.tools.audit_lbp_surface_texture_readiness`, six focused tests, and
  the explicit `scikit-image==0.24.0` dependency. The protocol denormalizes the
  exact runtime crop, erodes the transformed object box by 15%, ROI-aligns a
  `128x128` interior, and concatenates rotation-invariant uniform LBP histograms
  at `(P=8,R=1)` and `(P=16,R=2)`. Histograms are normalized independently and
  the resulting 28D descriptor is evaluated with five source-grouped folds.
- Self-review rejected the first 4096-row preflight because a free bias plus
  summed cross-entropy allowed class-prior recalibration to dominate the LBP
  question. A corrected no-bias, natural-frequency mean-CE residual with
  `C=0.3` removed that confound, but its prefix remained class-biased. The final
  decision therefore uses all `9,215/2,606` train/validation rows and
  `8,064/2,577` source groups, with zero train/validation or OOF fit/hold source
  overlap. Test and raw-data writes are prohibited.
- The fixed descriptor is low-dimensional in practice: standardized effective
  rank is only `4.386422`. A standalone LBP readout reaches validation
  macro/class1 F1 `0.368847/0.000000`. Adding the zero-initialized LBP residual
  changes the train in-sample keeper/grouped-OOF residual macro-class1
  `0.939922/0.815729 -> 0.923667/0.814516` and validation
  `0.884073/0.684058 -> 0.864965/0.645963`; candidate class1 precision/recall
  is `0.608187/0.688742` versus keeper `0.608247/0.781457`.
- Full transition audit confirms another class1 suppressor. On validation it
  changes 115 rows with `29/68` corrections/harms, removes/creates `14/5`
  class1 false positives, rescues one false negative, and breaks 15 true
  positives. OOF changes are similarly unsafe: `64/226` corrections/harms and
  `1/25` FN-rescue/TP-break. FN-versus-FP direction AUROC is effectively random
  and stable in the wrong sense (`0.484949/0.460128` OOF/validation).
- The reviewed 13-row RGB/LBP8/LBP16 contact sheet shows speckles, scratches,
  wrinkles, lesions, shading boundaries, and smooth-skin microtexture, but
  class1 false positives and false negatives visibly overlap. Strong responses
  also follow shadow and crop-edge transitions. The descriptor therefore does
  not provide recall-safe maturity evidence even when restricted to the fruit
  interior.
- Decision: 15 fixed checks fail and `image_smoke_permission=false`. Do not
  sweep RGB/opponent LBP channels, points/radii, ROI erosion/size, residual C,
  optimizer iterations, folds/class balancing, or nearby LTP/CLBP variants.
  Reopen local order-pattern texture only after genuinely new representation or
  independent supervision changes direct keeper FN/FP support.
- Final evidence is retained at
  `runs\diagnostic_lbp_surface_texture_full_20260712`: 20 payloads,
  `6,496,709` bytes, manifest SHA-256
  `195c8a0ea8b337a2afb955a5585a1eb09b921b54cd70c29cebb632a6c9f52919`,
  no model/checkpoint/test payload. It includes compact protocol/fold/summary
  evidence for both superseded preflights. Guarded cleanup manifest
  `runs\cleanup_manifest_20260712_lbp_surface_preflights_superseded.json`
  deleted exactly 20 original files (`6,665,614` bytes; observed free-space
  gain `6,713,344` bytes). Retention then covered 572 directories with
  `blockers=[]`; keeper and current-best command remain unchanged.
- Closure passed py-compile, compileall, focused LBP tests `6/6`, complete
  pytest `688/688`, artifact/cleanup JSON and payload-hash verification,
  `git diff --check`, and explicit protected-path worktree review. Current-best
  command SHA-256 remains
  `3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.

## Diagnostic 2026-07-12 - BioCLIP Domain Representation Rejected Before Transfer Smoke

- Re-read the primary [BioCLIP CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Stevens_BioCLIP_A_Vision_Foundation_Model_for_the_Tree_of_Life_CVPR_2024_paper.html),
  [official repository](https://github.com/Imageomics/bioclip), official
  [Hugging Face release](https://huggingface.co/imageomics/bioclip), and
  [OpenCLIP](https://github.com/mlfoundations/open_clip). This stage asks one
  narrow question: does TreeOfLife-10M domain adaptation add a transferable
  biological surface signal beyond a capacity- and pixel-matched OpenAI CLIP
  control? It is a frozen diagnostic, not a proposal to replace scratch TRKH
  with a pretrained final model.
- Added `trkh.tools.audit_bioclip_domain_feature_readiness`, nine focused
  tests, and locked `open_clip_torch==2.32.0`. The exact official revisions are
  OpenAI control `977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d` and BioCLIP
  `ce901ab3c6a913f9e9ef94ce6d27761069f4f01c`; checkpoint SHA-256 values are
  `4b869929...02b820f` and `e380384f...5f0ef4`. Both vision towers have
  `86,192,640` parameters, emit normalized 512D features, and use bit-identical
  224px transforms. Loading now fails closed if either downloaded checkpoint
  hash differs or if a non-class1 focus is requested. The official OpenAI
  release uses QuickGELU while BioCLIP
  uses GELU, so the comparison is capacity/pixel matched but not claimed to
  isolate training domain from activation choice.
- Protocol self-review caught two alignment hazards before accepting evidence.
  `ImageFolder` numeric class IDs cannot be compared directly with YOLO IDs
  because folder order is alphabetical; the guard now compares canonical class
  names. Source groups are also canonicalized case-insensitively before overlap
  checks. The final run covers all `9,215/2,606` train/validation objects and
  `8,064/2,577` source groups, with zero train/validation and fold fit/hold
  overlap. Test, raw-data writes, prompts, class weights, validation selection,
  model/checkpoint output, and trainable manifests are absent.
- BioCLIP looks mildly better only inside grouped train OOF. OpenAI versus
  BioCLIP OOF macro/class1 is `0.832440/0.447581 -> 0.839207/0.474383`, gains
  `+0.006767/+0.026803`, with four of five folds improving class1. The effect
  reverses on independent validation: `0.851003/0.552448 ->
  0.819665/0.444444`, while direct keeper remains `0.884073/0.684058`.
  BioCLIP class1 precision/recall is only `0.467153/0.423841` versus keeper
  `0.608247/0.781457`.
- Full transitions reject transfer. Against keeper, BioCLIP changes 281 rows
  with `93/169` corrections/harms, removes/creates class1 FP `49/46`, and
  rescues/breaks only `8/62` FN/TP. BioCLIP-minus-OpenAI class1 probability does
  show stable FN-versus-FP AUROC `0.828627/0.854911` on OOF/validation, but this
  is a control-relative direction. It is not evidence that the same delta is a
  valid keeper-relative action.
- A post-audit inventory verified that
  `runs\yolof_oof_folds_train5_20260704` contains five source-grouped hardlink
  data splits only. There are no fold checkpoints or prediction payloads for
  the exact current keeper, and the reusable keeper train cache is explicitly
  in-sample. A BioCLIP residual/router fitted against those keeper probabilities
  would therefore be protocol-invalid. The attractive domain-direction AUROC
  does not authorize another residual, router, threshold, or distillation
  smoke.
- Reviewed the complete 12-row mean-occlusion contact sheet. OpenAI positive
  class1 evidence has mean center mass `0.442268`; BioCLIP falls to `0.162520`.
  OpenAI usually uses central fruit surface, whereas BioCLIP frequently shifts
  evidence to fruit edges, corners, outer context, or diffuse periphery. It can
  rescue isolated class1 cases but breaks many real class1 examples and creates
  class1 predictions on yellow/overripe fruit. The learned bias is consistent
  with organism taxonomy/shape, not a stable mango-ripeness interior-surface
  cue.
- Decision: nine fixed checks fail and
  `representation_transfer_smoke_permission=false`. Do not sweep BioCLIP
  versions/image sizes, prompts, readout C/class weights/folds, residual
  coefficients, routers/validation thresholds, or feature-distillation
  adapters. Reopen only after true source-grouped current-keeper OOF
  predictions exist and a fixed train-only keeper-relative action independently
  preserves validation class1 recall.
- Final evidence remains at
  `runs\diagnostic_bioclip_domain_feature_full_20260712`: nine payloads,
  `5,908,596` bytes, manifest SHA-256
  `3ebb6b8a0f90e96ceb0908d0fa005e97c2b964450caeaeb97893f3ad7ec4b1c1`,
  no model/checkpoint/test payload. Guarded cleanup
  `runs\cleanup_manifest_20260712_bioclip_embedding_cache_rejected.json`
  removed only the reproducible 34,215,279-byte feature matrix, with observed
  free-space gain `34,140,160` bytes. Retention
  `runs\artifact_retention_audit_after_bioclip_domain_cleanup_20260712` passed
  over 574 directories with `blockers=[]`. Keeper and current-best command are
  unchanged; command SHA-256 remains
  `3c71813000970a9776cfde974895358be2dda214ca9d6eb0e6a89dbc31d657dd`.
- Closure passed py-compile, compileall, focused BioCLIP tests `9/9`, complete
  pytest `697/697`, JSON/payload-hash verification, `git diff --check`, and
  explicit protected-path review. No full train, test evaluation, deploy
  pointer, or current-best command update was permitted. `pip check` reports
  only the already documented shared-environment MambaVision pin mismatches and
  OpenCV's NumPy>=2 requirement; OpenCLIP introduced no additional conflict.

## Diagnostic 2026-07-12 - Deep-TEN Stem-Interior Texture Rejected Before Image Smoke

- Re-read the primary Deep TEN CVPR 2017 and DEP CVPR 2018 papers and inspected
  the official `PyTorch-Encoding` implementation at exact commit
  `ac748410dfc8d7d70a2ce7f5add08050af2fae20`. This fixed diagnostic tests the
  official learnable residual-encoding operation on the frozen keeper's final
  CNN stem; it is distinct from prior LBP/HOG/wavelet, bilinear/covariance,
  hard-cluster histogram, and high-frequency expert failures.
- Added `trkh.tools.audit_deepten_stem_texture_readiness` and 11 focused tests.
  The candidate uses a `12x12` final-stem grid, fixed orthogonal `256->32`
  projection, 12%-eroded bbox/valid-mask interior, `K=8` masked Deep-TEN,
  64D readout, and a zero-init no-bias keeper-log-probability residual. The
  matched control uses mean+std over exactly the same descriptors. Both use
  natural-frequency batches, five source folds, AdamW, and 20 epochs.
- The first class-biased prefix preflight found one eight-token eroded mask.
  Self-review replaced the unsafe all-valid fallback with the nine nearest
  valid bbox-center tokens, added a regression test, and reran the protocol
  preflight. Throughput improved from `178.46s` to `81.82s` with four workers
  and batch 192. Prefix metrics remained explicitly non-decisional.
- The full no-test run covers all `9,215/2,606` train/validation rows and
  `8,064/2,577` source groups with zero train/validation or OOF fit/hold source
  overlap. Deep-TEN is worse than its matched control in all five folds. Direct
  keeper/control/candidate validation macro-class1 is
  `0.884675/0.686047 -> 0.794480/0.363636 -> 0.779022/0.316327`; candidate
  class1 recall collapses from `0.781457` to `0.205298`.
- Candidate-versus-keeper transitions are decisively unsafe: `82/199`
  corrections/harms, class1 FP remove/create `62/1`, and FN-rescue/TP-break
  `0/87`. Codeword-use entropy recovers to `0.896705`, but maximum codeword
  cosine is `0.995986` and final residual magnitude is large. The branch learns
  a global class1 suppressor rather than missing positive maturity evidence.
- XAI confirms the same failure. Candidate positive center mass falls
  `0.356843 -> 0.278909` versus control and border mass rises
  `0.030891 -> 0.040745`; multiple true class1 examples receive broad negative
  surface evidence. Candidate-minus-control FN/FP AUROC `0.848099/0.922566`
  cannot be reused as a keeper action because candidate-minus-keeper validation
  AUROC is `0.446869` and exact keeper OOF predictions do not exist.
- Decision: ten locked checks fail and image smoke is closed. Do not sweep
  codewords/grid/projection/erosion/readout/LR/residual scale/folds/epochs, and
  do not fit a router, threshold, or calibrator from this control-relative
  signal. The current-best command remains unchanged.
- Final evidence is
  `runs\diagnostic_deepten_stem_texture_full_20260712`: 11 payloads,
  `5,671,831` bytes, manifest SHA
  `24ee35631bade5d70a1b9bca8714614d45eb81f1e2b983f6cf9ff4b852b46924`,
  no model/checkpoint/test. A guarded two-phase manifest removed both prefix
  roots, exactly 20 files and `5,866,265` bytes, with observed free gain
  `5,902,336` bytes. Retention passed across 576 directories with no blockers;
  the exact official reference clone was also deleted after commit verification.
- Closure passed py-compile, compileall, focused tests `11/11`, full pytest
  `708/708`, artifact/cleanup/retention hash verification, staged
  `git diff --check`, and explicit protected-path review. Keeper and current-best
  command SHA values remain unchanged.

## Diagnostic 2026-07-12 - VICRegL Crop-Local Objective Rejected Before Trainer Smoke

- Re-read the VICRegL NeurIPS 2022 paper and inspected its official code at
  commit `803ae4c8cd1649a820f03afb4793763e95317620`. Unlike prior global VICReg,
  pixel MIM, paired `class_f/yolo_f` feature consistency, and rejected hflip
  consistency, VICRegL explicitly matches local features by original-image
  coordinates and feature distance. The official recipe needs 100-400 epochs
  and large batches, so a fixed readiness audit was required under TRKH's
  30-epoch/8-GB constraint.
- Added `trkh.tools.audit_vicregl_crop_match_readiness` and nine focused tests.
  The frozen no-test protocol makes two fixed overlapping 240px crops from each
  256px runtime object view, transforms bbox/valid-mask metadata exactly, maps
  post-pruning tokens through `patch_indices` to source coordinates, and keeps
  the official 20 closest symmetric interior location matches. It writes no
  projector, model, checkpoint, cache, or trainable manifest.
- A 128-row implementation pilot and class-biased 1,024/512 preflight were
  protocol-only. They verified geometry, XAI, throughput, and fixed thresholds
  before the sole decision run. Full support is `9,215/2,606` rows and
  `8,064/2,577` groups with zero train/validation overlap.
- The final patch representation is already locally equivariant. Train/validation
  location cosine is `0.966519/0.965089`, and feature retrieval of the
  location-matched token is `0.893692/0.895635`; every row supplies all 20
  matches. Class1 FN mismatch is lower than TP mismatch
  (`-0.003113/-0.001721` FN-minus-TP), while FN-vs-FP AUROC is only
  `0.470264/0.411313`.
- Crop views are not a recall-safe action. Direct keeper versus crop-average
  validation macro-class1 changes `0.884675/0.686047 -> 0.881340/0.670487`.
  Among 78 changed rows there are `35/38` corrections/harms, class1 FP
  remove/create `12/18`, and FN-rescue/TP-break `5/6`.
- The reviewed 12-case mismatch sheet has mean mismatch `0.010515`; border mass
  `0.215986` exceeds center mass `0.149665`. Residual mismatch follows crop
  edges, padding, hands, stems, and object boundaries rather than missing
  interior maturity evidence. A local consistency loss would optimize nuisance
  invariance without targeting the class1 FN/FP boundary.
- Decision: 11 fixed checks fail and trainer/image smoke is closed. Do not sweep
  crop geometry, gamma, thresholds, projector, local/global weight, optimizer,
  or epochs, and do not use mismatch as a router/sample-weight/margin signal.
  Current-best commands remain unchanged.
- Final evidence is
  `runs\diagnostic_vicregl_crop_match_full_20260712`: nine payloads,
  `4,791,831` bytes, SHA
  `f249f51c8302454269ec17ae7253d44864d659ef110bf179e327d512170f6281`,
  no model/checkpoint/test. A guarded manifest deleted the one superseded
  preflight root, eight files and `1,776,284` bytes, with observed free gain
  `1,794,048` bytes. Retention passed across 578 directories with no blockers;
  the exact official reference clone was removed after commit verification.
- Closure passed py-compile, compileall, focused tests `9/9`, full pytest
  `717/717`, artifact/cleanup/retention hash verification, staged
  `git diff --check`, and explicit protected-path review. Keeper and current-best
  command SHA values remain unchanged.

## Diagnostic 2026-07-12 - Original-Resolution Surface Tiles Rejected

- No-repeat architecture triage rejected a blind FastViT-SA12 launch. Local
  `timm 1.0.27` defines layers `2/2/6/2`, RepMixer in the first three stages,
  attention only in the last stage, and default RepMixer kernel 3. It is another
  staged stock hybrid after the controlled stock-backbone closure, not a new
  class1-positive target.
- The retained hypothesis tested whether the keeper's 256 px global crop loses
  local source-resolution evidence. The fixed no-test tool uses exact global
  checkpoint preprocessing plus five 70%-scale original-resolution bbox-crop
  tiles. Bbox/crop-bbox/valid-mask/token-prior metadata remain aligned; no model,
  threshold, or weight is fitted. Fixed fusion is equal global/tile-mean
  probability averaging.
- Full coverage is `9215/2606` rows and `8064/2577` source groups with zero
  train/validation overlap. Global validation reproduces keeper macro/class1
  `0.884675/0.686046`; tile mean reaches only `0.862581/0.629080`, and fixed
  fusion falls to `0.880824/0.666667`, class1 P/R `0.601064/0.748344`.
- Fusion changes 48 validation decisions: 21 corrections, 24 harms, and 3
  neutral changes. It removes/creates class1 FP `10/10`, rescues/breaks FN/TP
  `0/5`, and fails six behavioral gates. Core smoke permission is false.
- Tile-minus-global p1 direction is stable but non-actionable: FN-vs-FP AUROC
  is `0.844347/0.840000` train/validation, while mean FN/FP deltas are
  `+0.003567/-0.034688` and `+0.011426/-0.029362`. A fixed matched binary
  readout sanity check remains below keeper (`0.869291/0.640212` validation),
  so the AUROC does not authorize a router or residual.
- Contact-sheet review shows correct geometry and magnified fruit surface,
  spots, color transitions, lesions, glare, stems, and crop context, but no
  stable class1-positive tile. Only `9.27%` of validation class1 crops have raw
  minimum side at least 384 px, and class1 FN are not concentrated in small
  crops. Global resize loss is not the missing signal.
- Decision: do not sweep tile fraction/positions/count, fusion weight, image
  size, crop thresholds, or tile routers. A next local route must change the
  spatial objective itself; Finer-CAM-style target-versus-confuser attribution
  is distinct enough for a locked no-test readiness audit before training.
- Evidence `runs\diagnostic_highres_surface_tiles_full_20260712` contains eight
  payloads (`8462921` bytes), SHA
  `010d368e87ab3214a5b1f09f0bcc048032890e5c8663bc901a191a85e0760de2`,
  no model/checkpoint/test. The superseded preflight was deleted after full
  preservation. Its cleanup manifest retains an explicit PowerShell 5.1
  provenance warning because the first hash-capture script used an unavailable
  API; no hashes were fabricated.
- Closure passed py-compile, compileall, focused tests `7/7`, full pytest
  `724/724`, stable payload-manifest verification, and retention across 580
  run directories with `blockers=[]`. Keeper and current-best command hashes
  remain unchanged.

## Diagnostic 2026-07-12 - Finer-CAM Class-Contrastive Attribution Rejected Before Smoke

- Re-read the Finer-CAM CVPR 2025 paper and current official implementation.
  Unlike every prior standard Grad-CAM/rollout audit, the fixed target
  differentiates class-1 logit contrast against three per-sample logit-nearest
  references before ReLU. The current official probability-weighted objective
  and default `alpha=1` were locked without comparing the paper's experimental
  `gamma=0.6`.
- Added a no-test FP32 readiness audit plus a batch-context-preserving changed-
  case reviewer and 13 focused tests. It uses the existing `patch_embed.proj`
  XAI layer, exact bbox/crop-bbox/valid-mask/token-prior metadata, and mean-RGB
  deletion of the top 5% valid pixels. Ground truth never selects the target,
  reference, heatmap, mask, or fixed action.
- Full support is `9215/2606` rows and `8064/2577` source groups with zero
  overlap. Finer relative confidence drop exceeds standard Grad-CAM on train
  `0.005317 > 0.005041` and validation `0.006777 > 0.006346`, but the contrast-
  gain FN-vs-FP AUROC is only `0.574541/0.616501`; direction does not pass both
  splits.
- The one fixed top-2 class1 residual changes FP32 keeper validation
  macro/class1 `0.882925/0.678261 -> 0.884183/0.682216`, class1 precision
  `0.603093 -> 0.609375`, and leaves recall `0.774834`. Four gates still fail:
  no `+0.005` class1 gain, no `0.70` milestone, no train-direction transfer,
  and no positive recall action.
- All 14 validation transitions were reviewed after exact batch-32
  reproduction over 334 context rows (maximum RD difference `6.56e-7`): 8/6
  corrections/harms, class1 FP remove/create `6/4`, and FN-rescue/TP-break
  `2/2`. Finer-CAM lowers validation interior mass
  `0.358258 -> 0.323801` and raises border mass `0.181833 -> 0.191826`.
  Visual maps emphasize silhouettes, endpoints, stems, hands, padding, bright
  boundaries, and broad color bands rather than a stable class1-positive cue.
- The prior tile anchor difference is explained, not ignored: tile inference
  used AMP and reproduced `0.884675/0.686046`; this gradient audit uses FP32 and
  reproduces the locked independent reload `0.882925/0.678261`. Labels/paths
  are identical; only six predictions differ from AMP. Candidate comparison is
  strictly against the matched FP32 control.
- Decision: `13/17` checks pass but smoke remains closed. Do not sweep alpha,
  layer, references, mask settings, CAM backend, residual scale, thresholds, or
  fit a router/filter. No current-keeper OOF signal exists, and a val-derived
  `1->0` suppression rule would repeat an already rejected validation trap.
- Retained full evidence
  `runs\diagnostic_finer_cam_class1_full_20260712` (8 payloads,
  `8770683` bytes, SHA `dfa7a1a5...40db`) and exact changed-case review
  `runs\review_finer_cam_class1_full_changed14_batch32_v2_20260712` (4 payloads,
  `1032981` bytes, SHA `10f48a65...67b9`), with no model/checkpoint/test. Keeper
  and current-best command SHA values remain unchanged.
- Two exact-hash cleanup manifests removed 41 superseded files (`3312956`
  bytes; observed free gain `3383296` bytes), including both implementation
  prefixes, the throughput prefix, selected-only review, pre-guard batch-32
  review, and obsolete retention snapshot. Final retention passed across 583
  directories with `blockers=[]`.
- Closure passed py-compile, compileall, focused tests `13/13`, full pytest
  `737/737`, artifact/cleanup/retention hashes, `git diff --check`, and explicit
  protected-path review. Current-best commands remain unchanged.
