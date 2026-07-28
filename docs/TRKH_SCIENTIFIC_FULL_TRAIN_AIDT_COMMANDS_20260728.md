# TRKH scientific full-train, audit, AIDT/TIMM inference commands

Date: 2026-07-28; scientific-state revision: 2026-07-29

Repository state reviewed through DDF engine commit
`fb9880c77057d887d9e3b3ab91e9dbb55e47cfe9`, resume/promotion fix commit
`2456dde06e9a2aa6de15c0f60a0af5c951beb1d6`, and pretrained-export safety
commit `2ee8d032f0c82bb8ada4bcbdc3ceae1ac685ddd2`. The v1-supersession/v2-fold/
protocol/geometry/presentation boundary is pushed at
`742fce03700b17f8e8f4449eac8a49dc0a03e211`. Registry schema v2 is derived
only from that Git snapshot; its current SHA-256 is
`e1b2b7de5aaf262bb3674fdc8a5694699cdec3458d189f3088483aec55bb8d93`
and `authorizes_formal_or_gpu=false`.

Machine used to size the commands:

- NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB VRAM;
- Intel Core i7-12700H, 14 cores / 20 logical processors;
- 15.6 GiB RAM;
- TRKH run `full_v8_yolof_current_best_30e_20260728_184850` measured about 5,975 MiB peak allocated and 7,250 MiB peak reserved with batch 32 and accumulation 2.

The commands below therefore use TRKH batch `32`, accumulation `2`, train workers `4`, evaluation workers `2`. The only OOM fallback that preserves effective batch 64 is batch `16`, accumulation `4`.

## 1. Forensic verdict for run 20260728_184850

The run did not fail to optimize for 30 epochs. It was stopped correctly at epoch 5 by patience 3. The apparent plateau has four concrete causes.

1. It was a warm start from an already strong 2-epoch checkpoint, not a random full train. The source checkpoint already had training-time validation macro/class-1 F1 `0.884675/0.686046` and fair selection score `0.7629487169`.
2. `--resume-reset-scheduler` did not actually leave the new scheduler at its initial state. `_load_training_checkpoint` unconditionally stepped the new scheduler to the source checkpoint epoch. The first child epoch therefore ran at `8e-5`; with a warmup start factor of `0.1`, the intended initial LR was `8e-6`.
3. Resetting the phase epoch replayed the curriculum. Attention-view training was off at child epoch 1 and jumped to about `0.6015` of samples at epoch 2. The recorded aggregate train loss consequently jumped from `0.9868` to `1.7137`; the two losses are not directly comparable because an auxiliary loss became active.
4. The child checkpoint comparator started fresh instead of comparing against the source checkpoint. The child best fair score was only `0.7559386593`, below the source `0.7629487169`, but it was still named `best.pt` inside the child phase. Macro F1 alone rose slightly to `0.885912`, while the class-1 trade-off worsened and the fair score did not improve.

Observed child history:

| Epoch | LR | Macro F1 | Class-1 F1 | Fair selection |
|---:|---:|---:|---:|---:|
| source | - | 0.884675 | 0.686046 | 0.762949 |
| 1 | 8.000e-5 | 0.884111 | 0.680233 | 0.751187 |
| 2 | 7.977e-5 | 0.885912 | 0.681818 | 0.755939 |
| 3 | 7.908e-5 | 0.881678 | 0.666667 | 0.730845 |
| 4 | 7.793e-5 | 0.880207 | 0.670360 | 0.738052 |
| 5 | 7.635e-5 | 0.879915 | 0.672131 | 0.744274 |

Conclusion: the source checkpoint remains the keeper. Do not promote the child run merely because its raw macro F1 was slightly higher in one evaluation.

## 2. Correctness changes now applied

- `trkh/training/train.py`
  - a reset scheduler is no longer stepped with the old checkpoint epoch;
  - `best.pt` receives `checkpoint_kind=best` without mutating the metadata saved in `last.pt`.
- `scripts/run_trkh_5class_attention_views_v8.ps1`
  - explicit modes: `Scratch`, `WarmStart`, `StatefulResume`;
  - scratch clears the historical checkpoint default and rejects any
    explicitly supplied non-empty checkpoint;
  - warm start loads weights but resets phase/optimizer/scheduler/scaler;
  - stateful resume must use the same run's `checkpoints\last.pt` and restores state.
  - when not explicitly overridden, a warm-start phase keeps attention-view
    training active from phase epoch 1 instead of replaying an off/on curriculum.
- `scripts/run_trkh_current_best_full_pipeline.ps1`
  - exposes the hardware-sensitive training parameters;
  - rejects reuse of an existing directory for scratch/warm-start runs;
  - refuses stateful resume of a normally completed run;
  - test/promotion now requires independent raw keeper thresholds and the exact fair metric name, direction, and score.
- Ten direct scratch/OOF launchers now pass their resume mode explicitly.
- AIDT and TIMM validation-only trainers no longer construct a test dataset.
- AIDT/TIMM prediction exporters require `--allow-test` with `--split test`.
- AIDT/TIMM training uses a fresh output directory so artifacts from separate runs cannot be mixed.
- The legacy comparison-table join was changed from row order to normalized path identity and now derives the focus-class metrics from predictions.

## 3. Scientific role of each model

Use these roles in the thesis/paper:

- **TRKH current v8**: reproducibility/control model. A continuation of it is not a new architecture.
- **Research candidate direction**: a scratch DDF sidecar with pairwise evidence
  heads for TRKH. DDF itself is an existing operator; the defensible potential
  contribution is the application/composition, causal controls, and
  reproducibility protocol, not a new general DDF operator.
- **AIDT and TIMM**: pretrained external comparison baselines only. They must not enter the TRKH inference graph or provide inference-time inputs.

Pair-Surface DDF currently has a synthetic engineering status of
`passed_mechanism_unproven`. The 9,380-parameter sidecar, FP64 oracle, ONNX
Runtime parity, TensorRT build, and synthetic causal checks pass. However, the
2026-07-25 A0 fold lock is superseded before formal execution. Although exact
source stems are disjoint, 150 of 531 dataset `leakage_group` values cross
folds, covering 330 cohort rows; numeric `Image_N` window-3 adjacency creates
974 cross-fold row pairs covering 659/763 rows. No formal or replay run was
consumed. The v1 lock must never be used for an OOF claim.

The formal runner/replay/authorization and production integration do not yet
exist. Therefore there is deliberately no executable DDF full-train command in
this document. Claiming otherwise would turn a prospective composition into an
unvalidated implementation claim.

The first three prospective, non-GPU steps are now implemented locally and
await their containing pushed boundary: immutable v1 supersession, the
candidate-blind 158-component v2 fold manifest, and the fully numeric v2
protocol/geometry preflight. They consumed no candidate score, validation,
test, formal run or replay. The remaining allowed scientific sequence is:

1. commit/push the pinned no-repeat registry and update the installed skill
   from that pushed lineage;
2. implement the v2 engine, machine lock, access ledger, authorization and
   exact replay harness without reading validation/test;
3. execute exactly one formal train-only A0 plus one replay only after all
   preflight gates pass;
4. integrate DDF with a parameter-matched static control;
5. run matched scratch smoke and then a full-validation probe with focus-class
   F1 at least `0.70` and protected precision/recall;
6. run locked full static and DDF roles plus XAI/robustness/inference gates;
7. use a newly sealed source/time/site holdout for confirmation. The legacy
   test already seen in earlier work cannot become a new paper-final test.

## 4. Verification and preflight

```powershell
Set-Location D:\DataAI\AIEx\TRKH
$py = 'D:\DataAI\.venv\Scripts\python.exe'
$stamp = Get-Date -Format yyyyMMdd_HHmmss

& $py -m pytest `
  tests\test_detection_calibration.py `
  tests\test_v8_launcher_scheduler.py `
  tests\test_current_best_pipeline_scripts.py `
  tests\test_pretrained_export_safety.py `
  tests\test_audit_validation_information_ceiling.py `
  tests\test_audit_pair_surface_ddf_fold_leakage.py `
  tests\test_build_trkh_pair_surface_ddf_v2_fold_manifest.py `
  tests\test_audit_pair_surface_ddf_v2_geometry_metadata.py `
  tests\test_audit_trkh_artifact_retention.py -q
if ($LASTEXITCODE -ne 0) { throw 'Focused tests failed.' }

# Reproduce the train-table-only v1 leakage evidence in a fresh artifact.
& $py -m trkh.tools.audit_pair_surface_ddf_fold_leakage `
  --cohort-arrays .\runs\audit_cross_colour_ratio_surface_a0_materialized_20260725\cohort_arrays.npz `
  --yolo-manifest D:\DataAI\AIEx\newdataset\yolo_f\manifest.csv `
  --output-json ".\runs\audit_pair_surface_ddf_v1_fold_leakage_recheck_$stamp\summary.json"
if ($LASTEXITCODE -ne 0) { throw 'V1 fold-leakage audit failed.' }

# Replay-check the candidate-blind v2 mapping; default/check-only writes nothing.
& $py .\scripts\build_trkh_pair_surface_ddf_v2_fold_manifest.py --check-only
if ($LASTEXITCODE -ne 0) { throw 'V2 fold-manifest replay failed.' }

# Reproduce the validity/bbox audit, including the wrong-endian regression.
& $py -m trkh.tools.audit_pair_surface_ddf_v2_geometry_metadata `
  --cohort-arrays .\runs\audit_cross_colour_ratio_surface_a0_materialized_20260725\cohort_arrays.npz `
  --valid-masks-packbits .\runs\audit_cross_colour_ratio_surface_a0_materialized_20260725\image_valid_masks_packbits.npy `
  --output-dir ".\runs\audit_pair_surface_ddf_v2_geometry_recheck_$stamp"
if ($LASTEXITCODE -ne 0) { throw 'V2 geometry audit failed.' }

& powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_trkh_current_best_full_pipeline.ps1 `
  -RunName "preflight_resume_fixed_$stamp" `
  -TrainingMode WarmStart `
  -PreflightOnly
if ($LASTEXITCODE -ne 0) { throw 'TRKH preflight failed.' }
```

## 5. Warm-start correction status

The scheduler/reset defect is covered by source inspection and focused tests, but no post-fix warm-start run has yet exercised it end to end. Do not spend six epochs on the previous warm-start command for the comparison table: it is neither a random-initialized model nor a one-variable scientific ablation, and the trainer does not compare the child checkpoint against the source checkpoint at epoch zero.

The coherent fix set is committed at `2456dde`. If runtime validation is later
required, define a separate bounded engineering-only smoke. It must keep test
locked and must not be presented as a new model result. There is intentionally
no recommended GPU command here.

## 6. Closed scratch-natural branch and retained scratch baseline

The former `-DisableBalancedEpochSampling` probe was executed and rejected:

- run: `runs\probe_v8_scratch_natural_120b_10e_20260728_223907`;
- execution: completed 10 epochs, 120 batches per epoch, no resume, no test;
- independent full validation: macro/class-1 F1 `0.754579/0.246154`;
- class-1 precision/recall `0.545455/0.158940`;
- pipeline status: `validation_gate_rejected`.

This reproduces the already-closed July 20--21 natural-only result (`0.767070/0.291262` after five full-data epochs). The rare class is suppressed by insufficient exposure. Do not resume this run, repeat it with more epochs, sweep the sampler, or launch the deleted 24-epoch command.

The retained full random-initialized student baseline already exists at:

```text
D:\DataAI\AIEx\TRKH\runs\full_v8_yolof_randominit_30e_20260714_105524\checkpoints\best.pt
```

It used strict balanced train exposure, stopped at epoch 23 with best epoch 20, and reached independent validation macro/class-1 F1 `0.874172/0.654639`. It has full validation confusion, boundary, forensics, robustness, and XAI artifacts under `final_audit_manual`. Report its provenance precisely: random-initialized student, no external pretrained weights, with a fixed train-only internal no-pretrain TRKH teacher target.

This run is a historical TRKH scratch baseline. It is **not** the parameter-matched static sidecar required by the Pair-Surface DDF protocol. The latter must be integrated and locked together with the DDF candidate after formal A0 authorization.

## 7. Historical Pair-Surface DDF v1 engineering check

This is synthetic-only. It does not train or read train/validation/test images.

```powershell
Set-Location D:\DataAI\AIEx\TRKH
$py = 'D:\DataAI\.venv\Scripts\python.exe'
$stamp = Get-Date -Format yyyyMMdd_HHmmss

& $py -m pytest `
  tests\test_pair_surface_ddf_a0_engine.py `
  tests\test_preflight_pair_surface_ddf_a0.py `
  tests\test_pair_surface_ddf_a0_engineering_evidence.py -q
if ($LASTEXITCODE -ne 0) { throw 'DDF engineering tests failed.' }

& $py -m trkh.tools.preflight_pair_surface_ddf_a0 `
  --output ".\runs\audit_pair_surface_ddf_a0_synthetic_$stamp\summary.json"
if ($LASTEXITCODE -ne 0) { throw 'DDF synthetic preflight failed.' }
```

Stop here. This command is retained only to reproduce synthetic engineering.
It does not authorize a formal run against the superseded v1 fold lock. Build
and commit the manifest-group/numeric-neighborhood-disjoint v2 protocol and
lock first.

The check was rerun on 2026-07-29 and passed (`14/14` tests plus preflight): equation max error `8.88e-16`, ONNX Runtime max error `1.49e-8`, TensorRT engine `900,212` bytes, batch-1 mean/p95 `0.175/0.219 ms`, and batch-32 mean/p95 `0.547/0.744 ms`. Artifact:

```text
D:\DataAI\AIEx\TRKH\runs\audit_pair_surface_ddf_a0_synthetic_20260729_003200\summary.json
```

This does not change the status from `passed_mechanism_unproven` because no train-only candidate metric, formal replay, or production integration was created.

## 8. Pretrained comparison inventory and selected benchmark

`D:\DataAI\AIDT` contains one main architecture: ResNet50 + ViT-B/16 feature fusion. The many other pretrained directions are in `D:\DataAI\AIEx\image_baseline_experiments`, whose catalog contains 18 TIMM models, 2 MambaVision variants, and HOG.

For the current main table, use the compact set that already has complete checkpoints:

1. ResNet-50;
2. MobileNetV3-Large;
3. EfficientNetV2-S;
4. ConvNeXt-Tiny;
5. AIDT ResNet50 + ViT-B/16 fusion.

Add both the deployable TRKH keeper and the completed random-initialized TRKH full run with distinct labels. Standalone ViT-B/16 and Swin-Tiny probes were rejected (`0.7791/0.3303` and about `0.8343/0.4519` macro/class-1 F1); they are appendix negative evidence, not complete main-table baselines. Train a standalone transformer only if a reviewer-facing, prospectively registered coverage rationale requires it. VGG, DenseNet, Inception, EfficientNet-B0/B3, MobileNetV2 and the remaining catalog models can stay in an appendix.

## 9. Existing complete TIMM checkpoints and optional reproducibility rerun

The four selected TIMM checkpoints already completed 30 epochs and do not need retraining:

```text
D:\DataAI\AIEx\image_baseline_experiments\outputs\resnet50\best.pt
D:\DataAI\AIEx\image_baseline_experiments\outputs\mobilenetv3\best.pt
D:\DataAI\AIEx\image_baseline_experiments\outputs\efficientnetv2_s\best.pt
D:\DataAI\AIEx\image_baseline_experiments\outputs\convnext_tiny\best.pt
```

The following block is an optional fresh reproducibility rerun, not the recommended next GPU action. Keep test locked and use a new output root.

```powershell
Set-Location D:\DataAI\AIEx\image_baseline_experiments
$py = 'D:\DataAI\.venv\Scripts\python.exe'
$train = '.\scripts\02_train_timm_classifier.py'
$data = 'D:\DataAI\AIEx\newdataset\class_f'
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$outRoot = "D:\DataAI\AIEx\image_baseline_experiments\outputs_paper_clean_$stamp"
if (Test-Path -LiteralPath $outRoot) { throw "Output exists: $outRoot" }
$env:HF_HOME = 'D:\DataAI\AIEx\image_baseline_experiments\.hf_cache'
$env:TORCH_HOME = 'D:\DataAI\AIEx\image_baseline_experiments\.torch_cache'

$runs = @(
  @{ id='resnet50';       model='resnet50.a1_in1k';                              paper='ResNet-50';        family='Residual CNN' },
  @{ id='mobilenetv3';    model='mobilenetv3_large_100.ra_in1k';                 paper='MobileNetV3-Large'; family='Mobile CNN' },
  @{ id='efficientnetv2_s'; model='tf_efficientnetv2_s.in21k_ft_in1k';           paper='EfficientNetV2-S'; family='Efficient CNN' },
  @{ id='convnext_tiny';  model='convnext_tiny.fb_in22k_ft_in1k';                paper='ConvNeXt-Tiny';    family='Modern CNN' }
)

foreach ($r in $runs) {
  & $py $train `
    --data $data `
    --model $r.model `
    --paper-name $r.paper `
    --family $r.family `
    --outdir "$outRoot\$($r.id)" `
    --epochs 30 `
    --batch-size 16 `
    --workers 4 `
    --lr 3e-4 `
    --weight-decay 0.05 `
    --seed 42 `
    --amp `
    --patience 3 `
    --best-metric objective `
    --focus-class-name Xoai_Song_ChuaNhe_CoNguyCo `
    --focus-class-weight 0.35 `
    --skip-test
  if ($LASTEXITCODE -ne 0) { throw "TIMM train failed: $($r.id)" }
}

"OUT_ROOT=$outRoot"
```

For a paper-grade variance estimate, freeze the recipe and later repeat seeds `3407` and `2026`. Never inspect test to choose a seed.

## 10. AIDT checkpoint and optional clean retrain

The two full AIDT run folders contain equivalent learned tensors, so retraining is not required for the comparison. Use:

```text
D:\DataAI\AIDT\runs\resnet50_vit_b16_class_f_5class_pretrained\best.pt
```

If a clean reproducibility run is desired, use a new output root and keep test locked:

```powershell
Set-Location D:\DataAI\AIDT
$py = 'D:\DataAI\.venv\Scripts\python.exe'
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$aidtRun = "D:\DataAI\AIDT\runs\paper_clean_aidt_$stamp"

& $py .\train.py `
  --data D:\DataAI\AIEx\newdataset\class_f `
  --outdir $aidtRun `
  --epochs 30 `
  --batch-size 6 `
  --grad-accum 4 `
  --workers 4 `
  --image-size 224 `
  --backbone-lr 1e-5 `
  --head-lr 3e-4 `
  --weight-decay 0.05 `
  --warmup-epochs 2 `
  --patience 3 `
  --label-smoothing 0.05 `
  --class-balance sqrt_inverse `
  --seed 42 `
  --pretrained `
  --amp `
  --amp-dtype bf16 `
  --skip-final-test
if ($LASTEXITCODE -ne 0) { throw 'AIDT train failed.' }
```

## 11. Validation inference, strict class remap, and comparison table

This section uses FP32 inference for common scientific metrics. `class_f` and `yolo_f` do not share raw class-index order; strict name/path remapping is mandatory.

The default map below reuses the completed checkpoints. Replace a path only when intentionally evaluating a separately registered fresh rerun.

```powershell
Set-Location D:\DataAI\AIEx\TRKH
$py = 'D:\DataAI\.venv\Scripts\python.exe'
$classF = 'D:\DataAI\AIEx\newdataset\class_f'
$yolo = 'D:\DataAI\AIEx\newdataset\yolo_f\data.yaml'
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$predRoot = "D:\DataAI\AIEx\TRKH\runs\paper_pretrained_compare_val_$stamp"
$checkpoints = [ordered]@{
  resnet50 = 'D:\DataAI\AIEx\image_baseline_experiments\outputs\resnet50\best.pt'
  mobilenetv3 = 'D:\DataAI\AIEx\image_baseline_experiments\outputs\mobilenetv3\best.pt'
  efficientnetv2_s = 'D:\DataAI\AIEx\image_baseline_experiments\outputs\efficientnetv2_s\best.pt'
  convnext_tiny = 'D:\DataAI\AIEx\image_baseline_experiments\outputs\convnext_tiny\best.pt'
}

foreach ($entry in $checkpoints.GetEnumerator()) {
  $id = $entry.Key
  & $py -m trkh.tools.export_timm_predictions `
    --checkpoint $entry.Value `
    --data $classF `
    --split val `
    --output-dir "$predRoot\$id" `
    --batch-size 32 `
    --workers 2
  if ($LASTEXITCODE -ne 0) { throw "TIMM validation inference failed: $id" }

  & $py -m trkh.tools.remap_classification_teacher_to_yolo `
    --teacher-csv "$predRoot\$id\predictions_val.csv" `
    --source-metrics-json "$predRoot\$id\metrics_val.json" `
    --yolo-data $yolo `
    --split val `
    --output-csv "$predRoot\$id\predictions_yolof_val_aligned.csv" `
    --strict
  if ($LASTEXITCODE -ne 0) { throw "Strict validation remap failed: $id" }
}

$aidt = "$predRoot\aidt_resnet50_vit_b16"
& $py -m trkh.tools.export_aidt_predictions `
  --checkpoint D:\DataAI\AIDT\runs\resnet50_vit_b16_class_f_5class_pretrained\best.pt `
  --data $classF `
  --split val `
  --output-dir $aidt `
  --batch-size 8 `
  --workers 2
if ($LASTEXITCODE -ne 0) { throw 'AIDT validation inference failed.' }

& $py -m trkh.tools.remap_classification_teacher_to_yolo `
  --teacher-csv "$aidt\predictions_val.csv" `
  --source-metrics-json "$aidt\metrics_val.json" `
  --yolo-data $yolo `
  --split val `
  --output-csv "$aidt\predictions_yolof_val_aligned.csv" `
  --strict
if ($LASTEXITCODE -ne 0) { throw 'Strict AIDT validation remap failed.' }

$trkhKeeper = "$predRoot\trkh_keeper"
& $py -m trkh.evaluation.evaluate `
  --checkpoint D:\DataAI\AIEx\TRKH\runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt `
  --data $yolo `
  --class-name-mode raw `
  --expected-num-classes 5 `
  --split val `
  --batch-size 32 `
  --num-workers 2 `
  --output-dir $trkhKeeper `
  --paper-name TRKH-keeper `
  --family TRKH-no-pretrain
if ($LASTEXITCODE -ne 0) { throw 'TRKH keeper validation inference failed.' }

$trkhRandomInitFull = 'D:\DataAI\AIEx\TRKH\runs\full_v8_yolof_randominit_30e_20260714_105524\final_audit_manual\val_raw_fp32\predictions_detailed.csv'

& $py -m trkh.tools.audit_validation_information_ceiling `
  --input "TRKH:trkh_keeper=$trkhKeeper\predictions_detailed.csv" `
  --input "TRKH:trkh_randominit_full=$trkhRandomInitFull" `
  --input "CNN:resnet50=$predRoot\resnet50\predictions_yolof_val_aligned.csv" `
  --input "Mobile:mobilenetv3=$predRoot\mobilenetv3\predictions_yolof_val_aligned.csv" `
  --input "Efficient:efficientnetv2_s=$predRoot\efficientnetv2_s\predictions_yolof_val_aligned.csv" `
  --input "ModernCNN:convnext_tiny=$predRoot\convnext_tiny\predictions_yolof_val_aligned.csv" `
  --input "AIDT:aidt_fusion=$aidt\predictions_yolof_val_aligned.csv" `
  --base-name trkh_keeper `
  --expected-samples 2606 `
  --focus-class-index 1 `
  --bootstrap-iterations 2000 `
  --bootstrap-seed 20260711 `
  --bootstrap-mode source_group `
  --output-dir "$predRoot\comparison_val"
if ($LASTEXITCODE -ne 0) { throw 'Validation comparison failed.' }

"PRED_ROOT=$predRoot"
Get-Content "$predRoot\comparison_val\model_metrics.md"
```

The table artifact is `comparison_val\model_metrics.md`. Bootstrap and paired artifacts are `bootstrap_delta_vs_base.csv` and `pairwise_vs_base.csv`. Oracle rows are label-assisted diagnostics, not deployable scores.

Completed FP32 validation artifact (2026-07-29):

```text
D:\DataAI\AIEx\TRKH\runs\paper_pretrained_compare_val_reuse_20260729_001455\comparison_val_complete_models
```

| Role/model | Accuracy | Macro F1 | Class-1 F1 |
|---|---:|---:|---:|
| AIDT ResNet50 + ViT-B/16 | 0.944743 | 0.910267 | 0.718563 |
| MobileNetV3-Large | 0.940522 | 0.904089 | 0.707792 |
| ResNet-50 | 0.939371 | 0.900284 | 0.690323 |
| EfficientNetV2-S | 0.938219 | 0.899833 | 0.692557 |
| ConvNeXt-Tiny | 0.936301 | 0.896462 | 0.682119 |
| TRKH keeper | 0.919033 | 0.882925 | 0.678261 |
| TRKH random-init full | 0.910975 | 0.874172 | 0.654639 |

Every newly exported external prediction file strict-remapped `2,606/2,606` validation objects with zero missing or duplicate keys. Confusion and prediction-forensics artifacts are stored beside each model in the same run root.

## 12. Confusion and prediction-forensics audits for every comparison model

```powershell
$csvById = [ordered]@{
  trkh_keeper = "$predRoot\trkh_keeper\predictions_detailed.csv"
  resnet50 = "$predRoot\resnet50\predictions_yolof_val_aligned.csv"
  mobilenetv3 = "$predRoot\mobilenetv3\predictions_yolof_val_aligned.csv"
  efficientnetv2_s = "$predRoot\efficientnetv2_s\predictions_yolof_val_aligned.csv"
  convnext_tiny = "$predRoot\convnext_tiny\predictions_yolof_val_aligned.csv"
  aidt_resnet50_vit_b16 = "$predRoot\aidt_resnet50_vit_b16\predictions_yolof_val_aligned.csv"
}

$imageStatsCache = "$predRoot\trkh_keeper\forensics_val\predictions_forensics.csv"
foreach ($entry in $csvById.GetEnumerator()) {
  $id = $entry.Key
  $csv = $entry.Value
  & $py -m trkh.tools.audit_class_confusions `
    --predictions $csv `
    --data $yolo `
    --focus-class-index 1 `
    --top-k-images 30 `
    --output-dir "$predRoot\$id\confusions_val"
  if ($LASTEXITCODE -ne 0) { throw "Confusion audit failed: $id" }

  $forensicsArgs = @(
    '-m', 'trkh.tools.audit_prediction_forensics',
    '--predictions', $csv,
    '--focus-class-index', '1',
    '--image-stats-mode', 'foreground',
    '--top-k-images', '25',
    '--output-dir', "$predRoot\$id\forensics_val"
  )
  if ($id -ne 'trkh_keeper') {
    if (-not (Test-Path -LiteralPath $imageStatsCache)) { throw "Missing image-stat cache: $imageStatsCache" }
    $forensicsArgs += @('--image-stats-cache', $imageStatsCache)
  } else {
    $forensicsArgs += @('--image-stats-workers', '2')
  }
  & $py @forensicsArgs
  if ($LASTEXITCODE -ne 0) { throw "Forensics audit failed: $id" }
}
```

The completed full scratch baseline already has these audits under `full_v8_yolof_randominit_30e_20260714_105524\final_audit_manual`. The shared image-stat cache avoids decoding the same 2,606 images once per model while preserving per-model prediction analysis.

## 13. Final-test status: blocked for a new scientific claim

No new TRKH proposed model has passed the formal A0, matched integration probe, and locked validation gates. Therefore there is currently no authorized final-test command. Do not reopen test for the rejected natural probe, the warm-start control, the DDF synthetic sidecar, or every external baseline.

Historical test artifacts already exist for the keeper, the July 14 full scratch run, the June TIMM models, and AIDT. They may be reported only as retrospective evidence with their original precision/preprocessing metadata. A future confirmatory comparison must first freeze one selected candidate and all checkpoint hashes, then use a new sealed external/temporal holdout. Test outcomes must never trigger another model, threshold, seed, TTA, or ensemble-weight choice.

## 14. Reporting caveat

The current test split has already been opened by historical experiments, including run `20260728_184850`. It can still be reported as a retrospective benchmark, but it is not a pristine confirmatory holdout. A strong scientific claim should add source-grouped cross-validation or a new sealed holdout collected after the architecture and protocol are frozen.
