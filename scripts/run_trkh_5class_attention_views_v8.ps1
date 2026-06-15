param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RunName = "mango_cls_256_5class_attention_views_v8_30e",
    [string]$ResumeCheckpoint = "runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\checkpoints\best.pt",
    [int]$Epochs = 30,
    [int]$Patience = 3,
    [int]$BatchSize = 32,
    [int]$GradAccumSteps = 2,
    [int]$NumWorkers = 4,
    [int]$EvalNumWorkers = 2,
    [int]$MaxTrainBatches = 0,
    [int]$MaxValBatches = 0,
    [double]$AttentionViewLossWeight = 0.35,
    [double]$AttentionCropProbability = 0.40,
    [double]$AttentionDropProbability = 0.20,
    [int]$AttentionViewStartEpoch = 2,
    [ValidateSet("learned_attention", "surface_detail", "hybrid")]
    [string]$AttentionViewScoreSource = "learned_attention",
    [double]$AttentionViewForegroundWeight = 0.40,
    [double]$AttentionDropMinAreaRatio = 0.06,
    [double]$AttentionDropMaxAreaRatio = 0.16,
    [double]$ElrLossWeight = 0.0,
    [double]$ElrBeta = 0.70,
    [int]$ElrStartEpoch = 2,
    [double]$BoundaryContrastiveLossWeight = 0.0,
    [string]$BoundaryContrastivePairs = "0-1,1-2,2-3,4-rest",
    [string]$BoundaryContrastiveSources = "head,patch",
    [double]$BoundaryContrastiveMargin = 0.12,
    [double]$BoundaryContrastiveTemperature = 0.20,
    [int]$BoundaryContrastiveMaxPairs = 128,
    [string]$SampleWeightManifest = "",
    [double]$SampleWeightFactor = 1.0,
    [double]$SampleWeightMax = 5.0,
    [bool]$ForegroundSurfaceFusion = $false,
    [double]$ForegroundSurfaceFusionDropout = 0.08,
    [bool]$BilinearPatchFusion = $false,
    [int]$BilinearPatchRank = 32,
    [double]$BilinearPatchDropout = 0.08,
    [bool]$FrequencySelectivePooling = $false,
    [int]$FrequencySelectiveTopK = 1,
    [double]$FrequencySelectiveBlend = 1.0,
    [double]$FrequencySelectiveForegroundThreshold = 0.35,
    [bool]$PairwiseMarginRouting = $false,
    [double]$PairwiseMarginRouteMaxProbabilityMargin = 0.20,
    [switch]$PreflightOnly,
    [switch]$Smoke,
    [switch]$SkipFinalTest,
    [bool]$TraceArchitecture = $true
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:PYTHONPATH = $ProjectRoot
$env:TRKH_AMP_DTYPE = "bf16"
$env:OMP_NUM_THREADS = "4"
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = "1"
Remove-Item Env:TRKH_ALLOW_WINDOWS_PIN_MEMORY -ErrorAction SilentlyContinue
Remove-Item Env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS -ErrorAction SilentlyContinue

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Khong tim thay Python: $Python"
}

& $Python -m py_compile `
    trkh\training\train.py `
    trkh\tools\trace_architecture.py `
    trkh\models\model.py `
    trkh\data\dataset.py
if ($LASTEXITCODE -ne 0) {
    throw "Python compile preflight failed."
}
& $Python -m trkh.training.train --help | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Training CLI import/help preflight failed."
}

$DataYaml = "D:\DataAI\AIEx\newdataset\class_f\data.yaml"
if (-not (Test-Path -LiteralPath $DataYaml)) {
    throw "Khong tim thay dataset YAML: $DataYaml"
}
if (-not [string]::IsNullOrWhiteSpace($ResumeCheckpoint) -and -not (Test-Path -LiteralPath $ResumeCheckpoint)) {
    throw "Khong tim thay resume checkpoint: $ResumeCheckpoint"
}
if (-not [string]::IsNullOrWhiteSpace($SampleWeightManifest) -and -not (Test-Path -LiteralPath $SampleWeightManifest)) {
    throw "Khong tim thay sample weight manifest: $SampleWeightManifest"
}

$SplitCounts = Get-ChildItem -Recurse -File "D:\DataAI\AIEx\newdataset\class_f" |
    Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp|webp)$' } |
    Group-Object { $_.Directory.Parent.Name + "/" + $_.Directory.Name } |
    Sort-Object Name |
    ForEach-Object {
        [ordered]@{
            split_class = $_.Name
            count = $_.Count
        }
    }
$GpuStatus = & nvidia-smi `
    --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,pstate `
    --format=csv,noheader

if ($PreflightOnly) {
    [ordered]@{
        status = "ok"
        python = $Python
        data_yaml = $DataYaml
        resume_checkpoint = $ResumeCheckpoint
        split_counts = $SplitCounts
        gpu = $GpuStatus
        batch_size = $BatchSize
        grad_accum_steps = $GradAccumSteps
        effective_batch_size = $BatchSize * $GradAccumSteps
        attention_view_loss_weight = $AttentionViewLossWeight
        attention_crop_probability = $AttentionCropProbability
        attention_drop_probability = $AttentionDropProbability
        attention_view_start_epoch = $AttentionViewStartEpoch
        attention_view_score_source = $AttentionViewScoreSource
        attention_view_foreground_weight = $AttentionViewForegroundWeight
        attention_drop_min_area_ratio = $AttentionDropMinAreaRatio
        attention_drop_max_area_ratio = $AttentionDropMaxAreaRatio
        elr_loss_weight = $ElrLossWeight
        elr_beta = $ElrBeta
        elr_start_epoch = $ElrStartEpoch
        boundary_contrastive_loss_weight = $BoundaryContrastiveLossWeight
        boundary_contrastive_pairs = $BoundaryContrastivePairs
        boundary_contrastive_sources = $BoundaryContrastiveSources
        boundary_contrastive_margin = $BoundaryContrastiveMargin
        boundary_contrastive_temperature = $BoundaryContrastiveTemperature
        boundary_contrastive_max_pairs = $BoundaryContrastiveMaxPairs
        sample_weight_manifest = $SampleWeightManifest
        sample_weight_factor = $SampleWeightFactor
        sample_weight_max = $SampleWeightMax
        bilinear_patch_fusion = [bool]$BilinearPatchFusion
        bilinear_patch_rank = $BilinearPatchRank
        bilinear_patch_dropout = $BilinearPatchDropout
        frequency_selective_pooling = [bool]$FrequencySelectivePooling
        frequency_selective_top_k = $FrequencySelectiveTopK
        frequency_selective_blend = $FrequencySelectiveBlend
        frequency_selective_foreground_threshold = $FrequencySelectiveForegroundThreshold
        pairwise_margin_routing = [bool]$PairwiseMarginRouting
        pairwise_margin_route_max_probability_margin = $PairwiseMarginRouteMaxProbabilityMargin
        skip_final_test = [bool]$SkipFinalTest
    } | ConvertTo-Json -Depth 5
    exit 0
}

if ($Smoke) {
    if (-not $PSBoundParameters.ContainsKey("RunName")) {
        $RunName = "smoke_mango_cls_256_5class_attention_views_v8"
    }
    $Epochs = 1
    $Patience = 1
    $GradAccumSteps = 1
    if ($MaxTrainBatches -le 0) {
        $MaxTrainBatches = 2
    }
    if ($MaxValBatches -le 0) {
        $MaxValBatches = 2
    }
}
$EffectiveAttentionViewStartEpoch = if ($Smoke) { 1 } else { $AttentionViewStartEpoch }

$RunDir = Join-Path $ProjectRoot "runs\$RunName"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$startedAt = Get-Date
$exitCode = 1
$transcriptStarted = $false

try {
    Start-Transcript -Path (Join-Path $RunDir "launcher_transcript.txt") -Append | Out-Null
    $transcriptStarted = $true

    $TrainArgs = @(
        "--data", $DataYaml,
        "--class-name-mode", "raw",
        "--expected-num-classes", "5",
        "--run-name", $RunName,
        "--output-dir", "runs",
        "--seed", "42",
        "--model-type", "vit_registers",
        "--no-pretrained",
        "--no-pretrained-distillation",
        "--image-size", "256",
        "--patch-size", "16",
        "--stem-channels", "32",
        "--cnn-feature-fusion",
        "--cnn-fusion-dropout", "0.10",
        "--fine-grained-pooling",
        "--fine-grained-pooling-dropout", "0.08",
        "--multi-branch-fusion",
        "--branch-color-tokens", "1",
        "--branch-edge-tokens", "1",
        "--branch-cnn-tokens", "0",
        "--branch-token-dropout", "0.08",
        "--detail-patch-enhancement",
        "--detail-patch-dropout", "0.05",
        "--token-pruning",
        "--token-prune-layers", "2,5",
        "--token-keep-rates", "0.85,0.65",
        "--token-prune-foreground-weight", "0.45",
        "--pairwise-margin-head",
        "--pairwise-margin-pairs", "0-1,1-2,2-3,4-rest",
        "--pairwise-margin-logit-scale", "0.25",
        "--pairwise-margin-dropout", "0.05",
        "--embed-dim", "256",
        "--depth", "8",
        "--num-heads", "8",
        "--num-registers", "4",
        "--register-positional-embedding",
        "--head-pooling", "cls_branch_register_mean",
        "--dropout", "0.12",
        "--attention-dropout", "0.03",
        "--drop-path-rate", "0.10",
        "--batch-size", "$BatchSize",
        "--grad-accum-steps", "$GradAccumSteps",
        "--epochs", "$Epochs",
        "--scheduler-total-epochs", "$Epochs",
        "--patience", "$Patience",
        "--learning-rate", "8e-5",
        "--min-learning-rate", "1e-6",
        "--warmup-epochs", "1",
        "--weight-decay", "0.05",
        "--grad-clip-norm", "0.7",
        "--max-nonfinite-grad-steps", "4",
        "--model-ema",
        "--model-ema-decay", "0.995",
        "--num-workers", "$NumWorkers",
        "--eval-num-workers", "$EvalNumWorkers",
        "--train-image-cache-mb", "0",
        "--eval-image-cache-mb", "0",
        "--balanced-epoch-multiplier", "1.0",
        "--balanced-epoch-tolerance", "0.10",
        "--disable-imbalance-auto-tune",
        "--disable-class-weights",
        "--disable-class-aware-augmentation",
        "--disable-rare-class-repeat",
        "--disable-rare-class-recall-guard",
        "--best-metric", "fair_macro_f1",
        "--fair-f1-gap-target", "0.08",
        "--classification-loss", "ldam_focal",
        "--ldam-max-margin", "0.30",
        "--ldam-scale", "18",
        "--focal-loss-gamma", "1.0",
        "--focal-loss-mix", "0.10",
        "--label-smoothing", "0.02",
        "--metric-learning-loss-weight", "0.04",
        "--metric-learning-temperature", "0.16",
        "--metric-learning-sources", "head,patch",
        "--boundary-contrastive-loss-weight", "$BoundaryContrastiveLossWeight",
        "--boundary-contrastive-pairs", "$BoundaryContrastivePairs",
        "--boundary-contrastive-sources", "$BoundaryContrastiveSources",
        "--boundary-contrastive-margin", "$BoundaryContrastiveMargin",
        "--boundary-contrastive-temperature", "$BoundaryContrastiveTemperature",
        "--boundary-contrastive-max-pairs", "$BoundaryContrastiveMaxPairs",
        "--foreground-consistency-loss-weight", "0.025",
        "--foreground-consistency-margin", "0.07",
        "--attention-view-loss-weight", "$AttentionViewLossWeight",
        "--attention-crop-probability", "$AttentionCropProbability",
        "--attention-drop-probability", "$AttentionDropProbability",
        "--attention-view-start-epoch", "$EffectiveAttentionViewStartEpoch",
        "--attention-crop-threshold", "0.55",
        "--attention-drop-threshold", "0.72",
        "--attention-crop-padding-ratio", "0.08",
        "--attention-crop-min-area-ratio", "0.25",
        "--attention-view-foreground-weight", "$AttentionViewForegroundWeight",
        "--attention-view-score-source", "$AttentionViewScoreSource",
        "--attention-drop-blur-kernel", "15",
        "--attention-drop-dilation-kernel", "5",
        "--attention-drop-min-area-ratio", "$AttentionDropMinAreaRatio",
        "--attention-drop-max-area-ratio", "$AttentionDropMaxAreaRatio",
        "--elr-loss-weight", "$ElrLossWeight",
        "--elr-beta", "$ElrBeta",
        "--elr-start-epoch", "$ElrStartEpoch",
        "--sample-weight-factor", "$SampleWeightFactor",
        "--sample-weight-max", "$SampleWeightMax",
        "--register-diversity-loss-weight", "0.0",
        "--pairwise-margin-loss-weight", "0.04",
        "--hard-sample-manifest", "runs\mango_cls_256_5class_defectstat_v3_30e\hard_mining_train_only\hard_samples_train_only.csv",
        "--hard-sample-repeat-factor", "1.6",
        "--resize-mode", "pad",
        "--train-scale-min", "0.88",
        "--train-scale-crop-probability", "0.35",
        "--brightness", "0.04",
        "--contrast", "0.04",
        "--saturation", "0.02",
        "--hue", "0.01",
        "--illumination-normalization",
        "--illumination-normalization-strength", "0.35",
        "--background-suppression-mode", "desaturate_blur",
        "--background-suppression-probability", "0.8",
        "--background-suppression-margin", "0.08",
        "--background-suppression-blur-radius", "7",
        "--local-exposure-probability", "0.15",
        "--local-exposure-strength", "0.25",
        "--obstacle-probability", "0.04",
        "--obstacle-max-area", "0.08",
        "--random-erasing-probability", "0",
        "--random-affine-degrees", "3",
        "--random-affine-translate", "0.02",
        "--random-affine-scale-min", "0.96",
        "--horizontal-flip-probability", "0.5",
        "--vertical-flip-probability", "0",
        "--rotate90-probability", "0.03",
        "--lighting-probability", "0",
        "--batch-mix-probability", "0",
        "--mosaic-probability", "0",
        "--mixup-probability", "0",
        "--cutmix-probability", "0",
        "--copy-paste-probability", "0",
        "--targeted-copy-paste-probability", "0"
    )

    if ([string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
        $TrainArgs += @("--disable-resume")
    }
    else {
        $TrainArgs += @(
            "--resume", $ResumeCheckpoint,
            "--resume-use-cli-config",
            "--resume-reset-epoch",
            "--resume-reset-optimizer",
            "--resume-reset-scheduler",
            "--resume-reset-scaler"
        )
    }
    if ($MaxTrainBatches -gt 0) {
        $TrainArgs += @("--max-train-batches", "$MaxTrainBatches")
    }
    if ($MaxValBatches -gt 0) {
        $TrainArgs += @("--max-val-batches", "$MaxValBatches")
    }
    if (-not [string]::IsNullOrWhiteSpace($SampleWeightManifest)) {
        if (-not (Test-Path -LiteralPath $SampleWeightManifest)) {
            throw "Khong tim thay sample weight manifest: $SampleWeightManifest"
        }
        $TrainArgs += @("--sample-weight-manifest", $SampleWeightManifest)
    }
    if ($ForegroundSurfaceFusion) {
        $TrainArgs += @(
            "--foreground-surface-fusion",
            "--foreground-surface-fusion-dropout", "$ForegroundSurfaceFusionDropout"
        )
    }
    if ($BilinearPatchFusion) {
        $TrainArgs += @(
            "--bilinear-patch-fusion",
            "--bilinear-patch-rank", "$BilinearPatchRank",
            "--bilinear-patch-dropout", "$BilinearPatchDropout"
        )
    }
    if ($FrequencySelectivePooling) {
        $TrainArgs += @(
            "--frequency-selective-pooling",
            "--frequency-selective-top-k", "$FrequencySelectiveTopK",
            "--frequency-selective-blend", "$FrequencySelectiveBlend",
            "--frequency-selective-foreground-threshold", "$FrequencySelectiveForegroundThreshold"
        )
    }
    if ($PairwiseMarginRouting) {
        $TrainArgs += @(
            "--pairwise-margin-routing",
            "--pairwise-margin-route-max-probability-margin",
            "$PairwiseMarginRouteMaxProbabilityMargin"
        )
    }
    if ($Smoke -or $SkipFinalTest) {
        $TrainArgs += @("--skip-final-test")
    }
    if ($TraceArchitecture) {
        $TrainArgs += @(
            "--trace-architecture",
            "--trace-architecture-seed", "42",
            "--trace-architecture-device", "cuda"
        )
    }

    [ordered]@{
        run_name = $RunName
        started_at = $startedAt.ToString("o")
        smoke = [bool]$Smoke
        skip_final_test = [bool]$SkipFinalTest
        no_pretrained = $true
        resume_checkpoint = $ResumeCheckpoint
        batch_size = $BatchSize
        grad_accum_steps = $GradAccumSteps
        effective_batch_size = $BatchSize * $GradAccumSteps
        num_workers = $NumWorkers
        eval_num_workers = $EvalNumWorkers
        frequency_selective_pooling = [bool]$FrequencySelectivePooling
        frequency_selective_top_k = $FrequencySelectiveTopK
        frequency_selective_blend = $FrequencySelectiveBlend
        frequency_selective_foreground_threshold = $FrequencySelectiveForegroundThreshold
        pairwise_margin_routing = [bool]$PairwiseMarginRouting
        pairwise_margin_route_max_probability_margin = $PairwiseMarginRouteMaxProbabilityMargin
        attention_view_loss_weight = $AttentionViewLossWeight
        attention_crop_probability = $AttentionCropProbability
        attention_drop_probability = $AttentionDropProbability
        attention_view_start_epoch = $EffectiveAttentionViewStartEpoch
        attention_view_score_source = $AttentionViewScoreSource
        attention_view_foreground_weight = $AttentionViewForegroundWeight
        attention_drop_min_area_ratio = $AttentionDropMinAreaRatio
        attention_drop_max_area_ratio = $AttentionDropMaxAreaRatio
        elr_loss_weight = $ElrLossWeight
        elr_beta = $ElrBeta
        elr_start_epoch = $ElrStartEpoch
        boundary_contrastive_loss_weight = $BoundaryContrastiveLossWeight
        boundary_contrastive_pairs = $BoundaryContrastivePairs
        boundary_contrastive_sources = $BoundaryContrastiveSources
        boundary_contrastive_margin = $BoundaryContrastiveMargin
        boundary_contrastive_temperature = $BoundaryContrastiveTemperature
        boundary_contrastive_max_pairs = $BoundaryContrastiveMaxPairs
        sample_weight_manifest = $SampleWeightManifest
        sample_weight_factor = $SampleWeightFactor
        sample_weight_max = $SampleWeightMax
        foreground_surface_fusion = [bool]$ForegroundSurfaceFusion
        foreground_surface_fusion_dropout = $ForegroundSurfaceFusionDropout
        bilinear_patch_fusion = [bool]$BilinearPatchFusion
        bilinear_patch_rank = $BilinearPatchRank
        bilinear_patch_dropout = $BilinearPatchDropout
        train_args = $TrainArgs
    } | ConvertTo-Json -Depth 6 |
        Set-Content -Path (Join-Path $RunDir "launcher_args.json")

    & $Python -m trkh.training.train @TrainArgs
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Training command exited with code $exitCode"
    }
}
catch {
    $_ | Out-String | Set-Content -Path (Join-Path $RunDir "launcher_exception.txt")
    if ($exitCode -eq 0) {
        $exitCode = 1
    }
}
finally {
    [ordered]@{
        run_name = $RunName
        started_at = $startedAt.ToString("o")
        finished_at = (Get-Date).ToString("o")
        exit_code = $exitCode
    } | ConvertTo-Json |
        Set-Content -Path (Join-Path $RunDir "launcher_status.json")
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}

exit $exitCode
