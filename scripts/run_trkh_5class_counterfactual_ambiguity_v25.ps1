param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\class_f\data.yaml",
    [string]$RunName = "mango_cls_256_5class_counterfactual_ambiguity_v25_30e",
    [string]$ResumeCheckpoint = "runs\mango_cls_256_5class_attention_views_bounded_v8_30e\checkpoints\best.pt",
    [string]$SampleWeightManifest = "runs\boundary_sample_weights_v11_train_only_20260612\sample_weights_train_only.csv",
    [string]$TrainPredictionsCsv = "runs\mango_cls_256_5class_defectstat_v3_30e\train_maskfix_for_hard_mining\predictions_detailed.csv",
    [string]$AmbiguousSoftTargetManifest = "runs\ambiguous_soft_targets_v25_train_only_20260625\ambiguous_soft_targets_train_only.csv",
    [string]$AmbiguousPairs = "0-1,1-2,2-3",
    [double]$AmbiguousSoftTargetAlpha = 0.22,
    [double]$AmbiguousMarginThreshold = 0.08,
    [int]$AmbiguousMaxSamples = 600,
    [switch]$SkipBuildAmbiguousManifest,
    [int]$Epochs = 30,
    [int]$Patience = 3,
    [int]$BatchSize = 32,
    [int]$GradAccumSteps = 2,
    [int]$NumWorkers = 4,
    [int]$EvalNumWorkers = 2,
    [int]$MaxTrainBatches = 0,
    [int]$MaxValBatches = 0,
    [double]$BackgroundCounterfactualConsistencyWeight = 0.03,
    [double]$BackgroundCounterfactualProbability = 0.35,
    [ValidateSet("gray", "blur", "mean", "desaturate_blur")]
    [string]$BackgroundCounterfactualMode = "desaturate_blur",
    [double]$BackgroundCounterfactualMargin = 0.08,
    [int]$BackgroundCounterfactualBlurKernel = 15,
    [double]$BackgroundCounterfactualTemperature = 1.2,
    [switch]$Probe,
    [switch]$PreflightOnly,
    [switch]$Smoke,
    [switch]$SkipFinalTest,
    [bool]$TraceArchitecture = $true
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$env:PYTHONPATH = $ProjectRoot

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Khong tim thay Python: $Python"
}

if (-not $SkipBuildAmbiguousManifest) {
    if (-not (Test-Path -LiteralPath $TrainPredictionsCsv)) {
        throw "Khong tim thay train prediction CSV de build ambiguous manifest: $TrainPredictionsCsv"
    }
    $ManifestDir = Split-Path -Parent $AmbiguousSoftTargetManifest
    if (-not [string]::IsNullOrWhiteSpace($ManifestDir)) {
        New-Item -ItemType Directory -Force -Path $ManifestDir | Out-Null
    }
    & $Python -m trkh.tools.build_ambiguous_soft_targets `
        --predictions $TrainPredictionsCsv `
        --output $AmbiguousSoftTargetManifest `
        --pairs $AmbiguousPairs `
        --alpha $AmbiguousSoftTargetAlpha `
        --margin-threshold $AmbiguousMarginThreshold `
        --max-samples $AmbiguousMaxSamples
    if ($LASTEXITCODE -ne 0) {
        throw "Build ambiguous soft-target manifest failed."
    }
}

if ($Probe) {
    if (-not $PSBoundParameters.ContainsKey("RunName")) {
        $RunName = "probe_mango_cls_256_5class_counterfactual_ambiguity_v25_120b_6e"
    }
    if (-not $PSBoundParameters.ContainsKey("Epochs")) {
        $Epochs = 6
    }
    if (-not $PSBoundParameters.ContainsKey("MaxTrainBatches")) {
        $MaxTrainBatches = 120
    }
    if (-not $PSBoundParameters.ContainsKey("MaxValBatches")) {
        $MaxValBatches = 0
    }
    if (-not $PSBoundParameters.ContainsKey("Patience")) {
        $Patience = 3
    }
}

$BaseLauncher = Join-Path $PSScriptRoot "run_trkh_5class_attention_views_v8.ps1"
$LauncherArgs = @{
    Python = $Python
    DataYaml = $DataYaml
    RunName = $RunName
    ResumeCheckpoint = $ResumeCheckpoint
    SampleWeightManifest = $SampleWeightManifest
    SampleWeightFactor = 1.0
    SampleWeightMax = 2.5
    AmbiguousSoftTargetManifest = $AmbiguousSoftTargetManifest
    AmbiguousSoftTargetAlpha = $AmbiguousSoftTargetAlpha
    Epochs = $Epochs
    Patience = $Patience
    BatchSize = $BatchSize
    GradAccumSteps = $GradAccumSteps
    NumWorkers = $NumWorkers
    EvalNumWorkers = $EvalNumWorkers
    MaxTrainBatches = $MaxTrainBatches
    MaxValBatches = $MaxValBatches
    AttentionViewLossWeight = 0.18
    AttentionCropProbability = 0.25
    AttentionDropProbability = 0.10
    AttentionViewStartEpoch = 2
    AttentionViewScoreSource = "surface_detail"
    AttentionViewForegroundWeight = 0.80
    AttentionDropMinAreaRatio = 0.06
    AttentionDropMaxAreaRatio = 0.14
    BoundaryContrastiveLossWeight = 0.08
    BoundaryContrastivePairs = "0-1,1-2,2-3"
    BoundaryContrastiveSources = "head,patch"
    BoundaryContrastiveMargin = 0.12
    BoundaryContrastiveTemperature = 0.20
    BoundaryContrastiveMaxPairs = 128
    PairwiseMarginRouting = $true
    PairwiseMarginRouteMaxProbabilityMargin = 0.20
    ClassificationLoss = "ldam_focal"
    FocalLossGamma = 1.0
    FocalLossMix = 0.10
    LabelSmoothing = 0.015
    LdamMaxMargin = 0.28
    LdamScale = 18.0
    BackgroundSuppressionMode = "desaturate_blur"
    BackgroundSuppressionProbability = 0.65
    BackgroundSuppressionMargin = 0.08
    BackgroundSuppressionBlurRadius = 7.0
    BackgroundCounterfactualConsistencyWeight = $BackgroundCounterfactualConsistencyWeight
    BackgroundCounterfactualProbability = $BackgroundCounterfactualProbability
    BackgroundCounterfactualMode = $BackgroundCounterfactualMode
    BackgroundCounterfactualMargin = $BackgroundCounterfactualMargin
    BackgroundCounterfactualBlurKernel = $BackgroundCounterfactualBlurKernel
    BackgroundCounterfactualTemperature = $BackgroundCounterfactualTemperature
    ForegroundBackgroundMixProbability = 0.0
    Sam = $false
    TraceArchitecture = $TraceArchitecture
}

if ($PreflightOnly) {
    $LauncherArgs.PreflightOnly = $true
}
if ($Smoke) {
    $LauncherArgs.Smoke = $true
}
if ($SkipFinalTest -or $Probe) {
    $LauncherArgs.SkipFinalTest = $true
}

& $BaseLauncher @LauncherArgs
exit $LASTEXITCODE
