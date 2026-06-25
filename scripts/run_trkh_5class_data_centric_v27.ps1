param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\class_f\data.yaml",
    [string]$RunName = "mango_cls_256_5class_data_centric_v27_30e",
    [string]$ResumeCheckpoint = "runs\mango_cls_256_5class_attention_views_bounded_v8_30e\checkpoints\best.pt",
    [string]$TrainPredictionsCsv = "runs\mango_cls_256_5class_defectstat_v3_30e\train_maskfix_for_hard_mining\predictions_detailed.csv",
    [string]$BaseSampleWeightManifest = "runs\boundary_sample_weights_v11_train_only_20260612\sample_weights_train_only.csv",
    [string]$DataCentricOutputDir = "runs\data_centric_weights_v27_train_only_20260625",
    [int]$ImageSize = 256,
    [string]$BoundaryPairs = "0-1,1-2,2-3,4-rest",
    [double]$HighConfidenceThreshold = 0.55,
    [double]$LowSelfConfidenceThreshold = 0.25,
    [double]$AmbiguousMarginThreshold = 0.06,
    [double]$HighConfidenceMultiplier = 0.30,
    [double]$LowSelfConfidenceMultiplier = 0.45,
    [double]$AmbiguousMultiplier = 0.85,
    [int]$MaxIssues = 420,
    [int]$MaxPerReason = 220,
    [int]$MaxPerPair = 180,
    [switch]$SkipBuildManifest,
    [switch]$CopyReviewImages,
    [int]$Epochs = 30,
    [int]$Patience = 3,
    [int]$BatchSize = 32,
    [int]$GradAccumSteps = 2,
    [int]$NumWorkers = 4,
    [int]$EvalNumWorkers = 2,
    [int]$MaxTrainBatches = 0,
    [int]$MaxValBatches = 0,
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

$SampleWeightManifest = Join-Path $DataCentricOutputDir "sample_weights_train_only.csv"

if (-not $SkipBuildManifest) {
    if (-not (Test-Path -LiteralPath $TrainPredictionsCsv)) {
        throw "Khong tim thay train prediction CSV de build data-centric manifest: $TrainPredictionsCsv"
    }
    if (-not (Test-Path -LiteralPath $BaseSampleWeightManifest)) {
        throw "Khong tim thay base sample-weight manifest: $BaseSampleWeightManifest"
    }
    New-Item -ItemType Directory -Force -Path $DataCentricOutputDir | Out-Null

    $BuildArgs = @(
        "-m", "trkh.tools.build_data_centric_sample_weights",
        "--predictions", $TrainPredictionsCsv,
        "--output-dir", $DataCentricOutputDir,
        "--base-sample-weight-manifest", $BaseSampleWeightManifest,
        "--boundary-pairs", $BoundaryPairs,
        "--high-confidence-threshold", "$HighConfidenceThreshold",
        "--low-self-confidence-threshold", "$LowSelfConfidenceThreshold",
        "--ambiguous-margin-threshold", "$AmbiguousMarginThreshold",
        "--high-confidence-multiplier", "$HighConfidenceMultiplier",
        "--low-self-confidence-multiplier", "$LowSelfConfidenceMultiplier",
        "--ambiguous-multiplier", "$AmbiguousMultiplier",
        "--max-issues", "$MaxIssues",
        "--max-per-reason", "$MaxPerReason",
        "--max-per-pair", "$MaxPerPair"
    )
    if ($CopyReviewImages) {
        $BuildArgs += "--copy-images"
    }

    & $Python @BuildArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Build data-centric sample-weight manifest failed."
    }
}

if (-not (Test-Path -LiteralPath $SampleWeightManifest)) {
    throw "Khong tim thay data-centric sample-weight manifest: $SampleWeightManifest"
}

if ($Smoke) {
    if (-not $PSBoundParameters.ContainsKey("RunName")) {
        $RunName = "smoke_mango_cls_256_5class_data_centric_v27"
    }
}

if ($Probe) {
    if (-not $PSBoundParameters.ContainsKey("RunName")) {
        $RunName = "probe_mango_cls_256_5class_data_centric_v27_120b_6e"
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

$BaseLauncher = Join-Path $PSScriptRoot "run_trkh_5class_routed_pairwise_v16.ps1"
$LauncherArgs = @{
    Python = $Python
    DataYaml = $DataYaml
    RunName = $RunName
    ResumeCheckpoint = $ResumeCheckpoint
    SampleWeightManifest = $SampleWeightManifest
    ImageSize = $ImageSize
    Epochs = $Epochs
    Patience = $Patience
    BatchSize = $BatchSize
    GradAccumSteps = $GradAccumSteps
    NumWorkers = $NumWorkers
    EvalNumWorkers = $EvalNumWorkers
    MaxTrainBatches = $MaxTrainBatches
    MaxValBatches = $MaxValBatches
    BackgroundSuppressionMode = "desaturate_blur"
    BackgroundSuppressionProbability = 0.65
    BackgroundSuppressionMargin = 0.08
    BackgroundSuppressionBlurRadius = 7.0
    SkipFinalTest = [bool]($SkipFinalTest -or $Probe)
    TraceArchitecture = $TraceArchitecture
}

if ($PreflightOnly) {
    $LauncherArgs.PreflightOnly = $true
}
if ($Smoke) {
    $LauncherArgs.Smoke = $true
}

& $BaseLauncher @LauncherArgs
exit $LASTEXITCODE
