param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\class_f\data.yaml",
    [string]$RunName = "mango_cls_256_5class_foreground_crop_v30_30e",
    [string]$ResumeCheckpoint = "runs\mango_cls_256_5class_attention_views_bounded_v8_30e\checkpoints\best.pt",
    [string]$SampleWeightManifest = "runs\boundary_sample_weights_v11_train_only_20260612\sample_weights_train_only.csv",
    [int]$ImageSize = 256,
    [int]$Epochs = 30,
    [int]$Patience = 3,
    [int]$BatchSize = 32,
    [int]$GradAccumSteps = 2,
    [int]$NumWorkers = 4,
    [int]$EvalNumWorkers = 2,
    [int]$MaxTrainBatches = 0,
    [int]$MaxValBatches = 0,
    [ValidateSet("none", "pseudo", "grabcut")]
    [string]$ForegroundCropMode = "pseudo",
    [double]$ForegroundCropProbability = 0.70,
    [double]$ForegroundCropMarginRatio = 0.10,
    [double]$ForegroundCropMinMaskAreaRatio = 0.03,
    [double]$ForegroundCropMaxMaskAreaRatio = 0.92,
    [double]$ForegroundCropMaxCropAreaRatio = 0.98,
    [string]$BackgroundSuppressionMode = "desaturate_blur",
    [double]$BackgroundSuppressionProbability = 0.65,
    [switch]$Probe,
    [switch]$PreflightOnly,
    [switch]$Smoke,
    [switch]$SkipFinalTest,
    [bool]$TraceArchitecture = $true
)

$launcher = Join-Path $PSScriptRoot "run_trkh_5class_routed_pairwise_v16.ps1"

if ($Probe) {
    if (-not $PSBoundParameters.ContainsKey("RunName")) {
        $RunName = "probe_mango_cls_256_5class_foreground_crop_v30_120b_6e"
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
if ($Smoke -and -not $PSBoundParameters.ContainsKey("RunName")) {
    $RunName = "smoke_mango_cls_256_5class_foreground_crop_v30"
}
if ($PreflightOnly -and -not $PSBoundParameters.ContainsKey("RunName")) {
    $RunName = "preflight_mango_cls_256_5class_foreground_crop_v30"
}

$launcherArgs = @{
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
    ForegroundCropMode = $ForegroundCropMode
    ForegroundCropProbability = $ForegroundCropProbability
    ForegroundCropMarginRatio = $ForegroundCropMarginRatio
    ForegroundCropMinMaskAreaRatio = $ForegroundCropMinMaskAreaRatio
    ForegroundCropMaxMaskAreaRatio = $ForegroundCropMaxMaskAreaRatio
    ForegroundCropMaxCropAreaRatio = $ForegroundCropMaxCropAreaRatio
    BackgroundSuppressionMode = $BackgroundSuppressionMode
    BackgroundSuppressionProbability = $BackgroundSuppressionProbability
    SkipFinalTest = [bool]$SkipFinalTest
    TraceArchitecture = $TraceArchitecture
}
if ($PreflightOnly) {
    $launcherArgs.PreflightOnly = $true
}
if ($Smoke) {
    $launcherArgs.Smoke = $true
}

& $launcher @launcherArgs
exit $LASTEXITCODE
