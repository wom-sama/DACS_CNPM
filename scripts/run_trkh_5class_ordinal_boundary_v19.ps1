param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\class_f\data.yaml",
    [string]$RunName = "mango_cls_256_5class_ordinal_boundary_v19_30e",
    [string]$ResumeCheckpoint = "runs\mango_cls_256_5class_attention_views_bounded_v8_30e\checkpoints\best.pt",
    [string]$SampleWeightManifest = "runs\boundary_sample_weights_v11_train_only_20260612\sample_weights_train_only.csv",
    [int]$Epochs = 30,
    [int]$Patience = 3,
    [int]$BatchSize = 32,
    [int]$GradAccumSteps = 2,
    [int]$NumWorkers = 4,
    [int]$EvalNumWorkers = 2,
    [int]$MaxTrainBatches = 0,
    [int]$MaxValBatches = 0,
    [double]$BoundaryContrastiveLossWeight = 0.08,
    [double]$PairwiseMarginRouteMaxProbabilityMargin = 0.20,
    [double]$OrdinalBoundaryLossWeight = 0.08,
    [string]$OrdinalBoundaryClasses = "0,1,2,3",
    [string]$OrdinalBoundaryThresholdWeights = "1.35,1.35,1.0",
    [double]$OrdinalBoundaryTemperature = 1.0,
    [int]$OrdinalBoundaryStartEpoch = 1,
    [string]$BackgroundSuppressionMode = "desaturate_blur",
    [double]$BackgroundSuppressionProbability = 0.80,
    [double]$BackgroundSuppressionMargin = 0.08,
    [double]$BackgroundSuppressionBlurRadius = 7.0,
    [switch]$Probe,
    [switch]$PreflightOnly,
    [switch]$Smoke,
    [switch]$SkipFinalTest,
    [bool]$TraceArchitecture = $true
)

$launcher = Join-Path $PSScriptRoot "run_trkh_5class_routed_pairwise_v16.ps1"

if ($Probe) {
    if (-not $PSBoundParameters.ContainsKey("RunName")) {
        $RunName = "probe_mango_cls_256_5class_ordinal_boundary_v19_120b_6e"
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

$launcherArgs = @{
    Python = $Python
    DataYaml = $DataYaml
    RunName = $RunName
    ResumeCheckpoint = $ResumeCheckpoint
    SampleWeightManifest = $SampleWeightManifest
    Epochs = $Epochs
    Patience = $Patience
    BatchSize = $BatchSize
    GradAccumSteps = $GradAccumSteps
    NumWorkers = $NumWorkers
    EvalNumWorkers = $EvalNumWorkers
    MaxTrainBatches = $MaxTrainBatches
    MaxValBatches = $MaxValBatches
    BoundaryContrastiveLossWeight = $BoundaryContrastiveLossWeight
    PairwiseMarginRouteMaxProbabilityMargin = $PairwiseMarginRouteMaxProbabilityMargin
    OrdinalBoundaryLossWeight = $OrdinalBoundaryLossWeight
    OrdinalBoundaryClasses = $OrdinalBoundaryClasses
    OrdinalBoundaryThresholdWeights = $OrdinalBoundaryThresholdWeights
    OrdinalBoundaryTemperature = $OrdinalBoundaryTemperature
    OrdinalBoundaryStartEpoch = $OrdinalBoundaryStartEpoch
    BackgroundSuppressionMode = $BackgroundSuppressionMode
    BackgroundSuppressionProbability = $BackgroundSuppressionProbability
    BackgroundSuppressionMargin = $BackgroundSuppressionMargin
    BackgroundSuppressionBlurRadius = $BackgroundSuppressionBlurRadius
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
