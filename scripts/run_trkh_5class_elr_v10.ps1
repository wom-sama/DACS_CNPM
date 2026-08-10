param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RunName = "mango_cls_256_5class_surface_detail_elr_v10_30e",
    [string]$ResumeCheckpoint = "runs\mango_cls_256_5class_attention_views_bounded_v8_30e\checkpoints\best.pt",
    [int]$Epochs = 30,
    [int]$Patience = 3,
    [int]$BatchSize = 32,
    [int]$GradAccumSteps = 2,
    [int]$NumWorkers = 4,
    [int]$EvalNumWorkers = 2,
    [int]$MaxTrainBatches = 0,
    [int]$MaxValBatches = 0,
    [double]$AttentionViewLossWeight = 0.25,
    [double]$AttentionCropProbability = 0.35,
    [double]$AttentionDropProbability = 0.15,
    [int]$AttentionViewStartEpoch = 2,
    [double]$AttentionViewForegroundWeight = 0.80,
    [double]$AttentionDropMinAreaRatio = 0.06,
    [double]$AttentionDropMaxAreaRatio = 0.14,
    [double]$ElrLossWeight = 0.10,
    [double]$ElrBeta = 0.70,
    [int]$ElrStartEpoch = 2,
    [switch]$Probe,
    [switch]$PreflightOnly,
    [switch]$Smoke,
    [bool]$TraceArchitecture = $true
)

$launcher = Join-Path $PSScriptRoot "run_trkh_5class_attention_views_v8.ps1"

if ($Probe) {
    if (-not $PSBoundParameters.ContainsKey("RunName")) {
        $RunName = "probe_mango_cls_256_5class_surface_detail_elr_v10_120b_6e"
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
    RunName = $RunName
    ResumeCheckpoint = $ResumeCheckpoint
    Epochs = $Epochs
    Patience = $Patience
    BatchSize = $BatchSize
    GradAccumSteps = $GradAccumSteps
    NumWorkers = $NumWorkers
    EvalNumWorkers = $EvalNumWorkers
    MaxTrainBatches = $MaxTrainBatches
    MaxValBatches = $MaxValBatches
    AttentionViewLossWeight = $AttentionViewLossWeight
    AttentionCropProbability = $AttentionCropProbability
    AttentionDropProbability = $AttentionDropProbability
    AttentionViewStartEpoch = $AttentionViewStartEpoch
    AttentionViewScoreSource = "surface_detail"
    AttentionViewForegroundWeight = $AttentionViewForegroundWeight
    AttentionDropMinAreaRatio = $AttentionDropMinAreaRatio
    AttentionDropMaxAreaRatio = $AttentionDropMaxAreaRatio
    ElrLossWeight = $ElrLossWeight
    ElrBeta = $ElrBeta
    ElrStartEpoch = $ElrStartEpoch
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
