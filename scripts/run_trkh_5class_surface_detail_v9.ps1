param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RunName = "mango_cls_256_5class_surface_detail_v9_30e",
    [string]$ResumeCheckpoint = "runs\mango_cls_256_5class_attention_views_bounded_v8_30e\checkpoints\best.pt",
    [int]$Epochs = 30,
    [int]$Patience = 3,
    [int]$BatchSize = 32,
    [int]$GradAccumSteps = 2,
    [int]$NumWorkers = 4,
    [int]$EvalNumWorkers = 2,
    [int]$MaxTrainBatches = 0,
    [int]$MaxValBatches = 0,
    [double]$AttentionViewLossWeight = 0.30,
    [double]$AttentionCropProbability = 0.40,
    [double]$AttentionDropProbability = 0.20,
    [int]$AttentionViewStartEpoch = 2,
    [double]$AttentionViewForegroundWeight = 0.80,
    [double]$AttentionDropMinAreaRatio = 0.06,
    [double]$AttentionDropMaxAreaRatio = 0.16,
    [switch]$PreflightOnly,
    [switch]$Smoke,
    [bool]$TraceArchitecture = $true
)

$launcher = Join-Path $PSScriptRoot "run_trkh_5class_attention_views_v8.ps1"
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
