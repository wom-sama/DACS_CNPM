param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RunName = "mango_cls_256_5class_boundary_contrastive_v12_30e",
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
    [double]$AttentionViewLossWeight = 0.25,
    [double]$AttentionCropProbability = 0.35,
    [double]$AttentionDropProbability = 0.15,
    [int]$AttentionViewStartEpoch = 2,
    [double]$AttentionViewForegroundWeight = 0.80,
    [double]$AttentionDropMinAreaRatio = 0.06,
    [double]$AttentionDropMaxAreaRatio = 0.14,
    [double]$BoundaryContrastiveLossWeight = 0.08,
    [string]$BoundaryContrastivePairs = "0-1,1-2,2-3,4-rest",
    [string]$BoundaryContrastiveSources = "head,patch",
    [double]$BoundaryContrastiveMargin = 0.12,
    [double]$BoundaryContrastiveTemperature = 0.20,
    [int]$BoundaryContrastiveMaxPairs = 128,
    [double]$SampleWeightFactor = 1.0,
    [double]$SampleWeightMax = 2.50,
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
    [switch]$Probe,
    [switch]$PreflightOnly,
    [switch]$Smoke,
    [switch]$SkipFinalTest,
    [bool]$TraceArchitecture = $true
)

$launcher = Join-Path $PSScriptRoot "run_trkh_5class_attention_views_v8.ps1"

if ($Probe) {
    if (-not $PSBoundParameters.ContainsKey("RunName")) {
        $RunName = "probe_mango_cls_256_5class_boundary_contrastive_v12_120b_6e"
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
    ElrLossWeight = 0.0
    BoundaryContrastiveLossWeight = $BoundaryContrastiveLossWeight
    BoundaryContrastivePairs = $BoundaryContrastivePairs
    BoundaryContrastiveSources = $BoundaryContrastiveSources
    BoundaryContrastiveMargin = $BoundaryContrastiveMargin
    BoundaryContrastiveTemperature = $BoundaryContrastiveTemperature
    BoundaryContrastiveMaxPairs = $BoundaryContrastiveMaxPairs
    SampleWeightManifest = $SampleWeightManifest
    SampleWeightFactor = $SampleWeightFactor
    SampleWeightMax = $SampleWeightMax
    ForegroundSurfaceFusion = $ForegroundSurfaceFusion
    ForegroundSurfaceFusionDropout = $ForegroundSurfaceFusionDropout
    BilinearPatchFusion = $BilinearPatchFusion
    BilinearPatchRank = $BilinearPatchRank
    BilinearPatchDropout = $BilinearPatchDropout
    FrequencySelectivePooling = $FrequencySelectivePooling
    FrequencySelectiveTopK = $FrequencySelectiveTopK
    FrequencySelectiveBlend = $FrequencySelectiveBlend
    FrequencySelectiveForegroundThreshold = $FrequencySelectiveForegroundThreshold
    PairwiseMarginRouting = $PairwiseMarginRouting
    PairwiseMarginRouteMaxProbabilityMargin = $PairwiseMarginRouteMaxProbabilityMargin
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
