param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\class_f\data.yaml",
    [string]$RunName = "mango_cls_256_5class_routed_pairwise_v16_30e",
    [string]$ResumeCheckpoint = "runs\mango_cls_256_5class_attention_views_bounded_v8_30e\checkpoints\best.pt",
    [string]$SampleWeightManifest = "runs\boundary_sample_weights_v11_train_only_20260612\sample_weights_train_only.csv",
    [string]$HardSampleManifest = "runs\mango_cls_256_5class_defectstat_v3_30e\hard_mining_train_only\hard_samples_train_only.csv",
    [double]$HardSampleRepeatFactor = 1.6,
    [int]$ImageSize = 256,
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
    [double]$PairwiseMarginRouteMaxProbabilityMargin = 0.20,
    [double]$AngularMarginLossWeight = 0.0,
    [double]$AngularMargin = 0.12,
    [double]$AngularMarginScale = 16.0,
    [int]$AngularMarginStartEpoch = 2,
    [string]$AngularMarginClasses = "0,1,2,3",
    [double]$OrdinalBoundaryLossWeight = 0.0,
    [string]$OrdinalBoundaryClasses = "0,1,2,3",
    [string]$OrdinalBoundaryThresholdWeights = "1.25,1.25,1.0",
    [double]$OrdinalBoundaryTemperature = 1.0,
    [int]$OrdinalBoundaryStartEpoch = 1,
    [double]$PairwiseConfusionLossWeight = 0.0,
    [string]$PairwiseConfusionSources = "head",
    [int]$PairwiseConfusionStartEpoch = 1,
    [bool]$PairwiseConfusionNormalize = $true,
    [ValidateSet("ldam_focal", "balanced_softmax")]
    [string]$ClassificationLoss = "ldam_focal",
    [double]$BalancedSoftmaxTau = 1.0,
    [double]$FocalLossGamma = 1.0,
    [double]$FocalLossMix = 0.10,
    [double]$LabelSmoothing = 0.02,
    [double]$LdamMaxMargin = 0.30,
    [double]$LdamScale = 18.0,
    [double]$MutualChannelLossWeight = 0.0,
    [int]$MutualChannelTopK = 8,
    [double]$MutualChannelDiversityWeight = 0.20,
    [int]$MutualChannelStartEpoch = 1,
    [ValidateSet("none", "pseudo", "grabcut")]
    [string]$ForegroundCropMode = "none",
    [double]$ForegroundCropProbability = 0.0,
    [double]$ForegroundCropMarginRatio = 0.08,
    [double]$ForegroundCropMinMaskAreaRatio = 0.03,
    [double]$ForegroundCropMaxMaskAreaRatio = 0.92,
    [double]$ForegroundCropMaxCropAreaRatio = 0.98,
    [bool]$Sam = $false,
    [double]$SamRho = 0.03,
    [bool]$SamAdaptive = $false,
    [bool]$MixStyle = $false,
    [double]$MixStyleProbability = 0.5,
    [double]$MixStyleAlpha = 0.1,
    [string]$BackgroundSuppressionMode = "desaturate_blur",
    [double]$BackgroundSuppressionProbability = 0.80,
    [double]$BackgroundSuppressionMargin = 0.08,
    [double]$BackgroundSuppressionBlurRadius = 7.0,
    [double]$ForegroundBackgroundMixProbability = 0.0,
    [double]$ForegroundBackgroundMixMargin = 0.08,
    [double]$ForegroundBackgroundMixMinForegroundFraction = 0.06,
    [double]$ForegroundBackgroundMixMaxForegroundFraction = 0.88,
    [double]$ForegroundBackgroundMixSoftness = 5.0,
    [switch]$Probe,
    [switch]$PreflightOnly,
    [switch]$Smoke,
    [switch]$SkipFinalTest,
    [bool]$TraceArchitecture = $true
)

$launcher = Join-Path $PSScriptRoot "run_trkh_5class_boundary_contrastive_v12.ps1"

if ($Probe) {
    if (-not $PSBoundParameters.ContainsKey("RunName")) {
        $RunName = "probe_mango_cls_256_5class_routed_pairwise_v16_120b_6e"
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
    HardSampleManifest = $HardSampleManifest
    HardSampleRepeatFactor = $HardSampleRepeatFactor
    ImageSize = $ImageSize
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
    AttentionViewForegroundWeight = $AttentionViewForegroundWeight
    AttentionDropMinAreaRatio = $AttentionDropMinAreaRatio
    AttentionDropMaxAreaRatio = $AttentionDropMaxAreaRatio
    BoundaryContrastiveLossWeight = $BoundaryContrastiveLossWeight
    AngularMarginLossWeight = $AngularMarginLossWeight
    AngularMargin = $AngularMargin
    AngularMarginScale = $AngularMarginScale
    AngularMarginStartEpoch = $AngularMarginStartEpoch
    AngularMarginClasses = $AngularMarginClasses
    OrdinalBoundaryLossWeight = $OrdinalBoundaryLossWeight
    OrdinalBoundaryClasses = $OrdinalBoundaryClasses
    OrdinalBoundaryThresholdWeights = $OrdinalBoundaryThresholdWeights
    OrdinalBoundaryTemperature = $OrdinalBoundaryTemperature
    OrdinalBoundaryStartEpoch = $OrdinalBoundaryStartEpoch
    PairwiseConfusionLossWeight = $PairwiseConfusionLossWeight
    PairwiseConfusionSources = $PairwiseConfusionSources
    PairwiseConfusionStartEpoch = $PairwiseConfusionStartEpoch
    PairwiseConfusionNormalize = $PairwiseConfusionNormalize
    ClassificationLoss = $ClassificationLoss
    BalancedSoftmaxTau = $BalancedSoftmaxTau
    FocalLossGamma = $FocalLossGamma
    FocalLossMix = $FocalLossMix
    LabelSmoothing = $LabelSmoothing
    LdamMaxMargin = $LdamMaxMargin
    LdamScale = $LdamScale
    MutualChannelLossWeight = $MutualChannelLossWeight
    MutualChannelTopK = $MutualChannelTopK
    MutualChannelDiversityWeight = $MutualChannelDiversityWeight
    MutualChannelStartEpoch = $MutualChannelStartEpoch
    ForegroundCropMode = $ForegroundCropMode
    ForegroundCropProbability = $ForegroundCropProbability
    ForegroundCropMarginRatio = $ForegroundCropMarginRatio
    ForegroundCropMinMaskAreaRatio = $ForegroundCropMinMaskAreaRatio
    ForegroundCropMaxMaskAreaRatio = $ForegroundCropMaxMaskAreaRatio
    ForegroundCropMaxCropAreaRatio = $ForegroundCropMaxCropAreaRatio
    Sam = $Sam
    SamRho = $SamRho
    SamAdaptive = $SamAdaptive
    MixStyle = $MixStyle
    MixStyleProbability = $MixStyleProbability
    MixStyleAlpha = $MixStyleAlpha
    ForegroundSurfaceFusion = $false
    BilinearPatchFusion = $false
    FrequencySelectivePooling = $false
    PairwiseMarginRouting = $true
    PairwiseMarginRouteMaxProbabilityMargin = $PairwiseMarginRouteMaxProbabilityMargin
    BackgroundSuppressionMode = $BackgroundSuppressionMode
    BackgroundSuppressionProbability = $BackgroundSuppressionProbability
    BackgroundSuppressionMargin = $BackgroundSuppressionMargin
    BackgroundSuppressionBlurRadius = $BackgroundSuppressionBlurRadius
    ForegroundBackgroundMixProbability = $ForegroundBackgroundMixProbability
    ForegroundBackgroundMixMargin = $ForegroundBackgroundMixMargin
    ForegroundBackgroundMixMinForegroundFraction = $ForegroundBackgroundMixMinForegroundFraction
    ForegroundBackgroundMixMaxForegroundFraction = $ForegroundBackgroundMixMaxForegroundFraction
    ForegroundBackgroundMixSoftness = $ForegroundBackgroundMixSoftness
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
