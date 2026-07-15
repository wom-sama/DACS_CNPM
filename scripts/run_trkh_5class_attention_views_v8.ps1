param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\class_f\data.yaml",
    [string]$ClassificationFolderYoloData = "",
    [string]$AuxiliaryTrainData = "",
    [string]$AuxiliaryTrainClassificationFolderYoloData = "",
    [double]$AuxiliaryTrainWeight = 1.0,
    [bool]$PairedViewTrain = $false,
    [string]$RunName = "mango_cls_256_5class_attention_views_v8_30e",
    [string]$ResumeCheckpoint = "runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\checkpoints\best.pt",
    [int]$ImageSize = 256,
    [ValidateSet("conv_pool", "coatnet_mbconv")]
    [string]$StemArchitecture = "conv_pool",
    [ValidateSet("max", "soft", "max_soft")]
    [string]$StemPoolingMode = "max",
    [double]$StemSoftpoolBlend = 0.15,
    [bool]$ShiftedPatchTokenization = $false,
    [int]$ShiftedPatchShift = 1,
    [double]$ShiftedPatchResidualScale = 0.10,
    [double]$CropMarginRatio = 0.05,
    [double]$ClassCropMarginScaleThreshold = 1.5,
    [double]$ClassCropMarginMaxRatio = 0.16,
    [string]$ClassConditionalAugmentationScales = "",
    [bool]$ClassificationSourceContext = $false,
    [ValidateSet("dim", "gray", "blur", "mean", "desaturate_blur", "blur_gray")]
    [string]$ClassificationSourceContextMode = "desaturate_blur",
    [ValidateSet("full", "crop_inset")]
    [string]$ClassificationSourceContextLayout = "full",
    [double]$ClassificationSourceContextMarginRatio = 0.12,
    [double]$ClassificationSourceContextBackgroundAlpha = 0.35,
    [double]$ClassificationSourceContextBlurRadius = 7.0,
    [double]$ClassificationSourceContextInsetScale = 0.34,
    [bool]$ClassificationSourceContextAux = $false,
    [double]$SourceContextAuxLossWeight = 0.0,
    [double]$SourceContextAuxClassificationWeight = 0.0,
    [double]$SourceContextAuxConsistencyWeight = 0.0,
    [double]$SourceContextAuxBboxMarginRatio = 0.04,
    [double]$SourceContextAuxAttentionTemperature = 0.20,
    [bool]$TokenPruning = $true,
    [double]$TokenPruneBboxWeight = 0.0,
    [double]$TokenPruneBboxMarginRatio = 0.04,
    [ValidateSet("bbox", "crop_bbox")]
    [string]$BboxTokenPriorSource = "bbox",
    [double]$EarlyTokenMaskKeepRate = 1.0,
    [int]$Epochs = 30,
    [int]$SchedulerTotalEpochs = 0,
    [int]$Patience = 3,
    [int]$BatchSize = 32,
    [int]$GradAccumSteps = 2,
    [int]$NumWorkers = 4,
    [int]$EvalNumWorkers = 2,
    [int]$MaxTrainBatches = 0,
    [int]$MaxValBatches = 0,
    [int]$Seed = 42,
    [bool]$DisableBalancedEpochSampling = $false,
    [bool]$DataCartography = $false,
    [string]$DataCartographyOutput = "",
    [double]$LearningRate = 8e-5,
    [double]$MinLearningRate = 1e-6,
    [string]$TrainableModulePrefixes = "",
    [bool]$ClassificationMlpHead = $false,
    [int]$ClassificationMlpHiddenDim = 512,
    [double]$ClassificationMlpDropout = 0.08,
    [double]$ClassificationMlpResidualScale = 0.20,
    [int]$BranchCnnTokens = 0,
    [int]$WarmupEpochs = 1,
    [double]$WeightDecay = 0.05,
    [double]$BackboneLrScale = 1.0,
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
    [double]$SelfAdaptiveTargetLossWeight = 0.0,
    [double]$SelfAdaptiveTargetBeta = 0.90,
    [int]$SelfAdaptiveTargetStartEpoch = 2,
    [double]$SelfAdaptiveTargetHardWeight = 0.10,
    [double]$SelfAdaptiveTargetConfidencePower = 1.0,
    [double]$SelfAdaptiveTargetMinConfidence = 0.0,
    [double]$RDropLossWeight = 0.0,
    [double]$RDropTemperature = 1.0,
    [double]$AugmixConsistencyLossWeight = 0.0,
    [double]$AugmixConsistencyProbability = 0.0,
    [double]$AugmixConsistencySeverity = 0.18,
    [int]$AugmixConsistencyWidth = 2,
    [int]$AugmixConsistencyDepth = 2,
    [double]$AugmixConsistencyAlpha = 1.0,
    [double]$AugmixConsistencyTemperature = 1.0,
    [double]$IlluminationConsistencyLossWeight = 0.0,
    [double]$IlluminationConsistencyProbability = 0.0,
    [double]$IlluminationConsistencyBrightness = 0.08,
    [double]$IlluminationConsistencyContrast = 0.08,
    [double]$IlluminationConsistencyGamma = 0.12,
    [double]$IlluminationConsistencyTemperature = 1.0,
    [double]$ForegroundChromaConsistencyLossWeight = 0.0,
    [double]$ForegroundChromaConsistencyProbability = 0.0,
    [double]$ForegroundChromaConsistencySaturationDelta = 0.10,
    [double]$ForegroundChromaConsistencyHueDelta = 0.015,
    [double]$ForegroundChromaConsistencyBboxMarginRatio = 0.02,
    [double]$ForegroundChromaConsistencyTemperature = 1.0,
    [string]$ForegroundChromaConsistencyClasses = "0,1,2,3,4",
    [double]$FriendlyAdversarialLossWeight = 0.0,
    [double]$FriendlyAdversarialEpsilon = (2.0 / 255.0),
    [double]$FriendlyAdversarialStepSize = (1.0 / 255.0),
    [int]$FriendlyAdversarialSteps = 2,
    [double]$FriendlyAdversarialBboxErodeRatio = 0.10,
    [int]$FriendlyAdversarialFocusClass = 1,
    [string]$FriendlyAdversarialNegativeClasses = "0,2,4",
    [int]$FriendlyAdversarialMaxPerDirection = 8,
    [int]$FriendlyAdversarialStartEpoch = 1,
    [double]$SemanticAttributeLossWeight = 0.0,
    [string]$SemanticAttributeSpecs = "maturity:0,1|2,3|4;transport:0,2|1|3,4;quality:0,1,2|3|4",
    [double]$ConfusionPairMixupLossWeight = 0.0,
    [double]$ConfusionPairMixupAlpha = 0.40,
    [string]$ConfusionPairMixupPairs = "0-1,1-2,1-4,2-3",
    [int]$ConfusionPairMixupMaxPairs = 64,
    [int]$ConfusionPairMixupStartEpoch = 1,
    [double]$ForegroundSnapmixLossWeight = 0.0,
    [double]$ForegroundSnapmixProbability = 0.0,
    [double]$ForegroundSnapmixAlpha = 1.0,
    [string]$ForegroundSnapmixPairs = "0-1,1-2,1-4,2-3",
    [double]$ForegroundSnapmixMinAreaRatio = 0.06,
    [double]$ForegroundSnapmixMaxAreaRatio = 0.22,
    [double]$ForegroundSnapmixBboxMarginRatio = 0.02,
    [double]$ForegroundCounterexampleMixLossWeight = 0.0,
    [double]$ForegroundCounterexampleMixProbability = 0.0,
    [int]$ForegroundCounterexampleMixSourceClass = 1,
    [string]$ForegroundCounterexampleMixTargetClasses = "0,2,4",
    [double]$ForegroundCounterexampleMixAlpha = 1.0,
    [double]$ForegroundCounterexampleMixMinAreaRatio = 0.04,
    [double]$ForegroundCounterexampleMixMaxAreaRatio = 0.12,
    [double]$ForegroundCounterexampleMixBboxMarginRatio = 0.01,
    [int]$ForegroundCounterexampleMixStartEpoch = 1,
    [double]$DclRegionShuffleLossWeight = 0.0,
    [double]$DclRegionShuffleProbability = 0.0,
    [int]$DclRegionShuffleGridSize = 4,
    [double]$DclRegionShuffleBboxMarginRatio = 0.02,
    [string]$DclRegionShuffleClasses = "0,1,2,3,4",
    [int]$DclRegionShuffleStartEpoch = 1,
    [double]$QuantizedLabelCpuLossWeight = 0.0,
    [string]$QuantizedLabelCpuClasses = "0,1,2,4",
    [double]$QuantizedLabelCpuMinPrior = 0.02,
    [double]$QuantizedLabelCpuMaxPrior = 0.60,
    [double]$QuantizedLabelCpuNegativeWeight = 1.0,
    [double]$QuantizedLabelCpuNonNegativeBeta = 0.0,
    [int]$QuantizedLabelCpuStartEpoch = 1,
    [double]$SelfPacedLossWeight = 0.0,
    [double]$SelfPacedLossPercentile = 0.80,
    [double]$SelfPacedLossGamma = 1.0,
    [double]$SelfPacedLossMinWeight = 0.35,
    [int]$SelfPacedLossStartEpoch = 1,
    [bool]$SelfPacedLossClassBalanced = $false,
    [double]$CyflodLossDampingWeight = 0.0,
    [double]$CyflodLossDampingDelta = 0.25,
    [int]$CyflodLossDampingCycleEpochs = 2,
    [double]$CyflodLossDampingMinWeight = 0.10,
    [int]$CyflodLossDampingStartEpoch = 1,
    [double]$MetricLearningLossWeight = 0.04,
    [double]$MetricLearningTemperature = 0.16,
    [string]$MetricLearningSources = "head,patch",
    [double]$TeacherGuidedContrastiveLossWeight = 0.0,
    [double]$TeacherGuidedContrastiveTemperature = 0.16,
    [string]$TeacherGuidedContrastiveSources = "head,patch",
    [string]$TeacherGuidedContrastiveClasses = "0,1,2,4",
    [double]$TeacherGuidedContrastiveTeacherMinConfidence = 0.70,
    [bool]$TeacherGuidedContrastiveRequireAgreement = $true,
    [bool]$TeacherGuidedContrastiveClassBalanced = $true,
    [ValidateSet("filter", "confidence", "confidence_margin")]
    [string]$TeacherGuidedContrastiveWeightMode = "filter",
    [double]$TeacherGuidedContrastiveStochasticStd = 0.0,
    [double]$TeacherGuidedContrastiveMinReliability = 0.0,
    [double]$TeacherGuidedContrastiveTeacherConfidencePower = 1.0,
    [int]$TeacherGuidedContrastiveMemoryQueueSize = 0,
    [int]$TeacherGuidedContrastiveMemoryMinCount = 2,
    [double]$BoundaryContrastiveLossWeight = 0.0,
    [string]$BoundaryContrastivePairs = "0-1,1-2,2-3,4-rest",
    [string]$BoundaryContrastiveSources = "head,patch",
    [double]$BoundaryContrastiveMargin = 0.12,
    [double]$BoundaryContrastiveTemperature = 0.20,
    [int]$BoundaryContrastiveMaxPairs = 128,
    [double]$BoundaryCenterLossWeight = 0.0,
    [string]$BoundaryCenterPairs = "0-1,1-2,2-3,1-4",
    [string]$BoundaryCenterSources = "head,patch",
    [double]$BoundaryCenterMargin = 0.10,
    [double]$BoundaryCenterTemperature = 0.20,
    [double]$BoundaryCenterCompactnessWeight = 0.10,
    [double]$BoundaryCenterTeacherMinConfidence = 0.0,
    [bool]$BoundaryCenterRequireAgreement = $false,
    [ValidateSet("filter", "confidence", "confidence_margin")]
    [string]$BoundaryCenterTeacherWeightMode = "filter",
    [double]$BorderAttentionSuppressionLossWeight = 0.0,
    [int]$BorderAttentionSuppressionFrameWidth = 1,
    [double]$BorderAttentionSuppressionBboxBand = 0.12,
    [double]$BorderAttentionSuppressionBboxWeight = 0.50,
    [double]$BorderAttentionSuppressionTemperature = 0.20,
    [string]$BorderAttentionSuppressionClasses = "0,1,2,4",
    [int]$BorderAttentionSuppressionStartEpoch = 1,
    [double]$RegisterAttentionAlignmentLossWeight = 0.0,
    [string]$RegisterAttentionAlignmentClasses = "0,1,2,4",
    [double]$RegisterAttentionAlignmentBboxMarginRatio = 0.0,
    [double]$RegisterAttentionAlignmentAgreementWeight = 1.0,
    [double]$RegisterAttentionAlignmentForegroundWeight = 0.25,
    [int]$RegisterAttentionAlignmentStartEpoch = 1,
    [double]$RegisterDiversityLossWeight = 0.0,
    [double]$ForegroundSurfaceAuxLossWeight = 0.0,
    [double]$ForegroundSurfacePairwiseLossWeight = 0.0,
    [double]$InteriorBoundaryPairwiseLossWeight = 0.0,
    [double]$AngularMarginLossWeight = 0.0,
    [double]$AngularMargin = 0.12,
    [double]$AngularMarginScale = 16.0,
    [int]$AngularMarginStartEpoch = 2,
    [string]$AngularMarginClasses = "0,1,2,3",
    [double]$SubcenterProxyLossWeight = 0.0,
    [int]$SubcenterProxySubcenters = 3,
    [double]$SubcenterProxyMargin = 0.08,
    [double]$SubcenterProxyScale = 12.0,
    [string]$SubcenterProxyClasses = "0,1,2,4",
    [int]$SubcenterProxyStartEpoch = 1,
    [double]$SubcenterProxyDropout = 0.0,
    [double]$SubcenterProxyInitStd = 0.002,
    [double]$DeepAbstentionLossWeight = 0.0,
    [double]$DeepAbstentionPenalty = 1.30,
    [int]$DeepAbstentionStartEpoch = 2,
    [double]$DeepAbstentionDropout = 0.05,
    [double]$DeepAbstentionInitialProbability = 0.01,
    [double]$OrdinalBoundaryLossWeight = 0.0,
    [string]$OrdinalBoundaryClasses = "0,1,2,3",
    [string]$OrdinalBoundaryThresholdWeights = "1.25,1.25,1.0",
    [double]$OrdinalBoundaryTemperature = 1.0,
    [int]$OrdinalBoundaryStartEpoch = 1,
    [double]$PairwiseConfusionLossWeight = 0.0,
    [string]$PairwiseConfusionSources = "head",
    [int]$PairwiseConfusionStartEpoch = 1,
    [bool]$PairwiseConfusionNormalize = $true,
    [double]$ConfusionSpectralLossWeight = 0.0,
    [double]$ConfusionSpectralEmaMomentum = 0.5,
    [double]$ConfusionSpectralFrequencySmoothing = 0.2,
    [double]$ConfusionSpectralMargin = 0.1,
    [int]$ConfusionSpectralStartEpoch = 1,
    [bool]$ConfusionSpectralBidirectional = $false,
    [string]$SampleWeightManifest = "",
    [double]$SampleWeightFactor = 1.0,
    [double]$SampleWeightMax = 5.0,
    [string]$QualityGroupManifest = "",
    [double]$GroupDroLossWeight = 0.0,
    [double]$GroupDroTemperature = 0.35,
    [int]$GroupDroMinSamples = 1,
    [string]$HardSampleManifest = "",
    [double]$HardSampleRepeatFactor = 1.6,
    [string]$AmbiguousSoftTargetManifest = "",
    [double]$AmbiguousSoftTargetAlpha = 0.25,
    [string]$TargetedMarginManifest = "",
    [double]$TargetedMarginLossWeight = 0.0,
    [double]$TargetedMarginDefaultMargin = 0.12,
    [double]$TargetedMarginDefaultWeight = 1.0,
    [double]$TargetedMarginMaxWeight = 3.0,
    [string]$FocusNeighborBinaryManifest = "",
    [double]$FocusNeighborBinaryLossWeight = 0.0,
    [int]$FocusNeighborBinaryFocusClass = 1,
    [string]$FocusNeighborBinaryNeighborClasses = "0,2,4",
    [double]$FocusNeighborBinaryDefaultWeight = 1.0,
    [double]$FocusNeighborBinaryMaxWeight = 3.0,
    [double]$FocusedFalsePositiveMarginLossWeight = 0.0,
    [int]$FocusedFalsePositiveClass = 1,
    [string]$FocusedFalsePositiveNegativeClasses = "0,4",
    [double]$FocusedFalsePositiveMargin = 0.10,
    [double]$FocusedFalsePositiveMinProbability = 0.05,
    [double]$FocusedFalsePositiveProbabilityPower = 1.5,
    [double]$BackgroundCounterfactualConsistencyWeight = 0.0,
    [double]$BackgroundCounterfactualProbability = 0.0,
    [ValidateSet("gray", "blur", "mean", "desaturate_blur")]
    [string]$BackgroundCounterfactualMode = "desaturate_blur",
    [double]$BackgroundCounterfactualMargin = 0.08,
    [int]$BackgroundCounterfactualBlurKernel = 15,
    [double]$BackgroundCounterfactualTemperature = 1.0,
    [double]$BackgroundFocusSuppressionLossWeight = 0.0,
    [double]$BackgroundFocusSuppressionProbability = 0.0,
    [int]$BackgroundFocusSuppressionFocusClass = 1,
    [string]$BackgroundFocusSuppressionNegativeClasses = "0,4",
    [double]$BackgroundFocusSuppressionMargin = 0.015,
    [double]$BackgroundFocusSuppressionMinProbability = 0.05,
    [double]$BackgroundFocusSuppressionProbabilityPower = 1.5,
    [double]$SourceContextFocusSuppressionLossWeight = 0.0,
    [double]$SourceContextFocusSuppressionProbability = 0.0,
    [int]$SourceContextFocusSuppressionFocusClass = 1,
    [string]$SourceContextFocusSuppressionNegativeClasses = "0,2,4",
    [double]$SourceContextFocusSuppressionMargin = 0.015,
    [double]$SourceContextFocusSuppressionMinProbability = 0.05,
    [double]$SourceContextFocusSuppressionProbabilityPower = 1.5,
    [double]$SurfaceCounterfactualConsistencyWeight = 0.0,
    [double]$SurfaceCounterfactualProbability = 0.0,
    [ValidateSet("rgb_unsharp", "luma_residual", "foreground_unsharp", "foreground_luma")]
    [string]$SurfaceCounterfactualMode = "foreground_luma",
    [double]$SurfaceCounterfactualStrength = 0.18,
    [int]$SurfaceCounterfactualBlurKernel = 7,
    [double]$SurfaceCounterfactualTemperature = 1.0,
    [double]$SurfaceAmplifiedSupervisedLossWeight = 0.0,
    [double]$SurfaceAmplifiedBoundaryMarginLossWeight = 0.0,
    [double]$SurfaceAmplifiedProbability = 0.0,
    [ValidateSet("rgb_unsharp", "luma_residual", "foreground_unsharp", "foreground_luma")]
    [string]$SurfaceAmplifiedMode = "foreground_luma",
    [double]$SurfaceAmplifiedStrength = 0.22,
    [int]$SurfaceAmplifiedBlurKernel = 7,
    [string]$SurfaceAmplifiedBoundaryPairs = "0-1,1-2,2-3,1-4,4-rest",
    [double]$SurfaceAmplifiedBoundaryMargin = 0.10,
    [double]$PairedViewSupervisedLossWeight = 0.0,
    [double]$PairedViewConsistencyWeight = 0.0,
    [double]$PairedViewFeatureConsistencyWeight = 0.0,
    [double]$PairedViewFusionLossWeight = 0.0,
    [double]$PairedViewFusionConsistencyWeight = 0.0,
    [double]$PairedViewTemperature = 1.0,
    [ValidateSet("head", "cnn", "patch", "patch_tokens", "registers")]
    [string]$PairedViewFeatureSource = "head",
    [double]$MaskedReconstructionLossWeight = 0.0,
    [double]$MaskedReconstructionMaskRatio = 0.45,
    [double]$MaskedReconstructionForegroundWeight = 0.70,
    [double]$MaskedReconstructionDetailWeight = 0.25,
    [double]$MaskedReconstructionBboxWeight = 0.35,
    [double]$MaskedReconstructionBboxMarginRatio = 0.04,
    [string]$DistillationTeacherCsv = "",
    [double]$DistillationWeight = 0.10,
    [double]$DistillationTemperature = 2.0,
    [int]$DistillationFocusClassIndex = 1,
    [double]$DistillationFocusClassWeight = 1.5,
    [double]$TeacherNonTargetDistillationLossWeight = 0.0,
    [string]$TeacherNonTargetDistillationClasses = "0,1,2,4",
    [double]$TeacherNonTargetDistillationTemperature = 2.0,
    [double]$TeacherNonTargetDistillationTeacherMinConfidence = 0.0,
    [bool]$TeacherNonTargetDistillationRequireAgreement = $true,
    [double]$TeacherFocusMarginLossWeight = 0.0,
    [int]$TeacherFocusMarginFocusClass = 1,
    [string]$TeacherFocusMarginNegativeClasses = "0,2,4",
    [double]$TeacherFocusMarginTeacherMaxProbability = 0.20,
    [double]$TeacherFocusMarginMargin = 0.08,
    [double]$TeacherFocusMarginMinProbability = 0.05,
    [double]$TeacherFocusMarginProbabilityPower = 1.5,
    [bool]$TeacherFocusMarginRequireAgreement = $true,
    [double]$TeacherFocusBinaryLossWeight = 0.0,
    [int]$TeacherFocusBinaryFocusClass = 1,
    [string]$TeacherFocusBinaryClasses = "0,1,2,4",
    [double]$TeacherFocusBinaryTeacherMinConfidence = 0.0,
    [double]$TeacherFocusBinaryErrorPower = 0.0,
    [double]$TeacherFocusBinaryHardTargetBlend = 0.0,
    [bool]$TeacherFocusBinaryRequireAgreement = $true,
    [double]$TeacherPairwiseMarginLossWeight = 0.0,
    [double]$TeacherPairwiseMarginTeacherMassThreshold = 0.0,
    [double]$TeacherPairwiseMarginErrorPower = 0.0,
    [double]$TeacherPairwiseMarginHardTargetBlend = 0.0,
    [bool]$TeacherPairwiseMarginRequireAgreement = $true,
    [string]$TeacherFeatureNpz = "",
    [double]$TeacherFeatureRkdLossWeight = 0.0,
    [double]$TeacherFeatureRkdDistanceWeight = 1.0,
    [double]$TeacherFeatureRkdAngleWeight = 0.0,
    [ValidateSet("head", "cnn", "patch", "registers")]
    [string]$TeacherFeatureRkdSource = "head",
    [ValidateSet("all", "boundary", "intra", "boundary_or_intra")]
    [string]$TeacherFeatureRkdPairMode = "all",
    [string]$TeacherFeatureRkdPairs = "0-1,1-2,1-4,2-3",
    [double]$TeacherFeatureContrastiveLossWeight = 0.0,
    [double]$TeacherFeatureContrastiveTemperature = 0.20,
    [int]$TeacherFeatureContrastiveProjectionDim = 128,
    [bool]$TeacherFeatureContrastiveUseProjectionAdapter = $false,
    [double]$TeacherFeatureContrastiveAdapterDropout = 0.0,
    [ValidateSet("head", "cnn", "patch", "registers")]
    [string]$TeacherFeatureContrastiveSource = "head",
    [ValidateSet("all", "boundary", "intra", "boundary_or_intra")]
    [string]$TeacherFeatureContrastivePairMode = "boundary",
    [string]$TeacherFeatureContrastivePairs = "0-1,1-2,1-4,2-3",
    [int]$TeacherFeatureContrastiveStartEpoch = 1,
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
    [bool]$ColorStatFusion = $false,
    [double]$ColorStatFusionDropout = 0.08,
    [bool]$DefectStatFusion = $false,
    [double]$DefectStatFusionDropout = 0.08,
    [bool]$ForegroundSurfaceFusion = $false,
    [double]$ForegroundSurfaceFusionDropout = 0.08,
    [bool]$ForegroundSurfacePairwiseHead = $false,
    [string]$ForegroundSurfacePairwisePairs = "0-1,1-2,1-4,2-3",
    [double]$ForegroundSurfacePairwiseLogitScale = 0.18,
    [double]$ForegroundSurfacePairwiseDropout = 0.05,
    [bool]$ForegroundSurfacePairwiseRouting = $true,
    [double]$ForegroundSurfacePairwiseRouteMaxProbabilityMargin = 0.22,
    [bool]$InteriorBoundaryPairwiseHead = $false,
    [string]$InteriorBoundaryPairwisePairs = "0-1,4-1",
    [double]$InteriorBoundaryPairwiseLogitScale = 0.14,
    [double]$InteriorBoundaryPairwiseDropout = 0.05,
    [int]$InteriorBoundaryPairwiseHiddenDim = 128,
    [int]$InteriorBoundaryPairwiseErodeKernel = 9,
    [bool]$InteriorBoundaryPairwiseRouting = $true,
    [double]$InteriorBoundaryPairwiseRouteMaxProbabilityMargin = 0.22,
    [bool]$BBoxSpatialFusion = $false,
    [int]$BBoxSpatialFusionHiddenDim = 64,
    [double]$BBoxSpatialFusionDropout = 0.05,
    [double]$BBoxSpatialFusionLogitScale = 0.20,
    [bool]$PatchObjectnessGuidedHead = $false,
    [int]$PatchObjectnessHiddenDim = 128,
    [double]$PatchObjectnessDropout = 0.05,
    [double]$PatchObjectnessLogitScale = 0.12,
    [double]$PatchObjectnessTemperature = 0.75,
    [double]$PatchObjectnessLossWeight = 0.0,
    [double]$PatchObjectnessPositiveWeight = 1.0,
    [double]$BboxTokenLabelLossWeight = 0.0,
    [double]$BboxTokenLabelMinPrior = 0.45,
    [double]$BboxTokenLabelPriorPower = 1.0,
    [string]$BboxTokenLabelClasses = "0,1,2,4",
    [int]$BboxTokenLabelFocusClass = 1,
    [double]$BboxTokenLabelFocusWeight = 1.25,
    [bool]$BboxTokenLabelDetachClassifier = $true,
    [int]$BboxTokenLabelStartEpoch = 1,
    [bool]$PatchEvidenceRouterHead = $false,
    [string]$PatchEvidenceRouterPair = "0-1",
    [int]$PatchEvidenceRouterHiddenDim = 128,
    [int]$PatchEvidenceRouterTopK = 4,
    [double]$PatchEvidenceRouterBboxWeight = 0.75,
    [double]$PatchEvidenceRouterDropout = 0.05,
    [double]$PatchEvidenceRouterLogitScale = 0.12,
    [string]$PatchEvidenceRouterMarginPriorMode = "none",
    [double]$PatchEvidenceRouterMarginPriorScale = 0.0,
    [bool]$PatchEvidenceRouterSummaryStats = $false,
    [bool]$PatchEvidenceRouterRouting = $true,
    [double]$PatchEvidenceRouterRouteMaxProbabilityMargin = 0.25,
    [double]$PatchEvidenceRouterRouteMinPairProbability = 0.02,
    [double]$PatchEvidenceRouterLossWeight = 0.0,
    [double]$PatchEvidenceRouterPositiveWeight = 1.0,
    [int]$PatchEvidenceRouterStartEpoch = 1,
    [string]$PatchEvidenceRouterTeacherCsv = "",
    [double]$PatchEvidenceRouterTeacherLossWeight = 0.0,
    [double]$PatchEvidenceRouterTeacherMinConfidence = 0.60,
    [double]$PatchEvidenceRouterTeacherMinPairMass = 0.20,
    [double]$PatchEvidenceRouterTeacherPositiveWeight = 1.0,
    [string]$PatchEvidenceLinearVerifierJson = "",
    [string]$PatchEvidenceLinearVerifierPair = "0-1",
    [double]$PatchEvidenceLinearVerifierMinPairProbability = 0.02,
    [double]$PatchEvidenceLinearVerifierMaxPairMargin = 0.40,
    [double]$PatchEvidenceLinearVerifierConfidenceThreshold = 0.60,
    [double]$PatchEvidenceLinearVerifierLogitBoost = 0.01,
    [double]$PatchEvidenceLinearVerifierProtectRightMinProbability = 0.0,
    [bool]$PatchEvidenceLinearVerifierTrainingSoftAdjustment = $false,
    [double]$PatchEvidenceLinearVerifierTrainingSoftLogitScale = 0.05,
    [double]$PatchEvidenceLinearVerifierTrainingSoftGateTemperature = 0.05,
    [bool]$BBoxPriorPatchContextHead = $false,
    [int]$BBoxPriorPatchContextHiddenDim = 128,
    [double]$BBoxPriorPatchContextDropout = 0.05,
    [double]$BBoxPriorPatchContextLogitScale = 0.12,
    [double]$BBoxPriorPatchContextTemperature = 0.50,
    [bool]$SourceContextFeatureFusion = $false,
    [int]$SourceContextFusionHiddenDim = 128,
    [double]$SourceContextFusionDropout = 0.05,
    [double]$SourceContextFusionLogitScale = 0.20,
    [double]$SourceContextFusionGateBias = -2.0,
    [bool]$PairedViewFeatureFusion = $false,
    [int]$PairedViewFusionHiddenDim = 128,
    [double]$PairedViewFusionDropout = 0.05,
    [double]$PairedViewFusionLogitScale = 0.12,
    [double]$PairedViewFusionGateBias = -2.0,
    [bool]$BilinearPatchFusion = $false,
    [int]$BilinearPatchRank = 32,
    [double]$BilinearPatchDropout = 0.08,
    [bool]$ComplementaryPatchSuppressionHead = $false,
    [int]$ComplementaryPatchSuppressionTopK = 6,
    [int]$ComplementaryPatchSuppressionHiddenDim = 128,
    [double]$ComplementaryPatchSuppressionDropout = 0.05,
    [double]$ComplementaryPatchSuppressionTemperature = 0.20,
    [double]$ComplementaryPatchSuppressionStrength = 0.85,
    [double]$ComplementaryPatchSuppressionBboxWeight = 0.35,
    [double]$ComplementaryPatchSuppressionLogitScale = 0.12,
    [bool]$MicroDetailPatchExpert = $false,
    [int]$MicroDetailTopK = 8,
    [int]$MicroDetailHiddenDim = 128,
    [double]$MicroDetailDropout = 0.08,
    [double]$MicroDetailTemperature = 0.12,
    [double]$MicroDetailForegroundPower = 1.0,
    [double]$MicroDetailLogitScale = 0.18,
    [bool]$MicroDetailRouting = $true,
    [string]$MicroDetailRoutePairs = "0-1,1-2,2-3,1-4,4-rest",
    [double]$MicroDetailRouteMaxProbabilityMargin = 0.25,
    [double]$MicroDetailAuxLossWeight = 0.0,
    [bool]$PartTokenLearner = $false,
    [int]$PartTokenCount = 4,
    [int]$PartTokenHiddenDim = 128,
    [double]$PartTokenDropout = 0.08,
    [double]$PartTokenTemperature = 0.70,
    [double]$PartTokenForegroundPower = 1.0,
    [double]$PartTokenBboxWeight = 0.75,
    [double]$PartTokenLogitScale = 0.16,
    [bool]$PartTokenRouting = $true,
    [string]$PartTokenRoutePairs = "0-1,1-2,4-1,2-3",
    [double]$PartTokenRouteMaxProbabilityMargin = 0.25,
    [double]$PartTokenAuxLossWeight = 0.0,
    [bool]$PartTokenPairwiseHead = $false,
    [string]$PartTokenPairwisePairs = "0-1,1-2,4-1,2-3",
    [double]$PartTokenPairwiseLogitScale = 0.18,
    [double]$PartTokenPairwiseDropout = 0.08,
    [bool]$PartTokenPairwiseRouting = $true,
    [double]$PartTokenPairwiseRouteMaxProbabilityMargin = 0.22,
    [double]$PartTokenPairwiseLossWeight = 0.04,
    [bool]$LocalZoomImageExpert = $false,
    [int]$LocalZoomCropSize = 128,
    [double]$LocalZoomCropScale = 0.48,
    [ValidateSet("foreground_detail", "defect_spot", "defect_spot_interior")]
    [string]$LocalZoomScoreMode = "foreground_detail",
    [int]$LocalZoomHiddenDim = 128,
    [double]$LocalZoomDropout = 0.08,
    [double]$LocalZoomLogitScale = 0.16,
    [bool]$LocalZoomRouting = $true,
    [string]$LocalZoomRoutePairs = "0-1,1-2,2-3,1-4,4-rest",
    [double]$LocalZoomRouteMaxProbabilityMargin = 0.25,
    [double]$LocalZoomAuxLossWeight = 0.0,
    [bool]$HighFrequencyTextureExpert = $false,
    [int]$HighFrequencyTextureHiddenDim = 128,
    [double]$HighFrequencyTextureDropout = 0.08,
    [int]$HighFrequencyTextureAnalysisSize = 96,
    [double]$HighFrequencyTextureLogitScale = 0.16,
    [bool]$HighFrequencyTextureRouting = $true,
    [string]$HighFrequencyTextureRoutePairs = "0-1,1-2,2-3,1-4,4-rest",
    [double]$HighFrequencyTextureRouteMaxProbabilityMargin = 0.25,
    [double]$HighFrequencyTextureAuxLossWeight = 0.0,
    [double]$HighFrequencyTexturePairwiseLossWeight = 0.0,
    [string]$HighFrequencyTexturePairwisePairs = "0-1,1-2,2-3,1-4",
    [bool]$MultiGranularityAuxHeads = $false,
    [string]$MultiGranularityAuxLayers = "2,5,8",
    [double]$MultiGranularityAuxDropout = 0.08,
    [double]$MultiGranularityAuxLossWeight = 0.0,
    [double]$MultiGranularityRefinementLossWeight = 0.0,
    [double]$MultiGranularityRefinementTemperature = 32.0,
    [double]$MultiGranularityContrastiveLossWeight = 0.0,
    [double]$MultiGranularityContrastiveTemperature = 0.18,
    [string]$MultiGranularityContrastivePairs = "0-1,1-2,1-4,2-3",
    [double]$MultiGranularityContrastiveTeacherMinConfidence = 0.0,
    [bool]$MultiGranularityContrastiveRequireAgreement = $true,
    [ValidateSet("filter","confidence","confidence_margin")]
    [string]$MultiGranularityContrastiveWeightMode = "filter",
    [double]$MultiGranularityContrastiveTeacherConfidencePower = 1.0,
    [bool]$SelfBoostingAttentionHead = $false,
    [double]$SelfBoostingAttentionLossWeight = 0.0,
    [double]$SelfBoostingAttentionTemperature = 0.40,
    [string]$SelfBoostingAttentionClasses = "0,1,2,3,4",
    [int]$SelfBoostingAttentionStartEpoch = 1,
    [bool]$BlockLocalPatchMixer = $false,
    [string]$BlockLocalPatchMixerLayers = "6,7,8",
    [double]$BlockLocalPatchMixerDropout = 0.0,
    [double]$BlockLocalPatchMixerScale = 0.10,
    [bool]$LocallyEnhancedFfn = $false,
    [string]$LocallyEnhancedFfnLayers = "1,2,3,4",
    [int]$LocallyEnhancedFfnKernelSize = 3,
    [bool]$ConcurrentLocalGlobalCoupling = $false,
    [string]$ConcurrentLocalGlobalLayers = "1,2,3,4,5,6,7,8",
    [int]$ConcurrentLocalGlobalDim = 64,
    [int]$ConcurrentLocalGlobalKernelSize = 3,
    [bool]$GatedRelativePositionAttention = $false,
    [string]$GatedRelativePositionAttentionLayers = "1,2,3,4",
    [double]$GatedRelativePositionAttentionMaxMix = 0.25,
    [double]$GatedRelativePositionAttentionLocalityStrength = 1.0,
    [bool]$VisualContrastAttention = $false,
    [string]$VisualContrastAttentionLayers = "1,2,3,4,5,6,7,8",
    [int]$VisualContrastTokens = 64,
    [bool]$CrossCovarianceAttention = $false,
    [string]$CrossCovarianceAttentionLayers = "2,5",
    [double]$CrossCovarianceAttentionResidualScale = 0.10,
    [bool]$DynamicGraphMixer = $false,
    [string]$DynamicGraphMixerLayers = "2,5",
    [int]$DynamicGraphMixerBottleneckDim = 64,
    [int]$DynamicGraphMixerK = 9,
    [bool]$PatchStyleRecalibration = $false,
    [string]$PatchStyleRecalibrationLayers = "2,5",
    [bool]$LayerTokenFusion = $false,
    [string]$LayerTokenFusionLayers = "2,4,6",
    [int]$LayerTokenFusionTopK = 4,
    [double]$LayerTokenFusionBlend = 0.12,
    [double]$LayerTokenFusionAttentionTemperature = 0.20,
    [double]$LayerTokenFusionBboxWeight = 0.20,
    [double]$LayerTokenFusionForegroundWeight = 0.10,
    [bool]$FrequencySelectivePooling = $false,
    [int]$FrequencySelectiveTopK = 1,
    [double]$FrequencySelectiveBlend = 1.0,
    [double]$FrequencySelectiveForegroundThreshold = 0.35,
    [bool]$PatchMemoryAdapter = $false,
    [double]$PatchMemoryAdapterDropout = 0.0,
    [bool]$LateClassAttentionPooling = $false,
    [int]$LateClassAttentionHeads = 4,
    [double]$LateClassAttentionDropout = 0.05,
    [double]$LateClassAttentionMlpRatio = 2.0,
    [double]$LateClassAttentionResidualScale = 0.10,
    [bool]$MixStyle = $false,
    [double]$MixStyleProbability = 0.5,
    [double]$MixStyleAlpha = 0.1,
    [string]$PairwiseMarginPairs = "0-1,1-2,2-3,4-rest",
    [double]$PairwiseMarginLogitScale = 0.25,
    [bool]$PairwiseMarginRouting = $false,
    [double]$PairwiseMarginRouteMaxProbabilityMargin = 0.20,
    [bool]$TopKReassessmentHead = $false,
    [int]$TopKReassessmentTopK = 2,
    [int]$TopKReassessmentHiddenDim = 128,
    [double]$TopKReassessmentDropout = 0.05,
    [double]$TopKReassessmentLogitScale = 0.15,
    [bool]$TopKReassessmentRouting = $true,
    [string]$TopKReassessmentRoutePairs = "0-1,1-2,2-3,4-rest",
    [double]$TopKReassessmentRouteMaxProbabilityMargin = 0.30,
    [double]$TopKReassessmentAuxLossWeight = 0.0,
    [double]$TopKReassessmentAuxRouteMinWeight = 0.05,
    [bool]$FocusClassHead = $false,
    [int]$FocusClassIndex = 1,
    [double]$FocusClassLogitScale = 0.20,
    [double]$FocusClassDropout = 0.05,
    [bool]$FocusClassRouting = $true,
    [double]$FocusClassRouteMaxProbabilityMargin = 0.35,
    [double]$FocusClassRouteMinProbability = 0.08,
    [double]$FocusClassAuxLossWeight = 0.0,
    [double]$FocusClassAuxPositiveWeight = 1.0,
    [bool]$ClassIndependentHead = $false,
    [double]$ClassIndependentDropout = 0.05,
    [double]$ClassIndependentLossWeight = 0.0,
    [double]$ClassIndependentPositiveWeight = 1.0,
    [double]$FocusTverskyLossWeight = 0.0,
    [int]$FocusTverskyClass = 1,
    [double]$FocusTverskyAlpha = 0.70,
    [double]$FocusTverskyBeta = 0.30,
    [double]$FocusTverskyGamma = 1.0,
    [double]$FocusTverskyProbabilityPower = 1.0,
    [int]$FocusTverskyStartEpoch = 1,
    [double]$FocusAucRankLossWeight = 0.0,
    [int]$FocusAucRankClass = 1,
    [string]$FocusAucRankNegativeClasses = "0,2,4",
    [double]$FocusAucRankMargin = 0.04,
    [double]$FocusAucRankTemperature = 0.12,
    [double]$FocusAucRankHardFraction = 0.50,
    [int]$FocusAucRankStartEpoch = 1,
    [double]$FocusPartialAucLossWeight = 0.0,
    [int]$FocusPartialAucClass = 1,
    [string]$FocusPartialAucNegativeClasses = "0,2,4",
    [double]$FocusPartialAucMargin = 0.04,
    [double]$FocusPartialAucTemperature = 0.12,
    [double]$FocusPartialAucNegativeFraction = 0.25,
    [double]$FocusPartialAucPositiveFraction = 0.30,
    [double]$FocusPartialAucPositiveWeight = 0.45,
    [double]$FocusPartialAucMinNegativeProbability = 0.05,
    [int]$FocusPartialAucStartEpoch = 1,
    [double]$BBoxForegroundDropoutLossWeight = 0.0,
    [double]$BBoxForegroundDropoutConsistencyWeight = 0.0,
    [double]$BBoxForegroundDropoutProbability = 0.0,
    [double]$BBoxForegroundDropoutMinAreaRatio = 0.04,
    [double]$BBoxForegroundDropoutMaxAreaRatio = 0.14,
    [ValidateSet("random", "interior", "boundary_band")]
    [string]$BBoxForegroundDropoutMode = "random",
    [ValidateSet("mean", "gray", "zero", "noise")]
    [string]$BBoxForegroundDropoutFill = "mean",
    [double]$BBoxForegroundDropoutTemperature = 1.0,
    [double]$BBoxObjectErasureNegativeLossWeight = 0.0,
    [double]$BBoxObjectErasureProbability = 0.0,
    [double]$BBoxObjectErasureMarginRatio = 0.04,
    [ValidateSet("mean", "gray", "zero", "noise", "blur")]
    [string]$BBoxObjectErasureFill = "mean",
    [int]$BBoxObjectErasureBlurKernel = 15,
    [double]$BBoxObjectErasureTemperature = 1.0,
    [bool]$OrdinalMaturityHead = $false,
    [string]$OrdinalMaturityClasses = "0,1,2,3",
    [double]$OrdinalMaturityLogitScale = 0.20,
    [double]$OrdinalMaturityDropout = 0.05,
    [double]$OrdinalMaturityLossWeight = 0.0,
    [bool]$CumulativeOrdinalHead = $false,
    [string]$CumulativeOrdinalClasses = "0,1,2,3",
    [double]$CumulativeOrdinalLogitScale = 0.25,
    [double]$CumulativeOrdinalDropout = 0.05,
    [double]$CumulativeOrdinalLossWeight = 0.0,
    [string]$CumulativeOrdinalThresholdWeights = "1.4,1.4,1.0",
    [double]$OrdinalDistributionLossWeight = 0.0,
    [string]$OrdinalDistributionClasses = "0,1,2,3",
    [double]$OrdinalDistributionTargetSigma = 0.0,
    [int]$OrdinalDistributionStartEpoch = 1,
    [ValidateSet("cross_entropy", "ce", "ldam_focal", "balanced_softmax", "gce", "ldam_gce", "ldr_kl", "seesaw", "logit_norm", "symmetric_cross_entropy", "sce")]
    [string]$ClassificationLoss = "ldam_focal",
    [double]$BalancedSoftmaxTau = 1.0,
    [double]$GceQ = 0.7,
    [double]$LdrMargin = 2.0,
    [double]$LdrTemperature = 1.0,
    [double]$LogitNormTemperature = 0.04,
    [double]$SymmetricCeAlpha = 0.1,
    [double]$SymmetricCeBeta = 1.0,
    [double]$SymmetricCeEpsilon = 1e-4,
    [double]$SeesawMitigationPower = 0.8,
    [double]$SeesawCompensationPower = 2.0,
    [double]$FocalLossGamma = 1.0,
    [double]$FocalLossMix = 0.10,
    [double]$LabelSmoothing = 0.02,
    [double]$LdamMaxMargin = 0.30,
    [double]$LdamScale = 18.0,
    [string]$ClassLossMultipliers = "",
    [double]$MutualChannelLossWeight = 0.0,
    [int]$MutualChannelTopK = 8,
    [double]$MutualChannelDiversityWeight = 0.20,
    [int]$MutualChannelStartEpoch = 1,
    [double]$ComplementEntropyLossWeight = 0.0,
    [string]$ComplementEntropyClasses = "all",
    [int]$ComplementEntropyStartEpoch = 1,
    [string]$BackgroundSuppressionMode = "desaturate_blur",
    [double]$BackgroundSuppressionProbability = 0.80,
    [double]$BackgroundSuppressionMargin = 0.08,
    [double]$BackgroundSuppressionBlurRadius = 7.0,
    [ValidateSet("none", "unsharp", "rgb_unsharp", "foreground_unsharp", "luma", "luma_residual", "foreground_luma")]
    [string]$SurfaceDetailAmplificationMode = "none",
    [double]$SurfaceDetailAmplificationProbability = 0.0,
    [double]$SurfaceDetailAmplificationStrength = 0.0,
    [double]$SurfaceDetailAmplificationBlurRadius = 1.25,
    [double]$SurfaceDetailAmplificationForegroundWeight = 0.85,
    [bool]$EvalSurfaceDetailAmplification = $false,
    [double]$ForegroundBackgroundMixProbability = 0.0,
    [double]$ForegroundBackgroundMixMargin = 0.08,
    [double]$ForegroundBackgroundMixMinForegroundFraction = 0.06,
    [double]$ForegroundBackgroundMixMaxForegroundFraction = 0.88,
    [double]$ForegroundBackgroundMixSoftness = 5.0,
    [ValidateSet("pseudo", "bbox", "crop_bbox")]
    [string]$ForegroundBackgroundMixMaskSource = "pseudo",
    [double]$LocalExposureProbability = 0.15,
    [double]$LocalExposureStrength = 0.25,
    [double]$ObstacleProbability = 0.04,
    [double]$ObstacleMaxArea = 0.08,
    [int]$RandAugmentNumOps = 0,
    [int]$RandAugmentMagnitude = 0,
    [switch]$PreflightOnly,
    [switch]$DryRun,
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

function Normalize-OptionalPathString {
    param([string]$Value)
    $Text = "$Value".Trim()
    if ($Text.ToLowerInvariant() -in @("none", "off", "false", "null", "na", "n/a")) {
        return ""
    }
    return $Value
}

$HardSampleManifest = Normalize-OptionalPathString $HardSampleManifest
$SampleWeightManifest = Normalize-OptionalPathString $SampleWeightManifest
$QualityGroupManifest = Normalize-OptionalPathString $QualityGroupManifest
$AmbiguousSoftTargetManifest = Normalize-OptionalPathString $AmbiguousSoftTargetManifest
$TargetedMarginManifest = Normalize-OptionalPathString $TargetedMarginManifest
$FocusNeighborBinaryManifest = Normalize-OptionalPathString $FocusNeighborBinaryManifest

if (-not (Test-Path -LiteralPath $DataYaml)) {
    throw "Khong tim thay dataset YAML: $DataYaml"
}
if (-not [string]::IsNullOrWhiteSpace($ClassificationFolderYoloData) -and -not (Test-Path -LiteralPath $ClassificationFolderYoloData)) {
    throw "Khong tim thay classification-folder paired YOLO YAML: $ClassificationFolderYoloData"
}
if (-not [string]::IsNullOrWhiteSpace($AuxiliaryTrainData) -and -not (Test-Path -LiteralPath $AuxiliaryTrainData)) {
    throw "Khong tim thay auxiliary train YAML: $AuxiliaryTrainData"
}
if (-not [string]::IsNullOrWhiteSpace($AuxiliaryTrainClassificationFolderYoloData) -and -not (Test-Path -LiteralPath $AuxiliaryTrainClassificationFolderYoloData)) {
    throw "Khong tim thay auxiliary train paired YOLO YAML: $AuxiliaryTrainClassificationFolderYoloData"
}
if ([string]::IsNullOrWhiteSpace($AuxiliaryTrainData) -and -not [string]::IsNullOrWhiteSpace($AuxiliaryTrainClassificationFolderYoloData)) {
    throw "AuxiliaryTrainClassificationFolderYoloData chi dung khi co AuxiliaryTrainData."
}
if ($AuxiliaryTrainWeight -lt 0.0) {
    throw "AuxiliaryTrainWeight phai >= 0."
}
if ($PairedViewTrain -and [string]::IsNullOrWhiteSpace($AuxiliaryTrainData)) {
    throw "PairedViewTrain can AuxiliaryTrainData."
}
if (($PairedViewSupervisedLossWeight -gt 0.0 -or $PairedViewConsistencyWeight -gt 0.0 -or $PairedViewFeatureConsistencyWeight -gt 0.0 -or $PairedViewFusionLossWeight -gt 0.0 -or $PairedViewFusionConsistencyWeight -gt 0.0) -and -not $PairedViewTrain) {
    throw "PairedView loss can bat PairedViewTrain."
}
if (($PairedViewFusionLossWeight -gt 0.0 -or $PairedViewFusionConsistencyWeight -gt 0.0) -and -not $PairedViewFeatureFusion) {
    throw "PairedView fusion loss can bat PairedViewFeatureFusion."
}
if ($PairedViewFusionHiddenDim -lt 1 -or $PairedViewFusionDropout -lt 0.0 -or $PairedViewFusionLogitScale -lt 0.0 -or $PairedViewFusionLossWeight -lt 0.0 -or $PairedViewFusionConsistencyWeight -lt 0.0) {
    throw "Tham so PairedView fusion khong hop le."
}
if ($PairedViewTemperature -le 0.0) {
    throw "PairedViewTemperature phai > 0."
}
if ($LdrMargin -lt 0.0 -or $LdrTemperature -le 0.0) {
    throw "LdrMargin phai >= 0 va LdrTemperature phai > 0."
}
if ($LogitNormTemperature -le 0.0) {
    throw "LogitNormTemperature phai > 0."
}
if ($SymmetricCeAlpha -lt 0.0 -or $SymmetricCeBeta -lt 0.0) {
    throw "SymmetricCeAlpha/Beta phai >= 0."
}
if (($SymmetricCeAlpha + $SymmetricCeBeta) -le 0.0) {
    throw "SymmetricCeAlpha + SymmetricCeBeta phai > 0."
}
if ($SymmetricCeEpsilon -le 0.0 -or $SymmetricCeEpsilon -gt 1.0) {
    throw "SymmetricCeEpsilon phai nam trong (0, 1]."
}
if ($EarlyTokenMaskKeepRate -le 0.0 -or $EarlyTokenMaskKeepRate -gt 1.0) {
    throw "EarlyTokenMaskKeepRate phai nam trong (0, 1]."
}
if ($MaskedReconstructionLossWeight -lt 0.0) {
    throw "MaskedReconstructionLossWeight phai >= 0."
}
if ($MaskedReconstructionMaskRatio -le 0.0 -or $MaskedReconstructionMaskRatio -ge 0.95) {
    throw "MaskedReconstructionMaskRatio phai nam trong (0, 0.95)."
}
if ($MaskedReconstructionForegroundWeight -lt 0.0 -or $MaskedReconstructionDetailWeight -lt 0.0 -or $MaskedReconstructionBboxWeight -lt 0.0 -or $MaskedReconstructionBboxMarginRatio -lt 0.0) {
    throw "Masked reconstruction weights/margin phai >= 0."
}
if (-not [string]::IsNullOrWhiteSpace($DistillationTeacherCsv) -and -not (Test-Path -LiteralPath $DistillationTeacherCsv)) {
    throw "Khong tim thay distillation teacher CSV: $DistillationTeacherCsv"
}
if ($DistillationWeight -lt 0.0) {
    throw "DistillationWeight phai >= 0."
}
if ($DistillationTemperature -le 0.0) {
    throw "DistillationTemperature phai > 0."
}
if ($DistillationFocusClassIndex -lt 0) {
    throw "DistillationFocusClassIndex phai >= 0."
}
if ($DistillationFocusClassWeight -le 0.0) {
    throw "DistillationFocusClassWeight phai > 0."
}
if ($TeacherGuidedContrastiveStochasticStd -lt 0.0 -or $TeacherGuidedContrastiveMinReliability -lt 0.0 -or $TeacherGuidedContrastiveTeacherConfidencePower -lt 0.0) {
    throw "TeacherGuidedContrastive stochastic/reliability tham so phai >= 0."
}
if ($TeacherGuidedContrastiveMemoryQueueSize -lt 0 -or $TeacherGuidedContrastiveMemoryMinCount -lt 1) {
    throw "TeacherGuidedContrastive memory queue size phai >= 0 va min-count phai >= 1."
}
if ($MultiGranularityAuxDropout -lt 0.0 -or $MultiGranularityAuxLossWeight -lt 0.0) {
    throw "MultiGranularityAux dropout/loss weight phai >= 0."
}
if ($MultiGranularityContrastiveLossWeight -lt 0.0 -or $MultiGranularityContrastiveTeacherMinConfidence -lt 0.0 -or $MultiGranularityContrastiveTeacherMinConfidence -gt 1.0 -or $MultiGranularityContrastiveTeacherConfidencePower -lt 0.0) {
    throw "MultiGranularityContrastive loss/confidence/power tham so khong hop le."
}
if ($MultiGranularityContrastiveTemperature -le 0.0) {
    throw "MultiGranularityContrastiveTemperature phai > 0."
}
if ($MultiGranularityContrastiveLossWeight -gt 0.0 -and -not $MultiGranularityAuxHeads) {
    throw "MultiGranularityContrastiveLossWeight > 0 yeu cau MultiGranularityAuxHeads."
}
if ($MultiGranularityContrastiveLossWeight -gt 0.0 -and ($MultiGranularityContrastiveRequireAgreement -or $MultiGranularityContrastiveTeacherMinConfidence -gt 0.0 -or $MultiGranularityContrastiveWeightMode -ne "filter") -and [string]::IsNullOrWhiteSpace($DistillationTeacherCsv)) {
    throw "Teacher-gated MultiGranularityContrastive yeu cau DistillationTeacherCsv."
}
if ($SelfBoostingAttentionLossWeight -lt 0.0 -or $SelfBoostingAttentionTemperature -le 0.0 -or $SelfBoostingAttentionStartEpoch -lt 1) {
    throw "SelfBoostingAttention tham so khong hop le."
}
$SelfBoostingAttentionHeadEffective = [bool]($SelfBoostingAttentionHead -or ($SelfBoostingAttentionLossWeight -gt 0.0))
if ($BlockLocalPatchMixerDropout -lt 0.0 -or $BlockLocalPatchMixerScale -lt 0.0) {
    throw "BlockLocalPatchMixer tham so khong hop le."
}
if ($LocallyEnhancedFfnKernelSize -lt 3 -or ($LocallyEnhancedFfnKernelSize % 2) -eq 0) {
    throw "LocallyEnhancedFfnKernelSize phai le va >= 3."
}
if ($ConcurrentLocalGlobalDim -le 0) {
    throw "ConcurrentLocalGlobalDim phai > 0."
}
if ($ConcurrentLocalGlobalKernelSize -lt 3 -or ($ConcurrentLocalGlobalKernelSize % 2) -eq 0) {
    throw "ConcurrentLocalGlobalKernelSize phai le va >= 3."
}
if ($GatedRelativePositionAttentionMaxMix -le 0.0 -or $GatedRelativePositionAttentionMaxMix -gt 1.0 -or $GatedRelativePositionAttentionLocalityStrength -le 0.0) {
    throw "GatedRelativePositionAttention max-mix/locality-strength khong hop le."
}
$visualContrastSide = [int][Math]::Sqrt([double]$VisualContrastTokens)
if ($VisualContrastTokens -le 0 -or ($visualContrastSide * $visualContrastSide) -ne $VisualContrastTokens) {
    throw "VisualContrastTokens phai la so chinh phuong duong."
}
if ($VisualContrastAttention -and $TokenPruning) {
    throw "VisualContrastAttention yeu cau TokenPruning=`$false de giu dense patch grid."
}
if ($VisualContrastAttention -and $EarlyTokenMaskKeepRate -lt 1.0) {
    throw "VisualContrastAttention yeu cau EarlyTokenMaskKeepRate=1.0."
}
if ($VisualContrastAttention -and $GatedRelativePositionAttention) {
    throw "VisualContrastAttention khong the dung cung GatedRelativePositionAttention."
}
if ($CrossCovarianceAttentionResidualScale -lt 0.0) {
    throw "CrossCovarianceAttentionResidualScale phai >= 0."
}
if ($CrossCovarianceAttention -and $VisualContrastAttention) {
    throw "CrossCovarianceAttention khong the dung cung VisualContrastAttention."
}
if ($DynamicGraphMixerBottleneckDim -le 0) {
    throw "DynamicGraphMixerBottleneckDim phai > 0."
}
if ($DynamicGraphMixerK -le 0) {
    throw "DynamicGraphMixerK phai > 0."
}
if ($DynamicGraphMixer -and $VisualContrastAttention) {
    throw "DynamicGraphMixer khong the dung cung VisualContrastAttention."
}
if ($DynamicGraphMixer -and $CrossCovarianceAttention) {
    throw "DynamicGraphMixer khong the dung cung CrossCovarianceAttention."
}
if ($PatchStyleRecalibration -and $LocallyEnhancedFfn) {
    throw "PatchStyleRecalibration khong the dung cung LocallyEnhancedFfn."
}
if ($BoundaryCenterTeacherMinConfidence -lt 0.0) {
    throw "BoundaryCenterTeacherMinConfidence phai >= 0."
}
if ($BoundaryCenterLossWeight -gt 0.0 -and ($BoundaryCenterTeacherMinConfidence -gt 0.0 -or $BoundaryCenterRequireAgreement -or $BoundaryCenterTeacherWeightMode -ne "filter") -and [string]::IsNullOrWhiteSpace($DistillationTeacherCsv)) {
    throw "Teacher-gated boundary center yeu cau DistillationTeacherCsv."
}
if ($BorderAttentionSuppressionLossWeight -lt 0.0 -or $BorderAttentionSuppressionFrameWidth -lt 0 -or $BorderAttentionSuppressionBboxBand -lt 0.0 -or $BorderAttentionSuppressionBboxWeight -lt 0.0) {
    throw "BorderAttentionSuppression loss/frame/bbox tham so phai >= 0."
}
if ($BorderAttentionSuppressionTemperature -le 0.0) {
    throw "BorderAttentionSuppressionTemperature phai > 0."
}
if ($BorderAttentionSuppressionStartEpoch -lt 1) {
    throw "BorderAttentionSuppressionStartEpoch phai >= 1."
}
if ($ConfusionSpectralLossWeight -lt 0.0) {
    throw "ConfusionSpectralLossWeight phai >= 0."
}
if ($ConfusionSpectralEmaMomentum -lt 0.0 -or $ConfusionSpectralEmaMomentum -ge 1.0) {
    throw "ConfusionSpectralEmaMomentum phai nam trong [0, 1)."
}
if ($ConfusionSpectralFrequencySmoothing -le 0.0) {
    throw "ConfusionSpectralFrequencySmoothing phai > 0."
}
if ($ConfusionSpectralStartEpoch -lt 1) {
    throw "ConfusionSpectralStartEpoch phai >= 1."
}
if ($ConfusionSpectralLossWeight -gt 0.0 -and $DisableBalancedEpochSampling) {
    throw "CAR/BiCAR yeu cau strict balanced epoch sampling."
}
if ($RegisterAttentionAlignmentLossWeight -lt 0.0 -or $RegisterAttentionAlignmentBboxMarginRatio -lt 0.0 -or $RegisterAttentionAlignmentAgreementWeight -lt 0.0 -or $RegisterAttentionAlignmentForegroundWeight -lt 0.0) {
    throw "RegisterAttentionAlignment tham so phai >= 0."
}
if ($RegisterAttentionAlignmentStartEpoch -lt 1) {
    throw "RegisterAttentionAlignmentStartEpoch phai >= 1."
}
if ($RegisterDiversityLossWeight -lt 0.0) {
    throw "RegisterDiversityLossWeight phai >= 0."
}
if ($SubcenterProxyLossWeight -lt 0.0 -or $SubcenterProxySubcenters -lt 1 -or $SubcenterProxyMargin -lt 0.0 -or $SubcenterProxyDropout -lt 0.0 -or $SubcenterProxyInitStd -lt 0.0) {
    throw "SubcenterProxy loss/subcenters/margin/dropout/init tham so khong hop le."
}
if ($SubcenterProxyScale -le 0.0) {
    throw "SubcenterProxyScale phai > 0."
}
if ($SubcenterProxyStartEpoch -lt 1) {
    throw "SubcenterProxyStartEpoch phai >= 1."
}
if ($DeepAbstentionLossWeight -lt 0.0 -or $DeepAbstentionLossWeight -gt 1.0 -or $DeepAbstentionPenalty -lt 0.0 -or $DeepAbstentionStartEpoch -lt 1 -or $DeepAbstentionDropout -lt 0.0 -or $DeepAbstentionInitialProbability -le 0.0 -or $DeepAbstentionInitialProbability -ge 1.0) {
    throw "DeepAbstention tham so khong hop le."
}
if ($TeacherNonTargetDistillationLossWeight -lt 0.0 -or $TeacherNonTargetDistillationTeacherMinConfidence -lt 0.0) {
    throw "Teacher non-target distillation weights/probabilities phai >= 0."
}
if ($TeacherNonTargetDistillationTemperature -le 0.0) {
    throw "TeacherNonTargetDistillationTemperature phai > 0."
}
if ($TeacherNonTargetDistillationLossWeight -gt 0.0 -and [string]::IsNullOrWhiteSpace($DistillationTeacherCsv)) {
    throw "TeacherNonTargetDistillationLossWeight > 0 yeu cau DistillationTeacherCsv."
}
if ($TeacherFocusMarginLossWeight -lt 0.0 -or $TeacherFocusMarginTeacherMaxProbability -lt 0.0 -or $TeacherFocusMarginMargin -lt 0.0 -or $TeacherFocusMarginMinProbability -lt 0.0 -or $TeacherFocusMarginProbabilityPower -lt 0.0) {
    throw "Teacher focus-margin weights/probabilities phai >= 0."
}
if ($TeacherFocusMarginLossWeight -gt 0.0 -and [string]::IsNullOrWhiteSpace($DistillationTeacherCsv)) {
    throw "TeacherFocusMarginLossWeight > 0 yeu cau DistillationTeacherCsv."
}
if ($TeacherFocusBinaryLossWeight -lt 0.0 -or $TeacherFocusBinaryFocusClass -lt 0 -or $TeacherFocusBinaryTeacherMinConfidence -lt 0.0 -or $TeacherFocusBinaryErrorPower -lt 0.0) {
    throw "Teacher focus-binary weights/probabilities phai >= 0."
}
if ($TeacherFocusBinaryHardTargetBlend -lt 0.0 -or $TeacherFocusBinaryHardTargetBlend -gt 1.0) {
    throw "TeacherFocusBinaryHardTargetBlend phai nam trong [0, 1]."
}
if ($TeacherFocusBinaryLossWeight -gt 0.0 -and [string]::IsNullOrWhiteSpace($DistillationTeacherCsv)) {
    throw "TeacherFocusBinaryLossWeight > 0 yeu cau DistillationTeacherCsv."
}
if ($TeacherPairwiseMarginLossWeight -lt 0.0 -or $TeacherPairwiseMarginTeacherMassThreshold -lt 0.0 -or $TeacherPairwiseMarginErrorPower -lt 0.0) {
    throw "Teacher pairwise-margin weights/probabilities phai >= 0."
}
if ($TeacherPairwiseMarginHardTargetBlend -lt 0.0 -or $TeacherPairwiseMarginHardTargetBlend -gt 1.0) {
    throw "TeacherPairwiseMarginHardTargetBlend phai nam trong [0, 1]."
}
if ($TeacherPairwiseMarginLossWeight -gt 0.0 -and [string]::IsNullOrWhiteSpace($DistillationTeacherCsv)) {
    throw "TeacherPairwiseMarginLossWeight > 0 yeu cau DistillationTeacherCsv."
}
if ($TeacherFeatureRkdLossWeight -lt 0.0 -or $TeacherFeatureRkdDistanceWeight -lt 0.0 -or $TeacherFeatureRkdAngleWeight -lt 0.0) {
    throw "TeacherFeatureRkd weights phai >= 0."
}
if ($TeacherFeatureContrastiveLossWeight -lt 0.0) {
    throw "TeacherFeatureContrastiveLossWeight phai >= 0."
}
if ($TeacherFeatureContrastiveTemperature -le 0.0) {
    throw "TeacherFeatureContrastiveTemperature phai > 0."
}
if ($TeacherFeatureContrastiveProjectionDim -lt 1) {
    throw "TeacherFeatureContrastiveProjectionDim phai >= 1."
}
if ($TeacherFeatureContrastiveAdapterDropout -lt 0.0) {
    throw "TeacherFeatureContrastiveAdapterDropout phai >= 0."
}
if ($TeacherFeatureContrastiveStartEpoch -lt 0) {
    throw "TeacherFeatureContrastiveStartEpoch phai >= 0."
}
if (($TeacherFeatureRkdLossWeight -gt 0.0 -or $TeacherFeatureContrastiveLossWeight -gt 0.0) -and [string]::IsNullOrWhiteSpace($TeacherFeatureNpz)) {
    throw "Teacher feature losses > 0 yeu cau TeacherFeatureNpz."
}
if ($TopKReassessmentTopK -lt 1 -or $TopKReassessmentHiddenDim -lt 16 -or $TopKReassessmentDropout -lt 0.0 -or $TopKReassessmentLogitScale -lt 0.0 -or $TopKReassessmentRouteMaxProbabilityMargin -lt 0.0 -or $TopKReassessmentAuxLossWeight -lt 0.0 -or $TopKReassessmentAuxRouteMinWeight -lt 0.0) {
    throw "TopKReassessment tham so khong hop le."
}
if ($FocusTverskyLossWeight -lt 0.0 -or $FocusTverskyClass -lt 0 -or $FocusTverskyAlpha -lt 0.0 -or $FocusTverskyBeta -lt 0.0 -or $FocusTverskyGamma -le 0.0 -or $FocusTverskyProbabilityPower -le 0.0 -or $FocusTverskyStartEpoch -lt 1) {
    throw "FocusTversky tham so khong hop le."
}
if (($FocusTverskyAlpha + $FocusTverskyBeta) -le 0.0) {
    throw "FocusTverskyAlpha + FocusTverskyBeta phai > 0."
}
if ($FocusAucRankLossWeight -lt 0.0 -or $FocusAucRankClass -lt 0 -or $FocusAucRankMargin -lt 0.0 -or $FocusAucRankTemperature -le 0.0 -or $FocusAucRankHardFraction -le 0.0 -or $FocusAucRankHardFraction -gt 1.0 -or $FocusAucRankStartEpoch -lt 1) {
    throw "FocusAucRank tham so khong hop le."
}
if ($FocusPartialAucLossWeight -lt 0.0 -or $FocusPartialAucClass -lt 0 -or $FocusPartialAucMargin -lt 0.0 -or $FocusPartialAucTemperature -le 0.0 -or $FocusPartialAucNegativeFraction -le 0.0 -or $FocusPartialAucNegativeFraction -gt 1.0 -or $FocusPartialAucPositiveFraction -le 0.0 -or $FocusPartialAucPositiveFraction -gt 1.0 -or $FocusPartialAucPositiveWeight -lt 0.0 -or $FocusPartialAucMinNegativeProbability -lt 0.0 -or $FocusPartialAucStartEpoch -lt 1) {
    throw "FocusPartialAuc tham so khong hop le."
}
if ($ClassIndependentDropout -lt 0.0 -or $ClassIndependentLossWeight -lt 0.0 -or $ClassIndependentPositiveWeight -le 0.0) {
    throw "ClassIndependent tham so khong hop le."
}
if ($ClassIndependentLossWeight -gt 0.0 -and -not $ClassIndependentHead) {
    throw "ClassIndependentLossWeight > 0 yeu cau ClassIndependentHead."
}
if ($PatchObjectnessHiddenDim -lt 1 -or $PatchObjectnessDropout -lt 0.0 -or $PatchObjectnessLogitScale -lt 0.0 -or $PatchObjectnessTemperature -le 0.0 -or $PatchObjectnessLossWeight -lt 0.0 -or $PatchObjectnessPositiveWeight -le 0.0) {
    throw "PatchObjectness tham so khong hop le."
}
if ($PatchObjectnessLossWeight -gt 0.0 -and -not $PatchObjectnessGuidedHead) {
    throw "PatchObjectnessLossWeight > 0 yeu cau PatchObjectnessGuidedHead."
}
if ($BboxTokenLabelLossWeight -lt 0.0 -or $BboxTokenLabelMinPrior -lt 0.0 -or $BboxTokenLabelMinPrior -gt 1.0 -or $BboxTokenLabelPriorPower -lt 0.0 -or $BboxTokenLabelFocusClass -lt 0 -or $BboxTokenLabelFocusWeight -le 0.0 -or $BboxTokenLabelStartEpoch -lt 1) {
    throw "BboxTokenLabel tham so khong hop le."
}
$validPatchEvidenceRouterMarginPriorModes = @("none", "weighted", "max", "topk_mean", "weighted_minus_mean")
if ($validPatchEvidenceRouterMarginPriorModes -notcontains $PatchEvidenceRouterMarginPriorMode) {
    throw "PatchEvidenceRouterMarginPriorMode khong hop le: $PatchEvidenceRouterMarginPriorMode."
}
if ($PatchEvidenceRouterHiddenDim -lt 16 -or $PatchEvidenceRouterTopK -lt 1 -or $PatchEvidenceRouterBboxWeight -lt 0.0 -or $PatchEvidenceRouterDropout -lt 0.0 -or $PatchEvidenceRouterLogitScale -lt 0.0 -or $PatchEvidenceRouterMarginPriorScale -lt 0.0 -or $PatchEvidenceRouterRouteMaxProbabilityMargin -lt 0.0 -or $PatchEvidenceRouterRouteMinPairProbability -lt 0.0 -or $PatchEvidenceRouterRouteMinPairProbability -gt 1.0 -or $PatchEvidenceRouterLossWeight -lt 0.0 -or $PatchEvidenceRouterPositiveWeight -le 0.0 -or $PatchEvidenceRouterStartEpoch -lt 1) {
    throw "PatchEvidenceRouter tham so khong hop le."
}
if ($PatchEvidenceRouterLossWeight -gt 0.0 -and -not $PatchEvidenceRouterHead) {
    throw "PatchEvidenceRouterLossWeight > 0 yeu cau PatchEvidenceRouterHead."
}
if ($PatchEvidenceRouterTeacherLossWeight -lt 0.0 -or $PatchEvidenceRouterTeacherMinConfidence -lt 0.0 -or $PatchEvidenceRouterTeacherMinConfidence -gt 1.0 -or $PatchEvidenceRouterTeacherMinPairMass -lt 0.0 -or $PatchEvidenceRouterTeacherMinPairMass -gt 1.0 -or $PatchEvidenceRouterTeacherPositiveWeight -le 0.0) {
    throw "PatchEvidenceRouterTeacher tham so khong hop le."
}
if ($PatchEvidenceRouterTeacherLossWeight -gt 0.0 -and -not $PatchEvidenceRouterHead) {
    throw "PatchEvidenceRouterTeacherLossWeight > 0 yeu cau PatchEvidenceRouterHead."
}
if ($PatchEvidenceRouterTeacherLossWeight -gt 0.0 -and [string]::IsNullOrWhiteSpace($PatchEvidenceRouterTeacherCsv)) {
    throw "PatchEvidenceRouterTeacherLossWeight > 0 yeu cau PatchEvidenceRouterTeacherCsv."
}
if (-not [string]::IsNullOrWhiteSpace($PatchEvidenceRouterTeacherCsv) -and -not (Test-Path -LiteralPath $PatchEvidenceRouterTeacherCsv)) {
    throw "Khong tim thay PatchEvidenceRouterTeacherCsv: $PatchEvidenceRouterTeacherCsv"
}
if (-not [string]::IsNullOrWhiteSpace($PatchEvidenceLinearVerifierJson) -and -not (Test-Path -LiteralPath $PatchEvidenceLinearVerifierJson)) {
    throw "Khong tim thay PatchEvidenceLinearVerifierJson: $PatchEvidenceLinearVerifierJson"
}
if ($PatchEvidenceLinearVerifierMinPairProbability -lt 0.0 -or $PatchEvidenceLinearVerifierMinPairProbability -gt 1.0 -or $PatchEvidenceLinearVerifierMaxPairMargin -lt 0.0 -or $PatchEvidenceLinearVerifierConfidenceThreshold -lt 0.0 -or $PatchEvidenceLinearVerifierConfidenceThreshold -gt 1.0 -or $PatchEvidenceLinearVerifierLogitBoost -lt 0.0 -or $PatchEvidenceLinearVerifierProtectRightMinProbability -lt 0.0 -or $PatchEvidenceLinearVerifierProtectRightMinProbability -gt 1.0 -or $PatchEvidenceLinearVerifierTrainingSoftLogitScale -lt 0.0 -or $PatchEvidenceLinearVerifierTrainingSoftGateTemperature -le 0.0) {
    throw "PatchEvidenceLinearVerifier tham so khong hop le."
}
if ($BBoxPriorPatchContextHiddenDim -lt 1 -or $BBoxPriorPatchContextDropout -lt 0.0 -or $BBoxPriorPatchContextLogitScale -lt 0.0 -or $BBoxPriorPatchContextTemperature -le 0.0) {
    throw "BBoxPriorPatchContext tham so khong hop le."
}
if ($TopKReassessmentAuxLossWeight -gt 0.0 -and -not $TopKReassessmentHead) {
    throw "TopKReassessmentAuxLossWeight > 0 yeu cau TopKReassessmentHead."
}
if ($BackgroundCounterfactualConsistencyWeight -lt 0.0) {
    throw "BackgroundCounterfactualConsistencyWeight phai >= 0."
}
if ($BackgroundCounterfactualProbability -lt 0.0 -or $BackgroundCounterfactualProbability -gt 1.0) {
    throw "BackgroundCounterfactualProbability phai nam trong [0, 1]."
}
if ($BackgroundCounterfactualMargin -lt 0.0) {
    throw "BackgroundCounterfactualMargin phai >= 0."
}
if ($BackgroundCounterfactualBlurKernel -lt 1) {
    throw "BackgroundCounterfactualBlurKernel phai >= 1."
}
if ($BackgroundCounterfactualTemperature -le 0.0) {
    throw "BackgroundCounterfactualTemperature phai > 0."
}
if ($BackgroundFocusSuppressionLossWeight -lt 0.0) {
    throw "BackgroundFocusSuppressionLossWeight phai >= 0."
}
if ($BackgroundFocusSuppressionProbability -lt 0.0 -or $BackgroundFocusSuppressionProbability -gt 1.0) {
    throw "BackgroundFocusSuppressionProbability phai nam trong [0, 1]."
}
if ($BackgroundFocusSuppressionFocusClass -lt 0) {
    throw "BackgroundFocusSuppressionFocusClass phai >= 0."
}
if ($BackgroundFocusSuppressionMargin -lt 0.0) {
    throw "BackgroundFocusSuppressionMargin phai >= 0."
}
if ($BackgroundFocusSuppressionMinProbability -lt 0.0) {
    throw "BackgroundFocusSuppressionMinProbability phai >= 0."
}
if ($BackgroundFocusSuppressionProbabilityPower -lt 0.0) {
    throw "BackgroundFocusSuppressionProbabilityPower phai >= 0."
}
if ($SourceContextFocusSuppressionLossWeight -lt 0.0) {
    throw "SourceContextFocusSuppressionLossWeight phai >= 0."
}
if ($SourceContextFocusSuppressionProbability -lt 0.0 -or $SourceContextFocusSuppressionProbability -gt 1.0) {
    throw "SourceContextFocusSuppressionProbability phai nam trong [0, 1]."
}
if ($SourceContextFocusSuppressionFocusClass -lt 0) {
    throw "SourceContextFocusSuppressionFocusClass phai >= 0."
}
if ($SourceContextFocusSuppressionMargin -lt 0.0) {
    throw "SourceContextFocusSuppressionMargin phai >= 0."
}
if ($SourceContextFocusSuppressionMinProbability -lt 0.0) {
    throw "SourceContextFocusSuppressionMinProbability phai >= 0."
}
if ($SourceContextFocusSuppressionProbabilityPower -lt 0.0) {
    throw "SourceContextFocusSuppressionProbabilityPower phai >= 0."
}
if ($SourceContextFocusSuppressionLossWeight -gt 0.0 -and -not $ClassificationSourceContextAux) {
    throw "SourceContextFocusSuppressionLossWeight > 0 can ClassificationSourceContextAux."
}
if (-not [string]::IsNullOrWhiteSpace($ResumeCheckpoint) -and -not (Test-Path -LiteralPath $ResumeCheckpoint)) {
    throw "Khong tim thay resume checkpoint: $ResumeCheckpoint"
}
if (-not [string]::IsNullOrWhiteSpace($SampleWeightManifest) -and -not (Test-Path -LiteralPath $SampleWeightManifest)) {
    throw "Khong tim thay sample weight manifest: $SampleWeightManifest"
}
if (-not [string]::IsNullOrWhiteSpace($QualityGroupManifest) -and -not (Test-Path -LiteralPath $QualityGroupManifest)) {
    throw "Khong tim thay quality-group manifest: $QualityGroupManifest"
}
if (-not [string]::IsNullOrWhiteSpace($HardSampleManifest) -and -not (Test-Path -LiteralPath $HardSampleManifest)) {
    throw "Khong tim thay hard sample manifest: $HardSampleManifest"
}
if (-not [string]::IsNullOrWhiteSpace($AmbiguousSoftTargetManifest) -and -not (Test-Path -LiteralPath $AmbiguousSoftTargetManifest)) {
    throw "Khong tim thay ambiguous soft-target manifest: $AmbiguousSoftTargetManifest"
}
if (-not [string]::IsNullOrWhiteSpace($TargetedMarginManifest) -and -not (Test-Path -LiteralPath $TargetedMarginManifest)) {
    throw "Khong tim thay targeted-margin manifest: $TargetedMarginManifest"
}
if (-not [string]::IsNullOrWhiteSpace($FocusNeighborBinaryManifest) -and -not (Test-Path -LiteralPath $FocusNeighborBinaryManifest)) {
    throw "Khong tim thay focus-neighbor binary manifest: $FocusNeighborBinaryManifest"
}
if ($FocusNeighborBinaryLossWeight -lt 0.0 -or $FocusNeighborBinaryFocusClass -lt 0 -or $FocusNeighborBinaryDefaultWeight -lt 0.0 -or $FocusNeighborBinaryMaxWeight -le 0.0) {
    throw "FocusNeighborBinary tham so khong hop le."
}
if ($FocusNeighborBinaryMaxWeight -lt $FocusNeighborBinaryDefaultWeight) {
    throw "FocusNeighborBinaryMaxWeight phai >= FocusNeighborBinaryDefaultWeight."
}
if ($RDropLossWeight -lt 0.0) {
    throw "RDropLossWeight phai >= 0."
}
if ($RDropTemperature -le 0.0) {
    throw "RDropTemperature phai > 0."
}
if ($AugmixConsistencyLossWeight -lt 0.0) {
    throw "AugmixConsistencyLossWeight phai >= 0."
}
if ($AugmixConsistencyProbability -lt 0.0 -or $AugmixConsistencyProbability -gt 1.0) {
    throw "AugmixConsistencyProbability phai nam trong [0, 1]."
}
if ($AugmixConsistencySeverity -lt 0.0) {
    throw "AugmixConsistencySeverity phai >= 0."
}
if ($AugmixConsistencyWidth -le 0) {
    throw "AugmixConsistencyWidth phai > 0."
}
if ($AugmixConsistencyDepth -le 0) {
    throw "AugmixConsistencyDepth phai > 0."
}
if ($AugmixConsistencyAlpha -le 0.0) {
    throw "AugmixConsistencyAlpha phai > 0."
}
if ($AugmixConsistencyTemperature -le 0.0) {
    throw "AugmixConsistencyTemperature phai > 0."
}
if ($IlluminationConsistencyLossWeight -lt 0.0) {
    throw "IlluminationConsistencyLossWeight phai >= 0."
}
if ($IlluminationConsistencyProbability -lt 0.0 -or $IlluminationConsistencyProbability -gt 1.0) {
    throw "IlluminationConsistencyProbability phai nam trong [0, 1]."
}
if ($IlluminationConsistencyBrightness -lt 0.0) {
    throw "IlluminationConsistencyBrightness phai >= 0."
}
if ($IlluminationConsistencyContrast -lt 0.0) {
    throw "IlluminationConsistencyContrast phai >= 0."
}
if ($IlluminationConsistencyGamma -lt 0.0) {
    throw "IlluminationConsistencyGamma phai >= 0."
}
if ($IlluminationConsistencyTemperature -le 0.0) {
    throw "IlluminationConsistencyTemperature phai > 0."
}
if ($ForegroundChromaConsistencyLossWeight -lt 0.0) {
    throw "ForegroundChromaConsistencyLossWeight phai >= 0."
}
if ($ForegroundChromaConsistencyProbability -lt 0.0 -or $ForegroundChromaConsistencyProbability -gt 1.0) {
    throw "ForegroundChromaConsistencyProbability phai nam trong [0, 1]."
}
if ($ForegroundChromaConsistencySaturationDelta -lt 0.0) {
    throw "ForegroundChromaConsistencySaturationDelta phai >= 0."
}
if ($ForegroundChromaConsistencyHueDelta -lt 0.0 -or $ForegroundChromaConsistencyHueDelta -gt 0.5) {
    throw "ForegroundChromaConsistencyHueDelta phai nam trong [0, 0.5]."
}
if ($ForegroundChromaConsistencyBboxMarginRatio -lt 0.0) {
    throw "ForegroundChromaConsistencyBboxMarginRatio phai >= 0."
}
if ($ForegroundChromaConsistencyTemperature -le 0.0) {
    throw "ForegroundChromaConsistencyTemperature phai > 0."
}
if ($FriendlyAdversarialLossWeight -lt 0.0) {
    throw "FriendlyAdversarialLossWeight phai >= 0."
}
if ($FriendlyAdversarialEpsilon -le 0.0) {
    throw "FriendlyAdversarialEpsilon phai > 0."
}
if ($FriendlyAdversarialStepSize -le 0.0 -or $FriendlyAdversarialStepSize -gt $FriendlyAdversarialEpsilon) {
    throw "FriendlyAdversarialStepSize phai nam trong (0, epsilon]."
}
if ($FriendlyAdversarialSteps -lt 1 -or $FriendlyAdversarialSteps -gt 16) {
    throw "FriendlyAdversarialSteps phai nam trong [1, 16]."
}
if ($FriendlyAdversarialBboxErodeRatio -lt 0.0 -or $FriendlyAdversarialBboxErodeRatio -ge 0.5) {
    throw "FriendlyAdversarialBboxErodeRatio phai nam trong [0, 0.5)."
}
if ($FriendlyAdversarialFocusClass -lt 0 -or $FriendlyAdversarialFocusClass -ge 5) {
    throw "FriendlyAdversarialFocusClass phai nam trong [0, 5)."
}
if ($FriendlyAdversarialMaxPerDirection -le 0) {
    throw "FriendlyAdversarialMaxPerDirection phai > 0."
}
if ($FriendlyAdversarialStartEpoch -lt 1) {
    throw "FriendlyAdversarialStartEpoch phai >= 1."
}
if ($SemanticAttributeLossWeight -lt 0.0) {
    throw "SemanticAttributeLossWeight phai >= 0."
}
if ($SemanticAttributeLossWeight -gt 0.0 -and [string]::IsNullOrWhiteSpace($SemanticAttributeSpecs)) {
    throw "SemanticAttributeSpecs khong duoc rong khi bat SemanticAttributeLossWeight."
}
if ($ConfusionPairMixupLossWeight -lt 0.0) {
    throw "ConfusionPairMixupLossWeight phai >= 0."
}
if ($ConfusionPairMixupAlpha -le 0.0) {
    throw "ConfusionPairMixupAlpha phai > 0."
}
if ($ConfusionPairMixupMaxPairs -lt 0) {
    throw "ConfusionPairMixupMaxPairs phai >= 0."
}
if ($ConfusionPairMixupStartEpoch -lt 0) {
    throw "ConfusionPairMixupStartEpoch phai >= 0."
}
if ($ForegroundSnapmixLossWeight -lt 0.0) {
    throw "ForegroundSnapmixLossWeight phai >= 0."
}
if ($ForegroundSnapmixProbability -lt 0.0 -or $ForegroundSnapmixProbability -gt 1.0) {
    throw "ForegroundSnapmixProbability phai nam trong [0, 1]."
}
if ($ForegroundSnapmixAlpha -le 0.0) {
    throw "ForegroundSnapmixAlpha phai > 0."
}
if ($ForegroundSnapmixMinAreaRatio -le 0.0) {
    throw "ForegroundSnapmixMinAreaRatio phai > 0."
}
if ($ForegroundSnapmixMaxAreaRatio -lt $ForegroundSnapmixMinAreaRatio) {
    throw "ForegroundSnapmixMaxAreaRatio phai >= ForegroundSnapmixMinAreaRatio."
}
if ($ForegroundSnapmixBboxMarginRatio -lt 0.0) {
    throw "ForegroundSnapmixBboxMarginRatio phai >= 0."
}
if ($ForegroundCounterexampleMixLossWeight -lt 0.0) {
    throw "ForegroundCounterexampleMixLossWeight phai >= 0."
}
if ($ForegroundCounterexampleMixProbability -lt 0.0 -or $ForegroundCounterexampleMixProbability -gt 1.0) {
    throw "ForegroundCounterexampleMixProbability phai nam trong [0, 1]."
}
if ($ForegroundCounterexampleMixSourceClass -lt 0) {
    throw "ForegroundCounterexampleMixSourceClass phai >= 0."
}
if ($ForegroundCounterexampleMixAlpha -le 0.0) {
    throw "ForegroundCounterexampleMixAlpha phai > 0."
}
if ($ForegroundCounterexampleMixMinAreaRatio -le 0.0) {
    throw "ForegroundCounterexampleMixMinAreaRatio phai > 0."
}
if ($ForegroundCounterexampleMixMaxAreaRatio -lt $ForegroundCounterexampleMixMinAreaRatio) {
    throw "ForegroundCounterexampleMixMaxAreaRatio phai >= ForegroundCounterexampleMixMinAreaRatio."
}
if ($ForegroundCounterexampleMixBboxMarginRatio -lt 0.0) {
    throw "ForegroundCounterexampleMixBboxMarginRatio phai >= 0."
}
if ($ForegroundCounterexampleMixStartEpoch -lt 0) {
    throw "ForegroundCounterexampleMixStartEpoch phai >= 0."
}
if ($DclRegionShuffleLossWeight -lt 0.0) {
    throw "DclRegionShuffleLossWeight phai >= 0."
}
if ($DclRegionShuffleProbability -lt 0.0 -or $DclRegionShuffleProbability -gt 1.0) {
    throw "DclRegionShuffleProbability phai nam trong [0, 1]."
}
if ($DclRegionShuffleGridSize -lt 2) {
    throw "DclRegionShuffleGridSize phai >= 2."
}
if ($DclRegionShuffleBboxMarginRatio -lt 0.0) {
    throw "DclRegionShuffleBboxMarginRatio phai >= 0."
}
if ($DclRegionShuffleStartEpoch -lt 0) {
    throw "DclRegionShuffleStartEpoch phai >= 0."
}
if ($QuantizedLabelCpuLossWeight -lt 0.0) {
    throw "QuantizedLabelCpuLossWeight phai >= 0."
}
if ($QuantizedLabelCpuMinPrior -lt 0.0 -or $QuantizedLabelCpuMinPrior -gt 1.0) {
    throw "QuantizedLabelCpuMinPrior phai nam trong [0, 1]."
}
if ($QuantizedLabelCpuMaxPrior -lt 0.0 -or $QuantizedLabelCpuMaxPrior -gt 1.0) {
    throw "QuantizedLabelCpuMaxPrior phai nam trong [0, 1]."
}
if ($QuantizedLabelCpuMaxPrior -lt $QuantizedLabelCpuMinPrior) {
    throw "QuantizedLabelCpuMaxPrior phai >= QuantizedLabelCpuMinPrior."
}
if ($QuantizedLabelCpuNegativeWeight -lt 0.0) {
    throw "QuantizedLabelCpuNegativeWeight phai >= 0."
}
if ($QuantizedLabelCpuNonNegativeBeta -lt 0.0) {
    throw "QuantizedLabelCpuNonNegativeBeta phai >= 0."
}
if ($QuantizedLabelCpuStartEpoch -lt 0) {
    throw "QuantizedLabelCpuStartEpoch phai >= 0."
}
if ($SelfPacedLossWeight -lt 0.0 -or $SelfPacedLossWeight -gt 1.0) {
    throw "SelfPacedLossWeight phai nam trong [0, 1]."
}
if ($SelfPacedLossPercentile -lt 0.0 -or $SelfPacedLossPercentile -gt 1.0) {
    throw "SelfPacedLossPercentile phai nam trong [0, 1]."
}
if ($SelfPacedLossGamma -lt 0.0) {
    throw "SelfPacedLossGamma phai >= 0."
}
if ($SelfPacedLossMinWeight -lt 0.0 -or $SelfPacedLossMinWeight -gt 1.0) {
    throw "SelfPacedLossMinWeight phai nam trong [0, 1]."
}
if ($SelfPacedLossStartEpoch -lt 0) {
    throw "SelfPacedLossStartEpoch phai >= 0."
}
if ($BranchCnnTokens -lt 0) {
    throw "BranchCnnTokens phai >= 0."
}
if ($LayerTokenFusionTopK -lt 1) {
    throw "LayerTokenFusionTopK phai >= 1."
}
if ($LayerTokenFusionBlend -lt 0.0 -or $LayerTokenFusionBlend -gt 1.0) {
    throw "LayerTokenFusionBlend phai nam trong [0, 1]."
}
if ($LayerTokenFusionAttentionTemperature -le 0.0) {
    throw "LayerTokenFusionAttentionTemperature phai > 0."
}
if ($LayerTokenFusionBboxWeight -lt 0.0) {
    throw "LayerTokenFusionBboxWeight phai >= 0."
}
if ($LayerTokenFusionForegroundWeight -lt 0.0) {
    throw "LayerTokenFusionForegroundWeight phai >= 0."
}
if ($LateClassAttentionHeads -lt 1) {
    throw "LateClassAttentionHeads phai >= 1."
}
if ($LateClassAttentionDropout -lt 0.0) {
    throw "LateClassAttentionDropout phai >= 0."
}
if ($LateClassAttentionMlpRatio -le 0.0) {
    throw "LateClassAttentionMlpRatio phai > 0."
}
if ($LateClassAttentionResidualScale -lt 0.0) {
    throw "LateClassAttentionResidualScale phai >= 0."
}

function Get-DataYamlScalar {
    param(
        [string]$YamlPath,
        [string]$Key,
        [string]$DefaultValue
    )
    $pattern = "^\s*" + [regex]::Escape($Key) + "\s*:\s*(.+?)\s*$"
    $match = Select-String -LiteralPath $YamlPath -Pattern $pattern | Select-Object -First 1
    if ($null -eq $match) {
        return $DefaultValue
    }
    $value = $match.Matches[0].Groups[1].Value.Trim()
    if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    return $value
}

function Resolve-DataYamlPath {
    param(
        [string]$BasePath,
        [string]$Value
    )
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $Value))
}

$DataYamlResolved = (Resolve-Path -LiteralPath $DataYaml).Path
$DataYamlDir = Split-Path -Parent $DataYamlResolved
$DatasetRootValue = Get-DataYamlScalar -YamlPath $DataYamlResolved -Key "path" -DefaultValue "."
$DatasetRoot = Resolve-DataYamlPath -BasePath $DataYamlDir -Value $DatasetRootValue
$SplitCounts = @()
foreach ($SplitEntry in @(
    @{ Name = "test"; Value = Get-DataYamlScalar -YamlPath $DataYamlResolved -Key "test" -DefaultValue "" },
    @{ Name = "train"; Value = Get-DataYamlScalar -YamlPath $DataYamlResolved -Key "train" -DefaultValue "train" },
    @{ Name = "val"; Value = Get-DataYamlScalar -YamlPath $DataYamlResolved -Key "val" -DefaultValue "val" }
)) {
    $SplitRoot = Resolve-DataYamlPath -BasePath $DatasetRoot -Value $SplitEntry.Value
    if ($null -eq $SplitRoot -or -not (Test-Path -LiteralPath $SplitRoot)) {
        continue
    }
    foreach ($ClassDir in Get-ChildItem -LiteralPath $SplitRoot -Directory | Sort-Object Name) {
        $ImageCount = (
            Get-ChildItem -LiteralPath $ClassDir.FullName -Recurse -File |
            Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp|webp)$' } |
            Measure-Object
        ).Count
        $SplitCounts += [ordered]@{
            split_class = $SplitEntry.Name + "/" + $ClassDir.Name
            count = $ImageCount
        }
    }
}
$GpuStatus = & nvidia-smi `
    --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,pstate `
    --format=csv,noheader

if ($SchedulerTotalEpochs -lt 0) {
    throw "SchedulerTotalEpochs must be >= 0."
}
$SchedulerTotalEpochsEffective = if ($SchedulerTotalEpochs -gt 0) {
    $SchedulerTotalEpochs
} else {
    $Epochs
}

if ($DryRun) {
    $PreflightOnly = $true
}

if ($PreflightOnly) {
    [ordered]@{
        status = "ok"
        dry_run = [bool]$DryRun
        preflight_only = [bool]$PreflightOnly
        python = $Python
        data_yaml = $DataYaml
        classification_folder_yolo_data = $ClassificationFolderYoloData
        auxiliary_train_data = $AuxiliaryTrainData
        auxiliary_train_classification_folder_yolo_data = $AuxiliaryTrainClassificationFolderYoloData
        auxiliary_train_weight = $AuxiliaryTrainWeight
        paired_view_train = $PairedViewTrain
        resume_checkpoint = $ResumeCheckpoint
        split_counts = $SplitCounts
        gpu = $GpuStatus
        batch_size = $BatchSize
        grad_accum_steps = $GradAccumSteps
        effective_batch_size = $BatchSize * $GradAccumSteps
        scheduler_total_epochs = $SchedulerTotalEpochsEffective
        image_size = $ImageSize
        stem_architecture = $StemArchitecture
        stem_pooling_mode = $StemPoolingMode
        stem_softpool_blend = $StemSoftpoolBlend
        shifted_patch_tokenization = [bool]$ShiftedPatchTokenization
        shifted_patch_shift = $ShiftedPatchShift
        shifted_patch_residual_scale = $ShiftedPatchResidualScale
        crop_margin_ratio = $CropMarginRatio
        class_crop_margin_scale_threshold = $ClassCropMarginScaleThreshold
        class_crop_margin_max_ratio = $ClassCropMarginMaxRatio
        class_conditional_augmentation_scales = $ClassConditionalAugmentationScales
        classification_source_context = [bool]$ClassificationSourceContext
        classification_source_context_mode = $ClassificationSourceContextMode
        classification_source_context_layout = $ClassificationSourceContextLayout
        classification_source_context_margin_ratio = $ClassificationSourceContextMarginRatio
        classification_source_context_background_alpha = $ClassificationSourceContextBackgroundAlpha
        classification_source_context_blur_radius = $ClassificationSourceContextBlurRadius
        classification_source_context_inset_scale = $ClassificationSourceContextInsetScale
        classification_source_context_aux = [bool]$ClassificationSourceContextAux
        source_context_aux_loss_weight = $SourceContextAuxLossWeight
        source_context_aux_classification_weight = $SourceContextAuxClassificationWeight
        source_context_aux_consistency_weight = $SourceContextAuxConsistencyWeight
        source_context_aux_bbox_margin_ratio = $SourceContextAuxBboxMarginRatio
        source_context_aux_attention_temperature = $SourceContextAuxAttentionTemperature
        token_pruning = [bool]$TokenPruning
        token_prune_bbox_weight = $TokenPruneBboxWeight
        token_prune_bbox_margin_ratio = $TokenPruneBboxMarginRatio
        bbox_token_prior_source = $BboxTokenPriorSource
        early_token_mask_keep_rate = $EarlyTokenMaskKeepRate
        bbox_spatial_fusion = [bool]$BBoxSpatialFusion
        bbox_spatial_fusion_hidden_dim = $BBoxSpatialFusionHiddenDim
        bbox_spatial_fusion_dropout = $BBoxSpatialFusionDropout
        bbox_spatial_fusion_logit_scale = $BBoxSpatialFusionLogitScale
        patch_objectness_guided_head = [bool]$PatchObjectnessGuidedHead
        patch_objectness_hidden_dim = $PatchObjectnessHiddenDim
        patch_objectness_dropout = $PatchObjectnessDropout
        patch_objectness_logit_scale = $PatchObjectnessLogitScale
        patch_objectness_temperature = $PatchObjectnessTemperature
        patch_objectness_loss_weight = $PatchObjectnessLossWeight
        patch_objectness_positive_weight = $PatchObjectnessPositiveWeight
        bbox_token_label_loss_weight = $BboxTokenLabelLossWeight
        bbox_token_label_min_prior = $BboxTokenLabelMinPrior
        bbox_token_label_prior_power = $BboxTokenLabelPriorPower
        bbox_token_label_classes = $BboxTokenLabelClasses
        bbox_token_label_focus_class = $BboxTokenLabelFocusClass
        bbox_token_label_focus_weight = $BboxTokenLabelFocusWeight
        bbox_token_label_detach_classifier = [bool]$BboxTokenLabelDetachClassifier
        bbox_token_label_start_epoch = $BboxTokenLabelStartEpoch
        patch_evidence_router_head = [bool]$PatchEvidenceRouterHead
        patch_evidence_router_pair = $PatchEvidenceRouterPair
        patch_evidence_router_hidden_dim = $PatchEvidenceRouterHiddenDim
        patch_evidence_router_top_k = $PatchEvidenceRouterTopK
        patch_evidence_router_bbox_weight = $PatchEvidenceRouterBboxWeight
        patch_evidence_router_dropout = $PatchEvidenceRouterDropout
        patch_evidence_router_logit_scale = $PatchEvidenceRouterLogitScale
        patch_evidence_router_margin_prior_mode = $PatchEvidenceRouterMarginPriorMode
        patch_evidence_router_margin_prior_scale = $PatchEvidenceRouterMarginPriorScale
        patch_evidence_router_summary_stats = [bool]$PatchEvidenceRouterSummaryStats
        patch_evidence_router_routing = [bool]$PatchEvidenceRouterRouting
        patch_evidence_router_route_max_probability_margin = $PatchEvidenceRouterRouteMaxProbabilityMargin
        patch_evidence_router_route_min_pair_probability = $PatchEvidenceRouterRouteMinPairProbability
        patch_evidence_router_loss_weight = $PatchEvidenceRouterLossWeight
        patch_evidence_router_positive_weight = $PatchEvidenceRouterPositiveWeight
        patch_evidence_router_start_epoch = $PatchEvidenceRouterStartEpoch
        patch_evidence_router_teacher_csv = $PatchEvidenceRouterTeacherCsv
        patch_evidence_router_teacher_loss_weight = $PatchEvidenceRouterTeacherLossWeight
        patch_evidence_router_teacher_min_confidence = $PatchEvidenceRouterTeacherMinConfidence
        patch_evidence_router_teacher_min_pair_mass = $PatchEvidenceRouterTeacherMinPairMass
        patch_evidence_router_teacher_positive_weight = $PatchEvidenceRouterTeacherPositiveWeight
        patch_evidence_linear_verifier_json = $PatchEvidenceLinearVerifierJson
        patch_evidence_linear_verifier_pair = $PatchEvidenceLinearVerifierPair
        patch_evidence_linear_verifier_min_pair_probability = $PatchEvidenceLinearVerifierMinPairProbability
        patch_evidence_linear_verifier_max_pair_margin = $PatchEvidenceLinearVerifierMaxPairMargin
        patch_evidence_linear_verifier_confidence_threshold = $PatchEvidenceLinearVerifierConfidenceThreshold
        patch_evidence_linear_verifier_logit_boost = $PatchEvidenceLinearVerifierLogitBoost
        patch_evidence_linear_verifier_protect_right_min_probability = $PatchEvidenceLinearVerifierProtectRightMinProbability
        patch_evidence_linear_verifier_training_soft_adjustment = [bool]$PatchEvidenceLinearVerifierTrainingSoftAdjustment
        patch_evidence_linear_verifier_training_soft_logit_scale = $PatchEvidenceLinearVerifierTrainingSoftLogitScale
        patch_evidence_linear_verifier_training_soft_gate_temperature = $PatchEvidenceLinearVerifierTrainingSoftGateTemperature
        bbox_prior_patch_context_head = [bool]$BBoxPriorPatchContextHead
        bbox_prior_patch_context_hidden_dim = $BBoxPriorPatchContextHiddenDim
        bbox_prior_patch_context_dropout = $BBoxPriorPatchContextDropout
        bbox_prior_patch_context_logit_scale = $BBoxPriorPatchContextLogitScale
        bbox_prior_patch_context_temperature = $BBoxPriorPatchContextTemperature
        source_context_feature_fusion = [bool]$SourceContextFeatureFusion
        source_context_fusion_hidden_dim = $SourceContextFusionHiddenDim
        source_context_fusion_dropout = $SourceContextFusionDropout
        source_context_fusion_logit_scale = $SourceContextFusionLogitScale
        source_context_fusion_gate_bias = $SourceContextFusionGateBias
        paired_view_feature_fusion = [bool]$PairedViewFeatureFusion
        paired_view_fusion_hidden_dim = $PairedViewFusionHiddenDim
        paired_view_fusion_dropout = $PairedViewFusionDropout
        paired_view_fusion_logit_scale = $PairedViewFusionLogitScale
        paired_view_fusion_gate_bias = $PairedViewFusionGateBias
        learning_rate = $LearningRate
        min_learning_rate = $MinLearningRate
        disable_balanced_epoch_sampling = [bool]$DisableBalancedEpochSampling
        trainable_module_prefixes = $TrainableModulePrefixes
        classification_mlp_head = [bool]$ClassificationMlpHead
        classification_mlp_hidden_dim = $ClassificationMlpHiddenDim
        classification_mlp_dropout = $ClassificationMlpDropout
        classification_mlp_residual_scale = $ClassificationMlpResidualScale
        branch_cnn_tokens = $BranchCnnTokens
        warmup_epochs = $WarmupEpochs
        weight_decay = $WeightDecay
        backbone_lr_scale = $BackboneLrScale
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
        self_adaptive_target_loss_weight = $SelfAdaptiveTargetLossWeight
        self_adaptive_target_beta = $SelfAdaptiveTargetBeta
        self_adaptive_target_start_epoch = $SelfAdaptiveTargetStartEpoch
        self_adaptive_target_hard_weight = $SelfAdaptiveTargetHardWeight
        self_adaptive_target_confidence_power = $SelfAdaptiveTargetConfidencePower
        self_adaptive_target_min_confidence = $SelfAdaptiveTargetMinConfidence
        rdrop_loss_weight = $RDropLossWeight
        rdrop_temperature = $RDropTemperature
        augmix_consistency_loss_weight = $AugmixConsistencyLossWeight
        augmix_consistency_probability = $AugmixConsistencyProbability
        augmix_consistency_severity = $AugmixConsistencySeverity
        augmix_consistency_width = $AugmixConsistencyWidth
        augmix_consistency_depth = $AugmixConsistencyDepth
        augmix_consistency_alpha = $AugmixConsistencyAlpha
        augmix_consistency_temperature = $AugmixConsistencyTemperature
        illumination_consistency_loss_weight = $IlluminationConsistencyLossWeight
        illumination_consistency_probability = $IlluminationConsistencyProbability
        illumination_consistency_brightness = $IlluminationConsistencyBrightness
        illumination_consistency_contrast = $IlluminationConsistencyContrast
        illumination_consistency_gamma = $IlluminationConsistencyGamma
        illumination_consistency_temperature = $IlluminationConsistencyTemperature
        foreground_chroma_consistency_loss_weight = $ForegroundChromaConsistencyLossWeight
        foreground_chroma_consistency_probability = $ForegroundChromaConsistencyProbability
        foreground_chroma_consistency_saturation_delta = $ForegroundChromaConsistencySaturationDelta
        foreground_chroma_consistency_hue_delta = $ForegroundChromaConsistencyHueDelta
        foreground_chroma_consistency_bbox_margin_ratio = $ForegroundChromaConsistencyBboxMarginRatio
        foreground_chroma_consistency_temperature = $ForegroundChromaConsistencyTemperature
        foreground_chroma_consistency_classes = $ForegroundChromaConsistencyClasses
        friendly_adversarial_loss_weight = $FriendlyAdversarialLossWeight
        friendly_adversarial_epsilon = $FriendlyAdversarialEpsilon
        friendly_adversarial_step_size = $FriendlyAdversarialStepSize
        friendly_adversarial_steps = $FriendlyAdversarialSteps
        friendly_adversarial_bbox_erode_ratio = $FriendlyAdversarialBboxErodeRatio
        friendly_adversarial_focus_class = $FriendlyAdversarialFocusClass
        friendly_adversarial_negative_classes = $FriendlyAdversarialNegativeClasses
        friendly_adversarial_max_per_direction = $FriendlyAdversarialMaxPerDirection
        friendly_adversarial_start_epoch = $FriendlyAdversarialStartEpoch
        semantic_attribute_loss_weight = $SemanticAttributeLossWeight
        semantic_attribute_specs = $SemanticAttributeSpecs
        confusion_pair_mixup_loss_weight = $ConfusionPairMixupLossWeight
        confusion_pair_mixup_alpha = $ConfusionPairMixupAlpha
        confusion_pair_mixup_pairs = $ConfusionPairMixupPairs
        confusion_pair_mixup_max_pairs = $ConfusionPairMixupMaxPairs
        confusion_pair_mixup_start_epoch = $ConfusionPairMixupStartEpoch
        foreground_snapmix_loss_weight = $ForegroundSnapmixLossWeight
        foreground_snapmix_probability = $ForegroundSnapmixProbability
        foreground_snapmix_alpha = $ForegroundSnapmixAlpha
        foreground_snapmix_pairs = $ForegroundSnapmixPairs
        foreground_snapmix_min_area_ratio = $ForegroundSnapmixMinAreaRatio
        foreground_snapmix_max_area_ratio = $ForegroundSnapmixMaxAreaRatio
        foreground_snapmix_bbox_margin_ratio = $ForegroundSnapmixBboxMarginRatio
        foreground_counterexample_mix_loss_weight = $ForegroundCounterexampleMixLossWeight
        foreground_counterexample_mix_probability = $ForegroundCounterexampleMixProbability
        foreground_counterexample_mix_source_class = $ForegroundCounterexampleMixSourceClass
        foreground_counterexample_mix_target_classes = $ForegroundCounterexampleMixTargetClasses
        foreground_counterexample_mix_alpha = $ForegroundCounterexampleMixAlpha
        foreground_counterexample_mix_min_area_ratio = $ForegroundCounterexampleMixMinAreaRatio
        foreground_counterexample_mix_max_area_ratio = $ForegroundCounterexampleMixMaxAreaRatio
        foreground_counterexample_mix_bbox_margin_ratio = $ForegroundCounterexampleMixBboxMarginRatio
        foreground_counterexample_mix_start_epoch = $ForegroundCounterexampleMixStartEpoch
        dcl_region_shuffle_loss_weight = $DclRegionShuffleLossWeight
        dcl_region_shuffle_probability = $DclRegionShuffleProbability
        dcl_region_shuffle_grid_size = $DclRegionShuffleGridSize
        dcl_region_shuffle_bbox_margin_ratio = $DclRegionShuffleBboxMarginRatio
        dcl_region_shuffle_classes = $DclRegionShuffleClasses
        dcl_region_shuffle_start_epoch = $DclRegionShuffleStartEpoch
        quantized_label_cpu_loss_weight = $QuantizedLabelCpuLossWeight
        quantized_label_cpu_classes = $QuantizedLabelCpuClasses
        quantized_label_cpu_min_prior = $QuantizedLabelCpuMinPrior
        quantized_label_cpu_max_prior = $QuantizedLabelCpuMaxPrior
        quantized_label_cpu_negative_weight = $QuantizedLabelCpuNegativeWeight
        quantized_label_cpu_non_negative_beta = $QuantizedLabelCpuNonNegativeBeta
        quantized_label_cpu_start_epoch = $QuantizedLabelCpuStartEpoch
        self_paced_loss_weight = $SelfPacedLossWeight
        self_paced_loss_percentile = $SelfPacedLossPercentile
        self_paced_loss_gamma = $SelfPacedLossGamma
        self_paced_loss_min_weight = $SelfPacedLossMinWeight
        self_paced_loss_start_epoch = $SelfPacedLossStartEpoch
        self_paced_loss_class_balanced = [bool]$SelfPacedLossClassBalanced
        cyflod_loss_damping_weight = $CyflodLossDampingWeight
        cyflod_loss_damping_delta = $CyflodLossDampingDelta
        cyflod_loss_damping_cycle_epochs = $CyflodLossDampingCycleEpochs
        cyflod_loss_damping_min_weight = $CyflodLossDampingMinWeight
        cyflod_loss_damping_start_epoch = $CyflodLossDampingStartEpoch
        metric_learning_loss_weight = $MetricLearningLossWeight
        metric_learning_temperature = $MetricLearningTemperature
        metric_learning_sources = $MetricLearningSources
        teacher_guided_contrastive_loss_weight = $TeacherGuidedContrastiveLossWeight
        teacher_guided_contrastive_temperature = $TeacherGuidedContrastiveTemperature
        teacher_guided_contrastive_sources = $TeacherGuidedContrastiveSources
        teacher_guided_contrastive_classes = $TeacherGuidedContrastiveClasses
        teacher_guided_contrastive_teacher_min_confidence = $TeacherGuidedContrastiveTeacherMinConfidence
        teacher_guided_contrastive_require_agreement = [bool]$TeacherGuidedContrastiveRequireAgreement
        teacher_guided_contrastive_class_balanced = [bool]$TeacherGuidedContrastiveClassBalanced
        teacher_guided_contrastive_weight_mode = $TeacherGuidedContrastiveWeightMode
        teacher_guided_contrastive_stochastic_std = $TeacherGuidedContrastiveStochasticStd
        teacher_guided_contrastive_min_reliability = $TeacherGuidedContrastiveMinReliability
        teacher_guided_contrastive_teacher_confidence_power = $TeacherGuidedContrastiveTeacherConfidencePower
        teacher_guided_contrastive_memory_queue_size = $TeacherGuidedContrastiveMemoryQueueSize
        teacher_guided_contrastive_memory_min_count = $TeacherGuidedContrastiveMemoryMinCount
        boundary_contrastive_loss_weight = $BoundaryContrastiveLossWeight
        boundary_contrastive_pairs = $BoundaryContrastivePairs
        boundary_contrastive_sources = $BoundaryContrastiveSources
        boundary_contrastive_margin = $BoundaryContrastiveMargin
        boundary_contrastive_temperature = $BoundaryContrastiveTemperature
        boundary_contrastive_max_pairs = $BoundaryContrastiveMaxPairs
        boundary_center_loss_weight = $BoundaryCenterLossWeight
        boundary_center_pairs = $BoundaryCenterPairs
        boundary_center_sources = $BoundaryCenterSources
        boundary_center_margin = $BoundaryCenterMargin
        boundary_center_temperature = $BoundaryCenterTemperature
        boundary_center_compactness_weight = $BoundaryCenterCompactnessWeight
        boundary_center_teacher_min_confidence = $BoundaryCenterTeacherMinConfidence
        boundary_center_require_agreement = [bool]$BoundaryCenterRequireAgreement
        boundary_center_teacher_weight_mode = $BoundaryCenterTeacherWeightMode
        border_attention_suppression_loss_weight = $BorderAttentionSuppressionLossWeight
        border_attention_suppression_frame_width = $BorderAttentionSuppressionFrameWidth
        border_attention_suppression_bbox_band = $BorderAttentionSuppressionBboxBand
        border_attention_suppression_bbox_weight = $BorderAttentionSuppressionBboxWeight
        border_attention_suppression_temperature = $BorderAttentionSuppressionTemperature
        border_attention_suppression_classes = $BorderAttentionSuppressionClasses
        border_attention_suppression_start_epoch = $BorderAttentionSuppressionStartEpoch
        register_attention_alignment_loss_weight = $RegisterAttentionAlignmentLossWeight
        register_attention_alignment_classes = $RegisterAttentionAlignmentClasses
        register_attention_alignment_bbox_margin_ratio = $RegisterAttentionAlignmentBboxMarginRatio
        register_attention_alignment_agreement_weight = $RegisterAttentionAlignmentAgreementWeight
        register_attention_alignment_foreground_weight = $RegisterAttentionAlignmentForegroundWeight
        register_attention_alignment_start_epoch = $RegisterAttentionAlignmentStartEpoch
        register_diversity_loss_weight = $RegisterDiversityLossWeight
        foreground_surface_aux_loss_weight = $ForegroundSurfaceAuxLossWeight
        foreground_surface_pairwise_loss_weight = $ForegroundSurfacePairwiseLossWeight
        interior_boundary_pairwise_loss_weight = $InteriorBoundaryPairwiseLossWeight
        angular_margin_loss_weight = $AngularMarginLossWeight
        angular_margin = $AngularMargin
        angular_margin_scale = $AngularMarginScale
        angular_margin_start_epoch = $AngularMarginStartEpoch
        angular_margin_classes = $AngularMarginClasses
        subcenter_proxy_loss_weight = $SubcenterProxyLossWeight
        subcenter_proxy_subcenters = $SubcenterProxySubcenters
        subcenter_proxy_margin = $SubcenterProxyMargin
        subcenter_proxy_scale = $SubcenterProxyScale
        subcenter_proxy_classes = $SubcenterProxyClasses
        subcenter_proxy_start_epoch = $SubcenterProxyStartEpoch
        subcenter_proxy_dropout = $SubcenterProxyDropout
        subcenter_proxy_init_std = $SubcenterProxyInitStd
        deep_abstention_loss_weight = $DeepAbstentionLossWeight
        deep_abstention_penalty = $DeepAbstentionPenalty
        deep_abstention_start_epoch = $DeepAbstentionStartEpoch
        deep_abstention_dropout = $DeepAbstentionDropout
        deep_abstention_initial_probability = $DeepAbstentionInitialProbability
        ordinal_boundary_loss_weight = $OrdinalBoundaryLossWeight
        ordinal_boundary_classes = $OrdinalBoundaryClasses
        ordinal_boundary_threshold_weights = $OrdinalBoundaryThresholdWeights
        ordinal_boundary_temperature = $OrdinalBoundaryTemperature
        ordinal_boundary_start_epoch = $OrdinalBoundaryStartEpoch
        pairwise_confusion_loss_weight = $PairwiseConfusionLossWeight
        pairwise_confusion_sources = $PairwiseConfusionSources
        pairwise_confusion_start_epoch = $PairwiseConfusionStartEpoch
        pairwise_confusion_normalize = [bool]$PairwiseConfusionNormalize
        confusion_spectral_loss_weight = $ConfusionSpectralLossWeight
        confusion_spectral_ema_momentum = $ConfusionSpectralEmaMomentum
        confusion_spectral_frequency_smoothing = $ConfusionSpectralFrequencySmoothing
        confusion_spectral_margin = $ConfusionSpectralMargin
        confusion_spectral_start_epoch = $ConfusionSpectralStartEpoch
        confusion_spectral_bidirectional = [bool]$ConfusionSpectralBidirectional
        sample_weight_manifest = $SampleWeightManifest
        sample_weight_factor = $SampleWeightFactor
        sample_weight_max = $SampleWeightMax
        quality_group_manifest = $QualityGroupManifest
        group_dro_loss_weight = $GroupDroLossWeight
        group_dro_temperature = $GroupDroTemperature
        group_dro_min_samples = $GroupDroMinSamples
        hard_sample_manifest = $HardSampleManifest
        hard_sample_repeat_factor = $HardSampleRepeatFactor
        ambiguous_soft_target_manifest = $AmbiguousSoftTargetManifest
        ambiguous_soft_target_alpha = $AmbiguousSoftTargetAlpha
        targeted_margin_manifest = $TargetedMarginManifest
        targeted_margin_loss_weight = $TargetedMarginLossWeight
        targeted_margin_default_margin = $TargetedMarginDefaultMargin
        targeted_margin_default_weight = $TargetedMarginDefaultWeight
        targeted_margin_max_weight = $TargetedMarginMaxWeight
        focus_neighbor_binary_manifest = $FocusNeighborBinaryManifest
        focus_neighbor_binary_loss_weight = $FocusNeighborBinaryLossWeight
        focus_neighbor_binary_focus_class = $FocusNeighborBinaryFocusClass
        focus_neighbor_binary_neighbor_classes = $FocusNeighborBinaryNeighborClasses
        focus_neighbor_binary_default_weight = $FocusNeighborBinaryDefaultWeight
        focus_neighbor_binary_max_weight = $FocusNeighborBinaryMaxWeight
        focused_false_positive_margin_loss_weight = $FocusedFalsePositiveMarginLossWeight
        focused_false_positive_class = $FocusedFalsePositiveClass
        focused_false_positive_negative_classes = $FocusedFalsePositiveNegativeClasses
        focused_false_positive_margin = $FocusedFalsePositiveMargin
        focused_false_positive_min_probability = $FocusedFalsePositiveMinProbability
        focused_false_positive_probability_power = $FocusedFalsePositiveProbabilityPower
        background_counterfactual_consistency_weight = $BackgroundCounterfactualConsistencyWeight
        background_counterfactual_probability = $BackgroundCounterfactualProbability
        background_counterfactual_mode = $BackgroundCounterfactualMode
        background_counterfactual_margin = $BackgroundCounterfactualMargin
        background_counterfactual_blur_kernel = $BackgroundCounterfactualBlurKernel
        background_counterfactual_temperature = $BackgroundCounterfactualTemperature
        background_focus_suppression_loss_weight = $BackgroundFocusSuppressionLossWeight
        background_focus_suppression_probability = $BackgroundFocusSuppressionProbability
        background_focus_suppression_focus_class = $BackgroundFocusSuppressionFocusClass
        background_focus_suppression_negative_classes = $BackgroundFocusSuppressionNegativeClasses
        background_focus_suppression_margin = $BackgroundFocusSuppressionMargin
        background_focus_suppression_min_probability = $BackgroundFocusSuppressionMinProbability
        background_focus_suppression_probability_power = $BackgroundFocusSuppressionProbabilityPower
        source_context_focus_suppression_loss_weight = $SourceContextFocusSuppressionLossWeight
        source_context_focus_suppression_probability = $SourceContextFocusSuppressionProbability
        source_context_focus_suppression_focus_class = $SourceContextFocusSuppressionFocusClass
        source_context_focus_suppression_negative_classes = $SourceContextFocusSuppressionNegativeClasses
        source_context_focus_suppression_margin = $SourceContextFocusSuppressionMargin
        source_context_focus_suppression_min_probability = $SourceContextFocusSuppressionMinProbability
        source_context_focus_suppression_probability_power = $SourceContextFocusSuppressionProbabilityPower
        surface_counterfactual_consistency_weight = $SurfaceCounterfactualConsistencyWeight
        surface_counterfactual_probability = $SurfaceCounterfactualProbability
        surface_counterfactual_mode = $SurfaceCounterfactualMode
        surface_counterfactual_strength = $SurfaceCounterfactualStrength
        surface_counterfactual_blur_kernel = $SurfaceCounterfactualBlurKernel
        surface_counterfactual_temperature = $SurfaceCounterfactualTemperature
        surface_amplified_supervised_loss_weight = $SurfaceAmplifiedSupervisedLossWeight
        surface_amplified_boundary_margin_loss_weight = $SurfaceAmplifiedBoundaryMarginLossWeight
        surface_amplified_probability = $SurfaceAmplifiedProbability
        surface_amplified_mode = $SurfaceAmplifiedMode
        surface_amplified_strength = $SurfaceAmplifiedStrength
        surface_amplified_blur_kernel = $SurfaceAmplifiedBlurKernel
        surface_amplified_boundary_pairs = $SurfaceAmplifiedBoundaryPairs
        surface_amplified_boundary_margin = $SurfaceAmplifiedBoundaryMargin
        paired_view_supervised_loss_weight = $PairedViewSupervisedLossWeight
        paired_view_consistency_weight = $PairedViewConsistencyWeight
        paired_view_feature_consistency_weight = $PairedViewFeatureConsistencyWeight
        paired_view_fusion_loss_weight = $PairedViewFusionLossWeight
        paired_view_fusion_consistency_weight = $PairedViewFusionConsistencyWeight
        paired_view_temperature = $PairedViewTemperature
        paired_view_feature_source = $PairedViewFeatureSource
        masked_reconstruction_loss_weight = $MaskedReconstructionLossWeight
        masked_reconstruction_mask_ratio = $MaskedReconstructionMaskRatio
        masked_reconstruction_foreground_weight = $MaskedReconstructionForegroundWeight
        masked_reconstruction_detail_weight = $MaskedReconstructionDetailWeight
        masked_reconstruction_bbox_weight = $MaskedReconstructionBboxWeight
        masked_reconstruction_bbox_margin_ratio = $MaskedReconstructionBboxMarginRatio
        distillation_teacher_csv = $DistillationTeacherCsv
        distillation_weight = $DistillationWeight
        distillation_temperature = $DistillationTemperature
        distillation_focus_class_index = $DistillationFocusClassIndex
        distillation_focus_class_weight = $DistillationFocusClassWeight
        teacher_non_target_distillation_loss_weight = $TeacherNonTargetDistillationLossWeight
        teacher_non_target_distillation_classes = $TeacherNonTargetDistillationClasses
        teacher_non_target_distillation_temperature = $TeacherNonTargetDistillationTemperature
        teacher_non_target_distillation_teacher_min_confidence = $TeacherNonTargetDistillationTeacherMinConfidence
        teacher_non_target_distillation_require_agreement = [bool]$TeacherNonTargetDistillationRequireAgreement
        teacher_focus_margin_loss_weight = $TeacherFocusMarginLossWeight
        teacher_focus_margin_focus_class = $TeacherFocusMarginFocusClass
        teacher_focus_margin_negative_classes = $TeacherFocusMarginNegativeClasses
        teacher_focus_margin_teacher_max_probability = $TeacherFocusMarginTeacherMaxProbability
        teacher_focus_margin_margin = $TeacherFocusMarginMargin
        teacher_focus_margin_min_probability = $TeacherFocusMarginMinProbability
        teacher_focus_margin_probability_power = $TeacherFocusMarginProbabilityPower
        teacher_focus_margin_require_agreement = [bool]$TeacherFocusMarginRequireAgreement
        teacher_focus_binary_loss_weight = $TeacherFocusBinaryLossWeight
        teacher_focus_binary_focus_class = $TeacherFocusBinaryFocusClass
        teacher_focus_binary_classes = $TeacherFocusBinaryClasses
        teacher_focus_binary_teacher_min_confidence = $TeacherFocusBinaryTeacherMinConfidence
        teacher_focus_binary_error_power = $TeacherFocusBinaryErrorPower
        teacher_focus_binary_hard_target_blend = $TeacherFocusBinaryHardTargetBlend
        teacher_focus_binary_require_agreement = [bool]$TeacherFocusBinaryRequireAgreement
        teacher_pairwise_margin_loss_weight = $TeacherPairwiseMarginLossWeight
        teacher_pairwise_margin_teacher_mass_threshold = $TeacherPairwiseMarginTeacherMassThreshold
        teacher_pairwise_margin_error_power = $TeacherPairwiseMarginErrorPower
        teacher_pairwise_margin_hard_target_blend = $TeacherPairwiseMarginHardTargetBlend
        teacher_pairwise_margin_require_agreement = [bool]$TeacherPairwiseMarginRequireAgreement
        teacher_feature_npz = $TeacherFeatureNpz
        teacher_feature_rkd_loss_weight = $TeacherFeatureRkdLossWeight
        teacher_feature_rkd_distance_weight = $TeacherFeatureRkdDistanceWeight
        teacher_feature_rkd_angle_weight = $TeacherFeatureRkdAngleWeight
        teacher_feature_rkd_source = $TeacherFeatureRkdSource
        teacher_feature_rkd_pair_mode = $TeacherFeatureRkdPairMode
        teacher_feature_rkd_pairs = $TeacherFeatureRkdPairs
        teacher_feature_contrastive_loss_weight = $TeacherFeatureContrastiveLossWeight
        teacher_feature_contrastive_temperature = $TeacherFeatureContrastiveTemperature
        teacher_feature_contrastive_projection_dim = $TeacherFeatureContrastiveProjectionDim
        teacher_feature_contrastive_use_projection_adapter = [bool]$TeacherFeatureContrastiveUseProjectionAdapter
        teacher_feature_contrastive_adapter_dropout = $TeacherFeatureContrastiveAdapterDropout
        teacher_feature_contrastive_source = $TeacherFeatureContrastiveSource
        teacher_feature_contrastive_pair_mode = $TeacherFeatureContrastivePairMode
        teacher_feature_contrastive_pairs = $TeacherFeatureContrastivePairs
        teacher_feature_contrastive_start_epoch = $TeacherFeatureContrastiveStartEpoch
        class_loss_multipliers = $ClassLossMultipliers
        foreground_crop_mode = $ForegroundCropMode
        foreground_crop_probability = $ForegroundCropProbability
        foreground_crop_margin_ratio = $ForegroundCropMarginRatio
        foreground_crop_min_mask_area_ratio = $ForegroundCropMinMaskAreaRatio
        foreground_crop_max_mask_area_ratio = $ForegroundCropMaxMaskAreaRatio
        foreground_crop_max_crop_area_ratio = $ForegroundCropMaxCropAreaRatio
        sam = [bool]$Sam
        sam_rho = $SamRho
        sam_adaptive = [bool]$SamAdaptive
        foreground_surface_fusion = [bool]$ForegroundSurfaceFusion
        foreground_surface_fusion_dropout = $ForegroundSurfaceFusionDropout
        foreground_surface_pairwise_head = [bool]$ForegroundSurfacePairwiseHead
        foreground_surface_pairwise_pairs = $ForegroundSurfacePairwisePairs
        foreground_surface_pairwise_logit_scale = $ForegroundSurfacePairwiseLogitScale
        foreground_surface_pairwise_dropout = $ForegroundSurfacePairwiseDropout
        foreground_surface_pairwise_routing = [bool]$ForegroundSurfacePairwiseRouting
        foreground_surface_pairwise_route_max_probability_margin = $ForegroundSurfacePairwiseRouteMaxProbabilityMargin
        interior_boundary_pairwise_head = [bool]$InteriorBoundaryPairwiseHead
        interior_boundary_pairwise_pairs = $InteriorBoundaryPairwisePairs
        interior_boundary_pairwise_logit_scale = $InteriorBoundaryPairwiseLogitScale
        interior_boundary_pairwise_dropout = $InteriorBoundaryPairwiseDropout
        interior_boundary_pairwise_hidden_dim = $InteriorBoundaryPairwiseHiddenDim
        interior_boundary_pairwise_erode_kernel = $InteriorBoundaryPairwiseErodeKernel
        interior_boundary_pairwise_routing = [bool]$InteriorBoundaryPairwiseRouting
        interior_boundary_pairwise_route_max_probability_margin = $InteriorBoundaryPairwiseRouteMaxProbabilityMargin
        bilinear_patch_fusion = [bool]$BilinearPatchFusion
        bilinear_patch_rank = $BilinearPatchRank
        bilinear_patch_dropout = $BilinearPatchDropout
        complementary_patch_suppression_head = [bool]$ComplementaryPatchSuppressionHead
        complementary_patch_suppression_top_k = $ComplementaryPatchSuppressionTopK
        complementary_patch_suppression_hidden_dim = $ComplementaryPatchSuppressionHiddenDim
        complementary_patch_suppression_dropout = $ComplementaryPatchSuppressionDropout
        complementary_patch_suppression_temperature = $ComplementaryPatchSuppressionTemperature
        complementary_patch_suppression_strength = $ComplementaryPatchSuppressionStrength
        complementary_patch_suppression_bbox_weight = $ComplementaryPatchSuppressionBboxWeight
        complementary_patch_suppression_logit_scale = $ComplementaryPatchSuppressionLogitScale
        micro_detail_patch_expert = [bool]$MicroDetailPatchExpert
        micro_detail_top_k = $MicroDetailTopK
        micro_detail_hidden_dim = $MicroDetailHiddenDim
        micro_detail_dropout = $MicroDetailDropout
        micro_detail_temperature = $MicroDetailTemperature
        micro_detail_foreground_power = $MicroDetailForegroundPower
        micro_detail_logit_scale = $MicroDetailLogitScale
        micro_detail_routing = [bool]$MicroDetailRouting
        micro_detail_route_pairs = $MicroDetailRoutePairs
        micro_detail_route_max_probability_margin = $MicroDetailRouteMaxProbabilityMargin
        micro_detail_aux_loss_weight = $MicroDetailAuxLossWeight
        part_token_learner = [bool]$PartTokenLearner
        part_token_count = $PartTokenCount
        part_token_hidden_dim = $PartTokenHiddenDim
        part_token_dropout = $PartTokenDropout
        part_token_temperature = $PartTokenTemperature
        part_token_foreground_power = $PartTokenForegroundPower
        part_token_bbox_weight = $PartTokenBboxWeight
        part_token_logit_scale = $PartTokenLogitScale
        part_token_routing = [bool]$PartTokenRouting
        part_token_route_pairs = $PartTokenRoutePairs
        part_token_route_max_probability_margin = $PartTokenRouteMaxProbabilityMargin
        part_token_aux_loss_weight = $PartTokenAuxLossWeight
        part_token_pairwise_head = [bool]$PartTokenPairwiseHead
        part_token_pairwise_pairs = $PartTokenPairwisePairs
        part_token_pairwise_logit_scale = $PartTokenPairwiseLogitScale
        part_token_pairwise_dropout = $PartTokenPairwiseDropout
        part_token_pairwise_routing = [bool]$PartTokenPairwiseRouting
        part_token_pairwise_route_max_probability_margin = $PartTokenPairwiseRouteMaxProbabilityMargin
        part_token_pairwise_loss_weight = $PartTokenPairwiseLossWeight
        local_zoom_image_expert = [bool]$LocalZoomImageExpert
        local_zoom_crop_size = $LocalZoomCropSize
        local_zoom_crop_scale = $LocalZoomCropScale
        local_zoom_score_mode = $LocalZoomScoreMode
        local_zoom_hidden_dim = $LocalZoomHiddenDim
        local_zoom_dropout = $LocalZoomDropout
        local_zoom_logit_scale = $LocalZoomLogitScale
        local_zoom_routing = [bool]$LocalZoomRouting
        local_zoom_route_pairs = $LocalZoomRoutePairs
        local_zoom_route_max_probability_margin = $LocalZoomRouteMaxProbabilityMargin
        local_zoom_aux_loss_weight = $LocalZoomAuxLossWeight
        high_frequency_texture_expert = [bool]$HighFrequencyTextureExpert
        high_frequency_texture_hidden_dim = $HighFrequencyTextureHiddenDim
        high_frequency_texture_dropout = $HighFrequencyTextureDropout
        high_frequency_texture_analysis_size = $HighFrequencyTextureAnalysisSize
        high_frequency_texture_logit_scale = $HighFrequencyTextureLogitScale
        high_frequency_texture_routing = [bool]$HighFrequencyTextureRouting
        high_frequency_texture_route_pairs = $HighFrequencyTextureRoutePairs
        high_frequency_texture_route_max_probability_margin = $HighFrequencyTextureRouteMaxProbabilityMargin
        high_frequency_texture_aux_loss_weight = $HighFrequencyTextureAuxLossWeight
        high_frequency_texture_pairwise_loss_weight = $HighFrequencyTexturePairwiseLossWeight
        high_frequency_texture_pairwise_pairs = $HighFrequencyTexturePairwisePairs
        multi_granularity_aux_heads = [bool]$MultiGranularityAuxHeads
        multi_granularity_aux_layers = $MultiGranularityAuxLayers
        multi_granularity_aux_dropout = $MultiGranularityAuxDropout
        multi_granularity_aux_loss_weight = $MultiGranularityAuxLossWeight
        multi_granularity_refinement_loss_weight = $MultiGranularityRefinementLossWeight
        multi_granularity_refinement_temperature = $MultiGranularityRefinementTemperature
        multi_granularity_contrastive_loss_weight = $MultiGranularityContrastiveLossWeight
        multi_granularity_contrastive_temperature = $MultiGranularityContrastiveTemperature
        multi_granularity_contrastive_pairs = $MultiGranularityContrastivePairs
        multi_granularity_contrastive_teacher_min_confidence = $MultiGranularityContrastiveTeacherMinConfidence
        multi_granularity_contrastive_require_agreement = [bool]$MultiGranularityContrastiveRequireAgreement
        multi_granularity_contrastive_weight_mode = $MultiGranularityContrastiveWeightMode
        multi_granularity_contrastive_teacher_confidence_power = $MultiGranularityContrastiveTeacherConfidencePower
        self_boosting_attention_head = [bool]$SelfBoostingAttentionHeadEffective
        self_boosting_attention_loss_weight = $SelfBoostingAttentionLossWeight
        self_boosting_attention_temperature = $SelfBoostingAttentionTemperature
        self_boosting_attention_classes = $SelfBoostingAttentionClasses
        self_boosting_attention_start_epoch = $SelfBoostingAttentionStartEpoch
        block_local_patch_mixer = [bool]$BlockLocalPatchMixer
        block_local_patch_mixer_layers = $BlockLocalPatchMixerLayers
        block_local_patch_mixer_dropout = $BlockLocalPatchMixerDropout
        block_local_patch_mixer_scale = $BlockLocalPatchMixerScale
        locally_enhanced_ffn = [bool]$LocallyEnhancedFfn
        locally_enhanced_ffn_layers = $LocallyEnhancedFfnLayers
        locally_enhanced_ffn_kernel_size = $LocallyEnhancedFfnKernelSize
        concurrent_local_global_coupling = [bool]$ConcurrentLocalGlobalCoupling
        concurrent_local_global_layers = $ConcurrentLocalGlobalLayers
        concurrent_local_global_dim = $ConcurrentLocalGlobalDim
        concurrent_local_global_kernel_size = $ConcurrentLocalGlobalKernelSize
        gated_relative_position_attention = [bool]$GatedRelativePositionAttention
        gated_relative_position_attention_layers = $GatedRelativePositionAttentionLayers
        gated_relative_position_attention_max_mix = $GatedRelativePositionAttentionMaxMix
        gated_relative_position_attention_locality_strength = $GatedRelativePositionAttentionLocalityStrength
        visual_contrast_attention = [bool]$VisualContrastAttention
        visual_contrast_attention_layers = $VisualContrastAttentionLayers
        visual_contrast_tokens = $VisualContrastTokens
        cross_covariance_attention = [bool]$CrossCovarianceAttention
        cross_covariance_attention_layers = $CrossCovarianceAttentionLayers
        cross_covariance_attention_residual_scale = $CrossCovarianceAttentionResidualScale
        dynamic_graph_mixer = [bool]$DynamicGraphMixer
        dynamic_graph_mixer_layers = $DynamicGraphMixerLayers
        dynamic_graph_mixer_bottleneck_dim = $DynamicGraphMixerBottleneckDim
        dynamic_graph_mixer_k = $DynamicGraphMixerK
        patch_style_recalibration = [bool]$PatchStyleRecalibration
        patch_style_recalibration_layers = $PatchStyleRecalibrationLayers
        layer_token_fusion = [bool]$LayerTokenFusion
        layer_token_fusion_layers = $LayerTokenFusionLayers
        layer_token_fusion_top_k = $LayerTokenFusionTopK
        layer_token_fusion_blend = $LayerTokenFusionBlend
        layer_token_fusion_attention_temperature = $LayerTokenFusionAttentionTemperature
        layer_token_fusion_bbox_weight = $LayerTokenFusionBboxWeight
        layer_token_fusion_foreground_weight = $LayerTokenFusionForegroundWeight
        frequency_selective_pooling = [bool]$FrequencySelectivePooling
        frequency_selective_top_k = $FrequencySelectiveTopK
        frequency_selective_blend = $FrequencySelectiveBlend
        frequency_selective_foreground_threshold = $FrequencySelectiveForegroundThreshold
        patch_memory_adapter = [bool]$PatchMemoryAdapter
        patch_memory_adapter_dropout = $PatchMemoryAdapterDropout
        late_class_attention_pooling = [bool]$LateClassAttentionPooling
        late_class_attention_heads = $LateClassAttentionHeads
        late_class_attention_dropout = $LateClassAttentionDropout
        late_class_attention_mlp_ratio = $LateClassAttentionMlpRatio
        late_class_attention_residual_scale = $LateClassAttentionResidualScale
        mixstyle = [bool]$MixStyle
        mixstyle_probability = $MixStyleProbability
        mixstyle_alpha = $MixStyleAlpha
        pairwise_margin_pairs = $PairwiseMarginPairs
        pairwise_margin_logit_scale = $PairwiseMarginLogitScale
        pairwise_margin_routing = [bool]$PairwiseMarginRouting
        pairwise_margin_route_max_probability_margin = $PairwiseMarginRouteMaxProbabilityMargin
        topk_reassessment_head = [bool]$TopKReassessmentHead
        topk_reassessment_top_k = $TopKReassessmentTopK
        topk_reassessment_hidden_dim = $TopKReassessmentHiddenDim
        topk_reassessment_dropout = $TopKReassessmentDropout
        topk_reassessment_logit_scale = $TopKReassessmentLogitScale
        topk_reassessment_routing = [bool]$TopKReassessmentRouting
        topk_reassessment_route_pairs = $TopKReassessmentRoutePairs
        topk_reassessment_route_max_probability_margin = $TopKReassessmentRouteMaxProbabilityMargin
        topk_reassessment_aux_loss_weight = $TopKReassessmentAuxLossWeight
        topk_reassessment_aux_route_min_weight = $TopKReassessmentAuxRouteMinWeight
        focus_class_head = [bool]$FocusClassHead
        focus_class_index = $FocusClassIndex
        focus_class_logit_scale = $FocusClassLogitScale
        focus_class_dropout = $FocusClassDropout
        focus_class_routing = [bool]$FocusClassRouting
        focus_class_route_max_probability_margin = $FocusClassRouteMaxProbabilityMargin
        focus_class_route_min_probability = $FocusClassRouteMinProbability
        focus_class_aux_loss_weight = $FocusClassAuxLossWeight
        focus_class_aux_positive_weight = $FocusClassAuxPositiveWeight
        class_independent_head = [bool]$ClassIndependentHead
        class_independent_dropout = $ClassIndependentDropout
        class_independent_loss_weight = $ClassIndependentLossWeight
        class_independent_positive_weight = $ClassIndependentPositiveWeight
        focus_tversky_loss_weight = $FocusTverskyLossWeight
        focus_tversky_class = $FocusTverskyClass
        focus_tversky_alpha = $FocusTverskyAlpha
        focus_tversky_beta = $FocusTverskyBeta
        focus_tversky_gamma = $FocusTverskyGamma
        focus_tversky_probability_power = $FocusTverskyProbabilityPower
        focus_tversky_start_epoch = $FocusTverskyStartEpoch
        focus_auc_rank_loss_weight = $FocusAucRankLossWeight
        focus_auc_rank_class = $FocusAucRankClass
        focus_auc_rank_negative_classes = $FocusAucRankNegativeClasses
        focus_auc_rank_margin = $FocusAucRankMargin
        focus_auc_rank_temperature = $FocusAucRankTemperature
        focus_auc_rank_hard_fraction = $FocusAucRankHardFraction
        focus_auc_rank_start_epoch = $FocusAucRankStartEpoch
        focus_partial_auc_loss_weight = $FocusPartialAucLossWeight
        focus_partial_auc_class = $FocusPartialAucClass
        focus_partial_auc_negative_classes = $FocusPartialAucNegativeClasses
        focus_partial_auc_margin = $FocusPartialAucMargin
        focus_partial_auc_temperature = $FocusPartialAucTemperature
        focus_partial_auc_negative_fraction = $FocusPartialAucNegativeFraction
        focus_partial_auc_positive_fraction = $FocusPartialAucPositiveFraction
        focus_partial_auc_positive_weight = $FocusPartialAucPositiveWeight
        focus_partial_auc_min_negative_probability = $FocusPartialAucMinNegativeProbability
        focus_partial_auc_start_epoch = $FocusPartialAucStartEpoch
        bbox_foreground_dropout_loss_weight = $BBoxForegroundDropoutLossWeight
        bbox_foreground_dropout_consistency_weight = $BBoxForegroundDropoutConsistencyWeight
        bbox_foreground_dropout_probability = $BBoxForegroundDropoutProbability
        bbox_foreground_dropout_min_area_ratio = $BBoxForegroundDropoutMinAreaRatio
        bbox_foreground_dropout_max_area_ratio = $BBoxForegroundDropoutMaxAreaRatio
        bbox_foreground_dropout_mode = $BBoxForegroundDropoutMode
        bbox_foreground_dropout_fill = $BBoxForegroundDropoutFill
        bbox_foreground_dropout_temperature = $BBoxForegroundDropoutTemperature
        bbox_object_erasure_negative_loss_weight = $BBoxObjectErasureNegativeLossWeight
        bbox_object_erasure_probability = $BBoxObjectErasureProbability
        bbox_object_erasure_margin_ratio = $BBoxObjectErasureMarginRatio
        bbox_object_erasure_fill = $BBoxObjectErasureFill
        bbox_object_erasure_blur_kernel = $BBoxObjectErasureBlurKernel
        bbox_object_erasure_temperature = $BBoxObjectErasureTemperature
        ordinal_maturity_head = [bool]$OrdinalMaturityHead
        ordinal_maturity_classes = $OrdinalMaturityClasses
        ordinal_maturity_logit_scale = $OrdinalMaturityLogitScale
        ordinal_maturity_dropout = $OrdinalMaturityDropout
        ordinal_maturity_loss_weight = $OrdinalMaturityLossWeight
        cumulative_ordinal_head = [bool]$CumulativeOrdinalHead
        cumulative_ordinal_classes = $CumulativeOrdinalClasses
        cumulative_ordinal_logit_scale = $CumulativeOrdinalLogitScale
        cumulative_ordinal_dropout = $CumulativeOrdinalDropout
        cumulative_ordinal_loss_weight = $CumulativeOrdinalLossWeight
        cumulative_ordinal_threshold_weights = $CumulativeOrdinalThresholdWeights
        ordinal_distribution_loss_weight = $OrdinalDistributionLossWeight
        ordinal_distribution_classes = $OrdinalDistributionClasses
        ordinal_distribution_target_sigma = $OrdinalDistributionTargetSigma
        ordinal_distribution_start_epoch = $OrdinalDistributionStartEpoch
        classification_loss = $ClassificationLoss
        balanced_softmax_tau = $BalancedSoftmaxTau
        gce_q = $GceQ
        ldr_margin = $LdrMargin
        ldr_temperature = $LdrTemperature
        logit_norm_temperature = $LogitNormTemperature
        symmetric_ce_alpha = $SymmetricCeAlpha
        symmetric_ce_beta = $SymmetricCeBeta
        symmetric_ce_epsilon = $SymmetricCeEpsilon
        seesaw_mitigation_power = $SeesawMitigationPower
        seesaw_compensation_power = $SeesawCompensationPower
        focal_loss_gamma = $FocalLossGamma
        focal_loss_mix = $FocalLossMix
        label_smoothing = $LabelSmoothing
        ldam_max_margin = $LdamMaxMargin
        ldam_scale = $LdamScale
        mutual_channel_loss_weight = $MutualChannelLossWeight
        mutual_channel_top_k = $MutualChannelTopK
        mutual_channel_diversity_weight = $MutualChannelDiversityWeight
        mutual_channel_start_epoch = $MutualChannelStartEpoch
        complement_entropy_loss_weight = $ComplementEntropyLossWeight
        complement_entropy_classes = $ComplementEntropyClasses
        complement_entropy_start_epoch = $ComplementEntropyStartEpoch
        background_suppression_mode = $BackgroundSuppressionMode
        background_suppression_probability = $BackgroundSuppressionProbability
        background_suppression_margin = $BackgroundSuppressionMargin
        background_suppression_blur_radius = $BackgroundSuppressionBlurRadius
        surface_detail_amplification_mode = $SurfaceDetailAmplificationMode
        surface_detail_amplification_probability = $SurfaceDetailAmplificationProbability
        surface_detail_amplification_strength = $SurfaceDetailAmplificationStrength
        surface_detail_amplification_blur_radius = $SurfaceDetailAmplificationBlurRadius
        surface_detail_amplification_foreground_weight = $SurfaceDetailAmplificationForegroundWeight
        eval_surface_detail_amplification = [bool]$EvalSurfaceDetailAmplification
        foreground_background_mix_probability = $ForegroundBackgroundMixProbability
        foreground_background_mix_margin = $ForegroundBackgroundMixMargin
        foreground_background_mix_min_foreground_fraction = $ForegroundBackgroundMixMinForegroundFraction
        foreground_background_mix_max_foreground_fraction = $ForegroundBackgroundMixMaxForegroundFraction
        foreground_background_mix_softness = $ForegroundBackgroundMixSoftness
        foreground_background_mix_mask_source = $ForegroundBackgroundMixMaskSource
        local_exposure_probability = $LocalExposureProbability
        local_exposure_strength = $LocalExposureStrength
        obstacle_probability = $ObstacleProbability
        obstacle_max_area = $ObstacleMaxArea
        randaugment_num_ops = $RandAugmentNumOps
        randaugment_magnitude = $RandAugmentMagnitude
        data_cartography = [bool]$DataCartography
        data_cartography_output = $DataCartographyOutput
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
$SchedulerTotalEpochsEffective = if ($SchedulerTotalEpochs -gt 0) {
    $SchedulerTotalEpochs
} else {
    $Epochs
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
        "--seed", "$Seed",
        "--model-type", "vit_registers",
        "--no-pretrained",
        "--no-pretrained-distillation",
        "--image-size", "$ImageSize",
        "--crop-margin-ratio", "$CropMarginRatio",
        "--class-crop-margin-scale-threshold", "$ClassCropMarginScaleThreshold",
        "--class-crop-margin-max-ratio", "$ClassCropMarginMaxRatio",
        "--classification-source-context-mode", "$ClassificationSourceContextMode",
        "--classification-source-context-layout", "$ClassificationSourceContextLayout",
        "--classification-source-context-margin-ratio", "$ClassificationSourceContextMarginRatio",
        "--classification-source-context-background-alpha", "$ClassificationSourceContextBackgroundAlpha",
        "--classification-source-context-blur-radius", "$ClassificationSourceContextBlurRadius",
        "--classification-source-context-inset-scale", "$ClassificationSourceContextInsetScale",
        "--patch-size", "16",
        "--stem-channels", "32",
        "--stem-architecture", "$StemArchitecture",
        "--stem-pooling-mode", "$StemPoolingMode",
        "--stem-softpool-blend", "$StemSoftpoolBlend",
        "--shifted-patch-shift", "$ShiftedPatchShift",
        "--shifted-patch-residual-scale", "$ShiftedPatchResidualScale",
        "--cnn-feature-fusion",
        "--cnn-fusion-dropout", "0.10",
        "--fine-grained-pooling",
        "--fine-grained-pooling-dropout", "0.08",
        "--multi-branch-fusion",
        "--branch-color-tokens", "1",
        "--branch-edge-tokens", "1",
        "--branch-cnn-tokens", "$BranchCnnTokens",
        "--branch-token-dropout", "0.08",
        "--detail-patch-enhancement",
        "--detail-patch-dropout", "0.05",
        "--token-prune-layers", "2,5",
        "--token-keep-rates", "0.85,0.65",
        "--token-prune-foreground-weight", "0.45",
        "--token-prune-bbox-weight", "$TokenPruneBboxWeight",
        "--token-prune-bbox-margin-ratio", "$TokenPruneBboxMarginRatio",
        "--bbox-token-prior-source", "$BboxTokenPriorSource",
        "--early-token-mask-keep-rate", "$EarlyTokenMaskKeepRate",
        "--pairwise-margin-head",
        "--pairwise-margin-pairs", "$PairwiseMarginPairs",
        "--pairwise-margin-logit-scale", "$PairwiseMarginLogitScale",
        "--pairwise-margin-dropout", "0.05",
        "--embed-dim", "256",
        "--depth", "8",
        "--num-heads", "8",
        "--num-registers", "4",
        "--register-positional-embedding",
        "--head-pooling", "cls_branch_register_mean",
        "--classification-mlp-hidden-dim", "$ClassificationMlpHiddenDim",
        "--classification-mlp-dropout", "$ClassificationMlpDropout",
        "--classification-mlp-residual-scale", "$ClassificationMlpResidualScale",
        "--dropout", "0.12",
        "--attention-dropout", "0.03",
        "--drop-path-rate", "0.10",
        "--batch-size", "$BatchSize",
        "--grad-accum-steps", "$GradAccumSteps",
        "--epochs", "$Epochs",
        "--scheduler-total-epochs", "$SchedulerTotalEpochsEffective",
        "--patience", "$Patience",
        "--learning-rate", "$LearningRate",
        "--backbone-lr-scale", "$BackboneLrScale",
        "--min-learning-rate", "$MinLearningRate",
        "--warmup-epochs", "$WarmupEpochs",
        "--weight-decay", "$WeightDecay",
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
        "--classification-loss", "$ClassificationLoss",
        "--balanced-softmax-tau", "$BalancedSoftmaxTau",
        "--gce-q", "$GceQ",
        "--ldr-margin", "$LdrMargin",
        "--ldr-temperature", "$LdrTemperature",
        "--logit-norm-temperature", "$LogitNormTemperature",
        "--symmetric-ce-alpha", "$SymmetricCeAlpha",
        "--symmetric-ce-beta", "$SymmetricCeBeta",
        "--symmetric-ce-epsilon", "$SymmetricCeEpsilon",
        "--seesaw-mitigation-power", "$SeesawMitigationPower",
        "--seesaw-compensation-power", "$SeesawCompensationPower",
        "--ldam-max-margin", "$LdamMaxMargin",
        "--ldam-scale", "$LdamScale",
        "--focal-loss-gamma", "$FocalLossGamma",
        "--focal-loss-mix", "$FocalLossMix",
        "--label-smoothing", "$LabelSmoothing",
        "--metric-learning-loss-weight", "$MetricLearningLossWeight",
        "--metric-learning-temperature", "$MetricLearningTemperature",
        "--metric-learning-sources", "$MetricLearningSources",
        "--teacher-guided-contrastive-loss-weight", "$TeacherGuidedContrastiveLossWeight",
        "--teacher-guided-contrastive-temperature", "$TeacherGuidedContrastiveTemperature",
        "--teacher-guided-contrastive-sources", "$TeacherGuidedContrastiveSources",
        "--teacher-guided-contrastive-classes", "$TeacherGuidedContrastiveClasses",
        "--teacher-guided-contrastive-teacher-min-confidence", "$TeacherGuidedContrastiveTeacherMinConfidence",
        "--teacher-guided-contrastive-weight-mode", "$TeacherGuidedContrastiveWeightMode",
        "--teacher-guided-contrastive-stochastic-std", "$TeacherGuidedContrastiveStochasticStd",
        "--teacher-guided-contrastive-min-reliability", "$TeacherGuidedContrastiveMinReliability",
        "--teacher-guided-contrastive-teacher-confidence-power", "$TeacherGuidedContrastiveTeacherConfidencePower",
        "--teacher-guided-contrastive-memory-queue-size", "$TeacherGuidedContrastiveMemoryQueueSize",
        "--teacher-guided-contrastive-memory-min-count", "$TeacherGuidedContrastiveMemoryMinCount",
        "--boundary-contrastive-loss-weight", "$BoundaryContrastiveLossWeight",
        "--boundary-contrastive-pairs", "$BoundaryContrastivePairs",
        "--boundary-contrastive-sources", "$BoundaryContrastiveSources",
        "--boundary-contrastive-margin", "$BoundaryContrastiveMargin",
        "--boundary-contrastive-temperature", "$BoundaryContrastiveTemperature",
        "--boundary-contrastive-max-pairs", "$BoundaryContrastiveMaxPairs",
        "--boundary-center-loss-weight", "$BoundaryCenterLossWeight",
        "--boundary-center-pairs", "$BoundaryCenterPairs",
        "--boundary-center-sources", "$BoundaryCenterSources",
        "--boundary-center-margin", "$BoundaryCenterMargin",
        "--boundary-center-temperature", "$BoundaryCenterTemperature",
        "--boundary-center-compactness-weight", "$BoundaryCenterCompactnessWeight",
        "--boundary-center-teacher-min-confidence", "$BoundaryCenterTeacherMinConfidence",
        "--boundary-center-teacher-weight-mode", "$BoundaryCenterTeacherWeightMode",
        "--foreground-surface-aux-loss-weight", "$ForegroundSurfaceAuxLossWeight",
        "--foreground-surface-pairwise-loss-weight", "$ForegroundSurfacePairwiseLossWeight",
        "--interior-boundary-pairwise-loss-weight", "$InteriorBoundaryPairwiseLossWeight",
        "--micro-detail-aux-loss-weight", "$MicroDetailAuxLossWeight",
        "--part-token-aux-loss-weight", "$PartTokenAuxLossWeight",
        "--part-token-pairwise-loss-weight", "$PartTokenPairwiseLossWeight",
        "--patch-objectness-loss-weight", "$PatchObjectnessLossWeight",
        "--patch-objectness-positive-weight", "$PatchObjectnessPositiveWeight",
        "--bbox-token-label-loss-weight", "$BboxTokenLabelLossWeight",
        "--bbox-token-label-min-prior", "$BboxTokenLabelMinPrior",
        "--bbox-token-label-prior-power", "$BboxTokenLabelPriorPower",
        "--bbox-token-label-classes", "$BboxTokenLabelClasses",
        "--bbox-token-label-focus-class", "$BboxTokenLabelFocusClass",
        "--bbox-token-label-focus-weight", "$BboxTokenLabelFocusWeight",
        "--bbox-token-label-start-epoch", "$BboxTokenLabelStartEpoch",
        "--patch-evidence-router-loss-weight", "$PatchEvidenceRouterLossWeight",
        "--patch-evidence-router-positive-weight", "$PatchEvidenceRouterPositiveWeight",
        "--patch-evidence-router-start-epoch", "$PatchEvidenceRouterStartEpoch",
        "--patch-evidence-router-teacher-loss-weight", "$PatchEvidenceRouterTeacherLossWeight",
        "--patch-evidence-router-teacher-min-confidence", "$PatchEvidenceRouterTeacherMinConfidence",
        "--patch-evidence-router-teacher-min-pair-mass", "$PatchEvidenceRouterTeacherMinPairMass",
        "--patch-evidence-router-teacher-positive-weight", "$PatchEvidenceRouterTeacherPositiveWeight",
        "--local-zoom-aux-loss-weight", "$LocalZoomAuxLossWeight",
        "--high-frequency-texture-aux-loss-weight", "$HighFrequencyTextureAuxLossWeight",
        "--high-frequency-texture-pairwise-loss-weight", "$HighFrequencyTexturePairwiseLossWeight",
        "--high-frequency-texture-pairwise-pairs", "$HighFrequencyTexturePairwisePairs",
        "--multi-granularity-aux-loss-weight", "$MultiGranularityAuxLossWeight",
        "--multi-granularity-refinement-loss-weight", "$MultiGranularityRefinementLossWeight",
        "--multi-granularity-refinement-temperature", "$MultiGranularityRefinementTemperature",
        "--multi-granularity-contrastive-loss-weight", "$MultiGranularityContrastiveLossWeight",
        "--self-boosting-attention-loss-weight", "$SelfBoostingAttentionLossWeight",
        "--self-boosting-attention-temperature", "$SelfBoostingAttentionTemperature",
        "--self-boosting-attention-classes", "$SelfBoostingAttentionClasses",
        "--self-boosting-attention-start-epoch", "$SelfBoostingAttentionStartEpoch",
        "--focus-class-aux-loss-weight", "$FocusClassAuxLossWeight",
        "--focus-class-aux-positive-weight", "$FocusClassAuxPositiveWeight",
        "--class-independent-loss-weight", "$ClassIndependentLossWeight",
        "--class-independent-positive-weight", "$ClassIndependentPositiveWeight",
        "--focus-tversky-loss-weight", "$FocusTverskyLossWeight",
        "--focus-tversky-class", "$FocusTverskyClass",
        "--focus-tversky-alpha", "$FocusTverskyAlpha",
        "--focus-tversky-beta", "$FocusTverskyBeta",
        "--focus-tversky-gamma", "$FocusTverskyGamma",
        "--focus-tversky-probability-power", "$FocusTverskyProbabilityPower",
        "--focus-tversky-start-epoch", "$FocusTverskyStartEpoch",
        "--focus-auc-rank-loss-weight", "$FocusAucRankLossWeight",
        "--focus-auc-rank-class", "$FocusAucRankClass",
        "--focus-auc-rank-negative-classes", "$FocusAucRankNegativeClasses",
        "--focus-auc-rank-margin", "$FocusAucRankMargin",
        "--focus-auc-rank-temperature", "$FocusAucRankTemperature",
        "--focus-auc-rank-hard-fraction", "$FocusAucRankHardFraction",
        "--focus-auc-rank-start-epoch", "$FocusAucRankStartEpoch",
        "--focus-partial-auc-loss-weight", "$FocusPartialAucLossWeight",
        "--focus-partial-auc-class", "$FocusPartialAucClass",
        "--focus-partial-auc-negative-classes", "$FocusPartialAucNegativeClasses",
        "--focus-partial-auc-margin", "$FocusPartialAucMargin",
        "--focus-partial-auc-temperature", "$FocusPartialAucTemperature",
        "--focus-partial-auc-negative-fraction", "$FocusPartialAucNegativeFraction",
        "--focus-partial-auc-positive-fraction", "$FocusPartialAucPositiveFraction",
        "--focus-partial-auc-positive-weight", "$FocusPartialAucPositiveWeight",
        "--focus-partial-auc-min-negative-probability", "$FocusPartialAucMinNegativeProbability",
        "--focus-partial-auc-start-epoch", "$FocusPartialAucStartEpoch",
        "--bbox-foreground-dropout-loss-weight", "$BBoxForegroundDropoutLossWeight",
        "--bbox-foreground-dropout-consistency-weight", "$BBoxForegroundDropoutConsistencyWeight",
        "--bbox-foreground-dropout-probability", "$BBoxForegroundDropoutProbability",
        "--bbox-foreground-dropout-min-area-ratio", "$BBoxForegroundDropoutMinAreaRatio",
        "--bbox-foreground-dropout-max-area-ratio", "$BBoxForegroundDropoutMaxAreaRatio",
        "--bbox-foreground-dropout-mode", "$BBoxForegroundDropoutMode",
        "--bbox-foreground-dropout-fill", "$BBoxForegroundDropoutFill",
        "--bbox-foreground-dropout-temperature", "$BBoxForegroundDropoutTemperature",
        "--bbox-object-erasure-negative-loss-weight", "$BBoxObjectErasureNegativeLossWeight",
        "--bbox-object-erasure-probability", "$BBoxObjectErasureProbability",
        "--bbox-object-erasure-margin-ratio", "$BBoxObjectErasureMarginRatio",
        "--bbox-object-erasure-fill", "$BBoxObjectErasureFill",
        "--bbox-object-erasure-blur-kernel", "$BBoxObjectErasureBlurKernel",
        "--bbox-object-erasure-temperature", "$BBoxObjectErasureTemperature",
        "--ordinal-maturity-loss-weight", "$OrdinalMaturityLossWeight",
        "--cumulative-ordinal-loss-weight", "$CumulativeOrdinalLossWeight",
        "--cumulative-ordinal-threshold-weights", "$CumulativeOrdinalThresholdWeights",
        "--ordinal-distribution-loss-weight", "$OrdinalDistributionLossWeight",
        "--ordinal-distribution-classes", "$OrdinalDistributionClasses",
        "--ordinal-distribution-target-sigma", "$OrdinalDistributionTargetSigma",
        "--ordinal-distribution-start-epoch", "$OrdinalDistributionStartEpoch",
        "--angular-margin-loss-weight", "$AngularMarginLossWeight",
        "--angular-margin", "$AngularMargin",
        "--angular-margin-scale", "$AngularMarginScale",
        "--angular-margin-start-epoch", "$AngularMarginStartEpoch",
        "--angular-margin-classes", "$AngularMarginClasses",
        "--subcenter-proxy-loss-weight", "$SubcenterProxyLossWeight",
        "--subcenter-proxy-subcenters", "$SubcenterProxySubcenters",
        "--subcenter-proxy-margin", "$SubcenterProxyMargin",
        "--subcenter-proxy-scale", "$SubcenterProxyScale",
        "--subcenter-proxy-classes", "$SubcenterProxyClasses",
        "--subcenter-proxy-start-epoch", "$SubcenterProxyStartEpoch",
        "--subcenter-proxy-dropout", "$SubcenterProxyDropout",
        "--subcenter-proxy-init-std", "$SubcenterProxyInitStd",
        "--deep-abstention-loss-weight", "$DeepAbstentionLossWeight",
        "--deep-abstention-penalty", "$DeepAbstentionPenalty",
        "--deep-abstention-start-epoch", "$DeepAbstentionStartEpoch",
        "--deep-abstention-dropout", "$DeepAbstentionDropout",
        "--deep-abstention-initial-probability", "$DeepAbstentionInitialProbability",
        "--ordinal-boundary-loss-weight", "$OrdinalBoundaryLossWeight",
        "--ordinal-boundary-classes", "$OrdinalBoundaryClasses",
        "--ordinal-boundary-threshold-weights", "$OrdinalBoundaryThresholdWeights",
        "--ordinal-boundary-temperature", "$OrdinalBoundaryTemperature",
        "--ordinal-boundary-start-epoch", "$OrdinalBoundaryStartEpoch",
        "--pairwise-confusion-loss-weight", "$PairwiseConfusionLossWeight",
        "--pairwise-confusion-sources", "$PairwiseConfusionSources",
        "--pairwise-confusion-start-epoch", "$PairwiseConfusionStartEpoch",
        "--confusion-spectral-loss-weight", "$ConfusionSpectralLossWeight",
        "--confusion-spectral-ema-momentum", "$ConfusionSpectralEmaMomentum",
        "--confusion-spectral-frequency-smoothing", "$ConfusionSpectralFrequencySmoothing",
        "--confusion-spectral-margin", "$ConfusionSpectralMargin",
        "--confusion-spectral-start-epoch", "$ConfusionSpectralStartEpoch",
        "--mutual-channel-loss-weight", "$MutualChannelLossWeight",
        "--mutual-channel-top-k", "$MutualChannelTopK",
        "--mutual-channel-diversity-weight", "$MutualChannelDiversityWeight",
        "--mutual-channel-start-epoch", "$MutualChannelStartEpoch",
        "--complement-entropy-loss-weight", "$ComplementEntropyLossWeight",
        "--complement-entropy-classes", "$ComplementEntropyClasses",
        "--complement-entropy-start-epoch", "$ComplementEntropyStartEpoch",
        "--foreground-consistency-loss-weight", "0.025",
        "--foreground-consistency-margin", "0.07",
        "--border-attention-suppression-loss-weight", "$BorderAttentionSuppressionLossWeight",
        "--border-attention-suppression-frame-width", "$BorderAttentionSuppressionFrameWidth",
        "--border-attention-suppression-bbox-band", "$BorderAttentionSuppressionBboxBand",
        "--border-attention-suppression-bbox-weight", "$BorderAttentionSuppressionBboxWeight",
        "--border-attention-suppression-temperature", "$BorderAttentionSuppressionTemperature",
        "--border-attention-suppression-classes", "$BorderAttentionSuppressionClasses",
        "--border-attention-suppression-start-epoch", "$BorderAttentionSuppressionStartEpoch",
        "--register-attention-alignment-loss-weight", "$RegisterAttentionAlignmentLossWeight",
        "--register-attention-alignment-classes", "$RegisterAttentionAlignmentClasses",
        "--register-attention-alignment-bbox-margin-ratio", "$RegisterAttentionAlignmentBboxMarginRatio",
        "--register-attention-alignment-agreement-weight", "$RegisterAttentionAlignmentAgreementWeight",
        "--register-attention-alignment-foreground-weight", "$RegisterAttentionAlignmentForegroundWeight",
        "--register-attention-alignment-start-epoch", "$RegisterAttentionAlignmentStartEpoch",
        "--background-counterfactual-consistency-weight", "$BackgroundCounterfactualConsistencyWeight",
        "--background-counterfactual-probability", "$BackgroundCounterfactualProbability",
        "--background-counterfactual-mode", "$BackgroundCounterfactualMode",
        "--background-counterfactual-margin", "$BackgroundCounterfactualMargin",
        "--background-counterfactual-blur-kernel", "$BackgroundCounterfactualBlurKernel",
        "--background-counterfactual-temperature", "$BackgroundCounterfactualTemperature",
        "--background-focus-suppression-loss-weight", "$BackgroundFocusSuppressionLossWeight",
        "--background-focus-suppression-probability", "$BackgroundFocusSuppressionProbability",
        "--background-focus-suppression-focus-class", "$BackgroundFocusSuppressionFocusClass",
        "--background-focus-suppression-negative-classes", "$BackgroundFocusSuppressionNegativeClasses",
        "--background-focus-suppression-margin", "$BackgroundFocusSuppressionMargin",
        "--background-focus-suppression-min-probability", "$BackgroundFocusSuppressionMinProbability",
        "--background-focus-suppression-probability-power", "$BackgroundFocusSuppressionProbabilityPower",
        "--source-context-focus-suppression-loss-weight", "$SourceContextFocusSuppressionLossWeight",
        "--source-context-focus-suppression-probability", "$SourceContextFocusSuppressionProbability",
        "--source-context-focus-suppression-focus-class", "$SourceContextFocusSuppressionFocusClass",
        "--source-context-focus-suppression-negative-classes", "$SourceContextFocusSuppressionNegativeClasses",
        "--source-context-focus-suppression-margin", "$SourceContextFocusSuppressionMargin",
        "--source-context-focus-suppression-min-probability", "$SourceContextFocusSuppressionMinProbability",
        "--source-context-focus-suppression-probability-power", "$SourceContextFocusSuppressionProbabilityPower",
        "--surface-counterfactual-consistency-weight", "$SurfaceCounterfactualConsistencyWeight",
        "--surface-counterfactual-probability", "$SurfaceCounterfactualProbability",
        "--surface-counterfactual-mode", "$SurfaceCounterfactualMode",
        "--surface-counterfactual-strength", "$SurfaceCounterfactualStrength",
        "--surface-counterfactual-blur-kernel", "$SurfaceCounterfactualBlurKernel",
        "--surface-counterfactual-temperature", "$SurfaceCounterfactualTemperature",
        "--surface-amplified-supervised-loss-weight", "$SurfaceAmplifiedSupervisedLossWeight",
        "--surface-amplified-boundary-margin-loss-weight", "$SurfaceAmplifiedBoundaryMarginLossWeight",
        "--surface-amplified-probability", "$SurfaceAmplifiedProbability",
        "--surface-amplified-mode", "$SurfaceAmplifiedMode",
        "--surface-amplified-strength", "$SurfaceAmplifiedStrength",
        "--surface-amplified-blur-kernel", "$SurfaceAmplifiedBlurKernel",
        "--surface-amplified-boundary-pairs", "$SurfaceAmplifiedBoundaryPairs",
        "--surface-amplified-boundary-margin", "$SurfaceAmplifiedBoundaryMargin",
        "--paired-view-supervised-loss-weight", "$PairedViewSupervisedLossWeight",
        "--paired-view-consistency-weight", "$PairedViewConsistencyWeight",
        "--paired-view-feature-consistency-weight", "$PairedViewFeatureConsistencyWeight",
        "--paired-view-fusion-loss-weight", "$PairedViewFusionLossWeight",
        "--paired-view-fusion-consistency-weight", "$PairedViewFusionConsistencyWeight",
        "--paired-view-temperature", "$PairedViewTemperature",
        "--paired-view-feature-source", "$PairedViewFeatureSource",
        "--masked-reconstruction-loss-weight", "$MaskedReconstructionLossWeight",
        "--masked-reconstruction-mask-ratio", "$MaskedReconstructionMaskRatio",
        "--masked-reconstruction-foreground-weight", "$MaskedReconstructionForegroundWeight",
        "--masked-reconstruction-detail-weight", "$MaskedReconstructionDetailWeight",
        "--masked-reconstruction-bbox-weight", "$MaskedReconstructionBboxWeight",
        "--masked-reconstruction-bbox-margin-ratio", "$MaskedReconstructionBboxMarginRatio",
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
        "--source-context-aux-loss-weight", "$SourceContextAuxLossWeight",
        "--source-context-aux-classification-weight", "$SourceContextAuxClassificationWeight",
        "--source-context-aux-consistency-weight", "$SourceContextAuxConsistencyWeight",
        "--source-context-aux-bbox-margin-ratio", "$SourceContextAuxBboxMarginRatio",
        "--source-context-aux-attention-temperature", "$SourceContextAuxAttentionTemperature",
        "--elr-loss-weight", "$ElrLossWeight",
        "--elr-beta", "$ElrBeta",
        "--elr-start-epoch", "$ElrStartEpoch",
        "--self-adaptive-target-loss-weight", "$SelfAdaptiveTargetLossWeight",
        "--self-adaptive-target-beta", "$SelfAdaptiveTargetBeta",
        "--self-adaptive-target-start-epoch", "$SelfAdaptiveTargetStartEpoch",
        "--self-adaptive-target-hard-weight", "$SelfAdaptiveTargetHardWeight",
        "--self-adaptive-target-confidence-power", "$SelfAdaptiveTargetConfidencePower",
        "--self-adaptive-target-min-confidence", "$SelfAdaptiveTargetMinConfidence",
        "--rdrop-loss-weight", "$RDropLossWeight",
        "--rdrop-temperature", "$RDropTemperature",
        "--augmix-consistency-loss-weight", "$AugmixConsistencyLossWeight",
        "--augmix-consistency-probability", "$AugmixConsistencyProbability",
        "--augmix-consistency-severity", "$AugmixConsistencySeverity",
        "--augmix-consistency-width", "$AugmixConsistencyWidth",
        "--augmix-consistency-depth", "$AugmixConsistencyDepth",
        "--augmix-consistency-alpha", "$AugmixConsistencyAlpha",
        "--augmix-consistency-temperature", "$AugmixConsistencyTemperature",
        "--illumination-consistency-loss-weight", "$IlluminationConsistencyLossWeight",
        "--illumination-consistency-probability", "$IlluminationConsistencyProbability",
        "--illumination-consistency-brightness", "$IlluminationConsistencyBrightness",
        "--illumination-consistency-contrast", "$IlluminationConsistencyContrast",
        "--illumination-consistency-gamma", "$IlluminationConsistencyGamma",
        "--illumination-consistency-temperature", "$IlluminationConsistencyTemperature",
        "--foreground-chroma-consistency-loss-weight", "$ForegroundChromaConsistencyLossWeight",
        "--foreground-chroma-consistency-probability", "$ForegroundChromaConsistencyProbability",
        "--foreground-chroma-consistency-saturation-delta", "$ForegroundChromaConsistencySaturationDelta",
        "--foreground-chroma-consistency-hue-delta", "$ForegroundChromaConsistencyHueDelta",
        "--foreground-chroma-consistency-bbox-margin-ratio", "$ForegroundChromaConsistencyBboxMarginRatio",
        "--foreground-chroma-consistency-temperature", "$ForegroundChromaConsistencyTemperature",
        "--foreground-chroma-consistency-classes", "$ForegroundChromaConsistencyClasses",
        "--friendly-adversarial-loss-weight", "$FriendlyAdversarialLossWeight",
        "--friendly-adversarial-epsilon", "$FriendlyAdversarialEpsilon",
        "--friendly-adversarial-step-size", "$FriendlyAdversarialStepSize",
        "--friendly-adversarial-steps", "$FriendlyAdversarialSteps",
        "--friendly-adversarial-bbox-erode-ratio", "$FriendlyAdversarialBboxErodeRatio",
        "--friendly-adversarial-focus-class", "$FriendlyAdversarialFocusClass",
        "--friendly-adversarial-negative-classes", "$FriendlyAdversarialNegativeClasses",
        "--friendly-adversarial-max-per-direction", "$FriendlyAdversarialMaxPerDirection",
        "--friendly-adversarial-start-epoch", "$FriendlyAdversarialStartEpoch",
        "--semantic-attribute-loss-weight", "$SemanticAttributeLossWeight",
        "--semantic-attribute-specs", "$SemanticAttributeSpecs",
        "--confusion-pair-mixup-loss-weight", "$ConfusionPairMixupLossWeight",
        "--confusion-pair-mixup-alpha", "$ConfusionPairMixupAlpha",
        "--confusion-pair-mixup-pairs", "$ConfusionPairMixupPairs",
        "--confusion-pair-mixup-max-pairs", "$ConfusionPairMixupMaxPairs",
        "--confusion-pair-mixup-start-epoch", "$ConfusionPairMixupStartEpoch",
        "--foreground-snapmix-loss-weight", "$ForegroundSnapmixLossWeight",
        "--foreground-snapmix-probability", "$ForegroundSnapmixProbability",
        "--foreground-snapmix-alpha", "$ForegroundSnapmixAlpha",
        "--foreground-snapmix-pairs", "$ForegroundSnapmixPairs",
        "--foreground-snapmix-min-area-ratio", "$ForegroundSnapmixMinAreaRatio",
        "--foreground-snapmix-max-area-ratio", "$ForegroundSnapmixMaxAreaRatio",
        "--foreground-snapmix-bbox-margin-ratio", "$ForegroundSnapmixBboxMarginRatio",
        "--foreground-counterexample-mix-loss-weight", "$ForegroundCounterexampleMixLossWeight",
        "--foreground-counterexample-mix-probability", "$ForegroundCounterexampleMixProbability",
        "--foreground-counterexample-mix-source-class", "$ForegroundCounterexampleMixSourceClass",
        "--foreground-counterexample-mix-target-classes", "$ForegroundCounterexampleMixTargetClasses",
        "--foreground-counterexample-mix-alpha", "$ForegroundCounterexampleMixAlpha",
        "--foreground-counterexample-mix-min-area-ratio", "$ForegroundCounterexampleMixMinAreaRatio",
        "--foreground-counterexample-mix-max-area-ratio", "$ForegroundCounterexampleMixMaxAreaRatio",
        "--foreground-counterexample-mix-bbox-margin-ratio", "$ForegroundCounterexampleMixBboxMarginRatio",
        "--foreground-counterexample-mix-start-epoch", "$ForegroundCounterexampleMixStartEpoch",
        "--dcl-region-shuffle-loss-weight", "$DclRegionShuffleLossWeight",
        "--dcl-region-shuffle-probability", "$DclRegionShuffleProbability",
        "--dcl-region-shuffle-grid-size", "$DclRegionShuffleGridSize",
        "--dcl-region-shuffle-bbox-margin-ratio", "$DclRegionShuffleBboxMarginRatio",
        "--dcl-region-shuffle-classes", "$DclRegionShuffleClasses",
        "--dcl-region-shuffle-start-epoch", "$DclRegionShuffleStartEpoch",
        "--quantized-label-cpu-loss-weight", "$QuantizedLabelCpuLossWeight",
        "--quantized-label-cpu-classes", "$QuantizedLabelCpuClasses",
        "--quantized-label-cpu-min-prior", "$QuantizedLabelCpuMinPrior",
        "--quantized-label-cpu-max-prior", "$QuantizedLabelCpuMaxPrior",
        "--quantized-label-cpu-negative-weight", "$QuantizedLabelCpuNegativeWeight",
        "--quantized-label-cpu-non-negative-beta", "$QuantizedLabelCpuNonNegativeBeta",
        "--quantized-label-cpu-start-epoch", "$QuantizedLabelCpuStartEpoch",
        "--self-paced-loss-weight", "$SelfPacedLossWeight",
        "--self-paced-loss-percentile", "$SelfPacedLossPercentile",
        "--self-paced-loss-gamma", "$SelfPacedLossGamma",
        "--self-paced-loss-min-weight", "$SelfPacedLossMinWeight",
        "--self-paced-loss-start-epoch", "$SelfPacedLossStartEpoch",
        "--cyflod-loss-damping-weight", "$CyflodLossDampingWeight",
        "--cyflod-loss-damping-delta", "$CyflodLossDampingDelta",
        "--cyflod-loss-damping-cycle-epochs", "$CyflodLossDampingCycleEpochs",
        "--cyflod-loss-damping-min-weight", "$CyflodLossDampingMinWeight",
        "--cyflod-loss-damping-start-epoch", "$CyflodLossDampingStartEpoch",
        "--sample-weight-factor", "$SampleWeightFactor",
        "--sample-weight-max", "$SampleWeightMax",
        "--targeted-margin-loss-weight", "$TargetedMarginLossWeight",
        "--targeted-margin-default-margin", "$TargetedMarginDefaultMargin",
        "--targeted-margin-default-weight", "$TargetedMarginDefaultWeight",
        "--targeted-margin-max-weight", "$TargetedMarginMaxWeight",
        "--focus-neighbor-binary-loss-weight", "$FocusNeighborBinaryLossWeight",
        "--focus-neighbor-binary-focus-class", "$FocusNeighborBinaryFocusClass",
        "--focus-neighbor-binary-neighbor-classes", "$FocusNeighborBinaryNeighborClasses",
        "--focus-neighbor-binary-default-weight", "$FocusNeighborBinaryDefaultWeight",
        "--focus-neighbor-binary-max-weight", "$FocusNeighborBinaryMaxWeight",
        "--focused-false-positive-margin-loss-weight", "$FocusedFalsePositiveMarginLossWeight",
        "--focused-false-positive-class", "$FocusedFalsePositiveClass",
        "--focused-false-positive-negative-classes", "$FocusedFalsePositiveNegativeClasses",
        "--focused-false-positive-margin", "$FocusedFalsePositiveMargin",
        "--focused-false-positive-min-probability", "$FocusedFalsePositiveMinProbability",
        "--focused-false-positive-probability-power", "$FocusedFalsePositiveProbabilityPower",
        "--teacher-focus-margin-loss-weight", "$TeacherFocusMarginLossWeight",
        "--teacher-focus-margin-focus-class", "$TeacherFocusMarginFocusClass",
        "--teacher-focus-margin-negative-classes", "$TeacherFocusMarginNegativeClasses",
        "--teacher-focus-margin-teacher-max-probability", "$TeacherFocusMarginTeacherMaxProbability",
        "--teacher-focus-margin-margin", "$TeacherFocusMarginMargin",
        "--teacher-focus-margin-min-probability", "$TeacherFocusMarginMinProbability",
        "--teacher-focus-margin-probability-power", "$TeacherFocusMarginProbabilityPower",
        "--teacher-focus-binary-loss-weight", "$TeacherFocusBinaryLossWeight",
        "--teacher-focus-binary-focus-class", "$TeacherFocusBinaryFocusClass",
        "--teacher-focus-binary-classes", "$TeacherFocusBinaryClasses",
        "--teacher-focus-binary-teacher-min-confidence", "$TeacherFocusBinaryTeacherMinConfidence",
        "--teacher-focus-binary-error-power", "$TeacherFocusBinaryErrorPower",
        "--teacher-focus-binary-hard-target-blend", "$TeacherFocusBinaryHardTargetBlend",
        "--register-diversity-loss-weight", "$RegisterDiversityLossWeight",
        "--pairwise-margin-loss-weight", "0.04",
        "--resize-mode", "pad",
        "--train-scale-min", "0.88",
        "--train-scale-crop-probability", "0.35",
        "--brightness", "0.04",
        "--contrast", "0.04",
        "--saturation", "0.02",
        "--hue", "0.01",
        "--illumination-normalization",
        "--illumination-normalization-strength", "0.35",
        "--foreground-crop-mode", "$ForegroundCropMode",
        "--foreground-crop-probability", "$ForegroundCropProbability",
        "--foreground-crop-margin-ratio", "$ForegroundCropMarginRatio",
        "--foreground-crop-min-mask-area-ratio", "$ForegroundCropMinMaskAreaRatio",
        "--foreground-crop-max-mask-area-ratio", "$ForegroundCropMaxMaskAreaRatio",
        "--foreground-crop-max-crop-area-ratio", "$ForegroundCropMaxCropAreaRatio",
        "--background-suppression-mode", "$BackgroundSuppressionMode",
        "--background-suppression-probability", "$BackgroundSuppressionProbability",
        "--background-suppression-margin", "$BackgroundSuppressionMargin",
        "--background-suppression-blur-radius", "$BackgroundSuppressionBlurRadius",
        "--surface-detail-amplification-mode", "$SurfaceDetailAmplificationMode",
        "--surface-detail-amplification-probability", "$SurfaceDetailAmplificationProbability",
        "--surface-detail-amplification-strength", "$SurfaceDetailAmplificationStrength",
        "--surface-detail-amplification-blur-radius", "$SurfaceDetailAmplificationBlurRadius",
        "--surface-detail-amplification-foreground-weight", "$SurfaceDetailAmplificationForegroundWeight",
        "--foreground-background-mix-probability", "$ForegroundBackgroundMixProbability",
        "--foreground-background-mix-margin", "$ForegroundBackgroundMixMargin",
        "--foreground-background-mix-min-foreground-fraction", "$ForegroundBackgroundMixMinForegroundFraction",
        "--foreground-background-mix-max-foreground-fraction", "$ForegroundBackgroundMixMaxForegroundFraction",
        "--foreground-background-mix-softness", "$ForegroundBackgroundMixSoftness",
        "--foreground-background-mix-mask-source", "$ForegroundBackgroundMixMaskSource",
        "--local-exposure-probability", "$LocalExposureProbability",
        "--local-exposure-strength", "$LocalExposureStrength",
        "--obstacle-probability", "$ObstacleProbability",
        "--obstacle-max-area", "$ObstacleMaxArea",
        "--randaugment-num-ops", "$RandAugmentNumOps",
        "--randaugment-magnitude", "$RandAugmentMagnitude",
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

    if ($DisableBalancedEpochSampling) {
        $TrainArgs += @("--disable-balanced-epoch-sampling")
    }

    if (-not [string]::IsNullOrWhiteSpace($ClassConditionalAugmentationScales)) {
        $TrainArgs += @(
            "--class-conditional-augmentation-scales",
            $ClassConditionalAugmentationScales
        )
    }

    if (-not $TeacherFocusMarginRequireAgreement) {
        $TrainArgs += @("--disable-teacher-focus-margin-require-agreement")
    }
    if (-not $TeacherFocusBinaryRequireAgreement) {
        $TrainArgs += @("--disable-teacher-focus-binary-require-agreement")
    }
    if (-not $TeacherNonTargetDistillationRequireAgreement) {
        $TrainArgs += @("--disable-teacher-non-target-distillation-require-agreement")
    }
    if (-not $TeacherPairwiseMarginRequireAgreement) {
        $TrainArgs += @("--disable-teacher-pairwise-margin-require-agreement")
    }
    if (-not $TeacherGuidedContrastiveRequireAgreement) {
        $TrainArgs += @("--disable-teacher-guided-contrastive-require-agreement")
    }
    if (-not $TeacherGuidedContrastiveClassBalanced) {
        $TrainArgs += @("--disable-teacher-guided-contrastive-class-balanced")
    }
    if ($BoundaryCenterRequireAgreement) {
        $TrainArgs += @("--boundary-center-require-agreement")
    }

    if (-not [string]::IsNullOrWhiteSpace($DistillationTeacherCsv)) {
        $TrainArgs += @(
            "--distillation-teacher-csv", "$DistillationTeacherCsv",
            "--distillation-weight", "$DistillationWeight",
            "--distillation-temperature", "$DistillationTemperature",
            "--distillation-focus-class-index", "$DistillationFocusClassIndex",
            "--distillation-focus-class-weight", "$DistillationFocusClassWeight",
            "--teacher-non-target-distillation-loss-weight", "$TeacherNonTargetDistillationLossWeight",
            "--teacher-non-target-distillation-classes", "$TeacherNonTargetDistillationClasses",
            "--teacher-non-target-distillation-temperature", "$TeacherNonTargetDistillationTemperature",
            "--teacher-non-target-distillation-teacher-min-confidence", "$TeacherNonTargetDistillationTeacherMinConfidence"
        )
    }
    if ($TeacherPairwiseMarginLossWeight -gt 0.0) {
        $TrainArgs += @(
            "--teacher-pairwise-margin-loss-weight", "$TeacherPairwiseMarginLossWeight",
            "--teacher-pairwise-margin-teacher-mass-threshold", "$TeacherPairwiseMarginTeacherMassThreshold",
            "--teacher-pairwise-margin-error-power", "$TeacherPairwiseMarginErrorPower",
            "--teacher-pairwise-margin-hard-target-blend", "$TeacherPairwiseMarginHardTargetBlend"
        )
    }
    if (-not [string]::IsNullOrWhiteSpace($TeacherFeatureNpz)) {
        if (-not (Test-Path -LiteralPath $TeacherFeatureNpz)) {
            throw "Khong tim thay teacher feature NPZ: $TeacherFeatureNpz"
        }
        $TrainArgs += @(
            "--teacher-feature-npz", "$TeacherFeatureNpz",
            "--teacher-feature-rkd-loss-weight", "$TeacherFeatureRkdLossWeight",
            "--teacher-feature-rkd-distance-weight", "$TeacherFeatureRkdDistanceWeight",
            "--teacher-feature-rkd-angle-weight", "$TeacherFeatureRkdAngleWeight",
            "--teacher-feature-rkd-source", "$TeacherFeatureRkdSource",
            "--teacher-feature-rkd-pair-mode", "$TeacherFeatureRkdPairMode",
            "--teacher-feature-rkd-pairs", "$TeacherFeatureRkdPairs",
            "--teacher-feature-contrastive-loss-weight", "$TeacherFeatureContrastiveLossWeight",
            "--teacher-feature-contrastive-temperature", "$TeacherFeatureContrastiveTemperature",
            "--teacher-feature-contrastive-projection-dim", "$TeacherFeatureContrastiveProjectionDim",
            "--teacher-feature-contrastive-source", "$TeacherFeatureContrastiveSource",
            "--teacher-feature-contrastive-pair-mode", "$TeacherFeatureContrastivePairMode",
            "--teacher-feature-contrastive-pairs", "$TeacherFeatureContrastivePairs",
            "--teacher-feature-contrastive-start-epoch", "$TeacherFeatureContrastiveStartEpoch"
        )
        if ($TeacherFeatureContrastiveUseProjectionAdapter) {
            $TrainArgs += @("--teacher-feature-contrastive-use-projection-adapter")
        }
        $TrainArgs += @("--teacher-feature-contrastive-adapter-dropout", "$TeacherFeatureContrastiveAdapterDropout")
    }
    elseif ($TeacherFeatureRkdLossWeight -gt 0.0 -or $TeacherFeatureContrastiveLossWeight -gt 0.0) {
        throw "Teacher feature losses > 0 yeu cau TeacherFeatureNpz."
    }

    if ($ClassificationMlpHead) {
        $TrainArgs += @("--classification-mlp-head")
    }

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
    if (-not [string]::IsNullOrWhiteSpace($ClassificationFolderYoloData)) {
        $TrainArgs += @("--classification-folder-yolo-data", "$ClassificationFolderYoloData")
    }
    if (-not [string]::IsNullOrWhiteSpace($AuxiliaryTrainData)) {
        $TrainArgs += @(
            "--auxiliary-train-data", "$AuxiliaryTrainData",
            "--auxiliary-train-weight", "$AuxiliaryTrainWeight"
        )
        if (-not [string]::IsNullOrWhiteSpace($AuxiliaryTrainClassificationFolderYoloData)) {
            $TrainArgs += @(
                "--auxiliary-train-classification-folder-yolo-data",
                "$AuxiliaryTrainClassificationFolderYoloData"
            )
        }
    }
    if ($PairedViewTrain) {
        $TrainArgs += @("--paired-view-train")
    }
    if ($ClassificationSourceContext) {
        $TrainArgs += @("--classification-source-context")
    }
    if ($ClassificationSourceContextAux) {
        $TrainArgs += @("--classification-source-context-aux")
    }
    if ($MaxTrainBatches -gt 0) {
        $TrainArgs += @("--max-train-batches", "$MaxTrainBatches")
    }
    if ($MaxValBatches -gt 0) {
        $TrainArgs += @("--max-val-batches", "$MaxValBatches")
    }
    if (-not [string]::IsNullOrWhiteSpace($ClassLossMultipliers)) {
        $TrainArgs += @("--class-loss-multipliers", $ClassLossMultipliers)
    }
    if ($DataCartography) {
        $TrainArgs += @("--data-cartography")
        if (-not [string]::IsNullOrWhiteSpace($DataCartographyOutput)) {
            $TrainArgs += @("--data-cartography-output", $DataCartographyOutput)
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($TrainableModulePrefixes)) {
        $TrainArgs += @("--trainable-module-prefixes", $TrainableModulePrefixes)
    }
    if ($SelfPacedLossClassBalanced) {
        $TrainArgs += @("--self-paced-loss-class-balanced")
    }
    if (-not [string]::IsNullOrWhiteSpace($SampleWeightManifest)) {
        if (-not (Test-Path -LiteralPath $SampleWeightManifest)) {
            throw "Khong tim thay sample weight manifest: $SampleWeightManifest"
        }
        $TrainArgs += @("--sample-weight-manifest", $SampleWeightManifest)
    }
    if (-not [string]::IsNullOrWhiteSpace($QualityGroupManifest)) {
        if (-not (Test-Path -LiteralPath $QualityGroupManifest)) {
            throw "Khong tim thay quality-group manifest: $QualityGroupManifest"
        }
        $TrainArgs += @(
            "--quality-group-manifest", $QualityGroupManifest,
            "--group-dro-loss-weight", "$GroupDroLossWeight",
            "--group-dro-temperature", "$GroupDroTemperature",
            "--group-dro-min-samples", "$GroupDroMinSamples"
        )
    }
    if (-not [string]::IsNullOrWhiteSpace($HardSampleManifest)) {
        if (-not (Test-Path -LiteralPath $HardSampleManifest)) {
            throw "Khong tim thay hard sample manifest: $HardSampleManifest"
        }
        $TrainArgs += @(
            "--hard-sample-manifest", $HardSampleManifest,
            "--hard-sample-repeat-factor", "$HardSampleRepeatFactor"
        )
    }
    if (-not [string]::IsNullOrWhiteSpace($AmbiguousSoftTargetManifest)) {
        if (-not (Test-Path -LiteralPath $AmbiguousSoftTargetManifest)) {
            throw "Khong tim thay ambiguous soft-target manifest: $AmbiguousSoftTargetManifest"
        }
        $TrainArgs += @(
            "--ambiguous-soft-target-manifest", $AmbiguousSoftTargetManifest,
            "--ambiguous-soft-target-alpha", "$AmbiguousSoftTargetAlpha"
        )
    }
    if (-not [string]::IsNullOrWhiteSpace($TargetedMarginManifest)) {
        if (-not (Test-Path -LiteralPath $TargetedMarginManifest)) {
            throw "Khong tim thay targeted-margin manifest: $TargetedMarginManifest"
        }
        $TrainArgs += @("--targeted-margin-manifest", $TargetedMarginManifest)
    }
    if (-not [string]::IsNullOrWhiteSpace($FocusNeighborBinaryManifest)) {
        if (-not (Test-Path -LiteralPath $FocusNeighborBinaryManifest)) {
            throw "Khong tim thay focus-neighbor binary manifest: $FocusNeighborBinaryManifest"
        }
        $TrainArgs += @("--focus-neighbor-binary-manifest", $FocusNeighborBinaryManifest)
    }
    if ($Sam) {
        $TrainArgs += @("--sam", "--sam-rho", "$SamRho")
        if ($SamAdaptive) {
            $TrainArgs += @("--sam-adaptive")
        }
    }
    if ($ColorStatFusion) {
        $TrainArgs += @(
            "--color-stat-fusion",
            "--color-stat-fusion-dropout", "$ColorStatFusionDropout"
        )
    }
    if ($DefectStatFusion) {
        $TrainArgs += @(
            "--defect-stat-fusion",
            "--defect-stat-fusion-dropout", "$DefectStatFusionDropout"
        )
    }
    if ($ForegroundSurfaceFusion) {
        $TrainArgs += @(
            "--foreground-surface-fusion",
            "--foreground-surface-fusion-dropout", "$ForegroundSurfaceFusionDropout"
        )
    }
    if ($ForegroundSurfacePairwiseHead) {
        $TrainArgs += @(
            "--foreground-surface-pairwise-head",
            "--foreground-surface-pairwise-pairs", "$ForegroundSurfacePairwisePairs",
            "--foreground-surface-pairwise-logit-scale", "$ForegroundSurfacePairwiseLogitScale",
            "--foreground-surface-pairwise-dropout", "$ForegroundSurfacePairwiseDropout",
            "--foreground-surface-pairwise-route-max-probability-margin",
            "$ForegroundSurfacePairwiseRouteMaxProbabilityMargin"
        )
        if (-not $ForegroundSurfacePairwiseRouting) {
            $TrainArgs += @("--disable-foreground-surface-pairwise-routing")
        }
    }
    if ($InteriorBoundaryPairwiseHead) {
        $TrainArgs += @(
            "--interior-boundary-pairwise-head",
            "--interior-boundary-pairwise-pairs", "$InteriorBoundaryPairwisePairs",
            "--interior-boundary-pairwise-logit-scale", "$InteriorBoundaryPairwiseLogitScale",
            "--interior-boundary-pairwise-dropout", "$InteriorBoundaryPairwiseDropout",
            "--interior-boundary-pairwise-hidden-dim", "$InteriorBoundaryPairwiseHiddenDim",
            "--interior-boundary-pairwise-erode-kernel", "$InteriorBoundaryPairwiseErodeKernel",
            "--interior-boundary-pairwise-route-max-probability-margin",
            "$InteriorBoundaryPairwiseRouteMaxProbabilityMargin"
        )
        if (-not $InteriorBoundaryPairwiseRouting) {
            $TrainArgs += @("--disable-interior-boundary-pairwise-routing")
        }
    }
    if ($BBoxSpatialFusion) {
        $TrainArgs += @(
            "--bbox-spatial-fusion",
            "--bbox-spatial-fusion-hidden-dim", "$BBoxSpatialFusionHiddenDim",
            "--bbox-spatial-fusion-dropout", "$BBoxSpatialFusionDropout",
            "--bbox-spatial-fusion-logit-scale", "$BBoxSpatialFusionLogitScale"
        )
    }
    if ($PatchObjectnessGuidedHead) {
        $TrainArgs += @(
            "--patch-objectness-guided-head",
            "--patch-objectness-hidden-dim", "$PatchObjectnessHiddenDim",
            "--patch-objectness-dropout", "$PatchObjectnessDropout",
            "--patch-objectness-logit-scale", "$PatchObjectnessLogitScale",
            "--patch-objectness-temperature", "$PatchObjectnessTemperature"
        )
    }
    if (-not $BboxTokenLabelDetachClassifier) {
        $TrainArgs += @("--no-bbox-token-label-detach-classifier")
    }
    if ($PatchEvidenceRouterHead) {
        $TrainArgs += @(
            "--patch-evidence-router-head",
            "--patch-evidence-router-pair", "$PatchEvidenceRouterPair",
            "--patch-evidence-router-hidden-dim", "$PatchEvidenceRouterHiddenDim",
            "--patch-evidence-router-top-k", "$PatchEvidenceRouterTopK",
            "--patch-evidence-router-bbox-weight", "$PatchEvidenceRouterBboxWeight",
            "--patch-evidence-router-dropout", "$PatchEvidenceRouterDropout",
            "--patch-evidence-router-logit-scale", "$PatchEvidenceRouterLogitScale",
            "--patch-evidence-router-margin-prior-mode", "$PatchEvidenceRouterMarginPriorMode",
            "--patch-evidence-router-margin-prior-scale", "$PatchEvidenceRouterMarginPriorScale",
            "--patch-evidence-router-route-max-probability-margin", "$PatchEvidenceRouterRouteMaxProbabilityMargin",
            "--patch-evidence-router-route-min-pair-probability", "$PatchEvidenceRouterRouteMinPairProbability"
        )
    if ($PatchEvidenceRouterSummaryStats) {
        $TrainArgs += @("--patch-evidence-router-summary-stats")
    }
    if (-not $PatchEvidenceRouterRouting) {
        $TrainArgs += @("--disable-patch-evidence-router-routing")
    }
}
if (-not [string]::IsNullOrWhiteSpace($PatchEvidenceRouterTeacherCsv)) {
    $TrainArgs += @("--patch-evidence-router-teacher-csv", "$PatchEvidenceRouterTeacherCsv")
}
if (-not [string]::IsNullOrWhiteSpace($PatchEvidenceLinearVerifierJson)) {
    $TrainArgs += @(
        "--patch-evidence-linear-verifier-json", "$PatchEvidenceLinearVerifierJson",
        "--patch-evidence-linear-verifier-pair", "$PatchEvidenceLinearVerifierPair",
        "--patch-evidence-linear-verifier-min-pair-probability", "$PatchEvidenceLinearVerifierMinPairProbability",
        "--patch-evidence-linear-verifier-max-pair-margin", "$PatchEvidenceLinearVerifierMaxPairMargin",
        "--patch-evidence-linear-verifier-confidence-threshold", "$PatchEvidenceLinearVerifierConfidenceThreshold",
        "--patch-evidence-linear-verifier-logit-boost", "$PatchEvidenceLinearVerifierLogitBoost",
        "--patch-evidence-linear-verifier-protect-right-min-probability", "$PatchEvidenceLinearVerifierProtectRightMinProbability",
        "--patch-evidence-linear-verifier-training-soft-logit-scale", "$PatchEvidenceLinearVerifierTrainingSoftLogitScale",
        "--patch-evidence-linear-verifier-training-soft-gate-temperature", "$PatchEvidenceLinearVerifierTrainingSoftGateTemperature"
    )
    if ($PatchEvidenceLinearVerifierTrainingSoftAdjustment) {
        $TrainArgs += @("--patch-evidence-linear-verifier-training-soft-adjustment")
    }
}
if ($BBoxPriorPatchContextHead) {
        $TrainArgs += @(
            "--bbox-prior-patch-context-head",
            "--bbox-prior-patch-context-hidden-dim", "$BBoxPriorPatchContextHiddenDim",
            "--bbox-prior-patch-context-dropout", "$BBoxPriorPatchContextDropout",
            "--bbox-prior-patch-context-logit-scale", "$BBoxPriorPatchContextLogitScale",
            "--bbox-prior-patch-context-temperature", "$BBoxPriorPatchContextTemperature"
        )
    }
    if ($SourceContextFeatureFusion) {
        $TrainArgs += @(
            "--source-context-feature-fusion",
            "--source-context-fusion-hidden-dim", "$SourceContextFusionHiddenDim",
            "--source-context-fusion-dropout", "$SourceContextFusionDropout",
            "--source-context-fusion-logit-scale", "$SourceContextFusionLogitScale",
            "--source-context-fusion-gate-bias", "$SourceContextFusionGateBias"
        )
    }
    if ($PairedViewFeatureFusion) {
        $TrainArgs += @(
            "--paired-view-feature-fusion",
            "--paired-view-fusion-hidden-dim", "$PairedViewFusionHiddenDim",
            "--paired-view-fusion-dropout", "$PairedViewFusionDropout",
            "--paired-view-fusion-logit-scale", "$PairedViewFusionLogitScale",
            "--paired-view-fusion-gate-bias", "$PairedViewFusionGateBias"
        )
    }
    if ($TopKReassessmentHead) {
        $TrainArgs += @(
            "--topk-reassessment-head",
            "--topk-reassessment-top-k", "$TopKReassessmentTopK",
            "--topk-reassessment-hidden-dim", "$TopKReassessmentHiddenDim",
            "--topk-reassessment-dropout", "$TopKReassessmentDropout",
            "--topk-reassessment-logit-scale", "$TopKReassessmentLogitScale",
            "--topk-reassessment-route-pairs", "$TopKReassessmentRoutePairs",
            "--topk-reassessment-route-max-probability-margin",
            "$TopKReassessmentRouteMaxProbabilityMargin",
            "--topk-reassessment-aux-loss-weight", "$TopKReassessmentAuxLossWeight",
            "--topk-reassessment-aux-route-min-weight", "$TopKReassessmentAuxRouteMinWeight"
        )
        if (-not $TopKReassessmentRouting) {
            $TrainArgs += @("--disable-topk-reassessment-routing")
        }
    }
    if ($FocusClassHead) {
        $TrainArgs += @(
            "--focus-class-head",
            "--focus-class-index", "$FocusClassIndex",
            "--focus-class-logit-scale", "$FocusClassLogitScale",
            "--focus-class-dropout", "$FocusClassDropout",
            "--focus-class-route-max-probability-margin", "$FocusClassRouteMaxProbabilityMargin",
            "--focus-class-route-min-probability", "$FocusClassRouteMinProbability"
        )
    if (-not $FocusClassRouting) {
        $TrainArgs += @("--disable-focus-class-routing")
    }
}
if ($ClassIndependentHead) {
    $TrainArgs += @(
        "--class-independent-head",
        "--class-independent-dropout", "$ClassIndependentDropout"
    )
}
    if ($OrdinalMaturityHead) {
        $TrainArgs += @(
            "--ordinal-maturity-head",
            "--ordinal-maturity-classes", "$OrdinalMaturityClasses",
            "--ordinal-maturity-logit-scale", "$OrdinalMaturityLogitScale",
            "--ordinal-maturity-dropout", "$OrdinalMaturityDropout"
        )
    }
    if ($CumulativeOrdinalHead) {
        $TrainArgs += @(
            "--cumulative-ordinal-head",
            "--cumulative-ordinal-classes", "$CumulativeOrdinalClasses",
            "--cumulative-ordinal-logit-scale", "$CumulativeOrdinalLogitScale",
            "--cumulative-ordinal-dropout", "$CumulativeOrdinalDropout"
        )
    }
    if ($BilinearPatchFusion) {
        $TrainArgs += @(
            "--bilinear-patch-fusion",
            "--bilinear-patch-rank", "$BilinearPatchRank",
            "--bilinear-patch-dropout", "$BilinearPatchDropout"
        )
    }
    if ($ComplementaryPatchSuppressionHead) {
        $TrainArgs += @(
            "--complementary-patch-suppression-head",
            "--complementary-patch-suppression-top-k", "$ComplementaryPatchSuppressionTopK",
            "--complementary-patch-suppression-hidden-dim", "$ComplementaryPatchSuppressionHiddenDim",
            "--complementary-patch-suppression-dropout", "$ComplementaryPatchSuppressionDropout",
            "--complementary-patch-suppression-temperature", "$ComplementaryPatchSuppressionTemperature",
            "--complementary-patch-suppression-strength", "$ComplementaryPatchSuppressionStrength",
            "--complementary-patch-suppression-bbox-weight", "$ComplementaryPatchSuppressionBboxWeight",
            "--complementary-patch-suppression-logit-scale", "$ComplementaryPatchSuppressionLogitScale"
        )
    }
    if ($MicroDetailPatchExpert) {
        $TrainArgs += @(
            "--micro-detail-patch-expert",
            "--micro-detail-top-k", "$MicroDetailTopK",
            "--micro-detail-hidden-dim", "$MicroDetailHiddenDim",
            "--micro-detail-dropout", "$MicroDetailDropout",
            "--micro-detail-temperature", "$MicroDetailTemperature",
            "--micro-detail-foreground-power", "$MicroDetailForegroundPower",
            "--micro-detail-logit-scale", "$MicroDetailLogitScale",
            "--micro-detail-route-pairs", "$MicroDetailRoutePairs",
            "--micro-detail-route-max-probability-margin", "$MicroDetailRouteMaxProbabilityMargin"
        )
        if (-not $MicroDetailRouting) {
            $TrainArgs += @("--disable-micro-detail-routing")
        }
    }
    if ($PartTokenLearner) {
        $TrainArgs += @(
            "--part-token-learner",
            "--part-token-count", "$PartTokenCount",
            "--part-token-hidden-dim", "$PartTokenHiddenDim",
            "--part-token-dropout", "$PartTokenDropout",
            "--part-token-temperature", "$PartTokenTemperature",
            "--part-token-foreground-power", "$PartTokenForegroundPower",
            "--part-token-bbox-weight", "$PartTokenBboxWeight",
            "--part-token-logit-scale", "$PartTokenLogitScale",
            "--part-token-route-pairs", "$PartTokenRoutePairs",
            "--part-token-route-max-probability-margin", "$PartTokenRouteMaxProbabilityMargin"
        )
        if (-not $PartTokenRouting) {
            $TrainArgs += @("--disable-part-token-routing")
        }
    }
    if ($PartTokenPairwiseHead) {
        $TrainArgs += @(
            "--part-token-pairwise-head",
            "--part-token-pairwise-pairs", "$PartTokenPairwisePairs",
            "--part-token-pairwise-logit-scale", "$PartTokenPairwiseLogitScale",
            "--part-token-pairwise-dropout", "$PartTokenPairwiseDropout",
            "--part-token-pairwise-route-max-probability-margin",
            "$PartTokenPairwiseRouteMaxProbabilityMargin"
        )
        if (-not $PartTokenPairwiseRouting) {
            $TrainArgs += @("--disable-part-token-pairwise-routing")
        }
    }
    if ($LocalZoomImageExpert) {
        $TrainArgs += @(
            "--local-zoom-image-expert",
            "--local-zoom-crop-size", "$LocalZoomCropSize",
            "--local-zoom-crop-scale", "$LocalZoomCropScale",
            "--local-zoom-score-mode", "$LocalZoomScoreMode",
            "--local-zoom-hidden-dim", "$LocalZoomHiddenDim",
            "--local-zoom-dropout", "$LocalZoomDropout",
            "--local-zoom-logit-scale", "$LocalZoomLogitScale",
            "--local-zoom-route-pairs", "$LocalZoomRoutePairs",
            "--local-zoom-route-max-probability-margin", "$LocalZoomRouteMaxProbabilityMargin"
        )
        if (-not $LocalZoomRouting) {
            $TrainArgs += @("--disable-local-zoom-routing")
        }
    }
    if ($HighFrequencyTextureExpert) {
        $TrainArgs += @(
            "--high-frequency-texture-expert",
            "--high-frequency-texture-hidden-dim", "$HighFrequencyTextureHiddenDim",
            "--high-frequency-texture-dropout", "$HighFrequencyTextureDropout",
            "--high-frequency-texture-analysis-size", "$HighFrequencyTextureAnalysisSize",
            "--high-frequency-texture-logit-scale", "$HighFrequencyTextureLogitScale",
            "--high-frequency-texture-route-pairs", "$HighFrequencyTextureRoutePairs",
            "--high-frequency-texture-route-max-probability-margin",
            "$HighFrequencyTextureRouteMaxProbabilityMargin"
        )
        if (-not $HighFrequencyTextureRouting) {
            $TrainArgs += @("--disable-high-frequency-texture-routing")
        }
    }
    if ($MultiGranularityAuxHeads) {
        $TrainArgs += @(
            "--multi-granularity-aux-heads",
            "--multi-granularity-aux-layers", "$MultiGranularityAuxLayers",
            "--multi-granularity-aux-dropout", "$MultiGranularityAuxDropout"
        )
    }
    if ($MultiGranularityContrastiveLossWeight -gt 0.0) {
        $TrainArgs += @(
            "--multi-granularity-contrastive-temperature", "$MultiGranularityContrastiveTemperature",
            "--multi-granularity-contrastive-pairs", "$MultiGranularityContrastivePairs",
            "--multi-granularity-contrastive-teacher-min-confidence", "$MultiGranularityContrastiveTeacherMinConfidence",
            "--multi-granularity-contrastive-weight-mode", "$MultiGranularityContrastiveWeightMode",
            "--multi-granularity-contrastive-teacher-confidence-power", "$MultiGranularityContrastiveTeacherConfidencePower"
        )
        if (-not $MultiGranularityContrastiveRequireAgreement) {
            $TrainArgs += @("--disable-multi-granularity-contrastive-require-agreement")
        }
    }
    if ($SelfBoostingAttentionHeadEffective) {
        $TrainArgs += @("--self-boosting-attention-head")
    }
    if ($BlockLocalPatchMixer) {
        $TrainArgs += @(
            "--block-local-patch-mixer",
            "--block-local-patch-mixer-layers", "$BlockLocalPatchMixerLayers",
            "--block-local-patch-mixer-dropout", "$BlockLocalPatchMixerDropout",
            "--block-local-patch-mixer-scale", "$BlockLocalPatchMixerScale"
        )
    }
    if ($LocallyEnhancedFfn) {
        $TrainArgs += @(
            "--locally-enhanced-ffn",
            "--locally-enhanced-ffn-layers", "$LocallyEnhancedFfnLayers",
            "--locally-enhanced-ffn-kernel-size", "$LocallyEnhancedFfnKernelSize"
        )
    }
    if ($ConcurrentLocalGlobalCoupling) {
        $TrainArgs += @(
            "--concurrent-local-global-coupling",
            "--concurrent-local-global-layers", "$ConcurrentLocalGlobalLayers",
            "--concurrent-local-global-dim", "$ConcurrentLocalGlobalDim",
            "--concurrent-local-global-kernel-size", "$ConcurrentLocalGlobalKernelSize"
        )
    }
    if ($GatedRelativePositionAttention) {
        $TrainArgs += @(
            "--gated-relative-position-attention",
            "--gated-relative-position-attention-layers", "$GatedRelativePositionAttentionLayers",
            "--gated-relative-position-attention-max-mix", "$GatedRelativePositionAttentionMaxMix",
            "--gated-relative-position-attention-locality-strength", "$GatedRelativePositionAttentionLocalityStrength"
        )
    }
    if ($TokenPruning) {
        $TrainArgs += @("--token-pruning")
    }
    if ($VisualContrastAttention) {
        $TrainArgs += @(
            "--visual-contrast-attention",
            "--visual-contrast-attention-layers", "$VisualContrastAttentionLayers",
            "--visual-contrast-tokens", "$VisualContrastTokens"
        )
    }
    if ($CrossCovarianceAttention) {
        $TrainArgs += @(
            "--cross-covariance-attention",
            "--cross-covariance-attention-layers", "$CrossCovarianceAttentionLayers",
            "--cross-covariance-attention-residual-scale", "$CrossCovarianceAttentionResidualScale"
        )
    }
    if ($DynamicGraphMixer) {
        $TrainArgs += @(
            "--dynamic-graph-mixer",
            "--dynamic-graph-mixer-layers", "$DynamicGraphMixerLayers",
            "--dynamic-graph-mixer-bottleneck-dim", "$DynamicGraphMixerBottleneckDim",
            "--dynamic-graph-mixer-k", "$DynamicGraphMixerK"
        )
    }
    if ($PatchStyleRecalibration) {
        $TrainArgs += @(
            "--patch-style-recalibration",
            "--patch-style-recalibration-layers", "$PatchStyleRecalibrationLayers"
        )
    }
    if ($ShiftedPatchTokenization) {
        $TrainArgs += @("--shifted-patch-tokenization")
    }
    if ($LayerTokenFusion) {
        $TrainArgs += @(
            "--layer-token-fusion",
            "--layer-token-fusion-layers", "$LayerTokenFusionLayers",
            "--layer-token-fusion-top-k", "$LayerTokenFusionTopK",
            "--layer-token-fusion-blend", "$LayerTokenFusionBlend",
            "--layer-token-fusion-attention-temperature", "$LayerTokenFusionAttentionTemperature",
            "--layer-token-fusion-bbox-weight", "$LayerTokenFusionBboxWeight",
            "--layer-token-fusion-foreground-weight", "$LayerTokenFusionForegroundWeight"
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
    if ($PatchMemoryAdapter) {
        $TrainArgs += @(
            "--patch-memory-adapter",
            "--patch-memory-adapter-dropout", "$PatchMemoryAdapterDropout"
        )
    }
    if ($LateClassAttentionPooling) {
        $TrainArgs += @(
            "--late-class-attention-pooling",
            "--late-class-attention-heads", "$LateClassAttentionHeads",
            "--late-class-attention-dropout", "$LateClassAttentionDropout",
            "--late-class-attention-mlp-ratio", "$LateClassAttentionMlpRatio",
            "--late-class-attention-residual-scale", "$LateClassAttentionResidualScale"
        )
    }
    if ($MixStyle) {
        $TrainArgs += @(
            "--mixstyle",
            "--mixstyle-probability", "$MixStyleProbability",
            "--mixstyle-alpha", "$MixStyleAlpha"
        )
    }
    if ($PairwiseMarginRouting) {
        $TrainArgs += @(
            "--pairwise-margin-routing",
            "--pairwise-margin-route-max-probability-margin",
            "$PairwiseMarginRouteMaxProbabilityMargin"
        )
    }
    if (-not $PairwiseConfusionNormalize) {
        $TrainArgs += @("--disable-pairwise-confusion-normalize")
    }
    if ($ConfusionSpectralBidirectional) {
        $TrainArgs += @("--confusion-spectral-bidirectional")
    }
    if ($EvalSurfaceDetailAmplification) {
        $TrainArgs += @("--eval-surface-detail-amplification")
    }
    if ($Smoke -or $SkipFinalTest) {
        $TrainArgs += @("--skip-final-test")
    }
    if ($TraceArchitecture) {
        $TrainArgs += @(
            "--trace-architecture",
            "--trace-architecture-seed", "$Seed",
            "--trace-architecture-device", "cuda"
        )
    }

    [ordered]@{
        run_name = $RunName
        started_at = $startedAt.ToString("o")
        smoke = [bool]$Smoke
        skip_final_test = [bool]$SkipFinalTest
        no_pretrained = $true
        data_yaml = $DataYaml
        classification_folder_yolo_data = $ClassificationFolderYoloData
        auxiliary_train_data = $AuxiliaryTrainData
        auxiliary_train_classification_folder_yolo_data = $AuxiliaryTrainClassificationFolderYoloData
        auxiliary_train_weight = $AuxiliaryTrainWeight
        paired_view_train = $PairedViewTrain
        resume_checkpoint = $ResumeCheckpoint
        batch_size = $BatchSize
        seed = $Seed
        disable_balanced_epoch_sampling = [bool]$DisableBalancedEpochSampling
        image_size = $ImageSize
        stem_architecture = $StemArchitecture
        stem_pooling_mode = $StemPoolingMode
        stem_softpool_blend = $StemSoftpoolBlend
        shifted_patch_tokenization = [bool]$ShiftedPatchTokenization
        shifted_patch_shift = $ShiftedPatchShift
        shifted_patch_residual_scale = $ShiftedPatchResidualScale
        crop_margin_ratio = $CropMarginRatio
        class_crop_margin_scale_threshold = $ClassCropMarginScaleThreshold
        class_crop_margin_max_ratio = $ClassCropMarginMaxRatio
        classification_source_context = [bool]$ClassificationSourceContext
        classification_source_context_mode = $ClassificationSourceContextMode
        classification_source_context_layout = $ClassificationSourceContextLayout
        classification_source_context_margin_ratio = $ClassificationSourceContextMarginRatio
        classification_source_context_background_alpha = $ClassificationSourceContextBackgroundAlpha
        classification_source_context_blur_radius = $ClassificationSourceContextBlurRadius
        classification_source_context_inset_scale = $ClassificationSourceContextInsetScale
        classification_source_context_aux = [bool]$ClassificationSourceContextAux
        source_context_aux_loss_weight = $SourceContextAuxLossWeight
        source_context_aux_classification_weight = $SourceContextAuxClassificationWeight
        source_context_aux_consistency_weight = $SourceContextAuxConsistencyWeight
        source_context_aux_bbox_margin_ratio = $SourceContextAuxBboxMarginRatio
        source_context_aux_attention_temperature = $SourceContextAuxAttentionTemperature
        token_pruning = [bool]$TokenPruning
        token_prune_bbox_weight = $TokenPruneBboxWeight
        token_prune_bbox_margin_ratio = $TokenPruneBboxMarginRatio
        bbox_token_prior_source = $BboxTokenPriorSource
        early_token_mask_keep_rate = $EarlyTokenMaskKeepRate
        bbox_spatial_fusion = [bool]$BBoxSpatialFusion
        bbox_spatial_fusion_hidden_dim = $BBoxSpatialFusionHiddenDim
        bbox_spatial_fusion_dropout = $BBoxSpatialFusionDropout
        bbox_spatial_fusion_logit_scale = $BBoxSpatialFusionLogitScale
        patch_objectness_guided_head = [bool]$PatchObjectnessGuidedHead
        patch_objectness_hidden_dim = $PatchObjectnessHiddenDim
        patch_objectness_dropout = $PatchObjectnessDropout
        patch_objectness_logit_scale = $PatchObjectnessLogitScale
        patch_objectness_temperature = $PatchObjectnessTemperature
        patch_objectness_loss_weight = $PatchObjectnessLossWeight
        patch_objectness_positive_weight = $PatchObjectnessPositiveWeight
        bbox_token_label_loss_weight = $BboxTokenLabelLossWeight
        bbox_token_label_min_prior = $BboxTokenLabelMinPrior
        bbox_token_label_prior_power = $BboxTokenLabelPriorPower
        bbox_token_label_classes = $BboxTokenLabelClasses
        bbox_token_label_focus_class = $BboxTokenLabelFocusClass
        bbox_token_label_focus_weight = $BboxTokenLabelFocusWeight
        bbox_token_label_detach_classifier = [bool]$BboxTokenLabelDetachClassifier
        bbox_token_label_start_epoch = $BboxTokenLabelStartEpoch
        patch_evidence_router_head = [bool]$PatchEvidenceRouterHead
        patch_evidence_router_pair = $PatchEvidenceRouterPair
        patch_evidence_router_hidden_dim = $PatchEvidenceRouterHiddenDim
        patch_evidence_router_top_k = $PatchEvidenceRouterTopK
        patch_evidence_router_bbox_weight = $PatchEvidenceRouterBboxWeight
        patch_evidence_router_dropout = $PatchEvidenceRouterDropout
        patch_evidence_router_logit_scale = $PatchEvidenceRouterLogitScale
        patch_evidence_router_margin_prior_mode = $PatchEvidenceRouterMarginPriorMode
        patch_evidence_router_margin_prior_scale = $PatchEvidenceRouterMarginPriorScale
        patch_evidence_router_summary_stats = [bool]$PatchEvidenceRouterSummaryStats
        patch_evidence_router_routing = [bool]$PatchEvidenceRouterRouting
        patch_evidence_router_route_max_probability_margin = $PatchEvidenceRouterRouteMaxProbabilityMargin
        patch_evidence_router_route_min_pair_probability = $PatchEvidenceRouterRouteMinPairProbability
        patch_evidence_router_loss_weight = $PatchEvidenceRouterLossWeight
        patch_evidence_router_positive_weight = $PatchEvidenceRouterPositiveWeight
        patch_evidence_router_start_epoch = $PatchEvidenceRouterStartEpoch
        patch_evidence_router_teacher_csv = $PatchEvidenceRouterTeacherCsv
        patch_evidence_router_teacher_loss_weight = $PatchEvidenceRouterTeacherLossWeight
        patch_evidence_router_teacher_min_confidence = $PatchEvidenceRouterTeacherMinConfidence
        patch_evidence_router_teacher_min_pair_mass = $PatchEvidenceRouterTeacherMinPairMass
        patch_evidence_router_teacher_positive_weight = $PatchEvidenceRouterTeacherPositiveWeight
        patch_evidence_linear_verifier_json = $PatchEvidenceLinearVerifierJson
        patch_evidence_linear_verifier_pair = $PatchEvidenceLinearVerifierPair
        patch_evidence_linear_verifier_min_pair_probability = $PatchEvidenceLinearVerifierMinPairProbability
        patch_evidence_linear_verifier_max_pair_margin = $PatchEvidenceLinearVerifierMaxPairMargin
        patch_evidence_linear_verifier_confidence_threshold = $PatchEvidenceLinearVerifierConfidenceThreshold
        patch_evidence_linear_verifier_logit_boost = $PatchEvidenceLinearVerifierLogitBoost
        patch_evidence_linear_verifier_protect_right_min_probability = $PatchEvidenceLinearVerifierProtectRightMinProbability
        patch_evidence_linear_verifier_training_soft_adjustment = [bool]$PatchEvidenceLinearVerifierTrainingSoftAdjustment
        patch_evidence_linear_verifier_training_soft_logit_scale = $PatchEvidenceLinearVerifierTrainingSoftLogitScale
        patch_evidence_linear_verifier_training_soft_gate_temperature = $PatchEvidenceLinearVerifierTrainingSoftGateTemperature
        bbox_prior_patch_context_head = [bool]$BBoxPriorPatchContextHead
        bbox_prior_patch_context_hidden_dim = $BBoxPriorPatchContextHiddenDim
        bbox_prior_patch_context_dropout = $BBoxPriorPatchContextDropout
        bbox_prior_patch_context_logit_scale = $BBoxPriorPatchContextLogitScale
        bbox_prior_patch_context_temperature = $BBoxPriorPatchContextTemperature
        source_context_feature_fusion = [bool]$SourceContextFeatureFusion
        source_context_fusion_hidden_dim = $SourceContextFusionHiddenDim
        source_context_fusion_dropout = $SourceContextFusionDropout
        source_context_fusion_logit_scale = $SourceContextFusionLogitScale
        source_context_fusion_gate_bias = $SourceContextFusionGateBias
        grad_accum_steps = $GradAccumSteps
        effective_batch_size = $BatchSize * $GradAccumSteps
        scheduler_total_epochs = $SchedulerTotalEpochsEffective
        learning_rate = $LearningRate
        min_learning_rate = $MinLearningRate
        trainable_module_prefixes = $TrainableModulePrefixes
        classification_mlp_head = [bool]$ClassificationMlpHead
        classification_mlp_hidden_dim = $ClassificationMlpHiddenDim
        classification_mlp_dropout = $ClassificationMlpDropout
        classification_mlp_residual_scale = $ClassificationMlpResidualScale
        branch_cnn_tokens = $BranchCnnTokens
        warmup_epochs = $WarmupEpochs
        weight_decay = $WeightDecay
        backbone_lr_scale = $BackboneLrScale
        num_workers = $NumWorkers
        eval_num_workers = $EvalNumWorkers
        frequency_selective_pooling = [bool]$FrequencySelectivePooling
        frequency_selective_top_k = $FrequencySelectiveTopK
        frequency_selective_blend = $FrequencySelectiveBlend
        frequency_selective_foreground_threshold = $FrequencySelectiveForegroundThreshold
        patch_memory_adapter = [bool]$PatchMemoryAdapter
        patch_memory_adapter_dropout = $PatchMemoryAdapterDropout
        late_class_attention_pooling = [bool]$LateClassAttentionPooling
        late_class_attention_heads = $LateClassAttentionHeads
        late_class_attention_dropout = $LateClassAttentionDropout
        late_class_attention_mlp_ratio = $LateClassAttentionMlpRatio
        late_class_attention_residual_scale = $LateClassAttentionResidualScale
        mixstyle = [bool]$MixStyle
        mixstyle_probability = $MixStyleProbability
        mixstyle_alpha = $MixStyleAlpha
        color_stat_fusion = [bool]$ColorStatFusion
        color_stat_fusion_dropout = $ColorStatFusionDropout
        defect_stat_fusion = [bool]$DefectStatFusion
        defect_stat_fusion_dropout = $DefectStatFusionDropout
        pairwise_margin_routing = [bool]$PairwiseMarginRouting
        pairwise_margin_route_max_probability_margin = $PairwiseMarginRouteMaxProbabilityMargin
        topk_reassessment_head = [bool]$TopKReassessmentHead
        topk_reassessment_top_k = $TopKReassessmentTopK
        topk_reassessment_hidden_dim = $TopKReassessmentHiddenDim
        topk_reassessment_dropout = $TopKReassessmentDropout
        topk_reassessment_logit_scale = $TopKReassessmentLogitScale
        topk_reassessment_routing = [bool]$TopKReassessmentRouting
        topk_reassessment_route_pairs = $TopKReassessmentRoutePairs
        topk_reassessment_route_max_probability_margin = $TopKReassessmentRouteMaxProbabilityMargin
        topk_reassessment_aux_loss_weight = $TopKReassessmentAuxLossWeight
        topk_reassessment_aux_route_min_weight = $TopKReassessmentAuxRouteMinWeight
        focus_class_head = [bool]$FocusClassHead
        focus_class_index = $FocusClassIndex
        focus_class_logit_scale = $FocusClassLogitScale
        focus_class_dropout = $FocusClassDropout
        focus_class_routing = [bool]$FocusClassRouting
        focus_class_route_max_probability_margin = $FocusClassRouteMaxProbabilityMargin
        focus_class_route_min_probability = $FocusClassRouteMinProbability
        focus_class_aux_loss_weight = $FocusClassAuxLossWeight
        focus_class_aux_positive_weight = $FocusClassAuxPositiveWeight
        class_independent_head = [bool]$ClassIndependentHead
        class_independent_dropout = $ClassIndependentDropout
        class_independent_loss_weight = $ClassIndependentLossWeight
        class_independent_positive_weight = $ClassIndependentPositiveWeight
        focus_tversky_loss_weight = $FocusTverskyLossWeight
        focus_tversky_class = $FocusTverskyClass
        focus_tversky_alpha = $FocusTverskyAlpha
        focus_tversky_beta = $FocusTverskyBeta
        focus_tversky_gamma = $FocusTverskyGamma
        focus_tversky_probability_power = $FocusTverskyProbabilityPower
        focus_tversky_start_epoch = $FocusTverskyStartEpoch
        focus_auc_rank_loss_weight = $FocusAucRankLossWeight
        focus_auc_rank_class = $FocusAucRankClass
        focus_auc_rank_negative_classes = $FocusAucRankNegativeClasses
        focus_auc_rank_margin = $FocusAucRankMargin
        focus_auc_rank_temperature = $FocusAucRankTemperature
        focus_auc_rank_hard_fraction = $FocusAucRankHardFraction
        focus_auc_rank_start_epoch = $FocusAucRankStartEpoch
        focus_partial_auc_loss_weight = $FocusPartialAucLossWeight
        focus_partial_auc_class = $FocusPartialAucClass
        focus_partial_auc_negative_classes = $FocusPartialAucNegativeClasses
        focus_partial_auc_margin = $FocusPartialAucMargin
        focus_partial_auc_temperature = $FocusPartialAucTemperature
        focus_partial_auc_negative_fraction = $FocusPartialAucNegativeFraction
        focus_partial_auc_positive_fraction = $FocusPartialAucPositiveFraction
        focus_partial_auc_positive_weight = $FocusPartialAucPositiveWeight
        focus_partial_auc_min_negative_probability = $FocusPartialAucMinNegativeProbability
        focus_partial_auc_start_epoch = $FocusPartialAucStartEpoch
        bbox_foreground_dropout_loss_weight = $BBoxForegroundDropoutLossWeight
        bbox_foreground_dropout_consistency_weight = $BBoxForegroundDropoutConsistencyWeight
        bbox_foreground_dropout_probability = $BBoxForegroundDropoutProbability
        bbox_foreground_dropout_min_area_ratio = $BBoxForegroundDropoutMinAreaRatio
        bbox_foreground_dropout_max_area_ratio = $BBoxForegroundDropoutMaxAreaRatio
        bbox_foreground_dropout_mode = $BBoxForegroundDropoutMode
        bbox_foreground_dropout_fill = $BBoxForegroundDropoutFill
        bbox_foreground_dropout_temperature = $BBoxForegroundDropoutTemperature
        bbox_object_erasure_negative_loss_weight = $BBoxObjectErasureNegativeLossWeight
        bbox_object_erasure_probability = $BBoxObjectErasureProbability
        bbox_object_erasure_margin_ratio = $BBoxObjectErasureMarginRatio
        bbox_object_erasure_fill = $BBoxObjectErasureFill
        bbox_object_erasure_blur_kernel = $BBoxObjectErasureBlurKernel
        bbox_object_erasure_temperature = $BBoxObjectErasureTemperature
        ordinal_maturity_head = [bool]$OrdinalMaturityHead
        ordinal_maturity_classes = $OrdinalMaturityClasses
        ordinal_maturity_logit_scale = $OrdinalMaturityLogitScale
        ordinal_maturity_dropout = $OrdinalMaturityDropout
        ordinal_maturity_loss_weight = $OrdinalMaturityLossWeight
        cumulative_ordinal_head = [bool]$CumulativeOrdinalHead
        cumulative_ordinal_classes = $CumulativeOrdinalClasses
        cumulative_ordinal_logit_scale = $CumulativeOrdinalLogitScale
        cumulative_ordinal_dropout = $CumulativeOrdinalDropout
        cumulative_ordinal_loss_weight = $CumulativeOrdinalLossWeight
        cumulative_ordinal_threshold_weights = $CumulativeOrdinalThresholdWeights
        ordinal_distribution_loss_weight = $OrdinalDistributionLossWeight
        ordinal_distribution_classes = $OrdinalDistributionClasses
        ordinal_distribution_target_sigma = $OrdinalDistributionTargetSigma
        ordinal_distribution_start_epoch = $OrdinalDistributionStartEpoch
        classification_loss = $ClassificationLoss
        balanced_softmax_tau = $BalancedSoftmaxTau
        gce_q = $GceQ
        ldr_margin = $LdrMargin
        ldr_temperature = $LdrTemperature
        logit_norm_temperature = $LogitNormTemperature
        symmetric_ce_alpha = $SymmetricCeAlpha
        symmetric_ce_beta = $SymmetricCeBeta
        symmetric_ce_epsilon = $SymmetricCeEpsilon
        seesaw_mitigation_power = $SeesawMitigationPower
        seesaw_compensation_power = $SeesawCompensationPower
        focal_loss_gamma = $FocalLossGamma
        focal_loss_mix = $FocalLossMix
        label_smoothing = $LabelSmoothing
        ldam_max_margin = $LdamMaxMargin
        ldam_scale = $LdamScale
        mutual_channel_loss_weight = $MutualChannelLossWeight
        mutual_channel_top_k = $MutualChannelTopK
        mutual_channel_diversity_weight = $MutualChannelDiversityWeight
        mutual_channel_start_epoch = $MutualChannelStartEpoch
        complement_entropy_loss_weight = $ComplementEntropyLossWeight
        complement_entropy_classes = $ComplementEntropyClasses
        complement_entropy_start_epoch = $ComplementEntropyStartEpoch
        background_suppression_mode = $BackgroundSuppressionMode
        background_suppression_probability = $BackgroundSuppressionProbability
        background_suppression_margin = $BackgroundSuppressionMargin
        background_suppression_blur_radius = $BackgroundSuppressionBlurRadius
        surface_detail_amplification_mode = $SurfaceDetailAmplificationMode
        surface_detail_amplification_probability = $SurfaceDetailAmplificationProbability
        surface_detail_amplification_strength = $SurfaceDetailAmplificationStrength
        surface_detail_amplification_blur_radius = $SurfaceDetailAmplificationBlurRadius
        surface_detail_amplification_foreground_weight = $SurfaceDetailAmplificationForegroundWeight
        eval_surface_detail_amplification = [bool]$EvalSurfaceDetailAmplification
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
        self_adaptive_target_loss_weight = $SelfAdaptiveTargetLossWeight
        self_adaptive_target_beta = $SelfAdaptiveTargetBeta
        self_adaptive_target_start_epoch = $SelfAdaptiveTargetStartEpoch
        self_adaptive_target_hard_weight = $SelfAdaptiveTargetHardWeight
        self_adaptive_target_confidence_power = $SelfAdaptiveTargetConfidencePower
        self_adaptive_target_min_confidence = $SelfAdaptiveTargetMinConfidence
        rdrop_loss_weight = $RDropLossWeight
        rdrop_temperature = $RDropTemperature
        augmix_consistency_loss_weight = $AugmixConsistencyLossWeight
        augmix_consistency_probability = $AugmixConsistencyProbability
        augmix_consistency_severity = $AugmixConsistencySeverity
        augmix_consistency_width = $AugmixConsistencyWidth
        augmix_consistency_depth = $AugmixConsistencyDepth
        augmix_consistency_alpha = $AugmixConsistencyAlpha
        augmix_consistency_temperature = $AugmixConsistencyTemperature
        illumination_consistency_loss_weight = $IlluminationConsistencyLossWeight
        illumination_consistency_probability = $IlluminationConsistencyProbability
        illumination_consistency_brightness = $IlluminationConsistencyBrightness
        illumination_consistency_contrast = $IlluminationConsistencyContrast
        illumination_consistency_gamma = $IlluminationConsistencyGamma
        illumination_consistency_temperature = $IlluminationConsistencyTemperature
        foreground_chroma_consistency_loss_weight = $ForegroundChromaConsistencyLossWeight
        foreground_chroma_consistency_probability = $ForegroundChromaConsistencyProbability
        foreground_chroma_consistency_saturation_delta = $ForegroundChromaConsistencySaturationDelta
        foreground_chroma_consistency_hue_delta = $ForegroundChromaConsistencyHueDelta
        foreground_chroma_consistency_bbox_margin_ratio = $ForegroundChromaConsistencyBboxMarginRatio
        foreground_chroma_consistency_temperature = $ForegroundChromaConsistencyTemperature
        foreground_chroma_consistency_classes = $ForegroundChromaConsistencyClasses
        friendly_adversarial_loss_weight = $FriendlyAdversarialLossWeight
        friendly_adversarial_epsilon = $FriendlyAdversarialEpsilon
        friendly_adversarial_step_size = $FriendlyAdversarialStepSize
        friendly_adversarial_steps = $FriendlyAdversarialSteps
        friendly_adversarial_bbox_erode_ratio = $FriendlyAdversarialBboxErodeRatio
        friendly_adversarial_focus_class = $FriendlyAdversarialFocusClass
        friendly_adversarial_negative_classes = $FriendlyAdversarialNegativeClasses
        friendly_adversarial_max_per_direction = $FriendlyAdversarialMaxPerDirection
        friendly_adversarial_start_epoch = $FriendlyAdversarialStartEpoch
        semantic_attribute_loss_weight = $SemanticAttributeLossWeight
        semantic_attribute_specs = $SemanticAttributeSpecs
        confusion_pair_mixup_loss_weight = $ConfusionPairMixupLossWeight
        confusion_pair_mixup_alpha = $ConfusionPairMixupAlpha
        confusion_pair_mixup_pairs = $ConfusionPairMixupPairs
        confusion_pair_mixup_max_pairs = $ConfusionPairMixupMaxPairs
        confusion_pair_mixup_start_epoch = $ConfusionPairMixupStartEpoch
        foreground_snapmix_loss_weight = $ForegroundSnapmixLossWeight
        foreground_snapmix_probability = $ForegroundSnapmixProbability
        foreground_snapmix_alpha = $ForegroundSnapmixAlpha
        foreground_snapmix_pairs = $ForegroundSnapmixPairs
        foreground_snapmix_min_area_ratio = $ForegroundSnapmixMinAreaRatio
        foreground_snapmix_max_area_ratio = $ForegroundSnapmixMaxAreaRatio
        foreground_snapmix_bbox_margin_ratio = $ForegroundSnapmixBboxMarginRatio
        foreground_counterexample_mix_loss_weight = $ForegroundCounterexampleMixLossWeight
        foreground_counterexample_mix_probability = $ForegroundCounterexampleMixProbability
        foreground_counterexample_mix_source_class = $ForegroundCounterexampleMixSourceClass
        foreground_counterexample_mix_target_classes = $ForegroundCounterexampleMixTargetClasses
        foreground_counterexample_mix_alpha = $ForegroundCounterexampleMixAlpha
        foreground_counterexample_mix_min_area_ratio = $ForegroundCounterexampleMixMinAreaRatio
        foreground_counterexample_mix_max_area_ratio = $ForegroundCounterexampleMixMaxAreaRatio
        foreground_counterexample_mix_bbox_margin_ratio = $ForegroundCounterexampleMixBboxMarginRatio
        foreground_counterexample_mix_start_epoch = $ForegroundCounterexampleMixStartEpoch
        dcl_region_shuffle_loss_weight = $DclRegionShuffleLossWeight
        dcl_region_shuffle_probability = $DclRegionShuffleProbability
        dcl_region_shuffle_grid_size = $DclRegionShuffleGridSize
        dcl_region_shuffle_bbox_margin_ratio = $DclRegionShuffleBboxMarginRatio
        dcl_region_shuffle_classes = $DclRegionShuffleClasses
        dcl_region_shuffle_start_epoch = $DclRegionShuffleStartEpoch
        quantized_label_cpu_loss_weight = $QuantizedLabelCpuLossWeight
        quantized_label_cpu_classes = $QuantizedLabelCpuClasses
        quantized_label_cpu_min_prior = $QuantizedLabelCpuMinPrior
        quantized_label_cpu_max_prior = $QuantizedLabelCpuMaxPrior
        quantized_label_cpu_negative_weight = $QuantizedLabelCpuNegativeWeight
        quantized_label_cpu_non_negative_beta = $QuantizedLabelCpuNonNegativeBeta
        quantized_label_cpu_start_epoch = $QuantizedLabelCpuStartEpoch
        self_paced_loss_weight = $SelfPacedLossWeight
        self_paced_loss_percentile = $SelfPacedLossPercentile
        self_paced_loss_gamma = $SelfPacedLossGamma
        self_paced_loss_min_weight = $SelfPacedLossMinWeight
        self_paced_loss_start_epoch = $SelfPacedLossStartEpoch
        self_paced_loss_class_balanced = [bool]$SelfPacedLossClassBalanced
        cyflod_loss_damping_weight = $CyflodLossDampingWeight
        cyflod_loss_damping_delta = $CyflodLossDampingDelta
        cyflod_loss_damping_cycle_epochs = $CyflodLossDampingCycleEpochs
        cyflod_loss_damping_min_weight = $CyflodLossDampingMinWeight
        cyflod_loss_damping_start_epoch = $CyflodLossDampingStartEpoch
        metric_learning_loss_weight = $MetricLearningLossWeight
        metric_learning_temperature = $MetricLearningTemperature
        metric_learning_sources = $MetricLearningSources
        teacher_guided_contrastive_loss_weight = $TeacherGuidedContrastiveLossWeight
        teacher_guided_contrastive_temperature = $TeacherGuidedContrastiveTemperature
        teacher_guided_contrastive_sources = $TeacherGuidedContrastiveSources
        teacher_guided_contrastive_classes = $TeacherGuidedContrastiveClasses
        teacher_guided_contrastive_teacher_min_confidence = $TeacherGuidedContrastiveTeacherMinConfidence
        teacher_guided_contrastive_require_agreement = [bool]$TeacherGuidedContrastiveRequireAgreement
        teacher_guided_contrastive_class_balanced = [bool]$TeacherGuidedContrastiveClassBalanced
        teacher_guided_contrastive_weight_mode = $TeacherGuidedContrastiveWeightMode
        teacher_guided_contrastive_stochastic_std = $TeacherGuidedContrastiveStochasticStd
        teacher_guided_contrastive_min_reliability = $TeacherGuidedContrastiveMinReliability
        teacher_guided_contrastive_teacher_confidence_power = $TeacherGuidedContrastiveTeacherConfidencePower
        teacher_guided_contrastive_memory_queue_size = $TeacherGuidedContrastiveMemoryQueueSize
        teacher_guided_contrastive_memory_min_count = $TeacherGuidedContrastiveMemoryMinCount
        boundary_contrastive_loss_weight = $BoundaryContrastiveLossWeight
        boundary_contrastive_pairs = $BoundaryContrastivePairs
        boundary_contrastive_sources = $BoundaryContrastiveSources
        boundary_contrastive_margin = $BoundaryContrastiveMargin
        boundary_contrastive_temperature = $BoundaryContrastiveTemperature
        boundary_contrastive_max_pairs = $BoundaryContrastiveMaxPairs
        boundary_center_loss_weight = $BoundaryCenterLossWeight
        boundary_center_pairs = $BoundaryCenterPairs
        boundary_center_sources = $BoundaryCenterSources
        boundary_center_margin = $BoundaryCenterMargin
        boundary_center_temperature = $BoundaryCenterTemperature
        boundary_center_compactness_weight = $BoundaryCenterCompactnessWeight
        boundary_center_teacher_min_confidence = $BoundaryCenterTeacherMinConfidence
        boundary_center_require_agreement = [bool]$BoundaryCenterRequireAgreement
        boundary_center_teacher_weight_mode = $BoundaryCenterTeacherWeightMode
        border_attention_suppression_loss_weight = $BorderAttentionSuppressionLossWeight
        border_attention_suppression_frame_width = $BorderAttentionSuppressionFrameWidth
        border_attention_suppression_bbox_band = $BorderAttentionSuppressionBboxBand
        border_attention_suppression_bbox_weight = $BorderAttentionSuppressionBboxWeight
        border_attention_suppression_temperature = $BorderAttentionSuppressionTemperature
        border_attention_suppression_classes = $BorderAttentionSuppressionClasses
        border_attention_suppression_start_epoch = $BorderAttentionSuppressionStartEpoch
        register_attention_alignment_loss_weight = $RegisterAttentionAlignmentLossWeight
        register_attention_alignment_classes = $RegisterAttentionAlignmentClasses
        register_attention_alignment_bbox_margin_ratio = $RegisterAttentionAlignmentBboxMarginRatio
        register_attention_alignment_agreement_weight = $RegisterAttentionAlignmentAgreementWeight
        register_attention_alignment_foreground_weight = $RegisterAttentionAlignmentForegroundWeight
        register_attention_alignment_start_epoch = $RegisterAttentionAlignmentStartEpoch
        register_diversity_loss_weight = $RegisterDiversityLossWeight
        foreground_surface_aux_loss_weight = $ForegroundSurfaceAuxLossWeight
        foreground_surface_pairwise_loss_weight = $ForegroundSurfacePairwiseLossWeight
        interior_boundary_pairwise_loss_weight = $InteriorBoundaryPairwiseLossWeight
        angular_margin_loss_weight = $AngularMarginLossWeight
        angular_margin = $AngularMargin
        angular_margin_scale = $AngularMarginScale
        angular_margin_start_epoch = $AngularMarginStartEpoch
        angular_margin_classes = $AngularMarginClasses
        subcenter_proxy_loss_weight = $SubcenterProxyLossWeight
        subcenter_proxy_subcenters = $SubcenterProxySubcenters
        subcenter_proxy_margin = $SubcenterProxyMargin
        subcenter_proxy_scale = $SubcenterProxyScale
        subcenter_proxy_classes = $SubcenterProxyClasses
        subcenter_proxy_start_epoch = $SubcenterProxyStartEpoch
        subcenter_proxy_dropout = $SubcenterProxyDropout
        subcenter_proxy_init_std = $SubcenterProxyInitStd
        deep_abstention_loss_weight = $DeepAbstentionLossWeight
        deep_abstention_penalty = $DeepAbstentionPenalty
        deep_abstention_start_epoch = $DeepAbstentionStartEpoch
        deep_abstention_dropout = $DeepAbstentionDropout
        deep_abstention_initial_probability = $DeepAbstentionInitialProbability
        ordinal_boundary_loss_weight = $OrdinalBoundaryLossWeight
        ordinal_boundary_classes = $OrdinalBoundaryClasses
        ordinal_boundary_threshold_weights = $OrdinalBoundaryThresholdWeights
        ordinal_boundary_temperature = $OrdinalBoundaryTemperature
        ordinal_boundary_start_epoch = $OrdinalBoundaryStartEpoch
        pairwise_confusion_loss_weight = $PairwiseConfusionLossWeight
        pairwise_confusion_sources = $PairwiseConfusionSources
        pairwise_confusion_start_epoch = $PairwiseConfusionStartEpoch
        pairwise_confusion_normalize = [bool]$PairwiseConfusionNormalize
        confusion_spectral_loss_weight = $ConfusionSpectralLossWeight
        confusion_spectral_ema_momentum = $ConfusionSpectralEmaMomentum
        confusion_spectral_frequency_smoothing = $ConfusionSpectralFrequencySmoothing
        confusion_spectral_margin = $ConfusionSpectralMargin
        confusion_spectral_start_epoch = $ConfusionSpectralStartEpoch
        confusion_spectral_bidirectional = [bool]$ConfusionSpectralBidirectional
        sample_weight_manifest = $SampleWeightManifest
        sample_weight_factor = $SampleWeightFactor
        sample_weight_max = $SampleWeightMax
        quality_group_manifest = $QualityGroupManifest
        group_dro_loss_weight = $GroupDroLossWeight
        group_dro_temperature = $GroupDroTemperature
        group_dro_min_samples = $GroupDroMinSamples
        hard_sample_manifest = $HardSampleManifest
        hard_sample_repeat_factor = $HardSampleRepeatFactor
        ambiguous_soft_target_manifest = $AmbiguousSoftTargetManifest
        ambiguous_soft_target_alpha = $AmbiguousSoftTargetAlpha
        targeted_margin_manifest = $TargetedMarginManifest
        targeted_margin_loss_weight = $TargetedMarginLossWeight
        targeted_margin_default_margin = $TargetedMarginDefaultMargin
        targeted_margin_default_weight = $TargetedMarginDefaultWeight
        targeted_margin_max_weight = $TargetedMarginMaxWeight
        focus_neighbor_binary_manifest = $FocusNeighborBinaryManifest
        focus_neighbor_binary_loss_weight = $FocusNeighborBinaryLossWeight
        focus_neighbor_binary_focus_class = $FocusNeighborBinaryFocusClass
        focus_neighbor_binary_neighbor_classes = $FocusNeighborBinaryNeighborClasses
        focus_neighbor_binary_default_weight = $FocusNeighborBinaryDefaultWeight
        focus_neighbor_binary_max_weight = $FocusNeighborBinaryMaxWeight
        focused_false_positive_margin_loss_weight = $FocusedFalsePositiveMarginLossWeight
        focused_false_positive_class = $FocusedFalsePositiveClass
        focused_false_positive_negative_classes = $FocusedFalsePositiveNegativeClasses
        focused_false_positive_margin = $FocusedFalsePositiveMargin
        focused_false_positive_min_probability = $FocusedFalsePositiveMinProbability
        focused_false_positive_probability_power = $FocusedFalsePositiveProbabilityPower
        background_counterfactual_consistency_weight = $BackgroundCounterfactualConsistencyWeight
        background_counterfactual_probability = $BackgroundCounterfactualProbability
        background_counterfactual_mode = $BackgroundCounterfactualMode
        background_counterfactual_margin = $BackgroundCounterfactualMargin
        background_counterfactual_blur_kernel = $BackgroundCounterfactualBlurKernel
        background_counterfactual_temperature = $BackgroundCounterfactualTemperature
        background_focus_suppression_loss_weight = $BackgroundFocusSuppressionLossWeight
        background_focus_suppression_probability = $BackgroundFocusSuppressionProbability
        background_focus_suppression_focus_class = $BackgroundFocusSuppressionFocusClass
        background_focus_suppression_negative_classes = $BackgroundFocusSuppressionNegativeClasses
        background_focus_suppression_margin = $BackgroundFocusSuppressionMargin
        background_focus_suppression_min_probability = $BackgroundFocusSuppressionMinProbability
        background_focus_suppression_probability_power = $BackgroundFocusSuppressionProbabilityPower
        source_context_focus_suppression_loss_weight = $SourceContextFocusSuppressionLossWeight
        source_context_focus_suppression_probability = $SourceContextFocusSuppressionProbability
        source_context_focus_suppression_focus_class = $SourceContextFocusSuppressionFocusClass
        source_context_focus_suppression_negative_classes = $SourceContextFocusSuppressionNegativeClasses
        source_context_focus_suppression_margin = $SourceContextFocusSuppressionMargin
        source_context_focus_suppression_min_probability = $SourceContextFocusSuppressionMinProbability
        source_context_focus_suppression_probability_power = $SourceContextFocusSuppressionProbabilityPower
        surface_counterfactual_consistency_weight = $SurfaceCounterfactualConsistencyWeight
        surface_counterfactual_probability = $SurfaceCounterfactualProbability
        surface_counterfactual_mode = $SurfaceCounterfactualMode
        surface_counterfactual_strength = $SurfaceCounterfactualStrength
        surface_counterfactual_blur_kernel = $SurfaceCounterfactualBlurKernel
        surface_counterfactual_temperature = $SurfaceCounterfactualTemperature
        surface_amplified_supervised_loss_weight = $SurfaceAmplifiedSupervisedLossWeight
        surface_amplified_boundary_margin_loss_weight = $SurfaceAmplifiedBoundaryMarginLossWeight
        surface_amplified_probability = $SurfaceAmplifiedProbability
        surface_amplified_mode = $SurfaceAmplifiedMode
        surface_amplified_strength = $SurfaceAmplifiedStrength
        surface_amplified_blur_kernel = $SurfaceAmplifiedBlurKernel
        surface_amplified_boundary_pairs = $SurfaceAmplifiedBoundaryPairs
        surface_amplified_boundary_margin = $SurfaceAmplifiedBoundaryMargin
        paired_view_supervised_loss_weight = $PairedViewSupervisedLossWeight
        paired_view_consistency_weight = $PairedViewConsistencyWeight
        paired_view_feature_consistency_weight = $PairedViewFeatureConsistencyWeight
        paired_view_fusion_loss_weight = $PairedViewFusionLossWeight
        paired_view_fusion_consistency_weight = $PairedViewFusionConsistencyWeight
        paired_view_temperature = $PairedViewTemperature
        paired_view_feature_source = $PairedViewFeatureSource
        masked_reconstruction_loss_weight = $MaskedReconstructionLossWeight
        masked_reconstruction_mask_ratio = $MaskedReconstructionMaskRatio
        masked_reconstruction_foreground_weight = $MaskedReconstructionForegroundWeight
        masked_reconstruction_detail_weight = $MaskedReconstructionDetailWeight
        masked_reconstruction_bbox_weight = $MaskedReconstructionBboxWeight
        masked_reconstruction_bbox_margin_ratio = $MaskedReconstructionBboxMarginRatio
        distillation_teacher_csv = $DistillationTeacherCsv
        distillation_weight = $DistillationWeight
        distillation_temperature = $DistillationTemperature
        distillation_focus_class_index = $DistillationFocusClassIndex
        distillation_focus_class_weight = $DistillationFocusClassWeight
        teacher_non_target_distillation_loss_weight = $TeacherNonTargetDistillationLossWeight
        teacher_non_target_distillation_classes = $TeacherNonTargetDistillationClasses
        teacher_non_target_distillation_temperature = $TeacherNonTargetDistillationTemperature
        teacher_non_target_distillation_teacher_min_confidence = $TeacherNonTargetDistillationTeacherMinConfidence
        teacher_non_target_distillation_require_agreement = [bool]$TeacherNonTargetDistillationRequireAgreement
        teacher_focus_margin_loss_weight = $TeacherFocusMarginLossWeight
        teacher_focus_margin_focus_class = $TeacherFocusMarginFocusClass
        teacher_focus_margin_negative_classes = $TeacherFocusMarginNegativeClasses
        teacher_focus_margin_teacher_max_probability = $TeacherFocusMarginTeacherMaxProbability
        teacher_focus_margin_margin = $TeacherFocusMarginMargin
        teacher_focus_margin_min_probability = $TeacherFocusMarginMinProbability
        teacher_focus_margin_probability_power = $TeacherFocusMarginProbabilityPower
        teacher_focus_margin_require_agreement = [bool]$TeacherFocusMarginRequireAgreement
        teacher_focus_binary_loss_weight = $TeacherFocusBinaryLossWeight
        teacher_focus_binary_focus_class = $TeacherFocusBinaryFocusClass
        teacher_focus_binary_classes = $TeacherFocusBinaryClasses
        teacher_focus_binary_teacher_min_confidence = $TeacherFocusBinaryTeacherMinConfidence
        teacher_focus_binary_error_power = $TeacherFocusBinaryErrorPower
        teacher_focus_binary_hard_target_blend = $TeacherFocusBinaryHardTargetBlend
        teacher_focus_binary_require_agreement = [bool]$TeacherFocusBinaryRequireAgreement
        teacher_pairwise_margin_loss_weight = $TeacherPairwiseMarginLossWeight
        teacher_pairwise_margin_teacher_mass_threshold = $TeacherPairwiseMarginTeacherMassThreshold
        teacher_pairwise_margin_error_power = $TeacherPairwiseMarginErrorPower
        teacher_pairwise_margin_hard_target_blend = $TeacherPairwiseMarginHardTargetBlend
        teacher_pairwise_margin_require_agreement = [bool]$TeacherPairwiseMarginRequireAgreement
        teacher_feature_npz = $TeacherFeatureNpz
        teacher_feature_rkd_loss_weight = $TeacherFeatureRkdLossWeight
        teacher_feature_rkd_distance_weight = $TeacherFeatureRkdDistanceWeight
        teacher_feature_rkd_angle_weight = $TeacherFeatureRkdAngleWeight
        teacher_feature_rkd_source = $TeacherFeatureRkdSource
        teacher_feature_rkd_pair_mode = $TeacherFeatureRkdPairMode
        teacher_feature_rkd_pairs = $TeacherFeatureRkdPairs
        teacher_feature_contrastive_loss_weight = $TeacherFeatureContrastiveLossWeight
        teacher_feature_contrastive_temperature = $TeacherFeatureContrastiveTemperature
        teacher_feature_contrastive_projection_dim = $TeacherFeatureContrastiveProjectionDim
        teacher_feature_contrastive_use_projection_adapter = [bool]$TeacherFeatureContrastiveUseProjectionAdapter
        teacher_feature_contrastive_adapter_dropout = $TeacherFeatureContrastiveAdapterDropout
        teacher_feature_contrastive_source = $TeacherFeatureContrastiveSource
        teacher_feature_contrastive_pair_mode = $TeacherFeatureContrastivePairMode
        teacher_feature_contrastive_pairs = $TeacherFeatureContrastivePairs
        teacher_feature_contrastive_start_epoch = $TeacherFeatureContrastiveStartEpoch
        class_loss_multipliers = $ClassLossMultipliers
        local_exposure_probability = $LocalExposureProbability
        local_exposure_strength = $LocalExposureStrength
        obstacle_probability = $ObstacleProbability
        obstacle_max_area = $ObstacleMaxArea
        randaugment_num_ops = $RandAugmentNumOps
        randaugment_magnitude = $RandAugmentMagnitude
        sam = [bool]$Sam
        sam_rho = $SamRho
        sam_adaptive = [bool]$SamAdaptive
        foreground_surface_fusion = [bool]$ForegroundSurfaceFusion
        foreground_surface_fusion_dropout = $ForegroundSurfaceFusionDropout
        foreground_surface_pairwise_head = [bool]$ForegroundSurfacePairwiseHead
        foreground_surface_pairwise_pairs = $ForegroundSurfacePairwisePairs
        foreground_surface_pairwise_logit_scale = $ForegroundSurfacePairwiseLogitScale
        foreground_surface_pairwise_dropout = $ForegroundSurfacePairwiseDropout
        foreground_surface_pairwise_routing = [bool]$ForegroundSurfacePairwiseRouting
        foreground_surface_pairwise_route_max_probability_margin = $ForegroundSurfacePairwiseRouteMaxProbabilityMargin
        interior_boundary_pairwise_head = [bool]$InteriorBoundaryPairwiseHead
        interior_boundary_pairwise_pairs = $InteriorBoundaryPairwisePairs
        interior_boundary_pairwise_logit_scale = $InteriorBoundaryPairwiseLogitScale
        interior_boundary_pairwise_dropout = $InteriorBoundaryPairwiseDropout
        interior_boundary_pairwise_hidden_dim = $InteriorBoundaryPairwiseHiddenDim
        interior_boundary_pairwise_erode_kernel = $InteriorBoundaryPairwiseErodeKernel
        interior_boundary_pairwise_routing = [bool]$InteriorBoundaryPairwiseRouting
        interior_boundary_pairwise_route_max_probability_margin = $InteriorBoundaryPairwiseRouteMaxProbabilityMargin
        bilinear_patch_fusion = [bool]$BilinearPatchFusion
        bilinear_patch_rank = $BilinearPatchRank
        bilinear_patch_dropout = $BilinearPatchDropout
        complementary_patch_suppression_head = [bool]$ComplementaryPatchSuppressionHead
        complementary_patch_suppression_top_k = $ComplementaryPatchSuppressionTopK
        complementary_patch_suppression_hidden_dim = $ComplementaryPatchSuppressionHiddenDim
        complementary_patch_suppression_dropout = $ComplementaryPatchSuppressionDropout
        complementary_patch_suppression_temperature = $ComplementaryPatchSuppressionTemperature
        complementary_patch_suppression_strength = $ComplementaryPatchSuppressionStrength
        complementary_patch_suppression_bbox_weight = $ComplementaryPatchSuppressionBboxWeight
        complementary_patch_suppression_logit_scale = $ComplementaryPatchSuppressionLogitScale
        micro_detail_patch_expert = [bool]$MicroDetailPatchExpert
        micro_detail_top_k = $MicroDetailTopK
        micro_detail_hidden_dim = $MicroDetailHiddenDim
        micro_detail_dropout = $MicroDetailDropout
        micro_detail_temperature = $MicroDetailTemperature
        micro_detail_foreground_power = $MicroDetailForegroundPower
        micro_detail_logit_scale = $MicroDetailLogitScale
        micro_detail_routing = [bool]$MicroDetailRouting
        micro_detail_route_pairs = $MicroDetailRoutePairs
        micro_detail_route_max_probability_margin = $MicroDetailRouteMaxProbabilityMargin
        micro_detail_aux_loss_weight = $MicroDetailAuxLossWeight
        part_token_learner = [bool]$PartTokenLearner
        part_token_count = $PartTokenCount
        part_token_hidden_dim = $PartTokenHiddenDim
        part_token_dropout = $PartTokenDropout
        part_token_temperature = $PartTokenTemperature
        part_token_foreground_power = $PartTokenForegroundPower
        part_token_bbox_weight = $PartTokenBboxWeight
        part_token_logit_scale = $PartTokenLogitScale
        part_token_routing = [bool]$PartTokenRouting
        part_token_route_pairs = $PartTokenRoutePairs
        part_token_route_max_probability_margin = $PartTokenRouteMaxProbabilityMargin
        part_token_aux_loss_weight = $PartTokenAuxLossWeight
        part_token_pairwise_head = [bool]$PartTokenPairwiseHead
        part_token_pairwise_pairs = $PartTokenPairwisePairs
        part_token_pairwise_logit_scale = $PartTokenPairwiseLogitScale
        part_token_pairwise_dropout = $PartTokenPairwiseDropout
        part_token_pairwise_routing = [bool]$PartTokenPairwiseRouting
        part_token_pairwise_route_max_probability_margin = $PartTokenPairwiseRouteMaxProbabilityMargin
        part_token_pairwise_loss_weight = $PartTokenPairwiseLossWeight
        local_zoom_image_expert = [bool]$LocalZoomImageExpert
        local_zoom_crop_size = $LocalZoomCropSize
        local_zoom_crop_scale = $LocalZoomCropScale
        local_zoom_score_mode = $LocalZoomScoreMode
        local_zoom_hidden_dim = $LocalZoomHiddenDim
        local_zoom_dropout = $LocalZoomDropout
        local_zoom_logit_scale = $LocalZoomLogitScale
        local_zoom_routing = [bool]$LocalZoomRouting
        local_zoom_route_pairs = $LocalZoomRoutePairs
        local_zoom_route_max_probability_margin = $LocalZoomRouteMaxProbabilityMargin
        local_zoom_aux_loss_weight = $LocalZoomAuxLossWeight
        high_frequency_texture_expert = [bool]$HighFrequencyTextureExpert
        high_frequency_texture_hidden_dim = $HighFrequencyTextureHiddenDim
        high_frequency_texture_dropout = $HighFrequencyTextureDropout
        high_frequency_texture_analysis_size = $HighFrequencyTextureAnalysisSize
        high_frequency_texture_logit_scale = $HighFrequencyTextureLogitScale
        high_frequency_texture_routing = [bool]$HighFrequencyTextureRouting
        high_frequency_texture_route_pairs = $HighFrequencyTextureRoutePairs
        high_frequency_texture_route_max_probability_margin = $HighFrequencyTextureRouteMaxProbabilityMargin
        high_frequency_texture_aux_loss_weight = $HighFrequencyTextureAuxLossWeight
        high_frequency_texture_pairwise_loss_weight = $HighFrequencyTexturePairwiseLossWeight
        high_frequency_texture_pairwise_pairs = $HighFrequencyTexturePairwisePairs
        multi_granularity_aux_heads = [bool]$MultiGranularityAuxHeads
        multi_granularity_aux_layers = $MultiGranularityAuxLayers
        multi_granularity_aux_dropout = $MultiGranularityAuxDropout
        multi_granularity_aux_loss_weight = $MultiGranularityAuxLossWeight
        multi_granularity_refinement_loss_weight = $MultiGranularityRefinementLossWeight
        multi_granularity_refinement_temperature = $MultiGranularityRefinementTemperature
        multi_granularity_contrastive_loss_weight = $MultiGranularityContrastiveLossWeight
        multi_granularity_contrastive_temperature = $MultiGranularityContrastiveTemperature
        multi_granularity_contrastive_pairs = $MultiGranularityContrastivePairs
        multi_granularity_contrastive_teacher_min_confidence = $MultiGranularityContrastiveTeacherMinConfidence
        multi_granularity_contrastive_require_agreement = [bool]$MultiGranularityContrastiveRequireAgreement
        multi_granularity_contrastive_weight_mode = $MultiGranularityContrastiveWeightMode
        multi_granularity_contrastive_teacher_confidence_power = $MultiGranularityContrastiveTeacherConfidencePower
        block_local_patch_mixer = [bool]$BlockLocalPatchMixer
        block_local_patch_mixer_layers = $BlockLocalPatchMixerLayers
        block_local_patch_mixer_dropout = $BlockLocalPatchMixerDropout
        block_local_patch_mixer_scale = $BlockLocalPatchMixerScale
        locally_enhanced_ffn = [bool]$LocallyEnhancedFfn
        locally_enhanced_ffn_layers = $LocallyEnhancedFfnLayers
        locally_enhanced_ffn_kernel_size = $LocallyEnhancedFfnKernelSize
        concurrent_local_global_coupling = [bool]$ConcurrentLocalGlobalCoupling
        concurrent_local_global_layers = $ConcurrentLocalGlobalLayers
        concurrent_local_global_dim = $ConcurrentLocalGlobalDim
        concurrent_local_global_kernel_size = $ConcurrentLocalGlobalKernelSize
        gated_relative_position_attention = [bool]$GatedRelativePositionAttention
        gated_relative_position_attention_layers = $GatedRelativePositionAttentionLayers
        gated_relative_position_attention_max_mix = $GatedRelativePositionAttentionMaxMix
        gated_relative_position_attention_locality_strength = $GatedRelativePositionAttentionLocalityStrength
        visual_contrast_attention = [bool]$VisualContrastAttention
        visual_contrast_attention_layers = $VisualContrastAttentionLayers
        visual_contrast_tokens = $VisualContrastTokens
        cross_covariance_attention = [bool]$CrossCovarianceAttention
        cross_covariance_attention_layers = $CrossCovarianceAttentionLayers
        cross_covariance_attention_residual_scale = $CrossCovarianceAttentionResidualScale
        dynamic_graph_mixer = [bool]$DynamicGraphMixer
        dynamic_graph_mixer_layers = $DynamicGraphMixerLayers
        dynamic_graph_mixer_bottleneck_dim = $DynamicGraphMixerBottleneckDim
        dynamic_graph_mixer_k = $DynamicGraphMixerK
        patch_style_recalibration = [bool]$PatchStyleRecalibration
        patch_style_recalibration_layers = $PatchStyleRecalibrationLayers
        layer_token_fusion = [bool]$LayerTokenFusion
        layer_token_fusion_layers = $LayerTokenFusionLayers
        layer_token_fusion_top_k = $LayerTokenFusionTopK
        layer_token_fusion_blend = $LayerTokenFusionBlend
        layer_token_fusion_attention_temperature = $LayerTokenFusionAttentionTemperature
        layer_token_fusion_bbox_weight = $LayerTokenFusionBboxWeight
        layer_token_fusion_foreground_weight = $LayerTokenFusionForegroundWeight
        data_cartography = [bool]$DataCartography
        data_cartography_output = $DataCartographyOutput
        train_args = $TrainArgs
    } | ConvertTo-Json -Depth 6 |
        Set-Content -Path (Join-Path $RunDir "launcher_args.json")

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $nativeErrorCountBefore = $Error.Count
    try {
        & $Python -m trkh.training.train @TrainArgs
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        $newNativeErrorCount = [Math]::Max(0, $Error.Count - $nativeErrorCountBefore)
        if ($newNativeErrorCount -gt 0) {
            $nativeErrors = @($Error | Select-Object -First $newNativeErrorCount)
            [Array]::Reverse($nativeErrors)
            $nativeErrors | Out-String |
                Set-Content -Path (Join-Path $RunDir "train_native_errors.txt")
        }
        throw (
            "Training command exited with code $exitCode. " +
            "Xem train_native_errors.txt neu native stderr khong hien trong transcript."
        )
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
