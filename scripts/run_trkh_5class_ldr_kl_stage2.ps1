param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
    [string]$RunName = "smoke_v8_yolof_keeper_ldrkl_m200_t100_60b_1e_20260711",
    [string]$ResumeCheckpoint = "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt",
    [int]$Epochs = 1,
    [int]$SchedulerTotalEpochs = 15,
    [int]$Patience = 3,
    [int]$BatchSize = 32,
    [int]$GradAccumSteps = 2,
    [int]$NumWorkers = 4,
    [int]$EvalNumWorkers = 2,
    [int]$MaxTrainBatches = 60,
    [int]$MaxValBatches = 0,
    [double]$LearningRate = 8e-5,
    [double]$LdrMargin = 2.0,
    [double]$LdrTemperature = 1.0,
    [int]$Seed = 42,
    [bool]$TraceArchitecture = $true,
    [switch]$PreflightOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $PSScriptRoot "run_trkh_5class_attention_views_v8.ps1"
$TeacherCsv = Join-Path $ProjectRoot "runs\weighted_ensemble_no_pretrain_6expert_val_gate_20260701\teacher_probs_train_sampleindex_focus005.csv"

foreach ($RequiredFile in @($Launcher, $DataYaml, $ResumeCheckpoint, $TeacherCsv)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required file not found: $RequiredFile"
    }
}
if ($Epochs -lt 1 -or $Epochs -gt 30) {
    throw "Epochs must be in [1, 30]."
}
if ($SchedulerTotalEpochs -lt $Epochs) {
    throw "SchedulerTotalEpochs must be at least Epochs."
}
if ($MaxTrainBatches -lt 0 -or $MaxValBatches -lt 0) {
    throw "Batch limits must be non-negative."
}
if ($LdrMargin -lt 0.0 -or $LdrTemperature -le 0.0) {
    throw "LdrMargin must be non-negative and LdrTemperature must be positive."
}

$Args = @{
    Python = $Python
    DataYaml = $DataYaml
    RunName = $RunName
    ResumeCheckpoint = $ResumeCheckpoint
    Epochs = $Epochs
    SchedulerTotalEpochs = $SchedulerTotalEpochs
    Patience = $Patience
    BatchSize = $BatchSize
    GradAccumSteps = $GradAccumSteps
    NumWorkers = $NumWorkers
    EvalNumWorkers = $EvalNumWorkers
    MaxTrainBatches = $MaxTrainBatches
    MaxValBatches = $MaxValBatches
    Seed = $Seed
    LearningRate = $LearningRate
    MinLearningRate = 1e-6
    WarmupEpochs = 1
    WeightDecay = 0.05
    BackboneLrScale = 1.0
    ImageSize = 256
    CropMarginRatio = 0.05
    BboxTokenPriorSource = "bbox"
    HardSampleManifest = ""
    HardSampleRepeatFactor = 1.0
    ClassificationLoss = "ldr_kl"
    LdrMargin = $LdrMargin
    LdrTemperature = $LdrTemperature
    LabelSmoothing = 0.02
    DistillationTeacherCsv = $TeacherCsv
    DistillationWeight = 0.0
    DistillationFocusClassWeight = 1.0
    TeacherFocusBinaryLossWeight = 0.015
    TeacherFocusBinaryFocusClass = 1
    TeacherFocusBinaryClasses = "0,1,2,4"
    TeacherFocusBinaryTeacherMinConfidence = 0.0
    TeacherFocusBinaryErrorPower = 0.0
    BBoxSpatialFusion = $true
    BBoxSpatialFusionHiddenDim = 64
    BBoxSpatialFusionDropout = 0.05
    BBoxSpatialFusionLogitScale = 0.20
    PairwiseMarginRouting = $true
    PairwiseMarginPairs = "0-1,1-2,2-3,4-rest"
    PairwiseMarginRouteMaxProbabilityMargin = 0.20
    MetricLearningLossWeight = 0.04
    MetricLearningTemperature = 0.16
    MetricLearningSources = "head,patch"
    BBoxForegroundDropoutLossWeight = 0.08
    BBoxForegroundDropoutConsistencyWeight = 0.03
    BBoxForegroundDropoutProbability = 0.45
    BBoxForegroundDropoutMinAreaRatio = 0.04
    BBoxForegroundDropoutMaxAreaRatio = 0.14
    BBoxForegroundDropoutMode = "boundary_band"
    BBoxForegroundDropoutFill = "mean"
    BBoxForegroundDropoutTemperature = 1.2
    AttentionViewLossWeight = 0.35
    AttentionCropProbability = 0.40
    AttentionDropProbability = 0.20
    AttentionViewStartEpoch = 2
    AttentionViewScoreSource = "learned_attention"
    AttentionViewForegroundWeight = 0.40
    AttentionDropMinAreaRatio = 0.06
    AttentionDropMaxAreaRatio = 0.16
    SkipFinalTest = $true
    TraceArchitecture = $TraceArchitecture
}
if ($PreflightOnly.IsPresent) {
    $Args.PreflightOnly = $true
}
if ($DryRun.IsPresent) {
    $Args.DryRun = $true
}

& $Launcher @Args
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
