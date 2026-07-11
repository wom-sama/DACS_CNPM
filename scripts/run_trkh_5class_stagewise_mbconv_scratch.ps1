param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
    [string]$RunName = "probe_v8_yolof_stagewise_mbconv_scratch_120b_5e",
    [int]$Epochs = 5,
    [int]$SchedulerTotalEpochs = 15,
    [int]$Patience = 3,
    [int]$BatchSize = 32,
    [int]$GradAccumSteps = 2,
    [int]$NumWorkers = 4,
    [int]$EvalNumWorkers = 2,
    [int]$MaxTrainBatches = 120,
    [int]$MaxValBatches = 0,
    [double]$LearningRate = 5e-4,
    [double]$MinLearningRate = 1e-6,
    [double]$AttentionViewLossWeight = 0.0,
    [int]$AttentionViewStartEpoch = 2,
    [int]$Seed = 42,
    [bool]$TraceArchitecture = $true,
    [switch]$PreflightOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $PSScriptRoot "run_trkh_5class_attention_views_v8.ps1"
$TeacherCsv = Join-Path $ProjectRoot "runs\weighted_ensemble_no_pretrain_6expert_val_gate_20260701\teacher_probs_train_sampleindex_focus005.csv"

if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
    throw "Base V8 launcher not found: $Launcher"
}
if (-not (Test-Path -LiteralPath $DataYaml -PathType Leaf)) {
    throw "Data YAML not found: $DataYaml"
}
if (-not (Test-Path -LiteralPath $TeacherCsv -PathType Leaf)) {
    throw "Teacher cache not found: $TeacherCsv"
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
if ($AttentionViewLossWeight -lt 0.0 -or $AttentionViewStartEpoch -lt 1) {
    throw "Attention-view settings are invalid."
}

$Args = @{
    Python = $Python
    DataYaml = $DataYaml
    RunName = $RunName
    ResumeCheckpoint = ""
    StemArchitecture = "coatnet_mbconv"
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
    MinLearningRate = $MinLearningRate
    WarmupEpochs = 1
    WeightDecay = 0.05
    BackboneLrScale = 1.0
    ImageSize = 256
    CropMarginRatio = 0.05
    BboxTokenPriorSource = "bbox"
    HardSampleManifest = ""
    HardSampleRepeatFactor = 1.0
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
    AttentionViewLossWeight = $AttentionViewLossWeight
    AttentionCropProbability = 0.40
    AttentionDropProbability = 0.20
    AttentionViewStartEpoch = $AttentionViewStartEpoch
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
