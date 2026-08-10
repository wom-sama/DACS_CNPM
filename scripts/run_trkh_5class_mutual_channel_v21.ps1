param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\class_f\data.yaml",
    [string]$RunName = "mango_cls_256_5class_mutual_channel_v21_30e",
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
    [double]$MutualChannelLossWeight = 0.04,
    [int]$MutualChannelTopK = 8,
    [double]$MutualChannelDiversityWeight = 0.20,
    [int]$MutualChannelStartEpoch = 1,
    [switch]$Probe,
    [switch]$PreflightOnly,
    [switch]$Smoke,
    [switch]$SkipFinalTest,
    [bool]$TraceArchitecture = $true
)

$launcher = Join-Path $PSScriptRoot "run_trkh_5class_routed_pairwise_v16.ps1"

if ($Probe) {
    if (-not $PSBoundParameters.ContainsKey("RunName")) {
        $RunName = "probe_mango_cls_256_5class_mutual_channel_v21_120b_6e"
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

& $launcher `
    -Python $Python `
    -DataYaml $DataYaml `
    -RunName $RunName `
    -ResumeCheckpoint $ResumeCheckpoint `
    -SampleWeightManifest $SampleWeightManifest `
    -Epochs $Epochs `
    -Patience $Patience `
    -BatchSize $BatchSize `
    -GradAccumSteps $GradAccumSteps `
    -NumWorkers $NumWorkers `
    -EvalNumWorkers $EvalNumWorkers `
    -MaxTrainBatches $MaxTrainBatches `
    -MaxValBatches $MaxValBatches `
    -BoundaryContrastiveLossWeight $BoundaryContrastiveLossWeight `
    -PairwiseMarginRouteMaxProbabilityMargin $PairwiseMarginRouteMaxProbabilityMargin `
    -ClassificationLoss "ldam_focal" `
    -MutualChannelLossWeight $MutualChannelLossWeight `
    -MutualChannelTopK $MutualChannelTopK `
    -MutualChannelDiversityWeight $MutualChannelDiversityWeight `
    -MutualChannelStartEpoch $MutualChannelStartEpoch `
    -PreflightOnly:$PreflightOnly `
    -Smoke:$Smoke `
    -SkipFinalTest:$SkipFinalTest `
    -TraceArchitecture $TraceArchitecture
exit $LASTEXITCODE
