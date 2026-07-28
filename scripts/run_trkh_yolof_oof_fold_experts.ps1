param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$FoldsRoot = "runs\yolof_oof_folds_train5_20260702",
    [string]$Folds = "0,1,2,3,4",
    [string]$RunPrefix = "oof_yolof_v8_scratch",
    [string]$ResumeCheckpoint = "",
    [switch]$AllowUnsafeResume,
    [int]$Epochs = 30,
    [int]$Patience = 3,
    [int]$BatchSize = 24,
    [int]$GradAccumSteps = 2,
    [int]$NumWorkers = 2,
    [int]$EvalNumWorkers = 1,
    [int]$MaxTrainBatches = 0,
    [int]$MaxValBatches = 0,
    [double]$LearningRate = 8e-5,
    [double]$MinLearningRate = 1e-6,
    [int]$WarmupEpochs = 1,
    [double]$WeightDecay = 0.05,
    [int]$EvalBatchSize = 64,
    [ValidateSet("last", "best")]
    [string]$ExportCheckpoint = "last",
    [switch]$Smoke,
    [switch]$SkipFinalTest,
    [switch]$NoExportValPredictions,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Khong tim thay Python: $Python"
}
if (-not (Test-Path -LiteralPath $FoldsRoot)) {
    throw "Khong tim thay FoldsRoot: $FoldsRoot"
}
if (-not [string]::IsNullOrWhiteSpace($ResumeCheckpoint) -and -not $AllowUnsafeResume) {
    throw "OOF fold experts khong duoc resume tu checkpoint full-train neu khong explicit -AllowUnsafeResume. Dung ResumeCheckpoint='' cho OOF hop le."
}
if (-not [string]::IsNullOrWhiteSpace($ResumeCheckpoint) -and -not (Test-Path -LiteralPath $ResumeCheckpoint)) {
    throw "Khong tim thay ResumeCheckpoint: $ResumeCheckpoint"
}

$TrainScript = Join-Path $PSScriptRoot "run_trkh_5class_attention_views_v8.ps1"
if (-not (Test-Path -LiteralPath $TrainScript)) {
    throw "Khong tim thay train launcher: $TrainScript"
}

$FoldIds = @()
foreach ($Item in ($Folds -split ",")) {
    $Text = $Item.Trim()
    if ($Text.Length -eq 0) {
        continue
    }
    $FoldIds += [int]$Text
}
if ($FoldIds.Count -eq 0) {
    throw "Danh sach Folds rong."
}

$Planned = @()
foreach ($FoldId in $FoldIds) {
    $FoldName = "fold_{0:D2}" -f $FoldId
    $DataYaml = Join-Path (Join-Path $FoldsRoot $FoldName) "data.yaml"
    if (-not (Test-Path -LiteralPath $DataYaml)) {
        throw "Khong tim thay fold data.yaml: $DataYaml"
    }

    $RunName = "{0}_{1}" -f $RunPrefix, $FoldName
    $TrainParams = @{
        DataYaml = $DataYaml
        RunName = $RunName
        ResumeMode = if ([string]::IsNullOrWhiteSpace($ResumeCheckpoint)) { "Scratch" } else { "WarmStart" }
        ResumeCheckpoint = $ResumeCheckpoint
        Epochs = $Epochs
        Patience = $Patience
        BatchSize = $BatchSize
        GradAccumSteps = $GradAccumSteps
        NumWorkers = $NumWorkers
        EvalNumWorkers = $EvalNumWorkers
        MaxTrainBatches = $MaxTrainBatches
        MaxValBatches = $MaxValBatches
        LearningRate = $LearningRate
        MinLearningRate = $MinLearningRate
        WarmupEpochs = $WarmupEpochs
        WeightDecay = $WeightDecay
        BBoxSpatialFusion = $true
        BBoxSpatialFusionLogitScale = 0.20
        PairwiseMarginHead = $true
        PairwiseMarginPairs = "0-1,1-2,2-3,4-rest"
        PairwiseMarginLogitScale = 0.25
        PairwiseMarginRouting = $true
        PairwiseMarginRouteMaxProbabilityMargin = 0.20
        MetricLearningLossWeight = 0.04
        MetricLearningTemperature = 0.16
        MetricLearningSources = "head,patch"
    }
    if ($Smoke) {
        $TrainParams["Smoke"] = $true
    }
    if ($SkipFinalTest) {
        $TrainParams["SkipFinalTest"] = $true
    }

    $ExportOutputDir = Join-Path (Join-Path "runs" $RunName) "eval_val_oof_export"
    $CheckpointName = if ($ExportCheckpoint -eq "best") { "best.pt" } else { "last.pt" }
    $ExportArgs = @(
        "-m", "trkh.evaluation.evaluate",
        "--checkpoint", (Join-Path (Join-Path (Join-Path "runs" $RunName) "checkpoints") $CheckpointName),
        "--data", $DataYaml,
        "--split", "val",
        "--expected-num-classes", "5",
        "--output-dir", $ExportOutputDir,
        "--batch-size", "$EvalBatchSize",
        "--num-workers", "$EvalNumWorkers",
        "--bbox-token-prior-source", "bbox"
    )

    $Planned += [PSCustomObject]@{
        fold = $FoldName
        data_yaml = $DataYaml
        run_name = $RunName
        train_script = $TrainScript
        train_params = $TrainParams
        export_predictions = -not $NoExportValPredictions
        export_checkpoint = $ExportCheckpoint
        export_args = $ExportArgs
        prediction_csv = (Join-Path $ExportOutputDir "predictions_detailed.csv")
    }
}

if ($DryRun) {
    $Planned | ConvertTo-Json -Depth 6
    return
}

foreach ($Plan in $Planned) {
    Write-Host ("=== Training {0} -> {1} ===" -f $Plan.fold, $Plan.run_name)
    $CurrentTrainParams = @{}
    foreach ($Key in $Plan.train_params.Keys) {
        $CurrentTrainParams[$Key] = $Plan.train_params[$Key]
    }
    & $TrainScript @CurrentTrainParams
    if ($LASTEXITCODE -ne 0) {
        throw "Train failed for $($Plan.fold) with exit code $LASTEXITCODE"
    }
    if ($Plan.export_predictions) {
        Write-Host ("=== Exporting fold-val predictions for {0} from {1}.pt ===" -f $Plan.fold, $Plan.export_checkpoint)
        $CurrentExportArgs = foreach ($Arg in $Plan.export_args) { $Arg }
        & $Python @CurrentExportArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Evaluate/export failed for $($Plan.fold) with exit code $LASTEXITCODE"
        }
    }
}

$StitchInputs = @()
foreach ($Plan in $Planned) {
    if ($Plan.export_predictions) {
        $StitchInputs += ("--input {0}={1}" -f $Plan.fold, $Plan.prediction_csv)
    }
}
if ($StitchInputs.Count -gt 0) {
    Write-Host "Stitch command:"
    Write-Host ("{0} -m trkh.tools.stitch_yolo_oof_predictions --data D:\DataAI\AIEx\newdataset\yolo_f\data.yaml {1} --output-dir runs\{2}_stitched --source-split train" -f $Python, ($StitchInputs -join " "), $RunPrefix)
}
