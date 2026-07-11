param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
    [ValidateSet("mobilevit_s", "mobilevit_xs", "mobilevit_xxs", "mobilevitv2_100", "edgenext_small", "edgenext_x_small", "coatnet_nano_rw_224", "efficientformerv2_s0")]
    [string]$ModelName = "mobilevit_s",
    [string]$RunName = "probe_compacthybrid_mobilevit_s_yolof_scratch_120b_5e",
    [int]$Epochs = 5,
    [int]$SchedulerTotalEpochs = 15,
    [int]$Patience = 3,
    [int]$BatchSize = 64,
    [int]$ImageSize = 256,
    [int]$MaxTrainBatches = 120,
    [int]$MaxValBatches = 0,
    [int]$NumWorkers = 4,
    [int]$EvalNumWorkers = 2,
    [int]$Seed = 42,
    [double]$LearningRate = 5e-4,
    [double]$MinLearningRate = 1e-6,
    [double]$WeightDecay = 0.05,
    [string]$TeacherCsv = "D:\DataAI\AIEx\TRKH\runs\weighted_ensemble_no_pretrain_6expert_val_gate_20260701\teacher_probs_train_sampleindex_focus005.csv",
    [double]$TeacherFocusBinaryLossWeight = 0.015,
    [string]$ResumeCheckpoint = "",
    [switch]$DisableTeacherFocusBinary,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $DataYaml -PathType Leaf)) {
    throw "Data YAML not found: $DataYaml"
}
if ($Epochs -lt 1 -or $Epochs -gt 30) {
    throw "Epochs must be in [1, 30]."
}
if ($SchedulerTotalEpochs -lt $Epochs) {
    throw "SchedulerTotalEpochs must be >= Epochs."
}
if ($BatchSize -lt 1 -or $MaxTrainBatches -lt 0 -or $MaxValBatches -lt 0) {
    throw "BatchSize must be positive and max-batch limits must be non-negative."
}
if ($ImageSize -lt 64 -or $ImageSize -gt 1024) {
    throw "ImageSize must be in [64, 1024]."
}
if ($LearningRate -le 0 -or $MinLearningRate -le 0 -or $MinLearningRate -gt $LearningRate) {
    throw "Learning-rate bounds are invalid."
}
$resumeEnabled = -not [string]::IsNullOrWhiteSpace($ResumeCheckpoint)
if ($resumeEnabled -and -not (Test-Path -LiteralPath $ResumeCheckpoint -PathType Leaf)) {
    throw "Resume checkpoint not found: $ResumeCheckpoint"
}

$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = "1"
Remove-Item Env:TRKH_ALLOW_WINDOWS_PIN_MEMORY -ErrorAction SilentlyContinue
Remove-Item Env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS -ErrorAction SilentlyContinue

$TrainArgs = @(
    "-m", "trkh.training.train",
    "--data", $DataYaml,
    "--class-name-mode", "raw",
    "--expected-num-classes", "5",
    "--run-name", $RunName,
    "--output-dir", "runs",
    "--seed", "$Seed",
    "--model-type", "timm_classifier",
    "--timm-model-name", $ModelName,
    "--no-pretrained",
    "--no-pretrained-distillation",
    "--image-size", "$ImageSize",
    "--crop-margin-ratio", "0.05",
    "--class-crop-margin-scale-threshold", "1.5",
    "--class-crop-margin-max-ratio", "0.16",
    "--batch-size", "$BatchSize",
    "--grad-accum-steps", "1",
    "--epochs", "$Epochs",
    "--scheduler-total-epochs", "$SchedulerTotalEpochs",
    "--patience", "$Patience",
    "--learning-rate", "$LearningRate",
    "--min-learning-rate", "$MinLearningRate",
    "--warmup-epochs", "1",
    "--warmup-start-factor", "0.1",
    "--weight-decay", "$WeightDecay",
    "--grad-clip-norm", "0.7",
    "--model-ema",
    "--model-ema-decay", "0.995",
    "--num-workers", "$NumWorkers",
    "--eval-num-workers", "$EvalNumWorkers",
    "--train-image-cache-mb", "0",
    "--eval-image-cache-mb", "0",
    "--classification-loss", "ldam_focal",
    "--focal-loss-gamma", "1.0",
    "--focal-loss-mix", "0.1",
    "--label-smoothing", "0.02",
    "--ldam-max-margin", "0.3",
    "--ldam-scale", "18",
    "--best-metric", "fair_macro_f1",
    "--fair-f1-gap-target", "0.05",
    "--fair-f1-gap-penalty", "1.5",
    "--fair-f1-min-weight", "0.25",
    "--resize-mode", "pad",
    "--train-scale-min", "0.88",
    "--train-scale-crop-probability", "0.35",
    "--brightness", "0.04",
    "--contrast", "0.04",
    "--saturation", "0.02",
    "--hue", "0.01",
    "--illumination-normalization",
    "--illumination-normalization-strength", "0.35",
    "--background-suppression-mode", "desaturate_blur",
    "--background-suppression-probability", "0.8",
    "--background-suppression-margin", "0.08",
    "--background-suppression-blur-radius", "7",
    "--local-exposure-probability", "0.15",
    "--local-exposure-strength", "0.25",
    "--obstacle-probability", "0.04",
    "--obstacle-max-area", "0.08",
    "--disable-class-aware-augmentation",
    "--disable-rare-class-repeat",
    "--batch-mix-probability", "0",
    "--mosaic-probability", "0",
    "--mixup-probability", "0",
    "--cutmix-probability", "0",
    "--copy-paste-probability", "0",
    "--targeted-copy-paste-probability", "0",
    "--max-train-batches", "$MaxTrainBatches",
    "--max-val-batches", "$MaxValBatches",
    "--skip-final-test"
)

if ($resumeEnabled) {
    $TrainArgs += @(
        "--resume", (Resolve-Path -LiteralPath $ResumeCheckpoint).Path,
        "--resume-use-cli-config"
    )
}

$teacherEnabled = -not $DisableTeacherFocusBinary.IsPresent -and $TeacherFocusBinaryLossWeight -gt 0
if ($teacherEnabled) {
    if (-not (Test-Path -LiteralPath $TeacherCsv -PathType Leaf)) {
        throw "Teacher CSV not found: $TeacherCsv"
    }
    $TrainArgs += @(
        "--distillation-teacher-csv", $TeacherCsv,
        "--distillation-weight", "0",
        "--distillation-temperature", "2",
        "--distillation-focus-class-index", "1",
        "--distillation-focus-class-weight", "1",
        "--teacher-focus-binary-loss-weight", "$TeacherFocusBinaryLossWeight",
        "--teacher-focus-binary-focus-class", "1",
        "--teacher-focus-binary-classes", "0,1,2,4",
        "--teacher-focus-binary-teacher-min-confidence", "0",
        "--teacher-focus-binary-error-power", "0"
    )
}

$launch = [ordered]@{
    run_name = $RunName
    model_name = $ModelName
    pretrained = $false
    data_yaml = (Resolve-Path -LiteralPath $DataYaml).Path
    epochs = $Epochs
    scheduler_total_epochs = $SchedulerTotalEpochs
    patience = $Patience
    batch_size = $BatchSize
    image_size = $ImageSize
    max_train_batches = $MaxTrainBatches
    max_val_batches = $MaxValBatches
    learning_rate = $LearningRate
    min_learning_rate = $MinLearningRate
    num_workers = $NumWorkers
    eval_num_workers = $EvalNumWorkers
    teacher_focus_binary_enabled = $teacherEnabled
    teacher_focus_binary_loss_weight = if ($teacherEnabled) { $TeacherFocusBinaryLossWeight } else { 0.0 }
    teacher_csv = if ($teacherEnabled) { $TeacherCsv } else { "" }
    resume_enabled = $resumeEnabled
    resume_checkpoint = if ($resumeEnabled) { (Resolve-Path -LiteralPath $ResumeCheckpoint).Path } else { "" }
    resume_policy = if ($resumeEnabled) { "preserve_epoch_optimizer_scheduler_scaler_ema" } else { "fresh" }
    skip_final_test = $true
    train_args = $TrainArgs
}

if ($DryRun) {
    $launch | ConvertTo-Json -Depth 6
    exit 0
}

$RunDir = Join-Path $ProjectRoot "runs\$RunName"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$launch | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $RunDir "launcher_args.json") -Encoding UTF8
$startedAt = Get-Date
$exitCode = 1
$transcriptStarted = $false

try {
    Start-Transcript -Path (Join-Path $RunDir "launcher_transcript.txt") -Append | Out-Null
    $transcriptStarted = $true
    $status = [ordered]@{
        state = "running"
        started_at = $startedAt.ToString("o")
        run_name = $RunName
        model_name = $ModelName
        resume_checkpoint = if ($resumeEnabled) { (Resolve-Path -LiteralPath $ResumeCheckpoint).Path } else { "" }
        skip_final_test = $true
    }
    $status | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $RunDir "launcher_status.json") -Encoding UTF8

    Push-Location $ProjectRoot
    try {
        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Python @TrainArgs
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $oldErrorActionPreference
    }
    finally {
        Pop-Location
    }
}
finally {
    $finishedAt = Get-Date
    $finalStatus = [ordered]@{
        state = if ($exitCode -eq 0) { "completed" } else { "failed" }
        started_at = $startedAt.ToString("o")
        finished_at = $finishedAt.ToString("o")
        elapsed_seconds = [math]::Round(($finishedAt - $startedAt).TotalSeconds, 3)
        exit_code = $exitCode
        run_name = $RunName
        model_name = $ModelName
        resume_checkpoint = if ($resumeEnabled) { (Resolve-Path -LiteralPath $ResumeCheckpoint).Path } else { "" }
        skip_final_test = $true
    }
    $finalStatus | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $RunDir "launcher_status.json") -Encoding UTF8
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}

exit $exitCode
