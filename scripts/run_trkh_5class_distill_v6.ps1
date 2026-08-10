param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RunName = "mango_cls_256_5class_v4_mobilenet_distill_v6_30e",
    [int]$Epochs = 30,
    [int]$Patience = 3,
    [int]$BatchSize = 48,
    [int]$NumWorkers = 4,
    [int]$EvalNumWorkers = 2,
    [bool]$PretrainedDistillation = $true,
    [string]$TeacherCheckpoint = "D:\DataAI\AIEx\image_baseline_experiments\outputs\mobilenetv3\best.pt",
    [string]$DistillationTeacherCsv = "",
    [double]$DistillationWeight = 0.08,
    [double]$DistillationTemperature = 2.0,
    [int]$DistillationFocusClassIndex = 1,
    [double]$DistillationFocusClassWeight = 1.5,
    [int]$MaxTrainBatches = 0,
    [int]$MaxValBatches = 0,
    [bool]$TraceArchitecture = $true,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ($Smoke -and -not $PSBoundParameters.ContainsKey("RunName")) {
    $RunName = "smoke_trkh_v4_mobilenet_distill_v6_script"
}
$RunDir = Join-Path $ProjectRoot "runs\$RunName"

Set-Location $ProjectRoot
$env:PYTHONPATH = $ProjectRoot
$env:TRKH_AMP_DTYPE = "bf16"
$env:OMP_NUM_THREADS = "4"
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = "1"
Remove-Item Env:TRKH_ALLOW_WINDOWS_PIN_MEMORY -ErrorAction SilentlyContinue
Remove-Item Env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS -ErrorAction SilentlyContinue

if ($Smoke) {
    $Epochs = 1
    if ($MaxTrainBatches -le 0) {
        $MaxTrainBatches = 2
    }
    if ($MaxValBatches -le 0) {
        $MaxValBatches = 2
    }
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$startedAt = Get-Date
$exitCode = 1
$transcriptStarted = $false

try {
    Start-Transcript -Path (Join-Path $RunDir "launcher_transcript.txt") -Append | Out-Null
    $transcriptStarted = $true

    $TrainArgs = @(
        "--data", "D:\DataAI\AIEx\newdataset\class_f\data.yaml",
        "--class-name-mode", "raw",
        "--expected-num-classes", "5",
        "--run-name", $RunName,
        "--output-dir", "runs",
        "--resume", "runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\checkpoints\best.pt",
        "--resume-use-cli-config",
        "--resume-reset-epoch",
        "--resume-reset-optimizer",
        "--resume-reset-scheduler",
        "--resume-reset-scaler",
        "--seed", "42",
        "--model-type", "vit_registers",
        "--no-pretrained",
        "--image-size", "256",
        "--patch-size", "16",
        "--stem-channels", "32",
        "--cnn-feature-fusion",
        "--cnn-fusion-dropout", "0.10",
        "--fine-grained-pooling",
        "--fine-grained-pooling-dropout", "0.08",
        "--multi-branch-fusion",
        "--branch-color-tokens", "1",
        "--branch-edge-tokens", "1",
        "--branch-cnn-tokens", "0",
        "--branch-token-dropout", "0.08",
        "--detail-patch-enhancement",
        "--detail-patch-dropout", "0.05",
        "--token-pruning",
        "--token-prune-layers", "2,5",
        "--token-keep-rates", "0.85,0.65",
        "--token-prune-foreground-weight", "0.45",
        "--pairwise-margin-head",
        "--pairwise-margin-pairs", "0-1,1-2,2-3,4-rest",
        "--pairwise-margin-logit-scale", "0.25",
        "--pairwise-margin-dropout", "0.05",
        "--embed-dim", "256",
        "--depth", "8",
        "--num-heads", "8",
        "--num-registers", "4",
        "--register-positional-embedding",
        "--head-pooling", "cls_branch_register_mean",
        "--dropout", "0.12",
        "--attention-dropout", "0.03",
        "--drop-path-rate", "0.10",
        "--batch-size", "$BatchSize",
        "--grad-accum-steps", "1",
        "--epochs", "$Epochs",
        "--scheduler-total-epochs", "$Epochs",
        "--patience", "$Patience",
        "--learning-rate", "8e-5",
        "--min-learning-rate", "1e-6",
        "--warmup-epochs", "1",
        "--weight-decay", "0.05",
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
        "--classification-loss", "ldam_focal",
        "--ldam-max-margin", "0.30",
        "--ldam-scale", "18",
        "--focal-loss-gamma", "1.0",
        "--focal-loss-mix", "0.10",
        "--label-smoothing", "0.02",
        "--metric-learning-loss-weight", "0.04",
        "--metric-learning-temperature", "0.16",
        "--metric-learning-sources", "head,patch",
        "--foreground-consistency-loss-weight", "0.025",
        "--foreground-consistency-margin", "0.07",
        "--register-diversity-loss-weight", "0.0",
        "--pairwise-margin-loss-weight", "0.04",
        "--hard-sample-manifest", "runs\mango_cls_256_5class_defectstat_v3_30e\hard_mining_train_only\hard_samples_train_only.csv",
        "--hard-sample-repeat-factor", "1.6",
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

    if (-not [string]::IsNullOrWhiteSpace($DistillationTeacherCsv)) {
        $TrainArgs += @(
            "--distillation-teacher-csv", $DistillationTeacherCsv,
            "--distillation-weight", "$DistillationWeight",
            "--distillation-temperature", "$DistillationTemperature",
            "--distillation-focus-class-index", "$DistillationFocusClassIndex",
            "--distillation-focus-class-weight", "$DistillationFocusClassWeight"
        )
    }
    elseif ($PretrainedDistillation) {
        $TrainArgs += @(
            "--pretrained-distillation",
            "--distillation-teacher-checkpoint", $TeacherCheckpoint,
            "--distillation-weight", "$DistillationWeight",
            "--distillation-temperature", "$DistillationTemperature",
            "--distillation-focus-class-index", "$DistillationFocusClassIndex",
            "--distillation-focus-class-weight", "$DistillationFocusClassWeight"
        )
    }
    else {
        $TrainArgs += @("--no-pretrained-distillation")
    }

    if ($MaxTrainBatches -gt 0) {
        $TrainArgs += @("--max-train-batches", "$MaxTrainBatches")
    }
    if ($MaxValBatches -gt 0) {
        $TrainArgs += @("--max-val-batches", "$MaxValBatches")
    }
    if ($Smoke) {
        $TrainArgs += @("--skip-final-test")
    }
    if ($TraceArchitecture) {
        $TrainArgs += @(
            "--trace-architecture",
            "--trace-architecture-seed", "42",
            "--trace-architecture-device", "cuda"
        )
    }

    @{
        run_name = $RunName
        started_at = $startedAt.ToString("o")
        smoke = [bool]$Smoke
        pretrained_distillation = [bool]$PretrainedDistillation
        teacher_checkpoint = $TeacherCheckpoint
        distillation_teacher_csv = $DistillationTeacherCsv
        distillation_weight = $DistillationWeight
        distillation_temperature = $DistillationTemperature
        distillation_focus_class_index = $DistillationFocusClassIndex
        distillation_focus_class_weight = $DistillationFocusClassWeight
        batch_size = $BatchSize
        num_workers = $NumWorkers
        eval_num_workers = $EvalNumWorkers
        epochs = $Epochs
        patience = $Patience
        max_train_batches = $MaxTrainBatches
        max_val_batches = $MaxValBatches
        trace_architecture = [bool]$TraceArchitecture
        train_args = $TrainArgs
    } |
        ConvertTo-Json -Depth 6 |
        Set-Content -Path (Join-Path $RunDir "launcher_args.json")

    & $Python -m trkh.training.train @TrainArgs
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Training command exited with code $exitCode"
    }
}
catch {
    $_ | Out-String | Set-Content -Path (Join-Path $RunDir "launcher_exception.txt")
    if ($exitCode -eq 0) {
        $exitCode = 1
    }
}
finally {
    @{
        run_name = $RunName
        started_at = $startedAt.ToString("o")
        finished_at = (Get-Date).ToString("o")
        exit_code = $exitCode
    } |
        ConvertTo-Json |
        Set-Content -Path (Join-Path $RunDir "launcher_status.json")

    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}

exit $exitCode
