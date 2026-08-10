param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunName = "mango_cls_256_5class_ordinal_maskaudit_v5_30e"
$RunDir = Join-Path $ProjectRoot "runs\$RunName"

Set-Location $ProjectRoot
$env:PYTHONPATH = $ProjectRoot
$env:TRKH_AMP_DTYPE = "bf16"
$env:OMP_NUM_THREADS = "4"
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = "1"
Remove-Item Env:TRKH_ALLOW_WINDOWS_PIN_MEMORY -ErrorAction SilentlyContinue
Remove-Item Env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$startedAt = Get-Date
$exitCode = 1
try {
    $TrainArgs = @(
        "--data", "D:\DataAI\AIEx\newdataset\class_f\data.yaml",
        "--class-name-mode", "raw",
        "--expected-num-classes", "5",
        "--run-name", $RunName,
        "--output-dir", "runs",
        "--disable-resume",
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
        "--pairwise-margin-pairs", "4-rest",
        "--pairwise-margin-logit-scale", "0.25",
        "--pairwise-margin-dropout", "0.05",
        "--ordinal-maturity-head",
        "--ordinal-maturity-classes", "0,1,2,3",
        "--ordinal-maturity-logit-scale", "0.20",
        "--ordinal-maturity-dropout", "0.05",
        "--embed-dim", "256",
        "--depth", "8",
        "--num-heads", "8",
        "--num-registers", "4",
        "--register-positional-embedding",
        "--head-pooling", "cls_branch_register_mean",
        "--dropout", "0.12",
        "--attention-dropout", "0.03",
        "--drop-path-rate", "0.10",
        "--batch-size", "64",
        "--grad-accum-steps", "1",
        "--epochs", "30",
        "--scheduler-total-epochs", "30",
        "--patience", "3",
        "--learning-rate", "2.5e-4",
        "--min-learning-rate", "1e-6",
        "--warmup-epochs", "4",
        "--weight-decay", "0.05",
        "--grad-clip-norm", "0.7",
        "--max-nonfinite-grad-steps", "4",
        "--model-ema",
        "--model-ema-decay", "0.995",
        "--num-workers", "4",
        "--eval-num-workers", "2",
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
        "--ordinal-maturity-loss-weight", "0.08",
        "--hard-sample-manifest", "runs\mango_cls_256_5class_defectstat_v3_30e\hard_mining_train_only\hard_samples_train_only.csv",
        "--hard-sample-repeat-factor", "1.6",
        "--resize-mode", "pad",
        "--train-scale-min", "0.88",
        "--train-scale-crop-probability", "0.35",
        "--brightness", "0.03",
        "--contrast", "0.03",
        "--saturation", "0.01",
        "--hue", "0.005",
        "--illumination-normalization",
        "--illumination-normalization-strength", "0.25",
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
        "--targeted-copy-paste-probability", "0",
        "--trace-architecture",
        "--trace-architecture-seed", "42",
        "--trace-architecture-device", "cuda"
    )
    & $Python -m trkh.training.train @TrainArgs
    $exitCode = $LASTEXITCODE
}
catch {
    $_ | Out-String | Set-Content -Path (Join-Path $RunDir "launcher_exception.txt")
    $exitCode = 1
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
}

exit $exitCode
