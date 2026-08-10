[CmdletBinding()]
param(
    [switch]$Run,
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "",
    [string]$B2Checkpoint = "",
    [string]$B2CacheDir = "",
    [string]$DinoCheckpoint = "",
    [string]$OutputDir = "",
    [ValidateRange(1, 128)][int]$BatchSize = 24,
    [ValidateRange(0, 16)][int]$Workers = 0,
    [ValidateRange(1, 32)][int]$TorchThreads = 4,
    [string]$Device = "cuda"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($DataYaml)) {
    $DataYaml = Join-Path $RepoRoot "configs\class_f_5class_dev.yaml"
}
if ([string]::IsNullOrWhiteSpace($B2Checkpoint)) {
    $B2Checkpoint = Join-Path $RepoRoot "runs\pretrained_dinov3_classf_tempered_p05_probe_20260731_local_b2_tempered_p05\checkpoints\best.pt"
}
if ([string]::IsNullOrWhiteSpace($B2CacheDir)) {
    $B2CacheDir = Join-Path $RepoRoot "runs\precheck_classf_b7_dinov3_prefix_residual_20260801_run4"
}
if ([string]::IsNullOrWhiteSpace($DinoCheckpoint)) {
    $DinoCheckpoint = "C:\Users\ADMIN\.cache\huggingface\hub\models--timm--vit_small_patch16_dinov3.lvd1689m\snapshots\3bf4720a82ec2066db88137180ff1f83a675cef0\model.safetensors"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot "runs\precheck_classf_b8_frozen_dino_np_2to1_20260801"
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)

foreach ($RequiredFile in @($Python, $DataYaml, $B2Checkpoint, $DinoCheckpoint)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required B8 file is missing: $RequiredFile"
    }
}
if (-not (Test-Path -LiteralPath $B2CacheDir -PathType Container)) {
    throw "Required B8 B2-cache directory is missing: $B2CacheDir"
}

$Arguments = @(
    "-m", "trkh.tools.precheck_dinov3_frozen_np_2to1_train_oof",
    "--data", $DataYaml,
    "--b2-checkpoint", $B2Checkpoint,
    "--b2-cache-dir", $B2CacheDir,
    "--dino-checkpoint", $DinoCheckpoint,
    "--output-dir", $OutputDir,
    "--class-name-mode", "raw",
    "--batch-size", [string]$BatchSize,
    "--workers", [string]$Workers,
    "--torch-threads", [string]$TorchThreads,
    "--device", $Device
)

if (-not $Run) {
    $Arguments += "--preflight-only"
    Write-Host "B8 mode: TRAIN-ONLY PREFLIGHT (dataset feature extraction is disabled)."
} else {
    $OutputDrive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($OutputDir))
    if ($OutputDrive.AvailableFreeSpace -lt 1GB) {
        throw "B8 requires at least 1 GiB free for atomic cache/artifacts."
    }
    Write-Host "B8 mode: EXPLICIT TRAIN-ONLY FROZEN-DINO READINESS RUN."
    Write-Host "No validation/test/full-training path exists in this launcher."
}

$PreviousPythonPath = $env:PYTHONPATH
try {
    if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
        $env:PYTHONPATH = $RepoRoot
    } else {
        $env:PYTHONPATH = "$RepoRoot;$PreviousPythonPath"
    }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "B8 readiness tool failed with exit code $LASTEXITCODE"
    }
} finally {
    $env:PYTHONPATH = $PreviousPythonPath
}

if (-not $Run) {
    Write-Host "Stopped after preflight. Review preflight.json, then pass -Run explicitly."
}
