[CmdletBinding()]
param(
    [switch]$Run,
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "",
    [string]$Checkpoint = "",
    [string]$OutputDir = "",
    [ValidateRange(1, 256)][int]$BatchSize = 32,
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
if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    $Checkpoint = Join-Path $RepoRoot "runs\pretrained_dinov3_classf_tempered_p05_probe_20260731_local_b2_tempered_p05\checkpoints\best.pt"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot "runs\precheck_classf_b6_dinov3_classconditional_deepsets_20260731"
}

foreach ($RequiredFile in @($Python, $DataYaml, $Checkpoint)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required B6 file is missing: $RequiredFile"
    }
}

$Arguments = @(
    "-m", "trkh.tools.precheck_dinov3_classconditional_deepsets_train_oof",
    "--data", $DataYaml,
    "--checkpoint", $Checkpoint,
    "--output-dir", $OutputDir,
    "--class-name-mode", "raw",
    "--batch-size", [string]$BatchSize,
    "--workers", [string]$Workers,
    "--torch-threads", [string]$TorchThreads,
    "--device", $Device
)

if (-not $Run) {
    $Arguments += "--preflight-only"
    Write-Host "B6 mode: TRAIN-ONLY PREFLIGHT (feature extraction and head training are disabled)."
} else {
    $OutputDrive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($OutputDir))
    $MinimumFreeBytes = 4GB
    if ($OutputDrive.AvailableFreeSpace -lt $MinimumFreeBytes) {
        throw (
            "B6 requires at least 4 GiB free for its 1.52 GiB token cache and " +
            "atomic artifacts; available bytes: $($OutputDrive.AvailableFreeSpace)"
        )
    }
    Write-Host "B6 mode: EXPLICIT TRAIN-ONLY READINESS RUN."
    Write-Host "No validation/test dataset or full-training path exists in this launcher."
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
        throw "B6 readiness tool failed with exit code $LASTEXITCODE"
    }
} finally {
    $env:PYTHONPATH = $PreviousPythonPath
}

if (-not $Run) {
    Write-Host "Stopped after preflight. Review preflight.json and disk capacity, then pass -Run explicitly."
}
