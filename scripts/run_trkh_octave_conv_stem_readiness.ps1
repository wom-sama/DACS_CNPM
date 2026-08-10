param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$Checkpoint = "runs\full_v8_yolof_randominit_30e_20260714_105524\checkpoints\best.pt",
    [string]$ResolvedConfig = "runs\full_v8_yolof_randominit_30e_20260714_105524\resolved_config.json",
    [string]$SourceLauncherArgs = "runs\full_v8_yolof_randominit_30e_20260714_105524\launcher_args.json",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
    [string]$OutputDir = "runs\audit_octave_conv_stem_stage_a_20260715",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$ProtocolDocument = "docs\TRKH_5CLASS_OCTAVE_CONV_STEM_READINESS_PROTOCOL_20260715.md"
$LockedCheckpointSha256 = "f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549"
$LockedResolvedConfigSha256 = "c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97"
$LockedSourceLauncherSha256 = "a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6"
$LockedDataSha256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
$LockedProtocolSha256 = "3f83adb7e6bffb9440f15dc6b60f706d80d164041e3bc5f3285a1f06b01be321"

function Assert-FileSha256 {
    param([string]$Path, [string]$Expected)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file not found: $Path"
    }
    $Observed = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($Observed -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path`: $Observed != $Expected"
    }
}

function Invoke-NativePython {
    param([string]$Step, [string[]]$Arguments)
    Write-Host "[$Step] $Python $($Arguments -join ' ')"
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @Arguments
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) {
        throw "$Step failed with exit code $ExitCode. Inspect $OutputDir\summary.json when present."
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
Assert-FileSha256 $Checkpoint $LockedCheckpointSha256
Assert-FileSha256 $ResolvedConfig $LockedResolvedConfigSha256
Assert-FileSha256 $SourceLauncherArgs $LockedSourceLauncherSha256
Assert-FileSha256 $DataYaml $LockedDataSha256
Assert-FileSha256 $ProtocolDocument $LockedProtocolSha256

Invoke-NativePython "import-preflight" @(
    "-m", "trkh.tools.audit_octave_conv_stem_readiness", "--help"
)

Write-Host "OctConv Stage-A preflight passed. Validation and test remain forbidden."
if ($PreflightOnly) {
    return
}

if (Test-Path -LiteralPath $OutputDir) {
    throw "Output directory already exists: $OutputDir"
}

Invoke-NativePython "octave-conv-stage-a" @(
    "-m", "trkh.tools.audit_octave_conv_stem_readiness",
    "--checkpoint", $Checkpoint,
    "--resolved-config", $ResolvedConfig,
    "--source-launcher-args", $SourceLauncherArgs,
    "--data", $DataYaml,
    "--output-dir", $OutputDir,
    "--batch-size", "32",
    "--fp32-batch-size", "2",
    "--num-workers", "4",
    "--seed", "42",
    "--benchmark-repeats", "3",
    "--max-peak-vram-gib", "3.25",
    "--max-runtime-ratio", "1.50",
    "--device", "cuda"
)

Write-Host "OctConv Stage-A gate passed: $OutputDir\summary.json"
