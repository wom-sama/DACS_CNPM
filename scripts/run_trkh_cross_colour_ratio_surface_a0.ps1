param(
    [ValidateSet("Preflight", "Formal", "Replay")]
    [string]$Mode = "Preflight",
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$Authorization = (
        "docs\TRKH_5CLASS_CROSS_COLOUR_RATIO_SURFACE_A0_" +
        "FIT_RECOVERY_AUTHORIZATION_20260725.json"
    ),
    [string]$CacheDir = (
        "runs\audit_cross_colour_ratio_surface_a0_" +
        "materialized_20260725"
    ),
    [string]$OutputDir = (
        "runs\audit_cross_colour_ratio_surface_a0_20260725"
    )
)

$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable does not exist: $Python"
}

$RunnerArgs = @(
    "-m",
    "trkh.tools.audit_cross_colour_ratio_surface_a0",
    "--cache-dir",
    $CacheDir,
    "--output-dir",
    $OutputDir
)

switch ($Mode) {
    "Preflight" {
        $RunnerArgs += "--preflight-only"
    }
    "Formal" {
        if (-not (Test-Path -LiteralPath $Authorization -PathType Leaf)) {
            throw "CCR fit authorization does not exist: $Authorization"
        }
        $RunnerArgs += @("--formal", "--authorization", $Authorization)
    }
    "Replay" {
        if (-not (Test-Path -LiteralPath $Authorization -PathType Leaf)) {
            throw "CCR fit authorization does not exist: $Authorization"
        }
        $RunnerArgs += @("--replay", "--authorization", $Authorization)
    }
}

& $Python @RunnerArgs
if ($LASTEXITCODE -ne 0) {
    throw "CCR $Mode failed with exit code $LASTEXITCODE"
}
