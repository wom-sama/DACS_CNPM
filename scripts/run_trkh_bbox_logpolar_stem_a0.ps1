param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [ValidateSet("Preflight", "Formal", "Replay")]
    [string]$Phase = "Preflight",
    [string]$OutputDir = "runs\audit_bbox_logpolar_stem_a0_20260720"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

function Invoke-LockedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "BBox log-polar A0 command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Get-ComputeProcesses {
    return Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @("python.exe", "pythonw.exe", "trtexec.exe", "ffmpeg.exe")
    } | Select-Object ProcessId, Name, CreationDate, CommandLine
}

function Show-ComputeState {
    $Processes = Get-ComputeProcesses
    if ($Processes) {
        Write-Host "Existing compute processes were detected; none will be stopped:"
        $Processes | Format-Table -AutoSize | Out-Host
    } else {
        Write-Host "No existing Python/TensorRT/FFmpeg process was detected."
    }
    $Memory = Get-CimInstance Win32_OperatingSystem
    $AvailableGiB = [math]::Round($Memory.FreePhysicalMemory / 1MB, 2)
    Write-Host "Available physical memory: $AvailableGiB GiB"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,pstate `
            --format=csv,noheader,nounits
    }
}

function Assert-FormalIsolation {
    $Processes = Get-ComputeProcesses
    if ($Processes) {
        $Processes | Format-Table -AutoSize | Out-Host
        throw "Formal bbox log-polar A0 requires no pre-existing Python/TensorRT/FFmpeg process."
    }
    $Memory = Get-CimInstance Win32_OperatingSystem
    $AvailableGiB = $Memory.FreePhysicalMemory / 1MB
    if ($AvailableGiB -lt 6.0) {
        throw "Formal bbox log-polar A0 requires at least 6 GiB available RAM."
    }
}

$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
$env:OPENBLAS_NUM_THREADS = "8"
$env:NUMEXPR_NUM_THREADS = "8"
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = "1"
$env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS = "1"
Remove-Item Env:TRKH_ALLOW_WINDOWS_PIN_MEMORY -ErrorAction SilentlyContinue

if ($Phase -in @("Preflight", "Formal")) {
    Invoke-LockedPython -Arguments @(
        "-m", "py_compile",
        "trkh\tools\audit_bbox_logpolar_stem_a0.py",
        "tests\test_audit_bbox_logpolar_stem_a0.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pyflakes",
        "trkh\tools\audit_bbox_logpolar_stem_a0.py",
        "tests\test_audit_bbox_logpolar_stem_a0.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pytest",
        "tests\test_audit_bbox_logpolar_stem_a0.py",
        "-q"
    )
}

if ($Phase -eq "Preflight") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Preflight must not create or reuse output: $OutputDir"
    }
    Show-ComputeState
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_bbox_logpolar_stem_a0",
        "--preflight-only",
        "--output-dir", $OutputDir,
        "--batch-size", "64",
        "--num-workers", "4",
        "--torch-threads", "8",
        "--device", "cuda"
    )
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Preflight unexpectedly created output: $OutputDir"
    }
    Write-Host "BBox log-polar A0 preflight passed without opening checkpoint/dataset or creating output."
    exit 0
}

$SummaryPath = Join-Path $OutputDir "summary.json"
if ($Phase -eq "Replay") {
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "BBox log-polar A0 summary is missing: $SummaryPath"
    }
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_bbox_logpolar_stem_a0",
        "--replay-summary", $SummaryPath,
        "--batch-size", "64",
        "--num-workers", "4",
        "--torch-threads", "8",
        "--device", "cuda"
    )
    exit 0
}

if (Test-Path -LiteralPath $OutputDir) {
    throw "Formal bbox log-polar A0 output already exists; rerun is forbidden: $OutputDir"
}
Show-ComputeState
Assert-FormalIsolation
Invoke-LockedPython -Arguments @(
    "-m", "trkh.tools.audit_bbox_logpolar_stem_a0",
    "--output-dir", $OutputDir,
    "--batch-size", "64",
    "--num-workers", "4",
    "--torch-threads", "8",
    "--device", "cuda"
)
Invoke-LockedPython -Arguments @(
    "-m", "trkh.tools.audit_bbox_logpolar_stem_a0",
    "--replay-summary", $SummaryPath,
    "--batch-size", "64",
    "--num-workers", "4",
    "--torch-threads", "8",
    "--device", "cuda"
)
$Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
$SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
$ManifestSha = (Get-FileHash -Algorithm SHA256 -LiteralPath (
    Join-Path $OutputDir "artifact_manifest.json"
)).Hash.ToLowerInvariant()
Write-Host "BBox log-polar A0 status: $($Summary.status)"
Write-Host "Summary SHA-256: $SummarySha"
Write-Host "Artifact manifest SHA-256: $ManifestSha"
Show-ComputeState
