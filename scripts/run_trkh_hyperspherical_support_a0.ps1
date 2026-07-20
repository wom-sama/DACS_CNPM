param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [ValidateSet("Preflight", "Formal", "FinalizeIncomplete", "Replay")]
    [string]$Phase = "Preflight",
    [string]$OutputDir = "runs\audit_hyperspherical_support_a0_20260721"
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
        throw "Hyperspherical-support A0 command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
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
        & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu `
            --format=csv,noheader,nounits
    }
}

function Assert-CpuIsolation {
    $Processes = Get-ComputeProcesses
    if ($Processes) {
        $Processes | Format-Table -AutoSize | Out-Host
        throw "Formal hyperspherical-support A0 requires no pre-existing Python/TensorRT/FFmpeg process."
    }
    $Memory = Get-CimInstance Win32_OperatingSystem
    $AvailableGiB = $Memory.FreePhysicalMemory / 1MB
    if ($AvailableGiB -lt 4.0) {
        throw "Formal hyperspherical-support A0 requires at least 4 GiB available RAM."
    }
}

$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
$env:OPENBLAS_NUM_THREADS = "8"
$env:NUMEXPR_NUM_THREADS = "8"

if ($Phase -in @("Preflight", "Formal", "FinalizeIncomplete")) {
    Invoke-LockedPython -Arguments @(
        "-m", "py_compile",
        "trkh\tools\audit_hyperspherical_support_a0.py",
        "tests\test_audit_hyperspherical_support_a0.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pyflakes",
        "trkh\tools\audit_hyperspherical_support_a0.py",
        "tests\test_audit_hyperspherical_support_a0.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pytest",
        "tests\test_audit_hyperspherical_support_a0.py",
        "-q"
    )
}

if ($Phase -eq "Preflight") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Preflight must not create or reuse output: $OutputDir"
    }
    Show-ComputeState
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_hyperspherical_support_a0",
        "--preflight-only",
        "--output-dir", $OutputDir
    )
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Preflight unexpectedly created output: $OutputDir"
    }
    Write-Host "Hyperspherical-support A0 preflight passed without opening the NPZ or creating output."
    exit 0
}

$SummaryPath = Join-Path $OutputDir "summary.json"
if ($Phase -eq "Replay") {
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "Hyperspherical-support A0 summary is missing: $SummaryPath"
    }
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_hyperspherical_support_a0",
        "--replay-summary", $SummaryPath
    )
    exit 0
}

if ($Phase -eq "FinalizeIncomplete") {
    if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) {
        throw "Incomplete hyperspherical-support A0 output is missing: $OutputDir"
    }
    if (Test-Path -LiteralPath $SummaryPath) {
        throw "Incomplete finalization refuses an existing summary: $SummaryPath"
    }
    $ManifestPath = Join-Path $OutputDir "artifact_manifest.json"
    if (Test-Path -LiteralPath $ManifestPath) {
        throw "Incomplete finalization refuses an existing manifest: $ManifestPath"
    }
    Show-ComputeState
    Assert-CpuIsolation
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_hyperspherical_support_a0",
        "--finalize-incomplete",
        "--output-dir", $OutputDir,
        "--blas-threads", "8"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_hyperspherical_support_a0",
        "--replay-summary", $SummaryPath
    )
    $Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
    $SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
    $ManifestSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $ManifestPath).Hash.ToLowerInvariant()
    Write-Host "Hyperspherical-support A0 finalized status: $($Summary.status)"
    Write-Host "Summary SHA-256: $SummarySha"
    Write-Host "Artifact manifest SHA-256: $ManifestSha"
    Show-ComputeState
    exit 0
}

if (Test-Path -LiteralPath $OutputDir) {
    throw "Formal hyperspherical-support A0 output already exists; rerun is forbidden: $OutputDir"
}
Show-ComputeState
Assert-CpuIsolation
Invoke-LockedPython -Arguments @(
    "-m", "trkh.tools.audit_hyperspherical_support_a0",
    "--output-dir", $OutputDir,
    "--blas-threads", "8"
)
Invoke-LockedPython -Arguments @(
    "-m", "trkh.tools.audit_hyperspherical_support_a0",
    "--replay-summary", $SummaryPath
)
$Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
$SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
$ManifestSha = (Get-FileHash -Algorithm SHA256 -LiteralPath (
    Join-Path $OutputDir "artifact_manifest.json"
)).Hash.ToLowerInvariant()
Write-Host "Hyperspherical-support A0 status: $($Summary.status)"
Write-Host "Summary SHA-256: $SummarySha"
Write-Host "Artifact manifest SHA-256: $ManifestSha"
Show-ComputeState
