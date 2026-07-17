param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [ValidateSet("Preflight", "Formal", "Finalize")]
    [string]$Phase = "Preflight",
    [string]$OutputDir = "runs\audit_dolg_orthogonal_local_global_signal_a0_20260717",
    [ValidateSet("pass", "fail")]
    [string]$VisualReviewResult = "fail",
    [string]$VisualReviewNote = "",
    [string]$ExpectedSummarySha256 = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

$MaxGpuUtilizationPercent = 20
$MaxGpuMemoryUsedMiB = 3000
$MaxGpuTemperatureC = 85

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Khong tim thay Python: $Python"
}

function Invoke-LockedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Assert-NoBackgroundModelProcesses {
    $Processes = @(
        Get-CimInstance Win32_Process -ErrorAction Stop |
            Where-Object { $_.Name -in @("python.exe", "trtexec.exe") }
    )
    if ($Processes.Count -gt 0) {
        $Details = @(
            $Processes |
                Select-Object ProcessId, CreationDate, Name, CommandLine
        ) | ConvertTo-Json -Compress
        throw (
            "Formal DOLG yeu cau khong co python.exe/trtexec.exe dang chay. " +
            "Khong tu dong dung tien trinh; hay kiem tra thu cong: $Details"
        )
    }
}

function Assert-IsolatedGpu {
    $Rows = @(
        & nvidia-smi `
            --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu `
            --format=csv,noheader,nounits
    )
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi failed with exit code ${LASTEXITCODE}."
    }
    if ($Rows.Count -lt 1) {
        throw "nvidia-smi khong tra ve GPU nao."
    }
    foreach ($Row in $Rows) {
        $Fields = @($Row -split ",")
        if ($Fields.Count -ne 4) {
            throw "Khong the parse GPU snapshot: $Row"
        }
        $Utilization = [int]$Fields[0].Trim()
        $MemoryUsed = [int]$Fields[1].Trim()
        $MemoryTotal = [int]$Fields[2].Trim()
        $Temperature = [int]$Fields[3].Trim()
        Write-Host (
            "GPU isolation snapshot: utilization=${Utilization}%, " +
            "memory=${MemoryUsed}/${MemoryTotal} MiB, temperature=${Temperature} C"
        )
        if (
            $Utilization -gt $MaxGpuUtilizationPercent -or
            $MemoryUsed -gt $MaxGpuMemoryUsedMiB -or
            $Temperature -gt $MaxGpuTemperatureC
        ) {
            throw (
                "GPU chua duoc co lap cho timing gate. Yeu cau utilization <= " +
                "${MaxGpuUtilizationPercent}%, memory <= ${MaxGpuMemoryUsedMiB} MiB, " +
                "temperature <= ${MaxGpuTemperatureC} C. Khong tu dong dong ung dung."
            )
        }
    }
}

if ($Phase -in @("Preflight", "Formal")) {
    Invoke-LockedPython -Arguments @(
        "-m", "py_compile",
        "trkh\tools\audit_dolg_orthogonal_local_global_signal.py",
        "tests\test_audit_dolg_orthogonal_local_global_signal.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pyflakes",
        "trkh\tools\audit_dolg_orthogonal_local_global_signal.py",
        "tests\test_audit_dolg_orthogonal_local_global_signal.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pytest",
        "tests\test_audit_dolg_orthogonal_local_global_signal.py",
        "-q"
    )
}

if ($Phase -eq "Preflight") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Preflight cam tao hoac dung output directory: $OutputDir"
    }
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_dolg_orthogonal_local_global_signal",
        "--preflight-only"
    )
    Write-Host "DOLG A0 preflight passed without creating an output directory."
    exit 0
}

if ($Phase -eq "Formal") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Formal output da ton tai; protocol cam overwrite/rerun: $OutputDir"
    }
    Assert-NoBackgroundModelProcesses
    Assert-IsolatedGpu
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_dolg_orthogonal_local_global_signal",
        "--output-dir", $OutputDir
    )
    $SummaryPath = Join-Path $OutputDir "summary.json"
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "Formal DOLG A0 khong tao summary: $SummaryPath"
    }
    $SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
    $Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
    if ($Summary.status -eq "awaiting_visual_review") {
        Write-Host "DOLG automated audit completed. Review all four contact sheets."
        Write-Host "Use this exact SHA with Finalize -ExpectedSummarySha256: $SummarySha"
    }
    else {
        Write-Host "DOLG formal audit completed with terminal status: $($Summary.status)"
        Write-Host "Summary SHA-256: $SummarySha"
    }
    exit 0
}

if ([string]::IsNullOrWhiteSpace($VisualReviewNote)) {
    throw "Finalize yeu cau -VisualReviewNote mo ta ket qua xem day du bon trang."
}
if ($ExpectedSummarySha256 -notmatch "^[0-9a-fA-F]{64}$") {
    throw "Finalize yeu cau -ExpectedSummarySha256 dung 64 ky tu hex tu Formal."
}
$SummaryPath = Join-Path $OutputDir "summary.json"
if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
    throw "Khong tim thay formal summary: $SummaryPath"
}
$ObservedSummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
$LockedSummarySha = $ExpectedSummarySha256.ToLowerInvariant()
if ($ObservedSummarySha -ne $LockedSummarySha) {
    throw "Formal summary SHA differs: $ObservedSummarySha != $LockedSummarySha"
}
$Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
if ($Summary.status -ne "awaiting_visual_review") {
    throw "Formal status khong yeu cau visual review: $($Summary.status)"
}
Invoke-LockedPython -Arguments @(
    "-m", "trkh.tools.audit_dolg_orthogonal_local_global_signal",
    "--output-dir", $OutputDir,
    "--finalize-visual-review",
    "--visual-review-result", $VisualReviewResult,
    "--visual-review-note", $VisualReviewNote,
    "--expected-summary-sha256", $LockedSummarySha
)
Write-Host "DOLG A0 visual review finalized from locked summary SHA-256 $LockedSummarySha."
