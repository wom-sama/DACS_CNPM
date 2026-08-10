param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [ValidateSet("Preflight", "Formal", "Finalize")]
    [string]$Phase = "Preflight",
    [string]$OutputDir = "runs\audit_learnable_polyphase_downsampling_a0_20260717",
    [ValidateSet("pass", "fail")]
    [string]$VisualReviewResult = "fail",
    [string]$VisualReviewNote = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

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

function Assert-IsolatedRuntime {
    $Unexpected = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @("python.exe", "pythonw.exe", "trtexec.exe")
    }
    if ($Unexpected) {
        $Details = ($Unexpected | Select-Object ProcessId, Name, CommandLine | Out-String).Trim()
        throw "LPD A0 found another Python/TensorRT process; none was terminated.`n$Details"
    }
    $GpuRows = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0 -or -not $GpuRows) {
        throw "Khong doc duoc nvidia-smi cho LPD A0."
    }
    foreach ($Row in $GpuRows) {
        $Fields = $Row.Split(",")
        if ($Fields.Count -ne 2) {
            throw "Khong parse duoc nvidia-smi row: $Row"
        }
        $Utilization = [int]$Fields[0].Trim()
        $Memory = [int]$Fields[1].Trim()
        if ($Utilization -gt 15 -or $Memory -gt 2200) {
            throw "GPU chua isolated cho formal timing: util=${Utilization}%, memory=${Memory} MiB."
        }
    }
}

if ($Phase -in @("Preflight", "Formal")) {
    Invoke-LockedPython -Arguments @(
        "-m", "py_compile",
        "trkh\tools\audit_learnable_polyphase_downsampling_a0.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pytest",
        "tests\test_audit_learnable_polyphase_downsampling_a0.py",
        "-q"
    )
}

if ($Phase -eq "Preflight") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Preflight cam tao hoac dung output directory: $OutputDir"
    }
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_learnable_polyphase_downsampling_a0",
        "--preflight-only"
    )
    Write-Host "LPD A0 preflight passed without creating an output directory."
    exit 0
}

if ($Phase -eq "Formal") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Formal output da ton tai; protocol cam overwrite/rerun: $OutputDir"
    }
    Assert-IsolatedRuntime
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_learnable_polyphase_downsampling_a0",
        "--output-dir", $OutputDir
    )
    $SummaryPath = Join-Path $OutputDir "summary.json"
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "Formal LPD A0 khong tao summary: $SummaryPath"
    }
    $SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
    $Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
    if ($Summary.status -eq "awaiting_visual_review") {
        Write-Host "LPD automated audit completed. Review all four contact sheets."
        Write-Host "Locked pre-review summary SHA-256: $SummarySha"
    }
    else {
        Write-Host "LPD formal audit completed with terminal status: $($Summary.status)"
        Write-Host "Summary SHA-256: $SummarySha"
    }
    exit 0
}

if ([string]::IsNullOrWhiteSpace($VisualReviewNote)) {
    throw "Finalize yeu cau -VisualReviewNote mo ta ket qua xem day du bon trang."
}
$SummaryPath = Join-Path $OutputDir "summary.json"
if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
    throw "Khong tim thay formal summary: $SummaryPath"
}
$SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
$Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
if ($Summary.status -ne "awaiting_visual_review") {
    throw "Formal status khong yeu cau visual review: $($Summary.status)"
}
Invoke-LockedPython -Arguments @(
    "-m", "trkh.tools.audit_learnable_polyphase_downsampling_a0",
    "--output-dir", $OutputDir,
    "--finalize-visual-review",
    "--visual-review-result", $VisualReviewResult,
    "--visual-review-note", $VisualReviewNote,
    "--expected-summary-sha256", $SummarySha
)
Write-Host "LPD A0 visual review finalized from summary SHA-256 $SummarySha."
