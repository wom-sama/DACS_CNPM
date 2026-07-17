param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [ValidateSet("Preflight", "Formal")]
    [string]$Phase = "Preflight",
    [string]$OutputDir = "runs\audit_dart_recurrent_aggregation_a0_20260717"
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
            "Formal DART yeu cau khong co python.exe/trtexec.exe dang chay. " +
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

Invoke-LockedPython -Arguments @(
    "-m", "py_compile",
    "trkh\tools\audit_dart_recurrent_aggregation_readiness.py",
    "tests\test_audit_dart_recurrent_aggregation_readiness.py"
)
Invoke-LockedPython -Arguments @(
    "-m", "pyflakes",
    "trkh\tools\audit_dart_recurrent_aggregation_readiness.py",
    "tests\test_audit_dart_recurrent_aggregation_readiness.py"
)
Invoke-LockedPython -Arguments @(
    "-m", "pytest",
    "tests\test_audit_dart_recurrent_aggregation_readiness.py",
    "-q"
)

if ($Phase -eq "Preflight") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Preflight cam tao hoac dung output directory: $OutputDir"
    }
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_dart_recurrent_aggregation_readiness",
        "--preflight-only"
    )
    Write-Host "DART A0 preflight passed without creating an output directory."
    exit 0
}

if (Test-Path -LiteralPath $OutputDir) {
    throw "Formal output da ton tai; protocol cam overwrite/rerun: $OutputDir"
}
Assert-NoBackgroundModelProcesses
Assert-IsolatedGpu
Invoke-LockedPython -Arguments @(
    "-m", "trkh.tools.audit_dart_recurrent_aggregation_readiness",
    "--output-dir", $OutputDir
)
$SummaryPath = Join-Path $OutputDir "summary.json"
if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
    throw "Formal DART A0 khong tao summary: $SummaryPath"
}
$SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
$Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
Write-Host "DART formal audit completed with status: $($Summary.status)"
Write-Host "Summary SHA-256: $SummarySha"
if ($Summary.status -eq "passed_nonvisual_xai_protocol_required") {
    Write-Host "Do not open validation yet. Lock and run the separate train-holdout XAI protocol."
}
