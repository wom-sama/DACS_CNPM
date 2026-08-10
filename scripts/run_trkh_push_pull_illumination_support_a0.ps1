param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RunName = "audit_push_pull_illumination_support_a0_20260717",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

function Get-ConflictingProcesses {
    return @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -in @("python.exe", "pythonw.exe", "trtexec.exe", "ffmpeg.exe")
            } |
            Select-Object ProcessId, Name, CreationDate, CommandLine
    )
}

function Assert-NoConflictingProcesses {
    param([string]$Phase)

    $Conflicts = @(Get-ConflictingProcesses)
    if ($Conflicts.Count -gt 0) {
        $Rendered = $Conflicts | Format-List | Out-String
        throw "Conflicting process detected before $Phase. No process was terminated.`n$Rendered"
    }
}

function Show-GpuSnapshot {
    param([string]$Phase)

    Write-Host "[$Phase] NVIDIA GPU snapshot"
    & nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi GPU snapshot failed with exit code $LASTEXITCODE."
    }
    & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi process snapshot failed with exit code $LASTEXITCODE."
    }
}

$BaseArguments = @(
    "-m",
    "trkh.tools.audit_push_pull_illumination_support_a0"
)

Assert-NoConflictingProcesses -Phase "push-pull A0 preflight"
& $Python @BaseArguments "--preflight-only"
if ($LASTEXITCODE -ne 0) {
    throw "Push-pull illumination-support A0 preflight failed with exit code $LASTEXITCODE."
}

if ($PreflightOnly) {
    return
}

Assert-NoConflictingProcesses -Phase "push-pull A0 formal audit"
Show-GpuSnapshot -Phase "before formal"
$OutputDir = Join-Path $RepoRoot "runs\$RunName"
$FormalExitCode = 0
try {
    & $Python @BaseArguments "--output-dir" $OutputDir
    $FormalExitCode = $LASTEXITCODE
}
finally {
    Show-GpuSnapshot -Phase "after formal"
}
if ($FormalExitCode -ne 0) {
    throw "Push-pull illumination-support A0 formal audit failed with exit code $FormalExitCode."
}

Assert-NoConflictingProcesses -Phase "push-pull A0 completion"
