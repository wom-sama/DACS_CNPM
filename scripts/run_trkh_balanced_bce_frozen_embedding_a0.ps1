param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [ValidateSet("Preflight", "Formal", "Replay", "FormalPass", "FormalReject")]
    [string]$Phase = "Preflight",
    [string]$OutputDir = "runs\audit_balanced_bce_frozen_embedding_a0_20260724",
    [string]$ReviewNote = "",
    [bool]$PauseWallpaperEngine = $true
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
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "Bal-BCE A0 failed with exit code ${ExitCode}: $($Arguments -join ' ')"
    }
}

function Assert-PowerShellParse {
    $Tokens = $null
    $ParseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile(
        $PSCommandPath,
        [ref]$Tokens,
        [ref]$ParseErrors
    )
    if ($ParseErrors.Count -gt 0) {
        $ParseErrors | Format-Table -AutoSize | Out-Host
        throw "Bal-BCE launcher has PowerShell parse errors."
    }
}

function Get-ComputeProcesses {
    return Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @("python.exe", "pythonw.exe", "trtexec.exe")
    } | Select-Object ProcessId, ParentProcessId, Name, CreationDate, CommandLine
}

function Show-ComputeState {
    $Processes = Get-ComputeProcesses
    if ($Processes) {
        Write-Host "Existing Python/TensorRT processes; none will be stopped:"
        $Processes | Format-Table -AutoSize | Out-Host
    } else {
        Write-Host "No existing Python/TensorRT process detected."
    }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,pstate `
            --format=csv,noheader,nounits
    }
}

function Set-WallpaperEngineState {
    param([ValidateSet("pause", "play")][string]$State)
    $Process = Get-Process -Name "wallpaper64", "wallpaper32" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $Process) {
        return $false
    }
    $Executable = $Process.Path
    if (-not $Executable -or -not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        return $false
    }
    Start-Process -FilePath $Executable -ArgumentList @("-control", $State) `
        -WorkingDirectory (Split-Path -Parent $Executable) -Wait -WindowStyle Hidden
    return $true
}

Assert-PowerShellParse

Invoke-LockedPython -Arguments @(
    "-m", "py_compile",
    "trkh\tools\balanced_bce_frozen_embedding_a0_engine.py",
    "trkh\tools\audit_balanced_bce_frozen_embedding_a0.py",
    "tests\test_balanced_bce_frozen_embedding_a0_engine.py",
    "tests\test_balanced_bce_frozen_embedding_a0_auditor.py"
)
Invoke-LockedPython -Arguments @(
    "-m", "pyflakes",
    "trkh\tools\balanced_bce_frozen_embedding_a0_engine.py",
    "trkh\tools\audit_balanced_bce_frozen_embedding_a0.py",
    "tests\test_balanced_bce_frozen_embedding_a0_engine.py",
    "tests\test_balanced_bce_frozen_embedding_a0_auditor.py"
)
Invoke-LockedPython -Arguments @(
    "-m", "pytest",
    "tests\test_balanced_bce_frozen_embedding_a0_lock.py",
    "tests\test_balanced_bce_frozen_embedding_a0_engine.py",
    "tests\test_balanced_bce_frozen_embedding_a0_auditor.py",
    "-q"
)

if ($Phase -eq "Preflight") {
    Show-ComputeState
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_balanced_bce_frozen_embedding_a0",
        "--preflight-only"
    )
    exit 0
}

if ($Phase -eq "Replay") {
    $SummaryPath = Join-Path $OutputDir "summary.json"
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "Formal summary is missing: $SummaryPath"
    }
    Show-ComputeState
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_balanced_bce_frozen_embedding_a0",
        "--replay-summary", $SummaryPath,
        "--device", "cuda"
    )
    exit 0
}

if ($Phase -in @("FormalPass", "FormalReject")) {
    $SummaryPath = Join-Path $OutputDir "summary.json"
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "Formal summary is missing: $SummaryPath"
    }
    if ([string]::IsNullOrWhiteSpace($ReviewNote)) {
        throw "ReviewNote is required for manual finalization."
    }
    $SummarySha = (Get-FileHash -LiteralPath $SummaryPath -Algorithm SHA256).Hash.ToLower()
    $Decision = if ($Phase -eq "FormalPass") { "pass" } else { "reject" }
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_balanced_bce_frozen_embedding_a0",
        "--finalize-visual-review",
        "--output-dir", $OutputDir,
        "--expected-summary-sha256", $SummarySha,
        "--decision", $Decision,
        "--review-note", $ReviewNote
    )
    exit 0
}

if (Test-Path -LiteralPath $OutputDir) {
    throw "Formal output already exists: $OutputDir"
}
$Existing = Get-ComputeProcesses
if ($Existing) {
    $Existing | Format-Table -AutoSize | Out-Host
    throw "Formal Bal-BCE A0 requires no pre-existing Python/TensorRT process."
}
$OperatingSystem = Get-CimInstance Win32_OperatingSystem
$AvailableGiB = ([double]$OperatingSystem.FreePhysicalMemory * 1KB) / 1GB
if ($AvailableGiB -lt 4.0) {
    throw "Formal Bal-BCE A0 requires at least 4 GiB available physical memory."
}

$WallpaperPaused = $false
try {
    if ($PauseWallpaperEngine) {
        $WallpaperPaused = Set-WallpaperEngineState -State "pause"
    }
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_balanced_bce_frozen_embedding_a0",
        "--run-a0",
        "--output-dir", $OutputDir,
        "--device", "cuda"
    )
} finally {
    if ($WallpaperPaused) {
        [void](Set-WallpaperEngineState -State "play")
    }
    Show-ComputeState
}
