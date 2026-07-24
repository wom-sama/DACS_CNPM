param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [ValidateSet(
        "Preflight",
        "Formal",
        "Replay",
        "Audit",
        "FinalizePass",
        "FinalizeReject"
    )]
    [string]$Phase = "Preflight",
    [string]$OutputDir = "runs\audit_cap_integral_region_context_a0_20260724",
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
        throw "CAP A0 failed with exit code ${ExitCode}: $($Arguments -join ' ')"
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
        throw "CAP launcher has PowerShell parse errors."
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
        & nvidia-smi @(
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,pstate",
            "--format=csv,noheader,nounits"
        )
    }
}

function Assert-NoForeignComputeProcess {
    $Existing = Get-ComputeProcesses
    if ($Existing) {
        $Existing | Format-Table -AutoSize | Out-Host
        throw "CAP A0 requires no pre-existing Python/TensorRT process."
    }
}

function Set-WallpaperEngineState {
    param([ValidateSet("pause", "play")][string]$State)
    $Process = Get-Process -Name "wallpaper64", "wallpaper32" `
        -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $Process) {
        return $false
    }
    $Executable = $Process.Path
    if (-not $Executable -or -not (
        Test-Path -LiteralPath $Executable -PathType Leaf
    )) {
        return $false
    }
    Start-Process -FilePath $Executable -ArgumentList @("-control", $State) `
        -WorkingDirectory (Split-Path -Parent $Executable) -Wait `
        -WindowStyle Hidden
    return $true
}

function Get-LockedCommit {
    $Head = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($Head)) {
        throw "Cannot resolve repository HEAD."
    }
    $Remote = (& git rev-parse origin/classification-only-research).Trim()
    if ($LASTEXITCODE -ne 0 -or $Remote -ne $Head) {
        throw "HEAD must be pushed to origin/classification-only-research."
    }
    $TrackedStatus = & git status --porcelain --untracked-files=no
    if ($LASTEXITCODE -ne 0 -or $TrackedStatus) {
        throw "Tracked worktree must be clean before CAP formal/replay."
    }
    return $Head
}

function Invoke-CapMode {
    param(
        [ValidateSet("formal", "replay")][string]$Mode,
        [string]$ExpectedCommit
    )
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_cap_integral_region_context_a0",
        "--mode", $Mode,
        "--output", $OutputDir,
        "--expected-commit", $ExpectedCommit,
        "--device", "cuda"
    )
}

Assert-PowerShellParse

Invoke-LockedPython -Arguments @(
    "-m", "py_compile",
    "trkh\tools\cap_integral_region_context_a0_engine.py",
    "trkh\tools\audit_cap_integral_region_context_a0.py",
    "tests\test_cap_integral_region_context_a0_engine.py",
    "tests\test_audit_cap_integral_region_context_a0.py"
)
Invoke-LockedPython -Arguments @(
    "-m", "pyflakes",
    "trkh\tools\cap_integral_region_context_a0_engine.py",
    "trkh\tools\audit_cap_integral_region_context_a0.py",
    "tests\test_cap_integral_region_context_a0_engine.py",
    "tests\test_audit_cap_integral_region_context_a0.py"
)
Invoke-LockedPython -Arguments @(
    "-m", "pytest",
    "tests\test_build_trkh_cap_integral_region_context_a0_lock.py",
    "tests\test_cap_integral_region_context_a0_engine.py",
    "tests\test_audit_cap_integral_region_context_a0.py",
    "-q"
)

if ($Phase -in @("FinalizePass", "FinalizeReject")) {
    $SummaryPath = Join-Path $OutputDir "summary.json"
    $ReplayPath = Join-Path $OutputDir "replay_summary.json"
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "Formal summary is missing: $SummaryPath"
    }
    if (-not (Test-Path -LiteralPath $ReplayPath -PathType Leaf)) {
        throw "Replay summary is missing: $ReplayPath"
    }
    if ([string]::IsNullOrWhiteSpace($ReviewNote)) {
        throw "ReviewNote is required for manual visual finalization."
    }
    $Arguments = @(
        "-m", "trkh.tools.audit_cap_integral_region_context_a0",
        "--mode", "finalize",
        "--output", $OutputDir,
        "--visual-notes", $ReviewNote
    )
    if ($Phase -eq "FinalizePass") {
        $Arguments += "--approve-visual"
    }
    Invoke-LockedPython -Arguments $Arguments
    exit 0
}

$ExpectedCommit = Get-LockedCommit

if ($Phase -eq "Preflight") {
    Show-ComputeState
    Assert-NoForeignComputeProcess
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_cap_integral_region_context_a0",
        "--mode", "preflight",
        "--expected-commit", $ExpectedCommit,
        "--device", "cuda"
    )
    exit 0
}

if ($Phase -in @("Formal", "Audit") -and (
    Test-Path -LiteralPath $OutputDir
)) {
    throw "Formal output already exists: $OutputDir"
}
if ($Phase -eq "Replay" -and -not (
    Test-Path -LiteralPath (Join-Path $OutputDir "summary.json") -PathType Leaf
)) {
    throw "Formal summary is missing in: $OutputDir"
}

$OperatingSystem = Get-CimInstance Win32_OperatingSystem
$AvailableGiB = (
    [double]$OperatingSystem.FreePhysicalMemory * 1KB
) / 1GB
if ($AvailableGiB -lt 3.5) {
    throw "CAP A0 requires at least 3.5 GiB available physical memory."
}

$WallpaperPaused = $false
try {
    Assert-NoForeignComputeProcess
    if ($PauseWallpaperEngine) {
        $WallpaperPaused = Set-WallpaperEngineState -State "pause"
    }
    if ($Phase -eq "Formal") {
        Invoke-CapMode -Mode "formal" -ExpectedCommit $ExpectedCommit
    } elseif ($Phase -eq "Replay") {
        Invoke-CapMode -Mode "replay" -ExpectedCommit $ExpectedCommit
    } else {
        Invoke-CapMode -Mode "formal" -ExpectedCommit $ExpectedCommit
        Assert-NoForeignComputeProcess
        Invoke-CapMode -Mode "replay" -ExpectedCommit $ExpectedCommit
    }
} finally {
    if ($WallpaperPaused) {
        [void](Set-WallpaperEngineState -State "play")
    }
    Show-ComputeState
}
