param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [ValidateSet("GeometryPreview", "GeometryPass", "GeometryReject", "Formal", "Replay")]
    [string]$Phase = "GeometryPreview",
    [string]$OutputDir = "runs\audit_nsa_class_pair_poisson_a0_geometry_preview_20260724",
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
    if ($LASTEXITCODE -ne 0) {
        throw "NSA class-pair Poisson A0 failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
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
        throw "NSA class-pair Poisson launcher has PowerShell parse errors."
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
    "trkh\tools\nsa_class_pair_poisson.py",
    "trkh\tools\audit_nsa_class_pair_poisson_a0.py",
    "tests\test_nsa_class_pair_poisson.py"
)
Invoke-LockedPython -Arguments @(
    "-m", "pyflakes",
    "trkh\tools\nsa_class_pair_poisson.py",
    "trkh\tools\audit_nsa_class_pair_poisson_a0.py",
    "tests\test_nsa_class_pair_poisson.py"
)
Invoke-LockedPython -Arguments @(
    "-m", "pytest",
    "tests\test_nsa_class_pair_poisson.py",
    "-q"
)

if ($Phase -eq "GeometryPreview") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Geometry preview output already exists: $OutputDir"
    }
    Show-ComputeState
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_nsa_class_pair_poisson_a0",
        "--geometry-preview",
        "--output-dir", $OutputDir
    )
    exit 0
}

if ($Phase -in @("GeometryPass", "GeometryReject")) {
    $Decision = if ($Phase -eq "GeometryPass") { "pass" } else { "reject" }
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_nsa_class_pair_poisson_a0",
        "--finalize-geometry-review",
        "--decision", $Decision,
        "--review-note", $ReviewNote,
        "--output-dir", $OutputDir
    )
    exit 0
}

if ($Phase -eq "Replay") {
    $SummaryPath = Join-Path $OutputDir "summary.json"
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "Formal summary is missing: $SummaryPath"
    }
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_nsa_class_pair_poisson_a0",
        "--replay-summary", $SummaryPath,
        "--output-dir", $OutputDir,
        "--device", "cuda",
        "--batch-size", "16",
        "--num-workers", "4"
    )
    exit 0
}

if (Test-Path -LiteralPath $OutputDir) {
    throw "Formal output already exists: $OutputDir"
}

$Existing = Get-ComputeProcesses
if ($Existing) {
    $Existing | Format-Table -AutoSize | Out-Host
    throw "Formal A0 requires no pre-existing Python/TensorRT process."
}

$OperatingSystem = Get-CimInstance Win32_OperatingSystem
$AvailableGiB = ([double]$OperatingSystem.FreePhysicalMemory * 1KB) / 1GB
if ($AvailableGiB -lt 5.0) {
    throw "Formal A0 requires at least 5 GiB available physical memory."
}

$WallpaperPaused = $false
$OldMultiprocessing = $env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING
$OldPinMemory = $env:TRKH_ALLOW_WINDOWS_PIN_MEMORY
$OldPersistentWorkers = $env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS
try {
    if ($PauseWallpaperEngine) {
        $WallpaperPaused = Set-WallpaperEngineState -State "pause"
    }
    $env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = "1"
    $env:TRKH_ALLOW_WINDOWS_PIN_MEMORY = "1"
    $env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS = "1"
    Invoke-LockedPython -Arguments @("-m", "pytest", "-q")
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_nsa_class_pair_poisson_a0",
        "--run-a0",
        "--output-dir", $OutputDir,
        "--device", "cuda",
        "--batch-size", "16",
        "--num-workers", "4"
    )
} finally {
    $env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = $OldMultiprocessing
    $env:TRKH_ALLOW_WINDOWS_PIN_MEMORY = $OldPinMemory
    $env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS = $OldPersistentWorkers
    if ($WallpaperPaused) {
        [void](Set-WallpaperEngineState -State "play")
    }
    Show-ComputeState
}
