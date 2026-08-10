param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [ValidateSet("Preflight", "Formal", "Replay")]
    [string]$Phase = "Preflight",
    [string]$OutputDir = "runs\audit_ielt_mhv_signal_a0_20260720",
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
        throw "IELT MHV command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Get-ComputeProcesses {
    return Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @("python.exe", "pythonw.exe", "trtexec.exe", "ffmpeg.exe")
    } | Select-Object ProcessId, ParentProcessId, Name, CreationDate, CommandLine
}

function Show-ComputeState {
    $Processes = Get-ComputeProcesses
    if ($Processes) {
        Write-Host "Existing compute processes were detected; none will be stopped:"
        $Processes | Format-Table -AutoSize | Out-Host
    } else {
        Write-Host "No existing Python/TensorRT/FFmpeg process was detected."
    }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,pstate `
            --format=csv,noheader,nounits
    }
}

function Assert-HostResources {
    $OperatingSystem = Get-CimInstance Win32_OperatingSystem
    $AvailableGiB = [math]::Round(
        ([double]$OperatingSystem.FreePhysicalMemory * 1KB) / 1GB,
        2
    )
    Write-Host "Available physical memory: $AvailableGiB GiB"
    if ($AvailableGiB -lt 6.0) {
        throw "Formal IELT MHV A0 requires at least 6 GiB available physical memory."
    }
}

function Assert-GpuIsolation {
    $Processes = Get-ComputeProcesses
    if ($Processes) {
        $Processes | Format-Table -AutoSize | Out-Host
        throw "Formal IELT MHV A0 requires no pre-existing Python/TensorRT/FFmpeg process."
    }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw "nvidia-smi is required for the formal GPU isolation gate."
    }
    $Samples = @()
    for ($Index = 0; $Index -lt 3; $Index++) {
        $Raw = & nvidia-smi --query-gpu=utilization.gpu,power.draw,pstate `
            --format=csv,noheader,nounits
        if ($LASTEXITCODE -ne 0) {
            throw "nvidia-smi isolation query failed."
        }
        $Fields = $Raw.Split(",")
        if ($Fields.Count -ne 3) {
            throw "Unexpected nvidia-smi isolation response: $Raw"
        }
        $Samples += [pscustomobject]@{
            Utilization = [int]$Fields[0].Trim()
            PowerWatts = [double]::Parse(
                $Fields[1].Trim(),
                [Globalization.CultureInfo]::InvariantCulture
            )
            PState = $Fields[2].Trim()
        }
        if ($Index -lt 2) {
            Start-Sleep -Seconds 1
        }
    }
    $MaximumUtilization = ($Samples.Utilization | Measure-Object -Maximum).Maximum
    $MaximumPower = ($Samples.PowerWatts | Measure-Object -Maximum).Maximum
    $LowPowerIdle = ($MaximumPower -le 10.0) -and (
        @($Samples | Where-Object { $_.PState -ne "P8" }).Count -eq 0
    )
    Write-Host "GPU utilization samples: $($Samples.Utilization -join ', ') percent"
    Write-Host "GPU power samples: $($Samples.PowerWatts -join ', ') W; states: $($Samples.PState -join ', ')"
    if (($MaximumUtilization -gt 20) -and (-not $LowPowerIdle)) {
        throw "Formal IELT MHV A0 is blocked by non-idle background GPU activity."
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

if ($Phase -in @("Preflight", "Formal")) {
    Invoke-LockedPython -Arguments @(
        "-m", "py_compile",
        "trkh\tools\audit_ielt_mhv_signal_a0.py",
        "tests\test_audit_ielt_mhv_signal_a0.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pyflakes",
        "trkh\tools\audit_ielt_mhv_signal_a0.py",
        "tests\test_audit_ielt_mhv_signal_a0.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pytest",
        "tests\test_audit_ielt_mhv_signal_a0.py",
        "-q"
    )
}

if ($Phase -eq "Preflight") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Preflight must not create or reuse output: $OutputDir"
    }
    Show-ComputeState
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_ielt_mhv_signal_a0",
        "--preflight-only",
        "--output-dir", $OutputDir,
        "--batch-size", "32",
        "--num-workers", "4",
        "--seed", "20260720",
        "--device", "cuda"
    )
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Preflight unexpectedly created output: $OutputDir"
    }
    Write-Host "IELT MHV A0 preflight passed with synthetic CUDA data only."
    exit 0
}

$SummaryPath = Join-Path $OutputDir "summary.json"
if ($Phase -eq "Replay") {
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "IELT MHV A0 summary is missing: $SummaryPath"
    }
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_ielt_mhv_signal_a0",
        "--replay-summary", $SummaryPath
    )
    exit 0
}

if (Test-Path -LiteralPath $OutputDir) {
    throw "Formal IELT MHV A0 output already exists; rerun is forbidden: $OutputDir"
}
Show-ComputeState
Assert-HostResources
$WallpaperPaused = $false
$OldMultiprocessing = $env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING
$OldPinMemory = $env:TRKH_ALLOW_WINDOWS_PIN_MEMORY
$OldPersistentWorkers = $env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS
try {
    if ($PauseWallpaperEngine) {
        $WallpaperPaused = Set-WallpaperEngineState -State "pause"
    }
    Assert-GpuIsolation
    $env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = "1"
    $env:TRKH_ALLOW_WINDOWS_PIN_MEMORY = "1"
    $env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS = "1"
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_ielt_mhv_signal_a0",
        "--output-dir", $OutputDir,
        "--batch-size", "32",
        "--num-workers", "4",
        "--seed", "20260720",
        "--device", "cuda"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_ielt_mhv_signal_a0",
        "--replay-summary", $SummaryPath
    )
    $Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
    $SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
    $ManifestPath = Join-Path $OutputDir "artifact_manifest.json"
    $ManifestSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $ManifestPath).Hash.ToLowerInvariant()
    Write-Host "IELT MHV A0 status: $($Summary.status)"
    Write-Host "Summary SHA-256: $SummarySha"
    Write-Host "Artifact manifest SHA-256: $ManifestSha"
} finally {
    $env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = $OldMultiprocessing
    $env:TRKH_ALLOW_WINDOWS_PIN_MEMORY = $OldPinMemory
    $env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS = $OldPersistentWorkers
    if ($WallpaperPaused) {
        [void](Set-WallpaperEngineState -State "play")
    }
    Show-ComputeState
}
