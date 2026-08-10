param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [ValidateSet("Preflight", "Formal", "Replay")]
    [string]$Phase = "Preflight",
    [string]$OutputDir = "runs\audit_c4_rotation_consensus_a0_20260720",
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
        throw "C4 rotation-consensus command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
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
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,pstate `
            --format=csv,noheader,nounits
    }
}

function Assert-GpuIsolation {
    $Processes = Get-ComputeProcesses
    if ($Processes) {
        $Processes | Format-Table -AutoSize | Out-Host
        throw "Formal C4 A0 requires no pre-existing Python/TensorRT/FFmpeg process."
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
        throw "Formal C4 A0 is blocked by non-idle background GPU activity."
    }
    if ($MaximumUtilization -gt 20) {
        Write-Host "Utilization exceeded 20 percent, but the locked low-power P8 idle fallback passed."
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
        "trkh\tools\audit_c4_rotation_consensus_a0.py",
        "tests\test_audit_c4_rotation_consensus_a0.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pyflakes",
        "trkh\tools\audit_c4_rotation_consensus_a0.py",
        "tests\test_audit_c4_rotation_consensus_a0.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pytest",
        "tests\test_audit_c4_rotation_consensus_a0.py",
        "-q"
    )
}

if ($Phase -eq "Preflight") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Preflight must not create or reuse output: $OutputDir"
    }
    Show-ComputeState
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_c4_rotation_consensus_a0",
        "--preflight-only",
        "--output-dir", $OutputDir
    )
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Preflight unexpectedly created output: $OutputDir"
    }
    Write-Host "C4 rotation-consensus A0 preflight passed without model/dataset loading or output creation."
    exit 0
}

$SummaryPath = Join-Path $OutputDir "summary.json"
if ($Phase -eq "Replay") {
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "C4 A0 summary is missing: $SummaryPath"
    }
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_c4_rotation_consensus_a0",
        "--replay-summary", $SummaryPath
    )
    exit 0
}

if (Test-Path -LiteralPath $OutputDir) {
    throw "Formal C4 A0 output already exists; rerun is forbidden: $OutputDir"
}
Show-ComputeState
$WallpaperPaused = $false
try {
    if ($PauseWallpaperEngine) {
        $WallpaperPaused = Set-WallpaperEngineState -State "pause"
    }
    Assert-GpuIsolation
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_c4_rotation_consensus_a0",
        "--output-dir", $OutputDir,
        "--batch-size", "64",
        "--num-workers", "4",
        "--folds", "5",
        "--seed", "20260714",
        "--device", "cuda"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_c4_rotation_consensus_a0",
        "--replay-summary", $SummaryPath
    )
    $Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
    $SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
    $ManifestSha = (Get-FileHash -Algorithm SHA256 -LiteralPath (
        Join-Path $OutputDir "artifact_manifest.json"
    )).Hash.ToLowerInvariant()
    Write-Host "C4 rotation-consensus A0 status: $($Summary.status)"
    Write-Host "Summary SHA-256: $SummarySha"
    Write-Host "Artifact manifest SHA-256: $ManifestSha"
} finally {
    if ($WallpaperPaused) {
        [void](Set-WallpaperEngineState -State "play")
    }
    Show-ComputeState
}
