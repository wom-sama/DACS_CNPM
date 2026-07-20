param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$OutputDir = "runs\audit_ibn_a_shallow_stem_stage_a_20260720",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$LockedFiles = [ordered]@{
    "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt" = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
    "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\resolved_config.json" = "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674"
    "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\launcher_args.json" = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
    "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml" = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
    "docs\TRKH_5CLASS_IBN_A_SHALLOW_STEM_PROTOCOL_20260720.md" = "7e329a7cf1bceea3d2dd8958fbdb1d5d34dd422b423c740032322511d167e576"
    "D:\DataAI\external_sources\papers\ibn_net_eccv2018.pdf" = "411fd8d8afd08e0d57f295821919f286c7206e7fe1ce982cedc0f06045de1b36"
    "D:\DataAI\external_sources\repositories\IBN-Net\ibnnet\modules.py" = "1578ca15ba7fbe6349d3e590c684c16236a2c0ccabf737a4762f6fed8a6b499c"
    "D:\DataAI\external_sources\repositories\IBN-Net\ibnnet\resnet_ibn.py" = "32af52f5f638bb0528b537b2e352da4ed65c02ccbf39b84fc0c5e06b35edd0f9"
    "D:\DataAI\external_sources\repositories\IBN-Net\LICENSE" = "4e2e849faed41630d067a8789edd12e4da452e8cd52b326c44e54b911b064532"
    "trkh\tools\audit_ibn_a_shallow_stem_readiness.py" = "74142d4e254fda1ff5acd07d5527ed424a6a86b32cb274efa4eac95cb6f8f09a"
    "trkh\models\model.py" = "91468f0417954b98e2db6f0bec6d966d70695f5680b8bb9d495bfb22a8289ec7"
    "trkh\core\config.py" = "f383d733c1da5a59e780a6052c76a5fe0fe0ce0f711b49a47e822c793374b0cb"
    "trkh\training\train.py" = "d01fee45c60dd7bf4ef9ec36e5cb68ddab2e3a88b428d045aed9476fa4306585"
    "scripts\run_trkh_5class_attention_views_v8.ps1" = "61f785a954fa21d242003debfb6ea34ef027b09ec9195e564df72aa4aa1e0e79"
    "tests\test_ibn_a_shallow_stem.py" = "b9398a0ab73eff246388be66bf087a184b803d2709a8d6bacf68b297e68441e7"
    "tests\test_audit_ibn_a_shallow_stem_readiness.py" = "ef8e84a2bce3582d8dce9f60694e221edb96a2fdb0a1fb4e1bf4acbdbc66ce3d"
    "docs\TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt" = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
    "docs\TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt" = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
}

function Assert-FileSha256 {
    param([string]$Path, [string]$Expected)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file not found: $Path"
    }
    $Observed = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($Observed -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path`: $Observed != $Expected"
    }
    return $Observed
}

function Get-GpuIsolationGate {
    $OwnedProcesses = @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -match '^(python|pythonw|trtexec)\.exe$' -and
                $_.CommandLine -match '(DataAI|TRKH)'
            } |
            Select-Object ProcessId, Name, CreationDate, CommandLine
    )
    if ($OwnedProcesses.Count -gt 0) {
        throw "A DataAI/TRKH compute process is already running: $($OwnedProcesses | ConvertTo-Json -Compress)"
    }

    $Samples = @()
    foreach ($Index in 0..2) {
        $Raw = & nvidia-smi --query-gpu=utilization.gpu,pstate,power.draw,memory.used,memory.total --format=csv,noheader,nounits
        if ($LASTEXITCODE -ne 0 -or -not $Raw) {
            throw "nvidia-smi GPU isolation query failed."
        }
        $Parts = @($Raw.Split(',') | ForEach-Object { $_.Trim() })
        $Samples += [pscustomobject][ordered]@{
            utilization_percent = [double]$Parts[0]
            pstate = [string]$Parts[1]
            power_watts = [double]$Parts[2]
            memory_used_mib = [double]$Parts[3]
            memory_total_mib = [double]$Parts[4]
        }
        if ($Index -lt 2) {
            Start-Sleep -Seconds 1
        }
    }
    $MaxUtilization = ($Samples | Measure-Object utilization_percent -Maximum).Maximum
    $MaxPower = ($Samples | Measure-Object power_watts -Maximum).Maximum
    $AllP8 = @($Samples | Where-Object { $_.pstate -ne 'P8' }).Count -eq 0
    $DisplayIdleFallback = $AllP8 -and $MaxPower -le 10.0
    $Passed = $MaxUtilization -le 20.0 -or $DisplayIdleFallback
    if (-not $Passed) {
        throw "GPU isolation failed: max utilization=$MaxUtilization%, max power=$MaxPower W, all P8=$AllP8."
    }
    return [ordered]@{
        passed = $true
        no_owned_compute_process = $true
        maximum_utilization_percent = $MaxUtilization
        maximum_power_watts = $MaxPower
        all_samples_p8 = $AllP8
        display_idle_fallback_used = [bool]($MaxUtilization -gt 20.0 -and $DisplayIdleFallback)
        samples = $Samples
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
$ObservedHashes = [ordered]@{}
foreach ($Entry in $LockedFiles.GetEnumerator()) {
    $ObservedHashes[$Entry.Key] = Assert-FileSha256 $Entry.Key $Entry.Value
}
$OfficialCommit = git -C "D:\DataAI\external_sources\repositories\IBN-Net" rev-parse HEAD
$OfficialTree = git -C "D:\DataAI\external_sources\repositories\IBN-Net" rev-parse 'HEAD^{tree}'
if ($OfficialCommit -ne "d1673389b36c1180cf9bc35ea8260d84046da915") {
    throw "Official IBN-Net commit drifted: $OfficialCommit"
}
if ($OfficialTree -ne "e113673c2aa64dc761a67f1585fd168a95d6ab4b") {
    throw "Official IBN-Net tree drifted: $OfficialTree"
}
$GpuGate = Get-GpuIsolationGate
$Head = git rev-parse HEAD
$Branch = git rev-parse --abbrev-ref HEAD
$Upstream = git rev-parse --abbrev-ref --symbolic-full-name '@{u}'

$Preflight = [ordered]@{
    status = "ok"
    mode = if ($PreflightOnly) { "preflight_only" } else { "formal" }
    repository_head = $Head
    branch = $Branch
    upstream = $Upstream
    observed_sha256 = $ObservedHashes
    official_commit = $OfficialCommit
    official_tree = $OfficialTree
    gpu_isolation = $GpuGate
    output_dir = $OutputDir
    batch_size = 32
    fp32_batch_size = 2
    requested_num_workers = 4
    benchmark_repeats = 3
    validation_allowed = $false
    test_allowed = $false
}
if ($PreflightOnly) {
    $Preflight | ConvertTo-Json -Depth 8
    exit 0
}

$TrackedStatus = @(git status --porcelain --untracked-files=no)
if ($TrackedStatus.Count -gt 0) {
    throw "Formal Stage A requires a clean tracked worktree: $($TrackedStatus -join '; ')"
}
$AheadBehind = @(git rev-list --left-right --count "$Upstream...HEAD") -split '\s+'
if ([int]$AheadBehind[0] -ne 0 -or [int]$AheadBehind[1] -ne 0) {
    throw "Formal Stage A requires HEAD and upstream to be synchronized."
}
if (Test-Path -LiteralPath $OutputDir) {
    throw "Refusing to overwrite Stage-A output: $OutputDir"
}

$PreviousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $Python -m trkh.tools.audit_ibn_a_shallow_stem_readiness `
        --output-dir $OutputDir `
        --batch-size 32 `
        --fp32-batch-size 2 `
        --num-workers 4 `
        --benchmark-repeats 3 `
        --seed 42 `
        --device cuda
    $AuditExitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $PreviousPreference
}

$SummaryPath = Join-Path $OutputDir "summary.json"
$ManifestPath = Join-Path $OutputDir "artifact_manifest.json"
if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
    throw "Stage-A audit did not produce summary.json (exit=$AuditExitCode)."
}
$EnvironmentPath = Join-Path $OutputDir "launcher_environment.json"
$Preflight | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $EnvironmentPath -Encoding UTF8
if (Test-Path -LiteralPath $ManifestPath -PathType Leaf) {
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    $Manifest.artifacts += [pscustomobject]@{
        path = (Resolve-Path $EnvironmentPath).Path
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $EnvironmentPath).Hash.ToLowerInvariant()
        role = "launcher_environment"
    }
    $Manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
}
if ($AuditExitCode -ne 0) {
    exit $AuditExitCode
}
Write-Host "IBN-a shallow-stem Stage A completed: $OutputDir"
