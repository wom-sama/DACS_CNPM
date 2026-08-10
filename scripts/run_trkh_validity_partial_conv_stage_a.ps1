param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$OutputDir = "runs\audit_validity_partial_conv_stem_stage_a_20260720",
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
    "docs\TRKH_5CLASS_VALIDITY_PARTIAL_CONV_STEM_PROTOCOL_20260720.md" = "1838135e00e2f4c38d9cbf1d19fbf851a6e715b2676f396236f2a6f010312cda"
    "D:\DataAI\external_sources\papers\partial_convolution_padding_arxiv1811.11718.pdf" = "6b6d1ceee5bf69e7b0a6b93ef26b6681c8d479601ba70e16c9968bbf82ccc93f"
    "D:\DataAI\external_sources\papers\partial_convolution_inpainting_eccv2018.pdf" = "9d902b164c4563c97cf9ca9f0a7d5508bb292ae4204ce4173cf3f8056009c73b"
    "D:\DataAI\external_sources\official\NVIDIA_partialconv\models\partialconv2d.py" = "ca92d642523b7b18e56c981d6198e29e6b7cdbfa7469b3a07f5f93e741435116"
    "D:\DataAI\external_sources\official\NVIDIA_partialconv\README.MD" = "7578fea2adcbaab9ac35ad4f1f19b16296a40d868d60243523c262d5ad077dd0"
    "D:\DataAI\external_sources\official\NVIDIA_partialconv\LICENSE" = "8dc73b75a37967ada9d2a4b7643b155fad5b17e409319f3aa7c5d6092bf0d3d0"
    "trkh\models\model.py" = "ea3002bbb25d1fa1a30e6390e29dfd5c7c93a02838cf744c5c53f86ed97cd595"
    "trkh\core\config.py" = "72b32a160ab6be2aecdd87670e2da325976f79ac1803a63d0b06b7f1f931e2fa"
    "trkh\training\train.py" = "763073051fe62dfac43eb5f509abad04991d03ccbcd893244d501720b7030c32"
    "scripts\run_trkh_5class_attention_views_v8.ps1" = "8512df324c1ddb196f8b579f2c1a268ead588799941ef236de237ac8c1b42f5a"
    "trkh\tools\audit_validity_partial_conv_stem_readiness.py" = "cadeb193ea4a09ec4776ed0b13089b93bc1cdef07f3d55fb670d7f85810cfd96"
    "tests\test_validity_partial_conv_stem.py" = "4aae8f8876d26331aac5ee6ff4742143c4b465763ba2b232e5bd32b85a5d892d"
    "tests\test_audit_validity_partial_conv_stem_readiness.py" = "ed4d4ad6751e9f4d1d59cf449bcd3859ec2ad1bacfdb3eac7981f3e17d0226cf"
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
$OfficialRepo = "D:\DataAI\external_sources\official\NVIDIA_partialconv"
$OfficialCommit = git -C $OfficialRepo rev-parse HEAD
$OfficialTree = git -C $OfficialRepo rev-parse 'HEAD^{tree}'
$OfficialStatus = @(git -C $OfficialRepo status --porcelain)
if ($OfficialCommit -ne "610d373f35257887d45adae84c86d0ce7ad808ec") {
    throw "Official NVIDIA partialconv commit drifted: $OfficialCommit"
}
if ($OfficialTree -ne "39e37de8bdce67efd5ceee3008fe0c94bdd9cd5f") {
    throw "Official NVIDIA partialconv tree drifted: $OfficialTree"
}
if ($OfficialStatus.Count -gt 0) {
    throw "Official NVIDIA partialconv checkout is dirty: $($OfficialStatus -join '; ')"
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
    official_status_clean = $true
    gpu_isolation = $GpuGate
    output_dir = $OutputDir
    batch_size = 32
    fp32_batch_size = 2
    requested_num_workers = 4
    benchmark_repeats = 3
    validation_allowed = $false
    test_allowed = $false
    full_train_allowed = $false
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
    & $Python -m trkh.tools.audit_validity_partial_conv_stem_readiness `
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
$ExternalReplayPath = Join-Path $OutputDir "external_process_replay.json"
$PreviousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $Python -m trkh.tools.audit_validity_partial_conv_stem_readiness `
        --replay-summary $SummaryPath > $ExternalReplayPath
    $ReplayExitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $PreviousPreference
}
if ($ReplayExitCode -ne 0) {
    throw "Independent Stage-A replay failed with exit code $ReplayExitCode."
}
$ExternalReplay = Get-Content -LiteralPath $ExternalReplayPath -Raw | ConvertFrom-Json
if (-not [bool]$ExternalReplay.exact) {
    throw "Independent Stage-A replay was not exact."
}

$EnvironmentPath = Join-Path $OutputDir "launcher_environment.json"
$Preflight | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $EnvironmentPath -Encoding UTF8
if (Test-Path -LiteralPath $ManifestPath -PathType Leaf) {
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    foreach ($ArtifactPath in @($EnvironmentPath, $ExternalReplayPath)) {
        $Manifest.artifacts += [pscustomobject]@{
            path = (Resolve-Path $ArtifactPath).Path
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ArtifactPath).Hash.ToLowerInvariant()
            size_bytes = (Get-Item -LiteralPath $ArtifactPath).Length
        }
    }
    $Manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
}
if ($AuditExitCode -ne 0) {
    exit $AuditExitCode
}
Write-Host "Validity partial-conv stem Stage A completed: $OutputDir"
