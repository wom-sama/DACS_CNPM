param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$SourceLauncherArgs = "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\launcher_args.json",
    [string]$KeeperCheckpoint = "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt",
    [string]$KeeperPredictions = "runs\eval_yolof_teacherfocusbinary015_best_val_20260702\predictions_detailed.csv",
    [string]$StageASummary = "runs\audit_ibn_a_shallow_stem_stage_a_20260720\summary.json",
    [string]$WorkerBenchmark = "runs\evidence_spectral_decoupling_rejected_20260720\worker_benchmark\summary.json",
    [string]$ControlRunName = "smoke_ibn_a_first_control_120b_2e_20260720",
    [string]$CandidateRunName = "smoke_ibn_a_first_candidate_120b_2e_20260720",
    [string]$PairOutputDir = "runs\audit_ibn_a_shallow_stem_matched_smoke_20260720",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot
$env:TRKH_AMP_DTYPE = "bf16"
$env:OMP_NUM_THREADS = "4"
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = "1"
Remove-Item Env:TRKH_ALLOW_WINDOWS_PIN_MEMORY -ErrorAction SilentlyContinue
Remove-Item Env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS -ErrorAction SilentlyContinue

$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"
$ProtocolDocument = "docs\TRKH_5CLASS_IBN_A_SHALLOW_STEM_PROTOCOL_20260720.md"
$ErratumDocument = "docs\TRKH_5CLASS_IBN_A_SHALLOW_STEM_SMOKE_ERRATUM_20260720.md"
$ExpectedMethod = "ibn_a_shallow_stem"
$ProtocolStage = "B_matched_keeper_resume_120b_2e_full_validation"
$NumWorkers = 4
$EvalNumWorkers = 2

$LockedFiles = [ordered]@{
    $SourceLauncherArgs = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
    $KeeperCheckpoint = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
    $KeeperPredictions = "9a3ee6b7ee1ab53ad8c140a864789d0dc29029ac17c78b55bca594c6af66a088"
    $StageASummary = "14b484ffaea0271d63e3c244f3f97eaf806d9e37fbfa78ce55e3ea0b859cb422"
    "runs\audit_ibn_a_shallow_stem_stage_a_20260720\artifact_manifest.json" = "8595174c6e5193311c6bc3b04a9a16d03027e57127f7b6870170c3ec6194dff6"
    "runs\audit_ibn_a_shallow_stem_stage_a_20260720\launcher_environment.json" = "b38c9067c529ada0d76615816a2a50dacb6b417198edb3c1478ba036fdf5e3f3"
    $WorkerBenchmark = "b55552f02ed51ffab0925f3e618fc9ad142e6dbde1de3917e4e38321402cc90a"
    $DataYaml = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
    $ProtocolDocument = "7e329a7cf1bceea3d2dd8958fbdb1d5d34dd422b423c740032322511d167e576"
    $ErratumDocument = "2ba11cfc71946e8fe5a31a8a98ee5b7e0a5b36e560f2fa1b23f6129fd2c3f91e"
    "trkh\models\model.py" = "91468f0417954b98e2db6f0bec6d966d70695f5680b8bb9d495bfb22a8289ec7"
    "trkh\core\config.py" = "f383d733c1da5a59e780a6052c76a5fe0fe0ce0f711b49a47e822c793374b0cb"
    "trkh\training\train.py" = "d01fee45c60dd7bf4ef9ec36e5cb68ddab2e3a88b428d045aed9476fa4306585"
    "scripts\run_trkh_5class_attention_views_v8.ps1" = "61f785a954fa21d242003debfb6ea34ef027b09ec9195e564df72aa4aa1e0e79"
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
    if ($MaxUtilization -gt 20.0 -and -not $DisplayIdleFallback) {
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

function Set-ArgumentValue {
    param(
        [System.Collections.Generic.List[string]]$Arguments,
        [string]$Name,
        [string]$Value
    )
    $Index = $Arguments.IndexOf($Name)
    if ($Index -lt 0) {
        $Arguments.Add($Name)
        $Arguments.Add($Value)
        return
    }
    if ($Index + 1 -ge $Arguments.Count -or $Arguments[$Index + 1].StartsWith("--")) {
        throw "Argument $Name does not have a value."
    }
    $Arguments[$Index + 1] = $Value
}

function Get-ArgumentValue {
    param([object[]]$Arguments, [string]$Name)
    $Index = [Array]::IndexOf([object[]]$Arguments, $Name)
    if ($Index -lt 0 -or $Index + 1 -ge $Arguments.Count) {
        throw "Required argument is missing: $Name"
    }
    $Value = [string]$Arguments[$Index + 1]
    if ($Value.StartsWith("--")) {
        throw "Argument $Name does not have a value."
    }
    return $Value
}

function Assert-ArgumentValue {
    param([object[]]$Arguments, [string]$Name, [string]$Expected)
    $Observed = Get-ArgumentValue $Arguments $Name
    if ($Observed -cne $Expected) {
        throw "Argument mismatch for $Name`: $Observed != $Expected"
    }
}

function New-LockedTrainArguments {
    param(
        [object[]]$SourceArguments,
        [string]$RunName,
        [string]$StemNormalization
    )
    $Arguments = [System.Collections.Generic.List[string]]::new()
    foreach ($Item in $SourceArguments) {
        $Arguments.Add([string]$Item)
    }
    Set-ArgumentValue $Arguments "--run-name" $RunName
    Set-ArgumentValue $Arguments "--resume" $KeeperCheckpoint
    Set-ArgumentValue $Arguments "--stem-normalization" $StemNormalization
    return $Arguments
}

function Normalize-MatchedArguments {
    param([object[]]$Arguments)
    $Comparable = @($Arguments)
    foreach ($Name in @("--run-name", "--stem-normalization")) {
        $Index = [Array]::IndexOf([object[]]$Comparable, $Name)
        if ($Index -lt 0 -or $Index + 1 -ge $Comparable.Count) {
            throw "Normalized arguments do not contain $Name."
        }
        $Comparable[$Index + 1] = "<matched-role>"
    }
    return $Comparable
}

function Invoke-NativePython {
    param([string]$Step, [string[]]$Arguments)
    Write-Host "[$Step] $Python $($Arguments -join ' ')"
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @Arguments
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) {
        throw "$Step failed with exit code $ExitCode."
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
$ObservedHashes = [ordered]@{}
foreach ($Entry in $LockedFiles.GetEnumerator()) {
    $ObservedHashes[$Entry.Key] = Assert-FileSha256 $Entry.Key $Entry.Value
}

$StageA = Get-Content -Raw -LiteralPath $StageASummary | ConvertFrom-Json
if (
    $StageA.method -ne $ExpectedMethod -or
    $StageA.status -ne "passed" -or
    -not [bool]$StageA.gate.smoke_permission -or
    [bool]$StageA.gate.full_train_permission
) {
    throw "Stage-A evidence does not authorize the IBN-a smoke pair."
}
if ([bool]$StageA.sources.validation_loaded -or [bool]$StageA.sources.test_loaded) {
    throw "Stage-A evidence is not train-only."
}

$Benchmark = Get-Content -Raw -LiteralPath $WorkerBenchmark | ConvertFrom-Json
$BestWorkers = $Benchmark.best_by_samples_per_second
if (
    [string]$Benchmark.device -ne "cuda" -or
    [int]$Benchmark.dataset_samples -ne 9215 -or
    [int]$Benchmark.batch_size -ne 32 -or
    [int]$Benchmark.image_size -ne 256 -or
    -not [bool]$Benchmark.disable_pin_memory -or
    -not [bool]$Benchmark.disable_persistent_workers -or
    [int]$BestWorkers.requested_workers -ne $NumWorkers -or
    [int]$BestWorkers.effective_workers -ne $NumWorkers -or
    [bool]$BestWorkers.pin_memory -or
    [bool]$BestWorkers.persistent_workers
) {
    throw "Worker benchmark does not authorize the locked 4/2 loader."
}

$Source = Get-Content -Raw -LiteralPath $SourceLauncherArgs | ConvertFrom-Json
$SourceArguments = @($Source.train_args)
if ([Array]::IndexOf([object[]]$SourceArguments, "--stem-normalization") -ge 0) {
    throw "Historical source unexpectedly contains --stem-normalization."
}
foreach ($RequiredFlag in @(
    "--no-pretrained",
    "--no-pretrained-distillation",
    "--resume-use-cli-config",
    "--resume-reset-epoch",
    "--resume-reset-optimizer",
    "--resume-reset-scheduler",
    "--resume-reset-scaler",
    "--skip-final-test",
    "--trace-architecture"
)) {
    if ([Array]::IndexOf([object[]]$SourceArguments, $RequiredFlag) -lt 0) {
        throw "Locked source arguments omit $RequiredFlag."
    }
}
if ([Array]::IndexOf([object[]]$SourceArguments, "--pretrained") -ge 0) {
    throw "Locked source unexpectedly enables pretrained weights."
}
if ([Array]::IndexOf([object[]]$SourceArguments, "--max-val-batches") -ge 0) {
    throw "Locked source must use the trainer's full-validation default."
}
foreach ($Expectation in @(
    @("--data", $DataYaml),
    @("--seed", "42"),
    @("--batch-size", "32"),
    @("--grad-accum-steps", "2"),
    @("--epochs", "2"),
    @("--scheduler-total-epochs", "2"),
    @("--patience", "2"),
    @("--max-train-batches", "120"),
    @("--num-workers", "$NumWorkers"),
    @("--eval-num-workers", "$EvalNumWorkers")
)) {
    Assert-ArgumentValue $SourceArguments $Expectation[0] $Expectation[1]
}

$ControlArguments = New-LockedTrainArguments $SourceArguments $ControlRunName "batch"
$CandidateArguments = New-LockedTrainArguments $SourceArguments $CandidateRunName "ibn_a_first"
$ComparableControl = Normalize-MatchedArguments @($ControlArguments)
$ComparableCandidate = Normalize-MatchedArguments @($CandidateArguments)
if (($ComparableControl | ConvertTo-Json -Compress) -cne ($ComparableCandidate | ConvertTo-Json -Compress)) {
    throw "Control and candidate arguments differ beyond the locked model role."
}
Assert-ArgumentValue $ControlArguments "--resume" $KeeperCheckpoint
Assert-ArgumentValue $CandidateArguments "--resume" $KeeperCheckpoint
Assert-ArgumentValue $ControlArguments "--stem-normalization" "batch"
Assert-ArgumentValue $CandidateArguments "--stem-normalization" "ibn_a_first"

$TrackedStatus = @(git status --porcelain --untracked-files=no)
$Head = git rev-parse HEAD
$Branch = git rev-parse --abbrev-ref HEAD
$Upstream = git rev-parse --abbrev-ref --symbolic-full-name '@{u}'
$AheadBehind = @(git rev-list --left-right --count "$Upstream...HEAD") -split '\s+'
if (-not $PreflightOnly) {
    if ($TrackedStatus.Count -gt 0) {
        throw "Formal smoke requires a clean tracked worktree: $($TrackedStatus -join '; ')"
    }
    if ([int]$AheadBehind[0] -ne 0 -or [int]$AheadBehind[1] -ne 0) {
        throw "Formal smoke requires HEAD and upstream to be synchronized."
    }
}
$PreGpuGate = Get-GpuIsolationGate

$Protocol = [ordered]@{
    method = $ExpectedMethod
    protocol_stage = $ProtocolStage
    protocol_document = (Resolve-Path $ProtocolDocument).Path
    protocol_document_sha256 = $ObservedHashes[$ProtocolDocument]
    erratum_document = (Resolve-Path $ErratumDocument).Path
    erratum_document_sha256 = $ObservedHashes[$ErratumDocument]
    source_launcher_args = (Resolve-Path $SourceLauncherArgs).Path
    source_launcher_args_sha256 = $ObservedHashes[$SourceLauncherArgs]
    source_resume_checkpoint_unavailable = "runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\checkpoints\best.pt"
    common_resume_replacement = (Resolve-Path $KeeperCheckpoint).Path
    common_resume_replacement_sha256 = $ObservedHashes[$KeeperCheckpoint]
    keeper_predictions = (Resolve-Path $KeeperPredictions).Path
    keeper_predictions_sha256 = $ObservedHashes[$KeeperPredictions]
    stage_a_summary = (Resolve-Path $StageASummary).Path
    stage_a_summary_sha256 = $ObservedHashes[$StageASummary]
    worker_benchmark = (Resolve-Path $WorkerBenchmark).Path
    worker_benchmark_sha256 = $ObservedHashes[$WorkerBenchmark]
    data_yaml = $DataYaml
    data_yaml_sha256 = $ObservedHashes[$DataYaml]
    test_allowed = $false
    seed = 42
    epochs = 2
    max_train_batches = 120
    max_val_batches = 0
    scheduler_total_epochs = 2
    patience = 2
    batch_size = 32
    grad_accum_steps = 2
    num_workers = $NumWorkers
    eval_num_workers = $EvalNumWorkers
    control_run_name = $ControlRunName
    candidate_run_name = $CandidateRunName
    control_arguments = @($ControlArguments)
    candidate_arguments = @($CandidateArguments)
}

$LauncherEnvironment = [ordered]@{
    status = if ($PreflightOnly) { "preflight_only" } else { "running" }
    repository_head = $Head
    branch = $Branch
    upstream = $Upstream
    ahead_behind = $AheadBehind
    tracked_status = $TrackedStatus
    observed_sha256 = $ObservedHashes
    pre_gpu_isolation = $PreGpuGate
    worker_benchmark = $BestWorkers
    raw_dataset_modified = $false
    validation_allowed = $true
    test_allowed = $false
}

if ($PreflightOnly) {
    [ordered]@{
        status = "ok"
        mode = "preflight_only"
        launcher_environment = $LauncherEnvironment
        protocol = $Protocol
    } | ConvertTo-Json -Depth 10
    exit 0
}

foreach ($Path in @(
    (Join-Path "runs" $ControlRunName),
    (Join-Path "runs" $CandidateRunName),
    $PairOutputDir
)) {
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to overwrite existing smoke artifact: $Path"
    }
}
New-Item -ItemType Directory -Path $PairOutputDir -Force | Out-Null
$ProtocolPath = Join-Path $PairOutputDir "locked_protocol.json"
$LauncherPath = Join-Path $PairOutputDir "launcher_environment.json"
$StatusPath = Join-Path $PairOutputDir "status.json"
$Protocol | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $ProtocolPath -Encoding UTF8
$LauncherEnvironment | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $LauncherPath -Encoding UTF8
$Status = [ordered]@{
    status = "running"
    control = "pending"
    candidate = "pending"
    control_validation = "pending"
    candidate_validation = "pending"
    comparison = "pending"
}
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8

Invoke-NativePython "train_control" (@("-m", "trkh.training.train") + @($ControlArguments))
$Status.control = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
Invoke-NativePython "train_candidate" (@("-m", "trkh.training.train") + @($CandidateArguments))
$Status.candidate = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8

$ControlRunDir = Join-Path "runs" $ControlRunName
$CandidateRunDir = Join-Path "runs" $CandidateRunName
$ControlVal = Join-Path $PairOutputDir "control_val"
$CandidateVal = Join-Path $PairOutputDir "candidate_val"
$EvaluationBase = @(
    "-m", "trkh.evaluation.evaluate",
    "--data", $DataYaml,
    "--class-name-mode", "raw",
    "--expected-num-classes", "5",
    "--split", "val",
    "--batch-size", "64",
    "--num-workers", "$EvalNumWorkers",
    "--amp",
    "--bbox-token-prior-source", "crop_bbox",
    "--family", "TRKH-IBN-a-shallow-matched-smoke"
)
Invoke-NativePython "evaluate_control_val" ($EvaluationBase + @(
    "--checkpoint", (Join-Path $ControlRunDir "checkpoints\best.pt"),
    "--paper-name", "TRKH-IBN-a-control-batch-120b-2e",
    "--output-dir", $ControlVal
))
$Status.control_validation = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
Invoke-NativePython "evaluate_candidate_val" ($EvaluationBase + @(
    "--checkpoint", (Join-Path $CandidateRunDir "checkpoints\best.pt"),
    "--paper-name", "TRKH-IBN-a-first-120b-2e",
    "--output-dir", $CandidateVal
))
$Status.candidate_validation = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8

$ComparisonDir = Join-Path $PairOutputDir "comparison"
Invoke-NativePython "compare_smoke_pair" @(
    "-m", "trkh.tools.audit_ibn_a_shallow_stem_smoke_pair",
    "--keeper-predictions", $KeeperPredictions,
    "--control-predictions", (Join-Path $ControlVal "predictions_detailed.csv"),
    "--candidate-predictions", (Join-Path $CandidateVal "predictions_detailed.csv"),
    "--control-run-dir", $ControlRunDir,
    "--candidate-run-dir", $CandidateRunDir,
    "--stage-a-summary", $StageASummary,
    "--locked-protocol", $ProtocolPath,
    "--output-dir", $ComparisonDir
)
$Status.comparison = "completed"
$Status.status = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8

$LauncherEnvironment["post_gpu_isolation"] = Get-GpuIsolationGate
$LauncherEnvironment.status = "completed"
$LauncherEnvironment | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $LauncherPath -Encoding UTF8

$ArtifactFiles = [ordered]@{
    "locked_protocol" = $ProtocolPath
    "launcher_environment" = $LauncherPath
    "status" = $StatusPath
    "comparison_summary" = (Join-Path $ComparisonDir "summary.json")
    "comparison_manifest" = (Join-Path $ComparisonDir "artifact_manifest.json")
    "changed_cases" = (Join-Path $ComparisonDir "changed_cases.csv")
    "source_group_forensics" = (Join-Path $ComparisonDir "source_group_forensics.csv")
    "control_checkpoint" = (Join-Path $ControlRunDir "checkpoints\best.pt")
    "candidate_checkpoint" = (Join-Path $CandidateRunDir "checkpoints\best.pt")
    "control_summary" = (Join-Path $ControlRunDir "summary.json")
    "candidate_summary" = (Join-Path $CandidateRunDir "summary.json")
    "control_config" = (Join-Path $ControlRunDir "resolved_config.json")
    "candidate_config" = (Join-Path $CandidateRunDir "resolved_config.json")
    "control_trace" = (Join-Path $ControlRunDir "architecture_trace\trace_summary.json")
    "candidate_trace" = (Join-Path $CandidateRunDir "architecture_trace\trace_summary.json")
    "control_predictions" = (Join-Path $ControlVal "predictions_detailed.csv")
    "candidate_predictions" = (Join-Path $CandidateVal "predictions_detailed.csv")
}
$Artifacts = @(
    foreach ($Entry in $ArtifactFiles.GetEnumerator()) {
        $Resolved = (Resolve-Path -LiteralPath $Entry.Value).Path
        [ordered]@{
            role = $Entry.Key
            path = $Resolved
            sha256 = (Get-FileHash -LiteralPath $Resolved -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
)
[ordered]@{
    artifacts = $Artifacts
    raw_dataset_modified = $false
    validation_used = $true
    test_used = $false
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $PairOutputDir "artifact_manifest.json") -Encoding UTF8

Write-Host "Matched shallow IBN-a smoke pair completed: $PairOutputDir"
