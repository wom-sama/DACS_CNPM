param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RawDataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
    [string]$Declaration = "runs\audit_cidt_readiness_full_train_20260714\predictions_all_conditions.csv",
    [string]$FoldRoot = "runs\yolof_cidt_fold0_trainonly_20260716",
    [string]$PreflightOutput = "runs\audit_dat_preflight_20260716",
    [string]$ControlRunName = "probe_dat_a1_control_5e_20260716",
    [string]$CandidateRunName = "probe_dat_a1_candidate_5e_20260716",
    [switch]$PreflightOnly,
    [switch]$RunPair
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not ($PreflightOnly -or $RunPair)) {
    throw "Chon -PreflightOnly hoac -RunPair."
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Khong tim thay Python: $Python"
}

$FoldDataYaml = Join-Path $FoldRoot "data.yaml"
$FoldSummary = Join-Path $FoldRoot "summary.json"
$RequiredFiles = @($RawDataYaml, $Declaration, $FoldDataYaml, $FoldSummary)
foreach ($Path in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Khong tim thay input da khoa: $Path"
    }
}

function Assert-FileSha256 {
    param([string]$Path, [string]$Expected)
    $Observed = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($Observed -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch: $Path observed=$Observed expected=$Expected"
    }
}

Assert-FileSha256 -Path $RawDataYaml -Expected "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
Assert-FileSha256 -Path $Declaration -Expected "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
Assert-FileSha256 -Path $FoldDataYaml -Expected "4195ba89995d4eee483cae31626717381df910f56657086e7116322ca971388e"
Assert-FileSha256 -Path $FoldSummary -Expected "2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763"

if ($PreflightOnly) {
    if (Test-Path -LiteralPath $PreflightOutput) {
        throw "Preflight output da ton tai; protocol cam overwrite: $PreflightOutput"
    }
    & $Python -m trkh.tools.audit_deformable_spatial_attention_preflight `
        --fold-data $FoldDataYaml `
        --fold-summary $FoldSummary `
        --raw-data $RawDataYaml `
        --declaration $Declaration `
        --output-dir $PreflightOutput `
        --batch-size 32 `
        --fp32-batch-size 2 `
        --num-workers 4 `
        --seed 42 `
        --warmup-iterations 2 `
        --timed-iterations 5
    if ($LASTEXITCODE -ne 0) {
        throw "DAT preflight rejected or failed with exit code $LASTEXITCODE."
    }
}

if ($RunPair) {
    $PreflightSummary = Join-Path $PreflightOutput "summary.json"
    if (-not (Test-Path -LiteralPath $PreflightSummary)) {
        throw "Khong tim thay preflight summary: $PreflightSummary"
    }
    $Preflight = Get-Content -Raw -LiteralPath $PreflightSummary | ConvertFrom-Json
    if (-not [bool]$Preflight.gate.formal_pair_permission) {
        throw "Preflight khong cap formal_pair_permission."
    }
    $TrackedStatus = git status --short --untracked-files=no
    if ($LASTEXITCODE -ne 0 -or -not [string]::IsNullOrWhiteSpace(($TrackedStatus -join "`n"))) {
        throw "Tracked worktree phai clean truoc formal pair."
    }
    $Head = (git rev-parse HEAD).Trim()
    $Upstream = (git rev-parse '@{upstream}').Trim()
    if ($LASTEXITCODE -ne 0 -or $Head -ne $Upstream) {
        throw "HEAD phai duoc push truoc formal pair."
    }
    if ($Preflight.git.head -ne $Head -or $Preflight.git.upstream -ne $Head) {
        throw "Preflight phai duoc tao tu chinh HEAD da push dang chay."
    }

    $ControlRunDir = Join-Path "runs" $ControlRunName
    $CandidateRunDir = Join-Path "runs" $CandidateRunName
    if ((Test-Path -LiteralPath $ControlRunDir) -or (Test-Path -LiteralPath $CandidateRunDir)) {
        throw "Formal run directory da ton tai; protocol cam overwrite/rerun."
    }

    $Launcher = Join-Path $PSScriptRoot "run_trkh_5class_attention_views_v8.ps1"
    $Shared = @{
        Python = $Python
        DataYaml = $FoldDataYaml
        ResumeCheckpoint = ""
        ImageSize = 256
        Epochs = 5
        SchedulerTotalEpochs = 5
        Patience = 3
        BatchSize = 32
        GradAccumSteps = 2
        NumWorkers = 4
        EvalNumWorkers = 2
        MaxTrainBatches = 0
        MaxValBatches = 0
        Seed = 42
        DisableBalancedEpochSampling = $true
        LearningRate = 2.5e-4
        MinLearningRate = 1e-6
        WarmupEpochs = 1
        WeightDecay = 0.05
        AttentionViewLossWeight = 0.0
        AttentionCropProbability = 0.0
        AttentionDropProbability = 0.0
        TeacherFocusBinaryLossWeight = 0.0
        DistillationTeacherCsv = ""
        EarlyTokenMaskKeepRate = 1.0
        BBoxSpatialFusion = $true
        DataCartography = $true
        FovealAggregatedAttention = $false
        DeformableSpatialAttentionLayers = "2"
        DeformableSpatialAttentionGroups = 2
        DeformableSpatialAttentionKernelSize = 5
        DeformableSpatialAttentionOffsetRange = 2.0
        TraceArchitecture = $true
        SkipFinalTest = $true
    }

    & $Launcher @Shared -RunName $ControlRunName -DeformableSpatialAttention $false
    if ($LASTEXITCODE -ne 0) {
        throw "DAT matched control failed with exit code $LASTEXITCODE"
    }
    & $Launcher @Shared -RunName $CandidateRunName -DeformableSpatialAttention $true
    if ($LASTEXITCODE -ne 0) {
        throw "DAT candidate failed with exit code $LASTEXITCODE"
    }

    $PairManifest = [ordered]@{
        method = "deformable_spatial_attention_a1_pair"
        protocol = "docs/TRKH_5CLASS_DEFORMABLE_ATTENTION_READINESS_PROTOCOL_20260716.md"
        git_head = $Head
        fold_data = (Resolve-Path -LiteralPath $FoldDataYaml).Path
        preflight_summary = (Resolve-Path -LiteralPath $PreflightSummary).Path
        control_run = (Resolve-Path -LiteralPath $ControlRunDir).Path
        candidate_run = (Resolve-Path -LiteralPath $CandidateRunDir).Path
        official_validation_used = $false
        test_used = $false
        current_best_command_updated = $false
    }
    $PairManifestPath = Join-Path "runs" "dat_a1_pair_manifest_20260716.json"
    $PairManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $PairManifestPath -Encoding utf8
    Write-Host "DAT A1 pair completed: $PairManifestPath"
}
