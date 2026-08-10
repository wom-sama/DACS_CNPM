param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RawDataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
    [string]$Declaration = "runs\audit_cidt_readiness_full_train_20260714\predictions_all_conditions.csv",
    [string]$FoldRoot = "runs\yolof_cidt_fold0_trainonly_20260716",
    [string]$PreflightOutput = "runs\audit_faa_preflight_20260716",
    [string]$PostAuditOutput = "runs\audit_faa_a1_pair_20260716",
    [string]$ControlRunName = "probe_faa_a1_control_5e_20260716",
    [string]$CandidateRunName = "probe_faa_a1_candidate_5e_20260716",
    [switch]$BuildFold,
    [switch]$PreflightOnly,
    [switch]$RunPair,
    [switch]$AuditPair,
    [switch]$FinalizeVisualReview,
    [ValidateSet("pass", "fail")]
    [string]$VisualReviewResult = "fail",
    [string]$VisualReviewNote = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Khong tim thay Python: $Python"
}
if (-not (Test-Path -LiteralPath $RawDataYaml)) {
    throw "Khong tim thay raw data YAML: $RawDataYaml"
}
if (-not (Test-Path -LiteralPath $Declaration)) {
    throw "Khong tim thay CIDT declaration: $Declaration"
}
if (-not ($BuildFold -or $PreflightOnly -or $RunPair -or $AuditPair -or $FinalizeVisualReview)) {
    throw "Chon it nhat mot mode: -BuildFold, -PreflightOnly, -RunPair, -AuditPair, hoac -FinalizeVisualReview."
}

$ExpectedDataSha = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
$ExpectedDeclarationSha = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
$ExpectedFitSha = "22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d"
$ExpectedHoldoutSha = "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"

function Assert-FileSha256 {
    param([string]$Path, [string]$Expected)
    $Observed = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($Observed -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch: $Path observed=$Observed expected=$Expected"
    }
}

Assert-FileSha256 -Path $RawDataYaml -Expected $ExpectedDataSha
Assert-FileSha256 -Path $Declaration -Expected $ExpectedDeclarationSha

if ($BuildFold) {
    if (Test-Path -LiteralPath $FoldRoot) {
        throw "Fold root da ton tai; khong overwrite artifact da khoa: $FoldRoot"
    }
    & $Python -m trkh.tools.build_yolo_declared_train_fold `
        --data $RawDataYaml `
        --declaration $Declaration `
        --output-root $FoldRoot `
        --fold 0 `
        --condition clean `
        --expected-data-sha256 $ExpectedDataSha `
        --expected-declaration-sha256 $ExpectedDeclarationSha `
        --expected-fit-index-sha256 $ExpectedFitSha `
        --expected-holdout-index-sha256 $ExpectedHoldoutSha
    if ($LASTEXITCODE -ne 0) {
        throw "Declared fold builder failed with exit code $LASTEXITCODE"
    }
}

$FoldDataYaml = Join-Path $FoldRoot "data.yaml"
$FoldSummary = Join-Path $FoldRoot "summary.json"
if (($PreflightOnly -or $RunPair -or $AuditPair) -and -not (Test-Path -LiteralPath $FoldDataYaml)) {
    throw "Khong tim thay generated fold data YAML: $FoldDataYaml"
}
if (($PreflightOnly -or $RunPair -or $AuditPair) -and -not (Test-Path -LiteralPath $FoldSummary)) {
    throw "Khong tim thay generated fold summary: $FoldSummary"
}

if ($PreflightOnly) {
    if (Test-Path -LiteralPath $PreflightOutput) {
        throw "Preflight output da ton tai; khong overwrite: $PreflightOutput"
    }
    & $Python -m trkh.tools.audit_foveal_aggregated_attention_preflight `
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
        throw "FAA preflight failed with exit code $LASTEXITCODE"
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
    $ControlRunDir = Join-Path "runs" $ControlRunName
    $CandidateRunDir = Join-Path "runs" $CandidateRunName
    if ((Test-Path -LiteralPath $ControlRunDir) -or (Test-Path -LiteralPath $CandidateRunDir)) {
        throw "Formal run directory da ton tai; protocol cam overwrite/rerun."
    }

    $Launcher = Join-Path $PSScriptRoot "run_trkh_5class_attention_views_v8.ps1"
    $Shared = @{
        Python = $Python
        DataYaml = $FoldDataYaml
        ResumeMode = "Scratch"
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
        BBoxSpatialFusion = $true
        DataCartography = $true
        FovealAggregatedAttentionLayers = "1"
        FovealAggregatedAttentionWindowSize = 3
        FovealAggregatedAttentionPoolSize = 4
        TraceArchitecture = $true
        SkipFinalTest = $true
    }

    & $Launcher @Shared -RunName $ControlRunName -FovealAggregatedAttention $false
    if ($LASTEXITCODE -ne 0) {
        throw "FAA matched control failed with exit code $LASTEXITCODE"
    }
    & $Launcher @Shared -RunName $CandidateRunName -FovealAggregatedAttention $true
    if ($LASTEXITCODE -ne 0) {
        throw "FAA candidate failed with exit code $LASTEXITCODE"
    }

    $PairManifest = [ordered]@{
        method = "foveal_aggregated_attention_a1_pair"
        protocol = "docs/TRKH_5CLASS_FOVEAL_AGGREGATED_ATTENTION_READINESS_PROTOCOL_20260716.md"
        git_head = $Head
        fold_data = (Resolve-Path -LiteralPath $FoldDataYaml).Path
        preflight_summary = (Resolve-Path -LiteralPath $PreflightSummary).Path
        control_run = (Resolve-Path -LiteralPath $ControlRunDir).Path
        candidate_run = (Resolve-Path -LiteralPath $CandidateRunDir).Path
        official_validation_used = $false
        test_used = $false
        current_best_command_updated = $false
    }
    $PairManifestPath = Join-Path "runs" "faa_a1_pair_manifest_20260716.json"
    $PairManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $PairManifestPath -Encoding utf8
    Write-Host "FAA A1 pair completed: $PairManifestPath"
}

if ($AuditPair) {
    if (Test-Path -LiteralPath $PostAuditOutput) {
        throw "Post-audit output da ton tai; khong overwrite: $PostAuditOutput"
    }
    & $Python -m trkh.tools.audit_foveal_aggregated_attention_pair `
        --control-checkpoint (Join-Path "runs" "$ControlRunName\checkpoints\best.pt") `
        --candidate-checkpoint (Join-Path "runs" "$CandidateRunName\checkpoints\best.pt") `
        --control-run-dir (Join-Path "runs" $ControlRunName) `
        --candidate-run-dir (Join-Path "runs" $CandidateRunName) `
        --fold-data $FoldDataYaml `
        --fold-summary $FoldSummary `
        --declaration $Declaration `
        --preflight-summary (Join-Path $PreflightOutput "summary.json") `
        --output-dir $PostAuditOutput `
        --batch-size 32 `
        --xai-batch-size 4 `
        --num-workers 4 `
        --seed 42
    if ($LASTEXITCODE -ne 0) {
        throw "FAA post-run audit rejected or failed with exit code $LASTEXITCODE. Inspect $PostAuditOutput\summary.json."
    }
}

if ($FinalizeVisualReview) {
    if (-not (Test-Path -LiteralPath (Join-Path $PostAuditOutput "summary.json"))) {
        throw "Khong tim thay post-audit summary: $PostAuditOutput\summary.json"
    }
    if ([string]::IsNullOrWhiteSpace($VisualReviewNote)) {
        throw "-VisualReviewNote la bat buoc khi finalize visual review."
    }
    & $Python -m trkh.tools.audit_foveal_aggregated_attention_pair `
        --output-dir $PostAuditOutput `
        --finalize-visual-review `
        --visual-review-result $VisualReviewResult `
        --visual-review-note $VisualReviewNote
    if ($LASTEXITCODE -ne 0) {
        throw "FAA visual review finalization failed with exit code $LASTEXITCODE"
    }
}
