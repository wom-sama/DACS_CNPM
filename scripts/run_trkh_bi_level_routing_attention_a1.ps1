param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RawDataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
    [string]$Declaration = "runs\audit_cidt_readiness_full_train_20260714\predictions_all_conditions.csv",
    [string]$FoldRoot = "runs\yolof_cidt_fold0_trainonly_20260716",
    [string]$PreflightOutput = "runs\audit_bra_preflight_20260716",
    [string]$ControlRunName = "probe_bra_a1_topk16_control_5e_20260716",
    [string]$CandidateRunName = "probe_bra_a1_topk4_candidate_5e_20260716",
    [switch]$PreflightOnly,
    [switch]$FinalizeVisualReview,
    [ValidateSet("pass", "fail")]
    [string]$VisualReviewResult = "pass",
    [string]$VisualReviewNote = "",
    [switch]$RunPair
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$SelectedModes = @($PreflightOnly, $FinalizeVisualReview, $RunPair) | Where-Object { $_ }
if ($SelectedModes.Count -ne 1) {
    throw "Chon dung mot mode: -PreflightOnly, -FinalizeVisualReview, hoac -RunPair."
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
    & $Python -m trkh.tools.audit_bi_level_routing_attention_preflight `
        --fold-data $FoldDataYaml `
        --fold-summary $FoldSummary `
        --raw-data $RawDataYaml `
        --declaration $Declaration `
        --output-dir $PreflightOutput `
        --batch-size 32 `
        --fp32-batch-size 2 `
        --selectivity-batch-size 16 `
        --num-workers 4 `
        --seed 42 `
        --warmup-iterations 2 `
        --timed-iterations 5
    if ($LASTEXITCODE -ne 0) {
        throw "BRA preflight rejected or failed with exit code $LASTEXITCODE."
    }
    Write-Host "BRA automated preflight passed. Review every route_overlay_page_*.png before finalization."
}

if ($FinalizeVisualReview) {
    if ([string]::IsNullOrWhiteSpace($VisualReviewNote)) {
        throw "FinalizeVisualReview yeu cau -VisualReviewNote."
    }
    & $Python -m trkh.tools.audit_bi_level_routing_attention_preflight `
        --output-dir $PreflightOutput `
        --finalize-visual-review `
        --visual-review-result $VisualReviewResult `
        --visual-review-note $VisualReviewNote
    if ($LASTEXITCODE -ne 0) {
        throw "BRA visual review finalization failed with exit code $LASTEXITCODE."
    }
}

if ($RunPair) {
    $PreflightSummary = Join-Path $PreflightOutput "summary.json"
    if (-not (Test-Path -LiteralPath $PreflightSummary)) {
        throw "Khong tim thay preflight summary: $PreflightSummary"
    }
    $Preflight = Get-Content -Raw -LiteralPath $PreflightSummary | ConvertFrom-Json
    if (-not [bool]$Preflight.gate.formal_pair_permission) {
        throw "Preflight va visual review khong cap formal_pair_permission."
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
        StemArchitecture = "conv_pool"
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
        ClassificationLoss = "ldam_focal"
        FocalLossGamma = 1.0
        FocalLossMix = 0.10
        LabelSmoothing = 0.02
        LdamMaxMargin = 0.30
        LdamScale = 18.0
        MetricLearningLossWeight = 0.04
        MetricLearningTemperature = 0.16
        AttentionViewLossWeight = 0.0
        AttentionCropProbability = 0.0
        AttentionDropProbability = 0.0
        TeacherFocusBinaryLossWeight = 0.0
        DistillationTeacherCsv = ""
        SampleWeightManifest = ""
        HardSampleManifest = ""
        AmbiguousSoftTargetManifest = ""
        TargetedMarginManifest = ""
        FocusNeighborBinaryManifest = ""
        ClassificationSourceContext = $false
        PairedViewTrain = $false
        AuxiliaryTrainData = ""
        MixStyle = $false
        TokenPruning = $true
        EarlyTokenMaskKeepRate = 1.0
        BBoxSpatialFusion = $true
        GatedRelativePositionAttention = $false
        VisualContrastAttention = $false
        FovealAggregatedAttention = $false
        DeformableSpatialAttention = $false
        CrossCovarianceAttention = $false
        DynamicGraphMixer = $false
        SoftMoePatchAdapter = $false
        DeepClassPrompt = $false
        LayerTokenFusion = $false
        PatchStyleRecalibration = $false
        BlockLocalPatchMixer = $false
        LocallyEnhancedFfn = $false
        ConcurrentLocalGlobalCoupling = $false
        BiLevelRoutingAttention = $true
        BiLevelRoutingAttentionLayers = "2"
        BiLevelRoutingAttentionRegionsPerAxis = 4
        BiLevelRoutingAttentionLocalContextKernelSize = 5
        DataCartography = $true
        TraceArchitecture = $true
        SkipFinalTest = $true
    }

    & $Launcher @Shared -RunName $ControlRunName -BiLevelRoutingAttentionTopK 16
    if ($LASTEXITCODE -ne 0) {
        throw "BRA topk16 matched control failed with exit code $LASTEXITCODE"
    }
    & $Launcher @Shared -RunName $CandidateRunName -BiLevelRoutingAttentionTopK 4
    if ($LASTEXITCODE -ne 0) {
        throw "BRA topk4 candidate failed with exit code $LASTEXITCODE"
    }

    $PairManifest = [ordered]@{
        method = "bi_level_routing_attention_a1_pair"
        protocol = "docs/TRKH_5CLASS_BILEVEL_ROUTING_ATTENTION_READINESS_PROTOCOL_20260716.md"
        git_head = $Head
        fold_data = (Resolve-Path -LiteralPath $FoldDataYaml).Path
        preflight_summary = (Resolve-Path -LiteralPath $PreflightSummary).Path
        control_run = (Resolve-Path -LiteralPath $ControlRunDir).Path
        candidate_run = (Resolve-Path -LiteralPath $CandidateRunDir).Path
        sole_causal_difference = "block2_bra_topk16_vs_topk4"
        official_validation_used = $false
        test_used = $false
        current_best_command_updated = $false
    }
    $PairManifestPath = Join-Path "runs" "bra_a1_pair_manifest_20260716.json"
    $PairManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $PairManifestPath -Encoding utf8
    Write-Host "BRA A1 pair completed: $PairManifestPath"
}
