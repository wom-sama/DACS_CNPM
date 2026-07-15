param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$Checkpoint = "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt",
    [string]$LauncherArgs = "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\launcher_args.json",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
    [string]$CidtSummary = "runs\audit_cidt_readiness_full_train_20260714\summary.json",
    [string]$CidtPredictions = "runs\audit_cidt_readiness_full_train_20260714\predictions_all_conditions.csv",
    [string]$Protocol = "docs\TRKH_5CLASS_SOFT_MOE_PATCH_ADAPTER_READINESS_PROTOCOL_20260715.md",
    [string]$VmoeRoot = "C:\Users\ADMIN\AppData\Local\Temp\trkh_soft_moe_primary_20260715\vmoe",
    [string]$SoftMoePaper = "C:\Users\ADMIN\AppData\Local\Temp\trkh_soft_moe_primary_20260715\soft_moe_arxiv_2308.00951.pdf",
    [string]$SweetSpotPaper = "C:\Users\ADMIN\AppData\Local\Temp\trkh_soft_moe_primary_20260715\vision_moe_sweet_spot_arxiv_2411.18322v2.pdf",
    [string]$OutputDir = "",
    [int]$NumWorkers = 4,
    [switch]$Preflight
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    if ([string]::IsNullOrWhiteSpace($OutputDir)) {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $OutputDir = "runs\audit_soft_moe_patch_adapter_stage_a_$stamp"
    }

    $RequiredPaths = [ordered]@{
        python = $Python
        checkpoint = $Checkpoint
        launcher_args = $LauncherArgs
        data_yaml = $DataYaml
        cidt_summary = $CidtSummary
        cidt_predictions = $CidtPredictions
        protocol = $Protocol
        vmoe_root = $VmoeRoot
        soft_moe_paper = $SoftMoePaper
        sweet_spot_paper = $SweetSpotPaper
    }
    foreach ($Entry in $RequiredPaths.GetEnumerator()) {
        if (-not (Test-Path -LiteralPath $Entry.Value)) {
            throw "Missing required $($Entry.Key): $($Entry.Value)"
        }
    }

    $Arguments = @(
        "-m", "trkh.tools.audit_soft_moe_patch_adapter_readiness",
        "--checkpoint", $Checkpoint,
        "--launcher-args", $LauncherArgs,
        "--data", $DataYaml,
        "--cidt-summary", $CidtSummary,
        "--cidt-predictions", $CidtPredictions,
        "--protocol", $Protocol,
        "--vmoe-root", $VmoeRoot,
        "--soft-moe-paper", $SoftMoePaper,
        "--sweet-spot-paper", $SweetSpotPaper,
        "--output-dir", $OutputDir,
        "--num-workers", "$NumWorkers"
    )

    if ($Preflight) {
        [ordered]@{
            python = $Python
            arguments = $Arguments
            output_dir = $OutputDir
            validation_used = $false
            test_used = $false
        } | ConvertTo-Json -Depth 4
        return
    }

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Soft-MoE patch-adapter Stage-A audit failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
