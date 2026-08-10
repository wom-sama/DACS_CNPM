param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$Checkpoint = "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt",
    [string]$LauncherArgs = "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\launcher_args.json",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
    [string]$CidtSummary = "runs\audit_cidt_readiness_full_train_20260714\summary.json",
    [string]$CidtPredictions = "runs\audit_cidt_readiness_full_train_20260714\predictions_all_conditions.csv",
    [string]$Protocol = "docs\TRKH_5CLASS_XCA_DUAL_AXIS_READINESS_PROTOCOL_20260715.md",
    [string]$OfficialXcitRoot = "C:\Users\ADMIN\AppData\Local\Temp\trkh_next_arch_sources_20260715\xcit",
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
        $OutputDir = "runs\audit_xca_dual_axis_readiness_$stamp"
    }
    $Arguments = @(
        "-m", "trkh.tools.audit_xca_dual_axis_readiness",
        "--checkpoint", $Checkpoint,
        "--launcher-args", $LauncherArgs,
        "--data", $DataYaml,
        "--cidt-summary", $CidtSummary,
        "--cidt-predictions", $CidtPredictions,
        "--protocol", $Protocol,
        "--official-xcit-root", $OfficialXcitRoot,
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
        throw "XCA Stage-A audit failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
