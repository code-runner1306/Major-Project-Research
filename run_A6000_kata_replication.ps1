[CmdletBinding()]
param(
    [string]$Python = 'python',
    [string]$OutputRoot = '',
    [ValidateRange(1,3)][int]$Concurrent = 3,
    [ValidateRange(901,39600)][int]$MaxSeconds = 39600,
    [switch]$Evaluate,
    [string]$EvaluateRoot = 'output/kata_replication',
    [switch]$PlanOnly
)
# KATA Table 2 replication (Gated DeltaNet, dh=64) plus the rewrite-shortcut test, on one A6000.
# Trains exactly 3 models (learning rates 1e-3, 3e-3, 1e-2; seed 42), all three sharing GPU 0. Nothing is published.
#   powershell.exe -File .\run_A6000_kata_replication.ps1             # train
#   powershell.exe -File .\run_A6000_kata_replication.ps1 -Evaluate   # evaluate after training
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if ($Evaluate) {
        $evalOutput = "research_results/kata_replication_eval_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        & $Python eval_kata_replication.py --root $EvaluateRoot --output $evalOutput --device cuda
        if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
        Write-Host "Report: $evalOutput/REPORT.md"
        return
    }
    if (-not $OutputRoot) {
        $id = (Get-Date -Format 'yyyyMMddTHHmmssfff') + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
        $OutputRoot = Join-Path $PSScriptRoot "output/kata_replication/$id"
    }
    $arguments = @('run_kata_replication.py','--output',$OutputRoot,'--concurrent',"$Concurrent",'--max-seconds',"$MaxSeconds",'--device','0')
    if ($PlanOnly) { $arguments += '--plan-only' }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Training failed or interrupted; inspect $OutputRoot" }
    if (-not $PlanOnly) { Write-Host "Session report: $OutputRoot/session_status.json" }
} finally { Pop-Location }
