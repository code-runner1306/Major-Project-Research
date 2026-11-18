[CmdletBinding()]
param(
    [string[]]$Seeds = @('42','43','44'),
    [string]$Variant = 'scalar',
    [string]$Python = 'python',
    [string]$OutputRoot = '',
    [ValidateRange(901,39600)][int]$MaxSeconds = 39600,
    [switch]$Evaluate,
    [switch]$CpuSmoke,
    [switch]$PlanOnly
)
# KATA-style repeated-key overwrite study (protocol kata-overwrite-v1) on one A6000.
# Default: 3 models (scalar x seeds 42, 43, 44), trained one after another on GPU 0. Nothing is published.
#   powershell.exe -File .\run_A6000_kata_overwrite.ps1             # train
#   powershell.exe -File .\run_A6000_kata_overwrite.ps1 -Evaluate   # evaluate after training
$ErrorActionPreference = 'Stop'
$Seeds = @($Seeds | ForEach-Object { "$_" -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($seed in $Seeds) { if ($seed -notmatch '^\d+$') { throw "Seeds must be nonnegative integers, got '$seed'" } }
if (@('scalar','channel-wise') -notcontains $Variant) { throw "Variant must be scalar or channel-wise" }
Push-Location $PSScriptRoot
try {
    if ($Evaluate) {
        $evalOutput = "research_results/kata_overwrite_eval_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        & $Python eval_kata_overwrite.py --output $evalOutput --seeds @Seeds
        if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
        Write-Host "Report: $evalOutput/REPORT.md"
        return
    }
    if (-not $OutputRoot) {
        $id = (Get-Date -Format 'yyyyMMddTHHmmssfff') + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
        $OutputRoot = Join-Path $PSScriptRoot "output/kata_overwrite_study/kata-overwrite-v1/$id"
    }
    $arguments = @('run_kata_overwrite_study.py','--devices','0','--variants',$Variant,'--output',$OutputRoot,
                   '--max-seconds',"$MaxSeconds",'--seeds') + $Seeds
    if ($CpuSmoke) { $arguments += '--cpu-smoke' }
    if ($PlanOnly) { $arguments += '--plan-only' }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Study failed or interrupted; inspect $OutputRoot" }
    if (-not $PlanOnly) { Write-Host "Session report: $OutputRoot/session_status.json" }
} finally { Pop-Location }
