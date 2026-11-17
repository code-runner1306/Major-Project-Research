[CmdletBinding()]
param(
    [ValidateSet('A0_control','A1_varied_distance','A2_no_block_first','A3_mixed_rewrites')]
    [string[]]$Arms = @('A2_no_block_first','A1_varied_distance','A3_mixed_rewrites'),
    [int[]]$Seeds = @(42,43,44),
    [string]$Devices = '0',
    [string]$Python = 'python',
    [string]$OutputRoot = '',
    [ValidateRange(901,39600)][int]$MaxSeconds = 39600,
    [switch]$CheckControl,
    [switch]$CpuSmoke,
    [switch]$PlanOnly
)
# Layout-fix training study (protocol layout-fix-v1). Publishing is never performed by this launcher.
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if ($CheckControl) {
        & $Python run_layout_fix_study.py --check-control --seeds @($Seeds | ForEach-Object { "$_" })
        if ($LASTEXITCODE -ne 0) { throw 'Control check failed' }
        return
    }
    if (-not $OutputRoot) {
        $id = (Get-Date -Format 'yyyyMMddTHHmmssfff') + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
        $OutputRoot = Join-Path $PSScriptRoot "output/layout_fix_study/layout-fix-v1/$id"
    }
    $arguments = @('run_layout_fix_study.py','--devices',$Devices,'--output',$OutputRoot,'--max-seconds',"$MaxSeconds",
                   '--arm') + $Arms + @('--seeds') + @($Seeds | ForEach-Object { "$_" })
    if ($CpuSmoke) { $arguments += '--cpu-smoke' }
    if ($PlanOnly) { $arguments += '--plan-only' }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Study failed or interrupted; inspect $OutputRoot" }
    if (-not $PlanOnly) { Write-Host "Session report: $OutputRoot/session_status.json" }
} finally { Pop-Location }
