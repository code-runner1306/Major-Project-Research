[CmdletBinding()]
param(
    [ValidateSet('mqar','overwrite')][string[]]$Family = @('mqar','overwrite'),
    [int[]]$Seeds,
    [ValidateRange(1,100)][int]$SeedCount = 3,
    [ValidateRange(0,2147483000)][int]$StartSeed = 42,
    [ValidateSet('explore')][string]$Stage = 'explore',
    [string]$Devices = '0',
    [string]$Python = 'python',
    [string]$OutputRoot = '',
    [ValidateRange(901,39600)][int]$MaxSeconds = 39600,
    [switch]$Publish = $false,
    [switch]$CpuSmoke,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
if ($PSBoundParameters.ContainsKey('Seeds') -and $PSBoundParameters.ContainsKey('SeedCount')) {
    throw 'Use Seeds or SeedCount, not both.'
}
if (-not $OutputRoot) {
    $id = (Get-Date -Format 'yyyyMMddTHHmmssfff') + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
    $OutputRoot = Join-Path $PSScriptRoot "output/joint_recipe_study/joint-timescale-sensitivity-v1/$id"
}
Push-Location $PSScriptRoot
try {
    $arguments = @('run_joint_recipe_study.py','--stage',$Stage,'--devices',$Devices,'--output',$OutputRoot,
                   '--max-seconds',"$MaxSeconds",'--family') + $Family + @('--initialization','joint')
    if ($Seeds) { $arguments += @('--seeds') + @($Seeds | ForEach-Object { "$_" }) }
    else { $arguments += @('--seed-count',"$SeedCount",'--start-seed',"$StartSeed") }
    if ($Publish -and -not $CpuSmoke) { $arguments += '--publish' } else { $arguments += '--no-publish' }
    if ($CpuSmoke) { $arguments += '--cpu-smoke' }
    if ($PlanOnly) { $arguments += '--plan-only' }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Study failed or interrupted; inspect $OutputRoot" }
    if (-not $PlanOnly) { Write-Host "Session report: $OutputRoot/session_status.json" }
} finally { Pop-Location }
