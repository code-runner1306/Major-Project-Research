[CmdletBinding()]
param(
    [ValidateSet('mqar','overwrite')][string[]]$Family = @('mqar','overwrite'),
    [int[]]$Seeds,
    [ValidateRange(1,100)][int]$SeedCount = 1,
    [ValidateRange(0,2147483000)][int]$StartSeed = 42,
    [ValidateSet('default','calibrated')][string[]]$Initialization = @('default','calibrated'),
    [ValidateSet('pilot','explore','study')][string]$Stage = 'pilot',
    [string]$Devices = '0',
    [string]$QualificationRoot = '',
    [string]$Python = 'python',
    [string]$OutputRoot = '',
    [ValidateRange(901,39600)][int]$MaxSeconds = 39600,
    [switch]$Publish = $true,
    [switch]$CpuSmoke,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
if ($PSBoundParameters.ContainsKey('Seeds') -and $PSBoundParameters.ContainsKey('SeedCount')) {
    throw 'Use Seeds or SeedCount, not both.'
}
if (-not $OutputRoot) {
    $id = (Get-Date -Format 'yyyyMMddTHHmmssfff') + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
    $OutputRoot = Join-Path $PSScriptRoot "output/calibrated_recipe_study/matched-decay-v1/$id"
}
Push-Location $PSScriptRoot
try {
    $arguments = @('run_recipe_study.py','--stage',$Stage,'--devices',$Devices,'--output',$OutputRoot,
                   '--max-seconds',"$MaxSeconds",'--family') + $Family + @('--initialization') + $Initialization
    if ($Seeds) { $arguments += @('--seeds') + @($Seeds | ForEach-Object { "$_" }) }
    elseif ($Stage -eq 'pilot' -and -not $PSBoundParameters.ContainsKey('StartSeed') -and -not $PSBoundParameters.ContainsKey('SeedCount')) {
        $arguments += @('--seeds','41')
    } else { $arguments += @('--seed-count',"$SeedCount",'--start-seed',"$StartSeed") }
    if ($QualificationRoot) { $arguments += @('--qualification-root',$QualificationRoot) }
    if ($Publish -and -not $CpuSmoke) { $arguments += '--publish' } else { $arguments += '--no-publish' }
    if ($CpuSmoke) { $arguments += '--cpu-smoke' }
    if ($PlanOnly) { $arguments += '--plan-only' }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Study failed or interrupted; inspect $OutputRoot" }
    if (-not $PlanOnly) { Write-Host "Session report: $OutputRoot/session_status.json" }
} finally { Pop-Location }
