<# Evaluation only. Run after the training job releases the GPU.
   .\run_A6000_decay_boundary.ps1
   .\run_A6000_decay_boundary.ps1 -Family overwrite -Seeds 42
   Publishing is disabled: this launcher has no upload or Git push operation.
#>
[CmdletBinding()]
param(
    [ValidateSet('mqar','overwrite')][string[]]$Family = @('mqar','overwrite'),
    [int[]]$Seeds = @(42,43,44),
    [string]$CheckpointRoot = 'output/calibrated_recipe_study/matched-decay-v1',
    [string]$JointRoot = '',
    [string[]]$RunId,
    [string[]]$JointRunId,
    [string]$OutputRoot = '',
    [string]$Python = 'python',
    [ValidateRange(1,128)][int]$BatchSize = 32,
    [ValidateRange(1,8192)][int]$Examples = 512,
    [switch]$Resume,
    [switch]$PlanOnly,
    [switch]$CpuSmoke
)
$ErrorActionPreference = 'Stop'
if ($Resume -and -not $OutputRoot) { throw 'Resume requires the original OutputRoot.' }
Push-Location $PSScriptRoot
$names = @('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS')
$saved = @{}
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
try {
    $env:CUDA_VISIBLE_DEVICES = '0'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    if (-not $OutputRoot) {
        $id = (Get-Date -Format 'yyyyMMddTHHmmssfff') + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
        $OutputRoot = Join-Path $PSScriptRoot "research_results/dynamic_decay_boundary_v1/$id"
    }
    $arguments = @('evaluate_decay_boundary.py','--root',$CheckpointRoot,'--output',$OutputRoot,
        '--batch',"$BatchSize",'--examples',"$Examples",'--family') + $Family + @('--seeds') + @($Seeds | ForEach-Object { "$_" })
    if ($JointRoot) { $arguments += @('--joint-root',$JointRoot) }
    if ($RunId) { $arguments += @('--run-id') + $RunId }
    if ($JointRunId) { $arguments += @('--joint-run-id') + $JointRunId }
    if ($Resume) { $arguments += '--resume' }
    if ($PlanOnly) { $arguments += '--plan-only' }
    if ($CpuSmoke) { $arguments += '--cpu-smoke' }
    Write-Host 'Evaluation only: one worker on GPU 0; publish=false.'
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed; inspect $OutputRoot and resume with the same settings." }
    if (-not $PlanOnly) { Write-Host "Report: $OutputRoot/REPORT.md" }
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name,$saved[$name],'Process') }
    Pop-Location
}
