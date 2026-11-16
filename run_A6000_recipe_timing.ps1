<# Single-GPU timing only: sequential versus two concurrent workers.
   Run from a CUDA-enabled Python environment.
   .\run_A6000_recipe_timing.ps1 -Family both -Python .\.venv\Scripts\python.exe
   Dependencies: use the repository requirements with a CUDA wheel compatible
   with this machine's driver. This launcher does not replace your PyTorch.
#>
[CmdletBinding()]
param(
    [ValidateSet('mqar','overwrite','both')][string]$Family = 'both',
    [string]$Python = 'python',
    [ValidateRange(50,196)][int]$Steps = 50,
    [ValidateRange(1,64)][int]$ProjectionEpochs = 64,
    [string]$OutputRoot = '',
    [switch]$CpuSmoke
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
$names = @('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS')
$saved = @{}
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
try {
    $env:CUDA_VISIBLE_DEVICES = '0'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    if (-not $OutputRoot) {
        $OutputRoot = Join-Path $PSScriptRoot ('output/a6000_recipe_timing_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
    }
    $arguments = @('benchmark_single_gpu_recipe.py', '--family', $Family, '--output', $OutputRoot,
                   '--steps', "$Steps", '--epochs', "$ProjectionEpochs")
    if ($CpuSmoke) { $arguments += '--cpu-smoke'; $arguments[$arguments.IndexOf('--steps') + 1] = '2' }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed; inspect logs and benchmark_report.json in $OutputRoot" }
    Write-Host "Completed. Report: $OutputRoot/benchmark_report.json"
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
    Pop-Location
}
