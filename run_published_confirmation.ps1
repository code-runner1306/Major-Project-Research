[CmdletBinding()]
param(
    [string]$Python = 'python',
    [string]$CheckpointRoot = $PSScriptRoot,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'auto',
    [string]$OutputRoot = '',
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not $OutputRoot) {
        $OutputRoot = Join-Path $PSScriptRoot ('research_results/published_confirmation_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
    }
    $arguments = @('evaluate_published_checkpoints.py','--checkpoint-root',$CheckpointRoot,
                   '--output',$OutputRoot,'--device',$Device)
    if ($PrepareOnly) { $arguments += '--prepare-only' }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Checkpoint preparation or confirmation evaluation failed; inspect the output.' }
    if (-not $PrepareOnly) { Write-Host "Results: $OutputRoot/confirmation_results.json" }
} finally {
    Pop-Location
}
