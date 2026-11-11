[CmdletBinding()]
param(
    [string]$OutputRoot = "output/granularity_init_study",
    [string]$Python = "python",
    [int]$TrainingWorkers = 1,
    [ValidateSet('E0','E1','E2','E3','E4','E5','E6')][string]$StartAt = 'E0',
    [ValidateSet('E0','E1','E2','E3','E4','E5','E6')][string]$StopAfter,
    [switch]$EvaluateOnly,
    [switch]$DryRun,
    [string]$StudyConfig = "study_configs/granularity_init_study.json"
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repo
try {
    $arguments = @('study_runner.py', '--study-config', $StudyConfig, '--output-root', $OutputRoot,
        '--python', $Python, '--training-workers', "$TrainingWorkers", '--start-at', $StartAt)
    if ($StopAfter) { $arguments += @('--stop-after', $StopAfter) }
    if ($EvaluateOnly) { $arguments += '--evaluate-only' }
    if ($DryRun) { $arguments += '--dry-run' }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Study runner failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
