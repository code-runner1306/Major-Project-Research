$ErrorActionPreference = 'Stop'
$script:Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $script:Root

if (-not $env:PYTHON_BIN) { $env:PYTHON_BIN = 'python' }
if (-not $env:OUTPUT_DIR) { $env:OUTPUT_DIR = 'output/full_research' }
if (-not $env:RECURRENCE_BACKEND) { $env:RECURRENCE_BACKEND = 'reference' }
if (-not $env:MAX_OPTIMIZER_STEPS) { $env:MAX_OPTIMIZER_STEPS = '1000' }

$outputPath = Join-Path $script:Root $env:OUTPUT_DIR
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$logPath = Join-Path $outputPath ("evaluate_10m_{0}.log" -f [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
Start-Transcript -Path $logPath | Out-Null

try {
    foreach ($family in @('tinystories', 'mqar', 'overwrite')) {
        foreach ($variant in @('scalar', 'channel-wise')) {
            foreach ($seed in @(42, 43, 44, 45, 46)) {
                Write-Host "[$([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))] Evaluating 10M: family=$family variant=$variant seed=$seed"
                $pythonArgs = @(
                    'run_all.py', '--mode', 'production', '--recurrence-backend', $env:RECURRENCE_BACKEND,
                    '--family', $family, '--models', '10M', '--seeds', "$seed", '--variant', $variant,
                    '--max-optimizer-steps', $env:MAX_OPTIMIZER_STEPS, '--output', $env:OUTPUT_DIR, '--evaluate-only'
                )
                & $env:PYTHON_BIN @pythonArgs
                if ($LASTEXITCODE -ne 0) {
                    throw "Evaluation failed: family=$family model=10M variant=$variant seed=$seed (exit $LASTEXITCODE). See $logPath"
                }
            }
        }
    }
    Write-Host "All requested 10M evaluation runs are complete. Log: $logPath"
} catch {
    Write-Error $_
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
