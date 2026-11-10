$ErrorActionPreference = 'Stop'
$script:Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $script:Root
$workerOverride = 0
$RunPilot = $false
$DryRun = $false
$showHelp = $false
for ($i = 0; $i -lt $args.Count; $i++) {
    switch -Regex ($args[$i]) {
        '^--workers$|^--parallel-workers$' {
            if ($i + 1 -ge $args.Count) { Write-Error "$($args[$i]) requires a positive integer."; exit 2 }
            $i++; if ($args[$i] -notmatch '^\d+$') { Write-Error 'Worker count must be an integer.'; exit 2 }
            $workerOverride = [int]$args[$i]; break
        }
        '^--workers=(\d+)$|^--parallel-workers=(\d+)$' {
            $workerOverride = [int]($args[$i] -replace '^.*=', ''); break
        }
        '^--pilot$' { $RunPilot = $true; break }
        '^--no-pilot$' { $RunPilot = $false; break }
        '^--dry-run$' { $DryRun = $true; break }
        '^-h$|^--help$' { $showHelp = $true; break }
        default { Write-Error "Unknown option: $($args[$i])"; exit 2 }
    }
}

function Show-Usage {
    @'
Usage: .\run_research_grid.ps1 [--workers N] [--pilot] [--dry-run]

The launcher trains every selected run across TinyStories, MQAR, and overwrite,
then evaluates/completes the selected runs across all three families.

Environment configuration (same defaults as run_research_grid.sh):
  MODELS="30M"                 Scales to include (default: 30M)
  SEEDS_10M="42,43,44,45,46"  Per-scale comma-separated seeds
  SEEDS_30M="42,43,44,45,46"
  SEEDS_55M="42"
  SEEDS_100M="42"
  MAX_OPTIMIZER_STEPS=1000    Training budget
  PARALLEL_WORKERS=1          Maximum concurrent training processes
  OUTPUT_DIR=output/full_research
  RECURRENCE_BACKEND=reference (or qualified cuda-fused)
  PYTHON_BIN=python           Python command; defaults to python
  PILOT_WARMUP=1 PILOT_STEPS=2
  PILOT_PARALLEL_SCALES=10M,30M PILOT_PARALLEL_WORKERS=2,3

--pilot runs the data-fed throughput and worker qualification trials, followed
by the CPU smoke suite. Parallel production runs automatically qualify first.
--dry-run prints planned commands without executing them.
'@ | Write-Host
}

if ($showHelp) { Show-Usage; exit 0 }
if ($workerOverride -gt 0) { $env:PARALLEL_WORKERS = "$workerOverride" }
if (-not $env:PYTHON_BIN) { $env:PYTHON_BIN = 'python' }
if (-not $env:MODELS) { $env:MODELS = '30M' }
if (-not $env:OUTPUT_DIR) { $env:OUTPUT_DIR = 'output/full_research' }
if (-not $env:MAX_OPTIMIZER_STEPS) { $env:MAX_OPTIMIZER_STEPS = '1000' }
if (-not $env:RECURRENCE_BACKEND) { $env:RECURRENCE_BACKEND = 'reference' }
if (-not $env:PARALLEL_WORKERS) { $env:PARALLEL_WORKERS = '1' }

if ($env:PARALLEL_WORKERS -notmatch '^([1-9][0-9]{0,2})$' -or [int]$env:PARALLEL_WORKERS -gt 128) {
    Write-Error 'PARALLEL_WORKERS must be an integer from 1 to 128.'; exit 2
}
$workerLimit = [int]$env:PARALLEL_WORKERS
if ($workerLimit -gt 1 -and -not $DryRun -and -not $RunPilot) {
    $RunPilot = $true
    Write-Host 'Parallel training requested; running the required capacity qualification pilot first.'
}

$models = @($env:MODELS -split '\s+' | Where-Object { $_ })
if ($models.Count -eq 0) { Write-Error 'MODELS must contain at least one scale.'; exit 2 }
if (@($models | Where-Object { $_ -notin @('10M','30M') }).Count -gt 0) {
    Write-Error 'This continuation plan supports only 10M and 30M.'; exit 2
}
$seedMap = @{
    '10M' = $(if ($env:SEEDS_10M) { $env:SEEDS_10M } else { '42,43,44,45,46' })
    '30M' = $(if ($env:SEEDS_30M) { $env:SEEDS_30M } else { '42,43,44,45,46' })
    '55M' = $(if ($env:SEEDS_55M) { $env:SEEDS_55M } else { '42' })
    '100M' = $(if ($env:SEEDS_100M) { $env:SEEDS_100M } else { '42' })
}
foreach ($model in $models) {
    if (-not $seedMap.ContainsKey($model)) { Write-Error "Unsupported model scale '$model'. Choose 10M, 30M, 55M, or 100M."; exit 2 }
    if (-not $seedMap[$model]) { Write-Error "No seeds configured for $model."; exit 2 }
}

function Timestamp { [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ') }
function Log-Progress([string]$Message) { Write-Host "[$(Timestamp)] $Message" }
function Invoke-Logged([string[]]$CommandArgs) {
    Log-Progress ('+' + ($CommandArgs -join ' '))
    if (-not $DryRun) {
        & $env:PYTHON_BIN @CommandArgs
        if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
    }
}

if (-not $DryRun) {
    $outputPath = Join-Path $script:Root $env:OUTPUT_DIR
    New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
    $script:RunLog = Join-Path $outputPath ("research_grid_{0}.log" -f [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
    Start-Transcript -Path $script:RunLog -Append | Out-Null
    Log-Progress "Run log: $script:RunLog"
}

try {
    if ($RunPilot) {
        $pilotScales = $models -join ','
        $estimateSeeds = @($models | ForEach-Object { $count = @($seedMap[$_] -split ',' | Where-Object { $_ }).Count; "${_}:$count" }) -join ','
        $warmup = if ($env:PILOT_WARMUP) { $env:PILOT_WARMUP } else { '1' }
        $steps = if ($env:PILOT_STEPS) { $env:PILOT_STEPS } else { '2' }
        $parallelScales = if ($env:PILOT_PARALLEL_SCALES) { $env:PILOT_PARALLEL_SCALES } else { '10M,30M' }
        $parallelCounts = if ($env:PILOT_PARALLEL_WORKERS) { $env:PILOT_PARALLEL_WORKERS } else { '2,3' }
        $extra = @()
        if ($workerLimit -gt 1) { $parallelScales = $pilotScales; $parallelCounts = "$workerLimit"; $extra = @('--pilot-require-parallel-workers', "$workerLimit") }
        Log-Progress "Starting pilot: scales=$pilotScales warmup=$warmup measured_steps=$steps data-fed families=tinystories,mqar,overwrite"
        Invoke-Logged (@('run_all.py','--pilot','--recurrence-backend',$env:RECURRENCE_BACKEND,'--output',$env:OUTPUT_DIR,'--max-optimizer-steps',$env:MAX_OPTIMIZER_STEPS,'--estimate-seeds',$estimateSeeds,'--pilot-scales',$pilotScales,'--pilot-warmup',$warmup,'--pilot-steps',$steps,'--pilot-data-fed','--pilot-parallel-scales',$parallelScales,'--pilot-parallel-workers',$parallelCounts) + $extra)
        $smokeOutput = Join-Path $env:OUTPUT_DIR ("pilot_smoke_{0}" -f [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
        Log-Progress "Starting post-pilot CPU smoke suite; output=$smokeOutput"
        Invoke-Logged @('run_all.py','--mode','smoke','--family','all','--output',$smokeOutput)
    }

    $families = @('tinystories','mqar','overwrite')
    $trainingTasks = [System.Collections.Generic.List[object]]::new()
    foreach ($family in @('mqar','overwrite')) {
        if ('10M' -in $models) {
            foreach ($seed in ($seedMap['10M'] -split ',' | Where-Object { $_ })) {
                foreach ($variant in @('scalar','channel-wise')) {
                    $trainingTasks.Add([pscustomobject]@{ Family=$family; Model='10M'; Seed=$seed; Variant=$variant })
                }
            }
        }
    }
    if ('30M' -in $models) {
        # This checkpoint has already completed training; retain it in the plan
        # so it is evaluated after the remaining 30M training finishes.
        $trainingTasks.Add([pscustomobject]@{ Family='tinystories'; Model='30M'; Seed='42'; Variant='channel-wise'; SkipTraining=$true })
        foreach ($family in $families) {
            foreach ($seed in ($seedMap['30M'] -split ',' | Where-Object { $_ })) {
                foreach ($variant in @('scalar','channel-wise')) {
                    if ($family -eq 'tinystories' -and $seed -eq '42' -and $variant -eq 'channel-wise') { continue }
                    $trainingTasks.Add([pscustomobject]@{ Family=$family; Model='30M'; Seed=$seed; Variant=$variant; SkipTraining=$false })
                }
            }
        }
    }
    if ($models.Count -eq 1 -and $models[0] -eq '10M') {
        Log-Progress '10M continuation plan includes MQAR and overwrite only; existing TinyStories runs are preserved.'
    }
    foreach ($task in $trainingTasks) {
        $family=$task.Family; $model=$task.Model; $seed=$task.Seed; $variant=$task.Variant
        if ($task.SkipTraining) {
            Log-Progress "Skipping already-trained run; will evaluate after training stage: dataset=$family model=$model variant=$variant seed=$seed"
            continue
        }
        Log-Progress "Training task dataset=$family model=$model variant=$variant seed=$seed"
        $argsList = @('run_all.py','--mode','production','--recurrence-backend',$env:RECURRENCE_BACKEND,'--family',$family,'--models',$model,'--seeds',$seed,'--variant',$variant,'--max-optimizer-steps',$env:MAX_OPTIMIZER_STEPS,'--output',$env:OUTPUT_DIR,'--concurrent-training-runs',"$workerLimit",'--train-only','--skip-report')
        Invoke-Logged $argsList
    }
    foreach ($family in $families) {
        Log-Progress "Starting evaluation/completion stage for dataset=$family"
        foreach ($model in $models) {
            $modelWorkers = $workerLimit
            if ($workerLimit -gt 1 -and -not $DryRun) {
                $qualificationPath = Join-Path (Join-Path $script:Root $env:OUTPUT_DIR) 'parallel_worker_qualification.json'
                $qualification = Get-Content -Raw $qualificationPath | ConvertFrom-Json
                $statuses = $qualification.scale_dataset_statuses.$model
                if (-not $statuses -or @($statuses.PSObject.Properties | Where-Object { $_.Value -ne 'complete' }).Count -gt 0) { $modelWorkers = 1 }
            }
            $evaluationTasks = @($trainingTasks | Where-Object { $_.Family -eq $family -and $_.Model -eq $model })
            foreach ($task in $evaluationTasks) {
                $variant=$task.Variant; $seed=$task.Seed
                Log-Progress "Evaluating dataset=$family model=$model variant=$variant seed=$seed"
                try {
                    Invoke-Logged @('run_all.py','--mode','production','--recurrence-backend',$env:RECURRENCE_BACKEND,'--family',$family,'--models',$model,'--seeds',$seed,'--variant',$variant,'--max-optimizer-steps',$env:MAX_OPTIMIZER_STEPS,'--output',$env:OUTPUT_DIR,'--concurrent-training-runs',"$modelWorkers",'--evaluate-only')
                } catch {
                    throw "Evaluation failed: dataset=$family model=$model variant=$variant seed=$seed. $($_.Exception.Message)"
                }
            }
        }
    }
    Write-Host "Research grid complete. Results are under: $env:OUTPUT_DIR"
} catch {
    Write-Error $_
    exit 1
} finally {
    if (-not $DryRun) { Stop-Transcript | Out-Null }
}
