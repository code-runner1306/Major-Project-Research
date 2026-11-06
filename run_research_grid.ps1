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

Stages run in this order: TinyStories -> MQAR -> overwrite. Each dataset is
trained first, then evaluated/completed sequentially before the next dataset.

Environment configuration (same defaults as run_research_grid.sh):
  MODELS="10M 30M 55M 100M"   Scales to include
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
if (-not $env:MODELS) { $env:MODELS = '10M 30M 55M 100M' }
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
    foreach ($family in $families) {
        Log-Progress "Starting training stage for dataset=$family"
        foreach ($model in $models) {
            $modelWorkers = $workerLimit
            if ($workerLimit -gt 1 -and -not $DryRun) {
                $qualificationPath = Join-Path (Join-Path $script:Root $env:OUTPUT_DIR) 'parallel_worker_qualification.json'
                if (-not (Test-Path $qualificationPath)) { throw "Missing parallel qualification report: $qualificationPath" }
                $qualification = Get-Content -Raw $qualificationPath | ConvertFrom-Json
                $statuses = $qualification.scale_dataset_statuses.$model
                if (-not $statuses -or @($statuses.PSObject.Properties | Where-Object { $_.Value -ne 'complete' }).Count -gt 0) {
                    $modelWorkers = 1
                    Log-Progress "Training dataset=$family model=$model sequentially; requested worker count failed a VRAM capacity gate."
                }
            }
            if ($modelWorkers -eq $workerLimit) { Log-Progress "Training dataset=$family model=$model with up to $modelWorkers workers." }
            $variants = @('scalar','channel-wise')
            if ($env:ENABLE_PARAMETER_MATCHED_SCALAR_CONTROL -match '^(1|true|yes)$') { $variants += 'scalar-matched' }
            $tasks = @()
            foreach ($seed in ($seedMap[$model] -split ',' | Where-Object { $_ })) {
                foreach ($variant in $variants) {
                    $argsList = @('run_all.py','--mode','production','--recurrence-backend',$env:RECURRENCE_BACKEND,'--family',$family,'--models',$model,'--seeds',$seed,'--variant',$variant,'--max-optimizer-steps',$env:MAX_OPTIMIZER_STEPS,'--output',$env:OUTPUT_DIR,'--concurrent-training-runs',"$modelWorkers",'--train-only','--skip-report')
                    Log-Progress "Queue training task dataset=$family model=$model variant=$variant seed=$seed (worker limit=$modelWorkers)"
                    if ($DryRun) { Invoke-Logged $argsList; continue }
                    while (@($tasks | Where-Object { -not $_.Process.HasExited }).Count -ge $modelWorkers) { Start-Sleep -Seconds 2 }
                    $taskDir = Join-Path $env:TEMP ("research_grid_{0}_{1}" -f $PID, [guid]::NewGuid().ToString('N'))
                    New-Item -ItemType Directory -Force -Path $taskDir | Out-Null
                    $outFile = Join-Path $taskDir 'stdout.log'; $errFile = Join-Path $taskDir 'stderr.log'
                    $quotedArgs = @($argsList | ForEach-Object { '"' + ($_ -replace '"','\"') + '"' }) -join ' '
                    Log-Progress ('+' + $env:PYTHON_BIN + ' ' + ($argsList -join ' '))
                    $proc = Start-Process -FilePath $env:PYTHON_BIN -ArgumentList $quotedArgs -WorkingDirectory $script:Root -RedirectStandardOutput $outFile -RedirectStandardError $errFile -PassThru
                    $tasks += [pscustomobject]@{ Process=$proc; Out=$outFile; Err=$errFile; Dir=$taskDir; Family=$family; Model=$model; Variant=$variant; Seed=$seed }
                }
            }
            foreach ($task in $tasks) {
                $task.Process.WaitForExit()
                $task.Process.Refresh()
                if (Test-Path $task.Out) { Get-Content $task.Out | ForEach-Object { Write-Host $_ } }
                if (Test-Path $task.Err) { Get-Content $task.Err | ForEach-Object { Write-Host $_ } }
                Remove-Item -LiteralPath $task.Dir -Recurse -Force -ErrorAction SilentlyContinue
                if ($null -eq $task.Process.ExitCode -or $task.Process.ExitCode -ne 0) { throw "Training worker failed: dataset=$($task.Family) model=$($task.Model) variant=$($task.Variant) seed=$($task.Seed), exit=$($task.Process.ExitCode). Stopping before evaluation." }
            }
        }
        Log-Progress "Starting evaluation/completion stage for dataset=$family"
        foreach ($model in $models) {
            $modelWorkers = $workerLimit
            if ($workerLimit -gt 1 -and -not $DryRun) {
                $qualificationPath = Join-Path (Join-Path $script:Root $env:OUTPUT_DIR) 'parallel_worker_qualification.json'
                $qualification = Get-Content -Raw $qualificationPath | ConvertFrom-Json
                $statuses = $qualification.scale_dataset_statuses.$model
                if (-not $statuses -or @($statuses.PSObject.Properties | Where-Object { $_.Value -ne 'complete' }).Count -gt 0) { $modelWorkers = 1 }
            }
            $variants = @('scalar','channel-wise')
            if ($env:ENABLE_PARAMETER_MATCHED_SCALAR_CONTROL -match '^(1|true|yes)$') { $variants += 'scalar-matched' }
            foreach ($variant in $variants) { foreach ($seed in ($seedMap[$model] -split ',' | Where-Object { $_ })) {
                Log-Progress "Evaluating dataset=$family model=$model variant=$variant seed=$seed"
                try {
                    Invoke-Logged @('run_all.py','--mode','production','--recurrence-backend',$env:RECURRENCE_BACKEND,'--family',$family,'--models',$model,'--seeds',$seed,'--variant',$variant,'--max-optimizer-steps',$env:MAX_OPTIMIZER_STEPS,'--output',$env:OUTPUT_DIR,'--concurrent-training-runs',"$modelWorkers",'--evaluate-only')
                } catch {
                    throw "Evaluation failed: dataset=$family model=$model variant=$variant seed=$seed. $($_.Exception.Message)"
                }
            } }
        }
    }
    Write-Host "Research grid complete. Results are under: $env:OUTPUT_DIR"
} catch {
    Write-Error $_
    exit 1
} finally {
    if (-not $DryRun) { Stop-Transcript | Out-Null }
}
