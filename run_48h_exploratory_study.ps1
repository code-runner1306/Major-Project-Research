[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('mqar', 'overwrite')]
    [string]$Family,
    [string]$Python = "python",
    [string]$OutputRoot,
    [switch]$DryRun,
    [switch]$EvaluateOnly,
    [string]$StudyConfig = "study_configs/granularity_init_study.json"
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repo
try {
    $studyConfigPath = [System.IO.Path]::GetFullPath((Join-Path $repo $StudyConfig))
    if (-not (Test-Path -LiteralPath $studyConfigPath -PathType Leaf)) {
        throw "Study config not found: $studyConfigPath"
    }
    $study = Get-Content -LiteralPath $studyConfigPath -Raw | ConvertFrom-Json
    if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
        $OutputRoot = "output/granularity_48h_exploratory_lr3e-3_$Family"
    }
    $root = [System.IO.Path]::GetFullPath((Join-Path $repo $OutputRoot))
    $outputBase = [System.IO.Path]::GetFullPath((Join-Path $repo 'output'))
    if (-not $root.StartsWith($outputBase + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "OutputRoot must be a new or existing subfolder of $outputBase"
    }

    $conditions = @(
        [ordered]@{ id = 'C1-scalar-default'; variant = 'scalar'; initialization = 'default' },
        [ordered]@{ id = 'C3-channel-default'; variant = 'channel-wise'; initialization = 'default' },
        [ordered]@{ id = 'C4-scalar-calibrated'; variant = 'scalar'; initialization = 'calibrated-geometric' },
        [ordered]@{ id = 'C6-channel-calibrated'; variant = 'channel-wise'; initialization = 'calibrated-geometric' }
    )
    $families = @($Family)
    $scale = '10M'
    $seed = 42
    $stepBudget = 1000
    $learningRate = 0.003
    $sourceBudget = $study.budgets.'10M'
    $trainingBudget = [ordered]@{
        max_optimizer_steps = $stepBudget
        batch_size = [int]$sourceBudget.batch_size
        grad_accum_steps = [int]$sourceBudget.grad_accum_steps
        context_length = [int]$sourceBudget.context_length
        validation_steps = 100
        evaluation_interval = 100
        learning_rate = $learningRate
    }
    $calibration = $study.calibration
    $retrievalPilot = $study.retrieval | ConvertTo-Json -Depth 20 | ConvertFrom-Json
    $retrievalPilot.validation_examples = 512
    $retrievalPilot.held_out_examples = 512
    $tasks = @()
    foreach ($family in $families) {
        foreach ($condition in $conditions) {
            $runDir = Join-Path $root ("runs\{0}\{1}\{2}\seed_{3}" -f $family, $scale, $condition.id, $seed)
            $tasks += [pscustomobject]@{
                experiment_id = 'granularity-init-48h-exploratory-lr3e-3-v1'
                condition_id = $condition.id
                family = $family
                scale = $scale
                variant = $condition.variant
                initialization = $condition.initialization
                seed = $seed
                output_dir = [System.IO.Path]::GetFullPath($runDir)
                max_optimizer_steps = $trainingBudget.max_optimizer_steps
                batch_size = $trainingBudget.batch_size
                grad_accum_steps = $trainingBudget.grad_accum_steps
                context_length = $trainingBudget.context_length
                validation_steps = $trainingBudget.validation_steps
                evaluation_interval = $trainingBudget.evaluation_interval
                learning_rate = $trainingBudget.learning_rate
                calibration_min_half_life = [int]$calibration.min_half_life
                calibration_max_half_life = [int]$calibration.max_half_life
                calibration_examples = [int]$calibration.examples
                retrieval_config = $retrievalPilot
                run_mode = 'production'
                recurrence_backend = 'reference'
                state_accumulator_dtype = 'input'
                recurrence_chunk_size = 64
                validation_answer_accuracy = $true
                concurrent_training_runs = 1
            }
        }
    }

    Write-Host ("Exploratory study: {0} 10M runs; family={1}; conditions={2}; seed={3}; steps={4}; LR={5}; workers=1" -f `
        $tasks.Count, $Family, ($conditions.id -join ','), $seed, $stepBudget, $learningRate)
    Write-Host "Output root: $root"
    Write-Host "Each task trains for 1,000 steps and validates every 100 steps. Existing compatible tasks are reused; partial or incompatible task folders stop the script without overwrite."
    Write-Host "Old-timing baseline: about 9 h 20 min training only for this family's four runs. This excludes the revised 100-step validation overhead, calibration, and evaluation; total runtime is not guaranteed to fit within 48 hours."
    Write-Host "Synthetic validation and held-out evaluation use 512 examples each per run; the training set remains 50,000 examples."
    Write-Host "Learning rate 3e-3 is fixed for all runs as a literature-informed choice; it is not tuned on these runs."
    Write-Host "This is an exploratory, one-seed, 1,000-step result; it is not the full 5-seed / token-budget study."

    $master = Join-Path $root ("logs\run_48h_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
    $script:MasterLog = $master
    $script:Records = [System.Collections.Generic.List[object]]::new()

    function Write-RunLog([string]$Message) {
        $line = "[{0}] {1}" -f ([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')), $Message
        Write-Host $line
        if ($DryRun) { return }
        Add-Content -LiteralPath $script:MasterLog -Value $line -Encoding UTF8
    }

    function Test-TaskComplete($Task, [string]$Operation, [string]$TaskPath) {
        $probe = "import json,sys; from study_runner import task_complete; print('COMPLETE' if task_complete(json.load(open(sys.argv[1], encoding='utf-8-sig')), sys.argv[2]) else 'INCOMPLETE')"
        $result = & $Python -c $probe $TaskPath $Operation 2>&1
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            throw "Could not validate existing study artifacts with Python ($code): $($result -join ' ')"
        }
        $lastOutput = @($result | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ } | Select-Object -Last 1)
        return ($lastOutput.Count -eq 1 -and $lastOutput[0] -eq 'COMPLETE')
    }

    function Invoke-StudyTask($Task, [string]$Operation) {
        $taskDir = Join-Path $root 'task_configs'
        $logDir = Join-Path $root 'logs'
        New-Item -ItemType Directory -Force -Path $taskDir, $logDir | Out-Null
        $tag = "{0}_{1}_{2}_seed_{3}_{4}" -f $Task.family, $Task.scale, $Task.condition_id, $Task.seed, $Operation
        $taskPath = Join-Path $taskDir ($tag + '.json')
        $taskPayload = [ordered]@{}
        foreach ($property in $Task.PSObject.Properties) { $taskPayload[$property.Name] = $property.Value }
        $taskPayload.operation = $Operation
        $taskJson = $taskPayload | ConvertTo-Json -Depth 30
        if (-not $DryRun) {
            Set-Content -LiteralPath $taskPath -Value $taskJson -Encoding UTF8
        }
        $taskLog = Join-Path $logDir ($tag + '.log')
        if ($DryRun) {
            Write-RunLog ("DRY RUN {0} {1}/{2}/{3}/seed{4}; config={5}" -f $Operation, $Task.family, $Task.scale, $Task.condition_id, $Task.seed, $taskPath)
            return
        }

        $arguments = @('run_all.py', '--study-task-config', $taskPath)
        Write-RunLog ("START {0} {1}/{2}/{3}/seed{4}; log={5}" -f $Operation, $Task.family, $Task.scale, $Task.condition_id, $Task.seed, $taskLog)
        & $Python @arguments 2>&1 | ForEach-Object {
            $line = $_.ToString()
            Add-Content -LiteralPath $taskLog -Value $line -Encoding UTF8
            Add-Content -LiteralPath $script:MasterLog -Value $line -Encoding UTF8
            Write-Host $line
        }
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            Write-RunLog "FAILED rc=$exitCode task=$tag"
            throw "Study task failed: $tag (exit $exitCode). See $taskLog"
        }
        if (-not (Test-TaskComplete $Task $Operation $taskPath)) {
            throw "Task process returned success but its checkpoint/evaluation failed validation: $tag. See $taskLog"
        }
        Write-RunLog "COMPLETED $tag"
        $script:Records.Add([pscustomobject]@{ operation = $Operation; family = $Task.family; condition = $Task.condition_id; seed = $Task.seed; status = 'complete'; log = $taskLog })
    }

    if ($DryRun) {
        foreach ($family in $families) {
            Write-Host "PLAN train all conditions, then evaluate all conditions: $family"
            foreach ($task in ($tasks | Where-Object family -EQ $family)) {
                Write-Host ("  {0} {1} {2} seed={3} steps={4}" -f $task.variant, $task.initialization, $family, $seed, $stepBudget)
            }
            foreach ($task in ($tasks | Where-Object family -EQ $family)) {
                Invoke-StudyTask $task 'train'
            }
            foreach ($task in ($tasks | Where-Object family -EQ $family)) {
                Invoke-StudyTask $task 'evaluate'
            }
        }
        return
    }

    New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs'), (Join-Path $root 'task_configs') | Out-Null
    New-Item -ItemType File -Force -Path $master | Out-Null
    if (-not $EvaluateOnly) {
        $cudaCheck = & $Python -c "import sys, torch, torch.nn; print('Python:', sys.executable); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 3)" 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Python/PyTorch CUDA preflight failed; no model run was started. $($cudaCheck -join ' ')"
        }
        Write-RunLog ($cudaCheck -join ' ')
    }

    foreach ($family in $families) {
        $familyTasks = @($tasks | Where-Object { $_.family -eq $family })
        if (-not $EvaluateOnly) {
            Write-RunLog "Training all $($familyTasks.Count) conditions for family=$family"
            foreach ($task in $familyTasks) {
                $taskPath = Join-Path (Join-Path $root 'task_configs') ("{0}_{1}_{2}_seed_{3}_train.json" -f $task.family, $task.scale, $task.condition_id, $task.seed)
                $taskJson = ($task | ConvertTo-Json -Depth 30)
                Set-Content -LiteralPath $taskPath -Value $taskJson -Encoding UTF8
                if (Test-TaskComplete $task 'train' $taskPath) {
                    Write-RunLog "SKIP validated trained run: $($task.output_dir)"
                    continue
                }
                if ((Test-Path -LiteralPath $task.output_dir) -and @(Get-ChildItem -LiteralPath $task.output_dir -Force -ErrorAction SilentlyContinue).Count -gt 0) {
                    throw "Refusing to overwrite or guess the state of existing artifact: $($task.output_dir). Use -EvaluateOnly for a valid trained checkpoint, or choose a fresh -OutputRoot."
                }
                Invoke-StudyTask $task 'train'
            }
        }

        Write-RunLog "Evaluating all conditions for family=$family"
        foreach ($task in $familyTasks) {
            $taskPath = Join-Path (Join-Path $root 'task_configs') ("{0}_{1}_{2}_seed_{3}_evaluate.json" -f $task.family, $task.scale, $task.condition_id, $task.seed)
            Set-Content -LiteralPath $taskPath -Value ($task | ConvertTo-Json -Depth 30) -Encoding UTF8
            if (Test-TaskComplete $task 'evaluate' $taskPath) {
                Write-RunLog "SKIP validated evaluation: $($task.output_dir)"
                continue
            }
            if (-not $EvaluateOnly -and -not (Test-TaskComplete $task 'train' $taskPath)) {
                throw "Evaluation requires a validated trained checkpoint: $($task.output_dir)"
            }
            Invoke-StudyTask $task 'evaluate'
        }
    }

    $summary = [ordered]@{
        experiment_id = 'granularity-init-48h-exploratory-lr3e-3-v1'
        created_at_utc = [DateTime]::UtcNow.ToString('o')
        family = $Family
        run_count = $tasks.Count
        scale = $scale
        conditions = @($conditions | ForEach-Object { $_.id })
        seed = $seed
        max_optimizer_steps = $stepBudget
        learning_rate = $learningRate
        status = 'complete'
        records = @($script:Records)
        learning_rate_selection = 'Fixed literature-informed rate; not tuned on these exploratory runs.'
        interpretation = 'One-family, one-seed, 1,000-step exploratory comparison only; not the full planned study.'
    }
    $summary | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $root 'run_summary.json') -Encoding UTF8
    Write-RunLog "All requested training and evaluation tasks completed. Summary: $(Join-Path $root 'run_summary.json')"
} finally {
    Pop-Location
}
