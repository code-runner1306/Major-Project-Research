[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('mqar', 'overwrite', 'both')]
    [string]$Family,
    [string]$Python = "python",
    [string]$OutputRoot,
    [ValidateRange(1, 2147483647)]
    [int]$OptimizerSteps = 1000,
    [int[]]$Seeds = @(42, 43, 44, 45, 46),
    [int]$Workers = 1,
    [int]$QualifyWorkers = 0,
    [int]$QualificationSteps = 20,
    [string]$QualificationRoot,
    [string]$QualificationReport,
    [switch]$DryRun,
    [switch]$EvaluateOnly,
    [switch]$SmokeTest,
    [string]$StudyConfig = "study_configs/granularity_init_study.json"
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repo
try {
    if ($Workers -lt 1) { throw "Workers must be at least 1" }
    if ($Seeds.Count -lt 1 -or @($Seeds | Select-Object -Unique).Count -ne $Seeds.Count -or @($Seeds | Where-Object { $_ -lt 0 }).Count -gt 0) { throw 'Seeds must be a non-empty list of unique non-negative integers' }
    if ($SmokeTest -and ($QualifyWorkers -gt 0 -or $EvaluateOnly -or $Workers -ne 1)) { throw 'SmokeTest requires one worker and cannot be combined with qualification or EvaluateOnly' }
    if ($SmokeTest) { $StudyConfig = 'study_configs/granularity_init_smoke.json' }
    if ($QualifyWorkers -lt 0) { throw "QualifyWorkers cannot be negative" }
    if ($QualifyWorkers -gt 0 -and $EvaluateOnly) { throw "QualifyWorkers cannot be combined with EvaluateOnly" }
    if ($QualificationSteps -lt 10) { throw "QualificationSteps must be at least 10" }
    $families = if ($Family -eq 'both') { @('mqar', 'overwrite') } else { @($Family) }
    $maxWorkers = 4 * $families.Count
    if ($Workers -gt $maxWorkers) { throw "Workers cannot exceed $maxWorkers for this family selection" }
    if ($QualifyWorkers -gt $maxWorkers) { throw "QualifyWorkers cannot exceed $maxWorkers for this family selection" }
    if ($QualifyWorkers -gt 0 -and $QualificationReport) { throw "QualificationReport is for production, not qualification" }
    if ($Workers -gt 1 -and $QualifyWorkers -eq 0 -and -not $QualificationReport -and -not $DryRun) { throw "Parallel production requires QualificationReport from a successful QualifyWorkers run" }

    $studyConfigPath = [System.IO.Path]::GetFullPath((Join-Path $repo $StudyConfig))
    if (-not (Test-Path -LiteralPath $studyConfigPath -PathType Leaf)) { throw "Study config not found: $studyConfigPath" }
    $study = Get-Content -LiteralPath $studyConfigPath -Raw | ConvertFrom-Json
    if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
        $stepSuffix = if ($OptimizerSteps -eq 1000) { '' } else { "_steps$OptimizerSteps" }
        $OutputRoot = "output/granularity_48h_exploratory_lr3e-3${stepSuffix}_$Family"
        if ($SmokeTest) { $OutputRoot = "output/granularity_48h_launcher_smoke_$Family" }
    }
    $outputBase = [System.IO.Path]::GetFullPath((Join-Path $repo 'output'))
    if ($QualifyWorkers -gt 0) {
        if ([string]::IsNullOrWhiteSpace($QualificationRoot)) { $QualificationRoot = "output/granularity_48h_worker_qualification_{0}_{1}" -f $Family, (Get-Date -Format 'yyyyMMdd_HHmmss') }
        $root = [System.IO.Path]::GetFullPath((Join-Path $repo $QualificationRoot))
    } else { $root = [System.IO.Path]::GetFullPath((Join-Path $repo $OutputRoot)) }
    if (-not $root.StartsWith($outputBase + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) { throw "OutputRoot and QualificationRoot must be subfolders of $outputBase" }
    if ($QualifyWorkers -gt 0 -and (Test-Path -LiteralPath $root) -and @(Get-ChildItem -LiteralPath $root -Force).Count -gt 0) { throw "Qualification root must be new or empty: $root" }

    $conditions = @(
        [ordered]@{ id = 'C1-scalar-default'; variant = 'scalar'; initialization = 'default' },
        [ordered]@{ id = 'C3-channel-default'; variant = 'channel-wise'; initialization = 'default' },
        [ordered]@{ id = 'C4-scalar-calibrated'; variant = 'scalar'; initialization = 'calibrated-geometric' },
        [ordered]@{ id = 'C6-channel-calibrated'; variant = 'channel-wise'; initialization = 'calibrated-geometric' }
    )
    $scale = '10M'; $stepBudget = $OptimizerSteps; $learningRate = 0.003
    $sourceBudget = $study.budgets.'10M'; $calibration = $study.calibration
    $experimentId = 'granularity-init-48h-exploratory-lr3e-3-v1'; $qualificationId = "granularity-init-worker-pilot-$Family-$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    if ($SmokeTest) { $scale = 'tiny'; $stepBudget = 2; $sourceBudget = $study.budgets.tiny; $learningRate = [double]$sourceBudget.learning_rate; $experimentId = 'granularity-init-48h-launcher-smoke-v1'; $Seeds = @(42) }
    $script:MasterLog = $null

    function Write-RunLog([string]$Message) {
        $line = "[{0}] {1}" -f ([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')), $Message
        Write-Host $line
        if ($script:MasterLog -and -not $DryRun) { Add-Content -LiteralPath $script:MasterLog -Value $line -Encoding UTF8 }
    }

    function New-TaskSpec([string]$TaskFamily, $Condition, [int]$Seed, [int]$Steps, [int]$Examples, [string]$Operation, [string]$OutputBase, [string]$TaskExperimentId, [int]$Concurrency) {
        $retrieval = $study.retrieval | ConvertTo-Json -Depth 20 | ConvertFrom-Json
        if ($SmokeTest) { $retrieval | Add-Member -NotePropertyName vocabulary_size -NotePropertyValue 128 -Force }
        $retrieval.validation_examples = $Examples; $retrieval.held_out_examples = $Examples
        $runDir = Join-Path $OutputBase ("runs\{0}\{1}\{2}\seed_{3}" -f $TaskFamily, $scale, $Condition.id, $Seed)
        $task = [ordered]@{
            operation = $Operation; experiment_id = $TaskExperimentId; condition_id = $Condition.id; family = $TaskFamily; scale = $scale
            variant = $Condition.variant; initialization = $Condition.initialization; seed = $Seed; output_dir = [System.IO.Path]::GetFullPath($runDir)
            max_optimizer_steps = $Steps; batch_size = [int]$sourceBudget.batch_size; grad_accum_steps = [int]$sourceBudget.grad_accum_steps
            context_length = [int]$sourceBudget.context_length; validation_steps = if ($Steps -ge 100) { 100 } else { 4 }
            evaluation_interval = if ($Steps -ge 100) { 100 } else { [Math]::Max(5, [int]($Steps / 2)) }; learning_rate = $learningRate
            calibration_min_half_life = [int]$calibration.min_half_life; calibration_max_half_life = [int]$calibration.max_half_life
            calibration_examples = [int]$calibration.examples; retrieval_config = $retrieval; run_mode = if ($SmokeTest) { 'smoke' } else { 'production' }; recurrence_backend = 'reference'
            state_accumulator_dtype = 'input'; recurrence_chunk_size = 64; validation_answer_accuracy = $true; concurrent_training_runs = $Concurrency
        }
        if ($Steps -eq 1000 -and $TaskExperimentId -eq $experimentId) { $task.validation_at_steps = @(300, 650) }
        return [pscustomobject]$task
    }

    function Write-TaskConfig($Task, [string]$ConfigRoot, [string]$Prefix = '') {
        $configDir = Join-Path $ConfigRoot 'task_configs'; New-Item -ItemType Directory -Force -Path $configDir | Out-Null
        $tag = "{0}{1}_{2}_{3}_seed_{4}_{5}" -f $Prefix, $Task.family, $Task.scale, $Task.condition_id, $Task.seed, $Task.operation
        $path = Join-Path $configDir ($tag + '.json'); $payload = [ordered]@{}
        foreach ($property in $Task.PSObject.Properties) { $payload[$property.Name] = $property.Value }
        if ($Prefix) { $payload.qualification = $true; $payload.qualification_stage = 'worker-capacity-pilot'; $payload.qualification_scope = 'runtime-capacity-only' }
        $payload | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $path -Encoding UTF8
        return $path
    }

    function Test-TaskComplete($Task, [string]$Operation, [string]$TaskPath) {
        $probe = "import json,sys; from study_runner import task_complete; print('COMPLETE' if task_complete(json.load(open(sys.argv[1], encoding='utf-8-sig')), sys.argv[2]) else 'INCOMPLETE')"
        $result = & $Python -c $probe $TaskPath $Operation 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Could not validate artifact: $($result -join ' ')" }
        $last = @($result | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ } | Select-Object -Last 1)
        return ($last.Count -eq 1 -and $last[0] -eq 'COMPLETE')
    }

    function Write-NewTaskOutput($Item) {
        foreach ($stream in @(
            [pscustomobject]@{ Path = $Item.Stdout; Label = 'stdout'; Property = 'StdoutLines' },
            [pscustomobject]@{ Path = $Item.Stderr; Label = 'stderr'; Property = 'StderrLines' }
        )) {
            if (-not (Test-Path -LiteralPath $stream.Path -PathType Leaf)) { continue }
            try { $lines = @(Get-Content -LiteralPath $stream.Path -ErrorAction SilentlyContinue) } catch { continue }
            $seen = [int]$Item.($stream.Property)
            if ($lines.Count -le $seen) { continue }
            for ($lineIndex = $seen; $lineIndex -lt $lines.Count; $lineIndex++) {
                Write-RunLog "$($Item.Tag) [$($stream.Label)] $($lines[$lineIndex])"
            }
            $Item.($stream.Property) = $lines.Count
        }
    }

    function Invoke-Phase([string]$Phase, [array]$Configs, [int]$Limit, [string]$PhaseRoot) {
        if ($Configs.Count -eq 0) { return [pscustomobject]@{ Success = $true; Tasks = @(); Duration = 0 } }
        $phaseStart = Get-Date; $records = @(); $hadFailure = $false
        for ($offset = 0; $offset -lt $Configs.Count; $offset += $Limit) {
            $batch = @($Configs | Select-Object -Skip $offset -First $Limit); $active = @()
            foreach ($configPath in $batch) {
                $task = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json; $tag = [System.IO.Path]::GetFileNameWithoutExtension($configPath)
                $logDir = Join-Path $PhaseRoot 'logs'; New-Item -ItemType Directory -Force -Path $logDir | Out-Null
                $stdout = Join-Path $logDir ($tag + '.out.log'); $stderr = Join-Path $logDir ($tag + '.err.log'); Write-RunLog "QUEUED $Phase $tag"
                $started = Get-Date; $quotedConfigPath = '"{0}"' -f $configPath; $proc = Start-Process -FilePath $Python -ArgumentList @('run_all.py', '--study-task-config', $quotedConfigPath) -WorkingDirectory $repo -PassThru -NoNewWindow -RedirectStandardOutput $stdout -RedirectStandardError $stderr
                $active += [pscustomobject]@{ Process = $proc; Config = $configPath; Task = $task; Tag = $tag; Started = $started; Stdout = $stdout; Stderr = $stderr; StdoutLines = 0; StderrLines = 0 }
            }
            $running = $true
            while ($running) {
                $running = $false
                foreach ($item in $active) {
                    Write-NewTaskOutput $item
                    if (-not $item.Process.HasExited) { $running = $true }
                }
                if ($running) { Start-Sleep -Seconds $(if ($SmokeTest) { 1 } else { 10 }) }
            }
            foreach ($item in $active) {
                $item.Process.WaitForExit()
                $item.Process.Refresh()
                Write-NewTaskOutput $item
                $elapsed = ((Get-Date) - $item.Started).TotalSeconds
                $exitCode = $item.Process.ExitCode
                Write-RunLog "PROCESS EXIT $Phase $($item.Tag) pid=$($item.Process.Id) code=$exitCode elapsed=$([math]::Round($elapsed, 1))s"
                $artifactComplete = $false
                if ($null -eq $exitCode -or $exitCode -eq 0) {
                    $artifactComplete = Test-TaskComplete $item.Task $Phase $item.Config
                    Write-RunLog "ARTIFACT CHECK $Phase $($item.Tag): $(if ($artifactComplete) { 'COMPLETE' } else { 'INCOMPLETE' })"
                }
                # Some Windows PowerShell/.NET combinations return a Process object
                # without exposing ExitCode after redirected child output. A complete
                # artifact is independently verified for task identity, final step,
                # lifecycle, and checkpoint hash, so use that as the fallback only
                # when ExitCode is unavailable. A real nonzero exit always fails.
                $success = ($artifactComplete -and ($null -eq $exitCode -or $exitCode -eq 0))
                if ($null -eq $exitCode -and $artifactComplete) { Write-RunLog "PROCESS EXIT CODE UNAVAILABLE; accepting verified artifact for $($item.Tag)" }
                if (-not $success) { $hadFailure = $true; Write-RunLog "FAILED $Phase $($item.Tag) exit_code=$exitCode" } else { Write-RunLog "COMPLETED $Phase $($item.Tag)" }
                $records += [pscustomobject]@{ task = $item.Task; config = $item.Config; success = $success; elapsed_seconds = $elapsed; stdout = $item.Stdout; stderr = $item.Stderr }
            }
            if ($hadFailure) { break }
        }
        return [pscustomobject]@{ Success = (-not $hadFailure); Tasks = @($records); Duration = ((Get-Date) - $phaseStart).TotalSeconds }
    }

    function Get-PhaseSummary($PhaseResult, [string]$Phase, [int]$WorkerCount, [string]$SummaryPath) {
        $resourceKey = if ($Phase -eq 'train') { 'training_resources' } else { 'evaluation_resources' }; $tasks = @(); $hardware = $null; $safe = [bool]$PhaseResult.Success
        foreach ($record in $PhaseResult.Tasks) {
            $metricsPath = Join-Path $record.task.output_dir 'metrics.json'; $metrics = if (Test-Path -LiteralPath $metricsPath) { Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json } else { $null }
            $capture = if ($metrics) { $metrics.PSObject.Properties[$resourceKey].Value } else { $null }; $runtime = if ($capture) { $capture.runtime } else { $null }
            if (-not $hardware -and $runtime) { $hardware = [ordered]@{ gpu_uuid = $runtime.gpu_uuid; nvml_gpu_name = $runtime.nvml_gpu_name; torch_gpu_name = $runtime.torch_gpu_name; driver_version = $runtime.driver_version; torch_version = $runtime.torch_version; torch_cuda_runtime = $runtime.torch_cuda_runtime } }
            $samples = if ($capture) { @($capture.activity_samples | Where-Object { $_.available -eq $true }) } else { @() }
            $vram = @($samples | Where-Object { $_.memory_total_bytes -gt 0 } | ForEach-Object { 100.0 * $_.memory_used_bytes / $_.memory_total_bytes }); $other = @($samples | Where-Object { $_.other_compute_process_count -is [int] } | ForEach-Object { [int]$_.other_compute_process_count })
            $maxVram = if ($vram.Count) { ($vram | Measure-Object -Maximum).Maximum } else { $null }; $otherMax = if ($other.Count) { ($other | Measure-Object -Maximum).Maximum } else { 0 }
            $activity = if ($capture) { $capture.activity_summary } else { $null }
            $gpuUtilization = if ($activity) { $activity.gpu_utilization_percent } else { $null }
            $processCpu = if ($activity) { $activity.process_cpu_percent } else { $null }
            $systemCpu = if ($activity) { $activity.system_cpu_percent } else { $null }
            $processRss = if ($activity) { $activity.process_rss_bytes } else { $null }
            $systemMemory = if ($activity) { $activity.system_memory_used_bytes } else { $null }
            $cpuFrequency = if ($activity) { $activity.cpu_frequency_current_mhz } else { $null }
            $telemetrySafe = ($null -ne $runtime -and $runtime.cuda_available -and $runtime.nvml_available -and $vram.Count -gt 0 -and $maxVram -lt 95 -and $otherMax -le [Math]::Max(0, $WorkerCount - 1)); $safe = $safe -and $record.success -and $telemetrySafe
            $tasks += [ordered]@{ family = $record.task.family; condition_id = $record.task.condition_id; seed = $record.task.seed; task_complete = $record.success; max_whole_gpu_vram_percent = $maxVram; max_gpu_utilization_percent = if ($gpuUtilization) { $gpuUtilization.max } else { $null }; process_cpu_percent = $processCpu; system_cpu_percent = $systemCpu; process_rss_bytes = $processRss; system_memory_used_bytes = $systemMemory; cpu_frequency_current_mhz = $cpuFrequency; torch_memory = if ($capture) { $capture.torch_memory } else { $null }; nvml_samples = $samples.Count; other_compute_process_count_max = $otherMax; telemetry_safe = $telemetrySafe; task_wall_seconds = $record.elapsed_seconds }
        }
        $result = [ordered]@{ phase = $Phase; requested_workers = $WorkerCount; task_count = $tasks.Count; command_success = [bool]$PhaseResult.Success; qualified = $safe; telemetry_safe = $safe; phase_makespan_seconds = $PhaseResult.Duration; hardware = $hardware; tasks = $tasks; gpu_telemetry_limitation = 'GPU utilization and VRAM are whole-device readings; do not sum them.'; cpu_telemetry_limitation = 'CPU observations do not by themselves prove throttling.' }
        $result | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8; return [pscustomobject]$result
    }

    function Get-CurrentGpuUuid {
        $probe = & $Python -c "import json,torch; from resource_telemetry import DeviceResourceCapture; c=DeviceResourceCapture(torch.device('cuda')); print(json.dumps(c.runtime_metadata()))" 2>$null
        if ($LASTEXITCODE -ne 0) { return '' }; try { return (($probe | Select-Object -Last 1 | ConvertFrom-Json).gpu_uuid) } catch { return '' }
    }

    if ($SmokeTest) { Write-Host "Launcher CPU smoke: $(4 * $families.Count) tiny runs; family=$Family; steps=$stepBudget; evaluation examples=4"; Write-Host "Output root: $root" }
    elseif ($QualifyWorkers -gt 0) { Write-Host "Worker qualification: family=$Family; target=$QualifyWorkers; fallback=$($QualifyWorkers)..1"; Write-Host "Pilot root: $root" }
    else { Write-Host "Exploratory production: $((4 * $families.Count) * $Seeds.Count) 10M runs; family=$Family; workers=$Workers; seeds=$($Seeds -join ','); steps=$stepBudget; LR=$learningRate"; Write-Host "Output root: $root" }
    if (-not $SmokeTest) {
        Write-Host "Production optimizer-step budget: $stepBudget (exploratory setting, not a literature-derived convergence budget). Worker qualification uses $QualificationSteps steps per task."
        Write-Host 'Production evaluation uses 512 examples; the worker pilot uses 64 examples and short isolated tasks.'
    }
    if ($DryRun) { if ($QualifyWorkers -gt 0) { for ($n = $QualifyWorkers; $n -ge 1; $n--) { Write-Host "  QUALIFY $n total workers; then evaluate the same tasks" } } else { Write-Host "PLAN: train all selected families and seeds before evaluating any family, bounded to $Workers total workers" }; return }

    New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs'), (Join-Path $root 'task_configs') | Out-Null; $script:MasterLog = Join-Path $root ("logs\run_48h_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss')); New-Item -ItemType File -Force -Path $script:MasterLog | Out-Null
    $cudaCheck = & $Python -c "import sys,torch,torch.nn; print('Python:',sys.executable); print('PyTorch:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() or $([int][bool]$SmokeTest) else 3)" 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Python/PyTorch CUDA preflight failed: $($cudaCheck -join ' ')" }; Write-RunLog ($cudaCheck -join ' ')
    $configSha = (Get-FileHash -LiteralPath $studyConfigPath -Algorithm SHA256).Hash.ToLowerInvariant(); try { $gitRevision = (& git -C $repo rev-parse HEAD 2>$null | Select-Object -Last 1).Trim() } catch { $gitRevision = 'unknown' }

    if ($QualifyWorkers -gt 0) {
        $attempted = @(); $trainLimit = 0; $evalLimit = 0; $jointLimit = 0; $trialRows = @()
        for ($count = $QualifyWorkers; $count -ge 1; $count--) {
            $attempted += $count; $trialRoot = Join-Path $root "workers_$count"; New-Item -ItemType Directory -Force -Path $trialRoot | Out-Null; $trainConfigs = @(); $evalConfigs = @(); $slot = 0
            foreach ($condition in $conditions) { foreach ($trialFamily in $families) {
                if ($slot -ge $count) { continue }; $pilotSeed = 81000 + ($count * 10) + $slot
                $train = New-TaskSpec $trialFamily $condition $pilotSeed $QualificationSteps 64 'train' $trialRoot $qualificationId $count; $evaluate = New-TaskSpec $trialFamily $condition $pilotSeed $QualificationSteps 64 'evaluate' $trialRoot $qualificationId $count
                $trainConfigs += Write-TaskConfig $train $trialRoot 'pilot_'; $evalConfigs += Write-TaskConfig $evaluate $trialRoot 'pilot_'; $slot++
            } }
            Write-RunLog "Qualification trial workers=${count}: training $($trainConfigs.Count) independent tasks"; $trainRun = Invoke-Phase 'train' $trainConfigs $count $trialRoot; $trainSummary = Get-PhaseSummary $trainRun 'train' $count (Join-Path $trialRoot 'training.json'); $evalSummary = $null
            if ($trainSummary.qualified) { $trainLimit = $count; Write-RunLog "Qualification trial workers=${count}: evaluation begins"; $evalRun = Invoke-Phase 'evaluate' $evalConfigs $count $trialRoot; $evalSummary = Get-PhaseSummary $evalRun 'evaluate' $count (Join-Path $trialRoot 'evaluation.json'); if ($evalSummary.qualified) { $evalLimit = $count; $jointLimit = $count } }
            $trialRows += [ordered]@{ workers = $count; training = $trainSummary; evaluation = $evalSummary; jointly_qualified = ($null -ne $evalSummary -and $evalSummary.qualified) }; if ($jointLimit -gt 0) { break }
        }
        $report = [ordered]@{ schema_version = 1; experiment_id = 'granularity-init-worker-qualification-v1'; created_at_utc = [DateTime]::UtcNow.ToString('o'); family_selection = $Family; families = $families; requested_target_workers = $QualifyWorkers; attempted_worker_counts_in_order = $attempted; qualified_training_workers = $trainLimit; qualified_evaluation_workers = $evalLimit; qualified_workers = $jointLimit; status = if ($jointLimit -gt 0) { 'qualified' } else { 'not_qualified' }; pilot_protocol = [ordered]@{ scale = '10M'; optimizer_steps = $QualificationSteps; validation_examples = 64; held_out_examples = 64; operation = 'separate-process training and evaluation' }; study_config_sha256 = $configSha; git_revision = $gitRevision; trials = $trialRows }
        $reportPath = Join-Path $root 'qualification_report.json'; $report | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath $reportPath -Encoding UTF8; Write-RunLog "Worker qualification report: $reportPath"; Write-Host "Qualified training workers=$trainLimit; evaluation workers=$evalLimit; joint maximum=$jointLimit"
        if ($jointLimit -le 0) { throw "No worker count passed both qualification phases. See $reportPath" }; Write-Host "Next production command: .\run_48h_exploratory_study.ps1 -Family $Family -Workers $jointLimit -QualificationReport '$reportPath'"; return
    }

    if ($Workers -gt 1) {
        if (-not (Test-Path -LiteralPath $QualificationReport -PathType Leaf)) { throw "Qualification report not found: $QualificationReport" }; $report = Get-Content -LiteralPath $QualificationReport -Raw | ConvertFrom-Json; $currentGpuUuid = Get-CurrentGpuUuid
        if ($report.status -ne 'qualified' -or ($report.families -join ',') -ne ($families -join ',') -or [int]$report.qualified_workers -lt $Workers -or $report.study_config_sha256 -ne $configSha -or $report.git_revision -ne $gitRevision -or [string]::IsNullOrWhiteSpace($currentGpuUuid) -or $report.hardware.gpu_uuid -ne $currentGpuUuid) { throw 'Qualification report does not authorize this family, worker count, code revision, config, or GPU' }
    }

    $trainConfigs = @(); $evalConfigs = @()
    $examples = if ($SmokeTest) { 4 } else { 512 }
    foreach ($seed in $Seeds) { foreach ($condition in $conditions) { foreach ($taskFamily in $families) { $train = New-TaskSpec $taskFamily $condition $seed $stepBudget $examples 'train' $root $experimentId $Workers; $evaluate = New-TaskSpec $taskFamily $condition $seed $stepBudget $examples 'evaluate' $root $experimentId $Workers; $trainConfigs += Write-TaskConfig $train $root; $evalConfigs += Write-TaskConfig $evaluate $root } } }
    if (-not $EvaluateOnly) {
        $pending = @(); foreach ($config in $trainConfigs) { $task = Get-Content -LiteralPath $config -Raw | ConvertFrom-Json; if (Test-TaskComplete $task 'train' $config) { Write-RunLog "SKIP validated trained run: $($task.output_dir)"; continue }; if ((Test-Path -LiteralPath $task.output_dir -PathType Container) -and @(Get-ChildItem -LiteralPath $task.output_dir -Force).Count -gt 0) { throw "Refusing to overwrite partial/incompatible artifact: $($task.output_dir)" }; $pending += $config }
        $trainResult = Invoke-Phase 'train' $pending $Workers $root; if (-not $trainResult.Success) { throw 'Training failed; evaluation has not started. See per-task logs.' }
    }
    $pendingEval = @(); foreach ($config in $evalConfigs) { $task = Get-Content -LiteralPath $config -Raw | ConvertFrom-Json; if (Test-TaskComplete $task 'evaluate' $config) { Write-RunLog "SKIP validated evaluation: $($task.output_dir)"; continue }; if (-not (Test-TaskComplete $task 'train' $config)) { throw "Evaluation requires a validated trained checkpoint: $($task.output_dir)" }; $pendingEval += $config }
    $evalResult = Invoke-Phase 'evaluate' $pendingEval $Workers $root; if (-not $evalResult.Success) { throw 'Evaluation failed. See per-task logs.' }
    $summary = [ordered]@{ experiment_id = $experimentId; created_at_utc = [DateTime]::UtcNow.ToString('o'); family_selection = $Family; families = $families; run_count = (4 * $families.Count) * $Seeds.Count; scale = $scale; conditions = @($conditions | ForEach-Object { $_.id }); seeds = @($Seeds); max_optimizer_steps = $stepBudget; learning_rate = $learningRate; status = 'complete'; training_barrier_before_evaluation = $true; interpretation = if ($SmokeTest) { 'CPU launcher smoke only; not research results or CUDA qualification.' } else { "$($Seeds.Count) paired seeds and $stepBudget steps per condition; exploratory comparison only." } }
    $summary | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $root 'run_summary.json') -Encoding UTF8; Write-RunLog "All requested training and evaluation tasks completed. Summary: $(Join-Path $root 'run_summary.json')"
} finally { Pop-Location }
