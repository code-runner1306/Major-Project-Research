[CmdletBinding()]
param(
    [string]$Python = 'python',
    [ValidateSet('all','train','interventions','analyze')][string]$Stage = 'all',
    [string[]]$Experiments = @('E1','E4','E5'),
    [switch]$IncludeNoConv,
    [switch]$PlanOnly
)
# Scalar vs channel-wise decay in the Anatomy of Associative Recall harness (third_party/dualgoose, upstream 8ca624a
# plus our patch). Trains ONLY the required runs: armed_gdn (published, channel-wise) and armed_gdn_scalar (new),
# 10 seeds each, in three published cells (E1 MQAR K=32, E4 haystack+curriculum, E5 collision+curriculum) = 60 runs.
# The three experiments run concurrently on GPU 0. Then E6 decay interventions and the analysis. Nothing is published.
#   powershell.exe -File .\run_A6000_dualgoose_granularity.ps1
#   powershell.exe -File .\run_A6000_dualgoose_granularity.ps1 -Stage analyze
$ErrorActionPreference = 'Stop'
$Experiments = @($Experiments | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$root = Join-Path $PSScriptRoot 'third_party/dualgoose'
$seeds = @('0','1','2','3','4','5','6','7','8','9')
$arms = @('armed_gdn','armed_gdn_scalar')
$e1Arms = if ($IncludeNoConv) { $arms + @('gated_deltanet','gated_deltanet_scalar') } else { $arms }
$long = @('scripts/long_retrieval_probe.py','--models') + $arms + @('--seq-lens','64','--pairs','4','--seeds') + $seeds +
        @('--steps','3000','--batch','256','--layers','2','--heads','2','--dense','4','--curriculum','--fp32','--rebind-eval')
$commands = [ordered]@{
    E1 = @('scripts/mqar_probe.py','--models') + $e1Arms + @('--d-model','32','--layers','1','--ks','32','--seeds') + $seeds +
         @('--fp32','--rebind-eval','--tag','_gran','--save-checkpoint','checkpoints/e1')
    E4 = $long + @('--tag','_granE4','--save-checkpoint','checkpoints/e4')
    E5 = $long + @('--collision','--tag','_granE5','--save-checkpoint','checkpoints/e5')
}
Push-Location $root
$previousPath = $env:PYTHONPATH
$env:PYTHONPATH = $root
try {
    if ($PlanOnly) {
        foreach ($name in $Experiments) { Write-Host "$name : $Python $($commands[$name] -join ' ')" }
        return
    }
    if ($Stage -in @('all','train')) {
        & $Python -m pytest -q -p no:cacheprovider tests/test_scalar_decay.py
        if ($LASTEXITCODE -ne 0) { throw 'Equivalence tests failed; not training' }
        New-Item -ItemType Directory -Force logs | Out-Null
        $processes = @()
        foreach ($name in $Experiments) {
            $log = "logs/$name.log"
            Write-Host "Starting $name (log: third_party/dualgoose/$log)"
            $processes += Start-Process -FilePath $Python -ArgumentList $commands[$name] -NoNewWindow -PassThru `
                -RedirectStandardOutput $log -RedirectStandardError "logs/$name.err.log"
        }
        $processes | Wait-Process
        $failed = @($processes | Where-Object { $_.ExitCode -ne 0 })
        if ($failed.Count) { throw "An experiment process failed; see third_party/dualgoose/logs" }
        Write-Host 'Training complete. Lines starting FAIL in the logs mark individual runs that failed.'
    }
    if ($Stage -in @('all','interventions')) {
        if ($Experiments -contains 'E1') { & $Python scripts/decay_interventions.py --kind mqar --checkpoints 'checkpoints/e1/armed_gdn_k*.pt' }
        if ($Experiments -contains 'E4') { & $Python scripts/decay_interventions.py --kind long --checkpoints 'checkpoints/e4/armed_gdn_L*.pt' }
        if ($Experiments -contains 'E5') { & $Python scripts/decay_interventions.py --kind long --checkpoints 'checkpoints/e5/armed_gdn_L*.pt' }
    }
    if ($Stage -in @('all','analyze')) {
        & $Python scripts/analyze_granularity.py
        Write-Host 'Report: third_party/dualgoose/results/granularity_report.md'
    }
} finally {
    $env:PYTHONPATH = $previousPath
    Pop-Location
}
