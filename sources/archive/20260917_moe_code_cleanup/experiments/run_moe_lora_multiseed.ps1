param(
    [Parameter(Mandatory = $true)]
    [string]$Python,

    [int[]]$Seeds = @(20260923, 20260933),

    [string]$RunDate = "20260917",

    [string]$StatusPath = "output/moe_lora_multiseed_20260917.status.jsonl"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$statusFile = Join-Path $projectRoot $StatusPath

function Write-QueueStatus {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Event,

        [hashtable]$Details = @{}
    )

    $record = @{
        event = $Event
        timestamp = (Get-Date).ToString("o")
    }
    foreach ($entry in $Details.GetEnumerator()) {
        $record[$entry.Key] = $entry.Value
    }
    Add-Content -LiteralPath $statusFile -Value ($record | ConvertTo-Json -Compress) -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$requiredInputs = @(
    "checkpoints/h20.best.pt",
    "data/online_crp/lora_domains/large_scale_n100_150/adapt_train.json",
    "data/online_crp/lora_domains/large_scale_n100_150/validation.json"
)
foreach ($relativePath in $requiredInputs) {
    $inputPath = Join-Path $projectRoot $relativePath
    if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
        throw "Required input not found: $inputPath"
    }
}

$runs = foreach ($seed in $Seeds) {
    $relativeRoot = "output/moe_lora_long_${RunDate}_seed${seed}"
    $absoluteRoot = Join-Path $projectRoot $relativeRoot
    if (Test-Path -LiteralPath $absoluteRoot) {
        throw "Refusing to overwrite existing output: $absoluteRoot"
    }
    [pscustomobject]@{
        Seed = $seed
        RelativeRoot = $relativeRoot
        AbsoluteRoot = $absoluteRoot
    }
}

Write-QueueStatus -Event "queue_start" -Details @{
    seeds = @($Seeds)
    run_date = $RunDate
}

try {
    foreach ($run in $runs) {
        New-Item -ItemType Directory -Path $run.AbsoluteRoot | Out-Null

        $latestAdapter = "$($run.RelativeRoot)/latest.adapter.pt"
        $bestAdapter = "$($run.RelativeRoot)/best.adapter.pt"
        $metrics = "$($run.RelativeRoot)/metrics.jsonl"
        $curve = "$($run.RelativeRoot)/live_curve.png"
        $trainArguments = @(
            "-u", "-B", "-m", "experiments.adapt_lora",
            "--base-checkpoint", "checkpoints/h20.best.pt",
            "--train-manifest", "data/online_crp/lora_domains/large_scale_n100_150/adapt_train.json",
            "--validation-manifest", "data/online_crp/lora_domains/large_scale_n100_150/validation.json",
            "--domain", "large_scale_n100_150",
            "--adapter-kind", "moe_lora",
            "--target-modules", "stack_projection.2",
            "--rank", "8",
            "--alpha", "16",
            "--dropout", "0",
            "--num-experts", "2",
            "--router-balance-coefficient", "0.01",
            "--learning-rate", "3e-4",
            "--epochs", "5",
            "--batch-size", "16",
            "--rollouts-per-instance", "8",
            "--replay-decision-batch-size", "2048",
            "--entropy-coefficient", "0.002",
            "--gradient-clip-norm", "1",
            "--precision-mode", "fp16",
            "--environment-engine", "vectorized",
            "--seed", "$($run.Seed)",
            "--output", $latestAdapter,
            "--best-output", $bestAdapter,
            "--metrics-output", $metrics
        )

        $train = Start-Process -FilePath $Python `
            -ArgumentList $trainArguments `
            -WorkingDirectory $projectRoot `
            -RedirectStandardOutput (Join-Path $run.AbsoluteRoot "train.stdout.log") `
            -RedirectStandardError (Join-Path $run.AbsoluteRoot "train.stderr.log") `
            -WindowStyle Hidden `
            -PassThru

        $plotArguments = @(
            "-u", "-B", "-m", "experiments.plot_adaptation_live",
            $metrics,
            $curve,
            "--watch-pid", "$($train.Id)",
            "--poll-seconds", "10"
        )
        $plot = Start-Process -FilePath $Python `
            -ArgumentList $plotArguments `
            -WorkingDirectory $projectRoot `
            -RedirectStandardOutput (Join-Path $run.AbsoluteRoot "plot.stdout.log") `
            -RedirectStandardError (Join-Path $run.AbsoluteRoot "plot.stderr.log") `
            -WindowStyle Hidden `
            -PassThru

        Write-QueueStatus -Event "run_start" -Details @{
            seed = $run.Seed
            output = $run.RelativeRoot
            train_pid = $train.Id
            plot_pid = $plot.Id
        }

        $train.WaitForExit()
        $train.Refresh()
        $trainExitCode = $train.ExitCode
        $plot.WaitForExit()
        $plot.Refresh()
        $plotExitCode = $plot.ExitCode

        Write-QueueStatus -Event "run_exit" -Details @{
            seed = $run.Seed
            output = $run.RelativeRoot
            train_exit_code = $trainExitCode
            plot_exit_code = $plotExitCode
        }

        if ($null -ne $trainExitCode -and $trainExitCode -ne 0) {
            throw "Training failed for seed $($run.Seed) with exit code $trainExitCode"
        }
        if ($null -ne $plotExitCode -and $plotExitCode -ne 0) {
            throw "Live plotting failed for seed $($run.Seed) with exit code $plotExitCode"
        }

        $metricsPath = Join-Path $projectRoot $metrics
        $lastRecord = Get-Content -LiteralPath $metricsPath -Tail 1 | ConvertFrom-Json
        if ($lastRecord.record_type -ne "run_complete") {
            throw "Seed $($run.Seed) exited without a run_complete metrics record"
        }
        $curvePath = Join-Path $projectRoot $curve
        if (-not (Test-Path -LiteralPath $curvePath -PathType Leaf)) {
            throw "Seed $($run.Seed) exited without a live curve"
        }
    }

    Write-QueueStatus -Event "queue_complete" -Details @{
        seeds = @($Seeds)
    }
}
catch {
    Write-QueueStatus -Event "queue_failed" -Details @{
        error = $_.Exception.Message
    }
    throw
}
