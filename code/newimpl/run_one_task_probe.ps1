param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("sst2", "mrpc", "rte", "cola")]
    [string]$TaskName,
    [string]$ModelName = "roberta-base",
    [string]$ResultsRoot = "newimpl\results\nlu_probe",
    [int]$TopK = 8,
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

if ($Quick) {
    $probeSamples = 128
    $sampleArgs = @("--max_train_samples", "512", "--max_eval_samples", "256")
    $evalArgs = @("--max_eval_samples", "256")
    $epochs = 1
} else {
    $probeSamples = 1024
    $sampleArgs = @()
    $evalArgs = @()
    $epochsByTask = @{
        "sst2" = 3
        "mrpc" = 5
        "rte"  = 5
        "cola" = 5
    }
    $epochs = $epochsByTask[$TaskName]
}

$taskRoot = Join-Path $ResultsRoot $TaskName
New-Item -ItemType Directory -Force $taskRoot | Out-Null

$strategies = @("full", "random", "gradient", "activation", "actgrad")

foreach ($strategy in $strategies) {
    $outputDir = Join-Path $taskRoot "$($TaskName)_$strategy`_top$TopK"
    Write-Host "Running Task-Probed LoRA: task=$TaskName strategy=$strategy output=$outputDir" -ForegroundColor Green
    $cmd = @(
        "newimpl\train_task_probed_lora_nlu.py",
        "--task_name", $TaskName,
        "--model_name", $ModelName,
        "--output_dir", $outputDir,
        "--probe_strategy", $strategy,
        "--top_k", "$TopK",
        "--probe_samples", "$probeSamples",
        "--candidate_scope", "attention",
        "--target_modules", "query,value",
        "--lora_r", "4",
        "--lora_alpha", "32",
        "--lora_dropout", "0.1",
        "--epochs", "$epochs",
        "--batch_size", "16",
        "--learning_rate", "0.0002"
    ) + $sampleArgs
    python @cmd

    Write-Host "Evaluating strategy=$strategy task=$TaskName" -ForegroundColor DarkGreen
    $evalCmd = @(
        "newimpl\evaluate_task_probed_lora_nlu.py",
        "--checkpoint_dir", "$outputDir\checkpoint",
        "--output_dir", $outputDir
    ) + $evalArgs
    python @evalCmd
}

python newimpl\plot_probe_results.py --results_root $taskRoot

Write-Host "Done. Results written to $taskRoot" -ForegroundColor Cyan
Write-Host "Summary: $taskRoot\probe_summary_table.csv" -ForegroundColor Cyan
