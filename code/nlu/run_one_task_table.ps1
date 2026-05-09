param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("sst2", "mrpc", "rte", "cola")]
    [string]$TaskName,
    [string]$ModelName = "roberta-base",
    [string]$ResultsRoot = "my_NLU\results\by_task",
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

if ($Quick) {
    $sampleArgs = @("--max_train_samples", "1024", "--max_eval_samples", "512")
    $evalArgs = @("--max_eval_samples", "512")
    $epochs = 1
} else {
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

$methods = @(
    @{ Name = "ft"; Label = "FT"; Extra = @("--method", "ft", "--learning_rate", "0.00002") },
    @{ Name = "bitfit"; Label = "BitFit"; Extra = @("--method", "bitfit", "--learning_rate", "0.0001") },
    @{ Name = "adapter_0p3m"; Label = "Adapter0.3M"; Extra = @("--method", "adapter", "--adapter_size", "16", "--adapter_location", "output", "--adapter_dropout", "0.0", "--learning_rate", "0.0002") },
    @{ Name = "adapter_0p9m"; Label = "Adapter0.9M"; Extra = @("--method", "adapter", "--adapter_size", "48", "--adapter_location", "output", "--adapter_dropout", "0.0", "--learning_rate", "0.0002") },
    @{ Name = "lora"; Label = "LoRA"; Extra = @("--method", "lora", "--lora_r", "4", "--lora_alpha", "32", "--lora_dropout", "0.1", "--target_modules", "query,value", "--learning_rate", "0.0002") }
)

Write-Host "Preparing GLUE task $TaskName..." -ForegroundColor Cyan
python reimpl\prepare_glue.py --task_name $TaskName --output_dir "data\glue_$TaskName" --preview_samples 20

foreach ($method in $methods) {
    $outputDir = Join-Path $taskRoot "$($TaskName)_$($method.Name)"
    Write-Host "Running $($method.Label) on $TaskName, output=$outputDir" -ForegroundColor Green
    $cmd = @(
        "reimpl\train_my_lora_nlu.py",
        "--task_name", $TaskName,
        "--model_name", $ModelName,
        "--output_dir", $outputDir,
        "--epochs", "$epochs",
        "--batch_size", "16"
    ) + $method.Extra + $sampleArgs
    python @cmd

    Write-Host "Evaluating $($method.Label) on $TaskName" -ForegroundColor DarkGreen
    $evalCmd = @(
        "reimpl\evaluate_my_lora_nlu.py",
        "--checkpoint_dir", "$outputDir\checkpoint",
        "--output_dir", $outputDir
    ) + $evalArgs
    python @evalCmd
}

python reimpl\plot_results.py `
    --results_root $taskRoot `
    --paper_tasks $TaskName `
    --table_name "paper_style_$($TaskName)_table"

Write-Host "Done. Results written to $taskRoot" -ForegroundColor Cyan
Write-Host "Task table: $taskRoot\paper_style_$($TaskName)_table.md" -ForegroundColor Cyan
