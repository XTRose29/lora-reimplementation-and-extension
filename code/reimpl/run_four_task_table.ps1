param(
    [string]$ModelName = "roberta-base",
    [string]$ResultsRoot = "my_NLU\results\four_task_table",
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

if ($Quick) {
    $sampleArgs = @("--max_train_samples", "1024", "--max_eval_samples", "512")
    $evalArgs = @("--max_eval_samples", "512")
    $defaultEpochs = 1
} else {
    $sampleArgs = @()
    $evalArgs = @()
    $defaultEpochs = $null
}

$tasks = @(
    @{ Name = "sst2"; Epochs = 3; LR = "0.0002"; Note = "sentiment" },
    @{ Name = "mrpc"; Epochs = 5; LR = "0.0002"; Note = "paraphrase" },
    @{ Name = "rte";  Epochs = 5; LR = "0.0002"; Note = "entailment" },
    @{ Name = "cola"; Epochs = 5; LR = "0.0002"; Note = "grammar" }
)

$methods = @(
    @{ Name = "ft"; Label = "FT"; Extra = @("--method", "ft", "--learning_rate", "0.00002") },
    @{ Name = "bitfit"; Label = "BitFit"; Extra = @("--method", "bitfit", "--learning_rate", "0.0001") },
    @{ Name = "adapter_0p3m"; Label = "Adapter0.3M"; Extra = @("--method", "adapter", "--adapter_size", "16", "--adapter_location", "output", "--adapter_dropout", "0.0", "--learning_rate", "0.0002") },
    @{ Name = "adapter_0p9m"; Label = "Adapter0.9M"; Extra = @("--method", "adapter", "--adapter_size", "48", "--adapter_location", "output", "--adapter_dropout", "0.0", "--learning_rate", "0.0002") },
    @{ Name = "lora"; Label = "LoRA"; Extra = @("--method", "lora", "--lora_r", "4", "--lora_alpha", "32", "--lora_dropout", "0.1", "--target_modules", "query,value", "--learning_rate", "0.0002") }
)

New-Item -ItemType Directory -Force $ResultsRoot | Out-Null

foreach ($task in $tasks) {
    Write-Host "Preparing GLUE task $($task.Name)..." -ForegroundColor Cyan
    python reimpl\prepare_glue.py --task_name $task.Name --output_dir "data\glue_$($task.Name)" --preview_samples 20
}

foreach ($task in $tasks) {
    foreach ($method in $methods) {
        $epochs = if ($defaultEpochs -ne $null) { $defaultEpochs } else { $task.Epochs }
        $outputDir = Join-Path $ResultsRoot "$($task.Name)_$($method.Name)"
        Write-Host "Running $($method.Label) on $($task.Name), output=$outputDir" -ForegroundColor Green
        $cmd = @(
            "reimpl\train_my_lora_nlu.py",
            "--task_name", $task.Name,
            "--model_name", $ModelName,
            "--output_dir", $outputDir,
            "--epochs", "$epochs",
            "--batch_size", "16"
        ) + $method.Extra + $sampleArgs
        python @cmd

        Write-Host "Evaluating $($method.Label) on $($task.Name)" -ForegroundColor DarkGreen
        $evalCmd = @(
            "reimpl\evaluate_my_lora_nlu.py",
            "--checkpoint_dir", "$outputDir\checkpoint",
            "--output_dir", $outputDir
        ) + $evalArgs
        python @evalCmd
    }
}

python reimpl\plot_results.py --results_root $ResultsRoot --paper_tasks "sst2,mrpc,cola,rte"

Write-Host "Done. Results written to $ResultsRoot" -ForegroundColor Cyan
Write-Host "Main table: $ResultsRoot\paper_style_4task_table.md" -ForegroundColor Cyan
