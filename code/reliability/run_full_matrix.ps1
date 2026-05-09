param(
  [switch]$Quick
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if ($Quick) {
  $Epochs = 1
  $TrainSizes = @(256)
  $EvalMax = 256
  $OodMax = 256
} else {
  $Epochs = 5
  $TrainSizes = @(512, 2000, 8551)
  $EvalMax = $null
  $OodMax = 1000
}

function Run-CoLA {
  param(
    [string]$Name,
    [string]$Method,
    [int]$TrainSize,
    [int]$Rank = 4,
    [string]$Placement = "attention",
    [string]$Layers = "all"
  )

  $Args = @(
    ".\cola_reliability\run_cola_reliability.py",
    "--method", $Method,
    "--model_name", "roberta-base",
    "--output_dir", "cola_reliability\results\$Name",
    "--epochs", "$Epochs",
    "--batch_size", "16",
    "--max_train_samples", "$TrainSize",
    "--max_ood_samples", "$OodMax",
    "--abstention_threshold", "0.80"
  )
  if ($EvalMax) {
    $Args += @("--max_eval_samples", "$EvalMax")
  }
  if ($Method -eq "lora") {
    $Args += @("--lora_r", "$Rank", "--lora_placement", "$Placement", "--layer_indices", "$Layers")
  }
  python @Args
}

foreach ($Size in $TrainSizes) {
  Run-CoLA -Name "ft_train$Size" -Method "ft" -TrainSize $Size
  Run-CoLA -Name "lora_r4_attention_train$Size" -Method "lora" -TrainSize $Size -Rank 4 -Placement "attention"
}

foreach ($Rank in @(1, 4, 8, 16)) {
  Run-CoLA -Name "lora_r${Rank}_attention_train$($TrainSizes[-1])" -Method "lora" -TrainSize $TrainSizes[-1] -Rank $Rank -Placement "attention"
}

foreach ($Placement in @("attention", "mlp", "attention_mlp")) {
  Run-CoLA -Name "lora_r4_${Placement}_train$($TrainSizes[-1])" -Method "lora" -TrainSize $TrainSizes[-1] -Rank 4 -Placement $Placement
}

foreach ($Layers in @("0-3", "4-7", "8-11")) {
  $Clean = $Layers.Replace("-", "_")
  Run-CoLA -Name "lora_r4_attention_layers${Clean}_train$($TrainSizes[-1])" -Method "lora" -TrainSize $TrainSizes[-1] -Rank 4 -Placement "attention" -Layers $Layers
}

python .\cola_reliability\summarize_results.py --results_root cola_reliability\results
