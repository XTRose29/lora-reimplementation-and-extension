$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

python .\cola_reliability\run_cola_reliability.py `
  --method ft `
  --model_name roberta-base `
  --output_dir cola_reliability\results\quick_ft_128 `
  --epochs 1 `
  --batch_size 8 `
  --max_train_samples 128 `
  --max_eval_samples 128 `
  --max_ood_samples 128 `
  --abstention_threshold 0.80

python .\cola_reliability\run_cola_reliability.py `
  --method lora `
  --model_name roberta-base `
  --output_dir cola_reliability\results\quick_lora_r4_attn_128 `
  --lora_r 4 `
  --lora_placement attention `
  --epochs 1 `
  --batch_size 8 `
  --max_train_samples 128 `
  --max_eval_samples 128 `
  --max_ood_samples 128 `
  --abstention_threshold 0.80

python .\cola_reliability\summarize_results.py --results_root cola_reliability\results
