# New NLU Pipeline: Task-Probed LoRA

This pipeline tests the new idea in `newimpl/`: use a cheap task probe to decide where to place LoRA adapters in RoBERTa attention.

## What It Runs

For each GLUE task, we compare:

| Strategy | Meaning |
|---|---|
| `full` | LoRA on all attention query/value modules |
| `random` | Randomly choose `top_k` query/value modules |
| `gradient` | Choose `top_k` by gradient energy |
| `activation` | Choose `top_k` by activation energy |
| `actgrad` | Choose `top_k` by combined activation + gradient score |

Default tasks:

```text
sst2, mrpc, rte, cola
```

Default candidates:

```text
roberta.encoder.layer.*.attention.self.query
roberta.encoder.layer.*.attention.self.value
```

## Setup

From the repo root:

```powershell
pip install -r reimpl\requirements.txt
```

## Quick Smoke Test

This checks the pipeline only, not final numbers.

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName sst2 -ModelName roberta-base -TopK 8 -Quick
```

For a faster CPU/debug run:

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName sst2 -ModelName distilroberta-base -TopK 4 -Quick
```

## Full Runs

Single task:

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName rte -ModelName roberta-base -TopK 8
```

Four tasks:

```powershell
.\newimpl\run_four_task_probe.ps1 -ModelName roberta-base -TopK 8
```

## Five Main Commands

Use these five commands for the actual NLU probe experiments.

Combined four-task run:

```powershell
.\newimpl\run_four_task_probe.ps1 -ModelName roberta-base -TopK 8
```

Split task runs:

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName sst2 -ModelName roberta-base -TopK 8
```

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName mrpc -ModelName roberta-base -TopK 8
```

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName rte -ModelName roberta-base -TopK 8
```

```powershell
.\newimpl\run_one_task_probe.ps1 -TaskName cola -ModelName roberta-base -TopK 8
```

Background four-task run:

```powershell
$repo = (Get-Location).Path
Start-Job -ArgumentList $repo -ScriptBlock { param($repo) Set-Location $repo; .\newimpl\run_four_task_probe.ps1 -ModelName roberta-base -TopK 8 }
```

## Manual Single Strategy

```powershell
python newimpl\train_task_probed_lora_nlu.py `
  --task_name sst2 `
  --model_name roberta-base `
  --output_dir newimpl\results\nlu_probe\sst2\sst2_actgrad_top8 `
  --probe_strategy actgrad `
  --top_k 8 `
  --probe_samples 1024 `
  --candidate_scope attention `
  --target_modules query,value `
  --lora_r 4 `
  --lora_alpha 32 `
  --lora_dropout 0.1 `
  --epochs 3 `
  --batch_size 16 `
  --learning_rate 0.0002

python newimpl\evaluate_task_probed_lora_nlu.py `
  --checkpoint_dir newimpl\results\nlu_probe\sst2\sst2_actgrad_top8\checkpoint `
  --output_dir newimpl\results\nlu_probe\sst2\sst2_actgrad_top8
```

## Outputs

Each experiment folder contains:

| File | Meaning |
|---|---|
| `probe_scores.json` | activation, gradient, and ActGrad score for every candidate module |
| `probe_selection.json` | selected modules used for LoRA injection |
| `train_config.json` | full training/probe configuration |
| `parameter_count.json` | trainable parameter count |
| `metrics.json` / `eval.txt` | validation metrics |
| `checkpoint/` | trainable LoRA + classifier head checkpoint |

The summary script writes:

```text
newimpl/results/nlu_probe/probe_summary_table.csv
newimpl/results/nlu_probe/figures/
```

## Main Comparison

The cleanest report table is:

```text
Full LoRA
Random top-8
Gradient top-8
Activation top-8
ActGrad top-8
```

If `ActGrad top-8` is close to `Full LoRA` and better than `Random top-8`, the probe is useful.
