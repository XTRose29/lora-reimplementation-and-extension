# My NLU LoRA Reimplementation

This folder contains the shared NLU LoRA implementation used by the main NLU experiments. The NLG code also reuses `my_lora.py` for standard linear LoRA layers; GPT-2-specific `Conv1D` LoRA support lives in `../nlg/run_qwen_nlg_generation.py`.

The original Microsoft LoRA repo is kept in `../lora/` only as reference code.

## What We Need To Run

We focus on four GLUE NLU tasks. These are enough for a realistic course project on one local GPU while still covering different kinds of language understanding.

| Task | What It Tests | Example |
| --- | --- | --- |
| SST-2 | Sentiment classification | `"This movie is excellent." -> positive` |
| MRPC | Paraphrase detection | `"The company bought the startup."` vs `"The startup was acquired."` -> same meaning? |
| RTE | Textual entailment | Premise: `"A man is playing guitar."` Hypothesis: `"A person is making music."` -> entailment? |
| CoLA | Grammatical acceptability | `"She seems happy."` -> acceptable? |

These four tasks are a practical subset of the LoRA paper's GLUE table:

```text
MNLI SST-2 MRPC CoLA QNLI QQP RTE STS-B Avg
```

We do not need to run all eight tasks at first. MNLI and QQP are much larger and can take many hours on a single RTX 4060.

The four-task paper reference table is saved in:

```text
my_NLU/paper_reference_4task_table.md
```

## Setup

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r reimpl\requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Optional Data Preview

The training script downloads GLUE directly through Hugging Face `datasets`. This preview command is only for checking the data format.

```powershell
python reimpl\prepare_glue.py `
  --task_name sst2 `
  --output_dir data\glue_sst2 `
  --preview_samples 50
```

## Smoke Test

Run this first to verify the full pipeline.

```powershell
python reimpl\train_my_lora_nlu.py `
  --task_name sst2 `
  --model_name roberta-base `
  --output_dir my_NLU\results\roberta_sst2_r4_smoke `
  --lora_r 4 `
  --lora_alpha 32 `
  --lora_dropout 0.1 `
  --target_modules query,value `
  --epochs 1 `
  --batch_size 8 `
  --learning_rate 0.0002 `
  --max_train_samples 512 `
  --max_eval_samples 256
```

Evaluate the saved checkpoint:

```powershell
python reimpl\evaluate_my_lora_nlu.py `
  --checkpoint_dir my_NLU\results\roberta_sst2_r4_smoke\checkpoint `
  --output_dir my_NLU\results\roberta_sst2_r4_smoke `
  --max_eval_samples 256
```

## One-Command Full Four-Task Table

This runs all rows we need for the four selected task columns:

```text
FT
BitFit
Adapter, about 0.3M trainable params
Adapter, about 0.9M trainable params
LoRA
```

on:

```text
SST-2, MRPC, RTE, CoLA
```

Run in the foreground:

```powershell
powershell -ExecutionPolicy Bypass -File .\my_NLU\run_four_task_table.ps1
```

Run a quick subset version first:

```powershell
powershell -ExecutionPolicy Bypass -File .\my_NLU\run_four_task_table.ps1 -Quick
```

Start it in the background and write logs:

```powershell
powershell -ExecutionPolicy Bypass -File .\my_NLU\run_four_task_table_background.ps1
```

The background helper prints a job id and a log path. Check progress with:

```powershell
Receive-Job -Id <JOB_ID> -Keep
```

Expected final table:

```text
my_NLU/results/four_task_table/paper_style_4task_table.md
my_NLU/results/four_task_table/paper_style_4task_table.csv
```

Important: the full table can take several hours on one RTX 4060 because it trains 20 runs total: 4 tasks x 5 methods.

## Run One Task At A Time

Use this when teammates want to split the work. Each command runs all five methods for one task:

```text
FT
BitFit
Adapter, about 0.3M trainable params
Adapter, about 0.9M trainable params
LoRA
```

Quick check for one task:

```powershell
powershell -ExecutionPolicy Bypass -File .\my_NLU\run_one_task_table.ps1 -TaskName sst2 -Quick
```

Full single-task commands:

```powershell
powershell -ExecutionPolicy Bypass -File .\my_NLU\run_one_task_table.ps1 -TaskName sst2
powershell -ExecutionPolicy Bypass -File .\my_NLU\run_one_task_table.ps1 -TaskName mrpc
powershell -ExecutionPolicy Bypass -File .\my_NLU\run_one_task_table.ps1 -TaskName rte
powershell -ExecutionPolicy Bypass -File .\my_NLU\run_one_task_table.ps1 -TaskName cola
```

Start one task in the background:

```powershell
powershell -ExecutionPolicy Bypass -File .\my_NLU\run_one_task_table_background.ps1 -TaskName sst2
```

Output location:

```text
my_NLU/results/by_task/<task>/
```

Example output table:

```text
my_NLU/results/by_task/sst2/paper_style_sst2_table.md
```

## Main LoRA Experiments

These commands use `roberta-base` to match the paper setting more closely than `distilroberta-base`.

### SST-2

Purpose: fastest main task; validates sentiment classification.

```powershell
python reimpl\train_my_lora_nlu.py --task_name sst2 --model_name roberta-base --output_dir my_NLU\results\roberta_sst2_r4 --lora_r 4 --lora_alpha 32 --lora_dropout 0.1 --target_modules query,value --epochs 3 --batch_size 16 --learning_rate 0.0002
```

### MRPC

Purpose: tests sentence-pair paraphrase detection.

```powershell
python reimpl\train_my_lora_nlu.py --task_name mrpc --model_name roberta-base --output_dir my_NLU\results\roberta_mrpc_r4 --lora_r 4 --lora_alpha 32 --lora_dropout 0.1 --target_modules query,value --epochs 5 --batch_size 16 --learning_rate 0.0002
```

### RTE

Purpose: tests entailment on a small dataset.

```powershell
python reimpl\train_my_lora_nlu.py --task_name rte --model_name roberta-base --output_dir my_NLU\results\roberta_rte_r4 --lora_r 4 --lora_alpha 32 --lora_dropout 0.1 --target_modules query,value --epochs 5 --batch_size 16 --learning_rate 0.0002
```

### CoLA

Purpose: tests grammatical acceptability.

```powershell
python reimpl\train_my_lora_nlu.py --task_name cola --model_name roberta-base --output_dir my_NLU\results\roberta_cola_r4 --lora_r 4 --lora_alpha 32 --lora_dropout 0.1 --target_modules query,value --epochs 5 --batch_size 16 --learning_rate 0.0002
```

## LoRA Ablation Experiments

These are our main project analysis runs. They test how LoRA rank and target modules affect accuracy and parameter efficiency.

### Rank Ablation On SST-2

```powershell
python reimpl\train_my_lora_nlu.py --task_name sst2 --model_name roberta-base --output_dir my_NLU\results\roberta_sst2_r1 --lora_r 1 --lora_alpha 32 --lora_dropout 0.1 --target_modules query,value --epochs 3 --batch_size 16 --learning_rate 0.0002
python reimpl\train_my_lora_nlu.py --task_name sst2 --model_name roberta-base --output_dir my_NLU\results\roberta_sst2_r4 --lora_r 4 --lora_alpha 32 --lora_dropout 0.1 --target_modules query,value --epochs 3 --batch_size 16 --learning_rate 0.0002
python reimpl\train_my_lora_nlu.py --task_name sst2 --model_name roberta-base --output_dir my_NLU\results\roberta_sst2_r8 --lora_r 8 --lora_alpha 32 --lora_dropout 0.1 --target_modules query,value --epochs 3 --batch_size 16 --learning_rate 0.0002
```

### Target/Dropout Ablation On SST-2

```powershell
python reimpl\train_my_lora_nlu.py --task_name sst2 --model_name roberta-base --output_dir my_NLU\results\roberta_sst2_r4_query_only --lora_r 4 --lora_alpha 32 --lora_dropout 0.1 --target_modules query --epochs 3 --batch_size 16 --learning_rate 0.0002
python reimpl\train_my_lora_nlu.py --task_name sst2 --model_name roberta-base --output_dir my_NLU\results\roberta_sst2_r4_no_dropout --lora_r 4 --lora_alpha 32 --lora_dropout 0.0 --target_modules query,value --epochs 3 --batch_size 16 --learning_rate 0.0002
```

## Summarize Results

```powershell
python reimpl\plot_results.py --results_root my_NLU\results
```

Expected outputs:

- `my_NLU/results/<experiment>/metrics.json`
- `my_NLU/results/<experiment>/eval.txt`
- `my_NLU/results/<experiment>/parameter_count.json`
- `my_NLU/results/<experiment>/train_log.jsonl`
- `my_NLU/results/summary_table.csv`
- `my_NLU/results/figures/metric_vs_rank.png`
- `my_NLU/results/figures/trainable_params_vs_metric.png`

## What About FT, BitFit, And Adapters?

The LoRA paper compares several methods:

```text
RoBbase (FT)
RoBbase (BitFit)
RoBbase (AdptD)
RoBbase (LoRA)
```

The original Microsoft LoRA repo does contain code paths for these comparison methods:

- Full fine-tuning: default `run_glue.py` behavior when no PEFT method is enabled.
- BitFit: `--apply_bitfit`.
- Adapters: `--apply_adapter`, `--adapter_type`, `--adapter_size`.
- LoRA: `--apply_lora`.

Relevant original files:

- `../lora/examples/NLU/examples/text-classification/run_glue.py`
- `../lora/examples/NLU/src/transformers/models/adapter.py`
- `../lora/examples/NLU/src/transformers/models/roberta/modeling_roberta.py`
- `../lora/examples/NLU/adapter_houlsby_roberta_large_mnli.sh`
- `../lora/examples/NLU/adapter_pfeiffer_roberta_large_mnli.sh`

For this project, our own reimplementation focuses on LoRA. FT, BitFit, and adapters are useful baselines, but they are not the core mechanism we are claiming to reimplement.
