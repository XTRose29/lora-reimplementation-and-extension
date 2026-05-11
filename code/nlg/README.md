# NLG Experiments

This folder contains the portable Python code for the NLG part of the project. Cluster submission files and shell launch scripts are intentionally omitted from this repository folder; the commands below show the exact Python calls needed to reproduce the reported runs.

## What This Code Covers

- **GPT-2 Medium on E2E**: reproduction-style comparison with the LoRA paper's GPT-2 Medium E2E NLG table.
- **Qwen2.5-0.5B-Instruct on E2E/WebNLG**: extension for structured data-to-text generation with self-reported confidence.

The main script is `run_qwen_nlg_generation.py`. Despite the historical filename, it supports any Hugging Face causal language model, including `gpt2-medium` and `Qwen/Qwen2.5-0.5B-Instruct`.

## Files

- `run_qwen_nlg_generation.py`: train and evaluate one FT or LoRA NLG run.
- `summarize_nlg_results.py`: aggregate run directories into `summary_table.csv` and `summary_table.md`.
- `make_nlg_smoke_html.py`: optional helper for inspecting generated outputs.

## Setup

From the repository root:

```bash
pip install -r code/requirements.txt
export PYTHONPATH="$PWD/code:$PYTHONPATH"
```

The scripts download GEM datasets through Hugging Face `datasets`. Use a CUDA GPU for the full runs.

## GPT-2 Medium E2E Reproduction

These two runs reproduce the rows we compare against the LoRA paper's E2E NLG table. We report BLEU and ROUGE-L because those are the metrics shared by the paper table and our evaluation pipeline.

```bash
python code/nlg/run_qwen_nlg_generation.py \
  --task e2e \
  --model_name gpt2-medium \
  --method ft \
  --prompt_variant strict \
  --output_dir results/nlg/paper_gpt2_full/e2e/gpt2_medium_e2e_ft \
  --max_train_examples 5000 \
  --max_eval_examples 1000 \
  --epochs 2 \
  --batch_size 2 \
  --learning_rate 5e-5 \
  --max_length 384 \
  --max_new_tokens 64 \
  --seed 42

python code/nlg/run_qwen_nlg_generation.py \
  --task e2e \
  --model_name gpt2-medium \
  --method lora \
  --prompt_variant strict \
  --output_dir results/nlg/paper_gpt2_full/e2e/gpt2_medium_e2e_lora \
  --max_train_examples 5000 \
  --max_eval_examples 1000 \
  --epochs 2 \
  --batch_size 2 \
  --learning_rate 2e-4 \
  --max_length 384 \
  --max_new_tokens 64 \
  --lora_r 4 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --seed 42

python code/nlg/summarize_nlg_results.py --results_root results/nlg/paper_gpt2_full/e2e
```

## Qwen NLG Confidence Extension

Run Qwen FT/LoRA on E2E and WebNLG. Replace `e2e` with `webnlg` for the second task.

```bash
python code/nlg/run_qwen_nlg_generation.py \
  --task e2e \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --method ft \
  --prompt_variant strict \
  --output_dir results/nlg/full/e2e/qwen_e2e_strict_ft \
  --max_train_examples 5000 \
  --max_eval_examples 1000 \
  --epochs 2 \
  --batch_size 2 \
  --max_length 384 \
  --max_new_tokens 64 \
  --seed 42

python code/nlg/run_qwen_nlg_generation.py \
  --task e2e \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --method lora \
  --prompt_variant strict \
  --output_dir results/nlg/full/e2e/qwen_e2e_strict_lora \
  --max_train_examples 5000 \
  --max_eval_examples 1000 \
  --epochs 2 \
  --batch_size 2 \
  --max_length 384 \
  --max_new_tokens 64 \
  --lora_r 4 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --seed 42

python code/nlg/summarize_nlg_results.py --results_root results/nlg/full/e2e
```

## Quick Smoke Test

Use smaller sample counts before launching full runs:

```bash
python code/nlg/run_qwen_nlg_generation.py \
  --task e2e \
  --method lora \
  --prompt_variant strict \
  --output_dir results/nlg/smoke/e2e/qwen_e2e_strict_lora \
  --max_train_examples 128 \
  --max_eval_examples 64 \
  --epochs 1 \
  --batch_size 2 \
  --max_length 384 \
  --max_new_tokens 64 \
  --lora_r 4 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --seed 42
```
