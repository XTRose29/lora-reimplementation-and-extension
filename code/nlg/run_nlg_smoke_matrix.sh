#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASK="${1:?usage: run_nlg_smoke_matrix.sh <e2e|webnlg|dart>}"
ROOT="results/nlg/smoke/${TASK}"

mkdir -p "${ROOT}"

python code/nlg/run_qwen_nlg_generation.py \
  --task "${TASK}" \
  --method ft \
  --prompt_variant strict \
  --output_dir "${ROOT}/qwen_${TASK}_strict_ft" \
  --max_train_examples 128 \
  --max_eval_examples 64 \
  --epochs 1 \
  --batch_size 2 \
  --max_length 384 \
  --max_new_tokens 64 \
  --seed 42

python code/nlg/run_qwen_nlg_generation.py \
  --task "${TASK}" \
  --method lora \
  --prompt_variant strict \
  --output_dir "${ROOT}/qwen_${TASK}_strict_lora" \
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

python code/nlg/run_qwen_nlg_generation.py \
  --task "${TASK}" \
  --method ft \
  --prompt_variant abstain \
  --output_dir "${ROOT}/qwen_${TASK}_abstain_ft" \
  --max_train_examples 128 \
  --max_eval_examples 64 \
  --epochs 1 \
  --batch_size 2 \
  --max_length 384 \
  --max_new_tokens 64 \
  --seed 42

python code/nlg/run_qwen_nlg_generation.py \
  --task "${TASK}" \
  --method lora \
  --prompt_variant abstain \
  --output_dir "${ROOT}/qwen_${TASK}_abstain_lora" \
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

python code/nlg/summarize_nlg_results.py --results_root "${ROOT}"
